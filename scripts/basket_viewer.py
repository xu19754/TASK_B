"""Interactive URDF-limited kinematic posture + MuJoCo cargo diagnostic.

Robot links are prescribed mocap bodies, NOT a free-base robot controller.
Only cargo has free dynamics. No evaluation code or runtime assets are changed.
Run with the dedicated launcher; --self-test exercises physics and rendering.
"""
from __future__ import annotations

import argparse
from dataclasses import dataclass
from datetime import datetime
import hashlib
import json
import math
from pathlib import Path
import sys
import time
import xml.etree.ElementTree as ET

import glfw
import mujoco
import numpy as np
from PIL import Image, ImageDraw, ImageFont
from scipy.spatial import ConvexHull

ROOT = Path(__file__).resolve().parents[1]
ANALYSIS = ROOT / 'reports/basket_feasibility'
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ANALYSIS))
import geometry_scan as geometry
from contact_screen import OBJECT_NAMES, inside_rim, quat_wxyz, read_obj
from taskb_student.constants import ARM_Q0, JOINT_LOWER, JOINT_NAMES, JOINT_UPPER, LEG_Q0, URDF_PATH

SCENARIOS = ['mixed_three', *OBJECT_NAMES]
SCENARIO_LABELS = ['混合三件', '三个糖盒', '三个芥末瓶', '三根香蕉']
OUT = ANALYSIS / 'interactive_runs'
DT = .001


def numbers(values):
    return ' '.join(f'{x:.12g}' for x in np.asarray(values).ravel())


@dataclass
class Pose:
    pitch: float
    x: float
    height: float
    q: np.ndarray
    links: dict
    margin: float
    lip: float
    ground: float
    support: float
    foot_error: float


class PostureLimits:
    """Hard URDF limits plus explicit, conservative screening assumptions."""
    def __init__(self):
        self.params = geometry.leg_parameters()
        nominal = geometry.forward(LEG_Q0 + ARM_Q0)
        self.feet = np.array([nominal[f'{leg}_foot'][:3, 3] for leg in geometry.LEGS])
        self.height = float(.04 - self.feet[:, 2].mean())
        self.feet[:, 2] += self.height
        self.hull = ConvexHull(self.feet[:, :2]).equations
        self.collision = geometry.collision_vertices()
        self.inertials = [(link.get('name'), geometry.origin(link.find('inertial/origin'))[:3, 3],
                           float(link.find('inertial/mass').get('value')))
                          for link in geometry.ROBOT.findall('link') if link.find('inertial') is not None]
        joints = {j.get('name'): j for j in geometry.ROBOT.findall('joint')}
        self.velocity = np.array([float(joints[n].find('limit').get('velocity')) for n in JOINT_NAMES])
        for i, name in enumerate(JOINT_NAMES):
            limit = joints[name].find('limit')
            assert abs(float(limit.get('lower')) - JOINT_LOWER[i]) < 1e-8
            assert abs(float(limit.get('upper')) - JOINT_UPPER[i]) < 1e-8

    def solve(self, pitch, x, height, require_rim=True):
        theta = math.radians(pitch)
        r = geometry.rotation([0, 1, 0], -theta)
        q, valid, margins = geometry.inverse_batch(((self.feet - [x, 0, height]) @ r)[None], self.params)
        if not valid[0]:
            return None, '拒绝：腿部不可达或超出 URDF 关节限位'
        if margins[0] < math.radians(2) - 1e-10:
            return None, '拒绝：关节限位余量不足 2°'
        full = np.r_[q[0], ARM_Q0]
        links = geometry.forward(full, x, height, theta)
        ground = min(float((v @ links[n][:3, :3].T + links[n][:3, 3])[:, 2].min())
                     for n, v in self.collision.items())
        if ground < .01:
            return None, '拒绝：机身 / 髋 / 大腿离地不足 1 cm'
        basket = links['tail_basket']
        lip = float((basket[:3, :3] @ [-.139, 0, .1] + basket[:3, 3])[2])
        if require_rim and lip < .57 - 1e-10:
            return None, '拒绝：后出口低于桶沿 55 cm + 2 cm 余量'
        positions = [links[n][:3, :3] @ p + links[n][:3, 3] for n, p, _ in self.inertials]
        masses = [m for _, _, m in self.inertials]
        # Same conservative assumed cargo COM as the published geometry screen.
        positions.append(basket[:3, :3] @ [-.10, 0, .075] + basket[:3, 3])
        masses.append(1.5)
        com = np.average(positions, axis=0, weights=masses)
        support = float(np.min(-(self.hull[:, :2] @ com[:2] + self.hull[:, 2]) /
                               np.linalg.norm(self.hull[:, :2], axis=1)))
        if support < .02:
            return None, '拒绝：估算质心投影离支撑边界不足 2 cm'
        error = max(np.linalg.norm(links[f'{leg}_foot'][:3, 3] - self.feet[i])
                    for i, leg in enumerate(geometry.LEGS))
        assert error < 1e-8
        return Pose(pitch, x, height, full, links, math.degrees(margins[0]), lip,
                    ground, support, float(error)), ''

    def search(self, pitch, require_rim=True):
        # Exact same grid and prioritization as geometry_scan.py, vectorized IK.
        xx, zz = np.meshgrid(np.linspace(-.15, .15, 121), np.linspace(.30, .65, 141))
        translations = np.c_[xx.ravel(), np.zeros(xx.size), zz.ravel()]
        theta = math.radians(pitch)
        r = geometry.rotation([0, 1, 0], -theta)
        q, valid, margins = geometry.inverse_batch((self.feet[None] - translations[:, None]) @ r, self.params)
        valid &= margins >= math.radians(2)
        if require_rim:
            valid &= translations[:, 2] - .439 * math.sin(theta) + .23 * math.cos(theta) >= .57 - 1e-10
        candidates = np.flatnonzero(valid)
        scores = (translations[candidates, 0] / .15) ** 2 + ((translations[candidates, 2] - self.height) / .15) ** 2
        for idx in candidates[np.argsort(scores)]:
            pose, _ = self.solve(pitch, translations[idx, 0], translations[idx, 2], require_rim)
            if pose is not None:
                return pose, ''
        return None, f'拒绝：{pitch:.1f}° 在当前固定四足与筛查范围内无可用姿态'


def build_scene():
    """URDF geometry/inertials copied to independently prescribed link frames."""
    root = ET.Element('mujoco', model='TaskB_prescribed_posture_cargo_diagnostic')
    ET.SubElement(root, 'compiler', angle='radian', inertiafromgeom='auto')
    ET.SubElement(root, 'option', timestep=str(DT), gravity='0 0 -9.81', iterations='60', cone='elliptic')
    ET.SubElement(root, 'size', memory='64M')
    visual = ET.SubElement(root, 'visual')
    ET.SubElement(visual, 'global', offwidth='1600', offheight='1100')
    ET.SubElement(visual, 'headlight', diffuse='.65 .65 .65', ambient='.35 .35 .35', specular='.15 .15 .15')
    assets = ET.SubElement(root, 'asset')
    ET.SubElement(assets, 'texture', name='grid', type='2d', builtin='checker', width='256', height='256',
                  rgb1='.19 .22 .28', rgb2='.23 .26 .32')
    ET.SubElement(assets, 'material', name='floor_mat', texture='grid', texrepeat='5 5', reflectance='.05')
    world = ET.SubElement(root, 'worldbody')
    ET.SubElement(world, 'light', pos='-1 -1 3', dir='.2 .2 -1', diffuse='.65 .65 .65')
    ET.SubElement(world, 'geom', name='floor', type='plane', size='4 4 .05', material='floor_mat',
                  contype='1', conaffinity='2', group='0')
    mesh_assets = {}

    def add_geometry(parent, element, name, contact, color, group):
        transform = geometry.origin(element.find('origin'))
        attrib = dict(name=name, pos=numbers(transform[:3, 3]), quat=numbers(quat_wxyz(transform[:3, :3])),
                      contype='1' if contact else '0', conaffinity='2' if contact else '0',
                      group=str(group), rgba=numbers(color), friction='.5 .005 .0001', condim='3',
                      solref='.004 1', solimp='.95 .99 .001')
        geom = element.find('geometry')[0]
        if geom.tag == 'mesh':
            file = URDF_PATH.parent / geom.get('filename')
            scale = geom.get('scale', '1 1 1')
            key = (str(file), scale)
            if key not in mesh_assets:
                mesh_name = f'mesh_{len(mesh_assets)}'
                ET.SubElement(assets, 'mesh', name=mesh_name, file=str(file), scale=scale, inertia='convex')
                mesh_assets[key] = mesh_name
            attrib.update(type='mesh', mesh=mesh_assets[key])
        elif geom.tag == 'box':
            attrib.update(type='box', size=numbers(np.fromstring(geom.get('size'), sep=' ') / 2))
        elif geom.tag == 'cylinder':
            attrib.update(type='cylinder', size=f"{geom.get('radius')} {float(geom.get('length')) / 2}")
        elif geom.tag == 'sphere':
            attrib.update(type='sphere', size=geom.get('radius'))
        else:
            raise ValueError(geom.tag)
        ET.SubElement(parent, 'geom', **attrib)

    for link in geometry.ROBOT.findall('link'):
        name = link.get('name')
        body = ET.SubElement(world, 'body', name=name, mocap='true')
        inertial = link.find('inertial')
        if inertial is not None and float(inertial.find('mass').get('value')) > 0:
            t = geometry.origin(inertial.find('origin'))
            i = inertial.find('inertia')
            tensor = np.array([[float(i.get('ixx')), float(i.get('ixy')), float(i.get('ixz'))],
                               [float(i.get('ixy')), float(i.get('iyy')), float(i.get('iyz'))],
                               [float(i.get('ixz')), float(i.get('iyz')), float(i.get('izz'))]])
            tensor = t[:3, :3] @ tensor @ t[:3, :3].T
            ET.SubElement(body, 'inertial', pos=numbers(t[:3, 3]), mass=inertial.find('mass').get('value'),
                          fullinertia=numbers([tensor[0, 0], tensor[1, 1], tensor[2, 2],
                                              tensor[0, 1], tensor[0, 2], tensor[1, 2]]))
        for k, v in enumerate(link.findall('visual')):
            color = [.32, .35, .40, 1] if not name.startswith('link') else [.60, .63, .67, 1]
            if name.endswith('foot'):
                color = [.1, .12, .15, 1]
            if name == 'tail_basket':
                color = [.10, .48, .75, .28]
            explicit = v.find('material/color')
            if explicit is not None and name != 'tail_basket':
                color = np.fromstring(explicit.get('rgba'), sep=' ')
            add_geometry(body, v, f'visual_{name}_{k}', False, color, 2)
        for k, c in enumerate(link.findall('collision')):
            add_geometry(body, c, f'collision_{name}_{k}', True, [.1, .65, .9, .30], 3)
    objects = {}
    for name in OBJECT_NAMES:
        file = ANALYSIS / 'screen_meshes' / f'{name}.obj'
        if not file.exists():
            raise FileNotFoundError(f'Missing cached source-object mesh: {file}')
        vertices, _ = read_obj(file)
        objects[name] = dict(vertices=vertices, file=file)
        ET.SubElement(assets, 'mesh', name=name, file=str(file), inertia='convex')
    for i, name in enumerate(OBJECT_NAMES):
        body = ET.SubElement(world, 'body', name=f'cargo_{i}', pos=f'0 0 {1 + i * .2}')
        ET.SubElement(body, 'freejoint', name=f'cargo_joint_{i}')
        ET.SubElement(body, 'geom', name=f'cargo_geom_{i}', type='mesh', mesh=name, mass='.5',
                      group='0', contype='2', conaffinity='3', condim='3', friction='.5 .005 .0001',
                      solref='.004 1', solimp='.95 .99 .001', rgba=numbers([
                          [.92, .29, .12, 1], [.95, .70, .10, 1], [.92, .88, .20, 1]][i]))
    xml = ET.tostring(root, encoding='unicode')
    return mujoco.MjModel.from_xml_string(xml), objects, xml


class Diagnostic:
    def __init__(self):
        self.limits = PostureLimits()
        self.model, self.objects, self.xml = build_scene()
        self.data = mujoco.MjData(self.model)
        self.scenario = 0
        self.mu = .5
        self.require_rim = True
        self.auto = True
        self.paused = False
        self.speed = 1.0
        self.message = '姿态通过几何筛查；电机力矩与动态平衡尚未验证'
        self.pose, _ = self.limits.solve(0, 0, self.limits.height)
        assert self.pose is not None
        self.target = self.pose
        self.waypoints = []
        self.mocap = {n: self.model.body_mocapid[self.model.body(n).id] for n in self.pose.links}
        self.cargo_ids = [self.model.body(f'cargo_{i}').id for i in range(3)]
        self.geoms = [self.model.geom(f'cargo_geom_{i}').id for i in range(3)]
        self.qadr = [self.model.jnt_qposadr[self.model.body_jntadr[b]] for b in self.cargo_ids]
        self.escaped = [False] * 3
        self.assets = OBJECT_NAMES.copy()
        self.setup_cargo(0)

    def impose(self, pose):
        for n, t in pose.links.items():
            mid = self.mocap[n]
            self.data.mocap_pos[mid] = t[:3, 3]
            self.data.mocap_quat[mid] = quat_wxyz(t[:3, :3])

    def setup_cargo(self, index):
        self.scenario = index
        self.assets = ([*OBJECT_NAMES] if index == 0 else [OBJECT_NAMES[index - 1]] * 3)
        # Recompile per selection so MuJoCo recomputes hulls, mass, inertia and bounds.
        # Changing mesh references in a compiled model alone would leave stale AABBs.
        root = ET.fromstring(self.xml)
        for i, name in enumerate(self.assets):
            root.find(f".//geom[@name='cargo_geom_{i}']").set('mesh', name)
        self.model = mujoco.MjModel.from_xml_string(ET.tostring(root, encoding='unicode'))
        self.data = mujoco.MjData(self.model)
        self.mocap = {n: self.model.body_mocapid[self.model.body(n).id] for n in self.pose.links}
        self.cargo_ids = [self.model.body(f'cargo_{i}').id for i in range(3)]
        self.geoms = [self.model.geom(f'cargo_geom_{i}').id for i in range(3)]
        self.qadr = [self.model.jnt_qposadr[self.model.body_jntadr[b]] for b in self.cargo_ids]
        for bid in self.cargo_ids:
            adr = self.model.jnt_dofadr[self.model.body_jntadr[bid]]
            self.model.dof_damping[adr:adr + 3] = self.model.body_mass[bid] * 2
            self.model.dof_damping[adr + 3:adr + 6] = self.model.body_inertia[bid] * 4
        self.set_friction(self.mu)
        self.reset()

    def reset(self):
        mujoco.mj_resetData(self.model, self.data)
        self.impose(self.pose)
        sideways = np.array([[0, 0, 1], [1, 0, 0], [0, 1, 0]])
        if self.scenario == 0:
            placements = [([.075, 0, .030], np.eye(3)), ([-.025, 0, .055], np.eye(3)),
                          ([.075, 0, .090], sideways)]
        elif self.scenario == 1:
            placements = [([x, 0, z], np.eye(3)) for x, z in [(-.025, .040), (.075, .030), (.075, .085)]]
        elif self.scenario == 2:
            placements = [([x, 0, .057], geometry.rotation([0, 1, 0], math.pi / 2)) for x in [-.008, .052, .112]]
        else:
            placements = [([x, 0, .050], sideways) for x in [-.070, .009, .088]]
        t = self.pose.links['tail_basket']
        for adr, (p, r) in zip(self.qadr, placements):
            self.data.qpos[adr:adr + 3] = t[:3, :3] @ p + t[:3, 3]
            self.data.qpos[adr + 3:adr + 7] = quat_wxyz(t[:3, :3] @ r)
        self.escaped = [False] * 3
        self.target = self.pose
        self.waypoints = []
        mujoco.mj_forward(self.model, self.data)

    def home(self):
        self.pose, _ = self.limits.solve(0, 0, self.limits.height, self.require_rim)
        self.reset()

    def set_friction(self, value):
        self.mu = float(value)
        self.model.geom_friction[:, 0] = self.mu

    def request(self, pitch=None, x=None, height=None):
        current = self.target
        pitch = current.pitch if pitch is None else float(pitch)
        x = current.x if x is None else float(x)
        height = current.height if height is None else float(height)
        if not (0 <= pitch <= 40 and -.15 <= x <= .15 and .3 <= height <= .65):
            self.message = '拒绝：超出界面设定的姿态搜索范围'
            return False
        if self.auto and abs(pitch - current.pitch) > 1e-9:
            candidate, reason = self.limits.search(pitch, self.require_rim)
        else:
            candidate, reason = self.limits.solve(pitch, x, height, self.require_rim)
        if candidate is None:
            self.message = reason
            return False
        start = np.array([self.pose.pitch, self.pose.x, self.pose.height])
        end = np.array([candidate.pitch, candidate.x, candidate.height])
        routes = [[end], [np.array([start[0], end[1], end[2]]), end],
                  [np.array([end[0], start[1], start[2]]), end]]
        route_found = None
        for route in routes:
            previous = start
            valid_path = True
            for point in route:
                for fraction in np.linspace(0, 1, 101):
                    if self.limits.solve(*(previous + fraction * (point - previous)), self.require_rim)[0] is None:
                        valid_path = False
                        break
                if not valid_path:
                    break
                previous = point
            if valid_path:
                route_found = route
                break
        if route_found is None:
            self.message = '拒绝：终点可用，但未找到满足余量的过渡路径'
            return False
        self.target = candidate
        self.waypoints = [v for v in route_found if np.max(np.abs(v - start)) > 1e-10]
        self.message = '缓慢移动至目标姿态；每一步重新检查限位与筛查条件'
        return True

    def step(self):
        if self.paused:
            return
        old = self.pose
        point = self.waypoints[0] if self.waypoints else np.array([self.target.pitch, self.target.x, self.target.height])
        delta = point - np.array([old.pitch, old.x, old.height])
        # Deliberately slow: <= 4 deg/s and <= 2 cm/s; never instant tilt.
        count = max(1., float(np.max(np.abs(delta) / np.array([4 * DT, .02 * DT, .02 * DT]))))
        if np.max(np.abs(delta)) > 1e-9:
            v = np.array([old.pitch, old.x, old.height]) + delta / count
            candidate, reason = self.limits.solve(*v, self.require_rim)
            if candidate is None:
                self.target = self.pose
                self.waypoints = []
                self.message = '路径中断：' + reason.replace('拒绝：', '')
            elif np.any(np.abs(candidate.q - old.q) / DT > self.limits.velocity + 1e-8):
                self.target = self.pose
                self.waypoints = []
                self.message = '路径中断：超过 URDF 关节速度限制'
            else:
                self.pose = candidate
                self.impose(self.pose)
                if self.waypoints and np.max(np.abs(point - np.array([
                        self.pose.pitch, self.pose.x, self.pose.height]))) < 1e-9:
                    self.waypoints.pop(0)
        mujoco.mj_step(self.model, self.data)
        # mj_step integrates qpos; refresh derived poses before geometric metrics/rendering.
        mujoco.mj_forward(self.model, self.data)

    def cargo_local(self):
        basket = self.pose.links['tail_basket']
        out = []
        for bid, name in zip(self.cargo_ids, self.assets):
            world = self.objects[name]['vertices'] @ self.data.xmat[bid].reshape(3, 3).T + self.data.xpos[bid]
            out.append((world - basket[:3, 3]) @ basket[:3, :3])
        return out

    def metrics(self):
        local = self.cargo_local()
        rows = []
        for i, v in enumerate(local):
            rear = bool(v[:, 0].max() < -.145)
            self.escaped[i] |= rear
            rows.append(dict(asset=self.assets[i], below_rim=inside_rim(v), fully_rear_now=rear,
                             rear_exit_observed=self.escaped[i], rear_most_x_m=float(v[:, 0].min()),
                             front_most_x_m=float(v[:, 0].max()), highest_z_m=float(v[:, 2].max())))
        return rows

    def state(self):
        rows = self.metrics()
        return dict(scope='Prescribed URDF-limited link poses + MuJoCo free cargo; not motor/balance/scoring validation',
                    urdf=str(URDF_PATH), urdf_sha256=hashlib.sha256(URDF_PATH.read_bytes()).hexdigest(),
                    mujoco=mujoco.__version__, simulated_time_s=float(self.data.time), friction=self.mu,
                    scenario=SCENARIOS[self.scenario], actual_pitch_deg=self.pose.pitch,
                    target_pitch_deg=self.target.pitch, base_x_m=self.pose.x, base_height_m=self.pose.height,
                    joint_margin_deg=self.pose.margin, rear_lip_world_z_m=self.pose.lip,
                    illegal_body_ground_clearance_m=self.pose.ground, assumed_com_support_margin_m=self.pose.support,
                    fixed_foot_error_m=self.pose.foot_error, require_rim=self.require_rim,
                    q_rad=dict(zip(JOINT_NAMES, self.pose.q.tolist())), cargo=rows,
                    assumptions=['19 URDF joint angles bounded externally; robot uses mocap bodies, no motor dynamics',
                                 'basket mass 0.2 kg and exact 16 collisions; each cargo 0.5 kg, source-mesh convex hull',
                                 '2 degree reserve, 1 cm body clearance, 2 cm assumed COM support margin are screening assumptions',
                                 'no reaction-force balance, motor torque or foot-slip validation; no task bucket or score',
                                 'mocap motion has no dynamic robot feedback; use only slow tilt, not impact/throw analysis',
                                 'robot mesh contacts are convex hulls, not Isaac convex decomposition'])


class Window:
    PANEL = 420
    WIDTH, HEIGHT = 1440, 940

    def __init__(self, sim, visible=True):
        self.sim = sim
        if not glfw.init():
            raise RuntimeError('GLFW could not initialize a local display.')
        glfw.window_hint(glfw.VISIBLE, glfw.TRUE if visible else glfw.FALSE)
        self.window = glfw.create_window(self.WIDTH, self.HEIGHT, 'TaskB | 篮子姿态与倾倒交互分析（规定姿态）', None, None)
        if not self.window:
            glfw.terminate()
            raise RuntimeError('OpenGL window could not be created.')
        glfw.make_context_current(self.window)
        glfw.swap_interval(1)
        self.camera = mujoco.MjvCamera()
        self.options = mujoco.MjvOption()
        self.options.geomgroup[3] = 0
        self.scene = mujoco.MjvScene(sim.model, maxgeom=10000)
        self.context = mujoco.MjrContext(sim.model, mujoco.mjtFontScale.mjFONTSCALE_150)
        self.current_model = sim.model
        self.transparent = True
        self.collision = False
        self.drag_slider = None
        self.drag_camera = None
        self.last_cursor = (0., 0.)
        self.rects = []
        self.sliders = []
        self.last_panel = 0
        self.panel = None
        self.font_path = next((str(p) for p in [Path('/System/Library/Fonts/STHeiti Medium.ttc'),
                              Path('/System/Library/Fonts/Supplemental/Arial Unicode.ttf'),
                              Path('/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf')] if p.exists()), None)
        self.fonts = {n: ImageFont.truetype(self.font_path, n) if self.font_path else ImageFont.load_default()
                      for n in [13, 14, 15, 16, 18, 22]}
        self.view('tail')
        glfw.set_mouse_button_callback(self.window, self.mouse_button)
        glfw.set_cursor_pos_callback(self.window, self.mouse_move)
        glfw.set_scroll_callback(self.window, self.scroll)
        glfw.set_key_callback(self.window, self.key)

    def refresh_model(self):
        if self.current_model is not self.sim.model:
            self.context.free()
            self.scene = mujoco.MjvScene(self.sim.model, maxgeom=10000)
            self.context = mujoco.MjrContext(self.sim.model, mujoco.mjtFontScale.mjFONTSCALE_150)
            self.current_model = self.sim.model
        basket_geom = self.sim.model.geom('visual_tail_basket_0').id
        self.sim.model.geom_rgba[basket_geom, 3] = .25 if self.transparent else 1
        self.options.geomgroup[2] = int(not self.collision)
        self.options.geomgroup[3] = int(self.collision)

    def view(self, name):
        if name == 'whole':
            self.camera.lookat[:] = [-.05, 0, .43]
            self.camera.distance, self.camera.azimuth, self.camera.elevation = 1.65, 135, -22
        elif name == 'tail':
            self.camera.lookat[:] = [-.29, 0, .65]
            self.camera.distance, self.camera.azimuth, self.camera.elevation = .82, 150, -34
        else:
            self.camera.lookat[:] = [-.25, 0, .56]
            self.camera.distance, self.camera.azimuth, self.camera.elevation = 1.16, 90, -10

    def select_scenario(self, index):
        self.sim.home()
        self.sim.setup_cargo(index)
        self.last_panel = 0

    def toggle_rim(self):
        if not self.sim.require_rim:
            if self.sim.pose.lip < .57 or self.sim.target.lip < .57:
                self.sim.message = '请先回到初始姿态，再启用桶沿高度条件'
                return
        self.sim.require_rim = not self.sim.require_rim

    def make_panel(self):
        im = Image.new('RGB', (self.PANEL, self.HEIGHT), '#101a29')
        draw = ImageDraw.Draw(im)
        self.rects, self.sliders = [], []
        white, muted, blue = '#eef4fd', '#9caec5', '#68b6ff'

        def text_at(y, text, size=15, color=white, x=20):
            draw.text((x, y), text, font=self.fonts[size], fill=color)

        def button(x, y, w, label, action, selected=False):
            box = (x, y, x + w, y + 31)
            draw.rounded_rectangle(box, 6, fill='#245b91' if selected else '#24354b')
            draw.text((x + 9, y + 7), label, font=self.fonts[14], fill=white)
            self.rects.append((box, action))

        def slider(y, label, value, low, high, action, fmt):
            text_at(y, label, 15)
            text_at(y, fmt.format(value), 15, blue, x=287)
            a, b, sy = 24, self.PANEL - 25, y + 31
            draw.line((a, sy, b, sy), fill='#3b4c65', width=5)
            vx = a + (b - a) * np.clip((value - low) / (high - low), 0, 1)
            draw.line((a, sy, vx, sy), fill=blue, width=5)
            draw.ellipse((vx - 7, sy - 7, vx + 7, sy + 7), fill=white)
            self.sliders.append(((a - 9, sy - 12, b + 9, sy + 12), low, high, action))

        sim = self.sim
        text_at(15, '篮子姿态与倾倒', 22)
        text_at(47, 'URDF 限位姿态 + MuJoCo 货物接触', 15, blue)
        text_at(70, '机身被规定姿态；未验证电机力矩 / 动态平衡', 13, '#f4c97b')
        button(20, 98, 87, '继续' if sim.paused else '暂停', lambda: setattr(sim, 'paused', not sim.paused))
        button(117, 98, 132, '回正并重新装载', sim.home)
        button(259, 98, 140, '当前姿态重新装载', sim.reset)
        for i, angle in enumerate([0, 10, 20, 25]):
            button(20 + i * 97, 140, 87, f'后倾 {angle}°', lambda a=angle: sim.request(pitch=a))
        slider(183, '机身后倾（抬头）', sim.target.pitch, 0, 40, lambda x: sim.request(pitch=round(x * 2) / 2), '{:.1f}°')
        slider(239, '机身高度', sim.target.height * 100, 30, 65,
               lambda x: sim.request(height=round(x, 1) / 100), '{:.1f} cm')
        slider(295, '机身前后平移（前为正）', sim.target.x * 100, -15, 15,
               lambda x: sim.request(x=round(x, 1) / 100), '{:+.1f} cm')
        button(20, 350, 379, ('启用 · ' if sim.auto else '关闭 · ') + '后倾时自动匹配高度与前后位置',
               lambda: setattr(sim, 'auto', not sim.auto), sim.auto)
        button(20, 389, 379, ('启用 · ' if sim.require_rim else '关闭 · ') + '出口 ≥ 桶沿 55 cm + 余量 2 cm',
               self.toggle_rim, sim.require_rim)
        slider(433, '接触摩擦（敏感性假设）', sim.mu, .05, 1.0, lambda x: sim.set_friction(round(x, 2)), 'μ = {:.2f}')
        text_at(489, '装载方案（切换将回正、重新装载）', 14, muted)
        for i, label in enumerate(SCENARIO_LABELS):
            button(20 + (i % 2) * 194, 514 + (i // 2) * 38, 185, label,
                   lambda idx=i: self.select_scenario(idx), sim.scenario == i)
        button(20, 596, 119, '整机视角', lambda: self.view('whole'))
        button(149, 596, 119, '篮子特写', lambda: self.view('tail'))
        button(278, 596, 121, '侧面观察', lambda: self.view('side'))
        button(20, 635, 185, '透明篮子' if self.transparent else '不透明篮子',
               lambda: setattr(self, 'transparent', not self.transparent), self.transparent)
        button(214, 635, 185, '显示碰撞体' if not self.collision else '恢复外观',
               lambda: setattr(self, 'collision', not self.collision), self.collision)
        rows = sim.metrics()
        text_at(681, f'实际后倾 {sim.pose.pitch:4.1f}°    仿真 {sim.data.time:.1f} s    尾部排出 {sum(sim.escaped)}/3', 16)
        text_at(707, '  /  '.join(f'{i + 1}: ' + ('已排出' if sim.escaped[i] else '篮内' if r['below_rim'] else '在沿外')
                                for i, r in enumerate(rows)), 14, blue)
        text_at(732, f'关节最小余量 {sim.pose.margin:.2f}°    后篮沿高 {sim.pose.lip * 100:.1f} cm', 14)
        text_at(755, f'质心投影余量* {sim.pose.support * 100:.1f} cm    四足位置误差 < 0.01 mm', 13, muted)
        text_at(777, '* 按 1.5 kg 固定装载质心估算；未解支撑力', 13, muted)
        message = sim.message
        color = '#ffad9c' if '拒绝' in message or '中断' in message else '#9ed8ad'
        # Fixed-width wrapping works well for the Chinese font and avoids clipping.
        for i in range(0, len(message), 25):
            text_at(806 + i // 25 * 19, message[i:i + 25], 14, color)
        button(20, 857, 379, '保存当前画面与参数（JSON + PNG）', self.save)
        text_at(902, '鼠标拖转 / 滚轮缩放 · 空格暂停 · R 回正 · S 保存', 13, muted)
        self.panel = np.ascontiguousarray(np.asarray(im)[::-1])
        self.last_panel = time.monotonic()

    def cursor(self, x, y):
        w, h = glfw.get_window_size(self.window)
        return x * self.WIDTH / max(w, 1), y * self.HEIGHT / max(h, 1)

    def mouse_button(self, window, button, action, mods):
        x, y = self.cursor(*glfw.get_cursor_pos(window))
        if action == glfw.RELEASE:
            self.drag_slider = self.drag_camera = None
            return
        if button == glfw.MOUSE_BUTTON_LEFT and x < self.PANEL:
            self.panel_press(x, y)
            return
        if x >= self.PANEL:
            self.drag_camera = button
            self.last_cursor = (x, y)

    def panel_press(self, x, y):
        for rect, callback in self.rects:
            if rect[0] <= x <= rect[2] and rect[1] <= y <= rect[3]:
                callback()
                self.last_panel = 0
                return
        for slider in self.sliders:
            rect = slider[0]
            if rect[0] <= x <= rect[2] and rect[1] <= y <= rect[3]:
                self.drag_slider = slider
                self.update_slider(x)
                return

    def update_slider(self, x):
        _, low, high, callback = self.drag_slider
        fraction = np.clip((x - 24) / (self.PANEL - 49), 0, 1)
        callback(float(low + fraction * (high - low)))
        self.last_panel = 0

    def mouse_move(self, window, x, y):
        x, y = self.cursor(x, y)
        if self.drag_slider:
            self.update_slider(x)
        elif self.drag_camera is not None:
            oldx, oldy = self.last_cursor
            mode = mujoco.mjtMouse.mjMOUSE_ROTATE_V if self.drag_camera == glfw.MOUSE_BUTTON_LEFT else mujoco.mjtMouse.mjMOUSE_MOVE_V
            mujoco.mjv_moveCamera(self.sim.model, mode, (x - oldx) / self.HEIGHT,
                                 (y - oldy) / self.HEIGHT, self.camera)
            self.last_cursor = (x, y)

    def scroll(self, window, x, y):
        mujoco.mjv_moveCamera(self.sim.model, mujoco.mjtMouse.mjMOUSE_ZOOM, 0, -.05 * y, self.camera)

    def key(self, window, key, scancode, action, mods):
        if action not in (glfw.PRESS, glfw.REPEAT):
            return
        if key == glfw.KEY_SPACE:
            self.sim.paused = not self.sim.paused
        elif key == glfw.KEY_R:
            self.sim.home()
        elif key == glfw.KEY_S:
            self.save()
        elif key in (glfw.KEY_UP, glfw.KEY_DOWN):
            self.sim.request(pitch=np.clip(self.sim.target.pitch + (1 if key == glfw.KEY_UP else -1), 0, 40))
        elif key == glfw.KEY_ESCAPE:
            glfw.set_window_should_close(window, True)
        self.last_panel = 0

    def render(self):
        self.refresh_model()
        width, height = glfw.get_framebuffer_size(self.window)
        panel_width = round(width * self.PANEL / self.WIDTH)
        viewport = mujoco.MjrRect(panel_width, 0, width - panel_width, height)
        mujoco.mjv_updateScene(self.sim.model, self.sim.data, self.options, None,
                              self.camera, mujoco.mjtCatBit.mjCAT_ALL, self.scene)
        if hasattr(self, 'add_scene_overlay'):
            self.add_scene_overlay()
        # A visual marker, not an obstacle or a relocated target bucket.
        marker = self.scene.geoms[self.scene.ngeom]
        mujoco.mjv_initGeom(marker, mujoco.mjtGeom.mjGEOM_CAPSULE, np.zeros(3), np.zeros(3), np.eye(3).ravel(),
                           np.array([.4, .9, .95, .8], dtype=np.float32))
        mujoco.mjv_connector(marker, mujoco.mjtGeom.mjGEOM_CAPSULE, .002, [-.8, -.18, .55], [-.45, -.18, .55])
        self.scene.ngeom += 1
        mujoco.mjr_render(viewport, self.scene, self.context)
        mujoco.mjr_overlay(mujoco.mjtFontScale.mjFONTSCALE_100, mujoco.mjtGridPos.mjGRID_TOPLEFT,
                           viewport, getattr(self.sim, 'overlay', 'PRESCRIBED POSE + CARGO PHYSICS\nNot a motor / balance / score test'),
                           'Cyan line: bucket rim height reference (0.55 m)', self.context)
        if self.panel is None or time.monotonic() - self.last_panel > .15:
            self.make_panel()
        pixel = np.asarray(Image.fromarray(self.panel[::-1]).resize((panel_width, height)))[::-1].copy()
        mujoco.mjr_drawPixels(pixel.ravel(), None, mujoco.MjrRect(0, 0, panel_width, height), self.context)

    def save(self, path=None):
        OUT.mkdir(exist_ok=True)
        stem = Path(path) if path else OUT / datetime.now().strftime('%Y%m%d_%H%M%S_%f')
        self.render()
        width, height = glfw.get_framebuffer_size(self.window)
        rgb = np.empty((height, width, 3), dtype=np.uint8)
        mujoco.mjr_readPixels(rgb.ravel(), None, mujoco.MjrRect(0, 0, width, height), self.context)
        Image.fromarray(rgb[::-1]).save(stem.with_suffix('.png'))
        stem.with_suffix('.json').write_text(json.dumps(self.sim.state(), ensure_ascii=False, indent=2) + '\n')
        self.sim.message = '已保存：' + stem.name
        self.last_panel = 0
        print('SAVED', stem, flush=True)
        return stem

    def run(self):
        print('VIEWER_READY', flush=True)
        previous = time.monotonic()
        budget = 0.
        while not glfw.window_should_close(self.window):
            now = time.monotonic()
            budget += min(now - previous, .05) * self.sim.speed
            previous = now
            # Keep UI responsive even if physics is slower than real time.
            n = min(int(budget / DT), 35)
            for _ in range(n):
                self.sim.step()
            budget = min(budget - n * DT, .05)
            self.render()
            glfw.swap_buffers(self.window)
            glfw.poll_events()
            time.sleep(.001)

    def close(self):
        self.context.free()
        glfw.destroy_window(self.window)
        glfw.terminate()


def self_test():
    """Actual compiler/contact/render checks, not merely syntax or mocked GUI."""
    sim = Diagnostic()
    assert sim.model.nmocap == 30 and sim.model.nq == 21
    assert sum(n.startswith('collision_tail_basket_') for n in
               [sim.model.geom(i).name for i in range(sim.model.ngeom)]) == 16
    assert abs(sim.model.body('tail_basket').mass[0] - .2) < 1e-10
    assert len(sim.pose.q) == 19
    assert sim.limits.solve(0, 0, .3)[0] is None
    window = Window(sim, visible=False)
    results = []
    try:
        distance = window.camera.distance
        window.scroll(window.window, 0, 1)
        assert window.camera.distance != distance
        azimuth = window.camera.azimuth
        window.drag_camera = glfw.MOUSE_BUTTON_LEFT
        window.last_cursor = (800, 500)
        window.mouse_move(window.window, 820, 510)
        assert window.camera.azimuth != azimuth
        window.drag_camera = None
        window.view('tail')
        for _ in range(2000):
            sim.step()
        assert all(r['below_rim'] for r in sim.metrics())
        window.save(OUT / 'verified_initial')
        for angle in [10, 20, 25]:
            assert sim.request(pitch=angle)
            for _ in range(5000):
                sim.step()
            assert abs(sim.pose.pitch - angle) < 1e-6, sim.message
            assert sim.pose.margin >= 2 - 1e-6
            assert sim.pose.lip >= .57 - 1e-6
            assert np.all(sim.pose.q >= np.array(JOINT_LOWER) - 1e-10)
            assert np.all(sim.pose.q <= np.array(JOINT_UPPER) + 1e-10)
            results.append(sim.state())
        for _ in range(3000):
            sim.step()
        window.save(OUT / 'verified_25deg')
        before = sim.target
        assert not sim.request(pitch=35)
        assert sim.target is before
        # Exercise real button hit-testing: pause, home, and select three bananas.
        window.make_panel()
        window.panel_press(45, 112)
        assert sim.paused
        sim.paused = False
        for index in range(1, 4):
            window.select_scenario(index)
            for _ in range(2000):
                sim.step()
            assert all(abs(sim.model.body_mass[bid] - .5) < 1e-10 for bid in sim.cargo_ids)
            assert all(r['below_rim'] for r in sim.metrics()), (index, sim.metrics())
            window.render()
        OUT.mkdir(exist_ok=True)
        (OUT / 'verification.json').write_text(json.dumps(dict(passed=True, pose_cases=results,
            checks=['30 prescribed URDF links, 16 basket colliders, 19 bounded joint targets',
                    '0.2 kg basket and 3 x 0.5 kg dynamic convex-hull cargo',
                    '0 -> 10 -> 20 -> 25 degree path checked at every 1 ms step',
                    '35 degree request rejected; prior valid target preserved',
                    'all four cargo scenarios settle inside the basket at home',
                    'native OpenGL framebuffer rendered and captured; pause button hit-test passed',
                    'scroll zoom and camera-drag callbacks change the camera without errors']),
                    ensure_ascii=False, indent=2) + '\n')
        print('SELF_TEST_PASSED', flush=True)
    finally:
        window.close()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--self-test', action='store_true')
    args = parser.parse_args()
    if args.self_test:
        self_test()
        return
    window = Window(Diagnostic())
    try:
        window.run()
    finally:
        window.close()


if __name__ == '__main__':
    main()
