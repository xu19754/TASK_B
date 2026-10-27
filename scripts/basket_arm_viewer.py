"""Part-2 basket + torque-limited dynamic Z1 on a prescribed B2 carrier."""
from __future__ import annotations

from pathlib import Path
import argparse
import json
import math
import os
import sys
import time
import xml.etree.ElementTree as ET

import glfw
import mujoco
import numpy as np
from PIL import Image, ImageDraw
from scipy.optimize import least_squares
from basket_trajectory import JointTrajectory

import basket_viewer as base

ROOT = base.ROOT
REVISION = os.environ.get('TASKB_BASKET_REVISION', 'v2')
if not REVISION.isalnum():
    raise ValueError('TASKB_BASKET_REVISION must be alphanumeric')
OUT = ROOT / 'reports' / ('basket_' + REVISION)
VARIANT = ROOT / 'assets/b2z1' / ('b2z1_0815_with_basket_' + REVISION + '.urdf')
MODEL_SPEC = json.loads((OUT / 'geometry.json').read_text()) if (OUT / 'geometry.json').exists() else {}
MODEL_DIMS = MODEL_SPEC.get('dimensions_m', [.45, .25, .20])
RIM_POINT = np.array([MODEL_SPEC.get('bounds_m', [[-.225, -.125, 0]])[0][0],
                      0., MODEL_SPEC.get('rear_sill_m', .015)])
base.URDF_PATH = VARIANT
base.geometry.URDF_PATH = VARIANT
base.geometry.ROBOT = ET.parse(VARIANT).getroot()
base.OUT = OUT / 'interactive_runs'
ARM_NAMES = base.JOINT_NAMES[12:]
ARM_LINKS = ['link00', 'link01', 'link02', 'link03', 'link04', 'link05', 'link06', 'gripperStator', 'gripperMover']
LOWER, UPPER = np.array(base.JOINT_LOWER[12:]), np.array(base.JOINT_UPPER[12:])
# robot.py is the runtime authority for the gripper: stricter than its URDF.
EFFORT = np.array([30., 60., 30., 30., 30., 30., 5.])
VELOCITY = np.array([3., 3., 3., 3., 3., 3., .25])
KP = np.array([512., 768., 768., 512., 384., 256., 80.])
KD = np.array([25.6] * 6 + [8.])
TIP = np.array([.18, 0, -.02])  # inspection site on the supplied fixed finger
COLORS = [[.92, .29, .12, 1], [.95, .70, .10, 1], [.92, .88, .20, 1]]


class Limits(base.PostureLimits):
    def __init__(self, arm_provider):
        super().__init__()
        self.arm_provider = arm_provider
        # Only legs are prescribed here. Dynamic arm speeds are checked from
        # qvel, not differences between cached IK snapshots of arm angles.
        self.velocity[12:] = np.inf

    def solve(self, pitch, x, height, require_rim=True):
        r = base.geometry.rotation([0, 1, 0], -math.radians(pitch))
        q, valid, margins = base.geometry.inverse_batch(((self.feet - [x, 0, height]) @ r)[None], self.params)
        if not valid[0]:
            return None, '拒绝：腿部不可达或超出 URDF 限位'
        if margins[0] < math.radians(2) - 1e-10:
            return None, '拒绝：腿部关节余量不足 2°'
        full = np.r_[q[0], self.arm_provider()]
        links = base.geometry.forward(full, x, height, math.radians(pitch))
        ground = min(float((v @ links[n][:3, :3].T + links[n][:3, 3])[:, 2].min())
                     for n, v in self.collision.items())
        if ground < .01:
            return None, '拒绝：机身 / 髋 / 大腿离地不足 1 cm'
        t = links['tail_basket']
        lip = float((t[:3, :3] @ RIM_POINT + t[:3, 3])[2])
        if require_rim and lip < .57 - 1e-10:
            return None, '拒绝：新版低尾沿低于桶沿 55 cm + 2 cm'
        positions = [links[n][:3, :3] @ p + links[n][:3, 3] for n, p, _ in self.inertials]
        masses = [m for _, _, m in self.inertials]
        positions.append(t[:3, :3] @ [-.1, 0, .075] + t[:3, 3])
        masses.append(1.5)
        com = np.average(positions, axis=0, weights=masses)
        support = float(np.min(-(self.hull[:, :2] @ com[:2] + self.hull[:, 2]) /
                               np.linalg.norm(self.hull[:, :2], axis=1)))
        if support < .02:
            return None, '拒绝：估算质心离支撑边界不足 2 cm'
        error = max(np.linalg.norm(links[f'{n}_foot'][:3, 3] - self.feet[i]) for i, n in enumerate(base.geometry.LEGS))
        return base.Pose(pitch, x, height, full, links, math.degrees(margins[0]), lip, ground, support, float(error)), ''

    def search(self, pitch, require_rim=True):
        xx, zz = np.meshgrid(np.linspace(-.15, .15, 121), np.linspace(.30, .65, 141))
        tr = np.c_[xx.ravel(), np.zeros(xx.size), zz.ravel()]
        theta = math.radians(pitch)
        r = base.geometry.rotation([0, 1, 0], -theta)
        _, valid, margins = base.geometry.inverse_batch((self.feet[None] - tr[:, None]) @ r, self.params)
        valid &= margins >= math.radians(2)
        if require_rim:
            valid &= tr[:, 2] - .525 * math.sin(theta) + .145 * math.cos(theta) >= .57 - 1e-10
        candidates = np.flatnonzero(valid)
        score = (tr[candidates, 0] / .15) ** 2 + ((tr[candidates, 2] - self.height) / .15) ** 2
        for idx in candidates[np.argsort(score)]:
            p, _ = self.solve(pitch, tr[idx, 0], tr[idx, 2], require_rim)
            if p:
                return p, ''
        return None, f'拒绝：{pitch:.1f}° 未找到满足固定四足与出口条件的姿态'


def build_scene():
    _, objects, xml = base.build_scene()
    root = ET.fromstring(xml)
    root.find('option').set('integrator', 'implicitfast')
    world = root.find('worldbody')
    bodies = {b.get('name'): b for b in world.findall('body')}
    joints = {j.find('child').get('link'): j for j in base.geometry.ROBOT.findall('joint')}
    for name in ARM_LINKS:
        b = bodies[name]
        world.remove(b)
        b.attrib.pop('mocap', None)
        j = joints[name]
        t = base.geometry.origin(j.find('origin'))
        b.set('pos', base.numbers(t[:3, 3]))
        b.set('quat', base.numbers(base.quat_wxyz(t[:3, :3])))
        parent = j.find('parent').get('link')
        bodies[parent].append(b)
        if j.get('type') != 'fixed':
            i = ARM_NAMES.index(j.get('name'))
            ET.SubElement(b, 'joint', name=j.get('name'), type='hinge', axis=j.find('axis').get('xyz'),
                          limited='true', range=base.numbers([LOWER[i], UPPER[i]]), margin='0' if i == 6 else '.015',
                          solreflimit='.002 1', solimplimit='.99 .999 .0001')
        for g in b.findall('geom'):
            if g.get('name', '').startswith('collision_'):
                g.set('contype', '4')
                g.set('conaffinity', '3')
    ET.SubElement(bodies['gripperStator'], 'site', name='finger_tip', pos=base.numbers(TIP),
                  size='.008', rgba='.95 .25 .8 1', group='0')
    actuator = ET.SubElement(root, 'actuator')
    for i, (name, cap) in enumerate(zip(ARM_NAMES, EFFORT)):
        ET.SubElement(actuator, 'position', name='motor_' + name, joint=name, gear='1',
                      kp=str(KP[i]), kv=str(KD[i]),
                      ctrllimited='true', ctrlrange=base.numbers([LOWER[i], UPPER[i]]),
                      forcelimited='true', forcerange=base.numbers([-cap, cap]))
    for g in world.findall('.//geom'):
        if g.get('name', '').startswith('cargo_geom_'):
            g.set('conaffinity', '7')
        elif g.get('name') == 'floor':
            g.set('conaffinity', '6')
    xml = ET.tostring(root, encoding='unicode')
    return mujoco.MjModel.from_xml_string(xml), objects, xml


class ArmDiagnostic(base.Diagnostic):
    overlay = 'PART 2 | DYNAMIC Z1 + CARGO\nB2 carrier prescribed; full-robot balance unverified'
    def __init__(self):
        self.arm_target = np.array(base.ARM_Q0, dtype=float)
        self.arm_command = self.arm_target.copy()
        self.arm_rate = .20
        self.gravity_compensation = False
        self.smooth_motion = None
        self.smooth_elapsed = 0.
        self.smooth_peak = .04
        self.smooth_complete = False
        self.last_smooth_plan = None
        self.sequence = []
        self.recorded = []
        self.stage_started = 0.
        self.max_speed = np.zeros(7)
        self.max_torque = np.zeros(7)
        self.violation = None
        self.rejected_step = None
        self.limits = Limits(self.actual_arm)
        self.model, self.objects, self.xml = build_scene()
        self.data = mujoco.MjData(self.model)
        self.scenario, self.mu = 0, .5
        self.require_rim, self.auto, self.paused, self.speed = True, True, False, 1.
        self.message = '新版篮子已加载；机械臂受力矩限制，机身为规定姿态'
        self.pose, _ = self.limits.solve(0, 0, self.limits.height)
        self.target, self.waypoints = self.pose, []
        self.assets = base.OBJECT_NAMES.copy()
        self.escaped = [False] * 3
        self.setup_cargo(0)

    def bind_arm(self):
        self.arm_qadr = np.array([self.model.jnt_qposadr[self.model.joint(n).id] for n in ARM_NAMES])
        self.arm_dadr = np.array([self.model.jnt_dofadr[self.model.joint(n).id] for n in ARM_NAMES])
        self.arm_motor = np.array([self.model.actuator('motor_' + n).id for n in ARM_NAMES])

    def actual_arm(self):
        if hasattr(self, 'arm_qadr') and hasattr(self, 'data'):
            return self.data.qpos[self.arm_qadr].copy()
        return self.arm_command.copy()

    def impose(self, pose):
        for n, t in pose.links.items():
            mid = self.mocap[n]
            if mid >= 0:
                self.data.mocap_pos[mid] = t[:3, 3]
                self.data.mocap_quat[mid] = base.quat_wxyz(t[:3, :3])

    def reset(self):
        self.bind_arm()
        super().reset()
        self.data.qpos[self.arm_qadr] = base.ARM_Q0
        self.arm_target = np.array(base.ARM_Q0, dtype=float)
        self.arm_command = self.arm_target.copy()
        self.sequence = []
        self.smooth_motion = None
        self.smooth_elapsed = 0.
        self.smooth_complete = False
        self.max_speed[:], self.max_torque[:] = 0, 0
        self.violation = None
        self.rejected_step = None
        self.paused = False
        self.state_spec = mujoco.mjtState.mjSTATE_INTEGRATION
        self.state_buffer = np.empty(mujoco.mj_stateSize(self.model, self.state_spec))
        mujoco.mj_forward(self.model, self.data)

    def request_arm(self, values):
        values = np.array(values, dtype=float)
        if values.shape != (7,) or not np.all(np.isfinite(values)) or np.any(values < LOWER) or np.any(values > UPPER):
            self.message = '拒绝：机械臂目标超出原关节范围'
            return False
        self.arm_target = values
        self.message = '机械臂缓慢跟踪目标；接触时可能受阻，实际角度以反馈为准'
        return True

    def arm_joint(self, i, value):
        values = self.arm_target.copy()
        values[i] = value
        self.sequence = []
        self.smooth_motion = None
        return self.request_arm(values)

    def tip_local(self):
        t = self.pose.links['tail_basket']
        return (self.data.site('finger_tip').xpos - t[:3, 3]) @ t[:3, :3]

    def inverse_arm(self, target, seed=None):
        """Geometric candidate only; contact/torque limits still act during motion."""
        t = self.pose.links['tail_basket']
        wanted = t[:3, :3] @ target + t[:3, 3]
        direction = t[:3, :3] @ [-1, 0, 0]
        up = t[:3, :3] @ [0, 0, -1]
        leg = self.pose.q[:12]

        chain = []
        for name in ARM_LINKS[:-1]:
            joint = next(j for j in base.geometry.ROBOT.findall('joint') if j.find('child').get('link') == name)
            chain.append((base.geometry.origin(joint.find('origin')), None if joint.get('type') == 'fixed' else
                          (ARM_NAMES.index(joint.get('name')), np.fromstring(joint.find('axis').get('xyz'), sep=' '))))

        def fk(q):
            transform = self.pose.links['base'].copy()
            for origin, move in chain:
                transform = transform @ origin
                if move:
                    motion = np.eye(4)
                    motion[:3, :3] = base.geometry.rotation(move[1], q[move[0]])
                    transform = transform @ motion
            return transform

        def residual(q):
            g = fk(q)
            point = g[:3, :3] @ TIP + g[:3, 3]
            return np.r_[(point - wanted), .015 * (g[:3, 0] - direction), .005 * (g[:3, 2] - up)]

        seeds = [self.actual_arm()[:6] if seed is None else np.array(seed)[:6],
                 [0, .1, -3.2, 0, 0, 0], [0, .8, -2.8, -1, 0, 0],
                 [0, 2.6, -3.1, -1., 0, 0], [0, 2.8, -2, -1.4, 0, 0],
                 [.8, 1.5, -3.9, .3, -.5, 0], [-.8, 1.5, -3.9, .3, .5, 0]]
        best = None
        query = mujoco.MjData(self.model)
        query.qpos[:] = self.data.qpos
        query.mocap_pos[:] = self.data.mocap_pos
        query.mocap_quat[:] = self.data.mocap_quat
        for q0 in seeds:
            result = least_squares(residual, np.clip(q0, LOWER[:6] + .06, UPPER[:6] - .06),
                                   bounds=(LOWER[:6] + .06, UPPER[:6] - .06), max_nfev=130)
            error = np.linalg.norm(residual(result.x)[:3])
            q = np.r_[result.x, -.04]
            query.qpos[self.arm_qadr] = q
            mujoco.mj_forward(self.model, query)
            depth = max([max(0., -c.dist) for c in query.contact if set(self.model.geom_contype[c.geom]) == {1, 4}] or [0.])
            score = error + 8 * depth + .00005 * np.linalg.norm(q - self.actual_arm())
            if best is None or score < best[2]:
                best = (error, q, score, depth)
        if best[0] > .012 or best[3] > .002:
            return None, best[0]
        return best[1], best[0]

    def point_arm(self, point):
        self.smooth_motion = None
        q, error = self.inverse_arm(np.array(point))
        if q is None:
            self.message = f'拒绝：此末端点未找到解，位置误差 {error * 1000:.1f} mm'
            return False
        self.sequence = []
        return self.request_arm(q)

    def reset_at_window(self):
        q, error = self.inverse_arm(np.array([.19, 0, .22]))
        if q is None:
            self.message = '窗口前诊断初态未找到无穿透候选'
            return False
        self.reset()
        # Explicit test initialization, never used during an action trajectory.
        self.data.qpos[self.arm_qadr] = q
        self.arm_target, self.arm_command = q.copy(), q.copy()
        mujoco.mj_forward(self.model, self.data)
        self.message = '已重置窗口前诊断初态；这不代表已完成从收臂到窗口的动作'
        return True

    def record(self):
        self.recorded.append(self.arm_target.copy())
        self.message = f'已记录第 {len(self.recorded)} 个关节目标'

    def play(self, frames=None):
        self.smooth_motion = None
        frames = self.recorded if frames is None else frames
        if not frames:
            self.message = '先用关节滑块调整，再记录至少一个动作点'
            return False
        self.sequence = [np.array(x, dtype=float).copy() for x in frames]
        self.request_arm(self.sequence.pop(0))
        self.stage_started = self.data.time
        return True

    def stop(self):
        self.sequence = []
        self.smooth_motion = None
        self.arm_target = self.actual_arm()
        self.arm_command = self.arm_target.copy()
        self.message = '已停止回放，机械臂保持当前位置'

    def record_actual(self):
        self.recorded.append(self.actual_arm())
        self.message = f'已记录实际关节姿态 #{len(self.recorded)}，用于平滑连贯回放'

    def request_sweep_body(self):
        self.stop()
        auto = self.auto
        self.auto = False
        ok = self.request(pitch=15, x=.10, height=.575)
        self.auto = auto
        return ok

    def candidate_sweep(self):
        import hashlib
        folder = OUT / 'sweep_15deg_forward10cm'
        path = folder / 'candidate_default_path.json'
        if not path.exists():
            path = folder / 'candidate_path.json'
        candidate = json.loads(path.read_text())
        if candidate.get('urdf_sha256') != hashlib.sha256(VARIANT.read_bytes()).hexdigest():
            raise ValueError('扫篮轨迹与当前 URDF 不一致；请重新分析生成')
        return candidate

    def initialize_sweep(self):
        try:
            candidate = self.candidate_sweep()
        except (OSError, ValueError) as e:
            self.message = str(e)
            return False
        pose, reason = self.limits.solve(15, .10, .575, self.require_rim)
        if pose is None:
            self.message = reason
            return False
        self.pose = self.target = pose
        self.reset()
        q = np.asarray(candidate['q_nodes_rad'][0])
        self.data.qpos[self.arm_qadr] = q
        self.arm_target, self.arm_command = q.copy(), q.copy()
        self.gravity_compensation = True
        mujoco.mj_forward(self.model, self.data)
        self.message = '已设扫篮诊断初态：15° / 前移10 cm / 高57.5 cm；未验证入篮路径'
        return True

    def begin_smooth(self, nodes, parameters=None):
        try:
            trajectory = JointTrajectory(nodes, LOWER, UPPER, VELOCITY, self.smooth_peak, parameters)
            if np.max(np.abs(self.actual_arm() - trajectory.nodes[0])) > .08:
                self.message = '先缓慢回到首点或设置诊断初态，再播放平滑动作'
                return False
            from basket_sweep import Kinematics
            query = Kinematics(self)
            for q in trajectory.spline(np.linspace(0, 1, 201)):
                distance, _ = query.clearance(q)
                if distance < -.0005:
                    self.message = f'拒绝：平滑路径会与机身或篮子穿透 {abs(distance) * 1000:.1f} mm'
                    return False
        except (ValueError, OSError) as e:
            self.message = str(e)
            return False
        self.sequence = []
        self.smooth_motion = trajectory
        self.last_smooth_plan = trajectory.details()
        self.smooth_elapsed = 0.
        self.smooth_complete = False
        self.arm_command = trajectory.at(0).copy()
        self.message = f'开始共同时间轴的平滑动作，预计 {trajectory.duration:.1f} s；保持动力学与限力矩'
        return True

    def start_candidate_sweep(self):
        if abs(self.pose.pitch - 15) > .01 or abs(self.pose.x - .1) > .001 or abs(self.pose.height - .575) > .001:
            self.message = '请先设置 15° / 前移10 cm / 高57.5 cm 的扫篮姿态'
            return False
        try:
            candidate = self.candidate_sweep()
        except (OSError, ValueError) as e:
            self.message = str(e)
            return False
        return self.begin_smooth(candidate['q_nodes_rad'], candidate['parameters'])

    def start_recorded_smooth(self):
        return self.begin_smooth(self.recorded)

    def return_to_first(self):
        if not self.recorded:
            self.message = '请先记录实际姿态'
            return False
        if np.max(np.abs(self.actual_arm() - self.recorded[0])) < .02:
            self.message = '已在记录首点附近，可以播放平滑轨迹'
            return True
        return self.begin_smooth([self.actual_arm(), self.recorded[0]])

    def demo(self):
        # A local contact experiment, explicitly separate from an entry planner.
        if np.linalg.norm(self.tip_local() - [.19, 0, .22]) > .20:
            self.message = '先点窗口前初态重置，再试局部推料；完整入篮路径尚未验证'
            return False
        self.arm_rate = min(self.arm_rate, .10)
        frames = []
        seed = None
        for point in [[.10, 0, .17], [0., 0, .12], [-.10, 0, .095], [-.20, 0, .095]]:
            q, _ = self.inverse_arm(np.array(point), seed)
            if q is None:
                self.message = '推料试验候选点不可达；可用滑块逐关节调整'
                return False
            seed = q
            frames.append(q)
        self.play(frames)
        return True

    def step(self):
        if self.paused:
            return
        if self.violation:
            self.paused = True
            return
        mujoco.mj_getState(self.model, self.data, self.state_buffer, self.state_spec)
        previous_pose = self.pose
        previous_command = self.arm_command.copy()
        previous_elapsed = self.smooth_elapsed
        if self.smooth_motion is not None:
            self.arm_target = self.smooth_motion.at(self.smooth_elapsed)
        actual = self.actual_arm()
        velocity = self.data.qvel[self.arm_dadr]
        rate = np.minimum(VELOCITY * .75, np.r_[[self.arm_rate] * 6, .15])
        self.arm_command += np.clip(self.arm_target - self.arm_command, -rate * base.DT, rate * base.DT)
        desired = self.arm_command.copy()
        if self.gravity_compensation:
            # Equivalent PD + model bias compensation, inside the SAME capped
            # actuator. No external cargo/body forces or raised motor limits.
            desired += self.data.qfrc_bias[self.arm_dadr] / KP
        self.data.ctrl[self.arm_motor] = np.clip(desired, LOWER, UPPER)
        super().step()
        actual = self.actual_arm()
        velocity = self.data.qvel[self.arm_dadr]
        torque = self.data.actuator_force[self.arm_motor]
        if np.any(actual < LOWER - 1e-4) or np.any(actual > UPPER + 1e-4):
            self.violation = '实际机械臂角度超限'
        elif np.any(np.abs(velocity) > VELOCITY + 1e-4):
            self.violation = '实际机械臂速度超限'
        elif np.any(np.abs(torque) > EFFORT + 1e-7):
            self.violation = '实际电机力矩超限'
        if self.violation:
            self.rejected_step = dict(reason=self.violation, time_s=float(self.data.time),
                                      q_rad=actual.tolist(), velocity_rad_s=velocity.tolist(), torque_Nm=torque.tolist())
            mujoco.mj_setState(self.model, self.data, self.state_buffer, self.state_spec)
            self.pose, self.target = previous_pose, previous_pose
            self.waypoints = []
            self.arm_command = previous_command
            self.smooth_elapsed = previous_elapsed
            mujoco.mj_forward(self.model, self.data)
            self.paused = True
            self.sequence = []
            self.message = '下一步将越界，已暂停于有效状态：' + self.violation + '；需重置'
            return
        self.max_speed = np.maximum(self.max_speed, np.abs(velocity))
        self.max_torque = np.maximum(self.max_torque, np.abs(torque))
        if self.smooth_motion is not None:
            self.smooth_elapsed += base.DT
            if self.smooth_elapsed >= self.smooth_motion.duration + 3:
                self.smooth_motion = None
                self.smooth_complete = True
                self.metrics()
                self.message = f'连贯动作结束；尾部排出 {sum(self.escaped)}/3，需检查残留与实际关节误差'
        if self.sequence and np.max(np.abs(actual - self.arm_target)) < .055:
            self.request_arm(self.sequence.pop(0))
            self.stage_started = self.data.time
        elif self.sequence and self.data.time - self.stage_started > 15:
            self.stop()
            self.message = '回放停止：15 秒未到达目标，检查碰撞阻挡与力矩余量'

    def metrics(self):
        rows = []
        for i, v in enumerate(self.cargo_local()):
            self.escaped[i] |= bool(v[:, 0].max() < -.230)
            rows.append(dict(asset=self.assets[i], below_rim=bool(v[:, 0].min() >= -.2235 - .003 and
                        v[:, 0].max() <= .2235 + .003 and np.abs(v[:, 1]).max() <= .1235 + .003 and
                        v[:, 2].max() <= MODEL_DIMS[2] + .003 and v[:, 2].min() >= -.0015),
                        rear_exit_observed=self.escaped[i], front_most_x_m=float(v[:, 0].max()),
                        rear_most_x_m=float(v[:, 0].min()), highest_z_m=float(v[:, 2].max())))
        return rows

    def state(self):
        d = super().state()
        if hasattr(self, 'edge'):
            d['edge_diagnostic'] = self.edge.metrics()
        d.update(scope='Part-2 CAD, prescribed B2 carrier + real dynamic 7-joint Z1 + free cargo; not full-robot balance',
                 q_rad=dict(zip(base.JOINT_NAMES, np.r_[self.pose.q[:12], self.actual_arm()].tolist())),
                 arm_target_rad=self.arm_target.tolist(), arm_velocity_rad_s=self.data.qvel[self.arm_dadr].tolist(),
                 arm_torque_Nm=self.data.actuator_force[self.arm_motor].tolist(), arm_peak_speed_rad_s=self.max_speed.tolist(),
                 arm_peak_torque_Nm=self.max_torque.tolist(), arm_velocity_limit_rad_s=VELOCITY.tolist(),
                 arm_effort_limit_Nm=EFFORT.tolist(), constraint_violation=self.violation,
                 rejected_next_step=self.rejected_step,
                 model_bias_compensation=self.gravity_compensation,
                 smooth_motion=None if self.smooth_motion is None else self.smooth_motion.details(),
                 smooth_elapsed_s=self.smooth_elapsed, smooth_complete=self.smooth_complete,
                 last_smooth_motion=self.last_smooth_plan,
                 tip_basket_m=self.tip_local().tolist(), recorded_action_targets_rad=[x.tolist() for x in self.recorded],
                 basket_colliders=sum(self.model.geom(i).name.startswith('collision_tail_basket_')
                                      for i in range(self.model.ngeom)),
                 basket_revision=REVISION, arm_dynamic_joints=7, robot_prescribed_links=21)
        d['assumptions'] = ['19 original controllable joint names preserved in the URDF; legs prescribed by bounded IK',
                            '7 arm joints use actual rigid-body dynamics, torque caps and measured speed/position monitors',
                            'gripper uses runtime 5 Nm / 0.25 rad/s, stricter than URDF 30 Nm / 3 rad/s',
                            'no cargo force, weld, attachment or teleport during arm motion; reset initializes only',
                            'arm vs basket/body collisions enabled as extra diagnostic guard; runtime self_collision is off',
                            'base receives no arm reaction feedback; feet, torque/power and full-robot stability unverified',
                            f"{d['basket_colliders']} convex basket partitions of CAD solid, source-object hull contacts, MuJoCo not PhysX",
                            '0.2 kg basket mass retained as an assumption; conservative COM uses fixed 1.5 kg payload',
                            'geometry-only end-effector presets can be blocked during dynamic execution']
        return d


class Window(base.Window):
    PANEL, WIDTH, HEIGHT = 480, 1600, 990

    def __init__(self, sim, visible=True):
        self.tab = 'arm'
        super().__init__(sim, visible)
        glfw.set_window_title(self.window, 'TaskB | 零件2：新版篮子与机械臂辅助卸料')
        self.view('whole')

    def view(self, name):
        super().view(name)
        if name == 'whole':
            self.camera.lookat[:] = [.02, 0, .69]
            self.camera.distance, self.camera.azimuth, self.camera.elevation = 1.9, 120, -22
        elif name == 'tail':
            self.camera.lookat[:] = [-.24, 0, .68]
            self.camera.distance, self.camera.azimuth, self.camera.elevation = 1.08, 135, -30
        elif name == 'side':
            self.camera.lookat[:] = [-.02, 0, .70]
            self.camera.distance, self.camera.azimuth, self.camera.elevation = 1.55, 90, -10

    def arm_home(self):
        self.sim.smooth_motion = None
        self.sim.sequence = []
        self.sim.request_arm(base.ARM_Q0)

    def add_scene_overlay(self):
        if self.tab == 'edge' and hasattr(self.sim, 'edge'):
            self.sim.edge.overlay(self)
            return
        if self.tab != 'sweep':
            return
        try:
            if not hasattr(self, '_sweep_curve'):
                self._sweep_curve = np.asarray(self.sim.candidate_sweep()['geometric_tool_nodes_m_and_pitch_rad'])[:, :3]
        except (ValueError, OSError):
            return
        t = self.sim.pose.links['tail_basket']
        points = self._sweep_curve @ t[:3, :3].T + t[:3, 3]
        for p, q in zip(points[:-1], points[1:]):
            geom = self.scene.geoms[self.scene.ngeom]
            mujoco.mjv_initGeom(geom, mujoco.mjtGeom.mjGEOM_CAPSULE, np.zeros(3), np.zeros(3),
                               np.eye(3).ravel(), np.array([.95, .32, .84, .9], dtype=np.float32))
            mujoco.mjv_connector(geom, mujoco.mjtGeom.mjGEOM_CAPSULE, .0015, p, q)
            self.scene.ngeom += 1

    def make_panel(self):
        im = Image.new('RGB', (self.PANEL, self.HEIGHT), '#101a29')
        draw = ImageDraw.Draw(im)
        self.rects, self.sliders = [], []
        white, muted, blue = '#eef4fd', '#a8b8cc', '#68b6ff'

        def txt(y, text, size=15, color=white, x=20):
            draw.text((x, y), text, font=self.fonts[size], fill=color)

        def button(x, y, w, label, action, selected=False):
            box = (x, y, x + w, y + 31)
            draw.rounded_rectangle(box, 6, fill='#245b91' if selected else '#24354b')
            draw.text((x + 8, y + 7), label, font=self.fonts[14], fill=white)
            self.rects.append((box, action))

        def slider(y, label, value, lo, hi, callback, fmt='{:.1f}'):
            txt(y, label, 14)
            txt(y, fmt.format(value), 14, blue, x=365)
            a, b, sy = 24, self.PANEL - 25, y + 27
            draw.line((a, sy, b, sy), fill='#3b4c65', width=5)
            vx = a + (b - a) * np.clip((value - lo) / (hi - lo), 0, 1)
            draw.line((a, sy, vx, sy), fill=blue, width=5)
            draw.ellipse((vx - 6, sy - 6, vx + 6, sy + 6), fill=white)
            self.sliders.append(((a - 9, sy - 10, b + 9, sy + 10), lo, hi, callback))

        s = self.sim
        txt(14, '零件 2 · 机械臂辅助卸料', 22)
        dimensions = ' × '.join(f'{x * 100:.0f}' for x in MODEL_DIMS)
        txt(46, f'{dimensions} cm · {REVISION} / 尾沿 {RIM_POINT[2] * 1000:.0f} mm', 15, blue)
        txt(70, '机械臂有真实动力学；机身规定姿态，整机平衡待验证', 13, '#f4c97b')
        if hasattr(s, 'edge'):
            for x, name, label in [(20, 'arm', '关节'), (132, 'body', '机身 / 装载'),
                                    (244, 'sweep', '原候选'), (356, 'edge', '铲扫姿态')]:
                button(x, 98, 102, label, lambda t=name: setattr(self, 'tab', t), self.tab == name)
        else:
            button(20, 98, 138, '机械臂关节', lambda: setattr(self, 'tab', 'arm'), self.tab == 'arm')
            button(168, 98, 138, '机身 / 装载', lambda: setattr(self, 'tab', 'body'), self.tab == 'body')
            button(316, 98, 142, '连贯扫篮', lambda: setattr(self, 'tab', 'sweep'), self.tab == 'sweep')
        button(20, 139, 90, '继续' if s.paused else '暂停', lambda: setattr(s, 'paused', not s.paused))
        button(120, 139, 164, '回正、收臂并重装', s.home)
        button(294, 139, 164, '当前倾角收臂重装', s.reset)
        if self.tab == 'edge':
            s.edge.draw_panel(txt, button, slider)
        elif self.tab == 'arm':
            actual = np.degrees(s.actual_arm())
            torques = s.data.actuator_force[s.arm_motor]
            speeds = s.data.qvel[s.arm_dadr]
            for i, name in enumerate(ARM_NAMES):
                label = f'{name}  {actual[i]:+.1f}°  v{speeds[i]:+.2f}  τ{torques[i]:+.1f}'
                slider(185 + i * 48, label, math.degrees(s.arm_target[i]), math.degrees(LOWER[i]),
                       math.degrees(UPPER[i]), lambda x, j=i: s.arm_joint(j, math.radians(x)), '{:+.1f}°')
            txt(528, '实际角度 / v(rad/s) / τ(Nm)；滑块是目标值', 13, muted)
            button(20, 552, 138, '缓慢收臂', self.arm_home)
            button(168, 552, 138, '窗口前初态重置', s.reset_at_window)
            button(316, 552, 142, '窗口内（候选）', lambda: s.point_arm([.10, 0, .17]))
            button(20, 592, 214, '窗口内局部推料试验', s.demo)
            button(244, 592, 214, '停止动作并保持', s.stop)
            button(20, 632, 138, f'记录目标 #{len(s.recorded) + 1}', s.record)
            button(168, 632, 138, f'回放 {len(s.recorded)} 个点', s.play)
            button(316, 632, 142, '清空记录', lambda: s.recorded.clear())
            slider(677, '动作目标变化速度（rad/s）', s.arm_rate, .05, .5,
                   lambda x: setattr(s, 'arm_rate', round(x, 2)), '{:.2f}')
            txt(719, '力矩按项目限幅；夹爪速度上限 0.25 rad/s', 13, muted)
        elif self.tab == 'sweep':
            txt(185, '后倾 15° / 前移 10 cm / 高度 57.5 cm', 18, blue)
            button(20, 219, 438, '仅将机身缓慢移动到这组姿态', s.request_sweep_body)
            button(20, 259, 438, '设置扫篮准备初态（诊断重置，非入篮运动）', s.initialize_sweep)
            button(20, 299, 214, '播放连贯候选扫篮', s.start_candidate_sweep)
            button(244, 299, 214, '停止并保持', s.stop)
            slider(345, '平滑轨迹目标峰值速度（rad/s）', s.smooth_peak, .02, .15,
                   lambda x: setattr(s, 'smooth_peak', round(x, 3)), '{:.3f}')
            button(20, 398, 438, ('启用 · ' if s.gravity_compensation else '关闭 · ') + '模型重力前馈（仍受原力矩上限）',
                   lambda: setattr(s, 'gravity_compensation', not s.gravity_compensation), s.gravity_compensation)
            txt(445, '把你手动试通的实际姿态记录下来，统一时间平滑回放', 13, muted)
            button(20, 471, 214, f'记录实际姿态 #{len(s.recorded) + 1}', s.record_actual)
            button(244, 471, 214, '缓慢回到记录首点', s.return_to_first)
            button(20, 511, 214, '平滑回放实际记录', s.start_recorded_smooth)
            button(244, 511, 214, '清空记录', lambda: s.recorded.clear())
            progress = 0 if s.smooth_motion is None else min(1., s.smooth_elapsed / s.smooth_motion.duration)
            if s.smooth_complete:
                progress = 1.
            txt(562, f'共同时间轴进度 {progress * 100:.1f}%   累计 {s.smooth_elapsed:.1f} s', 16)
            draw.rounded_rectangle((24, 591, 455, 600), 4, fill='#35465e')
            if progress > .001:
                draw.rounded_rectangle((24, 591, 24 + 431 * progress, 600), 4, fill='#c761c8')
            txt(617, '夹爪横向对齐，前臂后摆，腕部同步补偿', 15)
            txt(645, '紫线是候选扫动中心；实际动作仍受碰撞和力矩影响', 13, muted)
            txt(672, '已验证有物体排出；仍需检查残留、完整入篮与整机平衡', 13, '#f4c97b')
            txt(701, '调速度后重新播放生效；暂停后可继续同一条时间轴', 13, muted)
        else:
            for i, a in enumerate([0, 5, 10, 15]):
                button(20 + i * 111, 185, 103, f'后倾 {a}°', lambda angle=a: s.request(pitch=angle))
            slider(231, '机身后倾（抬头）', s.target.pitch, 0, 35,
                   lambda x: s.request(pitch=round(x * 2) / 2), '{:.1f}°')
            slider(281, '机身高度', s.target.height * 100, 30, 65,
                   lambda x: s.request(height=round(x, 1) / 100), '{:.1f} cm')
            slider(331, '机身前后平移', s.target.x * 100, -15, 15,
                   lambda x: s.request(x=round(x, 1) / 100), '{:+.1f} cm')
            button(20, 385, 438, ('启用 · ' if s.auto else '关闭 · ') + '后倾时自动匹配高度与前后位置',
                   lambda: setattr(s, 'auto', not s.auto), s.auto)
            button(20, 425, 438, ('启用 · ' if s.require_rim else '关闭 · ') + '低尾沿 ≥ 桶沿 55 cm + 余量 2 cm',
                   self.toggle_rim, s.require_rim)
            slider(474, '接触摩擦 μ（敏感性假设）', s.mu, .05, 1., lambda x: s.set_friction(round(x, 2)), '{:.2f}')
            for i, label in enumerate(base.SCENARIO_LABELS):
                button(20 + (i % 2) * 224, 531 + (i // 2) * 40, 214, label,
                       lambda n=i: self.select_scenario(n), s.scenario == i)
            txt(623, f'外形体积 {np.prod(MODEL_DIMS) * 1000:.1f} L，开放口不允许装满', 14, muted)
            txt(650, '低尾沿利于卸料，也可能增加行走时漏料风险', 14, '#f4c97b')
            txt(680, '四足固定；腿部限位余量 2°、离地与质心筛查', 13, muted)
        button(20, 734, 138, '整机视角', lambda: self.view('whole'))
        button(168, 734, 138, '篮子 / 窗口', lambda: self.view('tail'))
        button(316, 734, 142, '侧面观察', lambda: self.view('side'))
        button(20, 774, 214, '透明篮子' if self.transparent else '不透明篮子',
               lambda: setattr(self, 'transparent', not self.transparent), self.transparent)
        button(244, 774, 214, '显示碰撞体' if not self.collision else '恢复外观',
               lambda: setattr(self, 'collision', not self.collision), self.collision)
        s.metrics()
        txt(818, f'后倾 {s.pose.pitch:.1f}°  尾沿高 {s.pose.lip * 100:.1f} cm  尾部排出 {sum(s.escaped)}/3', 16)
        tip = s.tip_local() * 100
        txt(846, f'末端篮内坐标：X {tip[0]:+.1f}  Y {tip[1]:+.1f}  Z {tip[2]:+.1f} cm', 13, muted)
        message = s.message
        color = '#ffad9c' if any(t in message for t in ['拒绝', '触发', '停止', '中断']) else '#a2ddb2'
        for i in range(0, min(len(message), 58), 29):
            txt(873 + i // 29 * 19, message[i:i + 29], 14, color)
        button(20, 925, 438, '保存画面、动作点与关节约束记录（PNG + JSON）', self.save)
        txt(967, f'仿真 {s.data.time:.1f} s · 拖转 / 滚轮 · 空格暂停 · S 保存', 13, muted)
        self.panel = np.ascontiguousarray(np.asarray(im)[::-1])
        self.last_panel = time.monotonic()


def self_test_sweep():
    output = OUT / 'sweep_15deg_forward10cm'
    s = ArmDiagnostic()
    assert s.request_sweep_body()
    for _ in range(10000):
        s.step()
    assert abs(s.pose.pitch - 15) < 1e-7 and abs(s.pose.x - .1) < 1e-8, s.message
    assert not s.violation
    assert s.initialize_sweep()
    for _ in range(2000):
        s.step()
    assert not s.violation
    s.record_actual()
    assert np.allclose(s.recorded[-1], s.actual_arm())
    assert s.start_candidate_sweep(), s.message
    trajectory = s.smooth_motion
    assert np.max(np.abs(trajectory.spline(0, 1))) < 1e-9
    assert np.max(np.abs(trajectory.spline(1, 1))) < 1e-9
    assert np.all(trajectory.peak_velocity <= np.minimum(VELOCITY * .5, s.smooth_peak) + 1e-8)
    window = Window(s, visible=False)
    contacts = set()
    traces = []
    try:
        window.tab = 'sweep'
        window.view('tail')
        window.make_panel()
        window.save(output / 'smooth_start')
        # Pause leaves the common timeline and physical state unchanged.
        s.paused = True
        t, elapsed = s.data.time, s.smooth_elapsed
        s.step()
        assert s.data.time == t and s.smooth_elapsed == elapsed
        s.paused = False
        for step in range(math.ceil((trajectory.duration + 4) / base.DT)):
            s.step()
            for c in s.data.contact:
                if set(s.model.geom_contype[c.geom]) == {2, 4}:
                    contacts.add(tuple(s.model.geom(int(g)).name for g in c.geom))
            if step % 1000 == 0:
                traces.append(s.state())
            if step == int(trajectory.duration * .65 / base.DT):
                window.last_panel = 0
                window.save(output / 'smooth_contact')
            if s.violation:
                break
        assert not s.violation, s.rejected_step
        assert s.smooth_complete
        assert np.all(s.max_speed <= VELOCITY + 1e-4)
        assert np.all(s.max_torque <= EFFORT + 1e-7)
        assert len(contacts) > 0
        s.metrics()
        window.last_panel = 0
        window.save(output / 'smooth_finish')
        evidence = s.state()
        evidence.update(passed=True, trace=traces, observed_arm_cargo_pairs=sorted(contacts),
                        trajectory=trajectory.details(),
                        checks=['15deg/+10cm/57.5cm carrier reached with a bounded continuous body path',
                                'sweep arm start is an explicit diagnostic initialization, not a solved entry path',
                                'C2 joint interpolation, common timeline, sampled angle/speed/acceleration preflight',
                                'actual pose recording, pause preserves timing, native overlay renders',
                                'full continuous sweep executed with same torque caps, joint limits and speed guard'])
        (output / 'continuous_verification.json').write_text(json.dumps(evidence, indent=2, ensure_ascii=False) + '\n')
        print('SMOOTH_SWEEP_SELF_TEST_PASSED', 'rear exits', sum(s.escaped), flush=True)
    finally:
        window.close()


def self_test():
    s = ArmDiagnostic()
    assert (s.model.nmocap, s.model.nq, s.model.nu) == (21, 28, 7)
    assert sum(s.model.geom(i).name.startswith('collision_tail_basket_') for i in range(s.model.ngeom)) == MODEL_SPEC.get('collision_convex_parts', 30)
    assert abs(s.model.body('tail_basket').mass[0] - .2) < 1e-10
    for _ in range(2000):
        s.step()
    assert not s.violation and all(x['below_rim'] for x in s.metrics())
    q_before = s.actual_arm()
    target = s.arm_target.copy()
    target[0] = .15
    assert s.request_arm(target)
    for _ in range(3000):
        s.step()
    assert not s.violation and s.actual_arm()[0] - q_before[0] > .10
    assert np.all(s.max_torque <= EFFORT + 1e-7)
    assert np.all(s.max_speed <= VELOCITY + 1e-4)
    bad = s.arm_target.copy()
    bad[0] = UPPER[0] + .01
    assert not s.request_arm(bad)
    assert s.recorded == []
    s.record()
    assert s.play()
    s.stop()
    s.home()
    window = Window(s, visible=False)
    try:
        window.make_panel()
        window.panel_press(260, 110)
        assert window.tab == 'body'
        window.make_panel()
        distance = window.camera.distance
        window.scroll(window.window, 0, 1)
        assert window.camera.distance != distance
        window.drag_camera = glfw.MOUSE_BUTTON_LEFT
        window.last_cursor = (800, 500)
        window.mouse_move(window.window, 820, 510)
        window.drag_camera = None
        window.view('whole')
        for _ in range(2000):
            s.step()
        window.save(base.OUT / 'verified_initial')
        for angle in [5, 10, 15]:
            assert s.request(pitch=angle)
            for _ in range(5000):
                s.step()
            assert not s.violation, s.message
            assert abs(s.pose.pitch - angle) < 1e-6, s.message
        assert not s.request(pitch=20)
        window.view('tail')
        window.save(base.OUT / 'verified_tilt_15')
        s.home()
        assert s.reset_at_window()
        for _ in range(2000):
            s.step()
        assert not s.violation
        window.tab = 'arm'
        window.last_panel = 0
        window.save(base.OUT / 'verified_arm_window')
        s.arm_rate = .10
        assert s.demo()
        contacts = set()
        for _ in range(30000):
            s.step()
            for c in s.data.contact:
                if set(s.model.geom_contype[c.geom]) == {2, 4}:
                    contacts.add(tuple(s.model.geom(int(g)).name for g in c.geom))
            if s.violation:
                break
        assert np.all(s.actual_arm() >= LOWER - 1e-4)
        assert np.all(s.actual_arm() <= UPPER + 1e-4)
        assert np.all(np.abs(s.data.qvel[s.arm_dadr]) <= VELOCITY + 1e-4)
        window.last_panel = 0
        window.save(base.OUT / 'verified_arm_contact')
        evidence = s.state()
        evidence.update(passed=True, observed_arm_cargo_pairs=sorted(contacts),
                        checks=['CAD CRC, closure, 30000 random solid/collider checks and aperture probes in geometry.json',
                                '21 prescribed links, 7 true dynamic arm joints, 30 basket colliders',
                                'actual arm moves under torque-bounded position servos; no qpos writes during steps',
                                '19 URDF joint contracts unchanged; runtime gripper 5 Nm / 0.25 rad/s',
                                '0 -> 5 -> 10 -> 15 degree carrier path; 20 degree bucket-height request rejected',
                                'out-of-range arm request rejected; record/play/stop exercised',
                                'GUI tabs, camera zoom/drag callbacks and native OpenGL frames rendered',
                                'constraint guard retains last valid state when a proposed next step would exceed a monitored limit'])
        (OUT / 'verification.json').write_text(json.dumps(evidence, ensure_ascii=False, indent=2) + '\n')
        print('ARM_VIEWER_SELF_TEST_PASSED', flush=True)
    finally:
        window.close()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--self-test', action='store_true')
    parser.add_argument('--actions', type=Path, help='Load recorded_action_targets_rad from a saved viewer JSON')
    parser.add_argument('--sweep', action='store_true', help='Open the explicitly initialized 15deg/+10cm sweep diagnostic')
    parser.add_argument('--self-test-sweep', action='store_true')
    parser.add_argument('--edge', action='store_true', help='Shovel edge parallel to floor; adjustable face inclination')
    args = parser.parse_args()
    if args.self_test:
        self_test()
        return
    if args.self_test_sweep:
        self_test_sweep()
        return
    sim = ArmDiagnostic()
    if args.edge:
        from basket_edge import EdgeControl
        sim.edge = EdgeControl(sim)
        if not sim.edge.initialize():
            parser.error(sim.message)
        for _ in range(2000):
            sim.step()
    if args.sweep:
        if not sim.initialize_sweep():
            parser.error(sim.message)
        for _ in range(2000):
            sim.step()
    if args.actions:
        frames = json.loads(args.actions.read_text())['recorded_action_targets_rad']
        for q in frames:
            values = np.asarray(q, dtype=float)
            if values.shape != (7,) or np.any(values < LOWER) or np.any(values > UPPER) or not np.all(np.isfinite(values)):
                parser.error('Saved action contains an invalid joint target')
        sim.recorded = [np.array(q, dtype=float) for q in frames]
        sim.message = f'已加载 {len(frames)} 个自定义动作点；点击回放执行'
    window = Window(sim)
    if args.sweep:
        window.tab = 'sweep'
        window.view('tail')
    if args.edge:
        window.tab = 'edge'
        window.view('tail')
        glfw.set_window_title(window.window, f'TaskB | {REVISION} 横边贴底铲扫：边线平行、铲面可倾斜')
    try:
        window.run()
    finally:
        window.close()


if __name__ == '__main__':
    sys.modules.setdefault('basket_arm_viewer', sys.modules[__name__])
    main()
