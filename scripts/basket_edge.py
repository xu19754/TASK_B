"""Transverse shovel edge parallel to basket floor; jaw face may tilt."""
from pathlib import Path
import hashlib
import json
import math

import mujoco
import numpy as np

import basket_arm_viewer as arm
from basket_sweep import Kinematics, set_candidate_posture

OUT = arm.OUT / 'edge_parallel'
# Outer transverse edge of Cube_08, in gripperMover's URDF frame.
# This is an existing collision edge, not an added push plate. Leave a small
# closing margin to avoid commanding the gripper exactly at its upper stop.
GRIP = -.08
EDGE = np.array([[.1387284, -.0385, .0118656], [.1387284, .0385, .0118656]])
FACE = np.array([[.1027284, -.0385, .0118656], [.1387284, -.0385, .0118656],
                 [.1387284, .0385, .0118656], [.1027284, .0385, .0118656]])
EDGE_CENTER = np.array([.049, 0., 0.]) + arm.base.geometry.rotation([0, 1, 0], GRIP) @ EDGE.mean(axis=0)


def floor_z(x):
    """CAD inner floor / 13 mm rear chamfer, then the 15 mm outer sill."""
    if 'floor_profile_xz_m' in arm.MODEL_SPEC:
        profile = np.asarray(arm.MODEL_SPEC['floor_profile_xz_m'])
        return np.interp(x, profile[:, 0], profile[:, 1])
    return .0015 + np.clip(-np.asarray(x) - .2105, 0., .0135)


class EdgeControl:
    def __init__(self, sim):
        self.sim = sim
        sim.smooth_peak = .02
        self.x, self.gap_mm, self.tilt = -.23, 78.5, 0.
        self.query = Kinematics(sim, EDGE_CENTER)
        self.preview = None
        self.preview_ok = False

    def target(self):
        return np.array([self.x, 0., float(floor_z(self.x)) + self.gap_mm / 1000])

    def solve(self):
        row = self.query.solve(self.target(), phi=-math.pi - math.radians(self.tilt) - GRIP,
                               roll=0., grip=GRIP, attempts=8)
        row.update(target_edge_basket_m=self.target().tolist(), requested_tilt_deg=self.tilt,
                   requested_gap_mm=self.gap_mm,
                   scope='transverse edge parallel to basket floor; face inclination allowed')
        self.preview = row
        self.preview_ok = (row['position_error_m'] <= .002 and row['orientation_error_rad'] <= .02
                           and row['min_contact_distance_m'] >= -.0002)
        row['accepted'] = self.preview_ok
        self.sim.message = ('目标通过几何检查；移动前还会检查整段路径' if self.preview_ok else
                            f"拒绝目标：位置误差{row['position_error_m'] * 1000:.1f}mm / "
                            f"姿态误差{math.degrees(row['orientation_error_rad']):.1f}° / "
                            f"穿透{max(0., -row['min_contact_distance_m']) * 1000:.1f}mm")
        return self.preview_ok

    def initialize(self):
        return self.initialize_candidate()

    def _set_initial(self, q):
        s = self.sim
        s.data.qpos[s.arm_qadr] = q
        s.data.qvel[s.arm_dadr] = 0.
        s.arm_target, s.arm_command = q.copy(), q.copy()
        s.gravity_compensation = True
        mujoco.mj_forward(s.model, s.data)

    def move(self):
        if not self.solve():
            return False
        return self.sim.begin_smooth([self.sim.actual_arm(), self.preview['q_rad']])

    def candidate(self):
        path = OUT / 'low_edge_candidate.json'
        row = json.loads(path.read_text())
        if row['urdf_sha256'] != hashlib.sha256(arm.VARIANT.read_bytes()).hexdigest():
            raise ValueError('贴底轨迹与当前 URDF 不一致')
        return row

    def initialize_candidate(self):
        try:
            row = self.candidate()
        except (OSError, ValueError) as e:
            self.sim.message = str(e)
            return False
        set_candidate_posture(self.sim)
        self._set_initial(np.asarray(row['q_nodes_rad'][0]))
        self.x, _, z = row['edge_nodes_basket_m'][0]
        self.gap_mm = (z - float(floor_z(self.x))) * 1000
        self.tilt = row['face_tilt_nodes_deg'][0]
        self.preview = None
        self.sim.message = '铲扫准备初态：红边横置，面可倾斜；入篮路径未验证'
        return True

    def play_candidate(self):
        try:
            row = self.candidate()
        except (OSError, ValueError) as e:
            self.sim.message = str(e)
            return False
        if abs(self.sim.pose.pitch - 15) > .01 or abs(self.sim.pose.x - .1) > .001:
            self.sim.message = '请先设置贴底对照初态'
            return False
        return self.sim.begin_smooth(row['q_nodes_rad'], row['parameters'])

    def metrics(self):
        s = self.sim
        b = s.pose.links['tail_basket']
        t = s.data.body('gripperMover')
        r = t.xmat.reshape(3, 3)
        local_r = b[:3, :3].T @ r
        edge = (EDGE @ r.T + t.xpos - b[:3, 3]) @ b[:3, :3]
        face = (FACE @ r.T + t.xpos - b[:3, 3]) @ b[:3, :3]
        angle = math.degrees(math.acos(np.clip(abs(local_r[2, 2]), 0, 1)))
        edge_angle = math.degrees(math.asin(np.clip(abs(edge[1, 2] - edge[0, 2]) / np.linalg.norm(EDGE[1] - EDGE[0]), 0, 1)))
        # Exact lowest vertex relative to the infinite flat inner-floor plane.
        # Outside the basket this is a reference height, NOT a contact distance.
        lowest = np.inf
        for i in range(s.model.ngeom):
            if s.model.geom(i).name.startswith(('collision_gripperStator_', 'collision_gripperMover_')):
                mid = s.model.geom_dataid[i]
                adr, count = s.model.mesh_vertadr[mid], s.model.mesh_vertnum[mid]
                v = s.model.mesh_vert[adr:adr + count]
                p = (v @ s.data.geom_xmat[i].reshape(3, 3).T + s.data.geom_xpos[i] - b[:3, 3]) @ b[:3, :3]
                lowest = min(lowest, float(p[:, 2].min() - .0015))
        contacts = []
        for c in s.data.contact:
            names = [s.model.geom(int(g)).name for g in c.geom]
            if any(n.startswith('collision_tail_basket_') for n in names) and set(s.model.geom_contype[c.geom]) == {1, 4}:
                contacts.append(dict(distance_m=float(c.dist), names=names))
        return dict(face_angle_to_floor_deg=angle, edge_angle_to_floor_deg=edge_angle,
                    edge_end_height_difference_m=float(abs(edge[1, 2] - edge[0, 2])),
                    edge_basket_m=edge.tolist(), face_basket_m=face.tolist(),
                    edge_height_above_local_floor_m=(edge[:, 2] - floor_z(edge[:, 0])).tolist(),
                    jaw_lowest_vertex_above_flat_floor_m=lowest, arm_basket_contacts=contacts,
                    selected_front_edge_width_m=float(np.linalg.norm(EDGE[1] - EDGE[0])),
                    requested_edge_x_m=self.x, requested_edge_gap_mm=self.gap_mm,
                    requested_face_tilt_deg=self.tilt, preview=self.preview,
                    reference='local basket floor, follows carrier pitch')

    def draw_panel(self, txt, button, slider):
        s = self.sim
        m = self.metrics()
        txt(183, '前缘横置贴底，铲面允许倾斜向尾部扫', 18, '#68b6ff')
        slider(218, '前缘 X（负方向为尾部，cm）', self.x * 100, -29, 20,
               lambda x: setattr(self, 'x', x / 100), '{:+.1f}')
        slider(268, '目标前缘离底（mm）', self.gap_mm, 2, 160,
               lambda x: setattr(self, 'gap_mm', x), '{:.1f}')
        slider(318, '铲面倾角（可倾斜，前缘保持横向）', self.tilt, 0, 85,
               lambda x: setattr(self, 'tilt', x), '{:.1f}°')
        button(20, 372, 214, '仅计算目标（不移动）', self.solve)
        button(244, 372, 214, '检查整段并平滑移动', self.move)
        button(20, 412, 438, '重设铲扫准备初态（重装料，非入篮运动）', self.initialize)
        txt(458, f"实际前缘夹角 {m['edge_angle_to_floor_deg']:.2f}°；两端高差 {m['edge_end_height_difference_m'] * 1000:.2f} mm", 15)
        gap = np.mean(m['edge_height_above_local_floor_m']) * 1000
        txt(485, f"前缘离底 {gap:.1f} mm；铲面倾角 {m['face_angle_to_floor_deg']:.1f}°", 15)
        txt(512, '橙框：实际铲面；绿边：手动请求；紫线：候选路径', 13, '#a8b8cc')
        button(20, 547, 438, '播放横边贴底铲扫（自动协调肩肘腕）', self.play_candidate)
        button(20, 587, 214, '记录当前实际姿态', s.record_actual)
        button(244, 587, 214, '停止并保持', s.stop)
        slider(630, '轨迹目标峰值速度（改后重新播放）', s.smooth_peak, .01, .1,
               lambda x: setattr(s, 'smooth_peak', round(x, 3)), '{:.3f}')
        txt(683, '前段目标离底 3 mm；尾沿 15 mm，末段提前抬升', 13, '#f4c97b')
        txt(707, '红边宽 77 mm，篮内宽 247 mm，侧边可能仍需补扫', 13, '#a8b8cc')

    def overlay(self, window):
        b = self.sim.pose.links['tail_basket']
        m = self.metrics()
        def lines(points, color, radius=.002):
            if len(points) < 2:
                return
            world = np.asarray(points) @ b[:3, :3].T + b[:3, 3]
            for p, q in zip(world[:-1], world[1:]):
                geom = window.scene.geoms[window.scene.ngeom]
                mujoco.mjv_initGeom(geom, mujoco.mjtGeom.mjGEOM_CAPSULE, np.zeros(3), np.zeros(3),
                                   np.eye(3).ravel(), np.asarray(color, dtype=np.float32))
                mujoco.mjv_connector(geom, mujoco.mjtGeom.mjGEOM_CAPSULE, radius, p, q)
                window.scene.ngeom += 1
        face = np.asarray(m['face_basket_m'])
        lines(np.vstack([face, face[0]]), [1., .65, .15, 1.])
        lines(m['edge_basket_m'], [1., .12, .08, 1.], .003)
        target = self.target()
        lines([target + [0, -.0385, 0], target + [0, .0385, 0]], [.2, 1., .4, 1.])
        if not hasattr(self, '_curve'):
            try:
                self._curve = self.candidate()['edge_nodes_basket_m']
            except (OSError, ValueError):
                self._curve = []
        lines(self._curve, [.95, .32, .84, .8], .001)


def build_analysis():
    """Reproducible requested-plane probes and a separate tilted comparison."""
    OUT.mkdir(exist_ok=True)
    s = arm.ArmDiagnostic()
    set_candidate_posture(s)
    k = Kinematics(s, EDGE_CENTER)
    probes = []
    for x in [.16, .08, 0., -.08, -.16, -.22, -.27]:
        for theta in [0, 10, 20, 30, 40, 50, 60, 70]:
            row = k.solve(np.array([x, 0, float(floor_z(x)) + .003]),
                          phi=-math.pi - math.radians(theta) - GRIP, roll=0, grip=GRIP, attempts=8)
            row.update(x_m=x, tilt_deg=theta, edge_gap_m=.003)
            row['accepted'] = (row['position_error_m'] <= .002 and row['orientation_error_rad'] <= .02
                               and row['min_contact_distance_m'] >= -.0002)
            probes.append(row)
            if row['accepted']:
                break
        print('EDGE_PROBE', x, row['accepted'], theta, flush=True)
    (OUT / 'workspace_probes.json').write_text(json.dumps(dict(
        rows=probes, angular_grid_deg=10, ik_joint_reserve_rad=.04,
        scope='edge always transverse; face angle may vary; numerical IK, not a global infeasibility proof'), indent=2) + '\n')
    anchors = np.array([[.16, .12, 65], [.16, .07, 70], [.16, .0045, 72],
                        [.12, .0045, 67], [.08, .0045, 62], [.04, .0045, 57],
                        [0, .0045, 52], [-.04, .0045, 47], [-.08, .0045, 42],
                        [-.12, .016, 37], [-.16, .030, 32], [-.18, .035, 27],
                        [-.20, .035, 22], [-.22, .035, 17], [-.24, .035, 12], [-.28, .035, 10]])
    nodes, points, tilts = [], [], []
    previous = None
    for i, (p, q) in enumerate(zip(anchors[:-1], anchors[1:])):
        for u in np.linspace(0, 1, 5, endpoint=i == len(anchors) - 2):
            x, z, theta = p * (1 - u) + q * u
            row = k.solve(np.array([x, 0, z]), phi=-math.pi - math.radians(theta) - GRIP, roll=0,
                          grip=GRIP, previous=previous, attempts=8 if previous is None else 2)
            if row['position_error_m'] > .002 or row['orientation_error_rad'] > .02 or row['min_contact_distance_m'] < -.0002:
                raise ValueError(f'Low edge candidate rejected: {x, z, theta}: {row}')
            previous = np.asarray(row['q_rad'])
            nodes.append(previous)
            points.append([x, 0., z])
            tilts.append(theta)
    path = arm.JointTrajectory(nodes, arm.LOWER, arm.UPPER, arm.VELOCITY, .02)
    for q in path.spline(np.linspace(0, 1, 1001)):
        clearance, pairs = k.clearance(q)
        if clearance < -.0002:
            raise ValueError(f'Low edge interpolated collision: {clearance}, {pairs}')
    result = path.details()
    result.update(urdf_sha256=hashlib.sha256(arm.VARIANT.read_bytes()).hexdigest(),
                  q_nodes_rad=np.asarray(nodes).tolist(), parameters=path.parameters.tolist(),
                  edge_nodes_basket_m=points, face_tilt_nodes_deg=tilts,
                  selected_edge_in_mover_m=EDGE.tolist(), selected_face_in_mover_m=FACE.tolist(),
                  body=dict(pitch_deg=15, x_m=.10, height_m=.575),
                  scope='low leading edge parallel to basket floor, tilted shovel face as requested',
                  gripper_opening_rad=GRIP, entry_path_verified=False)
    (OUT / 'low_edge_candidate.json').write_text(json.dumps(result, indent=2) + '\n')
    print('LOW_EDGE_CANDIDATE', path.duration, flush=True)


def verify():
    OUT.mkdir(exist_ok=True)
    s = arm.ArmDiagnostic()
    s.edge = EdgeControl(s)
    assert s.edge.initialize(), s.message
    for _ in range(2000):
        s.step()
    assert not s.violation, s.rejected_step
    m = s.edge.metrics()
    assert m['edge_angle_to_floor_deg'] < 2, m
    w = arm.Window(s, visible=False)
    w.tab = 'edge'
    w.view('tail')
    try:
        w.save(OUT / 'transverse_edge_initial')
        # A request may be infeasible even when a least-squares fit returns q.
        q_before = s.actual_arm()
        s.edge.x, s.edge.gap_mm, s.edge.tilt = .08, 3., 0.
        assert not s.edge.move()
        assert np.array_equal(s.actual_arm(), q_before)
        w.last_panel = 0
        w.save(OUT / 'extra_flat_face_request_rejected')
        assert s.edge.initialize_candidate(), s.message
        for _ in range(2000):
            s.step()
        assert not s.violation, s.rejected_step
        assert s.edge.play_candidate(), s.message
        duration = s.smooth_motion.duration
        trace, contacts = [], set()
        w.last_panel = 0
        w.save(OUT / 'low_edge_start')
        for step in range(math.ceil((duration + 4) / arm.base.DT)):
            s.step()
            for c in s.data.contact:
                if set(s.model.geom_contype[c.geom]) == {2, 4}:
                    contacts.add(tuple(s.model.geom(int(g)).name for g in c.geom))
            if step % 1000 == 0:
                trace.append(s.state())
            if step in [int(duration * .35 / arm.base.DT), int(duration * .7 / arm.base.DT)]:
                w.last_panel = 0
                w.save(OUT / ('low_edge_35' if step < duration * .5 / arm.base.DT else 'low_edge_70'))
            if step % 20000 == 0:
                print('LOW_EDGE_ROLLOUT', s.data.time, sum(s.escaped), s.violation, flush=True)
            if s.violation:
                break
        w.last_panel = 0
        w.save(OUT / 'low_edge_finish')
        evidence = s.state()
        assert np.all(s.actual_arm() >= arm.LOWER - 1e-4)
        assert np.all(s.actual_arm() <= arm.UPPER + 1e-4)
        assert np.all(np.abs(s.data.qvel[s.arm_dadr]) <= arm.VELOCITY + 1e-4)
        assert np.all(np.abs(s.data.actuator_force[s.arm_motor]) <= arm.EFFORT + 1e-7)
        evidence.update(passed=not bool(s.violation) and s.smooth_complete,
                        diagnostic_checks_passed=True, motion_completed=s.smooth_complete,
                        trace=trace, observed_arm_cargo_pairs=sorted(contacts),
                        checks=['transverse leading edge parallel to basket floor; face may tilt',
                                'real URDF moving-jaw face and transverse edge highlighted',
                                'infeasible 0deg front target rejected without moving qpos',
                                'candidate tilted in front; entry remains diagnostic initialization'])
        (OUT / 'verification.json').write_text(json.dumps(evidence, indent=2, ensure_ascii=False) + '\n')
        print('LOW_EDGE_RESULT', evidence['passed'], sum(s.escaped), s.violation, flush=True)
    finally:
        w.close()


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--analyze', action='store_true')
    parser.add_argument('--verify', action='store_true')
    args = parser.parse_args()
    if args.analyze:
        build_analysis()
    if args.verify:
        verify()
