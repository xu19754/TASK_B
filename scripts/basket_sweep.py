"""Continuous candidate sweep: 15 deg nose up, 10 cm forward.

Geometric continuation plus torque-limited rollout in the existing diagnostic.
No changes to robot assets, scoring, actuator caps or collision masks.
"""
from pathlib import Path
import json
import math
import hashlib

import mujoco
import numpy as np
from scipy.optimize import least_squares
from scipy.spatial.transform import Rotation
from scipy.interpolate import CubicSpline

import basket_arm_viewer as arm

OUT = arm.OUT / 'sweep_15deg_forward10cm'
TOOL_CENTER = np.array([.10, 0, .04])


class Kinematics:
    def __init__(self, sim, tool_center=None):
        self.sim = sim
        self.tool_center = TOOL_CENTER if tool_center is None else np.asarray(tool_center)
        self.chain = []
        for name in arm.ARM_LINKS[:-1]:
            joint = next(j for j in arm.base.geometry.ROBOT.findall('joint') if j.find('child').get('link') == name)
            move = None if joint.get('type') == 'fixed' else (arm.ARM_NAMES.index(joint.get('name')),
                   np.fromstring(joint.find('axis').get('xyz'), sep=' '))
            self.chain.append((arm.base.geometry.origin(joint.find('origin')), move))
        self.query = mujoco.MjData(sim.model)

    def fk(self, q):
        t = self.sim.pose.links['base'].copy()
        for origin, move in self.chain:
            t = t @ origin
            if move:
                r = np.eye(4)
                r[:3, :3] = arm.base.geometry.rotation(move[1], q[move[0]])
                t = t @ r
        return t

    def local_center(self, q):
        t = self.fk(q)
        b = self.sim.pose.links['tail_basket']
        return (t[:3, :3] @ self.tool_center + t[:3, 3] - b[:3, 3]) @ b[:3, :3]

    def clearance(self, q):
        s, d = self.sim, self.query
        d.qpos[:] = s.data.qpos
        d.qpos[s.arm_qadr] = q
        d.mocap_pos[:] = s.data.mocap_pos
        d.mocap_quat[:] = s.data.mocap_quat
        mujoco.mj_forward(s.model, d)
        pairs = []
        for c in d.contact:
            if set(s.model.geom_contype[c.geom]) == {1, 4}:
                pairs.append((float(c.dist), [s.model.geom(int(g)).name for g in c.geom]))
        return min([p[0] for p in pairs] + [1.]), pairs

    def solve(self, position, phi=-math.pi, roll=math.pi / 2, grip=-1.35, previous=None, attempts=8,
              rotation_basket=None):
        s = self.sim
        b = s.pose.links['tail_basket']
        target_p = b[:3, :3] @ position + b[:3, 3]
        local_r = (arm.base.geometry.rotation([0, 1, 0], phi) @ arm.base.geometry.rotation([1, 0, 0], roll)
                   if rotation_basket is None else np.asarray(rotation_basket))
        target_r = b[:3, :3] @ local_r
        preferred = np.array([0, .6, -3., -.7, 0, roll, grip]) if previous is None else np.asarray(previous)

        def residual(q):
            t = self.fk(q)
            p = t[:3, :3] @ self.tool_center + t[:3, 3]
            orientation = Rotation.from_matrix(target_r.T @ t[:3, :3]).as_rotvec()
            return np.r_[p - target_p, .10 * orientation, .0002 * (q - preferred[:6])]

        seeds = [preferred[:6], [0, .2, -3.1, -.2, 0, roll], [0, 1.6, -3.6, -.7, 0, roll],
                 [0, 2.4, -3.8, -1.4, 0, roll], [.4, 2., -4.4, -.7, .4, roll],
                 [-.4, 2., -4.4, -.7, -.4, roll], [0, 2.9, -2.7, -1.5, 0, roll],
                 [0, 1.6, -4.6, 0, 0, roll]]
        best = None
        for seed in seeds[:attempts]:
            fit = least_squares(residual, np.clip(seed, arm.LOWER[:6] + .04, arm.UPPER[:6] - .04),
                                bounds=(arm.LOWER[:6] + .04, arm.UPPER[:6] - .04), max_nfev=160)
            q = np.r_[fit.x, grip]
            t = self.fk(q)
            pe = float(np.linalg.norm(t[:3, :3] @ self.tool_center + t[:3, 3] - target_p))
            re = float(np.linalg.norm(Rotation.from_matrix(target_r.T @ t[:3, :3]).as_rotvec()))
            clearance, pairs = self.clearance(q)
            score = pe + .10 * re + 20 * max(0, -clearance) + .0001 * np.linalg.norm(q - preferred)
            row = dict(q_rad=q.tolist(), position_error_m=pe, orientation_error_rad=re,
                       min_contact_distance_m=clearance, collision_pairs=pairs, tool_center_basket_m=self.local_center(q).tolist())
            if best is None or score < best[0]:
                best = score, row
        return best[1]


def set_candidate_posture(sim):
    pose, reason = sim.limits.solve(15, .10, .575, True)
    if pose is None:
        raise RuntimeError(reason)
    # Explicit initialization for isolating the arm sweep, not a task action.
    sim.pose = pose
    sim.target = pose
    sim.reset()
    return pose


def explore():
    s = arm.ArmDiagnostic()
    set_candidate_posture(s)
    k = Kinematics(s)
    rows = []
    for phi in [-math.pi, -2.85, -2.55]:
        for p in [[.19, 0, .17], [.10, 0, .14], [0, 0, .09], [-.10, 0, .07], [-.20, 0, .07], [-.27, 0, .08]]:
            row = k.solve(np.array(p), phi=phi)
            row.update(target_basket_m=p, phi_rad=phi)
            rows.append(row)
            print('pose', p, 'phi', round(phi, 2), 'poserr', round(row['position_error_m'], 4),
                  'roterr', round(row['orientation_error_rad'], 3), 'clear', round(row['min_contact_distance_m'], 4),
                  'q', np.round(row['q_rad'], 2), flush=True)
    OUT.mkdir(exist_ok=True)
    (OUT / 'workspace.json').write_text(json.dumps(rows, indent=2) + '\n')


def build_candidate(sim, grip=-.8, tool_center=None, dip=False):
    k = Kinematics(sim, tool_center)
    anchors = np.array([[.19, .22, -4.1], [.15, .18, -4.1], [.08, .15, -4.1],
                        [0., .11, -4.1], [-.08, .095, -3.8], [-.16, .085, -3.5], [-.27, .07, -math.pi]])
    if dip:
        anchors = np.array([[.19, .22, -4.1], [.19, .13, -4.3], [.13, .125, -4.25],
                            [.08, .12, -4.2], [0., .11, -4.1], [-.08, .095, -3.8],
                            [-.16, .085, -3.5], [-.27, .07, -math.pi]])
    qrows, nodes = [], []
    previous = None
    for i in range(len(anchors) - 1):
        for u in np.linspace(0, 1, 7, endpoint=i == len(anchors) - 2):
            x, z, phi = (1 - u) * anchors[i] + u * anchors[i + 1]
            row = k.solve(np.array([x, 0, z]), phi=phi, grip=grip, previous=previous,
                          attempts=8 if previous is None else 2)
            if row['position_error_m'] > .003 or row['orientation_error_rad'] > .04 or row['min_contact_distance_m'] < -.0002:
                raise RuntimeError(f'Sweep geometry rejected at x={x:.4f}, z={z:.4f}: {row}')
            previous = np.array(row['q_rad'])
            qrows.append(previous)
            nodes.append([float(x), 0., float(z), float(phi)])
    params = np.linspace(0, 1, len(qrows))
    spline = CubicSpline(params, np.array(qrows), bc_type='clamped')
    sample = np.linspace(0, 1, 1001)
    allq = spline(sample)
    if np.any(allq < arm.LOWER) or np.any(allq > arm.UPPER):
        raise RuntimeError('Spline overshoots a joint limit')
    for q in allq[::5]:
        clearance, pairs = k.clearance(q)
        if clearance < -.0003:
            raise RuntimeError(f'Interpolated sweep collision {clearance}: {pairs}')
    duration = max(45., float(np.max(np.abs(spline(sample, 1))) / .04),
                   math.sqrt(float(np.max(np.abs(spline(sample, 2)))) / .10))
    return dict(urdf_sha256=hashlib.sha256(arm.VARIANT.read_bytes()).hexdigest(),
                q_nodes_rad=np.array(qrows).tolist(), parameters=params.tolist(), duration_s=duration,
                geometric_tool_nodes_m_and_pitch_rad=nodes, opening_rad=grip,
                tool_center_in_stator_m=k.tool_center.tolist(),
                body=dict(pitch_deg=15, x_m=.10, height_m=.575),
                spline='CubicSpline clamped endpoint velocities; common time for all joints',
                target_peak_velocity_rad_s=np.max(np.abs(spline(sample, 1)), axis=0).__truediv__(duration).tolist(),
                target_peak_acceleration_rad_s2=(np.max(np.abs(spline(sample, 2)), axis=0) / duration ** 2).tolist(),
                joint_and_static_collision_preflight=True)


def rollout(gravity_compensation=False, closed=False, dip=False):
    s = arm.ArmDiagnostic()
    set_candidate_posture(s)
    s.gravity_compensation = gravity_compensation
    candidate = build_candidate(s, grip=-.08 if closed else -.8, tool_center=[.10, 0, 0] if closed else None, dip=dip)
    OUT.mkdir(exist_ok=True)
    candidate_name = 'candidate' + ('_closed' if closed else '') + ('_dip' if dip else '') + '_path.json'
    (OUT / candidate_name).write_text(json.dumps(candidate, indent=2) + '\n')
    q0 = np.array(candidate['q_nodes_rad'][0])
    s.data.qpos[s.arm_qadr] = q0
    s.arm_command, s.arm_target = q0.copy(), q0.copy()
    mujoco.mj_forward(s.model, s.data)
    for _ in range(2000):
        s.step()
        if s.violation:
            break
    spline = CubicSpline(candidate['parameters'], np.array(candidate['q_nodes_rad']), bc_type='clamped')
    start = s.data.time
    observed = set()
    states = []
    for step in range(math.ceil((candidate['duration_s'] + 3) / arm.base.DT)):
        u = np.clip((s.data.time - start) / candidate['duration_s'], 0, 1)
        s.arm_target = spline(u)
        s.step()
        for c in s.data.contact:
            if set(s.model.geom_contype[c.geom]) == {2, 4}:
                observed.add(tuple(s.model.geom(int(g)).name for g in c.geom))
        if step % 1000 == 0:
            states.append(s.state())
        if step % 10000 == 0:
            print('sweep', round(s.data.time, 2), round(float(u), 3), 'exited', sum(s.escaped),
                  'gripper_speed', s.data.qvel[s.arm_dadr][-1], flush=True)
        if s.violation:
            break
    result = s.state()
    result.update(observed_arm_cargo_pairs=sorted(observed), trace=states,
                  candidate_path=candidate_name, initial_arm_pose_is_explicit_diagnostic_reset=True)
    name = 'rollout' + ('_closed' if closed else '') + ('_dip' if dip else '') + ('_bias_compensation' if gravity_compensation else '') + '.json'
    (OUT / name).write_text(json.dumps(result, indent=2, ensure_ascii=False) + '\n')
    print('RESULT', s.data.time, sum(s.escaped), s.violation, 'contacts', len(observed), flush=True)


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--gravity-compensation', action='store_true')
    parser.add_argument('--closed', action='store_true')
    parser.add_argument('--dip', action='store_true')
    args = parser.parse_args()
    rollout(args.gravity_compensation, args.closed, args.dip)
