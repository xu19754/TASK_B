"""Cavity volume and fixed-foot, joint-limited posture feasibility screening.

This is inverse kinematics and geometric screening, not a stability simulation
or a controller. It never changes the runtime URDF or evaluation environment.
"""
from pathlib import Path
import json
import math
import sys
import xml.etree.ElementTree as ET

import numpy as np
from scipy.spatial import ConvexHull

OUT = Path(__file__).resolve().parent
ROOT = OUT.parents[1]
sys.path.insert(0, str(ROOT))
from taskb_student.constants import (ARM_Q0, JOINT_LOWER, JOINT_NAMES, JOINT_UPPER,
                                     LEG_Q0, URDF_PATH)

ROBOT = ET.parse(URDF_PATH).getroot()
LEGS = ['FR', 'FL', 'RR', 'RL']


def rotation(axis, angle):
    axis = np.array(axis, dtype=float)
    axis /= np.linalg.norm(axis)
    x, y, z = axis
    skew = np.array([[0, -z, y], [z, 0, -x], [-y, x, 0]])
    return np.eye(3) + math.sin(angle) * skew + (1 - math.cos(angle)) * (skew @ skew)


def origin(element):
    transform = np.eye(4)
    if element is None:
        return transform
    roll, pitch, yaw = map(float, element.get('rpy', '0 0 0').split())
    transform[:3, :3] = rotation([0, 0, 1], yaw) @ rotation([0, 1, 0], pitch) @ rotation([1, 0, 0], roll)
    transform[:3, 3] = list(map(float, element.get('xyz', '0 0 0').split()))
    return transform


def forward(q, bx=0, bz=0, theta=0):
    links = {'base': np.eye(4)}
    links['base'][:3, :3] = rotation([0, 1, 0], -theta)
    links['base'][:3, 3] = [bx, 0, bz]
    targets = dict(zip(JOINT_NAMES, q))
    pending = list(ROBOT.findall('joint'))
    while pending:
        count = len(pending)
        for joint in pending[:]:
            parent, child = joint.find('parent').get('link'), joint.find('child').get('link')
            if parent not in links:
                continue
            motion = np.eye(4)
            if joint.get('type') != 'fixed':
                axis = list(map(float, joint.find('axis').get('xyz').split()))
                motion[:3, :3] = rotation(axis, targets[joint.get('name')])
            links[child] = links[parent] @ origin(joint.find('origin')) @ motion
            pending.remove(joint)
        assert len(pending) < count
    return links


def leg_parameters():
    data = []
    for leg in LEGS:
        transforms = [origin(ROBOT.find(f"joint[@name='{leg}_{name}_joint']/origin"))
                      for name in ['hip', 'thigh', 'calf', 'foot']]
        assert all(np.allclose(t[:3, :3], np.eye(3)) for t in transforms)
        hip, thigh, calf, foot = [t[:3, 3] for t in transforms]
        assert abs(calf[0]) < 1e-12 and abs(foot[0]) < 1e-12
        data.append((hip, thigh, calf, foot))
    return data


def inverse_batch(targets, parameters):
    """Exact 3-DoF leg inverse kinematics for this URDF's Rx/Ry/Ry chains."""
    count = targets.shape[0]
    q = np.zeros((count, 12))
    valid = np.ones(count, dtype=bool)
    for i, (hip, thigh, calf, foot) in enumerate(parameters):
        p = targets[:, i] - hip
        lateral = thigh[1] + calf[1] + foot[1]
        radius = np.hypot(p[:, 1], p[:, 2])
        valid &= radius >= abs(lateral)
        a = np.arctan2(p[:, 2], p[:, 1]) + np.arccos(np.clip(lateral / np.maximum(radius, 1e-12), -1, 1))
        a = (a + math.pi) % (2 * math.pi) - math.pi
        u = p[:, 0] - thigh[0]
        v = -np.sin(a) * p[:, 1] + np.cos(a) * p[:, 2] - thigh[2]
        l1, l2 = -calf[2], -foot[2]
        cosine = (u * u + v * v - l1 * l1 - l2 * l2) / (2 * l1 * l2)
        valid &= np.abs(cosine) <= 1
        c = -np.arccos(np.clip(cosine, -1, 1))
        b = np.arctan2(-u, -v) - np.arctan2(l2 * np.sin(c), l1 + l2 * np.cos(c))
        q[:, 3 * i:3 * i + 3] = np.c_[a, b, c]
    lower, upper = np.array(JOINT_LOWER[:12]), np.array(JOINT_UPPER[:12])
    margin = np.minimum(q - lower, upper - q).min(axis=1)
    valid &= margin >= -1e-10
    return q, valid, margin


def collision_vertices():
    result = {}
    for link in ROBOT.findall('link'):
        name = link.get('name')
        if not (name == 'base' or name.endswith('_hip') or name.endswith('_thigh')):
            continue
        parts = []
        for collision in link.findall('collision'):
            geometry = collision.find('geometry')
            mesh, box = geometry.find('mesh'), geometry.find('box')
            if mesh is not None:
                path = URDF_PATH.parent / mesh.get('filename')
                v = np.array([list(map(float, line.split()[1:4])) for line in path.read_text().splitlines() if line.startswith('v ')])
                v *= np.array(list(map(float, mesh.get('scale', '1 1 1').split())))
            elif box is not None:
                d = np.array(list(map(float, box.get('size').split()))) / 2
                v = np.array([[x, y, z] for x in [-d[0], d[0]] for y in [-d[1], d[1]] for z in [-d[2], d[2]]])
            else:
                continue
            t = origin(collision.find('origin'))
            parts.append(v @ t[:3, :3].T + t[:3, 3])
        if parts:
            result[name] = np.concatenate(parts)
    return result


def main():
    visual = URDF_PATH.parent / 'meshes/basket/basket_visual.obj'
    vertices = np.array([list(map(float, line.split()[1:4])) for line in visual.read_text().splitlines() if line.startswith('v ')])
    inner = vertices[np.abs(vertices[:, 1] + .123) < 1e-6][:, [0, 2]]
    hull = ConvexHull(inner)
    profile = inner[hull.vertices]
    cavity_litres = float(hull.volume * .246 * 1000)
    floor = inner[np.abs(inner[:, 1] - .002) < 1e-6]
    capacity = {'outer_dimensions_cm': [28, 25, 10], 'opening_cm': [27.6, 24.6],
                'inner_height_cm': 9.8, 'flat_floor_cm': [float(np.ptp(floor[:, 0]) * 100), 24.6],
                'inner_cavity_litres_to_rim': cavity_litres, 'inner_profile_xz_m': profile.tolist()}
    params = leg_parameters()
    nominal = forward(LEG_Q0 + ARM_Q0)
    feet = np.array([nominal[f'{leg}_foot'][:3, 3] for leg in LEGS])
    height = float(.04 - feet[:, 2].mean())
    feet[:, 2] += height
    check, valid, _ = inverse_batch(np.array([feet - [0, 0, height]]), params)
    np.testing.assert_allclose(check[0], LEG_Q0, atol=1e-10)
    assert valid[0]
    support = ConvexHull(feet[:, :2]).equations
    bx_grid, bz_grid = np.meshgrid(np.linspace(-.15, .15, 121), np.linspace(.30, .65, 141))
    translations = np.c_[bx_grid.ravel(), np.zeros(bx_grid.size), bz_grid.ravel()]
    collision = collision_vertices()
    results = []
    for angle in range(0, 46, 5):
        theta = math.radians(angle)
        r = rotation([0, 1, 0], -theta)
        targets = (feet[None, :, :] - translations[:, None, :]) @ r
        q, feasible, margins = inverse_batch(targets, params)
        lip_height = translations[:, 2] - .439 * math.sin(theta) + .23 * math.cos(theta)
        candidates = np.flatnonzero(feasible & (margins >= math.radians(2)) & (lip_height >= .57))
        selected = None
        if len(candidates):
            score = (translations[candidates, 0] / .15) ** 2 + ((translations[candidates, 2] - height) / .15) ** 2
            for idx in candidates[np.argsort(score)]:
                full_q = np.r_[q[idx], ARM_Q0]
                links = forward(full_q, translations[idx, 0], translations[idx, 2], theta)
                low = min(float((v @ links[name][:3, :3].T + links[name][:3, 3])[:, 2].min()) for name, v in collision.items())
                if low < .01:
                    continue
                positions, masses = [], []
                for link in ROBOT.findall('link'):
                    inertial = link.find('inertial')
                    if inertial is not None:
                        positions.append((links[link.get('name')] @ origin(inertial.find('origin')))[:3, 3])
                        masses.append(float(inertial.find('mass').get('value')))
                t = links['tail_basket']
                positions.append(t[:3, :3] @ np.array([-.10, 0, .075]) + t[:3, 3])
                masses.append(1.5)
                com = np.average(positions, axis=0, weights=masses)
                support_margin = float(np.min(-(support[:, :2] @ com[:2] + support[:, 2]) / np.linalg.norm(support[:, :2], axis=1)))
                if support_margin < .02:
                    continue
                error = float(max(np.linalg.norm(links[f'{leg}_foot'][:3, 3] - feet[i]) for i, leg in enumerate(LEGS)))
                assert error < 1e-8
                selected = {'base_x_m': float(translations[idx, 0]), 'base_height_m': float(translations[idx, 2]),
                            'rear_lip_height_m': float(lip_height[idx]), 'joint_limit_margin_deg': float(np.degrees(margins[idx])),
                            'illegal_body_min_height_m': low, 'com_support_margin_m': support_margin,
                            'com_world_m': com.tolist(), 'mass_with_payload_kg': sum(masses),
                            'foot_error_m': error, 'leg_targets_rad': q[idx].tolist()}
                break
        row = {'nose_up_deg': angle, 'hard_limit_grid_points': int(feasible.sum()),
               'rim_clearance_and_joint_reserve_grid_points': len(candidates), 'selected_pose': selected}
        results.append(row)
        print(json.dumps(row), flush=True)
    barrier = []
    for angle in [0, 15, 20, 30, 45, 60, 75]:
        u = np.array([math.sin(math.radians(angle)), math.cos(math.radians(angle))])
        lower_height = float((profile @ u).min())
        rim_height = float(np.array([-.138, .1]) @ u)
        barrier.append({'nose_up_deg': angle, 'ideal_point_potential_barrier_m': rim_height - lower_height})
    (OUT / 'geometry_scan.json').write_text(json.dumps({
        'capacity': capacity, 'nominal_base_height_m': height, 'fixed_foot_centres_world_m': feet.tolist(),
        'posture_scan_assumptions': {'base_x_range_m': [-.15, .15], 'base_z_range_m': [.30, .65], 'grid_step_m': .0025,
            'joint_limit_reserve_deg': 2, 'rim_min_height_m': .57, 'bucket_rim_height_m': .55,
            'body_collision_min_ground_clearance_m': .01, 'payload_kg': 1.5,
            'payload_com_basket_m': [-.10, 0, .075], 'foot_centres_fixed': True,
            'scope': 'geometric necessary conditions only; no torques, foot slip, support-force distribution, self-collision, or closed-loop control validation; empty result is not a universal impossibility proof'},
        'postures': results, 'ideal_point_barriers': barrier,
    }, indent=2) + '\n')
    print('CAPACITY', json.dumps(capacity), flush=True)


if __name__ == '__main__':
    main()
