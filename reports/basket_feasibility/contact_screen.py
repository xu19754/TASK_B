"""Independent MuJoCo cargo-contact screening; not an Isaac/robot-control test.

Run from any directory with numpy, scipy, and mujoco installed. The source USDs
are read through pxr if available, otherwise through the usdcat executable.
Basket geometry stays fixed while gravity rotates slowly in its frame. This
approximates a prescribed quasi-static basket tilt, without testing the dog.
"""
from pathlib import Path
import ast
import json
import math
import re
import shutil
import subprocess
import xml.etree.ElementTree as ET

import mujoco
import numpy as np
from scipy.spatial.transform import Rotation

OUT = Path(__file__).resolve().parent
ROOT = OUT.parents[1]
ASSET = ROOT / 'assets/b2z1'
OBJECT_NAMES = ['004_sugar_box', '006_mustard_bottle', '011_banana']
DT = .001
SETTLE, RAMP, HOLD = 2.0, 5.0, 3.0


def read_obj(path):
    v, f = [], []
    for line in path.read_text().splitlines():
        s = line.split()
        if s and s[0] == 'v':
            v.append(list(map(float, s[1:4])))
        elif s and s[0] == 'f':
            f.append([int(x.split('/')[0]) - 1 for x in s[1:]])
    return np.array(v), np.array(f)


def read_usd(path):
    try:
        from pxr import Usd, UsdGeom
        stage = Usd.Stage.Open(str(path))
        meshes = [p for p in stage.Traverse() if p.IsA(UsdGeom.Mesh)]
        assert len(meshes) == 1
        mesh = UsdGeom.Mesh(meshes[0])
        points = np.array(mesh.GetPointsAttr().Get(), dtype=float)
        transform = np.array(UsdGeom.XformCache().GetLocalToWorldTransform(meshes[0]))
        points = (np.c_[points, np.ones(len(points))] @ transform)[:, :3]
        points *= UsdGeom.GetStageMetersPerUnit(stage)
        counts = np.array(mesh.GetFaceVertexCountsAttr().Get())
        indices = np.array(mesh.GetFaceVertexIndicesAttr().Get())
    except ImportError:
        exe = shutil.which('usdcat')
        if not exe:
            raise RuntimeError('Install usd-core (pxr) or provide usdcat on PATH.')
        text = subprocess.check_output([exe, str(path)], text=True)
        def values(name):
            return ast.literal_eval('[' + re.search(r'\b' + re.escape(name) + r'\s*=\s*\[([^\]]*)\]', text).group(1) + ']')
        points = np.array(values('points'), dtype=float)
        scale = ast.literal_eval(re.search(r'xformOp:scale\s*=\s*(\([^\)]*\))', text).group(1))
        mpu = float(re.search(r'metersPerUnit\s*=\s*([\d.eE+-]+)', text).group(1))
        # These three supplied files have a single scale op and no other transforms.
        assert re.findall(r'xformOp:\w+\s*=', text) == ['xformOp:scale =']
        points *= np.array(scale) * mpu
        counts, indices = np.array(values('faceVertexCounts')), np.array(values('faceVertexIndices'))
    assert np.all(counts == 3)
    points -= (points.min(0) + points.max(0)) / 2
    return points, indices.reshape(-1, 3)


def write_obj(path, v, f):
    path.write_text('\n'.join(['# metres'] + ['v ' + ' '.join(f'{x:.10g}' for x in p) for p in v] +
                              ['f ' + ' '.join(str(int(x) + 1) for x in t) for t in f]) + '\n')


def quat_wxyz(matrix):
    q = Rotation.from_matrix(matrix).as_quat()
    return np.r_[q[3], q[:3]]


def prepare_objects():
    target = OUT / 'screen_meshes'
    target.mkdir(exist_ok=True)
    objects = {}
    for name in OBJECT_NAMES:
        v, f = read_usd(ROOT / 'assets/objects' / (name + '.usd'))
        path = target / (name + '.obj')
        write_obj(path, v, f)
        objects[name] = {'vertices': v, 'file': path, 'dimensions_m': v.max(0) - v.min(0)}
    return objects


def make_model(objects, scenario, mu):
    root = ET.Element('mujoco', model='prescribed_basket_tilt_screen')
    ET.SubElement(root, 'compiler', angle='radian', inertiafromgeom='true')
    ET.SubElement(root, 'option', timestep=str(DT), gravity='0 0 -9.81', iterations='60', cone='elliptic')
    defaults = ET.SubElement(root, 'default')
    ET.SubElement(defaults, 'geom', friction=f'{mu} 0.005 0.0001', condim='3', solref='0.004 1', solimp='0.95 0.99 0.001')
    assets = ET.SubElement(root, 'asset')
    world = ET.SubElement(root, 'worldbody')
    basket = ET.parse(ASSET / 'b2z1_0815_with_basket.urdf').getroot().find("link[@name='tail_basket']")
    for i, collision in enumerate(basket.findall('collision')):
        xyz = collision.find('origin').get('xyz')
        geometry = collision.find('geometry')
        box = geometry.find('box')
        if box is not None:
            half = np.array(list(map(float, box.get('size').split()))) / 2
            ET.SubElement(world, 'geom', name=f'basket_{i}', type='box', pos=xyz, size=' '.join(map(str, half)))
        else:
            path = ASSET / geometry.find('mesh').get('filename')
            ET.SubElement(assets, 'mesh', name=f'basket_mesh_{i}', file=str(path), inertia='convex')
            ET.SubElement(world, 'geom', name=f'basket_{i}', type='mesh', mesh=f'basket_mesh_{i}', pos=xyz)
    for name, ob in objects.items():
        ET.SubElement(assets, 'mesh', name=name, file=str(ob['file']), inertia='convex')
    mixed = [
        ('004_sugar_box', [.075, 0, .030], np.eye(3)),
        ('006_mustard_bottle', [-.025, 0, .055], np.eye(3)),
        ('011_banana', [.075, 0, .090], np.array([[0, 0, 1], [1, 0, 0], [0, 1, 0]])),
    ]
    if scenario == 'mixed_three':
        placements = mixed
    else:
        name = scenario
        if name == '006_mustard_bottle':
            orientation = Rotation.from_euler('y', math.pi / 2).as_matrix()
            placements = [(name, [x, 0, .057], orientation) for x in [-.008, .052, .112]]
        elif name == '004_sugar_box':
            placements = [(name, [x, 0, z], np.eye(3)) for x, z in [(-.025, .040), (.075, .030), (.075, .085)]]
        else:
            orientation = mixed[-1][2]
            placements = [(name, [x, 0, .050], orientation) for x in [-.070, .009, .088]]
    for i, (name, pos, mat) in enumerate(placements):
        body = ET.SubElement(world, 'body', name=f'cargo_{i}', pos=' '.join(map(str, pos)), quat=' '.join(map(str, quat_wxyz(mat))))
        ET.SubElement(body, 'freejoint', name=f'cargo_joint_{i}')
        ET.SubElement(body, 'geom', name=f'cargo_geom_{i}', type='mesh', mesh=name, mass='0.5')
    xml = ET.tostring(root, encoding='unicode')
    (OUT / f'contact_{scenario}.xml').write_text(xml)
    model = mujoco.MjModel.from_xml_string(xml)
    # Approximate the exercise's linear and angular damping rates, without claiming PhysX equivalence.
    for i in range(3):
        bid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, f'cargo_{i}')
        adr = model.jnt_dofadr[model.body_jntadr[bid]]
        model.dof_damping[adr:adr + 3] = model.body_mass[bid] * 2.0
        model.dof_damping[adr + 3:adr + 6] = model.body_inertia[bid] * 4.0
    return model, placements


def cargo_vertices(model, data, placements, objects):
    out = []
    for i, (name, _, _) in enumerate(placements):
        bid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, f'cargo_{i}')
        v = objects[name]['vertices'] @ data.xmat[bid].reshape(3, 3).T + data.xpos[bid]
        out.append(v)
    return out


def inside_rim(vertices):
    # Convex inner cavity profile, derived from the actual supplied basket.
    angles = np.linspace(math.pi, math.pi * 1.5, 13)
    poly = np.c_[-.04 + .098 * np.cos(angles), .10 + .098 * np.sin(angles)]
    poly = np.vstack([poly, [.138, .002], [.138, .100]])
    following = np.roll(poly, -1, axis=0)
    area = np.sum(poly[:, 0] * following[:, 1] - poly[:, 1] * following[:, 0]) / 2
    edge = np.roll(poly, -1, axis=0) - poly
    normal = np.c_[edge[:, 1], -edge[:, 0]] * np.sign(area)
    normal /= np.linalg.norm(normal, axis=1)[:, None]
    dist = np.einsum('pfi,fi->pf', vertices[:, None, [0, 2]] - poly[None, :, :], normal)
    return bool(np.all(dist <= .003) and np.max(np.abs(vertices[:, 1])) <= .126)


def run_case(model, placements, objects, angle):
    data = mujoco.MjData(model)
    model.opt.gravity[:] = [0, 0, -9.81]
    for _ in range(round(SETTLE / DT)):
        mujoco.mj_step(model, data)
    mujoco.mj_forward(model, data)
    start = cargo_vertices(model, data, placements, objects)
    initial_poses = []
    for i, (name, _, _) in enumerate(placements):
        bid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, f'cargo_{i}')
        initial_poses.append({'asset': name, 'position_m': data.xpos[bid].tolist(),
                              'quat_wxyz': data.xquat[bid].tolist()})
    initial_contained = [inside_rim(v) for v in start]
    retained = [bool(v[:, 0].min() > -.145 and v[:, 0].max() < .145 and np.max(np.abs(v[:, 1])) < .135 and v[:, 2].mean() > -.01) for v in start]
    rear_exit = [False] * 3
    threshold = math.radians(angle)
    for step in range(round((RAMP + HOLD) / DT)):
        u = min(step * DT / RAMP, 1)
        t = threshold * u * u * (3 - 2 * u)
        model.opt.gravity[:] = [-9.81 * math.sin(t), 0, -9.81 * math.cos(t)]
        mujoco.mj_step(model, data)
        if step % 25 == 0:
            mujoco.mj_forward(model, data)
            for i, v in enumerate(cargo_vertices(model, data, placements, objects)):
                if v[:, 0].max() < -.145:
                    rear_exit[i] = True
    mujoco.mj_forward(model, data)
    final = cargo_vertices(model, data, placements, objects)
    return {'tilt_deg': angle, 'initial_fully_below_rim': initial_contained, 'initial_retained': retained,
            'rear_exit': rear_exit, 'rear_exit_count': sum(rear_exit),
            'initial_body_poses': initial_poses,
            'initial_bounding_boxes_m': [[v.min(0).tolist(), v.max(0).tolist()] for v in start],
            'final_vertex_means_in_basket_frame_m': [v.mean(0).tolist() for v in final]}


def main():
    objects = prepare_objects()
    results = []
    for mu in [.2, .5, 1.0]:
        model, placements = make_model(objects, 'mixed_three', mu)
        for angle in [15, 20, 30, 45, 60, 75]:
            row = {'scenario': 'mixed_three', 'mu': mu, **run_case(model, placements, objects, angle)}
            results.append(row)
            print(json.dumps({k: v for k, v in row.items() if k in ['scenario', 'mu', 'tilt_deg', 'initial_fully_below_rim', 'rear_exit_count']}), flush=True)
    capacity = []
    for scenario in OBJECT_NAMES:
        model, placements = make_model(objects, scenario, .5)
        row = {'scenario': scenario, **run_case(model, placements, objects, 0)}
        capacity.append(row)
        print('CAPACITY', scenario, row['initial_fully_below_rim'], flush=True)
    model, placements = make_model(objects, 'mixed_three', .5)
    sanity = run_case(model, placements, objects, 120)
    assert sanity['rear_exit_count'] == 3, sanity
    (OUT / 'contact_screen.json').write_text(json.dumps({
        'engine': f'MuJoCo {mujoco.__version__}',
        'scope': 'cargo-contact screening with prescribed gravity rotation; not Isaac Sim, not a whole-robot control test',
        'object_collision': 'convex hull of the supplied USD mesh, 0.5 kg each',
        'basket_collision': 'the actual 16 URDF collision shapes, fixed in its own frame',
        'time_step_s': DT, 'settle_s': SETTLE, 'tilt_ramp_s': RAMP, 'hold_s': HOLD,
        'damping': '2/s approximate linear damping, 4/s approximate principal-axis angular damping',
        'friction': 'both contacting bodies use the same tested sliding coefficient; these are sensitivity cases, not measured plastic values',
        'limitations': ['no leg controller or robot stability simulation', 'contact solver differs from PhysX',
                        'one deliberate packing arrangement per scenario; no capacity maximum or success probability inferred',
                        'slow tilt omits basket translational/rotational inertial forcing',
                        'initial-contained uses 3 mm geometric tolerance'],
        'objects': {k: {'dimensions_m': v['dimensions_m'].tolist()} for k, v in objects.items()},
        'mixed_trials': results, 'capacity_trials': capacity, 'sanity_check_120deg': sanity,
    }, indent=2) + '\n')


if __name__ == '__main__':
    main()
