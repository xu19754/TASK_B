"""Independent slow-tilt screen for the new CAD, excluding robot and arm."""
import json
import math
from pathlib import Path
import sys
import xml.etree.ElementTree as ET

import mujoco
import numpy as np
from scipy.spatial import ConvexHull

sys.path.insert(0, str(Path(__file__).resolve().parent))
import basket_arm_viewer as arm


def main():
    source = arm.base.geometry.ROBOT.find("link[@name='tail_basket']")
    root = ET.Element('mujoco', model='part2_independent_slow_tilt')
    ET.SubElement(root, 'compiler', angle='radian', inertiafromgeom='true')
    ET.SubElement(root, 'option', timestep='.001', gravity='0 0 -9.81', iterations='60', cone='elliptic')
    default = ET.SubElement(root, 'default')
    ET.SubElement(default, 'geom', friction='.5 .005 .0001', condim='3', solref='.004 1', solimp='.95 .99 .001')
    assets, world = ET.SubElement(root, 'asset'), ET.SubElement(root, 'worldbody')
    for i, c in enumerate(source.findall('collision')):
        mesh = c.find('geometry/mesh')
        ET.SubElement(assets, 'mesh', name=f'b_{i}', file=str(arm.VARIANT.parent / mesh.get('filename')), inertia='convex')
        ET.SubElement(world, 'geom', name=f'basket_{i}', type='mesh', mesh=f'b_{i}')
    vertices = []
    for i, name in enumerate(arm.base.OBJECT_NAMES):
        file = arm.base.ANALYSIS / 'screen_meshes' / f'{name}.obj'
        v, _ = arm.base.read_obj(file)
        vertices.append(v[ConvexHull(v).vertices])
        ET.SubElement(assets, 'mesh', name=name, file=str(file), inertia='convex')
        position = [[.075, 0, .030], [-.025, 0, .055], [.075, 0, .090]][i]
        rotation = np.eye(3) if i != 2 else np.array([[0, 0, 1], [1, 0, 0], [0, 1, 0]])
        body = ET.SubElement(world, 'body', name=f'cargo_{i}', pos=arm.base.numbers(position),
                             quat=arm.base.numbers(arm.base.quat_wxyz(rotation)))
        ET.SubElement(body, 'freejoint')
        ET.SubElement(body, 'geom', type='mesh', mesh=name, mass='.5')
    model = mujoco.MjModel.from_xml_string(ET.tostring(root, encoding='unicode'))
    bids = [model.body(f'cargo_{i}').id for i in range(3)]
    for b in bids:
        adr = model.jnt_dofadr[model.body_jntadr[b]]
        model.dof_damping[adr:adr + 3] = model.body_mass[b] * 2
        model.dof_damping[adr + 3:adr + 6] = model.body_inertia[b] * 4
    rows = []
    for mu in [.2, .5, 1.]:
        model.geom_friction[:, 0] = mu
        for angle in [0, 5, 10, 15, 20, 30]:
            data = mujoco.MjData(model)
            exited = [False] * 3
            for step in range(1000):
                fraction = np.clip((data.time - 2) / 5, 0, 1)
                theta = math.radians(angle) * fraction * fraction * (3 - 2 * fraction)
                model.opt.gravity[:] = [-9.81 * math.sin(theta), 0, -9.81 * math.cos(theta)]
                mujoco.mj_step(model, data, nstep=10)
                mujoco.mj_forward(model, data)
                for i, (b, v) in enumerate(zip(bids, vertices)):
                    p = v @ data.xmat[b].reshape(3, 3).T + data.xpos[b]
                    exited[i] |= bool(p[:, 0].max() < -.23)
            row = dict(friction=mu, angle_deg=angle, rear_exit_count=sum(exited), exited=exited)
            rows.append(row)
            print(row, flush=True)
    x = np.linspace(-.2235, .2235, 100001)
    width = 2 * np.minimum(.1235, .1086425 + .1 * (x + .2235))
    floor = .0015 + np.maximum(-.2105 - x, 0)
    volumes = {str(h): float(np.trapezoid(width * np.maximum(h - floor, 0), x) * 1000) for h in [.015, .08, .20]}
    result = dict(scope='Prescribed gravity direction, fixed basket, no robot/arm, no task bucket or score',
                  timestep_s=.001, gravity_update_s=.010, settle_s=2, ramp_s=5, hold_s=3,
                  cargo='Source-object convex hulls, 3 x 0.5 kg, same mixed initial packing as original screen',
                  cavity_slice_volume_litres=volumes,
                  volume_caveat='Only 15 mm is the lowest unobstructed sill; 80/200 mm volumes are geometric slices, not liquid retention or safe load',
                  cases=rows, rear_exit_definition='All source hull vertices behind local X=-0.23 m at any sampled time')
    (arm.OUT / 'tilt_screen.json').write_text(json.dumps(result, indent=2) + '\n')


if __name__ == '__main__':
    main()
