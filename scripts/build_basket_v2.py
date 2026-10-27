"""Recover part 2's cached CAD mesh, verify it and make a separate URDF.

Build dependency: meshpy (TetGen). Runtime viewer does not need this package.
The original CAD and both earlier URDFs are read-only inputs.
"""
from pathlib import Path
import hashlib
import json
import sys
import xml.etree.ElementTree as ET

import meshpy.tet as tet
import numpy as np
from scipy.spatial import ConvexHull

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'tools/vendor'))
from sldprt.blocks import find_block, read_blocks
from sldprt.savedmesh import read_saved_mesh

ASSET = ROOT / 'assets/b2z1'
OUT = ROOT / 'reports/basket_v2'
MESH = ASSET / 'meshes/basket_v2'
SOURCE = Path('/Users/yxh/Desktop/share_wind/零件2.SLDPRT')
URDF = ASSET / 'b2z1_0815_with_basket_v2.urdf'
MASS = .2


def nums(values):
    return ' '.join(f'{x:.12g}' for x in np.asarray(values).ravel())


def write_obj(path, v, f):
    path.write_text('# metres; source CAD saved tessellation\n' + '\n'.join(
        ['v ' + nums(p) for p in v] + ['f ' + ' '.join(str(int(x) + 1) for x in face) for face in f]) + '\n')


def properties(v, f):
    ref = v.mean(0)
    t = v[f] - ref
    weights = np.einsum('ij,ij->i', t[:, 0], np.cross(t[:, 1], t[:, 2])) / 6
    summed = t.sum(1)
    volume = weights.sum()
    first = np.einsum('i,ij->j', weights, summed) / 4
    second = sum(np.einsum('i,ij,ik->jk', weights, t[:, k], t[:, k]) for k in range(3)) / 20
    second += np.einsum('i,ij,ik->jk', weights, summed, summed) / 20
    if volume < 0:
        volume, first, second = -volume, -first, -second
    centre = first / volume
    covariance = second / volume - np.outer(centre, centre)
    inertia = MASS * (np.trace(covariance) * np.eye(3) - covariance)
    return float(volume), ref + centre, inertia


def ray_inside(points, v, f):
    direction = np.array([.7593, .4481, .4712])
    direction /= np.linalg.norm(direction)
    tri = v[f]
    e1, e2 = tri[:, 1] - tri[:, 0], tri[:, 2] - tri[:, 0]
    h = np.cross(direction, e2)
    det = np.einsum('ij,ij->i', e1, h)
    inv = np.divide(1., det, out=np.zeros_like(det), where=np.abs(det) > 1e-12)
    s = points[:, None] - tri[None, :, 0]
    u = np.einsum('pfi,fi->pf', s, h) * inv
    q = np.cross(s, e1)
    vv = np.einsum('pfi,i->pf', q, direction) * inv
    t = np.einsum('pfi,fi->pf', q, e2) * inv
    return np.sum((np.abs(det) > 1e-12) & (u > 0) & (vv > 0) & (u + vv < 1) & (t > 1e-10), axis=1) % 2 == 1


def floor_profile(v, f):
    """Upper floor section at Y=0; omit vertical front/rear wall sections."""
    heights = {}
    for tri in v[f]:
        crossings = []
        for p, q in zip(tri, np.roll(tri, -1, axis=0)):
            if p[1] * q[1] < 0:
                crossings.append(p + (q - p) * (-p[1] / (q[1] - p[1])))
            elif abs(p[1]) < 1e-10:
                crossings.append(p)
        if len(crossings) < 2 or np.ptp(np.asarray(crossings)[:, 0]) < 1e-9:
            continue
        for p in crossings:
            if v[:, 0].min() - 1e-8 <= p[0] <= .2235 + 1e-8 and .001 <= p[2] < .05:
                key = round(float(p[0]), 10)
                heights[key] = max(heights.get(key, 0.), float(p[2]))
    return [[x, z] for x, z in sorted(heights.items())]


def main(revision='v2'):
    global OUT, MESH, URDF
    if revision != 'v2':
        if not revision.isalnum():
            raise ValueError('Revision must be alphanumeric')
        OUT = ROOT / 'reports' / ('basket_' + revision)
        MESH = ASSET / 'meshes' / ('basket_' + revision)
        URDF = ASSET / ('b2z1_0815_with_basket_' + revision + '.urdf')
    OUT.mkdir(exist_ok=True)
    MESH.mkdir(parents=True, exist_ok=True)
    data = SOURCE.read_bytes()
    if (OUT / 'geometry.json').exists():
        previous = json.loads((OUT / 'geometry.json').read_text())
        if previous.get('source_sha256') != hashlib.sha256(data).hexdigest():
            raise FileExistsError(f'{revision} already preserves a different CAD revision; use a new --revision')
    blocks = read_blocks(data)
    assert find_block(blocks, 'Contents/DisplayLists').verify()
    mesh = read_saved_mesh(blocks)
    assert mesh and mesh.is_closed()
    source_v = np.array(mesh.vertices) / 1000
    low, high = source_v.min(0), source_v.max(0)
    v = np.c_[source_v[:, 0] - (low[0] + high[0]) / 2,
              (low[2] + high[2]) / 2 - source_v[:, 2], source_v[:, 1] - low[1]]
    f = np.array(mesh.triangles)
    volume, centre, inertia = properties(v, f)
    assert np.linalg.eigvalsh(inertia).min() > 0
    preview = find_block(blocks, 'PreviewPNG')
    if preview:
        (OUT / 'source_preview.png').write_bytes(preview.inflate())
    write_obj(MESH / 'basket_visual.obj', v, f)
    info = tet.MeshInfo()
    info.set_points(v)
    info.set_facets(f.tolist())
    tets = tet.build(info, options=tet.Options('pYQ'))
    tv, te = np.array(tets.points), np.array(tets.elements)
    groups = [set(map(int, t)) for t in te]
    volumes = [ConvexHull(tv[sorted(g)]).volume for g in groups]
    assert abs(sum(volumes) - volume) < 1e-12
    # Merge only adjacent pieces whose convex hull adds no measurable volume.
    while True:
        merged = False
        for i in range(len(groups)):
            for j in range(i + 1, len(groups)):
                if len(groups[i] & groups[j]) < 3:
                    continue
                ids = groups[i] | groups[j]
                hull = ConvexHull(tv[sorted(ids)])
                if abs(hull.volume - volumes[i] - volumes[j]) < 1e-12:
                    groups[i] = ids
                    volumes[i] += volumes[j]
                    groups.pop(j)
                    volumes.pop(j)
                    merged = True
                    break
            if merged:
                break
        if not merged:
            break
    hulls = []
    for i, g in enumerate(groups):
        cv = tv[sorted(g)]
        hull = ConvexHull(cv)
        cf = hull.simplices.copy()
        tri = cv[cf]
        inward = np.einsum('ij,ij->i', np.cross(tri[:, 1] - tri[:, 0], tri[:, 2] - tri[:, 0]),
                          tri.mean(1) - cv.mean(0)) < 0
        cf[inward] = cf[inward][:, [0, 2, 1]]
        write_obj(MESH / f'collision_{i:02}.obj', cv, cf)
        hulls.append(hull)
    samples = np.random.default_rng(20261023).uniform(v.min(0) - .003, v.max(0) + .003, (30000, 3))

    def collision_inside(points):
        return np.any([np.all(points @ h.equations[:, :3].T + h.equations[:, 3] <= 1e-10, axis=1)
                       for h in hulls], axis=0)

    assert np.array_equal(collision_inside(samples), ray_inside(samples, v, f))
    front_clear = np.array([[x, y, z] for x in [.224, .225, .226]
                            for y in np.linspace(-.12, .12, 13) for z in [.085, .12, .19]])
    rear_clear = np.array([[x, y, z] for x in [-.226, -.225, -.224]
                           for y in np.linspace(-.10, .10, 13) for z in [.02, .08, .19]])
    assert not collision_inside(np.r_[front_clear, rear_clear]).any()
    original = ET.parse(ASSET / 'b2z1_0815.urdf').getroot()
    new = ET.parse(ASSET / 'b2z1_0815_with_basket.urdf').getroot()
    link = new.find("link[@name='tail_basket']")
    link.find('inertial/origin').set('xyz', nums(centre))
    link.find('inertial/mass').set('value', str(MASS))
    for name, i, j in [('ixx', 0, 0), ('iyy', 1, 1), ('izz', 2, 2), ('ixy', 0, 1), ('ixz', 0, 2), ('iyz', 1, 2)]:
        link.find('inertial/inertia').set(name, f'{inertia[i, j]:.12g}')
    mesh_relative = MESH.relative_to(ASSET).as_posix()
    link.find('visual/geometry/mesh').set('filename', mesh_relative + '/basket_visual.obj')
    for c in link.findall('collision'):
        link.remove(c)
    for i in range(len(groups)):
        c = ET.SubElement(link, 'collision', name=f'basket_{revision}_{i:02}')
        ET.SubElement(c, 'origin', xyz='0 0 0', rpy='0 0 0')
        ET.SubElement(ET.SubElement(c, 'geometry'), 'mesh', filename=f'{mesh_relative}/collision_{i:02}.obj', scale='1 1 1')
    # Earlier robot elements, including all 19 joint contracts, stay identical.
    for el in original:
        counterpart = next(x for x in new if x.tag == el.tag and x.get('name') == el.get('name'))
        assert ET.tostring(el).strip() == ET.tostring(counterpart).strip()
    ET.indent(new, space='  ')
    ET.ElementTree(new).write(URDF, encoding='utf-8', xml_declaration=True)
    result = dict(source=str(SOURCE), source_sha256=hashlib.sha256(data).hexdigest(),
                  variant=str(URDF), variant_sha256=hashlib.sha256(URDF.read_bytes()).hexdigest(),
                  converter='XRTC5/sldprt-export 9c91ea0fd94090ce88efd6d37b61ba091243c715',
                  dimensions_m=np.ptp(v, axis=0).tolist(), wall_thickness_m=.0015,
                  source_mesh_vertices=len(v), source_mesh_triangles=len(f), cad_mesh_closed=True, crc_verified=True,
                  rear_sill_m=.015, rear_clear_width_m=.210, front_sill_m=.080, front_clear_width_m=.247,
                  front_window_clear_height_m=float(np.ptp(v, axis=0)[2] - .080),
                  shell_volume_m3=volume, mass_kg=MASS, mass_assumption='Previous requested 0.2 kg retained; not measured',
                  com_m=centre.tolist(), inertia_kg_m2=inertia.tolist(), source_to_basket_rotation=[[1, 0, 0], [0, 0, -1], [0, 1, 0]],
                  source_to_basket_translation_m=[-(low[0] + high[0]) / 2, (low[2] + high[2]) / 2, -low[1]],
                  collision_tetrahedra=len(te), collision_convex_parts=len(groups),
                  collision_volume_m3=float(sum(volumes)), collision_visual_random_samples=len(samples),
                  collision_visual_mismatches=0, front_and_rear_aperture_samples_clear=True,
                  fixed_mount_xyz_m=[-.30, 0, .13], unchanged_controllable_joints=19,
                  display_mesh_limit='Checks validate saved planar tessellation, not a licensed CAD B-rep reconstruction')
    (OUT / 'geometry.json').write_text(json.dumps(result, ensure_ascii=False, indent=2) + '\n')
    (OUT / 'cad_mesh.json').write_text(json.dumps(dict(vertices_m=v.tolist(), faces=f.tolist()), indent=2) + '\n')
    if revision == 'v2':
        result['rear_chamfer_run_and_rise_m'] = [.013, .013]
    # The mid-plane section supplies actual floor heights for later revisions;
    # don't reuse the old 45 degree chamfer formula on a changed CAD profile.
    result['floor_profile_xz_m'] = floor_profile(v, f)
    result['bounds_m'] = [v.min(0).tolist(), v.max(0).tolist()]
    result['revision'] = revision
    snapshot = OUT / 'source'
    snapshot.mkdir(exist_ok=True)
    (snapshot / SOURCE.name).write_bytes(data)
    result['source_snapshot'] = str(snapshot / SOURCE.name)
    (OUT / 'geometry.json').write_text(json.dumps(result, ensure_ascii=False, indent=2) + '\n')
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--revision', default='v2')
    args = parser.parse_args()
    main(args.revision)
