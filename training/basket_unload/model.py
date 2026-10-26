"""Independent free-base MuJoCo B2+Z1 unloading model.

URDF link frames, inertias, collision primitives and all 103 basket collision
parts are retained. Robot internal contacts are disabled, matching Task B;
``self_collision_pairs`` permits separate forbidden-intersection screening.
Source cargo triangle meshes retain their visual silhouette, while MuJoCo
uses their convex hull for contacts. This is not PhysX validation.
"""
from __future__ import annotations

from itertools import combinations_with_replacement
import json
import math
from pathlib import Path
import xml.etree.ElementTree as ET

import mujoco
import numpy as np

from taskb_student.constants import (
    ARM_Q0, JOINT_LOWER, JOINT_NAMES, JOINT_UPPER, LEG_Q0, URDF_PATH,
)

ROOT = Path(__file__).resolve().parents[2]
OBJECT_NAMES = ("004_sugar_box", "006_mustard_bottle", "011_banana")
COMBINATIONS = tuple(combinations_with_replacement(range(3), 3))
KP = np.array([360.] * 12 + [512., 768., 768., 512., 384., 256., 80.])
KD = np.array([5.] * 12 + [25.6] * 6 + [8.])
EFFORT = np.array([300.] * 12 + [30., 60., 30., 30., 30., 30., 5.])
VELOCITY = np.array([23.] * 12 + [3.] * 6 + [.25])


def _numbers(values):
    return " ".join(f"{float(x):.12g}" for x in np.asarray(values).ravel())


def _rotation(rpy):
    r, p, y = rpy
    cr, sr, cp, sp, cy, sy = math.cos(r), math.sin(r), math.cos(p), math.sin(p), math.cos(y), math.sin(y)
    return np.array([[cy*cp, cy*sp*sr-sy*cr, cy*sp*cr+sy*sr],
                     [sy*cp, sy*sp*sr+cy*cr, sy*sp*cr-cy*sr],
                     [-sp, cp*sr, cp*cr]])


def _origin(element):
    if element is None:
        return np.zeros(3), np.eye(3)
    return (np.fromstring(element.get("xyz", "0 0 0"), sep=" "),
            _rotation(np.fromstring(element.get("rpy", "0 0 0"), sep=" ")))


def _quaternion(rotation):
    q = np.empty(4)
    mujoco.mju_mat2Quat(q, np.ascontiguousarray(rotation).ravel())
    return q


def read_vertices(path):
    """Read source OBJ vertices in its body frame, in metres."""
    return np.array([list(map(float, line.split()[1:4]))
                     for line in Path(path).read_text().splitlines() if line.startswith("v ")])


def build_model(combination=(0, 1, 2), timestep=.002):
    """Return ``(MjModel, metadata)`` for any three cargo classes, with repeats.

    Actuator controls are absolute joint positions in ``JOINT_NAMES`` order.
    Native position-actuator damping is integrated by ``implicitfast``;
    actuator force ranges preserve robot.py limits. MuJoCo has no native
    hard joint-speed constraint: callers must use ``prepare_targets`` and
    monitor actual speeds, including externally driven overspeed.
    """
    combination = tuple(int(x) for x in combination)
    if len(combination) != 3 or any(x not in range(3) for x in combination):
        raise ValueError("combination must contain exactly three class indices in [0, 2]")
    robot = ET.parse(URDF_PATH).getroot()
    links = {link.get("name"): link for link in robot.findall("link")}
    joints = {joint.get("name"): joint for joint in robot.findall("joint")}
    children = {name: [] for name in links}
    for joint in joints.values():
        children[joint.find("parent").get("link")].append(joint)
    root = ET.Element("mujoco", model="B2Z1_v3_free_base_unload")
    ET.SubElement(root, "compiler", angle="radian", inertiafromgeom="auto", fusestatic="false")
    ET.SubElement(root, "option", timestep=str(timestep), gravity="0 0 -9.81",
                  integrator="implicitfast", solver="Newton", iterations="40", cone="elliptic")
    ET.SubElement(root, "size", memory="128M")
    visual = ET.SubElement(root, "visual")
    ET.SubElement(visual, "global", offwidth="1280", offheight="960")
    ET.SubElement(visual, "map", znear="0.01", zfar="100")
    ET.SubElement(visual, "headlight", ambient=".4 .4 .4", diffuse=".65 .65 .65", specular=".1 .1 .1")
    assets = ET.SubElement(root, "asset")
    ET.SubElement(assets, "texture", name="checker", type="2d", builtin="checker", width="256", height="256",
                  rgb1=".20 .23 .28", rgb2=".27 .30 .34")
    ET.SubElement(assets, "material", name="floor_material", texture="checker", texrepeat="8 8")
    world = ET.SubElement(root, "worldbody")
    ET.SubElement(world, "light", pos="1 -2 4", dir="-.2 .3 -1", diffuse=".8 .8 .8")
    # Contact bits: ground=1, robot=2, cargo=4. No robot-robot contacts.
    ET.SubElement(world, "geom", name="floor", type="plane", size="5 5 .05", material="floor_material",
                  contype="1", conaffinity="6", friction="1 .005 .0001", condim="3")
    ET.SubElement(world, "camera", name="overview", pos="-1.6 -1.8 1.35", xyaxes=".747 -.664 0 .22 .25 .94")
    meshes = {}
    collision_names = []
    collision_by_link = {}

    def mesh_asset(path, scale="1 1 1"):
        key = (str(path), scale)
        if key not in meshes:
            name = f"mesh_{len(meshes)}"
            ET.SubElement(assets, "mesh", name=name, file=str(path), scale=scale, inertia="convex")
            meshes[key] = name
        return meshes[key]

    def add_geom(body, item, name, collision, color):
        pos, rot = _origin(item.find("origin"))
        geom = item.find("geometry")[0]
        attributes = dict(name=name, pos=_numbers(pos), quat=_numbers(_quaternion(rot)),
                          group="3" if collision else "2", rgba=_numbers(color),
                          contype="2" if collision else "0", conaffinity="5" if collision else "0",
                          friction="1 .005 .0001", condim="3", solref=".004 1", solimp=".95 .99 .001")
        if geom.tag == "mesh":
            attributes.update(type="mesh", mesh=mesh_asset(URDF_PATH.parent / geom.get("filename"), geom.get("scale", "1 1 1")))
        elif geom.tag == "box":
            attributes.update(type="box", size=_numbers(np.fromstring(geom.get("size"), sep=" ") / 2))
        elif geom.tag == "sphere":
            attributes.update(type="sphere", size=geom.get("radius"))
        elif geom.tag == "cylinder":
            attributes.update(type="cylinder", size=f"{geom.get('radius')} {float(geom.get('length')) / 2}")
        else:
            raise ValueError(f"Unsupported URDF geometry {geom.tag}")
        ET.SubElement(body, "geom", **attributes)
        if collision:
            collision_names.append(name)
            collision_by_link.setdefault(body.get("name"), []).append(name)

    def add_link(name, parent, joint=None):
        if joint is None:
            pos, rot = np.array([0., 0., .6]), np.eye(3)
        else:
            pos, rot = _origin(joint.find("origin"))
        body = ET.SubElement(parent, "body", name=name, pos=_numbers(pos), quat=_numbers(_quaternion(rot)))
        if joint is None:
            ET.SubElement(body, "freejoint", name="robot_root")
        elif joint.get("type") != "fixed":
            limit = joint.find("limit")
            ET.SubElement(body, "joint", name=joint.get("name"), type="hinge", axis=joint.find("axis").get("xyz"),
                          limited="true", range=f"{limit.get('lower')} {limit.get('upper')}", damping="0", armature="0")
        inertial = links[name].find("inertial")
        if inertial is not None and float(inertial.find("mass").get("value")) > 0:
            xyz, rotation = _origin(inertial.find("origin"))
            inertia = inertial.find("inertia")
            tensor = np.array([[float(inertia.get("ixx")), float(inertia.get("ixy")), float(inertia.get("ixz"))],
                               [float(inertia.get("ixy")), float(inertia.get("iyy")), float(inertia.get("iyz"))],
                               [float(inertia.get("ixz")), float(inertia.get("iyz")), float(inertia.get("izz"))]])
            tensor = rotation @ tensor @ rotation.T
            ET.SubElement(body, "inertial", pos=_numbers(xyz), mass=inertial.find("mass").get("value"),
                          fullinertia=_numbers([tensor[0,0], tensor[1,1], tensor[2,2], tensor[0,1], tensor[0,2], tensor[1,2]]))
        for i, item in enumerate(links[name].findall("visual")):
            color = [.32, .35, .40, 1.] if not name.startswith("link") else [.60, .63, .67, 1.]
            explicit = item.find("material/color")
            if explicit is not None:
                color = np.fromstring(explicit.get("rgba"), sep=" ")
            if name == "tail_basket":
                color = [.12, .48, .75, 1.]
            add_geom(body, item, f"visual_{name}_{i}", False, color)
        for i, item in enumerate(links[name].findall("collision")):
            add_geom(body, item, f"collision_{name}_{i}", True, [.1, .65, .9, .25])
        if name == "link06":
            # MuJoCo camera local -Z -> link +X, +Y -> link +Z.
            # Its right axis +X -> link -Y, preserving an upright image.
            camera_rotation = np.array([[0., 0., -1.], [-1., 0., 0.], [0., 1., 0.]])
            fovy = math.degrees(2 * math.atan((20.955 * 480 / 640) / (2 * 15.)))
            ET.SubElement(body, "camera", name="ee_camera", pos=".03 0 .07", quat=_numbers(_quaternion(camera_rotation)), fovy=str(fovy))
        for child in children[name]:
            add_link(child.find("child").get("link"), body, child)

    add_link("base", world)
    actuator = ET.SubElement(root, "actuator")
    for i, name in enumerate(JOINT_NAMES):
        ET.SubElement(actuator, "position", name=f"target_{name}", joint=name, kp=str(KP[i]), kv=str(KD[i]),
                      ctrllimited="true", ctrlrange=f"{JOINT_LOWER[i]} {JOINT_UPPER[i]}",
                      forcelimited="true", forcerange=f"{-EFFORT[i]} {EFFORT[i]}")
    source_vertices = []
    for i, cls in enumerate(combination):
        path = ROOT / "reports/basket_feasibility/screen_meshes" / f"{OBJECT_NAMES[cls]}.obj"
        vertices = read_vertices(path)
        source_vertices.append(vertices)
        body = ET.SubElement(world, "body", name=f"cargo_{i}", pos=f"-.25 0 {1 + .3*i}")
        ET.SubElement(body, "freejoint", name=f"cargo_joint_{i}")
        ET.SubElement(body, "geom", name=f"cargo_geom_{i}", type="mesh", mesh=mesh_asset(path), mass=".5",
                      group="2", contype="4", conaffinity="7", condim="3", friction="1 .005 .0001",
                      solref=".004 1", solimp=".95 .99 .001",
                      rgba=_numbers([[.92,.29,.12,1.], [.95,.7,.1,1.], [.92,.88,.2,1.]][cls]))
    xml = ET.tostring(root, encoding="unicode")
    model = mujoco.MjModel.from_xml_string(xml)
    # MuJoCo map clipping values are multiples of model extent.
    model.vis.map.znear = .05 / model.stat.extent
    model.vis.map.zfar = 50. / model.stat.extent
    ids = lambda kind, names: np.array([mujoco.mj_name2id(model, kind, n) for n in names], dtype=np.int32)
    joint_ids = ids(mujoco.mjtObj.mjOBJ_JOINT, JOINT_NAMES)
    cargo_joints = ids(mujoco.mjtObj.mjOBJ_JOINT, [f"cargo_joint_{i}" for i in range(3)])
    cargo_ids = ids(mujoco.mjtObj.mjOBJ_BODY, [f"cargo_{i}" for i in range(3)])
    for bid, jid in zip(cargo_ids, cargo_joints):
        dof = model.jnt_dofadr[jid]
        # Same damping rates as formal assets, approximate across engines.
        model.dof_damping[dof:dof+3] = model.body_mass[bid] * 2
        model.dof_damping[dof+3:dof+6] = model.body_inertia[bid] * 4
    geom_ids = ids(mujoco.mjtObj.mjOBJ_GEOM, collision_names)
    by_link = {name: ids(mujoco.mjtObj.mjOBJ_GEOM, names) for name, names in collision_by_link.items()}
    arm_links = ["link02", "link03", "link04", "link05", "link06", "gripperStator", "gripperMover"]
    obstacle_links = ["base", "tail_basket"] + [f"{leg}_{part}" for leg in ("FR", "FL", "RR", "RL") for part in ("hip", "thigh", "calf")]
    pairs = [(int(a), int(b)) for arm in arm_links for a in by_link.get(arm, [])
             for obstacle in obstacle_links for b in by_link.get(obstacle, [])]
    metadata = dict(
        joint_names=tuple(JOINT_NAMES), joint_ids=joint_ids, joint_qpos_adr=model.jnt_qposadr[joint_ids].copy(),
        joint_dof_adr=model.jnt_dofadr[joint_ids].copy(), actuator_ids=ids(mujoco.mjtObj.mjOBJ_ACTUATOR, [f"target_{n}" for n in JOINT_NAMES]),
        joint_lower=np.array(JOINT_LOWER), joint_upper=np.array(JOINT_UPPER), kp=KP.copy(), kd=KD.copy(),
        effort=EFFORT.copy(), velocity=VELOCITY.copy(), nominal_q=np.array(LEG_Q0+ARM_Q0),
        base_id=int(model.body("base").id), basket_id=int(model.body("tail_basket").id),
        ee_camera_id=int(model.camera("ee_camera").id), cargo_body_ids=cargo_ids,
        cargo_qpos_adr=model.jnt_qposadr[cargo_joints].copy(), cargo_dof_adr=model.jnt_dofadr[cargo_joints].copy(),
        cargo_vertices=source_vertices, combination=combination,
        foot_body_ids=ids(mujoco.mjtObj.mjOBJ_BODY, [f"{leg}_foot" for leg in ("FR", "FL", "RR", "RL")]),
        collision_geom_ids=geom_ids, collision_by_link=by_link,
        self_collision_pairs=np.asarray(pairs, dtype=np.int32), xml=xml,
        camera_intrinsics=np.array([[15./20.955*640, 0, 320], [0, 15./20.955*640, 240], [0,0,1]]),
        camera_resolution=(640,480), camera_clip=(.05,50.), control_mode="absolute_position",
        assumptions=["MuJoCo cargo contacts use source-mesh convex hulls, including banana cavity approximation",
                     "Cargo mass 0.5 kg, basket mass 0.2 kg are configuration values, not measurements",
                     "MuJoCo friction 1.0 and damping approximate formal configuration, not PhysX equivalence",
                     "Actuator speed limits require target-rate limiting and actual-speed monitoring"],
    )
    return model, metadata


def prepare_targets(metadata, requested, previous, actual, dt=.02):
    """Rate-limit absolute targets; preserve the formal gripper lead limit."""
    requested = np.asarray(requested, dtype=float)
    if requested.shape != (19,) or not np.isfinite(requested).all():
        raise ValueError("requested must contain 19 finite absolute joint positions")
    requested = np.clip(requested, metadata["joint_lower"], metadata["joint_upper"])
    previous = np.asarray(previous)
    slew = metadata["velocity"] * dt
    targets = previous + np.clip(requested-previous, -slew, slew)
    if requested[18] > actual[18]:
        targets[18] = min(targets[18], max(previous[18]-slew[18], actual[18]+.04))
    return np.clip(targets, metadata["joint_lower"], metadata["joint_upper"])


def initialize_data(model, metadata):
    """Reset-only nominal initialization, never an unloading control action."""
    data = mujoco.MjData(model)
    data.qpos[metadata["joint_qpos_adr"]] = metadata["nominal_q"]
    data.ctrl[metadata["actuator_ids"]] = metadata["nominal_q"]
    mujoco.mj_forward(model, data)
    feet = data.xpos[metadata["foot_body_ids"]]
    # Put nominal feet on ground; base remains a free dynamic body afterward.
    data.qpos[2] += .04 - float(feet[:,2].mean())
    mujoco.mj_forward(model, data)
    return data


def self_collision_distances(model, data, metadata, max_distance=.05):
    """CPU forbidden-pair distances even though robot self-contact is off."""
    return np.array([mujoco.mj_geomDistance(model, data, int(a), int(b), max_distance, None)
                     for a, b in metadata["self_collision_pairs"]])


def validate_standing(output=None, seconds=10.):
    """CPU nominal-target diagnostic with cargo initialized far from robot."""
    model, meta = build_model()
    data = initialize_data(model, meta)
    # Isolate empty-basket standing: cargo still dynamic, away from robot.
    for i, adr in enumerate(meta["cargo_qpos_adr"]):
        data.qpos[adr:adr+3] = [2+i*.3, 2, .3]
    mujoco.mj_forward(model, data)
    start = data.qpos[:3].copy()
    min_height, max_tilt, max_speed = 1e6, 0., np.zeros(19)
    min_collision, dangerous_contacts = 1., 0
    foot_ids = set(int(i) for i in meta["foot_body_ids"])
    robot_ids = set(int(i) for i in model.geom_bodyid[meta["collision_geom_ids"]])
    for step in range(round(seconds/model.opt.timestep)):
        mujoco.mj_step(model, data)
        min_height = min(min_height, float(data.qpos[2]))
        up = data.xmat[meta["base_id"]].reshape(3,3)[2,2]
        max_tilt = max(max_tilt, math.degrees(math.acos(np.clip(up, -1, 1))))
        max_speed = np.maximum(max_speed, np.abs(data.qvel[meta["joint_dof_adr"]]))
        for contact in data.contact:
            ga, gb = contact.geom
            ba, bb = int(model.geom_bodyid[ga]), int(model.geom_bodyid[gb])
            if (ga == model.geom("floor").id and bb in robot_ids and bb not in foot_ids) or (gb == model.geom("floor").id and ba in robot_ids and ba not in foot_ids):
                dangerous_contacts += 1
        if step % 100 == 0:
            min_collision = min(min_collision, float(self_collision_distances(model,data,meta).min()))
    report = dict(engine=f"MuJoCo {mujoco.__version__}", asset=str(URDF_PATH.relative_to(ROOT)),
                  initialization="empty basket; source cargo placed away on floor at reset only",
                  duration_s=float(data.time), free_base=True, joints=model.nu, nq=model.nq, nv=model.nv,
                  robot_mass_kg=float(sum(model.body_mass[model.body_rootid==meta["base_id"]])),
                  basket_collision_parts=len(meta["collision_by_link"]["tail_basket"]),
                  min_base_height_m=min_height, max_base_tilt_deg=max_tilt,
                  base_displacement_m=float(np.linalg.norm(data.qpos[:3]-start)),
                  max_joint_speed_rad_s=max_speed.tolist(), max_speed_ratio=float(np.max(max_speed/meta["velocity"])),
                  minimum_forbidden_self_distance_m=min_collision, dangerous_ground_contact_samples=dangerous_contacts,
                  stable=bool(min_height>.35 and max_tilt<15 and dangerous_contacts==0),
                  assumptions=meta["assumptions"])
    if output:
        path=Path(output); path.parent.mkdir(parents=True,exist_ok=True)
        path.write_text(json.dumps(report,indent=2,ensure_ascii=False)+"\n",encoding="utf-8")
    return report


if __name__ == "__main__":
    print(json.dumps(validate_standing(ROOT/"runs/basket_unload/model_validation.json"), indent=2))
