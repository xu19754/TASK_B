"""Camera calibration and Z1 forward kinematics using only measured joint angles."""

from functools import lru_cache
import math
import xml.etree.ElementTree as ET

import numpy as np

from .constants import ARM_JOINTS, ARM_Q0, URDF_PATH

# Same mounts and world-axis convention as env_cfg.py. No scene/object queries.
HEAD_POSITION = (0.4, 0.0, 0.12)
WRIST_POSITION = (0.03, 0.0, 0.07)
SCAN_ARM_TARGET = tuple(ARM_Q0[:3] + [-0.10] + ARM_Q0[4:])


def axis_rotation(axis, angle):
    axis = np.asarray(axis, dtype=float)
    axis = axis / np.linalg.norm(axis)
    x, y, z = axis
    skew = np.array(((0, -z, y), (z, 0, -x), (-y, x, 0)))
    return np.eye(3) + math.sin(angle) * skew + (1 - math.cos(angle)) * (skew @ skew)


def head_camera_transform():
    transform = np.eye(4)
    transform[:3, 3] = HEAD_POSITION
    return transform


def validate_camera_transform(value):
    value = np.asarray(value, dtype=float)
    if (value.shape != (4, 4) or not np.isfinite(value).all()
            or not np.allclose(value[3], (0, 0, 0, 1))
            or not np.allclose(value[:3, :3].T @ value[:3, :3], np.eye(3), atol=1e-6)
            or not math.isclose(np.linalg.det(value[:3, :3]), 1, abs_tol=1e-6)):
        raise ValueError("Camera transform must be a finite rigid 4x4 camera-to-base transform.")
    return value


@lru_cache(maxsize=1)
def _arm_chain():
    robot = ET.parse(URDF_PATH).getroot()
    joints = {joint.find("child").get("link"): joint for joint in robot.findall("joint")}
    chain, link = [], "link06"
    while link != "base":
        joint = joints[link]
        origin = joint.find("origin")
        xyz = np.fromstring(origin.get("xyz", "0 0 0"), sep=" ") if origin is not None else np.zeros(3)
        rpy = np.fromstring(origin.get("rpy", "0 0 0"), sep=" ") if origin is not None else np.zeros(3)
        transform = np.eye(4)
        transform[:3, 3] = xyz
        transform[:3, :3] = (axis_rotation((0, 0, 1), rpy[2]) @ axis_rotation((0, 1, 0), rpy[1])
                             @ axis_rotation((1, 0, 0), rpy[0]))
        axis = joint.find("axis")
        axis = np.fromstring(axis.get("xyz"), sep=" ") if axis is not None else np.array((1, 0, 0))
        kind = joint.get("type")
        if kind not in ("fixed", "revolute", "continuous"):
            raise ValueError(f"Unsupported camera-chain joint: {joint.get('name')} ({kind})")
        chain.append((joint.get("name"), kind, transform, axis))
        link = joint.find("parent").get("link")
    return tuple(reversed(chain))


def wrist_camera_transform(joint_positions):
    """Camera world-convention axes in base frame; accepts 19 observed or 7 arm joints.

    Never substitute commanded angles for measured angles: camera pose changes during
    settling. URDF joint origins/axes and the fixed link06 camera mount define the chain.
    """
    if hasattr(joint_positions, "detach"):
        joint_positions = joint_positions.detach().cpu().numpy()
    joints = np.asarray(joint_positions, dtype=float).reshape(-1)
    if joints.shape == (19,):
        joints = joints[12:]
    if joints.shape != (7,) or not np.isfinite(joints).all():
        raise ValueError("Wrist camera needs 19 observed joints or 7 finite arm joint angles.")
    angles = dict(zip(ARM_JOINTS, joints))
    transform = np.eye(4)
    for name, kind, origin, axis in _arm_chain():
        transform = transform @ origin
        if kind != "fixed":
            motion = np.eye(4)
            motion[:3, :3] = axis_rotation(axis, float(angles[name]))
            transform = transform @ motion
    mount = np.eye(4)
    mount[:3, 3] = WRIST_POSITION
    return transform @ mount
