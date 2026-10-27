"""Check portable assets and the actual supplied policy without Isaac Sim."""

import hashlib
from pathlib import Path
import sys
import xml.etree.ElementTree as ET

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import torch

from taskb_student.constants import (
    ARM_Q0, JOINT_NAMES, LEG_Q0, POLICY_PATH, POLICY_SHA256, ROOT, URDF_PATH, URDF_SHA256,
)
from taskb_student.low_level import LowLevelController


def main():
    for path, expected in ((URDF_PATH, URDF_SHA256), (POLICY_PATH, POLICY_SHA256)):
        if hashlib.sha256(path.read_bytes()).hexdigest() != expected:
            raise RuntimeError(f"Checksum mismatch: {path}")
        print(f"OK checksum: {path.relative_to(ROOT)}")
    robot = ET.parse(URDF_PATH).getroot()
    movable = {j.attrib["name"] for j in robot.findall("joint") if j.attrib["type"] != "fixed"}
    if movable != set(JOINT_NAMES):
        raise RuntimeError("Robot joint names do not match the 19-DoF contract.")
    for mesh in robot.findall(".//mesh"):
        path = URDF_PATH.parent / mesh.attrib["filename"]
        if not path.is_file():
            raise FileNotFoundError(path)
    basket = robot.find("link[@name='tail_basket']")
    mount = robot.find("joint[@name='base_to_tail_basket']")
    if (basket is None or mount is None or mount.attrib["type"] != "fixed"
            or mount.find("parent").attrib["link"] != "base"
            or mount.find("child").attrib["link"] != "tail_basket"):
        raise RuntimeError("The default robot must include the basket fixed to base.")
    mass = float(basket.find("inertial/mass").attrib["value"])
    if mass != 0.2:
        raise RuntimeError(f"Expected a 0.2 kg basket, got {mass} kg.")
    print(f"OK rear basket: {mass:.3f} kg, {len(basket.findall('collision'))} collision shapes")
    for name in ("004_sugar_box.usd", "006_mustard_bottle.usd", "011_banana.usd"):
        if not (ROOT / "assets/objects" / name).is_file():
            raise FileNotFoundError(name)
    print("OK robot joints and asset paths")
    proprio = {
        "base_ang_vel": torch.zeros(1, 3),
        "projected_gravity": torch.tensor([[0.0, 0.0, -1.0]]),
        "base_quat": torch.tensor([[1.0, 0.0, 0.0, 0.0]]),
        "joint_pos": torch.tensor([LEG_Q0 + ARM_Q0]),
        "joint_vel": torch.zeros(1, 19),
    }
    controller = LowLevelController()
    for command in ((0.0, 0.0, 0.0), (0.2, 0.0, 0.0), (0.0, 0.0, 0.3)):
        action = controller.step(proprio, command)
        if action.shape != (19,) or not torch.isfinite(torch.from_numpy(action)).all():
            raise RuntimeError("Policy inference failed.")
    print("OK actual TorchScript policy: 52 observations -> 12 leg actions -> 19 joint targets")
    print("Package checks passed. These checks do not run physics or certify locomotion.")


if __name__ == "__main__":
    main()
