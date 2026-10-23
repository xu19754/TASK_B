from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
URDF_PATH = ROOT / "assets/b2z1/b2z1_0815_with_basket_v3.urdf"
POLICY_PATH = ROOT / "models/locomotion/policy.pt"
POLICY_SHA256 = "f9983f63040d18cae743c20b1f7792f87b39246551b168678d8216901df32dca"
URDF_SHA256 = "eb54aefe9c7b34ae18adb3599a6772d20c48b58358a0fd89dfa6ce4cc837e413"

TASK_ID = "Student-TaskB-B2Z1-v0"
NUM_OBJECTS = 18
TARGET_CENTER = (-3.0, -10.0)
TARGET_RADIUS = 1.0
TARGET_Z_RANGE = (0.0, 0.5)
SPAWN_POSITION = (-10.0, -10.0, 0.6)
CONTROL_DT = 0.02

LEG_JOINTS = [
    f"{leg}_{joint}_joint"
    for leg in ("FR", "FL", "RR", "RL")
    for joint in ("hip", "thigh", "calf")
]
ARM_JOINTS = ["joint1", "joint2", "joint3", "joint4", "joint5", "joint6", "jointGripper"]
JOINT_NAMES = LEG_JOINTS + ARM_JOINTS
LEG_Q0 = [-0.1, 0.8, -1.5, 0.1, 0.8, -1.5, -0.1, 0.8, -1.5, 0.1, 0.8, -1.5]
ARM_Q0 = [0.0, 1.48, -0.63, -0.84, 0.0, 1.57, 0.0]
LEG_ACTION_SCALE = [0.4, 0.45, 0.45] * 4
JOINT_LOWER = [-0.3, 0.0, -2.6] * 4 + [-2.618, 0.0, -4.78, -1.744, -1.727, -2.79, -1.57]
JOINT_UPPER = [0.3, 1.4, -0.6] * 4 + [2.618, 3.14, 0.0, 1.57, 1.727, 2.79, 0.0]
