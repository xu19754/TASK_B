import hashlib
import json
import math
import xml.etree.ElementTree as ET

import numpy as np
import pytest
import torch

from taskb_student.constants import (
    ARM_Q0, CONTROL_DT, JOINT_LOWER, JOINT_NAMES, JOINT_UPPER, LEG_ACTION_SCALE,
    LEG_JOINTS, LEG_Q0, NUM_OBJECTS, POLICY_PATH, POLICY_SHA256, ROOT,
    SPAWN_POSITION, TARGET_CENTER, URDF_PATH, URDF_SHA256,
)
from taskb_student.geometry import build_arena_meshes, sample_object_positions
from taskb_student.low_level import LowLevelController, ObservationBuilder
from taskb_student.scoring import PlacementCounter, objects_inside_target


def outside_positions(batch=1):
    return torch.tensor([-10.0, -10.0, 0.15]).repeat(batch, NUM_OBJECTS, 1)


def nominal_proprio(yaw=0.0):
    return {
        "base_ang_vel": torch.zeros(1, 3),
        "projected_gravity": torch.tensor([[0.0, 0.0, -1.0]]),
        "base_quat": torch.tensor([[math.cos(yaw / 2), 0.0, 0.0, math.sin(yaw / 2)]]),
        "joint_pos": torch.tensor([LEG_Q0 + ARM_Q0]),
        "joint_vel": torch.zeros(1, 19),
    }


def test_objects_on_floor_do_not_score_merely_because_robot_is_near():
    # The scorer has no end-effector input: approaching any object cannot earn a point.
    counter = PlacementCounter(1)
    for _ in range(10):
        assert counter.update(outside_positions()).item() == 0


def test_each_delivered_object_scores_once_even_after_leaving_and_reentering():
    counter = PlacementCounter(1)
    positions = outside_positions()
    positions[0, 4] = torch.tensor([*TARGET_CENTER, 0.15])
    assert counter.update(positions).item() == 1
    assert counter.update(positions).item() == 0
    assert counter.update(outside_positions()).item() == 0
    assert counter.update(positions).item() == 0


def test_maximum_score_is_eighteen_and_partial_reset_is_isolated():
    counter = PlacementCounter(2)
    positions = torch.tensor([*TARGET_CENTER, 0.15]).repeat(2, NUM_OBJECTS, 1)
    torch.testing.assert_close(counter.update(positions), torch.tensor([18.0, 18.0]))
    counter.reset(torch.tensor([1]))
    torch.testing.assert_close(counter.update(positions), torch.tensor([0.0, 18.0]))
    counter.reset()
    torch.testing.assert_close(counter.update(positions), torch.tensor([18.0, 18.0]))


@pytest.mark.parametrize("dx,z,expected", [
    (0.0, 0.1, True), (1.0, 0.1, True), (1.01, 0.1, False),
    (0.0, 0.0, True), (0.0, 0.5, True), (0.0, 0.51, False),
    (0.0, -0.01, False), (float("nan"), 0.1, False),
])
def test_target_distance_and_height_boundaries(dx, z, expected):
    positions = outside_positions()
    positions[0, 0] = torch.tensor([TARGET_CENTER[0] + dx, TARGET_CENTER[1], z])
    assert bool(objects_inside_target(positions)[0, 0]) is expected


def test_score_is_relative_to_environment_origin_without_mutating_positions():
    local = outside_positions(2)
    local[:, 0] = torch.tensor([*TARGET_CENTER, 0.1])
    origins = torch.tensor([[25.0, 40.0, 1.0], [-25.0, -40.0, 2.0]])
    world = local + origins[:, None, :]
    original = world.clone()
    result = PlacementCounter(2).update(world, origins)
    torch.testing.assert_close(result, torch.ones(2))
    torch.testing.assert_close(world, original)


def test_layouts_are_seeded_and_avoid_robot_and_object_overlap():
    for seed in (0, 1, 42, 1234):
        positions = sample_object_positions(seed)
        assert positions.shape == (18, 3)
        np.testing.assert_array_equal(positions, sample_object_positions(seed))
        assert np.all((positions[:, :2] >= -15.0) & (positions[:, :2] <= -5.0))
        assert np.all(np.linalg.norm(positions[:, :2] - SPAWN_POSITION[:2], axis=1) >= 0.85)
        for i in range(18):
            assert np.all(np.linalg.norm(positions[i + 1:, :2] - positions[i, :2], axis=1) >= 0.4)
    assert not np.array_equal(sample_object_positions(42), sample_object_positions(43))


def test_bin_geometry_matches_scoring_region_and_does_not_fill_opening():
    meshes = dict(build_arena_meshes())
    np.testing.assert_allclose(meshes["ground"].bounds, [[-20, -20, -0.1], [0, 0, 0]])
    wall = meshes["bin_wall"]
    np.testing.assert_allclose(wall.bounds[:, 2], [0.05, 0.55])
    radius = np.linalg.norm(wall.vertices[:, :2] - TARGET_CENTER, axis=1)
    assert radius.min() == pytest.approx(0.98)
    assert radius.max() == pytest.approx(1.0)


def test_default_robot_uses_basket_variant_and_preserves_original_assets():
    manifest = json.loads((ROOT / "assets/b2z1/SOURCE.json").read_text())
    assert URDF_PATH == ROOT / manifest["runtime_urdf"]
    assert URDF_SHA256 == manifest["runtime_urdf_sha256"]
    assert URDF_PATH.name == "b2z1_0815_with_basket_v3.urdf"
    assert hashlib.sha256(URDF_PATH.read_bytes()).hexdigest() == URDF_SHA256
    robot = ET.parse(URDF_PATH).getroot()
    reference = ROOT / manifest["reference_urdf"]
    assert hashlib.sha256(reference.read_bytes()).hexdigest() == manifest["canonical_urdf_sha256"]
    nodes = {(node.tag, node.attrib.get("name")): node for node in robot}
    for node in ET.parse(reference).getroot():
        assert ET.tostring(nodes[node.tag, node.attrib.get("name")]).strip() == ET.tostring(node).strip()
    moving = {joint.attrib["name"]: joint for joint in robot.findall("joint") if joint.attrib["type"] != "fixed"}
    assert set(moving) == set(JOINT_NAMES)
    for i, name in enumerate(JOINT_NAMES):
        limit = moving[name].find("limit")
        assert float(limit.attrib["lower"]) == JOINT_LOWER[i]
        assert float(limit.attrib["upper"]) == JOINT_UPPER[i]
    for mesh in robot.findall(".//mesh"):
        assert (URDF_PATH.parent / mesh.attrib["filename"]).is_file()
    for name, digest in manifest["mesh_sha256"].items():
        assert hashlib.sha256((URDF_PATH.parent / "meshes" / name).read_bytes()).hexdigest() == digest


def test_default_basket_is_fixed_and_has_valid_200g_shell_inertia():
    robot = ET.parse(URDF_PATH).getroot()
    basket = robot.find("link[@name='tail_basket']")
    mount = robot.find("joint[@name='base_to_tail_basket']")
    assert basket is not None and mount is not None
    assert mount.attrib["type"] == "fixed"
    assert mount.find("parent").attrib["link"] == "base"
    assert mount.find("child").attrib["link"] == "tail_basket"
    assert float(basket.find("inertial/mass").attrib["value"]) == pytest.approx(0.2)
    inertia = basket.find("inertial/inertia").attrib
    matrix = np.array([
        [inertia["ixx"], inertia["ixy"], inertia["ixz"]],
        [inertia["ixy"], inertia["iyy"], inertia["iyz"]],
        [inertia["ixz"], inertia["iyz"], inertia["izz"]],
    ], dtype=float)
    principal = np.linalg.eigvalsh(matrix)
    assert np.all(principal > 0)
    assert principal[-1] <= principal[:2].sum()
    properties = json.loads((ROOT / json.loads((ROOT / "assets/b2z1/SOURCE.json").read_text())["runtime_variant_manifest"]).read_text())
    assert properties["revision"] == "v3"
    assert len(basket.findall("collision")) == properties["collision_convex_parts"] == 103
    for name, digest in properties["mesh_sha256"].items():
        assert hashlib.sha256((URDF_PATH.parent / "meshes" / name).read_bytes()).hexdigest() == digest
    assert hashlib.sha256((ROOT / properties["source"]).read_bytes()).hexdigest() == properties["source_sha256"]
    assert basket.find("visual/geometry/mesh").attrib["filename"] == "meshes/basket_v3/basket_visual.obj"


def test_policy_contract_matches_joint_order_and_scaling():
    assert hashlib.sha256(POLICY_PATH.read_bytes()).hexdigest() == POLICY_SHA256
    manifest = json.loads((POLICY_PATH.parent / "deployment_manifest.json").read_text())
    assert manifest["current_obs_dim"] == 52
    assert manifest["num_actions"] == 12
    contract = manifest["environment_contract"]
    assert contract["leg_joint_names"] == LEG_JOINTS
    assert contract["leg_action_scale"] == LEG_ACTION_SCALE
    assert [contract["leg_default_q0_by_joint"][name] for name in LEG_JOINTS] == LEG_Q0
    assert contract["fixed_arm_default_q0"] == ARM_Q0


def test_heading_reference_latches_on_entry_and_reversal():
    builder = ObservationBuilder()
    initial = builder.build(nominal_proprio(yaw=0.6), (0.2, 0, 0))
    assert initial.shape == (52,)
    np.testing.assert_allclose(initial[50:52], [0, 0], atol=1e-6)
    turn = builder.build(nominal_proprio(yaw=0.8), (0.2, 0, 0))
    np.testing.assert_allclose(turn[50:52], [math.sin(0.2), math.cos(0.2) - 1.0], atol=1e-6)
    reverse = builder.build(nominal_proprio(yaw=0.8), (-0.2, 0, 0))
    np.testing.assert_allclose(reverse[50:52], [0, 0], atol=1e-6)
    turning = builder.build(nominal_proprio(yaw=0.9), (0, 0, 0.3))
    np.testing.assert_array_equal(turning[50:52], [0, 0])
    standing = builder.build(nominal_proprio(), (0, 0, 0))
    assert standing[45] == 0


def test_invalid_motion_and_invalid_state_are_rejected():
    with pytest.raises(ValueError, match="cmd_y"):
        ObservationBuilder().build(nominal_proprio(), (0.2, 0.2, 0))
    proprio = nominal_proprio()
    proprio["joint_pos"][0, 0] = float("nan")
    with pytest.raises(ValueError):
        ObservationBuilder().build(proprio, (0.2, 0, 0))


def test_real_jit_inference_produces_nineteen_bounded_targets_and_resets():
    controller = LowLevelController()
    proprio = nominal_proprio()
    first = controller.step(proprio, (0, 0, 0))
    assert first.shape == (19,)
    for _ in range(30):
        targets = controller.step(proprio, (0.2, 0, 0))
        assert np.isfinite(targets).all()
        assert np.all(targets >= np.asarray(JOINT_LOWER) - 1e-6)
        assert np.all(targets <= np.asarray(JOINT_UPPER) + 1e-6)
    controller.reset()
    np.testing.assert_allclose(controller.step(proprio, (0, 0, 0)), first, atol=1e-6)
    previous_arm = controller.arm_target.copy()
    requested = np.asarray(ARM_Q0) - 0.5
    targets = controller.step(proprio, (0, 0, 0), requested)
    assert np.max(np.abs(targets[12:] - previous_arm)) <= 3.0 * CONTROL_DT + 1e-6


def test_previous_action_preserves_training_contract_when_joint_target_is_clipped():
    controller = LowLevelController()
    controller.policy = lambda observation: torch.full((1, 12), 10.0)
    targets = controller.step(nominal_proprio(), (0, 0, 0))
    np.testing.assert_allclose(targets[:12], JOINT_UPPER[:12])
    np.testing.assert_allclose(controller.builder.previous_action, 10.0)


def test_gripper_slew_is_slow_and_closing_target_cannot_run_ahead_of_contact():
    controller = LowLevelController()
    proprio = nominal_proprio()
    controller.arm_target[6] = -0.8
    proprio["joint_pos"][0, 18] = -0.8
    for _ in range(100):
        previous = controller.arm_target[6]
        targets = controller.step(proprio, (0, 0, 0), ARM_Q0)
        assert targets[18] <= -0.76 + 1e-6
        assert abs(targets[18] - previous) <= 0.25 * CONTROL_DT + 1e-6
    proprio["joint_pos"][0, 18] = -1.2
    previous = controller.arm_target[6]
    targets = controller.step(proprio, (0, 0, 0), ARM_Q0)
    assert abs(targets[18] - previous) <= 0.25 * CONTROL_DT + 1e-6


def test_locomotion_reset_preserves_arm_preload_and_clears_leg_history():
    controller = LowLevelController()
    controller.arm_target[6] = -0.57
    controller.builder.previous_action[:] = 0.5
    controller.builder.phase = 0.6
    before = controller.arm_target.copy()
    controller.reset_locomotion()
    np.testing.assert_array_equal(controller.arm_target, before)
    np.testing.assert_array_equal(controller.builder.previous_action, np.zeros(12))
    assert controller.builder.phase == 0
