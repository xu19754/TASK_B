"""Observed-motion regression; synthetic dynamics do not replace Isaac testing."""

import importlib.util
import json
import math
from pathlib import Path
import sys
import types
import unittest
from unittest.mock import patch

import numpy as np

from taskb_student.constants import CONTROL_DT
from taskb_student.localization import Localization
from taskb_student.traversal import Traversal


def proprio(yaw=0, velocity=(0, 0, 0)):
    return {"base_quat": np.array([[math.cos(yaw / 2), 0, 0, math.sin(yaw / 2)]]),
            "base_lin_vel": np.array([velocity], dtype=float)}


def solution_module():
    class Controller:
        def reset(self):
            pass

        def step(self, *args):
            return np.zeros(19)

    low = types.ModuleType("taskb_student.low_level")
    low.LowLevelController = Controller
    spec = importlib.util.spec_from_file_location("demo._feedback_test", Path(__file__).resolve().parents[1] / "demo/solution.py")
    module = importlib.util.module_from_spec(spec)
    with patch.dict(sys.modules, {"taskb_student.low_level": low}):
        spec.loader.exec_module(module)
    return module


class FeedbackLocalizationTests(unittest.TestCase):
    def test_world_rotation_sideways_motion_and_no_velocity_deadband(self):
        loc = Localization((0, 0, 0.6))
        # Forward motion at +90 degrees goes to +Y; sideways motion goes to -X.
        obs = proprio(math.pi / 2, (0.02, 0.03, 0))
        loc.update_from_proprio(obs, dt=0)
        self.assertEqual(loc.position, (0, 0, 0.6))
        loc.update_from_proprio(obs, dt=2)
        np.testing.assert_allclose(loc.position, (-0.06, 0.04, 0.6), atol=1e-12)
        self.assertAlmostEqual(loc.yaw, math.pi / 2)

    def test_full_three_dimensional_orientation_and_quaternion_normalization(self):
        loc = Localization((0, 0, 0.6))
        # Pitch +90: body Z is world +X. Planar-only yaw rotation would miss this.
        obs = {"base_quat": np.array([math.sqrt(2), 0, math.sqrt(2), 0]),
               "base_lin_vel": np.array([0, 0, 0.2])}
        loc.update_from_proprio(obs, dt=0)
        loc.update_from_proprio(obs, dt=1)
        np.testing.assert_allclose(loc.position, (0.2, 0, 0.6), atol=1e-12)

    def test_trapezoid_coasting_and_yaw_wrap(self):
        loc = Localization((0, 0, 0.6))
        loc.update_from_proprio(proprio(0, (0.4, 0, 0)), dt=0)
        loc.update_from_proprio(proprio(0, (0, 0, 0)), dt=0.1)
        self.assertAlmostEqual(loc.position[0], 0.02)
        loc.update_from_proprio(proprio(math.pi - 0.01), dt=0)
        loc.update_from_proprio(proprio(-math.pi + 0.01), dt=0.1)
        self.assertAlmostEqual(loc.diagnostics()["measured_yaw_rate"], 0.2)
        self.assertFalse(loc.is_stationary)
        loc.reset()
        self.assertEqual(loc.diagnostics()["samples"], 0)
        self.assertIsNone(loc.diagnostics()["world_linear_velocity"])

    def test_invalid_feedback_does_not_mutate_or_fall_back(self):
        loc = Localization()
        with self.assertRaisesRegex(ValueError, "dt=0"):
            loc.update_from_proprio(proprio())
        loc.update_from_proprio(proprio(), dt=0)
        before = (loc.position, loc.yaw, loc.diagnostics())
        for obs, dt in (({}, 0.02), (proprio(0, (float("nan"), 0, 0)), 0.02),
                        ({**proprio(), "base_quat": [0, 0, 0, 0]}, 0.02),
                        ({**proprio(), "base_lin_vel": [[0, 0, 0], [0, 0, 0]]}, 0.02),
                        (proprio(), -1), (proprio(), float("inf"))):
            with self.subTest(obs=obs, dt=dt), self.assertRaises(ValueError):
                loc.update_from_proprio(obs, dt)
            self.assertEqual((loc.position, loc.yaw, loc.diagnostics()), before)

    def test_real_solution_uses_feedback_once_before_navigation_including_scan(self):
        module = solution_module()
        alg = module.AlgSolution(command=(0.4, 0, 0), scan_before_start=False)
        alg.predicts({"proprio": proprio()}, 0)
        for _ in range(100):  # Command says move; measured robot is stuck.
            alg.predicts({"proprio": proprio()}, 0)
        self.assertEqual(alg.localization.position, (-10, -10, 0.6))
        self.assertGreater(alg.command_prediction.position[0], -9.21)
        alg.predicts({"proprio": proprio(math.pi / 2, (0.2, 0, 0))}, 0)
        self.assertAlmostEqual(alg.localization.position[1], -10 + 0.002)
        alg.predicts({"proprio": proprio(math.pi / 2, (0.2, 0, 0))}, 0)
        self.assertAlmostEqual(alg.localization.position[1], -10 + 0.006)
        with self.assertRaisesRegex(ValueError, "base_lin_vel"):
            alg.predicts({"proprio": {}}, 0)
        alg.reset()
        alg.predicts({"proprio": proprio(math.pi / 2, (0.2, 0, 0))}, 0)
        self.assertEqual(alg.localization.position, (-10, -10, 0.6))
        # Scan frames receive the same feedback update even with zero commands.
        class Finder:
            ready, done = False, False
            arm_target = [0] * 7

            def compute_command(self, obs):
                return (0, 0, 0)
        alg.object_finder = Finder()
        alg.predicts({"proprio": proprio(math.pi / 2, (0.2, 0, 0))}, 0)
        self.assertAlmostEqual(alg.localization.position[1], -10 + 0.004)

    def test_arrival_requires_stopping_and_dwell_restarts_on_motion(self):
        loc = Localization()
        nav = Traversal(loc, [(-10, -10, 0)], [(-8, -10)], pickup_enabled=False)
        loc.update_from_proprio(proprio(0, (0.2, 0, 0)), dt=0)
        self.assertEqual(nav.compute_command(), (0, 0, 0))
        self.assertFalse(nav.awaiting_pickup)
        loc.update_from_proprio(proprio(), dt=CONTROL_DT)
        nav.compute_command()
        self.assertTrue(nav.awaiting_pickup)
        for _ in range(100):
            loc.update_from_proprio(proprio())
            nav.compute_command()
        self.assertAlmostEqual(nav.dwell_elapsed, 2)
        loc.update_from_proprio(proprio(0, (0.1, 0, 0)))
        nav.compute_command()
        self.assertEqual(nav.dwell_elapsed, 0)
        self.assertFalse(nav.completed)
        for _ in range(150):
            loc.update_from_proprio(proprio())
            nav.compute_command()
        self.assertTrue(nav.completed)

    def test_saved_seed42_route_with_underturn_and_lateral_slip(self):
        data = json.loads((Path(__file__).parent / "fixtures/traversal_seed42_drift.json").read_text())
        outcomes = {}
        for mode in ("command", "proprio"):
            loc = Localization()
            nav = Traversal(loc, [i["target_pose"] for i in data["items"]], [i["position"] for i in data["items"]],
                            obstacle_radii=[i["obstacle_radius"] for i in data["items"]], pickup_enabled=False,
                            **data["navigation"])
            xy, yaw = np.array([-10., -10.]), 0.
            world_velocity = np.zeros(2)
            visit_errors = []
            if mode == "proprio":
                loc.update_from_proprio(proprio(), dt=0)
            for step in range(45000):
                index = nav.target_index
                command = nav.compute_command()
                self.assertIsNone(nav.blocked_reason, (mode, step, nav.diagnostics()))
                if nav.target_index != index:
                    visit_errors.append(math.dist(xy, nav.targets[index][:2]))
                if nav.completed:
                    break
                # Deliberately imperfect motion: 18% speed loss, 22% underturn,
                # and lateral slip. Integrate endpoint velocities independently.
                yaw += 0.78 * command[2] * CONTROL_DT
                body = np.array([0.82 * command[0], 0.04 * command[0], 0.0])
                c, s = math.cos(yaw), math.sin(yaw)
                next_velocity = np.array([c*body[0] - s*body[1], s*body[0] + c*body[1]])
                xy += 0.5 * (world_velocity + next_velocity) * CONTROL_DT
                world_velocity = next_velocity
                if mode == "command":
                    loc.update(command)
                else:
                    loc.update_from_proprio(proprio(yaw, body))
                    self.assertLess(math.dist(loc.position[:2], xy), 1e-8)
            self.assertTrue(nav.completed, (mode, nav.diagnostics()))
            self.assertEqual(len(visit_errors), 18)
            self.assertEqual(len(nav.active_obstacle_indices), 18)
            outcomes[mode] = visit_errors
        self.assertGreater(max(outcomes["command"][9:]), 1.0)
        self.assertLessEqual(max(outcomes["proprio"]), 0.05)


if __name__ == "__main__":
    unittest.main()
