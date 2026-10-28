import importlib.util
import math
from pathlib import Path
import sys
import types
import unittest
from unittest.mock import patch

import numpy as np

from taskb_student.localization import Localization
from taskb_student.constants import CONTROL_DT, SPAWN_POSITION


class LocalizationTests(unittest.TestCase):
    def test_forward_reverse_and_initial_heading(self):
        loc = Localization((1, 2, 0.6), math.pi / 2)
        np.testing.assert_allclose(loc.update((0.2, 0, 0), 2), (1, 2.4, 0.6))
        np.testing.assert_allclose(loc.update((-0.2, 0, 0), 2), (1, 2, 0.6))

    def test_turn_then_forward_and_reset(self):
        loc = Localization()
        loc.update((0, 0, 0.5), math.pi)
        self.assertEqual(loc.position, SPAWN_POSITION)
        self.assertAlmostEqual(loc.yaw, math.pi / 2)
        np.testing.assert_allclose(loc.update((0.4, 0, 0), 1), (-10, -9.6, 0.6))
        loc.reset()
        self.assertEqual(loc.position, SPAWN_POSITION)
        self.assertEqual(loc.yaw, 0)

    def test_circular_motion_both_directions(self):
        for direction in (-1, 1):
            with self.subTest(direction=direction):
                loc = Localization((0, 0, 0.6))
                # Half circle of radius 0.5 m.
                loc.update((0.25, 0, direction * 0.5), 2 * math.pi)
                np.testing.assert_allclose(loc.position, (0, direction, 0.6), atol=1e-12)
                self.assertTrue(-math.pi <= loc.yaw < math.pi)

    def test_subdividing_command_preserves_result(self):
        loc = Localization()
        expected = loc.update((0.3, 0, -0.4), 2)
        loc.reset()
        for _ in range(100):
            loc.update((0.3, 0, -0.4))
        np.testing.assert_allclose(loc.position, expected, atol=1e-12)

    def test_clipping_deadband_and_default_timestep(self):
        loc = Localization()
        loc.update((0.1, 0, -0.1), 10)
        self.assertEqual(loc.position, SPAWN_POSITION)
        self.assertEqual(loc.yaw, 0)
        loc.update((5, 0, 0))
        self.assertAlmostEqual(loc.position[0], -10 + 0.4 * CONTROL_DT)
        loc.update((0, 0, -5), 1)
        self.assertAlmostEqual(loc.yaw, -0.6)

    def test_invalid_update_preserves_pose(self):
        for command, dt in [
            ((0.2, 0.1, 0), 1), ((float('nan'), 0, 0), 1),
            ((0, 0), 1), ((0.2, 0, 0), -1), ((0.2, 0, 0), float('inf')),
        ]:
            with self.subTest(command=command, dt=dt):
                loc = Localization()
                with self.assertRaises(ValueError):
                    loc.update(command, dt)
                self.assertEqual(loc.position, SPAWN_POSITION)
                self.assertEqual(loc.yaw, 0)

    def test_heading_correction_preserves_position_and_initial_pose(self):
        loc = Localization()
        loc.update((0.2, 0, 0), 1)
        position = loc.position
        loc.correct_heading(5 * math.pi / 2)
        self.assertAlmostEqual(loc.yaw, math.pi / 2)
        self.assertEqual(loc.position, position)
        with self.assertRaises(ValueError):
            loc.correct_heading(float("nan"))
        loc.reset()
        self.assertEqual(loc.position, SPAWN_POSITION)
        self.assertEqual(loc.yaw, 0)

    def test_solution_integrates_previous_command_before_navigation(self):
        # Isolate command timing from Torch, the trained policy and Isaac Sim.
        class Controller:
            def reset(self):
                pass

            def step(self, *args):
                return np.zeros(19)

        low_level = types.ModuleType('taskb_student.low_level')
        low_level.LowLevelController = Controller
        spec = importlib.util.spec_from_file_location(
            'demo._solution_under_test', Path(__file__).resolve().parents[1] / 'demo/solution.py'
        )
        solution = importlib.util.module_from_spec(spec)
        with patch.dict(sys.modules, {'taskb_student.low_level': low_level}):
            spec.loader.exec_module(solution)
        alg = solution.AlgSolution(command=[0.4, 0, 0], scan_before_start=False)
        seen_positions = []

        def select_command(obs, score):
            seen_positions.append(alg.localization.position)
            return alg.command

        alg.select_command = select_command
        response = alg.predicts({'proprio': {}}, 0)
        self.assertEqual(len(response['action']), 19)
        self.assertEqual(seen_positions[-1], SPAWN_POSITION)
        alg.command[:] = [0, 0, 0]
        alg.predicts({'proprio': {}}, 0)
        self.assertAlmostEqual(seen_positions[-1][0], -10 + 0.4 * CONTROL_DT)
        alg.predicts({'proprio': {}}, 0)
        self.assertEqual(seen_positions[-1], seen_positions[-2])
        alg.reset()
        alg.predicts({'proprio': {}}, 0)
        self.assertEqual(alg.localization.position, SPAWN_POSITION)

        # Exercise the real target-navigation branch and automatic pose updates.
        navigation = solution.AlgSolution(target_position=(-9.8, -10), scan_before_start=False)
        navigation.predicts({'proprio': {}}, 0)
        self.assertGreater(navigation._previous_command[0], 0)
        for _ in range(500):
            navigation.predicts({'proprio': {}}, 0)
            if navigation.motion.is_at_target(navigation.target_position):
                break
        self.assertTrue(navigation.motion.is_at_target(navigation.target_position))
        np.testing.assert_array_equal(navigation._previous_command, (0, 0, 0))
        navigation.reset()
        self.assertEqual(navigation.localization.position, SPAWN_POSITION)
        navigation.target_position = None
        navigation.command = (0, 0, 0.3)
        self.assertEqual(navigation.select_command({}, 0), (0, 0, 0.3))

        navigation.set_traversal([(-10, -10, 0)], [(-8, -10)])
        navigation.predicts({'proprio': {}}, 0)
        self.assertTrue(navigation.traversal.awaiting_pickup)
        np.testing.assert_array_equal(navigation._previous_command, (0, 0, 0))
        navigation.traversal.confirm_pickup()
        self.assertTrue(navigation.traversal.completed)
        navigation.reset()
        self.assertFalse(navigation.traversal.completed)
        self.assertEqual(navigation.traversal.active_obstacle_indices, (0,))

        # Disabling pickup works in the real demo orchestration too, including
        # bypassing an arm hook that would otherwise attempt a grasp.
        navigation.set_traversal([(-10, -10, 0)], [(-8, -10)], pickup_enabled=False)
        navigation.select_arm_target = lambda *args: self.fail("Grasp arm hook ran with pickup disabled")
        for _ in range(150):
            navigation.predicts({'proprio': {}}, 0)
        self.assertFalse(navigation.traversal.completed)
        navigation.predicts({'proprio': {}}, 0)
        self.assertTrue(navigation.traversal.completed)
        self.assertEqual(navigation.traversal.active_obstacle_indices, (0,))


if __name__ == '__main__':
    unittest.main()
