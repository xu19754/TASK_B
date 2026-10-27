import math
import unittest

import numpy as np

from taskb_student.localization import Localization
from taskb_student.motion import Motion


class MotionTests(unittest.TestCase):
    def setUp(self):
        self.localization = Localization((0, 0, 0.6))
        self.motion = Motion(self.localization)

    def test_stop_within_tolerance_and_ignore_target_height(self):
        for target in ((0, 0), (0.1, 0), (0, 0, 5)):
            self.assertTrue(self.motion.is_at_target(target))
            self.assertEqual(self.motion.compute_command(target), (0, 0, 0))

    def test_forward_slowdown_and_deadband(self):
        far = self.motion.compute_command((2, 0))
        near = self.motion.compute_command((0.1001, 0))
        self.assertEqual(far, (0.4, 0, 0))
        self.assertTrue(0.1 < near[0] < far[0])
        np.testing.assert_allclose(Localization.prepare_command(near), near)

    def test_turn_direction_and_turn_in_place(self):
        for target, direction in (((0, 1), 1), ((0, -1), -1)):
            vx, vy, yaw_rate = self.motion.compute_command(target)
            self.assertEqual((vx, vy), (0, 0))
            self.assertGreater(direction * yaw_rate, 0)
        self.assertEqual(self.motion.compute_command((-1, 0))[0], 0)

    def test_small_heading_error_avoids_yaw_deadband(self):
        command = self.motion.compute_command((1, math.tan(0.04)))
        self.assertGreater(command[0], 0.1)
        self.assertGreater(command[2], 0.1)
        np.testing.assert_allclose(Localization.prepare_command(command), command)

    def test_heading_wrap_uses_short_turn(self):
        loc = Localization((0, 0, 0.6), math.pi - 0.05)
        target = (math.cos(-math.pi + 0.05), math.sin(-math.pi + 0.05))
        vx, _, yaw_rate = Motion(loc).compute_command(target)
        self.assertGreater(vx, 0)
        self.assertGreater(yaw_rate, 0)
        self.assertLess(yaw_rate, 0.6)

    def test_navigation_does_not_advance_localization(self):
        before = (self.localization.position, self.localization.yaw)
        for _ in range(10):
            self.motion.compute_command((1, 1))
        self.assertEqual((self.localization.position, self.localization.yaw), before)

    def test_closed_loop_reaches_targets_and_handles_target_changes(self):
        for target in ((2, 0), (2, 2), (-1, -1), (-1.2, -1), (0, 0)):
            with self.subTest(target=target):
                for _ in range(5000):
                    command = self.motion.compute_command(target)
                    self.assertTrue(0 <= command[0] <= 0.4)
                    self.assertEqual(command[1], 0)
                    self.assertLessEqual(abs(command[2]), 0.6)
                    if self.motion.is_at_target(target):
                        break
                    self.localization.update(command)
                self.assertTrue(self.motion.is_at_target(target))
                self.assertEqual(self.motion.compute_command(target), (0, 0, 0))

    def test_invalid_targets_and_tolerances(self):
        for target in ((1,), (1, 2, 3, 4), (float("nan"), 0), (0, float("inf"))):
            with self.subTest(target=target), self.assertRaises(ValueError):
                self.motion.compute_command(target)
        for tolerance in (0, -1, float("nan"), float("inf")):
            with self.subTest(tolerance=tolerance), self.assertRaises(ValueError):
                Motion(self.localization, position_tolerance=tolerance)


if __name__ == "__main__":
    unittest.main()
