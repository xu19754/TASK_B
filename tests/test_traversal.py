import math
import unittest

from taskb_student.constants import CONTROL_DT
from taskb_student.localization import Localization
from taskb_student.traversal import NoPathError, Traversal, plan_traversal


def segment_distance(start, end, center):
    dx, dy = end[0] - start[0], end[1] - start[1]
    length = dx * dx + dy * dy
    t = ((center[0] - start[0]) * dx + (center[1] - start[1]) * dy) / length if length else 0
    t = max(0, min(1, t))
    return math.hypot(start[0] + t * dx - center[0], start[1] + t * dy - center[1])


class TraversalTests(unittest.TestCase):
    def setUp(self):
        self.options = dict(robot_radius=0.2, obstacle_radii=0.3, safety_margin=0.1,
                            bounds=(-5, 5, -5, 5), grid_resolution=0.15)

    def test_route_avoids_item_until_pickup_then_can_cross_removed_item(self):
        targets = [(1, 0, math.pi), (-2, 0, 0)]
        obstacles = [(0, 0), (3, 2)]
        legs = plan_traversal((-2, 0), targets, obstacles, **self.options)
        self.assertGreater(len(legs[0].waypoints), 2)
        for start, end in zip(legs[0].waypoints, legs[0].waypoints[1:]):
            for obstacle in obstacles:
                self.assertGreater(segment_distance(start, end, obstacle), 0.6)
        self.assertEqual(legs[1].waypoints, ((1, 0), (-2, 0)))

    def test_live_route_aligns_pose_waits_for_pickup_and_finishes(self):
        localization = Localization((-2, 0, 0.6))
        targets = [(1, 0, math.pi), (-2, 0, -math.pi / 2)]
        obstacles = [(0, 0), (3, 2)]
        traversal = Traversal(localization, targets, obstacles, **self.options)
        with self.assertRaises(RuntimeError):
            traversal.confirm_pickup()
        for index, target in enumerate(targets):
            for _ in range(12000):
                before = localization.position
                command = traversal.compute_command()
                self.assertIsNone(traversal.blocked_reason)
                self.assertIn(index, traversal.active_obstacle_indices)
                if traversal.awaiting_pickup:
                    self.assertEqual(command, (0, 0, 0))
                    break
                localization.update(command)
                for active in traversal.active_obstacle_indices:
                    self.assertGreater(segment_distance(before, localization.position, obstacles[active]), 0.6)
            self.assertTrue(traversal.awaiting_pickup)
            self.assertLessEqual(math.dist(localization.position[:2], target[:2]), 0.05)
            error = (localization.yaw - target[2] + math.pi) % (2 * math.pi) - math.pi
            self.assertLessEqual(abs(error), 0.03)
            for _ in range(3):
                self.assertEqual(traversal.compute_command(), (0, 0, 0))
                self.assertEqual(traversal.target_index, index)
            self.assertEqual(traversal.confirm_pickup(), index)
            self.assertNotIn(index, traversal.active_obstacle_indices)
            with self.assertRaises(RuntimeError):
                traversal.confirm_pickup()
        self.assertTrue(traversal.completed)
        self.assertEqual(traversal.compute_command(), (0, 0, 0))
        self.assertEqual(traversal.active_obstacle_indices, ())

    def test_complete_eighteen_item_traversal(self):
        obstacles = [(x, y) for y in (-15, -10, -5) for x in (-18, -15, -12, -9, -6, -3)]
        targets = [(x, y - 1.2, math.pi / 2) for x, y in obstacles]
        localization = Localization((-18, -18, 0.6))
        traversal = Traversal(localization, targets, obstacles)
        legs = traversal.plan_remaining()
        self.assertEqual(len(legs), 18)
        self.assertEqual(traversal.active_obstacle_indices, tuple(range(18)))
        for leg in legs:
            for start, end in zip(leg.waypoints, leg.waypoints[1:]):
                for obstacle in obstacles[leg.target_index:]:
                    self.assertGreater(segment_distance(start, end, obstacle), 0.9)
        pickups = []
        for _ in range(40000):
            before = localization.position
            command = traversal.compute_command()
            self.assertIsNone(traversal.blocked_reason)
            if traversal.awaiting_pickup:
                pickups.append(traversal.confirm_pickup())
            if traversal.completed:
                break
            localization.update(command)
            for index in traversal.active_obstacle_indices:
                self.assertGreater(segment_distance(before, localization.position, obstacles[index]), 0.9)
        self.assertTrue(traversal.completed)
        self.assertEqual(pickups, list(range(18)))

    def test_blocked_target_stops_and_does_not_remove_item(self):
        loc = Localization((-2, 0, 0.6))
        nav = Traversal(loc, [(0, 0, 0)], [(0, 0)], **self.options)
        with self.assertRaises(NoPathError):
            nav.plan_remaining()
        self.assertEqual(nav.compute_command(), (0, 0, 0))
        self.assertIsNotNone(nav.blocked_reason)
        self.assertEqual(nav.active_obstacle_indices, (0,))
        self.assertFalse(nav.awaiting_pickup)
        nav.replan()
        self.assertIsNone(nav.blocked_reason)

    def test_obstacle_wall_is_unreachable(self):
        obstacles = [(0, y) for y in (-4.5, -3.5, -2.5, -1.5, -0.5, 0.5, 1.5, 2.5, 3.5, 4.5)]
        with self.assertRaises(NoPathError):
            plan_traversal((-2, 0), [(2, 0, 0)] * len(obstacles), obstacles, **self.options)

    def test_diagonal_segment_cannot_cut_through_obstacle(self):
        legs = plan_traversal((-2, -2), [(2, 2, 0)], [(0, 0)], **self.options)
        self.assertGreater(len(legs[0].waypoints), 2)
        for start, end in zip(legs[0].waypoints, legs[0].waypoints[1:]):
            self.assertGreater(segment_distance(start, end, (0, 0)), 0.6)

    def test_final_heading_wrap_and_reset(self):
        loc = Localization((0, 0, 0.6), math.pi - 0.05)
        nav = Traversal(loc, [(0, 0, -math.pi + 0.05)], [(3, 3)], **self.options)
        self.assertGreater(nav.compute_command()[2], 0)
        for _ in range(300):
            loc.update(nav.compute_command(), CONTROL_DT)
            if nav.awaiting_pickup:
                break
        self.assertTrue(nav.awaiting_pickup)
        nav.confirm_pickup()
        self.assertTrue(nav.completed)
        nav.reset()
        self.assertFalse(nav.completed)
        self.assertFalse(nav.awaiting_pickup)
        self.assertEqual(nav.active_obstacle_indices, (0,))

    def test_invalid_inputs_and_clearance(self):
        for targets, obstacles in [([], []), ([(1, 2)], [(0, 0)]),
                                   ([(1, 2, 0)], []), ([(1, 2, float("nan"))], [(0, 0)])]:
            with self.subTest(targets=targets), self.assertRaises(ValueError):
                plan_traversal((-2, 0), targets, obstacles, **self.options)
        for changes in ({"obstacle_radii": -1}, {"robot_radius": 0}, {"grid_resolution": 0},
                        {"bounds": (0, 0, 0, 0)}, {"obstacle_radii": [0.2, 0.3]}):
            with self.subTest(changes=changes), self.assertRaises(ValueError):
                plan_traversal((-2, 0), [(2, 0, 0)], [(0, 0)], **(self.options | changes))
        with self.assertRaises(NoPathError):
            plan_traversal((0, 0), [(2, 0, 0)], [(0, 0)], **self.options)
        with self.assertRaises(NoPathError):
            plan_traversal((-2, 0), [(4.9, 0, 0)], [(0, 0)], **self.options)

    def test_waiting_for_grasp_does_not_restart_motion_after_pose_drift(self):
        loc = Localization((0, 0, 0.6))
        nav = Traversal(loc, [(0, 0, 0)], [(3, 3)], **self.options)
        self.assertEqual(nav.compute_command(), (0, 0, 0))
        self.assertTrue(nav.awaiting_pickup)
        loc.update((0.4, 0, 0), 1)
        self.assertEqual(nav.compute_command(), (0, 0, 0))
        with self.assertRaises(RuntimeError):
            nav.confirm_pickup()
        self.assertEqual(nav.active_obstacle_indices, (0,))
        nav.replan()
        self.assertFalse(nav.awaiting_pickup)
        self.assertNotEqual(nav.compute_command(), (0, 0, 0))


if __name__ == "__main__":
    unittest.main()
