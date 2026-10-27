import importlib.util
import json
import math
from pathlib import Path
import sys
import tempfile
import types
import unittest
from unittest.mock import patch

import numpy as np
from PIL import Image

from taskb_student.constants import CONTROL_DT
from taskb_student.localization import Localization
from taskb_student.object_finder import ObjectFinder, ObjectMap
from taskb_student.rgbd_perception import (
    CLASS_NAMES, Detection, RGBDGeometryDetector, classify_component,
    project_head_depth, quaternion_matrix, read_head_frame,
)


def detection(center=(-12, -10, 0.1), label="sugar_box"):
    points = np.array([(x, y, z) for x in (-0.08, 0.08) for y in (-0.03, 0.03)
                       for z in (-0.03, 0.03)]) + center
    return Detection(points, {name: 0.8 if name == label else 0.1 for name in CLASS_NAMES}, (1, 1, 6, 6))


def observation(yaw=0.0, frame=0):
    rgb = np.full((1, 8, 8, 3), 100, dtype=np.uint8)
    rgb[0, 0, 0, 0] = frame % 255
    depth = np.ones((1, 8, 8, 1))
    depth[0, 0, 0, 0] = 1 + frame * 0.0001
    return {
        "proprio": {"base_quat": np.array([[math.cos(yaw / 2), 0, 0, math.sin(yaw / 2)]])},
        "image": {"head_rgb": rgb, "head_depth": depth,
                  "head_intrinsics": np.array([[[10, 0, 4], [0, 10, 4], [0, 0, 1]]])},
    }


class FakeDetector:
    """Synthetic detections isolate scan/fusion logic from recognition accuracy."""
    def __init__(self, count=18):
        self.count = count

    def detect(self, *args):
        return [detection((-14 + (i % 6) * 0.7, -14 + (i // 6) * 0.7, 0.1), CLASS_NAMES[i // 6])
                for i in range(self.count)]


class ObjectFinderTests(unittest.TestCase):
    def test_optical_depth_projection_camera_offset_and_world_rotation(self):
        k = np.array([[2, 0, 1], [0, 2, 1], [0, 0, 1]])
        depth = np.full((3, 3), 2.0)
        cloud = project_head_depth(depth, k, (10, 20, 0.6), (1, 0, 0, 0), stride=1)
        np.testing.assert_allclose(cloud[1, 1], (12.4, 20, 0.72))
        np.testing.assert_allclose(cloud[1, 2], (12.4, 19, 0.72))
        np.testing.assert_allclose(cloud[2, 1], (12.4, 20, -0.28))
        rotated = project_head_depth(depth, k, (10, 20, 0.6),
                                     (math.sqrt(0.5), 0, 0, math.sqrt(0.5)), stride=1)
        np.testing.assert_allclose(rotated[1, 1], (10, 22.4, 0.72))
        depth[0, :3] = [float("nan"), float("inf"), 0]
        cloud = project_head_depth(depth, k, (0, 0, 0), (1, 0, 0, 0), stride=1)
        self.assertTrue(np.isnan(cloud[0]).all())

    def test_rgbd_shapes_and_invalid_calibration(self):
        rgb, depth, k = read_head_frame(observation())
        self.assertEqual(rgb.shape, (8, 8, 3))
        self.assertEqual(depth.shape, (8, 8))
        self.assertEqual(k.shape, (3, 3))
        with self.assertRaises(ValueError):
            read_head_frame({})
        bad = observation()
        bad["image"]["head_intrinsics"][0, 0, 0] = 0
        with self.assertRaises(ValueError):
            read_head_frame(bad)
        with self.assertRaises(ValueError):
            quaternion_matrix((0, 0, 0, 0))

    def test_depth_foreground_segmentation_with_synthetic_ground_and_sugar_box(self):
        height, width = 120, 160
        rgb = np.full((height, width, 3), 80, dtype=np.uint8)
        v, _ = np.mgrid[:height, :width]
        depth = np.full((height, width), np.nan)
        depth[v > 60] = 72 / (v[v > 60] - 60)
        rgb[84:94, 75:85] = (240, 240, 240)
        rgb[84:89, 75:78] = (20, 30, 180)
        depth[84:94, 75:85] = 2.0
        k = np.array([[100, 0, 80], [0, 100, 60], [0, 0, 1]])
        detector = RGBDGeometryDetector(stride=1)
        objects = detector.detect(rgb, depth, k, (-10, -10, 0.6), (1, 0, 0, 0))
        self.assertEqual(len(objects), 1)
        self.assertGreater(objects[0].class_scores["sugar_box"], 0.65)
        self.assertTrue(np.all(objects[0].points_world[:, 2] > 0.018))
        self.assertAlmostEqual(float(np.median(objects[0].points_world[:, 0])), -7.6)

    def test_classifier_keeps_ambiguous_colors_unknown(self):
        cloud = detection().points_world
        yellow = np.tile((220, 180, 20), (len(cloud), 1))
        scores = classify_component(cloud, yellow)
        self.assertEqual(max(scores, key=scores.get), "banana")
        gray = np.tile((80, 80, 80), (len(cloud), 1))
        self.assertLess(max(classify_component(cloud, gray).values()), 0.65)
        mustard = yellow.copy()
        mustard[0] = (180, 10, 10)
        self.assertEqual(max(classify_component(cloud, mustard), key=classify_component(cloud, mustard).get), "mustard_bottle")

    def test_multiview_fusion_dedup_and_principal_axis_uncertainty(self):
        mapping = ObjectMap()
        mapping.update([detection()], 0, 0)
        mapping.update([detection()], 0, 0)
        self.assertEqual(mapping.objects[0].views, 1)
        self.assertFalse(mapping.objects[0].confirmed)
        mapping.update([detection((-11.99, -10, 0.1))], 1, 0.3)
        self.assertTrue(mapping.objects[0].confirmed)
        self.assertIsNone(mapping.objects[0].yaw_axis)
        mapping.update([detection()], 2, 0.6)
        item = mapping.objects[0]
        self.assertEqual(len(mapping.objects), 1)
        self.assertIsNotNone(item.yaw_axis)
        self.assertTrue(item.orientation_ambiguous)
        np.testing.assert_allclose(item.position, (-11.995, -10, 0.1), atol=0.02)

    def test_neighboring_items_are_not_merged_in_one_frame(self):
        mapping = ObjectMap()
        a, b = detection(), detection((-11.82, -10, 0.1))
        self.assertEqual(mapping.update([a, b], 0, 0), [0, 1])
        self.assertEqual(mapping.update([b, a], 1, 0.5), [1, 0])
        self.assertEqual(len(mapping.objects), 2)

    def run_scan(self, count=18, repeated=False):
        loc = Localization()
        scanner = ObjectFinder(loc, FakeDetector(count), scan_views=8, yaw_rate=0.5)
        yaw = 3.0  # Cross +/-pi during the scan.
        for frame in range(6000):
            obs = observation(yaw, 0 if repeated else frame)
            command = scanner.compute_command(obs)
            self.assertEqual(command[:2], (0, 0))
            loc.update(command)
            # Use measured rotation with 20% tracking error, not commanded time.
            yaw += command[2] * CONTROL_DT * 0.8
            if scanner.done:
                break
        self.assertTrue(scanner.done)
        self.assertAlmostEqual(scanner.total_yaw, 2 * math.pi, delta=0.006)
        return scanner

    def test_full_turn_and_eighteen_confirmed_tracks(self):
        scanner = self.run_scan()
        self.assertTrue(scanner.ready)
        self.assertEqual(scanner.report()["counts_by_class"], dict.fromkeys(CLASS_NAMES, 6))
        self.assertEqual(len(scanner.objects), 18)
        self.assertEqual(scanner.compute_command({}), (0, 0, 0))
        scanner.reset()
        self.assertFalse(scanner.done)
        self.assertEqual(scanner.objects, ())

    def test_occlusion_is_reported_as_incomplete_without_fabricating_items(self):
        scanner = self.run_scan(count=17)
        self.assertFalse(scanner.ready)
        self.assertEqual(scanner.status, "incomplete")
        self.assertEqual(len(scanner.objects), 17)
        self.assertEqual(scanner.report()["missing_by_class"]["banana"], 1)

    def test_stale_camera_frames_do_not_count_as_independent_views(self):
        scanner = self.run_scan(repeated=True)
        self.assertFalse(scanner.ready)
        self.assertGreater(scanner.repeated_frames, 0)
        self.assertEqual(scanner.report()["confirmed_count"], 0)

    def test_missing_camera_and_no_actual_rotation_stop_safely(self):
        scanner = ObjectFinder(Localization())
        self.assertEqual(scanner.compute_command({"proprio": {}}), (0, 0, 0))
        self.assertEqual(scanner.status, "failed")
        self.assertIn("--enable_cameras", scanner.blocked_reason)
        scanner = ObjectFinder(Localization(), FakeDetector(), max_duration=2)
        for frame in range(110):
            scanner.compute_command(observation(0, frame))
        self.assertEqual(scanner.status, "failed")
        self.assertEqual(scanner.total_yaw, 0)
        self.assertFalse(scanner.ready)

    def test_report_and_annotated_images_export(self):
        scanner = self.run_scan(count=17)
        test_directory = Path(__file__).resolve().parent
        with tempfile.TemporaryDirectory(dir=test_directory, prefix="scan-export-") as directory:
            self.assertEqual(Path(directory).resolve().parent, test_directory)
            path = scanner.save_results(directory)
            report = json.loads(path.read_text(encoding="utf-8"))
            self.assertEqual(report["status"], "incomplete")
            self.assertEqual(len(report["objects"]), 17)
            with Image.open(Path(directory) / "object_map.png") as image:
                self.assertEqual(image.size, (800, 840))
            self.assertTrue((Path(directory) / "view_00.png").is_file())

    def test_demo_scan_blocks_navigation_until_ready(self):
        class Controller:
            def reset(self):
                pass

            def step(self, *args):
                return np.zeros(19)

        low_level = types.ModuleType("taskb_student.low_level")
        low_level.LowLevelController = Controller
        spec = importlib.util.spec_from_file_location(
            "demo._scan_solution_test", Path(__file__).resolve().parents[1] / "demo/solution.py"
        )
        module = importlib.util.module_from_spec(spec)
        with patch.dict(sys.modules, {"taskb_student.low_level": low_level}):
            spec.loader.exec_module(module)
        alg = module.AlgSolution(command=(0.4, 0, 0), detector=FakeDetector(), scan_output_dir=None)
        alg.object_finder = ObjectFinder(alg.localization, FakeDetector(), scan_views=8)
        yaw = 0.0
        for frame in range(5000):
            with patch("builtins.print"):
                response = alg.predicts(observation(yaw, frame), 0)
            self.assertEqual(len(response["action"]), 19)
            self.assertEqual(alg._previous_command[0], 0)
            yaw += float(alg._previous_command[2]) * CONTROL_DT * 0.8
            if alg.object_finder.done:
                break
        self.assertTrue(alg.object_finder.ready)
        self.assertAlmostEqual(alg.localization.yaw, (yaw + math.pi) % (2 * math.pi) - math.pi, places=6)
        alg.predicts(observation(yaw, frame + 1), 0)
        self.assertGreater(alg._previous_command[0], 0)
        pose = (alg.localization.position, alg.localization.yaw)
        alg.restart_scan()
        self.assertEqual((alg.localization.position, alg.localization.yaw), pose)
        self.assertFalse(alg.object_finder.done)
        with patch("builtins.print"):
            alg.predicts({"proprio": {}}, 0)
            self.assertEqual(alg.object_finder.status, "failed")
            np.testing.assert_array_equal(alg._previous_command, (0, 0, 0))
            alg.predicts({"proprio": {}}, 0)
            np.testing.assert_array_equal(alg._previous_command, (0, 0, 0))
        alg.reset()
        self.assertFalse(alg.object_finder.done)


if __name__ == "__main__":
    unittest.main()
