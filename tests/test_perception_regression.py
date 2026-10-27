"""Perception regressions: actual crop summaries plus synthetic, calibrated RGB-D."""

import itertools
import json
import math
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch

import numpy as np

from taskb_student.localization import Localization
from taskb_student.object_finder import ObjectFinder, ObjectMap
from taskb_student.rgbd_perception import Detection, RGBDGeometryDetector, classify_component, estimate_ground_plane

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
from replay_object_recognition import replay


def tilted_floor_frame(slope=(0.006, -0.004), offset=0.035, banana=True):
    """Ray-plane intersections: floor and a 3.5 cm yellow top; no simulator needed."""
    height, width, focal = 120, 160, 100.0
    v, u = np.mgrid[:height, :width]
    a, b = slope
    denominator = (v-60)/focal + a - b*(u-80)/focal
    depth = np.full((height, width), np.nan)
    visible = denominator > 0.02
    depth[visible] = (0.72 - 0.4*a - offset) / denominator[visible]
    rgb = np.full((height, width, 3), 205, dtype=np.uint8)
    if banana:
        # The top can be flat in one view, while still being a real colored item.
        depth[89:93, 72:84] = (0.72 - 0.4*a - offset - 0.035) / denominator[89:93, 72:84]
        rgb[89:93, 72:84] = (225, 195, 35)
        rgb[89, 72:80] = (245, 245, 235)  # Specular highlight used to suppress the old banana rule.
    k = np.array([[focal, 0, 80], [0, focal, 60], [0, 0, 1]])
    return rgb, depth, k


class PerceptionRegressionTests(unittest.TestCase):
    def test_recorded_crop_color_and_observed_box_regressions(self):
        # Bounds approximate geometry only: this is NOT a replay of the original RGB-D scan.
        path = ROOT / "tests/fixtures/scan_20261028/color_geometry.npz"
        with np.load(path, allow_pickle=False) as samples:
            for object_id, rgb, bounds, expected in zip(samples["object_ids"], samples["colors"],
                                                        samples["bounds"], samples["expected_labels"]):
                corners = np.array(list(itertools.product((0, 1), repeat=3)))
                cloud = bounds[0] + corners * (bounds[1] - bounds[0])
                scores = classify_component(cloud, rgb)
                label = max(scores, key=scores.get) if max(scores.values()) >= 0.65 else "unknown"
                with self.subTest(object_id=int(object_id)):
                    self.assertEqual(label, expected)

    def test_tilted_bright_floor_does_not_create_objects(self):
        detector = RGBDGeometryDetector()
        for slope, offset in (((0, 0), 0.04), ((0.006, -0.004), 0.035), ((0.02, 0.012), -0.06)):
            with self.subTest(slope=slope):
                rgb, depth, k = tilted_floor_frame(slope, offset, banana=False)
                detections = detector.detect(rgb, depth, k, (-10, -10, 0.6), (1, 0, 0, 0))
                self.assertEqual(detections, [])
                self.assertIsNotNone(detector.last_diagnostics["ground_plane"])

    def test_low_banana_survives_ground_removal_and_white_highlights(self):
        detector = RGBDGeometryDetector()
        rgb, depth, k = tilted_floor_frame()
        detections = detector.detect(rgb, depth, k, (-10, -10, 0.6), (1, 0, 0, 0))
        self.assertEqual(len(detections), 1)
        self.assertGreater(detections[0].class_scores["banana"], 0.65)
        np.testing.assert_allclose(detections[0].points_world[:, 2], 0.035, atol=1e-5)

    def test_object_only_cloud_is_not_mistaken_for_ground(self):
        x, y = np.meshgrid(np.linspace(-0.08, 0.08, 20), np.linspace(-0.03, 0.03, 20))
        points = np.stack((x, y, np.full_like(x, 0.15)), axis=-1)
        self.assertIsNone(estimate_ground_plane(points, np.ones(x.shape, dtype=bool)))

    def test_adaptive_connection_does_not_cross_an_occlusion_depth_gap(self):
        # Touching image masks, but separate low objects with a 40 cm depth discontinuity.
        row, col = np.mgrid[:4, :8]
        points = np.stack((-7 + row * 0.03 + (col >= 4) * 0.4,
                           -10 + (col % 4) * 0.025, np.full(row.shape, 0.035)), axis=-1)
        rgb = np.tile(np.array((225, 195, 35), dtype=np.uint8), (4, 8, 1))
        k = np.array([[100, 0, 4], [0, 100, 2], [0, 0, 1]])
        with patch("taskb_student.rgbd_perception.project_head_depth", return_value=points):
            detections = RGBDGeometryDetector().detect(rgb, np.ones((4, 8)), k,
                                                       (-10, -10, 0.6), (1, 0, 0, 0))
        self.assertEqual(len(detections), 2)
        self.assertTrue(all(len(item.points_world) == 16 for item in detections))

    def test_unknown_reason_distinguishes_weak_evidence_and_conflict(self):
        mapping = ObjectMap()
        points = np.array(list(itertools.product((-0.08, 0.08), (-0.03, 0.03), (0.02, 0.07))))
        banana = Detection(points, {"banana": 0.85}, (1, 1, 5, 5))
        box = Detection(points, {"sugar_box": 0.85}, (1, 1, 5, 5))
        mapping.update([banana], 0, 0)
        self.assertEqual(mapping.objects[0].candidate_class, "banana")
        self.assertEqual(mapping.objects[0].unknown_reason, "class_seen_in_fewer_than_two_views")
        mapping.update([box], 1, 0.3)
        mapping.update([banana], 2, 0.6)
        self.assertFalse(mapping.objects[0].confirmed)
        self.assertEqual(mapping.objects[0].unknown_reason, "conflicting_class_evidence")

    def test_raw_frame_export_and_offline_replay_match(self):
        finder = ObjectFinder(Localization(), record_rgbd=True)
        rgb, depth, k = tilted_floor_frame()
        for view in range(3):
            rgb = rgb.copy()
            rgb[0, 0, 0] = view  # Distinct sensor frames; identical geometry.
            finder.ingest_frame(rgb, depth, k, (-10, -10, 0.6), (1, 0, 0, 0), view)
        finder.total_yaw = 2 * math.pi
        finder.finish_scan()
        expected = json.loads(json.dumps(finder.report()["objects"]))
        self.assertEqual(len(expected), 1)
        self.assertEqual(expected[0]["class_name"], "banana")
        with tempfile.TemporaryDirectory(dir=ROOT / "tests", prefix="rgbd-replay-") as name:
            parent = Path(name).resolve()
            self.assertEqual(parent.parent, ROOT / "tests")
            source = parent / "source"
            finder.save_results(source)
            (source / "summary.json").write_text(json.dumps({"seed": 42}), encoding="utf-8")
            self.assertEqual(len(list((source / "rgbd").glob("*.npz"))), 3)
            output, ready = replay(source, parent / "replayed")
            self.assertFalse(ready)  # One banana is not a successful 18-item scan.
            report = json.loads((output / "objects.json").read_text())
            self.assertEqual(report["objects"], expected)
            self.assertEqual(json.loads((output / "summary.json").read_text())["seed"], 42)
            # Missing raw frames must never upgrade an unfinished capture to a successful scan.
            original = finder.report()
            original["captured_views"] += 1
            (source / "objects.json").write_text(json.dumps(original), encoding="utf-8")
            partial_output, ready = replay(source, parent / "replayed")
            self.assertFalse(ready)
            self.assertIn("frames are missing", json.loads((partial_output / "objects.json").read_text())["reason"])
            with self.assertRaisesRegex(ValueError, "Old PNG/JSON-only"):
                replay(parent / "no-raw-data", parent / "replayed")


if __name__ == "__main__":
    unittest.main()
