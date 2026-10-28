"""Near-field geometry, measured-arm scan sequencing and cross-camera fusion."""

import itertools
import json
import math
from pathlib import Path
import sys
import tempfile
import unittest

import numpy as np

from taskb_student.camera_geometry import SCAN_ARM_TARGET, head_camera_transform, wrist_camera_transform
from taskb_student.constants import ARM_Q0, CONTROL_DT, JOINT_LOWER, JOINT_UPPER, LEG_Q0
from taskb_student.localization import Localization
from taskb_student.object_finder import ObjectFinder, ObjectMap
from taskb_student.rgbd_perception import CLASS_NAMES, Detection, RGBDGeometryDetector, project_head_depth

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
from replay_object_recognition import replay, replay_batch


def render_near_banana(mount):
    """Analytic ray intersections with a floor and small yellow box at 1 m range.

    This synthetic scene isolates the camera blind zone; it is not an Isaac run.
    """
    height, width, focal = 240, 320, 229.06227
    v, u = np.mgrid[:height, :width]
    rays = np.stack((np.ones_like(u), -(u-160)/focal, -(v-120)/focal), axis=-1) @ mount[:3, :3].T
    origin = np.array((-10, -10, 0.6)) + mount[:3, 3]
    depth = np.full((height, width), np.nan)
    downward = rays[..., 2] < -1e-6
    depth[downward] = -origin[2] / rays[..., 2][downward]
    rgb = np.full((height, width, 3), 205, dtype=np.uint8)
    low, high = np.array((-9.1, -10.035, 0.008)), np.array((-8.9, -9.965, 0.035))
    denominator = np.where(np.abs(rays) < 1e-12, 1e-12, rays)
    a, b = (low-origin)/denominator, (high-origin)/denominator
    entry, leave = np.minimum(a, b).max(axis=-1), np.maximum(a, b).min(axis=-1)
    hit = (entry > 0.05) & (leave >= entry) & (~np.isfinite(depth) | (entry < depth))
    depth[hit] = entry[hit]
    rgb[hit] = (225, 195, 35)
    k = np.array([[focal, 0, 160], [0, focal, 120], [0, 0, 1]])
    return rgb, depth, k


class CoverageDetector:
    """Synthetic detections check orchestration separately from image recognition."""
    def __init__(self, indices):
        self.indices = indices

    def detect(self, *args, **kwargs):
        corners = np.array(list(itertools.product((-0.05, 0.05), (-0.03, 0.03), (-0.02, 0.02))))
        return [Detection(corners + (-14 + i % 6 * 0.7, -14 + i // 6 * 0.7, 0.1),
                          {CLASS_NAMES[i // 6]: 0.85}, (1, 1, 6, 6)) for i in self.indices]


def observation(yaw, frame, arm):
    rgb = np.full((1, 8, 8, 3), 100, dtype=np.uint8)
    rgb[0, 0, 0, 0] = frame % 255
    depth = np.full((1, 8, 8, 1), 1 + frame * 0.0001)
    image = {f"{camera}_{key}": value for camera in ("head", "ee") for key, value in
             (("rgb", rgb), ("depth", depth), ("intrinsics", np.array([[[10, 0, 4], [0, 10, 4], [0, 0, 1]]])))}
    return {"image": image, "proprio": {"joint_pos": np.array([LEG_Q0 + list(arm)]),
                                         "base_quat": np.array([[math.cos(yaw/2), 0, 0, math.sin(yaw/2)]])}}


class WristScanTests(unittest.TestCase):
    def test_forward_kinematics_zero_pose_and_measured_joint_motion(self):
        zero = wrist_camera_transform(np.zeros(7))
        np.testing.assert_allclose(zero[:3, :3], np.eye(3), atol=1e-12)
        np.testing.assert_allclose(zero[:3, 3], (0.2172, 0, 0.3205), atol=1e-12)
        scan = wrist_camera_transform(SCAN_ARM_TARGET)
        self.assertLess(scan[2, 0], -0.6)  # Camera forward axis points down.
        self.assertGreater(scan[2, 3], 0.25)  # Wrist mount remains above base, not near the floor.
        self.assertFalse(np.allclose(scan, wrist_camera_transform(ARM_Q0)))
        np.testing.assert_allclose(scan, wrist_camera_transform(LEG_Q0 + list(SCAN_ARM_TARGET)))
        self.assertTrue(np.all(np.array(SCAN_ARM_TARGET) >= JOINT_LOWER[12:]))
        self.assertTrue(np.all(np.array(SCAN_ARM_TARGET) <= JOINT_UPPER[12:]))
        with self.assertRaises(ValueError):
            wrist_camera_transform(np.zeros(6))

    def test_wrist_projection_uses_full_extrinsics(self):
        mount = wrist_camera_transform(SCAN_ARM_TARGET)
        k = np.array([[2, 0, 1], [0, 2, 1], [0, 0, 1]])
        points = project_head_depth(np.ones((3, 3)), k, (0, 0, 0.6), (1, 0, 0, 0),
                                   stride=1, camera_transform=mount)
        np.testing.assert_allclose(points[1, 1], mount[:3, 3] + mount[:3, 0] + (0, 0, 0.6))
        np.testing.assert_allclose(points[1, 2] - points[1, 1], -0.5 * mount[:3, 1])

    def test_near_banana_is_outside_head_image_but_detected_by_wrist(self):
        for camera, mount in (("head", head_camera_transform()), ("ee", wrist_camera_transform(SCAN_ARM_TARGET))):
            rgb, depth, k = render_near_banana(mount)
            detections = RGBDGeometryDetector().detect(rgb, depth, k, (-10, -10, 0.6), (1, 0, 0, 0),
                                                       camera_transform=mount)
            with self.subTest(camera=camera):
                self.assertEqual(len(detections), 0 if camera == "head" else 1)
                if detections:
                    self.assertGreater(detections[0].class_scores["banana"], 0.65)
                    np.testing.assert_allclose(np.mean(detections[0].points_world[:, :2], axis=0), (-9, -10), atol=0.08)

    def test_two_cameras_do_not_count_as_two_independent_views(self):
        detection = CoverageDetector([0]).detect()[0]
        mapping = ObjectMap()
        mapping.update([detection], 0, 0)
        mapping.update([detection], 0, 0)
        self.assertEqual(len(mapping.objects), 1)
        self.assertEqual(mapping.objects[0].classification_views, 1)
        self.assertFalse(mapping.objects[0].confirmed)
        conflicting = Detection(detection.points_world, {"banana": 0.85}, detection.bbox)
        mapping.update([conflicting], 0, 0)
        self.assertEqual(mapping.objects[0].classification_views, 0)
        mapping.update([detection], 1, 0.3)
        self.assertFalse(mapping.objects[0].confirmed)
        mapping.update([detection], 2, 0.6)
        self.assertTrue(mapping.objects[0].confirmed)

    def test_combined_scan_waits_for_arm_and_restores_before_ready(self):
        finder = ObjectFinder(Localization(), detector=CoverageDetector(range(16)), scan_views=8)
        finder.wrist_detector = CoverageDetector(range(14, 18))
        arm, yaw, restored = np.array(ARM_Q0, dtype=float), 0.0, False
        for frame in range(6000):
            command = finder.compute_command(observation(yaw, frame, arm))
            self.assertEqual(command[:2], (0, 0))
            if finder.status in ("positioning_wrist_camera", "restoring_arm"):
                self.assertEqual(command[2], 0)
            if finder.status == "restoring_arm":
                self.assertFalse(finder.ready)
                restored = True
            arm += np.clip(np.array(finder.arm_target) - arm, -0.03, 0.03)
            yaw += command[2] * CONTROL_DT * 0.8
            if finder.done:
                break
        self.assertTrue(restored)
        self.assertTrue(finder.ready)
        self.assertEqual(len(finder.objects), 18)
        self.assertEqual(finder.report()["captured_views_by_camera"], {"head": 9, "ee": 9})
        self.assertAlmostEqual(finder.captured_turn_radians, 2 * math.pi, delta=0.006)
        self.assertTrue(all(item.views == 9 for item in finder.objects))
        np.testing.assert_allclose(arm, ARM_Q0)

    def test_stalled_arm_stops_before_rotation_and_head_only_is_available(self):
        finder = ObjectFinder(Localization(), max_duration=0.6)
        for frame in range(40):
            self.assertEqual(finder.compute_command(observation(0, frame, ARM_Q0)), (0, 0, 0))
        self.assertEqual(finder.status, "failed")
        self.assertIn("positioning_wrist_camera", finder.blocked_reason)
        self.assertEqual(finder.arm_target, tuple(ARM_Q0))
        head = ObjectFinder(Localization(), detector=CoverageDetector([]), use_wrist_camera=False)
        for frame in range(30):
            head.compute_command(observation(0, frame, ARM_Q0))
        self.assertFalse(head.report()["wrist_camera_enabled"])
        self.assertEqual(head.arm_target, tuple(ARM_Q0))

    def test_two_camera_raw_export_and_replay_keep_one_near_item(self):
        finder = ObjectFinder(Localization(), record_rgbd=True)
        for view in range(2):
            for camera, mount in (("head", head_camera_transform()), ("ee", wrist_camera_transform(SCAN_ARM_TARGET))):
                rgb, depth, k = render_near_banana(mount)
                rgb[0, 0, 0] = view
                finder.ingest_frame(rgb, depth, k, (-10, -10, 0.6), (1, 0, 0, 0), view,
                                    camera=camera, joint_positions=SCAN_ARM_TARGET if camera == "ee" else None)
        finder.captured_turn_radians = 2 * math.pi
        finder.total_yaw = 2 * math.pi - 0.05  # Small yaw drift during the subsequent arm restoration.
        finder.finish_scan()
        self.assertEqual(len(finder.objects), 1)
        self.assertTrue(finder.objects[0].confirmed)
        expected = json.loads(json.dumps(finder.report()["objects"]))
        with tempfile.TemporaryDirectory(dir=ROOT / "tests", prefix="wrist-replay-") as name:
            parent = Path(name).resolve()
            self.assertEqual(parent.parent, ROOT / "tests")
            source = parent / "source"
            finder.save_results(source)
            self.assertTrue((source / "ee_view_00.png").exists())
            self.assertEqual(len(list((source / "rgbd").glob("*.npz"))), 4)
            crops = finder.save_object_images(source / "items", confirmed_only=True)
            self.assertEqual(crops[0]["source_camera"], "ee")
            result, ready = replay(source, parent / "output")
            self.assertFalse(ready)
            report = json.loads((result / "objects.json").read_text())
            self.assertEqual(report["objects"], expected)
            self.assertEqual(report["captured_views_by_camera"], {"head": 2, "ee": 2})
            self.assertNotIn("frames are missing", report["reason"])
            batch, success = replay_batch(source, parent / "batches")
            self.assertFalse(success)
            rows = json.loads((batch / "summary.json").read_text())["runs"]
            self.assertEqual(len(rows), 1)
            self.assertEqual(rows[0]["after_confirmed"], 1)


if __name__ == "__main__":
    unittest.main()
