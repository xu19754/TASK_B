import contextlib
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

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
import feature_test_support as support
import test_object_recognition as recognition_script
import test_traversal_video as traversal_script
from taskb_student.localization import Localization
from taskb_student.object_finder import ObjectFinder
from taskb_student.rgbd_perception import CLASS_NAMES, Detection
from taskb_student.traversal import Traversal


def route_fixture():
    positions = [(x, y, 0.1) for y in (-15, -10, -5) for x in (-18, -15, -12, -9, -6, -3)]
    return {"schema": "taskb-navigation-v1", "seed": 42, "navigation": {},
            "items": [{"object_id": i, "class_name": CLASS_NAMES[i // 6],
                       "position": position, "obstacle_radius": 0.2}
                      for i, position in enumerate(positions)]}


class MemoryWriter:
    def __init__(self):
        self.frames, self.closed = [], False

    def append_data(self, array):
        self.frames.append(array.copy())

    def close(self):
        self.closed = True


class FakeSolution:
    """Ideal command integration for testing orchestration; no physical simulation."""
    def __init__(self, **kwargs):
        self.localization = Localization()
        self.traversal = None
        self._previous_command = np.zeros(3)

    def reset(self):
        self.localization.reset()

    def set_traversal(self, targets, positions, **options):
        self.traversal = Traversal(self.localization, targets, positions, **options)

    def predicts(self, obs, score):
        self.localization.update(self._previous_command)
        self._previous_command = np.asarray(self.traversal.compute_command())
        return {"action": [0.0] * 19, "giveup": False}


class FakeEnvironment:
    def __init__(self):
        self.unwrapped, self.closed = self, False

    def reset(self, seed):
        return {}, {}

    def render(self):
        return np.full((96, 128, 3), 100, dtype=np.uint8)

    def close(self):
        self.closed = True


class FeatureScriptTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory(dir=Path(__file__).resolve().parent, prefix="feature-test-")
        self.directory = Path(self.temp.name).resolve()
        self.assertEqual(self.directory.parent, Path(__file__).resolve().parent)
        self.addCleanup(self.temp.cleanup)

    def test_traversal_pickup_switch_and_three_second_default(self):
        parser = support.build_parser("traversal")
        defaults = parser.parse_args([])
        self.assertFalse(defaults.pickup)
        self.assertEqual(defaults.dwell_seconds, 3.0)
        self.assertTrue(parser.parse_args(["--pickup"]).pickup)
        explicit = parser.parse_args(["--no-pickup", "--dwell-seconds", "5"])
        self.assertFalse(explicit.pickup)
        self.assertEqual(explicit.dwell_seconds, 5.0)

    def test_class_named_crop_export_and_unknown_filter(self):
        finder = ObjectFinder(Localization())
        cloud = np.array([(x, y, z) for x in (-12.05, -11.95) for y in (-10.02, -9.98) for z in (0.05, 0.15)])
        known = Detection(cloud, {"sugar_box": 0.8}, (4, 4, 20, 20))
        unknown = Detection(cloud + (1, 0, 0), dict.fromkeys(CLASS_NAMES, 0.1), (24, 4, 30, 14))
        finder.object_map.update([known, unknown], 0, 0)
        finder.object_map.update([known, unknown], 1, 0.3)
        finder._snapshots = [(0, np.full((32, 40, 3), 100, dtype=np.uint8), [(0, known.bbox), (1, unknown.bbox)])]
        records = finder.save_object_images(self.directory / "items", confirmed_only=True)
        self.assertEqual(len(records), 1)
        self.assertIn("sugar_box_conf0.80_x-12.00_y-10.00", records[0]["image"])
        with Image.open(self.directory / "items" / records[0]["image"]) as crop:
            self.assertGreater(crop.width, 0)
            self.assertGreater(crop.height, 44)
        candidates = finder.save_object_images(self.directory / "candidates")
        self.assertEqual(len(candidates), 2)
        self.assertIn("unknown", candidates[1]["image"])

    def test_generated_route_faces_each_item_and_retains_all_obstacles(self):
        source = route_fixture() | {"source": "synthetic-test-fixture"}
        route = support.prepare_route(source)
        self.assertEqual(len(route["items"]), 18)
        self.assertEqual(route["mode"], "visit_only_keep_all_obstacles")
        for item in route["items"]:
            x, y, yaw = item["target_pose"]
            ox, oy = item["position"][:2]
            self.assertAlmostEqual(yaw, math.atan2(oy - y, ox - x))
            self.assertGreater(math.hypot(ox - x, oy - y), 0.9)
        for path in route["planned_paths"]:
            for a, b in zip(path, path[1:]):
                dx, dy = b[0] - a[0], b[1] - a[1]
                for item in route["items"]:
                    x, y = item["position"][:2]
                    length = dx * dx + dy * dy
                    t = max(0, min(1, ((x - a[0])*dx + (y - a[1])*dy)/length)) if length else 0
                    self.assertGreater(math.hypot(a[0] + t*dx - x, a[1] + t*dy - y), 0.9)

    def test_visit_confirmation_keeps_obstacle_and_pickup_removes_it(self):
        loc = Localization()
        nav = Traversal(loc, [(-10, -10, 0), (-10, -10, 0)], [(-8, -10), (-12, -10)])
        nav.compute_command()
        nav.confirm_visit()
        self.assertEqual(nav.active_obstacle_indices, (0, 1))
        nav.compute_command()
        nav.confirm_pickup()
        self.assertEqual(nav.active_obstacle_indices, (0,))
        self.assertTrue(nav.completed)

    def test_input_validation_and_scene_seed_metadata(self):
        path = self.directory / "route.json"
        source = route_fixture()
        support.write_json(path, source)
        loaded = support.load_navigation_source(path)
        self.assertEqual(loaded["seed"], 42)
        self.assertEqual(len(loaded["items"]), 18)
        source["items"][0]["target_pose"] = [-10, -10, 0]
        support.write_json(path, source)
        with self.assertRaises(ValueError):
            support.load_navigation_source(path)
        support.write_json(path, {"ready": False})
        with self.assertRaisesRegex(ValueError, "incomplete"):
            support.load_navigation_source(path)
        report = {"ready": True, "objects": [{**item, "confirmed": True} for item in route_fixture()["items"]]}
        support.write_json(self.directory / "objects.json", report)
        support.write_json(self.directory / "summary.json", {"seed": 17})
        self.assertEqual(support.load_navigation_source(self.directory)["seed"], 17)

    def test_video_frame_timing_and_close(self):
        writer = MemoryWriter()
        video = support.SimulationVideo(self.directory / "unused.mp4", fps=25, writer_factory=lambda *a, **k: writer)
        rgb = np.zeros((96, 128, 3), dtype=np.uint8)
        for tick in range(51):
            video.append(rgb, tick * 0.02, "simulation test")
        self.assertEqual(video.frames, 26)  # Initial frame plus one frame each 0.04 seconds.
        video.close()
        video.close()
        self.assertTrue(writer.closed)
        self.assertEqual(len(writer.frames), 26)
        with self.assertRaises(RuntimeError):
            video.append(rgb, 2, "closed")

    @unittest.skipUnless(importlib.util.find_spec("imageio_ffmpeg"), "Optional video dependencies are not installed")
    def test_real_mp4_encoding_and_decoding(self):
        imageio = support.require_video_dependencies()
        path = self.directory / "codec-test.mp4"
        video = support.SimulationVideo(path, fps=12)
        try:
            for index in range(25):
                frame = np.full((96, 128, 3), index * 10, dtype=np.uint8)
                video.append(frame, index / 12, "synthetic codec test")
        finally:
            video.close()
        self.assertGreater(path.stat().st_size, 1000)
        reader = imageio.get_reader(path)
        try:
            self.assertEqual(reader.count_frames(), 25)
            self.assertAlmostEqual(reader.get_meta_data()["fps"], 12)
            self.assertEqual(reader.get_data(0).shape, (96, 128, 3))
        finally:
            reader.close()

    def test_video_runner_completes_and_closes_on_failure(self):
        fake_torch = types.ModuleType("torch")
        fake_torch.inference_mode = contextlib.nullcontext
        fake_solution = types.ModuleType("demo.solution")
        fake_solution.AlgSolution = FakeSolution
        for terminate, pickup in ((False, False), (True, False), (False, True)):
            with self.subTest(terminate=terminate, pickup=pickup):
                source = route_fixture() | {"source": "synthetic-test-fixture"}
                for item in source["items"]:
                    item["target_pose"] = [-10, -10, 0]
                args = types.SimpleNamespace(output_dir=self.directory / f"{terminate}-{pickup}", seed=42,
                                             source=source, fps=25, max_seconds=10, max_walltime_s=None,
                                             pickup=pickup, per_target_timeout=0.02, dwell_seconds=0.04)
                env, writer = FakeEnvironment(), MemoryWriter()
                factory = lambda path, fps: support.SimulationVideo(path, fps, writer_factory=lambda *a, **k: writer)
                step_result = ({}, np.array([0.0]), np.array([terminate]), np.array([False]), {})
                with patch.dict(sys.modules, {"torch": fake_torch, "demo.solution": fake_solution}), \
                        patch.object(traversal_script, "create_environment", return_value=env), \
                        patch.object(traversal_script, "step_environment", return_value=step_result), \
                        patch.object(traversal_script, "SimulationVideo", side_effect=factory), \
                        patch("builtins.print"):
                    code = traversal_script.run(args, types.SimpleNamespace(is_running=lambda: True))
                self.assertTrue(env.closed)
                self.assertTrue(writer.closed)
                pointer = json.loads((args.output_dir / "latest.json").read_text())
                summary = json.loads((Path(pointer["directory"]) / "summary.json").read_text())
                success = not terminate and not pickup
                self.assertEqual(summary["success"], success)
                self.assertEqual(summary["visited_count"], 18 if success else 0)
                self.assertEqual(code, 0 if success else 2)
                self.assertFalse(summary["obstacles_removed"])
                self.assertEqual(summary["pickup_enabled"], pickup)
                if pickup:
                    self.assertEqual(summary["reason"], "pickup_timeout")
                    self.assertEqual(summary["final_navigation"]["status"], "awaiting_pickup")
                if success:
                    run_dir = Path(pointer["directory"])
                    visits = json.loads((run_dir / "visits.json").read_text())
                    self.assertEqual(visits[0]["sim_time"], 0.04)
                    self.assertEqual(visits[0]["completion"], "dwell_completed")
                    records = [json.loads(line) for line in (run_dir / "trajectory.jsonl").read_text().splitlines()]
                    self.assertEqual(records[0]["navigation"]["status"], "dwelling")
                    self.assertEqual(records[-1]["navigation"]["status"], "completed")

    def test_recognition_runner_exports_complete_and_partial_results(self):
        fake_torch = types.ModuleType("torch")
        fake_torch.inference_mode = contextlib.nullcontext
        for count in (18, 17):
            with self.subTest(count=count):
                finder = ObjectFinder(Localization())
                detections = []
                for index, item in enumerate(route_fixture()["items"][:count]):
                    points = np.array([(x, y, z) for x in (-0.05, 0.05) for y in (-0.03, 0.03)
                                       for z in (-0.02, 0.02)]) + item["position"]
                    detections.append(Detection(points, {item["class_name"]: 0.8}, (2, 2, 20, 20)))
                finder.object_map.update(detections, 0, 0)
                finder.object_map.update(detections, 1, 0.5)
                finder._snapshots = [(0, np.full((32, 32, 3), 100, dtype=np.uint8),
                                      [(i, detection.bbox) for i, detection in enumerate(detections)])]
                finder.status = "complete" if count == 18 else "incomplete"
                fake_solution = types.ModuleType("demo.solution")
                fake_solution.AlgSolution = lambda **kwargs: types.SimpleNamespace(
                    object_finder=finder, reset=lambda: None,
                    predicts=lambda *args: {"action": [0] * 19, "giveup": False})
                args = types.SimpleNamespace(output_dir=self.directory / f"recognition-{count}", seed=42,
                                             max_seconds=10, max_walltime_s=None)
                env = FakeEnvironment()
                with patch.dict(sys.modules, {"torch": fake_torch, "demo.solution": fake_solution}), \
                        patch.object(recognition_script, "create_environment", return_value=env), \
                        patch("builtins.print"):
                    code = recognition_script.run(args, types.SimpleNamespace(is_running=lambda: True))
                pointer = json.loads((args.output_dir / "latest.json").read_text())
                result = Path(pointer["directory"])
                summary = json.loads((result / "summary.json").read_text())
                self.assertEqual(len(list((result / "items").glob("*.png"))), count)
                self.assertEqual(summary["success"], count == 18)
                self.assertEqual(code, 0 if count == 18 else 2)
                self.assertTrue(env.closed)
                self.assertEqual((result / "candidates").is_dir(), count < 18)


if __name__ == "__main__":
    unittest.main()
