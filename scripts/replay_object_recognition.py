"""Re-run perception on recorded RGB-D without starting Isaac Sim or moving a robot."""

import argparse
import json
import math
from pathlib import Path

import numpy as np

from feature_test_support import ROOT, make_run_directory, publish_summary
from taskb_student.localization import Localization
from taskb_student.object_finder import ObjectFinder


def replay(source, output_parent):
    source = Path(source).expanduser().resolve()
    frames = list((source / "rgbd").glob("*frame_*.npz"))
    if not frames:
        raise ValueError("No rgbd/frame_*.npz recordings. Old PNG/JSON-only scans cannot be replayed; run a new recognition test.")
    original = json.loads((source / "objects.json").read_text(encoding="utf-8"))
    finder = ObjectFinder(Localization())
    indexed = []
    for path in frames:
        with np.load(path, allow_pickle=False) as frame:
            camera = str(frame["camera"].item()) if "camera" in frame else "head"
            indexed.append((int(frame["view_id"].item()), camera != "head", camera, path))
    view_ids, camera_counts = set(), {"head": 0, "ee": 0}
    for _, _, camera, path in sorted(indexed):
        with np.load(path, allow_pickle=False) as frame:
            view_id = int(frame["view_id"].item())
            if (camera, view_id) in view_ids:
                raise ValueError(f"Duplicate recorded view_id: {camera}/{view_id}")
            view_ids.add((camera, view_id))
            finder.ingest_frame(frame["rgb"], frame["depth"], frame["intrinsics"],
                                frame["base_position"], frame["base_quaternion"], view_id, camera=camera,
                                joint_positions=frame["joint_positions"] if camera == "ee" else None)
            camera_counts[camera] += 1
    finder.total_yaw = float(original.get("total_yaw_radians", 0.0))
    captured_turn = original.get("captured_turn_radians")
    captured_turn = finder.total_yaw if captured_turn is None else float(captured_turn)
    finder.captured_turn_radians = captured_turn
    complete_capture = (original.get("status") in ("complete", "incomplete")
                        and math.isfinite(captured_turn)
                        and captured_turn >= 2 * math.pi - 0.02
                        and camera_counts["head"] == original.get("captured_views")
                        and camera_counts == original.get("captured_views_by_camera", {"head": original.get("captured_views"), "ee": 0}))
    if complete_capture:
        finder.finish_scan()
    else:
        finder.status = "incomplete"
        finder.blocked_reason = "Source scan did not finish a full turn or recorded frames are missing."
    directory = make_run_directory(output_parent)
    finder.save_results(directory)
    images = finder.save_object_images(directory / "items", confirmed_only=True)
    if not finder.ready:
        finder.save_object_images(directory / "candidates")
    summary_path = source / "summary.json"
    metadata = json.loads(summary_path.read_text(encoding="utf-8")) if summary_path.exists() else {}
    publish_summary(directory, {"test": "object_recognition_replay", "source": str(source),
                                "seed": metadata.get("seed"), "success": finder.ready,
                                "reason": finder.status, "saved_confirmed_images": len(images),
                                "candidate_count": len(finder.objects), "report": "objects.json"})
    return directory, finder.ready


def replay_batch(source, output_parent):
    """Compare every recorded run in a directory, preserving each input and seed."""
    sources = sorted(path.parent for path in Path(source).expanduser().resolve().rglob("objects.json")
                     if (path.parent / "rgbd").is_dir())
    if not sources:
        raise ValueError("No recorded scan directories found under --batch-input.")
    directory = make_run_directory(output_parent)
    rows = []
    for source_run in sources:
        try:
            result, ready = replay(source_run, directory / "runs")
            before = json.loads((source_run / "objects.json").read_text(encoding="utf-8"))
            after = json.loads((result / "objects.json").read_text(encoding="utf-8"))
            metadata = json.loads((result / "summary.json").read_text(encoding="utf-8"))
            row = {"source": str(source_run), "result": str(result), "seed": metadata["seed"],
                   "success": ready, "before_confirmed": before["confirmed_count"],
                   "after_confirmed": after["confirmed_count"], "counts_by_class": after["counts_by_class"],
                   "captured_views_by_camera": after["captured_views_by_camera"]}
        except (ValueError, OSError, KeyError) as error:
            row = {"source": str(source_run), "success": False, "error": str(error)}
        rows.append(row)
    success = all(row["success"] for row in rows)
    publish_summary(directory, {"test": "object_recognition_replay_batch", "success": success, "runs": rows})
    return directory, success


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument("--input", type=Path, help="A scan directory containing recorded RGB-D.")
    source.add_argument("--batch-input", type=Path, help="Replay all recorded runs under this directory and compare counts.")
    parser.add_argument("--output-dir", type=Path, default=ROOT / "runs" / "recognition_replay")
    args = parser.parse_args()
    try:
        _, ready = (replay_batch(args.batch_input, args.output_dir) if args.batch_input
                    else replay(args.input, args.output_dir))
    except (ValueError, OSError, KeyError) as error:
        parser.exit(1, f"Replay failed: {error}\n")
    return 0 if ready else 2


if __name__ == "__main__":
    raise SystemExit(main())
