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
    frames = sorted((source / "rgbd").glob("frame_*.npz"))
    if not frames:
        raise ValueError("No rgbd/frame_*.npz recordings. Old PNG/JSON-only scans cannot be replayed; run a new recognition test.")
    original = json.loads((source / "objects.json").read_text(encoding="utf-8"))
    finder = ObjectFinder(Localization())
    view_ids = set()
    for path in frames:
        with np.load(path, allow_pickle=False) as frame:
            view_id = int(frame["view_id"].item())
            if view_id in view_ids:
                raise ValueError(f"Duplicate recorded view_id: {view_id}")
            view_ids.add(view_id)
            finder.ingest_frame(frame["rgb"], frame["depth"], frame["intrinsics"],
                                frame["base_position"], frame["base_quaternion"], view_id)
    finder.total_yaw = float(original.get("total_yaw_radians", 0.0))
    complete_capture = (original.get("status") in ("complete", "incomplete")
                        and math.isfinite(finder.total_yaw)
                        and finder.total_yaw >= 2 * math.pi - 0.02
                        and len(frames) == original.get("captured_views"))
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


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, required=True, help="A new scan directory containing rgbd/frame_*.npz.")
    parser.add_argument("--output-dir", type=Path, default=ROOT / "runs" / "recognition_replay")
    args = parser.parse_args()
    try:
        _, ready = replay(args.input, args.output_dir)
    except (ValueError, OSError, KeyError) as error:
        parser.exit(1, f"Replay failed: {error}\n")
    return 0 if ready else 2


if __name__ == "__main__":
    raise SystemExit(main())
