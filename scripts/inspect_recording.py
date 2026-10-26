"""Read back a Task B episode without Isaac; verify hashes, alignment and schema."""

import argparse
import hashlib
import json
from pathlib import Path
import sys

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from taskb_student.constants import CONTROL_DT, JOINT_LOWER, JOINT_NAMES, JOINT_UPPER
from taskb_student.recording import CameraAudit
from taskb_student.sensor_contract import OBSERVATION_SHAPES


def inspect_episode(directory):
    directory = Path(directory)
    manifest = json.loads((directory / "manifest.json").read_text(encoding="utf-8"))
    if manifest["format"] != "taskb-observation-action-v1":
        raise ValueError("Unsupported recording format")
    if manifest["metadata"]["joint_names"] != JOINT_NAMES:
        raise ValueError("Joint order mismatch")
    shapes = {f"obs/{group}/{key}": shape for group, terms in OBSERVATION_SHAPES.items() for key, shape in terms.items()}
    shapes.update(action=(1, 19), **{key: () for key in (
        "step", "sim_time", "wall_time", "score_before", "reward", "score_after", "terminated", "truncated",
    )})
    count, previous_score, previous_wall, ended = 0, None, -1, False
    lidar_ready_samples = 0
    audit = CameraAudit()
    for index, chunk in enumerate(manifest["chunks"]):
        if chunk["file"] != f"chunk_{index:06d}.npz":
            raise ValueError("Invalid chunk path/order")
        path = directory / chunk["file"]
        if hashlib.sha256(path.read_bytes()).hexdigest() != chunk["sha256"]:
            raise ValueError(f"Checksum mismatch: {path.name}")
        with np.load(path, allow_pickle=False) as arrays:
            if set(arrays.files) != set(shapes):
                raise ValueError("Missing prescribed data or unexpected fields (possible privileged data)")
            size = chunk["steps"]
            for key, shape in shapes.items():
                if arrays[key].shape != (size, *shape):
                    raise ValueError(f"Shape mismatch for {key}")
            for i in range(size):
                if ended:
                    raise ValueError("Rows cross an episode termination/reset")
                if arrays["step"][i] != count or not np.isclose(arrays["sim_time"][i], count * CONTROL_DT):
                    raise ValueError("Observation/action time alignment mismatch")
                for key in ("wall_time", "score_before", "reward", "score_after", "action"):
                    if not np.isfinite(arrays[key][i]).all():
                        raise ValueError(f"Nonfinite {key}")
                if arrays["wall_time"][i] < previous_wall:
                    raise ValueError("Wall time moved backwards")
                previous_wall = arrays["wall_time"][i]
                action = arrays["action"][i]
                if np.any(action < np.array(JOINT_LOWER) - 1e-6) or np.any(action > np.array(JOINT_UPPER) + 1e-6):
                    raise ValueError("Action outside joint limits")
                before, after = arrays["score_before"][i], arrays["score_after"][i]
                if previous_score is not None and not np.isclose(before, previous_score):
                    raise ValueError("Score continuity mismatch")
                if not np.isclose(after - before, arrays["reward"][i] / CONTROL_DT, atol=1e-5):
                    raise ValueError("Reward/score feedback mismatch")
                previous_score = after
                observation = {key: arrays[key][i] for key in shapes if key.startswith("obs/")}
                audit.observe(observation)
                lidar_ready_samples += int(bool(observation["obs/lidar/ready"].all()))
                ended = bool(arrays["terminated"][i] or arrays["truncated"][i])
                count += 1
    if count != manifest.get("persisted_steps", 0):
        raise ValueError("Persisted row count mismatch")
    if manifest["status"] == "closed":
        if count != manifest["steps"]:
            raise ValueError("Closed episode has missing rows")
        if previous_score is not None and not np.isclose(previous_score, manifest["final_score"]):
            raise ValueError("Final score mismatch")
    return {
        "integrity": "passed", "status": manifest["status"], "steps": count,
        "sim_duration_s": count * CONTROL_DT, "stop_reason": manifest.get("stop_reason", "unclean_shutdown"),
        "both_cameras_ready_samples": audit.ready_samples,
        "lidar_ready_samples": lidar_ready_samples,
        "source_kind": manifest["metadata"]["source_kind"],
        "demonstration_success": manifest["metadata"]["demonstration_success"],
        "scope": "Offline file/schema/alignment checks; does not certify camera geometry or demonstration quality.",
    }


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("directory", type=Path)
    args = parser.parse_args()
    print(json.dumps(inspect_episode(args.directory), ensure_ascii=False, indent=2))
