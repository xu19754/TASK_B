"""Export a successful demonstration's actuator commands as a residual prior.

Only joint command targets and their control period enter the runtime NPZ.
This is an open-loop command prior, not a new physics rollout or RL result.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import torch

from taskb_student.constants import JOINT_LOWER, JOINT_UPPER
from .behavior_clone import _load_dataset


def export_reference(source, output, world=0):
    source, output = Path(source), Path(output)
    report_path = output.with_suffix(".json")
    if output.suffix != ".npz":
        raise ValueError("Reference output must use .npz")
    if output.exists() or report_path.exists():
        raise FileExistsError("Use a new output path for the reference and provenance")
    # Export reads one world's command labels, not the multi-GB image tensor.
    data, valid = _load_dataset(source, mmap=True)
    if (data["environment"].get("action_mode", "delta") != "position"
            or data["environment"].get("stage", "unload") != "unload"):
        raise ValueError("Export requires accepted absolute-position demonstrations")
    if world not in valid.tolist():
        raise ValueError("Selected world must have valid_environment=true")
    count = int(data["valid_transition"][:, world].sum())
    action = data["action"][:count, world]
    if count < 2 or not bool(torch.isfinite(action).all()) or bool((action.abs() > 1.000001).any()):
        raise ValueError("Reference requires at least two finite bounded command labels")
    lower, upper = torch.tensor(JOINT_LOWER), torch.tensor(JOINT_UPPER)
    targets = (lower + (action + 1.) * .5 * (upper - lower)).float()
    if bool(((targets < lower) | (targets > upper)).any()):
        raise ValueError("Reference targets exceed unchanged joint limits")
    control_dt = float(data["environment"]["control_dt"])
    if not np.isfinite(control_dt) or control_dt <= 0:
        raise ValueError("control_dt must be finite and positive")
    output.parent.mkdir(parents=True, exist_ok=True)
    np.savez(output, targets=targets.numpy(), control_dt=np.float64(control_dt))
    with source.open("rb") as stream:
        digest = hashlib.file_digest(stream, "sha256").hexdigest()
    report = {
        "source": str(source), "source_sha256": digest, "source_world": world,
        "source_report": data.get("report"), "commands": count,
        "control_dt": control_dt, "command_duration_s": count * control_dt,
        "runtime_inputs": ["robot-owned episode step", "learned joint residual"],
        "runtime_data": ["targets", "control_dt"], "new_physical_episodes": 0,
        "scope": "fixed actuator-command prior; no state playback, new unloading success or RL claim",
    }
    report_path.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps({key: value for key, value in report.items() if key != "source_report"}, indent=2))
    return output


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--world", type=int, default=0)
    args = parser.parse_args()
    torch.set_num_threads(1)
    export_reference(**vars(args))
