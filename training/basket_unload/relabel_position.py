"""Relabel accepted delta demonstrations as executed absolute-position targets.

This CPU-only conversion preserves existing success evidence; it performs no
physics rollout. Images and proprioception stay unchanged except for the
previous-action columns, which must follow the new position-action semantics.
An optional final proprioception column exposes the robot's own episode clock.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch

from taskb_student.constants import ARM_Q0, JOINT_LOWER, JOINT_UPPER, LEG_Q0
from .model import ROOT, VELOCITY


def _set_targets(requested, previous, actual, lower, upper, velocity, dt):
    """Float32 CPU equivalent of GPUSimulation.set_targets, without CUDA."""
    requested = requested.clamp(lower, upper)
    slew = velocity * dt
    previous_gripper = previous[:, 18].clone()
    targets = previous.clone()
    targets.add_((requested - targets).clamp(-slew, slew))
    closing = requested[:, 18] > actual[:, 18]
    lead = torch.maximum(previous_gripper - slew[18], actual[:, 18] + .04)
    targets[:, 18] = torch.where(closing, torch.minimum(targets[:, 18], lead), targets[:, 18])
    return targets.clamp_(lower, upper)


def _path(path):
    path = Path(path)
    return path if path.is_absolute() else ROOT / path


def _display_path(path):
    path = path.resolve()
    return str(path.relative_to(ROOT)) if path.is_relative_to(ROOT) else str(path)


def _validate(data):
    for name in ("image", "proprio", "action", "episode_start", "valid_transition", "valid_environment"):
        if not isinstance(data.get(name), torch.Tensor):
            raise ValueError(f"Dataset requires tensor {name}")
    image = data["image"]
    if image.ndim != 5 or image.shape[2] != 5 or min(image.shape[:2]) < 1:
        raise ValueError("image must have nonempty shape [T,N,5,H,W]")
    steps, worlds = image.shape[:2]
    for name, shape in (("proprio", (steps, worlds, 67)), ("action", (steps, worlds, 19)),
                        ("episode_start", (steps, worlds)), ("valid_transition", (steps, worlds)),
                        ("valid_environment", (worlds,))):
        if tuple(data[name].shape) != shape:
            raise ValueError(f"{name} must have shape {shape}")
    if not image.is_floating_point() or any(data[name].dtype != torch.float32 for name in ("proprio", "action")):
        raise ValueError("image must be floating point; proprio and action must be float32")
    if any(data[name].dtype != torch.bool for name in ("episode_start", "valid_transition", "valid_environment")):
        raise ValueError("Episode and validity masks must be boolean")
    valid = torch.nonzero(data["valid_environment"], as_tuple=False).flatten()
    if not len(valid):
        raise ValueError("No valid_environment=true complete successful demonstrations")
    mask = data["valid_transition"][:, valid]
    if not bool(mask[0].all()) or bool((~mask[:-1] & mask[1:]).any()):
        raise ValueError("Each accepted world needs a nonempty contiguous valid_transition prefix")
    starts = data["episode_start"][:, valid]
    if not bool(starts[0].all()) or bool((starts[1:] & mask[1:]).any()):
        raise ValueError("Expected one accepted episode per world; later reset seeds are unavailable")
    config, report = data.get("environment"), data.get("report")
    if not isinstance(config, dict) or not isinstance(report, dict):
        raise ValueError("Dataset requires environment and report dictionaries")
    if config.get("action_mode", "delta") != "delta" or config.get("stage", "unload") != "unload":
        raise ValueError("Only delta-action unload demonstrations can be relabelled")
    if len(report.get("reset_seeds", [])) != worlds:
        raise ValueError("report.reset_seeds must identify every source world")
    return valid, mask.sum(0)


def _initial_targets(bank_path, seeds):
    with np.load(bank_path, allow_pickle=False) as bank:
        if bank["seed"].ndim != 1 or bank["ctrl"].shape != (len(bank["seed"]), 19):
            raise ValueError("Reset bank requires seed[N] and ctrl[N,19]")
        indices = []
        for seed in seeds:
            matches = np.flatnonzero(bank["seed"] == seed)
            if len(matches) != 1:
                raise ValueError(f"Reset seed {seed} must match exactly one bank row")
            indices.append(int(matches[0]))
        targets = torch.tensor(bank["ctrl"][indices], dtype=torch.float32)
    if not bool(torch.isfinite(targets).all()):
        raise ValueError("Nonfinite reset-bank control targets")
    return targets, indices


@torch.no_grad()
def relabel(source, output, *, observe_time=False):
    source, output = _path(source).resolve(), _path(output).resolve()
    destinations = (output, output.parent / "report.json", output.parent / "environment.json")
    if source == output:
        raise ValueError("Source demonstrations must not be overwritten")
    for path in destinations:
        if path.exists():
            raise FileExistsError(f"Use a new output path; already exists: {path}")
    data = torch.load(source, mmap=True, map_location="cpu", weights_only=False)
    valid, lengths = _validate(data)
    config = dict(data["environment"])
    if config.get("observe_time", False):
        raise ValueError("Expected the original 67-D source without an appended clock")
    selected = valid.tolist()
    seeds = [data["report"]["reset_seeds"][world] for world in selected]
    targets, bank_indices = _initial_targets(_path(config["reset_bank"]), seeds)
    lower = torch.tensor(JOINT_LOWER, dtype=torch.float32)
    upper = torch.tensor(JOINT_UPPER, dtype=torch.float32)
    velocity = torch.tensor(VELOCITY, dtype=torch.float32)
    nominal = torch.tensor(LEG_Q0 + ARM_Q0, dtype=torch.float32)
    dt = float(config.get("control_dt", .02))
    rate = torch.tensor([config.get("leg_target_rate", .2)] * 12
                        + [config.get("arm_target_rate", .35)] * 6 + [.25], dtype=torch.float32)
    if not np.isfinite(dt) or dt <= 0 or not bool(torch.isfinite(rate).all()) or bool((rate <= 0).any()):
        raise ValueError("Control period and target rates must be finite and positive")
    if bool(((targets < lower) | (targets > upper)).any()):
        raise ValueError("Initial control targets exceed joint limits")
    steps, worlds = int(lengths.max()), len(valid)
    # Advanced indexing copies only accepted worlds, without serializing the
    # source mmap storage or rejected candidate images into the output file.
    result = {name: data[name][:steps, valid].contiguous()
              for name in ("image", "proprio", "episode_start", "valid_transition")}
    result["action"] = torch.zeros((steps, worlds, 19), dtype=torch.float32)
    result["valid_environment"] = torch.ones(worlds, dtype=torch.bool)
    replay_targets = targets.clone()
    roundtrip_error = replay_error = max_action = 0.
    command_min, command_max = torch.full((19,), float("inf")), torch.full((19,), -float("inf"))
    for start in range(0, steps, 128):
        stop = min(start + 128, steps)
        mask = result["valid_transition"][start:stop]
        for name in ("image", "proprio"):
            if not bool(torch.isfinite(result[name][start:stop][mask]).all()):
                raise ValueError(f"Nonfinite {name} in accepted valid transitions")
    for step in range(steps):
        active = result["valid_transition"][step]
        action = data["action"][step, valid][active]
        if not bool(torch.isfinite(action).all()) or bool((action.abs() > 1).any()):
            raise ValueError("Source actions must be finite and in [-1,1]")
        actual = result["proprio"][step, active, :19] * (upper - lower).clamp_min(.1) + nominal
        requested = targets[active] + action * rate * dt
        executed = _set_targets(requested, targets[active], actual, lower, upper, velocity, dt)
        position = (2 * (executed - lower) / (upper - lower) - 1).clamp(-1, 1)
        decoded = lower + (position + 1.) * .5 * (upper - lower)
        roundtrip_error = max(roundtrip_error, float((decoded - executed).abs().max()))
        replay_requested = replay_targets[active] + (decoded - replay_targets[active]).clamp(-rate * dt, rate * dt)
        replayed = _set_targets(replay_requested, replay_targets[active], actual, lower, upper, velocity, dt)
        replay_error = max(replay_error, float((replayed - executed).abs().max()))
        targets[active], replay_targets[active] = executed, replayed
        result["action"][step, active] = position
        result["proprio"][step, :, 47:66] = 0
        if step:
            continuing = active & ~result["episode_start"][step]
            result["proprio"][step, continuing, 47:66] = result["action"][step - 1, continuing]
        max_action = max(max_action, float(position.abs().max()))
        command_min = torch.minimum(command_min, executed.amin(0))
        command_max = torch.maximum(command_max, executed.amax(0))
    if not bool(torch.isfinite(result["action"]).all()) or max_action > 1.:
        raise ValueError("Relabelled actions are nonfinite or outside [-1,1]")
    if max(roundtrip_error, replay_error) > 2e-6:
        raise ValueError(f"Position command reconstruction mismatch: roundtrip={roundtrip_error}, replay={replay_error}")
    if observe_time:
        duration = float(config["episode_seconds"])
        if not np.isfinite(duration) or duration <= 0:
            raise ValueError("episode_seconds must be finite and positive for the normalized clock")
        clock = (torch.arange(steps, dtype=torch.float32) * dt / duration)[:, None, None].expand(-1, worlds, -1)
        if not bool(torch.isfinite(clock).all()):
            raise ValueError("Nonfinite normalized episode clock")
        result["proprio"] = torch.cat((result["proprio"], clock), dim=-1)
    config.update(action_mode="position", num_envs=worlds, observe_time=observe_time)
    report = {
        "relabel_only": True, "new_physical_episodes": 0,
        "scope": "CPU deterministic action relabelling; no new physical success or policy evaluation",
        "source": _display_path(source), "source_action_mode": "delta", "action_mode": "position",
        "source_worlds": selected, "reset_seeds": seeds, "reset_bank_rows": bank_indices,
        "valid_steps_per_world": lengths.tolist(), "total_valid_steps": int(lengths.sum()),
        "all_valid_inputs_finite": True, "all_labels_finite": True,
        "max_abs_action": max_action, "max_roundtrip_command_error_rad": roundtrip_error,
        "max_position_replay_command_error_rad": replay_error,
        "command_min_rad": command_min.tolist(), "command_max_rad": command_max.tolist(),
        "command_within_joint_limits": bool(((command_min >= lower) & (command_max <= upper)).all()),
        "proprio_modified_columns": [47, 66],
        "proprio_previous_action": "previous position label; zero at episode starts and invalid padding",
        "proprio_dim": result["proprio"].shape[-1],
        "clock": {"enabled": observe_time, "column": 67 if observe_time else None,
                  "formula": "episode_steps * control_dt / episode_seconds",
                  "unit": "dimensionless episode fraction", "source": "robot-owned episode control clock",
                  "scope": "time conditioning does not establish unloading generalization"},
        "actor_inputs": ["image", "proprio"], "environment": config,
        "source_report": data["report"],
    }
    result.update(environment=config, report=report)
    output.parent.mkdir(parents=True, exist_ok=True)
    torch.save(result, output)
    (output.parent / "report.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    (output.parent / "environment.json").write_text(json.dumps(config, indent=2), encoding="utf-8")
    print(json.dumps({key: value for key, value in report.items() if key != "source_report"}, indent=2), flush=True)
    return report


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", required=True)
    parser.add_argument("--output", required=True, help="New demonstrations.pt path; source is never overwritten")
    parser.add_argument("--observe-time", action="store_true", help="Append normalized robot-owned episode time as proprio column 67")
    args = parser.parse_args()
    torch.set_num_threads(1)
    relabel(args.source, args.output, observe_time=args.observe_time)
