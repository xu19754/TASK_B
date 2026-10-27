"""Deterministic held-out evaluation for a trained visual unload policy.

The evaluator may read privileged episode diagnostics after an episode ends;
those values never enter ``policy.act``.  It is intentionally separate from
the training entry point so evaluation cannot alter rewards or termination.
"""
from __future__ import annotations

import argparse
import json
from collections import Counter, defaultdict
from dataclasses import asdict, fields
from pathlib import Path
import time

import numpy as np
import torch

from .env import BasketUnloadEnv, EnvConfig
from .policy import RecurrentActorCritic
from .progress import RUN_DIR, update
from .vision import DesktopPreview


def _evaluation_config(checkpoint: Path, reset_bank: str, overrides: dict):
    environment_path = checkpoint.parent / "environment.json"
    if not environment_path.is_file():
        raise FileNotFoundError(f"Checkpoint execution configuration is missing: {environment_path}")
    saved = json.loads(environment_path.read_text(encoding="utf-8"))
    names = {field.name for field in fields(EnvConfig)}
    values = {key: value for key, value in saved.items() if key in names}
    explicit = {key: value for key, value in overrides.items() if value is not None}
    values.update(explicit)
    # Evaluation keeps its independent held-out bank and deterministic bank order.
    values.update(reset_bank=reset_bank, sequential_reset=True)
    config = EnvConfig(**values)
    if config.num_envs < 1 or config.episode_seconds <= 0 or config.control_dt <= 0:
        raise ValueError("num_envs, episode_seconds and control_dt must be positive")
    provenance = {
        "checkpoint_environment": str(environment_path),
        "explicit_overrides": explicit,
        "evaluation_overrides": {"reset_bank": reset_bank, "sequential_reset": True},
        "defaulted_fields": sorted(names - saved.keys() - values.keys()),
        "ignored_checkpoint_fields": sorted(saved.keys() - names),
    }
    return config, provenance


def evaluate(checkpoint: str | Path, *, episodes: int = 160, num_envs: int | None = None,
             reset_bank: str = "runs/basket_unload/reset_bank_eval.npz",
             episode_seconds: float | None = None, width: int | None = None,
             height: int | None = None, stage: str | None = None,
             num_objects: int | None = None, leg_target_rate: float | None = None,
             arm_target_rate: float | None = None, control_dt: float | None = None,
             timestep: float | None = None, camera_period: float | None = None,
             clearance_period: float | None = None, seed: int | None = None,
             gripper_residual_scale: float | None = None,
             record_video: bool = False, image_mode: str = "normal",
             output: str | Path = RUN_DIR / "evaluation.json") -> dict:
    if episodes < 1:
        raise ValueError("episodes must be positive")
    if image_mode not in ("normal", "zero"):
        raise ValueError("image_mode must be normal or zero")
    checkpoint = Path(checkpoint)
    config, provenance = _evaluation_config(checkpoint, reset_bank, {
        "num_envs": num_envs, "episode_seconds": episode_seconds,
        "image_width": width, "image_height": height, "stage": stage,
        "num_objects": num_objects, "leg_target_rate": leg_target_rate,
        "arm_target_rate": arm_target_rate, "control_dt": control_dt,
        "timestep": timestep, "camera_period": camera_period,
        "clearance_period": clearance_period, "seed": seed,
        "gripper_residual_scale": gripper_residual_scale,
    })
    num_envs = config.num_envs
    output = Path(output)
    output.parent.mkdir(parents=True, exist_ok=True)
    success_label = "卸料成功" if config.stage == "unload" else "站立阶段完成（非卸料成功）"
    update("自主策略独立评估初始化", f"已完成 0/{episodes} episodes；{success_label} 0。"
           f"加载 {checkpoint.name}，{num_envs} 个世界，episode 上限 {config.episode_seconds:g}s；image_mode={image_mode}。",
           run_dir=output.parent, metrics={"episodes": 0, "target_episodes": episodes, "success": 0,
                                        "image_mode": image_mode})
    device = torch.device("cuda:0")
    state = torch.load(checkpoint, map_location=device, weights_only=False)
    policy = RecurrentActorCritic(**state["network_kwargs"]).to(device)
    policy.load_state_dict(state["model"])
    policy.eval()
    env = BasketUnloadEnv(config)
    counts = Counter()
    by_combo = defaultdict(lambda: Counter())
    combo_metrics = defaultdict(lambda: defaultdict(list))
    metric_values = defaultdict(list)
    episode_records = []
    evaluation_steps = 0
    finished = 0
    completed_successes = 0
    excluded_terminals = 0
    max_steps = round(config.episode_seconds / config.control_dt) + 1
    preview = DesktopPreview(env.model, env.meta, output.parent, record_video=record_video)
    last_preview = 0.0
    last_progress = time.monotonic()
    try:
        for batch_start in range(0, episodes, num_envs):
            batch_size = min(num_envs, episodes - batch_start)
            # Inactive worlds may auto-reset while a slower assigned episode
            # finishes. Restore the intended bank cursor before each batch so
            # those incidental resets cannot change the evaluated seed set.
            env.reset_cursor = batch_start
            obs = env.reset()
            hidden = policy.initial_hidden(num_envs, device)
            starts = torch.ones(num_envs, dtype=torch.bool, device=device)
            active = torch.arange(num_envs, device=device) < batch_size
            pending_events = {}
            for _ in range(max_steps):
                with torch.no_grad():
                    policy_obs = (obs if image_mode == "normal" else
                                  {**obs, "image": torch.zeros_like(obs["image"])})
                    raw, _, _, next_hidden = policy.act(policy_obs, hidden, starts, deterministic=True)
                    obs, _, done, info = env.step(raw.tanh())
                evaluation_steps += 1
                hidden = next_hidden
                starts = done
                now = time.monotonic()
                for name, value in info.get("events", {}).items():
                    value = torch.as_tensor(value, device=device).long()
                    if value.shape != (num_envs,):
                        raise ValueError(f"Event {name} must have shape [num_envs]")
                    if name not in pending_events:
                        pending_events[name] = torch.zeros_like(value)
                    pending_events[name] += value * active.long()
                if done.any():
                    terminal_ids = torch.nonzero(done, as_tuple=False).flatten()
                    selected_indices = torch.nonzero(active[terminal_ids], as_tuple=False).flatten()
                    selected_ids = terminal_ids[selected_indices]
                    excluded_terminals += len(terminal_ids) - len(selected_ids)
                    for name, value in pending_events.items():
                        counts[name] += int(value[selected_ids].sum().item())
                        value[terminal_ids] = 0
                    active[selected_ids] = False
                    combinations = info.get("episode_combination")
                    if combinations is None:
                        raise ValueError("Terminal episodes require episode_combination diagnostics")
                    if "episode_seed" not in info or "episode_bank_row" not in info:
                        raise ValueError("Terminal episodes require reset bank seed and row diagnostics")
                    terminal_seeds = torch.as_tensor(info["episode_seed"]).cpu().tolist()
                    terminal_bank_rows = torch.as_tensor(info["episode_bank_row"]).cpu().tolist()
                    failure_flags = {
                        name: torch.as_tensor(value, device=device)[terminal_ids].bool().cpu().tolist()
                        for name, value in info.get("events", {}).items() if name.startswith("failure_")
                    }
                    episodes_info = info.get("episode", {})
                    success = torch.as_tensor(episodes_info.get("success", []), device=device).bool()
                    exits = torch.as_tensor(episodes_info.get("correct_exit_count", []), device=device)
                    wrong = torch.as_tensor(episodes_info.get("wrong_exit_count", []), device=device)
                    combinations = combinations.detach().cpu().tolist()
                    success, exits, wrong = success.cpu().tolist(), exits.cpu().tolist(), wrong.cpu().tolist()
                    diagnostic_names = (
                        "return_", "length", "max_tilt_rad", "min_clearance_m",
                        "max_speed_ratio", "max_foot_slip_m_s",
                    )
                    diagnostics = {
                        name: torch.as_tensor(episodes_info.get(name, []), device=device).cpu().tolist()
                        for name in diagnostic_names
                    }
                    dynamics_diagnostics = {
                        name: torch.as_tensor(episodes_info[name], device=device).cpu().tolist()
                        for name in ("peak_speed_joint_index", "peak_speed_joint_ratio", "invalid_joint_limit",
                                     "invalid_nonfinite", "invalid_overflow") if name in episodes_info
                    }
                    if any(len(values) != len(terminal_ids)
                           for values in (combinations, success, exits, wrong, terminal_seeds,
                                          terminal_bank_rows, *diagnostics.values(), *dynamics_diagnostics.values(),
                                          *failure_flags.values())):
                        raise ValueError("Terminal episode diagnostics must match done environments")
                    terminal_worlds = terminal_ids.cpu().tolist()
                    for index in selected_indices.cpu().tolist():
                        row, ok, ex, bad = (values[index] for values in (combinations, success, exits, wrong))
                        episode_id = batch_start + terminal_worlds[index]
                        if int(terminal_bank_rows[index]) != episode_id % len(env.bank["qpos"]):
                            raise RuntimeError("Assigned evaluation episode reset bank row changed before terminal")
                        episode_records.append({
                            "episode_index": finished, "world_id": terminal_worlds[index],
                            "episode_id": episode_id,
                            "terminal_control_step": evaluation_steps,
                            "seed": int(terminal_seeds[index]), "bank_row": int(terminal_bank_rows[index]),
                            "combination": row, "success": bool(ok),
                            "correct_exit_count": int(ex), "wrong_exit_count": int(bad),
                            "failure_flags": {name: values[index] for name, values in failure_flags.items()},
                            **{name: float(values[index]) for name, values in diagnostics.items()},
                            **{name: values[index] for name, values in dynamics_diagnostics.items()},
                        })
                        key = "-".join(map(str, row[:config.num_objects]))
                        record = by_combo[key]
                        record["episodes"] += 1
                        record["success"] += int(ok)
                        completed_successes += int(ok)
                        record["all_correct_exits"] += int(ex == env.config.num_objects)
                        record["correct_object_exits"] += int(ex)
                        record["wrong_exits"] += int(bad)
                        for name in diagnostic_names:
                            if index < len(diagnostics[name]):
                                value = float(diagnostics[name][index])
                                combo_metrics[key][name].append(value)
                                metric_values[name].append(value)
                        finished += 1
                    if len(selected_ids):
                        partial = {
                            "status": "in_progress", "checkpoint": str(checkpoint),
                            "image_mode": image_mode, "completed_episodes": finished,
                            "target_episodes": episodes, "episode_records": episode_records,
                            "scope": "Accepted episode diagnostics in progress; not a final evaluation rate.",
                        }
                        temporary = output.parent / "partial_episodes.tmp.json"
                        temporary.write_text(json.dumps(partial, indent=2), encoding="utf-8")
                        temporary.replace(output.parent / "partial_episodes.json")
                if bool(active.any()) and now - last_preview >= .5 and env.last_image is not None:
                    world_id = int(torch.nonzero(active, as_tuple=False)[0, 0])
                    preview.update(env.sim.data, env.last_image, world_id=world_id,
                                   asset_gpu=env.sim.gpu_model,
                                   metrics={"evaluation_episodes": finished, "target_episodes": episodes,
                                            "image_mode": image_mode, "world_id": world_id,
                                            "episode_id": batch_start + world_id})
                    last_preview = now
                if now - last_progress >= 15.:
                    update("自主策略独立评估", f"已完成 {finished}/{episodes} episodes；"
                           f"{success_label} {completed_successes}/{finished}。"
                           f"image_mode={image_mode}；显示实际自主策略动力学与腕部 RGB-D。",
                           run_dir=output.parent,
                           metrics={"episodes": finished, "target_episodes": episodes,
                                    "success": completed_successes, "image_mode": image_mode})
                    last_progress = now
                if not bool(active.any()):
                    break
            if bool(active.any()):
                unfinished_ids = (torch.nonzero(active).flatten() + batch_start).cpu().tolist()
                raise RuntimeError(f"Assigned evaluation episodes did not terminate within max_steps: {unfinished_ids}")
    finally:
        preview.close()
    combo_report = {}
    for key, record in sorted(by_combo.items()):
        item = dict(record)
        item.update({f"mean_{name}": float(np.mean(values))
                     for name, values in combo_metrics[key].items() if values})
        combo_report[key] = item
    report = {
        "checkpoint": str(checkpoint), "reset_bank": str(reset_bank),
        "image_mode": image_mode,
        "image_ablation_scope": ("All image channels are zeroed only in policy.act input; environment images and desktop preview retain real sensor images."
                                 if image_mode == "zero" else "policy.act receives the original environment image."),
        "engine": "MuJoCo 3.17.0 / MuJoCo Warp", "num_envs": num_envs,
        "checkpoint_global_step": state.get("global_step"),
        "environment": asdict(config), "configuration_source": provenance,
        "evaluation_kind": ("short_horizon_diagnostic" if config.episode_seconds < 1.
                            else "held_out_policy_evaluation"),
        "short_horizon_threshold_s": 1.,
        "success_semantics": ("balance curriculum completion; not unloading success"
                              if config.stage == "balance" else "all active objects legally exited and recovered stable"),
        "episodes": finished, "events": dict(counts),
        "episode_records": episode_records,
        "sampling": ("Fixed assigned batches: episode_id i uses reset bank row i modulo bank size. "
                     "Each world contributes only its first assigned terminal per batch; inactive-world "
                     "auto-resets and events are ignored while slower assigned episodes finish. "
                     "The reset cursor is restored to the batch start before every explicit reset, "
                     "and the last batch assigns only the remaining requested episodes."),
        "episode_record_order": ("Terminal control-step order; simultaneous terminals use ascending world_id. "
                                 "Only accepted episodes are included; seed and bank_row describe the state "
                                 "before automatic reset."),
        "excluded_terminal_episodes": excluded_terminals,
        "event_scope": ("Events are accumulated over assigned episodes only. "
                        "Inactive-world events and auto-reset episodes are excluded."),
        "episode_metrics": {f"mean_{name}": float(np.mean(values))
                            for name, values in metric_values.items() if values},
        "by_combination": combo_report,
        "desktop_preview": True,
        "preview_video": ({
            "path": str(preview.video_path), "frame_index": str(preview.video_index_path),
            "frame_count": preview.video_frames, "encoded_fps": 2,
            "time_semantics": ("Actual overview RGB frames sampled by the wall-clock desktop preview "
                               "approximately every 0.5 s and encoded at fixed 2 fps. This is not "
                               "simulation real-time playback; each JSONL frame records simulation_time "
                               "and wall_elapsed_s, including episode resets."),
        } if record_video else None),
        "scope": ("short-horizon pipeline diagnostic; not evidence of unloading capability"
                  if config.episode_seconds < 1. else
                  "held-out MuJoCo evaluation; not formal Isaac/PhysX Task B score"),
    }
    output.write_text(json.dumps(report, indent=2), encoding="utf-8")
    update("自主策略独立评估完成", f"已完成 {finished}/{episodes} episodes；"
           f"{success_label} {completed_successes}/{finished}；image_mode={image_mode}。报告：{output.name}。",
           run_dir=output.parent,
           metrics={"episodes": finished, "target_episodes": episodes,
                    "success": completed_successes, "report": str(output), "image_mode": image_mode})
    return report


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--episodes", type=int, default=160)
    parser.add_argument("--num-envs", type=int)
    parser.add_argument("--reset-bank", default="runs/basket_unload/reset_bank_eval.npz")
    parser.add_argument("--episode-seconds", type=float)
    parser.add_argument("--width", type=int)
    parser.add_argument("--height", type=int)
    parser.add_argument("--stage", choices=("balance", "unload"))
    parser.add_argument("--num-objects", type=int, choices=(1, 3))
    parser.add_argument("--leg-target-rate", type=float)
    parser.add_argument("--arm-target-rate", type=float)
    parser.add_argument("--control-dt", type=float)
    parser.add_argument("--timestep", type=float)
    parser.add_argument("--camera-period", type=float)
    parser.add_argument("--clearance-period", type=float)
    parser.add_argument("--seed", type=int)
    parser.add_argument("--gripper-residual-scale", type=float,
                        help="Override learned gripper target offset scale for a single-channel ablation")
    parser.add_argument("--image-mode", choices=("normal", "zero"), default="normal",
                        help="Zero image channels only at policy input; preserve environment images and preview")
    parser.add_argument("--record-video", action="store_true",
                        help="Record actual overview frames at 2 fps; simulation times are indexed separately")
    parser.add_argument("--output", default=str(RUN_DIR / "evaluation.json"))
    args = parser.parse_args()
    print(json.dumps(evaluate(**vars(args)), indent=2))
