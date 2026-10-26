"""Collect candidate demonstrations through the unchanged CUDA environment.

The reference consumes robot reset geometry only. Cargo truth is confined to
episode acceptance/reporting; stored policy inputs are image/proprio only.
Failed trajectories are labelled invalid and cannot initialize behaviour cloning.
"""
from __future__ import annotations

import argparse
from dataclasses import asdict
import json
from pathlib import Path
import time

import numpy as np
import torch
import mujoco
import warp as wp

from .env import BasketUnloadEnv, EnvConfig
from .expert_rollout import SweepReference, _smooth
from .model import ROOT
from .progress import RUN_DIR, update
from .reference_experiment import _clearance
from .vision import DesktopPreview


def collect(args):
    torch.set_num_threads(1)
    output = Path(args.output)
    output.mkdir(parents=True, exist_ok=True)
    config = EnvConfig(
        num_envs=args.num_envs, stage="unload", num_objects=args.num_objects,
        episode_seconds=args.episode_seconds, image_width=args.width,
        image_height=args.height, reset_bank=args.reset_bank, seed=args.seed,
        leg_target_rate=args.leg_target_rate, arm_target_rate=args.arm_target_rate,
        sequential_reset=True, action_mode=args.action_mode,
    )
    (output / "environment.json").write_text(json.dumps(asdict(config), indent=2), encoding="utf-8")
    update("GPU 专家轨迹复验", "通过与PPO相同的物理、碰撞和尾窗检查；尚未验证完整成功。", output)
    env = BasketUnloadEnv(config)
    obs = env.reset()
    initial_combinations = env.class_ids.cpu().tolist()
    reset_seeds = env.bank["seed"][:config.num_envs].cpu().tolist()
    # These reset-only robot states are used by the joint-target teacher, never
    # saved as actor observations and never include cargo poses/contact labels.
    position = env.sim.qpos[:, :3].cpu().numpy()
    rotation = env.sim.xmat[:, env.meta["base_id"]].cpu().numpy()
    feet = env.sim.xpos[:, env.foot_ids].cpu().numpy()
    reference_path = Path(args.path)
    reference_path = reference_path if reference_path.is_absolute() else ROOT / reference_path
    teachers = [SweepReference(env.meta, p, r, f, path=reference_path, speed_scale=args.speed_scale,
                              approach_rate=args.approach_rate,
                              approach_before_tilt=args.approach_before_tilt)
                for p, r, f in zip(position, rotation, feet)]
    if args.tool_lift_taper_start is not None and not (
            np.isfinite(args.tool_lift_taper_start)
            and 0. <= args.tool_lift_taper_start < min(teacher.stop_fraction for teacher in teachers)):
        raise ValueError("tool_lift_taper_start must be finite and in [0, stop_fraction)")
    duration = min(config.episode_seconds, max(teacher.duration for teacher in teachers) + 2.)
    count = round(duration / config.control_dt)
    if args.max_seconds is not None:
        count = min(count, round(args.max_seconds / config.control_dt))
    # Precompute at camera frequency, then linearly interpolate at control
    # frequency. All entries are joint targets; no state/velocity is prescribed.
    table_dt = config.camera_period
    times = np.arange(0., count * config.control_dt + table_dt * 2, table_dt)
    table = torch.as_tensor(np.stack([
        np.stack([teacher.targets(float(t)) for teacher in teachers]) for t in times
    ]), device=env.device, dtype=torch.float32)
    rates = torch.tensor([config.leg_target_rate] * 12 + [config.arm_target_rate] * 6 + [.25],
                         device=env.device)
    kp = torch.as_tensor(env.meta["kp"], device=env.device)
    active = torch.ones(config.num_envs, device=env.device, dtype=torch.bool)
    valid = torch.zeros_like(active)
    starts = torch.ones_like(active)
    images, proprio, actions, boundaries, valid_transitions = [], [], [], [], []
    episode_records = [None] * config.num_envs
    device_events = {}
    events = {}
    started = time.monotonic()
    last_preview, last_status = 0., 0.
    preview = DesktopPreview(env.model, env.meta, output)
    completed_steps = 0
    try:
        for step in range(count):
            elapsed = step * config.control_dt
            coordinate = step * config.control_dt / table_dt
            lower = int(coordinate)
            target = torch.lerp(table[lower], table[lower + 1], coordinate - lower)
            reference = teachers[0]
            if elapsed < reference.approach_start:
                tool_fraction = 0.
            elif elapsed < reference.sweep_start:
                tool_fraction = np.clip((elapsed - reference.approach_start) / reference.approach_duration, 0, 1)
            elif elapsed < reference.reverse_end:
                tool_fraction = 1.
            elif elapsed < reference.fold_end:
                tool_fraction = 1. - np.clip((elapsed - reference.reverse_end) / reference.approach_duration, 0, 1)
            else:
                tool_fraction = 0.
            tool_fraction = 10 * tool_fraction ** 3 - 15 * tool_fraction ** 4 + 6 * tool_fraction ** 5
            lift_fraction = tool_fraction
            if (args.tool_lift_taper_start is not None
                    and reference.sweep_start <= elapsed < reference.reverse_end):
                parameter = reference.sweep_parameter(elapsed)
                fade = _smooth((parameter - args.tool_lift_taper_start)
                               / (reference.stop_fraction - args.tool_lift_taper_start))
                lift_fraction *= 1. - fade
            # A small pitch-preserving joint offset increases floor clearance.
            # It is a commanded trajectory change, never a geometry/state edit.
            target[:, 13] += args.tool_joint_lift * lift_fraction
            target[:, 15] -= args.tool_joint_lift * lift_fraction
            if args.gripper_target is not None:
                target[:, 18] = args.gripper_target * tool_fraction
            if args.arm_gravity_compensation:
                # Robot articulated bias is a function of its own feedback and
                # model; free cargo coordinates/contact forces are not inputs.
                # Express feedforward through normal position targets, keeping
                # the same actuator, torque, slew and gripper lead limits.
                target[:, 12:] += wp.to_torch(env.sim.data.qfrc_bias)[:, env.sim.dof_adr[12:]] / kp[12:]
            if config.action_mode == "position":
                action = (2. * (target - env.sim.lower) / (env.sim.upper - env.sim.lower) - 1.).clamp(-1, 1)
            else:
                action = ((target - env.sim.targets) / (rates * config.control_dt)).clamp(-1, 1)
            # Completed environments may auto-reset while peers finish. Their
            # later observations/actions are never accepted as demonstrations.
            action = torch.where(active[:, None], action, torch.zeros_like(action))
            images.append(obs["image"].half().cpu())
            proprio.append(obs["proprio"].cpu())
            actions.append(action.cpu())
            boundaries.append(starts.cpu())
            valid_transitions.append(active.cpu())
            obs, _, done, info = env.step(action)
            completed_steps = step + 1
            for name, value in info["events"].items():
                if name not in device_events:
                    device_events[name] = torch.zeros((), device=env.device, dtype=torch.long)
                device_events[name] += (value * active).sum()
            just_done = active & done
            if bool(just_done.any()):
                done_indices = torch.nonzero(done).flatten().cpu().tolist()
                for row, world in enumerate(done_indices):
                    if not bool(active[world]):
                        continue
                    record = {name: float(value[row]) for name, value in info["episode"].items()}
                    record["failure_flags"] = {
                        name: bool(value[world]) for name, value in info["events"].items()
                        if name.startswith("failure_")
                    }
                    # True success already requires valid dynamics, correct
                    # complete exits, recovery and 0.5 s stable hold in Env.
                    valid[world] = bool(record["success"])
                    record["steps"] = step + 1
                    feedback = info["terminal_observation"]["proprio"][world]
                    joint_positions = (feedback[:19] * (env.sim.upper - env.sim.lower).clamp_min(.1)
                                       + env.sim.nominal).cpu().numpy()
                    record["terminal_joint_positions"] = joint_positions.tolist()
                    fastest = int(feedback[19:38].abs().argmax())
                    record["terminal_fastest_joint"] = env.meta["joint_names"][fastest]
                    # Read-only CPU cross-check of the same robot joint pose.
                    # Root transforms do not change internal pair distances.
                    diagnosis = mujoco.MjData(env.model)
                    diagnosis.qpos[env.meta["joint_qpos_adr"]] = joint_positions
                    mujoco.mj_forward(env.model, diagnosis)
                    distance, pair = _clearance(env.model, diagnosis, env.meta)
                    record["cpu_terminal_clearance_m"] = distance
                    record["cpu_terminal_clearance_pair"] = pair
                    episode_records[world] = record
                active &= ~done
                (output / "partial_episodes.json").write_text(
                    json.dumps(episode_records, indent=2), encoding="utf-8")
            starts = done
            now = time.monotonic()
            if now - last_preview > .5:
                events = {name: int(value) for name, value in device_events.items()}
                live_world = int(torch.nonzero(active).flatten()[0]) if bool(active.any()) else 0
                preview.update(env.sim.data, env.last_image, world_id=live_world,
                               asset_gpu=env.sim.gpu_model,
                               metrics={"candidate_reference": True, "step": step + 1,
                                        "accepted_episodes": int(valid.sum()), "events": events})
                last_preview = now
            if now - last_status > 15.:
                update("GPU 专家轨迹复验", f"仿真 {completed_steps * config.control_dt:.1f}s；"
                       f"合法完整成功 {int(valid.sum())}/{config.num_envs}；"
                       f"尾窗首次排出事件 {events.get('correct_object_exit', 0)}。", output)
                last_status = now
            if not bool(active.any()):
                break
    finally:
        preview.close()
    report = {
        "engine": "MuJoCo / MuJoCo Warp CUDA", "reference_only": True, "rl_policy": False,
        "environment": asdict(config), "speed_scale": args.speed_scale,
        "arm_gravity_compensation": args.arm_gravity_compensation,
        "tool_joint_lift_rad": args.tool_joint_lift, "gripper_target": args.gripper_target,
        "tool_lift_taper_start": args.tool_lift_taper_start,
        "reference": {"interpolation": teachers[0].interpolation,
                      "approach_before_tilt": args.approach_before_tilt,
                      "path": str(reference_path.relative_to(ROOT) if reference_path.is_relative_to(ROOT) else reference_path),
                      "stop_fraction": teachers[0].stop_fraction,
                      "duration_s": teachers[0].duration},
        "candidate_duration_s": duration, "simulation_steps": completed_steps,
        "elapsed_wall_s": time.monotonic() - started,
        "accepted_episodes": int(valid.sum()), "candidate_episodes": config.num_envs,
        "events": {name: int(value) for name, value in device_events.items()},
        "episodes": episode_records, "reset_seeds": reset_seeds,
        "initial_combinations": initial_combinations,
        "asset": "assets/b2z1/b2z1_0815_with_basket_v3.urdf",
        "unfinished_environments": torch.nonzero(active).flatten().cpu().tolist(),
        "actor_inputs": ["image", "proprio"],
        "scope": "MuJoCo expert candidate; accepted only after complete environment success",
    }
    (output / "report.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    if bool(valid.any()):
        # Preserve the validity mask and episode boundaries. BC explicitly
        # filters invalid candidates and never reads evaluator metadata.
        torch.save({"image": torch.stack(images), "proprio": torch.stack(proprio),
                    "action": torch.stack(actions), "episode_start": torch.stack(boundaries),
                    "valid_transition": torch.stack(valid_transitions),
                    "valid_environment": valid.cpu(), "environment": asdict(config),
                    "report": report}, output / "demonstrations.pt")
    update("GPU 专家复验结束", f"完整通过 {int(valid.sum())}/{config.num_envs}；"
           "只有通过者可进入视觉行为克隆。", output)
    print(json.dumps(report, indent=2), flush=True)
    return report


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", default=str(RUN_DIR / "expert_gpu_single"))
    parser.add_argument("--path", default="reports/basket_v3/edge_parallel/low_edge_candidate.json",
                        help="Candidate arm joint path; normal dynamics and acceptance remain unchanged")
    parser.add_argument("--num-envs", type=int, default=8)
    parser.add_argument("--num-objects", type=int, choices=(1, 3), default=1)
    parser.add_argument("--reset-bank", default="runs/basket_unload/reset_bank_collected.npz")
    parser.add_argument("--episode-seconds", type=float, default=400.)
    parser.add_argument("--leg-target-rate", type=float, default=.2)
    parser.add_argument("--arm-target-rate", type=float, default=.35)
    parser.add_argument("--action-mode", choices=("delta", "position"), default="delta")
    parser.add_argument("--approach-rate", type=float, default=.25)
    parser.add_argument("--approach-before-tilt", action="store_true",
                        help="Independent candidate: approach the basket before leaning the body")
    parser.add_argument("--speed-scale", type=float, default=1.)
    parser.add_argument("--arm-gravity-compensation", action="store_true")
    parser.add_argument("--tool-joint-lift", type=float, default=0.)
    parser.add_argument("--tool-lift-taper-start", type=float, default=None,
                        help="Fade the tool lift to zero from this sweep parameter to stop_fraction; default keeps constant lift.")
    parser.add_argument("--gripper-target", type=float)
    parser.add_argument("--max-seconds", type=float)
    parser.add_argument("--width", type=int, default=64)
    parser.add_argument("--height", type=int, default=48)
    parser.add_argument("--seed", type=int, default=42)
    collect(parser.parse_args())
