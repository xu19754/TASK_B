"""Visible, reproducible launch of the isolated GPU experiment."""
from __future__ import annotations

import argparse
from collections import Counter
from dataclasses import asdict
import json
from pathlib import Path
import time

import torch
import warp as wp

from .env import BasketUnloadEnv, EnvConfig
from .ppo import PPOConfig, train
from .progress import RUN_DIR, update
from .vision import DesktopPreview


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--stage", choices=("balance", "unload"), default="balance")
    parser.add_argument("--num-envs", type=int, default=16)
    parser.add_argument("--num-objects", type=int, choices=(1, 3), default=3)
    parser.add_argument("--steps", type=int, default=100000)
    parser.add_argument("--episode-seconds", type=float)
    parser.add_argument("--leg-target-rate", type=float, default=.2)
    parser.add_argument("--arm-target-rate", type=float, default=.35)
    parser.add_argument("--action-mode", choices=("delta", "position", "reference_residual"), default="delta")
    parser.add_argument("--reference-targets", help="Validated joint-command NPZ for reference_residual mode")
    parser.add_argument("--reference-stride", type=int, default=1,
                        help="Reference samples per policy step; reference dt * stride must equal control dt")
    parser.add_argument("--leg-residual-scale", type=float, default=.04)
    parser.add_argument("--arm-residual-scale", type=float, default=.08)
    parser.add_argument("--gripper-residual-scale", type=float, default=.04)
    parser.add_argument("--clearance-penalty-weight", type=float, default=0.)
    parser.add_argument("--speed-margin-penalty-weight", type=float, default=0.)
    parser.add_argument("--observe-time", action="store_true")
    parser.add_argument("--control-dt", type=float, default=.02)
    parser.add_argument("--gamma", type=float, default=.9998)
    parser.add_argument("--gae-lambda", type=float, default=.99)
    parser.add_argument("--rollout-steps", type=int, default=256)
    parser.add_argument("--learning-rate", type=float, default=3e-4)
    parser.add_argument("--entropy-coef", type=float, default=.005)
    parser.add_argument("--width", type=int, default=128)
    parser.add_argument("--height", type=int, default=96)
    parser.add_argument("--reset-bank", default="runs/basket_unload/reset_bank_collected.npz")
    parser.add_argument("--seed", type=int, default=42)
    initialization = parser.add_mutually_exclusive_group()
    initialization.add_argument("--resume", help="Continue this run, including optimizer and counters")
    initialization.add_argument("--init-checkpoint", help="Initialize weights for a new curriculum stage")
    parser.add_argument("--smoke", action="store_true")
    parser.add_argument("--run-name", default="balance_v1")
    parser.add_argument("--preview-dir", type=Path,
                        help="Separate desktop status/images directory for concurrent experiments")
    parser.add_argument("--max-walltime", "--max-walltime-s", dest="max_walltime",
                        type=float, default=None)
    args = parser.parse_args()
    torch.set_num_threads(4)
    run_dir = RUN_DIR / args.run_name
    run_dir.mkdir(parents=True, exist_ok=True)
    preview_dir = args.preview_dir if args.preview_dir is not None else RUN_DIR
    duration = args.episode_seconds if args.episode_seconds is not None else (5. if args.stage == "balance" else 180.)
    cfg = EnvConfig(num_envs=args.num_envs, stage=args.stage, num_objects=args.num_objects,
                    episode_seconds=duration, image_width=args.width,
                    image_height=args.height, reset_bank=args.reset_bank, seed=args.seed,
                    leg_target_rate=args.leg_target_rate, arm_target_rate=args.arm_target_rate,
                    control_dt=args.control_dt, action_mode=args.action_mode,
                    observe_time=args.observe_time, reference_targets=args.reference_targets,
                    reference_stride=args.reference_stride,
                    leg_residual_scale=args.leg_residual_scale,
                    arm_residual_scale=args.arm_residual_scale,
                    gripper_residual_scale=args.gripper_residual_scale,
                    clearance_penalty_weight=args.clearance_penalty_weight,
                    speed_margin_penalty_weight=args.speed_margin_penalty_weight)
    (run_dir / "environment.json").write_text(json.dumps(asdict(cfg), indent=2), encoding="utf-8")
    update("阶段 5–7 / 9：初始化视觉 GPU 环境", f"{args.num_envs} 个世界，阶段={args.stage}，腕部图像={args.width}×{args.height}；首次编译可能需要数分钟。", run_dir=preview_dir)
    env = BasketUnloadEnv(cfg)
    preview = DesktopPreview(env.model, env.meta, preview_dir)
    last_preview, last_update = 0., 0.

    def callback(metrics, policy, live_env):
        nonlocal last_preview, last_update
        now = time.monotonic()
        if now - last_preview >= .5 and live_env.last_image is not None:
            preview.update(live_env.sim.data, live_env.last_image,
                           asset_gpu=live_env.sim.gpu_model, metrics=metrics)
            last_preview = now
        event = metrics.get("event", "")
        if event in ("update", "checkpoint", "complete") or now - last_update > 15:
            stage = "阶段 8 / 9：强化学习训练" if not args.smoke else "阶段 5–7 / 9：完整管线短测"
            detail = f"阶段={args.stage}，实际采样={metrics.get('global_step', 0)}；{event}。站立阶段的成功不计为卸料成功。"
            if "samples_per_second" in metrics:
                detail += f" 采样 {metrics['samples_per_second']:.1f} transitions/s。"
            update(stage, detail, run_dir=preview_dir, metrics=metrics)
            last_update = now

    try:
        if args.smoke:
            obs = env.reset()
            started = time.monotonic()
            events = Counter()
            returns = 0.
            for step in range(round(cfg.episode_seconds / cfg.control_dt)):
                obs, reward, done, info = env.step(torch.zeros((args.num_envs, 19), device="cuda"))
                assert obs["image"].shape == (args.num_envs, 5, args.height, args.width)
                assert obs["proprio"].shape == (args.num_envs, 67 + int(cfg.observe_time))
                assert torch.isfinite(reward).all() and torch.isfinite(obs["image"]).all()
                returns += float(reward.mean())
                events.update({k: int(v.sum()) for k, v in info["events"].items()})
                callback({"event": "smoke", "global_step": (step + 1) * args.num_envs}, None, env)
            wp.synchronize()
            report = dict(stage=args.stage, physics="CUDA MuJoCo Warp", policy_training=False,
                          environments=args.num_envs, duration_sim_s=cfg.episode_seconds,
                          elapsed_wall_s=time.monotonic() - started, mean_return=returns,
                          events=dict(events), image_shape=list(obs["image"].shape),
                          proprio_shape=list(obs["proprio"].shape))
            (run_dir / "smoke.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
            print(json.dumps(report, indent=2), flush=True)
            update("完整 GPU 视觉管线短测完成", "已记录实际动力学、图像和失败统计；接下来检查结果后进入 PPO。", run_dir=preview_dir, report=report)
        else:
            config = PPOConfig(total_timesteps=args.steps, seed=args.seed,
                               max_walltime_seconds=args.max_walltime,
                               gamma=args.gamma, gae_lambda=args.gae_lambda,
                               rollout_steps=args.rollout_steps,
                               learning_rate=args.learning_rate,
                               entropy_coef=args.entropy_coef)
            checkpoint = train(env, run_dir, config, callback, args.resume,
                               init_checkpoint=args.init_checkpoint)
            update("本轮训练已保存", f"{checkpoint}；接下来进行独立评估并据结果继续下一阶段。", run_dir=preview_dir)
    finally:
        preview.close()


if __name__ == "__main__":
    main()
