"""Evaluate one student Task B episode in Isaac Lab."""

import argparse
import importlib
import inspect
import json
import math
from pathlib import Path
import sys
import time
import traceback

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from taskb_student.constants import JOINT_LOWER, JOINT_UPPER, NUM_OBJECTS, TASK_ID, URDF_PATH
from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser(description="Task B placement-only exercise, B2+Z1.")
parser.add_argument("--task", default=TASK_ID, choices=[TASK_ID])
parser.add_argument("--num_envs", type=int, default=1, choices=[1])
parser.add_argument("--seed", type=int, default=42)
parser.add_argument("--steps", type=int, default=0, help="Optional step limit; 0 runs the episode.")
parser.add_argument("--episode-seconds", type=float, default=1200.0, help="Simulation-time limit.")
parser.add_argument("--max-walltime-s", type=float, default=None, help="Optional real wall-clock limit.")
parser.add_argument("--demo-command", nargs=3, type=float, default=[0.0, 0.0, 0.0], metavar=("VX", "VY", "WZ"))
parser.add_argument("--real-time", action="store_true")
parser.add_argument("--debug", action="store_true")
parser.add_argument("--diagnostics", type=Path, default=None, help="Optional JSONL state trace; never used as policy input.")
parser.add_argument("--sensor-output", type=Path, default=None, help="Save the first ready lidar/camera snapshot for inspection.")
parser.add_argument("--camera-check", type=Path, default=None, help="Check existing RGB-D outputs and save previews/report; requires --enable_cameras.")
parser.add_argument("--record-dir", type=Path, default=None, help="New episode directory for prescribed observations and executed actions.")
parser.add_argument("--record-chunk-steps", type=int, default=10, help="Bounded-memory compressed chunk size.")
parser.add_argument("--record-kind", choices=["smoke", "demonstration"], default="smoke", help="Intent label only; never certifies a successful demonstration.")
parser.add_argument("--solution", default="demo.solution:AlgSolution", help="Observation-only controller module:class; constructor accepts command=, plus reset() and predicts(obs, score).")
AppLauncher.add_app_launcher_args(parser)
parser.set_defaults(device="cpu")
args = parser.parse_args()
if args.steps < 0 or args.episode_seconds <= 0 or (args.max_walltime_s is not None and args.max_walltime_s <= 0):
    parser.error("Time limits must be positive; --steps must be nonnegative.")
if args.device != "cpu":
    parser.error("This PhysX-lidar exercise uses --device cpu for pose readback. RGB-D still renders on the GPU.")
if (args.record_dir is not None or args.camera_check is not None) and not args.enable_cameras:
    parser.error("Recording/camera checks require --enable_cameras for both prescribed RGB-D streams.")
if args.record_chunk_steps < 1:
    parser.error("--record-chunk-steps must be positive.")
for output in (args.record_dir, args.camera_check):
    if output is not None and output.exists():
        parser.error(f"Output already exists; choose a new directory: {output}")
# USD must discover the lidar schema before its registry is first initialized.
args.kit_args += " --enable isaacsim.sensors.physx"
print(f"[Robot] URDF={URDF_PATH}", flush=True)
app = AppLauncher(args).app


def save_sensor_snapshot(obs, directory):
    import numpy as np
    from PIL import Image

    directory.mkdir(parents=True, exist_ok=True)
    buffers = {}
    for group in ("lidar", "image"):
        for name, tensor in obs.get(group, {}).items():
            array = tensor.detach().cpu().numpy()
            buffers[f"{group}_{name}"] = array
            if group == "image" and name.endswith("rgb"):
                Image.fromarray(array[0, ..., :3].astype(np.uint8)).save(directory / f"{name}.png")
    np.savez_compressed(directory / "sensors.npz", **buffers)
    print(f"[Sensors] saved snapshot: {directory}", flush=True)


def play():
    import gymnasium as gym
    import torch

    import taskb_student.registration  # noqa: F401
    from taskb_student.env_cfg import TaskBEnvCfg
    from taskb_student.recording import CameraAudit, EpisodeRecorder, collection_metadata, prescribed_observation

    cfg = TaskBEnvCfg(seed=args.seed)
    cfg.sim.device = args.device
    cfg.episode_length_s = args.episode_seconds
    cfg.configure_cameras(args.enable_cameras)
    module, separator, class_name = args.solution.partition(":")
    if not separator or not class_name:
        raise ValueError("--solution must have the form module:class")
    solution_class = getattr(importlib.import_module(module), class_name)
    solution = solution_class(command=tuple(args.demo_command))
    env = gym.make(args.task, cfg=cfg)
    env.unwrapped._student_debug_sensors = args.debug
    diagnostics = None
    recorder = None
    camera_audit = CameraAudit() if args.record_dir is not None or args.camera_check is not None else None
    reason, score = "exception", 0.0
    try:
        obs, _ = env.reset(seed=args.seed)
        solution.reset()
        score = 0.0
        step = 0
        started = time.monotonic()
        low = torch.tensor(JOINT_LOWER, device=env.unwrapped.device)
        high = torch.tensor(JOINT_UPPER, device=env.unwrapped.device)
        dt = env.unwrapped.step_dt
        if args.record_dir is not None:
            from taskb_student.constants import CONTROL_DT
            if not math.isclose(dt, CONTROL_DT):
                raise ValueError("Recording requires the formal 0.02s control period")
            metadata = collection_metadata(args.seed, args.episode_seconds, args.record_kind)
            import hashlib
            source = Path(inspect.getfile(solution_class))
            metadata["solution"] = {"entrypoint": args.solution, "source_sha256": hashlib.sha256(source.read_bytes()).hexdigest()}
            recorder = EpisodeRecorder(args.record_dir, metadata, args.record_chunk_steps)
        sensor_snapshot_saved = False
        start_position = env.unwrapped.scene["robot"].data.root_pos_w[0, :2].clone() if args.debug or args.diagnostics else None
        if args.diagnostics is not None:
            args.diagnostics.parent.mkdir(parents=True, exist_ok=True)
            diagnostics = args.diagnostics.open("w", encoding="utf-8")
        reason = "exception"
        while True:
            if not app.is_running():
                reason = "simulation_closed"
                break
            tick = time.monotonic()
            if args.max_walltime_s is not None and tick - started >= args.max_walltime_s:
                reason = "wall_clock_limit"
                break
            with torch.inference_mode():
                if camera_audit is not None:
                    captured = prescribed_observation(obs)
                    camera_audit.observe(captured)
                    if recorder is not None:
                        recorder.begin(captured, score, tick - started)
                response = solution.predicts(obs, score)
                if response.get("giveup", False):
                    reason = "solution_giveup"
                    break
                action = torch.as_tensor(response["action"], dtype=torch.float32, device=env.unwrapped.device)
                if action.numel() != 19 or not torch.isfinite(action).all():
                    raise ValueError("Solution must return 19 finite absolute joint targets, in radians.")
                action = torch.clamp(action.reshape(1, 19), low, high)
                executed_action = action.detach().cpu().numpy().copy() if recorder is not None else None
                obs, reward, terminated, truncated, _ = env.step(action)
                # RewardManager multiplies rewards by dt; remove it exactly once for integer points.
                increment = float(reward[0].item()) / dt
                score += increment
                if recorder is not None:
                    recorder.finish_step(executed_action, float(reward[0].item()), score, bool(terminated[0]), bool(truncated[0]))
            step += 1
            state = obs["proprio"]
            if args.debug or diagnostics is not None:
                root = env.unwrapped.scene["robot"].data.root_pos_w[0]
                root_position = root.detach().cpu().tolist()
                drift = float(torch.linalg.vector_norm(root[:2] - start_position).item())
                arm_actual = state["joint_pos"][0, 12:]
                arm_error = float((arm_actual - action[0, 12:]).abs().max().item())
                arm_speed = float(state["joint_vel"][0, 12:].abs().max().item())
                gravity_z = float(state["projected_gravity"][0, 2].item())
                tilt_deg = math.degrees(math.acos(max(-1.0, min(1.0, -gravity_z))))
            if args.sensor_output is not None and not sensor_snapshot_saved and step >= 10:
                images_ready = not args.enable_cameras or all(
                    bool(obs["image"][name].any()) for name in ("head_rgb", "ee_rgb")
                )
                if bool(obs["lidar"]["ready"].all()) and images_ready:
                    save_sensor_snapshot(obs, args.sensor_output)
                    sensor_snapshot_saved = True
            if diagnostics is not None:
                diagnostics.write(json.dumps({
                    "step": step, "sim_time": step * dt, "wall_time": time.monotonic() - started,
                    "score": score, "root_position": root_position, "xy_drift": drift,
                    "tilt_deg": tilt_deg, "arm_error": arm_error, "arm_speed": arm_speed,
                    "joint_positions": state["joint_pos"][0].detach().cpu().tolist(),
                    "joint_targets": action[0].detach().cpu().tolist(),
                    "terminated": bool(terminated[0]), "truncated": bool(truncated[0]),
                }) + "\n")
            if increment > 0.5 or (args.debug and step % 50 == 0):
                print(f"[TaskB] step={step} score={score:.0f}/{NUM_OBJECTS} "
                      f"sim={step * dt:.2f}s wall={time.monotonic() - started:.2f}s", flush=True)
                if args.debug:
                    print(f"[State] root_z={root_position[2]:.3f} drift={drift:.3f} tilt={tilt_deg:.1f}deg "
                          f"arm_error={arm_error:.4f} arm_speed={arm_speed:.3f}", flush=True)
                    lidar = obs["lidar"]
                    print(f"[Sensors] lidar={tuple(lidar['ranges'].shape)} "
                          f"ready={bool(lidar['ready'].all())} hits={int(lidar['hit_mask'].sum())} "
                          f"images={list(obs.get('image', {}))}", flush=True)
            if bool(terminated[0]) or bool(truncated[0]):
                reason = "environment_termination_or_timeout"
                break
            if args.steps and step >= args.steps:
                reason = "step_limit"
                break
            if args.real_time:
                remaining = dt - (time.monotonic() - tick)
                if remaining > 0:
                    time.sleep(remaining)
        print(f"[TaskB] final_score={score:.0f}/{NUM_OBJECTS} sim={step * dt:.2f}s "
              f"wall={time.monotonic() - started:.2f}s stop={reason}", flush=True)
    finally:
        try:
            if recorder is not None:
                recorder.close(reason, score)
            if camera_audit is not None:
                directory = args.camera_check if args.camera_check is not None else args.record_dir / "camera_check"
                report = camera_audit.save(directory)
                print(f"[CameraCheck] {report['status']}: {directory}", flush=True)
                if report["status"] == "not_ready" and reason != "exception":
                    raise RuntimeError("Existing cameras did not pass output checks; inspect camera_check.json before collection.")
        finally:
            if diagnostics is not None:
                diagnostics.close()
            env.close()


if __name__ == "__main__":
    try:
        play()
    except Exception:
        traceback.print_exc()
        raise
    finally:
        app.close(wait_for_replicator=False)
