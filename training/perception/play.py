"""Open the standalone B2+Z1 training scene with three kinds of physical objects."""

import argparse
import json
import math
from pathlib import Path
import sys
import time
import traceback

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))


def parse_args():
    from isaaclab.app import AppLauncher

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--steps", type=int, default=0, help="Control steps; 0 runs until the window closes.")
    parser.add_argument("--real-time", action="store_true", help="Limit execution to real simulation time.")
    parser.add_argument("--report", type=Path, help="Optional final scene state for verification.")
    AppLauncher.add_app_launcher_args(parser)
    parser.set_defaults(device="cpu")
    args = parser.parse_args()
    if args.seed < 0 or args.steps < 0:
        parser.error("--seed and --steps must be nonnegative.")
    if args.device != "cpu":
        parser.error("Use --device cpu for this scene; graphical rendering still uses the GPU.")
    return args


def play(args, app):
    import torch

    import isaaclab.sim as sim_utils
    from isaaclab.scene import InteractiveScene

    from scene import OBJECT_ASSETS, make_scene_cfg
    from taskb_student.constants import CONTROL_DT, JOINT_NAMES
    from taskb_student.low_level import LowLevelController

    sim = sim_utils.SimulationContext(sim_utils.SimulationCfg(
        device="cpu", dt=0.005, render_interval=4, use_fabric=False,
        physics_material=sim_utils.RigidBodyMaterialCfg(
            static_friction=1.0, dynamic_friction=1.0, restitution=0.0,
        ),
    ))
    sim.set_camera_view(eye=(3.5, -4.0, 2.7), target=(0.6, 0.0, 0.25))
    scene = InteractiveScene(make_scene_cfg(args.seed))
    controller = LowLevelController()
    sim.reset()
    scene.reset()
    robot = scene["robot"]
    joint_ids, _ = robot.find_joints(JOINT_NAMES, preserve_order=True)
    step = 0
    print("[TrainingScene] B2+Z1 with basket; sugar_box=1, mustard_bottle=1, banana=1", flush=True)
    print(f"[TrainingScene] seed={args.seed}, robot=(0, 0, 0.6), objects in front of robot", flush=True)
    try:
        with torch.inference_mode():
            while app.is_running() and (not args.steps or step < args.steps):
                tick = time.monotonic()
                if not sim.is_playing():
                    sim.render()
                    time.sleep(0.02)
                    continue
                proprio = {
                    "base_ang_vel": robot.data.root_ang_vel_b,
                    "projected_gravity": robot.data.projected_gravity_b,
                    "base_quat": robot.data.root_quat_w,
                    "joint_pos": robot.data.joint_pos[:, joint_ids],
                    "joint_vel": robot.data.joint_vel[:, joint_ids],
                }
                targets = controller.step(proprio, (0.0, 0.0, 0.0))
                robot.set_joint_position_target(torch.from_numpy(targets).unsqueeze(0), joint_ids=joint_ids)
                for _ in range(4):
                    scene.write_data_to_sim()
                    sim.step()
                    scene.update(sim.get_physics_dt())
                step += 1
                if step % 250 == 0:
                    height = float(robot.data.root_pos_w[0, 2])
                    print(f"[TrainingScene] step={step}, sim={step * CONTROL_DT:.2f}s, base_z={height:.3f}m", flush=True)
                if args.real_time:
                    time.sleep(max(0.0, CONTROL_DT - (time.monotonic() - tick)))
    finally:
        if args.report is not None:
            gravity_z = float(robot.data.projected_gravity_b[0, 2])
            report = {
                "seed": args.seed, "steps": step, "sim_time": step * CONTROL_DT,
                "robot_position": robot.data.root_pos_w[0].cpu().tolist(),
                "robot_tilt_deg": math.degrees(math.acos(max(-1.0, min(1.0, -gravity_z)))),
                "joint_positions": robot.data.joint_pos[0, joint_ids].cpu().tolist(),
                "objects": [
                    {
                        "name": f"object_{index + 1:02d}", "asset": asset_name,
                        "position": scene[f"object_{index + 1:02d}"].data.root_pos_w[0].cpu().tolist(),
                        "linear_velocity": scene[f"object_{index + 1:02d}"].data.root_lin_vel_w[0].cpu().tolist(),
                    }
                    for index, asset_name in enumerate(OBJECT_ASSETS)
                ],
            }
            args.report.parent.mkdir(parents=True, exist_ok=True)
            args.report.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
            print(f"[TrainingScene] report={args.report.resolve()}", flush=True)
        # Remove Isaac Lab's standalone STOP callback before shutting down the timeline.
        sim.clear_instance()
        sim.stop()


if __name__ == "__main__":
    args = parse_args()
    from isaaclab.app import AppLauncher

    app = AppLauncher(args).app
    try:
        play(args, app)
    except Exception:
        traceback.print_exc()
        raise
    finally:
        app.close(wait_for_replicator=False)
