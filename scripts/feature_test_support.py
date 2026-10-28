"""Shared I/O, route preparation and video utilities for the Isaac feature tests."""

import argparse
from datetime import datetime
import json
import math
from pathlib import Path
import sys
import uuid

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from taskb_student.constants import CONTROL_DT, SPAWN_POSITION
from taskb_student.traversal import NoPathError, plan_path


def write_json(path, value):
    Path(path).write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False), encoding="utf-8")


def make_run_directory(parent):
    parent = Path(parent).expanduser().resolve()
    parent.mkdir(parents=True, exist_ok=True)
    directory = parent / f"{datetime.now():%Y%m%d_%H%M%S}_{uuid.uuid4().hex[:6]}"
    directory.mkdir()
    return directory


def publish_summary(directory, summary):
    write_json(directory / "summary.json", summary)
    write_json(directory.parent / "latest.json", {"directory": str(directory), "success": summary["success"]})
    print(f"[Result] {directory}", flush=True)


def build_parser(mode):
    parser = argparse.ArgumentParser(description=(
        "Rotate once and export one labeled image per detected item." if mode == "recognition"
        else "Visit 18 item stand-off poses and record rendered robot motion as MP4."
    ))
    parser.add_argument("--seed", type=int, default=42 if mode == "recognition" else None)
    parser.add_argument("--output-dir", type=Path, default=ROOT / "runs" / f"{mode}_test",
                        help="Parent directory; every run creates a unique child directory.")
    parser.add_argument("--max-seconds", type=float, default=120.0 if mode == "recognition" else 1200.0,
                        help="Maximum simulation seconds, not wall-clock time.")
    parser.add_argument("--max-walltime-s", type=float, default=None)
    if mode == "recognition":
        parser.add_argument("--head-only", action="store_true", help="Disable wrist near-field scanning for comparison.")
    if mode == "traversal":
        parser.add_argument("--input", type=Path, default=None,
                            help="Recognition run folder, objects.json, or custom route.json; default: latest recognition run.")
        parser.add_argument("--fps", type=int, default=25, help="MP4 frames per simulation second (1..50).")
        parser.add_argument("--pickup", action=argparse.BooleanOptionalAction, default=False,
                            help="Wait for external confirm_pickup() when enabled. Default --no-pickup visits only; no built-in grasp controller.")
        parser.add_argument("--dwell-seconds", type=float, default=3.0,
                            help="Simulation seconds to stop at each pose with --no-pickup (default: 3).")
        parser.add_argument("--per-target-timeout", type=float, default=120.0)
    return parser


def launch(mode, runner, argv=None):
    parser = build_parser(mode)
    argv = sys.argv[1:] if argv is None else argv
    if "--help" in argv or "-h" in argv:
        # Basic help must work without loading Kit or even having Isaac installed.
        parser.add_argument("--headless", action="store_true", help="Render offscreen without a GUI window.")
        parser.add_argument("--device", default="cpu", help="Must be cpu; RTX still uses the GPU.")
        parser.add_argument("--enable_cameras", action="store_true", help="Enabled automatically by both tests.")
        parser.parse_args(argv)
    try:
        from isaaclab.app import AppLauncher
    except ModuleNotFoundError:
        parser.error("Activate the Isaac Lab Python environment used by scripts/run_taskb.sh.")
    AppLauncher.add_app_launcher_args(parser)
    parser.set_defaults(device="cpu")
    args = parser.parse_args(argv)
    if not math.isfinite(args.max_seconds) or args.max_seconds <= 0:
        parser.error("--max-seconds must be finite and positive.")
    if args.max_walltime_s is not None and (not math.isfinite(args.max_walltime_s) or args.max_walltime_s <= 0):
        parser.error("--max-walltime-s must be finite and positive.")
    if args.device != "cpu":
        parser.error("This exercise requires --device cpu; RGB rendering still uses the GPU.")
    if mode == "traversal":
        if not 1 <= args.fps <= 50:
            parser.error("--fps must be between 1 and 50.")
        if not math.isfinite(args.dwell_seconds) or args.dwell_seconds < 0:
            parser.error("--dwell-seconds must be finite and nonnegative.")
        if not math.isfinite(args.per_target_timeout) or args.per_target_timeout <= 0:
            parser.error("--per-target-timeout must be finite and positive.")
        require_video_dependencies()
        args.source = load_navigation_source(args.input)
        if args.seed is not None and args.seed != args.source["seed"]:
            parser.error("--seed differs from the saved input seed; the scene would not match the items.")
        args.seed = args.source["seed"]
    args.enable_cameras = True
    args.kit_args += " --enable isaacsim.sensors.physx"
    app = AppLauncher(args).app
    try:
        return runner(args, app)
    finally:
        app.close(wait_for_replicator=False)


def create_environment(args, *, video=False):
    # Import only after AppLauncher initializes Kit.
    import gymnasium as gym
    import taskb_student.registration  # noqa: F401
    from taskb_student.constants import TASK_ID
    from taskb_student.env_cfg import TaskBEnvCfg

    cfg = TaskBEnvCfg(seed=args.seed)
    cfg.sim.device = "cpu"
    cfg.episode_length_s = args.max_seconds + 1.0
    cfg.configure_cameras(True)
    if video:
        cfg.viewer.eye = (-10.0, -24.0, 18.0)
        cfg.viewer.lookat = (-10.0, -10.0, 0.0)
        cfg.viewer.resolution = (1280, 720)
    env = gym.make(TASK_ID, cfg=cfg, render_mode="rgb_array" if video else None)
    if not math.isclose(env.unwrapped.step_dt, CONTROL_DT):
        env.close()
        raise RuntimeError("Environment timestep differs from the controller/localization timestep.")
    return env


def step_environment(env, response):
    import torch
    from taskb_student.constants import JOINT_LOWER, JOINT_UPPER

    action = torch.as_tensor(response["action"], dtype=torch.float32, device=env.unwrapped.device)
    if action.numel() != 19 or not torch.isfinite(action).all():
        raise ValueError("The controller must return 19 finite joint targets.")
    lower = torch.tensor(JOINT_LOWER, device=action.device)
    upper = torch.tensor(JOINT_UPPER, device=action.device)
    return env.step(torch.clamp(action.reshape(1, 19), lower, upper))


def load_navigation_source(source=None):
    if source is None:
        pointer = ROOT / "runs/recognition_test/latest.json"
        if not pointer.is_file():
            raise FileNotFoundError("Run the recognition test first, or provide --input <route.json>.")
        source = Path(json.loads(pointer.read_text(encoding="utf-8"))["directory"])
    source = Path(source).expanduser().resolve()
    if source.is_dir():
        source = source / "objects.json"
    data = json.loads(source.read_text(encoding="utf-8"))
    if data.get("schema") == "taskb-navigation-v1":
        seed, items = data["seed"], data["items"]
        options = data.get("navigation", {})
    else:
        if not data.get("ready"):
            raise ValueError("Recognition is incomplete. Inspect item images; 18 confirmed items are required.")
        items = [{"object_id": item["object_id"], "class_name": item["class_name"],
                  "position": item["position"], "obstacle_radius": 0.2}
                 for item in data["objects"] if item["confirmed"]]
        summary = json.loads((source.parent / "summary.json").read_text(encoding="utf-8"))
        seed, options = summary["seed"], {}
    if not isinstance(seed, int) or isinstance(seed, bool) or len(items) != 18:
        raise ValueError("Input must contain an integer scene seed and exactly 18 items.")
    allowed = {"robot_radius", "safety_margin", "grid_resolution", "bounds", "position_tolerance", "yaw_tolerance"}
    if set(options) - allowed:
        raise ValueError(f"Unsupported navigation options: {set(options) - allowed}")
    if len({item["object_id"] for item in items}) != 18:
        raise ValueError("Item IDs must be unique.")
    specified = ["target_pose" in item for item in items]
    if any(specified) and not all(specified):
        raise ValueError("Specify target_pose for all 18 items or for none of them.")
    for item in items:
        position = np.asarray(item["position"], dtype=float)
        radius = float(item.get("obstacle_radius", 0.2))
        if position.shape not in ((2,), (3,)) or not np.isfinite(position).all() or not math.isfinite(radius) or radius < 0:
            raise ValueError("Each item needs a finite XY/XYZ position and nonnegative radius.")
        if "target_pose" in item:
            pose = np.asarray(item["target_pose"], dtype=float)
            if pose.shape != (3,) or not np.isfinite(pose).all():
                raise ValueError("target_pose must be finite (x, y, yaw), in meters and radians.")
    return {"seed": seed, "items": items, "navigation": options, "source": str(source)}


def prepare_route(source):
    """Generate navigation-only stand-off poses, or validate explicitly supplied poses."""
    from taskb_student.localization import Localization
    from taskb_student.traversal import Traversal

    items, options = source["items"], source["navigation"]
    positions = [item["position"] for item in items]
    radii = [float(item.get("obstacle_radius", 0.2)) for item in items]
    targets, current = [], SPAWN_POSITION[:2]
    planning_options = {key: value for key, value in options.items()
                        if key not in ("position_tolerance", "yaw_tolerance")}
    stand_off = {"sugar_box": 1.10, "mustard_bottle": 1.05, "banana": 1.00}
    for index, item in enumerate(items):
        if "target_pose" in item:
            pose = tuple(item["target_pose"])
            plan_path(current, pose[:2], positions, obstacle_radii=radii, **planning_options)
        else:
            x, y = item["position"][:2]
            clearance = radii[index] + options.get("robot_radius", 0.6) + options.get("safety_margin", 0.1)
            distance = max(stand_off.get(item.get("class_name"), 1.10), clearance + 0.15)
            start_angle = math.atan2(current[1] - y, current[0] - x)
            candidates = [(x + distance * math.cos(start_angle + j * math.pi / 16),
                           y + distance * math.sin(start_angle + j * math.pi / 16)) for j in range(32)]
            pose = None
            for candidate in sorted(candidates, key=lambda point: math.dist(current, point)):
                try:
                    plan_path(current, candidate, positions, obstacle_radii=radii, **planning_options)
                except NoPathError:
                    continue
                pose = (*candidate, math.atan2(y - candidate[1], x - candidate[0]))
                break
            if pose is None:
                raise NoPathError(f"No reachable stand-off pose for item {item['object_id']}; supply explicit target_pose values.")
        targets.append(pose)
        current = pose[:2]
    nav = Traversal(Localization(), targets, positions, obstacle_radii=radii, **options)
    paths = nav.plan_remaining(remove_after_visit=False)
    return {"schema": "taskb-navigation-v1", "seed": source["seed"], "navigation": options,
            "mode": "visit_only_keep_all_obstacles", "source": source["source"],
            "items": [{**item, "target_pose": pose, "obstacle_radius": radius}
                      for item, pose, radius in zip(items, targets, radii)],
            "planned_paths": [leg.waypoints for leg in paths]}


def require_video_dependencies():
    try:
        import imageio.v2 as imageio
        import imageio_ffmpeg
        imageio_ffmpeg.get_ffmpeg_version()
    except (ImportError, RuntimeError, OSError) as error:
        raise RuntimeError("Install video dependencies in the Isaac Python environment: "
                           "python -m pip install imageio imageio-ffmpeg") from error
    return imageio


class SimulationVideo:
    """Encode on simulation time; slow rendering does not speed up the MP4."""
    def __init__(self, path, fps=25, writer_factory=None):
        if not math.isfinite(fps) or not 1 <= fps <= 1 / CONTROL_DT:
            raise ValueError("FPS must be between 1 and the control frequency.")
        if writer_factory is None:
            writer_factory = require_video_dependencies().get_writer
        self.writer = writer_factory(str(path), format="FFMPEG", fps=fps, codec="libx264",
                                     quality=7, macro_block_size=2, output_params=["-movflags", "+faststart"])
        self.fps, self.frames, self.next_time = fps, 0, 0.0
        self.closed = False

    def due(self, sim_time):
        return sim_time + 1e-9 >= self.next_time

    def append(self, rgb, sim_time, caption):
        from PIL import Image, ImageDraw

        if self.closed:
            raise RuntimeError("Video writer is closed.")
        if not self.due(sim_time):
            return
        array = np.asarray(rgb)
        if array.ndim != 3 or array.shape[2] < 3 or min(array.shape[:2]) < 2:
            raise RuntimeError("env.render() returned no RGB image; check offscreen RTX rendering.")
        if not np.isfinite(array).all():
            raise RuntimeError("Rendered frame contains non-finite pixels.")
        if np.issubdtype(array.dtype, np.floating) and array.max(initial=0) <= 1:
            array = array * 255
        frame = Image.fromarray(np.clip(array[..., :3], 0, 255).astype(np.uint8))
        draw = ImageDraw.Draw(frame)
        draw.rectangle((0, 0, frame.width, 30), fill="black")
        draw.text((10, 8), caption, fill="white")
        self.writer.append_data(np.asarray(frame))
        self.frames += 1
        self.next_time = self.frames / self.fps

    def close(self):
        if not self.closed:
            self.writer.close()
            self.closed = True
