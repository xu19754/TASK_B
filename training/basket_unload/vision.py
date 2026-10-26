"""Batched wrist RGB-D and a separate, diagnostic-only desktop preview.

The policy camera renders only the attached ``ee_camera``. The external camera
and all CPU state transfers below are for the human monitor and never enter the
actor observation. Depth is optical-axis depth in metres before normalization.
"""

from __future__ import annotations

import json
import math
import os
from pathlib import Path
import subprocess
import time

# The viewer is a Tk dashboard; MuJoCo needs only an offscreen GL context.
os.environ.setdefault("MUJOCO_GL", "egl")

import mujoco
import mujoco_warp as mjw
import numpy as np
from PIL import Image
import torch
from torch import Tensor
import warp as wp


def intrinsics(model, width: int, height: int) -> np.ndarray:
    """Pinhole intrinsics, with camera x right, y down and optical z forward."""
    camera_id = int(model.camera("ee_camera").id)
    focal = .5 * height / math.tan(math.radians(float(model.cam_fovy[camera_id])) / 2)
    return np.array([[focal, 0., width / 2], [0., focal, height / 2], [0., 0., 1.]])


class GPUCamera:
    """Render all worlds on CUDA, retaining the live GPU model reference.

    Changes to the supplied model's per-world ``geom_dataid`` therefore select
    the corresponding cargo mesh in both physics and images. Output channels
    are RGB in [0,1], clipped depth/2 m, and a .05..50 m validity mask.
    """

    def __init__(self, model, gpu_model, gpu_data, nworld, width=128, height=96):
        self.model = model
        self.gpu_model = gpu_model
        self.gpu_data = gpu_data
        self.nworld, self.width, self.height = int(nworld), int(width), int(height)
        self.device = gpu_data.qpos.device
        with wp.ScopedDevice(self.device):
            self.context = mjw.create_render_context(
                model, nworld=self.nworld, cam_res=(self.width, self.height),
                cam_active=["ee_camera"], render_rgb=True, render_depth=True,
                enabled_geom_groups=[0, 2], use_shadows=False,
            )
        self.intrinsics = intrinsics(model, self.width, self.height)

    @torch.no_grad()
    def render(self) -> Tensor:
        with wp.ScopedDevice(self.device):
            # render() does not refit the scene BVH in MuJoCo Warp 3.17.
            # Without this, moving geometry is traced against qpos0 bounds.
            mjw.refit_bvh(self.gpu_model, self.gpu_data, self.context)
            mjw.render(self.gpu_model, self.gpu_data, self.context)
        # Establish a dependency rather than synchronizing the entire device.
        torch_stream = torch.cuda.current_stream(wp.to_torch(self.gpu_data.qpos).device)
        torch_stream.wait_stream(wp.stream_to_torch(wp.get_stream(self.device)))
        packed = wp.to_torch(self.context.rgb_data).to(torch.int64).reshape(
            self.nworld, self.height, self.width
        )
        rgb = torch.stack(
            [((packed >> shift) & 255).float() / 255 for shift in (16, 8, 0)], dim=1
        )
        depth = wp.to_torch(self.context.depth_data).reshape(
            self.nworld, 1, self.height, self.width
        )
        valid = torch.isfinite(depth) & (depth >= .05) & (depth < 50.)
        normalized_depth = torch.where(valid, (depth / 2.).clamp(0, 1), 0.)
        return torch.cat((rgb, normalized_depth, valid.float()), dim=1)


def _save_png(array: np.ndarray, path: Path) -> None:
    temporary = path.with_suffix(".tmp.png")
    Image.fromarray(array).save(temporary)
    temporary.replace(path)


def _depth_picture(depth: np.ndarray, valid: np.ndarray) -> np.ndarray:
    # Diagnostic false colour; the network receives the numerical channel.
    near = 1 - np.clip(depth, 0, 1)
    colour = np.stack((near, 1 - np.abs(near * 2 - 1), 1 - near), axis=-1)
    colour[~valid] = 0
    return np.asarray(colour * 255, dtype=np.uint8)


class DesktopPreview:
    """One real external frame and the actual policy RGB-D, written atomically."""

    def __init__(self, model, meta, run_dir, width=800, height=600, record_video=False):
        self.model, self.meta = model, meta
        self.run_dir = Path(run_dir)
        self.run_dir.mkdir(parents=True, exist_ok=True)
        self.data = mujoco.MjData(model)
        self.renderer = mujoco.Renderer(model, height=height, width=width)
        self.camera = mujoco.MjvCamera()
        mujoco.mjv_defaultCamera(self.camera)
        self.camera.distance = 1.75
        self.camera.azimuth = 145
        self.camera.elevation = -22
        self.options = mujoco.MjvOption()
        self.options.geomgroup[:] = 0
        self.options.geomgroup[[0, 2]] = 1
        self.record_video = record_video
        self.video_path = self.run_dir / "preview.mp4" if record_video else None
        self.video_index_path = self.run_dir / "preview_frames.jsonl" if record_video else None
        self.video_frames = 0
        self._video_process = self._video_index = self._video_log = None
        self._video_started_at = time.monotonic()

    def _record_frame(self, rgb, stamp):
        """Encode actual overview samples at 2 fps, not simulation real time."""
        if not self.record_video:
            return
        if self._video_process is None:
            height, width = rgb.shape[:2]
            self._video_index = self.video_index_path.open("w", encoding="utf-8")
            self._video_log = (self.run_dir / "preview_ffmpeg.log").open("wb")
            try:
                self._video_process = subprocess.Popen([
                    "/usr/bin/ffmpeg", "-y", "-hide_banner", "-loglevel", "error",
                    "-f", "rawvideo", "-pix_fmt", "rgb24", "-video_size", f"{width}x{height}",
                    "-framerate", "2", "-i", "pipe:0", "-an", "-c:v", "libx264",
                    "-preset", "veryfast", "-vf", "pad=ceil(iw/2)*2:ceil(ih/2)*2",
                    "-pix_fmt", "yuv420p", "-movflags", "+faststart", str(self.video_path),
                ], stdin=subprocess.PIPE, stdout=subprocess.DEVNULL, stderr=self._video_log)
            except OSError:
                self._video_index.close()
                self._video_log.close()
                raise
        try:
            self._video_process.stdin.write(np.ascontiguousarray(rgb, dtype=np.uint8).tobytes())
        except BrokenPipeError as error:
            raise RuntimeError("Preview video encoder failed; see preview_ffmpeg.log") from error
        record = dict(stamp, frame_index=self.video_frames, video_time_s=self.video_frames / 2.,
                      wall_elapsed_s=time.monotonic() - self._video_started_at)
        self._video_index.write(json.dumps(record, default=str) + "\n")
        self._video_index.flush()
        self.video_frames += 1

    def update(self, gpu_data, image, world_id=0, metrics=None, asset_gpu=None):
        """Copy only the selected world for human viewing, then render its state.

        ``asset_gpu`` may be the GPU model with world-batched geometry arrays;
        this keeps the diagnostic CPU rendering consistent with cargo class
        randomization. Its state is never used as a policy input.
        """
        if asset_gpu is not None:
            for field in ("geom_dataid", "geom_pos", "geom_quat", "geom_size", "geom_rgba"):
                array = wp.to_torch(getattr(asset_gpu, field))
                selected = array[world_id % array.shape[0]].cpu().numpy()
                getattr(self.model, field)[:] = selected
        mjw.get_data_into(self.data, self.model, gpu_data, world_id=int(world_id))
        self.update_cpu(self.data, image[world_id], metrics=metrics)

    def update_cpu(self, data, image=None, metrics=None):
        """Render an actual CPU dynamics state, also used before GPU startup."""
        self.camera.lookat[:] = data.xpos[self.meta["base_id"]] + [-.12, 0, .03]
        self.renderer.update_scene(data, camera=self.camera, scene_option=self.options)
        overview = self.renderer.render().copy()
        _save_png(overview, self.run_dir / "latest_world.png")
        if image is None:
            # CPU-only startup path uses the same physical wrist camera.
            self.renderer.update_scene(data, camera="ee_camera", scene_option=self.options)
            rgb = self.renderer.render().copy()
            self.renderer.enable_depth_rendering()
            depth = self.renderer.render().copy()
            self.renderer.disable_depth_rendering()
            valid = np.isfinite(depth) & (depth >= .05) & (depth < 50. - .01)
            depth = np.where(valid, np.clip(depth / 2., 0, 1), 0)
        else:
            channels = image.detach().float().cpu().numpy() if isinstance(image, Tensor) else np.asarray(image)
            rgb = np.asarray(np.clip(channels[:3].transpose(1, 2, 0), 0, 1) * 255, dtype=np.uint8)
            depth, valid = channels[3], channels[4] > .5
        _save_png(rgb, self.run_dir / "latest_rgb.png")
        _save_png(_depth_picture(depth, valid), self.run_dir / "latest_depth.png")
        stamp = dict(simulation_time=float(data.time), external_camera="diagnostic_only",
                     policy_camera="ee_camera", world_frame_current=True)
        if metrics:
            stamp["metrics"] = metrics
        temporary = self.run_dir / "preview_frame.tmp.json"
        temporary.write_text(json.dumps(stamp, default=str), encoding="utf-8")
        temporary.replace(self.run_dir / "preview_frame.json")
        self._record_frame(overview, stamp)

    def close(self):
        encoder_error = None
        try:
            if self._video_process is not None:
                try:
                    self._video_process.stdin.close()
                except BrokenPipeError:
                    pass
                try:
                    code = self._video_process.wait(timeout=30)
                    if code:
                        encoder_error = f"Preview video encoder exited with {code}; see preview_ffmpeg.log"
                except subprocess.TimeoutExpired:
                    self._video_process.kill()
                    self._video_process.wait(timeout=5)
                    encoder_error = "Preview video encoder timed out; see preview_ffmpeg.log"
                self._video_process = None
        finally:
            if self._video_index is not None:
                self._video_index.close()
            if self._video_log is not None:
                self._video_log.close()
            self.renderer.close()
        if encoder_error:
            raise RuntimeError(encoder_error)


def initial_CPU_preview(model, meta, run_dir, duration_s=5., fps=10, loaded=False):
    """Display a 5 s floating-base PD diagnostic, with no prescribed base motion.

    Cargo placement here is an explicitly labelled reset-only diagnostic. Loaded
    cargo starts in separated near-flat poses, then falls and contacts naturally.
    This function does not execute a learned unloading policy.
    """
    from .model import initialize_data

    data = initialize_data(model, meta)
    if loaded:
        from .assets_gpu import _place_random_cargo

        rng = np.random.default_rng(0)
        for _ in range(100):
            if _place_random_cargo(model, data, meta, rng):
                break
        else:
            raise RuntimeError("Could not initialize separated diagnostic cargo")
    else:
        for index, address in enumerate(meta["cargo_qpos_adr"]):
            data.qpos[address:address + 3] = [2 + index * .3, 2, .3]
    mujoco.mj_forward(model, data)
    preview = DesktopPreview(model, meta, run_dir)
    initial_height = float(data.qpos[2])
    min_height = initial_height
    max_tilt = 0.
    frame_interval = max(1, int(round(1 / (fps * model.opt.timestep))))
    started = time.perf_counter()
    frames = 0
    try:
        preview.update_cpu(data)
        for step in range(int(round(duration_s / model.opt.timestep))):
            # ctrl remains the 19 nominal PD targets; gravity and contacts act.
            mujoco.mj_step(model, data)
            min_height = min(min_height, float(data.qpos[2]))
            max_tilt = max(max_tilt, math.acos(np.clip(data.xmat[meta["base_id"], 8], -1, 1)))
            if (step + 1) % frame_interval == 0:
                preview.update_cpu(data)
                frames += 1
                # Keep diagnostic animation observable in the existing desktop.
                remaining = (step + 1) * model.opt.timestep - (time.perf_counter() - started)
                if remaining > 0:
                    time.sleep(min(remaining, 1 / fps))
        report = dict(
            duration_s=float(data.time), frames=frames, loaded=loaded,
            initial_base_height_m=initial_height, minimum_base_height_m=min_height,
            final_base_height_m=float(data.qpos[2]), max_tilt_degrees=math.degrees(max_tilt),
            control="19 nominal PD targets; floating base; no learned policy",
            initialization="reset-only cargo placement" if loaded else "reset-only cargo away from robot",
            camera="physical link06 wrist RGB-D; external view diagnostic only",
        )
        if loaded:
            rotation = data.xmat[meta["basket_id"]].reshape(3, 3)
            vertices = [
                (source @ data.xmat[body].reshape(3, 3).T + data.xpos[body]
                 - data.xpos[meta["basket_id"]]) @ rotation
                for source, body in zip(meta["cargo_vertices"], meta["cargo_body_ids"])
            ]
            report["cargo_all_retained"] = all(
                np.all(np.abs(v[:, 0]) < .223) and np.all(np.abs(v[:, 1]) < .124)
                and np.all(v[:, 2] > -.002) and np.all(v[:, 2] < .18)
                for v in vertices
            )
        output = Path(run_dir) / ("preview_loaded.json" if loaded else "preview_empty.json")
        output.write_text(json.dumps(report, indent=2), encoding="utf-8")
        return report
    finally:
        preview.close()


def validate_camera_geometry(model, meta, run_dir, width=128, height=96):
    """Isolated reset-pose camera checks; never a control or unloading test."""
    from .assets_gpu import AssetBank, _place_random_cargo
    from .model import ROOT, initialize_data, prepare_targets

    output = Path(run_dir) / "camera_diagnostics"
    output.mkdir(parents=True, exist_ok=True)
    data = initialize_data(model, meta)
    if not _place_random_cargo(model, data, meta, np.random.default_rng(0)):
        raise RuntimeError("Diagnostic camera layout starts with overlapping cargo")
    for _ in range(round(3. / model.opt.timestep)):
        mujoco.mj_step(model, data)
    wp.init()
    wp.set_device("cuda:0")
    wp.set_stream(wp.stream_from_torch(torch.cuda.current_stream()))
    asset_bank = AssetBank.from_model(model, meta)
    gpu_model = mjw.put_model(model, batch_sizes=asset_bank.batch_sizes(2))
    gpu_data = mjw.put_data(model, data, nworld=2, nconmax=512, njmax=1024)
    mjw.forward(gpu_model, gpu_data)
    camera = GPUCamera(model, gpu_model, gpu_data, 2, width, height)
    renderer = mujoco.Renderer(model, height=height, width=width)
    options = mujoco.MjvOption()
    options.geomgroup[:] = 0
    options.geomgroup[[0, 2]] = 1
    nominal_qpos = data.qpos.copy()
    candidate = json.loads((ROOT / "reports/basket_v3/edge_parallel/low_edge_candidate.json").read_text())
    cases = [
        ("nominal_folded", None),
        ("isolated_old_sweep_preparation", np.asarray(candidate["q_nodes_rad"][0])),
        ("physical_wrist_pd_motion", "pd_wrist"),
    ]
    reports = []
    first_camera_position = None
    first_camera_matrix = None
    try:
        for name, arm_q in cases:
            data.qpos[:] = nominal_qpos
            is_pd_motion = isinstance(arm_q, str)
            if is_pd_motion:
                # Real joint-target control after diagnostic nominal reset.
                data.qvel[:] = 0
                requested = meta["nominal_q"].copy()
                requested[17] -= .3
                previous = meta["nominal_q"].copy()
                for _ in range(50):
                    previous = prepare_targets(
                        meta, requested, previous, data.qpos[meta["joint_qpos_adr"]], dt=.02
                    )
                    data.ctrl[meta["actuator_ids"]] = previous
                    for _ in range(round(.02 / model.opt.timestep)):
                        mujoco.mj_step(model, data)
            elif arm_q is not None:
                # Explicit isolated reset to inspect a field of view only.
                data.qpos[meta["joint_qpos_adr"][12:]] = arm_q
            mujoco.mj_forward(model, data)
            wp.to_torch(gpu_data.qpos).copy_(torch.as_tensor(data.qpos, device="cuda", dtype=torch.float32))
            mjw.forward(gpu_model, gpu_data)
            image = camera.render()
            gpu_depth = wp.to_torch(camera.context.depth_data)[0].reshape(height, width).cpu().numpy().copy()
            renderer.update_scene(data, camera="ee_camera", scene_option=options)
            rgb = renderer.render().copy()
            renderer.enable_depth_rendering()
            gl_depth = renderer.render().copy()
            renderer.disable_depth_rendering()
            renderer.enable_segmentation_rendering()
            segmentation = renderer.render().copy()
            renderer.disable_segmentation_rendering()
            valid = (gpu_depth >= .05) & (gpu_depth < 49.99) & (gl_depth >= .05) & (gl_depth < 49.99)
            # Remove depth discontinuities, where rasterization and ray tracing
            # can select opposite sides of an edge at a subpixel boundary.
            smooth = np.ones_like(valid)
            for axis in (0, 1):
                for direction in (-1, 1):
                    smooth &= np.abs(gl_depth - np.roll(gl_depth, direction, axis)) < .01
            comparison = valid & smooth
            errors = np.abs(gpu_depth[comparison] - gl_depth[comparison])
            geom_ids = segmentation[..., 0]
            geom_type = segmentation[..., 1] == int(mujoco.mjtObj.mjOBJ_GEOM)
            basket_geoms = np.flatnonzero(
                (model.geom_bodyid == meta["basket_id"]) & (model.geom_group == 2)
            )
            cargo_geoms = np.flatnonzero(np.isin(model.geom_bodyid, meta["cargo_body_ids"]))
            cid = meta["ee_camera_id"]
            gpu_position = wp.to_torch(gpu_data.cam_xpos)[0, cid].cpu().numpy()
            gpu_matrix = wp.to_torch(gpu_data.cam_xmat)[0, cid].cpu().numpy().reshape(3, 3)
            np.testing.assert_allclose(gpu_position, data.cam_xpos[cid], atol=2e-6)
            np.testing.assert_allclose(gpu_matrix, data.cam_xmat[cid].reshape(3, 3), atol=2e-6)
            report = dict(
                name=name, diagnostic_reset_only=not is_pd_motion,
                control="rate-limited wrist PD for 1 s" if is_pd_motion else "isolated reset pose",
                basket_pixels=int((geom_type & np.isin(geom_ids, basket_geoms)).sum()),
                cargo_pixels=int((geom_type & np.isin(geom_ids, cargo_geoms)).sum()),
                comparison_pixels=int(comparison.sum()),
                optical_depth_mean_abs_error_m=float(errors.mean()) if errors.size else None,
                optical_depth_p99_abs_error_m=float(np.quantile(errors, .99)) if errors.size else None,
                camera_world_position=data.cam_xpos[cid].tolist(),
                camera_world_optical_axis=(-data.cam_xmat[cid].reshape(3, 3)[:, 2]).tolist(),
            )
            if errors.size < 100 or float(errors.mean()) > .005 or float(np.quantile(errors, .99)) > .015:
                raise AssertionError(f"CPU/GPU optical depth check failed: {report}")
            if first_camera_position is None:
                first_camera_position = data.cam_xpos[cid].copy()
                first_camera_matrix = data.cam_xmat[cid].copy()
            else:
                report["wrist_camera_translation_change_m"] = float(np.linalg.norm(data.cam_xpos[cid] - first_camera_position))
                report["wrist_camera_rotation_matrix_change"] = float(np.linalg.norm(data.cam_xmat[cid] - first_camera_matrix))
            reports.append(report)
            channels = image[0].cpu().numpy()
            _save_png(rgb, output / f"{name}_gl_rgb.png")
            _save_png(np.asarray(channels[:3].transpose(1, 2, 0) * 255, dtype=np.uint8), output / f"{name}_gpu_rgb.png")
            _save_png(_depth_picture(channels[3], channels[4] > .5), output / f"{name}_gpu_depth.png")
        # Verify that a live, batched asset switch changes the rendered image.
        data.qpos[:] = nominal_qpos
        data.qpos[meta["joint_qpos_adr"][12:]] = np.asarray(candidate["q_nodes_rad"][0])
        wp.to_torch(gpu_data.qpos).copy_(torch.as_tensor(data.qpos, device="cuda", dtype=torch.float32))
        class_ids = torch.tensor([[0, 1, 2], [1, 2, 0]], device="cuda")
        asset_bank.apply(gpu_model, class_ids, gpu_data=gpu_data)
        mjw.forward(gpu_model, gpu_data)
        variants = camera.render()
        variant_difference = float((variants[0] - variants[1]).abs().mean().item())
        if variant_difference <= 1e-5:
            raise AssertionError("Batched cargo asset switch did not change the wrist images")
        for index in range(2):
            colour = variants[index, :3].cpu().numpy().transpose(1, 2, 0)
            _save_png(np.asarray(colour * 255, dtype=np.uint8), output / f"batch_asset_world_{index}.png")
        # The image must be resized with its corresponding intrinsics.
        scaled_reference = meta["camera_intrinsics"].copy()
        scaled_reference[0] *= width / meta["camera_resolution"][0]
        scaled_reference[1] *= height / meta["camera_resolution"][1]
        np.testing.assert_allclose(camera.intrinsics, scaled_reference, atol=1e-8)
        result = dict(
            engine=f"MuJoCo {mujoco.__version__} / MuJoCo Warp", nworld=2,
            width=width, height=height, intrinsics=camera.intrinsics.tolist(),
            depth_semantics="optical-axis metres; normalized by 2m only after rendering",
            geom_groups=[0, 2], camera_pose_cpu_gpu_match=True, intrinsics_match=True,
            batched_asset_image_mean_abs_difference=variant_difference,
            optical_depth_checks_passed=True,
            cases=reports,
            scope="Camera geometry/visibility diagnostic only; old sweep pose is a reset, not learned or dynamically reached",
        )
        (output / "report.json").write_text(json.dumps(result, indent=2), encoding="utf-8")
        return result
    finally:
        renderer.close()
