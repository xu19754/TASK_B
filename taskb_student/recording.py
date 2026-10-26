"""Bounded-memory recording of prescribed observations and executed actions.

No access to env, scene objects, physical state APIs or evaluator internals.
Each row is obs_t -> action_t -> feedback_t+1, never a reset observation
pretending to be the final state of a terminated episode.
"""

from datetime import datetime, timezone
import hashlib
from importlib.metadata import PackageNotFoundError, version
import json
from pathlib import Path
import platform
import xml.etree.ElementTree as ET

import numpy as np

from .constants import CONTROL_DT, JOINT_LOWER, JOINT_NAMES, JOINT_UPPER, POLICY_PATH, ROOT, TASK_ID, URDF_PATH
from .sensor_contract import CAMERAS, CAMERA_COMMON, IMAGE_SHAPES, OBSERVATION_SHAPES


def array_copy(value):
    if hasattr(value, "detach"):
        value = value.detach().cpu().numpy()
    return np.array(value, copy=True)


def prescribed_observation(obs):
    """Copy before predicts/step: simulator buffers may be mutated in place."""
    arrays = {}
    for group, terms in OBSERVATION_SHAPES.items():
        for key, shape in terms.items():
            value = array_copy(obs[group][key])
            if value.shape != shape or value.dtype.kind not in "bufi":
                raise ValueError(f"Invalid observation {group}/{key}: {value.shape}, {value.dtype}; expected {shape}")
            if group == "proprio" and not np.isfinite(value).all():
                raise ValueError(f"Nonfinite proprioception: {key}")
            arrays[f"obs/{group}/{key}"] = value
    return arrays


def write_json(path, value):
    path = Path(path)
    temporary = path.with_name(path.name + ".tmp")
    temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    temporary.replace(path)


def collection_metadata(seed, episode_seconds, source_kind):
    """Source fingerprints identify the formal scene without exporting object poses."""
    files = [
        "taskb_student/env_cfg.py", "taskb_student/sensor_contract.py", "taskb_student/robot.py",
        "taskb_student/arena.py", "taskb_student/geometry.py", "taskb_student/constants.py",
        "taskb_student/mdp.py", "taskb_student/lidar.py", "taskb_student/scoring.py",
        "taskb_student/recording.py", "taskb_student/low_level.py", "demo/solution.py",
        "scripts/play_taskb.py", "models/locomotion/deployment_manifest.json", "assets/b2z1/SOURCE.json",
        URDF_PATH.relative_to(ROOT).as_posix(), POLICY_PATH.relative_to(ROOT).as_posix(),
    ]
    for mesh in ET.parse(URDF_PATH).getroot().findall(".//mesh"):
        files.append((URDF_PATH.parent / mesh.attrib["filename"]).relative_to(ROOT).as_posix())
    for name in ("004_sugar_box", "006_mustard_bottle", "011_banana"):
        files.extend([f"assets/objects/{name}.usd", f"assets/objects/Materials/Textures/{name}_COLOR.png"])
    versions = {"python": platform.python_version()}
    for package in ("numpy", "torch", "isaaclab", "isaacsim"):
        try:
            versions[package] = version(package)
        except PackageNotFoundError:
            versions[package] = None
    return {
        "task": TASK_ID, "seed": seed, "episode_seconds": episode_seconds,
        "source_kind": source_kind, "demonstration_success": "unreviewed",
        "scene": "formal TaskBEnvCfg; no alternate collection scene",
        "physics_device": "cpu", "control_dt_s": CONTROL_DT,
        "joint_names": JOINT_NAMES, "action_units": "rad", "action_semantics": "absolute joint position after clamp",
        "cameras": {name: {**CAMERA_COMMON, **mount} for name, mount in CAMERAS.items()},
        "source_sha256": {name: hashlib.sha256((ROOT / name).read_bytes()).hexdigest() for name in sorted(set(files))},
        "runtime_versions": versions,
        "sensor_timing": "Policy observations at 50 Hz; cameras target 10 Hz. Repeated image samples are retained. No hardware/exposure timestamps are available in prescribed observations.",
        "depth_semantics": "Optical-axis depth in metres; retain original invalid values; use finite & depth > 0 when backprojecting with K.",
        "row_alignment": "obs_t and score_before are copied before predicts and env.step; action is the target submitted to env.step; reward/score_after/done are returned feedback. No next_obs is saved because Isaac auto-resets terminal environments.",
        "excluded": ["object ground truth", "object IDs", "scene state", "global overhead images", "teacher diagnostics"],
    }


class EpisodeRecorder:
    def __init__(self, directory, metadata, chunk_steps=10):
        if chunk_steps < 1:
            raise ValueError("chunk_steps must be positive")
        self.directory = Path(directory)
        self.directory.mkdir(parents=True, exist_ok=False)
        self.chunk_steps = chunk_steps
        self.rows = []
        self.pending = None
        self.closed = False
        self.episode_done = False
        self.manifest = {
            "format": "taskb-observation-action-v1", "created_utc": datetime.now(timezone.utc).isoformat(),
            "status": "recording", "metadata": metadata, "steps": 0, "chunks": [],
        }
        write_json(self.directory / "manifest.json", self.manifest)

    def begin(self, observation, score_before, wall_time):
        if self.closed or self.pending is not None or self.episode_done:
            raise RuntimeError("Recorder closed or preceding action has no feedback")
        expected = {f"obs/{group}/{key}": shape for group, terms in OBSERVATION_SHAPES.items() for key, shape in terms.items()}
        if set(observation) != set(expected) or any(observation[key].shape != shape for key, shape in expected.items()):
            raise ValueError("Recorder accepts only the prescribed observation schema")
        self.pending = {**observation, "step": np.int64(self.manifest["steps"]),
                        "sim_time": np.float64(self.manifest["steps"] * CONTROL_DT),
                        "wall_time": np.float64(wall_time), "score_before": np.float64(score_before)}

    def finish_step(self, action, reward, score_after, terminated, truncated):
        if self.pending is None or self.closed:
            raise RuntimeError("No pending observation")
        action = array_copy(action).reshape(1, 19)
        if not np.isfinite(action).all() or np.any(action < np.array(JOINT_LOWER) - 1e-6) or np.any(action > np.array(JOINT_UPPER) + 1e-6):
            raise ValueError("Record only finite executed 19-joint actions inside limits")
        self.pending.update(action=action, reward=np.float64(reward), score_after=np.float64(score_after),
                            terminated=np.bool_(terminated), truncated=np.bool_(truncated))
        self.rows.append(self.pending)
        self.pending = None
        self.episode_done = bool(terminated or truncated)
        self.manifest["steps"] += 1
        if len(self.rows) >= self.chunk_steps:
            self.flush()

    def flush(self):
        if not self.rows:
            return
        name = f"chunk_{len(self.manifest['chunks']):06d}.npz"
        target = self.directory / name
        temporary = target.with_suffix(".npz.tmp")
        arrays = {key: np.stack([row[key] for row in self.rows]) for key in self.rows[0]}
        with temporary.open("wb") as stream:
            np.savez_compressed(stream, **arrays)
        temporary.replace(target)
        self.manifest["chunks"].append({
            "file": name, "steps": len(self.rows),
            "sha256": hashlib.sha256(target.read_bytes()).hexdigest(),
        })
        self.rows.clear()
        self.manifest["persisted_steps"] = sum(chunk["steps"] for chunk in self.manifest["chunks"])
        write_json(self.directory / "manifest.json", self.manifest)

    def close(self, reason, final_score):
        if self.closed:
            return
        self.manifest["discarded_pending_step"] = self.pending is not None
        self.pending = None
        self.flush()
        self.manifest.update(status="closed", stop_reason=reason, final_score=float(final_score))
        write_json(self.directory / "manifest.json", self.manifest)
        self.closed = True


class CameraAudit:
    """Checks prescribed outputs only; cannot certify physical mount accuracy."""
    def __init__(self):
        self.samples = 0
        self.ready_samples = 0
        self.last = None
        self.preview = None
        self.previous_hashes = {}
        self.changes = {name: 0 for name in CAMERAS}

    def observe(self, arrays):
        for key, shape in IMAGE_SHAPES.items():
            if arrays[f"obs/image/{key}"].shape != shape:
                raise ValueError(f"Invalid camera shape: {key}")
        self.samples += 1
        checks = {}
        for name in CAMERAS:
            rgb = arrays[f"obs/image/{name}_rgb"][0]
            depth = arrays[f"obs/image/{name}_depth"][0, ..., 0]
            k = arrays[f"obs/image/{name}_intrinsics"][0]
            valid = np.isfinite(depth) & (depth >= CAMERA_COMMON["clipping_range"][0]) & (depth <= CAMERA_COMMON["clipping_range"][1])
            focal_px = CAMERA_COMMON["width"] * CAMERA_COMMON["focal_length"] / CAMERA_COMMON["horizontal_aperture"]
            expected = np.array([[focal_px, 0, 320], [0, focal_px, 240], [0, 0, 1]])
            k_ok = bool(np.isfinite(k).all() and k[0, 0] > 0 and k[1, 1] > 0
                        and np.allclose(k[:2], expected[:2], rtol=0.01, atol=1e-3)
                        and np.allclose(k[2], [0, 0, 1], rtol=0, atol=1e-5))
            rgb_ok = bool(np.isfinite(rgb).all() and rgb.min() >= 0 and rgb.max() <= 255 and rgb.max() > 0)
            digest = hashlib.sha256(rgb.tobytes() + depth.tobytes()).hexdigest()
            if name in self.previous_hashes and self.previous_hashes[name] != digest:
                self.changes[name] += 1
            self.previous_hashes[name] = digest
            checks[name] = {
                "intrinsics_match_existing_optics": k_ok, "intrinsics": k.tolist() if np.isfinite(k).all() else None,
                "rgb_nonblack_in_0_255": rgb_ok, "rgb_dtype": str(rgb.dtype),
                "valid_depth_fraction": float(valid.mean()), "invalid_depth_fraction": float((~valid).mean()),
                "depth_median_m": float(np.median(depth[valid])) if valid.any() else None,
                "data_ready": bool(k_ok and rgb_ok and valid.any()),
            }
        ready = all(check["data_ready"] for check in checks.values())
        self.ready_samples += int(ready)
        self.last = checks
        if ready and self.preview is None:
            self.preview = {key: value.copy() for key, value in arrays.items() if key.startswith("obs/image/")}
        return ready

    def save(self, directory):
        from PIL import Image

        directory = Path(directory)
        directory.mkdir(parents=True, exist_ok=True)
        passed = self.ready_samples > 0 and self.last is not None and all(c["data_ready"] for c in self.last.values())
        report = {
            "status": "data_checks_passed_geometry_unverified" if passed else "not_ready",
            "samples": self.samples, "both_ready_samples": self.ready_samples,
            "last_checks": self.last, "observed_image_changes": self.changes,
            "existing_configuration": {name: {**CAMERA_COMMON, **mount} for name, mount in CAMERAS.items()},
            "scope": "Shapes checked by prescribed_observation; optics/K, RGB and valid optical depth checked here. No mount movement or new cameras. Visual field of view, occlusion, extrinsics and depth accuracy require runtime review; unchanged images alone do not prove a stale sensor.",
            "preview_depth_scale_m": [0.05, 5.0], "preview_sample": "first_both_ready",
        }
        if self.preview is not None:
            np.savez_compressed(directory / "camera_observations.npz", **self.preview)
            for name in CAMERAS:
                Image.fromarray(self.preview[f"obs/image/{name}_rgb"][0].astype(np.uint8)).save(directory / f"{name}_rgb.png")
                depth = self.preview[f"obs/image/{name}_depth"][0, ..., 0]
                valid = np.isfinite(depth) & (depth >= 0.05) & (depth <= 50.0)
                preview = np.zeros(depth.shape, dtype=np.uint8)
                preview[valid] = (255 * (1 - np.clip((depth[valid] - 0.05) / 4.95, 0, 1))).astype(np.uint8)
                Image.fromarray(preview).save(directory / f"{name}_depth_preview.png")
        write_json(directory / "camera_check.json", report)
        return report
