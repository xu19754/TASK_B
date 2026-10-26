"""Synthetic IO tests; no physics or demonstration success is implied."""

import hashlib
import json
import runpy
import sys
from types import ModuleType, SimpleNamespace

import numpy as np
import pytest

from taskb_student.constants import ARM_Q0, JOINT_UPPER, LEG_Q0, ROOT
from taskb_student.recording import CameraAudit, EpisodeRecorder, collection_metadata, prescribed_observation
from taskb_student.sensor_contract import CAMERA_COMMON, OBSERVATION_SHAPES
from scripts.inspect_recording import inspect_episode


def observation():
    obs = {group: {key: np.zeros(shape, dtype=np.float32) for key, shape in terms.items()}
           for group, terms in OBSERVATION_SHAPES.items()}
    focal = 640 * CAMERA_COMMON["focal_length"] / CAMERA_COMMON["horizontal_aperture"]
    for name in ("head", "ee"):
        obs["image"][f"{name}_rgb"] = np.full((1, 480, 640, 3), 128, dtype=np.uint8)
        obs["image"][f"{name}_depth"].fill(2)
        obs["image"][f"{name}_intrinsics"][:] = [[focal, 0, 320], [0, focal, 240], [0, 0, 1]]
    obs["lidar"]["hit_mask"] = np.zeros((1, 360, 16), dtype=bool)
    obs["lidar"]["ready"] = np.ones((1, 1), dtype=bool)
    return obs


def test_copy_before_step_whitelist_alignment_and_partial_chunk(tmp_path):
    obs = observation()
    obs["object_positions"] = np.ones((18, 3))
    obs["proprio"]["root_pos_w"] = np.ones((1, 3))
    folder = tmp_path / "episode"
    recorder = EpisodeRecorder(folder, collection_metadata(42, 1200, "smoke"), chunk_steps=2)
    for i in range(3):
        obs["proprio"]["joint_pos"].fill(i)
        copied = prescribed_observation(obs)
        recorder.begin(copied, i, i * 0.03)
        obs["proprio"]["joint_pos"].fill(99)  # step mutates live buffers
        recorder.finish_step(np.array(LEG_Q0 + ARM_Q0), 0.02, i + 1, i == 2, False)
    recorder.close("environment_termination_or_timeout", 3)
    recorder.close("ignored", 999)
    report = inspect_episode(folder)
    assert report["steps"] == 3 and report["both_cameras_ready_samples"] == 3
    with np.load(folder / "chunk_000000.npz") as chunk:
        assert np.all(chunk["obs/proprio/joint_pos"][0] == 0)
        assert np.all(chunk["obs/proprio/joint_pos"][1] == 1)
        assert not any("object" in key or "root_pos" in key for key in chunk.files)
    assert len(json.loads((folder / "manifest.json").read_text())["chunks"]) == 2
    with pytest.raises(FileExistsError):
        EpisodeRecorder(folder, {})


def test_interrupted_pending_step_is_not_paired_with_wrong_action(tmp_path):
    recorder = EpisodeRecorder(tmp_path / "interrupted", collection_metadata(1, 1200, "smoke"))
    recorder.begin(prescribed_observation(observation()), 0, 0)
    recorder.close("exception", 0)
    manifest = json.loads((recorder.directory / "manifest.json").read_text())
    assert manifest["discarded_pending_step"] is True
    assert inspect_episode(recorder.directory)["steps"] == 0


def test_invalid_depth_is_preserved_and_camera_report_is_limited(tmp_path):
    obs = observation()
    obs["image"]["head_depth"][0, 0, 0, 0] = np.inf
    copied = prescribed_observation(obs)
    audit = CameraAudit()
    assert audit.observe(copied)
    report = audit.save(tmp_path)
    assert report["status"] == "data_checks_passed_geometry_unverified"
    assert (tmp_path / "head_rgb.png").is_file()
    assert np.isinf(copied["obs/image/head_depth"][0, 0, 0, 0])
    obs["image"]["ee_depth"].fill(0)
    assert not audit.observe(prescribed_observation(obs))
    assert audit.save(tmp_path)["status"] == "not_ready"


def test_missing_stream_bad_intrinsics_and_action_are_rejected(tmp_path):
    obs = observation()
    del obs["image"]["ee_rgb"]
    with pytest.raises(KeyError):
        prescribed_observation(obs)
    obs = observation()
    obs["image"]["ee_intrinsics"][0, 2, 2] = 2
    assert not CameraAudit().observe(prescribed_observation(obs))
    recorder = EpisodeRecorder(tmp_path / "invalid", collection_metadata(1, 1200, "smoke"))
    recorder.begin(prescribed_observation(observation()), 0, 0)
    with pytest.raises(ValueError):
        recorder.finish_step(np.full(19, np.nan), 0, 0, False, False)
    recorder.close("exception", 0)


def test_readback_rejects_corruption_and_rows_after_terminal(tmp_path):
    folder = tmp_path / "corrupt"
    recorder = EpisodeRecorder(folder, collection_metadata(42, 1200, "smoke"))
    for i in range(2):
        recorder.begin(prescribed_observation(observation()), 0, 0)
        recorder.finish_step(np.array(LEG_Q0 + ARM_Q0), 0, 0, i == 1, False)
    with pytest.raises(RuntimeError):
        recorder.begin(prescribed_observation(observation()), 0, 0)
    recorder.close("step_limit", 0)
    path = folder / "chunk_000000.npz"
    with np.load(path) as source:
        arrays = {key: source[key] for key in source.files}
    arrays["terminated"][0] = True
    np.savez_compressed(path, **arrays)
    manifest = json.loads((folder / "manifest.json").read_text())
    manifest["chunks"][0]["sha256"] = hashlib.sha256(path.read_bytes()).hexdigest()
    (folder / "manifest.json").write_text(json.dumps(manifest))
    with pytest.raises(ValueError, match="termination/reset"):
        inspect_episode(folder)
    path.write_bytes(path.read_bytes() + b"corrupt")
    with pytest.raises(ValueError, match="Checksum"):
        inspect_episode(folder)


@pytest.mark.parametrize("fail_on_second_step", [False, True])
def test_formal_launcher_wiring_and_exception_flush_without_scene_access(tmp_path, monkeypatch, fail_on_second_step):
    """Fake simulator tests wiring only; not a substitute for an Isaac run."""
    import torch

    live_obs = {group: {key: torch.from_numpy(value) for key, value in terms.items()}
                for group, terms in observation().items()}
    class FakeEnv:
        def __init__(self):
            self.unwrapped = self
            self.device, self.step_dt = "cpu", 0.02
            self.steps, self.closed = 0, False

        @property
        def scene(self):
            raise AssertionError("Recording must not read privileged scene state")

        def reset(self, seed):
            assert seed == 42
            return live_obs, {}

        def step(self, action):
            self.steps += 1
            if fail_on_second_step and self.steps == 2:
                raise RuntimeError("synthetic step failure")
            torch.testing.assert_close(action, torch.tensor([JOINT_UPPER]))
            live_obs["proprio"]["joint_pos"].fill_(self.steps * 10)
            return live_obs, torch.tensor([0.0]), torch.tensor([self.steps == 3]), torch.tensor([False]), {}

        def close(self):
            self.closed = True

    env = FakeEnv()
    class FakeCfg:
        def __init__(self, seed):
            assert seed == 42
            self.sim = SimpleNamespace(device="cpu")

        def configure_cameras(self, enabled):
            assert enabled

    class FakeSolution:
        def __init__(self, command):
            pass

        def reset(self):
            pass

        def predicts(self, obs, current_score):
            obs["proprio"]["joint_pos"].fill_(77)  # controller can also mutate its inputs
            return {"action": np.array(JOINT_UPPER) + 1, "giveup": False}

    class FakeAppLauncher:
        @staticmethod
        def add_app_launcher_args(parser):
            parser.add_argument("--enable_cameras", action="store_true")
            parser.add_argument("--device")
            parser.add_argument("--kit_args", default="")

        def __init__(self, args):
            self.app = SimpleNamespace(is_running=lambda: True, close=lambda **kwargs: None)

    for name, attributes in {
        "isaaclab": {}, "isaaclab.app": {"AppLauncher": FakeAppLauncher},
        "taskb_student.registration": {}, "taskb_student.env_cfg": {"TaskBEnvCfg": FakeCfg},
        "gymnasium": {"make": lambda *args, **kwargs: env},
        "fake_collect_controller": {"AlgSolution": FakeSolution},
    }.items():
        module = ModuleType(name)
        module.__dict__.update(attributes)
        monkeypatch.setitem(sys.modules, name, module)
    folder = tmp_path / "wired"
    monkeypatch.setattr(sys, "argv", ["play_taskb.py", "--enable_cameras", "--record-dir", str(folder),
                                      "--solution", "fake_collect_controller:AlgSolution"])
    if fail_on_second_step:
        with pytest.raises(RuntimeError, match="synthetic step failure"):
            runpy.run_path(str(ROOT / "scripts/play_taskb.py"), run_name="__main__")
    else:
        runpy.run_path(str(ROOT / "scripts/play_taskb.py"), run_name="__main__")
    assert env.closed
    report = inspect_episode(folder)
    assert report["steps"] == (1 if fail_on_second_step else 3)
    assert report["stop_reason"] == ("exception" if fail_on_second_step else "environment_termination_or_timeout")
    with np.load(folder / "chunk_000000.npz") as arrays:
        assert np.all(arrays["obs/proprio/joint_pos"][0] == 0)
        np.testing.assert_allclose(arrays["action"][0, 0], JOINT_UPPER)
        if not fail_on_second_step:
            assert np.all(arrays["obs/proprio/joint_pos"][1] == 10)
    assert (folder / "camera_check/camera_check.json").is_file()
