"""CPU bookkeeping regressions; fake dynamics do not establish unloading skill."""

import importlib
import json
from itertools import count
from types import SimpleNamespace

import pytest
import torch

pytest.importorskip("mujoco")
pytest.importorskip("warp")
pytest.importorskip("mujoco_warp")

evaluation = importlib.import_module("training.basket_unload.evaluate")


@pytest.fixture
def evaluate_cpu(monkeypatch, tmp_path):
    instances, policies, previews = [], [], []
    settings = {"delays": (1, 3), "never_done": False, "dynamics_diagnostics": True}
    config = evaluation.EnvConfig(num_envs=2, episode_seconds=.1, num_objects=3)
    original_device = torch.device

    class FakePolicy:
        def __init__(self, **kwargs):
            self.hidden_initializations = 0
            self.observations = []
            policies.append(self)

        def to(self, device):
            assert device.type == "cpu"
            return self

        def load_state_dict(self, state):
            pass

        def eval(self):
            pass

        def initial_hidden(self, count, device):
            self.hidden_initializations += 1
            return torch.zeros(count, 1, device=device)

        def act(self, obs, hidden, starts, deterministic):
            assert deterministic
            self.observations.append(dict(obs))
            if obs["batch_first"]:
                assert starts.all() and not hidden.any()
            return torch.zeros(2, 19), None, None, hidden + 1

    class FakePreview:
        def __init__(self, *args, **kwargs):
            self.closed = False
            self.images = []
            self.metrics = []
            self.world_ids = []
            previews.append(self)

        def update(self, data, image, *, world_id, asset_gpu, metrics):
            self.images.append(image)
            self.metrics.append(metrics)
            self.world_ids.append(world_id)

        def close(self):
            self.closed = True

    class FakeEnv:
        def __init__(self, config):
            self.config = config
            self.model = self.meta = None
            self.last_image = torch.full((2, 5, 2, 2), .5)
            self.proprio = torch.ones(2, 67)
            self.sim = SimpleNamespace(data=object(), gpu_model=object())
            self.bank = {"qpos": torch.zeros(3, 1), "seed": torch.tensor([100, 101, 102])}
            self.reset_cursor = 0
            self.rows = torch.zeros(2, dtype=torch.long)
            self.age = torch.zeros(2, dtype=torch.long)
            self.explicit_cursors = []
            self.step_calls = 0
            self.partial_snapshots = []
            instances.append(self)

        def reset(self):
            self.explicit_cursors.append(self.reset_cursor)
            self.rows = (torch.arange(2) + self.reset_cursor) % len(self.bank["qpos"])
            self.reset_cursor = (self.reset_cursor + 2) % len(self.bank["qpos"])
            self.age.zero_()
            return {"batch_first": True, "image": self.last_image, "proprio": self.proprio}

        def step(self, action):
            partial_path = tmp_path / "partial_episodes.json"
            if partial_path.is_file():
                self.partial_snapshots.append(json.loads(partial_path.read_text(encoding="utf-8")))
            self.step_calls += 1
            self.age += 1
            done = (torch.zeros(2, dtype=torch.bool) if settings["never_done"] else
                    self.age >= torch.tensor(settings["delays"]))
            ids = torch.nonzero(done).flatten()
            rows = self.rows[ids].clone()
            success = ids == 1
            events = {
                "correct_object_exit": torch.tensor([0, int(done[1]) * 3]),
                "failure_wrong_exit": torch.tensor([bool(done[0]), False]),
                "failure_timeout": torch.zeros(2, dtype=torch.bool),
            }
            episode = {"success": success, "correct_exit_count": success.long() * 3,
                       "wrong_exit_count": (ids == 0).long()}
            for offset, name in enumerate(("return_", "length", "max_tilt_rad", "min_clearance_m",
                                           "max_speed_ratio", "max_foot_slip_m_s")):
                episode[name] = rows.float() + offset / 10
            if settings["dynamics_diagnostics"]:
                episode.update(
                    peak_speed_joint_index=torch.tensor([18, 14])[ids],
                    peak_speed_joint_ratio=torch.tensor([1.2, .45])[ids],
                    invalid_joint_limit=ids == 0,
                    invalid_nonfinite=torch.zeros_like(success),
                    invalid_overflow=torch.zeros_like(success))
            info = {
                "events": events, "episode_combination": rows[:, None].expand(-1, 3).clone(),
                "episode_seed": self.bank["seed"][rows].clone(), "episode_bank_row": rows,
                "episode": episode,
            }
            # Consume cursor rows for every auto-reset, including inactive
            # worlds, exactly as the real sequential-reset environment does.
            self.rows[ids] = (torch.arange(len(ids)) + self.reset_cursor) % len(self.bank["qpos"])
            self.reset_cursor = (self.reset_cursor + len(ids)) % len(self.bank["qpos"])
            self.age[ids] = 0
            return {"batch_first": False, "image": self.last_image, "proprio": self.proprio}, torch.zeros(2), done, info

    monkeypatch.setattr(evaluation, "_evaluation_config", lambda *args: (config, {}))
    monkeypatch.setattr(evaluation, "RecurrentActorCritic", FakePolicy)
    monkeypatch.setattr(evaluation, "BasketUnloadEnv", FakeEnv)
    monkeypatch.setattr(evaluation, "DesktopPreview", FakePreview)
    monkeypatch.setattr(evaluation, "update", lambda *args, **kwargs: None)
    ticks = count()
    monkeypatch.setattr(evaluation.time, "monotonic", lambda: float(next(ticks)))
    monkeypatch.setattr(evaluation.torch, "load", lambda *args, **kwargs: {"network_kwargs": {}, "model": {}})
    monkeypatch.setattr(evaluation.torch, "device", lambda name:
                        original_device("cpu" if name == "cuda:0" else name))

    def run(episodes, *, delays=(1, 3), never_done=False, image_mode="normal", dynamics_diagnostics=True):
        settings.update(delays=delays, never_done=never_done, dynamics_diagnostics=dynamics_diagnostics)
        return evaluation.evaluate("synthetic_checkpoint.pt", episodes=episodes,
                                   image_mode=image_mode,
                                   output=tmp_path / "evaluation.json")

    return run, instances, policies, previews


def test_fast_repeated_failures_cannot_replace_slow_assigned_success(evaluate_cpu):
    run, instances, _, previews = evaluate_cpu
    report = run(2)
    records = report["episode_records"]
    assert [record["success"] for record in records] == [False, True]
    assert [record["episode_id"] for record in records] == [0, 1]
    assert [record["terminal_control_step"] for record in records] == [1, 3]
    assert [record["bank_row"] for record in records] == [0, 1]
    assert [record["seed"] for record in records] == [100, 101]
    assert [record["failure_flags"]["failure_wrong_exit"] for record in records] == [True, False]
    assert report["events"] == {"correct_object_exit": 3, "failure_wrong_exit": 1, "failure_timeout": 0}
    assert report["excluded_terminal_episodes"] == 2
    assert report["episode_metrics"]["mean_return_"] == .5
    assert instances[0].step_calls == 3
    assert previews[0].closed


@pytest.mark.parametrize("delays", [(1, 3), (1, 1)])
def test_fixed_batches_cover_bank_rows_wrap_and_partial_last_batch(evaluate_cpu, delays):
    run, instances, policies, _ = evaluate_cpu
    report = run(5, delays=delays)
    records = report["episode_records"]
    assert len(records) == report["episodes"] == 5
    assert [record["episode_id"] for record in records] == list(range(5))
    assert [record["bank_row"] for record in records] == [0, 1, 2, 0, 1]
    assert [record["seed"] for record in records] == [100, 101, 102, 100, 101]
    assert [record["world_id"] for record in records] == [0, 1, 0, 1, 0]
    assert instances[0].explicit_cursors == [0, 2, 4]
    assert policies[0].hidden_initializations == 3
    assert sum(record["success"] for record in records) == 2
    assert report["events"] == {"correct_object_exit": 6, "failure_wrong_exit": 3, "failure_timeout": 0}
    if delays == (1, 1):
        # Both worlds terminate together in the last batch, whose only
        # assigned episode belongs to world 0. World 1 must be excluded.
        assert report["excluded_terminal_episodes"] == 1


def test_assigned_episode_still_active_at_max_steps_raises(evaluate_cpu):
    run, instances, _, previews = evaluate_cpu
    with pytest.raises(RuntimeError, match="did not terminate within max_steps: "):
        run(2, never_done=True)
    assert instances[0].step_calls == 6
    assert previews[0].closed


@pytest.mark.parametrize("image_mode", ["normal", "zero"])
def test_image_ablation_changes_only_policy_image(evaluate_cpu, image_mode):
    run, instances, policies, previews = evaluate_cpu
    report = run(2, image_mode=image_mode)
    env, policy, preview = instances[0], policies[0], previews[0]
    assert report["image_mode"] == image_mode
    assert torch.equal(env.last_image, torch.full_like(env.last_image, .5))
    assert len(policy.observations) == env.step_calls == 3
    for obs in policy.observations:
        assert obs["proprio"] is env.proprio
        if image_mode == "normal":
            assert obs["image"] is env.last_image
        else:
            assert obs["image"] is not env.last_image
            assert obs["image"].shape == env.last_image.shape
            assert obs["image"].dtype == env.last_image.dtype
            assert obs["image"].device == env.last_image.device
            assert not obs["image"].any()
    assert preview.images
    assert all(image is env.last_image for image in preview.images)
    assert all(metrics["image_mode"] == image_mode for metrics in preview.metrics)
    assert preview.closed
    # Synthetic terminal bookkeeping remains independent of the policy image.
    assert [record["success"] for record in report["episode_records"]] == [False, True]


def test_invalid_image_mode_rejected_before_environment_creation(evaluate_cpu):
    run, instances, policies, previews = evaluate_cpu
    with pytest.raises(ValueError, match="image_mode must be normal or zero"):
        run(2, image_mode="invalid")
    assert not instances and not policies and not previews


def test_preview_follows_remaining_slow_world_and_skips_finished_partial_batch(evaluate_cpu):
    run, _, _, previews = evaluate_cpu
    run(3)
    preview = previews[0]
    # World 0 fails on the first step. Both previews must follow the still
    # assigned slow episode in world 1, including after world 0 fails again.
    # The final batch assigns only world 0, which terminates immediately;
    # there must be no frame of its automatic reset or unassigned world 1.
    assert preview.world_ids == [1, 1]
    assert [metrics["world_id"] for metrics in preview.metrics] == [1, 1]
    assert [metrics["episode_id"] for metrics in preview.metrics] == [1, 1]
    assert [metrics["evaluation_episodes"] for metrics in preview.metrics] == [1, 1]


def test_partial_episode_diagnostics_update_only_for_accepted_terminals(evaluate_cpu, tmp_path):
    run, instances, _, _ = evaluate_cpu
    report = run(2)
    snapshots = instances[0].partial_snapshots
    assert len(snapshots) == 2
    for snapshot in snapshots:
        assert snapshot["status"] == "in_progress"
        assert snapshot["completed_episodes"] == 1
        assert snapshot["target_episodes"] == 2
        assert snapshot["episode_records"] == report["episode_records"][:1]
        assert "not a final evaluation rate" in snapshot["scope"]
        assert "success_rate" not in snapshot
    partial = json.loads((tmp_path / "partial_episodes.json").read_text(encoding="utf-8"))
    assert partial["status"] == "in_progress"
    assert partial["completed_episodes"] == 2
    assert partial["episode_records"] == report["episode_records"]
    assert not (tmp_path / "partial_episodes.tmp.json").exists()


@pytest.mark.parametrize("delays", [(1, 3), (3, 1), (1, 1)])
def test_dynamics_diagnostics_follow_terminal_world_order_without_changing_success(evaluate_cpu, delays):
    run, _, _, _ = evaluate_cpu
    report = run(2, delays=delays)
    for record in report["episode_records"]:
        world = record["world_id"]
        assert record["peak_speed_joint_index"] == (18, 14)[world]
        assert isinstance(record["peak_speed_joint_index"], int)
        assert record["peak_speed_joint_ratio"] == pytest.approx((1.2, .45)[world])
        assert record["invalid_joint_limit"] is (world == 0)
        assert record["invalid_nonfinite"] is False
        assert record["invalid_overflow"] is False
        assert record["success"] is (world == 1)
    legacy = run(2, delays=delays, dynamics_diagnostics=False)
    assert legacy["events"] == report["events"]
    assert legacy["by_combination"] == report["by_combination"]
    assert legacy["episode_metrics"] == report["episode_metrics"]
    assert [record["success"] for record in legacy["episode_records"]] == [
        record["success"] for record in report["episode_records"]]
    assert all("peak_speed_joint_index" not in record for record in legacy["episode_records"])


def test_real_dynamic_state_retains_substep_causes_and_peaks_without_modifying_state(monkeypatch):
    environment = importlib.import_module("training.basket_unload.env")
    env = environment.BasketUnloadEnv.__new__(environment.BasketUnloadEnv)
    env.meta = {"base_id": 0}
    env.sim = SimpleNamespace(
        xmat=torch.eye(3).reshape(1, 1, 3, 3).repeat(3, 1, 1, 1),
        qpos=torch.zeros(3, 26), qvel=torch.zeros(3, 25),
        joint_adr=torch.arange(7, 26), dof_adr=torch.arange(6, 25),
        velocity=torch.tensor([23.] * 12 + [3.] * 6 + [.25]),
        lower=-torch.ones(19), upper=torch.ones(19), nominal=torch.zeros(19),
        data=SimpleNamespace(overflow=torch.tensor([0, 0, 1])))
    env.sim.qpos[:, 2] = .6
    env.sim.qpos[0, 7] = 1.021
    env.sim.qvel[0, 24] = .3  # Gripper peak is .3/.25, not the fastest raw joint.
    env.sim.qvel[1, 19] = 4.5
    env.sim.qvel[1, 0] = float("nan")
    env.sim.qvel[2, 10] = 11.5
    env.episode_peak_joint_speed_ratio = torch.zeros(3, 19)
    env.episode_invalid_joint_limit = torch.zeros(3, dtype=torch.bool)
    env.episode_invalid_nonfinite = torch.zeros(3, dtype=torch.bool)
    env.episode_invalid_overflow = torch.zeros(3, dtype=torch.bool)
    monkeypatch.setattr(environment.wp, "to_torch", lambda array: array)
    before_qpos, before_qvel = env.sim.qpos.clone(), env.sim.qvel.clone()
    _, fallen, invalid = env._dynamic_state()
    assert invalid.tolist() == [True, True, True]
    assert not fallen.any()
    assert env.episode_invalid_joint_limit.tolist() == [True, False, False]
    assert env.episode_invalid_nonfinite.tolist() == [False, True, False]
    assert env.episode_invalid_overflow.tolist() == [False, False, True]
    peaks, joints = env.episode_peak_joint_speed_ratio.max(-1)
    torch.testing.assert_close(peaks, torch.tensor([1.2, 1.5, .5]))
    assert joints.tolist() == [18, 13, 4]
    torch.testing.assert_close(env.sim.qpos, before_qpos)
    torch.testing.assert_close(env.sim.qvel, before_qvel, equal_nan=True)

    # Later substeps recover. The original dynamics outputs recover too,
    # while evaluator-only episode diagnostics retain the earlier causes.
    env.sim.qpos[:, 7:] = 0
    env.sim.qvel.zero_()
    env.sim.data.overflow.zero_()
    stable, fallen, invalid = env._dynamic_state()
    assert stable.all() and not fallen.any() and not invalid.any()
    torch.testing.assert_close(env.episode_peak_joint_speed_ratio.max(-1).values, peaks)
    assert env.episode_invalid_joint_limit.tolist() == [True, False, False]
    assert env.episode_invalid_nonfinite.tolist() == [False, True, False]
    assert env.episode_invalid_overflow.tolist() == [False, False, True]
