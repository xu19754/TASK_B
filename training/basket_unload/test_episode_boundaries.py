"""Real environment terminal bookkeeping with CPU stand-ins for physics."""

from types import SimpleNamespace

import pytest
import torch

pytest.importorskip("mujoco")
pytest.importorskip("warp")
pytest.importorskip("mujoco_warp")

from training.basket_unload import env as environment
from training.basket_unload.metrics import ExitResult
from training.basket_unload.reward import RewardWeights


@pytest.mark.parametrize("stage", ["unload", "balance"])
def test_success_at_deadline_is_terminal_and_only_pure_timeout_bootstraps(monkeypatch, stage):
    env = environment.BasketUnloadEnv.__new__(environment.BasketUnloadEnv)
    env.config = environment.EnvConfig(num_envs=3, num_objects=1, stage=stage,
                                       episode_seconds=.02)
    env.device, env.num_envs = torch.device("cpu"), 3
    env.meta, env.foot_ids = {"base_id": 0}, torch.tensor([1])
    env.steps, env.control_steps = torch.zeros(3, dtype=torch.long), 0
    env.camera_age, env.episode_return = torch.zeros(3), torch.zeros(3)
    for name in ("prev_action", "previous_targets", "previous_previous_targets",
                 "episode_peak_joint_speed_ratio"):
        setattr(env, name, torch.zeros(3, 19))
    for name in ("previous_tail_potential", "previous_aperture_potential"):
        setattr(env, name, torch.zeros(3, 1))
    for name in ("episode_max_tilt", "episode_max_speed_ratio", "episode_max_foot_slip"):
        setattr(env, name, torch.zeros(3))
    for name in ("episode_invalid_joint_limit", "episode_invalid_nonfinite", "episode_invalid_overflow"):
        setattr(env, name, torch.zeros(3, dtype=torch.bool))
    env.last_clearance = env.episode_min_clearance = torch.full((3,), .02)
    env.last_intersection = torch.zeros(3, dtype=torch.bool)
    env.last_image = torch.zeros(3, 5, 32, 32)
    env.previous_feet = torch.zeros(3, 1, 3)
    env.class_ids = torch.zeros(3, 3, dtype=torch.long)
    env.episode_bank_rows = torch.arange(3)
    env.bank = {"seed": torch.tensor([10, 11, 12])}
    env.reward_weights = RewardWeights()
    objects = torch.zeros(3, 1, dtype=torch.bool)
    complete = torch.tensor([[True], [False], [True]])
    flags = torch.zeros(3, dtype=torch.bool)
    env.result = ExitResult(
        state=objects.long(), first_exit=objects, wrong_exit=objects, reentry=objects,
        ever_exited=complete, exited=complete, wrong=objects, reentered=objects,
        invalid_initial=objects, invalid_trajectory=objects, invalid_dynamics=flags,
        all_clear=complete[:, 0], hold_time=torch.zeros(3), success=complete[:, 0])
    sim = SimpleNamespace(
        qpos=torch.zeros(3, 26), qvel=torch.zeros(3, 25),
        joint_adr=torch.arange(7, 26), dof_adr=torch.arange(6, 25),
        velocity=torch.ones(19), effort=torch.ones(19), nominal=torch.zeros(19),
        lower=-torch.ones(19), upper=torch.ones(19), targets=torch.zeros(3, 19),
        xpos=torch.zeros(3, 2, 3), xmat=torch.eye(3)[None, None].repeat(3, 2, 1, 1))
    sim.qpos[:, 2] = .6
    sim.data = SimpleNamespace(qpos=sim.qpos, actuator_force=torch.zeros(3, 19))
    sim.set_targets = lambda requested, dt: sim.targets.copy_(requested)
    sim.physics_step = lambda: None
    env.sim = sim
    env._dynamic_state = lambda: (torch.tensor([True, False, True]), flags, flags)
    dangerous = torch.tensor([False, False, True])
    env._ground_contact = lambda measure_slip=False: (
        (dangerous, torch.zeros(3)) if measure_slip else dangerous)
    env.clearance = SimpleNamespace(evaluate=lambda qpos: (flags, torch.full((3,), .02)))
    env.tracker = SimpleNamespace(update=lambda *args, **kwargs: env.result)
    env._vertices = lambda: torch.zeros(3, 1, 1, 3)
    env._unload_potentials = lambda vertices: (torch.zeros(3, 1), torch.zeros(3, 1))
    env.observation = lambda: {"image": env.last_image, "proprio": torch.zeros(3, 67)}
    env._reset = lambda mask: None
    env.camera = SimpleNamespace(render=lambda: env.last_image)
    monkeypatch.setattr(environment.wp, "to_torch", lambda value: value)

    _, reward, done, info = env.step(torch.zeros(3, 19))
    assert done.tolist() == [True, True, True]
    assert info["episode"]["success"].tolist() == [1., 0., 0.]
    assert info["timeouts"].tolist() == [False, True, False]
    assert info["events"]["failure_dangerous_collision"].tolist() == [False, False, True]
    assert "terminal_observation" in info
    assert torch.isfinite(reward).all()
