"""Reward accounting checks; synthetic inputs do not prove dynamics success."""

from dataclasses import replace

import pytest
import torch

from training.basket_unload.metrics import ExitResult
from training.basket_unload.reward import RewardWeights, compute_reward


def result_for(batch=1, *, device="cpu", complete=False):
    objects = torch.zeros(batch, 3, device=device, dtype=torch.bool)
    env = torch.zeros(batch, device=device, dtype=torch.bool)
    complete_objects = objects | complete
    return ExitResult(
        objects.long(), objects, objects, objects, complete_objects,
        complete_objects, objects, objects, objects, objects, env,
        env | complete, env.float(), env | complete)


def inputs(batch=1, *, device="cpu", dt=.02):
    joints = torch.zeros(batch, 19, device=device)
    flags = torch.zeros(batch, device=device, dtype=torch.bool)
    return dict(
        action=joints, previous_action=joints, previous_previous_action=joints,
        joint_pos=joints, joint_vel=joints, lower=-torch.ones(19, device=device),
        upper=torch.ones(19, device=device), effort_ratio=joints,
        foot_slip=torch.zeros(batch, device=device), dangerous_collision=flags,
        fallen=flags, self_intersection=flags, timeout=flags, stable=flags, dt=dt)


def test_success_and_three_exit_events_dominate_small_rate_costs():
    result = result_for(complete=True)
    result = replace(result, first_exit=torch.ones(1, 3, dtype=torch.bool))
    arguments = inputs()
    arguments["stable"] = torch.tensor([True])
    reward, terms = compute_reward(result, **arguments)
    assert reward.item() == pytest.approx(130 - .05 * .02)
    assert terms["is_success"].item()
    assert not terms["is_failure"].item()
    assert terms["success"].item() == 100
    assert terms["first_exit"].item() == 30


def test_one_exit_event_reward_does_not_repeat_without_new_event():
    result = result_for()
    result = replace(result, first_exit=torch.tensor([[True, False, False]]))
    first_reward, first_terms = compute_reward(result, **inputs())
    later_reward, later_terms = compute_reward(result_for(), **inputs())
    assert (first_reward - later_reward).item() == pytest.approx(10.)
    assert first_terms["first_exit"].item() == 10
    assert later_terms["first_exit"].item() == 0


def test_explicit_success_event_suppresses_repeated_completion_reward():
    arguments = inputs()
    arguments["stable"] = torch.tensor([True])
    reward, terms = compute_reward(result_for(complete=True), **arguments,
                                    success_event=torch.tensor([False]))
    assert terms["is_success"].item()
    assert terms["success"].item() == 0
    assert reward.item() < 0


def test_deliberate_tilt_or_stepping_is_allowed_before_recovery():
    reward, terms = compute_reward(result_for(complete=True), **inputs())
    assert not terms["is_success"].item()
    assert not terms["is_failure"].item()
    assert reward.item() == pytest.approx(-.05 * .02)


@pytest.mark.parametrize("reason,flag", [
    ("wrong_exit", "wrong"), ("reentry", "reentered"),
    ("invalid_initial", "invalid_initial"),
    ("invalid_trajectory", "invalid_trajectory"),
])
def test_metric_failure_blocks_simultaneous_success_and_exit_rewards(reason, flag):
    result = result_for(complete=True)
    result = replace(result, **{flag: torch.tensor([[True, False, False]])},
                     first_exit=torch.ones(1, 3, dtype=torch.bool))
    arguments = inputs()
    arguments["stable"] = torch.tensor([True])
    reward, terms = compute_reward(result, **arguments)
    assert terms["is_failure"].item()
    assert terms[f"failure_{reason}"].item()
    assert terms["success"].item() == 0
    assert terms["first_exit"].item() == 0
    assert reward.item() <= -25


@pytest.mark.parametrize("flag,minimum_penalty", [
    ("dangerous_collision", 25), ("fallen", 50), ("self_intersection", 50)])
def test_real_dynamics_failure_has_strong_terminal_penalty(flag, minimum_penalty):
    arguments = inputs()
    arguments[flag] = torch.tensor([True])
    reward, terms = compute_reward(result_for(), **arguments)
    assert terms["is_failure"].item()
    assert terms[f"failure_{flag}"].item()
    assert reward.item() <= -minimum_penalty


def test_timeout_penalty_scales_with_remaining_objects():
    arguments = inputs()
    arguments["timeout"] = torch.tensor([True])
    three = result_for()
    one = replace(three, exited=torch.tensor([[True, True, False]]))
    reward_three, terms_three = compute_reward(three, **arguments)
    reward_one, terms_one = compute_reward(one, **arguments)
    assert terms_three["failure_timeout"].item()
    assert terms_three["timeout_remaining"].item() == -15
    assert terms_one["timeout_remaining"].item() == -5
    assert (reward_one - reward_three).item() == pytest.approx(10)


def test_recovered_success_at_timeout_is_success():
    arguments = inputs()
    arguments["timeout"] = torch.tensor([True])
    arguments["stable"] = torch.tensor([True])
    reward, terms = compute_reward(result_for(complete=True), **arguments)
    assert terms["is_success"].item()
    assert not terms["is_failure"].item()
    assert terms["timeout"].item() == 0
    assert reward.item() > 99


def test_action_velocity_cost_tracks_physical_rate_across_control_frequencies():
    rates = []
    for dt in (.01, .02):
        arguments = inputs(dt=dt)
        arguments["action"] = torch.full((1, 19), 2 * dt)
        arguments["previous_previous_action"] = torch.full((1, 19), -2 * dt)
        _, terms = compute_reward(result_for(), **arguments)
        rates.append(terms["action_velocity"].item() / dt)
        assert terms["action_acceleration"].item() == pytest.approx(0, abs=1e-9)
    assert rates[0] == pytest.approx(rates[1])


def test_action_acceleration_cost_tracks_physical_rate_across_frequencies():
    rates = []
    acceleration = 4.
    for dt in (.01, .02):
        arguments = inputs(dt=dt)
        arguments["action"] = torch.full((1, 19), .5 * acceleration * dt ** 2)
        arguments["previous_previous_action"] = arguments["action"].clone()
        _, terms = compute_reward(result_for(), **arguments)
        rates.append(terms["action_acceleration"].item() / dt)
    assert rates[0] == pytest.approx(rates[1])


def test_limits_and_measured_slip_effort_use_small_dense_penalties():
    arguments = inputs()
    _, stationary = compute_reward(result_for(), **arguments)
    arguments["joint_pos"] = torch.full((1, 19), .99)
    arguments["joint_vel"] = torch.full((1, 19), .5)
    arguments["effort_ratio"] = torch.full((1, 19), .5)
    arguments["foot_slip"] = torch.tensor([.2])
    reward, moving = compute_reward(result_for(), **arguments)
    for term in ("joint_limit", "joint_velocity", "energy", "foot_slip"):
        assert moving[term].item() < stationary[term].item()
    assert not moving["is_failure"].item()
    assert -.1 < reward.item() < 0


def test_stability_shaping_allows_unloading_lean_but_penalizes_pre_fall_state():
    arguments = inputs()
    _, leaning = compute_reward(result_for(), **arguments, body_tilt=torch.tensor([.27]),
                                body_height=torch.tensor([.5]))
    reward, unstable = compute_reward(result_for(), **arguments, body_tilt=torch.tensor([.6]),
                                      body_height=torch.tensor([.34]),
                                      body_angular_speed=torch.tensor([.8]))
    assert leaning["body_tilt"].item() == 0
    assert leaning["body_height"].item() == 0
    assert not unstable["is_failure"].item()
    assert torch.isfinite(reward).all()
    for name in ("body_tilt", "body_height", "body_angular_velocity"):
        assert unstable[name].item() < leaning[name].item()


@pytest.mark.parametrize("corruption", ["nan_action", "nan_dt", "zero_dt", "invalid_limits"])
def test_invalid_numeric_inputs_produce_finite_terminal_reward(corruption):
    arguments = inputs()
    if corruption == "nan_action":
        arguments["action"] = torch.full((1, 19), float("nan"))
    elif corruption == "nan_dt":
        arguments["dt"] = float("nan")
    elif corruption == "zero_dt":
        arguments["dt"] = 0
    else:
        arguments["upper"] = arguments["lower"]
    reward, terms = compute_reward(result_for(), **arguments)
    assert torch.isfinite(reward).all()
    assert terms["is_failure"].item()
    assert terms["failure_nonfinite_input"].item()
    assert reward.item() <= -50


def test_configurable_weights_change_only_requested_term():
    arguments = inputs()
    result = replace(result_for(), first_exit=torch.tensor([[True, False, False]]))
    default, _ = compute_reward(result, **arguments)
    changed, _ = compute_reward(result, **arguments, weights=RewardWeights(first_exit=15))
    assert (changed - default).item() == pytest.approx(5)
    with pytest.raises(ValueError):
        RewardWeights(action_velocity=-1)


def test_soft_safety_options_default_off_and_safe_region_has_no_cost():
    arguments = inputs()
    baseline, base_terms = compute_reward(result_for(), **arguments)
    disabled, disabled_terms = compute_reward(
        result_for(), **arguments, speed_ratio_peak=torch.tensor([float("inf")]),
        clearance_min=torch.tensor([float("nan")]))
    torch.testing.assert_close(disabled, baseline)
    assert not disabled_terms["is_failure"].any()
    safe, safe_terms = compute_reward(
        result_for(), **arguments, weights=RewardWeights(speed_margin=2, clearance=1),
        speed_ratio_peak=torch.tensor([.6]), clearance_min=torch.tensor([.01]))
    torch.testing.assert_close(safe, baseline)
    for name in ("speed_margin", "clearance"):
        assert base_terms[name].item() == disabled_terms[name].item() == safe_terms[name].item() == 0


def test_soft_safety_cost_increases_before_unchanged_hard_failure_boundary():
    arguments = inputs(3)
    _, terms = compute_reward(
        result_for(3), **arguments, weights=RewardWeights(speed_margin=2, clearance=1),
        speed_ratio_peak=torch.tensor([.6, .8, 1.]),
        clearance_min=torch.tensor([.01, .005, 0.]))
    torch.testing.assert_close(terms["speed_margin"], torch.tensor([0., -.01, -.04]))
    torch.testing.assert_close(terms["clearance"], torch.tensor([0., -.005, -.02]))
    assert not terms["is_failure"].any()
    assert not terms["is_success"].any()


def test_soft_safety_cost_is_integrated_over_control_dt():
    costs = []
    for dt in (.01, .02):
        _, terms = compute_reward(
            result_for(), **inputs(dt=dt), weights=RewardWeights(speed_margin=2, clearance=1),
            speed_ratio_peak=torch.tensor([.8]), clearance_min=torch.tensor([.005]))
        costs.append(torch.stack([terms[name] / dt for name in ("speed_margin", "clearance")]))
    torch.testing.assert_close(costs[0], costs[1])


@pytest.mark.parametrize("field,value", [
    ("speed_ratio_peak", float("nan")), ("speed_ratio_peak", float("inf")),
    ("clearance_min", float("nan")), ("clearance_min", -float("inf")),
    ("speed_ratio_peak", 1e30), ("clearance_min", -1e30),
])
def test_invalid_enabled_soft_input_has_finite_terminal_reward(field, value):
    arguments = {field: torch.tensor([value])}
    reward, terms = compute_reward(
        result_for(complete=True), **inputs(), weights=RewardWeights(speed_margin=2, clearance=1),
        **arguments)
    assert torch.isfinite(reward).all()
    assert terms["failure_nonfinite_input"].item()
    assert terms["is_failure"].item()
    assert not terms["is_success"].item()
    assert reward.item() <= -50
    assert terms["speed_margin"].item() == terms["clearance"].item() == 0


@pytest.mark.parametrize("weight", [-1., float("nan"), float("inf")])
def test_soft_safety_weights_require_finite_nonnegative_values(weight):
    from training.basket_unload.env import EnvConfig

    for name in ("clearance", "speed_margin"):
        with pytest.raises(ValueError):
            RewardWeights(**{name: weight})
        with pytest.raises(ValueError):
            EnvConfig(**{f"{name}_penalty_weight": weight})


def test_environment_soft_cost_uses_control_interval_peaks_not_episode_history(monkeypatch):
    from types import SimpleNamespace
    from training.basket_unload import env as environment

    env = environment.BasketUnloadEnv.__new__(environment.BasketUnloadEnv)
    env.config = environment.EnvConfig(
        num_envs=1, stage="balance", num_objects=3, episode_seconds=100,
        speed_margin_penalty_weight=2, clearance_penalty_weight=1)
    env.device, env.num_envs = torch.device("cpu"), 1
    env.meta = {"base_id": 0}
    env.foot_ids = torch.tensor([1])
    env.steps = torch.zeros(1, dtype=torch.long)
    env.control_steps = 0
    env.camera_age = torch.zeros(1)
    env.prev_action = torch.zeros(1, 19)
    env.episode_return = torch.zeros(1)
    env.previous_targets = torch.zeros(1, 19)
    env.previous_previous_targets = torch.zeros(1, 19)
    env.previous_tail_potential = torch.zeros(1, 1)
    env.previous_aperture_potential = torch.zeros(1, 1)
    env.episode_max_tilt = torch.zeros(1)
    env.episode_max_speed_ratio = torch.zeros(1)
    env.episode_min_clearance = torch.full((1,), .02)
    env.episode_max_foot_slip = torch.zeros(1)
    env.last_clearance = torch.full((1,), .02)
    env.last_intersection = torch.zeros(1, dtype=torch.bool)
    env.last_image = torch.zeros(1, 5, 32, 32)
    env.result = result_for()
    env.reward_weights = RewardWeights(clearance=1, speed_margin=2)
    sim = SimpleNamespace(
        qpos=torch.zeros(1, 26), qvel=torch.zeros(1, 25),
        joint_adr=torch.arange(7, 26), dof_adr=torch.arange(6, 25),
        velocity=torch.tensor([10.] * 18 + [.25]), targets=torch.zeros(1, 19),
        lower=-torch.ones(19), upper=torch.ones(19), effort=torch.ones(19),
        nominal=torch.zeros(19), xmat=torch.eye(3)[None, None].expand(1, 2, 3, 3).clone(),
        xpos=torch.zeros(1, 2, 3))
    sim.qpos[:, 2] = .6
    sim.data = SimpleNamespace(qpos=sim.qpos, actuator_force=torch.zeros(1, 19))
    sim.set_targets = lambda requested, dt: sim.targets.copy_(requested)
    substeps = iter([.1, .9, .1, .1, .1, .1, .1, .1, .1, .1] + [.2] * 10)

    def physics_step():
        sim.qvel.zero_()
        sim.qvel[:, sim.dof_adr[-1]] = next(substeps) * sim.velocity[-1]

    sim.physics_step = physics_step
    env.sim = sim
    env._dynamic_state = lambda: (torch.ones(1, dtype=torch.bool),
                                  torch.zeros(1, dtype=torch.bool), torch.zeros(1, dtype=torch.bool))
    env._ground_contact = lambda measure_slip=False: (
        (torch.zeros(1, dtype=torch.bool), torch.zeros(1)) if measure_slip
        else torch.zeros(1, dtype=torch.bool))
    env._vertices = lambda: torch.zeros(1, 1, 1, 3)
    env._unload_potentials = lambda vertices: (torch.zeros(1, 1), torch.zeros(1, 1))
    env.observation = lambda: {"image": env.last_image, "proprio": torch.zeros(1, 67)}
    distances = iter([.004, .009, .018, .019])
    env.clearance = SimpleNamespace(evaluate=lambda qpos: (
        torch.zeros(1, dtype=torch.bool), torch.tensor([next(distances)])))
    monkeypatch.setattr(environment.wp, "to_torch", lambda tensor: tensor)
    recorded = []

    def reward_probe(result, **kwargs):
        recorded.append({key: kwargs[key].clone() for key in ("speed_ratio_peak", "clearance_min")})
        return compute_reward(result, **kwargs)

    monkeypatch.setattr(environment, "compute_reward", reward_probe)
    for _ in range(2):
        _, reward, done, _ = env.step(torch.zeros(1, 19))
        assert torch.isfinite(reward).all() and not done.any()
    assert [row["speed_ratio_peak"].item() for row in recorded] == pytest.approx([.9, .2])
    # Previous query is retained until refreshed, but the old interval's .004
    # minimum must not stick as an episode-level cost in the second interval.
    assert [row["clearance_min"].item() for row in recorded] == pytest.approx([.004, .009])
    assert env.episode_max_speed_ratio.item() == pytest.approx(.9)
    assert env.episode_min_clearance.item() == pytest.approx(.004)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA unavailable")
def test_reward_batch_cuda_cpu_agreement_and_independent_termination():
    outputs = []
    for device in ("cpu", "cuda"):
        arguments = inputs(2, device=device)
        arguments["fallen"] = torch.tensor([False, True], device=device)
        arguments["stable"] = torch.tensor([True, True], device=device)
        reward, terms = compute_reward(result_for(2, device=device, complete=True), **arguments)
        assert terms["is_success"].tolist() == [True, False]
        assert terms["is_failure"].tolist() == [False, True]
        assert reward.device.type == device
        outputs.append((reward.cpu(), {name: value.cpu() for name, value in terms.items()}))
    torch.testing.assert_close(outputs[0][0], outputs[1][0])
    for name in outputs[0][1]:
        torch.testing.assert_close(outputs[0][1][name], outputs[1][1][name])
