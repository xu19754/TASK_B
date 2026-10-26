"""Reference command timing checks; no unloading dynamics are simulated here."""

from types import SimpleNamespace

import pytest
import torch

pytest.importorskip("mujoco")
pytest.importorskip("warp")
pytest.importorskip("mujoco_warp")

from training.basket_unload.env import BasketUnloadEnv, EnvConfig


@pytest.mark.parametrize("control_dt,stride", [(.02, 1), (.1, 5)])
@pytest.mark.parametrize("elapsed,expected_row", [(0., 0), (.1, 5), (.2, 10), (.3, 12), (400., 12)])
def test_reference_commands_follow_physical_time_and_hold_final_target(control_dt, stride,
                                                                      elapsed, expected_row):
    # Thirteen 20 ms commands also exercise a final row that is not a
    # multiple of the decimation stride. Explicit expected rows are shared
    # by both control frequencies, including beyond the reference horizon.
    env = BasketUnloadEnv.__new__(BasketUnloadEnv)
    env.config = EnvConfig(action_mode="reference_residual", reference_targets="unused.npz",
                           control_dt=control_dt, reference_stride=stride)
    env.device, env.num_envs = torch.device("cpu"), 2
    env.steps = torch.tensor([round(elapsed / control_dt), 0])
    env.reference_targets = torch.arange(13).float()[:, None].expand(-1, 19) * .0001
    env.residual_scale = torch.zeros(19)
    captured = []

    class CommandCaptured(Exception):
        pass

    def capture(requested, dt):
        captured.append((requested.clone(), dt))
        raise CommandCaptured

    env.sim = SimpleNamespace(lower=-torch.ones(19), upper=torch.ones(19),
                              targets=torch.zeros(2, 19), set_targets=capture)
    # Stop at the actuator interface: this verifies the real step's command
    # selection without claiming anything about the subsequent physics.
    with pytest.raises(CommandCaptured):
        env.step(torch.zeros(2, 19))
    requested, dt = captured[0]
    torch.testing.assert_close(requested[0], torch.full((19,), expected_row * .0001))
    torch.testing.assert_close(requested[1], torch.zeros(19))
    assert dt == control_dt


@pytest.mark.parametrize("stride", [0, -1, 1.5, True, False])
def test_reference_stride_rejects_invalid_values(stride):
    with pytest.raises(ValueError, match="reference_stride must be a positive integer"):
        EnvConfig(reference_stride=stride)
