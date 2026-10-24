"""Evaluator-only unloading rewards; no privileged tensors enter the actor.

Event rewards are per event. Smoothness/effort/slip terms are rates integrated
over dt in seconds, so changing the control frequency does not silently change
their relative weight. The caller must OR first_exit across physics substeps,
and terminate/reset on is_success/is_failure. If it keeps stepping after
success, it must supply an explicit success_event rising-edge mask.
"""

from dataclasses import dataclass, fields
import math

import torch


@dataclass(frozen=True)
class RewardWeights:
    success: float = 100.
    first_exit: float = 10.
    wrong_exit: float = 25.
    reentry: float = 25.
    dangerous_collision: float = 25.
    fallen: float = 50.
    self_intersection: float = 50.
    invalid: float = 50.
    timeout: float = 10.
    timeout_remaining: float = 5.
    time: float = .05
    action_velocity: float = .0005
    action_acceleration: float = .000001
    joint_velocity: float = .0005
    joint_limit: float = .5
    energy: float = .01
    foot_slip: float = .1
    body_tilt: float = 2.
    body_angular_velocity: float = .1
    body_height: float = 2.
    tail_progress: float = 4.
    aperture_alignment: float = 1.
    clearance: float = 0.
    speed_margin: float = 0.
    limit_margin_fraction: float = .05

    def __post_init__(self):
        if any(not math.isfinite(getattr(self, field.name)) or getattr(self, field.name) < 0
               for field in fields(self)):
            raise ValueError("Reward weights must be finite and nonnegative")
        if not 0 < self.limit_margin_fraction <= .5:
            raise ValueError("limit_margin_fraction must be in (0,.5]")


@torch.no_grad()
def compute_reward(result, *, action, previous_action, previous_previous_action,
                   joint_pos, joint_vel, lower, upper, effort_ratio, foot_slip,
                   dangerous_collision, fallen, self_intersection, timeout,
                   stable, dt=.02, weights=None, success_event=None,
                   tail_progress=None, aperture_alignment=None,
                   body_tilt=None, body_angular_speed=None, body_height=None,
                   speed_ratio_peak=None, clearance_min=None):
    """Return (reward[B], terms), entirely on the action tensor's device.

    action/joint tensors are [B,19], lower/upper are [19] or [B,19].
    effort_ratio is actual torque divided by actuator limit, not target error.
    Its product with joint speed is only an effort/power *proxy*, not Joules.
    foot_slip[B] is nonnegative measured tangential foot speed (m/s) during
    contact. Boolean flags have shape [B]. `stable=False` permits intentional
    tilt/stepping: it blocks recovered success, and is not itself a failure.

    terms includes signed reward terms, is_success/is_failure, and individual
    failure_* masks. Failures are latched by metrics or terminal in the caller.
    Reward terms cannot certify simulation validity; the caller supplies real
    measured collision, fall and self-intersection flags.
    """
    if not isinstance(action, torch.Tensor) or action.ndim != 2 or action.shape[-1] != 19:
        raise ValueError("action must be a Torch tensor with shape [B,19]")
    weights = weights or RewardWeights()
    b = action.shape[0]
    device, dtype = action.device, action.dtype
    if not action.is_floating_point():
        raise ValueError("action must be floating point")

    def floating(value):
        return torch.as_tensor(value, device=device, dtype=dtype)

    def flag(value):
        return torch.as_tensor(value, device=device, dtype=torch.bool).expand(b)

    previous_action = floating(previous_action)
    previous_previous_action = floating(previous_previous_action)
    joint_pos, joint_vel = floating(joint_pos), floating(joint_vel)
    effort_ratio, foot_slip = floating(effort_ratio), floating(foot_slip).expand(b)
    tail_progress = (torch.zeros(b, device=device, dtype=dtype) if tail_progress is None
                     else floating(tail_progress).expand(b))
    aperture_alignment = (torch.zeros(b, device=device, dtype=dtype)
                          if aperture_alignment is None
                          else floating(aperture_alignment).expand(b))
    body_tilt = floating(0. if body_tilt is None else body_tilt).expand(b)
    body_angular_speed = floating(0. if body_angular_speed is None else body_angular_speed).expand(b)
    body_height = floating(.6 if body_height is None else body_height).expand(b)
    speed_ratio_peak = floating(0. if speed_ratio_peak is None else speed_ratio_peak).expand(b)
    clearance_min = floating(.02 if clearance_min is None else clearance_min).expand(b)
    lower, upper = floating(lower).expand(b, 19), floating(upper).expand(b, 19)
    for name, value in (("previous_action", previous_action),
                        ("previous_previous_action", previous_previous_action),
                        ("joint_pos", joint_pos), ("joint_vel", joint_vel),
                        ("effort_ratio", effort_ratio)):
        if value.shape != action.shape:
            raise ValueError(f"{name} must have shape [B,19]")
    dt = floating(dt).expand(b)
    stable = flag(stable)
    dangerous_collision, fallen = flag(dangerous_collision), flag(fallen)
    self_intersection, timeout = flag(self_intersection), flag(timeout)

    finite = torch.isfinite(foot_slip) & torch.isfinite(dt) & (dt > 0)
    finite &= torch.isfinite(tail_progress) & torch.isfinite(aperture_alignment)
    finite &= (torch.isfinite(body_tilt) & torch.isfinite(body_angular_speed)
               & torch.isfinite(body_height))
    # Disabled options retain the previous reward/termination semantics even
    # if an unused diagnostic is nonfinite. Enabled diagnostics use the same
    # numeric fault path as other reward inputs.
    if weights.speed_margin:
        finite &= torch.isfinite(speed_ratio_peak)
    if weights.clearance:
        finite &= torch.isfinite(clearance_min)
    speed_margin_cost = (((torch.nan_to_num(speed_ratio_peak) - .6).clamp_min(0) / .4).square()
                         if weights.speed_margin else torch.zeros_like(speed_ratio_peak))
    clearance_cost = (((.01 - torch.nan_to_num(clearance_min)).clamp_min(0) / .01).square()
                      if weights.clearance else torch.zeros_like(clearance_min))
    finite &= torch.isfinite(speed_margin_cost) & torch.isfinite(clearance_cost)
    for value in (action, previous_action, previous_previous_action, joint_pos,
                  joint_vel, effort_ratio, lower, upper):
        finite &= torch.isfinite(value).all(-1)
    finite &= (upper > lower).all(-1)

    failure = {
        "failure_wrong_exit": result.wrong.any(-1),
        "failure_reentry": result.reentered.any(-1),
        "failure_invalid_initial": result.invalid_initial.any(-1),
        "failure_invalid_trajectory": result.invalid_trajectory.any(-1),
        "failure_invalid_dynamics": result.invalid_dynamics,
        "failure_nonfinite_input": ~finite,
        "failure_dangerous_collision": dangerous_collision,
        "failure_fallen": fallen,
        "failure_self_intersection": self_intersection,
    }
    non_timeout_failure = torch.stack(tuple(failure.values()), -1).any(-1)
    is_success = result.success & stable & ~non_timeout_failure
    failure["failure_timeout"] = timeout & ~is_success
    is_failure = non_timeout_failure | failure["failure_timeout"]
    success_event = is_success if success_event is None else flag(success_event) & is_success

    # Faults terminate with a finite reward rather than poisoning PPO buffers.
    clean_dt = torch.nan_to_num(dt, nan=0., posinf=0., neginf=0.).clamp_min(0)
    safe_dt = clean_dt.clamp_min(1e-6)
    action = torch.nan_to_num(action)
    previous_action = torch.nan_to_num(previous_action)
    previous_previous_action = torch.nan_to_num(previous_previous_action)
    joint_pos, joint_vel = torch.nan_to_num(joint_pos), torch.nan_to_num(joint_vel)
    effort_ratio = torch.nan_to_num(effort_ratio)
    lower, upper = torch.nan_to_num(lower), torch.nan_to_num(upper)
    target_velocity = (action - previous_action) / safe_dt[:, None]
    target_acceleration = (action - 2 * previous_action + previous_previous_action) / safe_dt[:, None].square()
    span = (upper - lower).clamp_min(1e-6)
    normalized_margin = torch.minimum(joint_pos - lower, upper - joint_pos) / span
    limit_cost = ((weights.limit_margin_fraction - normalized_margin).clamp_min(0)
                  / weights.limit_margin_fraction).square().mean(-1)
    remaining = (~result.exited).sum(-1).to(dtype)
    invalid = (failure["failure_invalid_initial"] | failure["failure_invalid_trajectory"]
               | failure["failure_invalid_dynamics"] | failure["failure_nonfinite_input"])
    terms = {
        "success": weights.success * success_event.to(dtype),
        "first_exit": weights.first_exit * result.first_exit.sum(-1).to(dtype) * (~is_failure).to(dtype),
        "wrong_exit": -weights.wrong_exit * failure["failure_wrong_exit"].to(dtype),
        "reentry": -weights.reentry * failure["failure_reentry"].to(dtype),
        "dangerous_collision": -weights.dangerous_collision * dangerous_collision.to(dtype),
        "fallen": -weights.fallen * fallen.to(dtype),
        "self_intersection": -weights.self_intersection * self_intersection.to(dtype),
        "invalid": -weights.invalid * invalid.to(dtype),
        "timeout": -weights.timeout * failure["failure_timeout"].to(dtype),
        "timeout_remaining": -weights.timeout_remaining * remaining * failure["failure_timeout"].to(dtype),
        "time": -weights.time * clean_dt,
        "action_velocity": -weights.action_velocity * target_velocity.square().mean(-1) * clean_dt,
        "action_acceleration": -weights.action_acceleration * target_acceleration.square().mean(-1) * clean_dt,
        "joint_velocity": -weights.joint_velocity * joint_vel.square().mean(-1) * clean_dt,
        "joint_limit": -weights.joint_limit * limit_cost * clean_dt,
        "energy": -weights.energy * (effort_ratio * joint_vel).abs().mean(-1) * clean_dt,
        "foot_slip": -weights.foot_slip * torch.nan_to_num(foot_slip).clamp_min(0) * clean_dt,
        # Allow the intended ~15 degree unloading lean. Penalize excess tilt
        # above 20 degrees and low body height before the hard fall terminal.
        "body_tilt": -weights.body_tilt * torch.nan_to_num(body_tilt - .35).clamp_min(0).square() * clean_dt,
        "body_angular_velocity": -weights.body_angular_velocity * torch.nan_to_num(body_angular_speed).square() * clean_dt,
        "body_height": -weights.body_height * (torch.nan_to_num(.4 - body_height).clamp_min(0) / .1).square() * clean_dt,
        # These are evaluator-only potential differences derived from the
        # current basket-frame cargo geometry.  They are deliberately absent
        # from the actor observation and vanish for the balance curriculum.
        "tail_progress": weights.tail_progress * torch.nan_to_num(tail_progress),
        "aperture_alignment": weights.aperture_alignment * torch.nan_to_num(aperture_alignment),
        # Max across hardware-normalized joint speeds avoids diluting the
        # slow gripper's margin by the other 18 joints. Both inputs are peaks
        # within this control interval, not episode-lifetime extrema.
        "speed_margin": -weights.speed_margin * speed_margin_cost * clean_dt,
        "clearance": -weights.clearance * clearance_cost * clean_dt,
    }
    # Extreme invalid inputs may overflow intermediate squared quantities.
    # The invalid penalty supplies their terminal signal; dense terms are zero.
    for name in terms:
        if name in ("time", "action_velocity", "action_acceleration", "joint_velocity",
                    "joint_limit", "energy", "foot_slip", "tail_progress",
                    "aperture_alignment", "body_tilt", "body_angular_velocity", "body_height",
                    "speed_margin", "clearance"):
            terms[name] = torch.where(finite, terms[name], torch.zeros_like(terms[name]))
    reward = torch.stack(tuple(terms.values()), -1).sum(-1)
    terms.update(failure)
    terms["is_failure"] = is_failure
    terms["is_success"] = is_success
    return reward, terms
