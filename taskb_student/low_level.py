"""The supplied model_98499 policy: 52 current observations -> 12 leg actions."""

import hashlib
import math

import numpy as np
import torch

from .constants import (
    ARM_Q0, CONTROL_DT, JOINT_LOWER, JOINT_UPPER, LEG_ACTION_SCALE, LEG_Q0,
    POLICY_PATH, POLICY_SHA256,
)


def _vector(value, size):
    if isinstance(value, torch.Tensor):
        value = value.detach().cpu().numpy()
    result = np.asarray(value, dtype=np.float32)
    if result.shape == (1, size):
        result = result[0]
    if result.shape != (size,) or not np.isfinite(result).all():
        raise ValueError(f"Expected a finite {size}-D vector; got {result.shape}.")
    return result


class ObservationBuilder:
    def __init__(self):
        self.reset()

    def reset(self):
        self.phase = 0.0
        self.previous_action = np.zeros(12, dtype=np.float32)
        self.pure_vx_sign = 0.0
        self.segment_heading = 0.0

    def build(self, proprio, command):
        command = _vector(command, 3).copy()
        if abs(float(command[1])) > 1e-6:
            raise ValueError("model_98499 requires cmd_y=0; use forward motion and yaw to navigate.")
        command[0] = np.clip(command[0], -0.4, 0.4)
        command[2] = np.clip(command[2], -0.6, 0.6)
        command[np.abs(command) <= 0.1] = 0.0
        moving = bool(command[0] != 0.0 or command[2] != 0.0)
        self.phase = (self.phase + CONTROL_DT * 2.0) % 1.0 if moving else 0.0

        quaternion = _vector(proprio["base_quat"], 4)
        norm = float(np.linalg.norm(quaternion))
        if norm < 1e-8:
            raise ValueError("Invalid base orientation quaternion.")
        w, x, y, z = quaternion / norm
        yaw = math.atan2(2.0 * (w * z + x * y), 1.0 - 2.0 * (y * y + z * z))
        pure_vx = command[0] != 0.0 and command[2] == 0.0
        sign = math.copysign(1.0, float(command[0])) if pure_vx else 0.0
        if pure_vx and sign != self.pure_vx_sign:
            self.segment_heading = yaw
        self.pure_vx_sign = sign
        heading = np.zeros(2, dtype=np.float32)
        if pure_vx:
            error = yaw - self.segment_heading
            heading[:] = (math.sin(error), math.cos(error) - 1.0)
        clocks = np.sin(2.0 * math.pi * np.remainder(
            [self.phase + 0.5, self.phase, self.phase, self.phase + 0.5], 1.0
        ))
        q = _vector(proprio["joint_pos"], 19)[:12]
        dq = _vector(proprio["joint_vel"], 19)[:12]
        obs = np.concatenate((
            _vector(proprio["base_ang_vel"], 3),
            _vector(proprio["projected_gravity"], 3),
            command, q - np.asarray(LEG_Q0, dtype=np.float32), dq,
            self.previous_action, [self.phase], clocks, heading,
        )).astype(np.float32)
        return np.clip(obs, -100.0, 100.0)


class LowLevelController:
    def __init__(self):
        if hashlib.sha256(POLICY_PATH.read_bytes()).hexdigest() != POLICY_SHA256:
            raise ValueError("The supplied locomotion policy checksum does not match its contract.")
        self.policy = torch.jit.load(str(POLICY_PATH), map_location="cpu").eval()
        self.builder = ObservationBuilder()
        self.reset()

    def reset(self):
        self.reset_locomotion()
        self.arm_target = np.asarray(ARM_Q0, dtype=np.float32).copy()

    def reset_locomotion(self):
        """Reset leg history without unloading a grasp already held by the arm."""
        self.policy.reset()
        self.builder.reset()

    @torch.inference_mode()
    def step(self, proprio, command, arm_target=None):
        current = self.builder.build(proprio, command)
        raw = self.policy(torch.from_numpy(current).unsqueeze(0))
        raw = _vector(raw, 12)
        leg_q0 = np.asarray(LEG_Q0, dtype=np.float32)
        scale = np.asarray(LEG_ACTION_SCALE, dtype=np.float32)
        clipped_action = np.clip(raw, -100.0, 100.0)
        legs = np.clip(leg_q0 + scale * clipped_action, JOINT_LOWER[:12], JOINT_UPPER[:12])
        # The training/deployment observation uses the delayed policy action, before target clipping.
        self.builder.previous_action[:] = clipped_action
        requested = np.asarray(ARM_Q0, dtype=np.float32) if arm_target is None else _vector(arm_target, 7)
        requested = np.clip(requested, JOINT_LOWER[12:], JOINT_UPPER[12:])
        slew = np.asarray([3.0] * 6 + [0.25], dtype=np.float32) * CONTROL_DT
        previous_grip = float(self.arm_target[6])
        self.arm_target += np.clip(requested - self.arm_target, -slew, slew)
        actual_grip = float(_vector(proprio["joint_pos"], 19)[18])
        if requested[6] > actual_grip:
            # Do not accumulate a large closing error after the jaw contacts an object.
            self.arm_target[6] = min(self.arm_target[6], max(previous_grip - slew[6], actual_grip + 0.04))
        return np.concatenate((legs, self.arm_target)).astype(np.float32)
