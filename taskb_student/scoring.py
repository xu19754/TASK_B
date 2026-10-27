"""Placement geometry and one-time scoring, independent of the simulator."""

import torch

from .constants import NUM_OBJECTS, TARGET_CENTER, TARGET_RADIUS, TARGET_Z_RANGE


def objects_inside_target(positions, env_origins=None):
    """positions: (num_envs, 18, 3); coordinates are object-root positions."""
    if positions.ndim != 3 or tuple(positions.shape[1:]) != (NUM_OBJECTS, 3):
        raise ValueError(f"Expected (num_envs, {NUM_OBJECTS}, 3) object positions.")
    local = positions
    if env_origins is not None:
        if tuple(env_origins.shape) != (positions.shape[0], 3):
            raise ValueError("Environment origins must have shape (num_envs, 3).")
        local = positions - env_origins[:, None, :]
    center = positions.new_tensor(TARGET_CENTER)
    inside_xy = ((local[..., :2] - center).square().sum(dim=-1) <= TARGET_RADIUS**2)
    inside_z = (local[..., 2] >= TARGET_Z_RANGE[0]) & (local[..., 2] <= TARGET_Z_RANGE[1])
    return inside_xy & inside_z & torch.isfinite(local).all(dim=-1)


class PlacementCounter:
    """Each object contributes one point at most, until that episode resets."""

    def __init__(self, num_envs, device="cpu"):
        self.counted = torch.zeros(num_envs, NUM_OBJECTS, dtype=torch.bool, device=device)

    def reset(self, env_ids=None):
        if env_ids is None:
            self.counted.zero_()
        else:
            self.counted[env_ids] = False

    def update(self, positions, env_origins=None):
        inside = objects_inside_target(positions, env_origins)
        if inside.shape != self.counted.shape:
            raise ValueError("Scoring batch does not match the environment count.")
        newly_inside = inside & ~self.counted
        self.counted |= inside
        return newly_inside.sum(dim=1).to(torch.float32)
