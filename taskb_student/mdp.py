import torch
from isaaclab.managers.manager_base import ManagerTermBase

from .constants import NUM_OBJECTS
from .scoring import PlacementCounter, objects_inside_target


def object_positions(env):
    return torch.stack(
        [env.scene[f"object_{i}"].data.root_pos_w for i in range(1, NUM_OBJECTS + 1)],
        dim=1,
    )


class ObjectsInTarget(ManagerTermBase):
    def __init__(self, cfg, env):
        super().__init__(cfg, env)
        self.counter = PlacementCounter(env.num_envs, env.device)

    def reset(self, env_ids=None):
        self.counter.reset(env_ids)

    def __call__(self, env):
        return self.counter.update(object_positions(env), env.scene.env_origins)


def all_objects_in_target(env):
    return objects_inside_target(object_positions(env), env.scene.env_origins).all(dim=1)


def camera_intrinsics(env, sensor_cfg):
    return env.scene[sensor_cfg.name].data.intrinsic_matrices
