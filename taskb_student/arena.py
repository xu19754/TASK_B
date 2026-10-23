from typing import Callable

import isaaclab.sim as sim_utils
from isaaclab.sim.spawners.spawner_cfg import SpawnerCfg
from isaaclab.terrains.utils import create_prim_from_mesh
from isaaclab.utils import configclass
from isaacsim.core.utils.prims import create_prim

from .geometry import build_arena_meshes


def spawn_arena(prim_path, cfg, translation=None, orientation=None, **kwargs):
    root = create_prim(prim_path, "Xform", translation=translation, orientation=orientation)
    for name, mesh in build_arena_meshes():
        color = (0.27, 0.31, 0.32) if name == "ground" else (1.0, 0.42, 0.04)
        create_prim_from_mesh(
            f"{prim_path}/{name}", mesh,
            visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=color),
            physics_material=sim_utils.RigidBodyMaterialCfg(
                static_friction=1.0, dynamic_friction=1.0, restitution=0.0,
                friction_combine_mode="multiply", restitution_combine_mode="multiply",
            ),
        )
    return root


@configclass
class ArenaCfg(SpawnerCfg):
    func: Callable = spawn_arena
