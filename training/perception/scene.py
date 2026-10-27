"""Standalone B2+Z1 and object scene, without evaluation or perception logic."""

import math

import numpy as np
import isaaclab.sim as sim_utils
from isaaclab.assets import ArticulationCfg, AssetBaseCfg, RigidObjectCfg
from isaaclab.scene import InteractiveSceneCfg
from isaaclab.utils import configclass

from taskb_student.constants import ROOT
from taskb_student.robot import B2Z1_CFG


OBJECT_ASSETS = ("004_sugar_box", "006_mustard_bottle", "011_banana")


@configclass
class TrainingSceneCfg(InteractiveSceneCfg):
    ground = AssetBaseCfg(
        prim_path="/World/Ground",
        spawn=sim_utils.CuboidCfg(
            size=(20.0, 20.0, 0.1),
            collision_props=sim_utils.CollisionPropertiesCfg(),
            visual_material=sim_utils.PreviewSurfaceCfg(
                diffuse_color=(0.27, 0.31, 0.32), roughness=0.85,
            ),
        ),
        init_state=AssetBaseCfg.InitialStateCfg(pos=(0.0, 0.0, -0.05)),
    )
    robot: ArticulationCfg = B2Z1_CFG.copy()
    light = AssetBaseCfg(
        prim_path="/World/Light", spawn=sim_utils.DomeLightCfg(intensity=2000.0),
    )


def make_scene_cfg(seed):
    scene = TrainingSceneCfg(num_envs=1, env_spacing=25.0)
    scene.robot.init_state.pos = (0.0, 0.0, 0.6)
    rng = np.random.default_rng(seed)
    for index, name in enumerate(OBJECT_ASSETS):
        # A compact row in front of the robot, with small seeded layout variations.
        xy = np.array([1.5, 0.30 * (1 - index)]) + rng.uniform(-0.02, 0.02, 2)
        # Preserve the source assets' ground orientation from the Task B scene.
        rotation = (0.0, math.sqrt(0.5), 0.0, math.sqrt(0.5)) if index == 0 else (
            0.0, 0.0, -math.sqrt(0.5), math.sqrt(0.5)
        )
        cfg = RigidObjectCfg(
            prim_path=f"{{ENV_REGEX_NS}}/Object{index + 1:02d}",
            spawn=sim_utils.UsdFileCfg(
                usd_path=str(ROOT / "assets/objects" / f"{name}.usd"),
                rigid_props=sim_utils.RigidBodyPropertiesCfg(
                    linear_damping=2.0, angular_damping=4.0, max_depenetration_velocity=0.5,
                ),
                mass_props=sim_utils.MassPropertiesCfg(mass=0.5),
                collision_props=sim_utils.CollisionPropertiesCfg(contact_offset=0.01, rest_offset=0.0),
            ),
            init_state=RigidObjectCfg.InitialStateCfg(
                pos=(float(xy[0]), float(xy[1]), 0.15 if index == 0 else 0.10), rot=rotation,
            ),
        )
        setattr(scene, f"object_{index + 1:02d}", cfg)
    return scene
