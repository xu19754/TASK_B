"""Only Task B, with B2+Z1 and placement-only episode scores."""

import math

import isaaclab.sim as sim_utils
from isaaclab.assets import ArticulationCfg, AssetBaseCfg, RigidObjectCfg
from isaaclab.envs import ManagerBasedRLEnvCfg, mdp
from isaaclab.managers import EventTermCfg as EventTerm
from isaaclab.managers import ObservationGroupCfg as ObsGroup
from isaaclab.managers import ObservationTermCfg as ObsTerm
from isaaclab.managers import RewardTermCfg as RewTerm
from isaaclab.managers import SceneEntityCfg
from isaaclab.managers import TerminationTermCfg as DoneTerm
from isaaclab.scene import InteractiveSceneCfg
from isaaclab.sensors import CameraCfg, ContactSensorCfg
from isaaclab.utils import configclass

from .arena import ArenaCfg
from .constants import ARM_JOINTS, JOINT_NAMES, LEG_JOINTS, ROOT
from .geometry import sample_object_positions
from .lidar import LidarCfg, lidar_observation
from .mdp import ObjectsInTarget, all_objects_in_target, camera_intrinsics
from .robot import B2Z1_CFG


@configclass
class SensorSimulationCfg(sim_utils.SimulationCfg):
    # Validate perception against CPU PhysX poses; RGB-D still renders on the GPU.
    device: str = "cpu"
    use_fabric: bool = False
    enable_scene_query_support: bool = True


@configclass
class TaskBSceneCfg(InteractiveSceneCfg):
    arena = AssetBaseCfg(prim_path="/World/ground", spawn=ArenaCfg())
    robot: ArticulationCfg = B2Z1_CFG
    lidar = AssetBaseCfg(
        prim_path="{ENV_REGEX_NS}/Robot/base/lidar", spawn=LidarCfg(),
        init_state=AssetBaseCfg.InitialStateCfg(pos=(0.38, 0.0, 0.23)),
    )
    light = AssetBaseCfg(
        prim_path="/World/light", spawn=sim_utils.DomeLightCfg(intensity=2000.0),
    )
    contact_sensor = ContactSensorCfg(
        prim_path="{ENV_REGEX_NS}/Robot/.*", history_length=3, update_period=0.02,
    )
    head_camera = CameraCfg(
        prim_path="{ENV_REGEX_NS}/Robot/base/head_camera",
        update_period=0.1, height=480, width=640,
        data_types=["rgb", "distance_to_image_plane"],
        spawn=sim_utils.PinholeCameraCfg(
            focal_length=15.0, horizontal_aperture=20.955, clipping_range=(0.05, 50.0),
        ),
        offset=CameraCfg.OffsetCfg(
            pos=(0.4, 0.0, 0.12), rot=(1.0, 0.0, 0.0, 0.0), convention="world",
        ),
    )
    ee_camera = CameraCfg(
        prim_path="{ENV_REGEX_NS}/Robot/link06/ee_camera",
        update_period=0.1, height=480, width=640,
        data_types=["rgb", "distance_to_image_plane"],
        spawn=sim_utils.PinholeCameraCfg(
            focal_length=15.0, horizontal_aperture=20.955, clipping_range=(0.05, 50.0),
        ),
        offset=CameraCfg.OffsetCfg(
            pos=(0.03, 0.0, 0.07), rot=(1.0, 0.0, 0.0, 0.0), convention="world",
        ),
    )


@configclass
class ActionsCfg:
    joint_leg = mdp.JointPositionActionCfg(
        asset_name="robot", joint_names=LEG_JOINTS, preserve_order=True,
        scale=1.0, use_default_offset=False,
    )
    joint_arm = mdp.JointPositionActionCfg(
        asset_name="robot", joint_names=ARM_JOINTS, preserve_order=True,
        scale=1.0, use_default_offset=False,
    )


@configclass
class ObservationsCfg:
    @configclass
    class ProprioCfg(ObsGroup):
        base_lin_vel = ObsTerm(func=mdp.base_lin_vel)
        base_ang_vel = ObsTerm(func=mdp.base_ang_vel)
        projected_gravity = ObsTerm(func=mdp.projected_gravity)
        base_quat = ObsTerm(func=mdp.root_quat_w)
        joint_pos = ObsTerm(
            func=mdp.joint_pos,
            params={"asset_cfg": SceneEntityCfg("robot", joint_names=JOINT_NAMES, preserve_order=True)},
        )
        joint_vel = ObsTerm(
            func=mdp.joint_vel,
            params={"asset_cfg": SceneEntityCfg("robot", joint_names=JOINT_NAMES, preserve_order=True)},
        )

        def __post_init__(self):
            self.concatenate_terms = False
            self.enable_corruption = False

    @configclass
    class ImageCfg(ObsGroup):
        head_rgb = ObsTerm(func=mdp.image, params={
            "sensor_cfg": SceneEntityCfg("head_camera"), "data_type": "rgb", "normalize": False,
        })
        head_depth = ObsTerm(func=mdp.image, params={
            "sensor_cfg": SceneEntityCfg("head_camera"), "data_type": "distance_to_image_plane",
        })
        ee_rgb = ObsTerm(func=mdp.image, params={
            "sensor_cfg": SceneEntityCfg("ee_camera"), "data_type": "rgb", "normalize": False,
        })
        ee_depth = ObsTerm(func=mdp.image, params={
            "sensor_cfg": SceneEntityCfg("ee_camera"), "data_type": "distance_to_image_plane",
        })
        head_intrinsics = ObsTerm(func=camera_intrinsics, params={"sensor_cfg": SceneEntityCfg("head_camera")})
        ee_intrinsics = ObsTerm(func=camera_intrinsics, params={"sensor_cfg": SceneEntityCfg("ee_camera")})

        def __post_init__(self):
            self.concatenate_terms = False
            self.enable_corruption = False

    proprio: ProprioCfg = ProprioCfg()
    image: ImageCfg = ImageCfg()

    @configclass
    class LidarObsCfg(ObsGroup):
        ranges = ObsTerm(func=lidar_observation, params={"data_type": "ranges"})
        points = ObsTerm(func=lidar_observation, params={"data_type": "points"})
        hit_mask = ObsTerm(func=lidar_observation, params={"data_type": "hit_mask"})
        ready = ObsTerm(func=lidar_observation, params={"data_type": "ready"})

        def __post_init__(self):
            self.concatenate_terms = False
            self.enable_corruption = False

    lidar: LidarObsCfg = LidarObsCfg()


@configclass
class RewardsCfg:
    objects_in_target = RewTerm(func=ObjectsInTarget, weight=1.0)


@configclass
class TerminationsCfg:
    time_out = DoneTerm(func=mdp.time_out, time_out=True)
    illegal_contact = DoneTerm(func=mdp.illegal_contact, params={
        "sensor_cfg": SceneEntityCfg("contact_sensor", body_names=["base", ".*_hip", ".*_thigh"]),
        "threshold": 1.0,
    })
    fall = DoneTerm(func=mdp.root_height_below_minimum, params={"minimum_height": 0.0})
    all_delivered = DoneTerm(func=all_objects_in_target)


@configclass
class EventsCfg:
    reset_scene = EventTerm(func=mdp.reset_scene_to_default, mode="reset")


@configclass
class CommandsCfg:
    pass


@configclass
class TaskBEnvCfg(ManagerBasedRLEnvCfg):
    seed: int = 42
    sim: SensorSimulationCfg = SensorSimulationCfg()
    scene: TaskBSceneCfg = TaskBSceneCfg(num_envs=1, env_spacing=25.0)
    actions: ActionsCfg = ActionsCfg()
    observations: ObservationsCfg = ObservationsCfg()
    rewards: RewardsCfg = RewardsCfg()
    terminations: TerminationsCfg = TerminationsCfg()
    events: EventsCfg = EventsCfg()
    commands: CommandsCfg = CommandsCfg()

    def __post_init__(self):
        self.decimation = 4
        self.sim.dt = 0.005
        self.sim.render_interval = 4
        self.sim.enable_scene_query_support = True
        # The PhysX lidar reads USD parent poses, not GPU-only Fabric transforms.
        self.sim.use_fabric = False
        self.sim.physx.solver_type = 1
        self.episode_length_s = 1200.0
        self.viewer.eye = (-6.0, -15.0, 4.0)
        self.viewer.lookat = (-10.0, -10.0, 0.4)
        self.sim.physics_material = sim_utils.RigidBodyMaterialCfg(
            static_friction=1.0, dynamic_friction=1.0, restitution=0.0,
        )
        if self.scene.num_envs != 1:
            raise ValueError("This student evaluation arena supports num_envs=1 only.")
        files = ["004_sugar_box.usd"] * 6 + ["006_mustard_bottle.usd"] * 6 + ["011_banana.usd"] * 6
        for i, (position, filename) in enumerate(zip(sample_object_positions(self.seed), files), start=1):
            rotation = (0.0, math.sqrt(0.5), 0.0, math.sqrt(0.5)) if i <= 6 else (
                0.0, 0.0, -math.sqrt(0.5), math.sqrt(0.5)
            )
            cfg = RigidObjectCfg(
                prim_path=f"{{ENV_REGEX_NS}}/Object{i}",
                spawn=sim_utils.UsdFileCfg(
                    usd_path=str(ROOT / "assets/objects" / filename),
                    rigid_props=sim_utils.RigidBodyPropertiesCfg(
                        rigid_body_enabled=True, kinematic_enabled=False,
                        linear_damping=2.0, angular_damping=4.0, max_depenetration_velocity=0.5,
                    ),
                    mass_props=sim_utils.MassPropertiesCfg(mass=0.5),
                    collision_props=sim_utils.CollisionPropertiesCfg(contact_offset=0.01, rest_offset=0.0),
                ),
                init_state=RigidObjectCfg.InitialStateCfg(pos=tuple(position), rot=rotation),
            )
            setattr(self.scene, f"object_{i}", cfg)

    def configure_cameras(self, enabled):
        if not enabled:
            self.scene.head_camera = None
            self.scene.ee_camera = None
            self.observations.image = None
