"""Load the B2+Z1 URDF variant with the fixed 200 g rear basket."""

import isaaclab.sim as sim_utils
from isaaclab.actuators import DCMotorCfg, ImplicitActuatorCfg
from isaaclab.assets import ArticulationCfg
from isaaclab.sim.converters import UrdfConverterCfg

from .constants import ARM_JOINTS, ARM_Q0, JOINT_NAMES, LEG_Q0, SPAWN_POSITION, URDF_PATH

B2Z1_CFG = ArticulationCfg(
    prim_path="{ENV_REGEX_NS}/Robot",
    spawn=sim_utils.UrdfFileCfg(
        asset_path=str(URDF_PATH),
        fix_base=False,
        root_link_name="base",
        merge_fixed_joints=False,
        self_collision=False,
        collider_type="convex_decomposition",
        joint_drive=UrdfConverterCfg.JointDriveCfg(
            gains=UrdfConverterCfg.JointDriveCfg.PDGainsCfg(stiffness=0.0, damping=0.0),
        ),
        activate_contact_sensors=True,
        rigid_props=sim_utils.RigidBodyPropertiesCfg(
            disable_gravity=False, linear_damping=0.0, angular_damping=0.0,
            max_depenetration_velocity=1.0,
        ),
        articulation_props=sim_utils.ArticulationRootPropertiesCfg(
            enabled_self_collisions=False,
            solver_position_iteration_count=4,
            solver_velocity_iteration_count=1,
        ),
    ),
    init_state=ArticulationCfg.InitialStateCfg(
        pos=SPAWN_POSITION,
        joint_pos=dict(zip(JOINT_NAMES, LEG_Q0 + ARM_Q0)),
        joint_vel={".*": 0.0},
    ),
    soft_joint_pos_limit_factor=0.9,
    actuators={
        "legs": DCMotorCfg(
            joint_names_expr=[".*_hip_joint", ".*_thigh_joint", ".*_calf_joint"],
            effort_limit=300.0, saturation_effort=300.0, velocity_limit=23.0,
            stiffness=360.0, damping=5.0, friction=0.0,
        ),
        **{
            name: ImplicitActuatorCfg(
                joint_names_expr=[name],
                effort_limit_sim=5.0 if name == "jointGripper" else (60.0 if name == "joint2" else 30.0),
                velocity_limit_sim=0.25 if name == "jointGripper" else 3.0,
                stiffness=80.0 if name == "jointGripper" else kp,
                damping=8.0 if name == "jointGripper" else 25.6,
            )
            for name, kp in zip(ARM_JOINTS, [512.0, 768.0, 768.0, 512.0, 384.0, 256.0, 512.0])
        },
    },
)
