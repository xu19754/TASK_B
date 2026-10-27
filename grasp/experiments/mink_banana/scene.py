"""Independent diagnostic scene; imports require an already started AppLauncher."""
from pathlib import Path
import torch,numpy as np
import isaaclab.sim as sim_utils
from isaaclab.assets import AssetBaseCfg,RigidObjectCfg
from isaaclab.scene import InteractiveScene,InteractiveSceneCfg
from isaaclab.sensors import CameraCfg,ContactSensorCfg
from isaaclab.utils import configclass
from taskb_student.robot import B2Z1_CFG
from taskb_student.constants import ROOT
from pxr import Usd,UsdGeom,Gf
from scipy.spatial.transform import Rotation
from experiments.mink_wholebody.kinematics import quat_matrix

def asset_vertices(path):
    stage=Usd.Stage.Open(str(Path(path).resolve()))
    cache=UsdGeom.XformCache(Usd.TimeCode.Default());points=[]
    for prim in stage.Traverse():
        if prim.IsA(UsdGeom.Mesh):
            xf=cache.GetLocalToWorldTransform(prim)
            pp=UsdGeom.Mesh(prim).GetPointsAttr().Get()
            points.extend([list(xf.Transform(Gf.Vec3d(*map(float,p)))) for p in pp])
    return np.asarray(points,float)

@configclass
class GraspSceneCfg(InteractiveSceneCfg):
    ground=AssetBaseCfg(prim_path='/World/ground',spawn=sim_utils.CuboidCfg(size=(20,20,.1),collision_props=sim_utils.CollisionPropertiesCfg(),physics_material=sim_utils.RigidBodyMaterialCfg(static_friction=1.,dynamic_friction=1.,restitution=0.),visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(.25,.28,.3))),init_state=AssetBaseCfg.InitialStateCfg(pos=(0.,0.,-.05)))
    light=AssetBaseCfg(prim_path='/World/light',spawn=sim_utils.DomeLightCfg(intensity=1400.))
    robot=B2Z1_CFG.replace(prim_path='{ENV_REGEX_NS}/Robot')
    contacts=ContactSensorCfg(prim_path='{ENV_REGEX_NS}/Robot/.*',history_length=3,update_period=.02)
    jaw_fixed=ContactSensorCfg(prim_path='{ENV_REGEX_NS}/Robot/gripperStator',filter_prim_paths_expr=['{ENV_REGEX_NS}/Object.*'],max_contact_data_count_per_prim=64,update_period=.02)
    jaw_moving=ContactSensorCfg(prim_path='{ENV_REGEX_NS}/Robot/gripperMover',filter_prim_paths_expr=['{ENV_REGEX_NS}/Object.*'],max_contact_data_count_per_prim=64,update_period=.02)
    camera=CameraCfg(prim_path='/World/Camera',height=720,width=960,update_period=.1,data_types=['rgb'],spawn=sim_utils.PinholeCameraCfg(focal_length=24.,horizontal_aperture=36.,clipping_range=(.03,50.)))

def build_scene(config,cameras=True):
    cfg=GraspSceneCfg(num_envs=1,env_spacing=20.)
    cfg.robot.init_state.pos=(0.,0.,.6)
    path=ROOT/'assets/objects'/config['object_asset']
    vertices=asset_vertices(path)
    rot=quat_matrix(config['object_quat_wxyz'])
    z=-float((vertices@rot.T)[:,2].min())+.002
    cfg.object=RigidObjectCfg(prim_path='{ENV_REGEX_NS}/Object',spawn=sim_utils.UsdFileCfg(usd_path=str(path),rigid_props=sim_utils.RigidBodyPropertiesCfg(rigid_body_enabled=True,kinematic_enabled=False,linear_damping=2.,angular_damping=4.,max_depenetration_velocity=.5),mass_props=sim_utils.MassPropertiesCfg(mass=.5),collision_props=sim_utils.CollisionPropertiesCfg(contact_offset=.01,rest_offset=0.)),init_state=RigidObjectCfg.InitialStateCfg(pos=(*config['object_xy'],z),rot=tuple(config['object_quat_wxyz'])))
    if config.get('solver_iterations'):
        pos,vel=config['solver_iterations']
        cfg.robot.spawn.articulation_props.solver_position_iteration_count=int(pos)
        cfg.robot.spawn.articulation_props.solver_velocity_iteration_count=int(vel)
        cfg.object.spawn.rigid_props.solver_position_iteration_count=int(pos)
        cfg.object.spawn.rigid_props.solver_velocity_iteration_count=int(vel)
    if not cameras: cfg.camera=None
    return InteractiveScene(cfg),vertices,z
