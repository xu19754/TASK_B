"""Independent Isaac standing/camera/coexistence probe; no changes to Task B."""
import sys,argparse,json,time
from pathlib import Path
ROOT=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT))
from isaaclab.app import AppLauncher
p=argparse.ArgumentParser()
p.add_argument('--output',required=True)
p.add_argument('--steps',type=int,default=300)
AppLauncher.add_app_launcher_args(p)
p.set_defaults(device='cpu')
args=p.parse_args()
app=AppLauncher(args).app
import torch,numpy as np
import isaaclab.sim as sim_utils
from isaaclab.assets import AssetBaseCfg,RigidObjectCfg
from isaaclab.scene import InteractiveScene,InteractiveSceneCfg
from isaaclab.sensors import CameraCfg,ContactSensorCfg
from isaaclab.utils import configclass
from taskb_student.robot import B2Z1_CFG
from taskb_student.constants import JOINT_NAMES,ARM_Q0
from taskb_student.low_level import LowLevelController
from pxr import Usd,UsdGeom,Gf
from PIL import Image
import mink,mujoco,qpsolvers

@configclass
class ProbeScene(InteractiveSceneCfg):
    ground=AssetBaseCfg(prim_path='/World/ground',spawn=sim_utils.CuboidCfg(size=(20,20,.1),collision_props=sim_utils.CollisionPropertiesCfg(),physics_material=sim_utils.RigidBodyMaterialCfg(static_friction=1.,dynamic_friction=1.,restitution=0.),visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(.35,.35,.35))),init_state=AssetBaseCfg.InitialStateCfg(pos=(0.,0.,-.05)))
    light=AssetBaseCfg(prim_path='/World/light',spawn=sim_utils.DomeLightCfg(intensity=2000.))
    robot=B2Z1_CFG.replace(prim_path='{ENV_REGEX_NS}/Robot')
    contact=ContactSensorCfg(prim_path='{ENV_REGEX_NS}/Robot/.*',history_length=3,update_period=.02)
    camera=CameraCfg(prim_path='/World/Camera',height=480,width=640,update_period=.1,data_types=['rgb'],spawn=sim_utils.PinholeCameraCfg(focal_length=24.,horizontal_aperture=36.,clipping_range=(.03,50.)))

def main():
    out=Path(args.output);out.mkdir(parents=True,exist_ok=True)
    assets={}
    for name in ['004_sugar_box','006_mustard_bottle','011_banana']:
        stage=Usd.Stage.Open(str(ROOT/'assets/objects'/f'{name}.usd'))
        b=UsdGeom.BBoxCache(Usd.TimeCode.Default(),['default','render','proxy']).ComputeWorldBound(stage.GetPseudoRoot()).ComputeAlignedRange()
        assets[name]={'min':list(b.GetMin()),'max':list(b.GetMax()),'up_axis':str(UsdGeom.GetStageUpAxis(stage))}
    (out/'object_bounds.json').write_text(json.dumps(assets,indent=2))
    cfg=sim_utils.SimulationCfg(dt=.005,device='cpu',use_fabric=False,enable_scene_query_support=True,render_interval=4,physics_material=sim_utils.RigidBodyMaterialCfg(static_friction=1.,dynamic_friction=1.,restitution=0.))
    sim=sim_utils.SimulationContext(cfg)
    scene_cfg=ProbeScene(num_envs=1,env_spacing=20.)
    scene_cfg.robot.init_state.pos=(0.,0.,.6)
    scene=InteractiveScene(scene_cfg)
    sim.reset()
    robot=scene['robot'];ids,_=robot.find_joints(JOINT_NAMES,preserve_order=True)
    robot.write_root_state_to_sim(robot.data.default_root_state.clone())
    robot.write_joint_state_to_sim(robot.data.default_joint_pos.clone(),robot.data.default_joint_vel.clone())
    scene.reset()
    scene['camera'].set_world_poses_from_view(torch.tensor([[1.6,-2.,1.25]]),torch.tensor([[.25,0.,.35]]))
    control=LowLevelController()
    rows=[]
    for step in range(args.steps):
        proprio={'base_ang_vel':robot.data.root_ang_vel_b,'base_lin_vel':robot.data.root_lin_vel_b,'projected_gravity':robot.data.projected_gravity_b,'base_quat':robot.data.root_quat_w,'joint_pos':robot.data.joint_pos[:,ids],'joint_vel':robot.data.joint_vel[:,ids]}
        action=control.step(proprio,(0.,0.,0.),ARM_Q0)
        robot.set_joint_position_target(torch.from_numpy(action).reshape(1,19),joint_ids=ids)
        for _ in range(4):
            scene.write_data_to_sim();sim.step();scene.update(.005)
        if step%50==0:
            q=robot.data.joint_pos[:,ids][0].numpy()
            row={'step':step,'root':robot.data.root_pos_w[0].tolist(),'quat':robot.data.root_quat_w[0].tolist(),'arm':q[12:].tolist(),'gravity':robot.data.projected_gravity_b[0].tolist()}
            rows.append(row);print('[PROBE]',json.dumps(row),flush=True)
    rgb=scene['camera'].data.output['rgb'][0,:,:,:3].cpu().numpy()
    Image.fromarray(rgb).save(out/'standing_camera.png')
    # Run a real numerical Mink step in the same process as Isaac.
    mj=mujoco.MjModel.from_xml_string('<mujoco><worldbody><body><joint name="j" type="hinge"/><geom size=".1" mass="1"/><site name="ee" pos=".3 0 0"/></body></worldbody></mujoco>')
    conf=mink.Configuration(mj);task=mink.FrameTask('ee','site',position_cost=1.,orientation_cost=1.);target=conf.get_transform_frame_to_world('ee','site')
    task.set_target(target);vel=mink.solve_ik(conf,[task],dt=.02,solver='daqp')
    result={'rows':rows,'rgb_mean':float(rgb.mean()),'rgb_std':float(rgb.std()),'rgb_shape':list(rgb.shape),'mink_same_process_finite':bool(np.isfinite(vel).all()),'solvers':qpsolvers.available_solvers,'joint_names':robot.joint_names,'body_names':robot.body_names,'soft_limits':robot.data.soft_joint_pos_limits[0,ids].tolist()}
    (out/'probe.json').write_text(json.dumps(result,indent=2));print('[PROBE_DONE]',json.dumps(result),flush=True)
try:
    main()
finally:
    app.close(wait_for_replicator=False)
