"""Known-target single-object Isaac diagnostic; MuJoCo/Mink only for IK."""
import argparse,sys,json,time,hashlib,copy,csv
from pathlib import Path
ROOT=Path(__file__).resolve().parents[2];sys.path.insert(0,str(ROOT))
parser=argparse.ArgumentParser(description='已知目标位姿的抓取诊断')
parser.add_argument('--mode',choices=['ik-check','env-check','posture','reach','grasp'],default='grasp')
parser.add_argument('--config',type=Path,default=ROOT/'experiments/mink_wholebody/config.json')
parser.add_argument('--output',type=Path,default=ROOT/'runs/mink_grasp/run')
parser.add_argument('--record-video',action='store_true')
parser.add_argument('--trials',type=int,default=1)
parser.add_argument('--history-warmup',type=int,default=100,help='Initialize recurrent history with repeated measured nominal proprioception before physics.')
from isaaclab.app import AppLauncher
AppLauncher.add_app_launcher_args(parser);parser.set_defaults(device='cpu')
args=parser.parse_args()
if args.device!='cpu': parser.error('CPU PhysX required')
if args.record_video and not args.enable_cameras: parser.error('--record-video requires --enable_cameras')
if args.trials<1 or args.history_warmup<0: parser.error('Invalid count')
out=args.output
if out.exists(): out=out.with_name(out.name+'_'+time.strftime('%Y%m%d_%H%M%S'))
out.mkdir(parents=True)
config=json.loads(args.config.read_text())
(out/'config.json').write_text(json.dumps(config,indent=2,ensure_ascii=False))
(out/'invocation.json').write_text(json.dumps({'argv':sys.argv,'output':str(out),'history_warmup':args.history_warmup},indent=2))
print('[OUTPUT]',out,flush=True)
manifest={str(p.relative_to(ROOT)):hashlib.sha256(p.read_bytes()).hexdigest() for p in (ROOT/'experiments/mink_wholebody').glob('*.py')}
(out/'code_manifest.json').write_text(json.dumps(manifest,indent=2))

def ik_check():
    import numpy as np
    from scipy.spatial.transform import Rotation
    from experiments.mink_wholebody.wholebody_ik import WholeBodyIK
    from experiments.mink_wholebody.kinematics import wxyz
    from taskb_student.constants import LEG_Q0,ARM_Q0
    ik=WholeBodyIK(config['tcp_offset_link06']);fk=ik.check_model()
    q=np.array(LEG_Q0+ARM_Q0);base=np.array([0.,0.,.539]);quat=np.array([1.,0,0,0])
    ik.anchor(base,quat,q);goal=np.array([0.,0.,config['body_height']])
    sq,r1=ik.solve(base,quat,q,goal,quat,max_iters=200)
    virtual=ik.configuration.q.copy()
    q[:18]=sq
    ee=(np.array(config['grasp_world']),wxyz(Rotation.from_euler('xyz',config['grasp_rpy']).as_matrix()))
    arm,r2=ik.solve(virtual[:3],virtual[3:7],q,goal,quat,ee,max_iters=300)
    result={'fk':fk,'crouch':r1,'reach':r2,'nq':ik.model.nq,'nv':ik.model.nv,'success':r2['success'] and r2['ee_position_error_m']<.005}
    (out/'ik_check.json').write_text(json.dumps(result,indent=2));print('[WHOLEBODY_IK_CHECK]',json.dumps(result),flush=True)
    return 0 if result['success'] else 1

def run():
    import numpy as np,torch,carb
    from PIL import Image,ImageDraw
    import imageio.v2 as imageio
    from pxr import UsdGeom,Usd,Gf
    import isaaclab.sim as sim_utils
    from experiments.mink_wholebody.scene import build_scene
    from experiments.mink_wholebody.wholebody_ik import WholeBodyIK
    from experiments.mink_wholebody.kinematics import quat_matrix,wxyz
    from experiments.mink_wholebody.controller import GraspController
    from experiments.mink_wholebody.evaluator import Evaluator
    from experiments.mink_wholebody.contact_recorder import ContactRecorder
    from taskb_student.low_level import LowLevelController
    from taskb_student.constants import JOINT_NAMES,ARM_Q0,CONTROL_DT,JOINT_LOWER,JOINT_UPPER
    settings=carb.settings.get_settings()
    previous={k:settings.get(k) for k in ['/physics/updateToUsd','/physics/updateVelocitiesToUsd','/physics/fabricEnabled']}
    settings.set_bool('/physics/updateToUsd',True)
    settings.set_bool('/physics/updateVelocitiesToUsd',True)
    sim_cfg=sim_utils.SimulationCfg(dt=.005,device='cpu',use_fabric=False,enable_scene_query_support=True,render_interval=20,physics_material=sim_utils.RigidBodyMaterialCfg(static_friction=1.,dynamic_friction=1.,restitution=0.))
    sim=sim_utils.SimulationContext(sim_cfg)
    scene,vertices,initial_z=build_scene(config,args.enable_cameras)
    robot=scene['robot'];obj=scene['object']
    sim.reset()
    startup_guard.cancel()
    print('[READY] Isaac scene reset completed',flush=True)
    ids,_=robot.find_joints(JOINT_NAMES,preserve_order=True)
    wrist_id=robot.find_bodies('link06')[0][0]
    body_names=scene['contacts'].body_names
    bad_ids=[i for i,n in enumerate(body_names) if n=='base' or n.endswith('_hip') or n.endswith('_thigh')]
    ik=WholeBodyIK(config['tcp_offset_link06'],robot.data.soft_joint_pos_limits[0,ids][:18].numpy())
    meta={'contact_body_names':body_names,'illegal_contact_bodies':[body_names[i] for i in bad_ids],'previous_physics_settings':previous,'physics_settings':{k:settings.get(k) for k in previous},'object_initial_z':initial_z,'object_vertex_bounds':[vertices.min(axis=0).tolist(),vertices.max(axis=0).tolist()],'fk_comparison':ik.check_model(),'effective_joint_limits':ik.limits_array.tolist(),'object_rigid_paths':obj.root_physx_view.prim_paths,'jaw_filters':scene['jaw_fixed'].cfg.filter_prim_paths_expr,'mass_kg':.5,'fix_base':robot.is_fixed_base}
    slices=[]
    for center in np.linspace(vertices[:,1].min(),vertices[:,1].max(),15):
        v=vertices[abs(vertices[:,1]-center)<.007]
        if len(v): slices.append({'asset_y':float(center),'count':len(v),'x_width':float(np.ptp(v[:,0])),'z_width':float(np.ptp(v[:,2]))})
    meta['object_cross_sections']=slices
    (out/'scene_metadata.json').write_text(json.dumps(meta,indent=2))
    summaries=[]
    for trial in range(args.trials):
        td=out/f'trial_{trial+1:02d}';td.mkdir()
        robot.write_root_state_to_sim(robot.data.default_root_state.clone())
        robot.write_joint_state_to_sim(robot.data.default_joint_pos.clone(),robot.data.default_joint_vel.clone())
        obj.write_root_state_to_sim(obj.data.default_root_state.clone())
        scene.reset()
        robot.set_joint_effort_target(torch.zeros((1,19)),joint_ids=ids)
        low=LowLevelController()
        controller=GraspController(config,ik,args.mode)
        evaluation=Evaluator(config,vertices)
        native_contacts=ContactRecorder()
        if args.enable_cameras:
            scene['camera'].set_world_poses_from_view(torch.tensor([[1.25,-1.6,1.05]]),torch.tensor([[.42,0.,.33]]))
        def proprio():
            return {'base_ang_vel':robot.data.root_ang_vel_b,'base_lin_vel':robot.data.root_lin_vel_b,'projected_gravity':robot.data.projected_gravity_b,'base_quat':robot.data.root_quat_w,'joint_pos':robot.data.joint_pos[:,ids],'joint_vel':robot.data.joint_vel[:,ids]}
        measured=proprio()
        for _ in range(args.history_warmup): low.step(measured,(0.,0.,0.),ARM_Q0)
        writer=imageio.get_writer(str(td/'continuous.mp4'),fps=10,codec='libx264',quality=8,macro_block_size=16) if args.record_video else None
        started=time.monotonic();frames=0;last_state=None;frame_diffs=[];previous_frame=None
        trace=(td/'controller.jsonl').open('w');etrace=(td/'evaluation.jsonl').open('w')
        try:
            for step in range(int(config['max_sim_s']/CONTROL_DT)):
                if time.monotonic()-started>config['max_wall_s']:
                    controller.fail('wall_clock_limit');break
                p=proprio()
                contact=scene['contacts'].data.net_forces_w_history[0,:,bad_ids,:]
                illegal=bool((torch.linalg.vector_norm(contact,dim=-1)>1.).any()) if step>300 else False
                cp={k:v.detach().cpu().numpy().copy() for k,v in p.items()}
                cp['root_position']=robot.data.root_pos_w[0].numpy().copy();cp['illegal_contact']=illegal
                target=controller.step(cp) if args.mode!='env-check' else np.r_[np.zeros(12),ARM_Q0]
                action=low.step(p,(0.,0.,0.),target[12:])
                if controller.active:
                    blend=min(controller.active_time/config['handover_s'],1.)
                    action[:12]=(1.-blend)*action[:12]+blend*target[:12]
                    own_q=cp['joint_pos'].reshape(19)
                    own_rot=quat_matrix(cp['base_quat'].reshape(4))
                    body_goal=np.array(controller.last_ik.get('body_goal',controller.baseline),float)
                    if controller.state=='SQUAT':
                        u=min(controller.age/config['squat_duration_s'],1.);u=u*u*(3.-2.*u)
                        body_goal[2]=(1.-u)*controller.start_height+u*config['body_height']
                    elif controller.state not in ['LIFT','HOLD']: body_goal[2]=config['body_height']
                    ff,support=ik.support_torques(cp['root_position'],cp['base_quat'].reshape(4),own_q,body_goal,controller.body_quat,own_rot@cp['base_lin_vel'].reshape(3),own_rot@cp['base_ang_vel'].reshape(3))
                    gravity_all=robot.root_physx_view.get_gravity_compensation_forces()
                    gravity_offset=0 if robot.is_fixed_base else 6
                    gravity=gravity_all[0,np.array(ids[:12])+gravity_offset].numpy()
                    action[:12]+=(ff+gravity)*blend/360.
                    action[:12]=np.clip(action[:12],JOINT_LOWER[:12],JOINT_UPPER[:12])
                    gravity_arm=gravity_all[0,np.array(ids[12:18])+gravity_offset].numpy()
                    action[12:18]+=gravity_arm/np.array([512.,768.,768.,512.,384.,256.])
                    action[12:18]=np.clip(action[12:18],JOINT_LOWER[12:18],JOINT_UPPER[12:18])
                    controller.last_ik.update(support)
                    controller.last_ik.update(joint_gravity_ff=gravity.tolist(),joint_total_ff=((ff+gravity)*blend).tolist())
                if not np.isfinite(action).all(): raise RuntimeError('nonfinite action')
                robot.set_joint_position_target(torch.tensor(action).reshape(1,19),joint_ids=ids)
                native_contacts.clear()
                for _ in range(4):
                    scene.write_data_to_sim();sim.step(render=False);scene.update(.005)
                t=(step+1)*CONTROL_DT
                actual=robot.data.joint_pos[0,ids].numpy()
                if step==299:
                    foot_ids=robot.find_bodies(ik.feet,preserve_order=True)[0]
                    f=ik.fk.frames(actual);rmat=quat_matrix(robot.data.root_quat_w[0].numpy());rp=robot.data.root_pos_w[0].numpy()
                    native_gravity=robot.root_physx_view.get_gravity_compensation_forces()
                    debug={'feet_isaac':robot.data.body_pos_w[0,foot_ids].tolist(),'feet_urdf':[(rp+rmat@f[n][:3,3]).tolist() for n in ik.feet],'gravity_shape':list(native_gravity.shape),'gravity_raw':native_gravity[0].tolist(),'joint_ids_ordered':ids}
                    (td/'model_alignment.json').write_text(json.dumps(debug,indent=2));print('[MODEL_ALIGNMENT]',json.dumps(debug),flush=True)
                root=robot.data.root_pos_w[0].numpy()
                grav=robot.data.projected_gravity_b[0].numpy()
                tilt=float(np.rad2deg(np.arccos(np.clip(-grav[2],-1,1))))
                filtered_jaw_forces=[]
                net_jaw_forces=[]
                for name in ['jaw_fixed','jaw_moving']:
                    mat=scene[name].data.force_matrix_w
                    filtered_jaw_forces.append(float(torch.linalg.vector_norm(mat,dim=-1).sum()) if mat is not None else 0.)
                    net_jaw_forces.append(float(torch.linalg.vector_norm(scene[name].data.net_forces_w,dim=-1).sum()))
                jaw_forces=native_contacts.forces(CONTROL_DT)
                ev=evaluation.observe(t,controller.state,obj.data.root_pos_w[0].numpy(),obj.data.root_quat_w[0].numpy(),robot.data.body_pos_w[0,wrist_id].numpy(),robot.data.body_quat_w[0,wrist_id].numpy(),actual[18],root,tilt,illegal,jaw_forces)
                ev.update(filtered_jaw_forces=filtered_jaw_forces,net_jaw_forces=net_jaw_forces,native_contacts=native_contacts.events,native_contact_errors=native_contacts.errors)
                etrace.write(json.dumps(ev)+'\n')
                row={'sim_time':t,'state':controller.state,'root_position':root.tolist(),'base_quat':robot.data.root_quat_w[0].tolist(),'tilt_deg':tilt,'illegal_contact':illegal,'bad_contact_force_max':float(torch.linalg.vector_norm(contact,dim=-1).max()),'joint_actual':actual.tolist(),'joint_target':action.tolist(),'arm_actual':actual[12:].tolist(),'arm_target':target[12:].tolist(),'wholebody_active':controller.active,'ik':controller.last_ik}
                trace.write(json.dumps(row)+'\n')
                if args.enable_cameras and step%5==0:
                    sim.render()
                    rgb=scene['camera'].data.output['rgb'][0,:,:,:3].cpu().numpy().copy()
                    if previous_frame is not None: frame_diffs.append(float(np.abs(rgb.astype(float)-previous_frame).mean()))
                    previous_frame=rgb.astype(float)
                    image=Image.fromarray(rgb)
                    if frames%20==0: image.save(td/f'frame_{frames:04d}.png')
                    draw=ImageDraw.Draw(image);draw.rectangle((0,0,640,40),fill=(0,0,0))
                    draw.text((8,5),f't={t:.2f}s  {controller.state}  rise={ev["rise_m"]:.3f}m  hold={ev["valid_hold_s"]:.2f}s',fill=(255,255,255))
                    if writer is not None: writer.append_data(np.asarray(image))
                    frames+=1
                if controller.state!=last_state or step%100==0:
                    print('[STEP]',t,controller.state,'root',root.tolist(),'arm',actual[12:].tolist(),'ik',controller.last_ik,flush=True);last_state=controller.state
                if args.mode=='env-check' and t>=6.:
                    controller.change('DONE','environment_probe');break
                if controller.state in ['DONE','FAIL']: break
            else: controller.fail('simulation_time_limit')
        finally:
            trace.close();etrace.close();native_contacts.close()
            if writer is not None: writer.close()
        result=evaluation.result()
        result.update(trial=trial+1,mode=args.mode,state=controller.state,reason=controller.reason,history=controller.history,sim_time=t,wall_time=time.monotonic()-started,frames=frames,max_frame_difference=max(frame_diffs,default=0.),config_sha256=hashlib.sha256(args.config.read_bytes()).hexdigest(),last_ik=controller.last_ik)
        if args.mode in ['env-check','posture']:
            result['environment_pass']=bool(frames>5 and max(frame_diffs,default=0.)>.05 and tilt<15. and (args.mode!='posture' or (controller.state=='DONE' and abs(root[2]-config['body_height'])<.04)))
        actual_wrist=robot.data.body_pos_w[0,wrist_id].numpy()
        fk=ik.fk.frames(actual)['link06']
        fk_world=quat_matrix(robot.data.root_quat_w[0].numpy())@fk[:3,3]+root
        result['isaac_urdf_wrist_fk_error_m']=float(np.linalg.norm(actual_wrist-fk_world))
        stage=sim.stage
        usdx=UsdGeom.XformCache(Usd.TimeCode.Default()).GetLocalToWorldTransform(stage.GetPrimAtPath('/World/envs/env_0/Robot/base'))
        result['usd_base_translation']=list(usdx.ExtractTranslation())
        result['physx_base_translation']=root.tolist()
        result['usd_physx_translation_error_m']=float(np.linalg.norm(np.array(result['usd_base_translation'])-root))
        (td/'result.json').write_text(json.dumps(result,indent=2,ensure_ascii=False))
        summaries.append(result);print('[TRIAL_RESULT]',json.dumps(result,ensure_ascii=False),flush=True)
    summary={'title':config['title'],'mode':args.mode,'trials':len(summaries),'success_count':sum(bool(x['success']) for x in summaries),'thresholds':config['thresholds'],'results':summaries,'output':str(out)}
    (out/'summary.json').write_text(json.dumps(summary,indent=2,ensure_ascii=False))
    with (out/'trials.csv').open('w',newline='') as f:
        keys=['trial','mode','state','success','reason','sim_time','wall_time','max_rise_m','max_valid_hold_s','frames','isaac_urdf_wrist_fk_error_m','usd_physx_translation_error_m']
        writer=csv.DictWriter(f,keys,extrasaction='ignore');writer.writeheader();writer.writerows(summaries)
    print('[COMPLETE]',out,flush=True)
    sim.stop()

if args.mode=='ik-check':
    raise SystemExit(ik_check())
import threading,os
def startup_timeout():
    print('[STARTUP_TIMEOUT] no trial result claimed',flush=True)
    os._exit(124)
startup_guard=threading.Timer(180.,startup_timeout);startup_guard.daemon=True;startup_guard.start()
app=AppLauncher(args).app
try:
    run()
finally:
    import threading,os
    def shutdown_timeout():
        print('[SHUTDOWN_TIMEOUT] results were flushed; native shutdown exceeded 15s',flush=True)
        os._exit(124)
    guard=threading.Timer(15.,shutdown_timeout);guard.daemon=True;guard.start()
    app.close(wait_for_replicator=False)
    guard.cancel()
