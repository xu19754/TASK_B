"""Independent free-base PD feasibility reference, never an RL policy result.

Only joint targets change after the reset. The desired body pose is used to
compute leg IK targets; actual body motion follows foot contacts and dynamics.
An optional historical arm sweep is preflighted from the nominal folded arm,
including its complete approach. Unreachable/intersecting paths are rejected.
"""
from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
import time

import mujoco
import numpy as np
from scipy.spatial.transform import Rotation
import torch

from reports.basket_feasibility.geometry_scan import inverse_batch, leg_parameters
from .env import _hull_topology
from .metrics import TorchRearExitTracker
from .model import ROOT, build_model, prepare_targets
from .progress import update


def _smooth(u):
    u=np.clip(u,0.,1.)
    return 10*u**3-15*u**4+6*u**5


def _clearance(model,data,metadata,margin=.02):
    pairs=metadata["self_collision_pairs"]
    a,b=pairs[:,0],pairs[:,1]
    lower=np.linalg.norm(data.geom_xpos[a]-data.geom_xpos[b],axis=1)-model.geom_rbound[a]-model.geom_rbound[b]
    active=pairs[lower<margin]
    minimum=margin
    names=None
    for ga,gb in active:
        distance=mujoco.mj_geomDistance(model,data,int(ga),int(gb),margin,None)
        if distance<minimum:
            minimum=float(distance);names=[model.geom(int(ga)).name,model.geom(int(gb)).name]
    return minimum,names


def run(output,bank_path,pitch_deg=15.,forward=.1,height=.575,with_sweep=False,preview=True,record_index=None):
    model,meta=build_model()
    with np.load(bank_path) as bank:
        candidates=np.flatnonzero(np.all(bank["combination"]==[0,1,2],axis=1))
        index=int(candidates[0] if record_index is None else record_index)
        if not np.array_equal(bank["combination"][index],[0,1,2]):
            raise ValueError("Reference diagnostic currently requires mixed-three bank record")
        data=mujoco.MjData(model)
        data.qpos[:]=bank["qpos"][index];data.qvel[:]=bank["qvel"][index];data.ctrl[:]=bank["ctrl"][index]
        seed=int(bank["seed"][index])
    mujoco.mj_forward(model,data)
    output=Path(output);output.mkdir(parents=True,exist_ok=True)
    init_position=data.qpos[:3].copy()
    init_rotation=data.xmat[meta["base_id"]].reshape(3,3).copy()
    yaw=math.atan2(init_rotation[1,0],init_rotation[0,0])
    end_rotation=Rotation.from_euler('z',yaw).as_matrix()@Rotation.from_euler('y',-math.radians(pitch_deg)).as_matrix()
    delta_rotation=Rotation.from_matrix(init_rotation.T@end_rotation).as_rotvec()
    end_position=init_position.copy();end_position[0]+=forward;end_position[2]=height
    feet=data.xpos[meta["foot_body_ids"]].copy()
    parameters=leg_parameters()

    def leg_targets(fraction):
        weight=_smooth(fraction)
        rotation=init_rotation@Rotation.from_rotvec(weight*delta_rotation).as_matrix()
        position=init_position+weight*(end_position-init_position)
        q,valid,margin=inverse_batch(((feet-position)@rotation)[None],parameters)
        if not valid[0]:
            raise RuntimeError(f"Leg IK exceeds original limits at fraction {fraction:.4f}, margin {margin[0]}")
        return q[0]

    for fraction in np.linspace(0,1,201):
        leg_targets(fraction)
    approach_report=None
    arm_nodes=None
    if with_sweep:
        path=ROOT/'reports/basket_v3/edge_parallel/low_edge_candidate.json'
        candidate=json.loads(path.read_text())
        arm_nodes=np.asarray(candidate['q_nodes_rad'])
        planning=mujoco.MjData(model)
        planning.qpos[:]=data.qpos
        planning.qpos[meta['joint_qpos_adr'][:12]]=leg_targets(1.)
        minimum=.02;bad=None
        for fraction in np.linspace(0,1,401):
            q=meta['nominal_q'][12:]+_smooth(fraction)*(arm_nodes[0]-meta['nominal_q'][12:])
            if np.any(q<meta['joint_lower'][12:]) or np.any(q>meta['joint_upper'][12:]):
                bad=dict(reason='joint_limit',fraction=float(fraction));break
            planning.qpos[meta['joint_qpos_adr'][12:]]=q
            mujoco.mj_forward(model,planning)
            distance,pairs=_clearance(model,planning,meta)
            minimum=min(minimum,distance)
            if distance<.001:
                bad=dict(reason='forbidden_intersection_or_under_1mm_clearance',fraction=float(fraction),distance_m=distance,pairs=pairs);break
        approach_report=dict(valid=bad is None,minimum_clearance_m=minimum,rejection=bad,
                             scope='Straight smooth joint approach from folded arm; a rejected route requires replanning')
        (output/'arm_approach_preflight.json').write_text(json.dumps(approach_report,indent=2)+'\n')
        if bad is not None:
            arm_nodes=None

    hulls,masks,edges,edge_masks=_hull_topology(meta['cargo_vertices'])
    vertex_list=[hulls[i,masks[i]] for i in range(3)]
    tracker=TorchRearExitTracker()
    def vertices():
        rotation=data.xmat[meta['basket_id']].reshape(3,3)
        array=np.zeros((1,3,hulls.shape[1],3),dtype=np.float64)
        for i,(source,bid) in enumerate(zip(vertex_list,meta['cargo_body_ids'])):
            array[0,i,:len(source)]=(source@data.xmat[bid].reshape(3,3).T+data.xpos[bid]-data.xpos[meta['basket_id']])@rotation
        return torch.from_numpy(array)
    result=tracker.reset(vertices(),torch.from_numpy(masks),torch.from_numpy(edges),torch.from_numpy(edge_masks))
    if bool(result.invalid_initial.any()):
        raise RuntimeError('Reference reset rejected by unchanged tail-window metric')
    from .vision import DesktopPreview
    renderer=DesktopPreview(model,meta,output) if preview else None
    update('独立参考诊断：腿部 PD 后倾','不是学习策略；机身保持浮动，检查实际姿态与尾窗排出。',output)
    previous=meta['nominal_q'].copy()
    trace=[]
    minimum_clearance=.02;max_tilt=0.;max_pitch=0.;max_speed_ratio=0.;dangerous_count=0
    failure=None
    foot_ids=set(meta['foot_body_ids']);robot_ids=set(model.geom_bodyid[meta['collision_geom_ids']])
    floor=model.geom('floor').id
    jacp,jacr=np.zeros((3,model.nv)),np.zeros((3,model.nv))
    slip=np.zeros(4)
    tilt_duration,hold_duration,recover_duration=8.,4.,8.
    approach_duration=0.;sweep_duration=0.
    if arm_nodes is not None:
        approach_duration=max(20.,float(np.max(np.abs(arm_nodes[0]-meta['nominal_q'][12:]))/.12))
        sweep_duration=max(45.,float(candidate.get('duration_s',45.)))
    total=tilt_duration+hold_duration+approach_duration+sweep_duration+recover_duration+2.
    started=time.perf_counter()
    try:
        for step in range(round(total/model.opt.timestep)):
            elapsed=step*model.opt.timestep
            hold_end=tilt_duration+hold_duration+approach_duration+sweep_duration
            fraction=elapsed/tilt_duration if elapsed<tilt_duration else 1. if elapsed<hold_end else 1.-(elapsed-hold_end)/recover_duration
            if step%10==0:
                request=meta['nominal_q'].copy();request[:12]=leg_targets(np.clip(fraction,0,1))
                if arm_nodes is not None:
                    approach_start=tilt_duration+hold_duration
                    sweep_start=approach_start+approach_duration
                    if approach_start<=elapsed<sweep_start:
                        request[12:]=meta['nominal_q'][12:]+_smooth((elapsed-approach_start)/approach_duration)*(arm_nodes[0]-meta['nominal_q'][12:])
                    elif sweep_start<=elapsed<hold_end:
                        parameter=np.clip((elapsed-sweep_start)/sweep_duration,0,1)*(len(arm_nodes)-1)
                        lower=min(int(parameter),len(arm_nodes)-2);u=parameter-lower
                        request[12:]=(1-u)*arm_nodes[lower]+u*arm_nodes[lower+1]
                    elif elapsed>=hold_end:
                        request[12:]=arm_nodes[-1]+_smooth((elapsed-hold_end)/recover_duration)*(meta['nominal_q'][12:]-arm_nodes[-1])
                previous=prepare_targets(meta,request,previous,data.qpos[meta['joint_qpos_adr']],.02)
                data.ctrl[:]=previous
            mujoco.mj_step(model,data)
            mujoco.mj_forward(model,data)
            rotation=data.xmat[meta['base_id']].reshape(3,3)
            pitch=math.degrees(math.asin(np.clip(rotation[2,0],-1,1)))
            tilt=math.degrees(math.acos(np.clip(rotation[2,2],-1,1)))
            max_tilt=max(max_tilt,tilt);max_pitch=max(max_pitch,pitch)
            speed_ratio=float(np.max(np.abs(data.qvel[meta['joint_dof_adr']])/meta['velocity']))
            max_speed_ratio=max(max_speed_ratio,speed_ratio)
            clearance,pair=_clearance(model,data,meta)
            minimum_clearance=min(minimum_clearance,clearance)
            for contact in data.contact:
                a,b=contact.geom
                bid=int(model.geom_bodyid[b] if a==floor else model.geom_bodyid[a])
                if (a==floor or b==floor) and bid in robot_ids-foot_ids:
                    dangerous_count+=1
                if (a==floor or b==floor) and bid in foot_ids:
                    index=int(np.flatnonzero(meta['foot_body_ids']==bid)[0])
                    mujoco.mj_jac(model,data,jacp,jacr,contact.pos,bid)
                    slip[index]+=float(np.linalg.norm((jacp@data.qvel)[:2]))*model.opt.timestep
            if clearance<-.0005:
                failure=dict(reason='forbidden_intersection',clearance_m=clearance,pair=pair)
            elif data.qpos[2]<.28 or tilt>55 or dangerous_count:
                failure=dict(reason='unstable_or_dangerous_ground_contact')
            elif speed_ratio>1.15:
                failure=dict(reason='actual_joint_overspeed',ratio=speed_ratio)
            stable=tilt<10 and np.linalg.norm(data.qvel[:6])<.2 and np.max(np.abs(data.qpos[meta['joint_qpos_adr']][12:]-meta['nominal_q'][12:]))<.2
            result=tracker.update(vertices(),model.opt.timestep,stable=torch.tensor([stable]),valid_dynamics=torch.tensor([failure is None]))
            if step%10==0:
                trace.append(dict(time_s=float(data.time),root_qpos=data.qpos[:7].tolist(),joint_pos=data.qpos[meta['joint_qpos_adr']].tolist(),targets=previous.tolist(),pitch_deg=pitch,clearance_m=clearance,exited=int(result.exited.sum()),wrong=int(result.wrong.sum())))
            if renderer is not None and step%50==0:
                renderer.update_cpu(data,metrics=dict(reference_only=True,pitch_deg=round(pitch,2),exited=int(result.exited.sum()),wrong=int(result.wrong.sum())))
                remaining=elapsed-(time.perf_counter()-started)
                if remaining>0:time.sleep(min(remaining,.1))
            if failure is not None:break
        report=dict(engine=f'MuJoCo {mujoco.__version__}',asset='b2z1_0815_with_basket_v3.urdf',reference_only=True,rl_policy=False,
                    reset_seed=seed,reset_bank=str(bank_path),control='Free floating base; IK-derived leg PD targets; actual dynamics; no root writes after reset',
                    desired_nose_up_deg=pitch_deg,desired_translation_x_m=forward,desired_base_height_m=height,
                    measured_max_nose_up_deg=max_pitch,measured_max_tilt_deg=max_tilt,actual_max_joint_speed_ratio=max_speed_ratio,
                    minimum_forbidden_clearance_m=minimum_clearance,foot_contact_tangential_path_m=slip.tolist(),dangerous_contact_samples=dangerous_count,
                    elapsed_simulation_s=float(data.time),correctly_exited=int(result.exited.sum()),wrong_exit=int(result.wrong.sum()),
                    success=bool(result.success[0]),failure=failure,arm_sweep_requested=with_sweep,arm_sweep_executed=arm_nodes is not None,
                    arm_approach_preflight=approach_report,trace=trace)
        (output/'report.json').write_text(json.dumps(report,indent=2)+'\n')
        update('独立参考诊断完成',f"实际最大后倾 {max_pitch:.2f}°；正确排出 {int(result.exited.sum())}/3；此结果不是 RL 成绩。",output)
        return report
    finally:
        if renderer is not None:renderer.close()


if __name__=='__main__':
    parser=argparse.ArgumentParser()
    parser.add_argument('--output',type=Path,default=ROOT/'runs/basket_unload/reference_experiment')
    parser.add_argument('--bank',type=Path,default=ROOT/'runs/basket_unload/reset_bank_collected.npz')
    parser.add_argument('--pitch-deg',type=float,default=15.)
    parser.add_argument('--forward',type=float,default=.1)
    parser.add_argument('--height',type=float,default=.575)
    parser.add_argument('--with-sweep',action='store_true')
    parser.add_argument('--no-preview',action='store_true')
    args=parser.parse_args()
    report=run(args.output,args.bank,args.pitch_deg,args.forward,args.height,args.with_sweep,not args.no_preview)
    print(json.dumps({k:v for k,v in report.items() if k!='trace'},indent=2))
