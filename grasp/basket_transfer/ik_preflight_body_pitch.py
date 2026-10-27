"""One sequential upright-object carry candidate from saved grasp; no physics or collision check."""
import argparse,json
from pathlib import Path
import numpy as np
from scipy.spatial.transform import Rotation
from experiments.mink_wholebody.wholebody_ik import WholeBodyIK
from experiments.mink_wholebody.kinematics import quat_matrix,transform,wxyz
parser=argparse.ArgumentParser();parser.add_argument('--trial',type=Path,required=True);parser.add_argument('--output',type=Path,required=True);args=parser.parse_args()
if args.output.exists():raise SystemExit('Output already exists')
trial=args.trial; last=json.loads((trial/'controller.jsonl').read_text().splitlines()[-1]); config=json.loads((trial.parent/'config.json').read_text());meta=json.loads((trial.parent/'scene_metadata.json').read_text())
q=np.array(last['joint_actual']);base=np.array(last['root_position']);quat=np.array(last['base_quat']);rot=quat_matrix(quat);origin=base.copy();limits=np.array(meta['effective_joint_limits'])
ik=WholeBodyIK(config['tcp_offset_link06'],limits);ik.anchor(base,quat,q);ik.ee_task.set_orientation_cost(5.);body_final=Rotation.from_euler('y',-.15).as_matrix()@rot;body_goal_quat=wxyz(body_final)
tcp=ik.fk.frames(q)['link06']@transform(config['tcp_offset_link06']);start=base+rot@tcp[:3,3];ori=rot@tcp[:3,:3]
front=origin+rot@np.array([.46,0.,0.]);upper=origin+rot@np.array([.60,0.,0.]);side=origin+rot@np.array([0.,.30,0.]);basket=origin+body_final@np.array([-.20,.06,0.]);z=float(max((origin+body_final@np.array([x,y,z0]))[2] for x in (-.525,-.075) for y in (-.125,.125) for z0 in (.13,.29))+.15)
p0=start.copy();p1=np.array([front[0],front[1],start[2]]);p2=np.array([upper[0],upper[1],z]);p3=np.array([side[0],side[1],z]);p4=np.array([basket[0],basket[1],z]);waypoints=[]
def append_segment(label,a,b,count,y0,y1):
 for i in range(1,count+1):
  u=i/count;waypoints.append((label+str(i),a+(b-a)*u,y0+(y1-y0)*u))
append_segment('retract_',p0,p1,4,0.,0.);append_segment('raise_',p1,p2,6,0.,0.);append_segment('side_',p2,p3,4,0.,90.);append_segment('basket_',p3,p4,4,90.,135.)
report={'source_trial':str(trial),'orientation':'preserve grasp tilt; rotate about world vertical','collision_checked':False,'object_clearance_checked':False,'body_pitch_rad':-.15,'waypoints':[],'pass':False}
args.output.parent.mkdir(parents=True,exist_ok=True)
for name,pos,yaw in waypoints:
 target=Rotation.from_euler('z',yaw,degrees=True).as_matrix()@ori
 candidate,info=ik.solve(base,quat,q,origin,body_goal_quat,(pos,wxyz(target)),max_iters=300)
 packed=ik.configuration.q.copy(); margins=np.minimum(candidate-limits[:,0],limits[:,1]-candidate)
 passed=bool(info['success'] and info['ee_position_error_m']<=.005 and info['ee_angle_error_deg']<=5. and float(margins.min())>=0.)
 row={'name':name,'target':pos.tolist(),'yaw_deg':yaw,'position_error_m':info['ee_position_error_m'],'orientation_error_deg':info['ee_angle_error_deg'],'minimum_joint_margin_rad':float(margins.min()),'limiting_joint_index':int(margins.argmin()),'q':candidate.tolist(),'pass':passed,'solver':info};report['waypoints'].append(row)
 args.output.write_text(json.dumps(report,indent=2)+'\n');print(name,'pass',passed,'position_m',row['position_error_m'],'orientation_deg',row['orientation_error_deg'],'margin_rad',row['minimum_joint_margin_rad'],flush=True)
 if not passed:raise SystemExit(1)
 base=packed[:3].copy();quat=packed[3:7].copy();q[:18]=candidate
report['pass']=True;args.output.write_text(json.dumps(report,indent=2)+'\n')
