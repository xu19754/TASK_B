from pathlib import Path
import json,sys,numpy as np
from scipy.spatial.transform import Rotation
ROOT=Path(__file__).resolve().parents[2];sys.path.insert(0,str(ROOT))
from experiments.mink_banana.ik_adapter import BananaIK
from experiments.mink_wholebody.kinematics import wxyz,quat_matrix
c=json.loads((ROOT/'experiments/mink_banana/configs/centered_04.json').read_text())
r=json.loads((ROOT/'runs/mink_multiclass/nine_01/mustard_yaw_p00/trial_01/controller.jsonl').read_text().splitlines()[299]);base=np.array(r['root_position']);quat=np.array(r['base_quat']);q=np.array(r['joint_actual']);ik=BananaIK(c['tcp_offset_link06']);ik.anchor(base,quat,q)
body=base.copy();body[2]=c['body_height'];bq=wxyz(Rotation.from_euler('y',c['body_pitch_rad']).as_matrix());ik.solve(base,quat,q,body,bq,max_iters=240);packed=ik.configuration.q.copy();q[:18]=packed[ik.q_indices]
reports=[];nominal=Rotation.from_euler('xyz',c['grasp_rpy']).as_matrix()
for stage,grip,up in [('pre',-1.0,.12),('approach',-1.0,0.),('close_wide',-.8,0.),('close_mid',-.5,0.),('close_narrow',-.3,0.),('lift',-.5,.15)]:
 q[-1]=grip;center,alpha,moving=ik.update_pinch_frame(q);rotation=nominal@Rotation.from_euler('y',alpha).as_matrix()
 target=np.array(c['grasp_world']);target[2]+=up;bg=body.copy()
 if stage=='lift':bg[2]=c['lift_body_height']
 candidate,d=ik.solve(packed[:3],packed[3:7],q,bg,bq,(target,wxyz(rotation)),max_iters=350);packed=ik.configuration.q.copy();q[:18]=candidate
 pad_delta=rotation@(moving-ik.fixed_pad)
 record={'stage':stage,'gripper':grip,'pinch_center':center.tolist(),'level_angle_deg':float(np.degrees(alpha)),'pad_height_difference_m':float(pad_delta[2]),**d};reports.append(record)
 print(stage,'q',grip,'center',center.tolist(),'tilt',np.degrees(alpha),'poserr',d['ee_position_error_m'],'orierr',d['ee_angle_error_deg'],'pad_dz',pad_delta[2],flush=True)
assert all(x['success'] and x['ee_position_error_m']<.005 and x['ee_angle_error_deg']<5 and abs(x['pad_height_difference_m'])<1e-8 for x in reports)
(ROOT/'runs/mink_banana/centered_04_preflight.json').write_text(json.dumps(reports,indent=2))
