"""Prepare exactly three known object yaw angles; no Isaac dynamics or inference."""
import sys,json,copy
from pathlib import Path
import numpy as np
from scipy.spatial.transform import Rotation
ROOT=Path(__file__).resolve().parents[2];sys.path.insert(0,str(ROOT))
from experiments.mink_wholebody.wholebody_ik import WholeBodyIK
from experiments.mink_wholebody.kinematics import wxyz,quat_matrix
base_config=json.loads((ROOT/'experiments/mink_wholebody/config.json').read_text())
profiles={
 'mustard':dict(asset='006_mustard_bottle',local=[0.,-.07,0.],tcp=[.21,0.,.02],height=.42,target_z=.167),
 'sugar':dict(asset='004_sugar_box',local=[0.,-.05,0.],tcp=[.21,0.,.012],height=.42,target_z=.1396),
 'banana':dict(asset='011_banana',local=[0.,0.,-.01795],tcp=[.235,0.,.012],height=.34,target_z=.021),
}
angles=[0,-45,45];cases=[];planning=[]
rows=(ROOT/'runs/mink_wholebody/final_5/trial_01/controller.jsonl').read_text().splitlines()
standing=json.loads(rows[299]);base=np.array(standing['root_position']);quat=np.array(standing['base_quat']);joints=np.array(standing['joint_actual'])
body_quat=wxyz(Rotation.from_euler('z',Rotation.from_matrix(quat_matrix(quat)).as_euler('xyz')[2]).as_matrix())
for yaw in angles:
 for name,profile in profiles.items():
  c=copy.deepcopy(base_config);angle=Rotation.from_euler('z',yaw,degrees=True)
  c['object_asset']=profile['asset']+'.usd';c['object_xy']=[.55,0.]
  c['object_quat_wxyz']=wxyz((angle*Rotation.from_euler('x',-90,degrees=True)).as_matrix()).tolist()
  local=np.array(profile['local']);object_rot=quat_matrix(c['object_quat_wxyz']);offset=object_rot@local
  c['grasp_world']=[float(.55+offset[0]),float(offset[1]),profile['target_z']]
  c['grasp_rpy']=(angle*Rotation.from_euler('xyz',base_config['grasp_rpy'])).as_euler('xyz').tolist()
  c['tcp_offset_link06']=profile['tcp'];c['body_height']=profile['height'];c['lift_body_height']=profile['height']+.06
  c['grasp_local_point']=profile['local'];c['title']='三物体三朝向已知目标抓取诊断'
  c['case_id']=f'{name}_yaw_{"m" if yaw<0 else "p"}{abs(yaw):02d}';c['object_class']=name;c['object_yaw_deg']=yaw
  c['distance_tested']=False;c['root_distance_m']=.55;c['repeat_count']=1
  c['notes']='Exactly one physical attempt per case; no navigation, no distance sweep, no retries. Geometry-defined known grasp point. Thresholds and dynamics unchanged.'
  path=Path(__file__).parent/'configs'/f'{c["case_id"]}.json';path.write_text(json.dumps(c,indent=2,ensure_ascii=False))
  cases.append({'case_id':c['case_id'],'object':name,'yaw_deg':yaw,'config':str(path.relative_to(ROOT))})
  ik=WholeBodyIK(c['tcp_offset_link06']);ik.anchor(base,quat,joints)
  body=base.copy();body[2]=c['body_height'];q=joints.copy()
  _,rep=ik.solve(base,quat,q,body,body_quat,max_iters=240);packed=ik.configuration.q.copy();q[:18]=packed[ik.q_indices]
  stages=[]
  for stage,up in [('pregrasp',.12),('grasp',0.),('lift',.15)]:
   goal=np.array(c['grasp_world']);goal[2]+=up;bg=body.copy()
   if stage=='lift':bg[2]=c['lift_body_height']
   candidate,report=ik.solve(packed[:3],packed[3:7],q,bg,body_quat,(goal,wxyz(Rotation.from_euler('xyz',c['grasp_rpy']).as_matrix())),max_iters=300)
   packed=ik.configuration.q.copy();q[:18]=candidate
   stages.append({'stage':stage,**report,'joint6':float(candidate[17])})
  record={'case_id':c['case_id'],'crouch':rep,'stages':stages};planning.append(record)
  print(c['case_id'],'body',c['body_height'],'goal',c['grasp_world'],'errors',[(s['stage'],round(s['ee_position_error_m'],6),round(s['ee_angle_error_deg'],3)) for s in stages],flush=True)
manifest={'title':'3 objects x 3 yaw angles, exactly one attempt each','angles_deg':angles,'root_xy':[.55,0.],'navigation':False,'distance_sweep':False,'case_count':9,'cases':cases}
(Path(__file__).parent/'suite.json').write_text(json.dumps(manifest,indent=2))
(ROOT/'runs/mink_multiclass/kinematic_preflight.json').write_text(json.dumps(planning,indent=2))
