"""One deterministic kinematic preflight from an existing grasp trace; no physics."""
import argparse, json
from pathlib import Path
import numpy as np
from scipy.spatial.transform import Rotation
from experiments.mink_wholebody.wholebody_ik import WholeBodyIK
from experiments.mink_wholebody.kinematics import quat_matrix, transform, wxyz
from taskb_student.constants import ROOT

parser=argparse.ArgumentParser()
parser.add_argument('--trial',type=Path,required=True)
parser.add_argument('--output',type=Path,required=True)
args=parser.parse_args()
if args.output.exists(): raise SystemExit('Refusing to overwrite output')
trace=[json.loads(x) for x in (args.trial/'controller.jsonl').read_text().splitlines() if x.strip()]
last=trace[-1]
if last['state']!='DONE': raise SystemExit('Prior grasp did not reach DONE')
config=json.loads((args.trial.parent/'config.json').read_text())
meta=json.loads((args.trial.parent/'scene_metadata.json').read_text())
basket=json.loads((ROOT/'assets/b2z1/basket_v3/basket_properties.json').read_text())
q=np.array(last['joint_actual'],float); base=np.array(last['root_position'],float); quat=np.array(last['base_quat'],float)
rot=quat_matrix(quat)
ik=WholeBodyIK(config['tcp_offset_link06'], np.array(meta['effective_joint_limits']))
ik.anchor(base,quat,q)
tcp=ik.fk.frames(q)['link06']@transform(config['tcp_offset_link06'])
start=base+rot@tcp[:3,3]; orientation=rot@tcp[:3,:3]
# Basket-local upper bound plus an explicit 15 cm TCP clearance. This is not
# a guarantee that the carried object, fingers or links clear the basket.
top_local=basket['fixed_mount_xyz_m'][2]+basket['bounds_m'][1][2]
top_world=(base+rot@np.array([0.,0.,top_local]))[2]
z=float(top_world+.15)
center=base+rot@np.array([basket['fixed_mount_xyz_m'][0],0.,0.])
front=base+rot@np.array([.46,0.,0.])
side=base+rot@np.array([0.,.30,0.])
waypoints=[('retract_front',np.array([front[0],front[1],start[2]]),0.)]+[(f'raise_{i}',np.array([front[0],front[1],start[2]+(z-start[2])*i/6]),0.) for i in range(1,7)]+[('verticalize',np.array([front[0],front[1],z]),0.),('turn_side',np.array([side[0],side[1],z]),90.),('above_basket',np.array([center[0],center[1],z]),180.)]
report={'source_trial':str(args.trial.resolve()),'basket_top_estimate_world_m':float(top_world),'tcp_clearance_m':.15,'waypoints':[],'pass':False,'collision_checked':False,'object_clearance_checked':False,'orientation_policy':'position-only while raising; vertical tool axis checked afterward, roll unconstrained'}
for name,pos,yaw in waypoints:
 if name=='retract_front' or name.startswith('raise_'): ik.ee_task.set_orientation_cost(0.)
 else: ik.ee_task.set_orientation_cost([0.,5.,5.])
 if name.startswith('raise_'):
  fraction=int(name.split('_')[1])/6
  target_rot=retract_orientation@Rotation.from_rotvec(fraction*vertical_delta).as_matrix()
 else:
  target_rot=Rotation.from_euler('z',yaw,degrees=True).as_matrix()@orientation
 candidate,info=ik.solve(base,quat,q,base,quat,(pos,wxyz(target_rot)),max_iters=300)
 packed=ik.configuration.q.copy()
 actual_rotation=quat_matrix(packed[3:7])@ik.fk.frames(np.r_[candidate,q[18]])['link06'][:3,:3]
 axis_error=float(np.degrees(np.arccos(np.clip(actual_rotation[:,0]@np.array([0.,0.,1.]),-1.,1.))))
 passed=bool(info['success'] and info['ee_position_error_m']<=.005 and (name=='retract_front' or name.startswith('raise_') or axis_error<=5.))
 report['waypoints'].append({'name':name,'target_world_m':pos.tolist(),'yaw_change_deg':yaw,'vertical_axis_error_deg':axis_error,'result':info,'pass':passed})
 args.output.parent.mkdir(parents=True,exist_ok=True)
 args.output.write_text(json.dumps(report,indent=2)+'\n')
 print(name,json.dumps(report['waypoints'][-1]),flush=True)
 if not passed: raise SystemExit(1)
 base=packed[:3].copy();quat=packed[3:7].copy();q[:18]=candidate
 if name=='retract_front':
  actual=quat_matrix(quat)@ik.fk.frames(q)['link06'][:3,:3]
  align,_=Rotation.align_vectors([[0.,0.,1.]],[actual[:,0]])
  orientation=align.as_matrix()@actual
  retract_orientation=actual
  vertical_delta=Rotation.from_matrix(actual.T@orientation).as_rotvec()
  report['vertical_tool_axis'] = 'link06 local +X points up in world'
  report['vertical_target_rotation'] = orientation.tolist()
report['pass']=True
args.output.write_text(json.dumps(report,indent=2)+'\n')
