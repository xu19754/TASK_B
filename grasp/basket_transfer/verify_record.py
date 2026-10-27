"""Verify recorded grasp and basket retention; does not run simulation or collision audit."""
import argparse,hashlib,json,math
import numpy as np
from experiments.mink_wholebody.kinematics import URDFKinematics,transform
from pathlib import Path
p=argparse.ArgumentParser();p.add_argument('trial',type=Path);p.add_argument('--output',type=Path,required=True);a=p.parse_args()
if a.output.exists():raise SystemExit('Output exists')
trial=a.trial;root=trial.parent
read=lambda p:[json.loads(line) for line in p.read_text().splitlines() if line.strip()]
e=read(trial/'evaluation.jsonl');b=read(trial/'basket_evaluation.jsonl');c=read(trial/'controller.jsonl');result=json.loads((trial/'result.json').read_text());config=json.loads((root/'config.json').read_text())
if not (len(e)==len(b)==len(c)>0):raise SystemExit('Trace length mismatch')
carry=[x for x in e if x['state']=='CARRY_PLAN'];hold=[x for x in e if x['state']=='HOLD'];after=[x for x in b if x['state'] in ('VERIFY_BASKET','DONE')]
if not carry or not hold or not after:raise SystemExit('Missing required stage')
th=config['thresholds'];stable=0.;maxstable=0.
last_hold=next(x for x in reversed(c) if x['state']=='HOLD')
last_carry=next(x for x in reversed(c) if x['state']=='CARRY_PLAN')
fk=URDFKinematics()
def tcp_base(r):return (fk.frames(np.asarray(r['joint_actual']))['link06']@transform(config['tcp_offset_link06']))[:3,3]
v1,v2=tcp_base(last_hold),tcp_base(last_carry)
a1,a2=math.atan2(v1[1],v1[0]),math.atan2(v2[1],v2[0]);azimuth_deg=abs(math.degrees(math.atan2(math.sin(a2-a1),math.cos(a2-a1))))
turn_bottom=[b[i]['object_bounds_basket'][0][2] for i,r in enumerate(c) if r['state']=='CARRY_PLAN' and r['ik'].get('joint_waypoint_index',-1)>=10]
if not turn_bottom:raise SystemExit('Missing turn samples')
limits=np.array(json.loads((root/'scene_metadata.json').read_text())['effective_joint_limits'])
arm_margins=[float(np.min(np.minimum(np.array(r['joint_actual'][:18])-limits[:,0],limits[:,1]-np.array(r['joint_actual'][:18])))) for r in c if r['state'] in ('CARRY_PLAN','RELEASE','VERIFY_BASKET')]

for r in after:
 lo,hi=r['object_bounds_basket'];center=r['object_center_basket'];jaws=r['jaw_forces']
 ok=lo[0]>=-.225 and hi[0]<=.225 and lo[1]>=-.125 and hi[1]<=.125 and lo[2]>=-.01 and 0.<=center[2]<=.16 and r['speed_m_s']<.05 and max(jaws)<.25
 stable=stable+.02 if ok else 0.;maxstable=max(maxstable,stable)
contact_errors=sum(len(x.get('native_contact_errors',[])) for x in e)
checks={'done':result['state']=='DONE','grasp_hold':max(x['valid_hold_s'] for x in hold)>=th['hold_s'], 'carry_dual_contact':all(min(x['jaw_forces'])>.25 and x['in_jaws'] and x['rise_m']>=th['rise_m'] for x in carry), 'stable_in_basket':maxstable>=3., 'gripper_open':abs(c[-1]['joint_actual'][18]-config['gripper_open'])<.04,'no_illegal_contact':not any(x['illegal_contact'] for x in e if x['sim_time']>6.),'native_contact_errors_zero':contact_errors==0,'tilt_within_limit':max(x['tilt_deg'] for x in e)<=th['tilt_deg'],'drift_within_limit':max(x['drift_m'] for x in e)<=th['drift_m'],'video_present':(trial/'continuous.mp4').stat().st_size>0,'forward_to_rear_azimuth':azimuth_deg>=175.,'object_above_basket_rim_during_turn':min(turn_bottom)>=.16,'within_effective_joint_limits':min(arm_margins)>=0.}
summary={'checks':checks,'pass':all(checks.values()),'path_collision_audited':False,'task_policy_validated':False,'simulation':'Isaac PhysX, known target','max_valid_hold_s':result['max_valid_hold_s'],'basket_continuous_s':maxstable,'tcp_azimuth_change_deg':azimuth_deg,'min_object_clearance_above_rim_m':min(turn_bottom)-.16,'minimum_effective_joint_margin_rad':min(arm_margins),'carry_min_dual_force_N':min(min(x['jaw_forces']) for x in carry),'max_tilt_deg':max(x['tilt_deg'] for x in e),'max_drift_m':max(x['drift_m'] for x in e),'final_basket_center_m':after[-1]['object_center_basket'],'video_sha256':hashlib.sha256((trial/'continuous.mp4').read_bytes()).hexdigest()}
a.output.parent.mkdir(parents=True,exist_ok=True);a.output.write_text(json.dumps(summary,indent=2)+'\n');print(json.dumps(summary,indent=2));raise SystemExit(0 if summary['pass'] else 1)
