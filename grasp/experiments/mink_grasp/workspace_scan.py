"""Finite Mink candidate audit in the observed standing pose, not a custom IK."""
import sys,json,time
from pathlib import Path
ROOT=Path(__file__).resolve().parents[2];sys.path.insert(0,str(ROOT))
import numpy as np
from scipy.spatial.transform import Rotation
from experiments.mink_grasp.mink_adapter import MinkAdapter,wxyz,quat_matrix
from taskb_student.constants import ARM_Q0
out=Path(sys.argv[1]);out.mkdir(parents=True,exist_ok=True)
config=json.loads((ROOT/'experiments/mink_grasp/config.json').read_text())
row=json.loads((ROOT/'runs/mink_grasp/reach_02/trial_01/controller.jsonl').read_text().splitlines()[299])
base=np.array(row['root_position']);rotation=quat_matrix(row['base_quat'])
ik=MinkAdapter(config['tcp_offset_link06'])
results=[];started=time.monotonic()
seeds=[ARM_Q0[:6],[0.,2.95,-1.5,.12,0.,1.57]]
for x in [.55,.65,.75]:
    for z in [.04,.10,.167,.185,.24,.30]:
        for pitch in [.7,1.2,1.57,2.0]:
            for roll in [0.,1.57]:
                target_local=rotation.T@np.array([x,0.,z-base[2]])
                quat=wxyz(rotation.T@Rotation.from_euler('xyz',[roll,pitch,0.]).as_matrix())
                best=None
                for seed in seeds:
                    q,r=ik.solve(target_local,quat,seed,max_iters=80)
                    candidate={'x_relative_m':x,'target_world_z_m':z,'roll':roll,'pitch':pitch,'q':q.tolist(),**r}
                    if best is None or r['position_error']<best['position_error']: best=candidate
                    if r['success']: break
                results.append(best)
        if time.monotonic()-started>100: raise RuntimeError('Finite scan exceeded 100s')
# Triangle-inequality lower bound for TCP in an exactly level base.
lo,hi=ik.limits_array[1]
min_shoulder_sine=min(np.sin(lo),np.sin(hi))
remaining=np.linalg.norm([.218,0,.057])+.07+.0492+np.linalg.norm(ik.tcp_offset)
level_lower_bound=float(base[2]+.09+.0585+.045+.35*min_shoulder_sine-remaining)
summary={'base':base.tolist(),'base_quat':row['base_quat'],'config_tcp':config['tcp_offset_link06'],'sampled_pose_count':len(results),'seeds':2,'max_iters_per_seed':80,'elapsed_s':time.monotonic()-started,'level_base_tcp_z_lower_bound_m':level_lower_bound,'success_count':sum(r['success'] for r in results),'ground_grasp_success_count':sum(r['success'] and r['target_world_z_m']<=.185 for r in results),'successes':[r for r in results if r['success']],'best_ground_candidates':sorted([r for r in results if r['target_world_z_m']<=.185],key=lambda r:r['position_error'])[:10]}
(out/'all_candidates.json').write_text(json.dumps(results,indent=2))
(out/'summary.json').write_text(json.dumps(summary,indent=2))
print(json.dumps(summary,indent=2),flush=True)
