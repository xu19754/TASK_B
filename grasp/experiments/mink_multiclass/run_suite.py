"""Nine fixed cases, one process/attempt each. Never retries failed cases."""
import argparse,csv,hashlib,json,subprocess,time,os,signal
from pathlib import Path
ROOT=Path(__file__).resolve().parents[2]
p=argparse.ArgumentParser(description='3 objects x 3 yaw angles, exactly one physical attempt per case')
p.add_argument('--output',type=Path,required=True)
a=p.parse_args();out=a.output if a.output.is_absolute() else ROOT/a.output
out.mkdir(parents=True,exist_ok=False)
suite=json.loads((Path(__file__).parent/'suite.json').read_text())
assert len(suite['cases'])==9 and len(set(x['case_id'] for x in suite['cases']))==9
tracked=[*sorted((ROOT/'experiments/mink_wholebody').glob('*.py')),*sorted((ROOT/'experiments/mink_multiclass').rglob('*.py')),*[ROOT/x['config'] for x in suite['cases']],ROOT/'taskb_student/low_level.py',ROOT/'taskb_student/robot.py',ROOT/'assets/b2z1/b2z1_0815.urdf',ROOT/'models/locomotion/policy.pt']
manifest={str(x.relative_to(ROOT)):hashlib.sha256(x.read_bytes()).hexdigest() for x in tracked}
(out/'frozen_manifest.json').write_text(json.dumps(manifest,indent=2));(out/'suite.json').write_text(json.dumps(suite,indent=2))
rows=[];started=time.monotonic()
for index,case in enumerate(suite['cases'],1):
 name=case['case_id'];directory=out/name
 command=['bash',str(ROOT/'scripts/run_mink_wholebody.sh'),'--mode','grasp','--trials','1','--headless','--enable_cameras','--record-video','--config',str(ROOT/case['config']),'--output',str(directory)]
 marker=out/(name+'.started.json')
 with marker.open('x') as f:json.dump({'index':index,'case':case,'command':command},f,indent=2)
 for path,digest in manifest.items():
  if hashlib.sha256((ROOT/path).read_bytes()).hexdigest()!=digest:raise RuntimeError('Frozen file changed: '+path)
 print('[CASE_START]',index,name,flush=True)
 with (out/(name+'.log')).open('w') as log:
  child=subprocess.Popen(command,cwd=ROOT,stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
  try:code=child.wait(timeout=525.)
  except subprocess.TimeoutExpired:
   os.killpg(child.pid,signal.SIGTERM)
   try:child.wait(timeout=8.)
   except subprocess.TimeoutExpired:os.killpg(child.pid,signal.SIGKILL);child.wait()
   code=124
 result_file=directory/'trial_01/result.json'
 result=json.loads(result_file.read_text()) if result_file.exists() else {}
 row=dict(index=index,case_id=name,object=case['object'],yaw_deg=case['yaw_deg'],attempts=1,root_distance_m=.55,success=bool(result.get('success',False) and result.get('state')=='DONE'),state=result.get('state','NO_RESULT'),reason=result.get('reason','') or ('' if result else 'startup_or_process_failure'),max_rise_m=result.get('max_rise_m',0.),max_valid_hold_s=result.get('max_valid_hold_s',0.),exit_code=code,output=str(directory.relative_to(ROOT)))
 rows.append(row)
 (out/(name+'.completed.json')).write_text(json.dumps(row,indent=2))
 summary={'title':suite['title'],'case_count':9,'completed':len(rows),'physical_attempt_limit':9,'actual_process_invocations':len(rows),'no_retries':True,'angles_deg':suite['angles_deg'],'root_distance_m':.55,'navigation_tested':False,'distance_sweep':False,'success_count':sum(r['success'] for r in rows),'results':rows,'elapsed_s':time.monotonic()-started}
 (out/'summary.json').write_text(json.dumps(summary,indent=2))
 with (out/'trials.csv').open('w',newline='') as f:
  writer=csv.DictWriter(f,fieldnames=list(row));writer.writeheader();writer.writerows(rows)
 print('[CASE_RESULT]',json.dumps(row),flush=True)
print('[SUITE_COMPLETE]',sum(r['success'] for r in rows),'/9',flush=True)
