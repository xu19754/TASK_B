"""Run each of three banana yaw angles once, preserving failures and all videos."""
from pathlib import Path
import argparse,csv,datetime,hashlib,json,subprocess
ROOT=Path(__file__).resolve().parents[1]
p=argparse.ArgumentParser(description='香蕉 -45/0/+45 度各运行一次，不自动重试。')
p.add_argument('--all-objects',action='store_true',help='芥末瓶、糖盒、香蕉共九个组合，各运行一次。')
p.add_argument('--dry-run',action='store_true',help='只显示将执行的组合，不启动仿真。')
p.add_argument('--output',type=Path,default=Path('runs/mink_banana/three_'+datetime.datetime.now().strftime('%Y%m%d_%H%M%S')))
a=p.parse_args();out=a.output if a.output.is_absolute() else ROOT/a.output
if not a.dry_run:out.mkdir(parents=True,exist_ok=False)
cases=[('banana_yaw_m45',-45,'banana_yaw_m45.json'),('banana_yaw_p00',0,'centered_offset_13.json'),('banana_yaw_p45',45,'banana_yaw_p45.json')]
if a.all_objects:
 cases=[(f'{obj}_yaw_{suffix}',angle,f'{obj}_yaw_{suffix}.json') for obj in ['mustard','sugar'] for suffix,angle in [('m45',-45),('p00',0),('p45',45)]]+cases
if a.dry_run:
 print(json.dumps(dict(trials=len(cases),automatic_retries=0,cases=cases),indent=2));raise SystemExit(0)
rows=[];manifest={str(f.relative_to(ROOT)):hashlib.sha256(f.read_bytes()).hexdigest() for folder in ['mink_wholebody','mink_banana'] for f in (ROOT/'experiments'/folder).glob('*.py')}
(out/'code_manifest.json').write_text(json.dumps(manifest,indent=2))
for case,angle,filename in cases:
 folder='mink_banana' if case.startswith('banana') else 'mink_multiclass'
 config=ROOT/'experiments'/folder/'configs'/filename
 assert all(hashlib.sha256((ROOT/f).read_bytes()).hexdigest()==sha for f,sha in manifest.items()),'控制代码在运行中改变'
 print('START',case,flush=True)
 runner='run_mink_banana.sh' if case.startswith('banana') else 'run_mink_wholebody.sh'
 command=['bash',str(ROOT/'scripts'/runner),'--mode','grasp','--headless','--device','cpu','--enable_cameras','--record-video','--trials','1','--config',str(config),'--output',str(out/case)]
 (out/(case+'_invocation.json')).write_text(json.dumps(dict(command=command,config_sha256=hashlib.sha256(config.read_bytes()).hexdigest()),indent=2))
 with (out/(case+'.log')).open('w') as log:result=subprocess.run(command,cwd=ROOT,stdout=log,stderr=subprocess.STDOUT)
 path=out/case/'trial_01/result.json';r=json.loads(path.read_text()) if path.exists() else {}
 row=dict(case=case,yaw_deg=angle,success=bool(r.get('success',False)),max_rise_m=r.get('max_rise_m'),max_valid_hold_s=r.get('max_valid_hold_s'),state=r.get('state','NO_RESULT'),reason=r.get('reason','no_result'),exit_code=result.returncode,video=str((out/case/'trial_01/continuous.mp4').relative_to(out)))
 rows.append(row)
 summary=dict(requested_trials=len(cases),attempted_trials=len(rows),automatic_retries=0,successes=sum(x['success'] for x in rows),results=rows)
 (out/'summary.json').write_text(json.dumps(summary,indent=2,ensure_ascii=False))
 with (out/'trials.csv').open('w',newline='') as f:
  writer=csv.DictWriter(f,fieldnames=list(row));writer.writeheader();writer.writerows(rows)
 print('FINISH',json.dumps(row,ensure_ascii=False),flush=True)
print('OUTPUT',out,flush=True)
raise SystemExit(0 if all(x['success'] for x in rows) else 1)
