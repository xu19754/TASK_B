import json
from pathlib import Path
ROOT=Path(__file__).resolve().parents[2];out=ROOT/'runs/mink_multiclass/nine_01'
completed=[]
for c in json.loads((ROOT/'experiments/mink_multiclass/suite.json').read_text())['cases']:
 name=c['case_id'];marker=out/(name+'.completed.json')
 if marker.exists():
  d=json.loads(marker.read_text());completed.append(d)
  print('DONE',name,'success',d['success'],'state',d['state'],'reason',d['reason'],'rise',round(d['max_rise_m'],4),'hold',round(d['max_valid_hold_s'],2))
 elif (out/(name+'.started.json')).exists():
  f=out/name/'trial_01/evaluation.jsonl'
  if f.exists():
   for s in reversed(f.read_text().splitlines()):
    try:d=json.loads(s);break
    except:continue
   print('RUNNING',name,{k:d[k] for k in ['sim_time','state','rise_m','valid_hold_s','jaw_forces']})
  else:print('STARTING',name)
print('TOTAL',len(completed),'/9; success',sum(x['success'] for x in completed))
