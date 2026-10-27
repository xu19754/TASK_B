"""Read-only postprocessing of the nine existing trials; never launches physics."""
import argparse,csv,gzip,hashlib,json,shutil,zipfile
from pathlib import Path
ROOT=Path(__file__).resolve().parents[2]
p=argparse.ArgumentParser();p.add_argument('--output',type=Path,required=True);a=p.parse_args()
out=a.output if a.output.is_absolute() else ROOT/a.output;dest=out/'deliverables';dest.mkdir(exist_ok=True)
def load(p):return json.loads(p.read_text())
def rows(p):return [json.loads(s) for s in p.read_text().splitlines()]
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def writecsv(p,rs):
 with p.open('w',newline='') as f:
  w=csv.DictWriter(f,fieldnames=list(rs[0]));w.writeheader();w.writerows(rs)
suite=load(out/'suite.json');raw=load(out/'summary.json');freeze=load(out/'frozen_manifest.json')
assert raw['completed']==9 and raw['actual_process_invocations']==9
assert len(list(out.glob('*.started.json')))==9 and len(list(out.glob('*.completed.json')))==9
assert all(sha(ROOT/n)==digest for n,digest in freeze.items()),'Frozen files changed'
metrics=[];detail=[]
for c in suite['cases']:
 case=c['case_id'];directory=out/case;td=directory/'trial_01';rfile=td/'result.json';configured=load(ROOT/c['config'])
 r=load(rfile) if rfile.exists() else {};ev=rows(td/'evaluation.jsonl') if (td/'evaluation.jsonl').exists() else []
 hold=[d for d in ev if d['state']=='HOLD'];active=[d for d in ev if d['state']!='SETTLE'];audit=load(td/'path_audit.json') if (td/'path_audit.json').exists() else None
 consecutive=0.;longest=0.;th=configured['thresholds']
 for d in hold:
  valid=(d['rise_m']>=th['rise_m'] and d['bottom_m']>=th['bottom_m'] and d['in_jaws'] and d['dual_contact'] and d['tilt_deg']<=th['tilt_deg'] and d['drift_m']<=th['drift_m'] and not d['illegal_contact'])
  consecutive=consecutive+.02 if valid else 0.;longest=max(longest,consecutive)
 contact_errors=sum(len(d.get('native_contact_errors',[])) for d in ev)
 physical=bool(r.get('success',False) and r.get('state')=='DONE' and longest>=3. and contact_errors==0)
 path_clear=bool(audit and audit['clear'])
 strict=physical and path_clear
 reason=r.get('reason','')
 if not physical and not reason:reason='no_valid_3s_hold' if ev else 'no_trial_result'
 if physical and not path_clear:reason='conservative_path_overlap_or_audit_missing'
 m=dict(object=c['object'],yaw_deg=c['yaw_deg'],case_id=case,attempts=1,physical_grasp_success=physical,verified_success=strict,state=r.get('state','NO_RESULT'),reason=reason,max_rise_m=r.get('max_rise_m',0.),max_valid_hold_s=r.get('max_valid_hold_s',0.),valid_HOLD_s=longest,hold_min_bottom_m=min((d['bottom_m'] for d in hold),default=0.),hold_min_fixed_jaw_N=min((d['jaw_forces'][0] for d in hold),default=0.),hold_min_moving_jaw_N=min((d['jaw_forces'][1] for d in hold),default=0.),max_active_tilt_deg=max((d['tilt_deg'] for d in active),default=0.),max_active_drift_m=max((d['drift_m'] for d in active),default=0.),path_clear=path_clear,minimum_arm_floor_m=audit['minimum_arm_ground_clearance_m'] if audit else None,path_overlap_count=audit['conservative_overlap_count'] if audit else None,native_contact_errors=contact_errors,configured_root_distance_m=.55,body_height_m=configured['body_height'],sim_time_s=r.get('sim_time',0.),video=case+'.mp4')
 m['last_actual_tcp_error_m']=r.get('last_ik',{}).get('actual_position_error')
 m['last_model_tcp_error_m']=r.get('last_ik',{}).get('ee_position_error_m')
 m['last_fixed_jaw_force_N']=ev[-1]['jaw_forces'][0] if ev else 0.
 m['last_moving_jaw_force_N']=ev[-1]['jaw_forces'][1] if ev else 0.
 metrics.append(m);detail.append({'case':c,'configuration':configured,'result':r,'path_audit':audit})
 if (td/'continuous.mp4').exists():shutil.copy2(td/'continuous.mp4',dest/m['video'])
summary={'title':'三个物体、三个朝向，九次已知目标抓取测试','objects':['mustard','sugar','banana'],'angles_deg':[-45,0,45],'case_count':9,'process_invocations':9,'attempts_per_case':1,'retries':0,'navigation_tested':False,'distance_sweep':False,'initial_object_root_xy_m':[.55,0.],'physical_grasp_success_count':sum(m['physical_grasp_success'] for m in metrics),'verified_success_count':sum(m['verified_success'] for m in metrics),'frozen_files_unchanged':True,'thresholds':load(ROOT/suite['cases'][0]['config'])['thresholds'],'results':metrics,'details':detail}
(dest/'nine_summary.json').write_text(json.dumps(summary,indent=2,ensure_ascii=False));writecsv(dest/'nine_trials.csv',metrics)
labels={'mustard':'芥末瓶','sugar':'糖盒','banana':'香蕉'};matrix=[]
for name in labels:
 row={'物体':labels[name]}
 for yaw in [-45,0,45]:
  m=next(m for m in metrics if m['object']==name and m['yaw_deg']==yaw)
  row[str(yaw)+'度']='成功' if m['verified_success'] else ('抓起但路径待确认' if m['physical_grasp_success'] else '失败')
 matrix.append(row)
writecsv(dest/'nine_matrix.csv',matrix)
report=['# 三个物体、三个朝向的抓取测试','',f'本轮严格执行九个组合，每个只尝试一次：物理夹持抬升成功 {summary["physical_grasp_success_count"]}/9，连同保守轨迹检查确认 {summary["verified_success_count"]}/9。没有失败补测、重复取最好结果、距离档位或随机种子扫描。','', '| 物体 | -45 度 | 0 度 | +45 度 |','| --- | --- | --- | --- |']
for row in matrix:report.append('| '+' | '.join(str(v) for v in row.values())+' |')
report+=['','| 物体 | 初始朝向 | 物理成功 | 最大抬升 cm | 连续有效持物 s | 原因 |','| --- | --- | --- | --- | --- | --- |']
for m in metrics:report.append(f'| {labels[m["object"]]} | {m["yaw_deg"]} 度 | {"是" if m["physical_grasp_success"] else "否"} | {100*m["max_rise_m"]:.2f} | {m["max_valid_hold_s"]:.2f} | {m["reason"] or "满足判据"} |')
report+=['','距离与导航','', '当前版本不会主动走近或后退来调整抓取距离。它使用站立时的零速度策略，再由全身 Mink 原地协调腿、机身和机械臂完成下蹲与起身。移动到可抓范围属于后续导航；本轮不测试距离。物体初始根点均为 [0.55,0] m，机器人初始根点为 [0,0]，初始中心距离固定 0.55 m。站稳及抓取中的厘米级机身漂移另有记录，不算距离档位。','', '三个角度指在资产放稳朝向基础上，绕世界竖直轴旋转 -45、0、+45 度。芥末瓶和糖盒直立；香蕉平放。各类使用按原始几何选择的已知局部夹持点，夹爪目标朝向随配置一起旋转，未读取物体实时真值来修正控制。此结果不代表视觉识别或未知目标抓取。','', '本轮调整与保持不变的条件','', '芥末瓶夹瓶颈，糖盒夹上部窄面，机身下蹲高度均为 0.42 m；香蕉夹中段，机身下蹲到 0.34 m。香蕉 TCP 采用 link06 偏置 [0.235,0,0.012] m，靠近指尖以给地面留间隙；糖盒 TCP 为 [0.21,0,0.012] m，芥末瓶保留 [0.21,0,0.02] m。原始尺寸、0.5 kg 质量、碰撞、重力、夹爪 5 Nm 力矩和 0.25 rad/s 速度上限均未更改。没有附着、隐藏约束、运动中物体重置或机身瞬移。','', '成功要求同时满足根点抬升至少 0.10 m、最低点离地至少 0.02 m、物体处于双指夹持区且两侧各有大于 0.25 N 的真实接触、机身倾斜不超过 15 度、水平漂移不超过 0.10 m、无机身/髋/大腿异常触地，连续保持至少 3 s。本次严格复核还要求流程 DONE、完整 HOLD 内连续有效至少 3 s，以及采样路径审计无包围盒相交。仅 IK 可达、闭爪到位或程序退出不计成功。','', '每个组合只有一个样本，表中计数是本轮观察结果，不估计重复成功率。此前芥末瓶同姿态五次复验单独保留，不混入本轮九次。','', '数据与复现','', 'nine_summary.json 保存逐次配置、原结果及路径审计；nine_trials.csv 为可比较指标；每个 case_id.mp4 为该次完整连续录像。失败录像也保留。控制与评价 JSONL 分开记录，原生 PhysX 接触包含明确物体/夹指刚体对、接触点与冲量。','', '在原服务器 /root/Desktop/TaskB 或同环境下的仓库根目录执行：','', '~~~bash','bash scripts/run_mink_multiclass.sh --output runs/mink_multiclass/my_nine','~~~','', '输出目录必须不存在。入口按 suite.json 的九个配置依次各运行一次，不自动重试。配置在 experiments/mink_multiclass/configs/，原抓取算法仍为 experiments/mink_wholebody/。本轮实际运行目录为 runs/mink_multiclass/nine_01/，首次试验前的源码及配置哈希在 frozen_manifest.json，九个 started/completed 文件记录调用次数。','', '所有推理、IK、物理与渲染均在服务器完成；本机只连接、传输和查看。环境沿用已验证的 Isaac Sim 5.1、Isaac Lab 2.3.2、Mink 1.0.0 与 MuJoCo 3.3.7；MuJoCo 仅做运动学。原生关闭的超时回收继续独立记录在各次 process_status.json，不替代物理判据。','', '轨迹审计按每 0.1 s、状态转换和最终样本检查 URDF 保守包围盒，不是任意目标的连续避障证明。控制器、正式任务、行走权重和机器人 URDF 未为本轮改动；完整 Task B 的导航、感知、收集投放不在本轮范围。']
report += ['','失败定位（仅分析既有记录，不增加试验）','']
for m in metrics:
 if m['verified_success']:continue
 error=m['last_actual_tcp_error_m'];model_error=m['last_model_tcp_error_m']
 text=f'{labels[m["object"]]} {m["yaw_deg"]} 度：{m["reason"]}。'
 if error is not None:text+=f'最终实际 TCP 到目标误差 {1000*error:.2f} mm。'
 if model_error is not None:text+=f'同期运动学求解误差 {1000*model_error:.3f} mm。'
 text+=f'最后固定/活动指对物体的接触力约 {m["last_fixed_jaw_force_N"]:.2f}/{m["last_moving_jaw_force_N"]:.2f} N，未按通过处理。'
 report.append(text)
if any(m['object']=='banana' and m['reason']=='approach_timeout' for m in metrics):
 report += ['','香蕉接近超时的记录中，运动学目标可达，但实际接近被单侧接触阻挡，未形成双指夹持。当前低矮曲面物体的夹持点/接近几何仍需改进；本轮未放宽到位阈值，也未追加重试。该问题与本轮没有测试的导航距离调整分别记录。']
(dest/'nine_report.md').write_text('\n'.join(report)+'\n')
(out/'strict_verification.json').write_text(json.dumps(summary,indent=2,ensure_ascii=False))
files=[]
for folder in [ROOT/'experiments/mink_multiclass',ROOT/'experiments/mink_wholebody']:
 files.extend(x for x in folder.rglob('*') if x.is_file() and '__pycache__' not in x.parts)
files.extend([ROOT/'scripts/run_mink_multiclass.sh',ROOT/'scripts/run_mink_wholebody.sh',out/'suite.json',out/'frozen_manifest.json',out/'summary.json',out/'trials.csv',out/'strict_verification.json'])
archive=dest/'nine_source_evidence.zip'
with zipfile.ZipFile(archive,'w',zipfile.ZIP_DEFLATED,compresslevel=6) as z:
 for f in sorted(set(files)):z.write(f,f.relative_to(ROOT))
 for c in suite['cases']:
  name=c['case_id'];td=out/name/'trial_01'
  for f in td.glob('*'):
   if f.suffix=='.png':continue
   if f.suffix=='.jsonl':z.writestr(str(f.relative_to(ROOT))+'.gz',gzip.compress(f.read_bytes(),mtime=0))
   elif f.is_file():z.write(f,f.relative_to(ROOT))
  for f in [out/name/'config.json',out/name/'process_status.json',out/name/'code_manifest.json',out/name/'scene_metadata.json',out/(name+'.log'),out/(name+'.started.json'),out/(name+'.completed.json')]:
   if f.exists():z.write(f,f.relative_to(ROOT))
 for name in ['asset_geometry.json','gripper_geometry.json','kinematic_preflight.json']:
  f=ROOT/'runs/mink_multiclass'/name
  if f.exists():z.write(f,f.relative_to(ROOT))
 for f in dest.glob('nine_*.md'):z.write(f,'reports/'+f.name)
 for f in dest.glob('nine_*.csv'):z.write(f,'reports/'+f.name)
with zipfile.ZipFile(archive) as z:assert z.testzip() is None
(dest/'nine_sha256.json').write_text(json.dumps({f.name:sha(f) for f in dest.iterdir() if f.is_file() and f.name!='nine_sha256.json'},indent=2))
print(json.dumps({k:summary[k] for k in ['case_count','process_invocations','physical_grasp_success_count','verified_success_count']},ensure_ascii=False),flush=True)
