"""One attempt per object/yaw, using frozen controllers; stop on first failure."""
from pathlib import Path
import argparse, csv, datetime, hashlib, json, subprocess, sys, time, traceback

p = argparse.ArgumentParser()
p.add_argument('--output', type=Path, required=True)
p.add_argument('--expected', type=Path, required=True)
a = p.parse_args()
ROOT = Path('/root/Desktop/TaskB')
OUT = a.output.resolve()
OUT.mkdir(parents=True, exist_ok=False)
expected = json.loads(a.expected.read_text())

def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()

def unchanged():
    bad = [name for name, digest in expected.items() if not (ROOT/name).is_file() or sha(ROOT/name) != digest]
    if bad:
        raise RuntimeError('Frozen source/config mismatch: ' + repr(bad))

def save(path, data):
    path.write_text(json.dumps(data, indent=2, ensure_ascii=False) + '\n')

unchanged()
save(OUT/'frozen_manifest.json', expected)
(OUT/'driver.py').write_bytes(Path(__file__).read_bytes())
cases = []
for obj in ['mustard', 'sugar', 'banana']:
    for suffix, angle in [('m45', -45), ('p00', 0), ('p45', 45)]:
        folder = 'mink_banana' if obj == 'banana' else 'mink_multiclass'
        filename = 'centered_offset_13.json' if obj == 'banana' and angle == 0 else f'{obj}_yaw_{suffix}.json'
        cases.append((obj, angle, f'{obj}_yaw_{suffix}', ROOT/'experiments'/folder/'configs'/filename))
summary = dict(requested_trials=9, attempted_trials=0, successes=0, automatic_retries=0,
               stop_on_first_failure=True, initial_root_distance_m=0.55,
               source_commit='3bc5822401bd82e00f49b0c6dbb518f9d46d64ba',
               controller_sources_match_expected=True, completed=False,
               started_utc=datetime.datetime.now(datetime.timezone.utc).isoformat(), results=[])

def flush():
    save(OUT/'summary.json', summary)
    if summary['results']:
        with (OUT/'trials.csv').open('w', newline='') as f:
            writer = csv.DictWriter(f, fieldnames=list(summary['results'][0]))
            writer.writeheader()
            writer.writerows(summary['results'])

flush()
for obj, angle, case, config in cases:
    unchanged()
    runner = 'run_mink_banana.sh' if obj == 'banana' else 'run_mink_wholebody.sh'
    run = OUT/case
    command = ['bash', str(ROOT/'scripts'/runner), '--mode', 'grasp', '--headless', '--device', 'cpu',
               '--enable_cameras', '--record-video', '--trials', '1', '--config', str(config), '--output', str(run)]
    save(OUT/(case+'_started.json'), dict(command=command, config_sha256=sha(config), started_utc=datetime.datetime.now(datetime.timezone.utc).isoformat()))
    summary['attempted_trials'] += 1
    summary['active_case'] = case
    flush()
    print('CASE_START', case, flush=True)
    row = dict(object=obj, yaw_deg=angle, case=case, success=False, state='NO_RESULT', reason='',
               runner_exit_code=None, max_rise_m=None, max_valid_hold_s=None, independently_valid_hold_s=None,
               valid_HOLD_s=None, hold_stage_duration_s=None, min_HOLD_fixed_jaw_N=None,
               min_HOLD_moving_jaw_N=None, min_HOLD_bottom_m=None, max_active_tilt_deg=None,
               max_active_drift_m=None, native_contact_errors=None, path_clear=None,
               minimum_arm_floor_m=None, video=f'{case}/trial_01/continuous.mp4')
    try:
        with (OUT/(case+'.log')).open('w') as log:
            result = subprocess.run(command, cwd=ROOT, stdout=log, stderr=subprocess.STDOUT, timeout=540)
        row['runner_exit_code'] = result.returncode
        trial = run/'trial_01'
        recorded = json.loads((trial/'result.json').read_text())
        row.update(state=recorded.get('state'), reason=recorded.get('reason', ''),
                   max_rise_m=recorded.get('max_rise_m'), max_valid_hold_s=recorded.get('max_valid_hold_s'))
        if not recorded.get('success') or recorded.get('state') != 'DONE':
            raise RuntimeError('Physical grasp failed: ' + (row['reason'] or str(row['state'])))
        if result.returncode != 0:
            raise RuntimeError('Runner failed after recorded result: ' + str(result.returncode))
        rows = [json.loads(s) for s in (trial/'evaluation.jsonl').read_text().splitlines() if s.strip()]
        if not rows:
            raise RuntimeError('No evaluation trace')
        th = json.loads(config.read_text())['thresholds']
        def valid(r):
            return (r['rise_m'] >= th['rise_m'] and r['bottom_m'] >= th['bottom_m']
                    and r['in_jaws'] and min(r['jaw_forces']) > .25
                    and r['tilt_deg'] <= th['tilt_deg'] and r['drift_m'] <= th['drift_m']
                    and not r['illegal_contact'])
        max_valid = 0.
        current = 0.
        hold_max = 0.
        hold_current = 0.
        previous = None
        for r in rows:
            ok = valid(r)
            dt = max(0., r['sim_time']-previous['sim_time']) if previous else 0.
            current = current+dt if ok and previous and valid(previous) else 0.
            hold_current = hold_current+dt if ok and previous and valid(previous) and r['state']=='HOLD' and previous['state']=='HOLD' else 0.
            max_valid=max(max_valid,current)
            hold_max=max(hold_max,hold_current)
            previous=r
        hold = [r for r in rows if r['state']=='HOLD']
        active = [r for r in rows if r['state']!='SETTLE']
        history=recorded.get('history',[])
        enter=next((r['time'] for r in history if r['to']=='HOLD'),None)
        leave=next((r['time'] for r in history if r['from']=='HOLD' and r['to']=='DONE'),None)
        duration=leave-enter if enter is not None and leave is not None else 0.
        row.update(independently_valid_hold_s=max_valid, valid_HOLD_s=hold_max, hold_stage_duration_s=duration,
                   min_HOLD_fixed_jaw_N=min((r['jaw_forces'][0] for r in hold),default=0.),
                   min_HOLD_moving_jaw_N=min((r['jaw_forces'][1] for r in hold),default=0.),
                   min_HOLD_bottom_m=min((r['bottom_m'] for r in hold),default=0.),
                   max_active_tilt_deg=max((r['tilt_deg'] for r in active),default=0.),
                   max_active_drift_m=max((r['drift_m'] for r in active),default=0.),
                   native_contact_errors=sum(len(r.get('native_contact_errors',[])) for r in rows))
        if max_valid < th['hold_s']-.001 or hold_max < th['hold_s']-.001 or duration < 3.99:
            raise RuntimeError('Independent physical HOLD verification failed')
        if row['native_contact_errors'] or any(r['illegal_contact'] for r in active):
            raise RuntimeError('Native contact error or illegal body contact')
        video = trial/'continuous.mp4'
        if not video.is_file() or video.stat().st_size == 0 or recorded.get('frames',0) <= 5:
            raise RuntimeError('Video evidence missing')
        audit=ROOT/'experiments'/('mink_banana' if obj=='banana' else 'mink_wholebody')/'path_audit.py'
        with (OUT/(case+'_audit.log')).open('w') as log:
            ar=subprocess.run(['/isaac-sim/python.sh',str(audit),str(trial)],cwd=ROOT,stdout=log,stderr=subprocess.STDOUT,timeout=180)
        report=json.loads((trial/'path_audit.json').read_text())
        row.update(path_clear=report.get('clear'),minimum_arm_floor_m=report.get('minimum_arm_ground_clearance_m'))
        if ar.returncode != 0 or not report.get('clear'):
            raise RuntimeError('Executed-path audit failed')
        unchanged()
        row['success']=True
    except Exception as exc:
        row['reason']=(row['reason']+'; ' if row['reason'] else '')+str(exc)
        (OUT/(case+'_verification_error.txt')).write_text(traceback.format_exc())
    save(OUT/(case+'_completed.json'),row)
    summary['results'].append(row)
    summary['successes']=sum(r['success'] for r in summary['results'])
    summary['active_case']=None
    flush()
    print('CASE_END',json.dumps(row),flush=True)
    if not row['success']:
        summary['paused_on_failure']=case
        summary['finished_utc']=datetime.datetime.now(datetime.timezone.utc).isoformat()
        flush()
        print('PAUSED_ON_FIRST_FAILURE',case,flush=True)
        raise SystemExit(1)
summary['completed']=True
summary['finished_utc']=datetime.datetime.now(datetime.timezone.utc).isoformat()
flush()
print('ALL_NINE_VERIFIED',flush=True)
