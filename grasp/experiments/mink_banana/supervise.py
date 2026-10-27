"""External process bounds, including native shutdown that can hold Python's GIL.
Physics/IK run only in the Isaac Python child. A completed diagnostic result is
read from JSON; native teardown status is recorded independently.
"""
import os,sys,time,signal,subprocess,selectors,json
from pathlib import Path
ROOT=Path(__file__).resolve().parents[2]
arguments=sys.argv[1:]
trials=int(arguments[arguments.index('--trials')+1]) if '--trials' in arguments else 1
overall_timeout=180.+300.*max(1,trials)+15.
command=[os.environ.get('TASKB_PYTHON', sys.executable),'-u',str(ROOT/'experiments/mink_banana/bench.py'),*sys.argv[1:]]
child=subprocess.Popen(command,cwd=ROOT,stdout=subprocess.PIPE,stderr=subprocess.STDOUT,start_new_session=True)
selector=selectors.DefaultSelector();selector.register(child.stdout,selectors.EVENT_READ)
started=time.monotonic();ready=False;completed=None;directory=None;buffer='';forced=False;reason=None
def line_seen(line):
    global ready,completed,directory
    if '[OUTPUT] ' in line:
        value=line.split('[OUTPUT] ',1)[1].strip()
        directory=Path(value)
        if not directory.is_absolute(): directory=ROOT/directory
    if '[READY]' in line: ready=True
    if '[COMPLETE]' in line: completed=time.monotonic()
def stop_group():
    global forced
    forced=True
    try: os.killpg(child.pid,signal.SIGTERM)
    except ProcessLookupError: return
    try: child.wait(timeout=3.)
    except subprocess.TimeoutExpired:
        os.killpg(child.pid,signal.SIGKILL);child.wait(timeout=5.)
while True:
    for key,_ in selector.select(timeout=.25):
        chunk=os.read(key.fileobj.fileno(),65536)
        if not chunk:
            selector.unregister(key.fileobj);continue
        sys.stdout.buffer.write(chunk);sys.stdout.buffer.flush()
        buffer+=chunk.decode('utf-8',errors='replace')
        while '\n' in buffer:
            line,buffer=buffer.split('\n',1);line_seen(line)
    now=time.monotonic()
    if child.poll() is not None and not selector.get_map(): break
    if completed is not None and now-completed>5.:
        reason='native_teardown_exceeded_5s_after_results_flushed';stop_group()
    elif not ready and completed is None and now-started>180.:
        reason='startup_timeout';stop_group()
    elif now-started>overall_timeout:
        reason='overall_timeout';stop_group()
if buffer: line_seen(buffer)
native_exit=child.wait()
exit_code=native_exit if native_exit>=0 else 1
diagnostic_pass=None
if completed is not None and directory is not None and (directory/'summary.json').exists():
    result=json.loads((directory/'summary.json').read_text())
    if result['mode'] in ['env-check','posture']:
        diagnostic_pass=all(r.get('environment_pass',False) for r in result['results'])
    elif result['mode']=='reach':
        diagnostic_pass=all(r['state']=='DONE' for r in result['results'])
    else:
        diagnostic_pass=all(r['success'] for r in result['results'])
    exit_code=0 if diagnostic_pass else 1
elif reason is not None: exit_code=124
status={'command':command,'elapsed_s':time.monotonic()-started,'overall_timeout_s':overall_timeout,'ready':ready,'completed_result_marker':completed is not None,'native_exit_code':native_exit,'forced_native_cleanup':forced,'cleanup_reason':reason,'diagnostic_pass':diagnostic_pass,'exit_code':exit_code}
if directory is not None:
    (directory/'process_status.json').write_text(json.dumps(status,indent=2))
print('[PROCESS_STATUS]',json.dumps(status),flush=True)
raise SystemExit(exit_code)
