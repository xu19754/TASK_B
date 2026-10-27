"""Export compact, actual-pose replay data and verify this completed experiment."""
from pathlib import Path
import hashlib
import json
import os
import sys

os.environ['TASKB_BASKET_REVISION'] = 'v3'
ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT / 'scripts'))

import mujoco
import numpy as np
import basket_random_trials as trials

OUT = Path(__file__).resolve().parent
VIS = Path('/Users/yxh/.codex/visualizations/2026/10/26/01a177dd-b269-7971-9b4f-6b0c4809a352/basket-random-path.html')


def rounded(array):
    return np.asarray(array).round(5).tolist()


def main():
    summary = json.loads((OUT / 'summary.json').read_text())
    assert len(summary['cases']) == 12
    assert len({(r['scenario'], r['seed']) for r in summary['cases']}) == 12
    assert summary['protected_files_unchanged']
    assert summary['protected_files_sha256_before'] == trials.protected_files()
    assert hashlib.sha256((OUT / 'runner_used.py').read_bytes()).hexdigest() == summary['script_sha256']
    payload = dict(meshes={}, floor=trials.arm.MODEL_SPEC['floor_profile_xz_m'], cases=[])
    sims = {}
    for result in summary['cases']:
        scenario = result['scenario']
        if scenario not in sims:
            sims[scenario] = trials.make_sim(scenario)
        sim = sims[scenario]
        case = ROOT / result['folder']
        layout = json.loads((case / 'layout.json').read_text())
        sim.layout = layout['layout']
        # A cached model must restore the initial arm too, not retain the
        # preceding case's final arm while checking this case's initial cargo.
        assert sim.edge.initialize_candidate(), sim.message
        sim.apply_layout(sim.layout)
        assert all(row['below_rim'] for row in sim.metrics())
        assert not any(any(int(g) in sim.geoms for g in c.geom) and c.dist < -1e-6
                       for c in sim.data.contact)
        for i in range(3):
            for j in range(i):
                assert mujoco.mj_geomDistance(sim.model, sim.data, sim.geoms[i], sim.geoms[j], .004, None) >= .003 - 1e-9
        for name in sim.assets:
            payload['meshes'][name] = rounded(sim.hulls[name])
        if 'route' not in payload:
            payload['route'] = sim.edge.candidate()['edge_nodes_basket_m']
        trace = json.loads((case / 'trace.json').read_text())
        record = np.load(case / 'replay.npz')
        assert np.allclose(sim.data.qpos, record['initial_qpos'], atol=1e-10)
        assert len(trace) == len(record['qpos'])
        assert np.isfinite(record['qpos']).all()
        assert np.all(np.diff(record['time_s']) >= -1e-9)
        assert not np.any(np.array(result['peak_velocity_rad_s']) > trials.arm.VELOCITY + 1e-4)
        assert not np.any(np.array(result['peak_torque_Nm']) > trials.arm.EFFORT + 1e-7)
        assert not np.any(np.array(result['actual_joint_min_rad']) < trials.arm.LOWER - 1e-4)
        assert not np.any(np.array(result['actual_joint_max_rad']) > trials.arm.UPPER + 1e-4)
        assert result['exited_count'] == sum(result['fully_rear_at_end'])
        assert result['original_trajectory_duration_s'] == sim.edge.candidate()['duration_s']
        assert (result['motion_completed'] and not result['constraint_violation']) or result['rejected_next_step']
        b = sim.pose.links['tail_basket']
        frames = []
        for index in sorted(set([*range(0, len(record['qpos']), 10), len(record['qpos']) - 1])):
            sim.data.qpos[:] = record['qpos'][index]
            mujoco.mj_forward(sim.model, sim.data)
            cargo = []
            for i, bid in enumerate(sim.cargo_ids):
                center = (sim.data.xpos[bid] - b[:3, 3]) @ b[:3, :3]
                local_r = b[:3, :3].T @ sim.data.xmat[bid].reshape(3, 3)
                cargo.append(rounded(np.r_[center, trials.arm.base.quat_wxyz(local_r)]))
                assert np.allclose(center, trace[index]['cargo'][i]['center_basket_m'], atol=1e-8)
            mover = sim.data.body('gripperMover')
            actual_edge = (trials.edge.EDGE @ mover.xmat.reshape(3, 3).T
                           + mover.xpos - b[:3, 3]) @ b[:3, :3]
            frames.append(dict(t=round(float(record['time_s'][index]), 3), cargo=cargo,
                               edge=rounded(actual_edge),
                               exited=[r['rear_exit_observed'] for r in trace[index]['cargo']]))
        payload['cases'].append(dict(label=result['label'], scenario=scenario,
                                     seed=result['seed'], assets=sim.assets,
                                     exited=result['exited_count'], early=sum(result['escaped_before_sweep']),
                                     completed=result['motion_completed'], success=result['all_three_unloaded'],
                                     violation=result['constraint_violation'], frames=frames))
    encoded = json.dumps(payload, ensure_ascii=False, separators=(',', ':'), allow_nan=False)
    (OUT / 'replay_compact.json').write_text(encoded)
    fragment = VIS.read_text()
    start = fragment.index('>', fragment.index('<script type="application/json"')) + 1
    stop = fragment.index('</script>', start)
    fragment = fragment[:start] + encoded + fragment[stop:]
    assert len(fragment.encode()) < 1000000
    VIS.write_text(fragment)
    trials.report(OUT, summary)
    with (OUT / '测试报告.md').open('a') as stream:
        stream.write('\n## 本次重点复核\n\n混合 seed=20261026：糖盒在沉降阶段提前滑出；香蕉与夹爪接触后停在侧边，末态中心约 Y=8.6 cm。中央宽 7.7 cm 的铲边后续没有补扫。三个糖盒 seed=20261127 的末态虽然为 3/3，但包含一件提前滑出。严格达标的两组为三个芥末瓶 seed=20261227、三根香蕉 seed=20261328。\n\n独立 3D 回放可双击项目根目录的 `打开随机三件路径测试.command`；已完成相同的 2 秒沉降，点击“继续”执行原轨迹。聊天中的局部回放按 1 秒采样，原始姿态文件按 0.1 秒采样。\n\n第一组随机初态：\n\n![随机混合三件初态](mixed_three_seed_20261026/initial_scene.png)\n\n同组末态，香蕉留在篮子侧边：\n\n![香蕉侧边残留](mixed_three_seed_20261026/finish_scene.png)\n')
    validation = dict(cases=12, protected_files=len(trials.protected_files()),
                      protected_files_unchanged=True, runner_snapshot_matches=True,
                      actual_pose_trace_consistent=True, finite_poses=True,
                      all_initial_layouts_contained_and_nonoverlapping=True,
                      original_trajectory_duration_unchanged=True,
                      accepted_step_joint_limits_preserved=True,
                      all_motion_stops_have_recorded_rejection=True,
                      inline_replay_bytes=len(fragment.encode()),
                      complete_unloading_cases=sum(r['all_three_unloaded'] for r in summary['cases']),
                      guarded_cases=sum(bool(r['constraint_violation']) for r in summary['cases']))
    trials.write_json(OUT / 'review_validation.json', validation)
    print(json.dumps(validation))


if __name__ == '__main__':
    main()
