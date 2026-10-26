"""Random three-cargo tests of the existing v3 shovel trajectory.

This is an independent diagnostic: the B2 carrier is prescribed, Z1 and cargo
use MuJoCo dynamics. Initialization alone writes cargo poses. Evaluation,
robot assets, motor limits, collision masks and the trajectory are untouched.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
from pathlib import Path
import sys
import time

os.environ.setdefault('TASKB_BASKET_REVISION', 'v3')

import mujoco
import numpy as np
from scipy.spatial import ConvexHull

import basket_arm_viewer as arm
import basket_edge as edge

SCENARIOS = ('mixed_three', *arm.base.OBJECT_NAMES)
LABELS = dict(zip(SCENARIOS, ('混合三件', '三个糖盒', '三个芥末瓶', '三根香蕉')))
DT = arm.base.DT
SAMPLE_S = .1
EXIT_X = -.230


def write_json(path, value):
    path.write_text(json.dumps(value, indent=2, ensure_ascii=False, allow_nan=False) + '\n')


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def protected_files():
    paths = [arm.VARIANT, edge.OUT / 'low_edge_candidate.json',
             arm.ROOT / 'demo/solution.py', arm.ROOT / 'taskb_student/scoring.py',
             arm.ROOT / 'taskb_student/robot.py', arm.ROOT / 'taskb_student/env_cfg.py',
             arm.ROOT / 'scripts/basket_arm_viewer.py',
             arm.ROOT / 'scripts/basket_viewer.py', arm.ROOT / 'scripts/basket_edge.py']
    paths += sorted((arm.ROOT / 'assets/b2z1/meshes' / ('basket_' + arm.REVISION)).rglob('*.obj'))
    paths += sorted((arm.ROOT / 'reports/basket_feasibility/screen_meshes').glob('*.obj'))
    return {str(p.relative_to(arm.ROOT)): digest(p) for p in paths}


class RandomDiagnostic(arm.ArmDiagnostic):
    """Keep an explicitly saved random layout when the viewer resets."""
    def __init__(self):
        self.layout = None
        super().__init__()
        self.hulls = {n: o['vertices'][ConvexHull(o['vertices']).vertices]
                      for n, o in self.objects.items()}

    def reset(self):
        super().reset()
        if self.layout is not None and [p['asset'] for p in self.layout] == self.assets:
            self.apply_layout(self.layout)

    def apply_layout(self, layout):
        b = self.pose.links['tail_basket']
        for adr, row in zip(self.qadr, layout):
            rotation = np.asarray(row['rotation_basket'])
            self.data.qpos[adr:adr + 3] = b[:3, :3] @ row['center_basket_m'] + b[:3, 3]
            self.data.qpos[adr + 3:adr + 7] = arm.base.quat_wxyz(b[:3, :3] @ rotation)
        for bid in self.cargo_ids:
            adr = self.model.jnt_dofadr[self.model.body_jntadr[bid]]
            self.data.qvel[adr:adr + 6] = 0
        self.escaped = [False] * 3
        mujoco.mj_forward(self.model, self.data)

    def cargo_local(self):
        # Extrema over convex-hull vertices equal extrema over the whole mesh.
        # This affects measurement cost only, never physics/collision geometry.
        b = self.pose.links['tail_basket']
        return [(self.hulls[name] @ self.data.xmat[bid].reshape(3, 3).T
                 + self.data.xpos[bid] - b[:3, 3]) @ b[:3, :3]
                for bid, name in zip(self.cargo_ids, self.assets)]


def make_sim(scenario):
    sim = RandomDiagnostic()
    sim.setup_cargo(SCENARIOS.index(scenario))
    sim.edge = edge.EdgeControl(sim)
    if not sim.edge.initialize_candidate():
        raise RuntimeError(sim.message)
    return sim


def random_layout(sim, seed):
    """Rejection sample nonoverlapping poses, with no execution-based filtering."""
    rng = np.random.default_rng(seed)
    rotation = arm.base.geometry.rotation
    resting = [np.eye(3), rotation([1, 0, 0], math.pi),
               rotation([1, 0, 0], math.pi / 2), rotation([1, 0, 0], -math.pi / 2),
               rotation([0, 1, 0], math.pi / 2), rotation([0, 1, 0], -math.pi / 2)]
    b = sim.pose.links['tail_basket']
    # Unplaced cargo is outside the basket during initialization only.
    for i, adr in enumerate(sim.qadr):
        sim.data.qpos[adr:adr + 7] = [0, 0, 3 + i, 1, 0, 0, 0]
    order = rng.permutation(3).tolist()
    placements = [None] * 3
    attempts = [0] * 3
    placed = []
    for i in order:
        vertices = sim.hulls[sim.assets[i]]
        gid, adr = sim.geoms[i], sim.qadr[i]
        for attempt in range(1, 10001):
            r = rotation([0, 0, 1], rng.uniform(-math.pi, math.pi)) @ resting[rng.integers(6)]
            v = vertices @ r.T
            low, high = v.min(0), v.max(0)
            # 5.5 mm lateral reserve; entire initial object below the rim.
            lower_xy = np.array([-.218, -.118]) - low[:2]
            upper_xy = np.array([.218, .118]) - high[:2]
            if np.any(upper_xy <= lower_xy) or high[2] - low[2] > .150:
                continue
            xy = rng.uniform(lower_xy, upper_xy)
            z = float(np.max(edge.floor_z(v[:, 0] + xy[0]) - v[:, 2]) + .003)
            if high[2] + z > .155:
                continue
            p = np.r_[xy, z]
            sim.data.qpos[adr:adr + 3] = b[:3, :3] @ p + b[:3, 3]
            sim.data.qpos[adr + 3:adr + 7] = arm.base.quat_wxyz(b[:3, :3] @ r)
            mujoco.mj_forward(sim.model, sim.data)
            if any(gid in c.geom and c.dist < -1e-6 for c in sim.data.contact):
                continue
            if any(mujoco.mj_geomDistance(sim.model, sim.data, gid, sim.geoms[j], .004, None)
                   < .003 for j in placed):
                continue
            placements[i] = dict(asset=sim.assets[i], center_basket_m=p.tolist(),
                                 rotation_basket=r.tolist())
            attempts[i] = attempt
            placed.append(i)
            break
        else:
            raise RuntimeError(f'Could not pack cargo {i} for seed {seed}; no rollout was run')
    sim.layout = placements
    sim.apply_layout(placements)
    penetrations = [float(c.dist) for c in sim.data.contact
                    if any(int(g) in sim.geoms for g in c.geom) and c.dist < -1e-6]
    if penetrations:
        raise RuntimeError(f'Invalid initial cargo penetration: {penetrations}')
    return dict(layout=placements, placement_order=order, attempts=attempts,
                initial_penetration_m=0., minimum_cargo_gap_m=.003)


def sample(sim, stage, start_time):
    b = sim.pose.links['tail_basket']
    cargo = []
    for row, bid, vertices in zip(sim.metrics(), sim.cargo_ids, sim.cargo_local()):
        center = (sim.data.xpos[bid] - b[:3, 3]) @ b[:3, :3]
        hull = vertices[ConvexHull(vertices[:, :2]).vertices, :2]
        cargo.append(dict(**row, center_basket_m=center.tolist(), footprint_xy_m=hull.tolist()))
    return dict(time_s=round(float(sim.data.time), 6),
                sweep_elapsed_s=round(float(sim.data.time - start_time), 6), stage=stage,
                cargo=cargo, arm_q_rad=sim.actual_arm().tolist(),
                arm_velocity_rad_s=sim.data.qvel[sim.arm_dadr].tolist(),
                arm_torque_Nm=sim.data.actuator_force[sim.arm_motor].tolist())


def execute(sim, folder, metadata, scenario, seed):
    folder.mkdir(parents=True, exist_ok=False)
    write_json(folder / 'layout.json', dict(scenario=scenario, seed=seed, **metadata))
    states, replay_q, replay_times = [], [], []
    first_exit = [None] * 3
    arm_contact_depth = 0.
    arm_contact_pairs = set()
    cargo_contact_pairs = set()
    minimum_q, maximum_q = sim.actual_arm(), sim.actual_arm()
    q0 = sim.data.qpos.copy()
    start = time.perf_counter()
    sweep_start = 2.

    def observe(step, stage):
        nonlocal arm_contact_depth, minimum_q, maximum_q
        actual = sim.actual_arm()
        minimum_q = np.minimum(minimum_q, actual)
        maximum_q = np.maximum(maximum_q, actual)
        pairs = sim.data.contact.geom
        types = sim.model.geom_contype[pairs]
        arm_environment = (types.min(axis=1) == 1) & (types.max(axis=1) == 4)
        arm_cargo = (types.min(axis=1) == 2) & (types.max(axis=1) == 4)
        for idx in np.flatnonzero(arm_environment):
            arm_contact_depth = min(arm_contact_depth, float(sim.data.contact.dist[idx]))
            arm_contact_pairs.add(tuple(sim.model.geom(int(g)).name for g in pairs[idx]))
        for idx in np.flatnonzero(arm_cargo):
            cargo_contact_pairs.add(tuple(sim.model.geom(int(g)).name for g in pairs[idx]))
        # Exit detection is 50 Hz; joint protection remains at every 1 ms step.
        if step % 20 == 0 or sim.violation:
            for i, row in enumerate(sim.metrics()):
                if row['rear_exit_observed'] and first_exit[i] is None:
                    first_exit[i] = round(float(sim.data.time), 6)
        if step % round(SAMPLE_S / DT) == 0 or sim.violation:
            states.append(sample(sim, stage, sweep_start))
            replay_q.append(sim.data.qpos.copy())
            replay_times.append(float(sim.data.time))

    observe(0, 'initial')
    for step in range(1, 2001):
        sim.step()
        observe(step, 'settle')
        if sim.violation:
            break
    settle_cargo = sim.metrics()
    escaped_before_sweep = sim.escaped.copy()
    settle_q = sim.data.qpos.copy()
    duration = float(sim.edge.candidate()['duration_s'])
    started = False
    rejection = None
    if not sim.violation:
        started = sim.edge.play_candidate()
        if not started:
            rejection = sim.message
        else:
            if abs(sim.smooth_motion.duration - duration) > 1e-8:
                raise RuntimeError('The original trajectory duration changed')
            sweep_start = float(sim.data.time)
            for step in range(1, math.ceil((duration + 4) / DT) + 1):
                sim.step()
                observe(step, 'sweep')
                if sim.violation:
                    break
    states.append(sample(sim, 'final', sweep_start))
    replay_q.append(sim.data.qpos.copy())
    replay_times.append(float(sim.data.time))
    final_cargo = sim.metrics()
    strict_exit = [bool(row['front_most_x_m'] < EXIT_X) for row in final_cargo]
    full_success = bool(started and sim.smooth_complete and not sim.violation
                        and not any(escaped_before_sweep) and all(strict_exit))
    result = dict(scenario=scenario, label=LABELS[scenario], seed=seed,
                  original_trajectory_duration_s=duration, simulated_time_s=float(sim.data.time),
                  walltime_s=round(time.perf_counter() - start, 3),
                  trajectory_started=started, motion_completed=sim.smooth_complete,
                  all_three_unloaded=full_success, rear_exit_observed=sim.escaped,
                  fully_rear_at_end=strict_exit, rear_exit_time_s=first_exit,
                  exited_count=sum(strict_exit), escaped_before_sweep=escaped_before_sweep,
                  settled_cargo=settle_cargo, final_cargo=final_cargo,
                  constraint_violation=sim.violation, rejected_next_step=sim.rejected_step,
                  preflight_rejection=rejection,
                  actual_joint_min_rad=minimum_q.tolist(), actual_joint_max_rad=maximum_q.tolist(),
                  peak_velocity_rad_s=sim.max_speed.tolist(), peak_torque_Nm=sim.max_torque.tolist(),
                  velocity_limits_rad_s=arm.VELOCITY.tolist(), effort_limits_Nm=arm.EFFORT.tolist(),
                  deepest_arm_environment_contact_m=arm_contact_depth,
                  observed_arm_environment_pairs=sorted(arm_contact_pairs),
                  observed_arm_cargo_pairs=sorted(cargo_contact_pairs),
                  layout_file='layout.json', trace_file='trace.json', replay_file='replay.npz')
    write_json(folder / 'result.json', result)
    write_json(folder / 'trace.json', states)
    np.savez_compressed(folder / 'replay.npz', time_s=np.asarray(replay_times),
                        qpos=np.asarray(replay_q), initial_qpos=q0, settled_qpos=settle_q)
    return result


def replay(case_folder, visible):
    row = json.loads((case_folder / 'layout.json').read_text())
    sim = make_sim(row['scenario'])
    sim.layout = row['layout']
    sim.apply_layout(sim.layout)
    if visible:
        # Repeat the same two-second settling phase before arming the replay.
        for _ in range(2000):
            sim.step()
            if sim.violation:
                break
        sim.metrics()
        if not sim.violation and not sim.edge.play_candidate():
            raise RuntimeError(sim.message)
        sim.paused = True
        sim.message = f"seed={row['seed']}；已沉降2秒，点击继续执行原路径"
        window = arm.Window(sim)
        window.tab = 'edge'
        window.view('tail')
        try:
            window.run()
        finally:
            window.close()
    else:
        from types import SimpleNamespace
        from PIL import Image
        data = np.load(case_folder / 'replay.npz')
        sim.model.geom_rgba[sim.model.geom('visual_tail_basket_0').id, 3] = .4
        camera = mujoco.MjvCamera()
        camera.lookat[:] = [-.24, 0, .60]
        camera.distance, camera.azimuth, camera.elevation = 1.20, 135, -38
        options = mujoco.MjvOption()
        options.geomgroup[3] = 0
        renderer = mujoco.Renderer(sim.model, height=900, width=1200)
        try:
            for name, q, t in [('initial', data['initial_qpos'], 0.),
                              ('settled', data['settled_qpos'], 2.),
                              ('finish', data['qpos'][-1], data['time_s'][-1])]:
                sim.data.qpos[:] = q
                sim.data.qvel[:] = 0
                sim.data.time = t
                mujoco.mj_forward(sim.model, sim.data)
                renderer.update_scene(sim.data, camera=camera, scene_option=options)
                sim.edge.overlay(SimpleNamespace(scene=renderer.scene))
                Image.fromarray(renderer.render()).save(case_folder / (name + '_scene.png'))
        finally:
            renderer.close()


def report(folder, summary):
    rows = summary['cases']
    full = sum(r['all_three_unloaded'] for r in rows)
    guarded = sum(bool(r['constraint_violation']) for r in rows)
    completed = sum(r['motion_completed'] for r in rows)
    early = sum(sum(r['escaped_before_sweep']) for r in rows)
    final_exits = sum(r['exited_count'] for r in rows)
    active_exits = sum(sum(end and not before for end, before in
                          zip(r['fully_rear_at_end'], r['escaped_before_sweep'])) for r in rows)
    gripper_stops = sum(bool(r['rejected_next_step']) and
                        abs(r['rejected_next_step']['velocity_rad_s'][-1]) > arm.VELOCITY[-1]
                        for r in rows)
    lines = ['# v3 篮子：随机三件物料原路径测试', '',
             f'**当前抽样完整卸料达标 {full}/{len(rows)} 组，'
             f'关节保护停止 {guarded}/{len(rows)} 组。**', '',
             f"固定原轨迹，每组随机位置、朝向与放置顺序；共 {len(rows)} 组、{len(rows) * 3} 件。",
             '', '| 组合 | seed | 末态越界 | 其中提前滑出 | 动作完成 | 关节保护 | 完整卸料达标 |',
             '| --- | ---: | ---: | ---: | --- | --- | --- |']
    for r in rows:
        lines.append(f"| {r['label']} | {r['seed']} | {r['exited_count']}/3 | "
                     f"{sum(r['escaped_before_sweep'])} | {'是' if r['motion_completed'] else '否'} | "
                     f"{r['constraint_violation'] or r['preflight_rejection'] or '未触发'} | "
                     f"{'是' if r['all_three_unloaded'] else '否'} |")
    lines += ['', f'原动作完整执行 {completed}/{len(rows)} 组。末态完整越过尾部平面 {final_exits}/{len(rows) * 3} 件，'
              f'其中 {early} 件在动作前的沉降阶段已经滑出；动作后新增越界 {active_exits} 件。'
              '提前滑出不能当作原路径卸料成功。', '',
              '## 观察到的问题', '',
              f'- {gripper_stops} 次保护由 jointGripper 的下一步实际速度超过 0.25 rad/s 触发；'
              '保持原限速，回退到上一个有效状态后停止，没有放宽限制继续播放。',
              '- 检查各组 `trace.json` 的物料轮廓和位置，可区分侧边残留、尾部卡料与提前滑出。',
              '- 出现动作前滑出的组，需要单独验证装载保持；不能仅用末态越界数量判断路径成功。', '',
              '## 保护触发记录', '',
              '| seed | 拒绝步时间（s） | 夹爪拒绝步速度（rad/s） | 限值（rad/s） |',
              '| ---: | ---: | ---: | ---: |']
    for r in rows:
        rejected = r['rejected_next_step']
        if rejected:
            lines.append(f"| {r['seed']} | {rejected['time_s']:.3f} | {rejected['velocity_rad_s'][-1]:+.6f} | 0.25 |")
    lines += ['', '结果中的峰值速度/力矩统计接受的有效步；拒绝步单独保存在 `rejected_next_step`，'
              '因此有效步峰值低于限值与触发保护并不矛盾。', '',
              '## 试验设置', '',
              '- v3 原篮子与原关节轨迹；后倾 15°、前移 10 cm、机身高 57.5 cm。',
              '- 摩擦 0.5；每件质量沿用既有诊断的 0.5 kg；目标峰值速度 0.02 rad/s。',
              '- 每组始终是三件：一盒一瓶一香蕉，或三个同类。不同 seed 独立抽样。',
              '- 从六个轴向静置姿态和连续随机偏航角抽样；按物体尺寸拒绝放不下的姿态。',
              '- 初始整件位于篮内，离侧面至少约 5 mm，离底约 3 mm；物体之间至少 3 mm，初态无穿透。',
              '- 只按初始几何接受/拒绝摆放，不按执行结果重抽；先自由沉降 2 秒，再运行原路径。',
              '- 每 1 ms 检查实际关节角度、速度和力矩；超限时沿用原保护，停在上一个有效状态。',
              '- 尾部越界检测 50 Hz；回放姿态和曲线每 0.1 秒记录；失败的下一步另存完整信息。',
              '- 卸出定义：整件位于篮子局部 X < -0.230 m；全部成功还要求没有提前掉落、动作完成且保护未触发。', '',
              '## 复现', '', '```sh',
              f".venv-basket/bin/python scripts/basket_random_trials.py --seed {summary['master_seed']} --trials-per-scenario {summary['trials_per_scenario']}",
              f".venv-basket/bin/python scripts/basket_random_trials.py --view {rows[0]['folder']}",
              '```', '',
              '每组目录包含初态布局、完整结果、采样轨迹和实际姿态回放。`summary.json` 保存所有组及资产哈希校验。', '',
              '本次实际运行脚本快照为 `runner_used.py`，其哈希与 `summary.json` 中 `script_sha256` 一致。'
              f"篮子、原轨迹、物料网格、正式策略与规则等 {len(summary['protected_files_sha256_before'])} 个受保护文件"
              f"在测试前后哈希{'一致' if summary['protected_files_unchanged'] else '不一致'}。", '',
              '## 适用范围', '',
              '这是规定机身姿态下的 MuJoCo 机械臂/物料动力学试验。物料仅在明确初始化时设置姿态，'
              '运动中没有吸附、外力辅助或瞬移。没有目标桶，不能计为 Task B 入桶或得分；'
              '整机反作用平衡、行走、收臂到准备位及 Isaac/PhysX 仍需分别验证。'
              f"每个组合 {summary['trials_per_scenario']} 次的样本只用于发现问题，不构成对所有随机摆放的成功率估计。", '']
    (folder / '测试报告.md').write_text('\n'.join(lines))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--seed', type=int, default=20261026)
    parser.add_argument('--trials-per-scenario', type=int, default=3)
    parser.add_argument('--scenarios', nargs='+', choices=SCENARIOS, default=SCENARIOS)
    parser.add_argument('--output', type=Path)
    parser.add_argument('--view', type=Path, help='Open an existing exact layout in the native viewer')
    parser.add_argument('--render', type=Path, help='Render actual recorded initial/end states')
    args = parser.parse_args()
    if args.view or args.render:
        replay((args.view or args.render).resolve(), bool(args.view))
        return
    if args.trials_per_scenario < 1:
        parser.error('--trials-per-scenario must be positive')
    if arm.REVISION != 'v3':
        parser.error('This diagnostic is fixed to the existing v3 trajectory')
    folder = args.output or arm.OUT / f'random_three_{args.seed}_{time.strftime("%Y%m%d_%H%M%S")}'
    folder = folder.resolve()
    folder.mkdir(parents=True, exist_ok=False)
    (folder / 'runner_used.py').write_bytes(Path(__file__).read_bytes())
    before = protected_files()
    summary = dict(master_seed=args.seed, trials_per_scenario=args.trials_per_scenario,
                   revision=arm.REVISION, mujoco_version=mujoco.__version__,
                   original_trajectory_sha256=digest(edge.OUT / 'low_edge_candidate.json'),
                   scope='Prescribed B2 + dynamic Z1 + free cargo; no bucket/scoring/full-robot balance',
                   cases=[], protected_files_sha256_before=before, script_sha256=digest(Path(__file__)))
    for scenario in args.scenarios:
        for trial in range(args.trials_per_scenario):
            seed = args.seed + SCENARIOS.index(scenario) * 100 + trial
            name = f'{scenario}_seed_{seed}'
            sim = make_sim(scenario)
            metadata = random_layout(sim, seed)
            print(f'START {name} attempts={metadata["attempts"]}', flush=True)
            result = execute(sim, folder / name, metadata, scenario, seed)
            result['folder'] = str((folder / name).relative_to(arm.ROOT))
            summary['cases'].append(result)
            summary['protected_files_sha256_after'] = protected_files()
            summary['protected_files_unchanged'] = before == summary['protected_files_sha256_after']
            write_json(folder / 'summary.json', summary)
            report(folder, summary)
            print(f'RESULT {name} unloaded={result["exited_count"]}/3 '
                  f'completed={result["motion_completed"]} violation={result["constraint_violation"]} '
                  f'walltime={result["walltime_s"]}s', flush=True)
    if not summary['protected_files_unchanged']:
        raise RuntimeError('Protected source/asset integrity check failed')
    print(f'REPORT {folder / "测试报告.md"}', flush=True)


if __name__ == '__main__':
    main()
