"""Read-only-in-physics millisecond audit of the seven recorded speed stops.

Replays the exact layouts, trajectory, motor caps and contacts. The observer
copies mjData around mj_step without changing the original physics state.
Local timestep sensitivity runs use separate in-memory models and are not
counted as successful task/path tests.
"""
from pathlib import Path
import argparse
import copy
import json
import math
import os

os.environ['TASKB_BASKET_REVISION'] = 'v3'

import mujoco
import numpy as np
import basket_random_trials as trials

SOURCE = trials.arm.OUT / 'random_three_2026-10-26'
OUT = trials.arm.OUT / 'speed_audit_2026-10-26'


def contacts(model, data, gripper_dof):
    rows = []
    mask = np.zeros(data.nefc)
    contribution = np.zeros(model.nv)
    for i, contact in enumerate(data.contact):
        names = [model.geom(int(g)).name for g in contact.geom]
        if not any(n.startswith('collision_gripperMover_') for n in names):
            continue
        force = np.empty(6)
        mujoco.mj_contactForce(model, data, i, force)
        mask[:] = 0
        start = int(contact.efc_address)
        if start >= 0:
            stop = start + int(contact.dim)
            mask[start:stop] = data.efc_force[start:stop]
            mujoco.mj_mulJacTVec(model, data, contribution, mask)
            torque = float(contribution[gripper_dof])
        else:
            torque = 0.
        rows.append(dict(geoms=names, distance_m=float(contact.dist),
                         normal_force_N=float(force[0]),
                         gripper_generalized_torque_Nm=torque))
    return rows


def frame(sim, before, after):
    j = int(sim.arm_dadr[-1])
    qa = int(sim.arm_qadr[-1])
    a = int(sim.arm_motor[-1])
    return dict(time_before_s=float(before.time), time_after_s=float(after.time),
                q_before_rad=float(before.qpos[qa]), q_after_rad=float(after.qpos[qa]),
                v_before_rad_s=float(before.qvel[j]), v_after_rad_s=float(after.qvel[j]),
                command_rad=float(sim.arm_command[-1]), control_rad=float(before.ctrl[a]),
                command_error_rad=float(sim.arm_command[-1] - before.qpos[qa]),
                proportional_torque_Nm=float(trials.arm.KP[-1] * (before.ctrl[a] - before.qpos[qa])),
                damping_torque_Nm=float(-trials.arm.KD[-1] * before.qvel[j]),
                motor_torque_used_Nm=float(after.actuator_force[a]),
                constraint_torque_Nm=float(after.qfrc_constraint[j]),
                bias_torque_Nm=float(after.qfrc_bias[j]),
                smooth_accel_rad_s2=float(after.qacc_smooth[j]),
                constrained_accel_rad_s2=float(after.qacc[j]),
                velocity_delta_rad_s=float(after.qvel[j] - before.qvel[j]),
                grip_contacts=contacts(sim.model, after, j),
                q_all_before=before.qpos[sim.arm_qadr].tolist(),
                v_all_before=before.qvel[sim.arm_dadr].tolist(),
                v_all_after=after.qvel[sim.arm_dadr].tolist())


def sensitivity(sim, before, levels=(1, 2, 4, 8)):
    results = []
    for substeps in levels:
        model = copy.copy(sim.model)
        model.opt.timestep = trials.DT / substeps
        data = mujoco.MjData(model)
        mujoco.mj_copyData(data, model, before)
        peak = abs(float(data.qvel[sim.arm_dadr[-1]]))
        for _ in range(substeps):
            mujoco.mj_step(model, data)
            peak = max(peak, abs(float(data.qvel[sim.arm_dadr[-1]])))
            mujoco.mj_forward(model, data)
        results.append(dict(timestep_s=float(model.opt.timestep), substeps=substeps,
                            horizon_s=trials.DT, end_velocity_rad_s=float(data.qvel[sim.arm_dadr[-1]]),
                            peak_speed_rad_s=peak))
    return results


def audit(result):
    source = trials.arm.ROOT / result['folder']
    layout = json.loads((source / 'layout.json').read_text())
    sim = trials.make_sim(result['scenario'])
    sim.layout = layout['layout']
    sim.apply_layout(sim.layout)
    expected = result['rejected_next_step']
    cutoff = expected['time_s'] - .201
    before, after = mujoco.MjData(sim.model), mujoco.MjData(sim.model)
    native_step = mujoco.mj_step
    trace = []

    def observed_step(model, data, *args, **kwargs):
        if data is sim.data and data.time >= cutoff:
            mujoco.mj_copyData(before, model, data)
            native_step(model, data, *args, **kwargs)
            mujoco.mj_copyData(after, model, data)
            trace.append(frame(sim, before, after))
        else:
            native_step(model, data, *args, **kwargs)

    mujoco.mj_step = observed_step
    try:
        for _ in range(2000):
            sim.step()
        assert not sim.violation
        assert sim.edge.play_candidate(), sim.message
        for _ in range(math.ceil((expected['time_s'] + .1 - 2) / trials.DT)):
            sim.step()
            if sim.violation:
                break
    finally:
        mujoco.mj_step = native_step
    assert sim.violation == result['constraint_violation']
    assert abs(sim.rejected_step['time_s'] - expected['time_s']) < 1e-8
    assert np.allclose(sim.rejected_step['velocity_rad_s'], expected['velocity_rad_s'], atol=1e-9)
    assert trace and abs(trace[-1]['time_after_s'] - expected['time_s']) < 1e-8
    inertia = np.empty((sim.model.nv, sim.model.nv))
    mujoco.mj_fullM(sim.model, after, inertia)
    j = sim.arm_dadr[-1]
    effective_inertia = float(1 / np.linalg.inv(inertia)[j, j])
    state = np.empty(mujoco.mj_stateSize(sim.model, sim.state_spec))
    mujoco.mj_getState(sim.model, before, state, sim.state_spec)
    prefix = OUT / str(result['seed'])
    trials.write_json(prefix.with_suffix('.json'), dict(seed=result['seed'], scenario=result['scenario'],
                      last_200ms=trace, exact_original_rejection_reproduced=True))
    np.savez_compressed(prefix.with_suffix('.npz'), integration_state=state,
                        state_spec=int(sim.state_spec), qpos=before.qpos.copy(),
                        qvel=before.qvel.copy(), ctrl=before.ctrl.copy())
    substep_results = sensitivity(sim, before)
    assert abs(substep_results[0]['end_velocity_rad_s'] - expected['velocity_rad_s'][-1]) < 1e-9
    row = dict(seed=result['seed'], scenario=result['scenario'], label=result['label'],
               exact_original_rejection_reproduced=True, final_step=trace[-1],
               preceding_step=trace[-2], gripper_effective_inertia_kg_m2=effective_inertia,
               damping_time_constant_s=effective_inertia / trials.arm.KD[-1],
               free_servo_velocity_estimate_rad_s=trials.arm.KP[-1] / trials.arm.KD[-1]
                                                * trace[-1]['command_error_rad'],
               local_timestep_sensitivity=substep_results)
    print(json.dumps(dict(seed=row['seed'], before_v=trace[-1]['v_before_rad_s'],
                          after_v=trace[-1]['v_after_rad_s'], error=trace[-1]['command_error_rad'],
                          used_motor=trace[-1]['motor_torque_used_Nm'],
                          constraint=trace[-1]['constraint_torque_Nm'],
                          contacts=trace[-1]['grip_contacts'],
                          sensitivity=substep_results), ensure_ascii=False), flush=True)
    return row


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--refine', action='store_true', help='Refine saved one-millisecond snapshots only')
    args = parser.parse_args()
    OUT.mkdir(exist_ok=True)
    before = trials.protected_files()
    if args.refine:
        audit_result = json.loads((OUT / 'audit.json').read_text())
        for row in audit_result['cases']:
            sim = trials.make_sim(row['scenario'])
            snapshot = np.load(OUT / f"{row['seed']}.npz")
            data = mujoco.MjData(sim.model)
            mujoco.mj_setState(sim.model, data, snapshot['integration_state'], int(snapshot['state_spec']))
            original = sensitivity(sim, data, (1,))[0]
            assert abs(original['end_velocity_rad_s'] - row['final_step']['v_after_rad_s']) < 1e-9
            refined = sensitivity(sim, data, (16, 32, 64))
            row['local_timestep_sensitivity'] = [x for x in row['local_timestep_sensitivity'] if x['substeps'] < 16] + refined
            print(json.dumps(dict(seed=row['seed'], refined=refined)), flush=True)
        assert before == trials.protected_files()
        audit_result['protected_files_unchanged'] = True
        trials.write_json(OUT / 'audit.json', audit_result)
        return
    summary = json.loads((SOURCE / 'summary.json').read_text())
    results = []
    for result in summary['cases']:
        if result['rejected_next_step']:
            results.append(audit(result))
            trials.write_json(OUT / 'audit.json', dict(cases=results,
                              protected_files_unchanged=before == trials.protected_files(),
                              scope='Same-layout baseline replay plus separate local timestep sensitivity; no path repairs'))
    assert before == trials.protected_files()


if __name__ == '__main__':
    main()
