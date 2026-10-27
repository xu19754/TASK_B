"""Physics-only reference trajectory for the independent MuJoCo unload task.

The rollout is an *expert-data generator*, not a policy and not a shortcut in
the environment.  After reset it changes only the 19 actuator targets.  The
body tilt is obtained from bounded leg IK targets and therefore follows the
floating-base dynamics.  The arm path is the v3 low-edge sweep that was
previously checked against the basket geometry.  Cargo remains a free rigid
body and is never written after reset.

This module is intentionally separate from ``train.py``.  It evaluates a
candidate seed for behaviour cloning/PPO when random exploration cannot find
the long sweep (about 156 s at its original clock).  A successful geometry
screen does not establish dynamics success.  Failed rollouts are diagnostics,
never successful expert demonstrations.  Geometry/exit reports stay isolated
from actor observations.
"""
from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import mujoco
import numpy as np
import torch
from scipy.spatial.transform import Rotation
from scipy.interpolate import CubicSpline

from reports.basket_feasibility.geometry_scan import inverse_batch, leg_parameters

from .env import _hull_topology
from .metrics import TorchRearExitTracker
from .model import ROOT, build_model, prepare_targets
from .progress import update
from .reference_experiment import _clearance
from .vision import DesktopPreview


def _smooth(u: float) -> float:
    u = float(np.clip(u, 0.0, 1.0))
    return 10.0 * u**3 - 15.0 * u**4 + 6.0 * u**5


def _clock(u: float, ramp=.05) -> float:
    """Mostly linear clock with smooth zero velocity at both endpoints."""
    u = float(np.clip(u, 0., 1.))
    if u < ramp:
        integral = u / 2. - ramp * math.sin(math.pi * u / ramp) / (2. * math.pi)
    elif u > 1. - ramp:
        v = 1. - u
        integral = 1. - ramp - (v / 2. - ramp * math.sin(math.pi * v / ramp) / (2. * math.pi))
    else:
        integral = u - ramp / 2.
    return integral / (1. - ramp)


class SweepReference:
    """Stateless joint-target teacher, usable by CPU or GPU data collection.

    Construct once from the *reset* robot pose/feet, then call ``targets(t)``
    at the environment control frequency.  It consumes no cargo positions,
    contacts or exit truth.  The returned finite 19 absolute joint targets
    still need the environment's normal actuator slew limiting.  It is a
    candidate reference: callers must reject rollouts that fail actual
    dynamics/exit checks before adding them to a behaviour-cloning dataset.
    """

    def __init__(self, metadata, base_position, base_rotation, foot_positions,
                 path=ROOT / "reports/basket_v3/edge_parallel/low_edge_candidate.json",
                 pitch_deg=15., speed_scale=1., approach_rate=.225,
                 stop_fraction=.85, interpolation="linear_uniform",
                 approach_before_tilt=False):
        if speed_scale <= 0 or approach_rate <= 0:
            raise ValueError("speed_scale and approach_rate must be positive")
        if not 0 < stop_fraction <= 1:
            raise ValueError("stop_fraction must be in (0,1]")
        if interpolation not in ("linear_uniform", "source_cubic"):
            raise ValueError("interpolation must be linear_uniform or source_cubic")
        self.interpolation = interpolation
        self.meta = metadata
        self.path = Path(path)
        if not self.path.is_absolute():
            self.path = ROOT / self.path
        self.path_data = json.loads(self.path.read_text(encoding="utf-8"))
        self.nodes = np.asarray(self.path_data["q_nodes_rad"], dtype=float)
        self.nominal = np.asarray(metadata["nominal_q"], dtype=float)
        if self.nodes.ndim != 2 or self.nodes.shape[1] != 7 or not np.isfinite(self.nodes).all():
            raise ValueError("arm path must be finite [K,7]")
        if np.any(self.nodes < metadata["joint_lower"][12:]) or np.any(self.nodes > metadata["joint_upper"][12:]):
            raise ValueError("arm path exceeds the unchanged joint limits")
        self.arm_spline = CubicSpline(np.asarray(self.path_data.get(
            "parameters", np.linspace(0, 1, len(self.nodes)))), self.nodes,
            bc_type=((1, np.zeros(7)), (1, np.zeros(7))))
        # A spline may overshoot its nodes, so check its actual joint targets.
        interpolated = np.stack([self.arm_targets(parameter) for parameter in np.linspace(0, 1, 2001)])
        if np.any(interpolated < metadata["joint_lower"][12:]) or np.any(interpolated > metadata["joint_upper"][12:]):
            raise ValueError("interpolated arm path exceeds the unchanged joint limits")
        self.initial_position = np.asarray(base_position, dtype=float).copy()
        self.initial_rotation = np.asarray(base_rotation, dtype=float).reshape(3, 3).copy()
        yaw = math.atan2(self.initial_rotation[1, 0], self.initial_rotation[0, 0])
        target_rotation = (Rotation.from_euler("z", yaw).as_matrix() @
                           Rotation.from_euler("y", -math.radians(pitch_deg)).as_matrix())
        self.delta_rotation = Rotation.from_matrix(self.initial_rotation.T @ target_rotation).as_rotvec()
        self.target_position = self.initial_position + [.10, 0., 0.]
        self.target_position[2] = .575
        self.feet = np.asarray(foot_positions, dtype=float).copy()
        self.parameters = leg_parameters()
        self.source_duration = float(self.path_data.get("duration_s", 156.51))
        self.stop_fraction = float(stop_fraction)
        self.tilt_duration = 8.
        self.approach_before_tilt = bool(approach_before_tilt)
        # Quintic easing has a maximum slope of 1.875.
        self.approach_duration = max(12., float(
            1.875 * np.max(np.abs(self.nodes[0] - self.nominal[12:])) / approach_rate))
        self.tilt_start = self.approach_duration if self.approach_before_tilt else 0.
        self.approach_start = 0. if self.approach_before_tilt else self.tilt_duration + 4.
        self.sweep_start = self.tilt_duration + 4. + self.approach_duration
        self.sweep_duration = max(24., self.source_duration / speed_scale) * self.stop_fraction
        self.sweep_end = self.sweep_start + self.sweep_duration
        # Retrace the checked arm path in reverse while the body holds its
        # tilted stance.  Folding directly from the low rear pose cuts through
        # the rear sill; the complete source path's last 15% also intersects
        # it under floating-base PD tracking.  Stop before that segment.
        self.reverse_end = self.sweep_end + self.sweep_duration
        self.fold_end = self.reverse_end + self.approach_duration
        self.recover_duration = 8.
        self.duration = self.fold_end + self.recover_duration + 1.
        for fraction in np.linspace(0., 1., 101):
            self.leg_targets(float(fraction))

    def arm_targets(self, parameter):
        if self.interpolation == "source_cubic":
            return self.arm_spline(parameter)
        index = float(np.clip(parameter, 0., 1.)) * (len(self.nodes) - 1)
        lo = min(int(index), len(self.nodes) - 2)
        weight = index - lo
        return (1. - weight) * self.nodes[lo] + weight * self.nodes[lo + 1]

    def leg_targets(self, fraction):
        weight = _smooth(fraction)
        rotation = self.initial_rotation @ Rotation.from_rotvec(weight * self.delta_rotation).as_matrix()
        position = self.initial_position + weight * (self.target_position - self.initial_position)
        q, valid, margin = inverse_batch(((self.feet - position) @ rotation)[None], self.parameters)
        if not bool(valid[0]):
            raise RuntimeError(f"leg IK exceeded limits at fraction={fraction:.4f}, margin={margin[0]}")
        return q[0]

    def phase(self, elapsed):
        if self.approach_before_tilt:
            if elapsed < self.approach_duration:
                return "observe_approach"
            if elapsed < self.approach_duration + self.tilt_duration:
                return "tilt_after_look"
            if elapsed < self.sweep_start:
                return "observe_hold"
        if elapsed < self.tilt_duration:
            return "tilt"
        if elapsed < self.approach_start:
            return "hold"
        if elapsed < self.sweep_start:
            return "approach"
        if elapsed < self.sweep_end:
            return "sweep"
        if elapsed < self.reverse_end:
            return "reverse_sweep"
        if elapsed < self.fold_end:
            return "fold_arm"
        return "recover_body"

    def sweep_parameter(self, elapsed):
        """Current arm-path parameter, shared by the forward/reverse clock.

        Outside the sweep it is zero, matching the first arm node used for
        approach and folding.  Callers can use this parameter to attach a
        reversible tool-height profile without duplicating the clock.
        """
        if elapsed < self.sweep_start or elapsed >= self.reverse_end:
            return 0.
        if elapsed < self.sweep_end:
            parameter = self.stop_fraction * _clock((elapsed - self.sweep_start) / self.sweep_duration)
        else:
            parameter = self.stop_fraction * (1. - _clock((elapsed - self.sweep_end) / self.sweep_duration))
        return float(np.clip(parameter, 0., self.stop_fraction))

    def targets(self, elapsed):
        if elapsed < self.tilt_start + self.tilt_duration:
            body_fraction = (elapsed - self.tilt_start) / self.tilt_duration
        elif elapsed < self.fold_end:
            body_fraction = 1.
        else:
            body_fraction = 1. - (elapsed - self.fold_end) / self.recover_duration
        targets = self.nominal.copy()
        targets[:12] = self.leg_targets(float(np.clip(body_fraction, 0., 1.)))
        if elapsed < self.approach_start:
            pass
        elif elapsed < self.sweep_start:
            targets[12:] += _smooth((elapsed - self.approach_start) / self.approach_duration) * (
                self.nodes[0] - self.nominal[12:])
        elif elapsed < self.reverse_end:
            targets[12:] = self.arm_targets(self.sweep_parameter(elapsed))
        elif elapsed < self.fold_end:
            targets[12:] = self.nodes[0] + _smooth((elapsed - self.reverse_end) / self.approach_duration) * (
                self.nominal[12:] - self.nodes[0])
        else:
            targets[12:] = self.nominal[12:]
        return targets

    def preflight(self, model, initial_qpos, period=.05):
        """Isolated kinematic rejection screen; never used to claim success.

        Root and arm writes occur only in this separate planning ``MjData``.
        The returned minimum checks the complete forward *and* recovery path.
        Actual rollout state is never changed by this method.
        """
        planning = mujoco.MjData(model)
        planning.qpos[:] = initial_qpos
        minimum, first_bad = .02, None
        for elapsed in np.arange(0., self.duration + period, period):
            planning.qpos[self.meta["joint_qpos_adr"]] = self.targets(float(elapsed))
            mujoco.mj_forward(model, planning)
            distance, pair = _clearance(model, planning, self.meta)
            minimum = min(minimum, distance)
            if distance < .001:
                first_bad = dict(time_s=float(elapsed), phase=self.phase(elapsed),
                                 distance_m=distance, pair=pair)
                break
        return dict(valid=first_bad is None, minimum_clearance_m=minimum,
                    rejection=first_bad, sample_period_s=period,
                    scope="kinematic forbidden-pair preflight only; not dynamics acceptance")


def _vertices(data, model, meta, hulls):
    rotation = data.xmat[meta["basket_id"]].reshape(3, 3)
    max_vertices = max(len(source) for source in hulls)
    padded = np.zeros((3, max_vertices, 3), dtype=np.float32)
    for index, (source, body) in enumerate(zip(hulls, meta["cargo_body_ids"])):
        transformed = (source @ data.xmat[body].reshape(3, 3).T + data.xpos[body]
                       - data.xpos[meta["basket_id"]]) @ rotation
        padded[index, :len(source)] = transformed
    return torch.from_numpy(padded[None])


def run(output: str | Path,
        bank_path: str | Path = ROOT / "runs/basket_unload/reset_bank_collected.npz",
        path: str | Path = ROOT / "reports/basket_v3/edge_parallel/low_edge_candidate.json",
        record_index: int | None = None,
        pitch_deg: float = 15.0,
        speed_scale: float = 1.0,
        preview: bool = True,
        preview_period_s: float = 1.0,
        max_seconds: float | None = None,
        bias_compensation: bool = False,
        stop_fraction: float = .85) -> dict:
    """Run one mixed-three reset using only actuator target commands.

    ``speed_scale`` changes the clock of the existing path.  Actual joint
    velocities, including externally driven gripper motion, remain monitored.
    """
    if speed_scale <= 0:
        raise ValueError("speed_scale must be positive")
    # Small per-world tensors are slower with a large OpenMP thread pool.
    torch.set_num_threads(1)
    model, meta = build_model()
    bank_path = Path(bank_path)
    if not bank_path.is_absolute():
        bank_path = ROOT / bank_path
    with np.load(bank_path) as bank:
        candidates = np.flatnonzero(np.all(bank["combination"] == [0, 1, 2], axis=1))
        if not len(candidates):
            raise ValueError("bank has no mixed-three [0,1,2] reset")
        index = int(candidates[0] if record_index is None else record_index)
        if not np.array_equal(bank["combination"][index], [0, 1, 2]):
            raise ValueError("record_index must select a mixed-three reset")
        data = mujoco.MjData(model)
        data.qpos[:] = bank["qpos"][index]
        data.qvel[:] = bank["qvel"][index]
        data.ctrl[:] = bank["ctrl"][index]
        seed = int(bank["seed"][index])
    # This is reset initialization only.  There are no qpos/qvel writes below.
    mujoco.mj_forward(model, data)
    reference = SweepReference(meta, data.qpos[:3], data.xmat[meta["base_id"]],
                               data.xpos[meta["foot_body_ids"]], path, pitch_deg, speed_scale,
                               stop_fraction=stop_fraction)

    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    preflight = reference.preflight(model, data.qpos)
    (output / "preflight.json").write_text(json.dumps(preflight, indent=2), encoding="utf-8")
    if not preflight["valid"]:
        raise RuntimeError(f"reference rejected by forbidden-pair preflight: {preflight['rejection']}")
    renderer = DesktopPreview(model, meta, output) if preview else None
    hulls, masks, edges, edge_masks = _hull_topology(meta["cargo_vertices"])
    tracker = TorchRearExitTracker()
    vertices = _vertices(data, model, meta, [hulls[i, masks[i]] for i in range(3)])
    result = tracker.reset(vertices, torch.as_tensor(masks), torch.as_tensor(edges), torch.as_tensor(edge_masks))
    if bool(result.invalid_initial.any()):
        raise RuntimeError("reset bank rejected by unchanged tail-window predicate")

    total = reference.duration
    if max_seconds is not None:
        total = min(total, float(max_seconds))
    previous = meta["nominal_q"].copy()
    max_tilt = 0.0
    max_speed_ratio = 0.0
    min_clearance = 0.02
    dangerous_contacts = 0
    exited_at = {}
    next_preview = 0.0
    floor_id = int(model.geom("floor").id)
    robot_body_ids = set(model.geom_bodyid[meta["collision_geom_ids"]])
    foot_body_ids = set(meta["foot_body_ids"])
    failure = None
    trace = []
    # Keep the trajectory guard at the 2 ms physics rate.  Even already-exited
    # objects can reach 2--3 m/s during a free fall; 10 ms samples could exceed
    # the unchanged 20 mm displacement guard and wrongly invalidate them.
    metric_stride = 1
    update("独立专家轨迹：开始", "只发送19关节目标；浮动基座由腿部IK和接触动力学响应。", output)
    try:
        for step in range(round(total / model.opt.timestep)):
            elapsed = step * model.opt.timestep
            request = reference.targets(elapsed)
            if bias_compensation:
                # The historic arm diagnostic used model gravity feedforward.
                # A native position actuator can express the same finite
                # feedforward as a target offset, without changing torque caps
                # or injecting an external force.  Robot bias depends on robot
                # kinematics; cargo truth is neither read nor used here.
                request[12:] += (data.qfrc_bias[meta["joint_dof_adr"]][12:]
                                 / meta["kp"][12:])
            # Target preparation is the only control write after reset.
            if step % 10 == 0:
                previous = prepare_targets(meta, request, previous,
                                           data.qpos[meta["joint_qpos_adr"]], model.opt.timestep * 10)
                data.ctrl[:] = previous
            mujoco.mj_step(model, data)
            mujoco.mj_forward(model, data)

            rotation = data.xmat[meta["base_id"]].reshape(3, 3)
            tilt = math.acos(float(np.clip(rotation[2, 2], -1.0, 1.0)))
            max_tilt = max(max_tilt, tilt)
            speed_ratio = float(np.max(np.abs(data.qvel[meta["joint_dof_adr"]]) / meta["velocity"]))
            max_speed_ratio = max(max_speed_ratio, speed_ratio)
            for contact in data.contact:
                a, b = int(contact.geom[0]), int(contact.geom[1])
                body = int(model.geom_bodyid[b] if a == floor_id else model.geom_bodyid[a])
                if (a == floor_id or b == floor_id) and body in robot_body_ids - foot_body_ids:
                    dangerous_contacts += 1
            # Tracker is sampled at the physics timestep, so no crossing can
            # be skipped.  A failed physical state latches dynamics invalid.
            actual_joints = data.qpos[meta["joint_qpos_adr"]]
            if (not np.isfinite(data.qpos).all() or not np.isfinite(data.qvel).all()
                    or np.any(actual_joints < meta["joint_lower"] - .02)
                    or np.any(actual_joints > meta["joint_upper"] + .02)):
                failure = failure or dict(reason="nonfinite_or_joint_limit", time_s=float(data.time))
            if dangerous_contacts:
                failure = failure or dict(reason="dangerous_ground_contact", time_s=float(data.time))
            if tilt > math.radians(55.0) or data.qpos[2] < .28 or speed_ratio > 1.02:
                fastest = int(np.argmax(np.abs(data.qvel[meta["joint_dof_adr"]]) / meta["velocity"]))
                failure = failure or dict(reason="unstable_or_overspeed", time_s=float(data.time),
                                          tilt_deg=math.degrees(tilt), speed_ratio=speed_ratio,
                                          phase=reference.phase(elapsed),
                                          joint=meta["joint_names"][fastest],
                                          actual_joint_velocity=data.qvel[meta["joint_dof_adr"]].tolist())
            if (step + 1) % metric_stride == 0:
                clearance, pair = _clearance(model, data, meta)
                min_clearance = min(min_clearance, clearance)
                if clearance < -.0005:
                    failure = failure or dict(reason="forbidden_intersection", time_s=float(data.time),
                                              clearance_m=clearance, pair=pair)
                current = _vertices(data, model, meta, [hulls[i, masks[i]] for i in range(3)])
                valid = failure is None and dangerous_contacts == 0
                stable = (tilt < math.radians(10.0) and
                          np.linalg.norm(data.qvel[:6]) < .2 and
                          np.max(np.abs(data.qpos[meta["joint_qpos_adr"]][12:] - meta["nominal_q"][12:])) < .2)
                result = tracker.update(current, model.opt.timestep * metric_stride,
                                        stable=torch.tensor([bool(stable)]),
                                        valid_dynamics=torch.tensor([valid]))
                for obj in np.flatnonzero(result.first_exit[0].numpy()):
                    exited_at.setdefault(str(int(obj)), float(data.time))
                if bool((result.wrong | result.reentered | result.invalid_trajectory).any()):
                    failure = failure or dict(reason="invalid_tail_window_trajectory", time_s=float(data.time),
                                              wrong=result.wrong[0].tolist(),
                                              reentered=result.reentered[0].tolist(),
                                              invalid_trajectory=result.invalid_trajectory[0].tolist())
            if step % 10 == 0:
                if renderer is not None and elapsed + 1e-9 >= next_preview:
                    renderer.update_cpu(data, metrics=dict(expert=True, time_s=float(data.time),
                                                            tilt_deg=math.degrees(tilt),
                                                            exited=int(result.exited.sum())))
                    next_preview += preview_period_s
                if step % 100 == 0:
                    trace.append(dict(time_s=float(data.time), tilt_deg=math.degrees(tilt),
                                      exited=int(result.exited.sum()), wrong=int(result.wrong.sum()),
                                      speed_ratio=speed_ratio))
            if failure is not None:
                break
        report = dict(engine=f"MuJoCo {mujoco.__version__}", reference_only=True,
                      rl_policy=False, control="19 actuator targets only; floating-base body via leg IK",
                      reset_bank=str(bank_path), reset_index=index, reset_seed=seed,
                      bias_compensation=bias_compensation,
                      stop_fraction=reference.stop_fraction,
                      interpolation=reference.interpolation,
                      arm_path=str(reference.path), source_duration_s=reference.source_duration,
                      speed_scale=speed_scale, simulated_time_s=float(data.time),
                      correctly_exited=int(result.exited.sum()), wrong_exit=int(result.wrong.sum()),
                      exited_at_s=exited_at, success=bool(result.success[0]),
                      max_tilt_deg=math.degrees(max_tilt), max_joint_speed_ratio=max_speed_ratio,
                      minimum_clearance_m=min_clearance, dangerous_contact_samples=dangerous_contacts,
                      preflight=preflight,
                      invalid_trajectory=result.invalid_trajectory[0].tolist(),
                      invalid_dynamics=bool(result.invalid_dynamics[0]),
                      failure=failure, trace=trace,
                      scope="MuJoCo reference/expert data only; not Isaac/PhysX validation")
        (output / "report.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
        update("独立专家轨迹完成", f"正确尾窗 {report['correctly_exited']}/3；该结果仅用于专家数据，不是RL成绩。", output)
        return report
    finally:
        if renderer is not None:
            renderer.close()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, default=ROOT / "runs/basket_unload/expert_rollout")
    parser.add_argument("--bank", type=Path, default=ROOT / "runs/basket_unload/reset_bank_collected.npz")
    parser.add_argument("--path", type=Path, default=ROOT / "reports/basket_v3/edge_parallel/low_edge_candidate.json")
    parser.add_argument("--record-index", type=int)
    parser.add_argument("--pitch-deg", type=float, default=15.0)
    parser.add_argument("--speed-scale", type=float, default=1.0)
    parser.add_argument("--max-seconds", type=float)
    parser.add_argument("--no-preview", action="store_true")
    parser.add_argument("--bias-compensation", action="store_true")
    parser.add_argument("--stop-fraction", type=float, default=.85)
    args = parser.parse_args()
    result = run(args.output, args.bank, args.path, args.record_index, args.pitch_deg,
                 args.speed_scale, not args.no_preview, max_seconds=args.max_seconds,
                 bias_compensation=args.bias_compensation, stop_fraction=args.stop_fraction)
    print(json.dumps({k: v for k, v in result.items() if k != "trace"}, indent=2), flush=True)
