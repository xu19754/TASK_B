"""Generate independently settled single-cargo curriculum reset states on CPU.

Only reset initialization writes cargo poses. Other cargo is parked before the
one active item is dropped; all subsequent settling uses free-base dynamics.
This fixes removal of support from three-item reset banks, not policy success.
"""
from __future__ import annotations

import argparse
from collections import Counter
import hashlib
import json
import math
from pathlib import Path

import mujoco
import numpy as np
import torch

from .assets_gpu import orientation_distribution
from .env import _hull_topology
from .metrics import ExitGeometry, TorchRearExitTracker
from .model import OBJECT_NAMES, ROOT, _quaternion, build_model, initialize_data


def generate(output: str | Path, *, seeds_per_class: int = 16,
             seed_offset: int, settle_seconds: float = 3.,
             max_attempts: int = 100) -> dict:
    if seeds_per_class < 1 or max_attempts < 1 or settle_seconds <= 0:
        raise ValueError("Seed count, attempt limit and settling duration must be positive")
    torch.set_num_threads(1)
    output = Path(output)
    output.parent.mkdir(parents=True, exist_ok=True)
    records, diagnostics, attempts = [], [], []
    rejection_counts = Counter()
    geometry = ExitGeometry.from_v3_properties(ROOT / "assets/b2z1/basket_v3/basket_properties.json")
    final_quaternions = []
    for cls in range(3):
        combination = (cls, cls, cls)
        model, meta = build_model(combination)
        vertices, mask, edges, edge_mask = _hull_topology(meta["cargo_vertices"])
        tracker = TorchRearExitTracker(geometry)
        cargo_id = int(meta["cargo_body_ids"][0])
        cargo_qpos = int(meta["cargo_qpos_adr"][0])
        cargo_dof = int(meta["cargo_dof_adr"][0])
        floor = int(model.geom("floor").id)
        forbidden = set(model.geom_bodyid[meta["collision_geom_ids"]]) - set(meta["foot_body_ids"])

        def local_vertices(data):
            rotation = data.xmat[meta["basket_id"]].reshape(3, 3)
            return ((vertices[0] @ data.xmat[cargo_id].reshape(3, 3).T
                     + data.xpos[cargo_id] - data.xpos[meta["basket_id"]]) @ rotation)

        for seed_index in range(seeds_per_class):
            seed = seed_offset + cls * 10000 + seed_index
            rng = np.random.default_rng(seed)
            accepted = False
            for attempt in range(1, max_attempts + 1):
                data = initialize_data(model, meta)
                # Reset-only parking happens before the active cargo is placed.
                for slot in (1, 2):
                    adr = int(meta["cargo_qpos_adr"][slot])
                    dof = int(meta["cargo_dof_adr"][slot])
                    data.qpos[adr:adr + 3] = [2. + slot * .4, 2., .2]
                    data.qvel[dof:dof + 6] = 0
                mujoco.mj_forward(model, data)
                quaternion = rng.normal(size=4)
                quaternion /= np.linalg.norm(quaternion)
                rotation = np.empty(9)
                mujoco.mju_quat2Mat(rotation, quaternion)
                rotation = rotation.reshape(3, 3)
                transformed = meta["cargo_vertices"][0] @ rotation.T
                x_max = min(.09, .21 - float(np.max(np.abs(transformed[:, 0]))))
                y_max = min(.02, .115 - float(np.max(np.abs(transformed[:, 1]))))
                rejected = None
                if min(x_max, y_max) <= 0:
                    rejected = "initial_projection_outside"
                else:
                    local = np.array([rng.uniform(-x_max, x_max), rng.uniform(-y_max, y_max),
                                      .16 + rng.uniform(.025, .055) - transformed[:, 2].min()])
                    basket_rotation = data.xmat[meta["basket_id"]].reshape(3, 3)
                    data.qpos[cargo_qpos:cargo_qpos + 3] = (
                        data.xpos[meta["basket_id"]] + basket_rotation @ local)
                    data.qpos[cargo_qpos + 3:cargo_qpos + 7] = _quaternion(basket_rotation @ rotation)
                    data.qvel[cargo_dof:cargo_dof + 6] = 0
                    mujoco.mj_forward(model, data)
                    max_tilt, min_height, dangerous = 0., float("inf"), 0
                    # Same .9 s post-drop interval and 3 s settling as the
                    # collected bank, with only one active cargo throughout.
                    for _ in range(round((.9 + settle_seconds) / model.opt.timestep)):
                        mujoco.mj_step(model, data)
                        upright = data.xmat[meta["base_id"]].reshape(3, 3)[2, 2]
                        max_tilt = max(max_tilt, math.degrees(math.acos(np.clip(upright, -1, 1))))
                        min_height = min(min_height, float(data.qpos[2]))
                        for contact in data.contact:
                            a, b = contact.geom
                            if ((a == floor and model.geom_bodyid[b] in forbidden)
                                    or (b == floor and model.geom_bodyid[a] in forbidden)):
                                dangerous += 1
                    mujoco.mj_forward(model, data)
                    result = tracker.reset(
                        torch.from_numpy(local_vertices(data)[None, None]),
                        torch.from_numpy(mask[:1][None]), torch.from_numpy(edges[:1][None]),
                        torch.from_numpy(edge_mask[:1][None]))
                    speed = float(np.linalg.norm(data.qvel[cargo_dof:cargo_dof + 3]))
                    finite = bool(np.isfinite(data.qpos).all() and np.isfinite(data.qvel).all())
                    reasons = []
                    if not finite:
                        reasons.append("nonfinite")
                    if bool(result.invalid_initial.any()):
                        reasons.append("cargo_outside")
                    if max_tilt >= 10 or min_height <= .35 or dangerous:
                        reasons.append("unstable_body")
                    if speed >= .08:
                        reasons.append("cargo_not_settled")
                    rejected = ",".join(reasons) if reasons else None
                    detail = dict(max_tilt_deg=max_tilt, min_base_height_m=min_height,
                                  dangerous_ground_contact_samples=dangerous,
                                  final_cargo_linear_speed_m_s=speed)
                if rejected:
                    rejection_counts.update(rejected.split(","))
                    attempts.append(dict(seed=seed, class_id=cls, attempt=attempt,
                                         accepted=False, reasons=rejected.split(",")))
                    continue
                local_rotation = (data.xmat[meta["basket_id"]].reshape(3, 3).T
                                  @ data.xmat[cargo_id].reshape(3, 3))
                final_quaternion = _quaternion(local_rotation).tolist()
                final_quaternions.append(final_quaternion)
                records.append((data.qpos.copy(), data.qvel.copy(), data.ctrl.copy(), combination, seed))
                diagnostics.append(dict(seed=seed, class_id=cls, object_name=OBJECT_NAMES[cls],
                                        attempts=attempt, source_quaternion_wxyz=quaternion.tolist(),
                                        source_center_basket_m=local.tolist(),
                                        final_quaternion_basket_wxyz=final_quaternion,
                                        final_center_basket_m=((data.xpos[cargo_id] - data.xpos[meta["basket_id"]])
                                                              @ data.xmat[meta["basket_id"]].reshape(3, 3)).tolist(),
                                        **detail))
                attempts.append(dict(seed=seed, class_id=cls, attempt=attempt, accepted=True, reasons=[]))
                print(f"single reset {len(records)}/{3 * seeds_per_class}: class={cls}, seed={seed}, attempts={attempt}", flush=True)
                accepted = True
                break
            if not accepted:
                failure = dict(status="failed_seed_attempt_limit", seed=seed, class_id=cls,
                               max_attempts=max_attempts, attempt_records=attempts)
                output.with_suffix(".failed.json").write_text(json.dumps(failure, indent=2), encoding="utf-8")
                raise RuntimeError(f"Could not settle single-cargo reset for class={cls}, seed={seed}")
    temporary = output.with_name(output.stem + ".tmp.npz")
    np.savez_compressed(temporary, qpos=np.stack([r[0] for r in records]),
                        qvel=np.stack([r[1] for r in records]), ctrl=np.stack([r[2] for r in records]),
                        combination=np.asarray([r[3] for r in records], dtype=np.int32),
                        seed=np.asarray([r[4] for r in records], dtype=np.int64))
    temporary.replace(output)
    report = dict(engine=f"MuJoCo {mujoco.__version__} CPU", gpu_started=False,
                  asset="assets/b2z1/b2z1_0815_with_basket_v3.urdf", count=len(records),
                  num_active_objects=1, seeds_per_class=seeds_per_class, seed_offset=seed_offset,
                  settle_seconds=settle_seconds, post_drop_seconds=.9, max_attempts=max_attempts,
                  rejection_reason_counts=dict(rejection_counts), attempt_records=attempts,
                  initialization="reset-only single SO(3) free drop near basket center, with slots1/2 parked before placement; then free-base dynamics under unchanged nominal 19-joint PD",
                  acceptance="Unchanged tracker.reset predicate on active full convex hull; body tilt<10deg, height>.35m, no dangerous ground contacts, cargo linear speed<.08m/s",
                  pose_ranges="Uniform SO(3), x up to +/- .09m and y up to +/- .02m clipped by projected source bounds; bottom .025..055m above .16m rim",
                  combination_semantics="All slots use active object's class; only slot0 is in the basket, other free bodies remain dynamic on distant floor",
                  scope="Single-item curriculum initialization repair only; new seed distribution, preserve old results and do not pool success rates; no policy success claim",
                  bank_sha256=hashlib.sha256(output.read_bytes()).hexdigest(),
                  final_orientation_distribution=orientation_distribution(final_quaternions), records=diagnostics)
    output.with_suffix(".json").write_text(json.dumps(report, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    return report


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--seed-offset", type=int, required=True)
    parser.add_argument("--seeds-per-class", type=int, default=16)
    parser.add_argument("--settle-seconds", type=float, default=3.)
    parser.add_argument("--max-attempts", type=int, default=100)
    args = parser.parse_args()
    result = generate(**vars(args))
    print(json.dumps({key: value for key, value in result.items()
                      if key not in ("records", "attempt_records")}, indent=2), flush=True)
