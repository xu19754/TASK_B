"""Independent batched whole-body visual unloading environment.

Actor observations contain only wrist RGB-D, joint/body feedback and previous
actions. Geometry, contacts and cargo state remain inside rewards/diagnostics.
All qpos/qvel assignments are reset-only; normal steps only send joint targets.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass, replace
from pathlib import Path

import mujoco_warp as mjw
import numpy as np
from scipy.spatial import ConvexHull
import torch
import warp as wp

from .assets_gpu import AssetBank
from .gpu import GPUSimulation
from .metrics import ExitGeometry, ExitResult, TorchRearExitTracker, world_to_basket
from .model import ROOT, build_model
from .reward import RewardWeights, compute_reward
from .vision import GPUCamera


@dataclass
class EnvConfig:
    num_envs: int = 16
    timestep: float = .002
    control_dt: float = .02
    camera_period: float = .1
    image_width: int = 128
    image_height: int = 96
    episode_seconds: float = 180.
    # Collected bank uses independent random SO(3) cargo orientations and
    # sequential physical settling; the near-flat bank remains available for
    # the initial geometry smoke test only.
    reset_bank: str = "runs/basket_unload/reset_bank_collected.npz"
    seed: int = 42
    stage: str = "unload"
    num_objects: int = 3
    leg_target_rate: float = .2
    arm_target_rate: float = .35
    clearance_period: float = .01
    sequential_reset: bool = False
    action_mode: str = "delta"
    observe_time: bool = False
    reference_targets: str | None = None
    reference_stride: int = 1
    leg_residual_scale: float = .04
    arm_residual_scale: float = .08
    gripper_residual_scale: float = .04
    clearance_penalty_weight: float = 0.
    speed_margin_penalty_weight: float = 0.

    def __post_init__(self):
        if self.stage not in ("balance", "unload"):
            raise ValueError("stage must be balance or unload")
        if self.action_mode not in ("delta", "position", "reference_residual"):
            raise ValueError("action_mode must be delta, position or reference_residual")
        if self.action_mode == "reference_residual" and not self.reference_targets:
            raise ValueError("reference_residual requires reference_targets")
        if (not isinstance(self.reference_stride, int) or isinstance(self.reference_stride, bool)
                or self.reference_stride < 1):
            raise ValueError("reference_stride must be a positive integer")
        if not all(np.isfinite(scale) and scale >= 0 for scale in (
                self.leg_residual_scale, self.arm_residual_scale, self.gripper_residual_scale)):
            raise ValueError("Residual scales must be finite and nonnegative")
        if not all(np.isfinite(weight) and weight >= 0 for weight in (
                self.clearance_penalty_weight, self.speed_margin_penalty_weight)):
            raise ValueError("Soft safety penalty weights must be finite and nonnegative")
        if self.num_objects not in (1, 3):
            raise ValueError("num_objects must be 1 or 3")
        if min(self.episode_seconds, self.leg_target_rate, self.arm_target_rate) <= 0:
            raise ValueError("Episode duration and target rates must be positive")
        for period in (self.control_dt, self.camera_period, self.clearance_period):
            if not np.isclose(period / self.timestep, round(period / self.timestep)):
                raise ValueError("Periods must be integer multiples of physics timestep")


def _hull_topology(source_vertices):
    hulls, edges = [], []
    for source in source_vertices:
        hull = ConvexHull(source)
        selected = np.unique(hull.simplices)
        remap = np.full(len(source), -1, dtype=int)
        remap[selected] = np.arange(len(selected))
        triangles = remap[hull.simplices]
        pairs = np.concatenate((triangles[:, [0, 1]], triangles[:, [1, 2]], triangles[:, [2, 0]]))
        hulls.append(source[selected])
        edges.append(np.unique(np.sort(pairs, axis=1), axis=0))
    max_vertices, max_edges = max(map(len, hulls)), max(map(len, edges))
    vertices = np.zeros((3, max_vertices, 3), dtype=np.float32)
    edge_array = np.zeros((3, max_edges, 2), dtype=np.int64)
    vertex_mask = np.zeros((3, max_vertices), dtype=bool)
    edge_mask = np.zeros((3, max_edges), dtype=bool)
    for cls in range(3):
        vertices[cls, :len(hulls[cls])] = hulls[cls]
        vertex_mask[cls, :len(hulls[cls])] = True
        edge_array[cls, :len(edges[cls])] = edges[cls]
        edge_mask[cls, :len(edges[cls])] = True
    return vertices, vertex_mask, edge_array, edge_mask


class BasketUnloadEnv:
    def __init__(self, config=None):
        self.config = config or EnvConfig()
        cfg = self.config
        self.reward_weights = RewardWeights(clearance=cfg.clearance_penalty_weight,
                                           speed_margin=cfg.speed_margin_penalty_weight)
        self.num_envs, self.device = cfg.num_envs, torch.device("cuda:0")
        self.generator = torch.Generator(device=self.device).manual_seed(cfg.seed)
        model, meta = build_model(timestep=cfg.timestep)
        if cfg.action_mode == "reference_residual":
            reference_path = Path(cfg.reference_targets)
            reference_path = reference_path if reference_path.is_absolute() else ROOT / reference_path
            with np.load(reference_path, allow_pickle=False) as source:
                if not {"targets", "control_dt"}.issubset(source.files):
                    raise ValueError("Reference NPZ requires targets and control_dt")
                reference_targets = source["targets"]
                reference_dt = source["control_dt"]
            if (reference_targets.dtype != np.float32 or reference_targets.ndim != 2
                    or reference_targets.shape[1] != 19 or reference_targets.shape[0] < 2
                    or not np.isfinite(reference_targets).all()):
                raise ValueError("Reference targets must be finite float32 [T,19] with T >= 2")
            if (reference_dt.ndim != 0 or not np.isfinite(reference_dt)
                    or float(reference_dt) <= 0
                    or not np.isclose(float(reference_dt) * cfg.reference_stride,
                                      cfg.control_dt, rtol=1e-6, atol=1e-9)):
                raise ValueError("Reference control_dt * reference_stride must match the environment")
            if (np.any(reference_targets < np.asarray(meta["joint_lower"], dtype=np.float32))
                    or np.any(reference_targets > np.asarray(meta["joint_upper"], dtype=np.float32))):
                raise ValueError("Reference targets exceed joint limits")
        self.asset_bank = AssetBank.from_model(model, meta)
        self.sim = GPUSimulation(cfg.num_envs, cfg.timestep, self.asset_bank.batch_sizes(cfg.num_envs), (model, meta))
        self.model, self.meta = model, meta
        if cfg.action_mode == "reference_residual":
            self.reference_targets = torch.as_tensor(reference_targets, device=self.device)
            self.residual_scale = torch.tensor(
                [cfg.leg_residual_scale] * 12 + [cfg.arm_residual_scale] * 6 + [cfg.gripper_residual_scale],
                device=self.device, dtype=torch.float32)
        bank_path = Path(cfg.reset_bank)
        bank_path = bank_path if bank_path.is_absolute() else ROOT / bank_path
        with np.load(bank_path) as source:
            self.bank = {name: torch.as_tensor(source[name], device=self.device) for name in source.files}
        if "seed" not in self.bank:
            raise ValueError("Reset bank requires per-record seed diagnostics")
        self.bank["combination"] = self.bank["combination"].long()
        self.episode_bank_rows = torch.full((cfg.num_envs,), -1, device=self.device, dtype=torch.long)
        self.class_ids = torch.zeros((cfg.num_envs, 3), device=self.device, dtype=torch.long)
        topology = _hull_topology(meta["cargo_vertices"])
        self.hull_vertices, self.hull_mask, self.hull_edges, self.edge_mask = (
            torch.as_tensor(array, device=self.device) for array in topology)
        self.cargo_ids = torch.as_tensor(meta["cargo_body_ids"][:cfg.num_objects], device=self.device, dtype=torch.long)
        self.foot_ids = torch.as_tensor(meta["foot_body_ids"], device=self.device, dtype=torch.long)
        self.steps = torch.zeros(cfg.num_envs, device=self.device, dtype=torch.long)
        self.episode_return = torch.zeros(cfg.num_envs, device=self.device)
        self.prev_action = torch.zeros((cfg.num_envs, 19), device=self.device)
        self.previous_targets = self.sim.targets.clone()
        self.previous_previous_targets = self.sim.targets.clone()
        self.camera_age = torch.zeros(cfg.num_envs, device=self.device)
        self.episode_max_tilt = torch.zeros(cfg.num_envs, device=self.device)
        # ClearanceQuery is constructed below; its default horizon is
        # margin + gap = .02 m.  Keep the episode accumulator independent of
        # construction order and reset it from the query horizon afterwards.
        self.episode_min_clearance = torch.full((cfg.num_envs,), .02, device=self.device)
        self.episode_max_speed_ratio = torch.zeros(cfg.num_envs, device=self.device)
        self.episode_peak_joint_speed_ratio = torch.zeros((cfg.num_envs, 19), device=self.device)
        self.episode_invalid_joint_limit = torch.zeros(cfg.num_envs, device=self.device, dtype=torch.bool)
        self.episode_invalid_nonfinite = torch.zeros_like(self.episode_invalid_joint_limit)
        self.episode_invalid_overflow = torch.zeros_like(self.episode_invalid_joint_limit)
        self.episode_max_foot_slip = torch.zeros(cfg.num_envs, device=self.device)
        self.control_steps = 0
        self.reset_cursor = 0
        self.last_image = None
        self.tracker = TorchRearExitTracker(ExitGeometry.from_v3_properties(
            ROOT / "assets/b2z1/basket_v3/basket_properties.json"))
        self.previous_tail_potential = torch.zeros(
            (cfg.num_envs, cfg.num_objects), device=self.device)
        self.previous_aperture_potential = torch.zeros_like(self.previous_tail_potential)
        self.sim.capture_step()
        self.camera = GPUCamera(model, self.sim.gpu_model, self.sim.data, cfg.num_envs,
                                cfg.image_width, cfg.image_height)
        from .clearance import ClearanceQuery
        self.clearance = ClearanceQuery(model, meta, cfg.num_envs)
        # Compile and capture the read-only forbidden-contact query on the
        # same owned CUDA stream as physics.  Keeping this outside the graph
        # leaves qpos copies per step dynamic while avoiding a host launch for
        # every 2 ms physics substep.
        self.clearance.evaluate(self.sim.qpos)
        self.clearance.capture()
        self.last_clearance = torch.ones(cfg.num_envs, device=self.device)
        self.last_intersection = torch.zeros(cfg.num_envs, device=self.device, dtype=torch.bool)
        self._dangerous_geom = torch.zeros(model.ngeom, device=self.device, dtype=torch.bool)
        self._foot_geom = torch.zeros(model.ngeom, device=self.device, dtype=torch.bool)
        self._geom_body = torch.as_tensor(model.geom_bodyid.copy(), device=self.device, dtype=torch.long)
        for name, ids in meta["collision_by_link"].items():
            if name == "base" or name.endswith(("_hip", "_thigh")):
                self._dangerous_geom[torch.as_tensor(ids, device=self.device, dtype=torch.long)] = True
            if name.endswith("_foot"):
                self._foot_geom[torch.as_tensor(ids, device=self.device, dtype=torch.long)] = True
        self.floor_id = int(model.geom("floor").id)
        self._contact_indices = torch.arange(self.sim.data.naconmax, device=self.device)
        self.last_info = {}

    def _vertices(self):
        cls = self.class_ids[:, :self.config.num_objects]
        vertices = self.hull_vertices[cls]
        rotations = self.sim.xmat[:, self.cargo_ids]
        world = torch.einsum("bnvj,bnkj->bnvk", vertices, rotations) + self.sim.xpos[:, self.cargo_ids, None]
        return world_to_basket(world, self.sim.xpos[:, self.meta["basket_id"]],
                               self.sim.xmat[:, self.meta["basket_id"]])

    def _unload_potentials(self, vertices):
        """Evaluator-only shaping potentials in the moving basket frame."""
        geometry = self.tracker.geometry
        mask = self.hull_mask[self.class_ids[:, :self.config.num_objects]]
        centre = (vertices * mask[..., None]).sum(-2) / mask.sum(-1).clamp_min(1)[..., None]
        rear_distance = (centre[..., 0] - geometry.rear_x).clamp_min(0.)
        tail = -rear_distance
        gate = torch.sigmoid((geometry.rear_x + .12 - centre[..., 0]) / .025)
        z_centre = .5 * (geometry.sill_z + geometry.rim_z)
        y_score = (1. - centre[..., 1].abs() / geometry.rear_half_width).clamp(0., 1.)
        z_score = (1. - (centre[..., 2] - z_centre).abs()
                   / max(.5 * (geometry.rim_z - geometry.sill_z), 1e-6)).clamp(0., 1.)
        aperture = gate * .5 * (y_score + z_score)
        return tail, aperture

    @torch.no_grad()
    def _reset(self, mask):
        indices = torch.nonzero(mask, as_tuple=False).flatten()
        if not len(indices):
            return
        if self.config.sequential_reset:
            records = (torch.arange(len(indices), device=self.device) + self.reset_cursor) % len(self.bank["qpos"])
            self.reset_cursor = (self.reset_cursor + len(indices)) % len(self.bank["qpos"])
        else:
            records = torch.randint(len(self.bank["qpos"]), (len(indices),), generator=self.generator, device=self.device)
        self.episode_bank_rows[indices] = records
        self.class_ids[indices] = self.bank["combination"][records]
        self.asset_bank.apply(self.sim.gpu_model, self.class_ids[indices], indices)
        mjw.reset_data(self.sim.gpu_model, self.sim.data, wp.from_torch(mask))
        self.sim.qpos[indices] = self.bank["qpos"][records].float()
        self.sim.qvel[indices] = self.bank["qvel"][records].float()
        self.sim.ctrl[indices] = self.bank["ctrl"][records].float()
        self.sim.targets[indices] = self.sim.ctrl[indices]
        if self.config.stage == "balance" or self.config.num_objects == 1:
            # Reset-only curriculum isolation: balance is truly empty; the
            # single-object course retains slot 0. Parked cargo stays dynamic.
            parked = range(3) if self.config.stage == "balance" else (1, 2)
            for slot in parked:
                adr, dof = int(self.meta["cargo_qpos_adr"][slot]), int(self.meta["cargo_dof_adr"][slot])
                self.sim.qpos[indices, adr:adr+3] = torch.tensor([2. + slot * .4, 2., .2], device=self.device)
                self.sim.qvel[indices, dof:dof+6] = 0
        mjw.forward(self.sim.gpu_model, self.sim.data)
        # reset_data/forward updates body kinematics; camlight is a separate
        # MuJoCo Warp pass and must be refreshed before the first wrist image.
        mjw.kinematics(self.sim.gpu_model, self.sim.data)
        mjw.camlight(self.sim.gpu_model, self.sim.data)
        self.steps[indices] = 0
        self.episode_return[indices] = 0
        self.prev_action[indices] = 0
        self.previous_targets[indices] = self.sim.targets[indices]
        self.previous_previous_targets[indices] = self.sim.targets[indices]
        self.camera_age[indices] = 0
        self.last_intersection[indices] = False
        self.last_clearance[indices] = self.clearance.horizon
        self.episode_max_tilt[indices] = 0
        self.episode_min_clearance[indices] = self.clearance.horizon
        self.episode_max_speed_ratio[indices] = 0
        self.episode_peak_joint_speed_ratio[indices] = 0
        self.episode_invalid_joint_limit[indices] = False
        self.episode_invalid_nonfinite[indices] = False
        self.episode_invalid_overflow[indices] = False
        self.episode_max_foot_slip[indices] = 0
        if self.config.stage == "balance":
            # No cargo task exists in the empty-basket course. Its completion
            # is solely standing stability; it can never emit an unload event.
            objects = torch.zeros((self.num_envs, self.config.num_objects),
                                  dtype=torch.bool, device=self.device)
            worlds = torch.zeros(self.num_envs, dtype=torch.bool, device=self.device)
            self.result = ExitResult(
                state=objects.long(), first_exit=objects, wrong_exit=objects,
                reentry=objects, ever_exited=objects, exited=objects,
                wrong=objects, reentered=objects, invalid_initial=objects,
                invalid_trajectory=objects, invalid_dynamics=worlds,
                all_clear=worlds, hold_time=torch.zeros(self.num_envs, device=self.device),
                success=worlds)
            return
        cls = self.class_ids[:, :self.config.num_objects]
        vertices = self._vertices()
        self.result = self.tracker.reset(vertices, self.hull_mask[cls], self.hull_edges[cls],
                                         self.edge_mask[cls], env_mask=mask)
        tail, aperture = self._unload_potentials(vertices)
        self.previous_tail_potential[indices] = tail[indices]
        self.previous_aperture_potential[indices] = aperture[indices]
        if bool(self.result.invalid_initial[indices].any()):
            raise RuntimeError("Reset bank contains geometry outside the valid basket interior")

    @torch.no_grad()
    def reset(self):
        self._reset(torch.ones(self.num_envs, device=self.device, dtype=torch.bool))
        self.last_image = self.camera.render()
        self.previous_feet = self.sim.xpos[:, self.foot_ids].clone()
        return self.observation()

    def observation(self):
        rotation = self.sim.xmat[:, self.meta["base_id"]]
        qpos = self.sim.qpos[:, self.sim.joint_adr]
        qvel = self.sim.qvel[:, self.sim.dof_adr]
        joint_pos = (qpos - self.sim.nominal) / (self.sim.upper - self.sim.lower).clamp_min(.1)
        linear = torch.einsum("bij,bi->bj", rotation, self.sim.qvel[:, :3])
        angular = self.sim.qvel[:, 3:6]
        gravity = -rotation[:, 2, :]
        proprio = torch.cat((joint_pos, qvel / self.sim.velocity, linear, angular, gravity,
                              self.prev_action, self.camera_age[:, None]), dim=-1)
        if self.config.observe_time:
            elapsed = self.steps.float() * self.config.control_dt / self.config.episode_seconds
            proprio = torch.cat((proprio, elapsed[:, None]), dim=-1)
        return {"image": self.last_image, "proprio": torch.nan_to_num(proprio)}

    def _ground_contact(self, measure_slip=False):
        contacts = self.sim.data.contact
        geom = wp.to_torch(contacts.geom).long()
        world = wp.to_torch(contacts.worldid).long()
        distance = wp.to_torch(contacts.dist)
        active = self._contact_indices < wp.to_torch(self.sim.data.nacon)[0]
        ga, gb = geom[:, 0].clamp(0, self.model.ngeom - 1), geom[:, 1].clamp(0, self.model.ngeom - 1)
        bad = active & (distance < -.0005) & (
            ((ga == self.floor_id) & self._dangerous_geom[gb]) |
            ((gb == self.floor_id) & self._dangerous_geom[ga]))
        counts = torch.zeros(self.num_envs, device=self.device, dtype=torch.long)
        counts.scatter_add_(0, world.clamp(0, self.num_envs - 1), bad.long())
        if not measure_slip:
            return counts > 0
        foot_contact = active & (distance < .003) & (
            ((ga == self.floor_id) & self._foot_geom[gb]) |
            ((gb == self.floor_id) & self._foot_geom[ga]))
        body = self._geom_body[torch.where(ga == self.floor_id, gb, ga)]
        world = world.clamp(0, self.num_envs - 1)
        spatial = wp.to_torch(self.sim.data.cvel)[world, body]
        centre = wp.to_torch(self.sim.data.subtree_com)[world, self.meta["base_id"]]
        radius = wp.to_torch(contacts.pos) - centre
        point_velocity = spatial[:, 3:] + torch.cross(spatial[:, :3], radius, dim=-1)
        speed = torch.nan_to_num(point_velocity[:, :2].norm(dim=-1)) * foot_contact
        totals = torch.zeros(self.num_envs, device=self.device)
        samples = torch.zeros_like(totals)
        totals.scatter_add_(0, world, speed)
        samples.scatter_add_(0, world, foot_contact.float())
        return counts > 0, totals / samples.clamp_min(1)

    def _dynamic_state(self):
        rotation = self.sim.xmat[:, self.meta["base_id"]]
        upright = rotation[:, 2, 2]
        finite = torch.isfinite(self.sim.qpos).all(-1) & torch.isfinite(self.sim.qvel).all(-1)
        joint_speed_ratio = self.sim.qvel[:, self.sim.dof_adr].abs() / self.sim.velocity
        speed_ratio = joint_speed_ratio.amax(-1)
        angles = self.sim.qpos[:, self.sim.joint_adr]
        limits_bad = ((angles < self.sim.lower - .02) | (angles > self.sim.upper + .02)).any(-1)
        overflow = wp.to_torch(self.sim.data.overflow) != 0
        # Read-only episode diagnostics: retain finite speed peaks and the
        # actual invalid-state causes even if later substeps recover. These
        # accumulators never feed observations, rewards or termination.
        finite_joint_speed = torch.where(torch.isfinite(joint_speed_ratio), joint_speed_ratio, 0.)
        self.episode_peak_joint_speed_ratio.copy_(torch.maximum(
            self.episode_peak_joint_speed_ratio, finite_joint_speed))
        self.episode_invalid_joint_limit |= limits_bad
        self.episode_invalid_nonfinite |= ~finite
        self.episode_invalid_overflow |= overflow
        # The actuator target slew limiter is not a guarantee that the
        # realized joint velocity stays below its hardware limit.  Allow only
        # a small numerical margin and classify a measured overspeed as an
        # invalid dynamics transition.
        invalid = ~finite | (speed_ratio > 1.02) | limits_bad | overflow
        fallen = (self.sim.qpos[:, 2] < .28) | (upright < .574)
        arm_recovered = (angles[:, 12:] - self.sim.nominal[12:]).abs().amax(-1) < .20
        stable = ((upright > .985) & (self.sim.qvel[:, :3].norm(dim=-1) < .15)
                  & (self.sim.qvel[:, 3:6].norm(dim=-1) < .2) & arm_recovered)
        return stable, fallen, invalid

    @torch.no_grad()
    def step(self, action):
        cfg, sim = self.config, self.sim
        if action.shape != (self.num_envs, 19):
            raise ValueError("Action must be [num_envs,19]")
        bad_action = ~torch.isfinite(action).all(-1)
        action = torch.nan_to_num(action).clamp(-1, 1)
        rate = torch.tensor([cfg.leg_target_rate] * 12 + [cfg.arm_target_rate] * 6 + [.25], device=self.device)
        if cfg.stage == "balance":
            rate[:12] = min(cfg.leg_target_rate, .2)
            rate[12:] = 0
        if cfg.action_mode == "position":
            goal = sim.lower + (action + 1.) * .5 * (sim.upper - sim.lower)
            requested = sim.targets + (goal - sim.targets).clamp(
                -rate * cfg.control_dt, rate * cfg.control_dt)
            sim.set_targets(requested, cfg.control_dt)
        elif cfg.action_mode == "reference_residual":
            index = (self.steps * cfg.reference_stride).clamp(max=self.reference_targets.shape[0] - 1)
            goal = (self.reference_targets[index] + self.residual_scale * action).clamp(sim.lower, sim.upper)
            requested = sim.targets + (goal - sim.targets).clamp(
                -rate * cfg.control_dt, rate * cfg.control_dt)
            sim.set_targets(requested, cfg.control_dt)
        else:
            sim.set_targets(sim.targets + action * rate * cfg.control_dt, cfg.control_dt)
        event_exit = torch.zeros((self.num_envs, cfg.num_objects), device=self.device, dtype=torch.bool)
        dangerous = torch.zeros(self.num_envs, device=self.device, dtype=torch.bool)
        fallen, invalid = dangerous.clone(), bad_action.clone()
        control_speed_peak = torch.zeros(self.num_envs, device=self.device)
        control_clearance_min = self.last_clearance.clone()
        for substep in range(round(cfg.control_dt / cfg.timestep)):
            sim.physics_step()
            stable, fall_now, invalid_now = self._dynamic_state()
            rotation = self.sim.xmat[:, self.meta["base_id"]]
            tilt = torch.acos(rotation[:, 2, 2].clamp(-1, 1))
            speed_ratio = (self.sim.qvel[:, self.sim.dof_adr].abs() / self.sim.velocity).amax(-1)
            control_speed_peak = torch.maximum(control_speed_peak, speed_ratio)
            self.episode_max_tilt.copy_(torch.maximum(self.episode_max_tilt, tilt))
            self.episode_max_speed_ratio.copy_(torch.maximum(self.episode_max_speed_ratio, speed_ratio))
            dangerous |= self._ground_contact()
            fallen |= fall_now
            invalid |= invalid_now
            if (substep + 1) % round(cfg.clearance_period / cfg.timestep) == 0:
                self.last_intersection, self.last_clearance = self.clearance.evaluate(sim.data.qpos)
                control_clearance_min = torch.minimum(control_clearance_min, self.last_clearance)
                self.episode_min_clearance.copy_(torch.minimum(self.episode_min_clearance, self.last_clearance))
            valid_dynamics = ~(invalid | fallen | dangerous | self.last_intersection)
            if cfg.stage == "balance":
                self.result = replace(self.result, invalid_dynamics=~valid_dynamics)
            else:
                self.result = self.tracker.update(self._vertices(), cfg.timestep,
                                                  stable=stable, valid_dynamics=valid_dynamics)
            event_exit |= self.result.first_exit
        self.steps += 1
        self.control_steps += 1
        self.camera_age += cfg.control_dt
        camera_due = self.control_steps % round(cfg.camera_period / cfg.control_dt) == 0
        if camera_due:
            self.last_image = self.camera.render()
            self.camera_age.zero_()
        feet = sim.xpos[:, self.foot_ids]
        _, foot_slip = self._ground_contact(measure_slip=True)
        self.episode_max_foot_slip.copy_(torch.maximum(self.episode_max_foot_slip, foot_slip))
        self.previous_feet = feet.clone()
        vertices = self._vertices()
        tail_potential, aperture_potential = self._unload_potentials(vertices)
        valid_object = ~(self.result.wrong | self.result.reentered
                         | self.result.invalid_initial | self.result.invalid_trajectory
                         | self.result.invalid_dynamics[:, None])
        tail_progress = ((tail_potential - self.previous_tail_potential)
                         * valid_object).sum(-1)
        aperture_progress = ((aperture_potential - self.previous_aperture_potential)
                             * valid_object).sum(-1)
        self.previous_tail_potential.copy_(tail_potential)
        self.previous_aperture_potential.copy_(aperture_potential)
        if cfg.stage == "balance":
            tail_progress.zero_()
            aperture_progress.zero_()
        timeout = self.steps * cfg.control_dt >= cfg.episode_seconds
        control_result = replace(self.result, first_exit=event_exit)
        reward, terms = compute_reward(
            control_result, action=sim.targets, previous_action=self.previous_targets,
            previous_previous_action=self.previous_previous_targets,
            joint_pos=sim.qpos[:, sim.joint_adr], joint_vel=sim.qvel[:, sim.dof_adr],
            lower=sim.lower, upper=sim.upper,
            effort_ratio=wp.to_torch(sim.data.actuator_force) / sim.effort,
            foot_slip=foot_slip, dangerous_collision=dangerous, fallen=fallen,
            self_intersection=self.last_intersection, timeout=timeout, stable=stable, dt=cfg.control_dt,
            tail_progress=tail_progress, aperture_alignment=aperture_progress,
            body_tilt=tilt, body_angular_speed=sim.qvel[:, 3:6].norm(dim=-1),
            body_height=sim.qpos[:, 2], weights=self.reward_weights,
            speed_ratio_peak=control_speed_peak, clearance_min=control_clearance_min)
        if cfg.stage == "balance":
            # Explicit standing curriculum: its completion is never unloading success.
            other_failure = torch.stack([value for key, value in terms.items()
                                         if key.startswith("failure_") and key != "failure_timeout"], -1).any(-1)
            balance_success = timeout & stable & ~other_failure
            reward -= terms["timeout"] + terms["timeout_remaining"]
            upright = sim.xmat[:, self.meta["base_id"], 2, 2]
            reward += cfg.control_dt * (.5 - 4 * (1 - upright)
                                        - 2 * sim.qvel[:, 3:6].square().sum(-1))
            reward += 5 * balance_success.float() - 10 * (timeout & ~balance_success).float()
            terms["is_success"] = balance_success
            terms["is_failure"] = other_failure | (timeout & ~balance_success)
        done = terms["is_success"] | terms["is_failure"]
        self.episode_return += reward
        self.previous_previous_targets.copy_(self.previous_targets)
        self.previous_targets.copy_(sim.targets)
        self.prev_action.copy_(action)
        obs = self.observation()
        # A pure episode time limit is a truncation for PPO bootstrapping;
        # success, dynamics, geometry and collision outcomes are terminals.
        non_timeout_failure = torch.stack(
            [value for key, value in terms.items()
             if key.startswith("failure_") and key != "failure_timeout"], -1
        ).any(-1)
        info = {"timeouts": timeout & ~non_timeout_failure & ~terms["is_success"], "events": {}}
        for key, value in terms.items():
            if key.startswith("failure_"):
                info["events"][key] = value & done & terms["is_failure"]
        info["events"]["unload_success"] = self.result.success & done
        info["events"]["correct_object_exit"] = event_exit.sum(-1)
        if bool(done.any()):
            terminal = {key: value.clone() for key, value in obs.items()}
            info["terminal_observation"] = terminal
            # This is evaluator-only metadata captured before local reset;
            # it is never included in actor observations.
            info["episode_combination"] = self.class_ids[done, :3].clone()
            info["episode_bank_row"] = self.episode_bank_rows[done].clone()
            info["episode_seed"] = self.bank["seed"][info["episode_bank_row"]].clone()
            peak_joint_ratio, peak_joint_index = self.episode_peak_joint_speed_ratio[done].max(-1)
            info["episode"] = dict(return_=self.episode_return[done].clone(),
                                    length=self.steps[done].float().clone(),
                                    success=terms["is_success"][done].float(),
                                    unload_success=self.result.success[done].float(),
                                    correct_exit_count=self.result.exited[done].sum(-1).float(),
                                    wrong_exit_count=self.result.wrong[done].sum(-1).float(),
                                    active_object_count=torch.full_like(self.steps[done],
                                        0 if cfg.stage == "balance" else cfg.num_objects),
                                    max_tilt_rad=self.episode_max_tilt[done].clone(),
                                    min_clearance_m=self.episode_min_clearance[done].clone(),
                                    max_speed_ratio=self.episode_max_speed_ratio[done].clone(),
                                    max_foot_slip_m_s=self.episode_max_foot_slip[done].clone(),
                                    peak_speed_joint_index=peak_joint_index,
                                    peak_speed_joint_ratio=peak_joint_ratio,
                                    invalid_joint_limit=self.episode_invalid_joint_limit[done].clone(),
                                    invalid_nonfinite=self.episode_invalid_nonfinite[done].clone(),
                                    invalid_overflow=self.episode_invalid_overflow[done].clone())
            self._reset(done)
            self.last_image = self.camera.render()
            self.camera_age.zero_()
            self.previous_feet[done] = sim.xpos[done][:, self.foot_ids]
            obs = self.observation()
        self.last_info = {"stage": cfg.stage, "terms": terms, "result": control_result}
        return obs, reward, done, info

    def config_dict(self):
        return asdict(self.config)
