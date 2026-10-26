"""Batched cargo assets and reset-only, physically settled layout generation."""
from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import mujoco
import numpy as np

from .model import COMBINATIONS, ROOT, _quaternion, _rotation, build_model, initialize_data


class AssetBank:
    """Reuse three compiled source meshes without recompiling per GPU world.

    A mesh geom's compiled frame differs from its source OBJ frame. Copying
    just ``geom_dataid`` is incorrect; the full frame, bounds, body inertia
    frame and free-joint derived weights are copied together.
    """
    GEOM_FIELDS = ("geom_dataid", "geom_pos", "geom_quat", "geom_size", "geom_aabb", "geom_rbound", "geom_rgba")
    BODY_FIELDS = ("body_ipos", "body_iquat", "body_inertia", "body_invweight0")
    DOF_FIELDS = ("dof_damping", "dof_invweight0")
    # set_const may update these per-world values. The independent .5 kg free
    # cargo bodies cannot alter the robot's joint/camera constants.
    DERIVED_FIELDS = ("body_subtreemass", "actuator_acc0", "cam_pos0", "cam_poscom0", "cam_mat0",
                      "light_pos0", "light_poscom0", "light_dir0")

    @classmethod
    def from_model(cls, model, metadata):
        if tuple(metadata["combination"]) != (0, 1, 2):
            raise ValueError("AssetBank requires build_model((0, 1, 2)) so all source meshes exist")
        bank = cls()
        bank.geom_ids = np.array([model.geom(f"cargo_geom_{i}").id for i in range(3)])
        bank.body_ids = metadata["cargo_body_ids"].copy()
        bank.dof_ids = metadata["cargo_dof_adr"][:, None] + np.arange(6)[None]
        bank.templates = {name: np.array(getattr(model, name)[bank.geom_ids], copy=True) for name in cls.GEOM_FIELDS}
        bank.templates.update({name: np.array(getattr(model, name)[bank.body_ids], copy=True) for name in cls.BODY_FIELDS})
        bank.templates.update({name: np.array(getattr(model, name)[bank.dof_ids], copy=True) for name in cls.DOF_FIELDS})
        bank._device_templates = {}
        return bank

    def batch_sizes(self, nworld):
        return {name: int(nworld) for name in self.GEOM_FIELDS+self.BODY_FIELDS+self.DOF_FIELDS+self.DERIVED_FIELDS}

    def apply(self, gpu_model, class_ids, world_ids=None, gpu_data=None):
        """Assign assets for B worlds, optionally recomputing constants.

        ``class_ids`` has shape ``[B, 3]`` and indexes the source classes;
        ``world_ids`` selects B worlds (default all). This changes model
        properties only; the caller separately resets initial qpos/qvel.
        ``gpu_data`` requests ``mujoco_warp.set_const`` after the assignment.
        """
        import torch
        import warp as wp

        if class_ids.ndim != 2 or class_ids.shape[1] != 3:
            raise ValueError("class_ids must have shape [B, 3]")
        if bool(((class_ids < 0) | (class_ids > 2)).any()):
            raise ValueError("class indices must be in [0, 2]")
        device = class_ids.device
        if world_ids is None:
            world_ids = torch.arange(len(class_ids), device=device)
        else:
            world_ids = world_ids.to(device=device, dtype=torch.long)
        key = str(device)
        if key not in self._device_templates:
            self._device_templates[key] = {name: torch.as_tensor(value, device=device) for name, value in self.templates.items()}
        templates = self._device_templates[key]
        cls_ids = class_ids.to(dtype=torch.long)
        worlds = world_ids[:, None]
        for name in self.GEOM_FIELDS:
            target = wp.to_torch(getattr(gpu_model, name))
            slots = torch.as_tensor(self.geom_ids, device=device)[None]
            # CPU geom_aabb is [..., 6], Warp exposes [..., 2, 3].
            values = templates[name][cls_ids].reshape(tuple(cls_ids.shape)+tuple(target.shape[2:]))
            target[worlds, slots] = values.to(target.dtype)
        for name in self.BODY_FIELDS:
            target = wp.to_torch(getattr(gpu_model, name))
            slots = torch.as_tensor(self.body_ids, device=device)[None]
            target[worlds, slots] = templates[name][cls_ids].to(target.dtype)
        for name in self.DOF_FIELDS:
            target = wp.to_torch(getattr(gpu_model, name))
            slots = torch.as_tensor(self.dof_ids, device=device)[None]
            target[world_ids[:, None, None], slots] = templates[name][cls_ids].to(target.dtype)
        if gpu_data is not None:
            import mujoco_warp
            mujoco_warp.set_const(gpu_model, gpu_data)


def _place_random_cargo(model, data, metadata, rng):
    """Initialize separated near-flat collection layouts, then gravity settles."""
    basket = data.xpos[metadata["basket_id"]].copy()
    slots = rng.permutation(3)
    for i, (cls, adr) in enumerate(zip(metadata["combination"], metadata["cargo_qpos_adr"])):
        xyz = [(-.112, 0, .112)[slots[i]] + rng.uniform(-.006, .006),
               rng.uniform(-.009, .009), rng.uniform(.075, .100)]
        rpy = rng.uniform([-.10, -.10, -.12], [.10, .10, .12])
        if cls == 2:
            rpy[2] += math.pi/2
        data.qpos[adr:adr+3] = basket+xyz
        data.qpos[adr+3:adr+7] = _quaternion(_rotation(rpy))
    mujoco.mj_forward(model, data)
    geoms = [model.geom(f"cargo_geom_{i}").id for i in range(3)]
    for i in range(3):
        for j in range(i+1,3):
            if mujoco.mj_geomDistance(model, data, geoms[i], geoms[j], .01, None) < .002:
                return False
    return True


def _place_collected_cargo(model, data, metadata, rng):
    """Reset-only sequential free drops with uniform SO(3) orientations."""
    source_orientations=[]
    drop_stats=dict(max_tilt=0.,min_height=10.,dangerous_contacts=0)
    floor=model.geom("floor").id
    forbidden=set(model.geom_bodyid[metadata["collision_geom_ids"]])-set(metadata["foot_body_ids"])
    for i, adr in enumerate(metadata["cargo_qpos_adr"]):
        # Unloaded items remain dynamic on distant floor until initialized.
        data.qpos[adr:adr+3]=[3+i*.3,3,.3]
    mujoco.mj_forward(model,data)
    spawned=[]
    for i in rng.permutation(3):
        adr=metadata["cargo_qpos_adr"][i]
        quaternion=rng.normal(size=4); quaternion/=np.linalg.norm(quaternion)
        rotation=np.empty(9); mujoco.mju_quat2Mat(rotation,quaternion)
        rotation=rotation.reshape(3,3)
        transformed=metadata["cargo_vertices"][i]@rotation.T
        # Initial projection is inside the rim; gravity and contact determine
        # the accepted, potentially stacked final orientation.
        x_max=min(.09,.21-float(np.max(np.abs(transformed[:,0]))))
        y_max=min(.02,.115-float(np.max(np.abs(transformed[:,1]))))
        if min(x_max,y_max)<=0:
            return False, source_orientations, drop_stats
        basket_rotation=data.xmat[metadata["basket_id"]].reshape(3,3)
        highest=.16
        for previous in spawned:
            bid=metadata["cargo_body_ids"][previous]
            vertices=(metadata["cargo_vertices"][previous]@data.xmat[bid].reshape(3,3).T+
                      data.xpos[bid]-data.xpos[metadata["basket_id"]])@basket_rotation
            highest=max(highest,float(vertices[:,2].max()))
        local=np.array([rng.uniform(-x_max,x_max),rng.uniform(-y_max,y_max),
                        highest+rng.uniform(.025,.055)-float(transformed[:,2].min())])
        data.qpos[adr:adr+3]=data.xpos[metadata["basket_id"]]+basket_rotation@local
        data.qpos[adr+3:adr+7]=_quaternion(basket_rotation@rotation)
        dof=metadata["cargo_dof_adr"][i];data.qvel[dof:dof+6]=0
        source_orientations.append(dict(slot=int(i),quaternion_wxyz=quaternion.tolist(),center_basket_frame_m=local.tolist()))
        mujoco.mj_forward(model,data)
        for _ in range(round(.9/model.opt.timestep)):
            mujoco.mj_step(model,data)
            drop_stats["min_height"]=min(drop_stats["min_height"],float(data.qpos[2]))
            drop_stats["max_tilt"]=max(drop_stats["max_tilt"],math.degrees(math.acos(np.clip(data.xmat[metadata["base_id"]].reshape(3,3)[2,2],-1,1))))
            for c in data.contact:
                a,b=c.geom
                if (a==floor and model.geom_bodyid[b] in forbidden) or (b==floor and model.geom_bodyid[a] in forbidden):
                    drop_stats["dangerous_contacts"]+=1
        spawned.append(int(i))
    return True, source_orientations, drop_stats


def _reset_metric_valid(vertices, topology):
    """Use the actual evaluator's initial-state predicate without relaxing it."""
    import torch
    from .metrics import TorchRearExitTracker

    vcount=max(len(v) for v in vertices)
    ecount=max(len(e) for e in topology)
    padded=np.zeros((1,3,vcount,3));mask=np.zeros((1,3,vcount),dtype=bool)
    edges=np.zeros((1,3,ecount,2),dtype=np.int64);edge_mask=np.zeros((1,3,ecount),dtype=bool)
    for i,(v,e) in enumerate(zip(vertices,topology)):
        padded[0,i,:len(v)]=v;mask[0,i,:len(v)]=True
        edges[0,i,:len(e)]=e;edge_mask[0,i,:len(e)]=True
    tracker=TorchRearExitTracker()
    result=tracker.reset(torch.from_numpy(padded),torch.from_numpy(mask),torch.from_numpy(edges),torch.from_numpy(edge_mask))
    return not bool(result.invalid_initial.any())


def _source_edges(path):
    faces=[list(map(lambda x:int(x.split('/')[0])-1,line.split()[1:]))
           for line in path.read_text().splitlines() if line.startswith('f ')]
    edges={(min(a,b),max(a,b)) for face in faces for a,b in zip(face,face[1:]+face[:1])}
    return np.asarray(sorted(edges),dtype=np.int64)


def orientation_distribution(quaternions):
    """Summarize the actual accepted local orientation distribution."""
    eulers=[]
    for quaternion in quaternions:
        matrix=np.empty(9);mujoco.mju_quat2Mat(matrix,np.asarray(quaternion,dtype=float))
        matrix=matrix.reshape(3,3)
        eulers.append([math.atan2(matrix[2,1],matrix[2,2]),
                       math.asin(np.clip(-matrix[2,0],-1,1)),math.atan2(matrix[1,0],matrix[0,0])])
    if not eulers:
        return {"count":0}
    degrees=np.rad2deg(np.asarray(eulers))
    return dict(count=len(degrees),rpy_min_deg=degrees.min(0).tolist(),rpy_max_deg=degrees.max(0).tolist(),
                yaw_histogram_45deg=np.histogram(degrees[:,2],bins=np.arange(-180,181,45))[0].tolist(),
                roll_histogram_45deg=np.histogram(degrees[:,0],bins=np.arange(-180,181,45))[0].tolist(),
                pitch_histogram_30deg=np.histogram(degrees[:,1],bins=np.arange(-90,91,30))[0].tolist())


def generate_reset_bank(output, seeds_per_combination=8, settle_seconds=3., *, seed_offset=0, pose_mode="near_flat"):
    """Produce reset states from free-base PD settling, rejecting invalid loads.

    ``near_flat`` is a simple initial curriculum. ``collected`` sequentially
    drops uniformly oriented objects above the basket, then accepts only
    physically settled states passing the unchanged evaluator reset predicate.
    Seed offsets keep training and evaluation layouts independent.
    """
    if pose_mode not in ("near_flat","collected"):
        raise ValueError("pose_mode must be near_flat or collected")
    records, observations = [], []
    rejected, reasons = 0, {}
    accepted_initial_orientations=[]
    accepted_final_orientations=[]
    for combo_index, combo in enumerate(COMBINATIONS):
        model, meta = build_model(combo)
        topology=[_source_edges(ROOT/"reports/basket_feasibility/screen_meshes"/f"{name}.obj")
                  for name in ("004_sugar_box","006_mustard_bottle","011_banana")]
        topology=[topology[c] for c in combo]
        for seed_index in range(seeds_per_combination):
            seed = seed_offset+combo_index*10000+seed_index
            rng = np.random.default_rng(seed)
            for attempt in range(100):
                data = initialize_data(model, meta)
                initial_orientations=[]
                drop_stats=dict(max_tilt=0.,min_height=10.,dangerous_contacts=0)
                if pose_mode=="collected":
                    placed,initial_orientations,drop_stats=_place_collected_cargo(model,data,meta,rng)
                else:
                    placed=_place_random_cargo(model, data, meta, rng)
                if not placed:
                    rejected += 1; reasons["initial_overlap"] = reasons.get("initial_overlap",0)+1
                    continue
                foot_previous = None
                slip_distance = np.zeros(4)
                contact_slip = np.zeros(4)
                max_contact_slip_speed = np.zeros(4)
                jacp, jacr = np.zeros((3, model.nv)), np.zeros((3, model.nv))
                max_tilt, dangerous_contacts, min_height = drop_stats["max_tilt"],drop_stats["dangerous_contacts"],drop_stats["min_height"]
                foot_set = set(meta["foot_body_ids"])
                robot_set = set(model.geom_bodyid[meta["collision_geom_ids"]])
                floor = model.geom("floor").id
                settle_start=float(data.time)
                for step in range(round(settle_seconds/model.opt.timestep)):
                    mujoco.mj_step(model,data)
                    min_height = min(min_height,float(data.qpos[2]))
                    max_tilt = max(max_tilt,math.degrees(math.acos(np.clip(data.xmat[meta["base_id"]].reshape(3,3)[2,2],-1,1))))
                    for c in data.contact:
                        a,b = c.geom
                        if (a==floor and model.geom_bodyid[b] in robot_set-foot_set) or (b==floor and model.geom_bodyid[a] in robot_set-foot_set):
                            dangerous_contacts += 1
                    if data.time > settle_start+settle_seconds-.5:
                        feet = data.xpos[meta["foot_body_ids"],:2].copy()
                        if foot_previous is not None:
                            slip_distance += np.linalg.norm(feet-foot_previous,axis=1)
                        foot_previous = feet
                        for contact in data.contact:
                            a,b=contact.geom
                            bid=int(model.geom_bodyid[b] if a==floor else model.geom_bodyid[a])
                            if (a==floor or b==floor) and bid in foot_set:
                                foot_index=int(np.flatnonzero(meta["foot_body_ids"]==bid)[0])
                                mujoco.mj_jac(model,data,jacp,jacr,contact.pos,bid)
                                speed=float(np.linalg.norm((jacp@data.qvel)[:2]))
                                contact_slip[foot_index]+=speed*model.opt.timestep
                                max_contact_slip_speed[foot_index]=max(max_contact_slip_speed[foot_index],speed)
                rotation = data.xmat[meta["basket_id"]].reshape(3,3)
                vertices = [(v @ data.xmat[bid].reshape(3,3).T + data.xpos[bid] - data.xpos[meta["basket_id"]]) @ rotation
                            for v,bid in zip(meta["cargo_vertices"],meta["cargo_body_ids"])]
                all_inside = _reset_metric_valid(vertices,topology)
                cargo_speed=max(float(np.linalg.norm(data.qvel[adr:adr+3])) for adr in meta["cargo_dof_adr"])
                good = all_inside and max_tilt<10 and min_height>.35 and dangerous_contacts==0 and cargo_speed<.08
                if not good:
                    reason = "cargo_outside" if not all_inside else "unstable"
                    rejected += 1; reasons[reason]=reasons.get(reason,0)+1
                    continue
                records.append((data.qpos.copy(),data.qvel.copy(),data.ctrl.copy(),combo,seed))
                accepted_initial_orientations.extend(initial_orientations)
                final_orientations=[]
                for bid in meta["cargo_body_ids"]:
                    final_orientations.append(_quaternion(rotation.T@data.xmat[bid].reshape(3,3)).tolist())
                accepted_final_orientations.extend(final_orientations)
                observations.append(dict(combination=combo,seed=seed,attempts=attempt+1,max_tilt_deg=max_tilt,
                                         min_base_height_m=min_height,max_final_cargo_speed_m_s=cargo_speed,
                                         foot_planar_path_final_half_second_m=slip_distance.tolist(),
                                         foot_contact_slip_final_half_second_m=contact_slip.tolist(),
                                         foot_max_contact_slip_speed_final_half_second_m_s=max_contact_slip_speed.tolist(),
                                         final_cargo_quaternions_basket_wxyz=final_orientations,
                                         dangerous_ground_contact_samples=dangerous_contacts))
                print(f"reset bank {len(records)}/{len(COMBINATIONS)*seeds_per_combination}: combination={combo}, seed={seed}, attempts={attempt+1}", flush=True)
                break
            else:
                raise RuntimeError(f"Could not settle valid reset for combination={combo}, seed={seed}")
    output=Path(output); output.parent.mkdir(parents=True,exist_ok=True)
    temporary=output.with_name(output.stem+".tmp.npz")
    np.savez_compressed(temporary,qpos=np.stack([r[0] for r in records]),qvel=np.stack([r[1] for r in records]),
                        ctrl=np.stack([r[2] for r in records]),combination=np.asarray([r[3] for r in records],dtype=np.int32),
                        seed=np.asarray([r[4] for r in records],dtype=np.int64))
    temporary.replace(output)
    report=dict(engine=f"MuJoCo {mujoco.__version__}",asset="b2z1_0815_with_basket_v3.urdf",
                count=len(records),seeds_per_combination=seeds_per_combination,settle_seconds=settle_seconds,
                seed_offset=seed_offset,pose_mode=pose_mode,
                rejected_attempts=rejected,rejection_rate=rejected/(len(records)+rejected),rejection_reasons=reasons,
                initialization="reset-only qpos assignment followed by free-base dynamics and nominal joint PD; no cargo constraints",
                pose_ranges=("x slots -0.112,0,0.112 m permuted plus +/-0.006 m; y +/-0.009 m; roll/pitch +/-0.10 rad; yaw +/-0.12 rad around near-flat class orientation" if pose_mode=="near_flat" else
                             "uniform SO(3) initial quaternions; center x up to +/-0.09 m and y up to +/-0.02 m constrained by projected source bounds; bottom 0.025..0.055 m above max(rim,previous cargo top), 0.9 s free dynamics between each drop"),
                initial_state_filter="TorchRearExitTracker.reset invalid_initial must be false for every full-source-mesh object; above-rim stacks allowed by evaluator",
                foot_metric="both planar foot-center path and ground-contact-point tangential speed integrated over final 0.5 s; Jacobian*qvel at each contact",
                scope="reset initialization distribution only; does not establish unloading generalization",
                accepted_initial_orientations=accepted_initial_orientations,
                accepted_final_quaternions_basket_wxyz=accepted_final_orientations,
                accepted_initial_orientation_distribution=orientation_distribution([v["quaternion_wxyz"] for v in accepted_initial_orientations]),
                accepted_final_orientation_distribution=orientation_distribution(accepted_final_orientations),
                records=observations)
    output.with_suffix(".json").write_text(json.dumps(report,indent=2,ensure_ascii=False)+"\n",encoding="utf-8")
    return report


if __name__ == "__main__":
    parser=argparse.ArgumentParser()
    parser.add_argument("--output",type=Path,default=ROOT/"runs/basket_unload/reset_bank.npz")
    parser.add_argument("--seeds-per-combination",type=int,default=8)
    parser.add_argument("--settle-seconds",type=float,default=3.)
    parser.add_argument("--seed-offset",type=int,default=0)
    parser.add_argument("--pose-mode",choices=("near_flat","collected"),default="near_flat")
    args=parser.parse_args()
    result=generate_reset_bank(args.output,args.seeds_per_combination,args.settle_seconds,seed_offset=args.seed_offset,pose_mode=args.pose_mode)
    print(json.dumps({k:v for k,v in result.items() if k not in ("records","accepted_initial_orientations","accepted_final_quaternions_basket_wxyz")},indent=2))
