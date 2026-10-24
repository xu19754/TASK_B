"""Independent GPU arm/body clearance query; never advances or alters physics.

The main model retains its disabled internal contacts. A second model has all
ordinary contacts disabled and only the metadata's forbidden pairs enabled.
Its qpos is copied from the main state; only kinematics and convex narrowphase
run. No query force, constraint, position or velocity is copied back.

Distances describe the model's actual MuJoCo collision geoms (convex hulls
where those geoms are meshes), not sphere proxies or render silhouettes.
They are clipped at margin+gap, .02 m by default: no detected contact means
"at least .02 m", not an exact distance to a distant body.
"""

from __future__ import annotations

import xml.etree.ElementTree as ET

import mujoco
import mujoco_warp as mjw
import numpy as np
import torch
import warp as wp


@wp.kernel
def _reduce_contact_distance(
    ncontact: wp.array[int],
    contact_world: wp.array[int],
    contact_distance: wp.array[float],
    minimum_distance: wp.array[float],
):
    contact = wp.tid()
    if contact < ncontact[0]:
        world = contact_world[contact]
        if world >= 0 and world < minimum_distance.shape[0]:
            wp.atomic_min(minimum_distance, world, contact_distance[contact])


class ClearanceQuery:
    """Read-only-to-main-state batched forbidden-contact evaluator.

    evaluate(qposwp[B,nq]) -> (intersect[B] bool, minimum_distance[B] float).
    Also accepts a Torch tensor on the query device. Returned tensors are GPU
    views/results; `minimum_distance` is reused by the next evaluation.
    `near_clearance_cost` is an optional nonnegative unitless proximity cost.
    `overflow` is a per-world device flag; a truncated/failed narrowphase
    conservatively makes affected worlds intersect, never granting safe passage.
    """

    def __init__(self, model, meta, nworld, *, margin=.01, gap=.01,
                 penetration_threshold=-.001, contacts_per_world=512,
                 convex_pairs_per_world=256, device=None):
        if nworld < 1 or margin <= 0 or gap < 0 or min(contacts_per_world, convex_pairs_per_world) < 1:
            raise ValueError("nworld/margin must be positive and gap nonnegative")
        wp.init()
        self.nworld = int(nworld)
        self.margin, self.horizon = float(margin), float(margin + gap)
        self.penetration_threshold = float(penetration_threshold)
        self.device = wp.get_device(device)
        root = ET.fromstring(meta["xml"])
        for geom in root.iter("geom"):
            geom.set("contype", "0")
            geom.set("conaffinity", "0")
        # Only the explicit forbidden robot pairs belong to this query.
        for contact in root.findall("contact"):
            root.remove(contact)
        contact = ET.SubElement(root, "contact")
        pairs = np.asarray(meta["self_collision_pairs"], dtype=np.int32).reshape(-1, 2)
        for index, (a, b) in enumerate(pairs):
            ET.SubElement(contact, "pair", name=f"clearance_{index}",
                          geom1=model.geom(int(a)).name, geom2=model.geom(int(b)).name,
                          margin=str(margin), gap=str(gap), condim="1")
        self.model = mujoco.MjModel.from_xml_string(ET.tostring(root, encoding="unicode"))
        if (self.model.nq != model.nq or self.model.nv != model.nv
                or not np.array_equal(self.model.jnt_qposadr, model.jnt_qposadr)):
            raise ValueError("Clearance query must preserve the main qpos layout")
        if np.any(self.model.geom_contype) or np.any(self.model.geom_conaffinity):
            raise RuntimeError("Ordinary query contacts were not fully disabled")
        self.cpu_data = mujoco.MjData(self.model)
        with wp.ScopedDevice(self.device):
            self.gpu_model = mjw.put_model(self.model)
            # Capacity concerns simultaneous *nearby* pairs, not the number of
            # configured pairs. Sizing it as 16*all_pairs caused huge Warp CCD
            # scratch allocations. Every capacity overflow is checked below.
            self.data = mjw.put_data(self.model, self.cpu_data, nworld=self.nworld,
                                     nconmax=contacts_per_world,
                                     nccdmax=min(convex_pairs_per_world, contacts_per_world),
                                     njmax=0)
            self._minimum = wp.full(self.nworld, self.horizon, dtype=float)
        self.minimum_distance = wp.to_torch(self._minimum)
        self._contact_count = wp.to_torch(self.data.nacon)
        self.near_clearance_cost = torch.zeros_like(self.minimum_distance)
        self.overflow = torch.zeros(self.nworld, device=self.minimum_distance.device, dtype=torch.bool)
        self.graph = None

    def _run_query(self):
        mjw.kinematics(self.gpu_model, self.data)
        self.data.overflow.zero_()
        mjw.collision(self.gpu_model, self.data)
        self._minimum.fill_(self.horizon)
        wp.launch(_reduce_contact_distance, dim=self.data.naconmax,
                  inputs=[self.data.nacon, self.data.contact.worldid,
                          self.data.contact.dist, self._minimum])

    def capture(self):
        """Capture query kernels on the caller's owned, nondefault CUDA stream.

        It is safe to call after an ordinary evaluation has compiled kernels.
        qpos copying stays outside the graph, so later input arrays may change.
        No physics integration or constraint solving enters this graph.
        """
        with wp.ScopedDevice(self.device):
            self._run_query()
            wp.synchronize_device(self.device)
            with wp.ScopedCapture() as capture:
                self._run_query()
            self.graph = capture.graph

    @torch.no_grad()
    def evaluate(self, qposwp):
        if isinstance(qposwp, torch.Tensor):
            qposwp = wp.from_torch(qposwp.contiguous(), dtype=wp.float32)
        if tuple(qposwp.shape) != (self.nworld, self.model.nq):
            raise ValueError("qpos must have shape [nworld, query_model.nq]")
        if qposwp.device != self.data.qpos.device:
            raise ValueError("qpos and clearance query must share a device")
        with wp.ScopedDevice(self.device):
            wp.copy(self.data.qpos, qposwp)
            if self.graph is None:
                self._run_query()
            else:
                wp.capture_launch(self.graph)
        self.overflow = ((self._contact_count[0] > self.data.naconmax)
                         | (wp.to_torch(self.data.overflow) != 0))
        intersect = ((self.minimum_distance < self.penetration_threshold) | self.overflow)
        self.near_clearance_cost = ((self.margin - self.minimum_distance).clamp_min(0)
                                    / self.margin).square()
        return intersect, self.minimum_distance
