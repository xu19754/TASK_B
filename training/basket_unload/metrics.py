"""Tail-window trajectory metrics for the independent MuJoCo experiment.

These metrics consume evaluator-only collision geometry, never actor inputs.
Coordinates are in the *current* basket frame: x forward, y left, z up.
Call at physics rate (or at most every 10 ms with the displacement guard).
Mesh vertices must retain their ordering between calls. Pass actual mesh edges
or the edges of a convex decomposition, not object centers or bounding boxes.

The v3 CAD display mesh has a rectangular rear aperture at x=-.225 m:
y=+/- .105 m, z=.015.. .16 m. Its width is constant with height there;
the rear sidewalls narrow along x, not along z. Source: basket_properties.json
and basket_visual.obj rear-plane vertices. This is a tessellation metric,
not a replacement for the simulator's collision detection or dynamics checks.
"""

from dataclasses import dataclass
from enum import IntEnum
import json
from pathlib import Path

import numpy as np
import torch


class ObjectState(IntEnum):
    RETAINED = 0
    PASSING = 1
    EXITED = 2
    WRONG_EXIT = 3
    REENTERED = 4
    INVALID_INITIAL = 5
    INVALID_TRAJECTORY = 6


@dataclass(frozen=True)
class ExitGeometry:
    rear_x: float = -.22500000894069672
    front_x: float = .22500000894069672
    rear_half_width: float = .105
    sill_z: float = .015
    rim_z: float = .15999999642372137
    side_half_width: float = .125
    rear_side_half_width: float = .110
    taper_start_x: float = -.075
    wall_thickness: float = .0015

    @classmethod
    def from_v3_properties(cls, path):
        data = json.loads(Path(path).read_text(encoding="utf-8"))
        if data.get("revision") != "v3":
            raise ValueError("Rear aperture geometry is verified only for basket v3")
        return cls(
            rear_x=data["bounds_m"][0][0], front_x=data["bounds_m"][1][0],
            rear_half_width=data["rear_clear_width_m"] / 2,
            sill_z=data["rear_sill_m"], rim_z=data["bounds_m"][1][2],
            side_half_width=data["bounds_m"][1][1],
            wall_thickness=data["wall_thickness_m"],
        )


@dataclass(frozen=True)
class ExitResult:
    """Object fields have shape [B,N]; environment fields have shape [B].

    first_exit, wrong_exit, reentry are one-update events. ever_exited,
    wrong, reentered and invalid flags latch until reset. `exited` additionally
    requires no subsequent failure. all_clear requires every object exited;
    success requires all_clear AND caller-supplied stable/valid dynamics for
    success_hold_s continuously. It is deliberately not latched.
    """
    state: object
    first_exit: object
    wrong_exit: object
    reentry: object
    ever_exited: object
    exited: object
    wrong: object
    reentered: object
    invalid_initial: object
    invalid_trajectory: object
    invalid_dynamics: object
    all_clear: object
    hold_time: object
    success: object


def world_to_basket(vertices_world, basket_position, rotation_world_from_basket):
    """Transform [B,...,3] geometry using a fresh dynamic basket pose.

    Rotation matrices map basket column vectors to world column vectors.
    Both NumPy arrays and Torch tensors are supported without device transfer.
    """
    expand = (slice(None),) + (None,) * (vertices_world.ndim - 2)
    relative = vertices_world - basket_position[expand]
    if isinstance(vertices_world, torch.Tensor):
        return torch.einsum("b...j,bjk->b...k", relative, rotation_world_from_basket)
    return np.einsum("b...j,bjk->b...k", relative, rotation_world_from_basket)


class TorchRearExitTracker:
    """Vectorized evaluator that keeps all per-step state on the input device.

    reset(vertices_local[B,N,V,3], vertex_mask, edges, edge_mask, env_mask)
    accepts masks [N,V]/[B,N,V], edges [N,E,2]/[B,N,E,2] and masks
    [N,E]/[B,N,E]. Padding edges must use legal vertex indices and be masked.
    Geometry/topology can vary across environments and reset independently.
    Edges are required: vertex crossings alone miss an elongated object whose
    edge intersects the sill while no vertex currently lies on the plane.

    A newly observed complete lift above the rim is a wrong top departure.
    Objects initially stacked above the rim are allowed; they can descend into
    the basket, but still must cross the tail opening at its legal height.
    Partial passage may reverse before completion. After complete passage,
    any reentry is terminal for this episode and never earns another reward.
    """

    def __init__(self, geometry=None, *, max_vertex_step_m=.02,
                 max_dt_s=.01, success_hold_s=.5, tolerance_m=.0005):
        self.geometry = geometry or ExitGeometry()
        if min(max_vertex_step_m, max_dt_s, success_hold_s) <= 0:
            raise ValueError("Step limits and success hold must be positive")
        self.max_vertex_step_m = max_vertex_step_m
        self.max_dt_s = max_dt_s
        self.success_hold_s = success_hold_s
        self.tolerance_m = tolerance_m
        self.previous = None

    def _side_width(self, x, *, inner=False):
        g = self.geometry
        fraction = ((x - g.rear_x) / (g.taper_start_x - g.rear_x)).clamp(0, 1)
        width = g.rear_side_half_width + fraction * (
            g.side_half_width - g.rear_side_half_width)
        return width - (g.wall_thickness if inner else 0)

    def _expand_topology(self, vertices, vertex_mask, edges, edge_mask):
        b, n, v, _ = vertices.shape
        if vertex_mask is None:
            vertex_mask = torch.ones((b, n, v), device=vertices.device, dtype=torch.bool)
        vertex_mask = torch.as_tensor(vertex_mask, device=vertices.device,
                                      dtype=torch.bool).expand(b, n, v)
        if edges is None:
            raise ValueError("Actual mesh/convex-decomposition edges are required")
        if isinstance(edges, np.ndarray) and not edges.flags.writeable:
            edges = edges.copy()
        edges = torch.as_tensor(edges, device=vertices.device, dtype=torch.long)
        edges = edges.expand(b, n, edges.shape[-2], 2)
        if edge_mask is None:
            edge_mask = torch.ones(edges.shape[:-1], device=vertices.device,
                                   dtype=torch.bool)
        edge_mask = torch.as_tensor(edge_mask, device=vertices.device,
                                    dtype=torch.bool).expand(edges.shape[:-1])
        return vertex_mask, edges, edge_mask

    def _bounds(self, vertices):
        masked_min = vertices.masked_fill(~self.vertex_mask[..., None], float("inf"))
        masked_max = vertices.masked_fill(~self.vertex_mask[..., None], -float("inf"))
        return masked_min.amin(-2), masked_max.amax(-2)

    @torch.no_grad()
    def reset(self, vertices_local, vertex_mask=None, edges=None, edge_mask=None,
              env_mask=None):
        vertices = vertices_local.detach()
        if vertices.ndim != 4 or vertices.shape[-1] != 3:
            raise ValueError("vertices_local must have shape [B,N,V,3]")
        if min(vertices.shape[:3]) == 0:
            raise ValueError("At least one environment, object and vertex are required")
        if not vertices.is_floating_point():
            raise ValueError("vertices_local must be floating point")
        b, n = vertices.shape[:2]
        topology = self._expand_topology(vertices, vertex_mask, edges, edge_mask)
        fresh = self.previous is None
        if fresh:
            self.vertex_mask, self.edges, self.edge_mask = topology
            self.previous = vertices.clone()
            env_mask = torch.ones(b, device=vertices.device, dtype=torch.bool)
        else:
            if vertices.shape != self.previous.shape or vertices.device != self.previous.device:
                raise ValueError("Geometry shape/device must stay fixed; create a new tracker")
            env_mask = (torch.ones(b, device=vertices.device, dtype=torch.bool)
                        if env_mask is None else torch.as_tensor(
                            env_mask, device=vertices.device, dtype=torch.bool).expand(b))
            for name, value in zip(("vertex_mask", "edges", "edge_mask"), topology):
                old = getattr(self, name)
                if old.shape != value.shape:
                    raise ValueError("Padded topology shape must stay fixed across resets")
                mask = env_mask.reshape((b,) + (1,) * (value.ndim - 1))
                setattr(self, name, torch.where(mask, value, old))
            self.previous = torch.where(env_mask[:, None, None, None], vertices,
                                        self.previous)

        low, high = self._bounds(vertices)
        g, tol = self.geometry, self.tolerance_m
        finite = (torch.isfinite(vertices) | ~self.vertex_mask[..., None]).all((-1, -2))
        empty = ~self.vertex_mask.any(-1) | ~self.edge_mask.any(-1)
        edge_a_valid = torch.gather(self.vertex_mask, -1, self.edges[..., 0])
        edge_b_valid = torch.gather(self.vertex_mask, -1, self.edges[..., 1])
        invalid_edges = (self.edge_mask & ~(edge_a_valid & edge_b_valid)).any(-1)
        side_bad = ((vertices[..., 1].abs() > self._side_width(vertices[..., 0], inner=True) + tol)
                    & self.vertex_mask).any(-1)
        invalid_initial = (~finite | empty | invalid_edges | side_bad | (low[..., 0] < g.rear_x)
                           | (high[..., 0] > g.front_x) | (low[..., 2] < -tol))
        zero = torch.zeros((b, n), device=vertices.device, dtype=torch.bool)
        for name in ("ever_exited", "wrong", "reentered", "invalid_trajectory"):
            setattr(self, name, zero.clone() if fresh else torch.where(
                env_mask[:, None], zero, getattr(self, name)))
        self.invalid_initial = (invalid_initial if fresh else torch.where(
            env_mask[:, None], invalid_initial, self.invalid_initial))
        env_zero = zero[:, 0]
        self.invalid_dynamics = (env_zero.clone() if fresh else torch.where(
            env_mask, env_zero, self.invalid_dynamics))
        float_zero = torch.zeros(b, device=vertices.device, dtype=vertices.dtype)
        self.hold_time = (float_zero if fresh else torch.where(
            env_mask, float_zero, self.hold_time))
        return self._result(zero, zero, zero)

    def _aperture_bad(self, points):
        g, tol = self.geometry, self.tolerance_m
        return ((points[..., 1].abs() > g.rear_half_width + tol)
                | (points[..., 2] < g.sill_z - tol)
                | (points[..., 2] > g.rim_z + tol))

    def _section_bad(self, vertices):
        # Actual edges give the complete plane intersection of every triangle.
        a = torch.gather(vertices, -2, self.edges[..., 0, None].expand(*self.edges.shape[:-1], 3))
        b = torch.gather(vertices, -2, self.edges[..., 1, None].expand(*self.edges.shape[:-1], 3))
        dx = b[..., 0] - a[..., 0]
        plane = self.geometry.rear_x
        intersects = (((a[..., 0] <= plane) & (b[..., 0] >= plane))
                      | ((b[..., 0] <= plane) & (a[..., 0] >= plane)))
        alpha = (plane - a[..., 0]) / torch.where(dx.abs() > 1e-12, dx,
                                                torch.ones_like(dx))
        points = a + alpha[..., None] * (b - a)
        bad = self._aperture_bad(points) & intersects & self.edge_mask
        # Coplanar edges must have BOTH endpoints inside the rectangular aperture.
        coplanar = (dx.abs() <= 1e-12) & (a[..., 0] == plane)
        bad |= coplanar & self._aperture_bad(b) & self.edge_mask
        return bad.any(-1)

    @torch.no_grad()
    def update(self, vertices_local, dt, *, stable=None, valid_dynamics=None):
        if self.previous is None:
            raise RuntimeError("reset must be called before update")
        vertices = vertices_local.detach()
        if vertices.shape != self.previous.shape or vertices.device != self.previous.device:
            raise ValueError("Geometry shape/device changed without reset")
        b = vertices.shape[0]
        dt = torch.as_tensor(dt, device=vertices.device, dtype=vertices.dtype).expand(b)
        stable = (torch.zeros(b, device=vertices.device, dtype=torch.bool) if stable is None
                  else torch.as_tensor(stable, device=vertices.device, dtype=torch.bool).expand(b))
        dynamics = (torch.ones(b, device=vertices.device, dtype=torch.bool)
                    if valid_dynamics is None else torch.as_tensor(
                        valid_dynamics, device=vertices.device, dtype=torch.bool).expand(b))
        self.invalid_dynamics = self.invalid_dynamics | ~dynamics
        previous = self.previous
        g, tol = self.geometry, self.tolerance_m
        low, high = self._bounds(vertices)
        prev_low, _ = self._bounds(previous)
        delta = vertices - previous
        finite = (torch.isfinite(vertices) | ~self.vertex_mask[..., None]).all((-1, -2))
        displacement = delta.norm(dim=-1).masked_fill(~self.vertex_mask, 0).amax(-1)
        bad_time = ~torch.isfinite(dt) | (dt <= 0) | (dt > self.max_dt_s + 1e-8)
        invalid_step = ~finite | (displacement > self.max_vertex_step_m) | bad_time[:, None]
        self.invalid_trajectory = self.invalid_trajectory | invalid_step

        crossing = ((previous[..., 0] >= g.rear_x) & (vertices[..., 0] < g.rear_x)
                    & self.vertex_mask)
        dx = delta[..., 0]
        alpha = (g.rear_x - previous[..., 0]) / torch.where(
            dx.abs() > 1e-12, dx, torch.ones_like(dx))
        crossing_points = previous + alpha[..., None] * delta
        bad_crossing = (crossing & self._aperture_bad(crossing_points)).any(-1)
        # A midpoint section limits rotational aliasing between sampled meshes.
        # Physics collision and the displacement guard remain mandatory.
        bad_section = self._section_bad(vertices) | self._section_bad((previous + vertices) * .5)
        retained_vertex = self.vertex_mask & (vertices[..., 0] >= g.rear_x)
        side_bad = ((vertices[..., 1].abs() > self._side_width(vertices[..., 0]) + tol)
                    & retained_vertex).any(-1)
        front_bad = high[..., 0] > g.front_x + tol
        bottom_bad = ((vertices[..., 2] < -tol) & retained_vertex).any(-1)
        top_departure = ((prev_low[..., 2] <= g.rim_z + tol)
                         & (low[..., 2] > g.rim_z + tol))
        raw_wrong = ((bad_crossing | bad_section | side_bad | front_bad | bottom_bad
                      | top_departure) & ~self.ever_exited & ~self.invalid_initial
                     & ~self.invalid_trajectory)
        wrong_event = raw_wrong & ~self.wrong
        self.wrong = self.wrong | raw_wrong
        # A rear-plane crossing alone is not a return to the basket: after
        # discharge a falling/landed object can move forward in the recovering
        # basket frame while remaining entirely below the floor or to its side.
        # Require spatial overlap with the finite cavity as well.  Bounds of
        # the actual collision hull conservatively retain possible reentries;
        # they are not used to relax the full-shape legal-exit predicate above.
        cavity_overlap = ((low[..., 0] < g.front_x + tol)
                          & (low[..., 1] < g.side_half_width + tol)
                          & (high[..., 1] > -g.side_half_width - tol)
                          & (low[..., 2] < g.rim_z + tol)
                          & (high[..., 2] > -tol))
        reentry_event = (self.ever_exited & ~self.reentered
                         & (high[..., 0] > g.rear_x + tol) & cavity_overlap)
        self.reentered = self.reentered | reentry_event
        first_exit = ((high[..., 0] < g.rear_x - tol) & ~self.ever_exited
                      & ~self.wrong & ~self.reentered & ~self.invalid_initial
                      & ~self.invalid_trajectory & ~self.invalid_dynamics[:, None])
        self.ever_exited = self.ever_exited | first_exit
        self.previous = vertices.clone()
        all_clear = self._valid_exited().all(-1) & ~self.invalid_dynamics
        hold = all_clear & stable & dynamics
        self.hold_time = torch.where(hold, self.hold_time + dt.clamp_min(0),
                                     torch.zeros_like(self.hold_time))
        return self._result(first_exit, wrong_event, reentry_event)

    def _valid_exited(self):
        return (self.ever_exited & ~self.wrong & ~self.reentered
                & ~self.invalid_initial & ~self.invalid_trajectory)

    def _result(self, first_exit, wrong_event, reentry_event):
        g = self.geometry
        low, _ = self._bounds(self.previous)
        state = torch.where(low[..., 0] < g.rear_x, int(ObjectState.PASSING),
                            int(ObjectState.RETAINED))
        for flag, code in ((self.ever_exited, ObjectState.EXITED),
                           (self.wrong, ObjectState.WRONG_EXIT),
                           (self.reentered, ObjectState.REENTERED),
                           (self.invalid_initial, ObjectState.INVALID_INITIAL),
                           (self.invalid_trajectory, ObjectState.INVALID_TRAJECTORY)):
            state = torch.where(flag, int(code), state)
        exited = self._valid_exited()
        all_clear = exited.all(-1) & ~self.invalid_dynamics
        return ExitResult(
            state, first_exit, wrong_event, reentry_event, self.ever_exited,
            exited, self.wrong, self.reentered, self.invalid_initial,
            self.invalid_trajectory, self.invalid_dynamics, all_clear,
            self.hold_time, all_clear & (self.hold_time >= self.success_hold_s - 1e-7))


class RearExitTracker:
    """NumPy convenience wrapper; training should use TorchRearExitTracker.

    Supports [N,V,3] (unbatched) or [B,N,V,3] arrays. The Torch backend's
    shape, topology, reset, event and failure semantics are unchanged.
    """

    def __init__(self, *args, **kwargs):
        self.backend = TorchRearExitTracker(*args, **kwargs)
        self.unbatched = False

    def _convert(self, vertices):
        vertices = np.asarray(vertices)
        if vertices.ndim == 3:
            vertices = vertices[None]
        return torch.as_tensor(vertices)

    def _numpy(self, result):
        values = {}
        for name in result.__dataclass_fields__:
            value = getattr(result, name).cpu().numpy()
            values[name] = value[0] if self.unbatched else value
        return ExitResult(**values)

    def reset(self, vertices_local, **kwargs):
        self.unbatched = np.asarray(vertices_local).ndim == 3
        return self._numpy(self.backend.reset(self._convert(vertices_local), **kwargs))

    def update(self, vertices_local, dt, **kwargs):
        return self._numpy(self.backend.update(self._convert(vertices_local), dt, **kwargs))
