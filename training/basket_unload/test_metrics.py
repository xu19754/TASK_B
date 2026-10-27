"""Synthetic geometry trajectories only; these do not validate robot dynamics."""

from itertools import product
from pathlib import Path

import numpy as np
import pytest
import torch

from training.basket_unload.metrics import (
    ExitGeometry, ObjectState, RearExitTracker, TorchRearExitTracker,
    world_to_basket,
)


SIGNS = np.array(list(product((-1, 1), repeat=3)), dtype=float)
BOX_EDGES = np.array([(a, b) for a in range(8) for b in range(a + 1, 8)
                      if np.count_nonzero(SIGNS[a] != SIGNS[b]) == 1])


def boxes(centers, half=(.012, .012, .012)):
    return np.asarray(centers)[:, None, :] + SIGNS[None] * np.asarray(half)


def tracker(vertices, **kwargs):
    result = RearExitTracker(**kwargs)
    result.reset(vertices, edges=np.broadcast_to(BOX_EDGES, (vertices.shape[0], 12, 2)))
    return result


def move(evaluator, start, end, *, steps=40, dt=.01, stable=False):
    results = []
    for fraction in np.linspace(0, 1, steps + 1)[1:]:
        vertices = start + (end - start) * fraction
        results.append(evaluator.update(vertices, dt, stable=stable))
    return results


def test_aperture_matches_v3_cad_rear_inner_edges():
    root = Path(__file__).resolve().parents[2]
    geometry = ExitGeometry.from_v3_properties(
        root / "assets/b2z1/basket_v3/basket_properties.json")
    mesh = root / "assets/b2z1/meshes/basket_v3/basket_visual.obj"
    vertices = np.array([[float(value) for value in line.split()[1:]]
                         for line in mesh.read_text().splitlines() if line.startswith("v ")])
    rear = vertices[np.isclose(vertices[:, 0], geometry.rear_x, atol=1e-8)]
    for y, z in product((-geometry.rear_half_width, geometry.rear_half_width),
                         (geometry.sill_z, geometry.rim_z)):
        assert np.any(np.linalg.norm(rear[:, 1:] - (y, z), axis=1) < 1e-7)


def test_three_complete_passages_reward_once_then_require_stable_hold():
    initial = boxes([[0, -.05, .06], [0, 0, .06], [0, .05, .06]])
    evaluator = tracker(initial, success_hold_s=.5)
    final = initial + (-.28, 0, 0)
    results = move(evaluator, initial, final)
    assert np.array([result.first_exit for result in results]).sum(axis=0).tolist() == [1, 1, 1]
    assert results[-1].all_clear
    assert not results[-1].success  # Empty alone does not certify stability.
    for _ in range(49):
        result = evaluator.update(final, .01, stable=True)
    assert not result.success
    result = evaluator.update(final, .01, stable=True)
    assert result.success
    assert not result.first_exit.any()
    result = evaluator.update(final, .01, stable=False)
    assert not result.success
    assert result.hold_time == 0


@pytest.mark.parametrize("offset", [(0, .18, 0), (.30, 0, 0), (0, 0, .20)])
def test_side_front_or_top_escape_cannot_be_relabelled_as_rear(offset):
    initial = boxes([[0, 0, .07]])
    evaluator = tracker(initial)
    wrong_location = initial + offset
    first = move(evaluator, initial, wrong_location)
    assert sum(result.wrong_exit.sum() for result in first) == 1
    tail_location = initial + (-.30, 0, 0)
    second = move(evaluator, wrong_location, tail_location)
    assert second[-1].wrong[0]
    assert not any(result.first_exit.any() for result in second)
    assert not second[-1].all_clear


def test_partial_crossing_can_reverse_but_full_reentry_is_terminal():
    initial = boxes([[0, 0, .07]])
    evaluator = tracker(initial)
    partial = initial + (-.22, 0, 0)
    result = move(evaluator, initial, partial)[-1]
    assert result.state[0] == ObjectState.PASSING
    assert not result.first_exit[0]
    result = move(evaluator, partial, initial)[-1]
    assert not result.reentered[0]
    final = initial + (-.28, 0, 0)
    first = move(evaluator, initial, final)
    assert sum(result.first_exit.sum() for result in first) == 1
    back = move(evaluator, final, partial)
    assert sum(result.reentry.sum() for result in back) == 1
    assert back[-1].reentered[0]
    repeated = move(evaluator, partial, final)
    assert not any(result.first_exit.any() for result in repeated)
    assert not repeated[-1].all_clear


@pytest.mark.parametrize("outside_offset", [(0, 0, -.70), (0, .30, 0), (0, 0, .30)])
def test_exited_object_outside_cavity_is_not_reentered_on_body_recovery(outside_offset):
    initial = boxes([[0, 0, .07]])
    evaluator = tracker(initial)
    tail_exit = initial + (-.28, 0, 0)
    result = move(evaluator, initial, tail_exit)[-1]
    assert result.exited[0]

    # The object is already fully outside the tail.  It can then fall below
    # the basket (or remain beside/above it) while the recovering/moving body
    # carries the rear plane behind it.  Its x projection overlaps the cavity,
    # but its complete collision hull still does not enter the basket.
    outside = tail_exit + outside_offset
    move(evaluator, tail_exit, outside, steps=100)
    recovered_frame = outside + (.15, 0, 0)
    results = move(evaluator, outside, recovered_frame)
    assert recovered_frame[..., 0].max() > evaluator.backend.geometry.rear_x
    assert not any(result.reentry.any() for result in results)
    assert results[-1].exited[0]
    assert not results[-1].invalid_trajectory[0]
    for _ in range(50):
        result = evaluator.update(recovered_frame, .01, stable=True)
    assert result.success


@pytest.mark.parametrize("center", [[-.28, 0, .07], [0, .14, .07]])
def test_initial_outside_never_earns_exit(center):
    initial = boxes([center])
    evaluator = tracker(initial)
    result = evaluator.update(initial, .01, stable=True)
    assert result.invalid_initial[0]
    assert not result.first_exit[0]
    assert not result.all_clear


def test_initial_stack_above_rim_can_settle_and_exit():
    initial = boxes([[0, 0, .21]])
    evaluator = tracker(initial)
    settled = boxes([[0, 0, .07]])
    falling = move(evaluator, initial, settled)
    assert not falling[-1].invalid_initial[0]
    assert not falling[-1].wrong[0]
    final = settled + (-.28, 0, 0)
    results = move(evaluator, settled, final)
    assert results[-1].all_clear


def test_wide_front_region_is_valid_but_tapered_side_escape_is_wrong():
    initial = boxes([[.05, .109, .07]], half=(.005, .004, .008))
    evaluator = tracker(initial)
    assert not evaluator.update(initial, .01).invalid_initial[0]
    near_tail = initial + (-.265, 0, 0)
    result = move(evaluator, initial, near_tail)[-1]
    assert result.wrong[0]  # Rear side narrows even before the opening.


@pytest.mark.parametrize("z", [.009, .17])
def test_crossing_tail_at_wrong_height_fails(z):
    initial = boxes([[0, 0, z]], half=(.004, .004, .004))
    evaluator = tracker(initial)
    final = initial + (-.28, 0, 0)
    result = move(evaluator, initial, final)[-1]
    assert result.wrong[0]
    assert not result.exited[0]


def test_true_shape_section_not_center_catches_low_crossing_corner():
    # The center is at a legal height; one lower corner goes through the sill.
    initial = boxes([[0, 0, .04]], half=(.03, .012, .03))
    evaluator = tracker(initial)
    final = initial + (-.29, 0, 0)
    result = move(evaluator, initial, final)[-1]
    assert result.wrong[0]
    assert not result.all_clear


def test_edge_plane_section_catches_violation_without_vertex_crossing_this_step():
    # An object already straddles the plane legally, then rotates/translates so
    # its edge cuts the aperture bottom. No vertex crosses x in the second step.
    initial = boxes([[-.19, 0, .05]], half=(.03, .01, .02))
    evaluator = tracker(initial, max_vertex_step_m=.04)
    straddled = initial + (-.02, 0, 0)
    result = evaluator.update(straddled, .01)
    assert not result.wrong[0]
    lowered = straddled + (0, 0, -.025)
    result = evaluator.update(lowered, .01)
    assert result.wrong_exit[0]


@pytest.mark.parametrize("dt", [.01, .02])
def test_large_displacement_or_time_gap_invalidates_instead_of_awarding(dt):
    initial = boxes([[0, 0, .07]])
    evaluator = tracker(initial)
    final = initial + (-.30 if dt == .01 else -.001, 0, 0)
    result = evaluator.update(final, dt, stable=True)
    assert result.invalid_trajectory[0]
    assert not result.first_exit[0]
    assert not result.all_clear


def test_dynamics_failure_blocks_success_and_latches_until_reset():
    initial = boxes([[0, 0, .07]])
    evaluator = tracker(initial, success_hold_s=.01)
    final = initial + (-.28, 0, 0)
    move(evaluator, initial, final)
    result = evaluator.update(final, .01, stable=True, valid_dynamics=False)
    assert result.invalid_dynamics
    assert not result.success
    result = evaluator.update(final, .01, stable=True, valid_dynamics=True)
    assert not result.success


def test_dynamic_basket_coordinates_are_invariant_to_world_motion():
    local = boxes([[0, 0, .07]])[None]
    angle = .8
    rotation = np.array([[[np.cos(angle), 0, np.sin(angle)], [0, 1, 0],
                          [-np.sin(angle), 0, np.cos(angle)]]])
    position = np.array([[1., -2., .6]])
    world = np.einsum("b...k,bjk->b...j", local, rotation) + position[:, None, None]
    np.testing.assert_allclose(world_to_basket(world, position, rotation), local, atol=1e-12)
    tensor_local = world_to_basket(torch.tensor(world), torch.tensor(position), torch.tensor(rotation))
    np.testing.assert_allclose(tensor_local.numpy(), local, atol=1e-12)


def test_batch_local_reset_keeps_other_environment_history():
    initial = torch.tensor(boxes([[0, 0, .07]])[None].repeat(2, 0), dtype=torch.float32)
    evaluator = TorchRearExitTracker(success_hold_s=.01)
    edges = torch.tensor(BOX_EDGES[None])
    evaluator.reset(initial, edges=edges)
    for x in np.linspace(0, -.28, 41)[1:]:
        final = initial + torch.tensor([x, 0, 0], dtype=torch.float32)
        result = evaluator.update(final, .01, stable=True)
    assert result.success.tolist() == [True, True]
    mixed = torch.cat((initial[:1], final[1:]))
    result = evaluator.reset(mixed, edges=edges, env_mask=[True, False])
    assert result.ever_exited[:, 0].tolist() == [False, True]
    assert result.success.tolist() == [False, True]


def test_masked_padding_and_nan_failure_are_per_object():
    initial = boxes([[0, 0, .07], [0, .04, .07]])[None]
    padded = np.concatenate((initial, np.full((*initial.shape[:2], 1, 3), np.nan)), axis=-2)
    mask = np.array([[True] * 8 + [False]] * 2)
    evaluator = RearExitTracker()
    evaluator.reset(padded, vertex_mask=mask, edges=BOX_EDGES[None].repeat(2, 0))
    result = evaluator.update(padded, .01)
    assert not result.invalid_initial.any()
    corrupted = padded.copy()
    corrupted[0, 0, 0] = np.nan
    result = evaluator.update(corrupted, .01)
    assert result.invalid_trajectory.tolist() == [[True, False]]


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA unavailable")
def test_cuda_matches_cpu_without_moving_state_off_device():
    initial = torch.tensor(boxes([[0, 0, .07]])[None], dtype=torch.float32)
    cpu, gpu = TorchRearExitTracker(), TorchRearExitTracker()
    cpu.reset(initial, edges=BOX_EDGES[None])
    gpu.reset(initial.cuda(), edges=BOX_EDGES[None])
    for x in np.linspace(0, -.28, 41)[1:]:
        frame = initial + torch.tensor([x, 0, 0], dtype=torch.float32)
        expected = cpu.update(frame, .01, stable=True)
        actual = gpu.update(frame.cuda(), .01, stable=True)
        for field in expected.__dataclass_fields__:
            assert getattr(actual, field).is_cuda
            torch.testing.assert_close(getattr(actual, field).cpu(), getattr(expected, field))
