"""Real narrowphase/query isolation checks, not unloading motion acceptance."""

import numpy as np
import pytest
import torch

mujoco = pytest.importorskip("mujoco")
wp = pytest.importorskip("warp")
pytest.importorskip("mujoco_warp")

from training.basket_unload.clearance import ClearanceQuery


pytestmark = pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA unavailable")


def simple_model():
    # Use an actual mesh convex hull against a box, with ordinary main contacts
    # disabled. Explicit pairs in the *query* must still detect an overlap.
    xml = '''<mujoco>
      <asset><mesh name="cube" vertex="-.05 -.05 -.05 -.05 -.05 .05
        -.05 .05 -.05 -.05 .05 .05 .05 -.05 -.05 .05 -.05 .05
        .05 .05 -.05 .05 .05 .05"/></asset>
      <worldbody>
        <geom name="body" type="box" size=".05 .05 .05" contype="0" conaffinity="0"/>
        <body name="arm" pos=".2 0 0"><freejoint/>
          <geom name="arm_mesh" type="mesh" mesh="cube" mass="1" contype="0" conaffinity="0"/>
        </body>
      </worldbody>
    </mujoco>'''
    model = mujoco.MjModel.from_xml_string(xml)
    return model, dict(xml=xml, self_collision_pairs=np.array([[0, 1]]))


def test_true_mesh_clear_near_and_penetrating_distances_match_cpu():
    wp.init()
    wp.set_device("cuda:0")
    model, meta = simple_model()
    query = ClearanceQuery(model, meta, 3)
    data = mujoco.MjData(model)
    qpos = np.repeat(data.qpos[None], 3, axis=0)
    qpos[:, 0] = [.20, .105, .09]
    cpu_distance = []
    for q in qpos:
        data.qpos[:] = q
        mujoco.mj_forward(model, data)
        cpu_distance.append(mujoco.mj_geomDistance(model, data, 0, 1, .02, None))
    gpu_qpos = torch.tensor(qpos, device="cuda", dtype=torch.float32)
    intersect, distance = query.evaluate(gpu_qpos)
    np.testing.assert_allclose(distance.cpu(), cpu_distance, atol=1e-5)
    assert intersect.tolist() == [False, False, True]
    assert query.near_clearance_cost[1] > 0
    assert not bool(query.overflow.any())


def test_query_never_writes_to_main_model_or_state_and_never_advances_time():
    wp.init()
    wp.set_device("cuda:0")
    model, meta = simple_model()
    data = mujoco.MjData(model)
    data.qpos[0] = .09
    mujoco.mj_forward(model, data)
    masks = model.geom_contype.copy(), model.geom_conaffinity.copy()
    saved_qpos, saved_qvel = data.qpos.copy(), data.qvel.copy()
    qpos = torch.tensor(data.qpos[None], device="cuda", dtype=torch.float32)
    original = qpos.clone()
    query = ClearanceQuery(model, meta, 1)
    for _ in range(3):
        intersect, _ = query.evaluate(qpos)
        assert intersect.item()
    torch.testing.assert_close(qpos, original)
    np.testing.assert_array_equal(data.qpos, saved_qpos)
    np.testing.assert_array_equal(data.qvel, saved_qvel)
    np.testing.assert_array_equal(model.geom_contype, masks[0])
    np.testing.assert_array_equal(model.geom_conaffinity, masks[1])
    assert torch.count_nonzero(wp.to_torch(query.data.qvel)).item() == 0
    assert torch.count_nonzero(wp.to_torch(query.data.time)).item() == 0
    assert np.all(query.model.pair_margin == .01)
    assert np.all(query.model.pair_gap == .01)


def test_actual_b2z1_convex_pairs_match_cpu_for_held_out_arm_poses():
    from training.basket_unload.model import build_model, initialize_data, self_collision_distances
    wp.init()
    wp.set_device("cuda:0")
    model, meta = build_model()
    data = initialize_data(model, meta)
    rng = np.random.default_rng(61224)
    qpos, cpu_distances = [], []
    for index in range(6):
        if index:
            data.qpos[meta["joint_qpos_adr"][12:18]] = rng.uniform(
                meta["joint_lower"][12:18], meta["joint_upper"][12:18])
        mujoco.mj_forward(model, data)
        qpos.append(data.qpos.copy())
        cpu_distances.append(self_collision_distances(model, data, meta, max_distance=.02).min())
    query = ClearanceQuery(model, meta, len(qpos))
    intersect, distances = query.evaluate(torch.tensor(np.array(qpos), device="cuda", dtype=torch.float32))
    # MuJoCo CPU and Warp use distinct convex narrowphase implementations;
    # agreement here is on sampled poses, with a 0.5 mm numerical tolerance.
    np.testing.assert_allclose(distances.cpu(), cpu_distances, atol=.0005, rtol=0)
    assert intersect.tolist() == [value < -.001 for value in cpu_distances]
    assert not bool(query.overflow.any())


def test_captured_query_reads_new_qpos_without_integrating():
    wp.init()
    wp.set_device("cuda:0")
    stream = torch.cuda.Stream()
    with torch.cuda.stream(stream), wp.ScopedStream(wp.stream_from_torch(stream)):
        model, meta = simple_model()
        query = ClearanceQuery(model, meta, 1)
        qpos = torch.tensor(mujoco.MjData(model).qpos[None], device="cuda", dtype=torch.float32)
        intersect, _ = query.evaluate(qpos)
        assert not intersect.item()
        query.capture()
        qpos[:, 0] = .09
        intersect, distance = query.evaluate(qpos)
        assert intersect.item()
        assert distance.item() == pytest.approx(-.01, abs=1e-5)
        assert not wp.to_torch(query.data.time).any().item()
