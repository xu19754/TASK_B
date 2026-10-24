"""Check genuine batched CUDA physics and RGB-D before building the task."""
import json
from pathlib import Path
import time

import mujoco
import mujoco_warp as mjw
import numpy as np
import torch
import warp as wp

from .progress import RUN_DIR, update


def main():
    update("阶段 1 / 9：验证 GPU 物理和 RGB-D", "首次编译 CUDA 内核；使用隔离的小场景核对批量相机光轴深度。")
    torch.set_num_threads(2)
    wp.init()
    with wp.ScopedDevice("cuda:0"):
        model = mujoco.MjModel.from_xml_string('''<mujoco>
          <option timestep=".002" integrator="implicitfast"/>
          <worldbody><light pos="0 0 3"/><camera name="probe" pos="0 0 2"/>
            <geom type="plane" size="3 3 .01"/>
            <body pos="0 0 .5"><freejoint/><geom type="box" size=".1 .1 .1" mass=".5" rgba=".9 .2 .1 1"/></body>
          </worldbody></mujoco>''')
        cpu = mujoco.MjData(model)
        mujoco.mj_forward(model, cpu)
        gpu_model = mjw.put_model(model)
        data = mjw.put_data(model, cpu, nworld=4, nconmax=32, njmax=64)
        qpos = wp.to_torch(data.qpos)
        # Isolated diagnostic initialization: each world starts at a different X.
        qpos[:, 0] = torch.tensor([0., .3, -.3, .5], device="cuda")
        mjw.forward(gpu_model, data)
        context = mjw.create_render_context(model, nworld=4, cam_res=(64, 48), render_rgb=True, render_depth=True)
        mjw.render(gpu_model, data, context)
        wp.synchronize()
        depth = wp.to_torch(context.depth_data).reshape(4, 48, 64)
        centre = depth[:, 24, 32].cpu().numpy()
        np.testing.assert_allclose(centre, [1.4, 2., 2., 2.], atol=.02)
        # Initial compilation is excluded from the timed physics measurement.
        mjw.step(gpu_model, data)
        wp.synchronize()
        started = time.perf_counter()
        with wp.ScopedCapture() as capture:
            for _ in range(10):
                mjw.step(gpu_model, data)
        for _ in range(10):
            wp.capture_launch(capture.graph)
        wp.synchronize()
        elapsed = time.perf_counter() - started
        assert torch.isfinite(wp.to_torch(data.qpos)).all()
        report = dict(mujoco=mujoco.__version__, warp=wp.__version__, device=str(wp.get_device()),
                      nworld=4, optical_depth_m=centre.tolist(), physics_steps=100,
                      elapsed_s=elapsed, world_steps_per_s=400 / elapsed,
                      note="Small backend probe only; not the robot task throughput or unloading validation.")
        (RUN_DIR / "backend_probe.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
        print(json.dumps(report, indent=2), flush=True)
    update("阶段 1 / 9：GPU 后端验证通过", "四个 CUDA 物理世界及批量 RGB-D 已验证；光轴深度和世界间独立性符合预期。")


if __name__ == "__main__":
    main()
