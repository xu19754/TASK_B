"""CUDA MuJoCo simulation; normal transitions only change actuator targets."""
from __future__ import annotations

import time

import mujoco
import mujoco_warp as mjw
import numpy as np
import torch
import warp as wp

from .model import build_model, initialize_data


class GPUSimulation:
    def __init__(self, num_envs=16, timestep=.002, batch_sizes=None, model_meta=None):
        wp.init()
        wp.set_device("cuda:0")
        self.device = torch.device("cuda:0")
        # CUDA graph capture requires a non-default stream. Keep Torch and Warp
        # on the same owned stream so shared arrays have a defined dependency.
        self.torch_stream = torch.cuda.Stream(device=self.device)
        self.torch_stream.wait_stream(torch.cuda.current_stream(self.device))
        torch.cuda.set_stream(self.torch_stream)
        self.warp_stream = wp.stream_from_torch(self.torch_stream)
        wp.set_stream(self.warp_stream)
        self.num_envs = num_envs
        self.model, self.meta = model_meta or build_model(timestep=timestep)
        self.cpu_data = initialize_data(self.model, self.meta)
        self.gpu_model = mjw.put_model(self.model, batch_sizes=batch_sizes)
        self.data = mjw.put_data(self.model, self.cpu_data, nworld=num_envs, nconmax=512, njmax=1024)
        self.qpos = wp.to_torch(self.data.qpos)
        self.qvel = wp.to_torch(self.data.qvel)
        self.ctrl = wp.to_torch(self.data.ctrl)
        self.xpos = wp.to_torch(self.data.xpos)
        self.xmat = wp.to_torch(self.data.xmat)
        self.joint_adr = torch.as_tensor(self.meta["joint_qpos_adr"], device=self.device, dtype=torch.long)
        self.dof_adr = torch.as_tensor(self.meta["joint_dof_adr"], device=self.device, dtype=torch.long)
        self.nominal = torch.as_tensor(self.meta["nominal_q"], device=self.device, dtype=torch.float32)
        self.lower = torch.as_tensor(self.meta["joint_lower"], device=self.device, dtype=torch.float32)
        self.upper = torch.as_tensor(self.meta["joint_upper"], device=self.device, dtype=torch.float32)
        self.velocity = torch.as_tensor(self.meta["velocity"], device=self.device, dtype=torch.float32)
        self.effort = torch.as_tensor(self.meta["effort"], device=self.device, dtype=torch.float32)
        self.targets = self.nominal.expand(num_envs, -1).clone()
        self.graph = None
        mjw.forward(self.gpu_model, self.data)

    def capture_step(self):
        # Compile kernels before capture, then capture one genuine physics step.
        mjw.step(self.gpu_model, self.data)
        mjw.kinematics(self.gpu_model, self.data)
        mjw.camlight(self.gpu_model, self.data)
        wp.synchronize()
        with wp.ScopedCapture() as capture:
            mjw.step(self.gpu_model, self.data)
            mjw.kinematics(self.gpu_model, self.data)
            mjw.camlight(self.gpu_model, self.data)
        self.graph = capture.graph

    def physics_step(self):
        if self.graph is None:
            mjw.step(self.gpu_model, self.data)
            mjw.kinematics(self.gpu_model, self.data)
            mjw.camlight(self.gpu_model, self.data)
        else:
            wp.capture_launch(self.graph)

    def set_targets(self, requested, control_dt=.02):
        if requested.shape != (self.num_envs, 19):
            raise ValueError("Expected [num_envs,19] absolute joint targets")
        requested = requested.clamp(self.lower, self.upper)
        slew = self.velocity * control_dt
        previous_gripper = self.targets[:, 18].clone()
        self.targets.add_((requested - self.targets).clamp(-slew, slew))
        actual = self.qpos[:, self.joint_adr]
        closing = requested[:, 18] > actual[:, 18]
        lead = torch.maximum(previous_gripper - slew[18], actual[:, 18] + .04)
        self.targets[:, 18] = torch.where(closing, torch.minimum(self.targets[:, 18], lead), self.targets[:, 18])
        self.targets.clamp_(self.lower, self.upper)
        self.ctrl.copy_(self.targets)


def benchmark(num_envs=16, steps=100):
    from .progress import RUN_DIR, update
    import json
    update("阶段 2 / 9：真实机器人 GPU 编译", f"编译 {num_envs} 个完整浮动基座世界，保留 103 块篮筐碰撞体。")
    sim = GPUSimulation(num_envs)
    sim.capture_step()
    started = time.perf_counter()
    for _ in range(steps):
        sim.physics_step()
    wp.synchronize()
    elapsed = time.perf_counter() - started
    finite = bool(torch.isfinite(sim.qpos).all())
    report = dict(nworld=num_envs, steps=steps, elapsed_s=elapsed,
                  world_steps_per_second=num_envs * steps / elapsed, finite=finite,
                  base_height_m=sim.qpos[:, 2].cpu().tolist(),
                  note="GPU compilation and throughput only; initial cargo is not yet reset into basket.")
    (RUN_DIR / "robot_gpu_benchmark.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report, indent=2), flush=True)
    if not finite:
        raise RuntimeError("Nonfinite GPU robot state")
    update("阶段 2 / 9：真实机器人 GPU 基准完成", f"{num_envs} 个世界运行完成；正在接入随机载物、尾窗判据和腕部图像。")


if __name__ == "__main__":
    benchmark()
