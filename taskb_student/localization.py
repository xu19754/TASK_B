"""基于运动指令的平面航位推算，不依赖真实位置或传感器反馈。

速度为机体坐标系下的 (vx, 0, yaw_rate)，单位为 m/s、m/s、rad/s。
世界坐标系中 yaw=0 朝向 +X，正角速度逆时针转向 +Y。
仅估计 x、y 和 yaw；z 保持初始高度，打滑、碰撞和控制误差会积累漂移。
"""

import math

import numpy as np

from .constants import CONTROL_DT, SPAWN_POSITION


class Localization:
    """持续保存位置估计；每段指令实际执行完后调用一次 update。

    Example::

        localization = Localization(initial_position=(-10, -10, 0.6))
        localization.update((0.2, 0, 0), dt=1.0)
        x, y, z = localization.position  # 约 (-9.8, -10, 0.6)

    在默认 demo 中，每次 predicts 收到下一帧观测时自动累计上一帧指令。
    """

    def __init__(self, initial_position=SPAWN_POSITION, initial_yaw=0.0):
        position = np.asarray(initial_position, dtype=float)
        if position.shape != (3,) or not np.isfinite(position).all():
            raise ValueError("initial_position must contain three finite values (x, y, z).")
        initial_yaw = float(initial_yaw)
        if not math.isfinite(initial_yaw):
            raise ValueError("initial_yaw must be finite (radians).")
        self._initial_position = tuple(float(value) for value in position)
        self._initial_yaw = self._wrap_yaw(initial_yaw)
        self.reset()

    @staticmethod
    def _wrap_yaw(yaw):
        return (yaw + math.pi) % (2.0 * math.pi) - math.pi

    @staticmethod
    def prepare_command(command):
        """返回独立的命令副本，限速和死区与 ObservationBuilder 保持一致。"""
        command = np.asarray(command, dtype=np.float32).copy()
        if command.shape == (1, 3):
            command = command[0]
        if command.shape != (3,) or not np.isfinite(command).all():
            raise ValueError("command must contain three finite values (vx, 0, yaw_rate).")
        if abs(float(command[1])) > 1e-6:
            raise ValueError("model_98499 requires cmd_y=0; use forward motion and yaw to navigate.")
        command[0] = np.clip(command[0], -0.4, 0.4)
        command[2] = np.clip(command[2], -0.6, 0.6)
        command[np.abs(command) <= 0.1] = 0.0
        return command

    @property
    def position(self):
        """当前世界坐标 (x, y, z)，单位为米，返回不可变元组。"""
        return self._position

    @property
    def yaw(self):
        """当前估计航向，范围 [-pi, pi)，单位为弧度。"""
        return self._yaw

    def reset(self):
        """新 episode 开始时恢复构造时指定的初始位置与航向。"""
        self._position = self._initial_position
        self._yaw = self._initial_yaw

    def correct_heading(self, yaw):
        """用已知的世界航向校正积分漂移，不改变位置或 reset 的初始值。"""
        yaw = float(yaw)
        if not math.isfinite(yaw):
            raise ValueError("yaw must be finite (radians).")
        self._yaw = self._wrap_yaw(yaw)

    def update(self, command, dt=CONTROL_DT):
        """累计 command 已执行 dt 秒的位移，返回更新后的 (x, y, z)。

        dt 是仿真时间而非程序运行耗时；默认一个控制步（0.02 秒）。
        假定该时间段内速度恒定，同时前进和转向时按圆弧精确积分。
        """
        command = self.prepare_command(command)
        dt = float(dt)
        if not math.isfinite(dt) or dt < 0.0:
            raise ValueError("dt must be finite and non-negative (seconds).")
        vx, _, yaw_rate = (float(value) for value in command)
        half_turn = yaw_rate * dt / 2.0
        # sin(a)/a 在 a=0 处的极限为 1，同时覆盖直行和原地转向。
        sinc = math.sin(half_turn) / half_turn if half_turn else 1.0
        distance = vx * dt * sinc
        heading = self._yaw + half_turn
        x, y, z = self._position
        self._position = (
            x + distance * math.cos(heading),
            y + distance * math.sin(heading),
            z,
        )
        self._yaw = self._wrap_yaw(self._yaw + 2.0 * half_turn)
        return self.position
