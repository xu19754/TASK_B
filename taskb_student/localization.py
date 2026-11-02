"""平面定位：支持观测速度积分，也保留指令积分用于离线对照。

速度为机体坐标系下的 (vx, 0, yaw_rate)，单位为 m/s、m/s、rad/s。
世界坐标系中 yaw=0 朝向 +X，正角速度逆时针转向 +Y。
update_from_proprio 使用公开的 base_lin_vel 和 base_quat，不读取场景位置。
仅估计 x、y 和 yaw；z 保持初始高度，速度观测偏差仍会积累漂移。
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

    默认 demo 使用 update_from_proprio；update 保留原指令积分接口。
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
        self._previous_world_velocity = None
        self._body_velocity = None
        self._quaternion = None
        self._measured_yaw_rate = None
        self._source = "uninitialized"
        self._samples = 0

    @staticmethod
    def _observation_vector(value, size, name):
        if hasattr(value, "detach"):
            value = value.detach().cpu().numpy()
        vector = np.asarray(value, dtype=float)
        if vector.shape == (1, size):
            vector = vector[0]
        if vector.shape != (size,) or not np.isfinite(vector).all():
            raise ValueError(f"{name} must be a finite {size}-vector or single-environment batch.")
        return vector.copy()

    @property
    def is_stationary(self):
        """实测平面速度 <5 cm/s 且航向变化 <0.1 rad/s；指令模式无测速判据。"""
        return (self._previous_world_velocity is None
                or (np.linalg.norm(self._previous_world_velocity[:2]) <= 0.05
                    and abs(self._measured_yaw_rate) <= 0.1))

    def diagnostics(self):
        return {"source": self._source, "samples": self._samples,
                "base_quat": None if self._quaternion is None else self._quaternion.tolist(),
                "body_linear_velocity": None if self._body_velocity is None else self._body_velocity.tolist(),
                "world_linear_velocity": (None if self._previous_world_velocity is None
                                          else self._previous_world_velocity.tolist()),
                "measured_yaw_rate": self._measured_yaw_rate, "stationary": bool(self.is_stationary)}

    def update_from_proprio(self, proprio, dt=CONTROL_DT):
        """用相邻观测的世界速度作梯形积分，并直接使用实测航向。

        episode 初始观测必须 dt=0：只缓存初始速度/姿态，不虚构已发生的位移。
        之后每次 env.step 的观测调用一次。机体速度包含侧滑和停车余速，
        不应用运动命令的限速/死区；三维姿态参与速度转换，XY 用于平面定位。
        缺失或无效观测时报错，不静默回退到指令积分。
        """
        dt = float(dt)
        if not math.isfinite(dt) or dt < 0:
            raise ValueError("dt must be finite and non-negative.")
        if "base_lin_vel" not in proprio or "base_quat" not in proprio:
            raise ValueError("Feedback localization requires proprio.base_lin_vel and proprio.base_quat.")
        velocity = self._observation_vector(proprio["base_lin_vel"], 3, "base_lin_vel")
        q = self._observation_vector(proprio["base_quat"], 4, "base_quat")
        norm = np.linalg.norm(q)
        if norm < 1e-8 or not np.isfinite(norm):
            raise ValueError("base_quat must have nonzero finite norm (wxyz).")
        q /= norm
        w, x, y, z = q
        world = velocity + 2 * np.cross(q[1:], np.cross(q[1:], velocity) + w * velocity)
        yaw = math.atan2(2 * (w*z + x*y), 1 - 2 * (y*y + z*z))
        if dt > 0 and self._previous_world_velocity is None:
            raise ValueError("Initialize feedback localization with dt=0 before integrating observations.")
        position = np.asarray(self._position).copy()
        if dt:
            position[:2] += 0.5 * (self._previous_world_velocity[:2] + world[:2]) * dt
        if not np.isfinite(world).all() or not np.isfinite(position).all():
            raise ValueError("Feedback integration produced non-finite values.")
        yaw_rate = self._wrap_yaw(yaw - self._yaw) / dt if dt else 0.0
        self._position = tuple(float(v) for v in position)
        self._yaw = self._wrap_yaw(yaw)
        self._previous_world_velocity = world
        self._body_velocity = velocity
        self._quaternion = q
        self._measured_yaw_rate = yaw_rate
        self._source = "proprio"
        self._samples += 1
        return self.position

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
        self._previous_world_velocity = None
        self._body_velocity = None
        self._quaternion = None
        self._measured_yaw_rate = None
        self._source = "command"
        return self.position
