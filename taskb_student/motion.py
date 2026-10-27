"""根据 Localization 位姿生成平面目标点导航命令，不包含避障或路径规划。"""

import math

import numpy as np

from .localization import Localization


class Motion:
    """每个控制步读取最新定位，输出 (vx, 0, yaw_rate)。

    目标支持世界坐标 (x, y) 或 (x, y, z)，单位为米；只控制 x、y，
    三维输入的 z 不参与到达判定。航向使用 localization.yaw（弧度）。
    默认距目标不超过 0.1 m 时停止，不要求到达后的最终朝向。

    Example::

        localization = Localization()
        motion = Motion(localization)
        command = motion.compute_command((-8.0, -9.0))
        # 将 command 交给 LowLevelController.step；执行后再更新定位。

    本类只读取定位，不调用 update，避免与 demo 的定位更新重复积分。
    """

    def __init__(self, localization: Localization, position_tolerance=0.1):
        position_tolerance = float(position_tolerance)
        if not math.isfinite(position_tolerance) or position_tolerance <= 0.0:
            raise ValueError("position_tolerance must be finite and positive (meters).")
        self.localization = localization
        self.position_tolerance = position_tolerance

    def _target_error(self, target_position):
        target = np.asarray(target_position, dtype=float)
        if target.shape not in ((2,), (3,)) or not np.isfinite(target).all():
            raise ValueError("target_position must be finite world coordinates (x, y) or (x, y, z).")
        x, y, _ = self.localization.position
        dx, dy = float(target[0]) - x, float(target[1]) - y
        distance = math.hypot(dx, dy)
        if not math.isfinite(distance):
            raise ValueError("Distance to target must be finite.")
        desired_yaw = math.atan2(dy, dx)
        error = (desired_yaw - self.localization.yaw + math.pi) % (2.0 * math.pi) - math.pi
        return distance, error

    def is_at_target(self, target_position):
        """当前估计位置是否位于目标的水平距离容差内。"""
        distance, _ = self._target_error(target_position)
        return distance <= self.position_tolerance

    def compute_command(self, target_position):
        """返回速度命令：vx 单位 m/s，yaw_rate 单位 rad/s。

        朝向误差超过 45 度时原地转向，否则边前进边修正航向。
        接近目标时减速；非零速度至少为 0.12，以避开底层 <=0.1 的死区。
        到达后完全停止；每帧可以传入不同目标，无需重置导航器。
        """
        distance, heading_error = self._target_error(target_position)
        if distance <= self.position_tolerance:
            return (0.0, 0.0, 0.0)

        # 小于约 1.7 度无需转向，避免微小误差导致持续左右摆动。
        if abs(heading_error) <= 0.03:
            yaw_rate = 0.0
        else:
            yaw_rate = math.copysign(
                min(0.6, max(0.12, 1.5 * abs(heading_error))), heading_error
            )

        if abs(heading_error) > math.pi / 4.0:
            vx = 0.0
        else:
            vx = min(0.4, max(0.12, 0.8 * distance * math.cos(heading_error)))
        return (vx, 0.0, yaw_rate)
