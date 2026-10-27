"""按输入顺序访问物品的抓取位姿，并在确认拾取后移除对应障碍。

target_poses[i] 为 (x, y, yaw)，yaw 单位为弧度，不是高度。
obstacle_positions[i] 为对应物品的 (x, y) 或 (x, y, z)，高度忽略。
障碍用圆形包络表示；obstacle_radii 支持统一半径或每件物品的半径列表。
默认按 18 件物品使用，也支持其他数量，两组输入必须非空且数量一致。

单次规划::

    legs = plan_traversal(current_position, target_poses, obstacle_positions)
    # legs[i].waypoints 是第 i 段路径；假设每段结束后已成功拾取对应物品。

逐帧执行（推荐）::

    traversal = Traversal(localization, target_poses, obstacle_positions)
    command = traversal.compute_command()  # 交给底层控制器执行
    # 定位仍由原控制循环更新，本模块不积分、不执行抓取。
    if traversal.awaiting_pickup and grasp_succeeded:
        traversal.confirm_pickup()  # 只更新障碍地图，不删除仿真物体

机器人及障碍物尺寸是可配置的包络近似；指令积分定位的漂移会影响避障精度。
"""

from dataclasses import dataclass
import heapq
import math

import numpy as np

from .constants import CONTROL_DT
from .localization import Localization
from .motion import Motion


class NoPathError(RuntimeError):
    """给定边界、障碍包络及网格分辨率下未找到安全路径。"""


@dataclass(frozen=True)
class PlannedLeg:
    target_index: int
    target_pose: tuple[float, float, float]
    waypoints: tuple[tuple[float, float], ...]


def _xy(position):
    values = np.asarray(position, dtype=float)
    if values.shape not in ((2,), (3,)) or not np.isfinite(values).all():
        raise ValueError("Position must be finite (x, y) or (x, y, z).")
    return float(values[0]), float(values[1])


def _angle_difference(target, current):
    return (target - current + math.pi) % (2 * math.pi) - math.pi


class _ObstacleMap:
    def __init__(self, positions, radii, robot_radius, safety_margin, bounds, resolution):
        self.positions = tuple(_xy(position) for position in positions)
        radii = np.asarray(radii, dtype=float)
        if radii.ndim == 0:
            radii = np.full(len(self.positions), float(radii))
        if radii.shape != (len(self.positions),) or not np.isfinite(radii).all() or (radii < 0).any():
            raise ValueError("obstacle_radii must be non-negative finite radii, one per item or a scalar.")
        for value, name, allow_zero in (
            (robot_radius, "robot_radius", False), (safety_margin, "safety_margin", True),
            (resolution, "grid_resolution", False),
        ):
            if not math.isfinite(value) or value < 0 or (value == 0 and not allow_zero):
                raise ValueError(f"{name} is invalid.")
        bounds = np.asarray(bounds, dtype=float)
        if bounds.shape != (4,) or not np.isfinite(bounds).all():
            raise ValueError("bounds must be (xmin, xmax, ymin, ymax).")
        xmin, xmax, ymin, ymax = (float(value) for value in bounds)
        padding = robot_radius + safety_margin
        self.bounds = (xmin + padding, xmax - padding, ymin + padding, ymax - padding)
        xmin, xmax, ymin, ymax = self.bounds
        if xmin >= xmax or ymin >= ymax:
            raise ValueError("bounds must leave room for the robot and safety margin.")
        self.radii = tuple(float(radius) + padding for radius in radii)
        self.resolution = resolution
        self.nx = math.floor((xmax - xmin) / resolution) + 1
        self.ny = math.floor((ymax - ymin) / resolution) + 1
        if self.nx * self.ny > 250_000:
            raise ValueError("Grid is too large; increase grid_resolution or reduce bounds.")

    def segment_clear(self, start, end, active, extra=0.0):
        """连续线段与膨胀圆检测，防止只检测网格端点而穿过障碍。"""
        xmin, xmax, ymin, ymax = self.bounds
        for x, y in (start, end):
            if not (xmin + extra <= x <= xmax - extra and ymin + extra <= y <= ymax - extra):
                return False
        dx, dy = end[0] - start[0], end[1] - start[1]
        length_squared = dx * dx + dy * dy
        for index in active:
            cx, cy = self.positions[index]
            t = ((cx - start[0]) * dx + (cy - start[1]) * dy) / length_squared if length_squared else 0.0
            t = min(1.0, max(0.0, t))
            if math.hypot(start[0] + t * dx - cx, start[1] + t * dy - cy) <= self.radii[index] + extra:
                return False
        return True

    def plan(self, start, goal, active):
        start, goal = _xy(start), _xy(goal)
        if not self.segment_clear(start, start, active):
            raise NoPathError("Current position is inside an obstacle clearance area or outside bounds.")
        if not self.segment_clear(goal, goal, active):
            raise NoPathError("Target position is inside an obstacle clearance area or outside bounds.")
        if self.segment_clear(start, goal, active):
            return (start, goal)

        xmin, _, ymin, _ = self.bounds
        resolution = self.resolution
        start_node, goal_node = (-2, -2), (-1, -1)

        def point(node):
            if node == start_node:
                return start
            if node == goal_node:
                return goal
            return xmin + node[0] * resolution, ymin + node[1] * resolution

        def grid_neighbors(node):
            if node == start_node:
                ix = math.floor((start[0] - xmin) / resolution)
                iy = math.floor((start[1] - ymin) / resolution)
                candidates = ((ix + dx, iy + dy) for dx in (-1, 0, 1, 2) for dy in (-1, 0, 1, 2))
            else:
                candidates = ((node[0] + dx, node[1] + dy)
                              for dx in (-1, 0, 1) for dy in (-1, 0, 1) if dx or dy)
            return [n for n in candidates if 0 <= n[0] < self.nx and 0 <= n[1] < self.ny]

        costs, parents = {start_node: 0.0}, {}
        frontier = [(math.dist(start, goal), 0.0, start_node)]
        while frontier:
            _, cost, node = heapq.heappop(frontier)
            if cost > costs[node]:
                continue
            if node == goal_node:
                route = [goal]
                while node != start_node:
                    node = parents[node]
                    route.append(point(node))
                route.reverse()
                # 删除冗余中间点；每一条捷径也必须通过连续碰撞检查。
                simplified = [route[0]]
                index = 0
                while index < len(route) - 1:
                    next_index = len(route) - 1
                    while not self.segment_clear(route[index], route[next_index], active):
                        next_index -= 1
                    simplified.append(route[next_index])
                    index = next_index
                return tuple(simplified)
            origin = point(node)
            for neighbor in [*grid_neighbors(node), goal_node]:
                destination = point(neighbor)
                # 给网格绕行点留少量余量，避免跟踪时贴着物品边缘。
                extra = 0.02 if neighbor != goal_node and node != start_node else 0.0
                if not self.segment_clear(origin, destination, active, extra):
                    continue
                candidate = cost + math.dist(origin, destination)
                if candidate < costs.get(neighbor, math.inf):
                    costs[neighbor], parents[neighbor] = candidate, node
                    heapq.heappush(frontier, (candidate + math.dist(destination, goal), candidate, neighbor))
        raise NoPathError("No collision-free grid path found; check clearances, bounds or grid resolution.")


class Traversal:
    """有状态的逐点执行器：到位 -> 等待拾取确认 -> 移除障碍 -> 下一点。

    默认世界边界为场地 [-20, 0] × [-20, 0]。robot_radius=0.6 m 是
    可调整的机体圆形包络，obstacle_radii=0.2 m 是物品默认包络，需按实际尺寸设置。
    路径不可达时返回停止命令，并在 blocked_reason 中说明原因。
    """

    def __init__(self, localization, target_poses, obstacle_positions, *,
                 obstacle_radii=0.2, robot_radius=0.6, safety_margin=0.1,
                 bounds=(-20.0, 0.0, -20.0, 0.0), grid_resolution=0.2,
                 position_tolerance=0.05, yaw_tolerance=0.03):
        targets = np.asarray(target_poses, dtype=float)
        if targets.ndim != 2 or targets.shape[1] != 3 or len(targets) == 0 or not np.isfinite(targets).all():
            raise ValueError("target_poses must be a non-empty list of finite (x, y, yaw) poses.")
        if not math.isfinite(yaw_tolerance) or not 0 < yaw_tolerance < math.pi:
            raise ValueError("yaw_tolerance must be between zero and pi radians.")
        self.localization = localization
        self.targets = tuple(tuple(float(v) for v in row) for row in targets)
        self._map = _ObstacleMap(obstacle_positions, obstacle_radii, robot_radius,
                                 safety_margin, bounds, grid_resolution)
        if len(self.targets) != len(self._map.positions):
            raise ValueError("Each target pose must have exactly one corresponding obstacle.")
        self._motion = Motion(localization, position_tolerance=position_tolerance)
        self.yaw_tolerance = yaw_tolerance
        self.reset()

    def reset(self):
        """新 episode 恢复全部物品；定位由调用者单独 reset。"""
        self.target_index = 0
        self.awaiting_pickup = False
        self.blocked_reason = None
        self._active = set(range(len(self.targets)))
        self._waypoints = []

    @property
    def completed(self):
        return self.target_index == len(self.targets)

    @property
    def active_obstacle_indices(self):
        return tuple(sorted(self._active))

    @property
    def current_path(self):
        return tuple(self._waypoints)

    def replan(self):
        """取消等待拾取，清除阻塞和旧路径；按当前定位重试，保留已拾取进度。"""
        self.blocked_reason = None
        self.awaiting_pickup = False
        self._waypoints = []

    def _at_pose(self):
        x, y, yaw = self.targets[self.target_index]
        return (math.dist(self.localization.position[:2], (x, y)) <= self._motion.position_tolerance
                and abs(_angle_difference(yaw, self.localization.yaw)) <= self.yaw_tolerance)

    def confirm_pickup(self):
        """仅在已停靠且外部确认抓取成功后调用；返回被移除物品的原始索引。"""
        return self._confirm_arrival(remove_obstacle=True)

    def confirm_visit(self):
        """仅确认导航访问，保留物品障碍；用于不执行抓取的遍历测试。"""
        return self._confirm_arrival(remove_obstacle=False)

    def _confirm_arrival(self, remove_obstacle):
        if (self.completed or self.blocked_reason or not self.awaiting_pickup or not self._at_pose()
                or not self._map.segment_clear(_xy(self.localization.position),
                                               _xy(self.localization.position), self._active)):
            raise RuntimeError("Pickup can only be confirmed while stopped at the current target pose.")
        index = self.target_index
        if remove_obstacle:
            self._active.remove(index)
        self.target_index += 1
        self.awaiting_pickup = False
        self.replan()
        return index

    def plan_remaining(self, remove_after_visit=True):
        """规划剩余路径；默认假设到位后拾取，导航测试可保留全部障碍。"""
        start = _xy(self.localization.position)
        active = self._active.copy()
        result = []
        for index in range(self.target_index, len(self.targets)):
            pose = self.targets[index]
            try:
                path = self._map.plan(start, pose[:2], active)
            except NoPathError as error:
                raise NoPathError(f"Target {index}: {error}") from error
            result.append(PlannedLeg(index, pose, path))
            if remove_after_visit:
                active.discard(index)
            start = pose[:2]
        return tuple(result)

    def compute_command(self):
        """每个 CONTROL_DT 控制步调用一次，返回 (vx, 0, yaw_rate)。"""
        stop = (0.0, 0.0, 0.0)
        if self.completed or self.blocked_reason or self.awaiting_pickup:
            return stop
        start = _xy(self.localization.position)
        pose = self.targets[self.target_index]
        try:
            if not self._map.segment_clear(start, start, self._active):
                raise NoPathError("Current estimated position violates obstacle or boundary clearance.")
            if not self._waypoints:
                self._waypoints = list(self._map.plan(start, pose[:2], self._active)[1:])
            if math.dist(start, pose[:2]) <= self._motion.position_tolerance:
                error = _angle_difference(pose[2], self.localization.yaw)
                if abs(error) <= self.yaw_tolerance:
                    self.awaiting_pickup = True
                    return stop
                return (0.0, 0.0, math.copysign(min(0.6, max(0.12, 1.5 * abs(error))), error))

            while len(self._waypoints) > 1 and math.dist(start, self._waypoints[0]) <= self._motion.position_tolerance:
                self._waypoints.pop(0)
            waypoint = self._waypoints[0]
            if not self._map.segment_clear(start, waypoint, self._active):
                self._waypoints = list(self._map.plan(start, pose[:2], self._active)[1:])
                waypoint = self._waypoints[0]
            vx, _, yaw_rate = self._motion.compute_command(waypoint)
            # 先转向再直行，避免圆弧跟踪切入路径拐角内侧的障碍。
            if yaw_rate:
                return (0.0, 0.0, yaw_rate)
            end = (start[0] + vx * CONTROL_DT * math.cos(self.localization.yaw),
                   start[1] + vx * CONTROL_DT * math.sin(self.localization.yaw))
            if not self._map.segment_clear(start, end, self._active):
                raise NoPathError("Next motion step would violate clearance; stopped for replanning.")
            return (vx, 0.0, 0.0)
        except NoPathError as error:
            self.blocked_reason = f"Target {self.target_index}: {error}"
            return stop


def plan_traversal(current_position, target_poses, obstacle_positions, **options):
    """输入当前位置、18 个抓取位姿和18 件物品位置，返回按序避障路径。

    current_position 支持 (x, y) 或 (x, y, z)；目标必须为 (x, y, yaw)。
    options 与 Traversal 的关键字参数一致。无法规划时抛出 NoPathError。
    本函数仅规划路线；实际运动和拾取确认请使用 Traversal。
    """
    x, y = _xy(current_position)
    localization = Localization((x, y, 0.0))
    return Traversal(localization, target_poses, obstacle_positions, **options).plan_remaining()


def plan_path(current_position, target_position, obstacle_positions, *,
              obstacle_radii=0.2, robot_radius=0.6, safety_margin=0.1,
              bounds=(-20.0, 0.0, -20.0, 0.0), grid_resolution=0.2):
    """在保留所有物品的地图上规划单段路径，供导航停靠点选择和测试使用。"""
    obstacle_map = _ObstacleMap(obstacle_positions, obstacle_radii, robot_radius,
                                safety_margin, bounds, grid_resolution)
    return obstacle_map.plan(current_position, target_position, set(range(len(obstacle_map.positions))))
