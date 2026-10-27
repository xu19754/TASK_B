# 按序避障与拾取

`taskb_student/traversal.py` 将输入顺序作为拾取顺序，目标与物品按索引一一对应。
18 件物品需要传入 18 个目标位姿和 18 个物品位置。也支持其他相同数量的输入。

- 目标位姿：`(x, y, yaw)`，世界坐标米、朝向弧度；第三项不是高度。
- 物品位置：`(x, y)` 或 `(x, y, z)`；避障只使用水平坐标。
- 当前位置：来自 `Localization.position`，当前朝向来自 `Localization.yaw`。
- `obstacle_radii`：物品的水平圆形包络半径，可传一个值或逐件半径列表。
- `robot_radius`：机器人及当前载物的水平圆形包络半径。

## 只规划路径

```python
from taskb_student.traversal import plan_traversal

legs = plan_traversal(
    current_position,  # (x, y) 或 (x, y, z)
    target_poses,      # 18 个 (x, y, yaw)
    obstacle_positions,
    obstacle_radii=0.2,
    robot_radius=0.6,
    safety_margin=0.1,
)
for leg in legs:
    print(leg.target_index, leg.target_pose, leg.waypoints)
```

每段路径包含起点、绕行点和目标位置。后续段假设前面物品已成功拾取；
规划本身不会改变真实物体或执行器的拾取进度。无法找到路径时抛出 `NoPathError`。

## 接入 demo

```python
solution.set_traversal(target_poses, obstacle_positions, obstacle_radii=item_radii)
```

此后 `solution.predicts()` 自动读取定位并选择遍历命令。原有单目标和手动命令
仅在 `solution.traversal is None` 时使用。每次环境 reset 后仍应调用 `solution.reset()`。

抓取逻辑应检查以下状态，并在成功判定后调用确认接口：

```python
traversal = solution.traversal
if traversal.awaiting_pickup:
    item_index = traversal.target_index
    # 此时机器狗已到达目标位置和朝向，导航持续返回零速度。
    # 在这里驱动机械臂，并由实际抓取流程生成 grasp_succeeded。
    if grasp_succeeded:
        traversal.confirm_pickup()
```

确认后，物品仅从避障地图移除，下一控制步前往下一件物品。
抓取失败或尚未完成时不调用确认，机器人保持停止。
`completed` 表示所有物品已确认拾取；`active_obstacle_indices` 为尚未移除的物品索引。
`blocked_reason` 非空表示路径或下一运动步不安全，此时命令为零。
修正定位等外部条件后，可调用 `replan()` 重新尝试当前目标；该操作取消等待拾取，
但保留已拾取进度。不要在机械臂仍进行抓取时调用。

## 边界与精度

默认场地边界为 `(-20, 0, -20, 0)`，网格分辨率为 0.2 m，到位误差为 0.05 m、0.03 rad。
路径按“物品半径 + 机器人半径 + 安全余量”避障，也为场地边界保留机器人包络。
默认三者之和为 0.9 m；若抓取位姿与物品中心距离不足，会报告无法到达，
因此应按实际尺寸设置包络和安全余量，不能通过提前移除未拾取物品来绕过碰撞检测。

寻路使用八邻域 A*，所有路径段都做连续线段碰撞检测；执行时先转向再直行，
每步检查预测运动是否进入障碍包络。该检查基于静态物品地图与指令积分定位，
不包含实测位置纠偏、动态障碍检测或机械臂抓取算法。
