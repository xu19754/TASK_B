# 启动扫描与物品地图

`AlgSolution` 默认先进行原地 RGB-D 扫描，再执行单点导航、遍历或手动运动命令。
需要打开相机，例如在已有 Isaac Lab 环境中运行：

```bash
bash scripts/run_taskb.sh --enable_cameras --steps 5000
```

扫描每 15° 停稳至少 0.3 秒采图，转满一周。它使用 `proprio.base_quat` 累计实际转角，
不会用运动命令的持续时间假定已经转完，也不会提前结束扫描来凑够 18 件。
两台相机更新周期均为 0.1 秒，停稳采图用于减少图像、机体姿态和关节角不同步的问题。
默认同时使用头部和腕部 RGB-D：先保持机身不动，将 `joint4` 从收臂的 -0.84 rad
调整为 -0.10 rad，其余机械臂关节保持训练姿态，使腕部相机向下观察近处地面。
确认实测关节角到位且稳定后才开始旋转；一周结束后保持零速度，恢复收臂并停稳，才允许导航。
这使用已有相机和正常关节控制，不修改相机安装、物体布局或评价规则。
成功或失败的扫描结果会打印到终端，并保存到 `runs/object_scan/scan-<唯一编号>/`。

- `objects.json`：物品 ID、类别、估计位置、主轴、确认状态、缺失数量和扫描状态。
- `view_00.png` 等：RGB 图像上的物品框、ID 和类别。
- `ee_view_00.png` 等：同一方向腕部相机的标注图。
- `object_map.png`：世界 XY 物品地图。
- `detection_log.json`：每个方向的地面平面、检测框、类别分数和高度范围。

## 输入与坐标约定

模块只使用允许的 `obs["image"]`、`obs["proprio"]["base_quat"]`、
`obs["proprio"]["joint_pos"]` 和 `Localization.position`。
不读取 `env.scene`、物体真实位置、初始化 seed、物理查询或语义真值。

头部相机使用 `head_rgb`、`head_depth`、`head_intrinsics`。
深度是光轴距离，先反投影到光学坐标，再用相机外参 `(0.4, 0, 0.12)` 和
机体四元数变换到世界坐标。机体横滚和俯仰也参与变换。
扫描结束后，demo 用实际观测航向校正定位的 yaw；位置仍来自原有指令积分。

腕部相机使用 `ee_rgb`、`ee_depth`、`ee_intrinsics`，安装于 `link06`，偏移 `(0.03, 0, 0.07)`。
`camera_geometry.py` 从公开 URDF 读取固定安装关系和关节轴，结合**实测关节角**做前向运动学，
再与机体姿态组合。不能拿指令目标角代替实际角，也不能把头部外参套在腕部图像上。
腕部相机有旋转的图像坐标轴，投影处理完整三维旋转，不假定图像竖直方向等于世界竖直方向。

外参与 `env_cfg.py` 一致，Isaac Lab 相机 `world` convention 为 +X 向前、+Z 向上，
参见 [官方相机配置说明](https://isaac-sim.github.io/IsaacLab/v2.3.1/_modules/isaaclab/sensors/camera/camera_cfg.html)。
若更改相机安装，请同步修改 `camera_geometry.py` 中的外参。

## 识别与结果含义

当前仓库没有已训练的物品识别权重，因此默认 `RGBDGeometryDetector` 是几何/颜色基线：
深度反投影 → 地面剔除 → 三维连通分量 → 颜色/形状分类 → 多视角空间关联。
它只覆盖糖盒 `sugar_box`、芥末瓶 `mustard_bottle`、香蕉 `banana`。
默认搜索区域为公开的物品散布区域扩展 `[-15.5, -4.5] × [-15.5, -4.5]`。

地面用 RANSAC 平面拟合剔除，补偿机身高度误差及采图姿态不同步产生的小幅倾斜。
相比只减去一个地面高度，这能避免把远处地面和搜索区域边界碎片当成物品。
默认使用全部像素（`stride=1`），前景高度门槛为 9 mm，保留低矮香蕉边缘；
低矮同高度邻点的连接距离随像素对应的三维间距调整，上限 14 cm。
分类同时使用离地高度、黄色/白色比例和形状：白色反光不直接否定香蕉，
黄色高瓶身可以在看不到商标时识别，避免把白色背标当成糖盒。

每个物品至少在两个独立采图方向获得一致的分类证据，才标为 `confirmed`。
`confidence` 是启发式一致性分数，不是校准后的统计概率。
相同 RGB-D 帧不会增加独立视角数量；不会强制将物品分配成每类六件。
同一方向的头部和腕部观测先做空间关联，合并点云，只计一次独立分类证据；
两台相机在该方向给出冲突的强类别时，该方向不投票。
`candidate_count` 是全部候选数，`stable_candidate_count` 是至少两个方向看到的候选数，
均不同于 `confirmed_count`。候选还包含 `candidate_class`、`classification_views`、
`unknown_reason`，用于区分证据不足、只在一个方向分类成功和不同方向分类冲突。

**当前输出不是完整 6D 物体姿态：**

- `position`：多视角可见表面包围盒的中心估计，单位米，不是 CAD 原点或精确质心。
- `size`：观测点云范围，遮挡时会低估真实物体尺寸，不能直接作为安全避障包络。
- `yaw_axis`：有足够视角和长轴证据时输出的水平主轴，单位弧度，模 π；否则为 `null`。
- `orientation_ambiguous=true`：始终保留长轴正反向歧义。roll/pitch 和 CAD 朝向未知。

已用用户提供的裁剪图和原始 RGB-D 做回归：旧头部记录的 seed 42 回放保持 18/18，
seed 40 保持 16/18（缺芥末瓶、香蕉各一件），没有凭已知数量填补未看到的物品。
头部相机水平安装，近处地面会落在画面下方；只原地转圈不能消除这个盲区。
新增腕部扫描已通过合成 RGB-D 的近距覆盖测试，但旧记录没有腕部帧，必须重新采集才能
验证 seed 40 的整轮结果；腕部扫描姿态的实际仿真稳定性也需要在 Isaac 环境中验证。
香蕉与无标签的黄色瓶身、遮挡、光照、图像分辨率都会影响识别。
双相机仍可能遇到遮挡，原地一周不保证每个 seed 都完整识别。
`ready=True` 表示满足当前算法的数量和分类检查，并不证明实际识别无误。

## 原始数据与离线回放

`scripts/test_object_recognition.py` 会额外保存 `rgbd/frame_XX.npz`，包括未标注的 RGB、
米制深度、相机内参、积分位置和实测机体四元数。普通 demo 默认不记录这部分数据；
需要时在扫描前设置 `solution.object_finder.record_rgbd = True`。
仅保存停稳采集的方向帧，不保存每个控制步。
腕部帧另存为 `rgbd/ee_frame_XX.npz`，额外记录实测关节角及由前向运动学得到的相机外参。
裁剪图的 `items/index.json` 增加 `source_camera`，指出图片来自头部还是腕部。

新扫描结果可以在不安装 Isaac、不启动机器人时重新运行感知：

```bash
python scripts/replay_object_recognition.py --input runs/recognition_test/<运行编号>
```

需要 NumPy 和 Pillow，输出在 `runs/recognition_replay/<新运行编号>/`，不覆盖原始数据。
离线回放沿用采集时的位姿，只验证感知和融合，不验证运动、相机时序或定位漂移。
旧的 PNG/JSON 目录不能恢复原始深度，需要重新扫描。

可批量回放整个结果目录，生成每个 seed 的前后数量及相机采集数量汇总：

```bash
python scripts/replay_object_recognition.py --batch-input runs/recognition_test
```

批次的 `summary.json` 包含所有结果目录；单次未完成会使批次返回非零退出码，但其他场景仍继续处理。
只有头部图像的旧记录可以回放，但不能凭空补出腕部观测。
仅做对比时可使用 `AlgSolution(scan_use_wrist=False)` 或识别测试的 `--head-only`。
观测缺少腕部图像或关节角时自动使用头部模式，报告会注明覆盖限制。

## 状态与后续遍历

```python
finder = solution.object_finder
items = finder.objects       # ObjectEstimate 元组；ID 是扫描关联 ID，不是场景物体编号
report = finder.report()

if finder.ready:
    # 用 items 中的类别和位置，结合具体抓取策略确定停靠 (x, y, yaw)。
    # 再按相同顺序构造 target_poses 和 obstacle_positions 调用 set_traversal。
    pass
```

仅确认 18 件、各类别恰好 6 件且没有额外稳定候选时，扫描标为 `complete`，允许后续运动。
漏检或类别不确定时标为 `incomplete`，保留全部候选并保持零速度。
相机缺失、实际转动停滞超过 90 秒或姿态异常时标为 `failed`，也保持零速度。
腕部未到位或恢复收臂未完成时同样保持零速度，不会提前将扫描标为成功。
`blocked_reason` 与 `missing_by_class` 用于排查。
模块不会自行生成未经验证的抓取停靠位姿，也不会调用 `confirm_pickup()`。

只调试其他模块时，可明确关闭启动扫描：

```python
solution = AlgSolution(scan_before_start=False)
```

新 episode 调用 `solution.reset()`（会同时重置扫描、定位与遍历）。
同一 episode 中重新扫描，在控制步之间调用 `solution.restart_scan()`，保留当前定位与拾取进度，
重新生成扫描清单和导出目录。扫描结果不会自动覆盖现有抓取位姿或障碍地图。

## 替换为训练模型

可以注入带下列接口的检测器，保留扫描、融合和导出逻辑：

```python
class TrainedDetector:
    def detect(self, rgb, depth, intrinsics, base_position, base_quaternion):
        # 返回 list[taskb_student.rgbd_perception.Detection]：
        # points_world: N×3 世界坐标点云
        # class_scores: 三个类名到 [0, 1] 分数的字典
        # bbox: RGB 图上的 (left, top, right, bottom)
        ...

solution = AlgSolution(detector=trained_detector)
```

上面的注入只替换头部检测器。替换腕部检测器时设置 `solution.object_finder.wrist_detector`，
其 `detect` 还须接受关键字参数 `camera_transform`（相机 world-convention 轴到机体的 4×4 变换）。

若需要可靠的抓取姿态，应进一步使用训练好的实例分割与物体模型姿态估计，
并用实际 RGB-D 验证；不能将当前水平主轴直接当作机器人停靠朝向或机械臂末端姿态。
