# 启动扫描与物品地图

`AlgSolution` 默认先进行原地 RGB-D 扫描，再执行单点导航、遍历或手动运动命令。
需要打开相机，例如在已有 Isaac Lab 环境中运行：

```bash
bash scripts/run_taskb.sh --enable_cameras --steps 5000
```

扫描每 15° 停稳至少 0.3 秒采图，转满一周。它使用 `proprio.base_quat` 累计实际转角，
不会用运动命令的持续时间假定已经转完，也不会提前结束扫描来凑够 18 件。
头部相机更新周期为 0.1 秒，停稳采图用于减少相机数据和机器人姿态不同步的问题。
成功或失败的扫描结果会打印到终端，并保存到 `runs/object_scan/scan-<唯一编号>/`。

- `objects.json`：物品 ID、类别、估计位置、主轴、确认状态、缺失数量和扫描状态。
- `view_00.png` 等：RGB 图像上的物品框、ID 和类别。
- `object_map.png`：世界 XY 物品地图。
- `detection_log.json`：每个方向的地面平面、检测框、类别分数和高度范围。

## 输入与坐标约定

模块只使用允许的 `obs["image"]`、`obs["proprio"]["base_quat"]` 和 `Localization.position`。
不读取 `env.scene`、物体真实位置、初始化 seed、物理查询或语义真值。

头部相机使用 `head_rgb`、`head_depth`、`head_intrinsics`。
深度是光轴距离，先反投影到光学坐标，再用相机外参 `(0.4, 0, 0.12)` 和
机体四元数变换到世界坐标。机体横滚和俯仰也参与变换。
扫描结束后，demo 用实际观测航向校正定位的 yaw；位置仍来自原有指令积分。

外参与 `env_cfg.py` 一致，Isaac Lab 相机 `world` convention 为 +X 向前、+Z 向上，
参见 [官方相机配置说明](https://isaac-sim.github.io/IsaacLab/v2.3.1/_modules/isaaclab/sensors/camera/camera_cfg.html)。
若更改相机安装，请同步修改 `project_head_depth()` 中的外参。

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
`candidate_count` 是全部候选数，`stable_candidate_count` 是至少两个方向看到的候选数，
均不同于 `confirmed_count`。候选还包含 `candidate_class`、`classification_views`、
`unknown_reason`，用于区分证据不足、只在一个方向分类成功和不同方向分类冲突。

**当前输出不是完整 6D 物体姿态：**

- `position`：多视角可见表面包围盒的中心估计，单位米，不是 CAD 原点或精确质心。
- `size`：观测点云范围，遮挡时会低估真实物体尺寸，不能直接作为安全避障包络。
- `yaw_axis`：有足够视角和长轴证据时输出的水平主轴，单位弧度，模 π；否则为 `null`。
- `orientation_ambiguous=true`：始终保留长轴正反向歧义。roll/pitch 和 CAD 朝向未知。

已用用户提供的扫描裁剪图及旧几何摘要做分类回归，并用合成 RGB-D 检查地面剔除。
旧结果没有原始深度，尚不能据此声称真实整轮扫描达到 18/18 或测得识别准确率。
香蕉与无标签的黄色瓶身、遮挡、光照、图像分辨率都会影响识别。
原地一周也无法看到被其他物品遮住或位于相机近处盲区的物品。
`ready=True` 表示满足当前算法的数量和分类检查，并不证明实际识别无误。

## 原始数据与离线回放

`scripts/test_object_recognition.py` 会额外保存 `rgbd/frame_XX.npz`，包括未标注的 RGB、
米制深度、相机内参、积分位置和实测机体四元数。普通 demo 默认不记录这部分数据；
需要时在扫描前设置 `solution.object_finder.record_rgbd = True`。
仅保存停稳采集的方向帧，不保存每个控制步。

新扫描结果可以在不安装 Isaac、不启动机器人时重新运行感知：

```bash
python scripts/replay_object_recognition.py --input runs/recognition_test/<运行编号>
```

需要 NumPy 和 Pillow，输出在 `runs/recognition_replay/<新运行编号>/`，不覆盖原始数据。
离线回放沿用采集时的位姿，只验证感知和融合，不验证运动、相机时序或定位漂移。
旧的 PNG/JSON 目录不能恢复原始深度，需要重新扫描。

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

若需要可靠的抓取姿态，应进一步使用训练好的实例分割与物体模型姿态估计，
并用实际 RGB-D 验证；不能将当前水平主轴直接当作机器人停靠朝向或机械臂末端姿态。
