# 物品识别与十八点遍历测试

这两个程序需要在原有 **Isaac Lab / Isaac Sim 环境**中运行。先进入项目根目录，
激活能够运行 `scripts/run_taskb.sh` 的 Python 环境。Linux 推荐使用下面的 Bash 入口，
它保留现有启动脚本所需的 PyTorch 动态库搜索路径。

## 1. 准备视频依赖

在同一个 Isaac Python 环境中执行一次：

```bash
python -m pip install imageio imageio-ffmpeg
```

也可在安装本项目时使用 `pip install -e '.[video]'`。
如果通过 `TASKB_PYTHON` 指定解释器，安装依赖时也使用该解释器。
两个测试都会自动启用相机；`--headless` 只关闭窗口，仍需 NVIDIA GPU 做 RTX 渲染。

## 2. 测试物品识别

```bash
bash scripts/run_feature_test.sh recognition --headless --seed 42
```

对应程序：`scripts/test_object_recognition.py`。
它启动仿真，将腕部相机调整为俯视方向，停稳后原地旋转一周，同时采集头部与腕部 RGB-D。
结束后恢复收臂并停稳，保存结果并结束。仍不下达平移命令。
默认最多运行 120 秒仿真时间；内部扫描器的超时仍为 90 秒。
需要观察仿真窗口时，去掉 `--headless`。

结果默认保存在**项目根目录**：

```text
runs/recognition_test/<日期时间_运行编号>/
├── items/                    # 已确认物品，一件一张最大可见裁剪图
│   ├── item_00_sugar_box_conf0.80_x-12.34_y-8.76_z0.08.png
│   └── index.json            # 图片名称、物品 ID、类别、坐标和原始图片位置
├── candidates/               # 识别未完成时保存全部候选，包括 unknown
├── objects.json              # 多视角融合结果与缺失类别统计
├── detection_log.json        # 地面拟合、逐帧类别分数、检测框和高度范围
├── rgbd/frame_00.npz ...     # 原始 RGB-D、内参、估计位置和实测四元数，供离线回放
├── rgbd/ee_frame_00.npz ...  # 腕部 RGB-D，额外记录实测关节角与相机外参
├── object_map.png            # 世界 XY 分布图
├── view_00.png ...           # 全景扫描中各方向的框选标注图
├── ee_view_00.png ...        # 同方向腕部相机的框选标注图
├── scan_log.jsonl            # 扫描进度、实际累计转角
└── summary.json              # seed、是否通过、保存数量及停止原因
```

图片名称包含 `物品 ID_识别类别_分数_世界坐标`。
类别对应为 `sugar_box`（糖盒）、`mustard_bottle`（芥末瓶）、`banana`（香蕉）。
`unknown` 表示没有足够证据，不能作为已确认物品凑数。
识别成功时 `items/` 中应有 18 张 PNG；若漏检，只保存实际确认的数量，退出码为 2，
可查看 `candidates/`、`objects.json` 和 `summary.json` 排查。
这测试的是现有识别流程，没有读取物品真实标签计算准确率；图片用于人工检查。

`objects.json` 中的 `candidate_count` 是全部候选数，`confirmed_count` 才是已确认物品数。
`candidate_class`、`classification_views`、`unknown_reason` 可以解释为何一个候选仍是 `unknown`。
`captured_views_by_camera` 可以检查两台相机是否都采到了图；裁剪清单的 `source_camera` 标记图片来源。
不会通过截断候选数量或强行按每类 6 件分配来让测试通过。

新版记录的原始帧可在普通 Python 环境（NumPy、Pillow）中离线重跑识别：

```bash
python scripts/replay_object_recognition.py --input runs/recognition_test/<日期时间_运行编号>
```

结果另存到 `runs/recognition_replay/<新运行编号>/`。旧版本只有图片和 JSON 的结果无法回放，
需重新运行扫描测试生成深度数据。回放不启动 Isaac，不产生机器人运动。

为了确认不是只适配 seed 42，建议先复测当前失败的 40，再测试其他布局：

```bash
for seed in 40 42 0 1 43; do
  bash scripts/run_feature_test.sh recognition --headless --seed "$seed"
done
python scripts/replay_object_recognition.py --batch-input runs/recognition_test
```

批量回放的 `summary.json` 汇总每次输入的 seed、修改前后确认数量和每台相机的帧数。
所有场景通过才返回 0；存在未完成结果返回 2。上述批量仿真命令需在 Isaac 环境运行，
不表示这些 seed 都已完成验证。需要与原来的头部扫描对比时，附加 `--head-only`。

用户提供的旧记录中，seed 42 为 18/18、seed 40 为 16/18，二者都只有头部帧。
新版回放保留这个事实；腕部补充视野需要重跑仿真采集，无法从旧的头部帧恢复。

## 3. 测试遍历并保存视频

完成上一步、得到 18 件已确认物品后执行：

```bash
bash scripts/run_feature_test.sh traversal --headless --no-pickup
```

对应程序：`scripts/test_traversal_video.py`。
默认通过 `runs/recognition_test/latest.json` 找到最近一次识别结果，沿用其中的 seed 创建场景。
识别未完成时会在机器人运动前报错，不会填造缺失坐标。

也可以指定某一次识别结果：

```bash
bash scripts/run_feature_test.sh traversal --headless \
  --input runs/recognition_test/<日期时间_运行编号>
```

默认按识别 ID 的输入顺序生成 18 个面向物品的导航停靠位姿。不同类别使用不同的期望停靠距离，
并检查机器人包络、物品包络、场地边界和路径可达性。这些是**导航测试位姿，不是经过验证的抓取位姿**。
若无法找到安全停靠点，程序报告错误，不会越过障碍。

默认关闭夹取（`--no-pickup`）。每到达一个位置并对齐朝向，停留 **3 秒仿真时间**，然后继续下一件。
这是只访问、不抓取的测试：全部物品仍留在场地，也始终留在避障地图中。
测试不会删除仿真物品、修改评价规则或调用教师侧真实位姿接口。

使用 `--dwell-seconds 5` 可将停留改为 5 秒。导航超时不打断已经开始的到位停留，
但全局 `--max-seconds` 和环境终止条件仍有效。停留期间若估计位姿离开容差范围，会重新到位并重新计时。

`--pickup` 打开夹取确认模式：到位后等待外部夹取控制器调用 `confirm_pickup()`，
确认后才继续并从导航地图移除该物品。**当前默认机械臂控制器仅保持收臂，没有实现自动夹取**；
单独打开此开关不会产生抓取动作，未接入夹取控制器会报告 `pickup_timeout`。
关闭开关时也会跳过 `select_arm_target()` 中的夹取逻辑，保持收臂。

结果默认保存在：

```text
runs/traversal_test/<日期时间_运行编号>/
├── traversal.mp4             # 1280×720 场地视角，默认 25 FPS
├── route.json                # 此次使用的 18 个停靠位姿、障碍位置和预规划路径
├── trajectory.jsonl          # 每步估计位置、实测速度/姿态、指令推算对照、航点及停留进度
├── visits.json               # 每个物品的访问时间及到达位姿
└── summary.json              # 已访问数量、视频帧数、夹取开关、停止原因和最终导航诊断
```

视频带有已访问数量、当前目标、仿真时间。按仿真时间抽帧，仿真运行慢不会让成片加速。
正常结束或 Ctrl+C 中断时都会关闭编码器，保存已经录到的部分；异常断电或强制杀进程不保证 MP4 完整。
在路线校验或环境初始化阶段失败时，可能还没有视频帧，但会尽量保存失败摘要。

运行结束时终端会输出 `[Result]` 和实际结果目录；每个测试的父目录下也有 `latest.json` 指向最近一次运行。
每次运行创建新目录，不覆盖上一次图片或视频。

## 指定你自己的十八个停靠位姿

可复制某次生成的 `route.json`，修改各物品的 `target_pose` 后重新运行：

```bash
bash scripts/run_feature_test.sh traversal --headless --input /path/to/my_route.json
```

完整输入须包含 `schema: "taskb-navigation-v1"`、场景 `seed`、18 个 `items`，每项包括：

```json
{
  "object_id": 0,
  "class_name": "sugar_box",
  "position": [-12.0, -10.0, 0.1],
  "obstacle_radius": 0.2,
  "target_pose": [-13.1, -10.0, 0.0]
}
```

上面只是单项结构示例，不是该 seed 的真实物品坐标。应使用实际识别结果或你已有的地图。
`target_pose` 为世界坐标 `(x, y, yaw)`，米和弧度；第三项不是高度。
必须为所有物品指定目标，或全部不指定并让程序生成。
可在顶层 `navigation` 设置 `robot_radius`、`safety_margin`、`grid_resolution`、
`bounds`、`position_tolerance`、`yaw_tolerance`，含义与 `Traversal` 相同。

## 常用参数及 Windows 入口

```bash
# 调整运行上限、编码帧率、到位停留时间
bash scripts/run_feature_test.sh traversal --headless --no-pickup \
  --max-seconds 1800 --fps 25 --dwell-seconds 3 --per-target-timeout 120

# 自定义结果父目录；第二个程序使用 --input 指向其中具体的一次运行
bash scripts/run_feature_test.sh recognition --headless --output-dir /path/to/results
```

`--max-seconds` 和 `--per-target-timeout` 都按仿真时间计算；`--max-walltime-s` 可另设真实时间上限。
`--seed` 在遍历测试中一般不传；显式传入时必须与输入文件的 seed 一致。

`--localization proprio` 为识别和遍历的默认定位模式：用公开观测 `base_quat`
将 `base_lin_vel` 从机体坐标转换到世界坐标，相邻帧速度作梯形积分，航向每帧取实测姿态。
侧滑、转弯时位移及停车余速均参与 XY 定位，z 仍保留初始高度。
初始观测只缓存速度和姿态，不积分；重置 episode 时清空历史。观测缺失或无效会报错，
不会悄悄退回指令积分。纯离线接口 `Localization.update(command)` 仍然保留。

`--localization command` 仅供与旧算法对比，会重新启用容易累计漂移的指令积分，
不建议用于实际遍历。两种模式都会在摘要中写明 `localization_mode` 和 `completion_basis`。
新轨迹中的 `localization` 字段记录实测机体/世界速度、四元数，以及指令推算的位置、航向与差值；
差值用于诊断执行偏差，不是独立测量的定位误差。默认模式到位后还要求平面速度不超过
0.05 m/s、实测航向变化率不超过 0.1 rad/s，才开始/累计停留计时。

在自己的控制程序中可使用相同开关：

```python
solution.set_traversal(target_poses, object_positions,
                       pickup_enabled=False, dwell_seconds=3.0)
# predicts() 每 0.02 秒仿真时间调用一次，由 Traversal 自动完成到位停留和切换。
# 开启夹取则设 pickup_enabled=True，在实际夹取成功后调用：
# solution.traversal.confirm_pickup()
```

Windows 下在配置好的 Isaac Python 环境、项目根目录中可以直接运行：

```powershell
python scripts/test_object_recognition.py --headless --seed 42
python scripts/test_traversal_video.py --headless
```

当前这份工作区的默认结果根路径是：

```text
D:\Robotic Workspace\TaskB\runs\recognition_test\
D:\Robotic Workspace\TaskB\runs\traversal_test\
```

在 Linux 服务器上部署时，目录位于服务器上的项目根目录，程序会打印完整路径。

## 本次 seed 13 超时的原因与复测

提供的 `20261028_123854_665312` 日志中，15.46 秒之后一直为零运动命令，
`awaiting_visit` 从未为真，最终距第一个停靠位姿约 1.752 米。
原因是靠近中间绕障拐点时提前切换航点，捷径碰撞检查失败后又重规划回当前拐点，
而点控制器认为已在该拐点的 5 厘米容差内，因此持续停止。原测试本身没有执行夹取。

现在只有下一段直线通过避障检查时才允许跳过拐点；否则继续靠近拐点。
该次实际路线已加入离线回归，当时修复版按指令积分能在约 386.48 秒完成 18 次访问和每次 3 秒停留，
保留全部 18 个障碍；这是导航逻辑验证，仍需在 Isaac 中复测实际运动。
服务器上可以直接复用失败测试保存的路线：

```bash
bash scripts/run_feature_test.sh traversal --headless --no-pickup --dwell-seconds 3 \
  --input runs/traversal_test/20261028_123854_665312/route.json
```

## 2026-11-02：后半程偏离的原因与复测

提供的 seed 42 记录 `20261028_144339_c07423` 虽然报告在 394.62 秒完成 18/18，
视频后半程机器人已远离物品区。旧导航始终积分下达的速度，未使用现有实测速度和姿态，
因此控制执行误差会累计，而访问检查仍在同一套错误的估计坐标中通过。
旧日志没有实测速度/姿态，无法从中精确重建实际轨迹，也不能用旧记录证明新算法的仿真效果。

默认定位现已改用上述观测反馈。路径搜索、简化和航点切换也保留最多 5 cm 的额外跟踪余量，
减少侧滑时贴近障碍边缘；这不会缩小原障碍包络。合成回归在用户的这条路线中加入
18% 前进速度损失、22% 转向不足和侧滑：旧指令积分产生超过 1 米的到位偏差，
反馈模式完成 18 次访问，并将该合成模型中的到位误差保持在 5 cm 内。
这些结果验证控制逻辑，不能替代 Isaac 中的实际运动测试。

服务器更新代码后可先复用本次路线进行对照：

```bash
bash scripts/run_feature_test.sh traversal --headless --no-pickup --dwell-seconds 3 \
  --localization proprio --input runs/traversal_test/20261028_144339_c07423/route.json
```

之后建议重新运行识别测试，再使用新识别目录运行遍历；这样物品地图也会使用更新后的扫描定位。
物品识别位置、初始位姿、速度观测偏差仍可能影响结果。新结果需结合视频检查，不能只看访问数量。

## 判断测试结果

- 退出码 `0`：完成数量检查（识别 18 件，或按估计位姿访问 18 点）。
- 退出码 `2`：未完成，如漏检、不可达、跌倒、仿真终止或时间上限。
- 退出码 `1`：运行错误；`130`：用户中断。

位置到达判定使用 `localization.py` 的所选定位模式，默认观测速度积分，不能证明实际机器狗与目标无偏差。
视频和轨迹用于检查打滑、碰撞和误识别。完整物理测试需在你的 Isaac 环境执行。

录制接口依据 [Isaac Lab 的 RGB-array 渲染接口](https://isaac-sim.github.io/IsaacLab/v2.3.1/source/api/lab/isaaclab.envs.html)
和 [ImageIO FFmpeg 插件](https://imageio.readthedocs.io/en/v2.21.3/_autosummary/imageio.plugins.ffmpeg.html)。
