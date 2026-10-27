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
它启动仿真，原地旋转一周、停稳采集 RGB-D，完成后立即保存结果并结束。
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
├── object_map.png            # 世界 XY 分布图
├── view_00.png ...           # 全景扫描中各方向的框选标注图
├── scan_log.jsonl            # 扫描进度、实际累计转角
└── summary.json              # seed、是否通过、保存数量及停止原因
```

图片名称包含 `物品 ID_识别类别_分数_世界坐标`。
类别对应为 `sugar_box`（糖盒）、`mustard_bottle`（芥末瓶）、`banana`（香蕉）。
`unknown` 表示没有足够证据，不能作为已确认物品凑数。
识别成功时 `items/` 中应有 18 张 PNG；若漏检，只保存实际确认的数量，退出码为 2，
可查看 `candidates/`、`objects.json` 和 `summary.json` 排查。
这测试的是现有识别流程，没有读取物品真实标签计算准确率；图片用于人工检查。

## 3. 测试遍历并保存视频

完成上一步、得到 18 件已确认物品后执行：

```bash
bash scripts/run_feature_test.sh traversal --headless
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

每到达一个位置并对齐朝向，停留 1 秒，然后继续下一件。
这是只访问、不抓取的测试：全部物品仍留在场地，也始终留在避障地图中。
测试不会删除仿真物品、修改评价规则或调用教师侧真实位姿接口。

结果默认保存在：

```text
runs/traversal_test/<日期时间_运行编号>/
├── traversal.mp4             # 1280×720 场地视角，默认 25 FPS
├── route.json                # 此次使用的 18 个停靠位姿、障碍位置和预规划路径
├── trajectory.jsonl          # 每步估计位置、朝向、运动命令和当前目标
├── visits.json               # 每个物品的访问时间及到达位姿
└── summary.json              # 已访问数量、视频帧数、是否通过、停止原因
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
bash scripts/run_feature_test.sh traversal --headless \
  --max-seconds 1800 --fps 25 --dwell-seconds 1 --per-target-timeout 120

# 自定义结果父目录；第二个程序使用 --input 指向其中具体的一次运行
bash scripts/run_feature_test.sh recognition --headless --output-dir /path/to/results
```

`--max-seconds` 和 `--per-target-timeout` 都按仿真时间计算；`--max-walltime-s` 可另设真实时间上限。
`--seed` 在遍历测试中一般不传；显式传入时必须与输入文件的 seed 一致。

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

## 判断测试结果

- 退出码 `0`：完成数量检查（识别 18 件，或按估计位姿访问 18 点）。
- 退出码 `2`：未完成，如漏检、不可达、跌倒、仿真终止或时间上限。
- 退出码 `1`：运行错误；`130`：用户中断。

位置到达判定仍使用 `localization.py` 的指令积分估计，不能证明实际机器狗与目标无偏差。
视频和轨迹用于检查打滑、碰撞和误识别。完整物理测试需在你的 Isaac 环境执行。

录制接口依据 [Isaac Lab 的 RGB-array 渲染接口](https://isaac-sim.github.io/IsaacLab/v2.3.1/source/api/lab/isaaclab.envs.html)
和 [ImageIO FFmpeg 插件](https://imageio.readthedocs.io/en/v2.21.3/_autosummary/imageio.plugins.ffmpeg.html)。
