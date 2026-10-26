# Task B：B2 + Z1 物体收集作业

项目接手请先读 [HANDOFF.md](HANDOFF.md)：包含当前实现、验证边界、随机物料与速度超限结论，以及示范采集 → DP3 → 残差 PPO 的待办顺序。智能体工作约束见 [AGENTS.md](AGENTS.md)。

本工程是学生作业的基础工程，仅包含 Task B。指定机器人为 **Unitree B2 + Z1**，任务 ID 为 `Student-TaskB-B2Z1-v0`。

## 作业内容

工程提供机器人与场景、物理仿真、雷达与 RGB-D、区域计分器、底层行走 policy，以及 `demo/solution.py` 学生策略模板。

**学生需要自行实现感知、搜索导航、抓取、带物运输和投放，完成完整闭环。这里不提供完整抓取搬运答案，也不要求重新训练底层步态。** 默认模板只保持站立和收臂，不会自动收集物体；运行后得零分是预期现象，不代表计分器损坏。

建议先完成单物体闭环，再逐步扩展到多个物体，并在不同 seed 下测试。完整闭环是学生的作业目标，不是本包内预先完成的示例功能。

## 任务与计分

场地为 20 m × 20 m，世界坐标范围为 `x,y ∈ [-20, 0]`。机器人从 `(-10, -10, 0.6)` 出生，18 个物体分布在 `x,y ∈ [-15, -5]` 的 10 m × 10 m 区域内，包括 6 个糖盒、6 个芥末瓶、6 根香蕉。`--seed` 控制物体布局；初始化时避开机器人和其他物体，换 seed 可以测试泛化能力。

**总分 18 分，每个物体最多获得 1 分。** 原线上赛每个物体的“末端靠近/触碰奖励”已删除，不能再通过低位扫臂或仅靠近物体拿分。只保留物体进入固定投放区的分数，未对保留的分数再乘以 0.5。

投放区是橙色开放圆桶，中心 `(-3, -10)`，判分水平半径 `1.0 m`，物体 root 高度须在 `[0.0, 0.5] m`。同一个物体离开再进入不会重复得分；新 episode 才清零。所有物体同时位于投放区内时完成任务。

区域得分沿用官方的物体 root 几何判据，不额外判断是否松爪或静止。因此这是“固定区域投放计分”，不是夹持成功检测器。机器人靠近物体、夹爪闭合、物体抬高、装入随车篮子本身均没有单独奖励。

环境保留机身/髋/大腿异常接触和跌落终止。默认 episode 上限为 **1200 秒仿真时间**；`--max-walltime-s` 可另加真实时间上限，二者不会混为一谈。

## 指定机器人与控制接口

默认 URDF：`assets/b2z1/b2z1_0815_with_basket_v3.urdf`，已安装 200 克尾部篮子，附带 `meshes/` 和 `SOURCE.json`。原始参考模型保留为 `assets/b2z1/b2z1_0815.urdf`。运行时通过 Isaac Lab 的 URDF importer 加载，不依赖老师电脑上的绝对路径；启动日志会打印实际使用的 URDF 路径。

机器人有 **19 个可控关节**。动作全部是绝对关节位置，单位弧度，顺序如下：

```text
FR_hip_joint, FR_thigh_joint, FR_calf_joint,
FL_hip_joint, FL_thigh_joint, FL_calf_joint,
RR_hip_joint, RR_thigh_joint, RR_calf_joint,
RL_hip_joint, RL_thigh_joint, RL_calf_joint,
joint1, joint2, joint3, joint4, joint5, joint6, jointGripper
```

环境动作 `scale=1`、`use_default_offset=False`，不要再次乘 0.5，也不要按原 Piper 的 20 维动作拼接。Z1 的夹爪是单个旋转关节：`jointGripper=0` 为闭合端，负值向打开端运动，范围 `[-1.57, 0]`，不能按米或双指平移位移控制。

夹爪采用 `80 N·m/rad` 刚度、`8 N·m·s/rad` 阻尼、`5 N·m` 力矩上限和 `0.25 rad/s` 限速；底层适配器另外限制闭爪目标相对实际关节角的超前量。不要将其他臂关节的快速驱动参数直接套到夹爪，也不要认为发出全闭目标就一定已经抓住物体。抓取对准、闭合过程与抬升验证需要由学生策略实现；限速和限力不能保证任意轨迹均不会穿透或掉落。

## 老师提供的底层行走 policy

**老师提供现有 B2+Z1 行走 policy，学生不必从零训练机械狗步态。** 权重在 `models/locomotion/policy.pt`，为平地 `domain_rand_r0 / model_98499`，并附部署契约 `deployment_manifest.json`。该策略有直行段的航向参考反馈，可以作为导航底盘。

配套 `taskb_student/low_level.py` 负责关节重排、52 维观测、2 Hz 步态相位、直行段航向参考、策略内部历史/LSTM 重置，以及 12 维腿动作到关节目标的转换。不能把 `obs` 原样送入 `policy.pt`，也不要自行再堆叠 10 帧；导出的模型会管理内部历史。

新 episode 开始时用 `controller.reset()` 清理全部控制状态。抓取过程中若需要重新初始化步态历史，应使用 `controller.reset_locomotion()`，它保留机械臂与夹爪目标，避免重置瞬间卸掉夹持预紧力。这不是抓取成功检测器，也不代替学生的运输控制。

给底层的速度命令为 `(cmd_x, 0, cmd_yaw)`：此 policy 的横移命令固定为零，通过转向后前进完成导航。当前教学接口把前进/后退限制在 `[-0.4, 0.4] m/s`，转向限制在 `[-0.6, 0.6] rad/s`；绝对值不超过 `0.1` 的命令按训练契约归零。

低层 policy 的训练姿态为机械臂收拢：`[0, 1.48, -0.63, -0.84, 0, 1.57, 0]`。伸臂、装载篮筐、增加负载后仍需重新验证稳定性；提供行走权重不意味着任意机械臂姿态或载重下都已经验证。

当前导入配置关闭机器人内部自碰撞，机械臂规划仍须避开机身和腿，不能把穿过自身的轨迹当成可执行抓取。此开关与夹爪对物体的外部碰撞无关；夹爪与物体的碰撞体保持启用。

## 当前默认安装的尾部篮子 v3

正式入口默认加载最新的 `assets/b2z1/b2z1_0815_with_basket_v3.urdf`，包含机器人、最新篮子外观、103 块凸碰撞网格和重算后的质心/惯性。原始无篮子参考模型 `b2z1_0815.urdf` 保留。

篮子来自用户提供的最新版 `零件2.SLDPRT`，外尺寸约 **450 × 250 × 160 mm**，壁厚约 1.5 mm；尾沿约 15 mm，尾部主斜面约 30°，带底部过渡。质量沿用 **0.2 kg 的设定值，未经实测**；质心与惯性按该质量和 CAD 壳体网格计算。局部坐标 X 前、Y 左、Z 上，固定安装到 `base`，位姿为 `xyz=(-0.30, 0, 0.13) m`、`rpy=(0, 0, 0)`。

模型说明、源 CAD 与哈希清单位于 `assets/b2z1/basket_v3/`，网格位于 `assets/b2z1/meshes/basket_v3/`。机器人原有 19 个可控关节、限位及传感器挂载 link 保持原样；篮子不是目标得分区，装篮本身不计分。

可探索先装载多个物体再集中运输，但最新 v3 接入后的空篮站立、装载、带载行走、卸料与整机平衡仍需在 Isaac Sim/PhysX 中验证。独立试验中的卸料表现不等于正式环境已验证或已得分；本包不附带这些试验脚本、轨迹和报告，也没有新增自动装篮/卸料控制器。先完成空篮站立，再逐步增加负载并验证策略。

篮子尺寸、形状和容量不另设限制；安装在背后是建议。改装仍需真实底板、侧壁、碰撞体、合理质量与惯性，并检查机械臂和后腿活动空间及重心变化。不能使用瞬移、隐藏约束或修改得分区域代替运输。

## 修改与提交规则

除指定 B2+Z1 模型、删除末端接近得分以及允许篮子改装外，任务区域、保留的计分判据和异常终止规则沿用原 Task B。不要修改计分器、物体状态、碰撞规则、目标区或终止条件来获得分数。默认仿真时间上限仍是 1200 秒；真实时间上限由运行参数另行设置，不自动改成 30 分钟。

学生策略入口为 `demo/solution.py` 中的 `AlgSolution.predicts(obs, current_score)`，返回 `action` 和 `giveup`，不可改名。可在 `demo/` 下增加策略辅助代码和自定义模型，并随提交附带实际使用的权重与依赖说明；如安装篮子，同时提交修改后的机器人资产及其网格资源。不要修改评价脚本、环境计分和终止条件。本包为 Z1 明确约定 19 维绝对位置动作，不能直接使用原 Piper 的关节接口。

学生只能使用规定的策略观测和 `current_score` 反馈，不得从评价器、场景对象或物理接口读取物体真实位姿，也不得直接写入物体状态。目标桶中心是公开任务信息，不等于物体真实坐标。教师侧的真实位姿、抓取及物体瞬移诊断程序不包含在本学生包内。

## 安装与检查

本次服务器使用 **Isaac Sim 5.0 + Isaac Lab v2.2.1 + Python 3.11 + PyTorch 2.7**。请先按 NVIDIA 官方文档安装这组环境并激活对应 Python 环境，不要只安装 `isaaclab` 包而缺少 Isaac Sim。Isaac Sim/Isaac Lab、GPU 驱动不在作业压缩包内。所需 URDF、网格和三个计分物体资源已附带。物体材质引用的 `OmniPBR.mdl` 由 Isaac Sim 自带材质库提供。

从解压后的工程根目录执行：

```bash
conda activate isaaclab
python -m pip install -e .
python scripts/check_package.py
```

请保留工程目录布局并采用 editable 安装，模型与资源按工程根目录定位。本工程使用独立包名 `taskb_student`，不要求安装旧的 `atec_rl_lab`。

先用收臂、零速度的默认示例检查仿真：

```bash
bash scripts/run_taskb.sh --steps 500 --debug
```

短距离测试提供的行走底盘：

```bash
bash scripts/run_taskb.sh --demo-command 0.2 0 0 --steps 500 --debug
```

打开前视与腕部 RGB-D，运行学生实现：

```bash
bash scripts/run_taskb.sh \
  --task Student-TaskB-B2Z1-v0 --num_envs 1 \
  --seed 42 --enable_cameras --debug
```

设置真实 30 分钟上限的示例：

```bash
bash scripts/run_taskb.sh --enable_cameras --max-walltime-s 1800 --debug
```

真实计时从 `env.reset()` 后开始；仿真 episode、异常终止或主动 `giveup` 仍可能更早结束。该命令只示范真实时间限制的设置方式，不将默认 1200 秒仿真时间改成 30 分钟。云服务器需打开租赁平台的图形桌面并正确设置 `DISPLAY`，SSH 本身不会显示图形窗口；无图形桌面时添加 `--headless`。

`run_taskb.sh` 使用当前环境的 `python`，并补充 PyTorch 的动态库搜索路径，避免部分云镜像报 `libtorch_cuda_linalg.so` 找不到。可用 `TASKB_PYTHON=/完整路径/python` 指定解释器；不要用系统 Python 启动已经安装在 conda 中的 Isaac。直接运行 `python scripts/play_taskb.py` 也受支持，但需要该环境的动态库路径已配置正确。

## 雷达与摄像头

**提供一组机身 LiDAR，以及前视、腕部各一组 RGB-D 摄像头。** 它们在 `env_cfg.py` 中运行时挂载，不修改指定 URDF 的机械结构和 19 维关节契约。URDF 文件名及机身网格中的 `nolidar` 表示原始机械资产不自带雷达，不表示当前运行环境没有雷达。

LiDAR 默认启用，不需要 `--enable_cameras`。它使用 Isaac Sim PhysX 场景碰撞射线，能够检测带碰撞体的静态和动态物体，不是只检测地面的高度扫描器。规格：水平 360°、1° 采样；竖直 16 线、线间距 3.75°；量程 `[0.15, 12.0] m`。每次采样完整视场，未模拟转动扫描畸变、真实激光反射强度或硬件噪声，也不提供物体标签、语义或真实坐标。

`obs["lidar"]` 的接口如下：

| 键 | 形状 | 含义 |
| --- | --- | --- |
| `ranges` | `(1, 360, 16)` | 距离，单位米；未命中填 `12.0` |
| `points` | `(1, 360, 16, 3)` | 雷达局部坐标系中的点；未命中填零 |
| `hit_mask` | `(1, 360, 16)` | 布尔有效命中标记，处理点云前必须使用 |
| `ready` | `(1, 1)` | 传感器数据是否已初始化 |

雷达相对 `base` 平移为 `(0.38, 0, 0.23) m`，不附加旋转；局部轴为 x 前、y 左、z 上。可由 `point_base = point_lidar + [0.38, 0, 0.23]` 转到机身坐标。机械臂和机身可能遮挡视线，小物体也可能落在雷达线束间隙，近距离抓取应结合 RGB-D。不能把未命中点当作地面物体。

两组相机都需通过 `--enable_cameras` 启用；不加时不创建相机，也没有 `obs["image"]`。两组均为 `640×480`、目标更新周期 `0.1 s`，提供 RGB 和米制光轴深度。前视安装在 `base`，腕部安装在 `link06`，具体位姿见各 `CameraCfg.OffsetCfg`。这些是教学仿真外参，不是实机标定。

导出一次实际传感器快照：

```bash
bash scripts/run_taskb.sh --headless --enable_cameras --steps 500 \
  --sensor-output runs/sensors --diagnostics runs/standing.jsonl --debug
```

`runs/sensors/` 中保存 `head_rgb.png`、`ee_rgb.png` 和包含雷达、RGB、深度、内参的 `sensors.npz`。无图形窗口服务器加 `--headless`；只关闭窗口不会关闭相机。脚本会在相机和雷达数据就绪后再保存快照。

自定义启动器须在创建 `AppLauncher` 之前给 `args.kit_args` 添加 `--enable isaacsim.sensors.physx`，确保 USD 在初始化 schema registry 前加载雷达类型；仅在创建场景后启用扩展不够。

当前单环境教学包默认使用 **CPU PhysX + GPU 图像渲染**，启动器也会拒绝 `--device cuda:0`，避免这台镜像中雷达停在出生点的 GPU 位姿回读问题。CPU 并不意味着不用 Isaac，也不意味着用 CPU 渲染 RGB-D。不要直接套用大批量行走训练中关闭场景查询、只保留 GPU/Fabric 位姿的设置；若日后改成 GPU 物理，需要先验证传感器随机器人移动、动态物体更新和深度一致性，不能仅检查输出形状。

## 学生实现入口

### 示范采集准备

相机校验和录制复用上面的正式场景与启动入口，现有相机安装参数保持一致：

```bash
bash scripts/run_taskb.sh --headless --enable_cameras --seed 42 --steps 250 \
  --record-dir runs/recording_smoke_seed42_001
python scripts/inspect_recording.py runs/recording_smoke_seed42_001
```

默认录制站立模板的链路检查，按 `smoke` 保存，不代表成功示范。录制仅含规定的 proprio / LiDAR / RGB-D 观测、19 维实际提交关节目标及时间／得分／终止反馈，自动导出现有两组相机的校验报告和预览。相机输出检查不等于外参或深度精度标定。启动与自定义示范控制器接入、分块格式和验证边界见 [示范采集准备](docs/示范采集准备.md)。

在 `demo/solution.py` 中实现 `AlgSolution.select_command()` 与 `select_arm_target()`，已有底层控制器负责生成 19 维动作。也可以扩展 `predicts(obs, current_score)`，但需要保持返回格式：

```python
return {"action": nineteen_absolute_joint_targets, "giveup": False}
```

`obs["proprio"]` 是字典，各项含环境批维度 `1`：

| 键 | 形状 | 含义 |
| --- | --- | --- |
| `base_lin_vel` | `(1, 3)` | 机体坐标线速度 |
| `base_ang_vel` | `(1, 3)` | 机体坐标角速度 |
| `projected_gravity` | `(1, 3)` | 机体坐标重力方向 |
| `base_quat` | `(1, 4)` | 机身姿态，四元数 `wxyz` |
| `joint_pos` | `(1, 19)` | 按上述顺序的绝对关节角 |
| `joint_vel` | `(1, 19)` | 同顺序的关节角速度 |

带 `--enable_cameras` 时，`obs["image"]` 提供 `head_rgb`、`head_depth`、`ee_rgb`、`ee_depth` 和对应的 `head_intrinsics`、`ee_intrinsics`。RGB 形状为 `(1,480,640,3)`，深度为 `(1,480,640,1)`；内参矩阵 K 为 `(1,3,3)`。深度是 `distance_to_image_plane`，不是从相机中心到像素的欧氏距离；反投影时按光轴深度使用 K，并过滤非有限值/无效深度。

策略输入不包含物体真实坐标、物体索引或全局俯视图；评价器内部读取物体位置只用于计算分数。建议实现“观察 → 对准 → 停稳抓取 → 收臂运输 → 投放 → 继续搜索”的状态机，并用多个 seed 测试。

## 基础检查与边界

教师已分别检查原生 Isaac 中的站立、直行、左右转向、停车、雷达和两组 RGB-D，并在受控布局中实际夹取、抬起物体。区域计分器也单独验证了入区加 1 分及同一物体不重复计分。各项检查不等于已提供完整自主策略，也不表示任意伸臂轨迹和篮筐载重均稳定。

本包保留基础自动测试，覆盖最新 v3 模型、权重与资源哈希、19 维动作契约、区域边界、计分与重置、行走观测适配、闭爪限速和重置步态时保留夹持目标。测试中的合成物体坐标只用于离线验证计分函数，不是策略输入或可直接使用的比赛解法：

```bash
python -m pip install -e '.[test]'
python -m pytest -q
```

本包不包含 Isaac Sim、Isaac Lab、显卡驱动、底层 policy 的从零训练流程或完整抓取答案。已包含用于仿真的篮子资产。原始参考 URDF 另行保留；导入器可能报告 `imu_link` 质量/惯性近似警告，部分镜像有 Warp 提示，不能仅依据画面加载成功就判断其他传感器或任务已经完成。

学生需要自行验证策略在导航、伸臂、夹持、带载移动和投放过程中的稳定性，并按上述正式规则提交。源码与资源来源见 `THIRD_PARTY_NOTICES.md`。

## 发布包整理与重建

发布包仅包含正式 Task B 源码、学生入口、安装配置、现有行走权重、最新 v3 机器人与篮子资产、三个计分物体及其贴图、基础检查和测试。独立篮子试验的查看器、轨迹、报告、桌面启动器、旧版篮子、本机虚拟环境和缓存不进入发布包。

解压后可先运行 `python scripts/verify_release.py` 校验完整文件清单及 SHA-256（不需要安装 Isaac 或 PyTorch），再按上面的安装步骤运行 `python scripts/check_package.py` 检查真实行走权重推理。请保留解压后的工程目录布局。

维护者在工程根目录运行 `python scripts/build_release.py` 可重建发布目录和 ZIP，默认输出到上一级 `dist/TaskB_v3_2026-10-23/` 及同名 ZIP。重复运行应通过 `--output-dir` 指定新目录，脚本不会覆盖已有发布包。
