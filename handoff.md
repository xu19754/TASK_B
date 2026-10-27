# TaskB 项目交接

交接日期：2026-10-26，Asia/Shanghai。目标仓库：`git@github.com:yxh0527-ship-it/taskb.git`。

## 1. 项目要完成什么

使用 Unitree B2 + Z1，在正式 Task B 场景中完成感知、搜索导航、抓取、运输和向目标桶投放。正式环境 ID 为 `Student-TaskB-B2Z1-v0`，共 18 件物料：糖盒、芥末瓶、香蕉各 6 件。每件进入规定目标区域最多计 1 分，总分 18 分。

目标桶中心 `(-3, -10)`，水平判分半径 1 m，物体 root 高度须在 `[0, 0.5] m`；装入随车篮子或仅推出篮子不构成正式得分。默认 episode 为 1200 秒仿真时间，真实时间上限单独配置。

当前工作重点是：复用同一正式场景与相机采集示范，先训练视觉模仿策略，再按实测需要增加残差强化学习。正式统一场景中的 DP3 + 残差 PPO 是后续方案，尚未实现或启动训练。远端原有的独立 MuJoCo PPO／行为克隆实验已保留，不能与这一正式流程混为一谈。

## 2. 当前实现与验证边界

| 项目 | 当前状态 | 证据或入口 |
| --- | --- | --- |
| B2 + Z1 与 v3 尾篮 | 已接入默认 URDF；保留原始参考模型 | `assets/b2z1/SOURCE.json`、`assets/b2z1/basket_v3/` |
| 原有行走 policy 与底层适配器 | 已有权重，真实 TorchScript 推理通过 | `models/locomotion/`、`taskb_student/low_level.py` |
| 正式任务、传感器、计分和终止 | 代码已有；本次未运行正式 Isaac 入口 | `taskb_student/`、`scripts/play_taskb.py` |
| 场景统一 | 采集、相机校验复用正式 `TaskBEnvCfg`，不另建采集场景 | `docs/示范采集准备.md` |
| 连续录制与相机输出检查 | 已实现；合成 IO 与入口接线测试通过 | `taskb_student/recording.py`、`scripts/inspect_recording.py` |
| 正式 RGB-D 录制与相机几何验收 | 尚待实际 Isaac 环境运行和人工检查 | 相机输出检查通过不等于外参、视场、深度精度已标定 |
| 默认学生策略 | 仅站立收臂，零分符合预期 | `demo/solution.py` |
| 示范操作与成功数据 | 尚未提供任务示范控制器；现有记录不构成正式训练数据 | 下一项工作 |
| 正式场景 DP3 + 残差 PPO | 尚未接入统一采集与正式策略，无已验证任务权重 | 不可声称已经具备正式任务泛化能力 |
| 独立 MuJoCo PPO／行为克隆 | 保留远端原有源码、测试、计划与历史交接；本次未训练或重做 GPU 评估 | `training/basket_unload/`、`handoff_2026-10-24.md` |
| 独立感知场景 | 保留远端原有三物体场景，当前只用于搭建与查看 | `training/perception/`、`scripts/run_perception.sh` |
| 独立原路径／随机物料实验 | 已做 MuJoCo 诊断 | 与正式任务验证分开解释，见第 5 节 |

本次从 GitHub 取得的基线为 `b1fb462`，其中已有独立卸料 PPO 训练和 2026-10-24 的服务器实验快照；旧交接全文保存在 [handoff_2026-10-24.md](handoff_2026-10-24.md)。旧文中的“正在运行”、checkpoint 和评估数字均为当时采样，本次没有登录 GPU 服务器刷新。训练运行目录、reset bank 和 checkpoint 仍在 `runs/`，不随 Git 同步。

## 3. 关键接口与不能随意改动的内容

### 动作与底层控制

- `AlgSolution.predicts(obs, current_score)` 返回 `{"action": targets_19, "giveup": False}`。
- 19 维均为绝对关节位置（rad）：12 腿关节、`joint1..joint6`、`jointGripper`。正式控制周期 0.02 s（50 Hz）。不要再次缩放或拼成 Piper 的 20 维接口。
- 行走命令为 `(cmd_x, 0, cmd_yaw)`，不支持横移。由既有底层控制器生成腿部关节目标；现有 policy 接收适配后的 52 维观测并输出 12 维腿动作。
- 夹爪为单旋转关节，范围 `[-1.57, 0] rad`，0 为闭合端；驱动参数为刚度 80、阻尼 8、力矩上限 5 N·m、限速 0.25 rad/s。限目标变化率不能单独证明实际速度不会被接触外力推高。
- 保留限位、力矩约束、实际状态检查及已有行走权重契约。机器人自碰撞当前关闭，不能因此把穿过机身／腿的轨迹当作可执行动作。

### 模型与正式规则

- 默认模型：`assets/b2z1/b2z1_0815_with_basket_v3.urdf`。原始无篮参考模型保留为 `b2z1_0815.urdf`。
- v3 尾篮来自 `零件2.SLDPRT`，质量为 0.2 kg 设定值（不是实测），103 块凸碰撞网格；固定到 base，安装平移 `(-0.30, 0, 0.13) m`。
- 不通过改变计分区域、终止规则、碰撞、物体状态、隐藏约束或瞬移获取成功。不以提高速度／力矩阈值掩盖失败。
- 学生控制器只使用规定的 proprio、LiDAR、RGB-D 与 `current_score`。不得使用物体真实位姿、物体 ID、教师接触真值或全局俯视图作为策略输入。

### 图像与录制

- 前视相机在 base，腕部相机在 link06；两组 RGB-D 为 640×480，目标更新率 10 Hz。共享配置在 `taskb_student/sensor_contract.py`。
- 深度为米制光轴深度 `distance_to_image_plane`，不能按欧氏射线距离或深度预览 PNG 处理。
- 每行数据是动作前的规定观测 → 正式入口限位后提交的动作 → 执行后的反馈。episode 终止后的自动重置观测不作为终止画面保存。
- 相机 10 Hz、控制 50 Hz，重复图像正常；规定观测不提供曝光时间／帧号。不得伪造同步精度。
- 每个 episode 使用新目录，manifest 保存源文件哈希、版本、seed、相机配置、结束原因与块哈希。`source_kind=demonstration` 只表示采集意图，成功标签仍需人工审查。

## 4. 接手后按什么顺序做

1. **正式运行验收。** 在可运行 Isaac 的机器上，完成相机检查与站立录制，读回并查看两组 RGB／深度。动作过程中还需检查工作区是否可见、相机是否随动及是否受遮挡。
2. **示范控制器。** 在 `demo/` 新增实现同一 `AlgSolution` 接口的键盘／手柄／操作面板或可审查专家控制器。先完成一个短技能，优先当前研究的卸料过程；腿部继续复用既有底层控制器。演示应连续执行，不在动作中改写物理状态。
3. **首批 5 条合格完整示范。** 用来打通数据、训练和闭环部署，不代表样本量已经足够。逐步覆盖单件、三件同类和混合摆放。成功与失败分别审查，超限、卡死和提前掉料不得混成成功示范。
4. **DP3 数据适配与离线训练。** 从规定深度和 K 反投影点云；融合多相机前，用安装配置与允许的机器人状态/FK 转到同一坐标系，检查重影、时间错位和工作区裁剪。准备 point cloud、proprio、动作序列和 episode 边界；保留 19 维原始动作，机械臂学习可先使用末 7 维，再经既有控制器组合成正式 19 维输出。按完整 episode／布局划分训练和测试，不随机拆相邻帧。
5. **DP3 闭环评估。** 在未训练的布局上测任务完成、残留、提前掉料、速度保护和耗时；训练 loss 降低不能代替动作成功。
6. **按失败类型加入残差 PPO。** 先冻结可用的 DP3，再让 PPO 在线学习有界小幅纠偏；策略仅用允许观测、基础动作与自身状态。PPO 必须持续与仿真交互，不能只对示范文件离线训练就完成。保留原控制／约束，独立训练入口不得悄悄改写正式评测。
7. **完整 Task B 集成。** 验证收臂到准备位、装篮保持、负载行走、整机平衡、入桶和正式得分，再谈完整任务泛化。

Mac 当前用于代码、操作端和独立 MuJoCo 诊断；没有安装 Isaac Sim / Isaac Lab。本机可以远程操作服务器上的同一仿真，示范由运行该仿真的机器录制。DP3 离线训练和 PPO 在线交互训练放在兼容 GPU 环境进行；尚未检查新的 GPU 服务器是否可用。

参考实现：[DP3](https://github.com/YanjieZe/3D-Diffusion-Policy)、[冻结基础 diffusion policy 后用残差 PPO 纠偏的 ResiP](https://residual-assembly.github.io/)。这些不是现成的 TaskB 训练入口，需要数据、动作、观测及环境适配。

## 5. 已完成的独立试验与已知问题

### 随机三件物料原路径测试

报告：`reports/basket_v3/random_three_2026-10-26/测试报告.md`；复现实验：`scripts/basket_random_trials.py`。

12 组包含混合三件、三个糖盒、三个芥末瓶、三根香蕉，每类 3 个 seed。严格按该独立试验定义达标 2/12，原动作完成 5/12，7/12 因实际夹爪速度超限停止。末态完整越过尾平面 17/36 件，其中 3 件提前滑出，不能算路径卸料成功。每类仅 3 次，不是可靠的泛化成功率估计。

本实验固定机身姿态、使用原低速路径，没有目标桶。完整出篮不等于正式得分，不能替代整机反作用、负载行走、收臂入位或 Isaac/PhysX 验证。逐组布局、失败记录、轨迹与回放均保留；`runner_used.py` 是当时运行脚本快照。

### 夹爪速度超限原因

报告：`reports/basket_v3/speed_audit_2026-10-26/速度超限原因分析.md`；复现：`scripts/basket_speed_audit.py`。

7 个失败案例均为实际 `jointGripper` 速度超限。5 例在接触脱开后，固定 −0.08 rad 目标下的伺服快速消除受载位置误差；2 例持续接触且电机力矩饱和，其中有夹爪／物料／篮底卡料。夹爪偏转使铲边高度偏离原规划假设，个例降低约 18.65 mm。

7 例临界时间与速度已精确复现；1 ms 临界窗口细分到 1/64 后，6 例仍超限，1 例对步长明显敏感。未从初态细步长重跑整条路径，不能宣称完整收敛或实机峰值已被证明。本次没有修改控制器解决该问题。下一步控制应使用实际夹爪角／速度与真实铲边几何，处理退让、抬升和卡料，而不是放宽阈值。

### 历史服务器验证

2026-10-23 的独立记录显示：v3 导入、19 关节、103 篮子碰撞体、空篮站立和小球基础碰撞检查通过。当时实际环境为 Isaac Sim 5.1.0 / Isaac Lab 2.3.2.post1，与项目参考的 5.0 / 2.2.1 不同；原正式入口存在 PhysX 雷达 Xform 操作兼容错误，独立验证使用了仅在内存中的兼容包装。

该历史记录不能证明新录制入口、0.5 kg 任务物料、整机卸料、负载行走或入桶计分已通过。本次交接不连接或修改历史服务器，不把教师辅助相机／探针接入正式采集。

## 6. 运行与复现

### 本机离线检查

在工程根目录执行，使用隔离临时依赖环境，不修改现有 `.venv-basket`：

```bash
uv run --no-project --python .venv-basket/bin/python \
  --with numpy --with pytest --with torch --with trimesh --with gymnasium --with Pillow \
  python -m pytest -q

uv run --no-project --python .venv-basket/bin/python \
  --with numpy --with torch --with trimesh --with gymnasium --with Pillow \
  python scripts/check_package.py
```

在新机器没有 `.venv-basket` 时，使用已安装的 Python 3.11／3.12 或先建立隔离环境；正式 Isaac 运行必须采用对应版本环境。

### 正式相机与录制实测

在项目参考 Isaac 环境中，进入工程根目录，每次选择新的输出目录：

```bash
python -m pip install -e .
python scripts/check_package.py

bash scripts/run_taskb.sh --headless --enable_cameras --seed 42 --steps 250 \
  --record-dir runs/recording_smoke_seed42_001
python scripts/inspect_recording.py runs/recording_smoke_seed42_001
```

默认控制器只录站立检查。自定义示范控制器接入和完整数据格式见 `docs/示范采集准备.md`；文档示例中的 `demo.my_demonstrator` 尚未创建，不能直接当成可运行模块。

正式入口当前为单环境 CPU PhysX，RGB-D 使用 GPU 渲染。不能仅把启动参数改成 `--num_envs 4096 --device cuda` 就当成 PPO 训练器。训练入口、并行环境适配及其物理一致性需要另行实现和验收。

### 独立诊断

```bash
.venv-basket/bin/python scripts/basket_random_trials.py --view \
  reports/basket_v3/random_three_2026-10-26/mixed_three_seed_20261026
```

macOS 也可双击 `打开随机三件路径测试.command`。批量复现与速度审计的完整命令见各报告；避免无意覆盖已有实验结果，先阅读脚本输出目录行为。

## 7. 仓库、发布与交接检查

- 开发仓库包含正式工程、资产、原有权重、独立诊断代码及报告；正式发布包使用 `scripts/build_release.py` 的显式白名单，排除独立试验、桌面启动器、旧篮子版本和缓存。
- 既有发布包名仍为 `TaskB_v3_2026-10-23`。后续构建使用新输出目录，不覆盖原 `dist/`。`handoff.md` 与 `AGENTS.md` 作为说明纳入白名单；其中提及的开发报告、独立脚本及桌面工具仍不进入正式发布包。
- `.gitignore` 排除虚拟环境、缓存、运行数据和新增的历史嵌套 viewer ZIP；本地原文件与远端已经受控的历史 ZIP 均保留。研究报告中的大型原始轨迹属于可追溯证据，不属于成功训练集。
- 本地初始检查记录放在 `reports/handoff_2026-10-26/`；云端整合检查见 `reports/github_sync_2026-10-26/`。本地 commit 不代表云端已同步，最终以远端引用及 GitHub 页面为准。
- 整合后的离线检查为正式工程与独立卸料测试合计 123 项通过、6 项因本机无 CUDA 跳过；资产与真实行走权重推理通过。此次没有执行新的训练或正式 Isaac／GPU 动力学验收。
- 本地首个快照 `b3affac` 保留在 `local-handoff-2026-10-26` 分支；本次在 `main` 合并远端 `b1fb462` 历史。远端已有的独立训练、Git 工作流、受控资产和旧交接记录均保留，不强推替换历史。交接入口统一为小写 `handoff.md`，避免 macOS 与 Linux 的大小写差异。
- `origin` 仍关联用户提供的 `git@github.com:yxh0527-ship-it/taskb.git`；本机该仓库使用 HTTPS 传输和官方 GitHub CLI，认证保存在系统钥匙串，不进入项目文件。日常操作见 `docs/git-workflow.md`。

历史后处理 `reports/basket_v3/random_three_2026-10-26/make_review.py` 仍读取开发机的聊天可视化绝对路径。仓库保留了可直接查看的 `review_preview.html`；跨机器重建该聊天可视化时需先配置对应输入路径。正式入口与原始试验数据读取不依赖该路径。
