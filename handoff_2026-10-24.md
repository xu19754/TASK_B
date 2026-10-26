# Task B 项目交接

更新日期：2026-10-24（Asia/Shanghai）。运行进度采样于当日约 13:08；正在运行的任务会继续变化。

仓库：[yxh0527-ship-it/taskb](https://github.com/yxh0527-ship-it/taskb)，当前分支 `main`，本次交接前的基线提交为 `38d3a9f`。服务器工作目录为 `/root/gpufree-data/5810_20261023/TaskB`；代码按工程根目录定位资源，不应依赖此服务器绝对路径。

## 1. 接手时先知道的事

- 正式 Task B 基础工程、机器人资产、区域计分器、传感器和已有行走 policy 已提供。`demo/solution.py` 默认仅站立收臂，尚未实现自主感知、导航、抓取、运输和投放的完整闭环；默认零分符合预期。
- 独立感知场景已搭建，当前只包含机器人和三类物体各一件，没有识别、分割、标注、采集或感知模型训练功能。
- 最近的主要工作是独立 MuJoCo GPU 篮筐尾窗卸料训练。空篮平衡通过初步检查，单件糖盒课程仍在推进，芥末瓶、香蕉及三件课程尚未通过；未证明可靠的视觉闭环或正式 Isaac/PhysX 卸料。
- 当前有三路 PPO 训练及三路独立评估正在运行。接手先查看进程与输出，避免重复启动；本次交接只检查、记录和同步源码，没有停止或重启这些任务。
- GitHub 保存源码、文档和原有受控资产；训练 checkpoint、reset bank、录像和运行报告位于被忽略的 `runs/`，虚拟环境也不入库。**仅 clone 仓库不能恢复服务器上的训练状态和实验文件。**

先阅读 [AGENTS.md](AGENTS.md)、[正式任务说明](README.md) 和 [Git 工作流](docs/git-workflow.md)。正式契约以根目录 README 为准，独立实验计划不覆盖正式规则。

## 2. 三种场景和主要文件

| 范围 | 入口 / 关键文件 | 当前边界 |
| --- | --- | --- |
| 正式任务 | [run_taskb.sh](scripts/run_taskb.sh)、[play_taskb.py](scripts/play_taskb.py)、[学生策略](demo/solution.py) | 正式得分只由此环境产生；自主收集闭环待实现 |
| 正式基础设施 | `taskb_student/`、`models/locomotion/`、`tests/test_taskb.py` | 环境、计分、19 维动作、底层观测适配、执行器和传感器契约 |
| 独立感知场景 | [场景说明](training/perception/README.md)、`training/perception/scene.py`、`play.py`、`scripts/run_perception.sh` | 当前仅搭建和查看场景；不加载正式计分与终止 |
| 独立卸料训练 | [训练说明](training/basket_unload/README.md)、[阶段计划](training/basket_unload/PLAN.md)、`scripts/run_basket_unload.sh` | MuJoCo / MuJoCo Warp CUDA，全身浮动基座训练，不产生正式 Task B 得分 |
| 当前篮子资产 | [v3 说明](assets/b2z1/basket_v3/README.md)、`assets/b2z1/b2z1_0815_with_basket_v3.urdf` | 450 × 250 × 160 mm，103 块凸碰撞体，0.2 kg 为设定值 |
| 历史篮子诊断 | `reports/basket_feasibility/`、`reports/basket_v2/`、`reports/basket_v3/`、`scripts/basket_*.py` | 部分工具规定机身/腿部姿态，仅作对应版本的局部诊断 |
| 发布与完整性 | `scripts/check_package.py`、`build_release.py`、`verify_release.py` | 正式发布包使用白名单，不能直接将整个开发仓库打包发布 |

卸料训练目录内，`model.py` / `gpu.py` / `assets_gpu.py` 管理模型、CUDA 物理与资产；`env.py` / `vision.py` / `clearance.py` 管理环境、腕部图像与禁碰查询；`metrics.py` / `reward.py` 管理尾窗事件和奖励。`policy.py` / `ppo.py` / `train.py` 是学习入口，`collect_expert.py` / `behavior_clone.py` / `relabel_position.py` / `export_reference.py` 管理示范与关节命令先验；`evaluate.py` / `summarize.py` / `dashboard.py` 管理独立评估、汇总与桌面预览。

## 3. 必须保留的正式契约

- 指定机器人 Unitree B2 + Z1，任务 `Student-TaskB-B2Z1-v0`，单环境；出生点 `(-10, -10, 0.6)`，三类物体各 6 件，共 18 件。
- 每件物体每 episode 最多 1 分，总分 18。目标桶中心 `(-3, -10)`，水平半径 `1.0 m`，物体 root 高度 `[0.0, 0.5] m`，包含边界。全部物体同时入区才完成；没有靠近、闭爪、抬升或装篮奖励。
- 默认 episode 为 1200 秒仿真时间；`--max-walltime-s` 是 reset 后开始的独立真实时间上限。保留异常接触和跌落终止。
- 策略只能使用 `predicts(obs, current_score)` 的规定输入：`proprio`、`lidar`、可选 `image` 和分数。公开桶位置可用于规划，物体真值不能从场景、评价器、布局或日志回流策略。
- 返回 `{"action": nineteen_absolute_joint_targets, "giveup": False}`。19 个有限绝对角度，单位 rad，顺序为 FR、FL、RR、RL 的 hip/thigh/calf，再接 Z1 六关节和夹爪；不再缩放。
- 夹爪为旋转关节，范围 `[-1.57, 0]`，0 为闭合端；保留 80/8 的刚度/阻尼、5 N·m 力矩上限、0.25 rad/s 限速和闭爪目标超前限制。
- 使用现有 `LowLevelController` 适配 52 维观测和已有 policy；速度为 `(cmd_x, 0, cmd_yaw)`，横移为零，前后限幅 0.4 m/s、转向限幅 0.6 rad/s。新 episode 调用 `reset()`；仅清步态历史且需保留夹持时调用 `reset_locomotion()`。
- 全身移动、抓取与运输必须通过执行器和真实接触完成，不能写 root/货物状态或使用隐藏约束代替控制。初始化和隔离诊断设置初态必须与正常策略动作区分。
- 当前正式配置是 CPU PhysX + GPU 图像渲染、`use_fabric=False`、开启场景查询。LiDAR 要检查 `ready` 和 `hit_mask`；相机只有 `--enable_cameras` 时存在，深度为米制光轴深度。四元数顺序为 `wxyz`。

## 4. 环境与常用命令

正式环境基线：Isaac Sim 5.0、Isaac Lab v2.2.1、Python 3.11、PyTorch 2.7。本次检查使用服务器 `isaaclab` conda 环境，Python 3.11.15。以下命令均在工程根目录执行。

```bash
conda activate isaaclab
python -m pip install -e '.[test]'
python scripts/check_package.py
python -m pytest -q

# 正式空篮短测；有图形桌面时可以去掉 --headless
bash scripts/run_taskb.sh --headless --steps 500 --debug

# 正式相机与传感器快照
bash scripts/run_taskb.sh --headless --enable_cameras --steps 500 \
  --sensor-output runs/sensors --diagnostics runs/standing.jsonl --debug

# 独立三物体场景
bash scripts/run_perception.sh --real-time
bash scripts/run_perception.sh --headless --steps 500 \
  --report runs/perception/three_objects_handoff_check.json
```

`TASKB_PYTHON` 可指定正式/感知入口的 Python。图形窗口需要有效 `DISPLAY`；卸料启动器默认使用 `:20`，换机器需按实际桌面设置。

独立卸料依赖保存在 [requirements.txt](training/basket_unload/requirements.txt)：MuJoCo 3.17.0、MuJoCo Warp 3.17.0、warp-lang 1.18.0 等；`.venv-unload` 通过 `--system-site-packages` 复用已有 CUDA PyTorch。

新机器需要安装依赖并恢复或生成 reset bank，不能直接假定 `runs/` 已存在：

```bash
python -m venv --system-site-packages .venv-unload
.venv-unload/bin/python -m pip install -r training/basket_unload/requirements.txt

# 三件物体随机收集姿态初态；训练/评估使用独立 seed
.venv-unload/bin/python -m training.basket_unload.assets_gpu \
  --output runs/basket_unload/reset_bank_collected.npz \
  --pose-mode collected --seeds-per-combination 8 --seed-offset 0

# 独立单件初态：先停放其他物体，再让活动物体物理落稳
.venv-unload/bin/python -m training.basket_unload.single_reset_bank \
  --output runs/basket_unload/reset_bank_single_physical_train.npz \
  --seed-offset 4000000 --seeds-per-class 16
.venv-unload/bin/python -m training.basket_unload.single_reset_bank \
  --output runs/basket_unload/reset_bank_single_physical_eval.npz \
  --seed-offset 5000000 --seeds-per-class 16

# 新目录做管线短测，不覆盖已有运行
bash scripts/run_basket_unload.sh --smoke --stage balance --num-envs 4 \
  --episode-seconds 1 --width 64 --height 48 --run-name handoff_smoke
```

上面的生成和仿真命令是接续说明，本次交接未重新执行。当前实验所用糖盒子集、参考 NPZ、checkpoint 及对应 `environment.json` 应从服务器恢复并核对来源；新生成初态不自动等同于旧实验的完整复现。

## 5. 独立卸料的已完成证据

本节来源均在服务器 `runs/basket_unload/` 下，**不随本次 GitHub 源码同步上传**。这些数字是读取已有报告所得，本次没有重新运行相应动力学评估。

独立环境要求货物完整穿过尾窗、收臂并恢复稳定保持 0.5 秒；错误出口、回入、危险碰撞、自交、越限或无效动力学不能被忽略。actor 使用腕部 RGB-D、本体反馈、上一动作和可选自身回合时钟，货物真值仅用于初始化、奖励和评估。

| 检查 / 模型 | 已完成结果 | 证据路径（相对 `runs/basket_unload/`）与限制 |
| --- | --- | --- |
| 空篮平衡 | 32/32，10 秒 episode | `balance_empty_stage1_v3/evaluation.json`；只证明该空篮站立检查 |
| 旧参考零残差，修正单件初态 | 1/16 完整成功 | `reference_residual_baseline_stage2/evaluation_single_physical_fixed/evaluation.json` |
| stator150 固定参考，50 Hz | 9/16；1 错误出口、6 超时 | `reference_stator150_physical_baseline_stage2/evaluation_fixed/evaluation.json`；固定关节先验，不是 RL 成果 |
| 同参考，10 Hz | 9/16；1 错误出口、1 无效动力学、5 超时 | `reference_stator150_physical_10hz_baseline_stage2/evaluation_fixed/evaluation.json`；与 50 Hz 相同成功 seed，不能声称安全性改善 |
| v2 PPO，checkpoint `1114112` | 正常图像 1/16；屏蔽图像 0/16 | `ppo_single_physical_soft_stage2_v2/eval_000001114112_normal/evaluation.json` 及同级 `eval_000001114112_zero/evaluation.json`；尚未优于对应零残差基线 |
| 先观察后倾身参考 | 训练 seed 9/16 成功，3 失败，4 在参考时域内未完成 | `expert_stator150_look_first_physical_v1/report.json`；独立 held-out 评估进行中，不能当成视觉策略成功 |
| 扩展单件训练初态 | 三类各 64 行，192 行均通过 0.5 秒 CPU 保持诊断 | `single_train64_preparation_20261024/preparation_report.json`；只验证初态，当前运行任务仍用旧 16 行糖盒训练子集 |
| 最终泛化 bank | 10 种三件类别计数组合各 100 行，共 1000 行，已生成并检查覆盖 | `reset_bank_generalization_100.npz`；保留作冻结模型最终评估，尚无对应策略验收结果 |

其他已知结论：早期 BC 的低 MSE 没有带来自主卸料成功；旧版单件 bank 从三件落稳后移走两件，可能让保留物体失去支撑，不能与修正后的独立单件 bank 混合统计。旧 v1 最终正常图像/屏蔽图像评估均为 0/16，不能仅因网络接收图像就声称视觉有效。

独立感知场景的当前三物体历史报告是 `runs/perception/three_objects_check.json`：seed 42、500 步、10 秒仿真、三件物体，末态机身倾角约 2.20°。同目录 `scene_check.json` 等部分旧报告为 18 件版本，不能用于证明当前三件布局。当前代码与 README 的三件配置一致。

历史 v3 原轨迹曾在特定 MuJoCo 混合三件布局中排出 3/3，见 [原轨迹复核](reports/basket_v3/原轨迹复核报告.md)。该工具规定机身/腿部准备姿态，且未放目标桶；不证明全身平衡、带载行走、入桶得分或正式 Isaac 执行成功。

## 6. 当前运行快照与接续位置

2026-10-24 约 13:08 读取 `metrics.jsonl` 时：

| run（均在 `runs/basket_unload/`） | 已记录 transitions / 目标 | 主要设置 |
| --- | --- | --- |
| `ppo_single_physical_soft_stage2_v2` | 2,392,064 / 4,000,000 | 64 环境，50 Hz，旧参考，修正单件初态，继承 v1 权重 |
| `ppo_single_stator_physical_soft_stage2_v3` | 1,703,936 / 4,000,000 | 64 环境，50 Hz，stator150 参考，零残差初始化 |
| `ppo_single_stator_physical_10hz_stage2_v4` | 118,784 / 800,000 | 64 环境，10 Hz，`reference_stride=5`，零残差初始化 |

三者均为 CUDA 训练、单件糖盒、400 秒 episode；训练中的合法出口事件与完整成功计数不能当成 held-out 成功率。v4 使用 `.1 s` 控制周期、`.002 s` 物理步、`gamma=.999000399920008`、`gae_lambda=.9509900499`、64 步 rollout；参考取样仍按原物理时钟，不能把它看成与 50 Hz 仅改一项参数的比较。

同一快照下，以下三个目录只有进行中的状态，最终 `evaluation.json` 尚未生成：

- `ppo_single_stator_physical_soft_stage2_v3/eval_000001081344_normal/`
- `ppo_single_stator_physical_soft_stage2_v3/eval_000001081344_zero/`
- `reference_stator150_look_first_physical_baseline_stage2/evaluation_fixed/`

查看时优先检查：

```bash
pgrep -af 'training.basket_unload.(train|evaluate|dashboard)'
cat runs/basket_unload/stage_progress.json
tail -n 1 runs/basket_unload/ppo_single_stator_physical_soft_stage2_v3/metrics.jsonl
readlink -f runs/basket_unload/live
```

每个训练目录保留 `environment.json`、`ppo_config.json`、checkpoint 和 `metrics.jsonl`；评估目录有 `status.json`、`partial_episodes.json`，完成后才有 `evaluation.json`。阶段总表可能落后于逐 run 输出，应交叉核对实际文件和进程。当前 `live` 指向 v3 正常图像评估目录。

注意 `stage_progress.json` 中的 `session_id` 是历史工具会话标识，不是 Linux PID。已有进程使用启动时加载的代码；当前源码中 episode 截止时刻成功的终止/截断修复不会自动更新旧进程，不应将旧进程结果当成该修复的动力学验收。

评估完成后可在相同 seed/bank 分配下做正常图像、屏蔽图像和零残差对照。例如：

```bash
.venv-unload/bin/python -m training.basket_unload.summarize \
  runs/basket_unload/ppo_single_stator_physical_soft_stage2_v3/eval_000001081344_normal/evaluation.json \
  runs/basket_unload/ppo_single_stator_physical_soft_stage2_v3/eval_000001081344_zero/evaluation.json \
  --output runs/basket_unload/ppo_single_stator_physical_soft_stage2_v3/visual_summary.json
```

该命令要等两个最终报告都存在后执行。汇总工具拒绝进行中、缺失回合、重复 seed 或配对初态不一致的输入；失败标记可能重叠，不能把失败标记计数简单相加当作回合数。录像为墙钟采样的 2 fps 预览，仿真时间以逐帧索引为准。

继续原课程用 `--resume` 继承优化器与计数；新课程用 `--init-checkpoint` 仅继承权重。恢复时核对原 run 的配置、reset bank、参考 NPZ、动作模式及 `control_dt`，并沿用其余 CLI 参数；不能混用 `delta`、`position`、`reference_residual` 的权重语义。新增运行使用新的 run 名和独立 `--preview-dir`。

## 7. 下一步工作与验收顺序

1. 完成当前 v3 正常/屏蔽图像评估和先观察参考的固定 16 初态评估，保存逐 seed 结果、失败类型与配对汇总。继续记录 v2/v3/v4 到各自声明的训练目标，不能用中间训练奖励替代最终结果。
2. 对冻结 checkpoint 比较对应零残差先验、正常 RGB-D 和屏蔽图像。当前单件糖盒课程未通过，不能因 9/16 固定参考或训练累计成功就进入“已完成”状态。
3. 若需要增加训练覆盖，使用已准备的 `*_physical_train64.npz` 新建运行；保留原 16 行训练实验，验证 bank/seed 分离和实际动力学。再逐类推进芥末瓶、香蕉、三件同类和混合。
4. 三件课程最终目标为全部 10 种类别组合各至少 100 个独立初态、各组合成功率至少 90%，并报告区间、错误出口、残留、碰撞、速度、稳定性和耗时。这是独立卸料计划的目标，不是正式 Task B 新增计分规则。
5. MuJoCo 结果作为动作设计参考；后续在正式 Isaac/PhysX 中从空篮、伸臂、单件抓取/入篮、带载移动到投放逐步验证，并用多个 seed。正式完整闭环仍需在 `demo/` 中实现合法感知、搜索导航、抓取、运输和投放。

## 8. 本次交接验证与同步范围

本次实际执行：

| 检查 | 结果 |
| --- | --- |
| `python scripts/check_package.py` | 通过：v3 URDF 与 policy 哈希、篮子碰撞/质量、关节及路径、真实 TorchScript 推理 |
| `CUDA_VISIBLE_DEVICES='' OMP_NUM_THREADS=1 python -m pytest -q` | 23 passed |
| `CUDA_VISIBLE_DEVICES='' OMP_NUM_THREADS=1 .venv-unload/bin/python -m pytest -q training/basket_unload` | 93 passed，6 skipped；本次未执行 CUDA 检查 |

以上不包含新一轮 Isaac/PhysX 动力学或 MuJoCo 全回合验收；第 5 节是已有本地报告的核对结果。此前不同测试数量对应不同源码快照，以本节本次实际输出为准。

本次源码同步范围为新建交接文档，以及工作区已有的 `AGENTS.md`、README 独立场景说明、两个独立启动器和 `training/` 源码/测试。正式策略、计分、终止、机器人资产和底层权重不在本次修改范围。

若迁移训练，应另外保存需要的 `runs/basket_unload/` 运行目录、引用的 reset bank 和参考 NPZ、checkpoint 相邻配置、报告与哈希清单；Git 推送不等于训练产物备份。正式发布继续使用 `build_release.py` 白名单，输出到工程之外的新目录，再用 `verify_release.py` 验证，不顺手扩大正式发布内容。
