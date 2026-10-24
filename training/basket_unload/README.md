# MuJoCo GPU 篮筐尾窗卸料训练

这是与正式 Isaac Task B 隔离的 MuJoCo 训练/诊断目录。环境使用浮动基座 B2+Z1、v3 篮筐碰撞体和三个自由货物；正常 `step` 只发 19 个关节目标。货物位姿、碰撞和尾窗事件只用于 evaluator/reward，绝不进入 actor。

完整阶段、奖励设计及验收标准见 [训练计划](PLAN.md)。

训练按以下顺序推进：

1. 先跑 reset、资产切换、CUDA 物理、腕部 RGB-D 和禁碰查询短测。
2. 用空篮 balance 课程学习浮动基座站立，腿部只做受限小步/抬腿调整。
3. 用单物体课程学习正确尾窗排出。
4. 扩展到三物体，覆盖同类和混合类别、随机收集姿态。
5. 使用独立 seed、随机初始姿态/摩擦/视觉扰动评估泛化。
6. 最后再把 MuJoCo 结果作为动作设计参考，单独在 Isaac/PhysX 正式环境验证；两者结果不互相冒充。

尾窗 tracker 只接受完整穿过篮筐尾部开窗的顶点/边截面；侧面、顶部或错误方向离开会锁存失败。奖励含全清空稳定成功、逐件合法出口、错误出口/回入、危险接触、跌倒、自交、动力学无效、超时和残留惩罚，并对动作变化、实际关节速度、限位、能耗和足端接触滑移施加小的连续成本。

可选 `--speed-margin-penalty-weight` 和 `--clearance-penalty-weight` 默认均为零。前者对当前控制周期内最大实际关节速度/硬件限速超过 `0.6` 的部分施加平方成本；后者对该周期查询到的禁碰间隙低于 `0.01 m` 的部分施加平方成本，两者按 `control_dt` 积分。它们为硬失败前提供训练信号，不改变硬限速、碰撞、成功或 actor 观测。

## 运行

先确认隔离依赖和桌面窗口：

```bash
bash scripts/run_basket_unload.sh --smoke --stage balance --num-envs 4 \
  --episode-seconds 1 --width 64 --height 48 --run-name smoke
```

课程训练示例（新课程用 `--init-checkpoint` 只继承权重；`--resume` 用于原课程断点续训，会继承采样计数和优化器）：

并行运行多个训练时，为各进程设置不同的 `--preview-dir runs/basket_unload/<run-name>`，避免桌面状态及图像相互覆盖；将桌面 `live` 链接指向当前要观察的目录。未指定时保留根训练目录预览。

```bash
bash scripts/run_basket_unload.sh --stage balance --num-envs 16 \
  --steps 1000000 --run-name balance_v1
bash scripts/run_basket_unload.sh --stage unload --num-objects 1 --num-envs 16 \
  --steps 2000000 --episode-seconds 400 \
  --init-checkpoint runs/basket_unload/balance_v1/checkpoint_final.pt \
  --run-name unload_single_v1
```

单件课程应使用 `single_reset_bank.py` 独立生成的初态：先停放其他两件，仅投放活动物体，再按原几何和站立条件落稳。示例为 `python -m training.basket_unload.single_reset_bank --output runs/basket_unload/reset_bank_single_physical_train.npz --seed-offset 4000000 --seeds-per-class 16`；验证集使用独立 seed 区间。该 bank 含三类各 16 行，`num_objects=1` 时仅 slot0 在篮内。旧 `reset_bank_single_sugar_train/eval.npz` 从三件落稳后移走两件，部分剩余物体会失去支撑；旧结果保留，不能与独立单件初态的结果合并。

卸料默认 episode 为 `180 s`，平衡课程为 `5 s`。当前带反向收臂与机身恢复的参考动作约 `344 s`，采集及其后续训练/评估应显式设置 `--episode-seconds 400`。早期 `3 s` 卸料短训只验证了管线。默认腿/臂目标速率为 `.2/.35 rad/s`，可通过 `--leg-target-rate`、`--arm-target-rate` 修改独立实验；实际硬件关节限速与力矩限制仍保留。默认 PPO `gamma=.9998`、`gae_lambda=.99`，避免在 50 Hz 长时卸料中将末端奖励过早折减。课程是否完成仍以独立合法尾窗排出与恢复稳定的实际评估为准。

`collect_expert.py` 仅接收机器人反馈并发送关节目标，使用与 PPO 相同的 CUDA 环境验收。只有完整通过的 episode 才可进入 `behavior_clone.py`，有效环境和首次终止前的连续 transition 前缀均有 mask。可选 `--tool-joint-lift .04 --tool-lift-taper-start .60 --gripper-target -.04` 调整扫推间隙与夹爪余量；这些仍是待动力学验证的轨迹参数，不改变碰撞体、执行器或验收判据。

`collect_expert.py --path <candidate.json>` 可对独立候选关节路径进行同一环境动力学复验；几何筛查通过不等于完整卸料成功。`evaluate.py --gripper-residual-scale 0` 仅移除学习器的夹爪目标偏移，用于与原尺度同 seed 对照，夹爪仍保留参考曲线、自由受力及原执行器限制。

`collect_expert.py --approach-before-tilt` 是独立参考候选：先将腕部接近篮内，再倾身，之后按原时间和路径扫推、反扫、收臂及恢复。它只重排关节目标，不改物体、root 或硬件限制；默认关闭，不改变原参考。该顺序旨在测试倾身前获得图像的机会，须验证实际货物可见性和完整动力学，不能由运动学间隙宣称视觉闭环或清空成功。

视觉行为克隆可用 `--ignore-previous-action` 在训练期间将本体编码首层对应上一动作的权重列固定为零，防止网络只复制慢轨迹的上一帧动作。默认网络接收 67 维本体观测；后续 PPO 可正常优化全部权重。训练 MSE 不证明自主执行成功，必须另跑 `evaluate.py`。评估继承 checkpoint 同目录的 `environment.json`，预览与进度写在评估输出文件目录，桌面 `live` 链接应指向该目录。

`evaluate.py` 按固定初态分批评估：每个 world 每批只计首次终止，快失败后的自动 reset 回合不会挤占慢回合；报告保留实际 seed、bank 行及失败标记。视觉对照可用 `--image-mode zero`，仅把传入策略的五通道图像置零，本体反馈、循环状态、物理和真实桌面画面保持原逻辑；默认 `normal` 使用实际腕部图像。应使用同一 checkpoint、相同固定初态比较两种模式，不能因网络接收图像就宣称视觉闭环有效。

后续评估的桌面预览跟随本批仍未结束的首个 world，并记录 `world_id` 与分配的 `episode_id`；每个已接受回合会原子写入 `partial_episodes.json`，该文件明确标为进行中，不能当最终成功率。逐回合终止诊断另保存速度峰值关节及实际越限、非有限值和 overflow 标记；它们仅用于定位故障，不进入策略或改变判据。

独立环境支持 `--action-mode delta`（默认，兼容已有模型）和 `--action-mode position`。后者把网络的 `[-1,1]` 输出解码为关节限位内的绝对位置目标，再施加相同的腿/臂目标速率及硬件执行器约束，避免预测误差持续积分。两种模式不能混用权重的动作语义。`relabel_position.py` 可从完整成功示范的控制初态、动作与机器人反馈确定性重建已执行命令，保留成功环境和 transition mask，并更新上一动作观测；重标不增加物理成功次数。可用 `--startup-repeats 5 --startup-steps 400` 加强行为克隆的起步监督；全轨迹 MSE 仍按不重复样本统计。

`train.py --observe-time` 和 `relabel_position.py --observe-time` 可在本体观测末尾追加自身回合计时 `episode_steps * control_dt / episode_seconds`，启用时为 68 维。该计时由机器人自身维护，不包含物体真值；评估继承保存的设置。BC 从 67 维权重初始化 68 维网络时，仅在本体首层末尾追加一列零权重，其他权重严格加载。时钟可帮助区分扫推与回收阶段，但不能证明视觉纠偏或随机布局泛化。

可选 `--action-mode reference_residual --reference-targets <commands.npz>` 使用固定关节命令先验加视觉 PPO 残差，避免仅学习保持当前姿态。先验必须用 `export_reference.py` 从完整成功的 position 示范导出；运行时只读取 `targets[T,19]` 和相同的 `control_dt`，按自身回合步数取命令，结束后保持最后一个目标。默认腿、臂、夹爪残差幅度为 `.04/.08/.04 rad`，仍经过原关节限位、目标速率、执行器力矩和夹爪限制。网络只输入腕部 RGB-D、本体反馈和可选自身时钟；不会读取货物真值、参考示范图像或 cargo 状态。零残差仅是固定关节控制基线，不能作为强化学习结果；训练后的残差必须另做独立评估和零残差/视觉消融对照。参考只发关节目标，不回放 root 或物体状态。该模式不能直接加载 position actor 的输出权重作为残差，需显式将残差 actor 初始化为零，再训练。`--learning-rate` 和 `--entropy-coef` 可设置 PPO 的学习率与探索正则，默认保持原值。

默认 `--reference-stride 1` 保留每20ms更新一次的原行为。独立控制频率候选可显式设置 `--control-dt .1 --reference-stride 5`：每100ms取原20ms参考的第 `episode_steps*5` 个命令，参考动作的物理时钟不提速，最后保持末目标。仍以2ms积分和检查实际速度/稳定/尾窗事件，禁碰查询周期保持10ms。目标只在100ms控制边界更新，因此动力学与50Hz并不等价，须单独做零残差和学习策略的完整评估。PPO折扣按实际控制周期设置（保持原时间折扣可用 `gamma=.9998**5`），GAE与循环训练窗口的物理时长也会变化；旧50Hz验证结果不能沿用。

默认训练 bank 是 `reset_bank_collected.npz`；`reset_bank_eval.npz` 只用于独立评估。评估不会把组合或尾窗真值喂给策略：

```bash
.venv-unload/bin/python -m training.basket_unload.evaluate \
  --checkpoint runs/basket_unload/unload_single_v1/checkpoint_final.pt \
  --episodes 160 --num-envs 16 \
  --reset-bank runs/basket_unload/reset_bank_eval.npz \
  --output runs/basket_unload/evaluation.json
```

`evaluation.json` 会记录每种组合的成功、逐件正确尾窗事件和错误出口；当前仓库提供的评估 bank 是独立的 16 seeds/组合，若要达到每组合 100 次，应先用 `assets_gpu.py --pose-mode collected --seeds-per-combination 100` 生成新的评估 bank。

评估可显式添加 `--record-video` 保存实际 overview 的 `preview.mp4` 与 `preview_frames.jsonl`。视频按墙钟预览采样编码为 `2 fps`，不是仿真实时倍率回放；逐帧索引保留仿真时间、墙钟时间及 episode 重置。外部画面仅用于人工查看，策略仍只使用腕部 RGB-D。

完成评估后可用 `python -m training.basket_unload.summarize <evaluation.json> [paired-evaluation.json ...] --output runs/basket_unload/summary.json` 离线生成各组合成功率、Wilson 95% 区间、失败标记、耗时和逐 seed 配对差异。该工具拒绝进行中/短测、重复 seed、缺失回合及配对初态不一致的输入；区间只表达独立抽样 seed 下的统计不确定性，不能替代泛化覆盖或物理验收。

## 已验证范围

`python -m pytest -q training/basket_unload` 覆盖奖励、尾窗事件、禁碰 CUDA query、GPU/CPU 距离对照和 PPO 数值路径。相机诊断已验证动态腕部位姿、BVH 刷新、光轴深度以及批量资产切换。GPU throughput、几何 query 或一次参考动作都不等同于完整卸料成功；报告必须注明 MuJoCo 引擎、bank、seed 和尚未验证的 Isaac/PhysX 范围。
