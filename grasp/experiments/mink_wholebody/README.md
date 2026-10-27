> Historical upstream documentation. Path-audit code referenced below was removed from this integration; see `grasp/README.md` for current behavior.

# 已知目标位姿的全身协调抓取诊断

用户在固定机身的臂抓取失败后，明确要求协调全身、通过下蹲改变可达范围。本目录实现该后续方案；原 experiments/mink_grasp 的失败结论和记录均保留。

从 /root/Desktop/TaskB 执行已经实测的完整入口：

~~bash
bash scripts/run_mink_wholebody.sh --mode grasp --trials 5 --headless --enable_cameras --record-video --output runs/mink_wholebody/reproduce
~~

支持 --mode ik-check / env-check / posture / reach / grasp，以及 --config。已存在的输出目录自动改为带时间戳的新目录。详细结果见 runs/mink_wholebody/report.md 和 final_5/。

机器人使用工程原 B2Z1_CFG、URDF、执行器和行走权重。SETTLE 使用老师零速度策略，并用 100 次初始实测观测填充历史。之后两秒逐渐切换到全身站姿控制，腿部不再仅由原行走策略维持高度。Mink 同时求解浮动基座的六个自由度、十二个腿关节和六个臂关节，包含四足锚定、机身目标、质心水平位置和姿态正则。机身参考从约 0.539 m 降到 0.420 m；抓住后机身上升到 0.480 m，同时 TCP 上升 0.150 m。每帧从实际机器人状态开始求解，预抓取和接近均缓慢插值。

真实执行仍只发送 19 维绝对关节位置。站立支撑和关节重力补偿通过现有执行器刚度转为位置偏置，未施加外部机身力，也未写入运动中的根位姿。夹爪继续通过 LowLevelController 的限速与预紧处理，保留原力矩/速度上限。全程 Isaac CPU PhysX，GPU 渲染；MuJoCo 只用于运动学、雅可比和质心计算。

固定目标是原尺寸、0.5 kg 的芥末瓶，放于前方 x=0.55 m，竖直稳定放置。控制器只使用已知目标配置和机器人自身状态；物体真值与真实 PhysX 接触事件仅进入独立评价记录。摄像头仅观察。没有物体附着、隐藏固定约束、运动中重置物体或修改正式 Task B 计分。

验收阈值保持为上升至少 0.10 m、最低点离地至少 0.02 m、连续至少 3 s，两指实际接触且物体在夹持区域，机身倾斜不超过 15 度、水平漂移不超过 0.10 m，无机身/髋/大腿异常触地。双指接触采用 PhysX 原生接触事件精确匹配 Robot/gripperStator、Robot/gripperMover 与 Object/_06_mustard_bottle 两个刚体。旧过滤传感器矩阵仍为零，作为已知问题单独记录；原生接触冲量及独立净力均保存，不把传感器零值改写成接触。

runtime.sh 复用 /isaac-sim/python.sh 和 /workspace/isaaclab。锁定 Mink 1.0.0、MuJoCo 3.3.7、qpsolvers 4.8.2、DAQP 0.7.2；完整镜像快照 requirements.lock.txt 不能替代 Isaac 二进制镜像。环境报告和安装前后差异在 runs/mink_envcheck/20261023_081402。单次仿真上限 100 s、运行上限 300 s、启动上限 180 s；总外部上限为 180+300*trials+15 秒。原生关闭挂起时，结果全部落盘后等待 5 秒，仅回收本次进程组，明确记录于 process_status.json。

path_audit.py 对实际执行轨迹每 0.1 s、状态转换和最终样本做 URDF 碰撞包围盒检查；它是保守离线审计，不声称通用避障或任意目标规划。完整录像、控制与评价 JSONL、逐次 CSV、配置/源码哈希均在 run 目录。

参考与许可证见 THIRD_PARTY.md 和 reference/。本实验验证固定布局单物体抓取，不代表完整 18 物体收集、导航、视觉识别、正式 LiDAR 或实机部署通过。
