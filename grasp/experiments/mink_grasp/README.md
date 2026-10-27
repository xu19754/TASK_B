> Historical upstream documentation. Path-audit code referenced below was removed from this integration; see `grasp/README.md` for current behavior.

**已知目标位姿的抓取诊断**

本目录是独立单物体机械诊断。实际测试结论：环境、FK/IK 和预抓取到位通过，地面接近受软限位约束，未发生真实夹持抬升。详见 ../../runs/mink_grasp/report.md。CLOSE/LIFT/HOLD 尚未物理验收成功。

从 /root/Desktop/TaskB 执行：
~~bash
bash scripts/run_mink_grasp.sh --mode ik-check --output runs/mink_grasp/ik_new
bash scripts/run_mink_grasp.sh --mode env-check --headless --enable_cameras --record-video --output runs/mink_envcheck/new
bash scripts/run_mink_grasp.sh --mode grasp --headless --enable_cameras --record-video --config experiments/mink_grasp/config_55.json --output runs/mink_grasp/new
~~

抓取诊断目前预期返回 1，并保存明确失败结果。输出目录存在时使用新目录。最终验收阈值保持为：上升 0.10 m、最低点离地 0.02 m、持续 3 s、姿态偏斜 <=15°、抓取阶段漂移 <=0.10 m，同时检查夹持区域与两侧接触。退出码不替代这些物理判据。

runtime.sh 使用当前镜像的 /isaac-sim/python.sh，即 Python 3.11.13。supervise.py 使用系统标准库监督子进程；没有在系统 Python 中运行 Isaac、Torch 推理或 IK。启动上限 180 s，总上限 480 s，单次试验还有配置中的内部仿真/真实时间上限。原生关闭超过输出落盘后 5 s 时，仅回收本次进程组，在 process_status.json 中单独报告。

controller.py 只接收机器人自身状态和配置中的固定目标。evaluator.py 的物体真值不反馈给控制器。没有篮子、导航、视觉检测、身体高度/俯仰控制、物体附着或原 Task B 规则修改。

参考项目：iit-DLSLab/trash-collection-isaaclab，固定提交 76c51c4c08e5762d00c75f36cb07aa3e525d6990。保留原 ik_mink.py、state_machine.py、BSD-3-Clause 许可证及核对记录。reference/ 下 XML 仅供几何审阅，不作为运行模型，不附带无用网格。运行模型 models/z1_urdf_reduced.xml 由本项目原 URDF 的关节原点、轴和有效限位生成；与独立 URDF FK 及 Isaac 实际腕部交叉检查。

Mink 1.0.0、MuJoCo 3.3.7 为安装的运行依赖，保留各自包许可证。没有复制或替换外部 Aliengo 权重。实际使用老师原 models/locomotion/policy.pt，哈希与机器人原始 URDF 哈希均保持不变。原项目资产来源和权利说明继续以根目录 THIRD_PARTY_NOTICES.md、LICENSE 和 assets/b2z1/SOURCE.json 为准。

完整镜像依赖快照、安装前后差异、CPU/GPU/相机检查与故障日志在 runs/mink_envcheck/20261023_081402/。这些快照描述现有 Isaac 镜像，不能单靠 pip freeze 重建 Isaac 二进制发行版。
