# 来源与修改说明

本目录是独立的教学 Task B 工程，不是 ATEC 官方原版发布包，也不代表官方赛题或评分规则更新。

- 任务、18 个物体的种类、固定投放圆桶和一次性区域计分设计，参考官方 `atecup/ATEC2026_Simulation_Challenge`，固定提交 `4000378a9a6fc6ce3e57bcdd20a1582f6854e0dc`。官方 MIT 许可证保留在 `LICENSE`。
- 官方仓库：<https://github.com/atecup/ATEC2026_Simulation_Challenge>。本版删除末端靠近计分，仅保留每物体一次的目标区域计分，总分 18。
- B2+Z1 基准 URDF 和网格最初复制自老师此前抓取整理包的 `assets/b2z1/0815_usd_export/`（历史来源路径；本包参考 URDF 位于 `assets/b2z1/b2z1_0815.urdf`）。原始生成说明、原生 0815 USD 来源和逐网格 SHA-256 保留在 `assets/b2z1/SOURCE.json`。原始参考 URDF 的 SHA-256 为 `5f286dbd2644bc60a115f00f4ebd7442736bce4f3a6935e7979f8dbcd18b8e90`，该文件及其既有机械结构保留原样。
- 当前默认模型为 `assets/b2z1/b2z1_0815_with_basket_v3.urdf`，在基准上增加用户最新版 `零件2.SLDPRT` 的尾部篮子和固定关节，保留原有 19 个可控关节。源 CAD 随包保存在 `assets/b2z1/basket_v3/零件2.SLDPRT`，SHA-256 为 `75ff5e19bf2b46551fd26042d03f5224faa94e827795f2acbb84448dda3fbb25`。通过 [XRTC5/sldprt-export](https://github.com/XRTC5/sldprt-export)（MIT，提交 `9c91ea0fd94090ce88efd6d37b61ba091243c715`）提取保存的显示网格；来源与许可证保留在 `tools/vendor/sldprt/`。碰撞分块由 MeshPy/TetGen 构建工具预生成，运行时不需要这些构建工具。模型参数及逐网格哈希见 `assets/b2z1/basket_v3/basket_properties.json`，运行模型哈希见 `SOURCE.json`。质量沿用 0.2 kg 假设；本次打包不代表已完成新版篮子的 Isaac/PhysX 动态验证。
- 区域物体 USD/贴图来自老师已有的官方 `atec_robot_model/objects/task_b/` 资源副本。机器人和物体资产的第三方权利仍归各自权利人；保留原始出处，不将其描述为本工程新创作的模型。
- 随附行走权重来自老师现有 B2+Z1 `domain_rand_r0_model_98499/exported/`，SHA-256 为 `f9983f63040d18cae743c20b1f7792f87b39246551b168678d8216901df32dca`。原始部署契约保留在 `models/locomotion/deployment_manifest.json`。本次只打包并适配已有模型，没有重新训练，也没有宣称已复现此前线上赛分数。
- 观测适配遵循原部署契约的关节顺序、缩放、步态相位和直行段航向语义。初始整理参考 Isaac Lab v2.3.2，服务器兼容验证使用 v2.2.1：<https://github.com/isaac-sim/IsaacLab/tree/v2.2.1>。PhysX LiDAR 使用 Isaac Sim 自带 `isaacsim.sensors.physx` 扩展的公开 schema/数据接口，未打包模拟器二进制或 SDK。

与线上赛原工程的其他差异：仅注册一个 B2+Z1 教学任务；采用独立包名；去掉其他任务和机器人；场地用本地网格生成，避免外部材质服务器依赖；物体初始化避开机器人与其他物体；提供显式关节目标的控制器和新的教学运行脚本；默认使用带随车篮子的机器人变体，尚未新增自动装篮或姿态倾倒控制。
