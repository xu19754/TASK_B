参考项目 iit-DLSLab/trash-collection-isaaclab，固定提交 76c51c4c08e5762d00c75f36cb07aa3e525d6990。保留 deploy/ik_mink.py、deploy/state_machine.py 的固定提交副本、来源哈希与 BSD-3-Clause 许可证，见 reference/。复用其 FrameTask、PostureTask、ConfigurationLimit、DAQP solve_ik 和分阶段抓取的用法；全身模型按本工程 B2+Z1 URDF 生成，没有照用 Aliengo 几何、编码器偏移或控制权重。mink_adapter.py 保留早期臂适配及几何工具，当前控制入口为 wholebody_ik.py。

Mink、MuJoCo 为已安装的运行依赖，遵循各自发行包许可证。本项目机器人、物体和老师模型的权利声明继续以工程根目录 LICENSE、THIRD_PARTY_NOTICES.md、assets/b2z1/SOURCE.json 为准。
