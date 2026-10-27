# 机械狗与物体训练场景

独立的 Isaac Lab 场景，包含 20 m × 20 m 地面、B2+Z1 机械狗（保留当前尾部篮子），以及糖盒、芥末瓶、香蕉各 1 个，共 3 个物体。机械狗在 `(0, 0, 0.6)` 出生，使用已有底层 policy 保持站立和收臂；物体集中摆在机身原点正前方约 `1.5 m`，横向中心间距约 `0.30 m`，便于后续视觉与雷达观察。物体启用重力和物理碰撞，`--seed` 仅控制各位置在水平面上不超过 `±0.02 m` 的小幅扰动。

这一阶段仅搭建场景，没有识别、分割、自动标注、数据采集或模型训练逻辑。它使用独立的场景配置与启动器，不加载正式比赛的计分器和终止条件。

使用已安装 Isaac Sim / Isaac Lab 的 Python 环境，在工程根目录启动：

```bash
conda activate isaaclab
bash scripts/run_perception.sh --real-time
```

启动后可在 Isaac Sim 窗口中查看机械狗和物体，使用视口操作缩放、旋转和平移；关闭窗口退出。

换一组物体布局：

```bash
bash scripts/run_perception.sh --seed 123 --real-time
```

无窗口运行 500 个控制步（10 秒仿真时间），并保存最终状态用于检查：

```bash
bash scripts/run_perception.sh --headless --steps 500 \
  --report runs/perception/scene_check.json
```

默认单环境、CPU PhysX，窗口图像由 GPU 渲染；无需重新安装 Python 环境或重新训练步态。可用 `TASKB_PYTHON=/完整路径/python` 指定解释器。需要图形窗口时，应在具备有效 `DISPLAY` 的桌面会话中执行。

场景配置位于 `training/perception/scene.py`；启动与站立控制位于 `training/perception/play.py`。生成的状态报告存放在已被 Git 忽略的 `runs/` 下。
