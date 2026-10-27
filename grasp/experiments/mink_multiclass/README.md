> Historical upstream documentation. Path-audit code referenced below was removed from this integration; see `grasp/README.md` for current behavior.

# 三物体、三朝向抓取测试

固定物体中心在机器人初始位置正前方 x=0.55 m；芥末瓶、糖盒、香蕉各测试竖直轴朝向 -45、0、+45 度。每个组合仅运行一次，共九次，没有距离档位、随机种子扩展、自动重试或失败后补测。

运行：

~~bash
bash scripts/run_mink_multiclass.sh --output runs/mink_multiclass/my_nine
~~

使用同一服务器 Isaac 环境及 scripts/run_mink_wholebody.sh。输出目录必须不存在；拒绝复用目录，防止误重跑。每个组合有 started/completed 记录、一个 trial_01、连续录像、控制和评价轨迹。总表在 summary.json、trials.csv。

所有实际抓取使用上版相同的 wholebody/controller、Mink IK、Isaac 场景、执行器、原始质量/重力和成功阈值，仅替换已知物体资产、初始化朝向、预设夹持点、夹爪姿态/TCP 和所需机身高度。控制器不接收物体实时真值。

| 物体 | 放置方式 | 下蹲高度 | TCP 在 link06 的偏置 m | 夹持位置 |
| --- | --- | --- | --- | --- |
| 芥末瓶 | 直立 | 0.42 m | [0.21,0,0.02] | 上部瓶颈 |
| 糖盒 | 直立 | 0.42 m | [0.21,0,0.012] | 上部窄面 |
| 香蕉 | 自然平放 | 0.34 m | [0.235,0,0.012] | 中段；靠近指尖避免触地 |

三物体使用各自预设局部夹持点；夹爪目标随配置中的物体初始朝向共同旋转。这是在已知目标位姿条件下检查物理可抓性，不是视觉检测、未知姿态估计或自动选择抓取部位。

当前控制器能原地协调腿、机身和机械臂，会下蹲和起身；没有主动走近/后退的距离调节功能。走到可抓范围属于后续导航，所以本轮不测试距离。

prepare.py 是原服务器上的几何/运动学预检查与配置生成记录，读取前轮站立轨迹；本轮复现直接使用已保存的 configs/ 和 suite.json，无需重新生成配置。audit_suite.py 只检查已有轨迹，不增加物理试验。
