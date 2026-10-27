> Historical upstream documentation. Path-audit code referenced below was removed from this integration; see `grasp/README.md` for current behavior.

# 平放香蕉抓取

成功参数：四脚支撑，机身0.30m，前俯0.10rad，夹持点沿香蕉长度移动25mm。保留原资产、质量、摩擦、限力、限速及预紧。

```bash
bash scripts/grasp_banana.sh --output runs/banana_once
bash scripts/grasp_banana_angles.sh --output runs/banana_three
```

三朝向各一次，实测3/3。[完整结果](../../results/mink_banana_repair/report.md)。configs其他文件为调试候选，失败数据和逐次源码保留在结果目录。控制器不读取物体真实位姿。
