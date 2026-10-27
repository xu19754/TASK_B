# 尾部篮子 URDF 变体

新模型：`../b2z1_0815_with_basket.urdf`。原模型 `../b2z1_0815.urdf` 保持原样。

## 几何来源与方向

- 来源为用户提供的 `/Users/yxh/Desktop/share_wind/零件1.SLDPRT`。
- 读取 SolidWorks 文件中保存的显示三角网格，并验证 `Contents/DisplayLists` 数据块的 CRC；没有通过图片重建，也没有改动篮子尺寸。
- 网格提取使用 [XRTC5/sldprt-export](https://github.com/XRTC5/sldprt-export)，版本及源文件 SHA-256 记录在 `basket_properties.json`。
- 外尺寸约 280 × 250 × 100 mm，壁厚约 2 mm；弧面外半径约 100 mm。
- 导出的 OBJ 使用米，URDF 的 mesh scale 为 `1 1 1`。
- 篮子局部轴与机身一致：X 前、Y 左、Z 上。弧形端朝机身尾部（-X），开口朝上。
- 篮子局部原点在外底面高度、外廓 XY 中心，范围约 X=[-0.14,0.14]、Y=[-0.125,0.125]、Z=[0,0.10] m。

## 安装

`base_to_tail_basket` 为固定关节，父 link 为 `base`，子 link 为 `tail_basket`。

```xml
<origin xyz="-0.3 0 0.13" rpy="0 0 0" />
```

这个安装位姿根据现有机身几何选择：篮子位于机身后部上方、左右居中；底面位于 base 坐标的 Z=0.13 m，高于机身视觉网格最高点约 13 mm。若需要调整位置，可修改新 URDF 末尾这一固定关节的 origin。

此版本仅增加一个篮子 link 和一个固定关节。原有关节、关节限位、19 维可控关节接口，以及 `base`、`link06` 等传感器挂载 link 均保持不变。

## 质量与惯性

质量采用用户指定的 **0.2 kg**。假设质量均匀分布于原 CAD 壳体实体，以封闭三角网格的有符号四面体积分计算质心和惯性，再归一化到 0.2 kg。没有把整个篮子按实心长方体估算。

篮子局部质心约为 `(0.0135781, 0, 0.0340719)` m。以该质心为原点、轴与篮子轴一致时，惯性张量约为，单位 kg·m²：

```text
[ 0.00189722954,  0,               0.0000829978622 ]
[ 0,              0.00199733114,   0               ]
[ 0.0000829978622, 0,              0.00346714252   ]
```

URDF 中保留了计算得到的微小非零项，完整精度见 `basket_properties.json`。质量按照指定值设置，不通过塑料密度反推，因此 JSON 中的 effective_density 只是归一化结果。

## 碰撞

共 **16 个碰撞体**：

- 底板、前侧直壁各一个 box，共 2 个。
- 左右侧壁各一个贴合弧形轮廓的凸网格，共 2 个。
- 弧形底面分为 12 个独立的凸网格段。

它们共同保留内部空腔和顶部开口。篮子整体没有使用一个包围盒或单个凸包，否则会把容纳物体的空间填满。

碰撞分块与 CAD 显示网格使用相同的弧面分段。分块总体积与视觉壳体网格体积的相对差异约 0.000094%，来自 CAD 缓存坐标的浮点精度。每个碰撞网格均为闭合凸体。

URDF 和 OBJ 的蓝色塑料材质用于外观。标准 URDF material 不定义静摩擦、动摩擦或恢复系数，这些接触参数由 Isaac Sim 中实际绑定的 physics material 决定；本次没有修改环境或目标物体的物理参数。

## 加载与文件保留

项目的 `taskb_student/constants.py` 已默认指向 `b2z1_0815_with_basket.urdf`，`robot.py` 的 `UrdfFileCfg.asset_path` 通过该常量加载模型。现有 `scripts/run_taskb.sh` 和 `scripts/play_taskb.py` 因此会直接使用带篮子的机器人，并在启动时打印 URDF 路径；无需另外手动导入篮子。

默认资产哈希和 `SOURCE.json` 的运行模型记录已同步更新，原始参考 URDF 和原始网格的哈希校验继续保留。机器人沿用现有的浮动基座、固定关节保留和 convex_decomposition 配置。

在已安装 Isaac Sim / Isaac Lab 的环境中，从 TaskB 工程根目录执行：

```bash
python scripts/check_package.py
bash scripts/run_taskb.sh --steps 500 --real-time --debug
```

第一条执行离线资产与策略推理检查；第二条以实时节奏运行约 10 秒仿真，用于查看空篮机器人。需要前视和腕部相机时添加 `--enable_cameras`。单独使用 Isaac Sim 图形界面时，也可以在 URDF 导入器中选择新文件。

移动新 URDF 时必须同时保留其引用的整个 `meshes/` 目录，包括新增 `meshes/basket/`，不能只复制 URDF 单文件。

## 已验证与未验证

已检查：源几何数据块校验、网格闭合与绕序、碰撞凸性和开放容纳空间、碰撞分块与视觉形状一致性、质心与惯性合法性、全部网格路径、19 个可控关节，以及原 URDF 文件和既有元素保持不变。

安装预览由 Blender 根据新 URDF 的实际网格、关节变换和工程默认关节姿态渲染，属于几何预览。当前未在 Isaac Sim 中执行加载、站立、带载行走、机械臂装篮或姿态倾倒测试；也没有添加倾倒控制器。全关节运动范围内的间隙和实际卸货效果仍需在目标仿真环境验证。

预览文件：`robot_with_basket.png`、`robot_with_basket_side.png`、`basket_detail.png`、`basket_collisions.png`。
