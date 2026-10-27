"""RGB-D 点云分割及可替换的颜色/形状分类基线，不读取场景真实物体状态。"""

from dataclasses import dataclass
import math

import numpy as np


CLASS_NAMES = ("sugar_box", "mustard_bottle", "banana")


def as_numpy(value):
    if hasattr(value, "detach"):
        value = value.detach().cpu().numpy()
    return np.asarray(value)


def quaternion_matrix(quaternion):
    q = as_numpy(quaternion).astype(float).reshape(-1)
    if q.shape != (4,) or not np.isfinite(q).all() or np.linalg.norm(q) < 1e-8:
        raise ValueError("base_quat must be a finite nonzero (w, x, y, z) quaternion.")
    w, x, y, z = q / np.linalg.norm(q)
    return np.array([
        [1 - 2 * (y*y + z*z), 2 * (x*y - w*z), 2 * (x*z + w*y)],
        [2 * (x*y + w*z), 1 - 2 * (x*x + z*z), 2 * (y*z - w*x)],
        [2 * (x*z - w*y), 2 * (y*z + w*x), 1 - 2 * (x*x + y*y)],
    ])


def read_head_frame(obs):
    """接收官方单环境 tensor 或 numpy 观测，返回 HWC RGB、HW 米制深度及 K。"""
    group = obs.get("image", {})
    if any(key not in group for key in ("head_rgb", "head_depth", "head_intrinsics")):
        raise ValueError("Head RGB-D is missing; start the simulator with --enable_cameras.")
    rgb = as_numpy(group["head_rgb"])
    depth = as_numpy(group["head_depth"])
    intrinsics = as_numpy(group["head_intrinsics"]).astype(float)
    if rgb.ndim == 4 and rgb.shape[0] == 1:
        rgb = rgb[0]
    if depth.ndim == 4 and depth.shape[0] == 1:
        depth = depth[0]
    if depth.ndim == 3 and depth.shape[-1] == 1:
        depth = depth[..., 0]
    if intrinsics.shape == (1, 3, 3):
        intrinsics = intrinsics[0]
    if rgb.ndim != 3 or rgb.shape[-1] not in (3, 4) or depth.shape != rgb.shape[:2]:
        raise ValueError("RGB-D must be aligned HxWx3/4 RGB and HxW optical-axis depth.")
    if (intrinsics.shape != (3, 3) or not np.isfinite(intrinsics).all()
            or intrinsics[0, 0] <= 0 or intrinsics[1, 1] <= 0
            or not np.allclose(intrinsics[2], (0, 0, 1))):
        raise ValueError("Invalid camera intrinsics.")
    rgb = rgb[..., :3]
    if not np.isfinite(rgb).all():
        raise ValueError("RGB contains non-finite values.")
    if np.issubdtype(rgb.dtype, np.floating) and rgb.max(initial=0) <= 1:
        rgb = rgb * 255
    rgb = np.clip(rgb, 0, 255).astype(np.uint8)
    depth = depth.astype(float)
    if not np.any(np.isfinite(depth) & (depth > 0.05) & (depth < 50)):
        raise ValueError("Camera depth has not initialized or contains no valid measurements.")
    return rgb, depth, intrinsics


def project_head_depth(depth, intrinsics, base_position, base_quaternion, stride=2):
    """反投影 distance_to_image_plane，返回采样后的世界点云 (H', W', 3)。

    env_cfg 中 head_camera 的平移为 (0.4, 0, 0.12)，旋转为单位四元数，
    convention='world'：+X 向前、+Z 向上。光学坐标 (右, 下, 前)
    因此转换为机体 (前, 左, 上)=(z, -x, -y)，再加相机平移并乘机体旋转。
    """
    if not isinstance(stride, int) or stride < 1:
        raise ValueError("stride must be a positive integer.")
    depth = np.asarray(depth, dtype=float)
    intrinsics = np.asarray(intrinsics, dtype=float)
    position = np.asarray(base_position, dtype=float)
    if depth.ndim != 2 or position.shape != (3,) or not np.isfinite(position).all():
        raise ValueError("Expected HW depth and finite (x, y, z) base_position.")
    if (intrinsics.shape != (3, 3) or not np.isfinite(intrinsics).all()
            or intrinsics[0, 0] <= 0 or intrinsics[1, 1] <= 0):
        raise ValueError("Invalid camera intrinsics.")
    v, u = np.mgrid[0:depth.shape[0]:stride, 0:depth.shape[1]:stride]
    d = depth[::stride, ::stride].copy()
    d[~np.isfinite(d) | (d <= 0.05) | (d >= 50)] = np.nan
    rays = np.stack((u, v, np.ones_like(u)), axis=-1) @ np.linalg.inv(intrinsics).T
    optical = rays * d[..., None]
    body = np.stack((optical[..., 2], -optical[..., 0], -optical[..., 1]), axis=-1)
    body += (0.4, 0.0, 0.12)
    return body @ quaternion_matrix(base_quaternion).T + position


@dataclass
class Detection:
    points_world: np.ndarray
    class_scores: dict[str, float]
    bbox: tuple[int, int, int, int]


def classify_component(points, colors):
    """颜色比例结合离地高度分类；分数是启发式证据，不是模型概率。

    points 的 z 须已校正到地面。白色反光不等于糖盒标签，瓶背面也不一定
    能看到红/蓝商标；低矮弯曲香蕉不要求每一帧都有很高的 PCA 长宽比。
    """
    colors = np.asarray(colors, dtype=float) / 255.0
    r, g, b = colors.T
    high, low = colors.max(axis=1), colors.min(axis=1)
    saturation = (high - low) / np.maximum(high, 1e-6)
    yellow = float(np.mean((r > 0.25) & (g > 0.18) & (b < 0.78 * np.minimum(r, g))
                           & (r > 0.85 * g) & (g > 0.55 * r)))
    white = float(np.mean((low > 0.5) & (saturation < 0.18)))
    blue = float(np.mean((b > 0.25) & (b > 1.25 * r) & (b > 1.1 * g)))
    red = float(np.mean((r > 0.3) & (r > 1.5 * g) & (r > 1.5 * b)))
    low_point, high_point = np.percentile(points, (2, 98), axis=0)
    extent = high_point - low_point
    eig = np.linalg.eigvalsh(np.cov(points[:, :2].T))
    elongation = math.sqrt(max(eig[-1], 1e-9) / max(eig[0], 1e-9))
    scores = dict.fromkeys(CLASS_NAMES, 0.1)
    # 高的黄色瓶身：允许白色背标、蓝色浅色商标，以及完全看不到商标的侧面。
    if high_point[2] > 0.105 and extent[2] > 0.085 and yellow > 0.30 and yellow > 0.9 * white:
        scores["mustard_bottle"] = 0.88
    # 低矮、黄色占主导的物体：少量白色高光或棕色果柄不会否定香蕉。
    elif (high_point[2] < 0.10 and extent[2] < 0.085 and extent[:2].max() > 0.07
          and yellow > 0.48 and yellow > 1.15 * white):
        scores["banana"] = 0.85 if elongation > 1.4 else 0.80
    elif white > 0.18 and white > 0.65 * yellow and (blue > 0.005 or yellow > 0.04):
        scores["sugar_box"] = 0.85
    elif high_point[2] > 0.10 and yellow > 0.40 and (red > 0.015 or blue > 0.02):
        # 部分遮挡的瓶身仍可利用商标，但不把低矮香蕉的果柄当作红色标签。
        scores["mustard_bottle"] = 0.75
    return scores


def estimate_ground_plane(points, region):
    """从 RGB-D 的大面积地面拟合 z=a*x+b*y+c，容忍高度和小幅姿态误差。

    RANSAC 不使用场景坐标真值；固定随机序列仅为离线回放可复现。
    点数/覆盖范围/内点比例不足时返回 None，不将小物体表面当成地面。
    """
    sample = points[region & (points[..., 2] > -0.35) & (points[..., 2] < 0.35)]
    if len(sample) < 200:
        return None
    if len(sample) > 6000:
        sample = sample[np.linspace(0, len(sample) - 1, 6000, dtype=int)]
    origin = np.median(sample[:, :2], axis=0)
    xy = sample[:, :2] - origin
    design = np.column_stack((xy, np.ones(len(sample))))
    z = sample[:, 2]
    best, count = None, 0
    rng = np.random.default_rng(0)
    for _ in range(48):
        indices = rng.choice(len(sample), 3, replace=False)
        triangle = design[indices]
        if abs(np.linalg.det(triangle)) < 0.05:
            continue
        plane = np.linalg.solve(triangle, z[indices])
        if np.linalg.norm(plane[:2]) > 0.25:
            continue
        inliers = np.abs(design @ plane - z) < 0.006
        support = int(inliers.sum())
        if support > count:
            best, count = inliers, support
    if count < max(200, 0.35 * len(sample)):
        return None
    for _ in range(3):
        plane = np.linalg.lstsq(design[best], z[best], rcond=None)[0]
        best = np.abs(design @ plane - z) < 0.006
        if int(best.sum()) < max(200, 0.35 * len(sample)):
            return None
    if np.linalg.norm(plane[:2]) > 0.25 or np.linalg.eigvalsh(np.cov(xy[best].T))[0] < 0.02:
        return None
    return np.array((plane[0], plane[1], plane[2] - plane[:2] @ origin))


class RGBDGeometryDetector:
    """无需额外权重的初版检测器；可注入同签名 detect() 的训练模型替换。

    search_bounds 是公开的物品散布区域扩展，不依赖 seed 或真实物品坐标。
    分割依靠离地高度及三维相邻点距离；分类在遮挡/无纹理侧面可能为未知。
    """

    def __init__(self, stride=1, min_points=8, search_bounds=(-15.5, -4.5, -15.5, -4.5)):
        if not isinstance(stride, int) or stride < 1 or min_points < 3:
            raise ValueError("Invalid sampling stride or minimum cluster size.")
        bounds = np.asarray(search_bounds, dtype=float)
        if bounds.shape != (4,) or not np.isfinite(bounds).all() or bounds[0] >= bounds[1] or bounds[2] >= bounds[3]:
            raise ValueError("Invalid search_bounds.")
        self.stride, self.min_points = stride, min_points
        self.search_bounds = tuple(bounds)
        self.last_diagnostics = {}

    def detect(self, rgb, depth, intrinsics, base_position, base_quaternion):
        intrinsics = np.asarray(intrinsics, dtype=float)
        points = project_head_depth(depth, intrinsics, base_position, base_quaternion, self.stride)
        colors = rgb[::self.stride, ::self.stride]
        xmin, xmax, ymin, ymax = self.search_bounds
        finite = np.isfinite(points).all(axis=-1)
        region = (finite & (points[..., 0] >= xmin) & (points[..., 0] <= xmax)
                  & (points[..., 1] >= ymin) & (points[..., 1] <= ymax))
        # 单个高度直方图无法去除略倾斜的地面，会在搜索区域边缘留下假物体。
        plane = estimate_ground_plane(points, region)
        if plane is not None:
            points[..., 2] -= points[..., 0] * plane[0] + points[..., 1] * plane[1] + plane[2]
        self.last_diagnostics = {"ground_plane": None if plane is None else plane.tolist(),
                                 "foreground_pixels": 0, "rejected_flat_components": 0}
        distance = np.linalg.norm(points[..., :2] - np.asarray(base_position)[:2], axis=-1)
        # 保留香蕉约 1 cm 高的边缘，减少低分辨率、远距离下的碎片和漏检。
        mask = (region & (points[..., 2] > 0.009) & (points[..., 2] < 0.4)
                & (distance > 0.75) & (distance < 12.0))
        self.last_diagnostics["foreground_pixels"] = int(mask.sum())
        camera = np.asarray(base_position) + quaternion_matrix(base_quaternion) @ np.array((0.4, 0.0, 0.12))
        camera_height = camera[2] if plane is None else camera[2] - (camera[:2] @ plane[:2] + plane[2])
        # 俯视角很小时，相邻像素在低矮物体表面的间距可能超过固定的 7 cm。
        # 由距离和像素角度估算采样间距，仅在相邻点高度一致时放宽，且上限 14 cm。
        sampling_gap = distance**2 * self.stride / (intrinsics[1, 1] * np.maximum(camera_height - points[..., 2], 0.15))
        flat_limits = np.clip(1.2 * sampling_gap, 0.07, 0.14)
        # 八邻域连通分量，同时限制三维间距，分开前后遮挡的物品。
        visited = np.zeros(mask.shape, dtype=bool)
        height, width = mask.shape
        detections = []
        for row, col in zip(*np.nonzero(mask)):
            if visited[row, col]:
                continue
            pending, component = [(row, col)], []
            visited[row, col] = True
            while pending:
                r, c = pending.pop()
                component.append((r, c))
                for dr, dc in ((-1, -1), (-1, 0), (-1, 1), (0, -1), (0, 1), (1, -1), (1, 0), (1, 1)):
                    nr, nc = r + dr, c + dc
                    if 0 <= nr < height and 0 <= nc < width and mask[nr, nc] and not visited[nr, nc]:
                        limit = 0.07
                        if abs(points[r, c, 2] - points[nr, nc, 2]) < 0.008:
                            limit = max(flat_limits[r, c], flat_limits[nr, nc])
                        if np.linalg.norm(points[r, c] - points[nr, nc]) <= limit:
                            visited[nr, nc] = True
                            pending.append((nr, nc))
            if len(component) < self.min_points:
                continue
            rr, cc = np.asarray(component).T
            cloud = points[rr, cc]
            extent = np.ptp(cloud, axis=0)
            if extent.max() > 0.45 or extent.max() < 0.035 or np.percentile(cloud[:, 2], 98) < 0.018:
                continue
            scores = classify_component(cloud, colors[rr, cc])
            # 当地面不足以拟合时，仍拒绝无任何类别证据、几乎零厚度的平面碎片。
            if extent[2] < 0.004 and max(scores.values()) < 0.65:
                self.last_diagnostics["rejected_flat_components"] += 1
                continue
            box = (int(cc.min() * self.stride), int(rr.min() * self.stride),
                   min(rgb.shape[1] - 1, int((cc.max() + 1) * self.stride)),
                   min(rgb.shape[0] - 1, int((rr.max() + 1) * self.stride)))
            detections.append(Detection(cloud, scores, box))
        return detections
