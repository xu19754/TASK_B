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
    """启发式分数，不是训练模型概率；没有明显证据时保留低分/未知类别。"""
    colors = np.asarray(colors, dtype=float) / 255.0
    r, g, b = colors.T
    high, low = colors.max(axis=1), colors.min(axis=1)
    saturation = (high - low) / np.maximum(high, 1e-6)
    yellow = float(np.mean((r > 0.35) & (g > 0.25) & (b < 0.65 * g) & (r > 0.85 * g)))
    white = float(np.mean((low > 0.5) & (saturation < 0.25)))
    blue = float(np.mean((b > 0.25) & (b > 1.25 * r) & (b > 1.1 * g)))
    red = float(np.mean((r > 0.3) & (r > 1.5 * g) & (r > 1.5 * b)))
    extent = np.ptp(points, axis=0)
    eig = np.linalg.eigvalsh(np.cov(points[:, :2].T))
    elongation = math.sqrt(max(eig[-1], 1e-9) / max(eig[0], 1e-9))
    scores = dict.fromkeys(CLASS_NAMES, 0.1)
    # 糖盒：白色面/蓝色印刷；芥末瓶：黄色瓶身和红/蓝标签。
    if white > 0.15 and (blue > 0.005 or yellow > 0.1):
        scores["sugar_box"] = 0.8
    if yellow > 0.35 and (red > 0.01 or blue > 0.02) and white < 0.15:
        scores["mustard_bottle"] = 0.8
    # 香蕉基线只接受低矮、细长、以黄色为主且没有瓶身标签的点云。
    if yellow > 0.65 and red < 0.01 and blue < 0.01 and white < 0.08:
        if extent[2] < 0.10 and elongation > 2.2:
            scores["banana"] = 0.75
    return scores


class RGBDGeometryDetector:
    """无需额外权重的初版检测器；可注入同签名 detect() 的训练模型替换。

    search_bounds 是公开的物品散布区域扩展，不依赖 seed 或真实物品坐标。
    分割依靠离地高度及三维相邻点距离；分类在遮挡/无纹理侧面可能为未知。
    """

    def __init__(self, stride=2, min_points=8, search_bounds=(-15.5, -4.5, -15.5, -4.5)):
        if not isinstance(stride, int) or stride < 1 or min_points < 3:
            raise ValueError("Invalid sampling stride or minimum cluster size.")
        bounds = np.asarray(search_bounds, dtype=float)
        if bounds.shape != (4,) or not np.isfinite(bounds).all() or bounds[0] >= bounds[1] or bounds[2] >= bounds[3]:
            raise ValueError("Invalid search_bounds.")
        self.stride, self.min_points = stride, min_points
        self.search_bounds = tuple(bounds)

    def detect(self, rgb, depth, intrinsics, base_position, base_quaternion):
        points = project_head_depth(depth, intrinsics, base_position, base_quaternion, self.stride)
        colors = rgb[::self.stride, ::self.stride]
        xmin, xmax, ymin, ymax = self.search_bounds
        finite = np.isfinite(points).all(axis=-1)
        region = (finite & (points[..., 0] >= xmin) & (points[..., 0] <= xmax)
                  & (points[..., 1] >= ymin) & (points[..., 1] <= ymax))
        # 用主要地面高度补偿原地步态造成的机身高度估计误差，不拟合物品真实位置。
        floor_samples = points[..., 2][region & (points[..., 2] > -0.25) & (points[..., 2] < 0.25)]
        if len(floor_samples) >= 200:
            histogram, edges = np.histogram(floor_samples, bins=100, range=(-0.25, 0.25))
            peak = int(histogram.argmax())
            if histogram[peak] > 0.1 * len(floor_samples):
                near_floor = floor_samples[np.abs(floor_samples - (edges[peak] + 0.0025)) < 0.015]
                points[..., 2] -= float(np.median(near_floor))
        distance = np.linalg.norm(points[..., :2] - np.asarray(base_position)[:2], axis=-1)
        mask = (region & (points[..., 2] > 0.018) & (points[..., 2] < 0.4)
                & (distance > 0.75) & (distance < 12.0))
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
                        if np.linalg.norm(points[r, c] - points[nr, nc]) <= 0.07:
                            visited[nr, nc] = True
                            pending.append((nr, nc))
            if len(component) < self.min_points:
                continue
            rr, cc = np.asarray(component).T
            cloud = points[rr, cc]
            extent = np.ptp(cloud, axis=0)
            if extent.max() > 0.45 or extent.max() < 0.035:
                continue
            box = (int(cc.min() * self.stride), int(rr.min() * self.stride),
                   min(rgb.shape[1] - 1, int((cc.max() + 1) * self.stride)),
                   min(rgb.shape[0] - 1, int((rr.max() + 1) * self.stride)))
            detections.append(Detection(cloud, classify_component(cloud, colors[rr, cc]), box))
        return detections
