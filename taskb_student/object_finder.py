"""启动前原地扫描一周，融合 RGB-D 物品种类、位置和水平主轴估计。

只使用规定图像、机体姿态、实测关节角和 Localization，不读取物体真值。
默认检测器是可替换的启发式基线，不能保证一次扫描识别全部遮挡物品。
position 为可见表面包围盒中心估计；yaw_axis 为模 pi 的水平长轴，
不是物品 CAD 坐标系朝向，不提供未经估计的 roll/pitch 或完整 6D 姿态。
"""

from dataclasses import asdict, dataclass, field
import hashlib
import json
import math
from pathlib import Path

import numpy as np

from .constants import ARM_Q0, CONTROL_DT
from .camera_geometry import SCAN_ARM_TARGET, wrist_camera_transform
from .rgbd_perception import CLASS_NAMES, Detection, RGBDGeometryDetector, as_numpy, quaternion_matrix, read_head_frame, read_rgbd_frame


@dataclass(frozen=True)
class ObjectEstimate:
    object_id: int
    class_name: str
    position: tuple[float, float, float]
    size: tuple[float, float, float]
    yaw_axis: float | None
    confidence: float
    views: int
    confirmed: bool
    orientation_ambiguous: bool = True
    candidate_class: str | None = None
    classification_views: int = 0
    unknown_reason: str | None = None


@dataclass
class _Track:
    object_id: int
    points: np.ndarray
    views: set = field(default_factory=set)
    class_votes: dict = field(default_factory=lambda: {name: [] for name in CLASS_NAMES})
    view_angles: list = field(default_factory=list)
    view_scores: dict = field(default_factory=dict)

    def estimate(self):
        low, high = np.percentile(self.points, (2, 98), axis=0)
        position = tuple(float(v) for v in (low + high) / 2)
        size = tuple(float(v) for v in high - low)
        winner = max(CLASS_NAMES, key=lambda name: sum(self.class_votes[name]))
        votes = self.class_votes[winner]
        total = sum(sum(scores) for scores in self.class_votes.values())
        confidence = float(np.mean(votes)) * sum(votes) / total if votes else 0.0
        confirmed = len(self.views) >= 2 and len(votes) >= 2 and confidence >= 0.65
        class_name = winner if confirmed else "unknown"
        reason = None
        if not votes:
            reason = "insufficient_color_or_geometry_evidence"
        elif len(votes) < 2:
            reason = "class_seen_in_fewer_than_two_views"
        elif not confirmed:
            reason = "conflicting_class_evidence"
        yaw = None
        # 至少三个独立方向、20 度视角跨度才尝试输出水平轴，始终保留 180 度歧义。
        angles = np.asarray(self.view_angles)
        diverse = len(angles) >= 3 and np.max(np.abs(np.angle(np.exp(1j * (angles - angles[0]))))) >= math.radians(20)
        if diverse and len(self.points) >= 12:
            values, vectors = np.linalg.eigh(np.cov(self.points[:, :2].T))
            if values[-1] > 1.8 * max(values[0], 1e-9):
                axis = vectors[:, -1]
                yaw = float(math.atan2(axis[1], axis[0]) % math.pi)
        return ObjectEstimate(self.object_id, class_name, position, size, yaw,
                              confidence, len(self.views), confirmed, candidate_class=winner if votes else None,
                              classification_views=len(votes), unknown_reason=reason)


class ObjectMap:
    """按空间邻近做每帧一对一关联，不按已知数量强制归类或截断到 18 件。"""

    def __init__(self, association_distance=0.22):
        if not math.isfinite(association_distance) or association_distance <= 0:
            raise ValueError("association_distance must be finite and positive.")
        self.association_distance = association_distance
        self._tracks = []

    @property
    def objects(self):
        return tuple(track.estimate() for track in self._tracks)

    def update(self, detections, view_id, view_angle):
        centers = []
        for detection in detections:
            points = np.asarray(detection.points_world, dtype=float)
            if points.ndim != 2 or points.shape[1] != 3 or len(points) < 3 or not np.isfinite(points).all():
                raise ValueError("Detector points_world must contain at least three finite XYZ points.")
            if any(name not in CLASS_NAMES or not math.isfinite(score) or not 0 <= score <= 1
                   for name, score in detection.class_scores.items()):
                raise ValueError("Detector class_scores must use known classes with scores in [0, 1].")
            low, high = np.percentile(points, (2, 98), axis=0)
            centers.append((low + high) / 2)
        candidates = []
        track_positions = [track.estimate().position for track in self._tracks]
        for detection_index, center in enumerate(centers):
            for track_index, position in enumerate(track_positions):
                distance = math.dist(center[:2], position[:2])
                if distance <= self.association_distance:
                    candidates.append((distance, detection_index, track_index))
        assigned, used_tracks = {}, set()
        for _, detection_index, track_index in sorted(candidates):
            if detection_index not in assigned and track_index not in used_tracks:
                assigned[detection_index] = track_index
                used_tracks.add(track_index)
        object_ids = []
        for index, detection in enumerate(detections):
            if index not in assigned:
                assigned[index] = len(self._tracks)
                self._tracks.append(_Track(len(self._tracks), np.empty((0, 3))))
            track = self._tracks[assigned[index]]
            object_ids.append(track.object_id)
            if view_id not in track.views:
                track.views.add(view_id)
                track.view_angles.append(float(view_angle))
            points = np.concatenate((track.points, detection.points_world))
            _, unique = np.unique(np.floor(points / 0.01).astype(np.int64), axis=0, return_index=True)
            points = points[np.sort(unique)]
            if len(points) > 2500:
                points = points[np.linspace(0, len(points) - 1, 2500, dtype=int)]
            track.points = points
            # 同一方向的双相机只算一票；点云都融合，类别冲突时该方向不投票。
            previous = track.view_scores.get(view_id, {})
            track.view_scores[view_id] = {name: max(previous.get(name, 0.0), detection.class_scores.get(name, 0.0))
                                         for name in CLASS_NAMES}
            track.class_votes = {name: [] for name in CLASS_NAMES}
            for scores in track.view_scores.values():
                ranking = sorted(((scores[name], name) for name in CLASS_NAMES), reverse=True)
                if ranking[0][0] >= 0.65 and ranking[0][0] - ranking[1][0] >= 0.15:
                    track.class_votes[ranking[0][1]].append(float(ranking[0][0]))
        return object_ids


class ObjectFinder:
    """扫描状态机；compute_command 每个 CONTROL_DT 调用一次。

    原地分段转满一周，在每个方向停稳超过两次相机刷新周期后采集。
    转角根据 proprio.base_quat 的实际变化累计，不能由下达指令的时长代替。
    完成时 done=True；只有 18 件均确认、各类别恰好 6 件才 ready=True。
    默认 24 个方向，另采初始方向；全程不下达平移命令。
    有腕部图像和关节观测时，先调整腕部俯角，双相机同步采集，结束后恢复收臂。
    """

    def __init__(self, localization, detector=None, *, scan_views=24, yaw_rate=0.4,
                 settle_time=0.3, max_duration=90.0, record_rgbd=False, use_wrist_camera=True):
        if not isinstance(scan_views, int) or scan_views < 8:
            raise ValueError("scan_views must be an integer >= 8.")
        if not math.isfinite(yaw_rate) or not 0.1 < yaw_rate <= 0.6:
            raise ValueError("yaw_rate must exceed the controller deadband and be <= 0.6.")
        if not math.isfinite(settle_time) or settle_time < 0.2:
            raise ValueError("settle_time must cover at least two 0.1-second camera updates.")
        if not math.isfinite(max_duration) or max_duration <= 0:
            raise ValueError("max_duration must be finite and positive.")
        self.localization = localization
        self.detector = detector if detector is not None else RGBDGeometryDetector()
        self.scan_views, self.yaw_rate = scan_views, yaw_rate
        self.settle_time, self.max_duration = settle_time, max_duration
        self.record_rgbd = record_rgbd
        self.use_wrist_camera = use_wrist_camera
        self.wrist_detector = RGBDGeometryDetector()
        self.reset()

    def reset(self):
        self.object_map = ObjectMap()
        self.status = "waiting_for_camera"
        self.blocked_reason = None
        self.total_yaw = 0.0
        self.elapsed = 0.0
        self._last_yaw = None
        self._sector = 0
        self._settled = 0.0
        self._seen_frames = set()
        self._snapshots = []
        self._wrist_snapshots = []
        self._wrist_enabled = None
        self._capture_finished = False
        self.captured_turn_radians = None
        self._arm_stable = 0.0
        self._last_arm_joints = None
        self._rgbd_frames = []
        self._view_diagnostics = []
        self.repeated_frames = 0

    @property
    def objects(self):
        return self.object_map.objects

    @property
    def done(self):
        return self.status in ("complete", "incomplete", "failed")

    @property
    def ready(self):
        return self.status == "complete"

    @property
    def observed_yaw(self):
        """最近一次观测到的世界航向，可用于扫描结束时校正定位航向。"""
        return self._last_yaw

    @property
    def arm_target(self):
        if self._wrist_enabled and not self._capture_finished and not self.done:
            return SCAN_ARM_TARGET
        return tuple(ARM_Q0)

    def _arm_is_settled(self, obs):
        joints = as_numpy(obs["proprio"]["joint_pos"]).astype(float).reshape(-1)
        if joints.shape != (19,) or not np.isfinite(joints).all():
            raise ValueError("Wrist scanning requires 19 finite measured joint positions.")
        arm = joints[12:]
        movement = 0.0 if self._last_arm_joints is None else float(np.max(np.abs(arm - self._last_arm_joints)))
        self._last_arm_joints = arm.copy()
        stationary = np.max(np.abs(arm - self.arm_target)) < 0.04 and movement < 0.004
        self._arm_stable = self._arm_stable + CONTROL_DT if stationary else 0.0
        return self._arm_stable >= self.settle_time

    def _fail(self, reason):
        self.status, self.blocked_reason = "failed", str(reason)
        return (0.0, 0.0, 0.0)

    def compute_command(self, obs):
        stop = (0.0, 0.0, 0.0)
        if self.done:
            return stop
        self.elapsed += CONTROL_DT
        if self.elapsed > self.max_duration:
            return self._fail(f"Scan timed out in {self.status}: check camera readiness, measured arm joints and robot rotation.")
        if "image" not in obs or any(key not in obs["image"] for key in ("head_rgb", "head_depth", "head_intrinsics")):
            return self._fail("Head RGB-D is missing; start with --enable_cameras.")
        try:
            quaternion = obs["proprio"]["base_quat"]
            rotation = quaternion_matrix(quaternion)
        except (KeyError, ValueError) as error:
            return self._fail(f"Missing/invalid base orientation: {error}")
        yaw = math.atan2(rotation[1, 0], rotation[0, 0])
        delta = 0.0 if self._last_yaw is None else (yaw - self._last_yaw + math.pi) % (2 * math.pi) - math.pi
        self._last_yaw = yaw
        if abs(delta) > 0.2:
            return self._fail("Base orientation jumped between control frames; reset the scanner.")
        self.total_yaw += delta
        if self._wrist_enabled is None:
            self._wrist_enabled = bool(self.use_wrist_camera and "joint_pos" in obs["proprio"]
                                       and all(f"ee_{key}" in obs["image"] for key in ("rgb", "depth", "intrinsics")))
        if self._wrist_enabled:
            try:
                arm_ready = self._arm_is_settled(obs)
            except ValueError as error:
                return self._fail(str(error))
            if not arm_ready:
                self.status = "restoring_arm" if self._capture_finished else "positioning_wrist_camera"
                self._settled = 0.0
                return stop
        if self._capture_finished:
            self.finish_scan()
            return stop
        target = self._sector * 2 * math.pi / self.scan_views
        error = target - self.total_yaw
        if abs(error) > 0.005:
            self.status, self._settled = "rotating", 0.0
            speed = min(self.yaw_rate, max(0.12, 1.5 * abs(error)))
            return (0.0, 0.0, math.copysign(speed, error))
        self.status = "settling"
        self._settled = self._settled + CONTROL_DT if abs(delta) < 0.002 else 0.0
        if self._settled < self.settle_time:
            return stop
        try:
            rgb, depth, intrinsics = read_head_frame(obs)
            if self._wrist_enabled:
                wrist_rgb, wrist_depth, wrist_k = read_rgbd_frame(obs, "ee")
        except ValueError as error:
            self.status, self.blocked_reason = "waiting_for_camera", str(error)
            return stop
        self.blocked_reason = None
        try:
            self.ingest_frame(rgb, depth, intrinsics, self.localization.position, quaternion, self._sector)
            if self._wrist_enabled:
                self.ingest_frame(wrist_rgb, wrist_depth, wrist_k, self.localization.position, quaternion,
                                  self._sector, camera="ee", joint_positions=obs["proprio"]["joint_pos"])
        except ValueError as error:
            return self._fail(f"Invalid detector result: {error}")
        if self._sector == self.scan_views:
            self._capture_finished = True
            self.captured_turn_radians = self.total_yaw
            self._arm_stable = 0.0
            if self._wrist_enabled:
                self.status = "restoring_arm"
            else:
                self.finish_scan()
            return stop
        self._sector += 1
        self._settled = 0.0
        return stop

    def ingest_frame(self, rgb, depth, intrinsics, base_position, base_quaternion, view_id, *,
                     camera="head", joint_positions=None):
        """处理实测帧或离线记录；不生成机器人运动，不读取仿真场景真值。"""
        rgb, depth, intrinsics = read_head_frame({"image": {
            "head_rgb": rgb, "head_depth": depth, "head_intrinsics": intrinsics,
        }})
        rotation = quaternion_matrix(base_quaternion)
        yaw = math.atan2(rotation[1, 0], rotation[0, 0])
        # 相机原始深度为 float32，统一精度保证 NPZ 保存/回放具有相同去重行为。
        depth = depth.astype(np.float32)
        if camera not in ("head", "ee"):
            raise ValueError("Supported cameras are head and ee.")
        fingerprint = (camera, hashlib.sha256(rgb.tobytes() + depth.tobytes()).digest())
        if fingerprint in self._seen_frames:
            self.repeated_frames += 1
            return
        detector = self.detector if camera == "head" else self.wrist_detector
        if camera == "ee":
            mount = wrist_camera_transform(joint_positions)
            detections = detector.detect(rgb, depth, intrinsics, base_position, base_quaternion, camera_transform=mount)
            self._wrist_enabled = True
        else:
            detections = detector.detect(rgb, depth, intrinsics, base_position, base_quaternion)
        object_ids = self.object_map.update(detections, view_id, yaw)
        self._seen_frames.add(fingerprint)
        rgb = rgb.copy()
        snapshots = self._snapshots if camera == "head" else self._wrist_snapshots
        snapshots.append((view_id, rgb, [(object_id, detection.bbox)
                          for object_id, detection in zip(object_ids, detections)]))
        self._view_diagnostics.append({
            "view_id": view_id, "camera": camera, "yaw": yaw, **getattr(detector, "last_diagnostics", {}),
            "detections": [{"object_id": object_id, "bbox": list(detection.bbox),
                            "class_scores": detection.class_scores, "point_count": len(detection.points_world),
                            "touches_image_border": (detection.bbox[0] <= 1 or detection.bbox[1] <= 1
                                                     or detection.bbox[2] >= rgb.shape[1] - 2
                                                     or detection.bbox[3] >= rgb.shape[0] - 2),
                            "height_range": np.percentile(detection.points_world[:, 2], (2, 98)).tolist()}
                           for object_id, detection in zip(object_ids, detections)],
        })
        if self.record_rgbd:
            frame = {"view_id": view_id, "camera": camera, "rgb": rgb, "depth": depth.copy(),
                                     "intrinsics": intrinsics.copy(),
                                     "base_position": np.array(base_position, dtype=float),
                                     "base_quaternion": as_numpy(base_quaternion).astype(float).reshape(4)}
            if camera == "ee":
                frame["joint_positions"] = as_numpy(joint_positions).astype(float).reshape(-1).copy()
                frame["camera_transform"] = mount.copy()
            self._rgbd_frames.append(frame)

    def finish_scan(self):
        """一周采集完成后检查证据；不按期望数量截断或强行分配类别。"""
        estimates = self.objects
        counts = {name: sum(item.confirmed and item.class_name == name for item in estimates) for name in CLASS_NAMES}
        reliable_tracks = sum(item.views >= 2 for item in estimates)
        complete = all(count == 6 for count in counts.values()) and reliable_tracks == 18
        self.status = "complete" if complete else "incomplete"
        self.blocked_reason = None if complete else (
            "Scan finished without 18 unambiguous items (six per class); inspect the report or rescan.")

    def report(self):
        estimates = self.objects
        counts = {name: sum(item.confirmed and item.class_name == name for item in estimates) for name in CLASS_NAMES}
        return {
            "status": self.status, "ready": self.ready, "reason": self.blocked_reason,
            "expected_count": 18, "confirmed_count": sum(item.confirmed for item in estimates),
            "candidate_count": len(estimates), "stable_candidate_count": sum(item.views >= 2 for item in estimates),
            "counts_by_class": counts, "missing_by_class": {name: max(0, 6 - count) for name, count in counts.items()},
            "excess_by_class": {name: max(0, count - 6) for name, count in counts.items()},
            "total_yaw_radians": self.total_yaw, "captured_views": len(self._snapshots),
            "captured_turn_radians": self.captured_turn_radians,
            "captured_views_by_camera": {"head": len(self._snapshots), "ee": len(self._wrist_snapshots)},
            "wrist_camera_enabled": bool(self._wrist_enabled),
            "scan_arm_target": list(SCAN_ARM_TARGET) if self._wrist_enabled else None,
            "coverage_note": ("Near-field wrist RGB-D included; occlusion may still hide items." if self._wrist_snapshots
                              else "Head-only data: nearby floor and occluded items may be outside the image."),
            "repeated_frames": self.repeated_frames,
            "position_reference": "estimated visible-surface bounding-box center in world coordinates, meters",
            "orientation_reference": "horizontal principal axis in radians modulo pi; roll/pitch and CAD orientation unknown",
            "detector": type(self.detector).__name__,
            "detectors_by_camera": {"head": type(self.detector).__name__, "ee": type(self.wrist_detector).__name__},
            "objects": [asdict(item) for item in estimates],
        }

    def save_results(self, directory):
        """保存 JSON 清单、带 ID/类别框的 RGB 视图，以及 XY 物品地图。"""
        from PIL import Image, ImageDraw

        directory = Path(directory)
        directory.mkdir(parents=True, exist_ok=True)
        (directory / "objects.json").write_text(json.dumps(self.report(), indent=2, ensure_ascii=False), encoding="utf-8")
        (directory / "detection_log.json").write_text(json.dumps(self._view_diagnostics, indent=2), encoding="utf-8")
        if self._rgbd_frames:
            frame_directory = directory / "rgbd"
            frame_directory.mkdir(exist_ok=True)
            for frame in self._rgbd_frames:
                prefix = "" if frame["camera"] == "head" else "ee_"
                np.savez_compressed(frame_directory / f"{prefix}frame_{frame['view_id']:02d}.npz", **frame)
        estimates = {item.object_id: item for item in self.objects}
        palette = {"sugar_box": "cyan", "mustard_bottle": "orange", "banana": "yellow", "unknown": "red"}
        for camera, (sector, rgb, boxes) in self._camera_snapshots():
            frame = Image.fromarray(rgb)
            draw = ImageDraw.Draw(frame)
            for object_id, bbox in boxes:
                item = estimates[object_id]
                color = palette[item.class_name]
                draw.rectangle(bbox, outline=color, width=2)
                draw.text((bbox[0], max(0, bbox[1] - 12)), f"#{object_id} {item.class_name}", fill=color, stroke_width=1, stroke_fill="black")
            prefix = "" if camera == "head" else "ee_"
            frame.save(directory / f"{prefix}view_{sector:02d}.png")
        canvas = Image.new("RGB", (800, 840), "white")
        draw = ImageDraw.Draw(canvas)
        draw.text((20, 10), f"{self.status}: {sum(item.confirmed for item in estimates.values())}/18 confirmed", fill="black")
        draw.text((20, 26), "World XY: -20..0 m; arrows are ambiguous principal axes, not grasp poses", fill="black")
        for world in range(-20, 1, 2):
            pixel = 40 + (world + 20) * 36
            draw.line((pixel, 80, pixel, 800), fill="#dddddd")
            draw.line((40, 80 + (0 - world) * 36, 760, 80 + (0 - world) * 36), fill="#dddddd")
        for item in estimates.values():
            x = 40 + (item.position[0] + 20) * 36
            y = 80 - item.position[1] * 36
            color = palette[item.class_name]
            draw.ellipse((x - 5, y - 5, x + 5, y + 5), fill=color, outline="black")
            draw.text((x + 7, y - 6), f"{item.object_id}: {item.class_name}", fill="black")
            if item.yaw_axis is not None:
                dx, dy = 12 * math.cos(item.yaw_axis), -12 * math.sin(item.yaw_axis)
                draw.line((x - dx, y - dy, x + dx, y + dy), fill="black", width=2)
        canvas.save(directory / "object_map.png")
        return directory / "objects.json"

    def _camera_snapshots(self):
        for camera, snapshots in (("head", self._snapshots), ("ee", self._wrist_snapshots)):
            for snapshot in snapshots:
                yield camera, snapshot

    def save_object_images(self, directory, *, confirmed_only=False):
        """每个扫描 ID 保存一张最大可见裁剪图，文件名包含类别、分数及位置。"""
        from PIL import Image, ImageDraw

        directory = Path(directory)
        directory.mkdir(parents=True, exist_ok=True)
        best = {}
        for camera, (sector, rgb, boxes) in self._camera_snapshots():
            height, width = rgb.shape[:2]
            for object_id, bbox in boxes:
                left, top, right, bottom = (int(value) for value in bbox)
                left, top = max(0, left), max(0, top)
                right, bottom = min(width, right), min(height, bottom)
                area = (right - left) * (bottom - top)
                if right <= left or bottom <= top:
                    continue
                if object_id not in best or area > best[object_id][0]:
                    best[object_id] = (area, sector, rgb, (left, top, right, bottom), camera)
        records = []
        for item in self.objects:
            if confirmed_only and not item.confirmed:
                continue
            if item.object_id not in best:
                continue
            _, sector, rgb, bbox, camera = best[item.object_id]
            x, y, z = item.position
            filename = (f"item_{item.object_id:02d}_{item.class_name}_conf{item.confidence:.2f}"
                        f"_x{x:.2f}_y{y:.2f}_z{z:.2f}.png")
            crop = Image.fromarray(rgb).crop(bbox)
            # 放大较小物品便于检查，但不声称提高原始识别分辨率。
            scale = min(4.0, max(1.0, 160.0 / max(crop.size)))
            crop = crop.resize((max(1, round(crop.width * scale)), max(1, round(crop.height * scale))))
            canvas = Image.new("RGB", (max(280, crop.width), crop.height + 44), "white")
            canvas.paste(crop, ((canvas.width - crop.width) // 2, 0))
            draw = ImageDraw.Draw(canvas)
            draw.text((5, crop.height + 4), f"#{item.object_id} {item.class_name} score={item.confidence:.2f}", fill="black")
            draw.text((5, crop.height + 20), f"XYZ=({x:.2f}, {y:.2f}, {z:.2f}) m", fill="black")
            canvas.save(directory / filename)
            records.append({**asdict(item), "image": filename, "source_view": sector, "source_bbox": bbox,
                            "source_camera": camera})
        (directory / "index.json").write_text(json.dumps(records, ensure_ascii=False, indent=2), encoding="utf-8")
        return records
