"""Existing sensor configuration shared by the scene, audit and recorder.

These are teaching simulation mounts, not measured hardware calibration.
No simulator imports: dataset inspection also works without Isaac.
"""

PROPRIO_SHAPES = {
    "base_lin_vel": (1, 3), "base_ang_vel": (1, 3),
    "projected_gravity": (1, 3), "base_quat": (1, 4),
    "joint_pos": (1, 19), "joint_vel": (1, 19),
}
LIDAR_SHAPES = {
    "ranges": (1, 360, 16), "points": (1, 360, 16, 3),
    "hit_mask": (1, 360, 16), "ready": (1, 1),
}
CAMERAS = {
    "head": {"parent": "base", "pos": (0.4, 0.0, 0.12)},
    "ee": {"parent": "link06", "pos": (0.03, 0.0, 0.07)},
}
CAMERA_COMMON = {
    "width": 640, "height": 480, "update_period": 0.1,
    "rot_wxyz": (1.0, 0.0, 0.0, 0.0), "convention": "world",
    "focal_length": 15.0, "horizontal_aperture": 20.955,
    "clipping_range": (0.05, 50.0),
    "depth_type": "distance_to_image_plane", "depth_units": "m",
}
IMAGE_SHAPES = {
    f"{camera}_{kind}": shape
    for camera in CAMERAS
    for kind, shape in (
        ("rgb", (1, 480, 640, 3)), ("depth", (1, 480, 640, 1)),
        ("intrinsics", (1, 3, 3)),
    )
}
OBSERVATION_SHAPES = {"proprio": PROPRIO_SHAPES, "lidar": LIDAR_SHAPES, "image": IMAGE_SHAPES}
