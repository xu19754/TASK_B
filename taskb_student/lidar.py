"""PhysX scene lidar: raycasts against colliders, including moving objects."""

from typing import Callable

import numpy as np
import torch
import isaaclab.sim as sim_utils
from isaaclab.sim.spawners.spawner_cfg import SpawnerCfg
from isaaclab.utils import configclass

AZIMUTH_BINS = 360
ELEVATION_BINS = 16
MIN_RANGE = 0.15
MAX_RANGE = 12.0


@sim_utils.clone
def spawn_lidar(prim_path, cfg, translation=None, orientation=None, **kwargs):
    from isaacsim.core.utils.extensions import enable_extension

    enable_extension("isaacsim.sensors.physx")
    import omni.isaac.RangeSensorSchema as RangeSensorSchema
    import omni.usd
    from pxr import Gf, UsdGeom

    # Publish CPU PhysX poses to USD for the attached lidar.
    context = sim_utils.SimulationContext.instance()
    context.get_physics_context().enable_fabric(False)
    # Define the schema directly: Isaac Lab can disable Kit's nested undo commands.
    lidar = RangeSensorSchema.Lidar.Define(omni.usd.get_context().get_stage(), prim_path)
    if not lidar:
        from pxr import Usd

        prim = lidar.GetPrim()
        raise RuntimeError(
            f"Could not create PhysX lidar at {prim_path}: prim={prim!r}, "
            f"type={prim.GetTypeName() if prim else None}, "
            f"schema={Usd.SchemaRegistry.GetTypeFromName('RangeSensorLidar')}."
        )
    lidar.CreateEnabledAttr(True)
    lidar.CreateMinRangeAttr(MIN_RANGE)
    lidar.CreateMaxRangeAttr(MAX_RANGE)
    lidar.CreateHorizontalFovAttr(360.0)
    lidar.CreateVerticalFovAttr(3.75 * (ELEVATION_BINS - 1))
    lidar.CreateHorizontalResolutionAttr(1.0)
    lidar.CreateVerticalResolutionAttr(3.75)
    lidar.CreateRotationRateAttr(0.0)
    lidar.CreateHighLodAttr(True)
    lidar.CreateYawOffsetAttr(0.0)
    lidar.CreateDrawPointsAttr(False)
    lidar.CreateDrawLinesAttr(False)
    lidar.CreateEnableSemanticsAttr(False)
    prim = lidar.GetPrim()
    xform = UsdGeom.Xformable(prim)
    xform.AddTranslateOp().Set(Gf.Vec3d(*(translation or (0.0, 0.0, 0.0))))
    if orientation is not None:
        xform.AddOrientOp().Set(Gf.Quatf(float(orientation[0]), Gf.Vec3f(*orientation[1:])))
    return prim


@configclass
class LidarCfg(SpawnerCfg):
    func: Callable = spawn_lidar


def _scan(env):
    if str(env.device) != "cpu":
        raise RuntimeError("Use CPU PhysX for the student lidar pose readback; RGB-D still renders on the GPU.")
    # Observation terms share one copied scan per control step, not four SDK reads.
    key = env.common_step_counter
    cached = getattr(env, "_student_lidar_scan", None)
    if cached is not None and cached[0] == key:
        return cached[1]
    from omni.physx import get_physx_interface
    from isaacsim.sensors.physx import _range_sensor

    # Lab's manual physics stepping does not publish USD transforms automatically.
    get_physx_interface().update_transformations(False, True)
    if env.sim.render_mode == env.sim.RenderMode.NO_GUI_OR_RENDERING:
        import omni.kit.app

        # Even without images, Kit must publish transforms and update sensor buffers.
        setting = "/app/player/playSimulations"
        previous = env.sim.get_setting(setting)
        env.sim.set_setting(setting, False)
        try:
            omni.kit.app.get_app().update()
        finally:
            env.sim.set_setting(setting, previous)
    if not hasattr(env, "_student_lidar_interface"):
        env._student_lidar_interface = _range_sensor.acquire_lidar_sensor_interface()
    interface = env._student_lidar_interface
    paths = sim_utils.find_matching_prim_paths(env.cfg.scene.lidar.prim_path)
    if len(paths) != 1 or env.num_envs != 1:
        raise RuntimeError("The student lidar arena supports exactly one sensor/environment.")
    path = paths[0]
    ranges = np.array(interface.get_linear_depth_data(path), dtype=np.float32, copy=True)
    points = np.array(interface.get_point_cloud_data(path), dtype=np.float32, copy=True)
    intensity = np.array(interface.get_intensity_data(path), copy=True)
    if getattr(env, "_student_debug_sensors", False) and key == 50:
        import omni.usd
        from pxr import UsdGeom

        stage = omni.usd.get_context().get_stage()
        position = UsdGeom.XformCache().GetLocalToWorldTransform(stage.GetPrimAtPath(path)).ExtractTranslation()
        print(f"[Sensors] lidar_world={tuple(position)} "
              f"base_world={env.scene['robot'].data.root_pos_w[0].tolist()} "
              f"raw_range={float(ranges.min()):.3f}..{float(ranges.max()):.3f} "
              f"intensity={np.unique(intensity).tolist()}", flush=True)
    shape = (1, AZIMUTH_BINS, ELEVATION_BINS)
    count = AZIMUTH_BINS * ELEVATION_BINS
    ready = ranges.size != 0 and points.size != 0 and intensity.size != 0
    if ready:
        if ranges.size != count or points.size != 3 * count or intensity.size != count:
            raise RuntimeError(f"Unexpected lidar buffers: {ranges.shape}, {points.shape}, {intensity.shape}.")
        ranges = torch.tensor(ranges.reshape(shape), device=env.device)
        points = torch.tensor(points.reshape(*shape, 3), device=env.device)
        hit = torch.tensor(intensity.reshape(shape) > 0, device=env.device)
        hit &= torch.isfinite(ranges) & (ranges >= MIN_RANGE) & (ranges < MAX_RANGE)
        hit &= torch.isfinite(points).all(dim=-1)
    else:
        ranges = torch.full(shape, MAX_RANGE, device=env.device)
        points = torch.zeros((*shape, 3), device=env.device)
        hit = torch.zeros(shape, dtype=torch.bool, device=env.device)
    scan = {
        "ranges": torch.where(hit, ranges, MAX_RANGE),
        "points": torch.where(hit.unsqueeze(-1), points, 0.0),
        "hit_mask": hit,
        "ready": torch.tensor([[ready]], dtype=torch.bool, device=env.device),
    }
    env._student_lidar_scan = (key, scan)
    return scan


def lidar_observation(env, data_type):
    return _scan(env)[data_type]
