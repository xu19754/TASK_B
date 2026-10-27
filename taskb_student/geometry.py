"""Task B geometry, without simulator imports."""

import numpy as np
import trimesh

from .constants import NUM_OBJECTS, SPAWN_POSITION, TARGET_CENTER


def sample_object_positions(seed):
    rng = np.random.default_rng(seed)
    positions = []
    for _ in range(10000):
        xy = rng.uniform(-15.0, -5.0, size=2)
        if np.linalg.norm(xy - np.asarray(SPAWN_POSITION[:2])) < 0.85:
            continue
        if any(np.linalg.norm(xy - old[:2]) < 0.4 for old in positions):
            continue
        z = 0.15 if len(positions) < 6 else 0.10
        positions.append(np.array([xy[0], xy[1], z]))
        if len(positions) == NUM_OBJECTS:
            return np.asarray(positions)
    raise RuntimeError("Could not sample a non-overlapping Task B layout.")


def build_arena_meshes():
    ground = trimesh.creation.box(
        extents=(20.0, 20.0, 0.10),
        transform=trimesh.transformations.translation_matrix((-10.0, -10.0, -0.05)),
    )
    x, y = TARGET_CENTER
    bottom = trimesh.creation.cylinder(
        radius=1.0, height=0.05, sections=64,
        transform=trimesh.transformations.translation_matrix((x, y, 0.025)),
    )
    wall = trimesh.creation.annulus(
        r_min=0.98, r_max=1.0, height=0.5, sections=64,
        transform=trimesh.transformations.translation_matrix((x, y, 0.30)),
    )
    return [("ground", ground), ("bin_bottom", bottom), ("bin_wall", wall)]
