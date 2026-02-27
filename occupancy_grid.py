"""
VRP Planner – 3D Occupancy Grid

Builds an inflated voxel occupancy grid from the environment mesh
(Duke of Lancaster shipwreck + static cuboid obstacles) using trimesh.

The grid is the central data structure shared by:
  • gpu_distance_matrix.py  (cuGraph / A* cost-matrix input)
  • traffic_light.py        (path corridor overlap checks)
  • trajectory_planner.py   (OMPL RRT* obstacle input)

All coordinates below use the **world** frame (metres).
"""

from __future__ import annotations

import os
import pickle
from dataclasses import dataclass, field
from pathlib import Path
from typing import List, Optional, Tuple

import numpy as np

from VRP.config import (
    INFLATION_VOXELS,
    MESH_PATH,
    ROBOT_RADIUS,
    STATIC_OBSTACLES,
    VOXEL_RESOLUTION,
)


@dataclass
class OccupancyGrid:
    """3D binary occupancy grid in voxel coordinates.

    Attributes
    ----------
    grid : np.ndarray, dtype=bool, shape (Nx, Ny, Nz)
        ``True`` = occupied / collision, ``False`` = free.
    origin : np.ndarray (3,)
        World position of voxel (0, 0, 0).
    resolution : float
        Metres per voxel edge.
    """

    grid: np.ndarray
    origin: np.ndarray
    resolution: float = VOXEL_RESOLUTION

    # ── Coordinate transforms ─────────────────────────────────────────────────

    def world_to_voxel(self, world_xyz: np.ndarray) -> np.ndarray:
        """Convert world-frame ``(..., 3)`` coords → integer voxel indices.

        Returned indices are *not* clipped; callers should use
        ``is_valid_voxel`` before indexing into the grid.
        """
        return np.floor(
            (world_xyz - self.origin) / self.resolution
        ).astype(int)

    def voxel_to_world(self, voxel_ijk: np.ndarray) -> np.ndarray:
        """Convert integer voxel indices ``(..., 3)`` → world-frame centre (m)."""
        return voxel_ijk.astype(float) * self.resolution + self.origin + self.resolution * 0.5

    def is_valid_voxel(self, ijk: np.ndarray) -> bool:
        """Return True if ``ijk`` is inside the grid bounds."""
        ijk = np.asarray(ijk)
        return bool(
            np.all(ijk >= 0) and np.all(ijk < np.array(self.grid.shape))
        )

    def is_free_world(self, world_xyz: np.ndarray) -> bool:
        """Return True if the world-frame point is in a free voxel."""
        ijk = self.world_to_voxel(world_xyz)
        if not self.is_valid_voxel(ijk):
            return False
        return not bool(self.grid[tuple(ijk)])

    def world_to_flat_index(self, world_xyz: np.ndarray) -> int:
        """Return the flat (C-order) grid index for a world-frame point."""
        ijk = self.world_to_voxel(world_xyz)
        return int(np.ravel_multi_index(tuple(ijk), self.grid.shape))

    def flat_index_to_world(self, flat_idx: int) -> np.ndarray:
        """Inverse of ``world_to_flat_index``."""
        ijk = np.array(np.unravel_index(flat_idx, self.grid.shape))
        return self.voxel_to_world(ijk)

    @property
    def shape(self) -> Tuple[int, int, int]:
        return tuple(self.grid.shape)  # type: ignore[return-value]

    @property
    def num_free(self) -> int:
        return int(np.sum(~self.grid))

    @property
    def num_occupied(self) -> int:
        return int(np.sum(self.grid))

    # ── Sampling ──────────────────────────────────────────────────────────────

    def sample_random_free_points(
        self, n: int, rng: Optional[np.random.RandomState] = None
    ) -> np.ndarray:
        """Return ``(n, 3)`` random world-frame points inside free voxels."""
        if rng is None:
            rng = np.random.RandomState()
        free_ijk = np.argwhere(~self.grid)           # (F, 3)
        if len(free_ijk) < n:
            raise ValueError(
                f"Grid has only {len(free_ijk)} free voxels; requested {n}."
            )
        chosen = free_ijk[rng.choice(len(free_ijk), n, replace=False)]
        # Random sub-voxel offset for variety
        offsets = rng.uniform(0.0, self.resolution, size=(n, 3))
        return chosen.astype(float) * self.resolution + self.origin + offsets

    # ── Persistence ───────────────────────────────────────────────────────────

    def save(self, path: str) -> None:
        """Pickle the grid to disk for caching."""
        with open(path, "wb") as f:
            pickle.dump(self, f)

    @classmethod
    def load(cls, path: str) -> "OccupancyGrid":
        with open(path, "rb") as f:
            return pickle.load(f)


# ── Grid builder ─────────────────────────────────────────────────────────────

def _inflate_grid(grid: np.ndarray, inflation_voxels: int) -> np.ndarray:
    """Morphological dilation of the obstacle grid by *inflation_voxels* voxels.

    Uses scipy's binary_dilation which is equivalent to a 3D sphere structuring
    element of radius ``inflation_voxels``.  This expands every obstacle by
    the robot's collision radius so that path planners can treat the robot
    as a point.
    """
    from scipy.ndimage import binary_dilation
    # Build a spherical structuring element
    r = inflation_voxels
    d = 2 * r + 1
    se = np.zeros((d, d, d), dtype=bool)
    cx, cy, cz = r, r, r
    for ix in range(d):
        for iy in range(d):
            for iz in range(d):
                if (ix - cx) ** 2 + (iy - cy) ** 2 + (iz - cz) ** 2 <= r ** 2:
                    se[ix, iy, iz] = True
    return binary_dilation(grid, structure=se)


def _add_cuboid_obstacles(
    grid: np.ndarray,
    origin: np.ndarray,
    resolution: float,
    obstacles: dict,
) -> None:
    """Mark cuboid obstacle voxels in-place."""
    shape = np.array(grid.shape)
    for name, obs in obstacles.items():
        dims = np.asarray(obs["dims"], dtype=float)
        pose = obs["pose"]
        centre = np.array(pose[:3], dtype=float)
        half   = dims / 2.0
        min_w  = centre - half
        max_w  = centre + half
        ijk_min = np.floor((min_w - origin) / resolution).astype(int)
        ijk_max = np.ceil((max_w - origin) / resolution).astype(int)
        ijk_min = np.clip(ijk_min, 0, shape - 1)
        ijk_max = np.clip(ijk_max, 0, shape - 1)
        grid[
            ijk_min[0]: ijk_max[0] + 1,
            ijk_min[1]: ijk_max[1] + 1,
            ijk_min[2]: ijk_max[2] + 1,
        ] = True


def build_occupancy_grid(
    mesh_path: str = MESH_PATH,
    resolution: float = VOXEL_RESOLUTION,
    inflation_voxels: int = INFLATION_VOXELS,
    padding: float = 1.0,
    cache_path: Optional[str] = None,
    force_rebuild: bool = False,
) -> OccupancyGrid:
    """Build a 3D occupancy grid from the environment mesh + static obstacles.

    Pipeline (follows PDF spec):
    1. Load mesh with trimesh.
    2. Voxelize at ``resolution`` using trimesh's ``VoxelGrid``.
    3. Mark static cuboid obstacles (rocks, coral).
    4. Inflate the raw grid by ``inflation_voxels`` (robot radius / resolution).
    5. Wrap in an ``OccupancyGrid`` object.

    Parameters
    ----------
    mesh_path
        Path to the ``.glb`` / ``.obj`` / ``.stl`` environment mesh.
    resolution
        Voxel edge length in metres.
    inflation_voxels
        Dilation radius in voxels (should equal ceil(robot_radius / resolution)).
    padding
        Extra space (metres) added around the mesh bounding box.
    cache_path
        If given, load from disk when available; save after building.
    force_rebuild
        Ignore any cached grid and rebuild from scratch.

    Returns
    -------
    OccupancyGrid
    """
    import trimesh

    # ── Cache hit ─────────────────────────────────────────────────────────────
    if cache_path and not force_rebuild and os.path.exists(cache_path):
        print(f"[OccupancyGrid] Loading cached grid from {cache_path}")
        return OccupancyGrid.load(cache_path)

    print(f"[OccupancyGrid] Building grid from {mesh_path} "
          f"(res={resolution}m, inflation={inflation_voxels}vox) …")

    # ── 1. Load mesh ──────────────────────────────────────────────────────────
    if os.path.exists(mesh_path):
        scene_or_mesh = trimesh.load(mesh_path, force="mesh")
        if isinstance(scene_or_mesh, trimesh.Scene):
            # Merge all scene geometries into a single mesh
            mesh = trimesh.util.concatenate(
                [g for g in scene_or_mesh.geometry.values()]
            )
        else:
            mesh = scene_or_mesh
        print(f"[OccupancyGrid] Mesh loaded: {len(mesh.vertices)} verts, "
              f"{len(mesh.faces)} faces")
        mesh_available = True
    else:
        print(f"[OccupancyGrid] WARNING: mesh not found at {mesh_path}. "
              f"Using obstacle-only grid.")
        mesh_available = False

    # ── 2. Determine grid bounds ───────────────────────────────────────────────
    if mesh_available:
        bounds_min = mesh.bounds[0] - padding
        bounds_max = mesh.bounds[1] + padding
    else:
        # Fall back to a sensible default workspace
        bounds_min = np.array([-5.0, -5.0,  0.0])
        bounds_max = np.array([15.0, 15.0, 10.0])

    origin = bounds_min.copy()
    grid_shape = np.ceil((bounds_max - bounds_min) / resolution).astype(int)
    grid_shape = np.maximum(grid_shape, 1)      # guard against zero-size
    print(f"[OccupancyGrid] Grid shape: {tuple(grid_shape)}  "
          f"({np.prod(grid_shape)/1e6:.1f} M voxels)")

    raw_grid = np.zeros(grid_shape, dtype=bool)

    # ── 3. Voxelise mesh ──────────────────────────────────────────────────────
    if mesh_available:
        # trimesh voxel grid pitch = resolution
        vg = mesh.voxelized(pitch=resolution)
        # vg.matrix is a dense bool array; indices are relative to vg.origin
        vox_matrix = vg.matrix
        vox_origin_ijk = np.floor(
            (np.asarray(vg.origin) - origin) / resolution
        ).astype(int)
        # Copy into raw_grid at the correct offset
        dst_min = np.maximum(vox_origin_ijk, 0)
        src_min = np.maximum(-vox_origin_ijk, 0)
        dst_max = np.minimum(vox_origin_ijk + np.array(vox_matrix.shape), grid_shape)
        src_max = src_min + (dst_max - dst_min)
        raw_grid[
            dst_min[0]: dst_max[0],
            dst_min[1]: dst_max[1],
            dst_min[2]: dst_max[2],
        ] = vox_matrix[
            src_min[0]: src_max[0],
            src_min[1]: src_max[1],
            src_min[2]: src_max[2],
        ]
        print(f"[OccupancyGrid] Mesh voxels occupied: {int(raw_grid.sum())}")

    # ── 4. Mark static cuboid obstacles ───────────────────────────────────────
    _add_cuboid_obstacles(raw_grid, origin, resolution, STATIC_OBSTACLES)
    print(f"[OccupancyGrid] After cuboids: {int(raw_grid.sum())} occupied voxels")

    # ── 5. Inflate by robot radius ───────────────────────────────────────────
    if inflation_voxels > 0:
        inflated_grid = _inflate_grid(raw_grid, inflation_voxels)
    else:
        inflated_grid = raw_grid
    print(f"[OccupancyGrid] After inflation: {int(inflated_grid.sum())} occupied  "
          f"({int((~inflated_grid).sum())} free)")

    og = OccupancyGrid(grid=inflated_grid, origin=origin, resolution=resolution)

    # ── Cache ─────────────────────────────────────────────────────────────────
    if cache_path:
        os.makedirs(os.path.dirname(os.path.abspath(cache_path)), exist_ok=True)
        og.save(cache_path)
        print(f"[OccupancyGrid] Saved to {cache_path}")

    return og
