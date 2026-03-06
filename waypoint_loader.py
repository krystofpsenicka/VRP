"""
VRP Planner – Waypoint Loader

Supports loading waypoints from multiple sources:
  1. Random free-space sampling (via OccupancyGrid)
  2. JSON / NPZ manual specification
  3. 3D-Inspection ViewpointResult objects (from methods_analysis/utils.py)
  4. CSV file (x,y,z per row)

All loaders return a list of ``[x, y, z, qw, qx, qy, qz]`` waypoints
(identity quaternion when orientation is not specified) suitable for
both VRP solving and cuRobo trajectory planning.
"""

from __future__ import annotations

import json
import os
import sys
from typing import List, Optional

import numpy as np

from VRP.config import (
    MESH_PATH,
    MESH_POSE,
    MESH_TARGET_LENGTH,
    PROJECT_ROOT,
    ROBOT_RADIUS,
)


# ── Mesh proximity filter (cached) ───────────────────────────────────────────

_SCALED_MESH = None   # trimesh.Trimesh, lazily loaded once


def _get_scaled_mesh():
    """Return the ship mesh with the same scale + pose as the occupancy grid.

    The result is cached so subsequent calls are free.
    """
    global _SCALED_MESH
    if _SCALED_MESH is not None:
        return _SCALED_MESH

    import trimesh
    from scipy.spatial.transform import Rotation as R

    raw = trimesh.load(MESH_PATH, force="mesh")
    if isinstance(raw, trimesh.Scene):
        raw = trimesh.util.concatenate(list(raw.geometry.values()))

    longest = float(raw.extents.max())
    if longest > 0:
        raw.apply_scale(MESH_TARGET_LENGTH / longest)

    T = np.eye(4)
    T[:3, 3] = MESH_POSE[:3]
    qw, qx, qy, qz = MESH_POSE[3:7]
    T[:3, :3] = R.from_quat([qx, qy, qz, qw]).as_matrix()
    raw.apply_transform(T)

    _SCALED_MESH = raw
    return _SCALED_MESH


# ── Identity quaternion helper ────────────────────────────────────────────────

def _with_identity_quat(xyz: np.ndarray) -> List[List[float]]:
    """Append identity quaternion [qw=1, qx=0, qy=0, qz=0] to (N,3) positions."""
    N = len(xyz)
    quat = np.tile([1.0, 0.0, 0.0, 0.0], (N, 1))
    return np.concatenate([xyz, quat], axis=1).tolist()


def _with_random_quats(
    xyz: np.ndarray,
    rng: np.random.RandomState,
    yaw_range: tuple = (0.0, 2 * np.pi),
    pitch_range: tuple = (-np.pi / 4, np.pi / 4),
) -> List[List[float]]:
    """Append random yaw + pitch quaternions to an (N, 3) position array.

    Orientation is a ZY Euler rotation: yaw about world-Z followed by pitch
    about body-Y.  The resulting quaternion is [qw, qx, qy, qz].

    Parameters
    ----------
    yaw_range   : uniform random range for yaw (radians).
    pitch_range : uniform random range for pitch/elevation (radians).
    """
    N = len(xyz)
    thetas = rng.uniform(*yaw_range, size=N)    # yaw angles
    phis   = rng.uniform(*pitch_range, size=N)  # pitch/elevation angles

    # Half-angles
    ct = np.cos(thetas / 2); st = np.sin(thetas / 2)
    cp = np.cos(phis   / 2); sp = np.sin(phis   / 2)

    # Compound quaternion: q_yaw * q_pitch
    #   q_yaw   = (ct, 0,  0,  st)
    #   q_pitch = (cp, 0, sp,   0)
    # Product → (ct*cp, -st*sp, ct*sp, st*cp)
    qw = ct * cp
    qx = -st * sp
    qy =  ct * sp
    qz =  st * cp

    quats = np.column_stack([qw, qx, qy, qz])
    return np.concatenate([xyz, quats], axis=1).tolist()


# ── Source 1: Random free-space sampling ─────────────────────────────────────

def load_random_waypoints(
    n: int,
    og,  # OccupancyGrid
    seed: Optional[int] = 42,
    z_min: float = 0.5,
    z_max: Optional[float] = None,
    x_min: Optional[float] = None,
    x_max: Optional[float] = None,
    y_min: Optional[float] = None,
    y_max: Optional[float] = None,
    mesh_clearance: Optional[float] = None,
    max_attempts: int = 50_000,
) -> List[List[float]]:
    """Sample ``n`` random free-space waypoints via rejection sampling.

    Uses fast rejection sampling instead of enumerating every free voxel,
    which is critical for large grids (>10 M voxels).

    Parameters
    ----------
    n       : number of waypoints.
    og      : OccupancyGrid instance.
    seed    : random seed for reproducibility.
    z_min   : minimum Z height (metres); points below are rejected.
    z_max   : maximum Z height; ``None`` = derived from grid bounds.
    x_min, x_max : X range; ``None`` = derived from grid bounds.
    y_min, y_max : Y range; ``None`` = derived from grid bounds.
    mesh_clearance : minimum distance to the ship mesh surface (metres).
        Defaults to ``ROBOT_RADIUS * 2``.  Set to 0 to disable.
    max_attempts : total random samples before giving up.
    """
    rng = np.random.RandomState(seed)

    # Derive default bounds from the occupancy grid extents
    grid_min = og.origin
    grid_max = og.origin + np.array(og.grid.shape) * og.resolution
    if x_min is None:
        x_min = float(grid_min[0])
    if x_max is None:
        x_max = float(grid_max[0])
    if y_min is None:
        y_min = float(grid_min[1])
    if y_max is None:
        y_max = float(grid_max[1])
    if z_max is None:
        z_max = float(grid_max[2])
    if mesh_clearance is None:
        mesh_clearance = ROBOT_RADIUS * 2.0

    print(f"[WaypointLoader] Sampling {n} waypoints  "
          f"x=[{x_min:.1f},{x_max:.1f}] y=[{y_min:.1f},{y_max:.1f}] "
          f"z=[{z_min:.1f},{z_max:.1f}]  clearance={mesh_clearance:.2f}m")

    # Lazy-load mesh proximity checker only when needed
    prox = None
    if mesh_clearance > 0 and os.path.isfile(MESH_PATH):
        import trimesh
        mesh = _get_scaled_mesh()
        prox = trimesh.proximity.ProximityQuery(mesh)

    # ── Rejection sampling (fast for large grids) ─────────────────────
    grid_shape = np.array(og.grid.shape)
    collected: List[np.ndarray] = []
    batch_size = max(n * 50, 500)
    total_tried = 0

    while len(collected) < n and total_tried < max_attempts:
        # Generate random world-frame points within the sampling box
        pts = np.column_stack([
            rng.uniform(x_min, x_max, batch_size),
            rng.uniform(y_min, y_max, batch_size),
            rng.uniform(z_min, z_max, batch_size),
        ])
        total_tried += batch_size

        # Convert to voxel indices and check occupancy grid
        ijk = og.world_to_voxel(pts)
        in_bounds = np.all(ijk >= 0, axis=1) & np.all(ijk < grid_shape, axis=1)
        pts = pts[in_bounds]
        ijk = ijk[in_bounds]
        free_mask = ~og.grid[ijk[:, 0], ijk[:, 1], ijk[:, 2]]
        pts = pts[free_mask]

        # Mesh proximity filter (only on the few surviving candidates)
        if prox is not None and len(pts) > 0:
            _, dists, _ = prox.on_surface(pts)
            pts = pts[dists >= mesh_clearance]

        for p in pts:
            if len(collected) >= n:
                break
            collected.append(p)

    if len(collected) < n:
        raise ValueError(
            f"Only {len(collected)} valid points found after {total_tried} "
            f"attempts; requested {n}.  Try relaxing the bounds or "
            f"reducing mesh_clearance."
        )

    chosen = np.array(collected[:n])
    print(f"[WaypointLoader] Sampled {n} waypoints in {total_tried} attempts")
    return _with_random_quats(chosen, rng)


# ── Source 2: JSON / NPZ file ─────────────────────────────────────────────────

def load_waypoints_from_json(path: str) -> List[List[float]]:
    """Load waypoints from a JSON file.

    Expected format (either list of lists or dict with 'waypoints' key):
    ::

        [
            [x, y, z],              // identity quaternion assumed
            [x, y, z, qw, qx, qy, qz],
            ...
        ]
    """
    with open(path) as f:
        data = json.load(f)
    if isinstance(data, dict):
        data = data["waypoints"]
    waypoints = []
    for item in data:
        item = list(item)
        if len(item) == 3:
            item += [1.0, 0.0, 0.0, 0.0]
        assert len(item) == 7, f"Expected 3 or 7 values per waypoint, got {len(item)}"
        waypoints.append([float(v) for v in item])
    return waypoints


def load_waypoints_from_npz(path: str) -> List[List[float]]:
    """Load waypoints from an NPZ file.

    Expects either a 'waypoints' key (shape N×7 or N×3) or a 'positions' key
    (shape N×3, identity quaternion assumed).
    """
    data = np.load(path, allow_pickle=True)
    if "waypoints" in data:
        arr = data["waypoints"]
    elif "positions" in data:
        arr = data["positions"]
    else:
        raise KeyError(f"NPZ file {path} has no 'waypoints' or 'positions' key.")
    arr = np.asarray(arr, dtype=np.float32)
    if arr.shape[1] == 3:
        return _with_identity_quat(arr)
    return arr.tolist()


def load_waypoints_from_csv(path: str) -> List[List[float]]:
    """Load waypoints from a CSV file (x,y,z per row; optional qw,qx,qy,qz columns)."""
    arr = np.loadtxt(path, delimiter=",", dtype=np.float32)
    if arr.ndim == 1:
        arr = arr.reshape(1, -1)
    if arr.shape[1] == 3:
        return _with_identity_quat(arr)
    if arr.shape[1] == 7:
        return arr.tolist()
    raise ValueError(f"CSV must have 3 or 7 columns; got {arr.shape[1]}.")


# ── Source 3: 3D-Inspection ViewpointResult ───────────────────────────────────

def load_waypoints_from_inspection(
    models_dir: str,
    result_file: Optional[str] = None,
) -> List[List[float]]:
    """Load waypoints from 3D-Inspection optimisation results.

    Locates ViewpointResult objects produced by the methods_analysis pipeline
    and extracts their ``position`` and ``direction`` attributes to build
    7-DOF waypoints (position + orientation derived from view direction).

    Parameters
    ----------
    models_dir
        Path to the ``3D-Inspection/methods_analysis/models/`` directory.
    result_file
        Path to a specific NPZ result file.  When ``None``, scans
        ``models_dir`` for ``*.npz`` result files.
    """
    # Add 3D-Inspection to sys.path so we can import utils.py
    inspection_root = os.path.join(PROJECT_ROOT, "3D-Inspection", "methods_analysis")
    if inspection_root not in sys.path:
        sys.path.insert(0, inspection_root)

    try:
        from utils import ViewpointResult, OptimizationResult  # type: ignore
    except ImportError as e:
        raise ImportError(
            f"Cannot import 3D-Inspection utils from {inspection_root}: {e}"
        )

    # Locate result files
    if result_file:
        result_files = [result_file]
    else:
        result_files = []
        for fname in os.listdir(models_dir):
            if fname.endswith(".npz") or fname.endswith(".pkl"):
                result_files.append(os.path.join(models_dir, fname))

    if not result_files:
        raise FileNotFoundError(
            f"No NPZ/PKL result files found in {models_dir}. "
            "Run 3D-Inspection optimisation first."
        )

    all_viewpoints: List[ViewpointResult] = []
    for rpath in result_files:
        try:
            data = np.load(rpath, allow_pickle=True)
            if "viewpoints" in data:
                vps = data["viewpoints"].tolist()
                all_viewpoints.extend(vps)
            elif "result" in data:
                result_obj = data["result"].item()
                if hasattr(result_obj, "viewpoints"):
                    all_viewpoints.extend(result_obj.viewpoints)
        except Exception as e:
            print(f"[WaypointLoader] Warning: could not load {rpath}: {e}")

    if not all_viewpoints:
        raise ValueError("No ViewpointResult objects found in the result files.")

    waypoints = []
    for vp in all_viewpoints:
        pos = np.asarray(vp.position, dtype=np.float32)
        # Derive quaternion from view direction if available
        quat = _direction_to_quat(np.asarray(vp.direction, dtype=np.float32))
        waypoints.append([
            float(pos[0]), float(pos[1]), float(pos[2]),
            float(quat[0]), float(quat[1]), float(quat[2]), float(quat[3]),
        ])

    print(f"[WaypointLoader] Loaded {len(waypoints)} viewpoints from "
          f"{len(result_files)} file(s).")
    return waypoints


def _direction_to_quat(direction: np.ndarray) -> np.ndarray:
    """Compute a quaternion that rotates the +X axis onto ``direction``.

    Returns identity quaternion if ``direction`` is zero.
    """
    d = direction.astype(np.float64)
    norm = np.linalg.norm(d)
    if norm < 1e-6:
        return np.array([1.0, 0.0, 0.0, 0.0])
    d /= norm
    x_axis = np.array([1.0, 0.0, 0.0])
    cross  = np.cross(x_axis, d)
    cross_norm = np.linalg.norm(cross)
    dot        = np.dot(x_axis, d)
    if cross_norm < 1e-6:
        if dot > 0:
            return np.array([1.0, 0.0, 0.0, 0.0])
        else:
            # 180-degree rotation around Z
            return np.array([0.0, 0.0, 0.0, 1.0])
    angle = np.arctan2(cross_norm, dot)
    axis  = cross / cross_norm
    s     = np.sin(angle / 2.0)
    return np.array([
        np.cos(angle / 2.0),
        axis[0] * s,
        axis[1] * s,
        axis[2] * s,
    ])


# ── Universal loader ──────────────────────────────────────────────────────────

def load_waypoints(
    source: str,
    n_random: Optional[int] = None,
    og=None,
    random_seed: int = 42,
) -> List[List[float]]:
    """Universal dispatcher for waypoint loading.

    Parameters
    ----------
    source
        One of:
        * ``"random"`` – requires ``n_random`` and ``og``.
        * path ending in ``.json`` – loaded as JSON.
        * path ending in ``.npz``  – loaded as NPZ.
        * path ending in ``.csv``  – loaded as CSV.
        * path to a directory      – treated as 3D-Inspection models dir.
    n_random
        Number of random waypoints (only used when ``source == "random"``).
    og
        OccupancyGrid instance (only used when ``source == "random"``).
    random_seed
        Seed for random sampling.
    """
    if source == "random":
        if og is None or n_random is None:
            raise ValueError("'random' source requires og and n_random.")
        return load_random_waypoints(n_random, og, seed=random_seed)
    if source.endswith(".json"):
        return load_waypoints_from_json(source)
    if source.endswith(".npz"):
        return load_waypoints_from_npz(source)
    if source.endswith(".csv"):
        return load_waypoints_from_csv(source)
    if os.path.isdir(source):
        return load_waypoints_from_inspection(source)
    raise ValueError(
        f"Cannot determine waypoint source type from: '{source}'. "
        "Use 'random', a .json/.npz/.csv file, or a directory."
    )
