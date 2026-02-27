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

from VRP.config import PROJECT_ROOT


# ── Identity quaternion helper ────────────────────────────────────────────────

def _with_identity_quat(xyz: np.ndarray) -> List[List[float]]:
    """Append identity quaternion [qw=1, qx=0, qy=0, qz=0] to (N,3) positions."""
    N = len(xyz)
    quat = np.tile([1.0, 0.0, 0.0, 0.0], (N, 1))
    return np.concatenate([xyz, quat], axis=1).tolist()


# ── Source 1: Random free-space sampling ─────────────────────────────────────

def load_random_waypoints(
    n: int,
    og,  # OccupancyGrid
    seed: Optional[int] = 42,
    z_min: float = 0.3,
    z_max: Optional[float] = None,
) -> List[List[float]]:
    """Sample ``n`` random free-space waypoints from the occupancy grid.

    Parameters
    ----------
    n       : number of waypoints.
    og      : OccupancyGrid instance.
    seed    : random seed for reproducibility.
    z_min   : minimum Z height (metres); points below are rejected.
    z_max   : maximum Z height; ``None`` = no upper bound.
    """
    rng = np.random.RandomState(seed)

    # Gather free voxels that satisfy Z constraints
    free_ijk = np.argwhere(~og.grid)                          # (F, 3)
    centres  = og.voxel_to_world(free_ijk)                   # (F, 3)

    z_mask = centres[:, 2] >= z_min
    if z_max is not None:
        z_mask &= centres[:, 2] <= z_max
    centres = centres[z_mask]

    if len(centres) < n:
        raise ValueError(
            f"Only {len(centres)} voxels satisfy Z constraints; requested {n}."
        )

    chosen = centres[rng.choice(len(centres), n, replace=False)]
    # Add sub-voxel jitter
    jitter = rng.uniform(-og.resolution * 0.4, og.resolution * 0.4, size=(n, 3))
    chosen = chosen + jitter
    return _with_identity_quat(chosen)


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
