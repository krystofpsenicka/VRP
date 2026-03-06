"""
VRP Planner – Shared Utilities

Ported helpers from brov_auv_curobo/run_multi_auv_waypoints.py:
  • load_local_robot_config()
  • generate_random_waypoints()
  • build_world_config_for_robot()
  • resample_positions_to_horizon()
  • aabb_overlap()
  • find_trajectory_collisions()
"""

from __future__ import annotations
import os
import sys
from typing import List, Optional, Tuple

import numpy as np

from VRP.config import (
    ASSETS_PATH,
    BROV_CUBOID_DIMS,
    CONFIGS_PATH,
    ROBOT_CFG_DIR,
    STATIC_OBSTACLES,
)
# ---------------------------------------------------------------------------
# cuRobo imports (optional – utils should still be importable without them
# to allow waypoint / grid work without a full cuRobo GPU env)
# ---------------------------------------------------------------------------
try:
    from curobo.geom.types import VoxelGrid, WorldConfig
    from curobo.types.math import Pose
    from curobo.util_file import load_yaml
    _CUROBO_AVAILABLE = True
except ImportError:
    _CUROBO_AVAILABLE = False


# ── Robot config ─────────────────────────────────────────────────────────────

def load_local_robot_config(robot_file: str = "brov.yml") -> dict:
    """Load a robot YAML config from the brov_auv_curobo configs directory.

    Patches ``external_asset_path`` and ``collision_spheres`` to absolute
    paths so the returned dict is ready for ``MotionGenConfig.load_from_robot_config``.
    """
    if not _CUROBO_AVAILABLE:
        raise RuntimeError("cuRobo is not available in this environment.")
    config_path = os.path.join(ROBOT_CFG_DIR, robot_file)
    robot_cfg = load_yaml(config_path)["robot_cfg"]
    robot_cfg["kinematics"]["external_asset_path"] = ASSETS_PATH
    spheres_file = robot_cfg["kinematics"].get("collision_spheres", "spheres/brov.yml")
    robot_cfg["kinematics"]["collision_spheres"] = os.path.join(
        ROBOT_CFG_DIR, spheres_file
    )
    return robot_cfg


def get_joint_names(robot_file: str = "brov.yml") -> List[str]:
    """Return cspace joint names from a robot config YAML."""
    cfg = load_local_robot_config(robot_file)
    return cfg["kinematics"]["cspace"]["joint_names"]


def get_default_config(robot_file: str = "brov.yml") -> np.ndarray:
    """Return the retract (home) joint configuration."""
    cfg = load_local_robot_config(robot_file)
    return np.array(cfg["kinematics"]["cspace"]["retract_config"], dtype=np.float32)


# ── Waypoint generation ───────────────────────────────────────────────────────

def generate_random_waypoints(
    num_waypoints: int,
    x_range: Tuple[float, float] = (-2.0, 8.0),
    y_range: Tuple[float, float] = (-2.0, 8.0),
    z_range: Tuple[float, float] = (0.5, 3.0),
    seed: Optional[int] = None,
) -> List[List[float]]:
    """Generate random waypoints as ``[x, y, z, qw, qx, qy, qz]``.

    Uses identity quaternion so the AUV body faces world +X.
    """
    rng = np.random.RandomState(seed)
    waypoints = []
    for _ in range(num_waypoints):
        x = rng.uniform(*x_range)
        y = rng.uniform(*y_range)
        z = rng.uniform(*z_range)
        waypoints.append([x, y, z, 1.0, 0.0, 0.0, 0.0])
    return waypoints


# ── World configuration ───────────────────────────────────────────────────────

def compute_esdf(og) -> "torch.Tensor":
    """Compute a signed distance field from the occupancy grid.

    Uses ``scipy.ndimage.distance_transform_edt`` on the *raw* (pre-inflation)
    obstacle grid so cuRobo's collision sphere radii are not double-counted.

    Convention (cuRobo):
        * **positive** inside obstacles
        * **negative** in free space

    Returns
    -------
    torch.Tensor
        Flat float32 tensor of length ``prod(raw_grid.shape)`` (C-order).
    """
    import torch
    from scipy.ndimage import distance_transform_edt

    raw = og.raw_grid if og.raw_grid is not None else og.grid
    # Distance from every free voxel to nearest obstacle surface
    outside_dist = distance_transform_edt(~raw) * og.resolution
    # Distance from every obstacle voxel to nearest free surface
    inside_dist  = distance_transform_edt(raw)  * og.resolution
    # cuRobo convention: positive inside obstacle, negative outside
    esdf = (inside_dist - outside_dist).astype(np.float32)
    return torch.from_numpy(esdf.ravel())


def build_esdf_voxel_grid(og) -> "VoxelGrid":
    """Build a cuRobo :class:`VoxelGrid` from an :class:`OccupancyGrid`.

    The ESDF is computed from the raw (pre-inflation) grid.  The returned
    ``VoxelGrid`` can be embedded into a ``WorldConfig`` and consumed by
    ``CollisionCheckerType.VOXEL``.
    """
    if not _CUROBO_AVAILABLE:
        raise RuntimeError("cuRobo is not available in this environment.")
    import torch

    esdf_tensor = compute_esdf(og)
    raw = og.raw_grid if og.raw_grid is not None else og.grid
    # cuRobo's get_grid_shape: grid_cells[i] = 1 + robust_floor(dims[i] / voxel_size)
    # With dims = s*res this yields s+1 cells → feature_tensor shape mismatch.
    # Using (s - 0.5)*res makes robust_floor return s-1 → 1+(s-1) = s cells ✓
    dims = [(float(s) - 0.5) * og.resolution for s in raw.shape]   # metres
    # Centre is the true physical mid-point of the voxel array in world frame
    phys_extent = [float(s) * og.resolution for s in raw.shape]
    centre = og.origin + np.array(phys_extent) / 2.0
    pose = [float(centre[0]), float(centre[1]), float(centre[2]),
            1.0, 0.0, 0.0, 0.0]

    vg = VoxelGrid(
        name="ship_esdf",
        pose=pose,
        dims=dims,
        voxel_size=og.resolution,
        feature_tensor=esdf_tensor,
        feature_dtype=torch.float32,
    )
    print(f"[ESDF] VoxelGrid built: dims={[f'{d:.1f}' for d in dims]}m  "
          f"voxel_size={og.resolution}m  "
          f"tensor={esdf_tensor.shape[0]} elements  "
          f"ESDF range=[{esdf_tensor.min():.3f}, {esdf_tensor.max():.3f}]")
    return vg


def build_world_config_for_robot(
    robot_idx: int = 0,
    num_robots: int = 1,
    robot_start_xyzs: Optional[list] = None,
    esdf_voxel_grid=None,
):
    """Build a WorldConfig with static environment obstacles.

    Includes rocks/coral from ``config.STATIC_OBSTACLES`` plus a static
    cuboid at every *other* robot's depot/start position.  Placing these
    obstacles in cuRobo's world prevents trajectories from being routed
    through the starting positions of sibling robots.

    Parameters
    ----------
    robot_idx:
        Index of the robot this config is built for.
    num_robots:
        Total number of robots (used when ``robot_start_xyzs`` is None).
    robot_start_xyzs:
        List of ``(3,)`` start XYZ arrays, one per robot.  When provided,
        a Minkowski-sum cuboid (``BROV_CUBOID_DIMS`` inflated by 1.5×) is
        placed at each other robot's position.
    esdf_voxel_grid:
        Optional cuRobo ``VoxelGrid`` with a pre-computed ESDF of the
        environment mesh.  When given, the returned ``WorldConfig``
        includes it so the VOXEL collision checker will use it.
    """
    if not _CUROBO_AVAILABLE:
        raise RuntimeError("cuRobo is not available in this environment.")

    cuboids = dict(STATIC_OBSTACLES)

    if robot_start_xyzs is not None:
        # Use a cuboid slightly larger than the robot body so there is
        # comfortable clearance, but not as large as the full Minkowski sum.
        obs_dims = [d * 1.5 for d in BROV_CUBOID_DIMS]
        for j, xyz in enumerate(robot_start_xyzs):
            if j == robot_idx:
                continue
            x, y, z = float(xyz[0]), float(xyz[1]), float(xyz[2])
            cuboids[f"depot_obs_{j}"] = {
                "dims": obs_dims,
                "pose": [x, y, z, 1.0, 0.0, 0.0, 0.0],
            }

    # Build cuboid objects via the dict helper, then construct WorldConfig
    # with both cuboids and (optionally) the ESDF voxel grid.
    cuboid_wc = WorldConfig.from_dict({"cuboid": cuboids})
    return WorldConfig(
        cuboid=cuboid_wc.cuboid,
        voxel=[esdf_voxel_grid] if esdf_voxel_grid is not None else None,
    )


# ── Trajectory resampling ─────────────────────────────────────────────────────

def resample_positions_to_horizon(
    positions: np.ndarray, horizon: int
) -> np.ndarray:
    """Uniformly resample ``(T, 3)`` position array to ``(horizon, 3)``.

    Uses linear interpolation to match CuRobo's trajectory-OBB horizon.
    """
    T = positions.shape[0]
    if T == 0:
        return np.zeros((horizon, 3), dtype=np.float32)
    if T == 1:
        return np.tile(positions[0], (horizon, 1)).astype(np.float32)
    src_idx = np.linspace(0, T - 1, horizon)
    idx_lo  = np.floor(src_idx).astype(int)
    idx_hi  = np.minimum(idx_lo + 1, T - 1)
    alpha   = (src_idx - idx_lo).reshape(-1, 1)
    return ((1 - alpha) * positions[idx_lo] + alpha * positions[idx_hi]).astype(np.float32)


# ── AABB collision check ──────────────────────────────────────────────────────

def aabb_overlap(
    pos_a: np.ndarray,
    pos_b: np.ndarray,
    half_dims: np.ndarray,
) -> bool:
    """Return True if two identical AABB boxes centered at pos_a / pos_b overlap.

    ``half_dims`` are the half-extents of ONE box's dimensions.
    Two identical boxes overlap when the distance between centres is less
    than the combined (doubled) half-extents on ALL axes.
    """
    diff = np.abs(pos_a - pos_b)
    gap  = diff - 2.0 * half_dims   # negative means overlapping on that axis
    return bool(np.all(gap < 0))


def find_trajectory_collisions(
    all_traj_positions: List[List[np.ndarray]],
    dims: Optional[np.ndarray] = None,
) -> List[Tuple[int, int, int, float]]:
    """Scan replay trajectories for AABB inter-robot collisions.

    Returns
    -------
    collisions : list of (step, robot_a, robot_b, penetration_depth)
    """
    if dims is None:
        dims = np.array(BROV_CUBOID_DIMS, dtype=np.float32)
    dims = np.asarray(dims, dtype=np.float32)
    half = dims / 2.0

    num_robots  = len(all_traj_positions)
    total_steps = max(len(t) for t in all_traj_positions)
    collisions  = []

    for step in range(total_steps):
        for a in range(num_robots):
            if step >= len(all_traj_positions[a]):
                pos_a = all_traj_positions[a][-1][:3]
            else:
                pos_a = all_traj_positions[a][step][:3]
            for b in range(a + 1, num_robots):
                if step >= len(all_traj_positions[b]):
                    pos_b = all_traj_positions[b][-1][:3]
                else:
                    pos_b = all_traj_positions[b][step][:3]
                diff = np.abs(pos_a - pos_b)
                gap  = diff - 2.0 * half
                if np.all(gap < 0):
                    penetration = float(-np.max(gap))
                    collisions.append((step, a, b, penetration))
    return collisions


# ---------------------------------------------------------------------------
# Solution persistence
# ---------------------------------------------------------------------------

def save_solution(result: "ExecutionResult", path: str) -> None:  # noqa: F821
    """Pickle an :class:`~route_executor.ExecutionResult` to *path*.

    The file can later be loaded by :func:`load_solution` in the Isaac Sim
    environment without needing cuRobo or OR-Tools to be installed.

    Parameters
    ----------
    result:
        The completed execution result returned by
        :class:`~route_executor.RouteExecutor`.
    path:
        Destination file path (``*.pkl`` recommended).
    """
    import pickle

    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    with open(path, "wb") as fh:
        pickle.dump(result, fh, protocol=pickle.HIGHEST_PROTOCOL)


def load_solution(path: str) -> "ExecutionResult":  # noqa: F821
    """Load an :class:`~route_executor.ExecutionResult` from a pickle file.

    Parameters
    ----------
    path:
        Path to the ``.pkl`` file written by :func:`save_solution`.

    Returns
    -------
    ExecutionResult
    """
    import pickle

    with open(path, "rb") as fh:
        return pickle.load(fh)
