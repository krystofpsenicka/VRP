"""
VRP Planner – Shared Utilities

Ported helpers from brov_auv_curobo/run_multi_auv_waypoints.py:
  • load_local_robot_config()
  • generate_random_waypoints()
  • build_world_config_for_robot()
  • _resample_positions_to_horizon()
  • _build_traj_obb_tensors()
  • aabb_overlap()
"""

from __future__ import annotations
import os
import sys
from typing import Dict, List, Optional, Tuple

import numpy as np

from VRP.config import (
    ASSETS_PATH,
    BROV_CUBOID_DIMS,
    CONFIGS_PATH,
    OBSTACLE_CUBOID_DIMS,
    ROBOT_CFG_DIR,
    STATIC_OBSTACLES,
    TRAJOPT_HORIZON,
)

# ---------------------------------------------------------------------------
# cuRobo imports (optional – utils should still be importable without them
# to allow waypoint / grid work without a full cuRobo GPU env)
# ---------------------------------------------------------------------------
try:
    from curobo.geom.types import WorldConfig
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

def build_world_config_for_robot(robot_idx: int, num_robots: int):
    """Build a WorldConfig with static obstacles + per-robot placeholder cuboids.

    Each robot gets a ``WorldConfig`` that contains:
    • All static environment obstacles (rocks, coral).
    • One inflated placeholder cuboid for every *other* robot, spawned
      far away at ``[100 + j, 100, 100]`` and moved to real positions
      at planning time via ``update_obstacle_pose``.
    """
    if not _CUROBO_AVAILABLE:
        raise RuntimeError("cuRobo is not available in this environment.")
    cuboids = dict(STATIC_OBSTACLES)
    for j in range(num_robots):
        if j == robot_idx:
            continue
        cuboids[f"robot_obs_{j}"] = {
            "dims": OBSTACLE_CUBOID_DIMS,
            "pose": [100.0 + j, 100.0, 100.0, 1.0, 0.0, 0.0, 0.0],
        }
    return WorldConfig.from_dict({"cuboid": cuboids})


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


# ── Trajectory OBB tensor builder ─────────────────────────────────────────────

def build_traj_obb_tensors(
    committed_xyz: Dict[int, List[np.ndarray]],
    planner_idx: int,
    planner_start_step: int,
    tensor_args,
    horizon: int = TRAJOPT_HORIZON,
):
    """Build trajectory OBB tensors from cumulative committed trajectories.

    For each robot ``k != planner_idx`` whose committed trajectory extends
    beyond ``planner_start_step``, extract the temporally-relevant window
    and resample it to *horizon* optimiser steps.

    Returns
    -------
    traj_obb_dims, traj_obb_poses, traj_obb_enable, n_traj_obbs
        Ready to pass to ``world_coll_checker.set_trajectory_obstacles``.
        All four are ``None / 0`` when there are no relevant obstacles.
    """
    import torch

    dims_list  = []
    poses_list = []

    for k, xyz_list in committed_xyz.items():
        if k == planner_idx:
            continue
        total_steps = len(xyz_list)
        if total_steps <= planner_start_step:
            continue   # robot k finishes before planner_idx starts this segment

        window    = np.array(xyz_list[planner_start_step:], dtype=np.float32)  # (W, 3)
        resampled = resample_positions_to_horizon(window, horizon)              # (H, 3)

        dims_list.append(OBSTACLE_CUBOID_DIMS + [0.0])   # [lx, ly, lz, 0]

        # OBB pose per horizon step: [x, y, z, qw, qx, qy, qz, 0]
        # CuRobo expects the **inverse** object-in-world pose.
        h_poses = np.zeros((horizon, 8), dtype=np.float32)
        h_poses[:, :3] = -resampled     # negated XYZ (obj_w convention)
        h_poses[:, 3]  = 1.0            # qw = 1 (identity rotation)
        poses_list.append(h_poses)

    n = len(dims_list)
    if n == 0:
        return None, None, None, 0

    traj_obb_dims  = tensor_args.to_device(
        torch.tensor(np.array(dims_list, dtype=np.float32))
    )   # (n, 4)
    traj_obb_poses = tensor_args.to_device(
        torch.tensor(np.stack(poses_list, axis=0))
    )   # (n, H, 8)
    traj_obb_enable = torch.ones(n, dtype=torch.uint8, device=traj_obb_dims.device)
    return traj_obb_dims, traj_obb_poses, traj_obb_enable, n


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
