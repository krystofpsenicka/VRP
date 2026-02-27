"""
VRP Planner – Route Executor

Executes per-vehicle waypoint routes produced by the VRP solver with
traffic-light wait injections.  The planner priority order is preserved
from the original ``run_multi_auv_waypoints.py`` reference:

* Higher-priority (lower index) robot's trajectory is committed first.
* All committed trajectories feed into the next robot's planner as
  trajectory OBB obstacles (via :func:`~utils.build_traj_obb_tensors`).
* Post-execution AABB collision check triggers re-planning on conflict.

Returns lists of per-step joint positions and velocities ready for the
Isaac Sim visualisation phase.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch

from .config import (
    BROV_CUBOID_DIMS,
    CUROBO_NUM_GRAPH_SEEDS,
    CUROBO_NUM_TRAJOPT_SEEDS,
    ROBOT_RADIUS,
    TRAJOPT_HORIZON,
)
from .traffic_light import TrafficPlan
from .trajectory_planner import HybridTrajectoryPlanner, SegmentResult
from .utils import (
    aabb_overlap,
    build_traj_obb_tensors,
    find_trajectory_collisions,
)

logger = logging.getLogger(__name__)

# Number of hold-steps appended after each waypoint
WAIT_STEPS = 25

# Max re-plan attempts per segment when AABB post-check fires
MAX_REPLAN_ATTEMPTS = 3


# ─── Result dataclass ────────────────────────────────────────────────────────

@dataclass
class ExecutionResult:
    """Full trajectory for every robot after route execution.

    Attributes
    ----------
    all_traj_positions:
        ``[num_robots][num_steps]`` – joint position arrays (8-DOF).
    all_traj_velocities:
        ``[num_robots][num_steps]`` – joint velocity arrays (8-DOF).
    all_waypoints:
        ``[num_robots][num_waypoints]`` – original waypoint lists sent to
        the executor (replicated for the Isaac Sim visualisation phase).
    initial_positions:
        ``[num_robots]`` – start XYZ.
    joint_names:
        List of joint name strings.
    fail_counts:
        Per-robot count of completely failed waypoints.
    """
    all_traj_positions:  List[List[np.ndarray]]
    all_traj_velocities: List[List[np.ndarray]]
    all_waypoints:       List[List[List[float]]]
    initial_positions:   List[np.ndarray]
    joint_names:         List[str]
    fail_counts:         List[int]


# ─── Executor ────────────────────────────────────────────────────────────────

class RouteExecutor:
    """Plans and commits trajectories for all vehicles according to their
    VRP routes + traffic-light plan.

    Parameters
    ----------
    motion_gens:
        List of ``MotionGen`` instances, one per robot.
    start_configs:
        ``(num_robots, D)`` initial joint configurations.
    joint_names:
        Ordered joint name list.
    og:
        Occupancy grid for RRT* fallback.
    """

    def __init__(
        self,
        motion_gens:    List,
        start_configs:  List[np.ndarray],
        joint_names:    List[str],
        og,
    ):
        self.motion_gens   = motion_gens
        self.num_robots    = len(motion_gens)
        self.start_configs = start_configs
        self.joint_names   = joint_names
        self.og            = og

        # Build per-robot hybrid planners
        self.planners = [
            HybridTrajectoryPlanner(mg, og, robot_name=f"AUV-{i}")
            for i, mg in enumerate(motion_gens)
        ]

    # ──────────────────────────────────────────────────────────────────

    def execute(
        self,
        routes:          List[List[int]],
        waypoints_world: np.ndarray,
        traffic_plan:    Optional[TrafficPlan] = None,
    ) -> ExecutionResult:
        """Execute all routes and return the full trajectory arrays.

        Parameters
        ----------
        routes:
            Per-vehicle ordered waypoint index lists (from VRP solver).
        waypoints_world:
            ``(N, 7)`` array ``[x, y, z, qw, qx, qy, qz]``.
        traffic_plan:
            Optional :class:`~traffic_light.TrafficPlan` with per-step
            wait counts.
        """
        num_robots  = self.num_robots
        n_robots    = num_robots

        # Build all_waypoints list (per-robot list of [x,y,z,qw,qx,qy,qz])
        all_waypoints: List[List[List[float]]] = []
        for v, route in enumerate(routes):
            all_waypoints.append([waypoints_world[wp_idx].tolist()
                                   for wp_idx in route])

        initial_positions = [cfg[:3].copy() for cfg in self.start_configs]

        # ── Tracking buffers ──────────────────────────────────────────
        all_traj_positions:  List[List[np.ndarray]] = [[] for _ in range(n_robots)]
        all_traj_velocities: List[List[np.ndarray]] = [[] for _ in range(n_robots)]
        committed_xyz:        Dict[int, List]        = {i: [] for i in range(n_robots)}
        current_js     = [c.copy() for c in self.start_configs]
        fail_counts    = [0] * n_robots

        # Convenient helpers -------------------------------------------------

        def _append_traj(robot_i: int, pos_np: np.ndarray, vel_np: np.ndarray):
            for sp, sv in zip(pos_np, vel_np):
                all_traj_positions[robot_i].append(sp)
                all_traj_velocities[robot_i].append(sv)
            committed_xyz[robot_i].extend(pos_np[:, :3].tolist())

        def _append_hold(robot_i: int, steps: int):
            cfg      = current_js[robot_i]
            zero_vel = np.zeros_like(cfg)
            for _ in range(steps):
                all_traj_positions[robot_i].append(cfg.copy())
                all_traj_velocities[robot_i].append(zero_vel.copy())
            committed_xyz[robot_i].extend([cfg[:3].tolist()] * steps)

        def _update_obstacle_poses(planner_idx: int):
            """Place static cuboids for other robots at their current pos."""
            try:
                from curobo.types.math import Pose  # type: ignore
                tensor_args = self.motion_gens[planner_idx].tensor_args
                for j in range(n_robots):
                    if j == planner_idx:
                        continue
                    obs_pose = Pose(
                        position=tensor_args.to_device(current_js[j][:3]),
                        quaternion=tensor_args.to_device(
                            np.array([1.0, 0.0, 0.0, 0.0])
                        ),
                    )
                    self.motion_gens[planner_idx].world_coll_checker \
                        .update_obstacle_pose(
                            name=f"robot_obs_{j}", w_obj_pose=obs_pose
                        )
            except Exception as exc:
                logger.debug("[executor] update_obstacle_poses skipped: %s", exc)

        def _set_traj_obstacles(planner_idx: int):
            """Feed trajectory OBBs from committed robots into planner."""
            try:
                from curobo.types.base import TensorDeviceType  # type: ignore
                tensor_args = self.motion_gens[planner_idx].tensor_args
                planner_start_step = len(all_traj_positions[planner_idx])
                dims, poses, enable, n = build_traj_obb_tensors(
                    committed_xyz, planner_idx, planner_start_step,
                    tensor_args, TRAJOPT_HORIZON,
                )
                wcc = self.motion_gens[planner_idx].world_coll_checker
                if n == 0:
                    wcc.clear_trajectory_obstacles()
                else:
                    wcc.set_trajectory_obstacles(dims, poses, enable)
            except Exception as exc:
                logger.debug("[executor] set_traj_obstacles skipped: %s", exc)

        # ── Get wait counts from traffic plan ─────────────────────────
        wait_counts: List[List[int]] = (
            traffic_plan.wait_counts
            if traffic_plan is not None
            else [[0] * len(r) for r in routes]
        )

        # ── Main planning loop – one waypoint at a time ───────────────
        max_wps = max(len(r) for r in all_waypoints) if all_waypoints else 0
        for wp_idx in range(max_wps):
            logger.info("── Waypoint %d ──", wp_idx)

            for i in range(n_robots):
                route_i = all_waypoints[i]
                if wp_idx >= len(route_i):
                    continue   # robot has fewer waypoints

                wp        = route_i[wp_idx]
                wait_here = (wait_counts[i][wp_idx]
                             if wp_idx < len(wait_counts[i]) else 0)

                _update_obstacle_poses(i)
                _set_traj_obstacles(i)

                goal_pose = np.array(wp[:7], dtype=np.float32)
                start_js  = current_js[i]

                # ── Traffic-light pre-wait ────────────────────────────
                if wait_here > 0:
                    logger.info("  Robot %d: holding %d steps before wp %d.",
                                i, wait_here, wp_idx)
                    _append_hold(i, wait_here)

                # ── Plan segment with hybrid planner ──────────────────
                seg: Optional[SegmentResult] = None
                for attempt in range(MAX_REPLAN_ATTEMPTS):
                    seg = self.planners[i].plan_segment(start_js, goal_pose)
                    if seg.success:
                        break
                    logger.warning("  Robot %d, wp %d attempt %d failed.",
                                   i, wp_idx, attempt)

                if seg is None or not seg.success:
                    fail_counts[i] += 1
                    logger.error("  Robot %d → wp %d FAILED after %d attempts.",
                                 i, wp_idx, MAX_REPLAN_ATTEMPTS)
                    _append_hold(i, WAIT_STEPS)
                    continue

                # Commit trajectory
                if seg.joint_trajectory is not None:
                    pos_np = seg.joint_trajectory.astype(np.float32)
                    vel_np = np.zeros_like(pos_np)
                    if len(pos_np) > 1:
                        vel_np[:-1] = np.diff(pos_np, axis=0)
                else:
                    # RRT* path – encode as prismatic moves (xyz joints 0-2)
                    pos_np = np.tile(current_js[i], (len(seg.positions), 1))
                    for s_idx, xyz in enumerate(seg.positions):
                        pos_np[s_idx, 0] = float(xyz[0])
                        pos_np[s_idx, 1] = float(xyz[1])
                        pos_np[s_idx, 2] = float(xyz[2])
                    vel_np = np.zeros_like(pos_np)
                    if len(pos_np) > 1:
                        vel_np[:-1] = np.diff(pos_np, axis=0)

                _append_traj(i, pos_np, vel_np)
                current_js[i] = pos_np[-1].copy()

                _append_hold(i, WAIT_STEPS)
                self.motion_gens[i].world_coll_checker.clear_trajectory_obstacles()

                logger.info(
                    "  Robot %d → wp %d OK (planner=%s, steps=%d).",
                    i, wp_idx, seg.planner, len(pos_np),
                )

        # ── Post-planning AABB collision check ────────────────────────
        _run_post_check(all_traj_positions, fail_counts)

        return ExecutionResult(
            all_traj_positions  = all_traj_positions,
            all_traj_velocities = all_traj_velocities,
            all_waypoints       = all_waypoints,
            initial_positions   = initial_positions,
            joint_names         = self.joint_names,
            fail_counts         = fail_counts,
        )


# ─── Post-planning AABB check ────────────────────────────────────────────────

def _run_post_check(
    all_traj_positions: List[List[np.ndarray]],
    fail_counts:        List[int],
) -> None:
    """Log AABB inter-robot collisions detected in the committed trajectories."""
    # Collect all robot XYZ per step
    min_len = min(len(t) for t in all_traj_positions)
    if min_len == 0:
        return

    robot_xyz: List[np.ndarray] = []
    for traj in all_traj_positions:
        xyz = np.array([s[:3] for s in traj[:min_len]], dtype=np.float32)
        robot_xyz.append(xyz)

    collisions = find_trajectory_collisions(robot_xyz, BROV_CUBOID_DIMS)
    if collisions:
        logger.warning(
            "[post_check] %d AABB inter-robot collision(s) detected:",
            len(collisions),
        )
        for step, a, b, pen in collisions:
            logger.warning("  step=%d  robots (%d, %d)  penetration=%.3fm",
                           step, a, b, pen)
    else:
        logger.info("[post_check] No AABB collisions detected.")
