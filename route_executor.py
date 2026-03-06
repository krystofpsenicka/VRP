"""
VRP Planner – Route Executor (Priority-Based Sequential Planning)

Builds per-vehicle trajectories that are **collision-free by construction**.
Robots are planned one at a time in priority order (longest route first).
Each robot's route is planned via Space-Time A* on a coarse 4-D grid so it
avoids both static obstacles *and* previously committed robots'
trajectories.  The coarse space-time path is then smoothly interpolated at
the replay time-step (``TRAJ_DT``) and converted to 8-DOF joint space.

Pipeline
--------
1. Down-sample occupancy grid → coarse grid for Space-Time A*.
2. Initialise a dense 4-D reservation table.
3. Sort robots by estimated route cost (longest first = highest priority).
4. For each robot (priority order):
   a.  Plan each route leg with Space-Time A*, building a coarse
       time-stamped XYZ path.
   b.  Commit the trajectory to the reservation table so later robots
       avoid it.
   c.  Smooth-interpolate the coarse path at TRAJ_DT for replay.
5. Run a final AABB safety check (should be clean; log any residual).
"""

from __future__ import annotations

import logging
import math
from dataclasses import dataclass
from typing import List, Optional

import numpy as np

from .config import (
    AUV_CRUISE_SPEED,
    BROV_CUBOID_DIMS,
    ROBOT_RADIUS,
    SPACE_TIME_DT,
    SPACE_TIME_DWELL_S,
    SPACE_TIME_MAX_HORIZON_S,
    SPACE_TIME_RESOLUTION,
    SPLINE_SAFETY_VOXELS,
)
from .space_time_astar import (
    ReservationTable,
    downsample_occupancy_grid,
    plan_robot_route_st,
)
from .utils import find_trajectory_collisions

logger = logging.getLogger(__name__)

# ── Trajectory interpolation defaults ─────────────────────────────────────────
TRAJ_DT = 0.02  # seconds – replay sample period


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
    """Priority-Based Sequential Planner from Space-Time A* paths.

    Parameters
    ----------
    start_configs:
        ``(num_robots, D)`` initial joint configurations.
    joint_names:
        Ordered joint name list.
    og:
        Occupancy grid (used to build the coarse grid for Space-Time A*).
    """

    def __init__(
        self,
        start_configs:  List[np.ndarray],
        joint_names:    List[str],
        og=None,
    ):
        self.num_robots    = len(start_configs)
        self.start_configs = start_configs
        self.joint_names   = joint_names
        self.og            = og

    # ──────────────────────────────────────────────────────────────────

    def execute(
        self,
        routes:          List[List[int]],
        waypoints_world: np.ndarray,
        traffic_plan=None,
        path_cache: Optional[dict] = None,
        dwell_s: float = SPACE_TIME_DWELL_S,
    ) -> ExecutionResult:
        """Build collision-free trajectories via Priority-Based Sequential
        Planning and return an :class:`ExecutionResult`.
        """
        n_robots = self.num_robots
        n_dof    = len(self.start_configs[0])  # 8

        # Build per-robot waypoint list for visualisation
        all_waypoints: List[List[List[float]]] = [
            [waypoints_world[wp_idx].tolist() for wp_idx in route]
            for route in routes
        ]
        initial_positions = [cfg[:3].copy() for cfg in self.start_configs]

        # ── 1. Coarse grid + reservation table ────────────────────────
        if self.og is not None:
            coarse_grid, coarse_origin, coarse_res = downsample_occupancy_grid(
                self.og.grid, self.og.origin, self.og.resolution,
                coarse_res=SPACE_TIME_RESOLUTION,
            )
        else:
            # No occupancy grid available – create an empty one
            logger.warning("No occupancy grid; collision avoidance with "
                           "static obstacles is disabled.")
            coarse_grid = np.zeros((10, 10, 10), dtype=np.bool_)
            coarse_origin = np.zeros(3)
            coarse_res = SPACE_TIME_RESOLUTION

        T_max = max(1, int(math.ceil(SPACE_TIME_MAX_HORIZON_S / SPACE_TIME_DT)))
        # Robot AABB half-extents in coarse voxels (≥1 so at least the
        # centre voxel is always reserved).  We add SPLINE_SAFETY_VOXELS
        # to compensate for cubic-spline overshoot during smooth interp.
        dims = np.array(BROV_CUBOID_DIMS, dtype=np.float64)
        half_v = np.maximum(np.ceil(dims / (2.0 * coarse_res)).astype(np.intp), 1)
        half_v += SPLINE_SAFETY_VOXELS
        reservation = ReservationTable(coarse_grid.shape, T_max, half_v)

        logger.info("[executor] Coarse grid %s  T=%d  half_v=%s  "
                    "reservation %.1f MB",
                    coarse_grid.shape, T_max, half_v,
                    reservation._data.nbytes / 1e6)

        # ── 2. Priority ordering: longest route first ─────────────────
        # Estimate route cost from path_cache or Euclidean distance
        route_costs: List[float] = []
        for i, route in enumerate(routes):
            cost = 0.0
            for leg in range(1, len(route)):
                a, b = route[leg - 1], route[leg]
                if path_cache and (a, b) in path_cache:
                    pc = path_cache[(a, b)]
                    cost += float(np.sum(np.linalg.norm(np.diff(pc, axis=0), axis=1)))
                else:
                    cost += float(np.linalg.norm(
                        waypoints_world[b, :3] - waypoints_world[a, :3]))
            route_costs.append(cost)

        priority_order = sorted(range(n_robots), key=lambda i: -route_costs[i])
        logger.info("[executor] Priority order (longest first): %s  costs=%s",
                    priority_order, [f"{route_costs[i]:.1f}" for i in priority_order])

        # ── 3. Plan each robot sequentially ───────────────────────────
        # Stores per-robot coarse-resolution results
        robot_world_paths: List[Optional[np.ndarray]] = [None] * n_robots
        robot_coarse_times: List[Optional[np.ndarray]] = [None] * n_robots
        robot_wp_schedules: List[Optional[list]] = [None] * n_robots
        for priority, robot_idx in enumerate(priority_order):
            route = routes[robot_idx]
            logger.info("[executor] Planning robot %d (priority %d/%d, "
                        "%d legs, est. %.1fm) …",
                        robot_idx, priority + 1, n_robots,
                        len(route) - 1, route_costs[robot_idx])

            world_xyz, coarse_t, wp_schedule = plan_robot_route_st(
                coarse_grid, coarse_origin, coarse_res,
                reservation, route, waypoints_world,
                path_cache=path_cache,
                dwell_s=dwell_s,
                dt=SPACE_TIME_DT,
                fine_og=self.og,
                robot_radius=ROBOT_RADIUS,
            )

            if len(world_xyz) == 0:
                logger.warning("  Robot %d: ST planning returned empty path!",
                               robot_idx)
            else:
                logger.info("  Robot %d: %d coarse samples, t=[%d..%d] "
                            "(%.1f s)",
                            robot_idx, len(world_xyz),
                            int(coarse_t[0]), int(coarse_t[-1]),
                            float(coarse_t[-1]) * SPACE_TIME_DT)

            robot_world_paths[robot_idx] = world_xyz
            robot_coarse_times[robot_idx] = coarse_t
            robot_wp_schedules[robot_idx] = wp_schedule

        # ── 4. Smooth-interpolate to replay resolution ────────────────
        all_traj_positions:  List[List[np.ndarray]] = [[] for _ in range(n_robots)]
        all_traj_velocities: List[List[np.ndarray]] = [[] for _ in range(n_robots)]
        fail_counts = [0] * n_robots

        for i in range(n_robots):
            w_xyz = robot_world_paths[i]
            c_t   = robot_coarse_times[i]
            cfg   = self.start_configs[i].copy()  # 8-DOF
            wp_sched = robot_wp_schedules[i] or []
            # Convert coarse time-step schedule → seconds
            wp_sched_s = [
                (int(ts) * SPACE_TIME_DT, int(te) * SPACE_TIME_DT, node)
                for ts, te, node in wp_sched
            ]

            if w_xyz is None or len(w_xyz) == 0:
                # Nothing planned – hold at home
                fail_counts[i] = len(routes[i]) - 1
                for _ in range(50):
                    all_traj_positions[i].append(cfg.astype(np.float32).copy())
                    all_traj_velocities[i].append(np.zeros_like(cfg, dtype=np.float32))
                continue

            # Convert coarse time-steps → continuous seconds
            t_seconds = c_t.astype(np.float64) * SPACE_TIME_DT
            t0, tf = float(t_seconds[0]), float(t_seconds[-1])
            duration = tf - t0

            if duration < 1e-6:
                traj_js = np.tile(cfg.astype(np.float64), (1, 1))
                traj_js[0, :3] = w_xyz[-1]
                t_dense = np.array([t0])
            else:
                n_steps = max(2, int(np.ceil(duration / TRAJ_DT)))
                t_dense = np.linspace(t0, tf, n_steps)
                # Remove duplicate time stamps from dwells
                _, unique_idx = np.unique(t_seconds, return_index=True)
                unique_idx = np.sort(unique_idx)
                t_uniq = t_seconds[unique_idx]
                xyz_uniq = w_xyz[unique_idx]
                # Linear densification – path is already OMPL-smooth
                xyz_dense = np.column_stack([
                    np.interp(t_dense, t_uniq, xyz_uniq[:, d]) for d in range(3)
                ])
                traj_js = np.tile(cfg.astype(np.float64), (n_steps, 1))
                traj_js[:, :3] = xyz_dense

            # Orientation: interpolate yaw/pitch between consecutive waypoints
            _apply_heading_orientation(
                traj_js,
                t_dense=t_dense,
                dt=TRAJ_DT,
                wp_schedule_s=wp_sched_s,
                waypoints_world=waypoints_world,
            )

            # Finite-difference velocities
            vel = np.zeros_like(traj_js)
            if len(traj_js) > 1:
                vel[:-1] = np.diff(traj_js, axis=0) / TRAJ_DT

            for sp, sv in zip(traj_js, vel):
                all_traj_positions[i].append(sp.astype(np.float32))
                all_traj_velocities[i].append(sv.astype(np.float32))

        # ── 5. Safety AABB check (should be clean) ────────────────────
        collisions = find_trajectory_collisions(all_traj_positions,
                                                np.array(BROV_CUBOID_DIMS, dtype=np.float32))
        if collisions:
            logger.warning("[executor] %d residual AABB collision(s) after "
                           "priority-based planning (logged for diagnostics).",
                           len(collisions))
        else:
            logger.info("[executor] No residual AABB collisions – clean plan.")

        return ExecutionResult(
            all_traj_positions  = all_traj_positions,
            all_traj_velocities = all_traj_velocities,
            all_waypoints       = all_waypoints,
            initial_positions   = initial_positions,
            joint_names         = self.joint_names,
            fail_counts         = fail_counts,
        )


# ─── Orientation: interpolate yaw between consecutive waypoints ─────────────

def _apply_heading_orientation(
    traj: np.ndarray,
    t_dense: np.ndarray,
    dt: float = TRAJ_DT,
    wp_schedule_s: Optional[list] = None,
    waypoints_world: Optional[np.ndarray] = None,
) -> np.ndarray:
    """Set yaw (joint 3) and camera pitch (joint 7) on a dense trajectory.

    During dwell windows the robot holds the waypoint's viewing direction
    derived from the waypoint quaternion.  Between dwells the yaw and camera
    pitch are cosine-eased between successive waypoint values.

    Joint layout: [x, y, z, yaw, pitch, roll, cam_yaw, cam_pitch]
    """
    N = len(traj)
    if N < 2:
        return traj

    yaw          = np.empty(N)
    camera_pitch = np.zeros(N)

    if not wp_schedule_s or waypoints_world is None:
        yaw[:] = traj[0, 3]
        traj[:, 3] = (yaw + math.pi) % (2 * math.pi) - math.pi
        return traj

    # ── Extract per-waypoint yaw and camera pitch ────────────────────────────
    wp_yaws:   list = []
    wp_cpitch: list = []
    wp_t_ds:   list = []
    wp_t_de:   list = []

    for (t_ds, t_de, node_idx) in wp_schedule_s:
        wp7 = waypoints_world[node_idx]
        qw = float(wp7[3]); qx = float(wp7[4])
        qy = float(wp7[5]); qz = float(wp7[6])
        # Rotate +X by quaternion → camera forward direction
        fx = 1.0 - 2.0 * (qy * qy + qz * qz)
        fy = 2.0 * (qx * qy + qw * qz)
        fz = 2.0 * (qx * qz - qw * qy)
        xy_norm = math.sqrt(fx * fx + fy * fy)
        wp_yaws.append(math.atan2(fy, fx))
        wp_cpitch.append(-math.atan2(fz, xy_norm) if xy_norm > 1e-9 else 0.0)
        wp_t_ds.append(t_ds)
        wp_t_de.append(t_de)

    # Unwrap waypoint yaw sequence so interpolation always takes the short arc
    wp_yaws = list(np.unwrap(wp_yaws))

    n_wp = len(wp_schedule_s)

    def _dense_idx(t: float, side: str = "left") -> int:
        return max(0, min(int(np.searchsorted(t_dense, t, side=side)), N))

    # ── Before first waypoint: hold first waypoint yaw ──────────────────────
    d_start_0 = _dense_idx(wp_t_ds[0])
    yaw[:d_start_0]          = wp_yaws[0]
    camera_pitch[:d_start_0] = 0.0

    # ── Fill dwell windows and transit segments ──────────────────────────────
    for i in range(n_wp):
        d_start = _dense_idx(wp_t_ds[i])
        d_end   = _dense_idx(wp_t_de[i], side="right")

        # Dwell: exact waypoint orientation
        yaw[d_start:d_end]          = wp_yaws[i]
        camera_pitch[d_start:d_end] = wp_cpitch[i]

        if i < n_wp - 1:
            # Transit: cosine-ease from wp[i] to wp[i+1]
            next_start = _dense_idx(wp_t_ds[i + 1])
            if next_start > d_end:
                ts   = t_dense[d_end:next_start]
                t0_t = t_dense[d_end]
                t1_t = t_dense[min(next_start, N - 1)]
                dur  = t1_t - t0_t
                if dur > 0:
                    alpha = (ts - t0_t) / dur
                    ease  = 0.5 * (1.0 - np.cos(math.pi * alpha))
                    yaw[d_end:next_start]          = (wp_yaws[i]
                        + ease * (wp_yaws[i + 1] - wp_yaws[i]))
                    camera_pitch[d_end:next_start] = (wp_cpitch[i]
                        + ease * (wp_cpitch[i + 1] - wp_cpitch[i]))
                else:
                    yaw[d_end:next_start]          = wp_yaws[i]
                    camera_pitch[d_end:next_start] = wp_cpitch[i]
        else:
            # After last waypoint: hold last waypoint yaw
            yaw[d_end:]          = wp_yaws[i]
            camera_pitch[d_end:] = 0.0

    # ── Write back ───────────────────────────────────────────────────────────
    # Write unwrapped yaw directly — do NOT wrap to [-π, π].
    # Wrapping creates a discontinuity when the interpolated yaw crosses ±π,
    # causing the joint controller to reverse direction mid-transit.
    traj[:, 3] = yaw
    if traj.shape[1] > 7:
        traj[:, 7] = camera_pitch
    return traj

