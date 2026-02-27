"""
VRP Planner – Pipeline Orchestrator

:class:`VRPPipeline` wires all modules together in the correct order:

1. Build / load occupancy grid from the wreck mesh.
2. Load / sample waypoints.
3. Compute GPU distance matrix (cuGraph Dijkstra) or CPU A* fallback.
4. Solve VRP (cuOpt GPU → OR-Tools CPU fallback).
5. Resolve spatio-temporal conflicts (traffic-light protocol).
6. Execute trajectories per robot (cuRobo hardened + RRT* warm-start).
7. (Optional) Replay in Isaac Sim.
"""

from __future__ import annotations

import logging
import os
from dataclasses import dataclass
from typing import List, Optional

import numpy as np

from .config import (
    ASSETS_PATH,
    CONFIGS_PATH,
    CUOPT_SERVICE_TIME,
    RAPIDS_PYTHON,
    ROBOT_RADIUS,
    STEPS_PER_WAYPOINT,
)
from .gpu_distance_matrix import compute_distance_matrix
from .occupancy_grid import OccupancyGrid, build_occupancy_grid
from .route_executor import ExecutionResult, RouteExecutor
from .traffic_light import resolve_conflicts
from .trajectory_planner import build_motion_gen
from .utils import build_world_config_for_robot, load_local_robot_config
from .vrp_solver import VRPResult, solve_vrp
from .waypoint_loader import load_waypoints

logger = logging.getLogger(__name__)


@dataclass
class PipelineConfig:
    """All user-facing settings for one VRP planning run."""
    num_robots:           int   = 2
    solver_backend:       str   = "auto"      # "auto" | "cuopt" | "ortools"
    rapids_python:        str   = RAPIDS_PYTHON
    service_time:         float = CUOPT_SERVICE_TIME
    gpu_timeout:          int   = 300
    ortools_time_limit:   int   = 60
    waypoint_source:      str   = "random"    # path or "random"
    n_random_waypoints:   int   = 5
    random_seed:          int   = 42
    headless:             bool  = True
    replay_in_isaac:      bool  = False
    depot:                int   = 0


class VRPPipeline:
    """End-to-end collision-free VRP pipeline.

    Parameters
    ----------
    cfg:
        :class:`PipelineConfig` describing the full run.
    """

    def __init__(self, cfg: PipelineConfig):
        self.cfg = cfg

    # ──────────────────────────────────────────────────────────────────

    def run(self) -> ExecutionResult:
        """Execute all pipeline stages and return trajectory buffers."""
        cfg = self.cfg
        logger.info("=" * 60)
        logger.info("VRP Pipeline  (%d robots)", cfg.num_robots)
        logger.info("=" * 60)

        # ── Stage 1: Occupancy grid ───────────────────────────────────
        logger.info("[1/6] Building occupancy grid …")
        og: OccupancyGrid = build_occupancy_grid()
        logger.info("      Grid shape: %s  resolution: %.2fm",
                    og.grid.shape, og.resolution)

        # ── Stage 2: Load waypoints ───────────────────────────────────
        logger.info("[2/6] Loading waypoints …")
        waypoints = load_waypoints(
            source      = cfg.waypoint_source,
            n_random    = cfg.n_random_waypoints,
            og          = og,
            random_seed = cfg.random_seed,
        )
        waypoints_world = np.array(waypoints, dtype=np.float32)  # (N, 7)
        N = len(waypoints_world)
        logger.info("      %d waypoints loaded.", N)

        # ── Stage 3: Distance matrix ──────────────────────────────────
        logger.info("[3/6] Computing %dx%d distance matrix (cuGraph) …", N, N)
        dist_matrix = compute_distance_matrix(og, waypoints_world[:, :3])
        logger.info("      Distance matrix computed.  max_dist=%.2fm",
                    float(np.max(dist_matrix[np.isfinite(dist_matrix)])))

        # ── Stage 4: VRP solve ────────────────────────────────────────
        logger.info("[4/6] Solving VRP (%s) …", cfg.solver_backend)
        vrp_result: VRPResult = solve_vrp(
            dist_matrix   = dist_matrix,
            num_vehicles  = cfg.num_robots,
            depot         = cfg.depot,
            backend       = cfg.solver_backend,
            rapids_python = cfg.rapids_python,
            service_time  = cfg.service_time,
            time_limit    = cfg.ortools_time_limit,
            gpu_timeout   = cfg.gpu_timeout,
        )
        logger.info("      VRP status=%s  total_cost=%.2f  solver=%s",
                    vrp_result.status, vrp_result.total_cost, vrp_result.solver)
        logger.info("      Routes: %s", vrp_result.routes)

        routes = vrp_result.routes
        if not any(routes):
            logger.error("VRP produced empty routes – aborting.")
            raise RuntimeError(f"VRP failed: {vrp_result.status}")

        # ── Stage 5: Traffic-light conflict resolution ─────────────────
        logger.info("[5/6] Resolving spatio-temporal conflicts …")
        traffic_plan = resolve_conflicts(
            routes          = routes,
            waypoints_world = waypoints_world,
            steps_per_leg   = STEPS_PER_WAYPOINT,
            og              = og,
        )
        logger.info("      Resolved in %d round(s).", traffic_plan.num_rounds)

        # ── Stage 6: Trajectory execution ────────────────────────────
        logger.info("[6/6] Planning trajectories (cuRobo + RRT*) …")

        robot_cfg   = load_local_robot_config("brov.yml")
        j_names     = robot_cfg["kinematics"]["cspace"]["joint_names"]
        default_cfg = robot_cfg["kinematics"]["cspace"]["retract_config"]

        # Build start configs (spread robots along X axis)
        spacing = 3.0
        start_configs: List[np.ndarray] = []
        for i in range(cfg.num_robots):
            pos = np.array([i * spacing, 0.0, 1.5], dtype=np.float32)
            s   = list(default_cfg)
            s[0], s[1], s[2] = float(pos[0]), float(pos[1]), float(pos[2])
            start_configs.append(np.array(s, dtype=np.float32))

        from curobo.types.base import TensorDeviceType           # type: ignore

        tensor_args = TensorDeviceType()
        motion_gens = []
        for i in range(cfg.num_robots):
            logger.info("  Warming up MotionGen %d …", i)
            world_cfg = build_world_config_for_robot(i, cfg.num_robots)
            mg = build_motion_gen(robot_cfg, world_cfg, tensor_args)
            motion_gens.append(mg)

        executor = RouteExecutor(
            motion_gens  = motion_gens,
            start_configs = start_configs,
            joint_names  = j_names,
            og           = og,
        )
        exec_result: ExecutionResult = executor.execute(
            routes          = routes,
            waypoints_world = waypoints_world,
            traffic_plan    = traffic_plan,
        )

        logger.info("=" * 60)
        logger.info("Pipeline complete.  Fail counts: %s", exec_result.fail_counts)
        logger.info("=" * 60)

        # ── Optional: Isaac Sim replay ────────────────────────────────
        if cfg.replay_in_isaac:
            from .visualization import replay_in_isaac_sim
            replay_in_isaac_sim(exec_result, headless=cfg.headless)

        return exec_result
