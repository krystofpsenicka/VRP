"""
VRP Planner – Hybrid Trajectory Planner

Implements the two-layer planning strategy from the PDF:

Layer 1 – **cuRobo** (hardened):
    * ``num_trajopt_seeds = 1024``, ``num_graph_seeds = 1024``
    * ``trajopt_fix_terminal_action = True``
    * dt-ladder retry: 0.02 → 0.05 → 0.10 s
    * ``reach_partial_pose = True`` for position-only fallback

Layer 2 – **OMPL RRT*** (probabilistically complete backup):
    * Runs on the 3-D occupancy grid (no IK).
    * Returns a geometrically smooth joint-space path.
    * Path joints are resampled to the cuRobo horizon and used as a
      *warm-start seed* for a final cuRobo polish pass.
    * Only if the polish pass also fails is the raw RRT* waypoints list
      returned to the caller.
"""

from __future__ import annotations

import logging
import math
from dataclasses import dataclass, field
from typing import List, Optional, Tuple

import numpy as np
import torch

from .config import (
    BROV_CUBOID_DIMS,
    CUROBO_DT_LADDER,
    CUROBO_FIX_TERMINAL_ACTION,
    CUROBO_NUM_GRAPH_SEEDS,
    CUROBO_NUM_TRAJOPT_SEEDS,
    CUROBO_POS_ONLY_WEIGHTS,
    TRAJOPT_HORIZON,
)
from .utils import resample_positions_to_horizon

logger = logging.getLogger(__name__)


# ─── Result dataclass ────────────────────────────────────────────────────────

@dataclass
class SegmentResult:
    """Trajectory for a single waypoint-to-waypoint segment.

    Attributes
    ----------
    positions:
        ``(H, 3)`` world-frame XYZ trajectory sampled at ``H`` steps.
    joint_trajectory:
        ``(H, D)`` joint angle trajectory (None when RRT* path is raw).
    success:
        True when a collision-free trajectory was found.
    planner:
        ``"curobo"`` | ``"curobo_warmed"`` | ``"rrt_star"``
    dt:
        Time step used by cuRobo (None for RRT*).
    """
    positions:        np.ndarray                    # (H, 3)
    joint_trajectory: Optional[np.ndarray] = None  # (H, D)
    success:          bool                 = True
    planner:          str                  = "curobo"
    dt:               Optional[float]      = None


# ─── RRT* helper (OMPL) ──────────────────────────────────────────────────────

def _rrt_star_plan(
    start_xyz:   np.ndarray,
    goal_xyz:    np.ndarray,
    og,                          # OccupancyGrid
    radius:      float = 0.35,
    timeout:     float = 5.0,
) -> Optional[np.ndarray]:
    """Plan a collision-free path with OMPL RRT* on the occupancy grid.

    Parameters
    ----------
    start_xyz, goal_xyz:
        3-D positions in world frame.
    og:
        :class:`~occupancy_grid.OccupancyGrid` used for collision checking.
    radius:
        Robot bounding-sphere radius for grid-based collision checks.
    timeout:
        Time budget (seconds) for the RRT* planner.

    Returns
    -------
    ``(M, 3)`` numpy array of path waypoints, or ``None`` if planning failed.
    """
    try:
        import ompl.base as ob  # type: ignore
        import ompl.geometric as og_ompl  # type: ignore
    except ImportError:
        logger.warning("[RRT*] pyompl not installed – falling back to straight-line path.")
        return np.stack([start_xyz, goal_xyz])

    class ValidityChecker(ob.StateValidityChecker):
        def __init__(self, si):
            super().__init__(si)

        def isValid(self, state):  # type: ignore[override]
            xyz = np.array([state[0], state[1], state[2]])
            return bool(og.is_free_world(xyz))

    # Build 3-D SE(3)-free space
    space = ob.RealVectorStateSpace(3)
    bounds = ob.RealVectorBounds(3)
    lo, hi = np.array(og.origin), np.array(og.origin) + np.array(og.grid.shape) * og.resolution
    for dim in range(3):
        bounds.setLow(dim, float(lo[dim]))
        bounds.setHigh(dim, float(hi[dim]))
    space.setBounds(bounds)

    si = ob.SpaceInformation(space)
    si.setStateValidityChecker(ValidityChecker(si))
    si.setStateValidityCheckingResolution(radius / max(hi - lo))
    si.setup()

    def _to_state(xyz):
        s = space.allocState()
        s[0], s[1], s[2] = float(xyz[0]), float(xyz[1]), float(xyz[2])
        return s

    pdef = ob.ProblemDefinition(si)
    pdef.setStartAndGoalStates(_to_state(start_xyz), _to_state(goal_xyz), 0.1)
    pdef.setOptimizationObjective(ob.PathLengthOptimizationObjective(si))

    planner = og_ompl.RRTstar(si)
    planner.setProblemDefinition(pdef)
    planner.setup()

    solved = planner.solve(timeout)
    if not solved:
        logger.warning("[RRT*] No solution found within %.1fs.", timeout)
        return None

    path = pdef.getSolutionPath()
    path.interpolate(max(10, int(path.length() / og.resolution)))

    states = path.getStates()
    waypoints = np.array([[s[0], s[1], s[2]] for s in states], dtype=np.float32)
    logger.info("[RRT*] Found path with %d waypoints.", len(waypoints))
    return waypoints


# ─── Hybrid planner ──────────────────────────────────────────────────────────

class HybridTrajectoryPlanner:
    """Per-robot trajectory planner combining cuRobo + RRT*.

    Parameters
    ----------
    motion_gen:
        Initialised ``cuRobo.MotionGen`` instance for this robot.
    og:
        :class:`~occupancy_grid.OccupancyGrid` for RRT* collision checking.
    robot_name:
        Label used in log messages.
    """

    def __init__(self, motion_gen, og, robot_name: str = "AUV"):
        self.motion_gen  = motion_gen
        self.og          = og
        self.robot_name  = robot_name

    # ------------------------------------------------------------------ API

    def plan_segment(
        self,
        start_js:   np.ndarray,      # (D,) joint angles
        goal_pose:  np.ndarray,      # (7,) [x, y, z, qw, qx, qy, qz]
        retract_js: Optional[np.ndarray] = None,
    ) -> SegmentResult:
        """Plan a single segment from joint state to target pose.

        Tries in order:
        1. cuRobo with dt-ladder.
        2. RRT* on occupancy grid → warm-start cuRobo.
        3. Raw RRT* path (position-only).

        Parameters
        ----------
        start_js:
            Current joint angles ``(D,)``.
        goal_pose:
            Target end-effector pose ``[x, y, z, qw, qx, qy, qz]``.
        retract_js:
            Optional retract/home configuration for cuRobo IK initialisation.
        """
        # ── Layer 1: cuRobo ───────────────────────────────────────────
        result = self._try_curobo(start_js, goal_pose, seed_traj=None)
        if result is not None:
            return result

        # ── Layer 2: RRT* warm-start ──────────────────────────────────
        start_xyz = self._fk_position(start_js)
        goal_xyz  = goal_pose[:3]

        rrt_path = _rrt_star_plan(start_xyz, goal_xyz, self.og)
        if rrt_path is not None:
            # Resample RRT* xyz to cuRobo horizon for warm-start
            seed_xyz = resample_positions_to_horizon(rrt_path, TRAJOPT_HORIZON)
            result = self._try_curobo(start_js, goal_pose,
                                      seed_traj=seed_xyz)
            if result is not None:
                result.planner = "curobo_warmed"
                return result

            # cuRobo still failed – return RRT* path as fallback
            positions = rrt_path
            if len(positions) < TRAJOPT_HORIZON:
                positions = resample_positions_to_horizon(positions, TRAJOPT_HORIZON)
            logger.warning(
                "[%s] cuRobo failed after warm-start; using raw RRT* path.",
                self.robot_name,
            )
            return SegmentResult(
                positions        = positions,
                joint_trajectory = None,
                success          = False,
                planner          = "rrt_star",
                dt               = None,
            )

        # ── Complete failure ──────────────────────────────────────────
        logger.error("[%s] All planners failed for goal %s.", self.robot_name, goal_pose)
        # Return straight-line path as last resort
        fallback = np.linspace(start_xyz, goal_xyz, TRAJOPT_HORIZON)
        return SegmentResult(
            positions        = fallback,
            joint_trajectory = None,
            success          = False,
            planner          = "fallback_linear",
            dt               = None,
        )

    # ---------------------------------------------------------------- private

    def _try_curobo(
        self,
        start_js:  np.ndarray,
        goal_pose: np.ndarray,
        seed_traj: Optional[np.ndarray],
    ) -> Optional[SegmentResult]:
        """Attempt cuRobo planning with the dt-ladder.

        Returns :class:`SegmentResult` on success, ``None`` on failure.
        """
        from curobo.types.math import Pose              # type: ignore
        from curobo.types.robot import JointState       # type: ignore
        from curobo.util_file import (                  # type: ignore
            TensorDeviceType,
        )
        from curobo.wrap.reacher.motion_gen import (    # type: ignore
            MotionGenPlanConfig,
        )

        goal_xyz = goal_pose[:3]
        goal_quat_wxyz = goal_pose[3:]    # [qw, qx, qy, qz]

        for dt in CUROBO_DT_LADDER:
            try:
                plan_cfg = MotionGenPlanConfig(
                    max_attempts                  = 1,
                    num_trajopt_seeds             = CUROBO_NUM_TRAJOPT_SEEDS,
                    num_graph_seeds               = CUROBO_NUM_GRAPH_SEEDS,
                    enable_graph                  = True,
                    timeout                       = 60.0,
                    enable_finetune_trajopt       = True,
                    finetune_trajopt_iters        = 300,
                    trajopt_dt                    = dt,
                    fix_terminal_action           = CUROBO_FIX_TERMINAL_ACTION,
                    reach_partial_pose            = True,
                    partial_pose_goal_weight      = CUROBO_POS_ONLY_WEIGHTS,
                )

                start_state = JointState.from_position(
                    torch.tensor(start_js, dtype=torch.float32).unsqueeze(0)
                )
                goal = Pose(
                    position    = torch.tensor(goal_xyz, dtype=torch.float32).unsqueeze(0),
                    quaternion  = torch.tensor(goal_quat_wxyz, dtype=torch.float32).unsqueeze(0),
                )

                if seed_traj is not None:
                    plan_cfg.seed_trajectory = torch.tensor(
                        seed_traj, dtype=torch.float32
                    ).unsqueeze(0)

                result = self.motion_gen.plan_single(start_state, goal, plan_cfg)

                if result.success.item():
                    traj_js = result.get_interpolated_plan()
                    positions = traj_js.position.squeeze(0).cpu().numpy()
                    xyz_traj = self._batch_fk(positions)
                    logger.info(
                        "[%s] cuRobo success (dt=%.3f, steps=%d).",
                        self.robot_name, dt, len(positions),
                    )
                    return SegmentResult(
                        positions        = xyz_traj,
                        joint_trajectory = positions,
                        success          = True,
                        planner          = "curobo",
                        dt               = dt,
                    )

            except Exception as exc:
                logger.debug("[%s] cuRobo dt=%.3f error: %s", self.robot_name, dt, exc)
                continue

        logger.warning("[%s] All cuRobo dt attempts failed.", self.robot_name)
        return None

    # ---------------------------------------------------------------- FK

    def _fk_position(self, js: np.ndarray) -> np.ndarray:
        """Return world-frame XYZ for a joint configuration."""
        try:
            from curobo.types.robot import JointState  # type: ignore

            state = JointState.from_position(
                torch.tensor(js, dtype=torch.float32).unsqueeze(0)
            )
            ee = self.motion_gen.kinematics.get_state(state)
            xyz = ee.ee_position.squeeze(0).cpu().numpy()
            return xyz
        except Exception:
            return np.zeros(3)

    def _batch_fk(self, positions_js: np.ndarray) -> np.ndarray:
        """Batch FK: ``(H, D)`` → ``(H, 3)`` EE XYZ."""
        try:
            from curobo.types.robot import JointState  # type: ignore

            state = JointState.from_position(
                torch.tensor(positions_js, dtype=torch.float32)
            )
            ee = self.motion_gen.kinematics.get_state(state)
            return ee.ee_position.cpu().numpy()
        except Exception:
            # Return interpolated fallback if FK fails
            n = len(positions_js)
            return np.zeros((n, 3))


# ─── Factory ─────────────────────────────────────────────────────────────────

def build_motion_gen(robot_cfg_path: str, world_cfg, tensor_args) -> object:
    """Construct and warmup a cuRobo ``MotionGen`` instance.

    Parameters
    ----------
    robot_cfg_path:
        Path to the robot YAML config file (brov.yml).
    world_cfg:
        cuRobo ``WorldConfig`` object with obstacle geometry.
    tensor_args:
        cuRobo ``TensorDeviceType`` for GPU allocation.
    """
    from curobo.types.base import TensorDeviceType               # type: ignore
    from curobo.wrap.reacher.motion_gen import (                # type: ignore
        MotionGen,
        MotionGenConfig,
    )

    mg_cfg = MotionGenConfig.load_from_robot_config(
        robot_cfg_path,
        world_cfg,
        tensor_args,
        trajopt_dt          = CUROBO_DT_LADDER[0],
        num_trajopt_seeds   = CUROBO_NUM_TRAJOPT_SEEDS,
        num_graph_seeds     = CUROBO_NUM_GRAPH_SEEDS,
        optimize_dt         = True,
    )
    mg = MotionGen(mg_cfg)
    mg.warmup()
    return mg
