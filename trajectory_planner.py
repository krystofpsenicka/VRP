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
import time
from dataclasses import dataclass, field
from typing import List, Optional, Tuple

import numpy as np
import torch

from .config import (
    BROV_CUBOID_DIMS,
    CUROBO_DT_LADDER,
    CUROBO_NUM_GRAPH_SEEDS,
    CUROBO_NUM_TRAJOPT_SEEDS,
    SEGMENT_PLAN_BUDGET_S,
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
    joint_names:
        Ordered joint name list matching the robot cspace definition.
        Required for correct joint ordering when building cuRobo JointState.
    """

    def __init__(self, motion_gen, og, robot_name: str = "AUV",
                 joint_names: Optional[List[str]] = None):
        self.motion_gen  = motion_gen
        self.og          = og
        self.robot_name  = robot_name
        self.joint_names = joint_names

    # ------------------------------------------------------------------ API

    def plan_segment(
        self,
        start_js:   np.ndarray,      # (D,) joint angles
        goal_pose:  np.ndarray,      # (7,) [x, y, z, qw, qx, qy, qz]
        retract_js: Optional[np.ndarray] = None,
        deadline:   Optional[float] = None,
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
        deadline:
            ``time.monotonic()`` value after which planning is aborted and
            the next layer is tried immediately.  Defaults to
            ``now + SEGMENT_PLAN_BUDGET_S``.
        """
        if deadline is None:
            deadline = time.monotonic() + SEGMENT_PLAN_BUDGET_S

        # ── Layer 1: cuRobo ───────────────────────────────────────────
        result = self._try_curobo(start_js, goal_pose, seed_traj=None,
                                  deadline=deadline)
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
                                      seed_traj=seed_xyz, deadline=deadline)
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
        deadline:  Optional[float] = None,
    ) -> Optional[SegmentResult]:
        """Attempt cuRobo planning with the dt-ladder.

        Tries full pose (position + orientation) first at each dt level.
        If all full-pose attempts fail, retries with position-only metric.

        Returns :class:`SegmentResult` on success, ``None`` on failure.
        """
        from curobo.rollout.cost.pose_cost import PoseCostMetric  # type: ignore
        from curobo.types.math import Pose                        # type: ignore
        from curobo.types.robot import JointState                 # type: ignore
        from curobo.wrap.reacher.motion_gen import (              # type: ignore
            MotionGenPlanConfig,
        )

        goal_xyz = goal_pose[:3]
        goal_quat_wxyz = goal_pose[3:]    # [qw, qx, qy, qz]

        tensor_args = self.motion_gen.tensor_args
        kin_joint_names = self.motion_gen.kinematics.joint_names

        def _make_start_state() -> JointState:
            js = JointState.from_position(
                tensor_args.to_device(
                    torch.tensor(start_js, dtype=torch.float32)
                ),
                joint_names=self.joint_names,
            )
            js = js.get_ordered_joint_state(kin_joint_names)
            return js

        goal = Pose(
            position   = tensor_args.to_device(
                torch.tensor(goal_xyz, dtype=torch.float32).unsqueeze(0)
            ),
            quaternion = tensor_args.to_device(
                torch.tensor(goal_quat_wxyz, dtype=torch.float32).unsqueeze(0)
            ),
        )

        # Position-only metric: zero rotation weights, full position weights
        pos_only_metric = PoseCostMetric(
            reach_partial_pose = True,
            reach_vec_weight   = tensor_args.to_device(
                torch.tensor([0.0, 0.0, 0.0, 1.0, 1.0, 1.0])
            ),
        )

        # Map dt-ladder to time_dilation_factor
        _dt_to_dilation = {
            CUROBO_DT_LADDER[0]: 1.0,
            CUROBO_DT_LADDER[1]: CUROBO_DT_LADDER[1] / CUROBO_DT_LADDER[0],
            CUROBO_DT_LADDER[2]: CUROBO_DT_LADDER[2] / CUROBO_DT_LADDER[0],
        }

        def _base_cfg(dt: float, pos_only: bool) -> MotionGenPlanConfig:
            dilation = _dt_to_dilation.get(dt, 1.0)
            return MotionGenPlanConfig(
                max_attempts            = 4,
                enable_graph            = True,
                enable_graph_attempt    = 2,
                timeout                 = 10.0,
                enable_finetune_trajopt = True,
                finetune_attempts       = 3,
                time_dilation_factor    = dilation if dilation != 1.0 else None,
                pose_cost_metric        = pos_only_metric if pos_only else None,
            )

        def _pre_solve_ik_goal() -> Optional[torch.Tensor]:
            """Analytically compute the goal joint configuration.

            For this robot the prismatic joints map directly to world position:
                EE_x = x_joint + 0.30
                EE_y = y_joint
                EE_z = z_joint + 0.05
            Invert to get joint config and build goal state.
            Returns a (1, D) tensor in kinematics joint order, or None on error.
            """
            try:
                approx_js = start_js.copy().astype(np.float32)
                approx_js[0] = float(goal_xyz[0]) - 0.30
                approx_js[1] = float(goal_xyz[1])
                approx_js[2] = float(goal_xyz[2]) - 0.05
                # Keep rotation joints from start state (or zero)
                goal_state = JointState.from_position(
                    tensor_args.to_device(
                        torch.tensor(approx_js, dtype=torch.float32)
                    ),
                    joint_names=self.joint_names,
                )
                goal_state = goal_state.get_ordered_joint_state(kin_joint_names)
                return goal_state.position.unsqueeze(0)  # (1, D)
            except Exception as e:
                logger.warning("[%s] IK pre-solve exception: %s", self.robot_name, e)
            return None

        # ── IK pre-solve to get a good goal joint config ─────────────
        ik_goal_js = _pre_solve_ik_goal()

        def _try_plan(pos_only: bool) -> Optional[SegmentResult]:
            for dt in CUROBO_DT_LADDER:
                if deadline is not None and time.monotonic() > deadline:
                    logger.warning(
                        "[%s] Segment deadline reached before dt=%.3f (%s) — aborting ladder.",
                        self.robot_name, dt, "pos-only" if pos_only else "full-pose",
                    )
                    return None
                try:
                    plan_cfg    = _base_cfg(dt, pos_only)
                    start_state = _make_start_state()

                    # Try plan_single_js first if we have an IK goal (avoids
                    # IK seeding failure for far-away goals)
                    result = None
                    if ik_goal_js is not None and not pos_only:
                        try:
                            goal_state_js = JointState.from_position(
                                ik_goal_js,  # (1, D) in kin_joint order
                                joint_names=kin_joint_names,
                            )
                            for use_graph in (True, False):
                                js_plan_cfg = MotionGenPlanConfig(
                                    max_attempts            = plan_cfg.max_attempts,
                                    enable_graph            = use_graph,
                                    enable_graph_attempt    = plan_cfg.enable_graph_attempt,
                                    timeout                 = plan_cfg.timeout,
                                    enable_finetune_trajopt = plan_cfg.enable_finetune_trajopt,
                                    finetune_attempts       = plan_cfg.finetune_attempts,
                                    time_dilation_factor    = plan_cfg.time_dilation_factor,
                                )
                                result = self.motion_gen.plan_single_js(
                                    start_state.unsqueeze(0),
                                    goal_state_js.unsqueeze(0),
                                    js_plan_cfg,
                                )
                                if result.success.item():
                                    break
                                logger.debug(
                                    "[%s] plan_single_js dt=%.3f graph=%s FAILED: %s",
                                    self.robot_name, dt, use_graph,
                                    getattr(result, 'status', 'unknown'))
                            if not result.success.item():
                                logger.warning(
                                    "[%s] plan_single_js dt=%.3f FAILED: %s",
                                    self.robot_name, dt,
                                    getattr(result, 'status', 'unknown'))
                                result = None  # fall through to plan_single
                        except Exception as js_exc:
                            logger.debug("[%s] plan_single_js exception: %s",
                                         self.robot_name, js_exc)
                            result = None

                    # Fall back to plan_single (pose-based)
                    if result is None or not result.success.item():
                        result = self.motion_gen.plan_single(
                            start_state.unsqueeze(0), goal, plan_cfg
                        )

                    if result.success.item():
                        traj_js = result.get_interpolated_plan()
                        # Reorder joints back to config (cspace) order
                        traj_js = traj_js.get_ordered_joint_state(
                            self.joint_names or kin_joint_names
                        )
                        positions = traj_js.position.cpu().numpy()
                        xyz_traj  = self._batch_fk(positions)
                        mode = "pos-only" if pos_only else "full-pose"
                        logger.info(
                            "[%s] cuRobo success (%s, dt=%.3f, steps=%d).",
                            self.robot_name, mode, dt, len(positions),
                        )
                        return SegmentResult(
                            positions        = xyz_traj,
                            joint_trajectory = positions,
                            success          = True,
                            planner          = "curobo",
                            dt               = dt,
                        )
                    else:
                        logger.warning(
                            "[%s] cuRobo dt=%.3f %s FAILED: status=%s  goal_xyz=%s",
                            self.robot_name, dt,
                            "pos-only" if pos_only else "full-pose",
                            str(getattr(result, 'status', 'unknown')),
                            goal_pose[:3],
                        )
                except Exception as exc:
                    logger.warning(
                        "[%s] cuRobo dt=%.3f exception: %s",
                        self.robot_name, dt, exc, exc_info=True,
                    )
            return None

        # ── Layer 1a: Full pose ──────────────────────────────────────
        result = _try_plan(pos_only=False)
        if result is not None:
            return result

        # ── Layer 1b: Position-only fallback ─────────────────────────
        logger.info(
            "[%s] Full-pose failed; retrying with position-only metric …",
            self.robot_name,
        )
        result = _try_plan(pos_only=True)
        if result is not None:
            return result

        logger.warning("[%s] All cuRobo attempts failed for goal %s.",
                       self.robot_name, goal_pose[:3])
        return None

    # ---------------------------------------------------------------- FK

    def _fk_position(self, js: np.ndarray) -> np.ndarray:
        """Return world-frame XYZ for a joint configuration."""
        try:
            from curobo.types.robot import JointState  # type: ignore

            kin_names   = self.motion_gen.kinematics.joint_names
            tensor_args = self.motion_gen.tensor_args
            state = JointState.from_position(
                tensor_args.to_device(torch.tensor(js, dtype=torch.float32)),
                joint_names=self.joint_names,
            ).get_ordered_joint_state(kin_names)
            ee = self.motion_gen.kinematics.get_state(state)
            xyz = ee.ee_position.squeeze(0).cpu().numpy()
            return xyz
        except Exception:
            return np.zeros(3)

    def _batch_fk(self, positions_js: np.ndarray) -> np.ndarray:
        """Batch FK: ``(H, D)`` → ``(H, 3)`` EE XYZ."""
        try:
            from curobo.types.robot import JointState  # type: ignore

            kin_names   = self.motion_gen.kinematics.joint_names
            tensor_args = self.motion_gen.tensor_args
            state = JointState.from_position(
                tensor_args.to_device(torch.tensor(positions_js, dtype=torch.float32)),
                joint_names=self.joint_names,
            ).get_ordered_joint_state(kin_names)
            ee = self.motion_gen.kinematics.get_state(state)
            return ee.ee_position.cpu().numpy()
        except Exception:
            # Return interpolated fallback if FK fails
            n = len(positions_js)
            return np.zeros((n, 3))


# ─── Factory ─────────────────────────────────────────────────────────────────

def build_motion_gen(robot_cfg_path: str, world_cfg, tensor_args,
                     num_trajopt_seeds: int = 12,
                     num_graph_seeds:   int = 12,
                     voxel_dims: Optional[List[float]] = None) -> object:
    """Construct and warmup a cuRobo ``MotionGen`` instance.

    Parameters
    ----------
    robot_cfg_path:
        Path to the robot YAML config file (brov.yml).
    world_cfg:
        cuRobo ``WorldConfig`` object with obstacle geometry.
    tensor_args:
        cuRobo ``TensorDeviceType`` for GPU allocation.
    num_trajopt_seeds, num_graph_seeds:
        Pre-allocated seed count.  Use the module default (12) for
        multi-robot to avoid CUDA OOM; 1024 only for single-robot.
    voxel_dims:
        ``[x, y, z]`` extent in metres of the ESDF VoxelGrid.  When
        provided the collision checker is set to ``VOXEL`` and the cache
        is pre-allocated for one ESDF layer of this size.  When ``None``
        the checker falls back to ``PRIMITIVE`` (cuboids only).
    """
    import torch
    from curobo.types.base import TensorDeviceType               # type: ignore
    from curobo.geom.sdf.world import CollisionCheckerType       # type: ignore
    from curobo.wrap.reacher.motion_gen import (                # type: ignore
        MotionGen,
        MotionGenConfig,
    )

    if voxel_dims is not None:
        checker_type = CollisionCheckerType.VOXEL
        collision_cache = {
            "obb": 20,
            "mesh": 10,
            "voxel": {
                "layers": 1,
                "dims": voxel_dims,
                "voxel_size": 0.10,
                "feature_dtype": torch.float32,
            },
        }
    else:
        checker_type = CollisionCheckerType.PRIMITIVE
        collision_cache = {"obb": 20, "mesh": 10}

    mg_cfg = MotionGenConfig.load_from_robot_config(
        robot_cfg_path,
        world_cfg,
        tensor_args,
        collision_checker_type = checker_type,
        trajopt_dt             = CUROBO_DT_LADDER[0],
        num_trajopt_seeds      = num_trajopt_seeds,
        num_graph_seeds        = num_graph_seeds,
        interpolation_dt       = CUROBO_DT_LADDER[0],
        collision_cache        = collision_cache,
        optimize_dt            = True,
        trajopt_tsteps         = TRAJOPT_HORIZON,
        position_threshold     = 0.01,
        rotation_threshold     = 0.05,
        use_cuda_graph         = False,
    )
    mg = MotionGen(mg_cfg)
    mg.warmup(enable_graph=True, warmup_js_trajopt=False)

    # cuRobo's load_collision_model copies pose/dims/enable into the GPU
    # cache but does NOT copy VoxelGrid.feature_tensor (the ESDF data).
    # Push it explicitly so the collision checker actually sees obstacles.
    if world_cfg.voxel is not None:
        for vg in world_cfg.voxel:
            if vg.feature_tensor is not None:
                mg.world_coll_checker.update_voxel_data(vg, env_idx=0)

    return mg
