"""
VRP Planner – Traffic-Light Collision Avoidance

Detects spatio-temporal conflicts between vehicle routes and resolves them
by injecting ``WAIT`` steps at the appropriate waypoints.

Algorithm
---------
1. For each route leg (edge from waypoint A → waypoint B) compute a time
   window ``[t_enter, t_exit]`` and a spatial corridor (bounding cylinder
   centred on the A* path).
2. Detect any pair of legs (different vehicles) whose time windows overlap
   **and** whose corridors intersect.
3. For each conflicting pair delay the *lower-priority* vehicle by inserting
   ``WAIT_STEPS_PER_WAYPOINT`` idle steps at its current hold waypoint.
4. Repeat until no conflicts remain (up to ``MAX_TRAFFIC_LIGHT_ROUNDS``).
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import List, Optional, Tuple

import numpy as np

from .config import (
    TRAFFIC_CORRIDOR_WIDTH,
    WAIT_STEPS_PER_WAYPOINT,
)

logger = logging.getLogger(__name__)

# Maximum re-resolution rounds to prevent infinite loops
MAX_TRAFFIC_LIGHT_ROUNDS = 10


# ─── Data types ──────────────────────────────────────────────────────────────

@dataclass
class Leg:
    """A single travel segment for one vehicle.

    Attributes
    ----------
    vehicle_idx:  Robot index.
    leg_idx:      Index of this leg within the vehicle's route.
    start_wp:     Start waypoint index in the global waypoints array.
    end_wp:       End waypoint index.
    path:         ``(M, 3)`` world-frame A* path for this leg (metres).
    t_start:      Simulation step at which travel begins.
    t_end:        Simulation step at which the vehicle arrives at ``end_wp``.
    """
    vehicle_idx: int
    leg_idx:     int
    start_wp:    int
    end_wp:      int
    path:        np.ndarray            # (M, 3)
    t_start:     float
    t_end:       float


@dataclass
class Conflict:
    """A detected spatio-temporal conflict between two legs."""
    vehicle_a:  int
    leg_a:      int
    vehicle_b:  int
    leg_b:      int
    overlap_t:  float   # magnitude of time overlap (steps)


@dataclass
class TrafficPlan:
    """Output of :func:`resolve_conflicts`.

    Attributes
    ----------
    routes:
        Per-vehicle list of waypoint indices (order preserved from VRP
        solution), with ``WAIT`` entries inserted where needed.
        ``WAIT`` is encoded as ``-1``.
    wait_counts:
        Per-vehicle list of wait-step counts *before* each waypoint
        (length = ``len(route)``).
    num_rounds:
        How many resolution rounds were needed.
    """
    routes:      List[List[int]]
    wait_counts: List[List[int]]
    num_rounds:  int = 0


# ─── Public API ──────────────────────────────────────────────────────────────

def build_legs(
    routes:          List[List[int]],
    waypoints_world: np.ndarray,
    steps_per_leg:   float,
    og,                              # OccupancyGrid – imported lazily to avoid circular
) -> List[List[Leg]]:
    """Compute :class:`Leg` objects for every edge in every route.

    Parameters
    ----------
    routes:
        Per-vehicle ordered waypoint index lists (from VRP solver).
    waypoints_world:
        ``(N, 3)`` or ``(N, 7)`` array of waypoint xyz (+ quaternion).
    steps_per_leg:
        Estimated number of simulation steps per waypoint leg; used to
        assign ``[t_start, t_end]`` windows.
    og:
        :class:`~occupancy_grid.OccupancyGrid`; used to compute A* paths.
    """
    from .gpu_distance_matrix import extract_astar_path  # lazy import

    xyz = waypoints_world[:, :3] if waypoints_world.ndim == 2 else waypoints_world

    all_legs: List[List[Leg]] = []
    for v_idx, route in enumerate(routes):
        legs: List[Leg] = []
        t_cursor = float(v_idx * WAIT_STEPS_PER_WAYPOINT)   # stagger starts
        full_route = route  # depot handling done upstream
        for l_idx, (wp_a, wp_b) in enumerate(zip(full_route[:-1], full_route[1:])):
            pos_a = xyz[wp_a]
            pos_b = xyz[wp_b]
            path  = extract_astar_path(og, pos_a, pos_b)
            if path is None or len(path) < 2:
                path = np.stack([pos_a, pos_b], axis=0)

            leg_dist = float(np.linalg.norm(path[-1] - path[0]))
            # Duration proportional to path length; minimum = steps_per_leg
            duration = max(steps_per_leg, leg_dist / og.resolution)

            legs.append(Leg(
                vehicle_idx = v_idx,
                leg_idx     = l_idx,
                start_wp    = wp_a,
                end_wp      = wp_b,
                path        = path.astype(np.float32),
                t_start     = t_cursor,
                t_end       = t_cursor + duration,
            ))
            t_cursor += duration + WAIT_STEPS_PER_WAYPOINT   # service time gap

        all_legs.append(legs)
    return all_legs


def detect_conflicts(all_legs: List[List[Leg]]) -> List[Conflict]:
    """Find all spatio-temporal conflicts across vehicles.

    A conflict exists when:
    * Two legs belong to *different* vehicles.
    * Their time windows ``[t_start, t_end]`` overlap.
    * Their spatial corridors (buffered approximation) intersect.
    """
    conflicts: List[Conflict] = []
    flat: List[Leg] = [leg for legs in all_legs for leg in legs]

    for i in range(len(flat)):
        for j in range(i + 1, len(flat)):
            la, lb = flat[i], flat[j]
            if la.vehicle_idx == lb.vehicle_idx:
                continue
            if not _time_windows_overlap(la, lb):
                continue
            if not _corridors_intersect(la, lb):
                continue
            overlap = min(la.t_end, lb.t_end) - max(la.t_start, lb.t_start)
            conflicts.append(Conflict(
                vehicle_a = la.vehicle_idx,
                leg_a     = la.leg_idx,
                vehicle_b = lb.vehicle_idx,
                leg_b     = lb.leg_idx,
                overlap_t = overlap,
            ))
    return conflicts


def resolve_conflicts(
    routes:          List[List[int]],
    waypoints_world: np.ndarray,
    steps_per_leg:   float,
    og,
) -> TrafficPlan:
    """Iteratively detect and resolve conflicts by inserting WAIT steps.

    Higher-indexed vehicles yield to lower-indexed ones (simple priority).

    Parameters
    ----------
    routes:
        Ordered waypoint index lists per vehicle.
    waypoints_world:
        ``(N ≥ 7)`` or ``(N, 3)`` world-frame positions.
    steps_per_leg:
        Estimated simulation steps per leg.
    og:
        Occupancy grid for path lookups.

    Returns
    -------
    :class:`TrafficPlan`
    """
    wait_counts: List[List[int]] = [[0] * len(r) for r in routes]

    # Step offsets per vehicle accumulated across rounds
    vehicle_delay = [0.0] * len(routes)

    for round_no in range(MAX_TRAFFIC_LIGHT_ROUNDS):
        all_legs = build_legs(routes, waypoints_world, steps_per_leg, og)

        # Apply accumulated delays to leg time windows
        for v_idx, legs in enumerate(all_legs):
            d = vehicle_delay[v_idx]
            for leg in legs:
                leg.t_start += d
                leg.t_end   += d

        conflicts = detect_conflicts(all_legs)
        if not conflicts:
            logger.info("[traffic_light] No conflicts after round %d.", round_no)
            break

        logger.info("[traffic_light] Round %d: %d conflicts detected.",
                    round_no, len(conflicts))

        # Resolve: delay the higher-index vehicle's problematic leg
        resolved = set()
        for c in conflicts:
            loser = max(c.vehicle_a, c.vehicle_b)
            if loser in resolved:
                continue
            wait_at = c.leg_a if loser == c.vehicle_a else c.leg_b
            vehicle_delay[loser] += WAIT_STEPS_PER_WAYPOINT
            if wait_at < len(wait_counts[loser]):
                wait_counts[loser][wait_at] += WAIT_STEPS_PER_WAYPOINT
            resolved.add(loser)

    else:
        logger.warning(
            "[traffic_light] Reached max rounds (%d); residual conflicts may remain.",
            MAX_TRAFFIC_LIGHT_ROUNDS,
        )

    return TrafficPlan(
        routes=routes,
        wait_counts=wait_counts,
        num_rounds=round_no + 1,
    )


# ─── Internal helpers ─────────────────────────────────────────────────────────

def _time_windows_overlap(la: Leg, lb: Leg) -> bool:
    """Return True iff the two time windows have any overlap."""
    return la.t_start < lb.t_end and lb.t_start < la.t_end


def _corridors_intersect(la: Leg, lb: Leg) -> bool:
    """Return True iff the two path corridors are within collision distance.

    Uses a simple min-distance check between polyline samples.
    """
    radius = TRAFFIC_CORRIDOR_WIDTH
    # Downsample paths to avoid O(M^2) for long paths
    path_a = _downsample_path(la.path, max_pts=20)
    path_b = _downsample_path(lb.path, max_pts=20)

    # Vectorised: compute all pairwise distances
    # path_a: (Mm, 3), path_b: (Mn, 3)
    diff = path_a[:, None, :] - path_b[None, :, :]   # (Mm, Mn, 3)
    dists = np.linalg.norm(diff, axis=-1)              # (Mm, Mn)
    return bool(dists.min() < radius)


def _downsample_path(path: np.ndarray, max_pts: int) -> np.ndarray:
    """Return at most ``max_pts`` evenly-spaced points from ``path``."""
    if len(path) <= max_pts:
        return path
    indices = np.linspace(0, len(path) - 1, max_pts, dtype=int)
    return path[indices]
