"""
VRP Planner – Space-Time A* for Multi-Robot Collision Avoidance
================================================================

Priority-Based Sequential Planning: robots are planned one at a time in
priority order.  Each robot's route legs are planned with A* on a **coarse**
4-D ``(x, y, z, t)`` graph so that the resulting trajectory avoids:

  • Static obstacles (down-sampled occupancy grid), and
  • Previously committed robots' trajectories (reservation table).

Resolution strategy
-------------------
The occupancy grid (0.10 m) is far too fine for a dense 4-D reservation
table.  We down-sample to ``SPACE_TIME_RESOLUTION`` (default 0.50 m) so
the dense ``(T, Nx, Ny, Nz)`` uint8 table consumes ~80 MB.  At 0.50 m
resolution with ``SPACE_TIME_DT = 0.25 s`` the robot moves **one coarse
voxel per time step** at 2 m/s cruise speed.

Implementation notes
--------------------
* Priority queue: :mod:`heapq`  (CPython C implementation).
* Occupancy / reservation lookups: NumPy uint8 arrays (C-backed O(1)).
* 26-connected spatial moves **+** wait action → 27 successors per node.
* g-scores: Python dict keyed on ``(x, y, z, t)`` – only *expanded*
  nodes are stored, so memory is bounded by the small search frontier.
"""

from __future__ import annotations

import heapq
import logging
import math
from typing import Dict, List, Optional, Tuple

import numpy as np

from .config import (
    BROV_CUBOID_DIMS,
    CAMERA_OFFSET_FORWARD,
    CAMERA_OFFSET_UP,
    SPACE_TIME_DT,
    SPACE_TIME_DWELL_S,
    SPACE_TIME_MAX_HORIZON_S,
    SPACE_TIME_MAX_WAIT,
    SPACE_TIME_RESOLUTION,
)

logger = logging.getLogger(__name__)


# ── Camera offset helper ──────────────────────────────────────────────────

def _robot_xyz_from_waypoint(wp7: np.ndarray) -> np.ndarray:
    """Return robot body-centre XYZ for a 7-DOF waypoint [x,y,z,qw,qx,qy,qz].

    Inspection waypoints encode the desired *camera* position and viewing
    direction.  The robot body centre must be offset backward so that the
    camera optical frame arrives at the waypoint position.

    For home nodes (identity quaternion) the position is returned unchanged
    because those coordinates already represent the robot body centre.
    """
    xyz = np.array(wp7[:3], dtype=np.float64)
    qw, qx, qy, qz = float(wp7[3]), float(wp7[4]), float(wp7[5]), float(wp7[6])
    # Identity quaternion → home/depot node, no offset needed
    if abs(qw - 1.0) < 1e-3 and abs(qx) < 1e-3 and abs(qy) < 1e-3 and abs(qz) < 1e-3:
        return xyz
    # Rotate +X by quaternion to get camera forward direction
    fx = 1.0 - 2.0 * (qy * qy + qz * qz)
    fy = 2.0 * (qx * qy + qw * qz)
    # Robot body is CAMERA_OFFSET_FORWARD metres *behind* the camera along fwd
    xyz[0] -= CAMERA_OFFSET_FORWARD * fx
    xyz[1] -= CAMERA_OFFSET_FORWARD * fy
    xyz[2] -= CAMERA_OFFSET_UP
    return xyz


# ── 26-connected spatial offsets + wait action ───────────────────────────
_SPATIAL_OFFSETS: List[Tuple[int, int, int]] = [
    (di, dj, dk)
    for di in (-1, 0, 1)
    for dj in (-1, 0, 1)
    for dk in (-1, 0, 1)
    if not (di == 0 and dj == 0 and dk == 0)
]
_SPATIAL_WEIGHTS: np.ndarray = np.array(
    [math.sqrt(di * di + dj * dj + dk * dk) for di, dj, dk in _SPATIAL_OFFSETS],
    dtype=np.float64,
)
# Append the *wait* action: (0, 0, 0) with zero spatial cost
_OFFSETS_27 = _SPATIAL_OFFSETS + [(0, 0, 0)]
_WEIGHTS_27 = np.append(_SPATIAL_WEIGHTS, 0.0)


# ── Grid down-sampling ───────────────────────────────────────────────────────

def downsample_occupancy_grid(
    fine_grid: np.ndarray,
    fine_origin: np.ndarray,
    fine_res: float,
    coarse_res: float = SPACE_TIME_RESOLUTION,
) -> Tuple[np.ndarray, np.ndarray, float]:
    """Down-sample an occupancy grid for Space-Time A*.

    A coarse voxel is **occupied** if **any** of its constituent fine
    voxels is occupied (conservative – no false free-space).

    Fully vectorised with ``np.pad`` + ``reshape`` + ``any``.

    Returns ``(coarse_grid, coarse_origin, coarse_res)``.
    """
    factor = max(1, int(round(coarse_res / fine_res)))
    Fx, Fy, Fz = fine_grid.shape

    # Pad each axis to the next multiple of *factor* so reshape is exact.
    # Padding with False keeps boundary voxels conservative (only real
    # fine-voxels contribute to the any() test).
    pad_x = (-Fx) % factor
    pad_y = (-Fy) % factor
    pad_z = (-Fz) % factor
    if pad_x or pad_y or pad_z:
        fine_padded = np.pad(
            fine_grid.astype(bool),
            [(0, pad_x), (0, pad_y), (0, pad_z)],
            constant_values=False,
        )
    else:
        fine_padded = fine_grid.astype(bool)

    Cx = fine_padded.shape[0] // factor
    Cy = fine_padded.shape[1] // factor
    Cz = fine_padded.shape[2] // factor

    coarse = (
        fine_padded
        .reshape(Cx, factor, Cy, factor, Cz, factor)
        .any(axis=(1, 3, 5))
    )

    coarse_origin = fine_origin.copy()
    actual_res = fine_res * factor
    logger.info(
        "[ST-A*] Downsampled grid: %s → %s  (factor=%d, coarse_res=%.2fm)",
        fine_grid.shape, coarse.shape, factor, actual_res,
    )
    return coarse, coarse_origin, actual_res


# ── Reservation table ────────────────────────────────────────────────────────

class ReservationTable:
    """Dense 4-D ``(T, Nx, Ny, Nz)`` table stored as a flat NumPy uint8
    array.  Every lookup is a single C-level array index – no Python
    ``dict`` or ``set`` overhead in the hot loop.

    Parameters
    ----------
    grid_shape : (Nx, Ny, Nz)
        Coarse spatial dimensions.
    max_time_steps : int
        Number of discrete time slots.
    robot_half_extents_voxels : (3,) int
        Per-axis half-width of the robot AABB measured in **coarse**
        voxels.  Every commit inflates the point trajectory by this box.
    """

    def __init__(
        self,
        grid_shape: Tuple[int, int, int],
        max_time_steps: int,
        robot_half_extents_voxels: np.ndarray,
    ):
        self.Nx, self.Ny, self.Nz = grid_shape
        self.T = max_time_steps
        size = self.T * self.Nx * self.Ny * self.Nz
        self._data = np.zeros(size, dtype=np.uint8)
        self._half = np.asarray(robot_half_extents_voxels, dtype=np.intp)
        self._stride_t = self.Nx * self.Ny * self.Nz
        self._stride_x = self.Ny * self.Nz
        self._stride_y = self.Nz

    # ── fast point query (called ~100 k× per A* leg) ────────────────────
    def is_reserved(self, x: int, y: int, z: int, t: int) -> bool:
        if t >= self.T or t < 0:
            return False
        idx = t * self._stride_t + x * self._stride_x + y * self._stride_y + z
        return bool(self._data[idx])

    # ── bulk commit ──────────────────────────────────────────────────────
    def commit_trajectory(
        self,
        positions_ijk: np.ndarray,
        time_steps: np.ndarray,
    ) -> None:
        """Mark all AABB voxels around each ``(t, cx, cy, cz)`` sample.

        Loops over the fixed-size AABB offsets (small, e.g. 3×3×3 = 27)
        and vectorises across all *K* path samples within each offset.
        """
        hx, hy, hz = int(self._half[0]), int(self._half[1]), int(self._half[2])
        K = len(positions_ijk)
        if K == 0:
            return

        ts  = np.asarray(time_steps, dtype=np.intp)
        pos = np.asarray(positions_ijk, dtype=np.intp)

        # Drop samples beyond the time horizon
        valid = ts < self.T
        if not valid.any():
            return
        ts  = ts[valid]
        pos = pos[valid]
        cx, cy, cz = pos[:, 0], pos[:, 1], pos[:, 2]

        st_t = self._stride_t
        st_x = self._stride_x
        st_y = self._stride_y

        for dx in range(-hx, hx + 1):
            xi = cx + dx
            mx = (xi >= 0) & (xi < self.Nx)
            for dy in range(-hy, hy + 1):
                yi = cy + dy
                mxy = mx & (yi >= 0) & (yi < self.Ny)
                for dz in range(-hz, hz + 1):
                    zi = cz + dz
                    m = mxy & (zi >= 0) & (zi < self.Nz)
                    if not m.any():
                        continue
                    idx = ts[m] * st_t + xi[m] * st_x + yi[m] * st_y + zi[m]
                    self._data[idx] = 1


# ── Coordinate helpers ───────────────────────────────────────────────────────

def world_to_coarse(xyz: np.ndarray, origin: np.ndarray, res: float) -> np.ndarray:
    """World-frame XYZ → coarse-grid voxel index (floor)."""
    return np.floor((np.asarray(xyz) - origin) / res).astype(np.intp)


def coarse_to_world(ijk: np.ndarray, origin: np.ndarray, res: float) -> np.ndarray:
    """Coarse voxel index → world-frame XYZ (voxel centre)."""
    return np.asarray(ijk, dtype=np.float64) * res + origin + res * 0.5


def _snap_coarse_to_free(grid: np.ndarray, ijk: np.ndarray) -> np.ndarray:
    """BFS-snap each voxel index to the nearest free coarse voxel."""
    from collections import deque

    Nx, Ny, Nz = grid.shape
    out = ijk.copy()
    for row in range(len(ijk)):
        ix, iy, iz = int(ijk[row, 0]), int(ijk[row, 1]), int(ijk[row, 2])
        ix = max(0, min(ix, Nx - 1))
        iy = max(0, min(iy, Ny - 1))
        iz = max(0, min(iz, Nz - 1))
        if not grid[ix, iy, iz]:
            out[row] = [ix, iy, iz]
            continue
        visited = set()
        q = deque()
        q.append((ix, iy, iz))
        visited.add((ix, iy, iz))
        found = False
        while q and not found:
            cx, cy, cz = q.popleft()
            for di, dj, dk in _SPATIAL_OFFSETS:
                nx, ny, nz = cx + di, cy + dj, cz + dk
                if not (0 <= nx < Nx and 0 <= ny < Ny and 0 <= nz < Nz):
                    continue
                if (nx, ny, nz) in visited:
                    continue
                visited.add((nx, ny, nz))
                if not grid[nx, ny, nz]:
                    out[row] = [nx, ny, nz]
                    found = True
                    break
                q.append((nx, ny, nz))
        if not found:
            logger.warning("[snap] Could not free voxel %s", ijk[row])
            out[row] = [ix, iy, iz]
    return out


# ── Space-Time A* ────────────────────────────────────────────────────────────

def space_time_astar(
    grid: np.ndarray,
    start_ijk: np.ndarray,
    goal_ijk: np.ndarray,
    t_start: int,
    reservation: ReservationTable,
    resolution: float,
    time_step_cost: float = 0.01,
    max_time_steps: int = 0,
    max_expansions: int = 500_000,
) -> Optional[Tuple[np.ndarray, np.ndarray]]:
    """Space-Time A* on a 26-connected coarse 3-D grid + wait.

    Returns ``(path_ijk, path_t)`` or ``None`` on failure.
    """
    Nx, Ny, Nz = grid.shape
    T_max = max_time_steps if max_time_steps > 0 else reservation.T

    sx, sy, sz = int(start_ijk[0]), int(start_ijk[1]), int(start_ijk[2])
    gx, gy, gz = int(goal_ijk[0]), int(goal_ijk[1]), int(goal_ijk[2])

    if grid[sx, sy, sz] or grid[gx, gy, gz]:
        logger.warning("[ST-A*] start or goal in obstacle.")
        return None

    if (sx, sy, sz) == (gx, gy, gz):
        return (
            np.array([[sx, sy, sz]], dtype=np.intp),
            np.array([t_start], dtype=np.intp),
        )

    def _h(x: int, y: int, z: int) -> float:
        dx = x - gx; dy = y - gy; dz = z - gz
        return math.sqrt(dx * dx + dy * dy + dz * dz) * resolution

    start_state = (sx, sy, sz, t_start)
    g: Dict[Tuple[int, int, int, int], float] = {start_state: 0.0}
    came_from: Dict[Tuple[int, int, int, int], Tuple[int, int, int, int]] = {}

    open_set: list = []
    heapq.heappush(open_set, (_h(sx, sy, sz), sx, sy, sz, t_start))
    _is_reserved = reservation.is_reserved

    expansions = 0
    while open_set:
        f, cx, cy, cz, ct = heapq.heappop(open_set)
        state = (cx, cy, cz, ct)

        if cx == gx and cy == gy and cz == gz:
            # Reconstruct
            path: list = []
            s = state
            while s in came_from:
                path.append(s)
                s = came_from[s]
            path.append(start_state)
            path.reverse()
            return (
                np.array([(s[0], s[1], s[2]) for s in path], dtype=np.intp),
                np.array([s[3] for s in path], dtype=np.intp),
            )

        cur_g = g.get(state)
        if cur_g is None or f - _h(cx, cy, cz) > cur_g + 1e-9:
            continue

        expansions += 1
        if expansions >= max_expansions:
            logger.warning("[ST-A*] Max expansions (%d) reached.", max_expansions)
            break

        nt = ct + 1
        if nt >= T_max:
            continue

        for idx in range(27):
            di, dj, dk = _OFFSETS_27[idx]
            nx, ny, nz = cx + di, cy + dj, cz + dk
            if not (0 <= nx < Nx and 0 <= ny < Ny and 0 <= nz < Nz):
                continue
            if grid[nx, ny, nz]:
                continue
            if _is_reserved(nx, ny, nz, nt):
                continue

            w = float(_WEIGHTS_27[idx])
            new_g = cur_g + w * resolution + time_step_cost

            nbr = (nx, ny, nz, nt)
            old_g = g.get(nbr)
            if old_g is None or new_g < old_g:
                g[nbr] = new_g
                came_from[nbr] = state
                heapq.heappush(open_set, (new_g + _h(nx, ny, nz), nx, ny, nz, nt))

    logger.warning("[ST-A*] No path found (start=%s goal=%s t=%d, %d exp.).",
                   start_ijk, goal_ijk, t_start, expansions)
    return None


# ── OMPL path simplification ────────────────────────────────────────────────

def simplify_path_ompl(
    path_xyz: np.ndarray,
    og,
    robot_radius: float = 0.35,
    max_time: float = 0.5,
) -> np.ndarray:
    """Simplify a 3-D path using OMPL ``PathSimplifier``.

    Applies ``shortcutPath()`` to remove grid-aligned detours followed by
    ``smoothBSpline()`` to round corners.  The result is guaranteed
    obstacle-free with respect to the fine-resolution occupancy grid.

    Parameters
    ----------
    path_xyz : (M, 3)
        World-frame path waypoints (e.g. from coarse A* converted to
        world coordinates).
    og : OccupancyGrid
        Fine-resolution occupancy grid for collision checking.
    robot_radius : float
        Robot bounding-sphere radius for validity check resolution.
    max_time : float
        Time budget (seconds) for the shortcut simplification step.

    Returns
    -------
    (K, 3) simplified path waypoints (K ≤ M).  Falls back to a copy of
    *path_xyz* when OMPL is unavailable or the path is too short.
    """
    if len(path_xyz) < 3:
        return path_xyz.copy()

    try:
        import ompl.base as ob
        import ompl.geometric as og_ompl
    except ImportError:
        logger.warning("[simplify] OMPL not available – returning original path.")
        return path_xyz.copy()

    # -- state space --
    class _Checker(ob.StateValidityChecker):
        def __init__(self, si):
            super().__init__(si)

        def isValid(self, state):  # noqa: N802
            xyz = np.array([state[0], state[1], state[2]])
            return bool(og.is_free_world(xyz))

    space = ob.RealVectorStateSpace(3)
    bounds = ob.RealVectorBounds(3)
    lo = np.asarray(og.origin, dtype=np.float64)
    hi = lo + np.asarray(og.grid.shape, dtype=np.float64) * og.resolution
    for dim in range(3):
        bounds.setLow(dim, float(lo[dim]))
        bounds.setHigh(dim, float(hi[dim]))
    space.setBounds(bounds)

    si = ob.SpaceInformation(space)
    si.setStateValidityChecker(_Checker(si))
    si.setStateValidityCheckingResolution(
        float(robot_radius / max(hi - lo))
    )
    si.setup()

    # -- build geometric path from waypoints --
    path = og_ompl.PathGeometric(si)
    for pt in path_xyz:
        s = space.allocState()
        s[0], s[1], s[2] = float(pt[0]), float(pt[1]), float(pt[2])
        path.append(s)

    # -- simplify: removes detours + rounds corners --
    simplifier = og_ompl.PathSimplifier(si)
    simplifier.simplifyMax(path)

    states = path.getStates()
    result = np.array([[s[0], s[1], s[2]] for s in states], dtype=np.float64)
    logger.info("[simplify] OMPL: %d → %d waypoints", len(path_xyz), len(result))
    return result


def _arc_length_resample(path_xyz: np.ndarray, n_samples: int) -> np.ndarray:
    """Resample a 3-D path to *n_samples* points at uniform arc-length."""
    if len(path_xyz) < 2 or n_samples < 1:
        return np.tile(path_xyz[0], (max(n_samples, 1), 1))
    diffs = np.diff(path_xyz, axis=0)
    seg_lens = np.linalg.norm(diffs, axis=1)
    cum_len = np.concatenate([[0.0], np.cumsum(seg_lens)])
    total_len = cum_len[-1]
    if total_len < 1e-9:
        return np.tile(path_xyz[0], (n_samples, 1))
    target_s = np.linspace(0.0, total_len, n_samples)
    return np.column_stack([
        np.interp(target_s, cum_len, path_xyz[:, d]) for d in range(3)
    ])


# ── High-level: plan a single robot's full route ────────────────────────────

def plan_robot_route_st(
    coarse_grid: np.ndarray,
    coarse_origin: np.ndarray,
    coarse_res: float,
    reservation: ReservationTable,
    route: List[int],
    waypoints_world: np.ndarray,
    path_cache: Optional[dict] = None,
    dwell_s: float = SPACE_TIME_DWELL_S,
    dt: float = SPACE_TIME_DT,
    fine_og=None,
    robot_radius: float = 0.35,
) -> Tuple[np.ndarray, np.ndarray]:
    """Plan one robot through its full VRP route using Space-Time A*.

    For each leg:

      1. Plan hop-by-hop with :func:`space_time_astar`.
      2. Smooth the leg's spatial path with OMPL ``PathSimplifier`` (when
         *fine_og* is provided) to remove grid-aligned zig-zags.
      3. Re-sample the smoothed path at the original coarse time steps
         (arc-length parameterisation preserves the A* time span).
      4. Safety-check the smoothed path against the reservation table;
         fall back to the raw A* path on conflict.

    After all legs the entire trajectory is committed to the reservation
    table in one batch so later robots plan around it.

    Parameters
    ----------
    fine_og : OccupancyGrid, optional
        Fine-resolution occupancy grid passed to :func:`simplify_path_ompl`.
        If ``None`` OMPL smoothing is skipped.
    robot_radius : float
        Bounding-sphere radius for OMPL validity checks.

    Returns
    -------
    ``(world_xyz, coarse_time_steps, wp_schedule)`` where
    ``world_xyz`` is an ``(M, 3)`` world-frame XYZ array,
    ``coarse_time_steps`` is an ``(M,)`` coarse time-step index array, and
    ``wp_schedule`` is a list of ``(t_dwell_start, t_dwell_end, node_idx)``
    tuples recording when (in coarse time) each VRP waypoint is dwelled at.
    """
    wp_arr = np.asarray(waypoints_world)
    # Compute robot body positions accounting for camera offset
    xyz_all = np.array([_robot_xyz_from_waypoint(wp_arr[i]) for i in range(len(wp_arr))])
    coarse_positions: list = []   # list of (k, 3) voxel-index arrays
    world_positions: list = []    # list of (k, 3) world-frame XYZ arrays (OMPL-smooth)
    coarse_times: list = []       # list of (k,) int arrays
    wp_schedule: list = []        # list of (t_dwell_start, t_dwell_end, node_idx)
    t_cursor = 0

    hold_steps = max(1, int(round(dwell_s / dt)))

    for leg in range(1, len(route)):
        prev_node = route[leg - 1]
        curr_node = route[leg]

        # Get fine-resolution cached path if available
        if path_cache is not None and (prev_node, curr_node) in path_cache:
            leg_xyz = path_cache[(prev_node, curr_node)].copy()
            # Override endpoints with body-centre positions (cache stores camera pos)
            leg_xyz[0] = xyz_all[prev_node]
            leg_xyz[-1] = xyz_all[curr_node]
        else:
            leg_xyz = np.vstack([xyz_all[prev_node], xyz_all[curr_node]])

        # Convert to coarse voxel indices → sub-sample every N voxels
        leg_ijk = world_to_coarse(leg_xyz, coarse_origin, coarse_res)
        # Remove consecutive duplicates
        keep = np.ones(len(leg_ijk), dtype=bool)
        keep[1:] = np.any(leg_ijk[1:] != leg_ijk[:-1], axis=1)
        leg_ijk = leg_ijk[keep]
        # Clip to grid bounds
        for d, mx in enumerate(coarse_grid.shape):
            leg_ijk[:, d] = np.clip(leg_ijk[:, d], 0, mx - 1)
        # Snap occupied voxels to nearest free
        leg_ijk = _snap_coarse_to_free(coarse_grid, leg_ijk)

        # Sub-sample: keep every Kth waypoint to reduce the number of
        # Space-Time A* calls (each hop is short enough to plan quickly)
        stride_voxels = max(1, int(round(2.0 / coarse_res)))  # ~2 m hops
        sub_idx = list(range(0, len(leg_ijk), stride_voxels))
        if sub_idx[-1] != len(leg_ijk) - 1:
            sub_idx.append(len(leg_ijk) - 1)
        sub_ijk = leg_ijk[sub_idx]

        # ── A* planning for every hop of this leg ─────────────────────
        leg_plan_pos: list = []   # per-hop ijk arrays
        leg_plan_t: list = []     # per-hop time arrays

        for hop in range(1, len(sub_ijk)):
            s_ijk = sub_ijk[hop - 1]
            g_ijk = sub_ijk[hop]
            if np.array_equal(s_ijk, g_ijk):
                continue

            result = space_time_astar(
                coarse_grid, s_ijk, g_ijk, t_cursor, reservation, coarse_res,
            )

            if result is None:
                logger.warning(
                    "  [route] ST-A* failed hop %d→%d (leg %d→%d, t=%d). "
                    "Using straight-line fallback.",
                    hop - 1, hop, prev_node, curr_node, t_cursor,
                )
                n = max(1, int(np.linalg.norm(g_ijk - s_ijk)))
                fb_ijk = np.stack([
                    np.linspace(int(s_ijk[d]), int(g_ijk[d]), n + 1).astype(np.intp)
                    for d in range(3)
                ], axis=1)
                fb_t = np.arange(t_cursor, t_cursor + len(fb_ijk), dtype=np.intp)
                result = (fb_ijk, fb_t)

            seg_ijk, seg_t = result
            # Skip first point (duplicate) unless this is the very first
            # segment of the entire route.
            start_idx = 0 if (not coarse_positions and not leg_plan_pos) else 1
            if len(seg_ijk) > start_idx:
                leg_plan_pos.append(seg_ijk[start_idx:])
                leg_plan_t.append(seg_t[start_idx:])
            t_cursor = int(seg_t[-1])

        # -- nothing planned for this leg → dwell only -----------------
        if not leg_plan_pos:
            if coarse_positions:
                last = coarse_positions[-1][-1]
                last_world = world_positions[-1][-1]
                dwell_ijk = np.tile(last, (hold_steps, 1))
                dwell_world = np.tile(last_world, (hold_steps, 1))
                dwell_t = np.arange(
                    t_cursor + 1, t_cursor + hold_steps + 1, dtype=np.intp,
                )
                coarse_positions.append(dwell_ijk)
                world_positions.append(dwell_world)
                coarse_times.append(dwell_t)
                t_dwell_start = int(t_cursor) + 1
                t_cursor += hold_steps
                wp_schedule.append((t_dwell_start, int(t_cursor), curr_node))
            continue

        planned_ijk = np.concatenate(leg_plan_pos, axis=0)
        planned_t   = np.concatenate(leg_plan_t, axis=0)

        # ── OMPL smoothing (before commit) ────────────────────────────
        final_ijk = planned_ijk          # default: use raw A* path
        final_world = coarse_to_world(planned_ijk, coarse_origin, coarse_res)

        if fine_og is not None and len(planned_ijk) >= 3:
            planned_world = coarse_to_world(
                planned_ijk, coarse_origin, coarse_res,
            )
            smoothed_world = simplify_path_ompl(
                planned_world, fine_og, robot_radius, max_time=0.5,
            )

            if len(smoothed_world) >= 2:
                # Arc-length resample to match original time-step count
                resampled_world = _arc_length_resample(
                    smoothed_world, len(planned_t),
                )
                resampled_ijk = world_to_coarse(
                    resampled_world, coarse_origin, coarse_res,
                )
                for d, mx in enumerate(coarse_grid.shape):
                    resampled_ijk[:, d] = np.clip(
                        resampled_ijk[:, d], 0, mx - 1,
                    )

                # Safety: verify no reservation conflicts at new positions
                conflict = False
                for k in range(len(resampled_ijk)):
                    if reservation.is_reserved(
                        int(resampled_ijk[k, 0]),
                        int(resampled_ijk[k, 1]),
                        int(resampled_ijk[k, 2]),
                        int(planned_t[k]),
                    ):
                        conflict = True
                        break

                if not conflict:
                    final_ijk = resampled_ijk
                    final_world = resampled_world  # use the OMPL-smooth coords
                    logger.info(
                        "  [leg %d→%d] OMPL smoothed: %d→%d wpts",
                        prev_node, curr_node,
                        len(planned_ijk), len(smoothed_world),
                    )
                else:
                    logger.info(
                        "  [leg %d→%d] OMPL path has reservation conflict "
                        "– using A* path.",
                        prev_node, curr_node,
                    )

        coarse_positions.append(final_ijk)
        world_positions.append(final_world)
        coarse_times.append(planned_t)

        # ── Dwell at waypoint ─────────────────────────────────────────
        last = final_ijk[-1]
        dwell_ijk = np.tile(last, (hold_steps, 1))
        dwell_world = np.tile(final_world[-1], (hold_steps, 1))
        dwell_t = np.arange(
            t_cursor + 1, t_cursor + hold_steps + 1, dtype=np.intp,
        )
        coarse_positions.append(dwell_ijk)
        world_positions.append(dwell_world)
        coarse_times.append(dwell_t)
        t_dwell_start = int(t_cursor) + 1
        t_cursor += hold_steps
        wp_schedule.append((t_dwell_start, int(t_cursor), curr_node))

    if not coarse_positions:
        return np.empty((0, 3), dtype=np.float64), np.empty((0,), dtype=np.intp), []

    all_ijk = np.concatenate(coarse_positions, axis=0)
    all_t   = np.concatenate(coarse_times, axis=0)

    # Commit to reservation table
    reservation.commit_trajectory(all_ijk, all_t)

    # Use the OMPL-smooth world coordinates directly (not voxel-centre re-derivation)
    world_xyz = np.concatenate(world_positions, axis=0)
    return world_xyz, all_t, wp_schedule
