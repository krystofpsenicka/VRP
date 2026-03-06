"""
VRP Planner – GPU Distance Matrix (cuGraph Dijkstra + CPU A* fallback)

This module has two modes:

  1. **Standalone subprocess mode** (run inside the ``rapids_solver`` conda env):
     Uses NVIDIA cuGraph to run all-pairs Dijkstra on the GPU.
     Called by the main env via ``compute_distance_matrix()``.
     When run as ``__main__``, reads args from a JSON temp file and writes
     the resulting N×N matrix as a ``.npy`` file.

  2. **CPU fallback mode** (runs in any env):
     A vectorised A* implementation using a binary heap.  Slower for large
     grids but works without RAPIDS/CUDA.

Entry points (from the main env):
  • ``compute_distance_matrix(og, waypoints_world, ...)``
      Automatically picks GPU if RAPIDS_PYTHON is available, otherwise A*.
"""

from __future__ import annotations

import json
import os
import sys
import tempfile
import subprocess
from typing import List, Optional, Tuple

import numpy as np


# ─────────────────────────────────────────────────────────────────────────────
# ── CPU A* helpers ────────────────────────────────────────────────────────────
# ─────────────────────────────────────────────────────────────────────────────

# 26-connected 3D neighbourhood offsets and their Euclidean weights
_OFFSETS_26: List[Tuple[int, int, int]] = [
    (di, dj, dk)
    for di in (-1, 0, 1)
    for dj in (-1, 0, 1)
    for dk in (-1, 0, 1)
    if not (di == 0 and dj == 0 and dk == 0)
]
_WEIGHTS_26: np.ndarray = np.array(
    [np.sqrt(di**2 + dj**2 + dk**2) for di, dj, dk in _OFFSETS_26],
    dtype=np.float32,
)


def _astar_single(
    grid: np.ndarray,
    start_ijk: np.ndarray,
    goal_ijk: np.ndarray,
    resolution: float,
) -> float:
    """A* on a 26-connected 3D voxel grid.

    Returns the Euclidean-weighted path length in **world metres**
    (voxel count × resolution × step weight).
    Returns ``np.inf`` if no path exists.
    """
    import heapq

    Nx, Ny, Nz = grid.shape
    start = tuple(start_ijk)
    goal  = tuple(goal_ijk)

    if start == goal:
        return 0.0
    if grid[start] or grid[goal]:
        return np.inf

    g_cost  = {start: 0.0}
    open_set = [(0.0, start)]
    came_from: dict = {}

    def h(node):
        return (
            np.sqrt(sum((a - b) ** 2 for a, b in zip(node, goal))) * resolution
        )

    while open_set:
        f, current = heapq.heappop(open_set)
        if current == goal:
            return g_cost[current]
        # Pruning: skip if we already found a better path
        if g_cost.get(current, np.inf) < f - h(current) - 1e-9:
            continue
        ci, cj, ck = current
        for (di, dj, dk), w in zip(_OFFSETS_26, _WEIGHTS_26):
            ni, nj, nk = ci + di, cj + dj, ck + dk
            if not (0 <= ni < Nx and 0 <= nj < Ny and 0 <= nk < Nz):
                continue
            if grid[ni, nj, nk]:
                continue
            new_g = g_cost[current] + float(w) * resolution
            nbr   = (ni, nj, nk)
            if new_g < g_cost.get(nbr, np.inf):
                g_cost[nbr] = new_g
                heapq.heappush(open_set, (new_g + h(nbr), nbr))
    return np.inf


def compute_distance_matrix_cpu(
    grid: np.ndarray,
    origin: np.ndarray,
    resolution: float,
    waypoints_world: np.ndarray,
) -> np.ndarray:
    """Compute N×N A* distance matrix on the CPU.

    Parameters
    ----------
    grid : (Nx, Ny, Nz) bool array (True = obstacle).
    origin : world origin of voxel (0,0,0).
    resolution : metres per voxel.
    waypoints_world : (N, 3) world-frame waypoint positions.

    Returns
    -------
    matrix : (N, N) float32 distance matrix in metres.
    """
    N = len(waypoints_world)
    matrix = np.zeros((N, N), dtype=np.float32)

    waypoints_ijk = np.floor(
        (waypoints_world - origin) / resolution
    ).astype(int)

    # Snap waypoints that are in occupied voxels to nearest free
    waypoints_ijk = _snap_to_free(grid, waypoints_ijk)

    total_pairs = N * (N - 1) // 2
    done = 0
    for i in range(N):
        for j in range(i + 1, N):
            d = _astar_single(grid, waypoints_ijk[i], waypoints_ijk[j], resolution)
            matrix[i, j] = d
            matrix[j, i] = d
            done += 1
            if done % 20 == 0:
                print(f"  [A*] {done}/{total_pairs} pairs …", end="\r")
    print(f"  [A*] {total_pairs}/{total_pairs} pairs done.          ")
    return matrix


def _snap_to_free(
    grid: np.ndarray,
    ijk_array: np.ndarray,
    max_radius: int = 10,
) -> np.ndarray:
    """For each index in ``ijk_array``, snap to nearest free voxel if occupied.

    Uses a bounded BFS outward from the given index.
    """
    shape = np.array(grid.shape)
    result = ijk_array.copy()
    for n, ijk in enumerate(ijk_array):
        ijk = np.clip(ijk, 0, shape - 1)
        if not grid[tuple(ijk)]:
            result[n] = ijk
            continue
        # BFS to find nearest free
        found = False
        from collections import deque
        queue = deque([tuple(ijk)])
        visited = {tuple(ijk)}
        while queue:
            cur = queue.popleft()
            if not grid[cur]:
                result[n] = np.array(cur)
                found = True
                break
            ci, cj, ck = cur
            for di, dj, dk in _OFFSETS_26:
                ni, nj, nk = ci + di, cj + dj, ck + dk
                if not (0 <= ni < shape[0] and 0 <= nj < shape[1] and 0 <= nk < shape[2]):
                    continue
                nb = (ni, nj, nk)
                if nb not in visited:
                    visited.add(nb)
                    queue.append(nb)
        if not found:
            result[n] = np.clip(ijk, 0, shape - 1)
    return result


# ─────────────────────────────────────────────────────────────────────────────
# ── Main entry point (subprocess mode inside rapids_solver env) ───────────────
# ─────────────────────────────────────────────────────────────────────────────

def _cuGraph_distance_matrix_main():
    """Entry point when this script is executed as a subprocess.

    Reads a JSON config file path from argv[1].
    Config JSON keys:
      grid_path          : path to a .npy bool 3D array
      origin             : [x, y, z]
      resolution         : float
      waypoints_flat     : [x0,y0,z0, x1,y1,z1, ...] flat list
      out_matrix_path    : path to write the output .npy matrix
    """
    import json
    import numpy as np

    config_path = sys.argv[1]
    with open(config_path) as f:
        cfg = json.load(f)

    grid        = np.load(cfg["grid_path"])
    origin      = np.array(cfg["origin"], dtype=np.float64)
    resolution  = float(cfg["resolution"])
    wp_flat     = np.array(cfg["waypoints_flat"], dtype=np.float64)
    N           = len(wp_flat) // 3
    waypoints   = wp_flat.reshape(N, 3)
    out_path    = cfg["out_matrix_path"]

    print(f"[cuGraph] Grid shape: {grid.shape}, {N} waypoints")

    try:
        import cudf
        import cugraph
        import cupy as cp

        # ── Build edge list from free voxels on GPU ────────────────────────
        free_mask  = ~grid   # (Nx, Ny, Nz) bool
        Nx, Ny, Nz = grid.shape

        # Get free voxel indices on CPU, then create edges
        free_ijk = np.argwhere(free_mask)  # (F, 3)
        free_flat = np.ravel_multi_index(
            (free_ijk[:, 0], free_ijk[:, 1], free_ijk[:, 2]), grid.shape
        )  # flat node IDs for free voxels

        # Build a mapping: flat_idx -> sequential node id
        flat_to_node = {int(f): i for i, f in enumerate(free_flat)}
        F = len(free_flat)
        print(f"[cuGraph] Free voxels: {F}", flush=True)

        offsets = np.array(_OFFSETS_26, dtype=np.int32)
        weights_arr = np.array(_WEIGHTS_26, dtype=np.float32) * float(resolution)

        src_list, dst_list, wt_list = [], [], []
        BATCH = 50_000
        for start in range(0, len(free_ijk), BATCH):
            batch_ijk = free_ijk[start: start + BATCH]
            for (di, dj, dk), w in zip(offsets, weights_arr):
                nbr_ijk = batch_ijk + np.array([di, dj, dk])
                # Filter valid bounds
                valid = (
                    (nbr_ijk[:, 0] >= 0) & (nbr_ijk[:, 0] < Nx) &
                    (nbr_ijk[:, 1] >= 0) & (nbr_ijk[:, 1] < Ny) &
                    (nbr_ijk[:, 2] >= 0) & (nbr_ijk[:, 2] < Nz)
                )
                nbr_ijk = nbr_ijk[valid]
                src_ijk = batch_ijk[valid]
                # Filter free neighbours
                free_nbr = ~grid[nbr_ijk[:, 0], nbr_ijk[:, 1], nbr_ijk[:, 2]]
                nbr_ijk  = nbr_ijk[free_nbr]
                src_ijk  = src_ijk[free_nbr]
                if len(src_ijk) == 0:
                    continue
                src_flat = np.ravel_multi_index(
                    (src_ijk[:, 0], src_ijk[:, 1], src_ijk[:, 2]), grid.shape
                )
                dst_flat = np.ravel_multi_index(
                    (nbr_ijk[:, 0], nbr_ijk[:, 1], nbr_ijk[:, 2]), grid.shape
                )
                # Map to sequential node IDs
                src_nodes = np.array([flat_to_node.get(int(s), -1) for s in src_flat])
                dst_nodes = np.array([flat_to_node.get(int(d), -1) for d in dst_flat])
                mask = (src_nodes >= 0) & (dst_nodes >= 0)
                src_list.append(src_nodes[mask])
                dst_list.append(dst_nodes[mask])
                wt_list.append(np.full(mask.sum(), w, dtype=np.float32))

        src_arr = np.concatenate(src_list).astype(np.int32)
        dst_arr = np.concatenate(dst_list).astype(np.int32)
        wt_arr  = np.concatenate(wt_list).astype(np.float32)

        gdf = cudf.DataFrame({"src": src_arr, "dst": dst_arr, "weight": wt_arr})
        G   = cugraph.Graph()
        G.from_cudf_edgelist(gdf, source="src", destination="dst", edge_attr="weight")
        print(f"[cuGraph] Graph built: {G.number_of_nodes()} nodes, "
              f"{G.number_of_edges()} edges", flush=True)

        # ── Snap waypoints to free node IDs ───────────────────────────────
        wp_ijk  = np.floor((waypoints - origin) / resolution).astype(int)
        wp_ijk  = _snap_to_free(grid, wp_ijk)
        wp_flat_ids = np.ravel_multi_index(
            (wp_ijk[:, 0], wp_ijk[:, 1], wp_ijk[:, 2]), grid.shape
        )
        wp_node_ids = np.array([flat_to_node.get(int(f), 0) for f in wp_flat_ids])

        # ── All-pairs Dijkstra ────────────────────────────────────────────
        matrix = np.zeros((N, N), dtype=np.float32)
        for i, src_node in enumerate(wp_node_ids):
            df = cugraph.shortest_path(G, src_node)
            df = df.to_pandas().set_index("vertex")
            for j, dst_node in enumerate(wp_node_ids):
                if i == j:
                    continue
                dist = df.at[int(dst_node), "distance"] if int(dst_node) in df.index else np.inf
                matrix[i, j] = float(dist)
            print(f"  [cuGraph] Dijkstra {i+1}/{N} done", end="\r", flush=True)
        print(f"  [cuGraph] All-pairs done.                    ", flush=True)

    except Exception as e:
        print(f"[cuGraph] FAILED ({e}), falling back to CPU A*", flush=True)
        # Fallback inside subprocess
        matrix = compute_distance_matrix_cpu(grid, origin, resolution, waypoints)

    np.save(out_path, matrix)
    print(f"[cuGraph] Matrix saved to {out_path}", flush=True)


# ─────────────────────────────────────────────────────────────────────────────
# ── Driver (called from main env) ────────────────────────────────────────────
# ─────────────────────────────────────────────────────────────────────────────

def compute_distance_matrix(
    og,           # OccupancyGrid
    waypoints_world: np.ndarray,
    cache_path: Optional[str] = None,
    force_rebuild: bool = False,
    rapids_python: Optional[str] = None,
) -> np.ndarray:
    """Compute the N×N collision-free distance matrix for ``waypoints_world``.

    Tries to use the RAPIDS (cuGraph) subprocess first.  Falls back to CPU A*
    when the RAPIDS environment is unavailable.

    Parameters
    ----------
    og : OccupancyGrid
    waypoints_world : (N, 3) – world-frame waypoint XYZ positions.
    cache_path : optional path to cache the result as .npy.
    force_rebuild : ignore cache.
    rapids_python : path to the Python binary inside the RAPIDS conda env.

    Returns
    -------
    matrix : (N, N) float32 in metres.
    """
    if rapids_python is None:
        from VRP.config import RAPIDS_PYTHON
        rapids_python = RAPIDS_PYTHON

    if cache_path and not force_rebuild and os.path.exists(cache_path):
        print(f"[DistMatrix] Loading cached matrix from {cache_path}")
        return np.load(cache_path)

    N = len(waypoints_world)
    print(f"[DistMatrix] Computing {N}×{N} distance matrix …")

    # ── Try GPU subprocess ────────────────────────────────────────────────────
    gpu_ok = (
        os.path.exists(rapids_python) and
        _rapids_env_has_cugraph(rapids_python)
    )

    if gpu_ok:
        try:
            matrix = _compute_via_subprocess(og, waypoints_world, rapids_python)
        except Exception as e:
            print(f"[DistMatrix] cuGraph subprocess error: {e}  → falling back to CPU A* …")
            matrix = compute_distance_matrix_cpu(
                og.grid, og.origin, og.resolution, waypoints_world
            )
    else:
        print("[DistMatrix] cuGraph unavailable, using CPU A* …")
        matrix = compute_distance_matrix_cpu(
            og.grid, og.origin, og.resolution, waypoints_world
        )

    if cache_path:
        os.makedirs(os.path.dirname(os.path.abspath(cache_path)), exist_ok=True)
        np.save(cache_path, matrix)
        print(f"[DistMatrix] Saved to {cache_path}")

    return matrix


def _rapids_env_has_cugraph(rapids_python: str) -> bool:
    """Quick check whether the RAPIDS env has cugraph importable."""
    try:
        result = subprocess.run(
            [rapids_python, "-c", "import cugraph; print('ok')"],
            capture_output=True, text=True, timeout=15
        )
        return result.returncode == 0 and "ok" in result.stdout
    except Exception:
        return False


def _compute_via_subprocess(
    og,
    waypoints_world: np.ndarray,
    rapids_python: str,
) -> np.ndarray:
    """Call this script as a subprocess inside the RAPIDS env."""
    with tempfile.TemporaryDirectory() as tmpdir:
        grid_path   = os.path.join(tmpdir, "grid.npy")
        config_path = os.path.join(tmpdir, "config.json")
        out_path    = os.path.join(tmpdir, "matrix.npy")

        np.save(grid_path, og.grid)
        cfg = {
            "grid_path":       grid_path,
            "origin":          og.origin.tolist(),
            "resolution":      float(og.resolution),
            "waypoints_flat":  waypoints_world.flatten().tolist(),
            "out_matrix_path": out_path,
        }
        with open(config_path, "w") as f:
            json.dump(cfg, f)

        script_path = os.path.abspath(__file__)
        print(f"[DistMatrix] Launching cuGraph subprocess …")
        result = subprocess.run(
            [rapids_python, script_path, config_path],
            capture_output=False,
            text=True,
        )
        if result.returncode != 0:
            raise RuntimeError(
                f"cuGraph subprocess failed (exit {result.returncode}). "
                "Falling back to CPU A*."
            )
        return np.load(out_path)


# ─────────────────────────────────────────────────────────────────────────────
# ── Path extraction helper (for traffic-light corridors) ─────────────────────
# ─────────────────────────────────────────────────────────────────────────────

def extract_astar_path(
    og,
    start_world: np.ndarray,
    goal_world: np.ndarray,
) -> np.ndarray:
    """Run A* on the occupancy grid and return the world-frame path.

    Returns
    -------
    path : (M, 3) array of world-frame XYZ waypoints along the collision-free
        path, or shape ``(0, 3)`` if no path exists.
    """
    from VRP.occupancy_grid import OccupancyGrid

    grid       = og.grid
    resolution = og.resolution
    origin     = og.origin
    Nx, Ny, Nz = grid.shape

    start_ijk = np.floor((start_world - origin) / resolution).astype(int)
    goal_ijk  = np.floor((goal_world  - origin) / resolution).astype(int)
    start_ijk = _snap_to_free(grid, start_ijk.reshape(1, 3))[0]
    goal_ijk  = _snap_to_free(grid, goal_ijk.reshape(1,  3))[0]

    import heapq

    start = tuple(start_ijk)
    goal  = tuple(goal_ijk)

    if start == goal:
        return np.array([og.voxel_to_world(start_ijk)])

    if grid[start] or grid[goal]:
        return np.zeros((0, 3), dtype=np.float32)

    g_cost    = {start: 0.0}
    came_from = {}
    open_set  = [(0.0, start)]

    def h(node):
        return np.sqrt(sum((a - b)**2 for a, b in zip(node, goal))) * resolution

    while open_set:
        f, current = heapq.heappop(open_set)
        if current == goal:
            # Reconstruct path
            path_ijk = []
            node = current
            while node in came_from:
                path_ijk.append(node)
                node = came_from[node]
            path_ijk.append(start)
            path_ijk.reverse()
            path_world = np.array(
                [og.voxel_to_world(np.array(p)) for p in path_ijk],
                dtype=np.float32,
            )
            return path_world
        if g_cost.get(current, np.inf) < f - h(current) - 1e-9:
            continue
        ci, cj, ck = current
        for (di, dj, dk), w in zip(_OFFSETS_26, _WEIGHTS_26):
            ni, nj, nk = ci + di, cj + dj, ck + dk
            if not (0 <= ni < Nx and 0 <= nj < Ny and 0 <= nk < Nz):
                continue
            if grid[ni, nj, nk]:
                continue
            new_g = g_cost[current] + float(w) * resolution
            nbr   = (ni, nj, nk)
            if new_g < g_cost.get(nbr, np.inf):
                g_cost[nbr]    = new_g
                came_from[nbr] = current
                heapq.heappush(open_set, (new_g + h(nbr), nbr))

    return np.zeros((0, 3), dtype=np.float32)


def build_route_path_cache(
    og,
    waypoints_world: np.ndarray,
    routes: List[List[int]],
    sub_sample_dist: float = 2.0,
) -> dict:
    """Compute A* paths for every (i, j) segment pair used by the routes.

    Only the unique directed pairs actually traversed are computed — not all
    N² combinations.  Each path is sub-sampled to one point every
    ``sub_sample_dist`` metres so cuRobo plans short, obstacle-free hops.

    Parameters
    ----------
    og:
        Inflated :class:`OccupancyGrid` used for A* collision checking.
    waypoints_world:
        ``(N, 7)`` or ``(N, 3)`` world-frame node positions (only XYZ used).
    routes:
        Per-robot ordered node-index lists (depot included).
    sub_sample_dist:
        Target spacing between consecutive sub-waypoints in metres.

    Returns
    -------
    cache : dict[(int, int), np.ndarray shape (M, 3)]
    """
    # Collect unique directed pairs that the routes actually traverse
    pairs: set = set()
    for route in routes:
        for k in range(1, len(route)):
            pairs.add((route[k - 1], route[k]))

    xyz   = np.asarray(waypoints_world)[:, :3]
    cache: dict = {}

    for (i, j) in sorted(pairs):
        path = extract_astar_path(og, xyz[i], xyz[j])

        if len(path) == 0:
            print(f"[path_cache] WARNING: no A* path {i}\u2192{j} — will plan direct.")
            cache[(i, j)] = np.array([xyz[i], xyz[j]], dtype=np.float32)
            continue

        if len(path) <= 2:
            cache[(i, j)] = path
            continue

        # Cumulative arc-length along path
        seg_lens = np.linalg.norm(np.diff(path, axis=0), axis=1).astype(float)
        cum      = np.concatenate([[0.0], np.cumsum(seg_lens)])
        total    = cum[-1]

        if total < sub_sample_dist:
            cache[(i, j)] = path[[0, -1]]
            continue

        targets = np.arange(0.0, total, sub_sample_dist)
        picked  = np.unique(
            np.concatenate([[0],
                            np.searchsorted(cum, targets),
                            [len(path) - 1]])
        ).astype(int)
        cache[(i, j)] = path[picked]
        print(f"[path_cache] {i}\u2192{j}: {len(path)} A* pts "
              f"\u2192 {len(cache[(i, j)])} sub-pts  ({total:.1f} m)")

    print(f"[path_cache] Built paths for {len(cache)} route segments.")
    return cache


# ─────────────────────────────────────────────────────────────────────────────
# ── Subprocess entry point ────────────────────────────────────────────────────
# ─────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python gpu_distance_matrix.py <config.json>")
        sys.exit(1)
    _cuGraph_distance_matrix_main()
