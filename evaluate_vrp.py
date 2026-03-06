#!/usr/bin/env python3
"""
VRP Routing Module – Quantitative Evaluation
=============================================

Sweeps fleet sizes (1–6) × waypoint counts (10–100) × 3 random seeds
using the cuOpt solver, collects per-stage timing, route-quality metrics,
and generates publication-ready figures for the IROS paper.

Usage
-----
    cd /home/troja_robot_lab/Desktop/Krystof
    python -m VRP.evaluate_vrp [--output_dir VRP/eval_results]
    # Quick test:
    python -m VRP.evaluate_vrp --fleet_sizes 1 2 3 --waypoint_counts 10 20 --seeds 42

Outputs
-------
    <output_dir>/
        results.csv                     – raw per-run metrics
        fig_makespan_vs_fleet.pdf/.png
        fig_total_cost_vs_fleet.pdf/.png
        fig_route_balance_vs_fleet.pdf/.png
        fig_timing_breakdown.pdf/.png
        fig_scalability_waypoints.pdf/.png
        fig_speedup_vs_fleet.pdf/.png
"""

from __future__ import annotations

import argparse
import csv
import itertools
import logging
import os
import sys
import time
from dataclasses import dataclass, asdict
from typing import Dict, List, Tuple

import numpy as np

# ── Ensure VRP package is importable ──────────────────────────────────────────
_VRP_ROOT = os.path.dirname(os.path.abspath(__file__))
_PROJECT_ROOT = os.path.dirname(_VRP_ROOT)
if _PROJECT_ROOT not in sys.path:
    sys.path.insert(0, _PROJECT_ROOT)
# Some VRP modules use bare `from config import ...` (not relative imports),
# so the VRP directory itself must also be on sys.path.
if _VRP_ROOT not in sys.path:
    sys.path.insert(0, _VRP_ROOT)

from VRP.occupancy_grid import build_occupancy_grid, OccupancyGrid, get_mesh_world_bounds
from VRP.waypoint_loader import load_waypoints
from VRP.gpu_distance_matrix import compute_distance_matrix, build_route_path_cache
from VRP.vrp_solver import solve_vrp, VRPResult
from VRP.route_executor import RouteExecutor, ExecutionResult
from VRP.vrp_planner import _compute_start_grid
from VRP.utils import (
    load_local_robot_config,
    find_trajectory_collisions,
)
from VRP.config import (
    RAPIDS_PYTHON,
    CUOPT_SERVICE_TIME,
    BROV_CUBOID_DIMS,
)

logger = logging.getLogger("vrp_eval")

# ── Default sweep parameters ─────────────────────────────────────────────────

FLEET_SIZES     = [1, 2, 3, 4, 5, 6]
WAYPOINT_COUNTS = [10, 20, 30, 50, 75, 100]
SEEDS           = [42, 123, 7]
SOLVER_BACKEND  = "cuopt"


# ── Metrics dataclass ─────────────────────────────────────────────────────────

@dataclass
class RunMetrics:
    """All metrics collected from a single pipeline run."""
    fleet_size:          int   = 0
    n_waypoints:         int   = 0
    seed:                int   = 0
    solver:              str   = ""
    status:              str   = ""
    # ── Route quality ──
    total_cost:          float = 0.0   # sum of all route distances (m)
    makespan:            float = 0.0   # longest single-robot route (m)
    shortest_route:      float = 0.0   # shortest single-robot route (m)
    route_balance_ratio: float = 0.0   # makespan / shortest (1.0 = perfect)
    route_cost_std:      float = 0.0   # std dev of per-robot route costs (m)
    mean_route_cost:     float = 0.0   # mean per-robot route cost (m)
    avg_wps_per_robot:   float = 0.0
    max_wps_per_robot:   int   = 0
    min_wps_per_robot:   int   = 0
    # ── Trajectory quality ──
    total_traj_steps:    int   = 0     # max trajectory length across robots
    residual_collisions: int   = 0     # AABB collisions post-deconfliction
    fail_count_total:    int   = 0     # segments that failed to plan
    # ── Timing (seconds) ──
    t_grid:              float = 0.0
    t_dist_matrix:       float = 0.0
    t_vrp_solve:         float = 0.0
    t_path_cache:        float = 0.0
    t_trajectory:        float = 0.0
    t_total:             float = 0.0


# ── Single benchmark run ──────────────────────────────────────────────────────

def run_single(
    fleet_size: int,
    n_waypoints: int,
    seed: int,
    og: OccupancyGrid,
    robot_cfg: dict,
    mesh_bounds_min: np.ndarray,
    mesh_bounds_max: np.ndarray,
    solver_backend: str = "cuopt",
) -> RunMetrics:
    """Execute one pipeline configuration and collect metrics."""
    m = RunMetrics(fleet_size=fleet_size, n_waypoints=n_waypoints, seed=seed)
    t_total_start = time.perf_counter()

    try:
        # ── Waypoints ─────────────────────────────────────────────
        inspection_wps = np.array(
            load_waypoints(
                source="random",
                n_random=n_waypoints,
                og=og,
                random_seed=seed,
            ),
            dtype=np.float32,
        )  # (N, 7)
        N = len(inspection_wps)
        K = fleet_size

        robot_start_xyzs = _compute_start_grid(K, mesh_bounds_min, mesh_bounds_max)
        home_poses = np.array(
            [[*xyz, 1.0, 0.0, 0.0, 0.0] for xyz in robot_start_xyzs],
            dtype=np.float32,
        )
        waypoints_world = np.vstack([home_poses, inspection_wps])
        home_indices = list(range(K))

        # ── Distance matrix ───────────────────────────────────────
        t0 = time.perf_counter()
        dist_matrix = compute_distance_matrix(og, waypoints_world[:, :3])
        m.t_dist_matrix = time.perf_counter() - t0

        # ── VRP solve ─────────────────────────────────────────────
        t0 = time.perf_counter()
        vrp_result: VRPResult = solve_vrp(
            dist_matrix=dist_matrix,
            num_vehicles=K,
            depot=home_indices,
            backend=solver_backend,
            rapids_python=RAPIDS_PYTHON,
            service_time=CUOPT_SERVICE_TIME,
            time_limit=60,
            gpu_timeout=300,
        )
        m.t_vrp_solve = time.perf_counter() - t0
        m.solver = vrp_result.solver
        m.status = vrp_result.status
        m.total_cost = vrp_result.total_cost

        if not any(vrp_result.routes):
            m.status = "empty_routes"
            m.t_total = time.perf_counter() - t_total_start
            return m

        # Routes with home nodes at start and end
        routes = [
            [home_indices[i]] + list(r) + [home_indices[i]]
            for i, r in enumerate(vrp_result.routes)
        ]

        # ── Per-robot route costs ─────────────────────────────────
        route_costs = []
        for i, route in enumerate(routes):
            cost = 0.0
            for a, b in zip(route[:-1], route[1:]):
                d = dist_matrix[a, b]
                if np.isfinite(d):
                    cost += d
            route_costs.append(cost)

        rc = np.array(route_costs)
        m.makespan = float(rc.max())
        m.shortest_route = float(rc[rc > 0].min()) if np.any(rc > 0) else 0.0
        m.mean_route_cost = float(rc.mean())
        m.route_cost_std = float(rc.std())
        m.route_balance_ratio = (
            m.makespan / m.shortest_route if m.shortest_route > 0 else float("inf")
        )

        # Waypoints per robot (excluding 2 home nodes)
        wps_per_robot = [max(0, len(r) - 2) for r in routes]
        m.avg_wps_per_robot = float(np.mean(wps_per_robot))
        m.max_wps_per_robot = int(max(wps_per_robot))
        m.min_wps_per_robot = int(min(wps_per_robot))

        # ── Path cache (A* sub-waypoints) ─────────────────────────
        t0 = time.perf_counter()
        path_cache = build_route_path_cache(og, waypoints_world[:, :3], routes)
        m.t_path_cache = time.perf_counter() - t0

        # ── Trajectory execution + collision avoidance ────────────
        t0 = time.perf_counter()

        j_names = robot_cfg["kinematics"]["cspace"]["joint_names"]
        default_cfg = robot_cfg["kinematics"]["cspace"]["retract_config"]

        start_configs = []
        for i in range(K):
            s = list(default_cfg)
            xyz = robot_start_xyzs[i]
            s[0], s[1], s[2] = float(xyz[0]), float(xyz[1]), float(xyz[2])
            start_configs.append(np.array(s, dtype=np.float32))

        executor = RouteExecutor(
            start_configs=start_configs,
            joint_names=j_names,
            og=og,
        )
        exec_result: ExecutionResult = executor.execute(
            routes=routes,
            waypoints_world=waypoints_world,
            path_cache=path_cache,
        )
        m.t_trajectory = time.perf_counter() - t0

        # ── Trajectory-quality metrics ────────────────────────────
        m.total_traj_steps = max(
            len(traj) for traj in exec_result.all_traj_positions
        ) if exec_result.all_traj_positions else 0

        m.fail_count_total = sum(exec_result.fail_counts)

        collisions = find_trajectory_collisions(
            exec_result.all_traj_positions,
            dims=BROV_CUBOID_DIMS,
        )
        m.residual_collisions = len(collisions)

    except Exception as e:
        logger.error("Run failed (fleet=%d, wps=%d, seed=%d): %s",
                     fleet_size, n_waypoints, seed, e, exc_info=True)
        m.status = f"error: {e}"

    m.t_total = time.perf_counter() - t_total_start
    return m


# ── Plotting ──────────────────────────────────────────────────────────────────

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator
from collections import defaultdict

# IEEE single-column = 3.5 in
IEEE_COL_W = 3.5

plt.rcParams.update({
    "font.size": 8,
    "axes.labelsize": 9,
    "legend.fontsize": 7,
    "xtick.labelsize": 8,
    "ytick.labelsize": 8,
    "lines.linewidth": 1.2,
    "lines.markersize": 4,
    "figure.dpi": 300,
    "savefig.bbox": "tight",
    "savefig.pad_inches": 0.02,
})


def _group(rows: List[RunMetrics]) -> Dict[Tuple[int, int], List[RunMetrics]]:
    """Group successful rows by (fleet_size, n_waypoints)."""
    g: Dict[Tuple[int, int], List[RunMetrics]] = defaultdict(list)
    for r in rows:
        if r.status == "success":
            g[(r.fleet_size, r.n_waypoints)].append(r)
    return g


def _save(fig, out_dir: str, name: str):
    fig.savefig(os.path.join(out_dir, f"{name}.pdf"))
    fig.savefig(os.path.join(out_dir, f"{name}.png"))
    plt.close(fig)
    logger.info("  saved %s", name)


def _wp_colors(waypoint_counts):
    return plt.cm.viridis(np.linspace(0.15, 0.85, len(waypoint_counts)))


def _fleet_colors(fleet_sizes):
    return plt.cm.plasma(np.linspace(0.15, 0.85, len(fleet_sizes)))


# ─────────────────────────────────────────────────────────────────────────────

def plot_makespan_vs_fleet(rows, fleet_sizes, waypoint_counts, out_dir):
    """Fig 1: Makespan (longest route) vs fleet size, one curve per WP count."""
    groups = _group(rows)
    colors = _wp_colors(waypoint_counts)
    fig, ax = plt.subplots(figsize=(IEEE_COL_W, 2.4))

    for wi, nw in enumerate(waypoint_counts):
        xs, means, stds = [], [], []
        for k in fleet_sizes:
            vals = [r.makespan for r in groups.get((k, nw), [])]
            if vals:
                xs.append(k)
                means.append(np.mean(vals))
                stds.append(np.std(vals))
        if xs:
            ax.errorbar(xs, means, yerr=stds, marker="o", color=colors[wi],
                        label=f"{nw} wps", capsize=2)

    ax.set_xlabel("Fleet size (number of robots)")
    ax.set_ylabel("Makespan (m)")
    ax.set_title("Makespan vs. Fleet Size")
    ax.xaxis.set_major_locator(MaxNLocator(integer=True))
    ax.legend(ncol=2, loc="upper right")
    ax.grid(True, alpha=0.3)
    _save(fig, out_dir, "fig_makespan_vs_fleet")


def plot_total_cost_vs_fleet(rows, fleet_sizes, waypoint_counts, out_dir):
    """Fig 2: Total route cost vs fleet size."""
    groups = _group(rows)
    colors = _wp_colors(waypoint_counts)
    fig, ax = plt.subplots(figsize=(IEEE_COL_W, 2.4))

    for wi, nw in enumerate(waypoint_counts):
        xs, means, stds = [], [], []
        for k in fleet_sizes:
            vals = [r.total_cost for r in groups.get((k, nw), [])]
            if vals:
                xs.append(k)
                means.append(np.mean(vals))
                stds.append(np.std(vals))
        if xs:
            ax.errorbar(xs, means, yerr=stds, marker="s", color=colors[wi],
                        label=f"{nw} wps", capsize=2)

    ax.set_xlabel("Fleet size (number of robots)")
    ax.set_ylabel("Total route cost (m)")
    ax.set_title("Total Route Cost vs. Fleet Size")
    ax.xaxis.set_major_locator(MaxNLocator(integer=True))
    ax.legend(ncol=2, loc="upper right")
    ax.grid(True, alpha=0.3)
    _save(fig, out_dir, "fig_total_cost_vs_fleet")


def plot_route_balance(rows, fleet_sizes, waypoint_counts, out_dir):
    """Fig 3: Route balance (CV = std/mean of per-robot costs) vs fleet size."""
    groups = _group(rows)
    colors = _wp_colors(waypoint_counts)
    fig, ax = plt.subplots(figsize=(IEEE_COL_W, 2.4))

    for wi, nw in enumerate(waypoint_counts):
        xs, means, stds = [], [], []
        for k in fleet_sizes:
            if k == 1:
                continue  # CV undefined for a single robot
            vals = [r.route_cost_std / r.mean_route_cost
                    for r in groups.get((k, nw), [])
                    if r.mean_route_cost > 0]
            if vals:
                xs.append(k)
                means.append(np.mean(vals))
                stds.append(np.std(vals))
        if xs:
            ax.errorbar(xs, means, yerr=stds, marker="^", color=colors[wi],
                        label=f"{nw} wps", capsize=2)

    ax.set_xlabel("Fleet size (number of robots)")
    ax.set_ylabel("Route cost CV (std / mean)")
    ax.set_title("Route Balance vs. Fleet Size")
    ax.xaxis.set_major_locator(MaxNLocator(integer=True))
    ax.legend(ncol=2, loc="upper right")
    ax.grid(True, alpha=0.3)
    _save(fig, out_dir, "fig_route_balance_vs_fleet")


def plot_timing_breakdown(rows, fleet_sizes, waypoint_counts, out_dir):
    """Fig 4: Stacked bar – time breakdown by stage (fixed mid-range WP count)."""
    groups = _group(rows)
    mid_wps = waypoint_counts[len(waypoint_counts) // 2]

    stages = ["t_dist_matrix", "t_vrp_solve", "t_path_cache", "t_trajectory"]
    labels = ["Distance matrix", "VRP solve", "Path cache (A*)", "Trajectory gen."]
    bar_colors = ["#2196F3", "#FF9800", "#4CAF50", "#9C27B0"]

    fig, ax = plt.subplots(figsize=(IEEE_COL_W, 2.6))
    bar_w = 0.6

    for ki, k in enumerate(fleet_sizes):
        ok = groups.get((k, mid_wps), [])
        if not ok:
            continue
        bottom = 0.0
        for si, stage in enumerate(stages):
            val = float(np.mean([getattr(r, stage) for r in ok]))
            ax.bar(k, val, bar_w, bottom=bottom, color=bar_colors[si],
                   label=labels[si] if ki == 0 else "")
            bottom += val

    ax.set_xlabel("Fleet size (number of robots)")
    ax.set_ylabel("Time (s)")
    ax.set_title(f"Computation Time Breakdown ({mid_wps} waypoints)")
    ax.xaxis.set_major_locator(MaxNLocator(integer=True))
    ax.legend(loc="upper left", fontsize=6)
    ax.grid(True, alpha=0.3, axis="y")
    _save(fig, out_dir, "fig_timing_breakdown")


def plot_scalability_waypoints(rows, fleet_sizes, waypoint_counts, out_dir):
    """Fig 5: Total time vs waypoint count, one curve per fleet size."""
    groups = _group(rows)
    colors = _fleet_colors(fleet_sizes)
    fig, ax = plt.subplots(figsize=(IEEE_COL_W, 2.4))

    for ki, k in enumerate(fleet_sizes):
        xs, means, stds = [], [], []
        for nw in waypoint_counts:
            vals = [r.t_total for r in groups.get((k, nw), [])]
            if vals:
                xs.append(nw)
                means.append(np.mean(vals))
                stds.append(np.std(vals))
        if xs:
            ax.errorbar(xs, means, yerr=stds, marker="D", color=colors[ki],
                        label=f"{k} robots", capsize=2)

    ax.set_xlabel("Number of waypoints")
    ax.set_ylabel("Total time (s)")
    ax.set_title("Scalability with Waypoint Count")
    ax.legend(ncol=2, loc="upper left")
    ax.grid(True, alpha=0.3)
    _save(fig, out_dir, "fig_scalability_waypoints")


def plot_speedup_vs_fleet(rows, fleet_sizes, waypoint_counts, out_dir):
    """Fig 6: Makespan speedup relative to single-robot baseline."""
    groups = _group(rows)
    colors = _wp_colors(waypoint_counts)
    fig, ax = plt.subplots(figsize=(IEEE_COL_W, 2.4))

    for wi, nw in enumerate(waypoint_counts):
        baseline_vals = [r.makespan for r in groups.get((1, nw), [])]
        if not baseline_vals:
            continue
        baseline = float(np.mean(baseline_vals))
        if baseline <= 0:
            continue

        xs, means, stds = [], [], []
        for k in fleet_sizes:
            vals = [baseline / r.makespan
                    for r in groups.get((k, nw), [])
                    if r.makespan > 0]
            if vals:
                xs.append(k)
                means.append(np.mean(vals))
                stds.append(np.std(vals))
        if xs:
            ax.errorbar(xs, means, yerr=stds, marker="o", color=colors[wi],
                        label=f"{nw} wps", capsize=2)

    # Ideal linear speedup reference line
    ax.plot(fleet_sizes, fleet_sizes, "k--", alpha=0.4, label="Ideal linear")

    ax.set_xlabel("Fleet size (number of robots)")
    ax.set_ylabel("Makespan speedup (×)")
    ax.set_title("Makespan Speedup vs. Fleet Size")
    ax.xaxis.set_major_locator(MaxNLocator(integer=True))
    ax.legend(ncol=2, loc="upper left", fontsize=6)
    ax.grid(True, alpha=0.3)
    _save(fig, out_dir, "fig_speedup_vs_fleet")


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="Quantitative evaluation of the VRP routing module")
    parser.add_argument("--output_dir",
                        default=os.path.join(_VRP_ROOT, "eval_results"),
                        help="Directory for results CSV and figures")
    parser.add_argument("--fleet_sizes", type=int, nargs="+",
                        default=FLEET_SIZES)
    parser.add_argument("--waypoint_counts", type=int, nargs="+",
                        default=WAYPOINT_COUNTS)
    parser.add_argument("--seeds", type=int, nargs="+",
                        default=SEEDS)
    parser.add_argument("--solver", default=SOLVER_BACKEND,
                        choices=["cuopt", "ortools", "auto"],
                        help="VRP solver backend")
    parser.add_argument("--verbose", "-v", action="store_true")
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(asctime)s %(levelname)-8s %(name)s: %(message)s",
    )

    solver_backend = args.solver

    os.makedirs(args.output_dir, exist_ok=True)

    # ── Build occupancy grid once (shared) ────────────────────────
    logger.info("Building occupancy grid (shared across all runs) ...")
    t0 = time.perf_counter()
    # Probe mesh bounds first so the grid covers the depot positions of
    # all fleet sizes in the sweep (anchor to mesh top, not grid top).
    mesh_bounds_min, mesh_bounds_max = get_mesh_world_bounds()
    logger.info("  Mesh world bounds: min=%s  max=%s",
                mesh_bounds_min.round(2), mesh_bounds_max.round(2))
    max_fleet = max(args.fleet_sizes) if args.fleet_sizes else 1
    max_depot_xyzs = _compute_start_grid(max_fleet, mesh_bounds_min, mesh_bounds_max)
    og = build_occupancy_grid(
        extra_free_points=np.array(max_depot_xyzs, dtype=np.float32),
    )
    t_grid = time.perf_counter() - t0
    logger.info("  Grid shape: %s  resolution: %.2fm  built in %.1fs",
                og.grid.shape, og.resolution, t_grid)

    robot_cfg = load_local_robot_config("brov.yml")

    # ── Sweep configurations ──────────────────────────────────────
    configs = list(itertools.product(
        args.fleet_sizes, args.waypoint_counts, args.seeds))
    total = len(configs)
    logger.info("Starting evaluation: %d runs "
                "(%d fleets × %d wp-counts × %d seeds)",
                total, len(args.fleet_sizes),
                len(args.waypoint_counts), len(args.seeds))

    all_metrics: List[RunMetrics] = []

    csv_path = os.path.join(args.output_dir, "results.csv")
    fieldnames = ["run_id"] + list(RunMetrics.__dataclass_fields__.keys())

    with open(csv_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()

        for run_id, (k, nw, seed) in enumerate(configs, 1):
            logger.info(
                "=== Run %d/%d: fleet=%d  waypoints=%d  seed=%d ===",
                run_id, total, k, nw, seed,
            )

            m = run_single(
                fleet_size=k,
                n_waypoints=nw,
                seed=seed,
                og=og,
                robot_cfg=robot_cfg,
                mesh_bounds_min=mesh_bounds_min,
                mesh_bounds_max=mesh_bounds_max,
                solver_backend=solver_backend,
            )
            m.t_grid = t_grid

            all_metrics.append(m)

            row = asdict(m)
            row["run_id"] = run_id
            writer.writerow(row)
            f.flush()

            cv = (m.route_cost_std / m.mean_route_cost
                  if m.mean_route_cost > 0 else 0.0)
            logger.info(
                "    status=%s  cost=%.1f  makespan=%.1f  CV=%.3f  "
                "t_total=%.1fs  collisions=%d",
                m.status, m.total_cost, m.makespan, cv,
                m.t_total, m.residual_collisions,
            )

    logger.info("All %d runs complete. CSV → %s", total, csv_path)

    # ── Generate figures ──────────────────────────────────────────
    successful = [m for m in all_metrics if m.status == "success"]
    if not successful:
        logger.error("No successful runs — skipping figure generation.")
        return

    logger.info("Generating figures from %d successful runs ...",
                len(successful))

    plot_makespan_vs_fleet(
        successful, args.fleet_sizes, args.waypoint_counts, args.output_dir)
    plot_total_cost_vs_fleet(
        successful, args.fleet_sizes, args.waypoint_counts, args.output_dir)
    plot_route_balance(
        successful, args.fleet_sizes, args.waypoint_counts, args.output_dir)
    plot_timing_breakdown(
        successful, args.fleet_sizes, args.waypoint_counts, args.output_dir)
    plot_scalability_waypoints(
        successful, args.fleet_sizes, args.waypoint_counts, args.output_dir)
    plot_speedup_vs_fleet(
        successful, args.fleet_sizes, args.waypoint_counts, args.output_dir)

    logger.info("All figures saved to %s", args.output_dir)


if __name__ == "__main__":
    main()
