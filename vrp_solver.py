"""
VRP Planner – Solver Layer

Provides two backends:

``GPUSolver``   – runs :mod:`cuopt_subprocess` inside the ``rapids_solver``
                  conda environment via subprocess.
``ORToolsSolver`` – pure-Python CPU fallback using Google OR-Tools
                    ``RoutingModel`` + ``GUIDED_LOCAL_SEARCH``.

Both return a ``VRPResult`` dataclass with per-vehicle ordered waypoint
index lists plus the total route cost.
"""

from __future__ import annotations

import json
import logging
import math
import os
import subprocess
import sys
import tempfile
from dataclasses import dataclass, field
from pathlib import Path
from typing import List, Optional, Union

import numpy as np

from .config import (
    CUOPT_SERVICE_TIME,
    RAPIDS_PYTHON,
    VRP_ROOT as VRP_DIR,
)

logger = logging.getLogger(__name__)


# ─── Result dataclass ────────────────────────────────────────────────────────

@dataclass
class VRPResult:
    """Holds the solution returned by any VRP backend.

    Attributes
    ----------
    routes:
        List of per-vehicle waypoint index lists (0-based, excluding depot).
    total_cost:
        Sum of distances over all routes according to the distance matrix.
    solver:
        Name of the solver that produced this solution (``"cuopt"`` or
        ``"ortools"``).
    status:
        ``"success"`` or an error / fallback message.
    """
    routes:     List[List[int]]
    total_cost: float
    solver:     str     = "unknown"
    status:     str     = "success"


# ─── Shared helpers ──────────────────────────────────────────────────────────

def _normalise_depot(
    depot: Union[int, List[int]], num_vehicles: int
) -> List[int]:
    """Return a per-vehicle depot list regardless of input type."""
    if isinstance(depot, (list, tuple)):
        return [int(d) for d in depot]
    return [int(depot)] * num_vehicles


def _compute_route_cost(routes: List[List[int]],
                        dist_matrix: np.ndarray,
                        depot: Union[int, List[int]] = 0) -> float:
    """Sum travel distances for all vehicles."""
    depots = _normalise_depot(depot, len(routes))
    total = 0.0
    for v, route in enumerate(routes):
        if not route:
            continue
        d = depots[v]
        full = [d] + list(route) + [d]
        for a, b in zip(full[:-1], full[1:]):
            total += float(dist_matrix[a, b])
    return total


def _per_vehicle_costs(
    routes: List[List[int]],
    dist_matrix: np.ndarray,
    depot: Union[int, List[int]] = 0,
) -> List[float]:
    """Return per-vehicle travel distances (for makespan computation)."""
    depots = _normalise_depot(depot, len(routes))
    costs: List[float] = []
    for v, route in enumerate(routes):
        c = 0.0
        if route:
            d = depots[v]
            full = [d] + list(route) + [d]
            for a, b in zip(full[:-1], full[1:]):
                c += float(dist_matrix[a, b])
        costs.append(c)
    return costs


# ─── GPU / cuOpt backend ─────────────────────────────────────────────────────

class GPUSolver:
    """Calls ``cuopt_subprocess.py`` inside the ``rapids_solver`` conda env.

    Parameters
    ----------
    rapids_python:
        Path to the Python executable in the ``rapids_solver`` environment.
        Defaults to :data:`~config.RAPIDS_PYTHON`.
    service_time:
        Dwell time at each waypoint (seconds). Provides temporal separation.
    timeout:
        Maximum wall-clock seconds to wait for the subprocess.
    """

    def __init__(
        self,
        rapids_python:  str   = RAPIDS_PYTHON,
        service_time:   float = CUOPT_SERVICE_TIME,
        timeout:        int   = 300,
    ):
        self.rapids_python = os.path.expanduser(rapids_python)
        self.service_time  = service_time
        self.timeout       = timeout
        self._script       = str(Path(VRP_DIR) / "cuopt_subprocess.py")

    # ------------------------------------------------------------------

    def solve(
        self,
        dist_matrix:   np.ndarray,
        num_vehicles:  int,
        depot:         Union[int, List[int]] = 0,
    ) -> VRPResult:
        """Run cuOpt VRP inside the ``rapids_solver`` subprocess.

        Parameters
        ----------
        dist_matrix:
            Square ``(N, N)`` float32 array of pairwise distances.
        num_vehicles:
            Number of AUVs / robots.
        depot:
            Depot index (int) shared by all vehicles, or a list of
            per-vehicle depot indices.
        """
        depots = _normalise_depot(depot, num_vehicles)

        result = self._call_cuopt(dist_matrix, num_vehicles, depots)
        if result.status == "success":
            per_v    = _per_vehicle_costs(result.routes, dist_matrix, depots)
            makespan = max(per_v) if per_v else 0.0
            logger.info("[GPUSolver] makespan=%.2f  per_vehicle=%s",
                        makespan, [f"{c:.1f}" for c in per_v])
        return result

    # ── Internal cuOpt subprocess call ───────────────────────────────
    def _call_cuopt(
        self,
        dist_matrix: np.ndarray,
        num_vehicles: int,
        depots: List[int],
    ) -> VRPResult:
        """Single cuOpt subprocess invocation."""
        n = dist_matrix.shape[0]
        depot_set = set(depots)
        n_inspection = n - len(depot_set)
        # Exact ceil(n/k) — no slack, forces cuOpt to distribute stops evenly
        balanced_cap = math.ceil(n_inspection / num_vehicles)

        cfg = {
            "n":                n,
            "num_vehicles":     num_vehicles,
            "service_time":     self.service_time,
            "vehicle_capacity": balanced_cap,
            "depot":            depots,
            "distance_matrix":  dist_matrix.flatten().tolist(),
        }

        with tempfile.NamedTemporaryFile(
            mode="w", suffix=".json", delete=False
        ) as cfg_f:
            json.dump(cfg, cfg_f)
            cfg_path = cfg_f.name

        out_path = cfg_path.replace(".json", "_out.json")

        try:
            cmd = [self.rapids_python, self._script, cfg_path, out_path]
            logger.info("[GPUSolver] Running cuOpt subprocess: %s", " ".join(cmd))

            proc = subprocess.run(
                cmd,
                capture_output=True,
                text=True,
                timeout=self.timeout,
            )

            if proc.returncode != 0:
                logger.warning(
                    "[GPUSolver] cuOpt subprocess failed (rc=%d):\n%s",
                    proc.returncode, proc.stderr,
                )
                return VRPResult(routes=[], total_cost=float("inf"),
                                 solver="cuopt", status=f"subprocess_error rc={proc.returncode}")

            with open(out_path) as f:
                result = json.load(f)

            if result.get("status") != "success":
                logger.warning("[GPUSolver] cuOpt solution status: %s", result.get("status"))
                return VRPResult(routes=[], total_cost=float("inf"),
                                 solver="cuopt", status=result.get("status", "unknown"))

            routes    = result["routes"]
            total_c   = _compute_route_cost(routes, dist_matrix, depots)
            logger.info("[GPUSolver] cuOpt solved. total_cost=%.2f  routes=%s",
                        total_c, routes)
            return VRPResult(routes=routes, total_cost=total_c,
                             solver="cuopt", status="success")

        except subprocess.TimeoutExpired:
            logger.error("[GPUSolver] cuOpt subprocess timed out after %ds", self.timeout)
            return VRPResult(routes=[], total_cost=float("inf"),
                             solver="cuopt", status="timeout")
        except Exception as exc:
            logger.error("[GPUSolver] Unexpected error: %s", exc)
            return VRPResult(routes=[], total_cost=float("inf"),
                             solver="cuopt", status=f"error: {exc}")
        finally:
            for p in (cfg_path, out_path):
                try:
                    os.unlink(p)
                except FileNotFoundError:
                    pass


# ─── CPU / OR-Tools backend ──────────────────────────────────────────────────

class ORToolsSolver:
    """CPU fallback using Google OR-Tools ``RoutingModel``.

    Parameters
    ----------
    time_limit:
        Maximum seconds for the local-search phase.
    span_cost_coefficient:
        Weight applied to ``SetGlobalSpanCostCoefficient`` on the Distance
        dimension.  Higher values more aggressively equalise route lengths.
    """

    def __init__(self, time_limit: int = 60, span_cost_coefficient: int = 100):
        self.time_limit            = time_limit
        self.span_cost_coefficient = span_cost_coefficient

    # ------------------------------------------------------------------

    def solve(
        self,
        dist_matrix:  np.ndarray,
        num_vehicles: int,
        depot:        Union[int, List[int]] = 0,
    ) -> VRPResult:
        """Solve capacitated VRP with OR-Tools GUIDED_LOCAL_SEARCH.

        Parameters
        ----------
        dist_matrix:
            Square ``(N, N)`` float32 / int array.
        num_vehicles:
            Number of AUVs / robots.
        depot:
            Depot index (int) shared by all vehicles, or a list of
            per-vehicle depot indices.
        """
        depots = _normalise_depot(depot, num_vehicles)

        try:
            from ortools.constraint_solver import pywrapcp, routing_enums_pb2
        except ImportError as exc:
            logger.error("[ORToolsSolver] ortools not installed: %s", exc)
            return VRPResult(routes=[], total_cost=float("inf"),
                             solver="ortools", status=f"import_error: {exc}")

        n = dist_matrix.shape[0]
        # OR-Tools requires integer distances
        int_matrix = (dist_matrix * 1000).astype(int)

        # Multi-depot: each vehicle has its own start/end node
        manager = pywrapcp.RoutingIndexManager(n, num_vehicles, depots, depots)
        routing = pywrapcp.RoutingModel(manager)

        def distance_callback(from_idx, to_idx):
            f = manager.IndexToNode(from_idx)
            t = manager.IndexToNode(to_idx)
            return int(int_matrix[f, t])

        cb_idx = routing.RegisterTransitCallback(distance_callback)
        routing.SetArcCostEvaluatorOfAllVehicles(cb_idx)

        depot_set = set(depots)

        # Dimension: distance — also penalise imbalance via span cost
        routing.AddDimension(
            cb_idx,
            0,                # no slack
            10_000_000,       # large upper bound
            True,             # start cumul at zero
            "Distance",
        )
        dist_dim = routing.GetDimensionOrDie("Distance")
        dist_dim.SetGlobalSpanCostCoefficient(self.span_cost_coefficient)

        # Dimension: stop count — hard-cap each vehicle at balanced load
        n_inspection = n - len(depot_set)
        balanced_cap = math.ceil(n_inspection / num_vehicles) + 1

        def count_callback(from_idx, to_idx):
            node = manager.IndexToNode(to_idx)
            return 0 if node in depot_set else 1

        count_cb_idx = routing.RegisterTransitCallback(count_callback)
        routing.AddDimension(
            count_cb_idx,
            0,             # no slack
            balanced_cap,  # max stops per vehicle
            True,          # start at zero
            "StopCount",
        )

        search_params = pywrapcp.DefaultRoutingSearchParameters()
        search_params.first_solution_strategy   = (
            routing_enums_pb2.FirstSolutionStrategy.PATH_CHEAPEST_ARC
        )
        search_params.local_search_metaheuristic = (
            routing_enums_pb2.LocalSearchMetaheuristic.GUIDED_LOCAL_SEARCH
        )
        search_params.time_limit.seconds = self.time_limit

        logger.info("[ORToolsSolver] Solving with GLS (n=%d, vehicles=%d, limit=%ds)…",
                    n, num_vehicles, self.time_limit)
        solution = routing.SolveWithParameters(search_params)

        if not solution:
            logger.warning("[ORToolsSolver] No solution found.")
            return VRPResult(routes=[], total_cost=float("inf"),
                             solver="ortools", status="no_solution")

        routes: List[List[int]] = []
        for v in range(num_vehicles):
            route = []
            idx = routing.Start(v)
            while not routing.IsEnd(idx):
                node = manager.IndexToNode(idx)
                if node not in depot_set:
                    route.append(node)
                idx = solution.Value(routing.NextVar(idx))
            routes.append(route)

        total_c = _compute_route_cost(routes, dist_matrix, depots)
        logger.info("[ORToolsSolver] Solved. total_cost=%.2f  routes=%s",
                    total_c, routes)
        return VRPResult(routes=routes, total_cost=total_c,
                         solver="ortools", status="success")


# ─── Unified solver selector ─────────────────────────────────────────────────

def solve_vrp(
    dist_matrix:           np.ndarray,
    num_vehicles:          int,
    depot:                 Union[int, List[int]] = 0,
    backend:               str = "auto",
    rapids_python:         str = RAPIDS_PYTHON,
    service_time:          float = CUOPT_SERVICE_TIME,
    time_limit:            int = 60,
    gpu_timeout:           int = 300,
    span_cost_coefficient: int = 100,
) -> VRPResult:
    """Unified entry-point for VRP solving.

    Parameters
    ----------
    dist_matrix:
        Square ``(N, N)`` cost matrix.
    num_vehicles:
        Number of AUVs / robots.
    depot:
        Single depot index (int) for all robots, or a per-vehicle
        list of depot indices (length = num_vehicles).
    backend:
        ``"auto"`` – try cuOpt, fall back to OR-Tools on failure.
        ``"cuopt"`` – use cuOpt only.
        ``"ortools"`` – use OR-Tools only.
    rapids_python, service_time, gpu_timeout:
        Forwarded to :class:`GPUSolver`.
    time_limit:
        Seconds forwarded to :class:`ORToolsSolver`.
    """
    use_gpu    = backend in ("auto", "cuopt")
    use_ortools = backend in ("auto", "ortools")

    if use_gpu:
        gpu_solver = GPUSolver(
            rapids_python=rapids_python,
            service_time=service_time,
            timeout=gpu_timeout,
        )
        result = gpu_solver.solve(dist_matrix, num_vehicles, depot)
        if result.status == "success":
            return result
        logger.warning("[solve_vrp] cuOpt failed (%s), falling back to OR-Tools.",
                       result.status)

    if use_ortools:
        cpu_solver = ORToolsSolver(
            time_limit=time_limit,
            span_cost_coefficient=span_cost_coefficient,
        )
        return cpu_solver.solve(dist_matrix, num_vehicles, depot)

    return VRPResult(routes=[], total_cost=float("inf"),
                     solver="none", status="no_backend_selected")
