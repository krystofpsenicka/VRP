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
import os
import subprocess
import sys
import tempfile
from dataclasses import dataclass, field
from pathlib import Path
from typing import List, Optional

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


# ─── Shared helper ───────────────────────────────────────────────────────────

def _compute_route_cost(routes: List[List[int]],
                        dist_matrix: np.ndarray,
                        depot: int = 0) -> float:
    """Sum travel distances for all vehicles."""
    total = 0.0
    for route in routes:
        if not route:
            continue
        full = [depot] + list(route) + [depot]
        for a, b in zip(full[:-1], full[1:]):
            total += float(dist_matrix[a, b])
    return total


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
        depot:         int = 0,
    ) -> VRPResult:
        """Run cuOpt VRP inside the ``rapids_solver`` subprocess.

        Parameters
        ----------
        dist_matrix:
            Square ``(N, N)`` float32 array of pairwise distances.
        num_vehicles:
            Number of AUVs / robots.
        depot:
            Index of the depot node (all vehicles start and end here).
        """
        n = dist_matrix.shape[0]

        cfg = {
            "n":               n,
            "num_vehicles":    num_vehicles,
            "service_time":    self.service_time,
            "vehicle_capacity": n,
            "depot":           depot,
            "distance_matrix": dist_matrix.flatten().tolist(),
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
            total_c   = _compute_route_cost(routes, dist_matrix, depot)
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
    """

    def __init__(self, time_limit: int = 60):
        self.time_limit = time_limit

    # ------------------------------------------------------------------

    def solve(
        self,
        dist_matrix:  np.ndarray,
        num_vehicles: int,
        depot:        int = 0,
    ) -> VRPResult:
        """Solve capacitated VRP with OR-Tools GUIDED_LOCAL_SEARCH.

        Parameters
        ----------
        dist_matrix:
            Square ``(N, N)`` float32 / int array.
        num_vehicles:
            Number of AUVs / robots.
        depot:
            Index of the depot node.
        """
        try:
            from ortools.constraint_solver import pywrapcp, routing_enums_pb2
        except ImportError as exc:
            logger.error("[ORToolsSolver] ortools not installed: %s", exc)
            return VRPResult(routes=[], total_cost=float("inf"),
                             solver="ortools", status=f"import_error: {exc}")

        n = dist_matrix.shape[0]
        # OR-Tools requires integer distances
        int_matrix = (dist_matrix * 1000).astype(int)

        manager = pywrapcp.RoutingIndexManager(n, num_vehicles, depot)
        routing = pywrapcp.RoutingModel(manager)

        def distance_callback(from_idx, to_idx):
            f = manager.IndexToNode(from_idx)
            t = manager.IndexToNode(to_idx)
            return int(int_matrix[f, t])

        cb_idx = routing.RegisterTransitCallback(distance_callback)
        routing.SetArcCostEvaluatorOfAllVehicles(cb_idx)

        # Dimension: distance
        routing.AddDimension(
            cb_idx,
            0,                # no slack
            10_000_000,       # large upper bound
            True,             # start cumul at zero
            "Distance",
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
                if node != depot:
                    route.append(node)
                idx = solution.Value(routing.NextVar(idx))
            routes.append(route)

        total_c = _compute_route_cost(routes, dist_matrix, depot)
        logger.info("[ORToolsSolver] Solved. total_cost=%.2f  routes=%s",
                    total_c, routes)
        return VRPResult(routes=routes, total_cost=total_c,
                         solver="ortools", status="success")


# ─── Unified solver selector ─────────────────────────────────────────────────

def solve_vrp(
    dist_matrix:   np.ndarray,
    num_vehicles:  int,
    depot:         int = 0,
    backend:       str = "auto",
    rapids_python: str = RAPIDS_PYTHON,
    service_time:  float = CUOPT_SERVICE_TIME,
    time_limit:    int = 60,
    gpu_timeout:   int = 300,
) -> VRPResult:
    """Unified entry-point for VRP solving.

    Parameters
    ----------
    dist_matrix:
        Square ``(N, N)`` cost matrix.
    num_vehicles:
        Number of AUVs / robots.
    depot:
        Starting node index for all robots.
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
        cpu_solver = ORToolsSolver(time_limit=time_limit)
        return cpu_solver.solve(dist_matrix, num_vehicles, depot)

    return VRPResult(routes=[], total_cost=float("inf"),
                     solver="none", status="no_backend_selected")
