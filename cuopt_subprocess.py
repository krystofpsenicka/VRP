"""
VRP Planner – cuOpt Subprocess Script

This script runs **inside** the ``rapids_solver`` conda environment.
It is invoked by ``vrp_solver.py`` via a subprocess call with a
JSON config path as the sole argument.

API: RAPIDS 24/25 – ``from cuopt.routing import DataModel, Solve, SolverSettings``

Input JSON keys:
  distance_matrix : flat list (N*N floats, row-major)
  n               : number of waypoints (N)
  num_vehicles    : number of AUVs / robots
  service_time    : seconds to dwell at each waypoint (for temporal separation)
  depot           : list of per-robot depot indices (length = num_vehicles),
                    OR a single int repeated for all robots.
  vehicle_capacity: (optional) int capacity per vehicle; default = N

Output JSON keys (written to ``out_path``):
  routes          : list of lists – per-vehicle ordered waypoint indices
                    (excluding depot padding that cuOpt adds)
  status          : "success" or "no_solution"
  solver          : "cuopt"

If cuOpt fails or is unavailable this script exits with code 1.
"""

from __future__ import annotations

import json
import sys
import numpy as np


# ── Column-name candidates (different cuOpt versions use different names) ──
_VEHICLE_COLS = ("truck_id", "vehicle_id", "vehicle", "v_id")
_LOC_COLS     = ("route", "location", "node", "loc", "stop")


def _extract_routes(route_data, num_vehicles: int) -> list[list[int]]:
    """Convert the Assignment.get_route() cudf DataFrame into per-vehicle lists.

    Expected columns: ``truck_id``, ``location``, ``type``
    Rows with ``type == 'Depot'`` are depot stops and are excluded.
    """
    if hasattr(route_data, "to_pandas"):
        df = route_data.to_pandas()
    else:
        import pandas as pd
        df = pd.DataFrame(route_data)

    print(f"[cuOpt] route_df columns: {list(df.columns)}", flush=True)
    print(f"[cuOpt] route_df:\n{df}", flush=True)

    vehicle_col = next((c for c in df.columns if c in _VEHICLE_COLS), None)
    loc_col     = next((c for c in df.columns if c in _LOC_COLS),     None)

    if vehicle_col is None or loc_col is None:
        raise RuntimeError(
            f"Cannot identify vehicle/location columns; columns={list(df.columns)}"
        )

    # Filter out depot rows if a 'type' column is available
    if "type" in df.columns:
        df = df[df["type"] != "Depot"]

    routes: list[list[int]] = [[] for _ in range(num_vehicles)]
    for _, row in df.iterrows():
        vid = int(row[vehicle_col])
        loc = int(row[loc_col])
        if 0 <= vid < num_vehicles:
            routes[vid].append(loc)
    return routes


def _solve(matrix: np.ndarray, num_vehicles: int, n: int,
           service_time: float, depots: list[int], vehicle_capacity: int,
           ) -> list[list[int]]:
    """Solve VRP with cuOpt DataModel / Solve / SolverSettings API (RAPIDS 24/25)."""
    import cudf                                                          # type: ignore
    from cuopt.routing import DataModel, Solve, SolverSettings          # type: ignore

    data_model = DataModel(n, num_vehicles)

    # Cost matrix: cudf DataFrame (rows = origins, cols = destinations)
    data_model.add_cost_matrix(cudf.DataFrame(matrix))

    # Vehicle depot start / end
    data_model.set_vehicle_locations(
        cudf.Series(depots, dtype=np.int32),
        cudf.Series(depots, dtype=np.int32),
    )

    # Allow vehicles to skip the return-to-depot leg (open routes for AUVs)
    data_model.set_drop_return_trips(
        cudf.Series([1] * num_vehicles, dtype=np.int8)
    )

    # Capacity: each non-depot location has demand 1
    depot_set = set(depots)
    demands = [0 if i in depot_set else 1 for i in range(n)]
    data_model.add_capacity_dimension(
        "demand",
        cudf.Series(demands, dtype=np.int32),
        cudf.Series([vehicle_capacity] * num_vehicles, dtype=np.int32),
    )

    # Optional: per-location service time (dwell time)
    if service_time > 0:
        data_model.set_order_service_times(
            cudf.Series([float(service_time)] * n, dtype=np.float32)
        )

    solver_settings = SolverSettings()
    solver_settings.set_time_limit(60.0)
    solver_settings.set_verbose_mode(False)

    result = Solve(data_model, solver_settings)
    if result is None:
        raise RuntimeError("Solve() returned None")

    status = result.get_status()
    print(f"[cuOpt] status={status}", flush=True)

    # get_status() may return an int (0) or SolutionStatus enum; treat both
    status_val = int(status) if not isinstance(status, int) else status
    if status_val != 0:  # SolutionStatus.SUCCESS == 0
        msg = result.get_message() if hasattr(result, "get_message") else ""
        raise RuntimeError(f"cuOpt non-success: status={status} msg={msg}")

    routes = _extract_routes(result.get_route(), num_vehicles)
    return routes


def main() -> None:
    if len(sys.argv) < 3:
        print("Usage: cuopt_subprocess.py <config.json> <output.json>", file=sys.stderr)
        sys.exit(1)

    config_path = sys.argv[1]
    out_path    = sys.argv[2]

    with open(config_path) as f:
        cfg = json.load(f)

    n                = int(cfg["n"])
    num_vehicles     = int(cfg["num_vehicles"])
    service_time     = float(cfg.get("service_time", 10.0))
    vehicle_capacity = int(cfg.get("vehicle_capacity", n))
    flat_matrix      = cfg["distance_matrix"]
    depot_cfg        = cfg.get("depot", 0)

    matrix = np.array(flat_matrix, dtype=np.float32).reshape(n, n)

    if isinstance(depot_cfg, list):
        depots = [int(d) for d in depot_cfg]
    else:
        depots = [int(depot_cfg)] * num_vehicles

    try:
        routes = _solve(matrix, num_vehicles, n, service_time, depots,
                        vehicle_capacity)
        print(f"[cuOpt] Solved. routes={routes}", flush=True)
    except Exception as exc:
        print(f"[cuOpt] Solver error: {exc}", flush=True, file=sys.stderr)
        out = {"routes": [], "status": f"cuopt_error: {exc}", "solver": "cuopt"}
        with open(out_path, "w") as f:
            json.dump(out, f)
        sys.exit(1)

    out = {"routes": routes, "status": "success", "solver": "cuopt"}
    with open(out_path, "w") as f:
        json.dump(out, f)
    print(f"[cuOpt] Solution written to {out_path}", flush=True)


if __name__ == "__main__":
    main()

