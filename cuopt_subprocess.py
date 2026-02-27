"""
VRP Planner – cuOpt Subprocess Script

This script runs **inside** the ``rapids_solver`` conda environment.
It is invoked by ``vrp_solver.py`` via a subprocess call with a
JSON config path as the sole argument.

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


def main():
    if len(sys.argv) < 3:
        print("Usage: cuopt_subprocess.py <config.json> <output.json>")
        sys.exit(1)

    config_path = sys.argv[1]
    out_path    = sys.argv[2]

    with open(config_path) as f:
        cfg = json.load(f)

    n               = int(cfg["n"])
    num_vehicles    = int(cfg["num_vehicles"])
    service_time    = float(cfg.get("service_time", 10.0))
    vehicle_capacity= int(cfg.get("vehicle_capacity", n))
    flat_matrix     = cfg["distance_matrix"]
    depot_cfg       = cfg.get("depot", 0)

    matrix = np.array(flat_matrix, dtype=np.float32).reshape(n, n)

    # Depot: per-robot list or single index
    if isinstance(depot_cfg, list):
        depots = [int(d) for d in depot_cfg]
    else:
        depots = [int(depot_cfg)] * num_vehicles

    # ── cuOpt routing ─────────────────────────────────────────────────────────
    try:
        from cuopt import routing  # type: ignore

        solver = routing.Solver(num_vehicles, n)
        solver.set_distance_matrix(matrix)

        # Service times create temporal separation between vehicles
        service_times = np.full(n, service_time, dtype=np.float32)
        solver.set_service_times(service_times)

        # Per-vehicle settings
        solver.set_vehicle_capacity(np.full(num_vehicles, vehicle_capacity, dtype=np.int32))
        solver.set_start_locations(np.array(depots, dtype=np.int32))
        solver.set_end_locations(np.array(depots, dtype=np.int32))

        # Demand: 1 unit per waypoint so capacity constraints work
        demands = np.ones(n, dtype=np.int32)
        demands[depots[0]] = 0   # depot has zero demand
        solver.set_transit_demand(demands)

        result = solver.solve()
        if result is None:
            raise RuntimeError("cuOpt returned None solution")

        routes = []
        for v in range(num_vehicles):
            route = result.get_route(v)
            # cuOpt pads routes with depot index; strip leading/trailing depot
            route = [int(r) for r in route if r not in depots]
            routes.append(route)

        out = {"routes": routes, "status": "success", "solver": "cuopt"}

    except Exception as e:
        out = {"routes": [], "status": f"cuopt_error: {e}", "solver": "cuopt"}
        print(f"[cuOpt] ERROR: {e}", file=sys.stderr)
        with open(out_path, "w") as f:
            json.dump(out, f)
        sys.exit(1)

    with open(out_path, "w") as f:
        json.dump(out, f)
    print(f"[cuOpt] Solution written to {out_path}  routes={routes}")


if __name__ == "__main__":
    main()
