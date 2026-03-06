#!/usr/bin/env python3
"""
VRP Planner – Isaac Sim Visualization Entry Point
==================================================

Loads a pre-planned :class:`~route_executor.ExecutionResult` from a pickle
file and replays it inside Isaac Sim.  This script is intended to be run in
the **Isaac Sim conda environment** where cuRobo, cuOpt and OR-Tools are
*not* required.

Usage
-----
Activate the Isaac Sim environment first, then::

    python VRP/visualize_solution.py --solution_file vrp_solution.pkl

    # Headless (no GUI) mode:
    python VRP/visualize_solution.py --solution_file vrp_solution.pkl --headless

Generating the solution file
-----------------------------
In the planning environment (where cuRobo is available)::

    python run_vrp.py --num_robots 2 --random_waypoints 5 --save_solution vrp_solution.pkl

The ``--save_solution`` flag (default: ``vrp_solution.pkl``) is always active
in :mod:`run_vrp`, so running the planner always produces a replayable file.
"""

from __future__ import annotations

import argparse
import logging
import os
import sys

# Allow running as a top-level script as well as from inside the VRP package
_REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Replay a VRP solution in Isaac Sim",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument(
        "--solution_file", "-s",
        type=str,
        default="vrp_solution.pkl",
        help="Path to the .pkl file written by run_vrp.py --save_solution.",
    )
    p.add_argument(
        "--headless",
        action="store_true",
        help="Launch Isaac Sim without a GUI window.",
    )
    p.add_argument(
        "--oceansim",
        action="store_true",
        help=(
            "Use the OceanSim underwater renderer instead of the vanilla Isaac Sim "
            "visualizer.  Requires the isaaclab conda environment with the OceanSim "
            "extension installed."
        ),
    )
    p.add_argument(
        "--no_uw_camera",
        action="store_true",
        help=(
            "When --oceansim is active, skip attaching the OceanSim UW_Camera sensor "
            "(faster startup, useful for batch evaluation runs)."
        ),
    )
    p.add_argument(
        "--no_caustics",
        action="store_true",
        help=(
            "When --oceansim is active, skip enabling RTX caustics "
            "(useful on GPUs / drivers where caustics are unsupported)."
        ),
    )
    p.add_argument(
        "--uw_camera_robot",
        type=int,
        default=0,
        metavar="IDX",
        help="Robot index (0-based) that gets the UW_Camera attached (--oceansim only).",
    )
    p.add_argument(
        "--verbose", "-v",
        action="store_true",
        help="Enable DEBUG logging.",
    )
    return p.parse_args()


def main() -> None:
    args = parse_args()

    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(levelname)-8s %(name)s: %(message)s",
    )
    log = logging.getLogger(__name__)

    # ── Load solution ────────────────────────────────────────────────
    solution_path = os.path.abspath(args.solution_file)
    if not os.path.isfile(solution_path):
        log.error("Solution file not found: %s", solution_path)
        log.error(
            "Generate one first with:\n"
            "  python run_vrp.py --num_robots 2 --random_waypoints 5 "
            "--save_solution vrp_solution.pkl"
        )
        sys.exit(1)

    log.info("Loading solution from: %s", solution_path)
    from VRP.utils import load_solution
    exec_result = load_solution(solution_path)

    num_robots = len(exec_result.all_waypoints)
    total_wps  = sum(len(r) for r in exec_result.all_waypoints)
    total_steps = len(exec_result.all_traj_positions[0]) if exec_result.all_traj_positions else 0
    log.info(
        "Solution loaded: %d robot(s), %d total waypoints, %d trajectory steps",
        num_robots, total_wps, total_steps,
    )

    # ── Replay in Isaac Sim (vanilla or OceanSim) ────────────────────
    if args.oceansim:
        log.info(
            "Launching OceanSim underwater replay "
            "(headless=%s, uw_camera=%s, caustics=%s) …",
            args.headless,
            not args.no_uw_camera,
            not args.no_caustics,
        )
        from VRP.visualization_oceansim import replay_in_oceansim
        replay_in_oceansim(
            exec_result,
            headless=args.headless,
            uw_camera=not args.no_uw_camera,
            uw_camera_robot_idx=args.uw_camera_robot,
            caustics=not args.no_caustics,
        )
    else:
        log.info("Launching vanilla Isaac Sim replay (headless=%s) …", args.headless)
        from VRP.visualization import replay_in_isaac_sim
        replay_in_isaac_sim(exec_result, headless=args.headless)


if __name__ == "__main__":
    main()
