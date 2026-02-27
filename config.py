"""
VRP Planner – Central Configuration
All tuneable constants and paths in one place.
"""

import os

# ── Paths ──────────────────────────────────────────────────────────────────────
VRP_ROOT      = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT  = os.path.dirname(VRP_ROOT)
BROV_ROOT     = os.path.join(PROJECT_ROOT, "brov_auv_curobo")
ASSETS_PATH   = os.path.join(BROV_ROOT, "assets")
CONFIGS_PATH  = os.path.join(BROV_ROOT, "configs")
ROBOT_CFG_DIR = os.path.join(CONFIGS_PATH, "robot")

MESH_PATH = os.path.join(ASSETS_PATH, "environment", "duke_of_lancaster_uk.glb")

# ── Occupancy grid ─────────────────────────────────────────────────────────────
VOXEL_RESOLUTION = 0.10        # metres per voxel edge
ROBOT_RADIUS     = 0.35        # collision sphere radius (brov.yml)
# Inflate occupancy by this many voxels on each side
INFLATION_VOXELS = int(ROBOT_RADIUS / VOXEL_RESOLUTION) + 1   # ≥ 4

# ── Robot physical / planning constants (ported from run_multi_auv_waypoints.py) ─
BROV_CUBOID_DIMS   = [0.7, 0.5, 0.35]
OBSTACLE_CUBOID_DIMS = [d * 2 for d in BROV_CUBOID_DIMS]      # Minkowski sum
TRAJOPT_HORIZON    = 64

# Static obstacles in the environment (same as reference script)
STATIC_OBSTACLES = {
    "rock_1":         {"dims": [0.5, 0.5, 0.4], "pose": [2.0,  1.0, 0.5,  1.0, 0.0, 0.0, 0.0]},
    "rock_2":         {"dims": [0.4, 0.3, 0.3], "pose": [1.5, -1.0, 0.3,  1.0, 0.0, 0.0, 0.0]},
    "coral_structure":{"dims": [0.3, 0.3, 0.6], "pose": [3.0,  0.0, 0.4,  1.0, 0.0, 0.0, 0.0]},
}

# ── cuRobo hardened planner settings (per PDF spec) ───────────────────────────
CUROBO_NUM_TRAJOPT_SEEDS      = 1024
CUROBO_NUM_GRAPH_SEEDS        = 1024
CUROBO_INTERPOLATION_DT       = 0.02
CUROBO_TRAJOPT_TSTEPS         = 64
CUROBO_FIX_TERMINAL_ACTION    = True
CUROBO_MAX_ATTEMPTS           = 10

# dt-scaling retry ladder (interpolation_dt values to try in order)
CUROBO_DT_LADDER = [0.02, 0.05, 0.10]

# Position-only fallback orientation weights: [rot_x, rot_y, rot_z, pos_x, pos_y, pos_z]
CUROBO_POS_ONLY_WEIGHTS = [0.0, 0.0, 0.0, 1.0, 1.0, 1.0]

# ── cuOpt / cuGraph subprocess ─────────────────────────────────────────────────
# Path to the Python binary inside the RAPIDS conda env.
# Override at runtime with env-var RAPIDS_PYTHON if conda path differs.
_default_rapids_python = os.path.expanduser(
    "~/miniconda3/envs/rapids_solver/bin/python"
)
RAPIDS_PYTHON = os.environ.get("RAPIDS_PYTHON", _default_rapids_python)

# Service time per waypoint for cuOpt temporal separation (seconds)
CUOPT_SERVICE_TIME = 10

# ── OR-Tools fallback settings ─────────────────────────────────────────────────
ORTOOLS_TIME_LIMIT_S = 30
ORTOOLS_FIRST_SOLUTION = "PARALLEL_CHEAPEST_INSERTION"
ORTOOLS_LOCAL_SEARCH   = "GUIDED_LOCAL_SEARCH"

# ── Traffic-light conflict resolution ─────────────────────────────────────────
# Width of the bounding corridor around a path segment (metres)
TRAFFIC_CORRIDOR_WIDTH = ROBOT_RADIUS * 2.5   # 0.875 m
# Maximum wait-steps to inject before giving up and re-solving VRP
TRAFFIC_MAX_WAIT_STEPS = 200

# ── Isaac Sim replay ───────────────────────────────────────────────────────────
WAIT_STEPS_PER_WAYPOINT = 25   # sim steps to hold at each waypoint
STEPS_PER_WAYPOINT      = 64   # estimated motion steps per waypoint leg (for traffic-light timing)
