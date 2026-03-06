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

MESH_PATH = os.path.join(ASSETS_PATH, "environment", "duke_of_lancaster_uk_clipped.glb")
# Pose [x, y, z, qw, qx, qy, qz] for the mesh in the Isaac Sim stage
# 180° rotation about X-axis (qw=0, qx=1) to flip the GLB mesh right-side up.
MESH_POSE = [0, 0, 1.5, 0.0, 1.0, 0.0, 0.0]
# Target length (metres) of the ship along its longest axis.
# The mesh is uniformly scaled at load time so that
# mesh.extents.max() == MESH_TARGET_LENGTH.
MESH_TARGET_LENGTH = 40.0

# ── Occupancy grid ─────────────────────────────────────────────────────────────
VOXEL_RESOLUTION = 0.10        # metres per voxel edge
# ESDF voxel resolution (metres). Defaults to VOXEL_RESOLUTION but can be
# tuned independently for the cuRobo VOXEL collision checker.
ESDF_VOXEL_RESOLUTION = VOXEL_RESOLUTION
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

# Seeds used when constructing MotionGen (pre-allocated GPU buffer).
# 1024 is ideal for single-robot but OOMs on 24 GB with multiple robots.
# Match the reference script (run_multi_auv_waypoints.py) for multi-robot.
CUROBO_NUM_SEEDS_MULTI = 12

# Position-only fallback orientation weights: [rot_x, rot_y, rot_z, pos_x, pos_y, pos_z]
CUROBO_POS_ONLY_WEIGHTS = [0.0, 0.0, 0.0, 1.0, 1.0, 1.0]

# Hard wall-clock budget per segment (seconds). Planning is aborted after this
# regardless of how many dt-ladder / planner layers remain.
SEGMENT_PLAN_BUDGET_S = 45.0

# ── cuOpt / cuGraph subprocess ─────────────────────────────────────────────────
# Path to the Python binary inside the RAPIDS conda env.
# Override at runtime with env-var RAPIDS_PYTHON if conda path differs.
def _find_rapids_python() -> str:
    """Search common conda/mamba prefixes for a rapids_solver environment."""
    search_roots = [
        os.path.expanduser("~/miniconda3"),
        os.path.expanduser("~/anaconda3"),
        os.path.expanduser("~/miniforge3"),
        os.path.expanduser("~/mambaforge"),
        os.path.expanduser("~/.conda"),
    ]
    for root in search_roots:
        candidate = os.path.join(root, "envs", "rapids_solver", "bin", "python")
        if os.path.isfile(candidate):
            return candidate
    # fall back to the miniconda3 path; will produce a clear error at runtime
    return os.path.expanduser("~/miniconda3/envs/rapids_solver/bin/python")

RAPIDS_PYTHON = os.environ.get("RAPIDS_PYTHON", _find_rapids_python())

# Service time per waypoint for cuOpt temporal separation (seconds)
CUOPT_SERVICE_TIME = 2

# ── OR-Tools fallback settings ─────────────────────────────────────────────────
ORTOOLS_TIME_LIMIT_S = 30
ORTOOLS_FIRST_SOLUTION = "PARALLEL_CHEAPEST_INSERTION"
ORTOOLS_LOCAL_SEARCH   = "GUIDED_LOCAL_SEARCH"

# ── Traffic-light conflict resolution ─────────────────────────────────────────
# Width of the bounding corridor around a path segment (metres)
TRAFFIC_CORRIDOR_WIDTH = ROBOT_RADIUS * 2.5   # 0.875 m
# Maximum wait-steps to inject before giving up and re-solving VRP
TRAFFIC_MAX_WAIT_STEPS = 200
# ── Space-Time A* collision avoidance ──────────────────────────────────────
# Coarse spatial resolution for the 4-D reservation table (metres).
# Chosen so the dense 4-D array fits comfortably in RAM (~80 MB).
SPACE_TIME_RESOLUTION = 0.50
# Time step for the coarse Space-Time A* search (seconds).
# At 2 m/s cruise the robot moves exactly 1 coarse voxel per step.
SPACE_TIME_DT = SPACE_TIME_RESOLUTION / 2.0          # 0.25 s
# Maximum planning horizon (seconds).  Constrains reservation table size.
SPACE_TIME_MAX_HORIZON_S = 400.0
# Maximum contiguous wait steps before declaring a deadlock
SPACE_TIME_MAX_WAIT = 200
# Inspection-dwell hold time at each waypoint (seconds)
SPACE_TIME_DWELL_S = 2.0

# Extra reservation‐table inflation (in coarse voxels) to compensate for
# cubic-spline overshoot during smooth interpolation.  At 0.5 m resolution a
# margin of 1 adds 0.5 m clearance on each side – well above the ~0.15 m
# worst-case spline deviation.
SPLINE_SAFETY_VOXELS = 1

# ── Camera geometry (from URDF kinematic chain) ───────────────────────────────
# Total offset of the camera_optical_frame from base_link along the robot's
# forward (+X) axis at zero joint angles (0.20 + 0.02 + 0.02 + 0.06 m).
CAMERA_OFFSET_FORWARD = 0.30    # metres – camera is this far ahead of body centre
CAMERA_OFFSET_UP      = 0.05   # metres – camera is this far above body centre

# ── AUV dynamics ──────────────────────────────────────────────────────────────
AUV_CRUISE_SPEED = 2.0          # m/s nominal cruise speed
AUV_MAX_ACCEL    = 1.5          # m/s² (for future trapezoidal profile)
# ── Isaac Sim replay ───────────────────────────────────────────────────────────
WAIT_STEPS_PER_WAYPOINT = 25   # sim steps to hold at each waypoint
STEPS_PER_WAYPOINT      = 64   # estimated motion steps per waypoint leg (for traffic-light timing)

# ── OceanSim underwater visualisation ─────────────────────────────────────────
# Which BROV model to use from OceanSim assets (Bluerov/BROV_low.usd or BROV_high.usd).
# Use BROV_high.usd for fidelity when running a single robot; BROV_low.usd is faster
# with multiple robots.
OCEANSIM_BROV_USD = "BROV_low.usd"

# OceanSim UW_Camera rendering parameters (Akkaynak & Treibitz revised model):
#   [0:3]  backscatter_value  – RGB (0-1): colour of the scattered light haze
#   [3:6]  backscatter_coeff  – RGB: how quickly backscatter builds up with distance
#   [6:9]  attenuation_coeff  – RGB: how quickly direct signal attenuates with distance
# Defaults represent typical coastal / slightly turbid water.  Tune via the
# OceanSim Color Picker tool (OceanSim → Color Picker in the Isaac Sim UI).
OCEANSIM_UW_PARAMS: list = [
    0.0,  0.31, 0.24,   # backscatter value  (R, G, B)
    0.05, 0.05, 0.20,   # backscatter coeff  (R, G, B)
    0.05, 0.05, 0.05,   # attenuation coeff  (R, G, B)
]

# Z-coordinate of the water surface in the stage (metres, stage-up = +Z).
# The water-surface USD is translated to this height; the barometer sensor also
# uses this value.  Set to the approximate top of the scene / above the ship.
OCEANSIM_WATER_SURFACE_Z = 14.0
