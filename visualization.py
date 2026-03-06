"""
VRP Planner – Isaac Sim Visualisation

Replay Phase: drives N BlueROV2 articulations through the planned
trajectory buffers produced by :class:`~route_executor.RouteExecutor`.

Ported and refactored from ``brov_auv_curobo/run_multi_auv_waypoints.py``
(``replay_in_isaac_sim`` function).  Requires Isaac Sim to be running
and importable.

Only applies if you intend to use Isaac Sim as the renderer.  The module
gracefully skips if Isaac Sim is not present.
"""

from __future__ import annotations

import logging
import os
import sys
from typing import List, Optional, Tuple

import numpy as np

from .config import (
    ASSETS_PATH,
    CONFIGS_PATH,
    MESH_PATH,
    MESH_POSE,
    MESH_TARGET_LENGTH,
    STATIC_OBSTACLES,
)
from .route_executor import ExecutionResult
from .utils import load_local_robot_config

logger = logging.getLogger(__name__)


# ─── Waypoint-marker colours ─────────────────────────────────────────────────

_COLORS = [
    np.array([1.0, 0.2, 0.2]),   # red
    np.array([0.2, 0.6, 1.0]),   # blue
    np.array([0.2, 1.0, 0.2]),   # green
    np.array([1.0, 0.8, 0.0]),   # yellow
    np.array([1.0, 0.2, 1.0]),   # magenta
    np.array([0.0, 1.0, 1.0]),   # cyan
    np.array([1.0, 0.5, 0.0]),   # orange
    np.array([0.5, 0.0, 1.0]),   # purple
]


# ─── Pose helpers ────────────────────────────────────────────────────────────

def _traj8_to_pose(traj_pos: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """Convert one 8-DOF joint position to ``(xyz, quat_wxyz)``.

    The 8 DOFs are ``[x, y, z, yaw, pitch, roll, cam_yaw, cam_pitch]``.
    """
    from scipy.spatial.transform import Rotation as R
    xyz = traj_pos[:3].copy().astype(np.float64)
    yaw, pitch, roll = float(traj_pos[3]), float(traj_pos[4]), float(traj_pos[5])
    rot = R.from_euler("ZYX", [yaw, pitch, roll])
    qx, qy, qz, qw = rot.as_quat()
    return xyz, np.array([qw, qx, qy, qz], dtype=np.float64)


def _convert_trajectories(all_traj_positions: list) -> List[np.ndarray]:
    """Pre-convert all robot trajectories to ``(N, 7)`` arrays of ``[x,y,z,qw,qx,qy,qz]``."""
    result = []
    for robot_traj in all_traj_positions:
        converted = np.zeros((len(robot_traj), 7), dtype=np.float64)
        for step, tp in enumerate(robot_traj):
            xyz, qwxyz = _traj8_to_pose(tp)
            converted[step, :3] = xyz
            converted[step, 3:] = qwxyz
        result.append(converted)
    return result


def _set_robot_pose(rob_prim, xyz: np.ndarray, qwxyz: np.ndarray) -> None:
    """Teleport a USD prim by setting its xformOp:translate / xformOp:orient attributes."""
    from pxr import Gf  # type: ignore
    x, y, z = float(xyz[0]), float(xyz[1]), float(xyz[2])
    qw, qx, qy, qz = float(qwxyz[0]), float(qwxyz[1]), float(qwxyz[2]), float(qwxyz[3])

    translate_attr = rob_prim.GetAttribute("xformOp:translate")
    orient_attr    = rob_prim.GetAttribute("xformOp:orient")

    if translate_attr and translate_attr.IsValid():
        try:
            translate_attr.Set(Gf.Vec3f(x, y, z))
        except Exception:
            try:
                translate_attr.Set(Gf.Vec3d(x, y, z))
            except Exception:
                pass
    else:
        from pxr import UsdGeom  # type: ignore
        UsdGeom.Xformable(rob_prim).AddTranslateOp().Set(Gf.Vec3f(x, y, z))

    if orient_attr and orient_attr.IsValid():
        try:
            orient_attr.Set(Gf.Quatf(qw, qx, qy, qz))
        except Exception:
            try:
                orient_attr.Set(Gf.Quatd(qw, qx, qy, qz))
            except Exception:
                pass
    else:
        from pxr import UsdGeom  # type: ignore
        UsdGeom.Xformable(rob_prim).AddOrientOp().Set(Gf.Quatf(qw, qx, qy, qz))


# ─── Public entry-point ──────────────────────────────────────────────────────

def replay_in_isaac_sim(
    exec_result: ExecutionResult,
    headless:    bool = False,
) -> None:
    """Launch Isaac Sim and replay multi-robot trajectories.

    Parameters
    ----------
    exec_result:
        Result from :meth:`~route_executor.RouteExecutor.execute`.
    headless:
        If ``True``, run without a display (useful for CI / server use).
    """
    all_traj_positions = exec_result.all_traj_positions
    all_waypoints      = exec_result.all_waypoints
    num_robots         = len(all_traj_positions)
    total_steps        = max(len(t) for t in all_traj_positions) if all_traj_positions else 0
    traj_poses         = _convert_trajectories(all_traj_positions)

    # ── Bootstrap Isaac Sim ───────────────────────────────────────────
    try:
        from omni.isaac.kit import SimulationApp
    except ImportError:
        try:
            from isaacsim import SimulationApp
        except ImportError:
            logger.error(
                "[viz] Isaac Sim not found.  "
                "Skipping visual replay.  "
                "Trajectories saved to exec_result."
            )
            return

    simulation_app = SimulationApp(
        {"headless": headless, "width": "1920", "height": "1080"}
    )

    # ── Late imports (require Isaac Sim running) ──────────────────────
    from omni.isaac.core import World
    from omni.isaac.core.objects import cuboid as cuboid_mod
    from omni.isaac.core.robots import Robot
    from curobo.util.usd_helper import UsdHelper, set_prim_transform  # type: ignore

    ISAAC_SIM_45 = False
    try:
        from omni.importer.urdf import _urdf
    except ImportError:
        from isaacsim.asset.importer.urdf import _urdf  # type: ignore
        ISAAC_SIM_45 = True

    # ── Add cuRobo Isaac Sim extensions helper if present ─────────────
    try:
        import curobo as _cb  # type: ignore
        _cp  = os.path.dirname(os.path.dirname(_cb.__file__))
        _ex  = os.path.join(_cp, "examples", "isaac_sim")
        if os.path.exists(_ex):
            sys.path.insert(0, _ex)
            from helper import add_extensions  # type: ignore
            add_extensions(simulation_app, headless)
    except Exception:
        pass

    # ── World ─────────────────────────────────────────────────────────
    my_world = World(stage_units_in_meters=1.0)
    stage    = my_world.stage

    # Zero gravity (underwater AUVs)
    try:
        from pxr import UsdPhysics, Gf  # type: ignore
        ps = UsdPhysics.Scene.Get(stage, "/physicsScene")
        if not ps:
            ps = UsdPhysics.Scene.Define(stage, "/physicsScene")
        ps.GetGravityDirectionAttr().Set(Gf.Vec3f(0, 0, 0))
        ps.GetGravityMagnitudeAttr().Set(0.0)
    except Exception:
        pass

    # ── Dome light ────────────────────────────────────────────────────
    try:
        from pxr import UsdLux  # type: ignore
        dome_light = UsdLux.DomeLight.Define(stage, "/World/DomeLight")
        dome_light.GetIntensityAttr().Set(800.0)
        dome_light.GetColorAttr().Set(Gf.Vec3f(0.75, 0.85, 1.0))  # cool blue tint
        logger.info("[viz] Dome light added.")
    except Exception as _e:
        logger.warning("[viz] Could not add dome light: %s", _e)

    xform = stage.DefinePrim("/World", "Xform")
    stage.SetDefaultPrim(xform)
    stage.DefinePrim("/curobo", "Xform")

    # ── URDF import → temp USD ────────────────────────────────────────
    robot_cfg  = load_local_robot_config("brov.yml")
    urdf_iface = _urdf.acquire_urdf_interface()

    import_config = _urdf.ImportConfig()
    import_config.merge_fixed_joints       = False
    import_config.convex_decomp            = False
    import_config.fix_base                 = True
    import_config.make_default_prim        = True
    import_config.self_collision            = False
    import_config.create_physics_scene     = True
    import_config.import_inertia_tensor    = False
    import_config.default_drive_strength   = 100_000.0
    import_config.default_position_drive_damping = 10_000.0
    import_config.default_drive_type       = _urdf.UrdfJointTargetType.JOINT_DRIVE_POSITION
    import_config.distance_scale           = 1
    import_config.density                  = 0.0

    asset_path  = robot_cfg["kinematics"].get("external_asset_path", ASSETS_PATH)
    _kin        = robot_cfg["kinematics"]
    urdf_path   = os.path.join(asset_path, _kin["urdf_path"])
    robot_dir   = os.path.dirname(urdf_path)
    urdf_file   = os.path.basename(urdf_path)

    if ISAAC_SIM_45:
        import omni.kit.commands  # type: ignore
        dest_usd = os.path.join(
            robot_dir,
            os.path.splitext(urdf_file)[0] + "_vrp_temp.usd",
        )
        _, inner_prim_path = omni.kit.commands.execute(
            "URDFParseAndImportFile",
            urdf_path=os.path.join(robot_dir, urdf_file),
            import_config=import_config,
            dest_path=dest_usd,
        )
    else:
        imported = urdf_iface.parse_urdf(robot_dir, urdf_file, import_config)
        inner_prim_path = urdf_iface.import_robot(
            robot_dir, urdf_file, imported, import_config, ""
        )
        dest_usd = None

    xform = stage.GetPrimAtPath("/World") or stage.DefinePrim("/World", "Xform")
    stage.SetDefaultPrim(xform)

    # ── Spawn robots ──────────────────────────────────────────────────
    import omni.usd  # type: ignore

    robots    = []
    rob_prims = []
    for i in range(num_robots):
        dp = str(stage.GetDefaultPrim().GetPath())
        pp = omni.usd.get_stage_next_free_path(stage, dp + inner_prim_path, False)
        stage.OverridePrim(pp).GetReferences().AddReference(dest_usd)
        robot = Robot(prim_path=pp, name=f"brov_{i}")
        set_prim_transform(stage.GetPrimAtPath(pp), [0, 0, 0, 1, 0, 0, 0])
        robots.append(my_world.scene.add(robot))
        rob_prims.append(stage.GetPrimAtPath(pp))
        logger.info("[viz] Spawned robot %d  prim=%s", i, pp)

    # ── Static obstacles (mesh + cuboids) ─────────────────────────────
    from curobo.wrap.reacher.motion_gen import WorldConfig  # type: ignore

    usd_help = UsdHelper()
    usd_help.load_stage(stage)

    # Build WorldConfig that includes the environment mesh AND cuboid obstacles
    world_dict: dict = {"cuboid": STATIC_OBSTACLES}
    if os.path.isfile(MESH_PATH):
        # Compute the same uniform scale used by build_occupancy_grid()
        import trimesh as _tm
        _raw = _tm.load(MESH_PATH, force="mesh")
        if isinstance(_raw, _tm.Scene):
            _raw = _tm.util.concatenate(list(_raw.geometry.values()))
        _longest = float(_raw.extents.max())
        _mesh_scale = MESH_TARGET_LENGTH / _longest if _longest > 0 else 1.0

        world_dict["mesh"] = {
            "duke_of_lancaster": {
                "file_path": MESH_PATH,
                "pose": MESH_POSE,
                "scale": [_mesh_scale, _mesh_scale, _mesh_scale],
            }
        }
        logger.info("[viz] Adding mesh: %s  scale=%.4f", MESH_PATH, _mesh_scale)
    else:
        logger.warning("[viz] Mesh not found at %s – skipping.", MESH_PATH)

    usd_help.add_world_to_stage(
        WorldConfig.from_dict(world_dict),
        base_frame="/World",
    )
    my_world.scene.add_default_ground_plane()

    # ── Waypoint marker cubes ─────────────────────────────────────────
    for i in range(num_robots):
        color = _COLORS[i % len(_COLORS)]
        for wi, wp in enumerate(all_waypoints[i]):
            cuboid_mod.VisualCuboid(
                f"/World/wp_r{i}_{wi}",
                position=np.array(wp[:3], dtype=np.float64),
                orientation=np.array(wp[3:7], dtype=np.float64),
                color=color,
                size=0.08,
            )

    # ── Initialise physics ────────────────────────────────────────────
    if ISAAC_SIM_45:
        simulation_app.update()
        simulation_app.update()
        my_world.initialize_physics()

    logger.info(
        "[viz] Ready to replay %d steps for %d robots. Click PLAY.",
        total_steps, num_robots,
    )

    # ── Replay loop ───────────────────────────────────────────────────
    replay_idx = 0
    playing    = False

    while simulation_app.is_running():
        my_world.step(render=True)

        if not my_world.is_playing():
            playing = False
            continue

        if not playing:
            replay_idx = 0
            playing    = True
            logger.info("[viz] Replay started.")

        if replay_idx >= total_steps:
            logger.info("[viz] Replay finished – looping.")
            replay_idx = 0

        for i, prim in enumerate(rob_prims):
            safe_idx = min(replay_idx, len(traj_poses[i]) - 1)
            _set_robot_pose(prim, traj_poses[i][safe_idx, :3], traj_poses[i][safe_idx, 3:])

        replay_idx += 1
        if replay_idx % 500 == 0:
            logger.info("[viz] Replay step %d / %d", replay_idx, total_steps)

    simulation_app.close()
