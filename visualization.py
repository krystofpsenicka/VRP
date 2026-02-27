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
from typing import List, Optional

import numpy as np

from .config import (
    ASSETS_PATH,
    CONFIGS_PATH,
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
    all_traj_positions  = exec_result.all_traj_positions
    all_traj_velocities = exec_result.all_traj_velocities
    all_waypoints       = exec_result.all_waypoints
    j_names             = exec_result.joint_names
    num_robots          = len(all_traj_positions)
    total_steps         = len(all_traj_positions[0]) if all_traj_positions else 0

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
    from omni.isaac.core.utils.types import ArticulationAction
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

    robots            = []
    art_controllers   = []
    for i in range(num_robots):
        dp = str(stage.GetDefaultPrim().GetPath())
        pp = omni.usd.get_stage_next_free_path(stage, dp + inner_prim_path, False)
        stage.OverridePrim(pp).GetReferences().AddReference(dest_usd)
        robot = Robot(prim_path=pp, name=f"brov_{i}")
        set_prim_transform(stage.GetPrimAtPath(pp), [0, 0, 0, 1, 0, 0, 0])
        robots.append(my_world.scene.add(robot))
        art_controllers.append(None)
        logger.info("[viz] Spawned robot %d  prim=%s", i, pp)

    # ── Static obstacles (mesh + cuboids) ─────────────────────────────
    from curobo.wrap.reacher.motion_gen import WorldConfig  # type: ignore

    usd_help = UsdHelper()
    usd_help.load_stage(stage)
    usd_help.add_world_to_stage(
        WorldConfig.from_dict({"cuboid": STATIC_OBSTACLES}),
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
    idx_lists:    List[Optional[List[int]]] = [None] * num_robots
    replay_idx    = 0
    initialised   = False

    while simulation_app.is_running():
        my_world.step(render=True)
        if not my_world.is_playing():
            continue

        step_index = my_world.current_time_step_index

        # Init articulation controllers + drive to first frame
        if step_index <= 10:
            for i in range(num_robots):
                robots[i]._articulation_view.initialize()
                art_controllers[i] = robots[i].get_articulation_controller()

                il = []
                for jn in j_names:
                    try:
                        il.append(robots[i].get_dof_index(jn))
                    except Exception:
                        pass
                idx_lists[i] = il

                if il:
                    init_pos = all_traj_positions[i][0]
                    robots[i].set_joint_positions(init_pos.tolist(), il)
                    robots[i]._articulation_view.set_joint_position_targets(
                        positions=np.array(init_pos, dtype=np.float32).reshape(1, -1),
                        joint_indices=il,
                    )
                    robots[i]._articulation_view.set_joint_velocity_targets(
                        velocities=np.zeros((1, len(il)), dtype=np.float32),
                        joint_indices=il,
                    )
                    robots[i]._articulation_view.set_max_efforts(
                        values=np.array([50_000.0] * len(il)),
                        joint_indices=il,
                    )
            continue

        # Settling: keep driving robots to start position
        if step_index < 40:
            for i in range(num_robots):
                if idx_lists[i] and art_controllers[i]:
                    init_pos = all_traj_positions[i][0]
                    art_controllers[i].apply_action(ArticulationAction(
                        init_pos,
                        np.zeros_like(init_pos),
                        joint_indices=idx_lists[i],
                    ))
            continue

        if not initialised:
            initialised = True
            replay_idx  = 0
            logger.info("[viz] Replay started.")

        if replay_idx >= total_steps:
            logger.info("[viz] Replay finished – looping.")
            replay_idx = 0

        # Drive articulations
        for i in range(num_robots):
            if not idx_lists[i] or art_controllers[i] is None:
                continue
            tp = all_traj_positions[i][replay_idx]
            tv = all_traj_velocities[i][replay_idx]
            art_controllers[i].apply_action(ArticulationAction(
                tp, tv, joint_indices=idx_lists[i],
            ))

        # Sub-step physics for better tracking accuracy
        for _ in range(4):
            my_world.step(render=False)

        replay_idx += 1
        if replay_idx % 500 == 0:
            logger.info("[viz] Replay step %d / %d", replay_idx, total_steps)

    simulation_app.close()
