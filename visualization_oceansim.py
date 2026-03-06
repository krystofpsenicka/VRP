"""
VRP Planner – OceanSim Underwater Visualisation
================================================

Replays pre-planned :class:`~route_executor.ExecutionResult` trajectories inside
NVIDIA Isaac Sim with the OceanSim extension enabled, giving a physics-based
underwater rendering experience:

  * BlueROV2 rigid-body USD models from OceanSim assets (position/orient driven
    directly from the pre-computed trajectories, no joint articulation needed)
  * :class:`~isaacsim.oceansim.sensors.UW_Camera.UW_Camera`: GPU post-process
    applying depth-dependent attenuation + backscatter (Akkaynak & Treibitz model)
  * OceanSim water-surface USD (``mhl_water.usd``) with transparent refractive
    material for caustic-ready rendering
  * RTX caustics enabled programmatically (requires RTX Real-Time or Path-Traced
    renderer; gracefully skipped if setting path differs between IS versions)
  * Directional "sun" light with per-light caustics attribute + ambient dome light
    with a deep-blue underwater tint

Usage
-----
Run from the ``isaaclab`` conda environment::

    python VRP/visualize_solution.py --solution_file vrp_solution.pkl --oceansim

    # Headless (no GUI), no UW camera post-process:
    python VRP/visualize_solution.py --solution_file vrp_solution.pkl --oceansim \\
        --headless --no_uw_camera

Requirements
------------
* Isaac Sim 5.0 (isaaclab conda env)
* OceanSim extension installed under ``<isaaclab>/extsUser/OceanSim/``
* ``asset_path.json`` already registered (done once via
  ``python config/register_asset_path.py /path/to/OceanSim_assets``)
"""

from __future__ import annotations

import logging
import os
import sys
from typing import List, Optional, Tuple

import numpy as np

from .config import (
    ASSETS_PATH,
    MESH_PATH,
    MESH_POSE,
    MESH_TARGET_LENGTH,
    STATIC_OBSTACLES,
    OCEANSIM_BROV_USD,
    OCEANSIM_UW_PARAMS,
    OCEANSIM_WATER_SURFACE_Z,
)
from .route_executor import ExecutionResult

logger = logging.getLogger(__name__)

# ─── Per-robot waypoint-marker colours ───────────────────────────────────────

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


# ─── Helpers ─────────────────────────────────────────────────────────────────

def _traj8_to_pose(traj_pos: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """Convert one 8-DOF joint position to ``(xyz, quat_wxyz)``.

    The 8 DOFs are ``[x, y, z, yaw, pitch, roll, cam_yaw, cam_pitch]``.
    Camera joints are ignored for body-level pose control.

    Returns
    -------
    xyz : np.ndarray, shape (3,)
    quat_wxyz : np.ndarray, shape (4,)  -- ``[qw, qx, qy, qz]``
    """
    from scipy.spatial.transform import Rotation as R  # lazy
    xyz  = traj_pos[:3].copy().astype(np.float64)
    yaw, pitch, roll = float(traj_pos[3]), float(traj_pos[4]), float(traj_pos[5])
    rot  = R.from_euler("ZYX", [yaw, pitch, roll])
    qx, qy, qz, qw = rot.as_quat()   # scipy → [x,y,z,w]
    return xyz, np.array([qw, qx, qy, qz], dtype=np.float64)


def _convert_trajectories(
    all_traj_positions: list,
) -> List[np.ndarray]:
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


def _try_enable_oceansim_extension() -> None:
    """Try to enable the OceanSim extension so its Python modules are importable."""
    try:
        import omni.kit.app
        mgr = omni.kit.app.get_app().get_extension_manager()
        # The extension folder is named "OceanSim"; find it by scanning extsUser
        for ext_id in mgr.get_extensions():
            if "OceanSim" in ext_id or "oceansim" in ext_id.lower():
                mgr.set_extension_enabled_immediate(ext_id, True)
                logger.info("[oceansim] Extension enabled: %s", ext_id)
                return
        # If not found via manager, try direct enable by name
        mgr.set_extension_enabled_immediate("OceanSim", True)
        logger.info("[oceansim] Extension enabled by direct name.")
    except Exception as e:
        logger.warning("[oceansim] Could not enable OceanSim via extension manager: %s", e)


def _find_oceansim_ext_root() -> str | None:
    """Return the absolute path to the OceanSim extension root directory, or None."""
    # Strategy 1: derive from the isaacsim package location
    try:
        import isaacsim as _isc
        # isaacsim is typically at .../site-packages/isaacsim/__init__.py
        # extsUser sits alongside the package root
        for _base in [
            os.path.dirname(_isc.__file__),                           # .../isaacsim/
            os.path.dirname(os.path.dirname(_isc.__file__)),          # .../site-packages/
        ]:
            for _rel in ["extsUser/OceanSim", "isaacsim/extsUser/OceanSim"]:
                _c = os.path.join(_base, _rel)
                if os.path.isdir(_c):
                    return _c
    except Exception:
        pass

    # Strategy 2: brute-force common conda roots
    import sys as _sys
    _py_ver = f"python{_sys.version_info.major}.{_sys.version_info.minor}"
    for _conda_root in [
        os.path.expanduser("~/miniconda3"),
        os.path.expanduser("~/anaconda3"),
        os.path.expanduser("~/miniforge3"),
        os.path.expanduser("~/mambaforge"),
    ]:
        for _env in ["isaaclab"]:
            _c = os.path.join(
                _conda_root, "envs", _env,
                "lib", _py_ver, "site-packages",
                "isaacsim", "extsUser", "OceanSim",
            )
            if os.path.isdir(_c):
                return _c

    return None


def _ensure_oceansim_importable() -> bool:
    """Make sure ``isaacsim.oceansim`` is importable; try multiple strategies.

    ``isaacsim`` is a **namespace package** spread across multiple directories.
    Simply adding the OceanSim root to ``sys.path`` is not enough – Python has
    already resolved the ``isaacsim`` namespace package and won't re-scan new
    ``sys.path`` entries for its sub-packages.  The correct fix is to extend
    ``isaacsim.__path__`` with the ``isaacsim/`` sub-directory that lives inside
    the OceanSim extension folder.

    Returns ``True`` if OceanSim sensors can be imported.
    """
    try:
        import isaacsim.oceansim  # noqa: F401 – just a probe
        return True
    except ImportError:
        pass

    ext_root = _find_oceansim_ext_root()
    if ext_root is None:
        logger.error(
            "[oceansim] Could not locate OceanSim extension directory.\n"
            "  Expected path: <isaaclab_env>/lib/pythonX.Y/site-packages/"
            "isaacsim/extsUser/OceanSim\n"
            "  Make sure OceanSim is installed and asset_path.json is registered."
        )
        return False

    logger.info("[oceansim] Found OceanSim extension at: %s", ext_root)

    # The extension ships its Python code at <ext_root>/isaacsim/oceansim/.
    # We must graft that path onto the already-imported isaacsim namespace package
    # so that `import isaacsim.oceansim` resolves correctly.
    oceansim_isaacsim_dir = os.path.join(ext_root, "isaacsim")
    if os.path.isdir(oceansim_isaacsim_dir):
        try:
            import isaacsim as _isc
            if oceansim_isaacsim_dir not in _isc.__path__:
                _isc.__path__.append(oceansim_isaacsim_dir)  # type: ignore[union-attr]
                logger.info(
                    "[oceansim] Extended isaacsim.__path__ with: %s",
                    oceansim_isaacsim_dir,
                )
        except Exception as e:
            logger.warning("[oceansim] Could not extend isaacsim.__path__: %s", e)
            # Last resort: add the ext root to sys.path so a direct import works
            if ext_root not in sys.path:
                sys.path.insert(0, ext_root)
    else:
        logger.warning(
            "[oceansim] Expected isaacsim/ sub-dir not found inside %s", ext_root
        )
        if ext_root not in sys.path:
            sys.path.insert(0, ext_root)

    try:
        import isaacsim.oceansim  # noqa: F401
        logger.info("[oceansim] isaacsim.oceansim imported successfully.")
        return True
    except ImportError as e:
        logger.error(
            "[oceansim] Could not import isaacsim.oceansim: %s\n"
            "  Make sure OceanSim is installed in the isaaclab environment and "
            "asset_path.json has been registered.",
            e,
        )
        return False


def _enable_rtx_caustics() -> None:
    """Enable RTX caustics in the Omniverse render settings (best-effort)."""
    try:
        import carb

        settings = carb.settings.get_settings()

        # Different Isaac Sim / Kit versions use different paths – try both
        tried: list[str] = []
        for path in [
            "/rtx/raytracing/caustics/enabled",
            "/rtx/pathtracing/caustics/enabled",
            "/rtx/iray/caustics/enabled",
        ]:
            try:
                settings.set(path, True)
                # Verify it took effect
                if settings.get(path) is not None:
                    logger.info("[oceansim] RTX caustics enabled at: %s", path)
                    tried.append(path)
            except Exception:
                pass

        # Ensure we are in a ray-tracing mode that supports caustics
        try:
            mode = settings.get("/rtx/rendermode")
            if mode not in ("RaytracedLighting", "PathTracing"):
                settings.set("/rtx/rendermode", "RaytracedLighting")
                logger.info("[oceansim] Render mode set to RaytracedLighting for caustics.")
        except Exception:
            pass

        if not tried:
            logger.warning(
                "[oceansim] Could not locate caustics carb setting path. "
                "Caustics may need to be enabled manually: "
                "Render Settings → Ray Tracing → Caustics."
            )
    except Exception as e:
        logger.warning("[oceansim] RTX caustics setup failed (non-fatal): %s", e)


def _configure_underwater_viewport() -> None:
    """Apply RTX carb settings to give the main viewport a subtle underwater look."""
    try:
        import carb
        s = carb.settings.get_settings()
    except Exception as e:
        logger.warning("[oceansim] carb.settings unavailable – skipping viewport config: %s", e)
        return

    def _set(path: str, value, label: str = "") -> None:
        try:
            s.set(path, value)
        except Exception as exc:
            logger.debug("[oceansim] carb set failed %s: %s", path, exc)

    _set("/persistent/exts/omni.kit.environment.core/rtx/env/auto", False)

    # Subtle fog: starts 5 m away, ends 80 m, density 0.35 (not opaque)
    _set("/rtx/fog/enabled", True)
    _set("/rtx/fog/fogColor", [0.03, 0.10, 0.15])
    _set("/rtx/fog/fogColorIntensity", 0.8)
    _set("/rtx/fog/fogStartDist", 5.0)
    _set("/rtx/fog/fogEndDist", 80.0)
    _set("/rtx/fog/fogDensityAtEnd", 0.35)
    _set("/rtx/fog/fogHeight", 500.0)
    _set("/rtx/fog/fogHeightDensity", 0.0)
    _set("/rtx/fog/fogHeightFalloff", 0.0)

    # Gentle color correction: mild red attenuation only
    _set("/rtx/post/colorcorr/enabled", True)
    _set("/rtx/post/colorcorr/gain", [0.78, 0.92, 1.0])
    _set("/rtx/post/colorcorr/offset", [0.0, 0.0, 0.0])
    _set("/rtx/post/colorcorr/saturation", [1.0, 1.0, 1.0])

    # Disable color grading (was making everything too blue)
    _set("/rtx/post/colorgrad/enabled", False)

    logger.info("[oceansim] Underwater viewport configured: subtle teal fog 5–80 m.")


def _add_sun_light_with_caustics(stage) -> None:
    """Add a directional sun light shining straight down.

    A DistantLight is the correct light type for sunlight — but the water surface
    mesh occludes it, making the underwater scene black.  The occlusion is fixed in
    :func:`_load_water_surface` by disabling shadow-casting on the water mesh, so
    the directional light passes through as if the surface were transparent.

    Auto-created Isaac Sim default lights are removed so they don't compete.
    """
    try:
        from pxr import UsdLux, Gf, Sdf  # type: ignore

        # Remove Isaac Sim auto-created lights that may interfere
        for auto_path in ("/Environment", "/World/defaultLight", "/World/DistantLight"):
            prim = stage.GetPrimAtPath(auto_path)
            if prim.IsValid():
                stage.RemovePrim(auto_path)
                logger.info("[oceansim] Removed auto-created light: %s", auto_path)

        light = UsdLux.DistantLight.Define(stage, "/World/SunLight")
        light.GetIntensityAttr().Set(5000.0)
        light.GetColorAttr().Set(Gf.Vec3f(1.0, 1.0, 1.0))  # white — no color tint
        light.GetAngleAttr().Set(0.53)  # ~solar disk angle in degrees (soft shadows)

        # Enable caustics (attribute name varies by Kit version; best-effort)
        prim = light.GetPrim()
        for attr_name in ("caustics:enable", "inputs:enableCaustics"):
            try:
                prim.CreateAttribute(attr_name, Sdf.ValueTypeNames.Bool).Set(True)
            except Exception:
                pass

        logger.info("[oceansim] Sun light: DistantLight intensity=5000 pointing down.")
    except Exception as e:
        logger.warning("[oceansim] Could not add sun light: %s", e)


def _add_ambient_dome(stage) -> None:
    """Add a cool-blue dome light matching the standard (non-oceansim) visualization."""
    try:
        from pxr import UsdLux, Gf  # type: ignore

        # Remove Isaac Sim auto-created lights that may interfere
        for auto_path in ("/Environment", "/World/defaultLight", "/World/DistantLight"):
            prim = stage.GetPrimAtPath(auto_path)
            if prim.IsValid():
                stage.RemovePrim(auto_path)
                logger.info("[oceansim] Removed auto-created light: %s", auto_path)

        dome = UsdLux.DomeLight.Define(stage, "/World/DomeLight")
        dome.GetIntensityAttr().Set(800.0)
        logger.info("[oceansim] Dome light added: intensity=800.")
    except Exception as e:
        logger.warning("[oceansim] Could not add dome light: %s", e)


def _load_ship_mesh(stage, app) -> None:
    """Load the Duke of Lancaster ship mesh as a USD reference with proper scale."""
    if not os.path.isfile(MESH_PATH):
        logger.warning("[oceansim] Ship mesh not found at %s – skipping.", MESH_PATH)
        return

    try:
        import trimesh as _tm  # type: ignore
        from isaacsim.core.utils.stage import add_reference_to_stage  # type: ignore

        # Compute the same uniform scale used by build_occupancy_grid()
        raw = _tm.load(MESH_PATH, force="mesh")
        if isinstance(raw, _tm.Scene):
            raw = _tm.util.concatenate(list(raw.geometry.values()))
        longest = float(raw.extents.max())
        mesh_scale = MESH_TARGET_LENGTH / longest if longest > 0 else 1.0

        ship_prim_path = "/World/DukeOfLancaster"
        prim = add_reference_to_stage(usd_path=MESH_PATH, prim_path=ship_prim_path)

        # Apply scale and pose
        from pxr import Gf, UsdGeom  # type: ignore
        xf = UsdGeom.Xformable(stage.GetPrimAtPath(ship_prim_path))
        xf.ClearXformOpOrder()
        xf.AddTranslateOp().Set(Gf.Vec3f(*[float(v) for v in MESH_POSE[:3]]))
        xf.AddOrientOp().Set(Gf.Quatf(float(MESH_POSE[3]), float(MESH_POSE[4]),
                                        float(MESH_POSE[5]), float(MESH_POSE[6])))
        xf.AddScaleOp().Set(Gf.Vec3f(mesh_scale, mesh_scale, mesh_scale))

        # Enable collision and add OceanSim sonar reflectivity semantics
        try:
            from isaacsim.core.prims import SingleGeometryPrim  # type: ignore
            from isaacsim.core.utils.semantics import add_update_semantics  # type: ignore

            collider = SingleGeometryPrim(prim_path=ship_prim_path, collision=True)
            collider.set_collision_approximation("convexDecomposition")
            # Sonar reflectivity: 1.0 = steel/metal (ship hull)
            add_update_semantics(
                prim=stage.GetPrimAtPath(ship_prim_path),
                type_label="reflectivity",
                semantic_label="1.0",
            )
            logger.info(
                "[oceansim] Ship mesh loaded: scale=%.4f  path=%s",
                mesh_scale,
                ship_prim_path,
            )
        except Exception as e:
            logger.warning("[oceansim] Ship mesh collision/semantics setup partial: %s", e)

    except Exception as e:
        logger.error("[oceansim] Failed to load ship mesh: %s", e)


def _load_static_obstacles(stage) -> None:
    """Create cuboid obstacle prims from STATIC_OBSTACLES config."""
    try:
        from isaacsim.core.prims import SingleGeometryPrim  # type: ignore
        from pxr import UsdGeom, Gf, UsdPhysics  # type: ignore

        for name, spec in STATIC_OBSTACLES.items():
            dims = spec["dims"]
            pose = spec["pose"]  # [x,y,z, qw,qx,qy,qz]
            prim_path = f"/World/obstacles/{name}"

            # Define a cube then scale to dims
            cube = UsdGeom.Cube.Define(stage, prim_path)
            cube.GetSizeAttr().Set(1.0)
            xf = UsdGeom.Xformable(cube.GetPrim())
            xf.AddTranslateOp().Set(Gf.Vec3f(*[float(v) for v in pose[:3]]))
            xf.AddOrientOp().Set(Gf.Quatf(*[float(v) for v in pose[3:]]))
            xf.AddScaleOp().Set(Gf.Vec3f(*[float(v) for v in dims]))

            # Make it a static collider
            UsdPhysics.CollisionAPI.Apply(cube.GetPrim())
            logger.debug("[oceansim] Static obstacle: %s", name)

    except Exception as e:
        logger.warning("[oceansim] Could not create static obstacles: %s", e)


def _add_water_plane(stage) -> None:
    """Add a flat water-surface plane 1 m below the ship mesh origin."""
    try:
        from pxr import UsdGeom, Gf, Vt  # type: ignore
        water_z = float(MESH_POSE[2]) - 1.0
        sz = 500.0
        mesh = UsdGeom.Mesh.Define(stage, "/World/WaterPlane")
        mesh.GetPointsAttr().Set(Vt.Vec3fArray([
            Gf.Vec3f(-sz, -sz, 0.0), Gf.Vec3f(sz, -sz, 0.0),
            Gf.Vec3f(sz,  sz, 0.0),  Gf.Vec3f(-sz,  sz, 0.0),
        ]))
        mesh.GetFaceVertexCountsAttr().Set(Vt.IntArray([4]))
        mesh.GetFaceVertexIndicesAttr().Set(Vt.IntArray([0, 1, 2, 3]))
        mesh.GetDisplayColorAttr().Set(Vt.Vec3fArray([Gf.Vec3f(0.05, 0.25, 0.45)]))
        xf = UsdGeom.Xformable(mesh)
        xf.AddTranslateOp().Set(Gf.Vec3d(0.0, 0.0, water_z))
        logger.info("[oceansim] Water plane added at Z=%.2f m (1 m below ship mesh).", water_z)
    except Exception as e:
        logger.warning("[oceansim] Could not add water plane: %s", e)


def _load_water_surface(stage) -> None:
    """Load ``ocean.usda`` from the omni.warp extension for animated JONSWAP ocean waves.

    The ``omni.warp.WarpSampleOceanDeform`` ActionGraph node in that file implements
    a full TMA/JONSWAP spectrum wave deformation.  We reference the USD into the stage,
    translate to the water surface Z, and scale for open-water coverage.
    """
    import glob
    from isaacsim.core.utils.stage import add_reference_to_stage  # type: ignore
    from pxr import UsdGeom, Gf  # type: ignore

    # ── 1.  Locate ocean.usda inside the omni.warp extension ─────────────────
    try:
        import isaacsim as _isc
        ext_search_base = os.path.dirname(_isc.__file__)
    except Exception:
        ext_search_base = os.path.expanduser(
            "~/miniconda3/envs/isaaclab/lib/python3.11/site-packages/isaacsim"
        )

    pattern = os.path.join(
        ext_search_base, "extscache", "omni.warp-*", "data", "scenes", "ocean.usda"
    )
    matches = sorted(glob.glob(pattern))
    if not matches:
        logger.warning(
            "[oceansim] ocean.usda not found (pattern: %s). "
            "Water surface will be skipped.", pattern
        )
        return

    ocean_usda = matches[-1]  # newest installed version
    logger.info("[oceansim] Found ocean.usda: %s", ocean_usda)

    try:
        # ── 2.  Reference into the stage ──────────────────────────────────────
        add_reference_to_stage(usd_path=ocean_usda, prim_path="/World/WaterSurface")

        water_prim = stage.GetPrimAtPath("/World/WaterSurface")
        if not water_prim.IsValid():
            logger.warning("[oceansim] /World/WaterSurface prim not valid after add_reference.")
            return

        # ── 3.  Xform: place at water surface Z, scale for scene coverage ─────
        #    ocean.usda is Y-up so the wave plane lies in XZ by default.
        #    No rotation needed — user confirmed correct orientation without it.
        xf = UsdGeom.Xformable(water_prim)
        xf.ClearXformOpOrder()
        xf.AddTranslateOp().Set(Gf.Vec3d(0.0, 0.0, float(OCEANSIM_WATER_SURFACE_Z)))
        xf.AddScaleOp().Set(Gf.Vec3f(0.5, 0.5, 0.5))

        # ── 4.  Disable shadow casting on the water mesh ──────────────────────
        #    In RTX Real-Time mode a DistantLight above the water surface is
        #    blocked by the water geometry, making the underwater scene black.
        #    Setting primvars:doNotCastShadows lets sunlight pass straight
        #    through — correct for a transparent/refractive water surface.
        from pxr import Sdf  # type: ignore
        for prim in stage.Traverse():
            path_str = str(prim.GetPath())
            if path_str.startswith("/World/WaterSurface"):
                try:
                    prim.CreateAttribute(
                        "primvars:doNotCastShadows", Sdf.ValueTypeNames.Bool
                    ).Set(True)
                except Exception:
                    pass
        logger.info("[oceansim] Water surface: shadow casting disabled on all child prims.")

        logger.info(
            "[oceansim] Water surface loaded: ocean.usda  (Z=%.1f m, scale=0.5)",
            OCEANSIM_WATER_SURFACE_Z,
        )

    except Exception as e:
        logger.warning("[oceansim] Water surface setup failed (non-fatal): %s", e)


def _spawn_brov_models(stage, num_robots: int) -> list:
    """Spawn OceanSim BROV USD rigid-body models, one per robot.

    Returns a list of USD Prim objects (one per robot), used for direct
    xformOp position/orientation control during replay.
    """
    from isaacsim.oceansim.utils.assets_utils import get_oceansim_assets_path  # type: ignore
    from isaacsim.core.utils.stage import add_reference_to_stage  # type: ignore
    from isaacsim.core.prims import SingleRigidPrim, SingleGeometryPrim  # type: ignore
    from pxr import PhysxSchema  # type: ignore

    assets_root = get_oceansim_assets_path()
    brov_usd = os.path.join(assets_root, "OceanSim_assets", "Bluerov", OCEANSIM_BROV_USD)
    if not os.path.isfile(brov_usd):
        # Try without the sub-folder
        brov_usd = os.path.join(assets_root, "Bluerov", OCEANSIM_BROV_USD)

    if not os.path.isfile(brov_usd):
        raise FileNotFoundError(
            f"BROV USD not found at {brov_usd}. "
            "Make sure OceanSim assets are installed and asset_path.json is registered."
        )

    rob_prims = []
    for i in range(num_robots):
        prim_path = f"/World/brov_{i}"
        rob_prim = add_reference_to_stage(usd_path=brov_usd, prim_path=prim_path)

        # Disable gravity, add damping  (underwater neutral buoyancy)
        physx_api = PhysxSchema.PhysxRigidBodyAPI.Apply(
            stage.GetPrimAtPath(prim_path)
        )
        physx_api.CreateDisableGravityAttr(True)
        physx_api.GetLinearDampingAttr().Set(10.0)
        physx_api.GetAngularDampingAttr().Set(10.0)

        # Bounding-cube collision approximation (fast + good enough for replay)
        try:
            collider = SingleGeometryPrim(prim_path=prim_path, collision=True)
            collider.set_collision_approximation("boundingCube")
        except Exception:
            pass

        rob_prims.append(rob_prim)
        logger.info("[oceansim] Spawned BROV %d at %s", i, prim_path)

    return rob_prims


def _add_waypoint_markers(stage, all_waypoints: list) -> None:
    """Add small coloured cubes at each inspection waypoint."""
    try:
        from omni.isaac.core.objects import cuboid as cuboid_mod  # type: ignore
    except ImportError:
        try:
            from isaacsim.core.api.objects import cuboid as cuboid_mod  # type: ignore
        except ImportError:
            logger.warning("[oceansim] cuboid module not available – skipping waypoint markers.")
            return

    for i, waypoints in enumerate(all_waypoints):
        color = _COLORS[i % len(_COLORS)]
        for wi, wp in enumerate(waypoints):
            try:
                cuboid_mod.VisualCuboid(
                    f"/World/wp_r{i}_{wi}",
                    position=np.array(wp[:3], dtype=np.float64),
                    orientation=np.array(wp[3:7], dtype=np.float64),
                    color=color,
                    size=0.08,
                )
            except Exception:
                pass


def _set_robot_pose(rob_prim, xyz: np.ndarray, qwxyz: np.ndarray) -> None:
    """Directly set a BROV rigid-body prim's world pose via xformOp attributes.

    USD / Isaac Sim 5 may store ``xformOp:translate`` as either ``float3``
    (GfVec3f) or ``double3`` (GfVec3d), and ``xformOp:orient`` as ``quatf``
    or ``quatd``.  We try the single-precision variant first and fall back
    to double-precision so the code works across USD/Kit versions.
    """
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
        xf = UsdGeom.Xformable(rob_prim)
        try:
            xf.AddTranslateOp().Set(Gf.Vec3f(x, y, z))
        except Exception:
            pass

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
        xf = UsdGeom.Xformable(rob_prim)
        try:
            xf.AddOrientOp().Set(Gf.Quatf(qw, qx, qy, qz))
        except Exception:
            pass


# ─── Public entry-point ──────────────────────────────────────────────────────

def replay_in_oceansim(
    exec_result:           ExecutionResult,
    headless:              bool = False,
    uw_camera:             bool = True,
    uw_camera_robot_idx:   int  = 0,
    caustics:              bool = True,
) -> None:
    """Launch Isaac Sim with OceanSim and replay multi-robot trajectories.

    Parameters
    ----------
    exec_result:
        Result from :meth:`~route_executor.RouteExecutor.execute`.
    headless:
        If ``True``, run without a display window.
    uw_camera:
        Attach an OceanSim :class:`~isaacsim.oceansim.sensors.UW_Camera.UW_Camera`
        to one robot for physics-based underwater imagery.  Ignored in headless mode
        (viewport rendering is not available headless).
    uw_camera_robot_idx:
        Which robot (0-indexed) gets the UW camera.
    caustics:
        Attempt to enable RTX caustics.  Requires RTX Real-Time or Path-Traced
        renderer and may not activate on all GPU/driver combinations.
    """
    all_traj_positions  = exec_result.all_traj_positions
    all_traj_velocities = exec_result.all_traj_velocities   # not used – position control
    all_waypoints       = exec_result.all_waypoints
    num_robots          = len(all_traj_positions)
    total_steps         = max(len(t) for t in all_traj_positions) if all_traj_positions else 0

    if total_steps == 0:
        logger.error("[oceansim] No trajectory data in exec_result. Aborting.")
        return

    # Disable UW camera in headless mode (no display → no viewport provider)
    if headless and uw_camera:
        logger.info("[oceansim] Headless mode: disabling UW camera viewport.")
        uw_camera = False

    # ── Bootstrap Isaac Sim ───────────────────────────────────────────
    try:
        from isaacsim import SimulationApp
    except ImportError:
        try:
            from omni.isaac.kit import SimulationApp  # type: ignore
        except ImportError:
            logger.error(
                "[oceansim] Isaac Sim not found. "
                "Activate the isaaclab conda environment and re-run."
            )
            return

    simulation_app = SimulationApp(
        {"headless": headless, "width": "1920", "height": "1080"}
    )

    # ── Enable OceanSim extension ─────────────────────────────────────
    _try_enable_oceansim_extension()

    if not _ensure_oceansim_importable():
        logger.error(
            "[oceansim] OceanSim extension could not be imported. "
            "Falling back to vanilla Isaac Sim visualization."
        )
        simulation_app.close()
        from .visualization import replay_in_isaac_sim

        replay_in_isaac_sim(exec_result, headless=headless)
        return

    # ── Late imports ──────────────────────────────────────────────────
    try:
        from isaacsim.core.api import World  # type: ignore
    except ImportError:
        from omni.isaac.core import World  # type: ignore

    from pxr import UsdPhysics, Gf  # type: ignore

    # ── World ─────────────────────────────────────────────────────────
    my_world = World(stage_units_in_meters=1.0)
    stage    = my_world.stage

    # Zero gravity (underwater neutral buoyancy)
    try:
        ps = UsdPhysics.Scene.Get(stage, "/physicsScene")
        if not ps:
            ps = UsdPhysics.Scene.Define(stage, "/physicsScene")
        ps.GetGravityDirectionAttr().Set(Gf.Vec3f(0.0, 0.0, 0.0))
        ps.GetGravityMagnitudeAttr().Set(0.0)
    except Exception as e:
        logger.warning("[oceansim] Could not set zero gravity: %s", e)

    # ── RTX Real-Time Ray Tracing ─────────────────────────────────────
    # OceanSim uses RTX Real-Time ("RaytracedLighting"), not Path Tracing.
    # Path Tracing blocks all light below the water surface (renders black).
    # Real-Time Ray Tracing supports caustics and correctly illuminates the
    # underwater scene — this is what the OceanSim docs refer to when they
    # say "Render Settings - Ray Tracing - Caustics".
    try:
        import carb
        carb.settings.get_settings().set("/rtx/rendermode", "RaytracedLighting")
        logger.info("[oceansim] Renderer set to RTX Real-Time Ray Tracing.")
    except Exception as e:
        logger.warning("[oceansim] Could not set render mode: %s", e)

    # ── RTX caustics ─────────────────────────────────────────────────
    if caustics:
        _enable_rtx_caustics()

    # ── Lighting ─────────────────────────────────────────────────────
    _add_ambient_dome(stage)

    # ── Scene ─────────────────────────────────────────────────────────
    _load_ship_mesh(stage, simulation_app)
    _load_static_obstacles(stage)
    my_world.scene.add_default_ground_plane(z_position=float(MESH_POSE[2]) - 2.0)

    # ── Robots (OceanSim BROV USD, rigid-body) ────────────────────────
    try:
        rob_prims = _spawn_brov_models(stage, num_robots)
    except FileNotFoundError as e:
        logger.error("[oceansim] BROV model error: %s", e)
        simulation_app.close()
        return

    # Teleport each robot to its trajectory starting position
    traj_poses = _convert_trajectories(all_traj_positions)
    for i, prim in enumerate(rob_prims):
        _set_robot_pose(prim, traj_poses[i][0, :3], traj_poses[i][0, 3:])

    # ── Waypoint markers ──────────────────────────────────────────────
    _add_waypoint_markers(stage, all_waypoints)

    # ── Initialise physics ────────────────────────────────────────────
    simulation_app.update()
    simulation_app.update()
    my_world.initialize_physics()

    # ── UW Camera ────────────────────────────────────────────────────
    # Root cause of "kind=f/i, size=0" crash: when annotators attach to a
    # render product, the replicator orchestrator registers an on_update
    # callback that fires on the next simulation_app.update().  At that
    # point simTimesToWrite is an empty array because no simulation steps
    # have run yet (timeline is stopped / at frame 0).
    #
    # Fix: create the Camera prim now (needs to be on stage before physics),
    # but defer initialize() — which attaches the annotators — to the first
    # frame of an active replay, AFTER my_world.step(render=True) has been
    # called at least once.  This mirrors exactly how OceanSim's own
    # setup_scenario() works in the UI extension framework.
    _uw_cam_pending = None   # prim created, initialize() not yet called
    uw_cam_obj      = None   # fully initialized camera
    if uw_camera and uw_camera_robot_idx < num_robots:
        try:
            from isaacsim.oceansim.sensors.UW_Camera import UW_Camera  # type: ignore
            cam_prim_path = f"/World/brov_{uw_camera_robot_idx}/UW_camera"
            _uw_cam_pending = UW_Camera(
                prim_path=cam_prim_path,
                name=f"UW_Camera_{uw_camera_robot_idx}",
                resolution=(1920, 1080),
                translation=np.array([0.3, 0.0, 0.1]),
            )
            _uw_cam_pending.set_focal_length(2.1)
            _uw_cam_pending.set_clipping_range(0.1, 100.0)
            logger.info("[oceansim] UW_Camera prim created; initialize() deferred to first play frame.")
        except Exception as e:
            logger.warning("[oceansim] UW_Camera prim creation failed (non-fatal): %s", e)
            _uw_cam_pending = None

    logger.info(
        "[oceansim] Ready.  %d robot(s), %d total trajectory steps.  "
        "Press PLAY to start replay.",
        num_robots,
        total_steps,
    )

    # ── Replay loop ───────────────────────────────────────────────────
    replay_idx = 0
    playing    = False

    try:
        while simulation_app.is_running():
            my_world.step(render=True)

            if not my_world.is_playing():
                playing = False
                continue

            if not playing:
                # Reset to beginning each time PLAY is pressed
                replay_idx = 0
                playing    = True
                logger.info("[oceansim] Replay started.")

                # ── Deferred UW_Camera init (first play frame) ──────────
                # The simulation has now stepped at least once; the render
                # product has valid timestamps so annotator attach() works.
                if _uw_cam_pending is not None and uw_cam_obj is None:
                    try:
                        _uw_cam_pending.initialize(
                            UW_param=np.array(OCEANSIM_UW_PARAMS, dtype=np.float32),
                            viewport=True,
                        )
                        uw_cam_obj = _uw_cam_pending
                        _uw_cam_pending = None
                        logger.info("[oceansim] UW_Camera initialized on first play frame.")
                    except Exception as e:
                        logger.warning(
                            "[oceansim] UW_Camera initialize() failed (non-fatal): %s", e
                        )
                        _uw_cam_pending = None  # don't retry

            if replay_idx >= total_steps:
                logger.info("[oceansim] Replay finished – looping.")
                replay_idx = 0

            # Teleport all robots to their pose at this step
            for i, prim in enumerate(rob_prims):
                safe_idx = min(replay_idx, len(traj_poses[i]) - 1)
                _set_robot_pose(prim, traj_poses[i][safe_idx, :3], traj_poses[i][safe_idx, 3:])

            # Apply OceanSim underwater camera post-process
            if uw_cam_obj is not None:
                try:
                    uw_cam_obj.render()
                except Exception as e:
                    logger.debug("[oceansim] UW_Camera render error (skipping frame): %s", e)

            replay_idx += 1
            if replay_idx % 500 == 0:
                logger.info("[oceansim] Replay step %d / %d", replay_idx, total_steps)

    finally:
        # Clean up camera resources
        for cam in (uw_cam_obj, _uw_cam_pending):
            if cam is not None:
                try:
                    cam.close()
                except Exception:
                    pass
        simulation_app.close()
