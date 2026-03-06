"""
clip_ship_mesh.py
─────────────────
Load the Duke of Lancaster GLB mesh, apply clipping planes in scaled
coordinate space (where the longest axis = 40 m), and save as a new GLB.

Clip rules (in 40 m-scaled space):
  • Remove everything with z < -2     (hull bottom)
  • Remove everything with |y| > 3    (port/starboard extremes)

Output: duke_of_lancaster_uk_clipped.glb  (same directory as the input)

Usage:
    cd /home/troja_robot_lab/Desktop/Krystof/VRP
    python clip_ship_mesh.py
"""

import os
import numpy as np
import trimesh

# ── Paths ──────────────────────────────────────────────────────────────────────
VRP_ROOT    = os.path.dirname(os.path.abspath(__file__))
ASSETS_PATH = os.path.join(VRP_ROOT, "..", "brov_auv_curobo", "assets")
INPUT_GLB   = os.path.join(ASSETS_PATH, "environment", "duke_of_lancaster_uk.glb")
OUTPUT_GLB  = os.path.join(ASSETS_PATH, "environment", "duke_of_lancaster_uk_clipped.glb")

MESH_TARGET_LENGTH = 40.0   # metres — longest axis in scaled space

# Clip thresholds in scaled (40 m) space
# NOTE: Isaac Sim applies MESH_POSE = 180° rotation about X, meaning
#       Isaac_Y = -raw_Y  and  Isaac_Z = -raw_Z.
# "Remove Isaac Z < -2"  →  -raw_Z < -2  →  raw_Z > 2  →  keep raw_Z <= 2/scale
# "Remove |Isaac Y| > 3"  →  |raw_Y| > 3  →  keep |raw_Y| <= 3/scale
Z_MIN_SCALED  = -2.0   # remove Isaac Z < -2 m (ship bottom)
Y_HALF_SCALED =  3.0   # remove |Isaac Y| > 3 m (port/starboard)


def load_and_merge(path: str) -> trimesh.Trimesh:
    loaded = trimesh.load(path)
    if isinstance(loaded, trimesh.Scene):
        print(f"Scene with {len(loaded.geometry)} geometries — concatenating raw geometry (no scene transforms).")
        return trimesh.util.concatenate(list(loaded.geometry.values()))
    return loaded


def clip_mesh(mesh: trimesh.Trimesh, z_max: float, y_half: float) -> trimesh.Trimesh:
    # Remove raw_Z > z_max  →  removes Isaac_Z < -Z_MIN_SCALED (ship bottom)
    mesh = mesh.slice_plane(
        plane_origin=[0, 0, z_max],
        plane_normal=[0, 0, -1],   # keep raw_Z <= z_max
        cap=True,
    )
    # Remove raw_Y > +y_half  →  removes Isaac_Y < -Y_HALF_SCALED
    mesh = mesh.slice_plane(
        plane_origin=[0,  y_half, 0],
        plane_normal=[0, -1, 0],   # keep raw_Y <= +y_half
        cap=True,
    )
    # Remove raw_Y < -y_half  →  removes Isaac_Y > +Y_HALF_SCALED
    mesh = mesh.slice_plane(
        plane_origin=[0, -y_half, 0],
        plane_normal=[0,  1, 0],   # keep raw_Y >= -y_half
        cap=True,
    )
    return mesh


def main():
    print(f"Loading: {INPUT_GLB}")
    mesh = load_and_merge(INPUT_GLB)

    longest = float(mesh.extents.max())
    scale   = MESH_TARGET_LENGTH / longest if longest > 0 else 1.0
    print(f"Raw mesh longest extent: {longest:.4f} units")
    print(f"Scale factor to reach {MESH_TARGET_LENGTH} m: {scale:.6f}")

    # Convert scaled-space thresholds to raw mesh space
    z_min_raw  = Z_MIN_SCALED  / scale
    y_half_raw = Y_HALF_SCALED / scale
    print(f"Clip planes in raw mesh space:  z < {z_min_raw:.4f},  |y| > {y_half_raw:.4f}")

    bounds = mesh.bounds
    print(f"Bounds before clip:  min={np.round(bounds[0], 3)}, max={np.round(bounds[1], 3)}")

    clipped = clip_mesh(mesh, z_min_raw, y_half_raw)

    bounds2 = clipped.bounds
    print(f"Bounds after clip:   min={np.round(bounds2[0], 3)}, max={np.round(bounds2[1], 3)}")
    print(f"Vertices: {len(clipped.vertices)}  Faces: {len(clipped.faces)}")

    clipped.export(OUTPUT_GLB)
    print(f"Saved: {OUTPUT_GLB}")


if __name__ == "__main__":
    main()
