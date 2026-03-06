#!/usr/bin/env python3
"""
Visualize ESDF voxel grid — 2D slices (matplotlib) or interactive 3D (Open3D).

Usage — 2D slice:
    python visualize_esdf.py                           # default Z=1.5
    python visualize_esdf.py --z_slice 2.0
    python visualize_esdf.py --z_slice 1.5 --axis 1   # Y-slice

Usage — 3D interactive:
    python visualize_esdf.py --mode 3d
    python visualize_esdf.py --mode 3d --esdf_band 2.0
    python visualize_esdf.py --mode 3d --show_occupied --show_inflated

Colour convention (both modes):
  Red     = inside obstacle  (positive ESDF)
  White   = surface          (ESDF ≈ 0)
  Blue    = free space       (negative ESDF)
"""

from __future__ import annotations

import argparse
import sys

import numpy as np


# ═══════════════════════════════════════════════════════════════════════════════
# Shared helpers
# ═══════════════════════════════════════════════════════════════════════════════

def _build_og_and_esdf():
    """Build occupancy grid and compute ESDF.  Returns (og, raw, esdf_3d)."""
    from occupancy_grid import build_occupancy_grid
    from scipy.ndimage import distance_transform_edt

    print("Building occupancy grid …")
    og = build_occupancy_grid()
    raw = og.raw_grid if og.raw_grid is not None else og.grid
    print(f"Grid shape: {raw.shape}  origin: {og.origin}  res: {og.resolution}m")

    print("Computing ESDF …")
    outside_dist = distance_transform_edt(~raw) * og.resolution
    inside_dist  = distance_transform_edt(raw)  * og.resolution
    esdf = (inside_dist - outside_dist).astype(np.float32)
    print(f"ESDF range: [{esdf.min():.3f}, {esdf.max():.3f}]")
    return og, raw, esdf


def _load_scaled_mesh():
    """Load the ship mesh with the same scale + pose applied in occupancy_grid."""
    import trimesh
    from scipy.spatial.transform import Rotation as R
    from config import MESH_PATH, MESH_POSE, MESH_TARGET_LENGTH

    raw_mesh = trimesh.load(MESH_PATH, force="mesh")
    if isinstance(raw_mesh, trimesh.Scene):
        raw_mesh = trimesh.util.concatenate(list(raw_mesh.geometry.values()))
    longest = float(raw_mesh.extents.max())
    if longest > 0:
        raw_mesh.apply_scale(MESH_TARGET_LENGTH / longest)
    T = np.eye(4)
    T[:3, 3] = MESH_POSE[:3]
    qw, qx, qy, qz = MESH_POSE[3:7]
    T[:3, :3] = R.from_quat([qx, qy, qz, qw]).as_matrix()
    raw_mesh.apply_transform(T)
    return raw_mesh


def _esdf_to_rgb(values: np.ndarray, vmin: float, vmax: float) -> np.ndarray:
    """Map ESDF float values → (N, 3) RGB array via RdBu_r colourmap."""
    import matplotlib.cm as cm
    norm = np.clip((values - vmin) / (vmax - vmin + 1e-9), 0.0, 1.0)
    return cm.get_cmap("RdBu_r")(norm)[:, :3].astype(np.float64)


# ═══════════════════════════════════════════════════════════════════════════════
# 3D mode — Open3D interactive viewer
# ═══════════════════════════════════════════════════════════════════════════════

def _visualize_3d(args):
    try:
        import open3d as o3d
    except ImportError:
        print("ERROR: open3d not installed.  Run:  pip install open3d")
        sys.exit(1)

    og, raw, esdf = _build_og_and_esdf()
    geometries = []

    # ── 1. Ship mesh (grey) ───────────────────────────────────────────
    if args.show_mesh:
        try:
            tmesh = _load_scaled_mesh()
            o3d_mesh = o3d.geometry.TriangleMesh(
                vertices=o3d.utility.Vector3dVector(tmesh.vertices),
                triangles=o3d.utility.Vector3iVector(tmesh.faces),
            )
            o3d_mesh.compute_vertex_normals()
            o3d_mesh.paint_uniform_color([0.6, 0.6, 0.6])
            geometries.append(o3d_mesh)
            print(f"Mesh: {len(tmesh.vertices):,} verts, {len(tmesh.faces):,} faces")
        except Exception as e:
            print(f"Could not load mesh: {e}")

    # ── 2. Near-surface ESDF voxels coloured by distance ─────────────
    band = args.esdf_band
    mask = np.abs(esdf) < band
    ijk = np.argwhere(mask)
    if len(ijk) > 0:
        if len(ijk) > args.max_points:
            rng = np.random.RandomState(42)
            ijk = ijk[rng.choice(len(ijk), args.max_points, replace=False)]
        centres = og.origin + (ijk.astype(np.float64) + 0.5) * og.resolution
        values  = esdf[ijk[:, 0], ijk[:, 1], ijk[:, 2]]
        colors  = _esdf_to_rgb(values, vmin=-band, vmax=band * 0.5)

        pcd = o3d.geometry.PointCloud()
        pcd.points = o3d.utility.Vector3dVector(centres)
        pcd.colors = o3d.utility.Vector3dVector(colors)
        geometries.append(pcd)
        print(f"ESDF band  |d| < {band}m : {len(ijk):,} points")
    else:
        print("WARNING: no voxels fall within ESDF band — check grid / ESDF values")

    # ── 3. Raw occupied voxels (optional, red) ─────────────────────────
    if args.show_occupied:
        occ_ijk = np.argwhere(raw)
        if len(occ_ijk) > args.max_points:
            rng = np.random.RandomState(42)
            occ_ijk = occ_ijk[rng.choice(len(occ_ijk), args.max_points, replace=False)]
        pts = og.origin + (occ_ijk.astype(np.float64) + 0.5) * og.resolution
        pcd2 = o3d.geometry.PointCloud()
        pcd2.points = o3d.utility.Vector3dVector(pts)
        pcd2.paint_uniform_color([1.0, 0.3, 0.3])
        geometries.append(pcd2)
        print(f"Occupied voxels      : {len(occ_ijk):,} points")

    # ── 4. Inflation shell  (optional, blue) ─────────────────────────
    if args.show_inflated:
        shell = og.grid & ~raw
        shell_ijk = np.argwhere(shell)
        if len(shell_ijk) > args.max_points:
            rng = np.random.RandomState(99)
            shell_ijk = shell_ijk[rng.choice(len(shell_ijk), args.max_points, replace=False)]
        pts = og.origin + (shell_ijk.astype(np.float64) + 0.5) * og.resolution
        pcd3 = o3d.geometry.PointCloud()
        pcd3.points = o3d.utility.Vector3dVector(pts)
        pcd3.paint_uniform_color([0.3, 0.3, 1.0])
        geometries.append(pcd3)
        print(f"Inflation shell      : {len(shell_ijk):,} points")

    # ── 5. Coordinate frame ───────────────────────────────────────────
    geometries.append(o3d.geometry.TriangleMesh.create_coordinate_frame(size=2.0))

    print()
    print("Opening Open3D viewer …")
    print("  Grey mesh           = ship hull")
    print("  Red → White → Blue  = ESDF  (red = inside obstacle, blue = free space)")
    if args.show_occupied:
        print("  Red points          = raw occupied voxels")
    if args.show_inflated:
        print("  Blue points         = inflation shell")
    print("  Controls: left-drag = rotate | scroll = zoom | middle-drag = pan")

    o3d.visualization.draw_geometries(
        geometries,
        window_name="ESDF 3D — Open3D",
        width=1600,
        height=900,
    )


# ═══════════════════════════════════════════════════════════════════════════════
# 2D mode — original matplotlib slice viewer
# ═══════════════════════════════════════════════════════════════════════════════

def _visualize_2d(args):
    import matplotlib.pyplot as plt

    og, raw, esdf = _build_og_and_esdf()

    ax        = args.axis
    axis_names = ["X", "Y", "Z"]
    slice_idx  = int(round((args.z_slice - og.origin[ax]) / og.resolution))
    slice_idx  = int(np.clip(slice_idx, 0, raw.shape[ax] - 1))
    actual_pos = og.origin[ax] + (slice_idx + 0.5) * og.resolution
    print(f"Slicing at {axis_names[ax]}={actual_pos:.2f}m  (index {slice_idx})")

    if ax == 0:
        esdf_slice = esdf[slice_idx, :, :].T
        occ_slice  = raw[slice_idx, :, :].T
        xlabel, ylabel = "Y (m)", "Z (m)"
        x_origin, y_origin = og.origin[1], og.origin[2]
        nx, ny = raw.shape[1], raw.shape[2]
    elif ax == 1:
        esdf_slice = esdf[:, slice_idx, :].T
        occ_slice  = raw[:, slice_idx, :].T
        xlabel, ylabel = "X (m)", "Z (m)"
        x_origin, y_origin = og.origin[0], og.origin[2]
        nx, ny = raw.shape[0], raw.shape[2]
    else:
        esdf_slice = esdf[:, :, slice_idx].T
        occ_slice  = raw[:, :, slice_idx].T
        xlabel, ylabel = "X (m)", "Y (m)"
        x_origin, y_origin = og.origin[0], og.origin[1]
        nx, ny = raw.shape[0], raw.shape[1]

    extent = [x_origin, x_origin + nx * og.resolution,
              y_origin, y_origin + ny * og.resolution]

    fig, axes = plt.subplots(1, 2, figsize=(18, 7))

    ax1 = axes[0]
    im = ax1.imshow(esdf_slice, origin="lower", extent=extent,
                    cmap="RdBu_r", vmin=args.vmin, vmax=args.vmax, aspect="equal")
    ax1.contour(esdf_slice, levels=[0.0], colors="white", linewidths=1.5,
                origin="lower", extent=extent)
    plt.colorbar(im, ax=ax1, label="ESDF (m) — +ve inside obstacle")
    ax1.set_xlabel(xlabel); ax1.set_ylabel(ylabel)
    ax1.set_title(f"ESDF slice at {axis_names[ax]}={actual_pos:.2f}m")

    ax2 = axes[1]
    ax2.imshow(occ_slice.astype(float), origin="lower", extent=extent,
               cmap="Greys", vmin=0, vmax=1, aspect="equal")
    ax2.set_xlabel(xlabel); ax2.set_ylabel(ylabel)
    ax2.set_title(f"Raw occupancy (pre-inflation) at {axis_names[ax]}={actual_pos:.2f}m")

    if args.show_mesh:
        try:
            tmesh = _load_scaled_mesh()
            plane_origin = [0.0, 0.0, 0.0]
            plane_normal = [0.0, 0.0, 0.0]
            plane_origin[ax] = actual_pos
            plane_normal[ax] = 1.0
            cross = tmesh.section(plane_origin=plane_origin,
                                  plane_normal=plane_normal)
            if cross is not None:
                for entity in cross.entities:
                    pts = cross.vertices[entity.points]
                    if ax == 0:
                        ax1.plot(pts[:, 1], pts[:, 2], "lime", lw=0.8)
                        ax2.plot(pts[:, 1], pts[:, 2], "lime", lw=0.8)
                    elif ax == 1:
                        ax1.plot(pts[:, 0], pts[:, 2], "lime", lw=0.8)
                        ax2.plot(pts[:, 0], pts[:, 2], "lime", lw=0.8)
                    else:
                        ax1.plot(pts[:, 0], pts[:, 1], "lime", lw=0.8)
                        ax2.plot(pts[:, 0], pts[:, 1], "lime", lw=0.8)
                print("Mesh cross-section overlaid (green lines)")
            else:
                print("Mesh cross-section is empty at this slice height")
        except Exception as e:
            print(f"Mesh overlay failed: {e}")

    plt.tight_layout()
    out_path = f"esdf_slice_{axis_names[ax]}{actual_pos:.1f}.png"
    plt.savefig(out_path, dpi=150)
    print(f"Saved to {out_path}")
    plt.show()


# ═══════════════════════════════════════════════════════════════════════════════

def main():
    parser = argparse.ArgumentParser(
        description="ESDF visualization — 2D matplotlib slice or 3D Open3D")
    parser.add_argument("--mode", choices=["2d", "3d"], default="2d",
                        help="2d = matplotlib heatmap slice  |  3d = Open3D interactive")

    # ── 2D options ────────────────────────────────────────────────────
    parser.add_argument("--z_slice", type=float, default=1.5,
                        help="Slice position in world-frame metres (2D)")
    parser.add_argument("--axis", type=int, default=2, choices=[0, 1, 2],
                        help="Slice axis  0=X  1=Y  2=Z  (2D)")
    parser.add_argument("--vmin", type=float, default=-3.0,
                        help="Colour-bar lower bound in metres (2D)")
    parser.add_argument("--vmax", type=float, default=1.0,
                        help="Colour-bar upper bound in metres (2D)")

    # ── 3D options ────────────────────────────────────────────────────
    parser.add_argument("--esdf_band", type=float, default=2.0,
                        help="Render voxels where |ESDF| < band  metres (3D)")
    parser.add_argument("--show_occupied", action="store_true", default=False,
                        help="Overlay raw occupied voxels as red cloud (3D)")
    parser.add_argument("--show_inflated", action="store_true", default=False,
                        help="Overlay inflation shell as blue cloud (3D)")
    parser.add_argument("--max_points", type=int, default=300_000,
                        help="Max points per layer — subsampled if exceeded (3D)")

    # ── Shared ────────────────────────────────────────────────────────
    parser.add_argument("--show_mesh", action="store_true", default=True,
                        help="Render ship mesh (both modes)")
    parser.add_argument("--no_mesh", dest="show_mesh", action="store_false")

    args = parser.parse_args()

    if args.mode == "3d":
        _visualize_3d(args)
    else:
        _visualize_2d(args)


if __name__ == "__main__":
    main()
