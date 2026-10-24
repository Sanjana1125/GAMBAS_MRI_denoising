"""
landmark_validation.py

Step 3 of the staged plan: validate registration using landmarks, not just
visual overlays. Checkerboards catch gross misalignment well but are easy
to eyeball-pass on subtle drift (exactly what happened with the anchor-hop
bug earlier in this project -- individual checkerboards "looked fine").
A landmark distance in mm is a number you can threshold and track, not a
vibe.

This is a MANUAL landmark picker -- click 3-5 clearly identifiable
anatomical points (e.g. anterior commissure, posterior commissure, tip of
a clearly visible ventricle horn, a distinctive sulcus junction) on the
SAME physical locations in both the warped-L volume and the M volume, at
matched slice indices. It then reports point-to-point Euclidean distance
in mm (using real voxel spacing, not just voxel counts) as your
registration accuracy metric.

Why manual, not automatic: automatic landmark detection needs a trained
model or a well-defined intensity-based landmark scheme, which is its own
project. For validating N registration outputs by eye anyway, manual
picking of a handful of unambiguous points is faster to trust and easier
to sanity-check than debugging a landmark detector you don't have time to
build. If this needs to scale to many more subjects later, revisit.

Usage:
    python landmark_validation.py <warped_L_npy> <M_npy> <slice_idx> <voxel_spacing_mm x,y,z>

e.g.
    python landmark_validation.py \
        prepared_L_to_M/sub-01_lab1_T1_L_registered.npy \
        prepared_L_to_M/sub-01_lab1_T1_M_target.npy \
        90 1.0,1.0,1.0
"""

import sys
import numpy as np
import matplotlib.pyplot as plt


def pick_landmarks(vol_a, vol_b, slice_idx, n_points, title_a, title_b):
    """Click n_points on each panel, in the SAME anatomical order on both --
    e.g. always click AC first, PC second, etc. Order is what pairs them up,
    there's no automatic matching."""
    fig, axes = plt.subplots(1, 2, figsize=(12, 6))
    axes[0].imshow(vol_a[:, :, slice_idx], cmap='gray')
    axes[0].set_title(f"{title_a}\nClick {n_points} points, in a consistent order")
    axes[1].imshow(vol_b[:, :, slice_idx], cmap='gray')
    axes[1].set_title(f"{title_b}\nClick the SAME {n_points} points, SAME order")
    plt.tight_layout()

    print(f"Click {n_points} points on the LEFT panel first, then {n_points} on the RIGHT panel, "
          f"in the same anatomical order both times (e.g. AC, PC, ventricle tip...).")
    pts = plt.ginput(n_points * 2, timeout=0)
    plt.close(fig)

    pts_a = pts[:n_points]
    pts_b = pts[n_points:]
    return pts_a, pts_b


def compute_landmark_error(pts_a, pts_b, voxel_spacing):
    """pts are (x, y) in voxel/pixel coords from matplotlib's click coords
    (which are already in array-index space for imshow). voxel_spacing is
    (sx, sy, sz) mm -- we only have in-plane (x,y) here since points were
    picked on one slice each; z contributes 0 unless you extend this to
    pick points across different slices deliberately."""
    sx, sy, _ = voxel_spacing
    errors = []
    for (xa, ya), (xb, yb) in zip(pts_a, pts_b):
        dx_mm = (xa - xb) * sx
        dy_mm = (ya - yb) * sy
        dist_mm = np.sqrt(dx_mm**2 + dy_mm**2)
        errors.append(dist_mm)
    return errors


def main():
    warped_L_path, M_path, slice_idx, spacing_str = sys.argv[1:5]
    slice_idx = int(slice_idx)
    voxel_spacing = tuple(float(s) for s in spacing_str.split(","))

    vol_a = np.load(warped_L_path)
    vol_b = np.load(M_path)

    if vol_a.shape != vol_b.shape:
        print(f"[WARN] shapes differ: {vol_a.shape} vs {vol_b.shape} -- "
              f"landmark comparison across mismatched grids may not be meaningful.")

    n_points = 4
    pts_a, pts_b = pick_landmarks(vol_a, vol_b, slice_idx, n_points,
                                   "Warped 47mT (L in M space)", "Low-res 3T (M, target)")

    errors = compute_landmark_error(pts_a, pts_b, voxel_spacing)

    print("\n=== Landmark validation results ===")
    for i, e in enumerate(errors):
        flag = "" if e < 3.0 else "  <-- LARGE, investigate this point"
        print(f"  Point {i+1}: {e:.2f} mm{flag}")
    print(f"  Mean error: {np.mean(errors):.2f} mm")
    print(f"  Max error:  {np.max(errors):.2f} mm")

    print("\nRule of thumb: sub-2mm mean error is solid for this kind of "
          "cross-contrast registration. Consistently >3-4mm on multiple "
          "points means the affine registration itself needs attention, "
          "not just cosmetic tuning.")


if __name__ == "__main__":
    main()