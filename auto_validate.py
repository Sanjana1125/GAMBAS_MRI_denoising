"""
auto_validate_registration.py

Automated alternative/complement to manual landmark_validation.py --
scores EVERY session in prepared_L_to_M/ at once, no clicking required.
Use this to triage: run this first across the whole batch, then only
manually landmark-validate whichever sessions score worst here.

Two automated metrics, both computed directly from the already-saved
normalized .npy volumes (no re-registration, no GUI):

1. MASK DICE OVERLAP: brain foreground (> -0.9 in tanh-normalized space,
   i.e. not background) overlap between warped-L and M. A numeric version
   of "does the brain outline in the checkerboard line up."

2. EDGE CORRELATION: Sobel-gradient-magnitude correlation between the two
   volumes. Structural boundaries (skull, ventricles, sulci) show up as
   strong gradients in both modalities even though raw intensities differ
   a lot cross-contrast -- so this is a reasonable automated proxy for
   "do the tissue boundaries actually coincide," which is exactly what a
   checkerboard shows visually but a human has to eyeball tile-by-tile.

Neither metric replaces landmark validation's precision (mm error), but
both are free, instant, and cover 100% of your sessions instead of
whichever ones you have time to click through by hand.
"""

import os
import glob
import re
import numpy as np
from scipy.ndimage import sobel

PREPARED_DIR = "prepared_L_to_M"


def foreground_mask(vol, thresh=-0.9):
    return vol > thresh


def dice_overlap(mask_a, mask_b):
    inter = np.logical_and(mask_a, mask_b).sum()
    denom = mask_a.sum() + mask_b.sum()
    return 2 * inter / denom if denom > 0 else 0.0


def edge_magnitude(vol):
    gx = sobel(vol, axis=0)
    gy = sobel(vol, axis=1)
    gz = sobel(vol, axis=2)
    return np.sqrt(gx**2 + gy**2 + gz**2)


def edge_correlation(vol_a, vol_b):
    ea = edge_magnitude(vol_a).flatten()
    eb = edge_magnitude(vol_b).flatten()
    if ea.std() < 1e-8 or eb.std() < 1e-8:
        return 0.0
    return float(np.corrcoef(ea, eb)[0, 1])


def discover_pairs(prepared_dir):
    pairs = []
    for f in glob.glob(os.path.join(prepared_dir, "*_L_registered.npy")):
        fname = os.path.basename(f)
        stem = fname.replace("_L_registered.npy", "")
        target_path = os.path.join(prepared_dir, f"{stem}_M_target.npy")
        if os.path.exists(target_path):
            pairs.append((stem, f, target_path))
    return sorted(pairs)


def main():
    pairs = discover_pairs(PREPARED_DIR)
    if not pairs:
        print(f"No pairs found in {PREPARED_DIR}/ -- has batch_register_L_to_M.py "
              f"finished running?")
        return

    results = []
    for stem, input_path, target_path in pairs:
        vol_in = np.load(input_path)
        vol_tgt = np.load(target_path)

        if vol_in.shape != vol_tgt.shape:
            print(f"[FAIL] {stem}: shape mismatch {vol_in.shape} vs {vol_tgt.shape}")
            results.append((stem, None, None))
            continue

        mask_in = foreground_mask(vol_in)
        mask_tgt = foreground_mask(vol_tgt)
        dice = dice_overlap(mask_in, mask_tgt)
        edge_corr = edge_correlation(vol_in, vol_tgt)

        results.append((stem, dice, edge_corr))

    print(f"{'stem':<28} {'dice':>8} {'edge_corr':>10}")
    print("-" * 50)
    # Sort worst-first on dice, so the sessions most likely to need manual
    # landmark checking float to the top of the output.
    valid_results = [r for r in results if r[1] is not None]
    valid_results.sort(key=lambda r: r[1])

    for stem, dice, edge_corr in valid_results:
        flag = ""
        if dice < 0.85:
            flag = "  <-- LOW DICE, prioritize for manual landmark check"
        elif edge_corr < 0.3:
            flag = "  <-- LOW EDGE CORR, prioritize for manual landmark check"
        print(f"{stem:<28} {dice:>8.4f} {edge_corr:>10.4f}{flag}")

    dices = [d for _, d, _ in valid_results]
    edges = [e for _, _, e in valid_results]
    print("-" * 50)
    print(f"Mean dice: {np.mean(dices):.4f}   Mean edge corr: {np.mean(edges):.4f}")

    # Split by sequence (T1 vs T2) -- if failures cluster in one sequence,
    # that points to a systemic per-modality bug (e.g. skull-strip called
    # with the wrong 'modality' argument for that sequence) rather than
    # random per-session drift.
    print("\n--- Split by sequence ---")
    for seq in ("T1", "T2"):
        seq_results = [r for r in valid_results if f"_{seq}" in r[0]]
        if seq_results:
            seq_dices = [d for _, d, _ in seq_results]
            print(f"{seq}: n={len(seq_results)}, mean dice={np.mean(seq_dices):.4f}, "
                  f"min dice={np.min(seq_dices):.4f}")

    print(f"\nRule of thumb: dice > 0.85 and edge_corr > 0.3 is a reasonable initial bar "
          f"for this kind of cross-contrast registration -- adjust once you've cross-checked "
          f"a few of these numbers against actual landmark mm errors and know what 'good' "
          f"looks like on YOUR data specifically.")
    print(f"\nNext: manually landmark-validate the top few flagged sessions above, plus at "
          f"least one or two that scored well (to confirm high scores actually mean good "
          f"alignment here, not just that both metrics have a shared blind spot).")


if __name__ == "__main__":
    main()