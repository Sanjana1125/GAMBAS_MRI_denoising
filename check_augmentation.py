"""
check_augmentation.py

Throwaway visual sanity-check -- NOT part of the training pipeline.

Loads one real (input, target) slice pair from your prepared train_data,
applies augment_pair() several times, and saves a grid image so you can
eyeball whether the flips/rotations/noise look anatomically sane before
trusting augment_pair() inside your actual Dataset class.

Delete this file once you've confirmed things look right.
"""

import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from scipy.ndimage import rotate as ndi_rotate, shift as ndi_shift

# ============================================================
# Same augmentation function discussed earlier
# ============================================================

def augment_pair(input_slice, target_slice, rotation_range=10, noise_std=0.03):
    # Horizontal flip -- identical to both
    if np.random.rand() < 0.5:
        input_slice = np.flip(input_slice, axis=1).copy()
        target_slice = np.flip(target_slice, axis=1).copy()

    # Small rotation -- identical angle, both images
    angle = np.random.uniform(-rotation_range, rotation_range)
    if abs(angle) > 0.01:
        input_slice = ndi_rotate(input_slice, angle, reshape=False, order=1,
                                  mode='constant', cval=-1.0)
        target_slice = ndi_rotate(target_slice, angle, reshape=False, order=1,
                                   mode='constant', cval=-1.0)

    # Small translation -- identical shift, both images
    max_shift = 4
    dy, dx = np.random.randint(-max_shift, max_shift + 1, size=2)
    if dy != 0 or dx != 0:
        input_slice = ndi_shift(input_slice, (dy, dx), order=1, mode='constant', cval=-1.0)
        target_slice = ndi_shift(target_slice, (dy, dx), order=1, mode='constant', cval=-1.0)

    # Intensity jitter -- INPUT ONLY, simulates 47mT acquisition variability
    contrast = np.random.uniform(0.9, 1.1)
    brightness = np.random.uniform(-0.05, 0.05)
    input_slice = np.clip(input_slice * contrast + brightness, -1.0, 1.0)

    # Gaussian noise -- INPUT ONLY, more plausible LF noise realizations
    if noise_std > 0:
        noise = np.random.normal(0, noise_std, input_slice.shape).astype(np.float32)
        input_slice = np.clip(input_slice + noise, -1.0, 1.0)

    return input_slice, target_slice


# ============================================================
# Config -- point this at ONE real slice pair from your prepared data
# ============================================================

INPUT_PATH = Path("train_data/train/inputs/9110_icu01_T1w_sl060.npy")   # <-- adjust filename
TARGET_PATH = Path("train_data/train/targets/9110_icu01_T1w_sl060.npy")  # <-- adjust filename
N_SAMPLES = 8
OUT_PATH = Path("augmentation_check.png")


def main():
    input_slice = np.load(INPUT_PATH).astype(np.float32)
    target_slice = np.load(TARGET_PATH).astype(np.float32)

    print(f"Loaded input {INPUT_PATH.name}: shape={input_slice.shape}, "
          f"range=[{input_slice.min():.3f}, {input_slice.max():.3f}]")
    print(f"Loaded target {TARGET_PATH.name}: shape={target_slice.shape}, "
          f"range=[{target_slice.min():.3f}, {target_slice.max():.3f}]")

    fig, axes = plt.subplots(2, N_SAMPLES + 1, figsize=(3 * (N_SAMPLES + 1), 6))

    # First column: original, unaugmented pair
    axes[0, 0].imshow(input_slice, cmap="gray", vmin=-1, vmax=1)
    axes[0, 0].set_title("original input")
    axes[0, 0].axis("off")
    axes[1, 0].imshow(target_slice, cmap="gray", vmin=-1, vmax=1)
    axes[1, 0].set_title("original target")
    axes[1, 0].axis("off")

    # Remaining columns: augmented samples
    for i in range(1, N_SAMPLES + 1):
        aug_input, aug_target = augment_pair(input_slice.copy(), target_slice.copy())
        axes[0, i].imshow(aug_input, cmap="gray", vmin=-1, vmax=1)
        axes[0, i].set_title(f"aug input #{i}")
        axes[0, i].axis("off")
        axes[1, i].imshow(aug_target, cmap="gray", vmin=-1, vmax=1)
        axes[1, i].set_title(f"aug target #{i}")
        axes[1, i].axis("off")

    plt.tight_layout()
    plt.savefig(OUT_PATH, dpi=150)
    print(f"\nSaved: {OUT_PATH}")
    print("Check that:")
    print("  1. Rotated/flipped/shifted input and target still line up anatomically")
    print("  2. Background fill (corners from rotation/shift) is dark, not bright")
    print("  3. Contrast jitter and noise on inputs still look like plausible scans")
    print("  4. Targets are NEVER noisy/contrast-jittered -- only geometric changes")


if __name__ == "__main__":
    main()