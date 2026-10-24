"""
batch_register_L_to_M.py

Step 1-2-4 of the staged plan: register all 5 47mT conditions, both
subjects, to the low-resolution 3T scan (M) -- M stays the target, H is
not touched at all in this phase.

Uses robust, BRAIN-MASKED normalization (not whole-image percentiles):
percentiles are computed only over voxels inside the skull-stripped mask,
so background/skull remnants and any stray bright artifacts outside the
brain can't skew the intensity mapping -- this matters more here than it
did before, because at this stage the training target IS this normalized
volume, not an intermediate QC step.
"""

import os
import sys
import numpy as np
import ants
import antspynet

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
REPEAT_DIR = os.path.join(BASE_DIR, "repeatability_share_data")
OUT_DIR = os.path.join(BASE_DIR, "prepared_L_to_M")

SESSIONS = ["icu1", "lab1", "lab2", "lab3", "van1"]
SEQUENCES = ["T1", "T2"]

# subject_id -> (47mT session dir naming, low-res 3T path per sequence)
# EDIT the low-res 3T paths to your actual filenames.
SUBJECTS = {
    "sub-01": {
        "num": "9110",
        "lowres_3T": {
            "T1": os.path.join(REPEAT_DIR, "3T_Data_defaced/9110/anat/sub-9110_acq-lowres_T1w.nii.gz"),
            "T2": os.path.join(REPEAT_DIR, "3T_Data_defaced/9110/anat/sub-9110_acq-lowres_T2w.nii.gz"),
        },
    },
    "sub-02": {
        "num": "9133",
        "lowres_3T": {
            "T1": os.path.join(REPEAT_DIR, "3T_Data_defaced/9133/anat/sub-9133_acq-lowres_T1w.nii.gz"),
            "T2": os.path.join(REPEAT_DIR, "3T_Data_defaced/9133/anat/sub-9133_acq-lowres_T2w.nii.gz"),
        },
    },
}

SESSION_DIR_MAP = {
    "icu1": "ses-icu01", "lab1": "ses-lab01", "lab2": "ses-lab02",
    "lab3": "ses-lab03", "van1": "ses-van01",
}


def safe_print(*a, **k):
    print(*a, **k)
    sys.stdout.flush()


def bias_correct(img):
    return ants.n4_bias_field_correction(img)


def skull_strip(img, modality="t1"):
    prob_mask = antspynet.brain_extraction(img, modality=modality)
    mask = ants.threshold_image(prob_mask, 0.5, 1)
    return img * mask, mask


def robust_masked_normalize(volume_np, mask_np, p_low=0.5, p_high=99.5):
    """
    Percentiles computed ONLY over in-mask voxels -- robust to background,
    skull remnants, and any bright artifacts outside the brain that would
    otherwise skew a whole-image percentile calculation.
    """
    brain_voxels = volume_np[mask_np > 0]
    if brain_voxels.size == 0:
        raise ValueError("Empty brain mask -- skull-stripping likely failed for this volume.")
    lo, hi = np.percentile(brain_voxels, [p_low, p_high])
    if hi <= lo:
        raise ValueError(f"Degenerate intensity range in mask (lo={lo}, hi={hi}).")
    clipped = np.clip(volume_np, lo, hi)
    normalized = 2 * (clipped - lo) / (hi - lo) - 1
    # Force background (outside mask) to exactly -1, not whatever the clip
    # happened to produce -- keeps background truly uniform for the network.
    normalized[mask_np == 0] = -1.0
    return normalized.astype(np.float32)


def register_L_to_M(L_img, M_img):
    reg = ants.registration(fixed=M_img, moving=L_img, type_of_transform="Affine", aff_metric="mattes")
    return reg["fwdtransforms"], reg["warpedmovout"]


def save_qc_checkerboard(vol_a, vol_b, slice_idx, out_path, tile=16):
    import matplotlib.pyplot as plt
    a = vol_a[:, :, slice_idx]
    b = vol_b[:, :, slice_idx]
    checker = np.zeros_like(a)
    for i in range(0, a.shape[0], tile):
        for j in range(0, a.shape[1], tile):
            use_a = ((i // tile) + (j // tile)) % 2 == 0
            src = a if use_a else b
            checker[i:i+tile, j:j+tile] = src[i:i+tile, j:j+tile]
    plt.imsave(out_path, checker, cmap='gray')


def main():
    os.makedirs(OUT_DIR, exist_ok=True)
    total = len(SUBJECTS) * len(SESSIONS) * len(SEQUENCES)
    done = 0

    for subject_id, info in SUBJECTS.items():
        subj_num = info["num"]
        for seq in SEQUENCES:
            M_path = info["lowres_3T"][seq]
            if not os.path.exists(M_path):
                safe_print(f"[SKIP subject] {subject_id} {seq}: low-res 3T not found at {M_path}")
                continue
            M_img_raw = bias_correct(ants.image_read(M_path))
            M_img, M_mask = skull_strip(M_img_raw, modality=seq.lower())

            for session in SESSIONS:
                done += 1
                ses_dir = SESSION_DIR_MAP[session]
                L_path = os.path.join(
                    REPEAT_DIR, f"sub-{subj_num}/{ses_dir}/anat/sub-{subj_num}_{ses_dir}_{seq}w.nii.gz"
                )
                if not os.path.exists(L_path):
                    safe_print(f"[SKIP {done}/{total}] {subject_id} {session} {seq}: 47mT file not found")
                    continue

                prefix = f"{subject_id}_{session}_{seq}"
                out_input = os.path.join(OUT_DIR, f"{prefix}_L_registered.npy")
                out_target = os.path.join(OUT_DIR, f"{prefix}_M_target.npy")
                if os.path.exists(out_input) and os.path.exists(out_target):
                    safe_print(f"[{done}/{total}] {prefix}: already done -- skipping")
                    continue

                safe_print(f"[{done}/{total}] {prefix}: registering L -> M...")
                L_img_raw = bias_correct(ants.image_read(L_path))
                L_img, L_mask = skull_strip(L_img_raw, modality=seq.lower())

                fwd_transforms, warped_L = register_L_to_M(L_img, M_img)

                warped_L_np = warped_L.numpy()
                M_np = M_img.numpy()
                M_mask_np = M_mask.numpy()

                # Warp the L brain mask into M's space too, using the same
                # transform, so normalization uses a mask that's actually
                # correct for the warped image (not the pre-warp L mask).
                warped_L_mask = ants.apply_transforms(
                    fixed=M_img, moving=L_mask, transformlist=fwd_transforms, interpolator="nearestNeighbor"
                ).numpy()

                input_norm = robust_masked_normalize(warped_L_np, warped_L_mask)
                target_norm = robust_masked_normalize(M_np, M_mask_np)

                if input_norm.shape != target_norm.shape:
                    safe_print(f"[FAIL] {prefix}: shape mismatch after resample "
                               f"{input_norm.shape} vs {target_norm.shape} -- not saving.")
                    continue

                np.save(out_input, input_norm)
                np.save(out_target, target_norm)

                save_qc_checkerboard(input_norm, target_norm, input_norm.shape[2] // 2,
                                      os.path.join(OUT_DIR, f"{prefix}_qc_checkerboard.png"))
                safe_print(f"[{done}/{total}] {prefix}: done, QC checkerboard saved -- INSPECT IT.")

    safe_print("\nAll done. Next: landmark validation (landmark_validation.py) before trusting this "
               "data enough to train on.")


if __name__ == "__main__":
    main()