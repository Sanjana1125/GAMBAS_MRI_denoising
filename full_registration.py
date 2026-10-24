"""
prepare_new_dataset_direct3T.py

Fixes the anchor-hop compounding-error bug in the original
process_subject_multi_session(): previously, only the 'anchor' session
(lab1) was registered directly to 3T; every other session (icu1, lab2,
lab3, van1) was registered only to the anchor, so 3T and a non-anchor
session ended up on the same grid via two INDEPENDENT registration hops
that never reconciled with each other. Errors in each hop compound rather
than cancel -- confirmed by the identity-check diagnostic showing each
non-anchor session drifting by its own, different, inconsistent slice
offset (+8, -3, +1, +3, -9), which is the signature of independent
per-session drift, not a single shared bug.

This version registers EVERY session -- anchor included -- directly to 3T
in a single hop, using the same register_3T_to_47mT() call that was
previously reserved for the anchor only. More registration calls per
subject (5 instead of 1+4), but no compounding error, since every
session's grid alignment to 3T is now independent of every other
session's.
"""

import ants
import antspynet
import numpy as np
import matplotlib.pyplot as plt
import os
import time
import sys

os.environ["ITK_GLOBAL_DEFAULT_NUMBER_OF_THREADS"] = "8"

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
REPEAT_DIR = os.path.join(BASE_DIR, "repeatability_share_data")


def safe_print(*args, **kwargs):
    try:
        print(*args, **kwargs)
        sys.stdout.flush()
    except (BrokenPipeError, OSError):
        pass


def load_volume(path):
    return ants.image_read(path)


def bias_correct(img):
    return ants.n4_bias_field_correction(img)


def skull_strip(img):
    prob_mask = antspynet.brain_extraction(img, modality="t1")
    mask = ants.threshold_image(prob_mask, 0.5, 1)
    return img * mask, mask


def register_3T_to_47mT(fixed_47mT, moving_3T):
    reg = ants.registration(fixed=fixed_47mT, moving=moving_3T, type_of_transform="Affine", aff_metric="mattes")
    return reg["warpedmovout"], reg["fwdtransforms"]


def resample_to_reference(moving, reference):
    return ants.resample_image_to_target(moving, reference)


def normalize_to_tanh_range(volume_np, p_low=0.5, p_high=99.5):
    lo, hi = np.percentile(volume_np, [p_low, p_high])
    clipped = np.clip(volume_np, lo, hi)
    normalized = 2 * (clipped - lo) / (hi - lo) - 1
    return normalized


def save_qc_checkerboard(vol_a, vol_b, slice_idx, out_path):
    a = vol_a[:, :, slice_idx]
    b = vol_b[:, :, slice_idx]
    tile = 16
    checker = np.zeros_like(a)
    for i in range(0, a.shape[0], tile):
        for j in range(0, a.shape[1], tile):
            use_a = ((i // tile) + (j // tile)) % 2 == 0
            src = a if use_a else b
            checker[i:i+tile, j:j+tile] = src[i:i+tile, j:j+tile]
    plt.imsave(out_path, checker, cmap='gray')


def check_paths_exist(subjects):
    missing = []
    for subj_id, paths in subjects.items():
        for seq, p in paths["3T"].items():
            if not os.path.exists(p):
                missing.append(p)
        for sess, seq_paths in paths["sessions"].items():
            for seq, p in seq_paths.items():
                if not os.path.exists(p):
                    missing.append(p)
    if missing:
        safe_print(f"WARNING: {len(missing)} path(s) do not exist:")
        for m in missing:
            safe_print(f"  {m}")
    else:
        safe_print("All paths exist.")
    return missing


def process_subject_direct3T(subject_id, path_3T, session_paths, out_dir, sequence_name="T1"):
    """
    Registers EVERY session directly to the subject's 3T scan -- no anchor,
    no session-to-session hop. Each session's output is independent of
    every other session's registration outcome.
    """
    os.makedirs(out_dir, exist_ok=True)

    # Load + bias-correct + skull-strip the 3T scan ONCE per subject/sequence
    # (it's the fixed image for every session, no need to redo this per-session)
    img_3T = bias_correct(load_volume(path_3T))
    img_3T_brain, _ = skull_strip(img_3T)

    for session_name, session_path in session_paths.items():
        session_npy_path = os.path.join(out_dir, f"{subject_id}_{session_name}_{sequence_name}.npy")
        target_npy_path = os.path.join(out_dir, f"{subject_id}_3T_{sequence_name}_registered_{session_name}.npy")

        if os.path.exists(session_npy_path) and os.path.exists(target_npy_path):
            safe_print(f"[{subject_id}] session '{session_name}' ({sequence_name}) "
                       f"already done -- skipping.")
            continue

        img_session = bias_correct(load_volume(session_path))
        img_session_brain, _ = skull_strip(img_session)

        # Direct hop: this session -> 3T. Fixed=47mT session (defines the
        # output grid), moving=3T. Same call previously reserved for the
        # anchor only -- now every session gets it.
        warped_3T, _ = register_3T_to_47mT(img_session_brain, img_3T_brain)
        warped_3T = resample_to_reference(warped_3T, img_session_brain)

        vol_session_norm = normalize_to_tanh_range(img_session_brain.numpy())
        vol_3T_norm = normalize_to_tanh_range(warped_3T.numpy())

        # Sanity check before saving: shapes MUST match since warped_3T was
        # just resampled onto img_session_brain's own grid. If this ever
        # fails, something upstream is broken -- fail loudly, don't save.
        assert vol_session_norm.shape == vol_3T_norm.shape, (
            f"[{subject_id}/{session_name}] shape mismatch after resample: "
            f"{vol_session_norm.shape} vs {vol_3T_norm.shape} -- do not save this pair."
        )

        np.save(session_npy_path, vol_session_norm)
        np.save(target_npy_path, vol_3T_norm)

        save_qc_checkerboard(
            vol_session_norm, vol_3T_norm, vol_session_norm.shape[2] // 2,
            os.path.join(out_dir, f"{subject_id}_{session_name}_{sequence_name}_vs_3T_direct_qc.png")
        )
        safe_print(f"[{subject_id}] session '{session_name}' ({sequence_name}) "
                   f"registered DIRECTLY to 3T (no anchor hop).")


if __name__ == "__main__":
    # Same subjects dict as prepare_new_dataset.py -- copy that block in,
    # or import it, rather than retyping paths. Left as a placeholder here
    # since the actual paths live in the existing script.
    subjects = {
        "sub-01": {
            "3T": {
                "T1": os.path.join(REPEAT_DIR, "3T_Data_defaced/9110/anat/sub-9110_acq-lowres_T1w.nii.gz"),
                "T2": os.path.join(REPEAT_DIR, "3T_Data_defaced/9110/anat/sub-9110_acq-lowres_T2w.nii.gz"),
                "FLAIR": os.path.join(REPEAT_DIR, "3T_Data_defaced/9110/anat/sub-9110_acq-lowres_FLAIR.nii.gz"),
            },
            "sessions": {
                "icu1": {"T1": os.path.join(REPEAT_DIR, "sub-9110/ses-icu01/anat/sub-9110_ses-icu01_T1w.nii.gz"),
                        "T2": os.path.join(REPEAT_DIR, "sub-9110/ses-icu01/anat/sub-9110_ses-icu01_T2w.nii.gz")},
                "lab1": {"T1": os.path.join(REPEAT_DIR, "sub-9110/ses-lab01/anat/sub-9110_ses-lab01_T1w.nii.gz"),
                        "T2": os.path.join(REPEAT_DIR, "sub-9110/ses-lab01/anat/sub-9110_ses-lab01_T2w.nii.gz")},
                "lab2": {"T1": os.path.join(REPEAT_DIR, "sub-9110/ses-lab02/anat/sub-9110_ses-lab02_T1w.nii.gz"),
                        "T2": os.path.join(REPEAT_DIR, "sub-9110/ses-lab02/anat/sub-9110_ses-lab02_T2w.nii.gz")},
                "lab3": {"T1": os.path.join(REPEAT_DIR, "sub-9110/ses-lab03/anat/sub-9110_ses-lab03_T1w.nii.gz"),
                        "T2": os.path.join(REPEAT_DIR, "sub-9110/ses-lab03/anat/sub-9110_ses-lab03_T2w.nii.gz")},
                "van1": {"T1": os.path.join(REPEAT_DIR, "sub-9110/ses-van01/anat/sub-9110_ses-van01_T1w.nii.gz"),
                        "T2": os.path.join(REPEAT_DIR, "sub-9110/ses-van01/anat/sub-9110_ses-van01_T2w.nii.gz")},
            },
        },
        "sub-02": {
            "3T": {
                "T1": os.path.join(REPEAT_DIR, "3T_Data_defaced/9133/anat/sub-9133_acq-lowres_T1w.nii.gz"),
                "T2": os.path.join(REPEAT_DIR, "3T_Data_defaced/9133/anat/sub-9133_acq-lowres_T2w.nii.gz"),
                "FLAIR": os.path.join(REPEAT_DIR, "3T_Data_defaced/9133/anat/sub-9133_acq-lowres_FLAIR.nii.gz"),
            },
            "sessions": {
                "icu1": {"T1": os.path.join(REPEAT_DIR, "sub-9133/ses-icu01/anat/sub-9133_ses-icu01_T1w.nii.gz"),
                        "T2": os.path.join(REPEAT_DIR, "sub-9133/ses-icu01/anat/sub-9133_ses-icu01_T2w.nii.gz")},
                "lab1": {"T1": os.path.join(REPEAT_DIR, "sub-9133/ses-lab01/anat/sub-9133_ses-lab01_T1w.nii.gz"),
                        "T2": os.path.join(REPEAT_DIR, "sub-9133/ses-lab01/anat/sub-9133_ses-lab01_T2w.nii.gz")},
                "lab2": {"T1": os.path.join(REPEAT_DIR, "sub-9133/ses-lab02/anat/sub-9133_ses-lab02_T1w.nii.gz"),
                        "T2": os.path.join(REPEAT_DIR, "sub-9133/ses-lab02/anat/sub-9133_ses-lab02_T2w.nii.gz")},
                "lab3": {"T1": os.path.join(REPEAT_DIR, "sub-9133/ses-lab03/anat/sub-9133_ses-lab03_T1w.nii.gz"),
                        "T2": os.path.join(REPEAT_DIR, "sub-9133/ses-lab03/anat/sub-9133_ses-lab03_T2w.nii.gz")},
                "van1": {"T1": os.path.join(REPEAT_DIR, "sub-9133/ses-van01/anat/sub-9133_ses-van01_T1w.nii.gz"),
                        "T2": os.path.join(REPEAT_DIR, "sub-9133/ses-van01/anat/sub-9133_ses-van01_T2w.nii.gz")},
            },
        },
    } # noqa: reuse existing path dict

    out_dir = os.path.join(BASE_DIR, "prepared_3T_47mT_pairs_direct3T")
    sequences_with_3T_ground_truth = ["T1", "T2"]

    missing = check_paths_exist(subjects)
    if missing:
        raise SystemExit(
            f"\nAborting: {len(missing)} path(s) listed above do not exist. "
            f"Fix these before running registration."
        )

    total_units = len(subjects) * len(sequences_with_3T_ground_truth)
    completed_units = 0
    overall_start = time.time()

    for subject_id, paths in subjects.items():
        for seq in sequences_with_3T_ground_truth:
            unit_start = time.time()
            session_paths_this_seq = {
                session_name: seq_paths[seq]
                for session_name, seq_paths in paths["sessions"].items()
            }
            process_subject_direct3T(
                subject_id,
                paths["3T"][seq],
                session_paths_this_seq,
                out_dir,
                sequence_name=seq,
            )
            completed_units += 1
            unit_elapsed = time.time() - unit_start
            avg_per_unit = (time.time() - overall_start) / completed_units
            remaining = total_units - completed_units
            eta_minutes = (avg_per_unit * remaining) / 60
            safe_print(f"--- [{subject_id} {seq}] done in {unit_elapsed/60:.1f} min. "
                  f"{completed_units}/{total_units} complete. "
                  f"Estimated time remaining: {eta_minutes:.1f} min ---\n")