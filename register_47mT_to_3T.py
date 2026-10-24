from pathlib import Path
import ants
import shutil


# ============================================================
# Configuration
# ============================================================

DATASET_ROOT = Path(
    "repeatability_share_data"
)

HF_ROOT = DATASET_ROOT / "3T_Data_defaced"

OUTPUT_ROOT = DATASET_ROOT / "registered_LF_to_3T"

# LF modality -> corresponding 3T reference
REFERENCE_MAP = {
    "T1w": "acq-highres_T1w.nii.gz",
    "T2w": "acq-highres_T2w.nii.gz",
    "FLAIR": "acq-highres_FLAIR.nii.gz",
}


# ============================================================
# Registration function
# ============================================================

def register_image(path_moving, path_fixed, path_output):
    """
    Register moving LF image to fixed 3T image using rigid registration.

    The registered image is saved directly as a NIfTI file,
    preserving the fixed-image spatial grid and metadata.
    """

    print(f"    Fixed  : {path_fixed}")
    print(f"    Moving : {path_moving}")
    print(f"    Output : {path_output}")

    fixed = ants.image_read(str(path_fixed))
    moving = ants.image_read(str(path_moving))

    reg = ants.registration(
        fixed=fixed,
        moving=moving,
        type_of_transform="Rigid"
    )

    # Create output directory if needed
    path_output.parent.mkdir(parents=True, exist_ok=True)

    # Save ANTs image directly.
    # Do NOT convert to NumPy before saving.
    ants.image_write(
        reg["warpedmovout"],
        str(path_output)
    )

    return reg


# ============================================================
# Determine image modality
# ============================================================

def get_modality(filename):
    """
    Determine modality from the BIDS filename.

    Examples:
        sub-9110_ses-icu01_T1w.nii.gz        -> T1w
        sub-9110_ses-icu01_acq-IR_T1w.nii.gz -> T1w
        sub-9110_ses-icu01_T2w.nii.gz        -> T2w
    """

    for modality in REFERENCE_MAP:
        if filename.endswith(f"_{modality}.nii.gz"):
            return modality

    return None


# ============================================================
# Process complete dataset
# ============================================================

def register_all_sessions():

    # Find LF subject directories:
    # sub-9092, sub-9110, sub-9133, ...
    subject_dirs = sorted(DATASET_ROOT.glob("sub-*"))

    print(f"Found {len(subject_dirs)} LF subjects.")

    for subject_dir in subject_dirs:

        subject_id = subject_dir.name.replace("sub-", "")

        print("\n" + "=" * 70)
        print(f"Processing subject: {subject_id}")
        print("=" * 70)

        # Corresponding 3T directory
        hf_anat_dir = HF_ROOT / subject_id / "anat"

        if not hf_anat_dir.exists():
            print(f"  No 3T data available for subject {subject_id}. Skipping.")
            continue

        # Find all sessions
        session_dirs = sorted(subject_dir.glob("ses-*"))

        for session_dir in session_dirs:

            print(f"\n  Session: {session_dir.name}")

            lf_anat_dir = session_dir / "anat"

            if not lf_anat_dir.exists():
                print("    No anat directory. Skipping.")
                continue

            # Corresponding output directory
            relative_session_path = session_dir.relative_to(DATASET_ROOT)

            output_anat_dir = (
                OUTPUT_ROOT
                / relative_session_path
                / "anat"
            )

            output_anat_dir.mkdir(
                parents=True,
                exist_ok=True
            )

            # Find all LF NIfTI files
            lf_images = sorted(
                lf_anat_dir.glob("*.nii.gz")
            )

            for moving_path in lf_images:

                modality = get_modality(moving_path.name)

                if modality is None:
                    print(
                        f"    Unknown modality: "
                        f"{moving_path.name}. Skipping."
                    )
                    continue

                # Corresponding 3T reference filename
                reference_suffix = REFERENCE_MAP[modality]

                fixed_path = (
                    hf_anat_dir
                    / f"sub-{subject_id}_{reference_suffix}"
                )

                if not fixed_path.exists():
                    print(
                        f"    No 3T {modality} reference found. "
                        f"Skipping {moving_path.name}."
                    )
                    continue

                # Keep exactly the same LF filename
                output_path = (
                    output_anat_dir
                    / moving_path.name
                )

                # Avoid recomputing existing outputs
                if output_path.exists():
                    print(
                        f"    Already exists: "
                        f"{output_path.name}. Skipping."
                    )
                    continue

                try:
                    register_image(
                        path_moving=moving_path,
                        path_fixed=fixed_path,
                        path_output=output_path
                    )

                    print("    Registration successful.")

                except Exception as e:
                    print(
                        f"    Registration FAILED for "
                        f"{moving_path.name}"
                    )
                    print(f"    Error: {e}")

            # Copy JSON sidecars to preserve LF structure
            for json_file in lf_anat_dir.glob("*.json"):

                destination = (
                    output_anat_dir
                    / json_file.name
                )

                if not destination.exists():
                    shutil.copy2(
                        json_file,
                        destination
                    )


# ============================================================
# Run
# ============================================================

if __name__ == "__main__":
    register_all_sessions()