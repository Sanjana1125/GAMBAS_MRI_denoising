import os
import nibabel as nib
import ants

base_dir = "repeatability_share_data"
output_reg_dir = "./ants_registered_data"
os.makedirs(output_reg_dir, exist_ok=True)

paired_subjects = ["sub-9110", "sub-9133"]

def find_t1w_file(search_path, require_keyword=None, exclude_keywords=[]):
    if not os.path.exists(search_path):
        return None
    for root, _, files in os.walk(search_path):
        for file in files:
            if file.endswith(('.nii', '.nii.gz')) and '_T1w' in file:
                if require_keyword and require_keyword not in file:
                    continue
                if any(x in file for x in exclude_keywords):
                    continue
                return os.path.join(root, file)
    return None

# ---------------------------------------------------------------------------
# Recommended Pipeline: 47mT Moving -> 3T Highres Fixed Space
# ---------------------------------------------------------------------------
for sub in paired_subjects:
    print("\n==========================================")
    print(f"Registering 47mT to 3T Highres Space for {sub}...")
    print("==========================================")
    
    path_3t = os.path.join(base_dir, "3T_Data_defaced", sub.replace("sub-", ""))
    if not os.path.exists(path_3t):
        path_3t = os.path.join(base_dir, "3T_Data_defaced", sub)
        
    # Explicitly require 'highres' to get the best possible anatomical target
    file_3t = find_t1w_file(path_3t, require_keyword="highres") 
    if not file_3t:
        print(f"Skipping {sub}: Could not find 3T 'highres' template.")
        continue
        
    print(f"Using Fixed Template: {os.path.basename(file_3t)}")
    fixed_img = ants.image_read(file_3t)
    
    sub_path_47mt = os.path.join(base_dir, sub)
    if not os.path.exists(sub_path_47mt):
        continue
        
    sessions = sorted([d for d in os.listdir(sub_path_47mt) if os.path.isdir(os.path.join(sub_path_47mt, d)) and d.startswith("ses-")])
    
    for ses in sessions[:5]:
        ses_path = os.path.join(sub_path_47mt, ses)
        file_47mt = find_t1w_file(ses_path, exclude_keywords=["IR_T1w"])
        
        if file_47mt:
            print(f"Aligning {ses} to high-resolution template space...")
            moving_img = ants.image_read(file_47mt)
            
            # Rigid transformation aligns them without risking shape warping
            reg_result = ants.registration(
                fixed=fixed_img,
                moving=moving_img,
                type_of_transform='Rigid',
                multivariate_extras='MutualInformation'
            )
            
            sub_output_dir = os.path.join(output_reg_dir, sub, ses)
            os.makedirs(sub_output_dir, exist_ok=True)
            
            output_filename = os.path.join(sub_output_dir, f"{sub}_{ses}_space-3Thighres_T1w.nii.gz")
            ants.image_write(reg_result['warpedmovout'], output_filename)
            print(f"Saved perfectly aligned session file: {output_filename}")