# check_all_h5.py
import h5py, numpy as np
from pathlib import Path

base_off = Path("repeatability/noise_corr_off")

for subj in sorted(base_off.iterdir()):
    for h5_file in sorted(subj.glob("*.h5")):
        with h5py.File(str(h5_file), 'r') as f:
            raw = f['dataset/data'][:]
        
        all_ky = np.array([int(raw[i]['head']['idx']['kspace_encode_step_1']) for i in range(len(raw))])
        all_sl = np.array([int(raw[i]['head']['idx']['slice']) for i in range(len(raw))])
        
        n_slices = len(np.unique(all_sl))
        Ny       = int(all_ky.max()) + 1
        Nx       = int(raw[0]['head']['number_of_samples'])
        
        print(f"{subj.name}/{h5_file.name}: slices={n_slices}, Ny={Ny}, Nx={Nx}, total_acq={len(raw)}")
