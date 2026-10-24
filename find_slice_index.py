# find_slice_index.py
import h5py, numpy as np

h5_path = "repeatability/noise_corr_off/9092/T1w_TSE_Lab1.h5"

with h5py.File(h5_path, 'r') as f:
    raw = f['dataset/data'][:]

# Print all idx fields for first 10 acquisitions
print("idx fields:", raw[0]['head']['idx'].dtype.names)
print()
print(f"{'acq':>5} | {'ky':>5} | {'sl':>4} | {'rep':>4} | {'seg':>4} | {'avg':>4} | {'phase':>5} | {'set':>4} | {'user0':>6}")
print("-" * 65)
for i in range(20):
    idx = raw[i]['head']['idx']
    print(f"{i:>5} | {int(idx['kspace_encode_step_1']):>5} | "
          f"{int(idx['slice']):>4} | {int(idx['repetition']):>4} | "
          f"{int(idx['segment']):>4} | {int(idx['average']):>4} | "
          f"{int(idx['phase']):>5} | {int(idx['set']):>4} | "
          f"{int(idx['user'][0]):>6}")

# Also check total unique values per field across ALL acquisitions
print("\nUnique values per idx field across all acquisitions:")
for field in raw[0]['head']['idx'].dtype.names:
    if field == 'user':
        vals = np.unique([raw[i]['head']['idx']['user'][0] for i in range(len(raw))])
    else:
        vals = np.unique([raw[i]['head']['idx'][field] for i in range(len(raw))])
    if len(vals) > 1:
        print(f"  {field}: {len(vals)} unique values → {vals[:10]}")
