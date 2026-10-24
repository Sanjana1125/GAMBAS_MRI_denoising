import numpy as np, matplotlib.pyplot as plt
vol = np.load("/hdd/sanjana_intern/GAMBAS/train_data_L_to_M/lab1/train/inputs/sub-01_icu1_T1_sl4.npy")
plt.imshow(vol, cmap='gray'); plt.savefig("check_slice.png")