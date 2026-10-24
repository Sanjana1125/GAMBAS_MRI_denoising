# visualize_loco_condition.py
# Usage: python visualize_loco_condition.py --condition lab1 --netG resnet --name resnet_loco_lab1 --which_epoch best --csv_prefix holdout
import argparse
import numpy as np
import pandas as pd
import torch
import matplotlib.pyplot as plt
from pathlib import Path
from skimage.metrics import structural_similarity as ssim
from Models.gambas_model import GambasModel
from options import Options

parser = argparse.ArgumentParser()
parser.add_argument("--condition", required=True, choices=["icu1", "lab1", "lab2", "lab3", "van1"])
parser.add_argument("--netG", required=True)
parser.add_argument("--name", required=True)
parser.add_argument("--which_epoch", default="best")
parser.add_argument("--csv_prefix", default="holdout")
parser.add_argument("--rows_per_page", type=int, default=15)
args = parser.parse_args()

OUTPUT_DIR = Path("eval_results")
OUTPUT_DIR.mkdir(exist_ok=True)

# CHANGED: Manifest now correctly points to the subfolder directory matching eval_loco.py
val_manifest_path = Path("train_data_L_to_M") / args.condition / f"{args.csv_prefix}_{args.condition}_val.csv"
if not val_manifest_path.exists():
    raise SystemExit(f"{val_manifest_path} not found -- rerun make_slices.py for this condition first.")
df = pd.read_csv(val_manifest_path)
print(f"Loaded {len(df)} val slices for '{args.condition}'")


def to_01(arr): return np.clip((arr + 1) / 2, 0, 1)
def compute_brain_mask(y01, threshold=0.02): return y01 > threshold
def psnr_masked(y, pred, mask, data_range=1.0):
    mse = np.mean((y[mask].astype(np.float64) - pred[mask].astype(np.float64)) ** 2)
    return 100.0 if mse < 1e-12 else 10 * np.log10((data_range ** 2) / mse)
def ssim_masked(y, pred, mask, data_range=1.0):
    _, m = ssim(y, pred, data_range=data_range, full=True)
    return m[mask].mean()


opt = Options()
opt.isTrain = False
opt.continue_train = False
opt.netG = args.netG
opt.name = args.name
opt.which_epoch = args.which_epoch

model = GambasModel()
model.initialize(opt)
model.netG.eval()
print(f"Loaded checkpoint: {opt.name} (netG={opt.netG})")

results = []
for _, r in df.iterrows():
    x_raw = np.load(r["slice_input_path"]).astype(np.float32)
    y_raw = np.load(r["slice_target_path"]).astype(np.float32)

    x_tensor = torch.tensor(x_raw[None, None]).to(model.device)
    with torch.no_grad():
        pred = model.netG(x_tensor)
    pred_pm1 = pred.squeeze().cpu().numpy()

    x01, y01, pred01 = to_01(x_raw), to_01(y_raw), to_01(pred_pm1)
    mask = compute_brain_mask(y01)
    if mask.sum() < 20:
        continue

    # CHANGED: Now explicitly tracking input metrics alongside output metrics
    results.append({
        "subject_id": r["subject_id"], 
        "slice_idx": int(r["slice_idx"]),
        "psnr_in": psnr_masked(y01, x01, mask), 
        "psnr_out": psnr_masked(y01, pred01, mask),
        "ssim_out": ssim_masked(y01, pred01, mask),
        "x01": x01, 
        "pred01": pred01, 
        "y01": y01,
    })

results.sort(key=lambda r: r["psnr_out"])  # worst first

n = len(results)
n_pages = (n + args.rows_per_page - 1) // args.rows_per_page
for page in range(n_pages):
    page_rows = results[page * args.rows_per_page:(page + 1) * args.rows_per_page]
    fig, axes = plt.subplots(len(page_rows), 3, figsize=(9, 3 * len(page_rows)))
    if len(page_rows) == 1: axes = axes[None, :]
    for row, r in enumerate(page_rows):
        # CHANGED: Appended Input PSNR value to the Input image column title
        axes[row, 0].imshow(r["x01"], cmap='gray')
        axes[row, 0].set_title(f'{r["subject_id"]} sl{r["slice_idx"]}\nInput | PSNR:{r["psnr_in"]:.1f}', fontsize=8)
        
        axes[row, 1].imshow(r["pred01"], cmap='gray')
        axes[row, 1].set_title(f'Pred | PSNR:{r["psnr_out"]:.1f} SSIM:{r["ssim_out"]:.3f}', fontsize=8)
        
        axes[row, 2].imshow(r["y01"], cmap='gray')
        axes[row, 2].set_title('Target', fontsize=8)
        for c in range(3): axes[row, c].axis('off')
    plt.tight_layout()
    out_path = OUTPUT_DIR / f"loco_{args.condition}_{args.netG}_page{page+1}of{n_pages}.png"
    plt.savefig(out_path, dpi=150)
    plt.close()
    print(f"Saved: {out_path}")