# eval_loco.py
# Usage: python eval_loco.py --condition lab1 --netG resnet --name resnet_loco_lab1 --which_epoch best --csv_prefix holdout
import argparse
import numpy as np
import pandas as pd
import torch
from pathlib import Path
from skimage.metrics import structural_similarity as ssim
from Models.gambas_model import GambasModel
from options import Options

parser = argparse.ArgumentParser()
parser.add_argument("--condition", required=True, choices=["icu1", "lab1", "lab2", "lab3", "van1"])
parser.add_argument("--netG", required=True)
parser.add_argument("--name", required=True, help="checkpoint folder name for this fold")
parser.add_argument("--which_epoch", default="best")
parser.add_argument("--csv_prefix", default="holdout",
                     help="Manifest filename prefix used by make_slices.py's --out-prefix")
parser.add_argument("--results_csv", default="eval_results/loco_all_folds.csv",
                     help="Persistent results file -- lives OUTSIDE train_data_L_to_M, survives the wipe.")
args = parser.parse_args()

Path("eval_results").mkdir(exist_ok=True)

# CHANGED: manifest now lives under train_data_L_to_M/{condition}/, matching
# make_slices.py's per-condition output directories (train_data_L_to_M/train_data_L_to_M/holdout_X_val.csv is gone).
val_manifest_path = Path("train_data_L_to_M") / args.condition / f"{args.csv_prefix}_{args.condition}_val.csv"
if not val_manifest_path.exists():
    raise SystemExit(f"{val_manifest_path} not found -- run make_slices.py --held-out-condition "
                      f"{args.condition} first (with matching --out-prefix / --csv_prefix).")
df = pd.read_csv(val_manifest_path)
print(f"Loaded {len(df)} val slices for held-out condition '{args.condition}'")


def to_01(arr):
    return np.clip((arr + 1) / 2, 0, 1)


def compute_brain_mask(y_01, threshold=0.02):
    return y_01 > threshold


def psnr_masked(y, pred, mask, data_range=1.0):
    diff = y[mask].astype(np.float64) - pred[mask].astype(np.float64)
    mse = np.mean(diff ** 2)
    return 100.0 if mse < 1e-12 else 10 * np.log10((data_range ** 2) / mse)


def ssim_masked(y, pred, mask, data_range=1.0):
    _, ssim_map = ssim(y, pred, data_range=data_range, full=True)
    return ssim_map[mask].mean()


# Plain Options(), not Options.parse() -- avoids clashing with this script's
# own argparse parser reading the same sys.argv (both would otherwise try
# to consume --condition / --csv_prefix / etc. and error out).
opt = Options()
opt.isTrain = False
opt.continue_train = False
opt.netG = args.netG
opt.name = args.name
opt.which_epoch = args.which_epoch

model = GambasModel()
model.initialize(opt)
model.netG.eval()
print(f"Loaded checkpoint: {opt.name} (netG={opt.netG}, which_epoch={opt.which_epoch})")

rows = []
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

    rows.append({
        "netG": args.netG,
        "condition": args.condition,
        "subject_id": r["subject_id"],
        "sequence": r["sequence"],
        "slice_idx": int(r["slice_idx"]),
        "slice_position_norm": r["slice_position_norm"],
        "psnr_in": psnr_masked(y01, x01, mask),
        "psnr_out": psnr_masked(y01, pred01, mask),
        "ssim_in": ssim_masked(y01, x01, mask),
        "ssim_out": ssim_masked(y01, pred01, mask),
    })

fold_results = pd.DataFrame(rows)
fold_results["improvement"] = fold_results["psnr_out"] - fold_results["psnr_in"]

results_path = Path(args.results_csv)
if results_path.exists():
    existing = pd.read_csv(results_path)
    # drop any prior rows for this exact (netG, condition) so reruns don't duplicate
    existing = existing[~((existing["netG"] == args.netG) & (existing["condition"] == args.condition))]
    combined = pd.concat([existing, fold_results], ignore_index=True)
else:
    combined = fold_results
combined.to_csv(results_path, index=False)

print(f"\n[{args.condition}] n={len(fold_results)}  "
      f"input PSNR={fold_results['psnr_in'].mean():.2f} dB  "
      f"output PSNR={fold_results['psnr_out'].mean():.2f} dB  "
      f"improvement={fold_results['improvement'].mean():+.2f} dB")
print(f"Appended to {results_path} (now {len(combined)} total rows across all folds/models run so far)")