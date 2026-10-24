# eval_all_checkpoints.py
import torch
import numpy as np
import glob
import os
import csv
import matplotlib.pyplot as plt
from pathlib import Path
from skimage.metrics import peak_signal_noise_ratio as psnr
from skimage.metrics import structural_similarity as ssim
from models.gambas_model import GambasModel
from options import Options


def normalize_pm1(arr):
    """Normalize to [-1, 1] — matches Tanh output range, used as model input."""
    arr_min = arr.min()
    arr_max = arr.max()
    if arr_max - arr_min < 1e-8:
        return np.zeros_like(arr, dtype=np.float32)
    norm01 = (arr - arr_min) / (arr_max - arr_min)
    return (2 * norm01 - 1).astype(np.float32)


def to_01(arr):
    """Rescale [-1,1] -> [0,1] for metric computation/visualization."""
    return (arr + 1) / 2


def pad_to_multiple(arr, multiple=4):
    """Pad H, W up to the nearest multiple of `multiple` so the model's
    bottleneck (hardcoded for a specific downsampled size) doesn't break."""
    h, w = arr.shape
    pad_h = (multiple - h % multiple) % multiple
    pad_w = (multiple - w % multiple) % multiple
    return np.pad(arr, ((0, pad_h), (0, pad_w)), mode='edge')


def evaluate_one_checkpoint(run_name, checkpoints_dir, val_input_dir, val_target_dir, out_root):
    opt = Options()
    opt.isTrain = False
    opt.continue_train = False
    opt.name = run_name
    opt.checkpoints_dir = checkpoints_dir

    model = GambasModel()
    model.initialize(opt)

    checkpoint_path = Path(opt.checkpoints_dir) / opt.name / 'latest_net_G.pth'
    if not checkpoint_path.exists():
        print(f"  No checkpoint at {checkpoint_path} -- skipping {run_name}.")
        return None, []

    state_dict = torch.load(checkpoint_path, map_location=model.device)
    if list(state_dict.keys())[0].startswith('module.'):
        state_dict = {k[7:]: v for k, v in state_dict.items()}
    if isinstance(model.netG, torch.nn.DataParallel):
        model.netG.module.load_state_dict(state_dict)
    else:
        model.netG.load_state_dict(state_dict)
    model.netG.eval()
    print(f"  Loaded {checkpoint_path}")

    input_dir = Path(val_input_dir)
    target_dir = Path(val_target_dir)

    input_stems = {p.stem for p in input_dir.glob("*.npy")}
    target_stems = {p.stem for p in target_dir.glob("*.npy")}
    common_stems = sorted(input_stems & target_stems)

    mismatched = input_stems.symmetric_difference(target_stems)
    if mismatched:
        print(f"  WARNING: {len(mismatched)} stem(s) present in only one of "
              f"inputs/targets -- skipped: {sorted(mismatched)[:5]}"
              f"{' ...' if len(mismatched) > 5 else ''}")

    test_inputs = [str(input_dir / f"{s}.npy") for s in common_stems]
    test_targets = [str(target_dir / f"{s}.npy") for s in common_stems]

    out_dir = os.path.join(out_root, run_name)
    os.makedirs(out_dir, exist_ok=True)

    per_slice_rows = []
    psnr_in_list, psnr_out_list, ssim_in_list, ssim_out_list = [], [], [], []

    for i, (inp_path, tgt_path) in enumerate(zip(test_inputs, test_targets)):
        x_raw = np.load(inp_path).astype(np.float32)
        y_raw = np.load(tgt_path).astype(np.float32)

        x_pm1 = normalize_pm1(x_raw)
        y_pm1 = normalize_pm1(y_raw)
        x_pm1_padded = pad_to_multiple(x_pm1)

        x = to_01(x_pm1)
        y = to_01(y_pm1)

        if y.mean() < 0.05:
            continue

        x_tensor = torch.tensor(x_pm1_padded[None, None]).to(model.device)
        with torch.no_grad():
            pred = model.netG(x_tensor)

        orig_h, orig_w = x_raw.shape
        pred_pm1 = pred.squeeze().cpu().numpy()
        pred_pm1 = pred_pm1[:orig_h, :orig_w]
        pred_np = np.clip(to_01(pred_pm1), 0, 1)

        data_range = max(y.max() - y.min(), 1e-8)

        psnr_in = psnr(y, x, data_range=data_range)
        psnr_out = psnr(y, pred_np, data_range=data_range)
        ssim_in = ssim(y, x, data_range=data_range)
        ssim_out = ssim(y, pred_np, data_range=data_range)

        psnr_in_list.append(psnr_in)
        psnr_out_list.append(psnr_out)
        ssim_in_list.append(ssim_in)
        ssim_out_list.append(ssim_out)

        per_slice_rows.append({
            "run_name": run_name,
            "stem": Path(inp_path).stem,
            "psnr_input": psnr_in,
            "ssim_input": ssim_in,
            "psnr_output": psnr_out,
            "ssim_output": ssim_out,
            "psnr_gain": psnr_out - psnr_in,
            "ssim_gain": ssim_out - ssim_in,
            "foreground_frac": float(y.mean()),
        })

        if i % 50 == 0:
            stem = Path(inp_path).stem
            fig, axes = plt.subplots(1, 3, figsize=(15, 5))
            axes[0].imshow(x, cmap='gray')
            axes[0].set_title(f'Input (47mT)\nPSNR:{psnr_in:.2f} SSIM:{ssim_in:.3f}')
            axes[1].imshow(pred_np, cmap='gray')
            axes[1].set_title(f'{run_name} Output\nPSNR:{psnr_out:.2f} SSIM:{ssim_out:.3f}')
            axes[2].imshow(y, cmap='gray')
            axes[2].set_title('Target (3T ground truth)')
            fig.suptitle(stem, fontsize=10)
            plt.tight_layout()
            plt.savefig(os.path.join(out_dir, f"sample_{i:04d}.png"), dpi=150)
            plt.close(fig)

    if not psnr_out_list:
        print(f"  No valid (non-background) slices for {run_name}.")
        return None, []

    summary = {
        "run_name": run_name,
        "n_slices": len(psnr_out_list),
        "mean_psnr_input": float(np.mean(psnr_in_list)),
        "std_psnr_input": float(np.std(psnr_in_list)),
        "mean_ssim_input": float(np.mean(ssim_in_list)),
        "std_ssim_input": float(np.std(ssim_in_list)),
        "mean_psnr_output": float(np.mean(psnr_out_list)),
        "std_psnr_output": float(np.std(psnr_out_list)),
        "mean_ssim_output": float(np.mean(ssim_out_list)),
        "std_ssim_output": float(np.std(ssim_out_list)),
        "mean_psnr_gain": float(np.mean(psnr_out_list) - np.mean(psnr_in_list)),
        "mean_ssim_gain": float(np.mean(ssim_out_list) - np.mean(ssim_in_list)),
    }
    return summary, per_slice_rows


def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoints_dir", default="checkpoints")
    parser.add_argument("--val_input_dir", default="train_data_registered_final/val/inputs")
    parser.add_argument("--val_target_dir", default="train_data_registered_final/val/targets")
    parser.add_argument("--out_dir", default="eval_results_sweep")
    args = parser.parse_args()

    run_dirs = sorted(glob.glob(os.path.join(args.checkpoints_dir, "gambas_adv_*")))
    run_names = [os.path.basename(d) for d in run_dirs]
    print(f"Found {len(run_names)} checkpoint(s): {run_names}")

    os.makedirs(args.out_dir, exist_ok=True)
    summaries = []
    all_per_slice_rows = []

    for run_name in run_names:
        print(f"\n=== Evaluating {run_name} ===")
        summary, per_slice_rows = evaluate_one_checkpoint(
            run_name, args.checkpoints_dir, args.val_input_dir, args.val_target_dir, args.out_dir
        )
        if summary:
            summaries.append(summary)
            all_per_slice_rows.extend(per_slice_rows)
            print(f"  PSNR: {summary['mean_psnr_output']:.2f} (+{summary['mean_psnr_gain']:.2f} vs input)  "
                  f"SSIM: {summary['mean_ssim_output']:.4f} (+{summary['mean_ssim_gain']:.4f} vs input)")

    if not summaries:
        print("Nothing evaluated.")
        return

    summary_csv = os.path.join(args.out_dir, "summary.csv")
    with open(summary_csv, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(summaries[0].keys()))
        writer.writeheader()
        writer.writerows(summaries)

    per_slice_csv = os.path.join(args.out_dir, "per_slice_detail.csv")
    with open(per_slice_csv, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(all_per_slice_rows[0].keys()))
        writer.writeheader()
        writer.writerows(all_per_slice_rows)

    print(f"\nDone. Summary: {summary_csv}")
    print(f"Per-slice detail: {per_slice_csv}")
    print(f"Sample images: {args.out_dir}/<run_name>/")


if __name__ == "__main__":
    main()