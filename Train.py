import torch
import csv
from torch.utils.data import DataLoader
from datasets.dataset import LowFieldMRI2D
from models.gambas_model import GambasModel
from options import Options
import os

opt = Options.parse()
print(f"=== Run: {opt.name} | lambda_adv={opt.lambda_adv} | "
      f"lambda_A={opt.lambda_A} | lambda_perc={opt.lambda_perc} | "
      f"lambda_ssim={opt.lambda_ssim} ===")

train_set = LowFieldMRI2D(opt.data_path)
val_set = LowFieldMRI2D(opt.val_path)

train_loader = DataLoader(train_set, batch_size=opt.batch_size, shuffle=True, num_workers=opt.num_workers)
val_loader = DataLoader(val_set, batch_size=1, shuffle=False, num_workers=opt.num_workers)

print(f"Training samples: {len(train_set)}")
print(f"Validation samples: {len(val_set)}")

model = GambasModel()
model.initialize(opt)
model.setup(opt)

os.makedirs(os.path.join(opt.checkpoints_dir, opt.name), exist_ok=True)

# --- per-run metrics log, one row per epoch, so all lambda_adv runs can be
# compared afterward without parsing the tee'd stdout logs ---
metrics_path = os.path.join(opt.checkpoints_dir, opt.name, 'metrics.csv')
metrics_is_new = not os.path.exists(metrics_path)
metrics_file = open(metrics_path, 'a', newline='')
metrics_writer = csv.writer(metrics_file)
if metrics_is_new:
    #metrics_writer.writerow(['epoch', 'lambda_adv', 'D', 'G_GAN', 'G_L1', 'G_perc', 'G_SSIM', 'val_L1'])
    metrics_writer.writerow(['epoch', 'G_L1', 'val_L1'])

for epoch in range(opt.epoch_count, opt.niter + opt.niter_decay + 1):

    model.netG.train()
    #model.netD.train()

    for i, (real_A, real_B) in enumerate(train_loader):
        model.set_input((real_A, real_B))
        model.optimize_parameters()

        if i % 50 == 0:
            losses = model.get_current_losses()
            print(f"Epoch {epoch} | Batch {i}/{len(train_loader)} | "
                  #f"D: {losses['D']:.4f} | G_GAN: {losses['G_GAN']:.4f} | "
                  f"G_L1: {losses['G_L1']:.4f}")

    # --- Update learning rate ---
    model.update_learning_rate()

    # --- Save checkpoint ---
    if epoch % opt.save_epoch_freq == 0:
        print(f"Saving model at epoch {epoch}")
        model.save_networks(epoch)
        model.save_networks('latest')

    # --- Validation loss ---
    val_loss = 0
    model.eval()
    with torch.no_grad():
        for real_A, real_B in val_loader:
            model.set_input((real_A, real_B))
            model.forward()
            val_loss += torch.nn.L1Loss()(model.fake_B, model.real_B).item()
    val_loss /= len(val_loader)
    print(f"Epoch {epoch} | Val L1: {val_loss:.4f}")

    # --- Log this epoch's metrics ---
    losses = model.get_current_losses()
    #metrics_writer.writerow([epoch, opt.lambda_adv, losses['D'], losses['G_GAN'],losses['G_L1'], losses['G_perc'], val_loss])
    metrics_writer.writerow([epoch, losses['G_L1'], val_loss])
    metrics_file.flush()

    model.netG.train()
    #model.netD.train()

model.save_networks('latest')
metrics_file.close()
print("Training complete.")