# Corrected 2D GAMBAS code

This package is a cleaned, native-PyTorch replacement for the eight uploaded files.
It does **not** require `mamba-ssm`.

## Recommended controlled runs

1. `--netG resnet --lambda_adv 0 --lambda_perc 0` — plain 2D ResNet baseline.
2. `--netG gambas_identity --lambda_adv 0 --lambda_perc 0` — verifies Hilbert serialization integration.
3. `--netG gambas_gru --lambda_adv 0 --lambda_perc 0` — recommended stable Hilbert sequence model.
4. `--netG gambas_pure_mamba --lambda_adv 0 --lambda_perc 0` — experimental only.
5. Add `--lambda_adv` only after the L1-only result is anatomically correct.

Keep ordinary residual bottleneck blocks in all variants. The sequence branch is inserted at bottleneck positions 1, 5 and 9.

## Important fixes

- Correct `(H,W)` handling for Gilbert indices.
- Validate that the path is a true permutation with no duplicates or omissions.
- Compute inverse permutation over sequence positions.
- Restore Hilbert order before converting to 2D.
- Use stable bidirectional GRU as the no-install sequence backend.
- Keep CNN features dominant through a gate initialized near zero.
- Use L1 only as generator reconstruction loss; vanilla GAN uses BCE-with-logits, LSGAN uses MSE.
- Do not initialize a discriminator when `lambda_adv == 0`.
- Remove the doubly weighted and incorrectly normalized VGG loss from the default path.
- Remove evaluation-time resizing and use fixed `data_range=1.0`.

## Integration note

Your archive did not contain `options.py` or `datasets/dataset.py`; this package keeps their existing interfaces. Add the generator names above to the allowed values in your options parser if it validates `--netG` choices.

Run `python test_hilbert.py` before training.
