# Run: Residual Refiner v1 Epoch010 Test20

Date: 2026-05-20  
Purpose: Pilot deterministic residual refiner on top of frozen U-Net v1 before
implementing posterior residual diffusion.

## Training Command

```bash
/home/mingzhang/miniconda3/envs/cardiac-diffusion/bin/python \
  -m code.training.train_residual_refiner \
  --config code/training/configs/residual_refiner_v1.yaml \
  --epochs 10 \
  --no-wandb
```

## Evaluation Command

```bash
/home/mingzhang/miniconda3/envs/cardiac-diffusion/bin/python \
  scripts/python/evaluate_residual_refiner_full_volume.py \
  --ckpt experiments/checkpoints/residual_refiner_v1/epoch_010.pt \
  --val-cases 0 \
  --test-cases 20 \
  --figure-cases 8 \
  --out-dir experiments/runs/residual_refiner_v1/eval_epoch010_test20
```

## Model

- Frozen initializer: U-Net v1 epoch200 EMA
- Refiner input: `[corrupted, unet_pred, corrupted - unet_pred]`
- Refiner target: `clean - unet_pred`
- Refiner architecture: MONAI 3D BasicUNet, features `[32, 32, 64, 128, 256, 32]`
- Loss: final-image L1 + 0.05 gradient L1
- Training: full train split, precomputed pairs, 128^3 patches, 10 epochs

## Outputs

- Checkpoint: `experiments/checkpoints/residual_refiner_v1/epoch_010.pt`
- Metrics: `experiments/runs/residual_refiner_v1/eval_epoch010_test20/metrics.csv`
- Summary: `experiments/runs/residual_refiner_v1/eval_epoch010_test20/summary.json`
- Figures: `experiments/runs/residual_refiner_v1/eval_epoch010_test20/figures/`

## Training Signal

Mean train loss by epoch:

| Epoch | Total | Final L1 | Gradient |
| ---: | ---: | ---: | ---: |
| 1 | 0.03097 | 0.02974 | 0.02468 |
| 2 | 0.02045 | 0.01966 | 0.01567 |
| 3 | 0.01977 | 0.01901 | 0.01530 |
| 4 | 0.01943 | 0.01868 | 0.01505 |
| 5 | 0.01945 | 0.01870 | 0.01505 |
| 6 | 0.01934 | 0.01858 | 0.01515 |
| 7 | 0.01908 | 0.01833 | 0.01486 |
| 8 | 0.01932 | 0.01857 | 0.01508 |
| 9 | 0.01915 | 0.01841 | 0.01472 |
| 10 | 0.01952 | 0.01875 | 0.01535 |

## Test20 Results

| Metric | U-Net init | Refiner | Delta refiner - U-Net |
| --- | ---: | ---: | ---: |
| Global MAE HU | 38.4924 | 38.5111 | +0.0187 |
| PSNR | 33.1854 | 33.1865 | +0.0011 |
| SSIM | 0.944695 | 0.944655 | -0.000040 |
| NRMSE | 0.028236 | 0.028235 | -0.000001 |
| Heart MAE HU | 44.4160 | 44.3847 | -0.0313 |
| Heart RMSE HU | 68.7194 | 68.6683 | -0.0511 |
| Boundary MAE HU | 57.4521 | 57.4484 | -0.0037 |
| Boundary RMSE HU | 100.4064 | 100.4097 | +0.0033 |
| Boundary gradient L1 HU | 56.0384 | 56.0636 | +0.0252 |

## Interpretation

The deterministic residual refiner learns the patch-level residual objective,
but after 10 epochs it essentially matches the frozen U-Net initializer on
full-volume test20. It does not provide meaningful global or boundary gains.

This is useful evidence for the paper strategy:

- A plain second U-Net refiner is not enough to materially improve U-Net v1.
- The v2 method should not be framed as a deterministic cascade.
- The next model needs explicit posterior residual modeling, uncertainty or
  gate control, and artifact-aware losses.
- Heart MAE improved slightly while global MAE worsened slightly, suggesting
  region-specific objectives may be necessary.

## Next Design Consequence

Do not spend a long run on this exact deterministic refiner. The next
engineering step should add at least one of:

- heart and boundary weighted losses,
- residual magnitude/gate control,
- uncertainty-aware posterior residual diffusion,
- or a hybrid where deterministic refiner provides the mean and diffusion
  models residual uncertainty around it.

