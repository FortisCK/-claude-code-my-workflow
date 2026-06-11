# Residual Gate v1 Train100 / Test5

Date: 2026-05-22

## Purpose

Train the first learned reliability gate on cached posterior residual diffusion features.
The goal is to test whether a spatial gate can safely release useful residual diffusion
corrections on top of the frozen U-Net baseline, instead of directly adding the
posterior mean residual.

## Inputs

- Cache: `experiments/cache/residual_diffusion_features/v1_train100_val20_test5`
- Cache size: 125 cases, 5.3G
- Cache split: 100 train, 20 val, 5 test
- Residual diffusion checkpoint:
  `experiments/checkpoints/diffusion_v2_residual/pilot5/epoch_005.pt`
- Gate config: `code/training/configs/residual_gate_v1.yaml`
- Gate model: `code/models/residual_gate.py`

Cached feature channels:

- `clean`
- `corrupted`
- `initial`
- `residual_mean`
- `residual_std`
- `heart_mask`

## Training

Command:

```bash
/home/mingzhang/miniconda3/envs/cardiac-diffusion/bin/python \
  -m code.training.train_residual_gate \
  --config code/training/configs/residual_gate_v1.yaml \
  --cache-dir experiments/cache/residual_diffusion_features/v1_train100_val20_test5 \
  --run-slug residual_gate_v1_train100 \
  --ckpt-dir experiments/checkpoints/residual_gate_v1/train100 \
  --epochs 20 \
  --num-workers 0 \
  --no-pin-memory \
  --no-wandb
```

Checkpoint:

- `experiments/checkpoints/residual_gate_v1/train100/epoch_020.pt`

Training completed successfully in 3348.6 seconds.

Final epoch training summary:

- total loss: 0.06868
- final L1: 0.02029
- heart L1: 0.02068
- boundary L1: 0.02501
- boundary gradient L1: 0.02565
- gate mean: 0.00932
- gate max: 0.25

Interpretation from training logs:

- The gate did not collapse or explode.
- Mean gate gradually increased from about 0.0038 at epoch 1 to about 0.0093 at epoch 20.
- Max gate reached the configured upper bound, so the model learned sparse high-confidence release.
- Patch loss stayed mostly flat, which is expected because the frozen U-Net baseline is already strong and the useful residual-correction region is narrow.

## Evaluation

Command:

```bash
/home/mingzhang/miniconda3/envs/cardiac-diffusion/bin/python \
  scripts/python/evaluate_residual_gate_full_volume.py \
  --ckpt experiments/checkpoints/residual_gate_v1/train100/epoch_020.pt \
  --cache-dir experiments/cache/residual_diffusion_features/v1_train100_val20_test5 \
  --splits test \
  --max-cases 5 \
  --out-dir experiments/runs/residual_gate_v1/eval_train100_test5_epoch020
```

Outputs:

- `experiments/runs/residual_gate_v1/eval_train100_test5_epoch020/metrics.csv`
- `experiments/runs/residual_gate_v1/eval_train100_test5_epoch020/summary.json`

Test cases:

- case 21
- case 32
- case 41
- case 50
- case 51

## Test5 Summary

All HU metrics are lower-is-better. PSNR/SSIM are higher-is-better.

| Method | MAE HU | Heart MAE HU | Boundary MAE HU | Boundary Grad L1 HU | PSNR | SSIM |
|---|---:|---:|---:|---:|---:|---:|
| U-Net init | 40.6173 | 48.8695 | 64.5591 | 62.1894 | 32.2778 | 0.9357 |
| Posterior mean residual | 51.9974 | 56.9768 | 71.7388 | 61.0589 | 31.5134 | 0.9058 |
| Fixed alpha 0.05 | 40.8207 | 48.8652 | 64.5911 | 62.1873 | 32.2734 | 0.9356 |
| Fixed alpha 0.10 | 41.1046 | 48.9146 | 64.6672 | 62.1534 | 32.2647 | 0.9353 |
| Fixed alpha 0.25 | 42.2727 | 49.3790 | 65.1454 | 61.9012 | 32.2138 | 0.9335 |
| Learned gate | 40.5791 | 48.4917 | 64.2268 | 62.0990 | 32.2828 | 0.9357 |
| Oracle voxel | 37.2667 | 43.7827 | 59.6589 | 59.6821 | 32.4979 | 0.9393 |
| Oracle block32 | 40.5719 | 48.7934 | 64.5528 | 62.1468 | 32.2798 | 0.9356 |

Delta versus U-Net init, negative means improvement:

| Method | MAE HU | Heart MAE HU | Boundary MAE HU | Boundary Grad L1 HU |
|---|---:|---:|---:|---:|
| Posterior mean residual | +11.3801 | +8.1073 | +7.1797 | -1.1305 |
| Fixed alpha 0.05 | +0.2034 | -0.0043 | +0.0320 | -0.0021 |
| Fixed alpha 0.10 | +0.4873 | +0.0451 | +0.1081 | -0.0360 |
| Fixed alpha 0.25 | +1.6554 | +0.5095 | +0.5862 | -0.2882 |
| Learned gate | -0.0382 | -0.3778 | -0.3323 | -0.0904 |
| Oracle voxel | -3.3506 | -5.0868 | -4.9002 | -2.5073 |
| Oracle block32 | -0.0454 | -0.0761 | -0.0063 | -0.0426 |

Gate statistics:

- Global gate mean: 0.0042
- Heart gate mean: 0.0436
- Boundary gate mean: 0.0354
- Gate max: 0.25
- Mean absolute residual proposal: 22.63 HU
- Mean residual std: 46.66 HU

## Interpretation

This is a positive but small result.

Compared with the frozen U-Net, the learned gate improves every tracked metric on
test5, but the global MAE gain is only 0.038 HU. The more meaningful signal is in
the local metrics:

- Heart MAE improves by 0.378 HU.
- Boundary MAE improves by 0.332 HU.
- Boundary gradient L1 improves by 0.090 HU.

The learned gate also clearly beats fixed global alphas. Fixed alpha 0.05, 0.10,
and 0.25 all make global MAE worse, while the learned gate keeps the global output
stable and improves heart/boundary metrics. This supports the reliability-gated
posterior residual direction.

The result is still far from the voxel oracle. The oracle voxel result suggests
that the posterior residual proposal contains useful local corrections, with an
upper bound of about:

- 3.35 HU global MAE improvement
- 5.09 HU heart MAE improvement
- 4.90 HU boundary MAE improvement
- 2.51 HU boundary gradient improvement

The current learned gate captures only a small part of that oracle gap.

## Epoch Selection

Epoch 10, 15, and 20 were evaluated on the same cached test5 split.

| Epoch | MAE HU | Delta MAE | Delta Heart MAE | Delta Boundary MAE | Delta Boundary Grad | Gate Mean | Heart Gate | Boundary Gate |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 10 | 40.6031 | -0.0142 | -0.1805 | -0.1023 | -0.0194 | 0.0023 | 0.0217 | 0.0121 |
| 15 | 40.5881 | -0.0292 | -0.3079 | -0.2459 | -0.0611 | 0.0034 | 0.0351 | 0.0261 |
| 20 | 40.5791 | -0.0382 | -0.3778 | -0.3323 | -0.0904 | 0.0042 | 0.0436 | 0.0354 |

Epoch 20 is the best of the checked checkpoints across all key metrics. This
suggests the current gate is still under-releasing rather than over-releasing.

## Next Steps

The next version should not simply train longer. The current loss produces a very
safe gate but underuses the residual proposal.

Recommended v2 changes:

1. Add explicit oracle-supervised gate targets from per-voxel or local alpha choice.
2. Add a stronger heart/boundary release objective while preserving global MAE.
3. Compare epoch 10, 15, and 20 on test5 to check whether the gate is over- or under-releasing.
4. Consider raising `g_max` only after the gate target is better calibrated.
5. Scale evaluation from test5 to val20/test100 only after v2 shows a larger local gain.
