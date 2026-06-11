# Residual Gate v2 Oracle Pilot5 / Test5

Date: 2026-05-22

## Purpose

Test an auxiliary oracle gate loss for learned reliability gating. The v1 gate was
safe but under-released the residual proposal, so this pilot added a per-voxel
oracle target:

```text
g* = clamp(((clean - initial) * residual_mean) / residual_mean^2, 0, g_max)
```

The intent was to make the gate release more of the posterior residual in places
where the proposal points toward the clean target.

## Setup

- Config: `code/training/configs/residual_gate_v2_oracle.yaml`
- Cache: `experiments/cache/residual_diffusion_features/v1_train100_val20_test5`
- Checkpoint: `experiments/checkpoints/residual_gate_v2_oracle/pilot5/epoch_005.pt`
- Evaluation: `experiments/runs/residual_gate_v2_oracle/eval_pilot5_test5_epoch005`

Key loss settings:

- `oracle_gate_weight: 0.05`
- `oracle_gate_loss: bce`
- `oracle_gate_positive_weight: 4.0`
- `oracle_gate_heart_weight: 1.0`
- `oracle_gate_boundary_weight: 3.0`

## Training Summary

The 5-epoch pilot completed successfully.

Final epoch train metrics:

- total loss: 0.17094
- final L1: 0.01989
- heart L1: 0.02078
- boundary L1: 0.02482
- gate mean: 0.05058
- gate max: 0.14173
- oracle gate loss: 1.54636
- oracle target mean: 0.11314
- oracle target active fraction: 0.49350

The oracle target was very dense: nearly half of patch voxels had a non-trivial
positive gate target. This made the model release much more residual than v1.

## Test5 Result

All deltas are versus frozen U-Net. Negative is better.

| Run | MAE HU | Delta MAE | Delta Heart MAE | Delta Boundary MAE | Delta Boundary Grad | PSNR | SSIM | Gate Mean | Heart Gate | Boundary Gate |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| v1 epoch20 | 40.5791 | -0.0382 | -0.3778 | -0.3323 | -0.0904 | 32.2828 | 0.9357 | 0.0042 | 0.0436 | 0.0354 |
| v2 oracle pilot5 | 40.6316 | +0.0143 | -0.0113 | +0.0133 | -0.0034 | 32.2782 | 0.9356 | 0.0208 | 0.0511 | 0.0549 |

## Interpretation

This pilot is worse than v1.

The oracle BCE target made the model release residuals too broadly. It increased
the full-volume gate mean from 0.0042 to 0.0208, but did not convert that extra
release into useful heart or boundary gains. Global MAE became slightly worse
than U-Net, and the local gains almost disappeared.

The failure mode is useful: the dense per-voxel least-squares target is too
permissive. Many voxels receive a positive target even when the gain is tiny or
not robust enough to generalize. The next variant should supervise the gate only
where the oracle correction produces a clear local improvement.

## Next Variant

Use a sparse-improvement oracle target:

1. Compute the bounded per-voxel oracle gate as before.
2. Compute oracle improvement:
   `abs(initial - clean) - abs(initial + g* residual_mean - clean)`.
3. Set the target gate to zero unless improvement exceeds a small HU threshold.
4. Use a softer L1/MSE target loss instead of BCE.
5. Keep or increase gate L1 regularization so the model only releases high-confidence regions.

