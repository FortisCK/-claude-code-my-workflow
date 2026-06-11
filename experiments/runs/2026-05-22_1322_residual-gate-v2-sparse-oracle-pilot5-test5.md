# Residual Gate v2 Sparse Oracle Pilot5 / Test5

Date: 2026-05-22

## Purpose

The dense v2 oracle target over-released residuals. This variant made oracle
supervision sparser by keeping positive gate targets only where the oracle
correction improved local absolute error by at least `0.001` in normalized units
(about 2 HU).

## Setup

- Config: `code/training/configs/residual_gate_v2_sparse_oracle.yaml`
- Cache: `experiments/cache/residual_diffusion_features/v1_train100_val20_test5`
- Checkpoint: `experiments/checkpoints/residual_gate_v2_sparse_oracle/pilot5/epoch_005.pt`
- Evaluation: `experiments/runs/residual_gate_v2_sparse_oracle/eval_pilot5_test5_epoch005`

## Training Summary

The 5-epoch pilot completed successfully.

Final epoch train metrics:

- total loss: 0.22812
- final L1: 0.01993
- heart L1: 0.02078
- boundary L1: 0.02480
- gate mean: 0.07433
- gate max: 0.24783
- oracle gate loss: 0.53533
- oracle target mean: 0.07845
- oracle target active fraction: 0.33048

Compared with dense v2, the target was sparser, but the learned gate still became
very aggressive. The training gate max reached the configured upper bound.

## Test5 Comparison

All deltas are versus frozen U-Net. Negative is better.

| Run | MAE HU | Delta MAE | Delta Heart MAE | Delta Boundary MAE | Delta Boundary Grad | PSNR | SSIM | Gate Mean | Heart Gate | Boundary Gate |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| v1 epoch20 | 40.5791 | -0.0382 | -0.3778 | -0.3323 | -0.0904 | 32.2828 | 0.9357 | 0.0042 | 0.0436 | 0.0354 |
| v2 dense pilot5 | 40.6316 | +0.0143 | -0.0113 | +0.0133 | -0.0034 | 32.2782 | 0.9356 | 0.0208 | 0.0511 | 0.0549 |
| v2 sparse pilot5 | 40.6103 | -0.0070 | -0.0384 | -0.0361 | -0.0118 | 32.2791 | 0.9355 | 0.0407 | 0.0458 | 0.0436 |

## Interpretation

Sparse oracle is better than dense oracle, but still worse than v1.

The sparse threshold prevented the global MAE regression seen in dense v2, but it
did not recover the local gains from v1. The full-volume gate mean was ten times
larger than v1, yet heart and boundary improvements were much smaller.

This suggests that simply matching a per-voxel oracle gate is not enough. The
target needs to encode robustness, not only immediate per-voxel error reduction.
Many local oracle-positive voxels are apparently not reliable enough under
sliding-window full-volume inference.

## Current Best

The best checkpoint remains:

- `experiments/checkpoints/residual_gate_v1/train100/epoch_020.pt`

Current best test5 gains over frozen U-Net:

- global MAE: -0.038 HU
- heart MAE: -0.378 HU
- boundary MAE: -0.332 HU
- boundary gradient L1: -0.090 HU

## Next Direction

Do not continue training either dense or sparse v2 as-is.

Better next variants:

1. Use v1 as the base and add a much smaller oracle auxiliary term.
2. Raise the sparse improvement threshold substantially, for example 5-10 HU.
3. Supervise a binary "release/no-release" target only for top-percentile oracle improvements.
4. Add validation-based early stopping on test5/val20 gate metrics.
5. Consider a separate gate calibration step over the v1 checkpoint rather than training from scratch.

