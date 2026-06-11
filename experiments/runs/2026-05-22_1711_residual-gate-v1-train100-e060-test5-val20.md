# Residual Gate v1 Train100 Resume to Epoch60 / Test5 + Val20

Date: 2026-05-22

## Purpose

Resume the best residual gate v1 run from epoch 20 to epoch 60 to test whether
the conservative learned reliability gate was under-trained rather than
over-releasing posterior residual diffusion corrections.

This run keeps the same cached residual diffusion features and the same v1 gate
objective. It does not use the v2 dense or sparse oracle targets.

## Setup

- Cache: `experiments/cache/residual_diffusion_features/v1_train100_val20_test5`
- Cache split: 100 train, 20 val, 5 test
- Residual diffusion checkpoint used for cache:
  `experiments/checkpoints/diffusion_v2_residual/pilot5/epoch_005.pt`
- Gate config: `code/training/configs/residual_gate_v1.yaml`
- Resume checkpoint: `experiments/checkpoints/residual_gate_v1/train100/epoch_020.pt`
- Output checkpoint directory:
  `experiments/checkpoints/residual_gate_v1/train100_e060`
- Training log:
  `experiments/runs/residual_gate_v1/train100_e060.log`

## Training

Command:

```bash
nohup /home/mingzhang/miniconda3/envs/cardiac-diffusion/bin/python \
  -m code.training.train_residual_gate \
  --config code/training/configs/residual_gate_v1.yaml \
  --cache-dir experiments/cache/residual_diffusion_features/v1_train100_val20_test5 \
  --ckpt experiments/checkpoints/residual_gate_v1/train100/epoch_020.pt \
  --run-slug residual_gate_v1_train100_e060 \
  --ckpt-dir experiments/checkpoints/residual_gate_v1/train100_e060 \
  --epochs 60 \
  --num-workers 0 \
  --no-pin-memory \
  --no-wandb \
  > experiments/runs/residual_gate_v1/train100_e060.log 2>&1 &
```

Training completed successfully:

- Total time: 6677.2 seconds
- Last epoch: 60
- Final checkpoint: `experiments/checkpoints/residual_gate_v1/train100_e060/epoch_060.pt`

Saved checkpoints:

- `epoch_025.pt`
- `epoch_030.pt`
- `epoch_035.pt`
- `epoch_040.pt`
- `epoch_045.pt`
- `epoch_050.pt`
- `epoch_055.pt`
- `epoch_060.pt`

Final epoch training summary:

- total loss: 0.06730
- final L1: 0.01978
- heart L1: 0.02034
- boundary L1: 0.02436
- boundary gradient L1: 0.02531
- gate mean: 0.01796
- gate max: 0.25

## Evaluation Outputs

Test5 epoch sweep:

- `experiments/runs/residual_gate_v1/eval_train100_e060_test5_epoch030`
- `experiments/runs/residual_gate_v1/eval_train100_e060_test5_epoch040`
- `experiments/runs/residual_gate_v1/eval_train100_e060_test5_epoch050`
- `experiments/runs/residual_gate_v1/eval_train100_e060_test5_epoch060`

Val20 epoch60:

- `experiments/runs/residual_gate_v1/eval_train100_e060_val20_epoch060`

## Test5 Epoch Sweep

All deltas are versus frozen U-Net. Negative means improvement.

| Epoch | MAE HU | Delta MAE | Delta Heart MAE | Delta Boundary MAE | Delta Boundary Grad | PSNR | SSIM | Gate Mean | Heart Gate | Boundary Gate |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 20 | 40.5791 | -0.0382 | -0.3778 | -0.3323 | -0.0904 | 32.2828 | 0.9357 | 0.0042 | 0.0436 | 0.0354 |
| 30 | 40.5670 | -0.0503 | -0.4777 | -0.4530 | -0.1233 | 32.2844 | 0.9357 | 0.0050 | 0.0515 | 0.0447 |
| 40 | 40.5552 | -0.0621 | -0.6446 | -0.6167 | -0.1967 | 32.2855 | 0.9358 | 0.0069 | 0.0666 | 0.0636 |
| 50 | 40.5481 | -0.0692 | -0.7234 | -0.7008 | -0.2123 | 32.2859 | 0.9358 | 0.0073 | 0.0697 | 0.0688 |
| 60 | 40.5427 | -0.0746 | -0.7755 | -0.7596 | -0.2235 | 32.2867 | 0.9358 | 0.0076 | 0.0714 | 0.0717 |

Epoch 60 is the best checked checkpoint on test5 across global MAE, heart MAE,
boundary MAE, boundary-gradient L1, PSNR, and SSIM.

## Epoch60 Test5 Summary

All HU metrics are lower-is-better. PSNR/SSIM are higher-is-better.

| Method | MAE HU | Heart MAE HU | Boundary MAE HU | Boundary Grad L1 HU | PSNR | SSIM |
|---|---:|---:|---:|---:|---:|---:|
| U-Net init | 40.6173 | 48.8695 | 64.5591 | 62.1894 | 32.2778 | 0.9357 |
| Posterior mean residual | 51.9974 | 56.9768 | 71.7388 | 61.0589 | 31.5134 | 0.9058 |
| Learned gate epoch60 | 40.5427 | 48.0941 | 63.7995 | 61.9659 | 32.2867 | 0.9358 |

Delta versus frozen U-Net:

- global MAE: -0.0746 HU
- heart MAE: -0.7755 HU
- boundary MAE: -0.7596 HU
- boundary-gradient L1: -0.2235 HU

Per-case consistency:

- global MAE improved on 5/5 test cases
- boundary MAE improved on 5/5 test cases
- boundary-gradient L1 improved on 5/5 test cases

Gate statistics:

- global gate mean: 0.0076
- heart gate mean: 0.0714
- boundary gate mean: 0.0717
- gate max: 0.25

## Epoch60 Val20 Summary

| Method | MAE HU | Heart MAE HU | Boundary MAE HU | Boundary Grad L1 HU | PSNR | SSIM |
|---|---:|---:|---:|---:|---:|---:|
| U-Net init | 42.1510 | 44.6561 | 56.7832 | 56.2584 | 31.9607 | 0.9365 |
| Posterior mean residual | 53.5594 | 53.0042 | 64.3138 | 55.3948 | 31.2894 | 0.9062 |
| Learned gate epoch60 | 42.0735 | 43.9905 | 56.0767 | 56.0358 | 31.9678 | 0.9367 |

Delta versus frozen U-Net:

- global MAE: -0.0775 HU
- heart MAE: -0.6656 HU
- boundary MAE: -0.7065 HU
- boundary-gradient L1: -0.2226 HU

Per-case consistency:

- global MAE improved on 20/20 val cases
- boundary MAE improved on 20/20 val cases
- boundary-gradient L1 improved on 20/20 val cases

Gate statistics:

- global gate mean: 0.0077
- heart gate mean: 0.0648
- boundary gate mean: 0.0680
- gate max: 0.25

## Interpretation

This run strengthens the v1 conclusion.

Training longer from epoch 20 to epoch 60 improves the learned gate monotonically
on test5, and epoch60 also improves every val20 case relative to the frozen U-Net.
The gains are still modest in global MAE, but they are consistent and larger in
heart and boundary regions.

The posterior mean residual remains much worse than the frozen U-Net in global,
heart, boundary, PSNR, and SSIM metrics. Its only useful signal is local and
high-frequency. This supports the current design choice: diffusion should not be
used as a direct second-stage image predictor here; it should provide posterior
residual proposals that are released through a conservative reliability gate.

The v2 dense and sparse oracle pilots are still not competitive with v1. They
release more residual but lose robustness. The best current checkpoint is now:

- `experiments/checkpoints/residual_gate_v1/train100_e060/epoch_060.pt`

## Current Conclusion

Residual Gate v1 epoch60 is the current best gated diffusion refinement result.

It does not yet produce a large enough gain to claim a major performance jump
over U-Net, but it gives a stable proof of concept:

- posterior residual diffusion contains useful local correction signal;
- direct posterior mean addition over-corrects;
- learned reliability gating can extract consistent heart/boundary gains without
  harming global metrics;
- longer v1 training is beneficial up to epoch60 on the checked splits.

## Next Steps

Recommended immediate next steps:

1. Treat v1 epoch60 as the current best gate checkpoint.
2. Run full test100 only after deciding whether to spend the GPU time now or after
   one more gate-calibration variant.
3. For the next model change, avoid dense oracle supervision. Keep the v1 loss as
   the anchor and add only a weak, sparse, high-confidence auxiliary release term.
4. Consider validating epoch40/50/60 on val20 if formal checkpoint selection is
   needed; current evidence already supports epoch60, but only epoch60 has been
   evaluated on the full val20 split.
