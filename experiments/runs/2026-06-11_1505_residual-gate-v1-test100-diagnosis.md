# Residual Gate v1 Epoch60 / Test100 Qualitative Diagnosis

Date: 2026-06-11

## Purpose

Generate qualitative diagnosis panels for the full test100 residual-gate result.
The goal is to inspect where the learned reliability gate releases posterior
residual diffusion corrections, and to compare best-gain, worst-gain, high-error,
and original test5 cases.

This does not rerun diffusion sampling. It uses the cached full-volume posterior
features from:

- `experiments/cache/residual_diffusion_features/v1_test100`

and reruns only the lightweight GateNet sliding-window inference from:

- `experiments/checkpoints/residual_gate_v1/train100_e060/epoch_060.pt`

## Script

Reusable script:

- `scripts/python/make_residual_gate_diagnosis_figures.py`

Command:

```bash
/home/mingzhang/miniconda3/envs/cardiac-diffusion/bin/python \
  scripts/python/make_residual_gate_diagnosis_figures.py \
  --ckpt experiments/checkpoints/residual_gate_v1/train100_e060/epoch_060.pt \
  --cache-dir experiments/cache/residual_diffusion_features/v1_test100 \
  --metrics-csv experiments/runs/residual_gate_v1/eval_train100_e060_test100_epoch060/metrics.csv \
  --out-dir experiments/runs/residual_gate_v1/diagnosis_test100_epoch060 \
  --cases-per-group 5 \
  --device cuda
```

## Outputs

Output directory:

- `experiments/runs/residual_gate_v1/diagnosis_test100_epoch060`

Generated files:

- `selected_cases.csv`
- `diagnosis_manifest.json`
- 22 PNG diagnosis panels under `figures/`

The PNGs are local generated artifacts and are intentionally not tracked by git.

## Case Selection

The script selected 22 unique cases from:

- best 5 global MAE deltas
- worst 5 global MAE deltas
- highest 5 U-Net global MAE cases
- highest 5 U-Net boundary MAE cases
- original test5 sanity cases: `21`, `32`, `41`, `50`, `51`

Selected cases:

| Case | Groups | Delta MAE HU | Delta Heart HU | Delta Boundary HU | Severity |
|---|---|---:|---:|---:|---|
| 642 | best_global_delta | -0.1259 | -0.8840 | -0.8645 | severe |
| 468 | best_global_delta | -0.1179 | -0.7085 | -0.7341 | severe |
| 375 | best_global_delta | -0.1099 | -0.7150 | -0.8858 | medium |
| 483 | best_global_delta; highest_boundary_mae | -0.1056 | -0.9119 | -1.0974 | severe |
| 547 | best_global_delta; highest_boundary_mae | -0.1052 | -0.9486 | -0.9774 | severe |
| 732 | worst_global_delta | -0.0335 | -0.2740 | -0.5382 | mild |
| 943 | worst_global_delta | -0.0345 | -0.5247 | -0.5790 | mild |
| 251 | worst_global_delta | -0.0456 | -0.3888 | -0.4854 | mild |
| 817 | worst_global_delta | -0.0485 | -0.6122 | -0.5282 | severe |
| 630 | worst_global_delta | -0.0522 | -0.3799 | -0.4598 | mild |
| 821 | highest_boundary_mae; highest_unet_mae | -0.0772 | -0.2600 | -0.3655 | mild |
| 798 | highest_unet_mae | -0.0950 | -0.8552 | -0.6745 | severe |
| 952 | highest_unet_mae | -0.1029 | -0.7139 | -0.8660 | medium |
| 339 | highest_unet_mae | -0.0550 | -0.4316 | -0.5033 | mild |
| 733 | highest_unet_mae | -0.0756 | -0.6966 | -0.5820 | mild |
| 53 | highest_boundary_mae | -0.0928 | -1.0207 | -1.0774 | severe |
| 85 | highest_boundary_mae | -0.0821 | -1.1380 | -0.9996 | severe |
| 21 | original_test5 | -0.0695 | -0.5674 | -0.6778 | medium |
| 32 | original_test5 | -0.0738 | -0.6808 | -0.6400 | medium |
| 41 | original_test5 | -0.0658 | -0.8746 | -0.8587 | severe |
| 50 | original_test5 | -0.0732 | -0.7081 | -0.6952 | severe |
| 51 | original_test5 | -0.0967 | -0.9348 | -1.1070 | severe |

## Panel Contents

Each PNG panel contains:

- clean
- corrupted
- U-Net init
- gated refinement
- posterior mean
- corrupted error map
- U-Net error map
- gated refinement error map
- U-Net-to-gated error gain map
- gate map
- residual mean map
- residual std map
- `gate * residual` correction map
- `|residual_mean| / residual_std` map
- heart/boundary overlay

Heart contours are green. Boundary-band contours are cyan.

## Initial Visual Check

Two representative panels were inspected:

- `case_642_best_global_delta.png`
- `case_732_worst_global_delta.png`

The figures are non-empty and the maps are coherent. The gate is sparse globally
but more active in heart/boundary areas, while the posterior residual mean is
large and noisy. The actual applied correction remains small because it is
multiplied by the learned gate.

This visually supports the quantitative interpretation from full test100:

- direct posterior mean residual is too unreliable as an image predictor;
- the learned gate suppresses most residual proposals;
- useful corrections are released locally around heart/boundary structures;
- even worst-gain cases remain positive but conservative.
