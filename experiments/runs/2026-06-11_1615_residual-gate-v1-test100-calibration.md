# Residual Gate v1 Epoch60 / Test100 Calibration Analysis

Date: 2026-06-11

## Purpose

Analyze whether the learned reliability gate is actually calibrated: when it
releases more posterior residual, does the voxel-level correction become more
reliable?

This analysis reruns only GateNet sliding-window inference on cached posterior
features. It does not rerun diffusion sampling.

## Inputs

- Gate checkpoint:
  `experiments/checkpoints/residual_gate_v1/train100_e060/epoch_060.pt`
- Test100 residual diffusion cache:
  `experiments/cache/residual_diffusion_features/v1_test100`
- Full test100 metrics:
  `experiments/runs/residual_gate_v1/eval_train100_e060_test100_epoch060/metrics.csv`

## Script

Reusable script:

- `scripts/python/analyze_residual_gate_calibration.py`

Command:

```bash
/home/mingzhang/miniconda3/envs/cardiac-diffusion/bin/python \
  scripts/python/analyze_residual_gate_calibration.py \
  --ckpt experiments/checkpoints/residual_gate_v1/train100_e060/epoch_060.pt \
  --cache-dir experiments/cache/residual_diffusion_features/v1_test100 \
  --metrics-csv experiments/runs/residual_gate_v1/eval_train100_e060_test100_epoch060/metrics.csv \
  --splits test \
  --out-dir experiments/runs/residual_gate_v1/calibration_test100_epoch060 \
  --device cuda
```

## Outputs

Output directory:

- `experiments/runs/residual_gate_v1/calibration_test100_epoch060`

Files:

- `region_summary.csv`
- `bin_summary.csv`
- `case_summary.csv`
- `calibration_manifest.json`

The CSV summaries are small and are tracked as formal analysis artifacts.

## Region-Level Summary

Voxel-level improvement is defined as:

```text
abs(U-Net - clean) - abs(gated_refinement - clean)
```

Positive is better, in HU.

| Region | Voxels | Mean Improvement HU | Positive Fraction | U-Net Error HU | Gated Error HU | Posterior Mean Improvement HU | Mean Gate | Applied Correction HU |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| all | 707,788,800 | 0.0762 | 0.1163 | 38.0782 | 38.0020 | -11.4674 | 0.0076 | 0.2364 |
| heart | 52,456,734 | 0.6356 | 0.3705 | 42.7062 | 42.0706 | -8.2782 | 0.0628 | 1.9017 |
| boundary | 42,300,389 | 0.7103 | 0.3696 | 55.9476 | 55.2374 | -7.2540 | 0.0673 | 1.9467 |
| heart_interior | 33,342,790 | 0.5910 | 0.3610 | 39.4440 | 38.8530 | -8.2941 | 0.0569 | 1.7820 |
| non_heart | 655,332,066 | 0.0314 | 0.0960 | 37.7078 | 37.6763 | -11.7227 | 0.0032 | 0.1031 |

Main takeaway: the global improvement is small because most voxels are outside
the heart and receive almost no correction. Inside heart/boundary regions, the
mean improvement is an order of magnitude larger.

## Gate Calibration

Boundary-region gate bins:

| Gate Bin | Boundary Fraction | Mean Improvement HU | Positive Fraction | Posterior Mean Improvement HU | Applied Correction HU |
|---|---:|---:|---:|---:|---:|
| <0.001 | 55.06% | -0.000 | 0.155 | -12.393 | 0.00 |
| [0.001,0.005) | 6.04% | -0.000 | 0.499 | -7.734 | 0.05 |
| [0.005,0.01) | 2.39% | 0.006 | 0.524 | -6.939 | 0.16 |
| [0.01,0.02) | 2.34% | 0.021 | 0.540 | -6.447 | 0.32 |
| [0.02,0.05) | 3.26% | 0.072 | 0.556 | -6.041 | 0.75 |
| [0.05,0.1) | 2.95% | 0.195 | 0.573 | -5.446 | 1.70 |
| [0.1,0.15) | 2.38% | 0.369 | 0.585 | -4.838 | 2.97 |
| [0.15,0.2) | 2.76% | 0.578 | 0.597 | -4.256 | 4.33 |
| >=0.2 | 22.80% | 2.968 | 0.717 | +4.142 | 7.31 |

Heart-region gate bins show the same trend:

| Gate Bin | Heart Fraction | Mean Improvement HU | Positive Fraction | Posterior Mean Improvement HU | Applied Correction HU |
|---|---:|---:|---:|---:|---:|
| <0.001 | 55.53% | -0.000 | 0.167 | -12.170 | 0.00 |
| [0.05,0.1) | 3.18% | 0.252 | 0.580 | -7.937 | 1.99 |
| [0.1,0.15) | 2.56% | 0.457 | 0.588 | -7.192 | 3.44 |
| [0.15,0.2) | 2.95% | 0.686 | 0.598 | -6.321 | 4.93 |
| >=0.2 | 20.66% | 2.861 | 0.705 | +2.410 | 7.53 |

This is strong evidence that the learned gate is meaningful: high-gate voxels
are much more likely to produce useful residual correction.

## Applied-Correction Calibration

Boundary-region bins by `abs(gate * residual_mean)`:

| Applied Correction HU | Boundary Fraction | Mean Improvement HU | Positive Fraction | Posterior Mean Improvement HU |
|---|---:|---:|---:|---:|
| <0.05 | 60.46% | -0.000 | 0.186 | -11.215 |
| [0.5,1) | 2.98% | 0.158 | 0.606 | -4.355 |
| [1,2) | 4.05% | 0.476 | 0.659 | -1.983 |
| [2,5) | 9.76% | 1.358 | 0.696 | +1.985 |
| >=5 | 14.42% | 3.829 | 0.689 | +0.867 |

The correction magnitude is also calibrated: when the model applies larger
corrections, those voxels have substantially higher average improvement.

## Residual SNR Calibration

Boundary-region bins by `abs(residual_mean) / residual_std`:

| Residual SNR | Boundary Fraction | Mean Improvement HU | Positive Fraction | Posterior Mean Improvement HU | Applied Correction HU |
|---|---:|---:|---:|---:|---:|
| <0.1 | 12.13% | 0.030 | 0.295 | -0.078 | 0.09 |
| [0.1,0.25) | 17.91% | 0.215 | 0.353 | -0.785 | 0.49 |
| [0.25,0.5) | 27.87% | 0.625 | 0.377 | -4.803 | 1.60 |
| [0.5,1) | 31.15% | 1.125 | 0.392 | -13.723 | 3.20 |
| [1,2) | 9.58% | 1.310 | 0.399 | -13.738 | 3.68 |
| >=2 | 1.36% | about 1.33 | about 0.40 | about -12 to -13 | about 3.7 |

Residual SNR is useful, but it is not sufficient by itself. Higher SNR bins have
higher gated improvement, but direct posterior mean addition is still strongly
negative in most SNR bins. This supports using SNR as a gate feature, not as a
direct release rule.

## Interpretation

The calibration analysis supports the current reliability-gated diffusion story:

- The gate is not arbitrary. Higher gate values correspond to higher mean
  voxel-level improvement and higher positive-improvement fractions.
- Most non-heart voxels are suppressed, which explains why global MAE gains are
  small despite larger local gains.
- Direct posterior mean residual is usually unsafe. It is negative globally and
  in most gate/SNR bins, except in the highest-gate regions.
- Residual std behaves partly like an artifact/magnitude signal, not simply as
  inverse reliability. High-std boundary voxels can still improve substantially
  after gating, but posterior mean remains unsafe there.
- The strongest improvement appears where both gate and applied correction are
  high, especially in boundary/heart regions.

## Design Implications for v3

Recommended v3 changes:

1. Add explicit `heart_mask` and `boundary_band` inputs. The useful correction is
   much more concentrated in these regions.
2. Add residual SNR as an input channel:
   `abs(residual_mean) / (residual_std + eps)`.
3. Keep v1's conservative loss as the anchor. The current gate is well calibrated
   and should not be replaced by dense oracle supervision.
4. Add only a weak sparse high-confidence release auxiliary loss, focused on
   heart/boundary voxels with high gate/SNR or strong oracle improvement.
5. Do not blindly suppress high residual std. High std often coincides with
   high-error/high-artifact regions where gated correction can still help.

The immediate next model experiment should be:

```text
residual_gate_v3a:
  inputs = v1 inputs + heart_mask + boundary_band + residual_snr
  loss = v1 loss
  g_max = 0.25
```

Then compare against:

```text
residual_gate_v3b:
  v3a + weak sparse high-confidence oracle auxiliary loss
```
