# Plan: Outputs Cleanup

**Date:** 2026-05-21
**Status:** APPROVED
**Scope:** Remove obsolete Task 1 experiment outputs while preserving active and reference runs

## Current State

`outputs/task1` is about 237MB and contains smoke, pilot, full baseline, and
current overnight experiment directories.

Current active training process:

- config: `configs/task1/monai_unet_cldice_domain_balanced_640.yaml`
- output: `outputs/task1/monai_unet_cldice_domain_balanced_640`
- main PID observed: `647128`

Do not delete the active output directory while the process is running.

## Keep

- `outputs/task1/monai_unet_cldice_domain_balanced_640`
  - active Stage 1 overnight run
- `outputs/task1/monai_unet_dicece_full_512_overnight`
  - current best reference checkpoint and full domain-wise eval
- `outputs/task1/baseline_unet_multiclass_mild_weighted_full_512`
  - old non-MONAI full baseline reference; small enough to keep unless a stricter
    cleanup is requested

## Delete Candidate

These are smoke/pilot/temporary outputs and can be removed:

- `outputs/task1/baseline_unet_smoke`
- `outputs/task1/baseline_unet_multiclass_smoke`
- `outputs/task1/baseline_unet_multiclass_gpu_pilot`
- `outputs/task1/baseline_unet_multiclass_weighted_gpu_pilot`
- `outputs/task1/baseline_unet_multiclass_mild_weighted_gpu_pilot`
- `outputs/task1/baseline_unet_multiclass_mild_weighted_full_512_smoke`
- `outputs/task1/baseline_unet_multiclass_mild_weighted_full_512_bs64_smoke`
- `outputs/task1/monai_unet_dicece_smoke`
- `outputs/task1/monai_unet_cldice_domain_balanced_640_smoke`
- `outputs/task1/monai_unet_cldice_domain_balanced_640_launcher_smoke`

Estimated reclaim: about 160MB.

## Verification

After cleanup:

- confirm active training process still exists;
- confirm active output directory still exists;
- confirm reference MONAI overnight output still exists;
- re-run `du -h --max-depth=2 outputs`.

Approved by user request on 2026-05-21 to clean previous outputs.
