# Task2 Stage2AI Calibrated Policy Sweep Plan

Date: 2026-06-14

## Goal

After Stage2AE changed box geometry, re-sweep domain-aware score policies on calibrated boxes. The Stage2AB score policy was selected before box calibration, so it may no longer be optimal for mAP50-95.

## Steps

1. Add a calibrated score-policy sweep script that applies optional box transforms during AP computation.
2. Run the sweep with the Stage2AE animal-only strength 1.25 transform.
3. Export clean predictions for the selected score policy.
4. Add GT-free custom policy export support so the selected policy can be reproduced without validation GT.
5. Decide whether Stage2AI should replace Stage2AE or remain a metric-specific alternative.

## Acceptance Criteria

- Exported CSVs use the clean internal schema.
- GT-free re-export matches the sweep export exactly.
- Decision explicitly accounts for both mAP50 and mAP50-95, because the policy may trade one for the other.

