# Task2 Stage2AK Bottleneck Attribution Plan

Date: 2026-06-14

## Goal

Determine the dominant remaining bottleneck of the current Task2 champion family, especially on `valid_phantom`, before launching another training run.

The current public-validation champion family is:

- `stage2ae`: balanced/default candidate;
- `stage2ai`: high `mAP50-95` candidate with lower `mAP50`.

The next optimization should be chosen from evidence:

- candidate recall failure;
- box geometry/calibration failure;
- ranking/scoring failure;
- class/domain-specific imbalance.

## Steps

1. Inspect the current candidate and prediction schemas used by Stage2U/AB/AE/AI.
2. Implement a read-only diagnostic script that compares current predictions against oracle variants:
   - current score ranking;
   - oracle score by best IoU to GT;
   - oracle top-1/top-k candidate selection;
   - per-domain and per-class recall at IoU 0.50 and stricter thresholds.
3. Run the diagnostic on public validation, with emphasis on `valid_phantom`.
4. Record the result and next optimization decision.

## Acceptance Criteria

- The diagnostic does not train or tune on hidden test.
- It writes machine-readable JSON/CSV outputs.
- It reports per-domain and per-class evidence.
- It identifies whether the next high-value work is candidate generation, box refinement, or ranking/verifier training.

