# Task2 Stage2AB Verification Package

Date: 2026-06-14

## Decision

Stage2AB is now packaged as a machine-checkable internal prediction artifact, not just a one-off score sweep.

## Artifacts

- Sweep/export script: `scripts/task2/sweep_stage2v_domain_policy.py`
- Verifier: `scripts/task2/verify_stage2ab_domain_policy.py`
- Tests:
  - `tests/test_task2_stage2v_domain_policy.py`
  - `tests/test_task2_stage2ab_verify.py`
- Manifest: `outputs/task2/stage2v_domain_policy/stage2ab_manifest.json`
- Verification result: `outputs/task2/stage2v_domain_policy/stage2ab_verification.json`
- Prediction CSV directory:
  `outputs/task2/stage2v_domain_policy/yolo_only_domain_policy_predictions`

## Verification

Command:

```bash
python3 scripts/task2/verify_stage2ab_domain_policy.py \
  --artifact-dir outputs/task2/stage2v_domain_policy
```

Result:

```text
status: ok
verification: outputs/task2/stage2v_domain_policy/stage2ab_verification.json
```

The verifier checks:

- required split CSVs are present;
- CSV fields match the clean internal schema;
- validation/leakage fields are absent;
- policy score modes match the frozen Stage2AB domain/class policy;
- manifest row counts match CSV row counts;
- frozen `valid_combined` mAP50/mAP50-95 match the current public-validation best.

## Current Standing

Stage2AB remains the public-validation best:

- `valid_combined` mAP50: 0.21128164745000094
- `valid_combined` mAP50-95: 0.050628135044670744

Stage2V remains the conservative fallback:

- `valid_combined` mAP50: 0.20096803092334323
- `valid_combined` mAP50-95: 0.04924044637514719

