# Task2 Stage2AB GT-Free Export Result

Date: 2026-06-14

## Decision

Use Stage2AB as the current Task2 public-validation champion, but package it through a GT-free frozen-policy exporter rather than the original validation sweep script.

The new exporter is:

```bash
python3 scripts/task2/export_stage2ab_domain_policy_predictions.py \
  --run-dir outputs/task2/stage2u_quality_ranker/stage2u_warm_stage2o_fullvalid_yolo_only_eval \
  --output-dir outputs/task2/stage2v_domain_policy/stage2ab_gt_free_reexport
```

For official validation or hidden test, replace `--run-dir` with the directory containing the inference candidate rows and verifier prediction rows:

```bash
python3 scripts/task2/export_stage2ab_domain_policy_predictions.py \
  --run-dir <official_eval_stage2u_run_dir> \
  --output-dir <official_eval_stage2ab_export_dir>
```

The run directory must contain:

- `{split}_candidates_used.csv`
- `{split}_eval_prediction_rows.csv`

The candidate CSV must include clean inference metadata including `sample_id`, box coordinates, `source`, `source_rank`, and `domain`. Missing/unknown domains fail fast by default. `--fallback-domain phantom`, `--fallback-domain animal`, or `--fallback-domain human` may be used only when the official metadata genuinely lacks a domain field and the intended fallback has been explicitly chosen.

If official data includes `human` rows, the exporter preserves `domain=human` in the output and maps the score-policy lookup to `animal` by default:

```bash
python3 scripts/task2/export_stage2ab_domain_policy_predictions.py \
  --run-dir <official_eval_stage2u_run_dir> \
  --output-dir <official_eval_stage2ab_export_dir> \
  --human-policy-domain animal
```

This is explicit and configurable. `--human-policy-domain phantom` can be used for an ablation, but should not silently replace the default without validation evidence.

## Frozen Policy

Stage2AB applies the following score columns:

| Class | Domain | Score mode |
| --- | --- | --- |
| 0 / normal | phantom | `prob_iou75_source_rank_decay_roi` |
| 0 / normal | animal | `prob_iou75_source_rank_decay_roi` |
| 1 / collision | phantom | `rank_decay_roi` |
| 1 / collision | animal | `prob_iou75_roi` |

For `human`, the default export uses the same score modes as `animal` but keeps policy names like `class1_human_as_animal` so the mapping remains auditable.

## Verification

The GT-free export was run on the current public-validation Stage2U YOLO-only evaluation directory and compared against the existing Stage2AB champion CSVs.

Result:

- `valid_combined`: exact CSV match, 40,970 rows
- `valid_phantom`: exact CSV match, 39,338 rows
- `valid_animal`: exact CSV match, 1,632 rows

The re-export also passed the frozen Stage2AB verifier:

```bash
python3 scripts/task2/verify_stage2ab_domain_policy.py \
  --artifact-dir outputs/task2/stage2v_domain_policy \
  --prediction-dir outputs/task2/stage2v_domain_policy/stage2ab_gt_free_reexport \
  --output-json outputs/task2/stage2v_domain_policy/stage2ab_gt_free_reexport/stage2ab_reexport_verification.json
```

Public-validation metrics remain:

| Split | mAP50 | mAP50-95 | Rows |
| --- | ---: | ---: | ---: |
| valid_combined | 0.21128164745000094 | 0.050628135044670744 | 40,970 |
| valid_phantom | 0.10462265654206386 | 0.04301139057740514 | 39,338 |
| valid_animal | 0.4997355309073923 | 0.050008473801353336 | 1,632 |

## Tests

```bash
PYTHONDONTWRITEBYTECODE=1 python3 -m py_compile \
  scripts/task2/export_stage2ab_domain_policy_predictions.py \
  scripts/task2/verify_stage2ab_domain_policy.py \
  scripts/task2/export_clean_predictions.py \
  scripts/task2/sweep_stage2v_domain_policy.py

PYTHONDONTWRITEBYTECODE=1 /home/mingzhang/miniconda3/envs/cathaction-task1/bin/python -m pytest \
  tests/test_task2_stage2ab_gt_free_export.py \
  tests/test_task2_stage2ab_verify.py \
  tests/test_task2_stage2v_domain_policy.py \
  tests/test_task2_stage2v_champion_verify.py \
  tests/test_task2_export_clean_predictions.py
```

Result: 17 passed.

## Interpretation

This does not improve the model numerically; it hardens the current best Task2 result into a submission-safe export path. The main practical gain is that the current champion no longer depends on validation labels at export time.
