# Task2 Stage2AO Champion Pipeline With Stage2AN Result

Date: 2026-06-14

## Question

Can the champion export pipeline package the newly tested Stage2AN global score
calibration variant together with the existing AB/AD/AE/AI candidates?

## Implementation

Updated:

- `scripts/task2/run_stage2_champion_pipeline.py`
- `tests/test_task2_champion_pipeline.py`

The pipeline now exports five variants:

- `stage2ab`: domain-aware score policy, no box calibration;
- `stage2ad`: Stage2AB plus animal-only box calibration strength 1.0;
- `stage2ae`: Stage2AB plus animal-only box calibration strength 1.25;
- `stage2ai`: Stage2AE plus calibrated high-IoU score policy;
- `stage2an`: Stage2AE plus class/domain global score scaling.

Stage2AN is implemented as an inference-side clean-CSV score post-processing
step. It reads `best_policy` from:

`outputs/task2/stage2an_global_score_scale/stage2ae_fast_global_scale_mAP5095.json`

and scales scores by class/domain:

| Scale | Value |
| --- | ---: |
| class0 phantom | 1.5 |
| class0 animal | 0.5 |
| class1 phantom | 0.5 |
| class1 animal | 1.5 |

No boxes, class ids, domains, or schema fields are changed.

## Command

```bash
/home/mingzhang/miniconda3/envs/cathaction-task1/bin/python \
  scripts/task2/run_stage2_champion_pipeline.py \
  --run-dir outputs/task2/stage2u_quality_ranker/stage2u_warm_stage2o_fullvalid_yolo_only_eval \
  --output-dir outputs/task2/stage2ao_champion_pipeline_with_an/public_valid_ab_ad_ae_ai_an
```

Output:

`outputs/task2/stage2ao_champion_pipeline_with_an/public_valid_ab_ad_ae_ai_an`

## Public-Validation Metrics

| Variant | valid_combined mAP50 | valid_combined mAP50-95 | Role |
| --- | ---: | ---: | --- |
| Stage2AB | 0.21128164745000094 | 0.050628135044670744 | uncalibrated fallback |
| Stage2AD | 0.21128164745000094 | 0.06251332119288261 | calibrated fallback |
| Stage2AE | 0.21128164745000094 | 0.06327005557430047 | default balanced candidate |
| Stage2AI | 0.18439741617609534 | 0.08394956471533466 | high-IoU candidate |
| Stage2AN | 0.20037493968130657 | 0.06478975652205522 | mild global-score calibration |

Stage2AN exactly reproduces the earlier independent Stage2AN sweep result in
the unified export pipeline.

## Interpretation

Stage2AN provides a very small mAP50-95 gain over Stage2AE:

- `+0.00151970094775475` mAP50-95;
- `-0.01090670776869437` mAP50.

This is not enough to replace Stage2AE as the main candidate. The main value of
Stage2AO is operational: all current submission candidates are now exported by
one reproducible pipeline, with clean CSV schema validation and metric
recomputation.

## Decision

Keep the Task2 candidate ordering:

1. Stage2AE as the default balanced candidate.
2. Stage2AI if the official evaluator clearly rewards COCO-style high-IoU mAP.
3. Stage2AN only as a conservative high-IoU calibration backup.
4. Stage2AD/AB as fallbacks and ablation references.

Do not spend more time tuning global scalar score calibration unless official
validation indicates that high-IoU mAP is the dominant ranking criterion.

## Verification

Passed:

```bash
PYTHONDONTWRITEBYTECODE=1 python3 -m py_compile \
  scripts/task2/run_stage2_champion_pipeline.py \
  scripts/task2/export_stage2ab_domain_policy_predictions.py \
  scripts/task2/sweep_stage2an_fast_global_score_scale.py

/home/mingzhang/miniconda3/envs/cathaction-task1/bin/python -m pytest \
  tests/test_task2_champion_pipeline.py \
  tests/test_task2_stage2ab_gt_free_export.py \
  tests/test_task2_stage2ae_policy_sweep.py
```

After saving this decision file, the full Task2 regression suite also passed:

```bash
/home/mingzhang/miniconda3/envs/cathaction-task1/bin/python -m pytest tests/test_task2_*.py
```

Result: 84 passed.
