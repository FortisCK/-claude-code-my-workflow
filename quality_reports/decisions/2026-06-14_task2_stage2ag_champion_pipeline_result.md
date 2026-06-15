# Task2 Stage2AG Champion Pipeline Result

Date: 2026-06-14

## Decision

Use `scripts/task2/run_stage2_champion_pipeline.py` as the standard Task2 export/evaluation wrapper for the current champion family.

The pipeline exports:

- `stage2ab`: no box calibration fallback;
- `stage2ad`: animal-only train-fitted calibration strength 1.0 fallback;
- `stage2ae`: balanced public-validation champion, animal-only calibration strength 1.25;
- `stage2ai`: high mAP50-95 candidate, Stage2AE boxes plus calibrated score policy.

## Public-Validation Command

```bash
python3 scripts/task2/run_stage2_champion_pipeline.py \
  --run-dir outputs/task2/stage2u_quality_ranker/stage2u_warm_stage2o_fullvalid_yolo_only_eval \
  --output-dir outputs/task2/stage2aj_champion_pipeline_with_ai/public_valid_ab_ad_ae_ai
```

Output:

`outputs/task2/stage2aj_champion_pipeline_with_ai/public_valid_ab_ad_ae_ai`

## Official Validation / Hidden-Test Use

Once official candidate/verifier rows exist:

```bash
python3 scripts/task2/run_stage2_champion_pipeline.py \
  --run-dir <official_eval_stage2u_run_dir> \
  --output-dir <official_eval_champion_pipeline_dir>
```

The run directory must contain:

- `{split}_candidates_used.csv`
- `{split}_eval_prediction_rows.csv`

If candidate rows include GT fields, metrics are recomputed. If they do not, the pipeline still exports predictions and records `metrics_available=false`.

## Public-Validation Results

| Variant | valid_combined mAP50 | valid_combined mAP50-95 | Rows |
| --- | ---: | ---: | ---: |
| stage2ab | 0.21128164745000094 | 0.050628135044670744 | 40,970 |
| stage2ad | 0.21128164745000094 | 0.06251332119288261 | 40,970 |
| stage2ae | 0.21128164745000094 | 0.06327005557430047 | 40,970 |
| stage2ai | 0.18439741617609534 | 0.08394956471533466 | 40,970 |

Split metrics:

| Variant | Split | mAP50 | mAP50-95 |
| --- | --- | ---: | ---: |
| stage2ab | valid_phantom | 0.10462265654206386 | 0.04301139057740514 |
| stage2ab | valid_animal | 0.4997355309073923 | 0.050008473801353336 |
| stage2ad | valid_phantom | 0.10462265654206386 | 0.04301139057740514 |
| stage2ad | valid_animal | 0.4997355309073923 | 0.1269937435713481 |
| stage2ae | valid_phantom | 0.10462265654206386 | 0.04301139057740514 |
| stage2ae | valid_animal | 0.4997355309073923 | 0.1431702793939404 |
| stage2ai | valid_phantom | 0.09742357618031064 | 0.040369930409670254 |
| stage2ai | valid_animal | 0.48785015325856906 | 0.23876800871618054 |

## Verification

The pipeline-generated CSVs exactly match the previous hand-exported CSVs for all variants and splits checked before Stage2AI was added:

- `stage2ab`: `valid_combined`, `valid_phantom`, `valid_animal` exact match.
- `stage2ad`: `valid_combined`, `valid_phantom`, `valid_animal` exact match.
- `stage2ae`: `valid_combined`, `valid_phantom`, `valid_animal` exact match.

After Stage2AI was added, the updated four-variant pipeline reproduced the Stage2AI sweep/re-export metrics as well.

Tests:

```bash
PYTHONDONTWRITEBYTECODE=1 python3 -m py_compile \
  scripts/task2/run_stage2_champion_pipeline.py \
  scripts/task2/export_stage2ab_domain_policy_predictions.py \
  scripts/task2/evaluate_stage2ac_box_calibration.py \
  scripts/task2/sweep_stage2ae_calibrated_policy.py

PYTHONDONTWRITEBYTECODE=1 /home/mingzhang/miniconda3/envs/cathaction-task1/bin/python -m pytest \
  tests/test_task2_champion_pipeline.py \
  tests/test_task2_stage2ab_gt_free_export.py \
  tests/test_task2_stage2ae_policy_sweep.py
```

Result after Stage2AI integration: 15 passed for the focused pipeline/export tests. The latest broader Task2 export/metric regression passed with 29 tests.

## Interpretation

Stage2AG/Stage2AJ does not introduce a new trained model; it makes the current champion/fallback family reproducible and ready for official validation. This reduces operational risk: official candidate rows can be processed through one stable command, producing AB/AD/AE/AI side-by-side.

Metric selection rule:

- If official evaluation prioritizes AP50-like behavior, use Stage2AE.
- If official evaluation prioritizes COCO-style mAP50-95 or high-IoU mAP, evaluate Stage2AI seriously because it is much stronger on mAP50-95.
- Keep Stage2AD as the lower-risk calibration fallback and Stage2AB as the no-calibration fallback.
