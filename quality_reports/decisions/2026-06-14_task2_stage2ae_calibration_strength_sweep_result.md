# Task2 Stage2AE Calibration Strength Sweep Result

Date: 2026-06-14

## Question

Can the animal-only train-fitted box correction from Stage2AD be strengthened to further improve mAP50-95 without hurting mAP50?

## Setup

Base run:

`outputs/task2/stage2u_quality_ranker/stage2u_warm_stage2o_fullvalid_yolo_only_eval`

Evaluation script:

`scripts/task2/evaluate_stage2ac_box_calibration.py`

Sweep command pattern:

```bash
python3 scripts/task2/evaluate_stage2ac_box_calibration.py \
  --run-dir outputs/task2/stage2u_quality_ranker/stage2u_warm_stage2o_fullvalid_yolo_only_eval \
  --disable-class-specific \
  --min-fit-iou 0.30 \
  --calibrate-domain animal \
  --transform-strength <strength> \
  --eval-split valid_combined \
  --output-json outputs/task2/stage2ae_calibration_strength_sweep/animal_only_strength_<strength>_valid_combined.json
```

The score policy is still the frozen Stage2AB domain-aware policy. Only animal box coordinates are transformed; phantom uses identity transforms.

## Strength Sweep

| Strength | valid_combined mAP50 | valid_combined mAP50-95 | Delta mAP50-95 vs Stage2AB |
| ---: | ---: | ---: | ---: |
| 0.25 | 0.21128164745000094 | 0.05103238943927888 | +0.00040425439460813756 |
| 0.50 | 0.21128164745000094 | 0.051397733364284694 | +0.0007695983196139497 |
| 0.75 | 0.21128164745000094 | 0.05201474867833224 | +0.0013866136336614932 |
| 1.00 / Stage2AD | 0.21128164745000094 | 0.06251332119288261 | +0.011885186148211865 |
| 1.25 / Stage2AE | 0.21128164745000094 | 0.06327005557430047 | +0.01264192052962973 |
| 1.50 | 0.21128164745000094 | 0.0521712158911384 | +0.0015430808464676551 |

Best public-validation setting: `--transform-strength 1.25`.

## Final Stage2AE Metrics

| Method | valid_combined mAP50 | valid_combined mAP50-95 |
| --- | ---: | ---: |
| Stage2AB domain policy | 0.21128164745000094 | 0.050628135044670744 |
| Stage2AD animal-only strength 1.00 | 0.21128164745000094 | 0.06251332119288261 |
| Stage2AE animal-only strength 1.25 | 0.21128164745000094 | 0.06327005557430047 |

Split metrics from clean export:

| Split | mAP50 | mAP50-95 | class0 mAP50-95 | class1 mAP50-95 |
| --- | ---: | ---: | ---: | ---: |
| valid_combined | 0.21128164745000094 | 0.06327005557430047 | 0.06750764952433767 | 0.05903246162426329 |
| valid_phantom | 0.10462265654206386 | 0.04301139057740514 | 0.06980514469398438 | 0.016217636460825885 |
| valid_animal | 0.4997355309073923 | 0.1431702793939404 | 0.0 | 0.28634055878788073 |

## Clean Export

Stage2AE export:

```bash
python3 scripts/task2/export_stage2ab_domain_policy_predictions.py \
  --run-dir outputs/task2/stage2u_quality_ranker/stage2u_warm_stage2o_fullvalid_yolo_only_eval \
  --output-dir outputs/task2/stage2ae_calibration_strength_sweep/stage2ae_animal_strength125_export \
  --box-transform-json outputs/task2/stage2ae_calibration_strength_sweep/animal_only_strength_1.25_valid_combined.json
```

Output:

`outputs/task2/stage2ae_calibration_strength_sweep/stage2ae_animal_strength125_export`

Verification:

```bash
python3 scripts/task2/verify_stage2ab_domain_policy.py \
  --artifact-dir outputs/task2/stage2v_domain_policy \
  --prediction-dir outputs/task2/stage2ae_calibration_strength_sweep/stage2ae_animal_strength125_export \
  --output-json outputs/task2/stage2ae_calibration_strength_sweep/stage2ae_animal_strength125_export/schema_policy_verification.json \
  --skip-frozen-metric-check
```

Result: ok.

Recomputed metrics from clean CSV:

`outputs/task2/stage2ae_calibration_strength_sweep/stage2ae_animal_strength125_export/recomputed_metrics.json`

## Decision

Promote Stage2AE as the current public-validation Task2 champion.

Rationale:

- It preserves Stage2AB and Stage2AD valid_combined mAP50 exactly.
- It improves valid_combined mAP50-95 over Stage2AD by +0.00075673438141786.
- It improves valid_combined mAP50-95 over Stage2AB by +0.01264192052962973.
- Phantom remains unchanged because phantom uses identity transforms.

Risk:

- Strength 1.25 is public-validation tuned, so it carries more overfitting risk than Stage2AD strength 1.0.
- For official validation, evaluate Stage2AB, Stage2AD, and Stage2AE side by side if labels are available. If official validation is unavailable or domain metadata is uncertain, keep Stage2AB/Stage2AD as fallbacks.

