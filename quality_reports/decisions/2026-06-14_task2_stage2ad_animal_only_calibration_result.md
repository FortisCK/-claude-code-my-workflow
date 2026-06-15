# Task2 Stage2AD Animal-Only Box Calibration Result

Date: 2026-06-14

## Question

Can we keep the Stage2AC animal high-IoU gain while avoiding the small phantom degradation?

## Setup

Base run:

`outputs/task2/stage2u_quality_ranker/stage2u_warm_stage2o_fullvalid_yolo_only_eval`

Evaluation script:

`scripts/task2/evaluate_stage2ac_box_calibration.py`

Best command:

```bash
python3 scripts/task2/evaluate_stage2ac_box_calibration.py \
  --run-dir outputs/task2/stage2u_quality_ranker/stage2u_warm_stage2o_fullvalid_yolo_only_eval \
  --disable-class-specific \
  --min-fit-iou 0.30 \
  --calibrate-domain animal \
  --eval-split valid_combined \
  --output-json outputs/task2/stage2ad_animal_only_box_calibration/yolo_only_trainfit_animal_only_iou30_valid_combined.json
```

This uses the same frozen Stage2AB score policy, but applies box calibration only to animal rows. Phantom rows use identity transforms.

## Result

| Method | valid_combined mAP50 | valid_combined mAP50-95 |
| --- | ---: | ---: |
| Stage2AB domain policy | 0.21128164745000094 | 0.050628135044670744 |
| Stage2AC all-domain box calibration | 0.2090715934287487 | 0.06212640338404716 |
| Stage2AD animal-only calibration | 0.21128164745000094 | 0.06251332119288261 |

Stage2AD vs Stage2AB:

- mAP50: +0.0
- mAP50-95: +0.011885186148211865

Stage2AD vs Stage2AC:

- mAP50: +0.002210054021252239
- mAP50-95: +0.00038691780883545

Split results:

| Split | Stage2AD mAP50 | Stage2AD mAP50-95 | Note |
| --- | ---: | ---: | --- |
| valid_combined | 0.21128164745000094 | 0.06251332119288261 | New best combined mAP50-95 |
| valid_phantom | 0.10462265654206386 | 0.04301139057740514 | Matches Stage2AB |
| valid_animal | 0.4997355309073923 | 0.1269937435713481 | Keeps Stage2AC animal gain |

Class-level valid_combined mAP50-95:

| Class | Stage2AD |
| --- | ---: |
| class 0 / normal | 0.06750764952433767 |
| class 1 / collision | 0.05751899286142753 |

## Clean Export

Stage2AD clean export:

```bash
python3 scripts/task2/export_stage2ab_domain_policy_predictions.py \
  --run-dir outputs/task2/stage2u_quality_ranker/stage2u_warm_stage2o_fullvalid_yolo_only_eval \
  --output-dir outputs/task2/stage2ad_animal_only_box_calibration/stage2ad_animal_only_export \
  --box-transform-json outputs/task2/stage2ad_animal_only_box_calibration/yolo_only_trainfit_animal_only_iou30_valid_combined.json
```

Output:

`outputs/task2/stage2ad_animal_only_box_calibration/stage2ad_animal_only_export`

Schema/policy verification:

```bash
python3 scripts/task2/verify_stage2ab_domain_policy.py \
  --artifact-dir outputs/task2/stage2v_domain_policy \
  --prediction-dir outputs/task2/stage2ad_animal_only_box_calibration/stage2ad_animal_only_export \
  --output-json outputs/task2/stage2ad_animal_only_box_calibration/stage2ad_animal_only_export/schema_policy_verification.json \
  --skip-frozen-metric-check
```

Result: ok.

## Decision

Promote Stage2AD as the current Task2 champion for public validation.

Rationale:

- It preserves Stage2AB mAP50 exactly on valid_combined.
- It improves valid_combined mAP50-95 by +0.0119 absolute.
- It avoids Stage2AC's phantom degradation by keeping phantom boxes unchanged.
- The transform is fitted from train-side GT only, so the export path remains usable for official validation/hidden test once candidate rows and verifier scores are produced.

Keep Stage2AB as a no-calibration fallback for official validation if domain metadata is missing or if official validation contradicts the animal calibration assumption.

