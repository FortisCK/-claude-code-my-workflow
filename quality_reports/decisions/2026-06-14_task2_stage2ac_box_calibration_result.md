# Task2 Stage2AC Train-Fitted Box Calibration Result

Date: 2026-06-14

## Question

Can we improve the current Stage2AB result by correcting systematic candidate-box geometry using train-side GT only?

## Setup

Base run:

`outputs/task2/stage2u_quality_ranker/stage2u_warm_stage2o_fullvalid_yolo_only_eval`

Diagnostic script:

`scripts/task2/evaluate_stage2ac_box_calibration.py`

GT-free exporter:

`scripts/task2/export_stage2ab_domain_policy_predictions.py`

The calibration is fitted from `train_candidates_used.csv` and then applied to validation candidates while keeping the frozen Stage2AB domain-aware score policy unchanged.

Best tested setting:

```bash
python3 scripts/task2/evaluate_stage2ac_box_calibration.py \
  --run-dir outputs/task2/stage2u_quality_ranker/stage2u_warm_stage2o_fullvalid_yolo_only_eval \
  --disable-class-specific \
  --min-fit-iou 0.30 \
  --output-json outputs/task2/stage2ac_box_calibration/yolo_only_trainfit_domain_shared_iou30_valid_combined.json
```

This fits one transform per domain, shared across classes.

## Fitted Transforms

| Domain | n_fit | dx_center | dy_center | log_w | log_h |
| --- | ---: | ---: | ---: | ---: | ---: |
| animal | 1,099 | -0.0625 | 0.14285714285714285 | 0.09819413229448364 | 0.22314355131420976 |
| phantom | 7,980 | -0.0015988298880296556 | -0.0014246962952523417 | -0.020521861930100818 | -0.016629539449565422 |

Interpretation:

- Animal boxes are systematically shifted and undersized relative to GT.
- Phantom boxes need only a tiny correction, so the transform mostly affects animal high-IoU thresholds.

## Result

| Method | valid_combined mAP50 | valid_combined mAP50-95 |
| --- | ---: | ---: |
| Stage2AB | 0.21128164745000094 | 0.050628135044670744 |
| Stage2AC box-calibrated | 0.2090715934287487 | 0.06212640338404716 |
| Delta | -0.002210054021252239 | +0.011498268339376415 |

Split results:

| Split | Stage2AB mAP50 | Stage2AB mAP50-95 | Stage2AC mAP50 | Stage2AC mAP50-95 | Delta mAP50-95 |
| --- | ---: | ---: | ---: | ---: | ---: |
| valid_combined | 0.21128164745000094 | 0.050628135044670744 | 0.2090715934287487 | 0.06212640338404716 | +0.011498268339376415 |
| valid_phantom | 0.10462265654206386 | 0.04301139057740514 | 0.1028093335961711 | 0.04270632200614751 | -0.0003050685712576301 |
| valid_animal | 0.4997355309073923 | 0.050008473801353336 | 0.4997355309073923 | 0.1269937435713481 | +0.07698526976999476 |

Class-level combined mAP50-95:

| Class | Stage2AB | Stage2AC |
| --- | ---: | ---: |
| class 0 / normal | 0.06750764952433767 | 0.06681737667511815 |
| class 1 / collision | 0.03374862056500379 | 0.05743543009297615 |

## Clean Export

Stage2AC predictions were exported with the same clean internal schema as Stage2AB:

```bash
python3 scripts/task2/export_stage2ab_domain_policy_predictions.py \
  --run-dir outputs/task2/stage2u_quality_ranker/stage2u_warm_stage2o_fullvalid_yolo_only_eval \
  --output-dir outputs/task2/stage2ac_box_calibration/stage2ac_domain_shared_export \
  --box-transform-json outputs/task2/stage2ac_box_calibration/yolo_only_trainfit_domain_shared_iou30_valid_combined.json
```

Schema/policy verification passed:

```bash
python3 scripts/task2/verify_stage2ab_domain_policy.py \
  --artifact-dir outputs/task2/stage2v_domain_policy \
  --prediction-dir outputs/task2/stage2ac_box_calibration/stage2ac_domain_shared_export \
  --output-json outputs/task2/stage2ac_box_calibration/stage2ac_domain_shared_export/schema_policy_verification.json \
  --skip-frozen-metric-check
```

## Decision

Promote Stage2AC as the current **mAP50-95 / COCO-mAP champion** because it improves valid_combined mAP50-95 by +0.0115 absolute.

Keep Stage2AB as the **mAP50 fallback** because Stage2AC slightly lowers valid_combined mAP50 by -0.0022.

This is a useful result because it targets the observed failure mode directly: boxes ranked around the correct location but too imprecise at higher IoU thresholds, especially animal collision frames.

## Caveat

The transform is learned from the current train split and evaluated on public validation. It is less risky than validation-tuned box correction, but it is still a deterministic calibration that assumes animal/phantom box bias transfers to official validation/hidden test. It should be rechecked when the official validation set is released.

