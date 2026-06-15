# Task2 Stage2Y Result: Source-Aware Residual Ranker

Date: 2026-06-14
Status: completed diagnostic

## Question

Stage2X improved candidate oracle recall but failed because the image-only ranker could not score the new source. Can a source-aware ranker use proposal-source metadata to turn the five-source candidate pool into higher Task2 mAP?

## Implementation

Added an optional source-aware residual head to:

`scripts/task2/train_stage2u_quality_ranker.py`

The new mode is enabled with:

```bash
--source-aware
```

Design:

- keep the original ROI image model;
- add a residual metadata head:
  - `output = image_model(roi) + metadata_head(source_metadata)`;
- zero-initialize the final metadata layer, so the model starts exactly as the warm-started image model;
- load old Stage2U checkpoints into the wrapped `image_model.*` prefix automatically.

Inference-safe metadata only:

- source one-hot;
- source confidence;
- source rank decay;
- source priority transform;
- normalized candidate box center, size, area, and aspect.

The model does **not** use `gt_class` or `domain` as features.

## Smoke Verification

Smoke run:

`outputs/task2/stage2u_quality_ranker/smoke_stage2y_source_aware_limit96_valid16`

Result:

- old Stage2U checkpoint warm-start loaded `182` tensors;
- source-aware residual path ran on GPU;
- `--log-interval` progress output worked;
- checkpoint and metrics were written.

## Pilot

Run:

`outputs/task2/stage2u_quality_ranker/stage2y_source_aware_train60k_valid256_e2`

Log:

`quality_reports/logs/task2_stage2y_source_aware_train60k_valid256_e2.log`

Configuration:

- init checkpoint: `outputs/task2/stage2u_quality_ranker/stage2u_warm_stage2o_fulltrain_valid512_e6/checkpoints/best.pt`;
- train CSV: `outputs/task2/stage2o_candidate_pool/stage2x_train_subset_yolo_geometry_stage2x_top50/valid_combined_candidates.csv`;
- valid CSV: five-source valid pool;
- source-aware: on;
- epochs: 2;
- effective train candidates after stratified limit: 42,755;
- sampled valid: 256 combined / 256 phantom / all 107 animal;
- primary metric: `valid_combined/blend_rank_decay_roi/mAP50-95`.

Sampled-validation result:

| Epoch | valid_combined best mAP50 | valid_combined best mAP50-95 | primary `blend_rank_decay_roi` mAP50-95 |
| ---: | ---: | ---: | ---: |
| 1 | 0.5282 | 0.0622 | 0.0622 |
| 2 | 0.5599 | 0.0745 | 0.0738 |

This passed the sampled-validation gate and justified a full `valid_combined` eval.

## Full Valid Combined Eval

Run:

`outputs/task2/stage2u_quality_ranker/stage2y_source_aware_train60k_fullvalid_combined_eval`

Log:

`quality_reports/logs/task2_stage2y_source_aware_train60k_fullvalid_combined_eval.log`

Full `valid_combined` result:

| Mode | mAP50 | mAP50-95 |
| --- | ---: | ---: |
| `prob_iou75_rank_decay_roi` | 0.1557 | 0.0321 |
| `blend_rank_decay_roi` | 0.1388 | 0.0319 |
| `pred_iou_rank_decay_roi` | 0.1344 | 0.0313 |

For comparison:

| Policy | valid_combined mAP50 | valid_combined mAP50-95 |
| --- | ---: | ---: |
| Stage2Y source-aware full eval | 0.1557 | 0.0321 |
| Stage2V champion | 0.2010 | 0.0492 |

## Class-Aware Fusion Check

To test whether Stage2Y was useful only for class1, I fused:

- class0: Stage2V YOLO-only score `prob_iou75_source_rank_decay_roi`;
- class1: Stage2Y score `prob_iou75_rank_decay_roi`.

Output:

`outputs/task2/stage2u_quality_ranker/stage2y_class0_yolo_class1_sourceaware_fusion_eval/eval_metrics.json`

Result:

| Policy | valid_combined mAP50 | valid_combined mAP50-95 |
| --- | ---: | ---: |
| class0 YOLO + class1 Stage2Y | 0.1753 | 0.0439 |
| Stage2V champion | 0.2010 | 0.0492 |

Stage2Y does not improve the class-aware champion either.

## Decision

Do not promote Stage2Y.

The source-aware residual implementation is technically sound and useful for future experiments, but the current training setup still does not convert five-source oracle recall into AP. The sampled validation result was again too optimistic; full valid combined is the decisive result.

Current Task2 champion remains:

`outputs/task2/stage2u_quality_ranker/stage2v_class0_yolo_iou75_class1_yolo_roi_export`

with:

- valid_combined mAP50: `0.20097`;
- valid_combined mAP50-95: `0.04924`;
- valid_phantom mAP50-95: `0.04086`;
- valid_animal mAP50-95: `0.05000`.

## Implication

The main remaining issue is not just "ranker needs source metadata." The weak point is the training/evaluation formulation:

- candidate sources differ strongly between train and full validation;
- small validation truncation is misleading;
- adding many low-quality candidates can lower AP even when oracle recall rises;
- class1 high-IoU candidates exist, but confidence ordering is poor.

Next work should either:

1. harden Stage2V for submission and reporting; or
2. move to a true per-frame/listwise selector trained to choose among candidates within the same frame, not a per-candidate classifier scored independently.

Stage2Y code should remain available, but it is not part of the current submission-safe pipeline.
