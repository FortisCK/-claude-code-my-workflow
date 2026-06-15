# Task2 Stage2X Result: Class1 Center Localizer and Five-Source Ranker

Date: 2026-06-13
Status: completed diagnostic

## Question

Can a class1-specific temporal localizer improve the collision-box candidate pool enough to raise Task2 mAP over the current Stage2V class-aware fusion champion?

## Setup

Stage2X trained a class1-only sequence localizer on the Stage2L train subset:

- model: `smp_fpn` with `tu-convnext_tiny`;
- input: 5-frame temporal stack, 384 px;
- loss: CenterNet-style heatmap plus strong coordinate loss;
- split: `configs/task2/splits_stage2x_class1/train_v0_v1_val_v2_class1_labels.txt`;
- run: `outputs/task2/sequence_tip_localizer/stage2x_class1_tip384_convnext_centernet_coord20_e8`;
- log: `quality_reports/logs/task2_stage2x_class1_tip384_convnext_centernet_coord20_e8.log`.

The best checkpoint was exported with dense templates into:

`outputs/task2/sequence_tip_proposals/stage2x_class1_centernet_dense_templates_top30`

## Candidate Oracle

The five-source candidate union combined:

- `yolo_stage2l`;
- `task1_geometry_rect`;
- `sequence_tip`;
- `stage2w_dense`;
- `stage2x_class1`.

Oracle output:

`outputs/task2/proposal_union_oracle/yolo_geometry_oldseq_stage2w_dense_stage2x_class1_top50`

Key oracle metrics:

| Split | All R@0.50 | All R@0.75 | Class1 R@0.50 | Class1 R@0.75 |
| --- | ---: | ---: | ---: | ---: |
| valid_combined | 0.6649 | 0.3136 | 0.5456 | 0.1766 |
| valid_phantom | 0.6262 | 0.3544 | 0.4442 | 0.2160 |
| valid_animal | 0.9626 | 0.0000 | 1.0000 | 0.0000 |

Interpretation:

- Stage2X does add real oracle value for class1, especially on phantom.
- It does not solve animal high-IoU localization.
- The bottleneck shifts from "does a useful candidate exist?" to "can the ranker identify it?"

## Ranker Fine-Tune Diagnostic

A one-epoch smoke fine-tune was run from the Stage2U checkpoint:

- run: `outputs/task2/stage2u_quality_ranker/stage2x_ranker_smoke_train30k_valid128_e1`;
- checkpoint: `checkpoints/best.pt`;
- train candidates: 23,599;
- valid candidates in checkpoint selection: 128 samples per large split;
- command log: `quality_reports/logs/task2_stage2x_ranker_smoke_train30k_valid128_e1.log`.

Small-valid result looked promising:

| Split | Best mAP50 | Best mAP50-95 |
| --- | ---: | ---: |
| valid_combined, 128-sample subset | 0.5951 | 0.0954 |
| valid_phantom, 128-sample subset | 0.1162 | 0.0541 |
| valid_animal | 0.5003 | 0.0501 |

However, full `valid_combined` evaluation showed the small-valid result was optimistic:

| Eval | valid_combined mAP50 | valid_combined mAP50-95 |
| --- | ---: | ---: |
| Stage2X smoke ranker full-valid checkpoint | 0.1536 | 0.0279 |
| Current Stage2V champion | 0.2010 | 0.0492 |

The full-valid eval was stopped after `valid_combined`, because it was already below the current champion.

## Source Filter Diagnostic

Using the existing full five-source eval:

`outputs/task2/stage2u_quality_ranker/stage2x_five_source_fullvalid_eval`

the source-filter sweep was saved to:

`outputs/task2/stage2u_quality_ranker/stage2x_five_source_fullvalid_eval/source_filter_blend_rank_decay_roi.json`

Important `valid_combined` results:

| Policy | Rows | mAP50 | mAP50-95 |
| --- | ---: | ---: | ---: |
| all five sources | 206,635 | 0.1648 | 0.0303 |
| no Stage2X | 160,085 | 0.1663 | 0.0314 |
| YOLO + geometry | 66,985 | 0.1875 | 0.0414 |
| YOLO only | 20,485 | 0.1883 | 0.0416 |
| YOLO + Stage2X | 67,035 | 0.1852 | 0.0387 |
| Stage2X only | 46,550 | 0.0189 | 0.0046 |

Class1 did improve slightly when Stage2X was mixed with YOLO:

- YOLO-only class1 AP50-95: 0.0290;
- YOLO+Stage2X class1 AP50-95: 0.0319.

But class0 AP dropped enough that the combined metric got worse.

## Decision

Do not promote Stage2X into the current Task2 champion.

Stage2X is useful as a diagnostic because it proves that better class1 candidates can be generated, but it is not yet useful as a deployable component:

- candidate oracle improves;
- old ranker cannot score the new source correctly;
- short fine-tuning overfits the small validation subset and fails on full validation;
- simple source filtering or YOLO+Stage2X fusion does not beat Stage2V.

Current champion remains:

`outputs/task2/stage2u_quality_ranker/stage2v_class0_yolo_iou75_class1_yolo_roi_export`

with:

- valid_combined mAP50: 0.20097;
- valid_combined mAP50-95: 0.04924;
- valid_phantom mAP50-95: 0.04086;
- valid_animal mAP50-95: 0.05000.

## Next

The next Task2 improvement should not simply add more candidate sources. The evidence points to two more targeted directions:

1. Keep Stage2V as the submission-safe path and improve calibration/export around it.
2. If continuing model work, train a ranker specifically designed for source-aware candidate selection, with balanced positives from each candidate source and a validation split that is not distorted by sample-order truncation.

The immediate engineering change should be to add progress logging to `train_stage2u_quality_ranker.py`, because full candidate eval currently runs for many minutes with no batch-level visibility.
