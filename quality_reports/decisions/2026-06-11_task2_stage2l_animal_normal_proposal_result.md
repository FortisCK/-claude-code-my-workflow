# Task 2 Stage2L Animal-Normal-Aware Proposal Result

Date: 2026-06-11

## Experiment

Stage2L trained a one-class YOLO11s proposal detector with:

- train: original phantom `train_clean` plus `video_0_animal` and
  `video_1_animal`;
- validation for checkpoint selection: `video_2_animal` plus phantom
  balanced-small;
- labels: both original classes mapped to `0: tool_roi`.

Artifacts:

- Weights:
  `outputs/task2/yolo_stage2l_proposal/yolo11s_1024_agnostic_train_v0_v1_val_v2_combined_bal_e20/weights/best.pt`
- Training log:
  `quality_reports/logs/task2_yolo11s_1024_stage2l_agnostic_train_v0_v1_val_v2_combined_bal_e20.log`
- Proposal eval:
  `outputs/task2/yolo_proposal_eval/yolo11s_1024_stage2l_combined_bal_e20_stage2l_panels/summary_metrics.json`

Training stopped early after 19 epochs. The best checkpoint was from epoch 11.
YOLO's final best-checkpoint validation on the combined panel was:

- precision: 0.271
- recall: 0.214
- mAP50: 0.112
- mAP50-95: 0.0385

## Proposal Recall Comparison

All rows below use the same Stage2L panels and top-k proposal evaluation.

| Proposal source | Split | Top10 R@0.50 | Top50 R@0.50 | Top50 R@0.75 |
| --- | --- | ---: | ---: | ---: |
| Old class-agnostic | combined | 0.3727 | 0.4479 | 0.2030 |
| Stage2H animal-adapted | combined | 0.4329 | 0.4930 | 0.1439 |
| Stage2L combined-val | combined | 0.4715 | 0.5456 | 0.2170 |
| Old class-agnostic | phantom balanced-small | 0.4211 | 0.5061 | 0.2294 |
| Stage2H animal-adapted | phantom balanced-small | 0.3774 | 0.4454 | 0.1626 |
| Stage2L combined-val | phantom balanced-small | 0.4211 | 0.5049 | 0.2451 |
| Old class-agnostic | animal video_2 | 0.0000 | 0.0000 | 0.0000 |
| Stage2H animal-adapted | animal video_2 | 0.8598 | 0.8598 | 0.0000 |
| Stage2L combined-val | animal video_2 | 0.8598 | 0.8598 | 0.0000 |

Class-wise top50 R@0.50:

| Proposal source | Split | Class 0 | Class 1 |
| --- | --- | ---: | ---: |
| Old class-agnostic | combined | 0.7049 | 0.2302 |
| Stage2H animal-adapted | combined | 0.5878 | 0.4127 |
| Stage2L combined-val | combined | 0.7096 | 0.4067 |
| Old class-agnostic | phantom balanced-small | 0.7306 | 0.2816 |
| Stage2H animal-adapted | phantom balanced-small | 0.6092 | 0.2816 |
| Stage2L combined-val | phantom balanced-small | 0.7354 | 0.2743 |
| Old class-agnostic | animal video_2 | 0.0000 | 0.0000 |
| Stage2H animal-adapted | animal video_2 | 0.0000 | 1.0000 |
| Stage2L combined-val | animal video_2 | 0.0000 | 1.0000 |

## Interpretation

Stage2L improves the balanced combined proposal coverage relative to both old
proposal sources. It also preserves the animal collision coverage gained by
Stage2H while recovering phantom behavior back to roughly the old
class-agnostic level.

However, it does not solve the intended animal normal problem:

- animal `class 0` top50 R@0.50 remains 0.0000;
- animal `class 1` top50 R@0.50 remains 1.0000;
- animal R@0.75 remains 0.0000, so the proposals are coarse even when they
  count as covered at IoU 0.50.

The result is therefore useful as a more balanced proposal checkpoint, but it is
not sufficient for a final two-stage detector.

## Decision

Do not spend another overnight run on the same one-class YOLO split recipe. The
next iteration should target the specific failure mode:

- animal normal proposals need a different cue than the collision-like ROI;
- class-agnostic training alone cannot force the detector to cover both animal
  normal and animal collision;
- a better next experiment is likely either class-conditional proposal training
  with domain-balanced sampling, or a local ROI scoring/ranking method seeded by
  multiple proposal sources rather than one YOLO checkpoint.

