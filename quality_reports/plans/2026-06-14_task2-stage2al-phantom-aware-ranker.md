# Task2 Stage2AL Phantom-Aware Ranker Plan

Date: 2026-06-14

## Goal

Train a focused Stage2U ranker variant that targets the Stage2AK bottleneck:
phantom/class-aware candidate ranking, especially `valid_phantom` class1 AP50.

## Rationale

Stage2AK showed that the current Stage2AE yolo-only candidate pool has a large
oracle scoring gap:

- `valid_phantom` current mAP50: 0.10462265654206386;
- `valid_phantom` oracle-score mAP50: 0.4003400993223441;
- `valid_phantom` class1 current AP50: 0.045713631775941704;
- `valid_phantom` class1 oracle-score AP50: 0.24037906909174153.

The current exported candidate source is yolo-only, but the previous Stage2U
ranker was trained on a mixed yolo+geometry candidate distribution. Stage2AL
will align train/eval source distribution and select checkpoints by a
phantom/class-aware metric.

## Steps

1. Add `--train-source-keep` to `train_stage2u_quality_ranker.py`.
2. Add `--primary-metric-key` so checkpoint selection can target
   `valid_phantom/prob_iou75_rank_decay_roi/class1_ap50`.
3. Run a GPU sanity check with a small train/valid limit.
4. If sanity passes, launch a longer yolo-only ranker run.
5. Evaluate/export through the existing Stage2AB/AE/AI-compatible pipeline and
   re-run Stage2AK attribution.

## Acceptance Criteria

- Sanity run reaches evaluation without schema/runtime errors.
- Full run writes `*_best_prediction_rows.csv`.
- Promote only if it improves `valid_phantom` class1 AP50 over 0.045713631775941704
  and does not materially damage `valid_combined` mAP50 unless it becomes a
  high-`mAP50-95` alternate.

