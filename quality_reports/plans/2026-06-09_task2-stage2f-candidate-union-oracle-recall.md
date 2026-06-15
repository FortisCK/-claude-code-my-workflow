# Task 2 Stage2F: Candidate-Union Oracle Recall

Date: 2026-06-09

Status: approved by user request ("我们可以试试看")

## Objective

Before training a second-stage reranker, measure whether the existing coarse
detectors generate any usable candidate boxes. This is a low-cost gate:

- If the union top-k candidate pool contains boxes with IoU >= 0.5 to GT, a
  proposal-aware hard-negative reranker is worth training.
- If the union top-k candidate pool still misses GT, the first-stage localizer
  must change, likely toward a heatmap/CenterNet-style model.

## Candidate Sources

1. two-class YOLO11s:
   `outputs/task2/yolo/yolo11s_1024_clean_combined_e100/weights/best.pt`
2. class-agnostic YOLO11s:
   `outputs/task2/yolo_proposal/yolo11s_1024_agnostic_clean_combined_e100/weights/best.pt`
3. collision-only YOLO11s:
   `outputs/task2/yolo_collision/yolo11s_1024_collision_only_clean_combined_e80/weights/best.pt`

## Metrics

Report for valid_combined, valid_phantom, and valid_animal:

- top1/top5/top10/top20/top50 oracle recall at IoU >= 0.25, 0.50, 0.75
- same metrics split by GT class 0/1
- mean/median best IoU
- source contribution: which detector supplied the best candidate

## Scope

- Use validation splits only for diagnosis, not hidden-test tuning.
- Do not change model weights.
- Save CSV rows and summary JSON under
  `outputs/task2/candidate_union_oracle/`.

## Success Criteria

Reranker is worth trying if either condition is met:

- valid_combined class1 top50 IoU>=0.5 recall is materially above current top5
  detector recall; or
- valid_animal class1 top50 IoU>=0.5 recall is non-trivial.

If both remain near zero, move first to a heatmap/localizer.
