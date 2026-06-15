# Task 2 Stage2H: Animal Adaptation Splits

Date: 2026-06-09

Status: approved by user request ("可以，这样看看我们之前的模型会不会有起色")

## Objective

Test whether adding a small amount of labeled animal data to training improves
held-out animal detection/localization. This is a diagnostic for the observed
phantom-to-animal domain shift.

## Constraints

- Do not frame-randomize animal data.
- Split animal data at video/procedure level.
- Do not overwrite the conservative `configs/task2/splits/` clean evaluation
  setup.
- Treat this as an adaptation diagnostic, not an official no-leakage validation
  score for the public split.

## Animal Videos

From `valid_animal.txt`:

- `video_0_animal`: 108 frames, all class0.
- `video_1_animal`: 183 frames, 180 class1 and 3 class0.
- `video_2_animal`: 107 frames, 92 class1 and 15 class0.

## Plan

1. Generate split files under `configs/task2/splits_adapt_animal/`.
2. Generate YOLO YAMLs under `configs/task2/`.
3. Start a short YOLO11s adaptation diagnostic:
   - train: `train_clean + video_1_animal`;
   - val: `video_2_animal`;
   - keep `valid_phantom` available as a separate sanity validation.
4. Compare held-out animal class1 behavior to the previous phantom-only runs.

## Success Criteria

The adaptation idea is useful if held-out animal class1 candidate recall / AP
is no longer near zero.
