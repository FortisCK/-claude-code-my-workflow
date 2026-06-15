# Task 2 Stage2I: Reverse Animal Fold and Phantom Sanity Check

Date: 2026-06-10

Status: completed

## Objective

Check whether the Stage2H animal adaptation finding is robust and whether the
adapted detector sacrifices phantom-domain performance.

## Rationale

Stage2H showed that training on `video_1_animal` and validating on
`video_2_animal` gives strong held-out animal collision detection:

- `collision mAP50 = 0.973`
- `collision mAP50-95 = 0.292`
- `normal mAP50 = 0.000`

This supports a domain-shift diagnosis, but it must be checked in the reverse
direction and against phantom validation before changing the Task 2 roadmap.

## Plan

1. Run a phantom sanity validation of the Stage2H adapted `best.pt` on
   `valid_phantom`.
2. Start reverse fold YOLO11s training:
   - train: `train_clean + video_2_animal`
   - val: `video_1_animal`
3. Compare reverse fold class-wise metrics with Stage2H.
4. Record the results in `quality_reports/decisions/`.

## Success Criteria

The animal-adaptation route is robust if class 1 collision AP is also strong on
`video_1_animal` when training includes `video_2_animal`.

The route is practically useful only if phantom performance is not completely
destroyed or can be recovered with domain-balanced training/calibration.
