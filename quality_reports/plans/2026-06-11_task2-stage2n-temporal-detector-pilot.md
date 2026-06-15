# Task 2 Stage2N: Temporal First-Stage Detector Pilot

Date: 2026-06-11

Status: completed

## Objective

Test whether a real temporal/video object detector can outperform the current
single-frame YOLO Stage2L coarse detector for Task 2 collision detection.

This is distinct from the previous sequence-tip localizer. The localizer was a
custom heatmap-center model and failed as a box detector. Stage2N targets mature
video detection or temporally fused detection models that preserve a real object
detection head and bbox training objective.

## Motivation

CathAction collision detection is frame-wise box detection on fluoroscopy
videos. The paper explicitly benchmarks tiny/video detection methods such as
YOLOV and EFF, and the data has temporal continuity at 24 FPS. The current
single-frame detector has known failure modes:

- animal class0 top50 R@0.50 is near zero;
- phantom class1 proposal recall is weak;
- temporal smoothness exists but simple post-hoc smoothing cannot recover boxes
  that are never proposed.

## Experiment Plan

1. Re-read the CathAction collision-detection section and extract exact baseline
   methods, metrics, and reference details.
2. Find official/primary implementations for YOLOV and EFF.
3. Decide viability:
   - use official YOLOV if installable/adaptable within reasonable time;
   - otherwise build a controlled temporal detector pilot using a known
     detection head with feature/input temporal fusion.
4. Prepare CATHACTION Task2 data in the required video/sequence format while
   preserving case/video-level splits.
5. Run a small smoke train/eval first. **Done:** official YOLOV now trains,
   saves, and evaluates on CATHACTION smoke data.
6. Compare against current Stage2L YOLO on the same panels:
   - mAP/AP if available;
   - top1/top5/top10/top50 proposal R@0.50/R@0.75;
   - animal/phantom breakdown;
   - class0/class1 breakdown.

## Success Criteria

Continue the temporal-detector route if it improves any of:

- combined top50 R@0.50 over YOLO Stage2L's 0.5456;
- phantom top50 R@0.50 over 0.5049;
- animal class0 top50 R@0.50 above 0.0000;
- final ranked mAP/AP over the current single-frame detector.

## Verification

- Save implementation choices and commands under `quality_reports/decisions/`.
- Compile any new scripts.
- Run smoke train/eval before long training.
- Save all outputs under `outputs/task2/`.

## 2026-06-11 Setup Result

Recorded in:

- `quality_reports/decisions/2026-06-11_task2_stage2n_yolov_setup_result.md`

Key result: the official YOLOV route is technically runnable. A representative
3-epoch YOLOV pilot was completed and evaluated. It should not replace Stage2L
as the main first-stage detector, but it can be kept as a weak supplemental
proposal source if candidate budget allows.

Pilot summary:

- YOLOV pilot top50 R@0.50: combined 0.3629, phantom 0.3027, animal 0.8542.
- Stage2L top50 R@0.50: combined 0.5456, phantom 0.5049, animal 0.8598.
- Stage2L + YOLOV oracle union: combined 0.5542, phantom 0.5158, animal 0.8598.
- Animal class0 remains 0.0000, so the core failure mode is not solved by YOLOV.
