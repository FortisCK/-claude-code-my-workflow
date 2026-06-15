# Task 2 Stage2D: Class-Agnostic YOLO + ROI Fusion Check

Date: 2026-06-09

Status: approved by user request ("你先测一下看看吧")

## Objective

Check whether the Stage2C class-agnostic YOLO proposal detector improves final
Task 2 detector-style mAP when combined with the existing Stage2A ConvNeXt ROI
classifier.

## Scope

1. Run `scripts/task2/evaluate_yolo_roi_fusion.py` with:
   - YOLO weights:
     `outputs/task2/yolo_proposal/yolo11s_1024_agnostic_clean_combined_e100/weights/best.pt`
   - ROI classifier:
     `outputs/task2/roi_classifier/convnext_tiny_gtroi224_scale8_e30/checkpoints/best.pt`
   - candidate mode: top-k
   - score mode: `det_roi`
   - validation splits: combined, phantom, animal
2. Start with top5 because it matches the previous Stage2B best comparison.
3. Compare against the prior Stage2B best:
   - combined mAP50-95: 0.0406
   - combined best-candidate IoU>=0.5: 0.4931
4. If top5 is close or clearly better, consider top10. If top5 is clearly worse,
   do not spend extra time on top10 unless diagnostics suggest it may help.

## Non-Goals

- Do not retrain the detector or classifier in this stage.
- Do not use hidden-test feedback.
- Do not change split definitions.

## Success Criteria

The class-agnostic fusion should clearly exceed the old top5 `det_roi` fusion
on combined validation, or reveal a specific failure mode that guides the next
collision-aware localization stage.
