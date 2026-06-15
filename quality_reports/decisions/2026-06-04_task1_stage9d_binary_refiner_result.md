# Task 1 Stage9D Binary Hard-Negative Refiner Result

Date: 2026-06-04

## Decision

Stage9D is not promoted to the Task 1 champion stack.

The binary hard-negative ROI refiner was a reasonable attempt to reduce false positives under the MSLNet-style tolerance metrics, but the full released-eval gate result regressed both the challenge-style multiclass Dice and the MSLNet-style binary metrics relative to the current Stage9A champion.

## Compared Systems

Current champion:

- Prediction manifest: `outputs/task1/stage9a_seven_model_add_both_010_010/remove_small_min32/predictions.csv`
- Challenge-style Dice: 0.655196
- Label 1 Dice: 0.633971
- Label 2 Dice: 0.676422
- Animal Dice: 0.742323
- Phantom Dice: 0.631411
- MSLNet-style Dice: 0.621562
- MSLNet-style IoU: 0.460505
- MSLNet-style AHD: 1.371323
- MSLNet-style F1 r=3: 0.911578
- MSLNet-style precision r=3: 0.866395
- MSLNet-style recall r=3: 0.967729

Stage9D binary gate result:

- Prediction manifest: `outputs/task1/stage9d_binary_gate_full_t065/predictions.csv`
- Challenge-style Dice: 0.645414
- Label 1 Dice: 0.635857
- Label 2 Dice: 0.654971
- Animal Dice: 0.740680
- Phantom Dice: 0.619406
- MSLNet-style Dice: 0.621404
- MSLNet-style IoU: 0.463412
- MSLNet-style AHD: 1.534148
- MSLNet-style F1 r=3: 0.916587
- MSLNet-style precision r=3: 0.903981
- MSLNet-style recall r=3: 0.932747

MSLNet paper reference on CathAction binary segmentation:

- Dice: 0.6251
- IoU: 0.4658
- AHD: 1.5
- F1 r=3: 0.9305
- Precision r=3: 0.9152
- Recall r=3: 0.9462

## Interpretation

Stage9D did improve MSLNet-style precision compared with Stage9A, but it reduced recall too much. This is consistent with the refiner acting as an aggressive foreground gate: it removes some false positives, but it also erases true thin foreground or near-boundary foreground that the current champion keeps.

The regression is especially harmful for the official multiclass objective:

- Overall Dice drops by about 0.0098.
- Label 2 drops by about 0.0214.
- Phantom drops by about 0.0120.

The MSLNet-style F1 r=3 rises from 0.911578 to 0.916587, but this is still well below the paper's 0.9305 and does not compensate for the official Dice drop. Stage9D also worsens AHD relative to both Stage9A and the MSLNet paper reference.

## Consequence

Do not continue long training on this exact binary-gate formulation as the main route.

Retain the implementation as a diagnostic branch only:

- `src/cathaction/training/task1_baseline.py` binary loss and hard-negative sampling support
- `configs/task1/smp_fpn_convnext_small_512_binary_hardneg_stage9d.yaml`
- `scripts/task1/predict_binary_roi_gate_original_space.py`
- `scripts/task1/run_stage9d_binary_refiner_*.sh`

Recommended next direction:

1. Keep Stage9A as the current submission candidate.
2. If using Stage9D at all, only test low-risk threshold sweeps or soft fusion, not hard masking at 0.65.
3. Shift the main effort back to improving the champion ensemble/postprocess with changes that preserve recall: class-calibrated fusion, soft ROI refinement, or case/domain-specific thresholding.

