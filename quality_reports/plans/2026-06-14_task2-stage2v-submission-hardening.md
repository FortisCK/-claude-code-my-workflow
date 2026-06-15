# Task2 Stage2V Submission Hardening Plan

Date: 2026-06-14

## Goal

Freeze the current Task2 Stage2V validation champion into a reproducible, machine-checkable artifact so we can move from exploration to submission preparation without losing the exact policy and metrics.

## Current Champion

Path:

`outputs/task2/stage2u_quality_ranker/stage2v_class0_yolo_iou75_class1_yolo_roi_export`

Policy:

- class 0: `stage2u_warm_stage2o_fullvalid_yolo_only_eval`, score mode `prob_iou75_source_rank_decay_roi`
- class 1: `stage2u_warm_stage2o_fullvalid_yolo_only_eval`, score mode `roi`

Validation result:

- `valid_combined` mAP50: 0.20096803092334323
- `valid_combined` mAP50-95: 0.04924044637514719

## Steps

1. Add a verifier for the Stage2V champion export.
   - Check required prediction CSVs exist.
   - Check exported CSV columns match the clean internal schema.
   - Reject validation-only or leakage fields such as `gt_*`, `candidate_iou`, and `verifier_label`.
   - Check row counts match `eval_metrics.json`.
   - Check key mAP values against expected champion metrics.

2. Add a small test suite for the verifier.
   - Cover passing clean exports.
   - Cover leakage-field rejection.
   - Cover metric mismatch rejection.

3. Generate/update a champion manifest.
   - Record policy, source runs, score modes, split row counts, and metrics.
   - Keep this beside the champion CSVs for handoff and hidden-test adaptation.

4. Add a hidden-test preparation checklist.
   - State exactly which parts are frozen.
   - State what must change once the official result schema is published.

5. Run verification.
   - Python compile check for the new script.
   - Pytest for the new verifier tests and existing Task2 export tests.
   - Run the verifier on the current Stage2V champion artifact.

