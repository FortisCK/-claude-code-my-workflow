# Task2 Stage2O Plan: A-prime Multi-source Candidate Ranker

Date: 2026-06-11
Status: approved, Phase 1/2 completed, Phase 3 pilot completed, Phase 4 pending
Task: CATHACTION Task2 collision detection

## Decision Summary

Three subagent reviews converged on the same practical direction:

- Do not continue patching YOLO/YOLOV as the main route.
- Keep YOLO/YOLOV only as proposal sources and baselines.
- Treat Task2 as a tip-centric localization plus local state verification problem.
- The next best step is not a full detector rewrite, but a deployable pipeline that converts the existing multi-source oracle candidate pool into real ranked detections.

The selected route is Stage2O / A-prime:

`multi-source candidates -> local ROI verifier/ranker -> optional box refinement -> light temporal reranking -> submission-style mAP evaluation`

## Why This Route

Current evidence says the main failure is not that normal/collision state is intrinsically impossible to classify. The GT-ROI classifier worked well when given the correct crop. The failure appears when the system must localize and rank tiny tip boxes from noisy candidates.

The useful signals we already have are complementary:

- Stage2L YOLO remains the strongest generic proposal source.
- Task1 geometry helps recover animal class0 candidates that YOLO misses.
- Sequence-tip localizer has useful center evidence, but its raw boxes are weak.
- YOLOV is technically runnable, but its pilot did not beat Stage2L and should not become the main line.

Therefore the fastest credible route is to rank and refine the existing candidate pool rather than keep replacing first-stage detectors.

## Frozen Inputs

Use existing outputs first. Do not retrain proposal sources before the first Stage2O pilot.

Candidate sources:

- Stage2L YOLO proposals.
- Task1 geometry endpoint/rect proposals.
- Sequence-tip localizer proposals.
- YOLOV proposals as optional low-priority supplementary source.

Reference reports:

- `quality_reports/decisions/2026-06-11_task2_stage2n_yolov_setup_result.md`
- `quality_reports/decisions/2026-06-11_task2_task1_geometry_proposal_result.md`
- `quality_reports/decisions/2026-06-11_task2_stage2m_sequence_tip_localizer_result.md`
- `quality_reports/decisions/2026-06-10_task2_stage2k_proposal_roi_verifier_result.md`

## Implementation Phases

### Phase 1: Unified Candidate CSV

Create one standard candidate table from all proposal sources.

Required schema:

- `sample_id`
- `video_id` or equivalent case/procedure key
- `frame_id`
- `domain`
- `image_path`
- `image_width`
- `image_height`
- `gt_class`
- `gt_box`
- `source`
- `source_rank`
- `source_conf`
- `candidate_box`
- `candidate_iou`
- `matched_gt`

Verification:

- Parse all source CSVs without dropping samples silently.
- Report per-source candidate counts.
- Report oracle recall by domain/class at IoU 0.50 and 0.75.
- Confirm case/procedure split integrity.

### Phase 2: Candidate Pool Audit

Before training, measure whether the candidate pool has enough upper bound to justify a ranker.

Minimum go criteria:

- Combined top50 R@0.50 improves by at least +0.02 to +0.03 over Stage2L.
- Animal class0 top50 R@0.50 reaches at least 0.50.
- Phantom performance does not collapse relative to Stage2L.
- Source contribution is not entirely dominated by one noisy source.

Stop condition:

- If animal class0 top50 R@0.50 remains below 0.40, do not train another verifier yet. Fix candidate generation.

### Phase 3: CSV-driven ROI Verifier and Ranker

Adapt the existing ROI verifier to read the unified candidate CSV rather than generating candidates online from one YOLO model.

Training target:

- Three classes: background, normal, collision.
- Candidate IoU >= 0.50: positive normal/collision according to GT class.
- Candidate IoU <= 0.20: background.
- Intermediate IoU candidates: ignored or used only for quality calibration after a first pilot.

Sampling:

- Balance domain.
- Balance normal/collision.
- Balance background/positive.
- Balance proposal source enough to avoid source-prior overfitting.

Initial scoring:

- `score_normal = source_conf * p_normal`
- `score_collision = source_conf * p_collision`
- keep background probability for rejection and calibration.

Verification:

- Detector-style AP/mAP, not just ROI classification accuracy.
- Report combined, phantom, animal, class0, class1.
- Compare against YOLO baseline, Stage2K, and simple `source_conf * ROI_score`.

### Phase 4: Optional Local Box Refinement

Only add box refinement if Phase 2/3 shows the ranker is limited by IoU ceiling.

Candidate refinement options:

- Candidate crop regression for center offset and width/height.
- Source-specific template calibration.
- Sequence-tip center as an auxiliary feature, not as a standalone detector.

Go criteria:

- Matched candidate mean IoU improves by at least 0.05, or
- R@0.75 improves by at least +0.05 absolute.

Stop condition:

- If center is near GT but IoU does not improve, inspect bbox target convention and crop scale before adding more model capacity.

### Phase 5: Light Temporal Reranking

Temporal modeling should be a final reranking/calibration step, not the main detector.

Use:

- center jump penalty
- box size jump penalty
- state persistence
- collision onset/offset hysteresis
- Viterbi/HMM-style sequence selection over candidate scores

Verification:

- Compare raw ranker vs temporal ranker on mAP50-95 and mAP50.
- Reject temporal smoothing if it only improves visual smoothness but not AP/mAP.

## Primary Success Gates

The first Stage2O pilot is considered successful only if it clears these minimum gates:

- Combined mAP50-95 >= 0.055.
- Combined mAP50 >= 0.13.
- Class1/collision AP50-95 >= 0.010.
- Valid phantom and valid animal do not drop by more than 0.005 vs the frozen baseline unless explained by a planned tradeoff.
- Animal class0 candidate-stage top50 R@0.50 >= 0.50.
- Learned ranker beats ROI-only and simple `YOLO_conf * ROI_score`.

## Explicit No-go Items

Do not spend more time on these unless new evidence changes the situation:

- Full YOLOV training as the main route.
- Another YOLO recipe as the main route.
- Standalone sequence-tip localizer as a YOLO replacement.
- GT-ROI classifier as final evidence.
- Oracle-only improvements without detector-style mAP gain.
- Simple temporal smoothing over weak candidates.
- MONAI UNet-style segmentation detours for Task2.

## Expected Deliverables

- Unified candidate CSV generator.
- Candidate pool audit report.
- CSV-driven ROI verifier/ranker training script.
- Detector-style evaluation report.
- Failure montage for selected samples.
- Decision note: continue with ranking, add box refinement, or revisit candidate generation.

## Next Action After Approval

Implement Phase 1 and Phase 2 first. No model training should start until the candidate pool audit proves that the multi-source pool gives enough recall ceiling.

## Progress Update: 2026-06-11

Phase 1/2 has been completed.

Artifacts:

- `scripts/task2/build_stage2o_candidate_pool.py`
- `tests/test_task2_stage2o_candidate_pool.py`
- `outputs/task2/stage2o_candidate_pool/stage2o_aprime_default_sources_top50/summary_audit.json`
- `outputs/task2/stage2o_candidate_pool/stage2o_aprime_default_sources_top50/candidate_pool_audit.md`
- `quality_reports/decisions/2026-06-11_task2_stage2o_candidate_pool_audit_result.md`

Gate result: GO for Phase 3 CSV-driven ROI verifier/ranker.

Key audit values:

- Combined union R@0.50: 0.5789 vs Stage2L baseline 0.5456.
- Animal class0 union R@0.50: 0.6667 vs Stage2L baseline 0.0000.
- Phantom union R@0.50: 0.5316 vs Stage2L baseline 0.5049.

Remaining risk:

- Animal R@0.75 remains 0.0000, so Phase 4 box refinement will likely be needed if Phase 3 mAP50-95 remains low.

## Progress Update: 2026-06-11 Phase 3

Phase 3 CSV-driven ROI ranker pilot has been completed.

Artifacts:

- `scripts/task2/prepare_stage2o_ranker_train_subset.py`
- `scripts/task2/train_stage2o_candidate_ranker.py`
- `outputs/task2/stage2o_candidate_pool/stage2o_train_subset_yolo_geometry_top50/valid_combined_candidates.csv`
- `outputs/task2/stage2o_candidate_pool/stage2o_valid_yolo_geometry_top50/summary_audit.json`
- `outputs/task2/stage2o_ranker/convnext_tiny_stage2o_yolo_geometry_train_subset_valid256_e4`
- `outputs/task2/stage2o_ranker/convnext_tiny_stage2o_yolo_geometry_train_subset_full_eval/eval_metrics.json`
- `quality_reports/decisions/2026-06-11_task2_stage2o_ranker_pilot_result.md`

Main full-valid result for YOLO+geometry ranker:

- valid_combined rank_decay mAP50: 0.1887.
- valid_combined rank_decay mAP50-95: 0.0388.
- valid_phantom rank_decay mAP50-95: 0.0340.
- valid_animal rank_decay mAP50-95: 0.0467.

Interpretation:

- Ranker/background rejection now works when trained on matched source distributions.
- Combined mAP50 passes the planned 0.13 gate.
- Combined mAP50-95 fails the planned 0.055 gate.
- The next bottleneck is box quality, especially because animal R@0.75 remains 0.0000.

Decision:

- Move to Phase 4 box refinement.
- Do not spend the next iteration primarily on a bigger ROI classifier.
