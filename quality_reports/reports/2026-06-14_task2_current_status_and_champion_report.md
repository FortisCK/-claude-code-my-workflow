# Task2 Current Status and Champion Report

Date: 2026-06-14

## Executive Summary

Task2 is collision detection in endovascular intervention videos. The current conservative internal validation champion is:

`outputs/task2/stage2u_quality_ranker/stage2v_class0_yolo_iou75_class1_yolo_roi_export`

A newer public-validation best is the Stage2AB domain-aware score policy:

`outputs/task2/stage2v_domain_policy/yolo_only_domain_policy_sweep.json`

Current public-validation metrics:

| Split | mAP50 | mAP50-95 | Prediction rows |
| --- | ---: | ---: | ---: |
| Stage2V valid_combined | 0.20097 | 0.04924 | 40,970 |
| Stage2AB valid_combined | 0.21128 | 0.05063 | 40,970 |
| Stage2AB valid_phantom | 0.10462 | 0.04301 | 39,338 |
| Stage2AB valid_animal | 0.49974 | 0.05001 | 1,632 |

The active policy is class-aware but source-simple:

- class 0 / normal:
  - run: `outputs/task2/stage2u_quality_ranker/stage2u_warm_stage2o_fullvalid_yolo_only_eval`;
  - score mode: `prob_iou75_source_rank_decay_roi`;
- class 1 / collision:
  - run: `outputs/task2/stage2u_quality_ranker/stage2u_warm_stage2o_fullvalid_yolo_only_eval`;
  - score mode: `roi`.

The result is modest in absolute value. Stage2AB is the best public-validation metric so far, but Stage2V remains the conservative fallback because Stage2AB adds domain-aware validation calibration.

## Task Framing

Task2 evaluates detection boxes with mAP as the primary metric. In our local validation setup each frame/sample has one GT state/box, and AP is sensitive to:

- whether a candidate overlaps the target at IoU thresholds 0.50 to 0.95;
- whether high-IoU candidates are ranked above low-IoU candidates;
- class balance between normal/class0 and collision/class1;
- domain shift between phantom and animal.

The major practical issue is not only finding approximate locations. Some candidate pools have reasonable oracle recall, but AP remains low because scores do not rank the correct high-IoU candidates early enough.

## What We Tried

### YOLO Baselines

Early YOLO/YOLO-style detectors gave a usable coarse proposal baseline, but the boxes were noisy and class behavior was domain-dependent.

Key lesson:

- YOLO-like proposals are still the strongest practical source for the current champion.
- Adding many more non-YOLO proposal sources increases oracle recall but often hurts AP due to false positives and poor ranking.

### Two-Stage ROI/Quality Ranker

We built Stage2U, an image-aware ROI quality ranker that scores candidate boxes and predicts:

- background / normal / collision class;
- predicted IoU;
- probability of IoU >= 0.50;
- probability of IoU >= 0.75.

The ranker supports multiple score modes, including raw ROI class probability, source-confidence adjusted score, rank decay, and quality-adjusted variants.

The best outcome came not from a single global score mode, but from class-aware score selection.

### Stage2V: Current Champion

Stage2V fuses class-specific predictions from saved Stage2U eval runs.

The final chosen export is:

```bash
python3 scripts/task2/export_stage2v_class_aware_predictions.py \
  --class0-run-dir outputs/task2/stage2u_quality_ranker/stage2u_warm_stage2o_fullvalid_yolo_only_eval \
  --class0-score-mode prob_iou75_source_rank_decay_roi \
  --class1-run-dir outputs/task2/stage2u_quality_ranker/stage2u_warm_stage2o_fullvalid_yolo_only_eval \
  --class1-score-mode roi \
  --output-dir outputs/task2/stage2u_quality_ranker/stage2v_class0_yolo_iou75_class1_yolo_roi_export
```

Clean prediction CSVs contain only inference fields:

`sample_id, video_id, frame_index, domain, class_id, score, x1, y1, x2, y2, source, source_rank, score_mode, policy_name`

There are no `gt_*` validation-only fields in the exported prediction CSVs.

### Stage2X: Class1 Center Localizer

Stage2X trained a class1-only temporal localizer. It improved candidate oracle recall:

| Split | All R@0.50 | All R@0.75 | Class1 R@0.50 | Class1 R@0.75 |
| --- | ---: | ---: | ---: | ---: |
| valid_combined | 0.6649 | 0.3136 | 0.5456 | 0.1766 |
| valid_phantom | 0.6262 | 0.3544 | 0.4442 | 0.2160 |
| valid_animal | 0.9626 | 0.0000 | 1.0000 | 0.0000 |

But it did not improve AP when added to the scored candidate pool:

| Policy | valid_combined mAP50 | valid_combined mAP50-95 |
| --- | ---: | ---: |
| YOLO only | 0.1883 | 0.0416 |
| YOLO + Stage2X | 0.1852 | 0.0387 |
| Stage2X only | 0.0189 | 0.0046 |

Conclusion:

- Stage2X finds some useful boxes;
- existing score/ranker logic cannot rank them properly;
- do not promote Stage2X into the current champion.

### Stage2Y: Source-Aware Residual Ranker

Stage2Y added optional source metadata to Stage2U:

```text
output = image_model(roi) + metadata_head(source_metadata)
```

The source-aware head is zero-initialized and old Stage2U checkpoints load into the wrapped image model.

Sampled validation looked promising:

| Epoch | sampled valid_combined best mAP50 | sampled valid_combined best mAP50-95 |
| ---: | ---: | ---: |
| 1 | 0.5282 | 0.0622 |
| 2 | 0.5599 | 0.0745 |

But full validation failed:

| Mode | full valid_combined mAP50 | full valid_combined mAP50-95 |
| --- | ---: | ---: |
| `prob_iou75_rank_decay_roi` | 0.1557 | 0.0321 |
| `blend_rank_decay_roi` | 0.1388 | 0.0319 |

Class-aware fusion using Stage2Y for class1 also failed:

| Policy | valid_combined mAP50 | valid_combined mAP50-95 |
| --- | ---: | ---: |
| class0 YOLO + class1 Stage2Y | 0.1753 | 0.0439 |
| Stage2V champion | 0.2010 | 0.0492 |

Conclusion:

- source metadata is not enough;
- small validation truncation is misleading;
- full valid remains the decision source.

### Stage2Z: Existing Stage2T Listwise Selector

Stage2Z tested the existing tabular per-frame selector on the five-source full-valid pool.

Best result:

| Quality | Base score | k | valid_combined mAP50 | valid_combined mAP50-95 |
| --- | --- | ---: | ---: | ---: |
| `blend` | `roi` | 50 | 0.1645 | 0.0371 |

Conclusion:

- it improves over the weakest five-source independent scorer;
- it does not beat YOLO+geometry or Stage2V;
- a real listwise selector would need source-matched train-side prediction rows for all proposal sources.

### Stage2AA: Source-Matched Three-Source Selector

Stage2AA fixed the main Stage2Z validity issue by matching train and validation proposal sources:

- `yolo_stage2l`;
- `task1_geometry_rect`;
- `stage2x_class1`.

The source-match audit passed:

- train candidate/prediction rows: 278,006 / 278,006;
- valid candidate/prediction rows: 113,535 / 113,535;
- source set match: true.

Best result:

| Policy | valid_combined mAP50 | valid_combined mAP50-95 |
| --- | ---: | ---: |
| `pred_iou` + `rank_decay_roi`, k=50 | 0.1537 | 0.0390 |
| Stage2V champion | 0.2010 | 0.0492 |

Conclusion:

- the fair source-matched selector still does not beat Stage2V;
- do not promote Stage2AA;
- further tuning of this HistGradientBoosting selector is unlikely to be the right next use of time without changing the objective.

### Stage2AB: Domain-Aware Score Calibration

Stage2AB keeps the yolo-only candidate source and searches only score modes by class/domain.

Selected policy:

- class 0 / normal:
  - phantom: `prob_iou75_source_rank_decay_roi`;
  - animal: `prob_iou75_source_rank_decay_roi`;
- class 1 / collision:
  - phantom: `rank_decay_roi`;
  - animal: `prob_iou75_roi`.

Result:

| Method | valid_combined mAP50 | valid_combined mAP50-95 |
| --- | ---: | ---: |
| Stage2V conservative champion | 0.2010 | 0.0492 |
| Stage2AB domain-aware policy | 0.2113 | 0.0506 |

Conclusion:

- Stage2AB is the current public-validation best;
- the gain is small but achieved without adding candidate sources;
- keep Stage2V as a fallback because Stage2AB is more explicitly tuned to public-valid domain behavior.

## Current Technical Assets

Reusable scripts:

- `scripts/task2/export_stage2v_class_aware_predictions.py`
  - single-entry current champion export wrapper;
- `scripts/task2/sweep_stage2v_domain_policy.py`
  - searches and exports the current Stage2AB domain-aware policy;
- `scripts/task2/verify_stage2v_champion_export.py`
  - verifies the frozen Stage2V export schema, row counts, policy, and champion metrics;
  - writes `champion_manifest.json` beside the prediction CSVs;
- `scripts/task2/verify_stage2ab_domain_policy.py`
  - verifies the frozen Stage2AB policy, clean CSV schema, row counts, and public-valid metrics;
  - writes `stage2ab_verification.json`;
- `scripts/task2/train_stage2u_quality_ranker.py`
  - Stage2U image-aware ranker;
  - now also supports `--source-aware` and `--log-interval`;
- `scripts/task2/train_stage2t_tabular_selector.py`
  - tabular per-frame selector;
- `scripts/task2/evaluate_stage2u_class_fusion.py`
  - class-aware fusion evaluator;
- `scripts/task2/evaluate_stage2u_source_filter.py`
  - source filtering diagnostics;
- `scripts/task2/evaluate_candidate_oracle.py`
  - candidate oracle diagnostics.

Important decision files:

- `quality_reports/decisions/2026-06-13_task2_stage2v_score_mode_sweep_result.md`;
- `quality_reports/decisions/2026-06-13_task2_stage2x_class1_localizer_result.md`;
- `quality_reports/decisions/2026-06-14_task2_stage2aa_source_matched_selector_result.md`;
- `quality_reports/decisions/2026-06-14_task2_stage2ab_domain_policy_result.md`;
- `quality_reports/decisions/2026-06-14_task2_stage2ab_verification_package.md`;
- `quality_reports/decisions/2026-06-14_task2_stage2y_source_aware_ranker_result.md`;
- `quality_reports/decisions/2026-06-14_task2_stage2z_listwise_selector_result.md`.
- `quality_reports/decisions/2026-06-14_task2_stage2v_submission_hardening_result.md`.

## Main Lessons

1. Full validation is mandatory.

Sampled validation repeatedly overestimated results. Stage2Y looked excellent on 256 sampled combined frames, then dropped below Stage2V on full combined validation.

2. Candidate oracle recall is not the current decisive metric.

Five-source pools can contain better candidates, but AP drops if confidence ordering is poor.

3. Normal/class0 and collision/class1 behave differently.

Class-aware score selection is the only recent change that gave a reliable full-validation improvement.

4. Animal class0 remains weak.

The current champion has animal class1 AP near 1.0 at IoU 0.50, but animal class0 AP is essentially zero. This may reflect domain split difficulty, label distribution, or proposal construction.

5. The best current method is not the most complex one.

The current champion is simpler than Stage2X/Y/Z: YOLO-only candidate source plus class-specific score modes.

## Next Recommended Work

### Short-Term: Submission Hardening

Use Stage2V as the current Task2 deliverable and focus on:

1. freezing the exact champion config;
2. writing a single hidden-test inference checklist;
3. creating an official-format adapter once the result schema is known;
4. producing visual examples for advisor/challenge reporting;
5. documenting limitations and expected hidden-test risks.

### Medium-Term: True Listwise Redesign

If we continue model development, do not add another candidate source blindly. Build a source-matched listwise selector:

1. generate train-side prediction rows for all sources used at validation/test;
2. train a per-frame selector with candidates from the same frame as a group;
3. optimize for selecting top candidates per class within frame;
4. evaluate only on full valid combined.

This is the next plausible route to use Stage2X oracle gains. It is larger than the current Stage2T transfer and should be treated as a separate experiment, not a quick patch.

## Current Decision

For now, keep Stage2AB as the public-validation best and Stage2V as the conservative fallback. Stop promoting Stage2X/Y/Z/AA variants.

The next practical milestone is a clean, reproducible Stage2V submission/handoff package rather than another exploratory model tweak.

That milestone now has an initial implementation:

- verifier: `scripts/task2/verify_stage2v_champion_export.py`;
- manifest: `outputs/task2/stage2u_quality_ranker/stage2v_class0_yolo_iou75_class1_yolo_roi_export/champion_manifest.json`;
- checklist: `quality_reports/reports/2026-06-14_task2_stage2v_hidden_test_checklist.md`.
