# Decision Record — Task 2 mAP Interpretation and Tie-Break

**Date:** 2026-06-15
**Status:** ACTIVE (binding revisit 2026-07-10 when the official validation package + evaluator are released)
**Source of truth:** 2026-04-22 CATHACTION BIAS challenge PDF, Task 2 "Assessment Methods" (pp. 34-35).

## What the official PDF says (verbatim points)

- "Performance of the submitted algorithms for collision detection will be evaluated using **Average Precision (AP) and mean Average Precision (mAP)**."
- "AP is computed based on **precision–recall curves for collision event detection under different confidence thresholds**."
- "mAP is calculated by **averaging AP scores across all test cases and across defined detection settings**."
- "The **primary ranking metric will be mAP** on the private test set, while AP will be reported as a complementary indicator."
- Ranking method: "For each submission, **AP is computed per test case** ... Aggregate results across the entire test set to obtain **mean AP (mAP)**. Rank all submissions in descending order based on mAP."

## What is RESOLVED by the wording

1. **Primary metric = mAP** (AP secondary). Reaffirms INV-8.
2. **Aggregation = per-case averaging.** The ranking text is explicit: AP is computed *per test case*, then averaged → mAP. This is **macro-average over cases**, NOT a single global pool of all boxes. Our prior local evaluator (`compute_detection_map`) pools every box globally and is therefore frame-count weighted — a different number. We have added a sibling `compute_detection_map_per_case` (src/cathaction/metrics/detection.py) that matches the official per-case wording, with a unit test pinning that the two readings genuinely diverge (global 0.752 vs per-case 0.500 on the test scenario).
3. **Matching is box-IoU object detection**, with two classes `collision` / `normal`. Confirmed by the dataset paper (arXiv 2408.13126, §IV-D and Table V, which report AP/mAP for tiny-object detectors YOLOV/EFF) and the bounding-box annotation described in the PDF. **Do NOT build a self-derived per-frame/event-unit GT loader** — that targets a contradicted hypothesis.

## What remains GENUINELY AMBIGUOUS

**The IoU axis.** "Different confidence thresholds" describes the PR-curve sweep (standard AP). "Across defined detection settings" is undefined — it may or may not mean multiple IoU thresholds (COCO 0.50:0.05:0.95). So we cannot yet tell whether official mAP is:

- **AP50-like** (single IoU 0.50, loose localization), or
- **COCO mAP50-95** (multi-IoU, strict localization).

Under the loose reading our system looks competitive/ahead of the paper baselines (Stage2AQ local mAP50 23.66% vs EFF 16.91% AP-mean); under the strict reading it is behind (local mAP50-95 7-8% vs EFF 14.88% mAP-mean). The two readings invert the conclusion, so we will not commit to one.

## Decision

1. **Local evaluation reports BOTH** mAP50 and mAP50-95, computed with **per-case macro-averaging** (`compute_detection_map_per_case`) as the official-aligned primary, and the global-pool number retained only as a secondary diagnostic.
2. **Maintain two frozen candidates, do not deprioritize either:**
   - **Stage2AQ** — operational default (strongest balanced local mAP50, already wired into the submission entrypoint).
   - **Stage2AI** — COCO/strict hedge (best local mAP50-95).
3. **Tie-break decision tree** (applied once the official metric is known):
   - official ranking is AP50-like → submit **Stage2AQ**;
   - official ranking is COCO mAP50-95 → submit **Stage2AI**;
   - official ranking is domain-reweighted or otherwise unexpected → re-rank all frozen candidates on the official package before choosing.
4. **Binding revisit trigger:** when the official validation package + evaluator release (scheduled 2026-07-10), (a) align the local evaluator to the official code, (b) re-rank Stage2AE/AQ/AI/V on the official validation set, (c) pick the submission per the tree above. The first 48 h after release is re-ranking + schema-fill, not new design.

## Explicit non-actions (until 2026-07-10)

- Do not hard-tune confidence/top-k to chase mAP50-95 on the self-built (single-video phantom, see [[task2-split]]) panel — plateau-picks only.
- Do not implement an event-level GT loader.
- Do not collapse to a single champion before the evaluator is known.

## References

- `src/cathaction/metrics/detection.py` — `compute_detection_map` (global pool), `compute_detection_map_per_case` (official-aligned).
- `tests/test_task2_per_case_map.py` — divergence proof.
- `quality_reports/reports/2026-06-14_task2_metric_alignment_high_iou_route.md` — prior analysis.
- `quality_reports/plans/2026-06-15_prevalidation_roadmap.md` — Tier A item #4.
