# Decision Record — Task 2 Honest Baseline (held-out validation, pre-official)

**Date:** 2026-06-15
**Status:** ACTIVE result. Tier B foundation step ("honest baseline only" path).
**Tool:** `scripts/task2/evaluate_candidate_honest_baseline.py` (reuses the pipeline's proven GT/pred loaders; reports global-pool AND official-aligned per-case mAP). Per-candidate JSON in `quality_reports/decisions/2026-06-15_task2_honest_baseline_stage2{v,aq,ai}.json`.

## Headline numbers (same canonical GT for all three)

| Candidate | combined global mAP50 | combined global mAP50-95 | combined per-case mAP50 | combined per-case mAP50-95 |
|---|---:|---:|---:|---:|
| Stage2AQ (operational) | **0.2372** | 0.0692 | **0.3147** | 0.0966 |
| Stage2V (fallback) | 0.2010 | 0.0492 | 0.2979 | 0.0454 |
| Stage2AI (COCO hedge) | 0.1844 | **0.0839** | 0.2926 | **0.1396** |

Per-domain (single video each):

| Candidate | phantom (video_0) mAP50 / 50-95 | animal (video_2_animal) mAP50 / 50-95 |
|---|---|---|
| Stage2AQ | 0.1296 / 0.0500 | 0.4997 / 0.1432 |
| Stage2V | 0.0960 / 0.0409 | 0.4998 / 0.0500 |
| Stage2AI | 0.0974 / 0.0404 | 0.4879 / **0.2388** |

## Honest findings (these reframe the Task 2 standing)

1. **The validation is effectively TWO videos.** GT-from-candidates covers **video_0** (phantom, 824 GT samples) and **video_2_animal** (107 GT samples) — `num_cases=2`. valid_animal nominally lists 3 animal videos but only video_2_animal carries GT here. So BOTH domains are single-video; the signal is thin and high-variance. (Argues for the multi-video retrain if we want robust selection.)

2. **GT-from-candidates inflates recall.** Only 824 of video_0's ~11k frames carry GT (frames with no candidate are not counted). True full-frame-GT mAP would be **lower**. The reported pipeline numbers all share this optimism.

3. **Per-case averaging (official-aligned) materially raises the combined number** vs global pooling — e.g. AQ 0.2372 → 0.3147 — because the strong animal video gets equal weight instead of frame-count weight. The Tier A per-case evaluator was not academic: it changes the headline by ~10 points.

4. **The candidate hedge is confirmed on real data:** AQ wins mAP50, AI wins mAP50-95 (AI's animal mAP50-95 0.2388 vs AQ 0.143 / V 0.05). Keep both per `2026-06-15_task2_metric_interpretation.md`.

5. **Sobering correction to "we beat EFF/YOLOV".** On the phantom domain (the bulk), mAP50 is only **~0.10–0.13**, **below** the paper baselines (YOLOV 0.1589 / EFF 0.1691 AP-mean). The ~0.24 combined headline is **animal-inflated** (one strong 107-frame video). Honestly: we are at/below the paper baselines on phantom; we are not clearly ahead.

## Implications for Tier B

- The performance gap is on **phantom**, so the ranker + box-refinement work should target phantom class behavior, evaluated honestly (video_0, with the recall caveat in mind).
- Validation thinness (2 videos) is now the top measurement risk → the **multi-video phantom retrain** rises in priority (it requires retraining the detector, since the champion saw all other phantom videos).
- Do not trumpet the combined headline; report per-domain, per-video, and both metric readings.

## Caveats

- Same canonical GT (V's yolo_only candidates_used) used for all three for comparability; AQ's multisource predictions on samples outside that GT set count as FP (mildly conservative for AQ).
- Numbers are internal proxies; official evaluator (2026-07-10) is authoritative.
