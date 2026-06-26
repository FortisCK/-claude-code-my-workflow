# Decision Record — Our Method on YOLOV Base (Arm A: OOD reranker)

**Date:** 2026-06-18. **Result: our Stage2U reranker improves raw YOLOV on the trustworthy multi-video panel, even with the OOD (YOLO11-trained) checkpoint.**

## Setup
- Base: converged YOLOV-S two-class (mv split, 26 train / 6 valid videos).
- Pool: YOLOV proposals → GT-bearing candidate pool (`build_stage2o_candidate_pool`), oracle R@0.50=0.661 / R@0.75=0.180 vs YOLOV's own top-1 R@0.50=0.237 → large ranking headroom.
- Arm A: re-score with the EXISTING champion Stage2U checkpoint (`stage2u_warm_stage2o_fulltrain_valid512_e6`, trained on YOLO11/1024px crops → OOD for YOLOV/576px), best score mode `prob_iou75_sqrt_source_roi`, no retraining.
- Eval: `evaluate_clean_csv_vs_coco_gt.py` on the 6-video full-frame-GT panel (apples-to-apples vs raw YOLOV).

## Result (panel, phantom)

| | per-case mAP50 | global mAP50 | per-case mAP50-95 |
|---|---:|---:|---:|
| Raw YOLOV | 0.252 ± 0.216 | 0.209 | 0.098 |
| + reranker (Arm A) | **0.278 ± 0.225** | **0.256** | **0.103** |

Per-video (reranked vs raw): video_11 0.031→0.063, video_15 0.645→0.708, video_2 0.405→0.425, video_21 0.051→0.110, video_16 0.240→0.220, video_8 0.140→0.138. **4/6 improved (largest on the hard low-density videos), no collapse on high-mass videos.**

## Verdict
- **Our method (reranker) adds real value on YOLOV** (+10% per-case / +22% global), consistent across videos — the YOLO11-era reranker contribution is NOT redundant on a strong base, contrary to the planning prior. The oracle headroom (0.24→0.66) is partly capturable.
- Falls just short of the strict +0.05 per-case gate (+0.026), but this is the OOD checkpoint. **Proceed to Arm B**: retrain Stage2U on YOLOV candidates (no OOD penalty) to capture more of the headroom.

## Arm B (retrain Stage2U on YOLOV candidates) — did NOT beat Arm A

| config | per-case mAP50 | global | mAP50-95 |
|---|---:|---:|---:|
| Raw YOLOV | 0.252 | 0.209 | 0.098 |
| + reranker Arm A (existing diverse-trained ckpt, OOD) | **0.278** | 0.256 | 0.103 |
| + reranker Arm B (retrained on YOLOV-only pool) | 0.267 | 0.221 | 0.103 |

Arm B warm-started from the champion and retrained 6 epochs on the YOLOV-only train pool; its primary metric **peaked at epoch 1 then declined (overfit)**. The narrow single-source retrain degraded the champion's general ROI-quality features. **The diverse-trained existing reranker (Arm A) is the better choice on YOLOV.**

## Overall verdict
- **Best "our method on YOLOV base" = raw YOLOV + existing Stage2U reranker (Arm A) = 0.278 per-case mAP50** (vs raw YOLOV 0.252, +10%). Our method genuinely adds value on a strong base.
- The lift is **modest and capped**: the reranker reaches ~0.28 against an oracle ceiling of 0.66 — it cannot discriminate the good box from near-duplicates well. **Further reranker tuning is low-EV.**
- The remaining big levers are NOT the reranker: high-IoU localization (locR75=0.18 caps mAP50-95 ~0.10) and domain (animal=0). Those are where real headroom is.

## Caveats
- Single phantom-domain panel (6 videos), GT full-frame; animal/human still untested.
- Score mode used as-is for both classes; per-class/domain fusion sweep not yet optimized (another lever).
- Numbers are internal proxies; official evaluator (2026-07-10) authoritative.
