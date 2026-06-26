# Decision Record — RT-DETRv2-R18 Bare Base (Round 1, 640px) Result

**Date:** 2026-06-15
**Bake-off:** `2026-06-15_task2_strongest_base_bakeoff.md` — Gate A (bare base, no NWD/P2).
**Config:** RT-DETRv2-R18 (PekingU/rtdetr_v2_r18vd), 2-class, fine-tuned on video-disjoint train_clean (35,079 frames), 6 epochs, batch 12, lr 1e-4, **640px** (processor default). Loss converged 421 → 7.2.
**Eval:** `evaluate_candidate_honest_baseline.py` on the SAME frames as the champion GT (V's yolo_only candidates_used: 824 phantom + 107 animal).

## Result — Gate A FAIL

| metric | RT-DETR-R18 bare @640 | champion (V/AQ) | EFF (paper) |
|---|---:|---:|---:|
| phantom per-case mAP50 | **0.0276** | 0.096–0.130 | 0.169 (AP-mean) |
| animal per-case mAP50 | **0.000** | ~0.50 | — |
| combined per-case mAP50 | 0.0138 | 0.30–0.31 | — |

Gate A threshold was ≥~0.13. **0.028 is well below → NO-GO for this configuration.**

## Diagnosis (verified, not a coordinate bug)

Coordinate space confirmed correct: for video_0_00000102, GT (636,332,680,376) on a 1107×842 image, and RT-DETR's top box (630.8,329.5,679.4,375.4) nearly perfectly matches. So the failure is real, with two causes:

1. **Resolution.** Tip ≈ 2% of width ≈ 22px in a 1107px image → ~13px at 640px. Diagnostics over 824 phantom frames:
   - localization ceiling (best-IoU any det): **R@0.50 = 0.496, R@0.75 = 0.102** (tight localization is poor).
   - best-IoU among correct-class dets: R@0.50 = 0.444.
2. **Score/ranking collapse.** Top-SCORE detection IoU **median 0.147, mean 0.199**, only **12.9%** ≥0.5, class-correct 53.5%. The correct box is found in ~50% of frames but the model's confidence does not rank it first → AP collapses. Scores are flat and tiny (~0.02).
3. **Animal = 0.0**: domain skew (train 35,084 phantom vs 291 animal) + the above.

## Interpretation

This **validates the bake-off's core prediction**: a bare generic SOTA base does NOT clear the floor on its own; the EFF-beating levers are tip-scale resolution + P2 + NWD + score calibration. The 640px bare-base test under-tested RT-DETR (640px is too coarse for 13px tips) — it is informative ("RT-DETR-R18 @640 is not viable; localization ceiling mediocre") but does NOT settle whether RT-DETR + tip-scale resolution is viable.

## Next (decided)

Single-variable follow-up: **retrain RT-DETR-R18 at 1024px** (the tip-scale the plan specified). Decisive test of the resolution hypothesis:
- if R@0.50 / R@0.75 jump materially at 1024px → RT-DETR + graft is viable, proceed to Gate B (P2/NWD/score);
- if they barely move → the base is wrong; escalate to RTMDet/Co-DETR or reconsider.
~2–3 GPU-hr. Ranking/score-calibration is a separate, later fixable lever (reranker layer or training tweak), held constant this round.
