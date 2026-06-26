# Decision Record — EFF-Gap Calibration via Reproduced YOLOV

**Date:** 2026-06-17. **Headline:** the apparent "≈2× gap to EFF" is largely an **eval-ruler artifact**. On a common ruler, our champion is **at EFF level (mAP50)** and **above it (mAP50-95)**.

## Method
Reproduced YOLOV-S two-class on the video-disjoint train_clean (35,079 frames, leakage-safe), 24 epochs, converged (native COCOeval AP50 peaked ~0.126 @ epoch 11, then plateaued). Exported clean 14-col predictions on the SAME 824 phantom + 107 animal eval frames as the champion GT, scored with the same `evaluate_candidate_honest_baseline.py`. This makes YOLOV ⟷ champion apples-to-apples, and lets the paper's EFF/YOLOV numbers be projected onto our ruler.

## Numbers on our ruler (phantom per-case)

| method | mAP50 | mAP50-95 |
|---|---:|---:|
| reproduced YOLOV (24ep) | **0.131** | 0.039 |
| champion Stage2AQ | 0.130 | 0.050 |
| RT-DETR-R18 (converged) | 0.103 | 0.040 |
| paper YOLOV / EFF (official test) | 0.141 / 0.149 | — |

## Calibration
Eval-transfer ratio from YOLOV (the one method on both rulers): `EFF_ours = 0.1488 × (Y / 0.1411) = 1.0546·Y`.

- **mAP50 axis** (Y = 0.131): `EFF_ours ≈ 0.138`. Champion 0.130 is **~6% below** projected EFF — i.e. essentially **at EFF level**, NOT 2× below.
- **mAP50-95 axis** (Y = 0.039): `EFF_ours ≈ 0.041`. Champion AQ 0.050 is **~22% ABOVE** projected EFF → champion **beats** EFF on the strict axis.

The key empirical fact (independent of the projection): **our champion (0.130) matches a faithfully-reproduced, converged YOLOV (0.131)** on a common ruler, and on the strict IoU axis exceeds it. Since the paper places EFF only ~5% above YOLOV (14.88 vs 14.11), champion sits in the same band as the paper's best.

## Verdict
**The "2× to EFF" was a measurement mirage** — it compared our thin single-video proxy (0.13) to the paper's official-test number (0.169) across two different rulers. Calibrated onto one ruler:
- we are **not 2× behind**; we are **at EFF level (mAP50)** or **ahead (mAP50-95)**;
- the original RT-DETR "ceiling chase" to double 0.10→0.17 was chasing a gap that largely does not exist.

## Caveats
1. Thin validation: single phantom video (video_0, 824 GT) + GT-from-candidates (recall-optimistic). The **ratio cancels** much of this bias (it inflates Y and champion identically), which is why the ratio — not absolute Y — is the deliverable.
2. Reproduced-YOLOV is OUR training (may not equal the paper team's exact YOLOV), but it is a faithful, converged tiny-object video detector — a fair proxy.
3. Paper IoU axis not definitively pinned (Table V "AP"/"mAP" columns); hence the band over mAP50 vs mAP50-95. The official evaluator (2026-07-10) is authoritative.
4. Animal = 0.0 for ALL methods (domain skew, 291 animal train frames); calibration is phantom-only (INV-9: report per-domain). Human domain still untested.

## Implication for strategy
Stop treating "beat EFF" as a 2× moonshot. We are competitive on the phantom proxy. The higher-value pre-2026-07-10 work is now clearly: (a) trustworthy multi-video validation, (b) the untested human domain, (c) submission hardening — not squeezing a largely-illusory phantom gap. Keep YOLOV (0.131, Apache-2.0) as a viable submission candidate alongside Stage2AQ/AI; it also retires the AGPL-YOLO11 risk.
