# Decision Record — RT-DETRv2-R18 @640 Ceiling (recipe-fixed)

**Date:** 2026-06-16/17
**Bake-off S0** (`2026-06-15_task2_strongest_base_bakeoff.md`); recipe from the research panel (warmup-flat LR, EMA, 3 param-groups, num_queries 60, light aug, in-loop eval).

## Full convergence curve (phantom per-case mAP50, video_0 824-frame proxy)

| stage | mAP50 | note |
|---|---:|---|
| 6ep, broken recipe (OneCycle, no EMA) | 0.028 | original |
| recipe-v2 e4 / e8 / e12 | 0.038 / 0.067 / 0.076 | climbing |
| continuation e16 / **e20** / e24 / e28 / e32 | 0.087 / **0.103** / 0.097 / 0.064 / 0.053 | **peak 0.103**, then overfits → early-stop |

## Verdict

- **Recipe fix worked: 0.028 → 0.103 (~3.7x).** Confirms the original failure was largely recipe (OneCycleLR + no EMA), not the architecture. The warmup-flat + EMA + param-group + num_queries changes were the lever.
- **Confidence stayed ~0.02 throughout while mAP climbed to 0.103** → the flat confidence was a VFL/1-box-per-frame artifact, NOT a bug. The original confidence-based gate was mis-designed and was corrected to a mAP-trajectory gate.
- **Ceiling of R18@640 ≈ 0.103 = roughly MATCHES the champion (0.10–0.13 low end), does NOT beat it, and is far below EFF (0.169).** More epochs hurt (overfit). The remaining gap is **structural**: stride-8 features vs ~13px tips → tight-localization capped (%top-box IoU≥0.5 plateaued ~0.19).

## Implication

To beat EFF from 0.103 needs ~2x. The plan's S1 (resolution 1024, recipe-fixed) + S2 (P2 stride-4) + S3 (NWD) target the structural limit, but each is a multi-hour, uncertain step, and the validation is a single video (high-variance proxy). **A ~2x gain by stacking 3 uncertain levers on a thin proxy, weeks before the official evaluator (2026-07-10) resolves the metric axis, is low expected value.** See the strategy reassessment that follows this record.

## Artifacts
- `outputs/task2/rtdetrv2_r18_recipe_v2/` (e12), `outputs/task2/rtdetrv2_r18_recipe_v2_cont/` (best EMA @0.103), `train_history.json` in each.
- Scripts: `train_rtdetrv2_task2.py` (recipe), `export_rtdetrv2_predictions.py`, `evaluate_candidate_honest_baseline.py`.
