# Decision: Task 1 `0.65` Dice Is Mostly a Thin-Line Exactness Problem

Date: 2026-06-02

## Context

The current seven-model local champion is `0.6536723455148790`, but full PNG
predictions are currently available for the earlier five-model full prediction
set:

`outputs/task1/stage5_submission_smoke/predict_full/predictions.csv`

That five-model set scores `0.6516511688167698`, close enough to diagnose the
main failure mode before regenerating seven-model prediction PNGs.

## Diagnostic Script

Added:

`scripts/task1/diagnose_prediction_quality.py`

The script computes:

- exact multiclass Dice over labels `[1, 2]`;
- binary foreground Dice after collapsing labels `1|2`;
- class-swap Dice after swapping predicted labels `1 <-> 2`;
- oracle per-image class-swap Dice;
- tolerance Dice at radii `1`, `2`, and `3` pixels;
- animal / phantom breakdowns;
- per-sample and top-case CSVs.

## Full Released-Eval Result

Run:

`outputs/task1/diagnostics/stage5_full_prediction_quality/summary.json`

Full released eval, `4691` samples:

- exact multiclass Dice: `0.6516511688167698`
- label_1 Dice: `0.6305547443921731`
- label_2 Dice: `0.6727475932413666`
- binary foreground Dice: `0.6197076656852551`
- binary minus multiclass: `-0.031943503131514665`
- oracle class-swap Dice: `0.6530648529795398`
- oracle class-swap gain: `+0.0014136841627699819`
- tolerance radius 1 multiclass Dice: `0.7969024383552233`
- tolerance radius 2 multiclass Dice: `0.8776655042940429`
- tolerance radius 3 multiclass Dice: `0.9200622639285359`

Domain breakdown:

- animal exact Dice: `0.7373800564302811`
- animal tolerance r2 Dice: `0.9272817356177415`
- phantom exact Dice: `0.6282472988197025`
- phantom tolerance r2 Dice: `0.8641203404645611`

## Balanced 1024 Result

Run:

`outputs/task1/diagnostics/stage5_balanced1024_prediction_quality/summary.json`

Balanced 1024, `512` animal + `512` phantom:

- exact multiclass Dice: `0.6796473943260101`
- binary foreground Dice: `0.6620236114396353`
- oracle class-swap gain: `+0.002192898453233017`
- tolerance r1 multiclass Dice: `0.8257669028521625`
- tolerance r2 multiclass Dice: `0.8962868630712062`
- tolerance r3 multiclass Dice: `0.9285031810356551`

## Interpretation

The main bottleneck is not label_1 / label_2 class swapping:

- full oracle class-swap gain is only about `+0.0014`;
- balanced 1024 oracle class-swap gain is only about `+0.0022`.

The main bottleneck is also not simply that multiclass is much harder than
binary foreground:

- binary foreground Dice is lower than multiclass macro Dice in this evaluator,
  meaning foreground collapse does not reveal a hidden `0.8x` localization
  score.

The dominant signal is thin-line exactness:

- exact full Dice: `0.6517`;
- 1px tolerant Dice: `0.7969`;
- 2px tolerant Dice: `0.8777`;
- 3px tolerant Dice: `0.9201`.

This means the model is often close to the annotated tool centerline, but strict
pixel-level overlap on 3-5 pixel-wide structures is causing large Dice loss.

## Decision

Stop treating `0.65` as only a backbone-capacity problem. Ordinary architecture
swaps are unlikely to jump directly to `0.8x` under strict exact Dice unless they
also improve sub-pixel/line-width alignment.

Prioritize:

1. shape/boundary/centerline-aware refinement;
2. skeleton/endpoint and distance-transform based losses;
3. post-processing that snaps, thins, thickens, or calibrates line width;
4. high-resolution inference and ROI refinement only if it improves exact
   alignment without shifting the line;
5. phantom-specific handling, because phantom exact Dice remains lower.

Still keep a real architecture check on the roadmap, but do not assume that a
generic ViT/Mamba swap alone solves the observed failure mode.

