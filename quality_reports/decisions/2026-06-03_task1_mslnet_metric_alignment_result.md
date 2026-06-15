# Decision: MSLNet-Style Metric Alignment for Current Task 1 Champion

Date: 2026-06-03

## Question

Can our result be evaluated with the same metric style as the MSLNet paper, so
the comparison against their CathAction numbers is not distorted by our
three-class macro Dice evaluator?

## Implemented Evaluator

Added:

- `src/cathaction/metrics/mslnet_style.py`
- `scripts/task1/evaluate_mslnet_style.py`
- `tests/test_task1_mslnet_style_metrics.py`

The evaluator collapses all non-background labels into binary foreground and
computes:

- standard binary Dice;
- standard binary IoU;
- precision / recall / F1 at distance radii 0, 1, 2, 3, and 4 pixels;
- AHD as the average of prediction-to-target and target-to-prediction nearest
  foreground-pixel distances.

This matches the MSLNet paper's segmentation metric definition more closely
than our earlier multiclass macro Dice or dilation-based tolerant Dice
diagnostic.

## Current Champion Evaluated

Prediction set:

`outputs/task1/stage9a_seven_model_add_both_010_010/remove_small_min32/predictions.csv`

Output:

`outputs/task1/diagnostics/stage9a_seven_model_remove_small_min32_mslnet_style/eval_mslnet_style.json`

Per-sample diagnostics:

`outputs/task1/diagnostics/stage9a_seven_model_remove_small_min32_mslnet_style/per_sample_mslnet_style.csv`

## Result

Overall sample-mean binary foreground metrics:

| Metric | Ours |
| --- | ---: |
| Dice | `0.6215624175077626` |
| IoU | `0.4605049121513648` |
| AHD | `1.3713229393268282` |
| F1 radius 0 | `0.6215624175077626` |
| F1 radius 1 | `0.7607678985630797` |
| F1 radius 2 | `0.8514899720219553` |
| F1 radius 3 | `0.911577665104627` |
| F1 radius 4 | `0.944560505299149` |

Domain split:

| Domain | Dice | IoU | AHD | F1 r3 |
| --- | ---: | ---: | ---: | ---: |
| animal | `0.738274705153863` | `0.5897981116933185` | `0.4904227573235063` | `0.9827937447392838` |
| phantom | `0.5897001213416904` | `0.4252080441081611` | `1.6118074937624705` | `0.892135771994053` |

## Comparison to MSLNet Paper

MSLNet CathAction segmentation reports:

| Metric | MSLNet | Ours | Difference |
| --- | ---: | ---: | ---: |
| Dice | `0.6251` | `0.6216` | `-0.0035` |
| IoU | `0.4658` | `0.4605` | `-0.0053` |
| AHD | `1.5` | `1.37` | `+0.13` lower is better |
| F1 r0 | `0.6266` | `0.6216` | `-0.0050` |
| F1 r1 | `0.7679` | `0.7608` | `-0.0071` |
| F1 r2 | `0.8591` | `0.8515` | `-0.0076` |
| F1 r3 | `0.9305` | `0.9116` | `-0.0189` |
| F1 r4 | `0.9492` | `0.9446` | `-0.0046` |

The comparison is now close to apples-to-apples for binary segmentation, but
not identical statistically because MSLNet reports mean and standard deviation
over four independent runs while our current result is one ensemble.

## Interpretation

Under MSLNet-style binary metrics, our current champion is close to MSLNet on
strict Dice and IoU but below MSLNet on distance-tolerant F1, especially at the
3-pixel threshold. AHD is slightly better for our model.

The domain split shows the main weakness is phantom:

- animal F1 r3 is very high at `0.9828`;
- phantom F1 r3 is `0.8921`.

This means the next improvement route should not be another generic global loss
run. The higher-value direction is phantom-focused foreground recovery and
coarse-to-fine / ROI refinement, while preserving the current Stage 9A ensemble
as the fallback.

## Verification

- `python3 -m py_compile src/cathaction/metrics/mslnet_style.py scripts/task1/evaluate_mslnet_style.py tests/test_task1_mslnet_style_metrics.py` passed.
- Manual one-pixel-shift sanity check returned Dice `0.0`, IoU `0.0`, AHD `1.0`, F1 r0 `0.0`, F1 r1 `1.0`.
- Full released-eval MSLNet-style run completed over `4691` samples.
- `pytest` could not be run because the available local environments checked
  during this turn did not have pytest installed.
