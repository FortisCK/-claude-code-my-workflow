# Task2 Result: Stage2U Source-Filtered Inference Diagnostic

Date: 2026-06-13
Status: completed

## Question

After Stage2P refined candidates failed to improve full-validation AP, should inference use all candidate sources, or should some candidate sources be filtered/downweighted?

## Implementation

Added:

- `scripts/task2/evaluate_stage2u_source_filter.py`

The script reads saved Stage2U prediction rows and candidate rows, then recomputes mAP after source filtering, refined-source downweighting, and optional per-sample top-k filtering.

Verification:

```bash
PYTHONDONTWRITEBYTECODE=1 python3 -m py_compile scripts/task2/evaluate_stage2u_source_filter.py
```

Diagnostic run:

```bash
PYTHONDONTWRITEBYTECODE=1 python3 scripts/task2/evaluate_stage2u_source_filter.py \
  --run-dir outputs/task2/stage2u_quality_ranker/stage2u_warm_stage2o_fulltrain_fullvalid_stage2p_augmented_eval \
  --output-json outputs/task2/stage2u_quality_ranker/stage2u_warm_stage2o_fulltrain_fullvalid_stage2p_augmented_eval/source_filter_diagnostic.json
```

Note:

- This diagnostic uses stored candidate-vs-GT IoU and the saved prediction scores.
- It is exact for the current one-GT-per-sample Task2 validation files, but it should still be treated as a diagnostic until wired into the official submission/evaluation path.

## Results

Using score mode `blend_rank_decay_roi`:

| Policy | valid_combined mAP50 | valid_combined mAP50-95 | valid_phantom mAP50 | valid_phantom mAP50-95 | valid_animal mAP50 | valid_animal mAP50-95 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| all augmented sources | `0.1493` | `0.0308` | `0.0527` | `0.0216` | `0.4962` | `0.0537` |
| original sources | `0.1875` | `0.0414` | `0.0891` | `0.0342` | `0.5004` | `0.0502` |
| YOLO only | `0.1883` | `0.0416` | `0.0899` | `0.0345` | `0.4993` | `0.0500` |
| refined only | `0.1269` | `0.0341` | `0.0845` | `0.0314` | `0.2635` | `0.0304` |
| all sources, refined x0.2 | `0.1875` | `0.0414` | `0.0890` | `0.0342` | `0.5003` | `0.0502` |
| all sources, refined x0.05 | `0.1876` | `0.0414` | `0.0891` | `0.0343` | `0.5002` | `0.0501` |

## Interpretation

The current candidate stack benefits more from reducing noisy sources than from appending all available sources.

Main observations:

- `YOLO only` is slightly better than the current original mixed-source policy on valid_combined and valid_phantom.
- The gain is small: valid_combined mAP50-95 `0.0414 -> 0.0416`.
- `stage2p_refined` is not useless, but as a full-volume source it hurts ranking precision.
- Downweighting refined candidates mostly recovers the original mixed-source score; it does not create a meaningful gain.
- Task1 geometry candidates are not helping the current Stage2U score mode enough to justify their false-positive load.

## Decision

Treat source filtering as a real but low-ceiling optimization.

For the current Stage2U pipeline:

- keep `yolo_stage2l`-only inference as a lightweight candidate selection baseline;
- do not adopt all augmented sources;
- do not continue Stage2P refined-source training as the main path;
- do not expect source filtering alone to solve Task2.

## Next Direction

The next high-leverage work should be candidate generation redesign, not another Stage2U score-mode sweep.

Prioritized directions:

1. Build a source-filtered official evaluation/submission path so `yolo_stage2l`-only can be reproduced without post-hoc diagnostics.
2. Revisit candidate generation for phantom, where even the best current policy is only about `0.0345` mAP50-95.
3. Prefer structured proposal generation over adding more generic detector boxes:
   - Task1 mask / centerline proposals;
   - heatmap or keypoint interaction localizer;
   - temporal consistency proposal filtering;
   - phantom-specific localizer.
