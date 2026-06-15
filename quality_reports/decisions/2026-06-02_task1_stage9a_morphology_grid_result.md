# Decision: Use Small-Component Removal as a Cheap Task 1 Postprocess Candidate

Date: 2026-06-02

## Context

The Task 1 diagnostic pass showed that the current `0.65` Dice regime is mainly
a thin-line exactness problem: the five-model full prediction improves from
`0.6516511688167698` exact Dice to `0.7969024383552233` with a one-pixel
tolerance and `0.8776655042940429` with a two-pixel tolerance.

Stage 9A tested whether simple hard-mask morphology can recover some exact
Dice without retraining.

## Run

Prediction set:

`outputs/task1/stage5_submission_smoke/predict_full/predictions.csv`

Command:

```bash
PYTHONPATH=/home/mingzhang/cathaction/src:/home/mingzhang/cathaction \
/home/mingzhang/miniconda3/envs/cardiac-diffusion/bin/python -u \
scripts/task1/search_morphology_postprocess.py \
  --eval-manifest configs/task1/splits/released_eval.csv \
  --predictions-csv outputs/task1/stage5_submission_smoke/predict_full/predictions.csv \
  --output-dir outputs/task1/stage9a_morphology_grid/stage5_full_compact \
  --top-k 20 \
  --preset compact \
  --write-best-predictions
```

Artifacts:

- search summary:
  `outputs/task1/stage9a_morphology_grid/stage5_full_compact/summary.json`
- candidate table:
  `outputs/task1/stage9a_morphology_grid/stage5_full_compact/candidate_metrics.csv`
- best predictions:
  `outputs/task1/stage9a_morphology_grid/stage5_full_compact/best_predictions/predictions.csv`
- independent eval:
  `outputs/task1/stage9a_morphology_grid/stage5_full_compact/best_predictions_eval.json`
- fixed postprocess script:
  `scripts/task1/apply_morphology_postprocess.py`
- fixed postprocess full output:
  `outputs/task1/stage9a_morphology_grid/stage5_full_fixed_remove_small_min32/predictions.csv`
- fixed postprocess full eval:
  `outputs/task1/stage9a_morphology_grid/stage5_full_fixed_remove_small_min32/eval.json`
- seven-model rerun script for when CUDA is restored:
  `scripts/task1/run_stage9a_seven_model_postprocess.sh`

## Result

Best compact-grid candidate:

- candidate: `remove_small_min32`
- operation: remove connected components smaller than `32` pixels from the hard
  prediction mask
- no dilation, erosion, opening, or closing

Metrics:

| Metric | Baseline | Best | Delta |
| --- | ---: | ---: | ---: |
| mean Dice | `0.6516511688167698` | `0.6537494144529713` | `+0.0020982456362015` |
| label_1 Dice | `0.6305547443921731` | `0.6319584963034739` | `+0.0014037519113008` |
| label_2 Dice | `0.6727475932413666` | `0.6755403326024687` | `+0.0027927393611021` |
| animal Dice | `0.7373800564302811` | `0.7404724102193491` | `+0.0030923537890679` |
| phantom Dice | `0.6282472988197025` | `0.6300741542790294` | `+0.0018268554593268` |

The independent evaluator exactly matched the morphology-search metrics:

- mean Dice: `0.6537494144529713`
- IoU / mIoU: `0.5314944396606959`
- pixel accuracy: `0.9947703806932926`

The standalone fixed postprocess script was also validated on the full
five-model prediction set and matched the same metrics exactly. This script
does not read evaluation masks and is suitable for hidden-test inference after
the candidate has been fixed.

## Interpretation

This is a real but small gain. The winning candidate is not line thickening,
line thinning, or gap closing; it is small-component suppression. That means
the current ensemble still loses most of its Dice to exact alignment and line
shape, but it also produces enough tiny false-positive fragments that removing
components smaller than `32` pixels improves both labels and both domains.

The result is worth keeping because it is cheap, deterministic, and improves
animal and phantom simultaneously. It is not enough to call Stage 9A a major
method step.

## Decision

Keep `remove_small_min32` as the first fixed postprocessing candidate for final
Task 1 submission packaging.

Do not expand the morphology grid yet. The compact grid already shows that
simple morphology is only a small correction, so the next meaningful work should
return to exact-alignment improvements: boundary/centerline loss, high-resolution
ROI refinement, and better phantom hard-case handling.

## Next Action

Apply the same `remove_small_min32` postprocess to the current seven-model
champion once CUDA is visible again. This was completed after confirming that
the host GPU is available outside the Codex sandbox.

Seven-model artifacts:

- raw predictions:
  `outputs/task1/stage9a_seven_model_add_both_010_010/raw_predictions/predictions.csv`
- raw eval:
  `outputs/task1/stage9a_seven_model_add_both_010_010/raw_predictions/eval.json`
- postprocessed predictions:
  `outputs/task1/stage9a_seven_model_add_both_010_010/remove_small_min32/predictions.csv`
- postprocessed eval:
  `outputs/task1/stage9a_seven_model_add_both_010_010/remove_small_min32/eval.json`

Seven-model result:

| Metric | Raw seven-model | `remove_small_min32` | Delta |
| --- | ---: | ---: | ---: |
| mean Dice | `0.6536723455148790` | `0.6551964758686204` | `+0.0015241303537414` |
| label_1 Dice | `0.6327836065361310` | `0.6339706899994402` | `+0.0011870834633092` |
| label_2 Dice | `0.6745610844936268` | `0.6764222617378007` | `+0.0018611772441739` |
| animal Dice | `0.7394416932476355` | `0.7423225909297140` | `+0.0028808976820785` |
| phantom Dice | `0.6302574299601563` | `0.6314111646741944` | `+0.0011537347140381` |

## GPU Visibility Note

The first CUDA check failed inside the normal Codex sandbox:

- `nvidia-smi` could not communicate with the driver.
- `/dev/nvidia*` was not visible.
- in `cardiac-diffusion`, `torch.cuda.is_available()` returned `False`.

The host GPU was healthy. Running `nvidia-smi` outside the sandbox showed:

- `NVIDIA RTX 6000 Ada Generation`
- `49140 MB` total memory
- about `38 MB` used before inference

Interpretation: this was a sandbox device-visibility issue, not a machine GPU
failure. GPU jobs that need CUDA should be run through approved outside-sandbox
commands in this Codex environment.
