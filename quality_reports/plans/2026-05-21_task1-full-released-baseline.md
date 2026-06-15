# Plan: Task 1 Full Released-Split Baseline

**Date:** 2026-05-21
**Status:** COMPLETED
**Scope:** Train the first complete Task 1 segmentation baseline on all released animal/phantom data

---

## Goal

Run a complete, reproducible Task 1 segmentation baseline using all currently
released animal/phantom training and evaluation samples:

- train: `released_train.csv` = 18,758 samples
- eval: `released_eval.csv` = 4,691 samples
- exclude: `human_holdout.csv` for now, because its masks are binary and do not
  preserve the same `0/1/2` multiclass semantics.

This is not a hidden-test estimate. It is a released-folder baseline for
engineering validation, qualitative inspection, and future model comparisons.

---

## Proposed Configuration

- model: `TinyUNet`
- label mode: `multiclass_012`
- working label interpretation:
  - `0`: background
  - `1`: provisional catheter
  - `2`: provisional guidewire
- image size: `512x512`
- loss: weighted cross entropy
- class weights: `[0.02, 1.0, 2.0]`
- train samples: all 18,758 released train samples
- monitor eval samples during training: 512 released eval samples
- final eval samples after training: all 4,691 released eval samples
- epochs: 15
- batch size: 64
- device: CUDA through the `pointdet` environment
- checkpointing:
  - save `best_checkpoint.pt` by `eval.dice`
  - save final `checkpoint.pt`
- output directory:
  `outputs/task1/baseline_unet_multiclass_mild_weighted_full_512`

Rationale:

- `256x256` runs are useful for quick pilots, but catheter/guidewire masks are
  very thin. A `512x512` baseline should preserve more line structure.
- The mild class weights avoided both prior failure modes:
  - unweighted CE predicted no `label_2` pixels in the sampled outputs;
  - inverse-frequency CE severely over-predicted `label_2`.
- TinyUNet is intentionally simple. It gives a stable reference before adding
  Dice loss, pretrained encoders, or larger segmentation backbones.

---

## Execution Steps

1. Add a full-run config under `configs/task1/`.
2. Run `py_compile`, `pytest`, and `git diff --check`.
3. Start the full GPU training run in `pointdet`.
4. Monitor checkpoints using a capped 512-sample eval subset during training.
5. Evaluate the best checkpoint on all 4,691 released eval samples.
6. Export a representative prediction subset from the best checkpoint.
7. Generate qualitative overlays and diagnostic sheets.
8. Compute prediction-vs-GT pixel distributions for exported samples.
9. Record results in the session log and mark this plan complete.

---

## Verification

Required:

- `conda run -n cathaction-task1 python -m py_compile ...`
- `conda run -n cathaction-task1 pytest`
- `git diff --check`
- full training writes:
  - `metrics.json`
  - `best_checkpoint.pt`
  - `checkpoint.pt`
  - `resolved_config.json`
- best-checkpoint evaluation writes:
  - `eval_best_full.json`
- saved prediction evaluation reports:
  - matched predictions equals exported predictions
  - no missing sample IDs

Expected review:

- Compare full-run DSC/IoU against the 512-sample pilot, but do not call it a
  hidden-test score.
- Inspect overlays for common failures:
  - missing distal guidewire;
  - over-thick catheter masks;
  - confusion with vessel/contrast structures;
  - domain-specific failure on animal vs phantom frames.

---

## Runtime Adjustment

The first launch used batch size 8. `nvidia-smi` showed only about 4.0 GB used
out of 49.1 GB on the RTX 6000 Ada, so the run was stopped before completing
epoch 1 and the approved full baseline was adjusted to batch size 64.

The second launch used `conda run` with default output capture, which is not
suitable for long training because intermediate epoch logs are hidden. It was
stopped and the run was adjusted to:

- `conda run --no-capture-output`
- 15 epochs instead of 30 for the first complete reference run
- 512-sample eval during training, followed by one full 4,691-sample eval of
  `best_checkpoint.pt`

During the accepted `512x512`, batch-size-64 run, `nvidia-smi` reported about
29,506 MiB used out of 49,140 MiB with 100% GPU utilization. This leaves some
headroom, but a true original-resolution run still needs a padded or bucketed
collate path because the animal/phantom frames have heterogeneous sizes.

---

## Results

Completed on 2026-05-21.

Training:

- train samples: 18,758
- capped monitor eval samples during training: 512
- final full eval samples: 4,691
- best checkpoint: epoch 14 by capped `eval.dice`
- final checkpoint: epoch 15

Best checkpoint, full released eval:

- checkpoint: `outputs/task1/baseline_unet_multiclass_mild_weighted_full_512/best_checkpoint.pt`
- eval file: `outputs/task1/baseline_unet_multiclass_mild_weighted_full_512/eval_best_full.json`
- DSC: 0.4437008138855879
- IoU/mIoU: 0.3399198392117686
- per-class DSC:
  - `label_1`: 0.442940108746356
  - `label_2`: 0.4444615190248199
- pixel accuracy: 0.9904373834138739

Final checkpoint, full released eval:

- checkpoint: `outputs/task1/baseline_unet_multiclass_mild_weighted_full_512/checkpoint.pt`
- eval file: `outputs/task1/baseline_unet_multiclass_mild_weighted_full_512/eval_final_full.json`
- DSC: 0.271676945781372
- IoU/mIoU: 0.17325436992533275
- per-class DSC:
  - `label_1`: 0.3905182702943531
  - `label_2`: 0.15283562126839095
- pixel accuracy: 0.9850121024218583

Saved prediction subset from the best checkpoint:

- predictions: 64
- matched predictions: 64
- missing sample IDs: 0
- saved-prediction DSC: 0.5000091709365408
- saved-prediction IoU/mIoU: 0.3914642801116009
- per-class saved-prediction DSC:
  - `label_1`: 0.5299399969097738
  - `label_2`: 0.4700783449633079
- prediction pixels at prediction resolution:
  - label 0: 16,581,789
  - label 1: 165,449
  - label 2: 29,978
- resized GT pixels at prediction resolution:
  - label 0: 16,660,928
  - label 1: 93,908
  - label 2: 22,380

Qualitative outputs:

- overlays: `outputs/task1/baseline_unet_multiclass_mild_weighted_full_512/best_overlays/`
- contact sheet:
  `outputs/task1/baseline_unet_multiclass_mild_weighted_full_512/best_overlays/contact_sheet.png`

Interpretation:

- The full 512 baseline is usable as the first reference run.
- The best checkpoint is materially better than the final checkpoint, so
  checkpoint selection matters.
- The two full-eval class DSC values are balanced, but the 64-sample pixel
  distribution and overlays show over-segmentation, especially for `label_1`.
- Next model work should focus on improving boundary/thin-structure fidelity:
  likely CE+Dice loss, pretrained encoder/backbone, and eventually true
  original-resolution padding/bucketing.

---

## Stop / Adjust Criteria

Stop and adjust instead of blindly finishing if:

- GPU OOM occurs at `512x512` batch size 64. First adjustment: batch size 32.
- Training becomes clearly unstable or outputs all background for multiple
  epochs. First adjustment: lower learning rate to `5e-4`.
- Eval time becomes impractically long. First adjustment: keep training full,
  but evaluate every epoch on a documented capped eval subset, then run one
  full eval at the best/final checkpoint.

---

## Approval Needed

User approved the plan on 2026-05-21 before config creation and full GPU
training launch.
