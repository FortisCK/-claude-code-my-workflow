# Plan: Task 1 Top-Version Roadmap

**Date:** 2026-05-21
**Status:** APPROVED
**Scope:** Upgrade Task 1 from strong MONAI baseline to top-version candidate

---

## Current Baseline

The current strongest verified baseline is:

- config: `configs/task1/monai_unet_dicece_full_512_overnight.yaml`
- model: MONAI 2D UNet
- loss: MONAI DiceCE
- training size: all released animal/phantom train samples, 18,758 frames
- train-time monitor eval: capped 512 released eval samples
- full released eval: all 4,691 released eval frames

Best checkpoint full released eval:

- DSC: 0.5303603258369163
- IoU/mIoU: 0.40486750742246314
- `label_1` Dice: 0.5917695853589093
- `label_2` Dice: 0.4689510663149234

This is a released-folder engineering benchmark only, not hidden-test or
leaderboard performance.

---

## Research Synthesis

The external research report points to a high-confidence diagnosis:

- the task is extreme foreground sparsity;
- the target structures are thin, long, and topology-sensitive;
- animal and phantom domains are imbalanced and visibly different;
- most public fluoroscopy work is binary, so multiclass catheter/guidewire
  results must be adapted carefully;
- plain architecture swaps are unlikely to be enough by themselves.

Therefore the next version should combine:

1. a stronger pretrained or library backbone;
2. class-wise thin-structure loss;
3. better spatial strategy than direct square resize;
4. domain-balanced training and foreground-aware sampling;
5. full released eval plus per-domain reporting.

---

## Candidate Priority

### Stage 0: Evaluation and Diagnostics Foundation

Before adding a top model, add reporting that makes model comparisons meaningful:

- full eval split by domain:
  - animal subset
  - phantom subset
- per-class metrics for `label_1` and `label_2`;
- prediction-vs-GT pixel counts for exported samples;
- optional connected-component and skeleton diagnostics if implementation cost is
  low.

Rationale: the current full eval hides domain imbalance. Phantom has 3,685 eval
frames while animal has 1,006.

### Stage 1: Loss/Resolution/Sampling Upgrade on Current MONAI Backbone

First implement upgrades that are independent of SegFormer installation:

- class-wise soft-clDice for `multiclass_012`;
- combined loss:
  `0.5 * CE + 1.0 * Dice + 0.3 * classwise_soft_clDice`;
- optional schedule:
  - warmup with DiceCE for early epochs;
  - enable clDice after the model has learned coarse localization;
- aspect-ratio-preserving resize and pad to 640 or 768;
- domain-balanced sampler for animal/phantom;
- foreground-aware crop/pad strategy if direct full-frame 768 is inefficient.

This should answer whether the current model is bottlenecked mainly by loss and
sampling rather than backbone capacity.

### Stage 2: Pretrained Segmentation Backbone

Install and use a mature library rather than implementing a large model by hand.

Preferred options:

- `segmentation-models-pytorch` if it supports the desired pretrained encoders
  in the active environment;
- `transformers` for `SegformerForSemanticSegmentation`;
- `mmsegmentation` only if dependency compatibility is acceptable.

Initial target:

- SegFormer-B2 or SegFormer-B3 style model;
- pretrained encoder when allowed and available;
- 3 output classes;
- 640 or 768 aspect-ratio-preserving input;
- Dice/CE + class-wise soft-clDice;
- same released train/eval discipline.

Fallback target if SegFormer integration is slow:

- pretrained encoder U-Net/FPN/DeepLabV3+ through SMP;
- compare with MONAI UNet + clDice under the same data pipeline.

### Stage 3: Geometry-Aware Fine-Tuning

Only after Stage 1/2 are stable:

- shape-sensitive loss or signed-distance/boundary loss;
- small weight, late fine-tuning;
- compare against clDice-only result.

Rationale: these losses are more expensive and easier to overfit or destabilize.

### Stage 4: Inference and Postprocess

After a strong checkpoint exists:

- test-time augmentation:
  - horizontal flip only if visually and anatomically acceptable;
  - multi-scale inference at 512/640/768 if feasible;
- connected-component filtering for tiny false positives;
- class-specific morphology/skeleton pruning only after overlay review.

### Stage 5: External/Public Data and Human Binary Holdout

Second-stage research track:

- public fluoroscopy/guidewire datasets for binary foreground pretraining;
- synthetic or phantom pretraining;
- human binary masks as foreground-only auxiliary pretraining only if challenge
  rules allow and the usage is documented.

Do not mix binary human masks into multiclass training without a documented
auxiliary objective.

---

## First Implementation Proposal

Implement Stage 0 + Stage 1 first.

Why:

- no new external package is required;
- it isolates the effect of thin-structure loss and data strategy;
- it produces a better baseline even if SegFormer installation becomes slow;
- it gives per-domain metrics before we invest in a larger model.

Concrete first experiment:

- base model: MONAI UNet from current best config;
- input: `640x640` or `768x768` with aspect-ratio preserve + pad;
- sampler: animal/phantom domain-balanced batches;
- loss:
  - DiceCE warmup;
  - DiceCE + class-wise soft-clDice main phase;
- epochs: 120 or overnight budget;
- output:
  `outputs/task1/monai_unet_cldice_domain_balanced_640` or
  `outputs/task1/monai_unet_cldice_domain_balanced_768`.

Then run full 4,691-frame eval and compare against:

- MONAI UNet DiceCE best DSC: 0.5303603258369163.

Success threshold:

- worthwhile: full eval DSC >= 0.55;
- strong: full eval DSC >= 0.57;
- very strong for this iteration: full eval DSC >= 0.60;
- must also improve or at least not damage `label_2` Dice.

---

## Package/Environment Notes

Checked `cardiac-diffusion`:

- `monai`: available
- `timm`: available
- `transformers`: not installed
- `segmentation_models_pytorch`: not installed
- `mmseg/mmcv`: not installed

If Stage 2 is approved, install only one mature segmentation stack first.
Recommended first package:

- `segmentation-models-pytorch` for practical encoder-decoder experiments; or
- `transformers` if we commit specifically to SegFormer.

The install should be recorded in the environment notes and verified with a
small smoke run before long training.

---

## Approval Needed

User approval is needed before implementation because this plan changes the data
pipeline, sampler, losses, and potentially installs new model libraries.

Approved by user on 2026-05-21 for Stage 0 + Stage 1 implementation.
