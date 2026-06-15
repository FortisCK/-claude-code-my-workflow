# Plan: Task 1 Stage 2 Architecture Shootout

**Date:** 2026-05-21
**Status:** APPROVED
**Scope:** Move from Stage 1 UNet training-strategy exploration to strong
pretrained architecture screening.

## Decision

Stop the Stage 1 `monai_unet_cldice_domain_balanced_640` run and move to Stage
2 architecture screening.

Rationale:

- Stage 1 reached epoch 40 without beating the reference MONAI UNet baseline.
- Best Stage 1 full-eval Dice was epoch 5: `0.5240289630882011`.
- Reference MONAI UNet DiceCE 512 best remains: `0.5303603258369163`.
- The Stage 1 combination improved neither overall Dice nor stability enough to
justify a 120-epoch run.

## Reference Anchor

Current reference to beat:

- config: `configs/task1/monai_unet_dicece_full_512_overnight.yaml`
- checkpoint: `outputs/task1/monai_unet_dicece_full_512_overnight/best_checkpoint.pt`
- full released eval Dice: `0.5303603258369163`
- `label_1` Dice: `0.5917695853589093`
- `label_2` Dice: `0.4689510663149234`
- animal Dice: `0.6240706427597901`
- phantom Dice: `0.5047775364680124`

## Stage 2 Principle

Do not transfer every Stage 1 change to a Transformer/ViT-style model.

For the first architecture shootout, keep the stable baseline training setup:

- input: 512 direct resize
- loss: DiceCE-style multiclass segmentation loss
- sampler: normal random shuffle, no domain balancing
- eval: full released eval with overall, per-label, animal, and phantom metrics

Only change the model architecture first. If a new architecture beats the
reference anchor, apply focused ablations on that winner.

## Candidate Models

First wave:

1. `segmentation-models-pytorch` Unet or FPN with pretrained encoders:
   - EfficientNet-B3
   - ConvNeXt-Tiny
   - ResNet50
2. SegFormer-B2 through `transformers`, if integration is straightforward.

Fallback:

- If `transformers`/SegFormer integration is slow, run SMP candidates first and
  add SegFormer after the first winner/loser signal.

## Screening Rule

- `< 0.53`: not better than reference; do not deep-tune.
- `0.53-0.55`: possible candidate; inspect `label_2` and phantom.
- `0.55-0.57`: main candidate.
- `> 0.57`: winner; run resolution/loss/sampler/TTA ablations on it.

## Follow-Up Ablations Only On Winners

After a winner exists:

- 512 direct vs 640 aspect-pad
- DiceCE vs DiceCE + light clDice
- normal sampler vs domain-balanced/weighted sampler
- TTA and lightweight postprocessing

## Verification

For each candidate:

- smoke train for 1 epoch on a small sample;
- run at least a short full-eval training job;
- record metrics in `outputs/task1/<experiment>/metrics.json`;
- compare against the reference anchor using overall, per-label, animal, and
  phantom metrics.

Approved by user on 2026-05-21 after agreeing to prioritize architecture
shootout over further UNet ablation.
