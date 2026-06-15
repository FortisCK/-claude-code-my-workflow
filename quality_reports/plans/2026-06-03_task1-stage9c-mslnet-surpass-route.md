# Task 1 Stage 9C MSLNet-Surpass Route

Date: 2026-06-03

Status: proposed

## Goal

Use the MSLNet paper as a direct engineering target and push the current Stage
9A champion past MSLNet on CathAction-style binary segmentation metrics while
preserving the official multiclass Task 1 score.

## Current Gap

Current Stage 9A champion, evaluated with MSLNet-style binary foreground
metrics:

- Dice: `0.6215624175077626`
- IoU: `0.4605049121513648`
- AHD: `1.3713229393268282`
- F1 r3: `0.911577665104627`

MSLNet paper CathAction segmentation:

- Dice: `0.6251`
- IoU: `0.4658`
- AHD: `1.5`
- F1 r3: `0.9305`

The important diagnostic is r3 precision / recall:

- ours precision r3: `0.8663946640069181`
- ours recall r3: `0.9677293936026353`
- MSLNet precision r3: `0.9152`
- MSLNet recall r3: `0.9462`

So the main gap is not recall. Our model predicts too much foreground,
especially in phantom, and needs better foreground precision without losing
too much recall.

## MSLNet Ideas Worth Reusing

Do not copy the whole architecture blindly. The paper's useful ideas are:

1. Coarse-to-fine segmentation.
2. Patch-level foreground gating.
3. Fine loss restricted to foreground-promising patches.
4. Explicit segmentation metrics that tolerate small annotation offsets but
   punish off-structure false positives.

The exact MSLNet gain on CathAction is modest, and the paper's own ablation
says MSL is not materially helpful on CathAction segmentation. The more useful
transfer is the coarse-to-fine precision control.

## Proposed Experiments

### 9C.1 MSL-style precision calibration

Use the existing Stage 9A soft probabilities if available; otherwise regenerate
probability maps for the current ensemble. Search foreground confidence and
connected-component thresholds against MSLNet-style F1 r3 and Dice.

Promotion target:

- MSLNet-style Dice >= `0.6251`; or
- F1 r3 gain >= `+0.01` without multiclass Dice falling below `0.652`.

### 9C.2 Phantom-focused false-positive analysis

Use the per-sample MSLNet-style CSV to identify phantom cases with:

- high recall and low precision;
- high AHD;
- large prediction pixel count relative to target.

Generate overlays/contact sheets for the worst cases. Classify errors into:

- over-thick line;
- parallel false line / vessel or bone edge;
- detached small components;
- wrong foreground extent;
- label_1 / label_2 confusion.

### 9C.3 Coarse-to-fine ROI refiner

Train a small high-resolution patch refiner around Stage 9A foreground
candidates and hard negative line-like structures.

Inputs:

- original X-ray crop;
- Stage 9A foreground probability crop;
- optional predicted label crop.

Outputs:

- refined binary foreground gate;
- optional three-class correction head.

Loss:

- Dice + BCE for binary foreground;
- multiclass DiceCE only inside foreground-promising patches;
- stronger penalty on false positives than false negatives, because r3 recall
  is already high.

### 9C.4 Apply refiner as a precision gate

Use the refiner to remove or shrink low-confidence foreground regions from the
Stage 9A ensemble, then reassign surviving pixels back to `label_1` / `label_2`
from the original ensemble probabilities.

Primary metrics:

- MSLNet-style F1 r3;
- MSLNet-style Dice;
- current multiclass mean Dice.

## Expected Win

The realistic target is not a massive jump. To beat MSLNet's paper number, we
need roughly:

- `+0.0035` binary Dice;
- `+0.0053` binary IoU;
- `+0.0189` F1 r3.

The F1 r3 gap is precision-driven, so the most likely path is precision
calibration plus phantom-focused ROI refinement, not another generic backbone
run.

## Stop Rules

Stop this route if:

- precision gains reduce recall enough that F1 r3 does not improve;
- multiclass Task 1 mean Dice drops below the current champion by more than
  `0.003`;
- the refiner only improves local released-eval tuning cases and fails on
  held-out train-validation folds.

## Verification

Every promoted candidate must report:

- MSLNet-style Dice / IoU / AHD / F1 r0-r4;
- multiclass Task 1 Dice by label and domain;
- animal / phantom split metrics;
- overlays for the worst false-positive phantom cases.
