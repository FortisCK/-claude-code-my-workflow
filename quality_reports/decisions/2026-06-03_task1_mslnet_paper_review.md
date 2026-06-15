# Decision: MSLNet Paper Review for Task 1 Direction

Date: 2026-06-03

Paper:

- `master_supporting_docs/supporting_papers/MSLNet_and_Perceptual_Grouping_for_Guidewire_Segmentation_and_Localization_2025.pdf`
- Barbu, Adrian. "MSLNet and Perceptual Grouping for Guidewire Segmentation and Localization." Sensors, 2025.

## Why This Paper Matters

This is currently the most directly relevant paper for our Task 1 work because it evaluates on
CathAction segmentation data and explicitly discusses the same failure mode we observed:
thin 3-5 px line masks, strict Dice/IoU sensitivity, and animal/phantom domain differences.

## Key Facts Extracted

- CathAction split reported in the paper:
  - 18,758 training images;
  - 4,691 test images;
  - train: 4,021 animal and 14,737 phantom;
  - test: 1,006 animal and 3,685 phantom.
- CathAction masks are described as 3-5 px thick, very long line segments.
- Catheter pixels and guidewire pixels are annotated with different labels.
- The guidewire is not annotated inside the catheter.
- There is one annotation per image, and annotators may differ across images.
- The authors separate animal and phantom in cross-dataset experiments because the images are very different.

## Evaluation Insight

The paper explicitly argues that Dice/IoU are very sensitive for thin structures. It gives the
example that a visually correct 1-pixel-wide prediction shifted by one pixel could have Dice/IoU
near zero. CathAction is less extreme because the masks are 3-5 px thick, but the authors state
that Dice/IoU are still sensitive to some extent.

Their tolerance-F1 table supports our own tolerance diagnostic: scores rise rapidly from 0 px to
3 px tolerance on CathAction.

## Method Insight

MSLNet is a coarse-to-fine architecture:

- a ResNet encoder extracts a low-resolution feature map;
- a fine head predicts z by z segmentation patches directly from the encoded feature map;
- a coarse head predicts which low-resolution patches are promising foreground patches;
- final segmentation keeps fine predictions only where the coarse foreground head is positive.

Training uses a two-term loss:

- coarse Dice + weighted BCE on patch-level foreground;
- fine Dice + weighted BCE restricted to patches where the coarse target is positive.

This restriction increases the foreground fraction for fine loss from about 0.3% to about 7%,
which is the most practically useful idea for our next stage.

## Results Relevant to Our Work

On CathAction segmentation:

- Res-UNet F1: 92.48, Dice: 62.07, AHD: 1.7.
- nnU-Net F1: 92.62, Dice: 62.19, AHD: 1.7.
- MSLNet F1: 93.05, Dice: 62.51, AHD: 1.5.
- MSLNet-Lor F1: 92.26, Dice: 60.45, AHD: 1.5.

The paper's ablation says MSL training is important for the 1-pixel guidewire dataset, but not
materially helpful on CathAction segmentation:

- CathAction without MSL: F1 93.01, Dice 62.68.
- CathAction with MSL: F1 93.05, Dice 62.51.

The Lorenz loss is beneficial for the 1-pixel guidewire dataset but not for CathAction's thicker
3-5 px annotations:

- CathAction Dice+BCE: F1 93.05, Dice 62.51.
- CathAction Dice+Lorenz: F1 92.26, Dice 60.45.

This matches our Stage 9B result: extra geometry/noise-robust loss terms can hurt strict
CathAction Dice when the masks are thicker and the warm-start model is already strong.

## Implication

Do not spend more time on Stage 9B-style global topology/geometry losses unless a much more
targeted class-wise variant is proposed. The higher-value idea from MSLNet is not the exact
loss; it is foreground-focused refinement:

- train/evaluate high-resolution ROI or patch refinement around foreground candidates;
- restrict fine loss to foreground-promising patches;
- preserve our strong ConvNeXt full-image model as the coarse detector;
- solve strict Dice by correcting local width, edges, small gaps, and label_1/label_2 confusion.

## Next Recommended Route

Stage 9C should be a ConvNeXt-backed coarse-to-fine refinement experiment, not another
standalone loss experiment:

1. Stop Stage 9B if it is still running.
2. Generate train-time foreground-centered crops from training masks only.
3. Train a local refinement model or refinement head on patches around foreground and hard
   negative line-like structures.
4. Evaluate exact Dice, 1 px tolerant Dice, and 2 px tolerant Dice to confirm whether local
   alignment is improving.
5. Only then test full original-space hflip + fixed morphology postprocess.

