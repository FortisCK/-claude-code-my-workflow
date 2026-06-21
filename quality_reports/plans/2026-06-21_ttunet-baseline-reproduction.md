# Plan: reproduce TT-U-Net as a published baseline

Status: ACTIVE (goal-driven)
Date: 2026-06-21
Why: MICCAI/TMI referee panel showstopper #1 = "no published cardiac-MoCo baseline".
TT-U-Net (Deng et al., IEEE TMI 2023) is THE published method (local code at
external/TT-U-Net), recommends ImageCAS (our data). Reproducing it + running it through
our Dice-RCA + GT-lumen diagnostics = the highest-leverage publishability move and
reframes the paper offensively ("we test which published methods are safe on real motion").

## Source architecture (external/TT-U-Net/code/TTUNet/TTUNet_demo.ipynb)

Temporal-Transformer U-Net: input (B,1,S,H,W), S = temporal/phase axis (hardcoded 48).
Spatial convs [1,3,3] + temporal convs [k,1,1]; 3 spatial-downsample levels; at each level
a PatchEmbed (spatial /2) -> 2 Swin-style TransformerBlocks attending ALONG S (window_size=1,
relative-pos bias trivial) -> PatchUnEmbed -> temporal conv -> + skip. Residual output (y = ... + x).
WGAN-GP 2D PatchGAN discriminator + L1 in their full recipe.

## Faithful adaptation to our data (documented divergences)

1. Our data = single 3D volume pairs (clean, corrupted) at 192^3, NOT 48-phase dynamic
   sequences. **Adaptation: treat the z-axis (depth) as the temporal sequence S.** The
   temporal transformer then attends across axial slices — the natural single-volume use.
2. Replace hardcoded S=48 with dynamic S = depth (from tensor shape); compute per-level
   spatial token grid dynamically so it works on our patches (spatial must be divisible by 16).
3. Same normalization as our other models ([-1,1], NOT their /1000) so the comparison is
   apples-to-apples on identical data/split/eval.
4. PRIMARY run: TT-U-Net architecture + the SAME L1+gradient loss our U-Net used
   (isolates architecture; fair vs our U-Net 0.656 / diffusion). The GAN is their refinement;
   port the discriminator too but keep adversarial OFF for the primary run (note as variant).
5. Patch [64,128,128] (depth-64 video clip, 128^2 spatial) for memory; sliding-window at eval.

## Steps

1. code/models/ttunet.py — faithful port (TTUNet + blocks + Dis), dynamic S + spatial.
2. code/training/configs/ttunet_v1.yaml — mirror unet_v1 (patch [64,128,128], L1+grad, EMA).
3. code/training/train_ttunet.py — mirror train_unet_baseline (reuse build_paired_dataloader),
   corrupted -> TTUNet -> pred, L1+grad, EMA, ckpt with model_cfg.
4. Smoke: instantiate + 1 forward/backward (shape + memory check) before full train.
5. Run card (before training). Train (GPU, serialized w/ user process).
6. Eval: sliding-window inference -> save NIfTI -> coronary_dice_rca (TS + LEGACY + GT-lumen).
   Compare to U-Net 0.656 / diff_mean 0.654 / diff_sample 0.619 / DPS.

## Success / honesty

A faithful, fair TT-U-Net baseline scored on our Dice-RCA + GT-lumen harness. Whatever it
scores (likely ~U-Net class, since it is also a distortion-trained feed-forward capped by MMSE),
it clears the showstopper and anchors the paper. Gate on GT-lumen geometry, not LEGACY (shared lineage).
