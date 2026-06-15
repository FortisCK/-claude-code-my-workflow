# Decision: Task 1 Official Repository Inventory

**Date:** 2026-05-26
**Status:** Completed initial clone and inventory

## Purpose

Move Stage 4 from conservative incremental tuning to a stronger official-code
implementation push. The cloned repositories are references for model
definitions, training recipes, losses, and preprocessing strategies.

All repositories are under `external_repos/`, which is ignored by git.

## Cloned Repositories

| Local path | Upstream | Commit | License status | Intended use |
| --- | --- | --- | --- | --- |
| `external_repos/ConvNeXt` | `https://github.com/facebookresearch/ConvNeXt.git` | `048efce` | MIT | Official ConvNeXt semantic segmentation backbone/reference. |
| `external_repos/ConvNeXt-V2` | `https://github.com/facebookresearch/ConvNeXt-V2.git` | `2553895` | MIT; pretrained weights have separate CC-BY-NC note | ConvNeXt V2 model definitions and pretrained-weight URLs. |
| `external_repos/SegFormer` | `https://github.com/NVlabs/SegFormer.git` | `65fa8cf` | NVIDIA Source Code License; research/evaluation use | Official SegFormer implementation; uses MMSegmentation v0.13.0. |
| `external_repos/Swin-Unet` | `https://github.com/HuCaoFighting/Swin-Unet.git` | `f48f623` | No license file found | Official Swin-Unet medical segmentation implementation. |
| `external_repos/TransUNet` | `https://github.com/Beckschen/TransUNet.git` | `02ef001` | Apache License | Official TransUNet implementation. |
| `external_repos/FGA-Net-Guidewire-Segmentation` | `https://github.com/why-26/Guidewire-Segmentation.git` | `d7411a0` | No license file found | Fluoroscopy guidewire-specific model and sliding-window inference recipe. |
| `external_repos/MSLNet` | `https://github.com/barbua/MSLNet.git` | `4cfdb2f` | MIT | CathAction-relevant MSLNet implementation and nnU-Net-style workflow notes. |
| `external_repos/WT-CMUNeXt` | `https://github.com/pikopico/WT-CMUNeXt.git` | `df9db7f` | No license file found | Paper-listed repo, but currently contains only README/image and no usable code. |
| `external_repos/nnUNet` | `https://github.com/MIC-DKFZ/nnUNet.git` | `2932ced` | Apache License | Strong medical segmentation baseline framework and possible MSLNet integration base. |

## Source Verification

- SegFormer README identifies the repository as the official PyTorch
  implementation and states that it uses MMSegmentation v0.13.0.
- ConvNeXt README identifies the repository as the official PyTorch
  implementation and includes `semantic_segmentation/`.
- ConvNeXt V2 README identifies the repository as the official PyTorch
  implementation and provides model definitions plus pretrained-weight URLs.
- Swin-Unet README identifies the code as the implementation for
  "Swin-Unet: Unet-like Pure Transformer for Medical Image Segmentation".
- TransUNet README identifies the repository as code for TransUNet.
- FGA-Net article page points to `why-26/Guidewire-Segmentation`.
- MSLNet article page points to `barbua/MSLNet`.
- WT-CMUNeXt article page points to `pikopico/WT-CMUNeXt`, but the cloned repo
  currently has no model/training code.

## Environment Findings

Current environment checked: `cardiac-diffusion`.

Installed:

- Python `3.10.20`
- PyTorch `2.11.0+cu128`
- `segmentation_models_pytorch 0.5.0`
- `timm 1.0.22`
- `monai 1.5.2`
- `nnunetv2`

Missing:

- `mmcv`
- `mmseg`

Implication:

- Running the official SegFormer/ConvNeXt MMSeg configs directly would require
  a separate compatible OpenMMLab environment or a careful modern MMEngine port.
- For fastest challenge progress, use the official repositories as source of
  truth for model definitions/recipes, but integrate candidates into the
  existing CATHACTION data, split, metric, TTA, and original-space evaluation
  pipeline.

## Integration Decision

Do not replace the current training/evaluation pipeline with each repository's
standalone training script.

Instead:

1. Keep the CATHACTION pipeline as the source of truth for data loading,
   procedure-level split discipline, metrics, checkpointing, TTA, and
   original-space ensembling.
2. Port or wrap official model definitions into `src/cathaction/models/` only
   when their dependencies are compatible.
3. Use standalone official scripts only when the repository's full training
   recipe is essential and dependency-compatible.
4. Treat no-license repositories as local research references unless license
   status is clarified before any public release.

## Stage 4A Candidate Priority

1. **SegFormer/MiT**:
   - highest priority Transformer/hierarchical encoder candidate;
   - use official repo as reference;
   - first implementation should use a compatible `timm`/SMP path if possible,
     then escalate to a separate MMseg environment only if the compatible path
     underperforms or lacks pretrained weights.
2. **ConvNeXt V2 / larger ConvNeXt**:
   - likely easiest to integrate through current `timm`/SMP stack;
   - test larger capacity and V2 variants before over-investing in old MMseg.
3. **FGA-Net**:
   - fluoroscopy-specific and likely valuable;
   - port model blocks into the current training loop for multiclass output.
4. **Swin-Unet / TransUNet**:
   - official medical Transformer baselines;
   - old Python 3.7-era code, likely requires adaptation rather than direct use.
5. **nnU-Net / MSLNet**:
   - separate data-conversion branch;
   - useful as a strong medical baseline and for coarse-to-fine ideas, but not
     the quickest Stage 4A first run.
6. **WT-CMUNeXt**:
   - blocked for now because the cloned repo lacks implementation code.

## Immediate Next Step

Start with a Stage 4A model-availability probe inside `cardiac-diffusion`:

- enumerate SMP/timm encoders for MiT, ConvNeXt V2, larger ConvNeXt, and
  EfficientNet variants;
- instantiate candidate models with `out_channels=3`;
- run a tiny forward pass and config-load check;
- create the first aggressive shootout configs only for candidates that
  instantiate cleanly.

## Model Availability Probe

Current compatible candidates that instantiate and run a `256 x 256` forward
pass in the existing SMP/timm path with `encoder_weights=None`:

| Architecture | Encoder | Params | Probe |
| --- | --- | ---: | --- |
| `Segformer` | `mit_b3` | `44.6M` | passed |
| `Segformer` | `mit_b4` | `61.4M` | passed |
| `Segformer` | `mit_b5` | `82.0M` | passed |
| `Unet` | `mit_b3` | `47.4M` | passed |
| `Unet` | `mit_b4` | `64.1M` | passed |
| `Unet` | `mit_b5` | `84.7M` | passed |
| `FPN` | `tu-convnext_small` | `51.4M` | passed |
| `FPN` | `tu-convnext_base` | `89.7M` | passed |
| `FPN` | `tu-convnextv2_tiny` | `29.9M` | passed |
| `FPN` | `tu-convnextv2_base` | `89.8M` | passed |
| `Unet` | `efficientnet-b5` | `31.2M` | passed |
| `FPN` | `tu-maxvit_tiny_rw_256` | `30.4M` | passed |

Swin through the current SMP/timm path is usable only at fixed model sizes:

- `tu-swin_tiny_patch4_window7_224` passed at `224 x 224`;
- `tu-swin_small_patch4_window7_224` passed at `224 x 224`;
- `tu-swin_base_patch4_window12_384` passed at `384 x 384`.

Because the current best CATHACTION models use `512` or `640`, Swin is a later
candidate unless we intentionally run a fixed `384 x 384` branch or port the
official Swin-Unet more directly.

Initial Stage 4A configs created:

- `configs/task1/smp_segformer_mit_b3_512_stage4a.yaml`
- `configs/task1/smp_segformer_mit_b4_512_stage4a.yaml`
- `configs/task1/smp_fpn_convnext_small_512_stage4a.yaml`
- `configs/task1/smp_fpn_convnextv2_base_512_stage4a.yaml`
- `configs/task1/smp_unet_efficientnet_b5_512_stage4a.yaml`

These configs use the existing CATHACTION data/eval pipeline with ImageNet
normalization, `512 x 512` direct resize, MONAI DiceCE, and the released
train/eval-balanced manifests.

## Stage 4A Runner

Added:

- `scripts/task1/run_stage4a_shootout.sh`

The runner trains the five initial Stage 4A candidates sequentially and runs a
full released evaluation after each checkpoint finishes. Outputs:

- training logs under `quality_reports/logs/`;
- full-eval logs under `quality_reports/logs/*_full_eval.log`;
- full-eval JSON files under `outputs/task1/stage4a_eval/`.

Launch status:

- initial `nohup` launch exited early before child logs were created;
- relaunched as user systemd service:
  - `cathaction-task1-stage4a-shootout.service`
  - master log: `quality_reports/logs/task1_stage4a_shootout_master_systemd.log`
- service was confirmed active on 2026-05-26;
- first active candidate: `smp_segformer_mit_b3_512_stage4a`;
- GPU utilization was about `97%` with about `11.3GB` allocated.

## Stage 4A Completion

Completed on 2026-05-28. The user systemd service finished and became
`inactive`.

Full released results:

| Candidate | Mean Dice | Label 1 | Label 2 | Animal | Phantom |
| --- | ---: | ---: | ---: | ---: | ---: |
| `FPN + ConvNeXtV2-Base` | `0.634086861627` | `0.609813554681` | `0.658360168573` | `0.717981853848` | `0.611183642584` |
| `FPN + ConvNeXt-Small` | `0.633066549973` | `0.609565498660` | `0.656567601285` | `0.722420953553` | `0.608672919036` |
| `Segformer + MiT-B3` | `0.629093403226` | `0.605300341391` | `0.652886465061` | `0.711088410427` | `0.606708877515` |
| `UNet + EfficientNet-B5` | `0.628982855360` | `0.614394692940` | `0.643571017781` | `0.710592786740` | `0.606703454826` |
| `Segformer + MiT-B4` | `0.627912671935` | `0.599148031785` | `0.656677312084` | `0.708191497738` | `0.605996661417` |

Decision:

- Stage 4A did not find a new single-model champion.
- Keep the active provisional champion as:
  `ConvNeXt640/EfficientNet512 0.5/0.5 + hflip`, original-mask-space eval,
  full released Dice `0.6438677932546133`.
- Promote `FPN + ConvNeXtV2-Base` and `FPN + ConvNeXt-Small` to
  complementarity/ensemble testing only.
- Do not continue plain SegFormer/MiT under the same recipe unless changing the
  recipe substantially, such as using official MMSeg training, different LR
  schedule, higher resolution, or thin-structure losses.
