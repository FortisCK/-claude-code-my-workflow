# Plan: Task 1 MONAI Overnight Strong Baseline

**Date:** 2026-05-21
**Status:** COMPLETED
**Scope:** Run a stronger overnight Task 1 segmentation baseline with MONAI

---

## Goal

Start an overnight Task 1 multiclass segmentation run that is stronger than the
current `TinyUNet` baseline, using an existing library implementation rather
than a hand-written backbone.

Use the `cardiac-diffusion` Conda environment because it has:

- PyTorch with CUDA access
- MONAI 1.5.2
- timm

The current `pointdet` and `cathaction-task1` environments do not have MONAI.

---

## Configuration

Planned model:

- library: MONAI
- model: `monai.networks.nets.UNet`
- spatial dims: 2
- input channels: 3
- output channels: 3
- channels: `[32, 64, 128, 256, 512]`
- strides: `[2, 2, 2, 2]`
- residual units: 2

Planned loss:

- library: MONAI
- loss: `monai.losses.DiceCELoss`
- label mode: `multiclass_012`
- softmax: true
- one-hot target conversion: true
- class weights: `[0.02, 1.0, 2.0]`

Training:

- train manifest: `configs/task1/splits/released_train.csv`
- eval manifest: `configs/task1/splits/released_eval.csv`
- train samples: all 18,758 released train samples
- eval samples during training: capped at 512
- image size: `512x512`
- epochs: 100 unless smoke testing shows this is impractical
- batch size: start at 12 with AMP; reduce if OOM
- device: CUDA
- launcher: `nohup`
- output directory: `outputs/task1/monai_unet_dicece_full_512_overnight`

This is still released-folder evaluation, not hidden-test evaluation.

---

## Execution

1. Add MONAI model/loss support as lazy imports so the CPU env tests do not
   require MONAI.
2. Add a strong MONAI config under `configs/task1/`.
3. Run syntax/tests in `cathaction-task1`.
4. Run a short smoke train in `cardiac-diffusion` with the MONAI config.
5. If the smoke run fits memory, launch the full run with `nohup` in
   `cardiac-diffusion`.
6. Record PID, log path, output directory, and tomorrow's inspection commands.

---

## Verification

Before overnight launch:

- `py_compile`
- `pytest`
- `git diff --check`
- one short MONAI smoke run on a small sample subset

After launch:

- verify the `nohup` process is alive
- verify the log is being written
- verify the output directory contains `resolved_config.json` or active epoch
  logs once the run starts

---

## Approval

The user asked to use MONAI or a similar library, increase epochs, and run the
strong baseline overnight with `nohup`. The user then clarified to switch to the
environment that already has the artifact/library stack rather than
hand-implementing the model.

---

## Launch Record

Launched on 2026-05-21 at 01:46 CEST.

Implementation:

- config: `configs/task1/monai_unet_dicece_full_512_overnight.yaml`
- launcher script: `scripts/task1/run_monai_overnight.sh`
- service manager: `systemd --user`
- service name: `cathaction-monai-overnight.service`
- log:
  `quality_reports/logs/task1_monai_unet_dicece_full_512_overnight.log`
- output directory:
  `outputs/task1/monai_unet_dicece_full_512_overnight`

Reason for `systemd --user` instead of plain `nohup`:

- In this execution environment, plain `nohup ... &`, `setsid ... &`, and
  `screen -dmS ...` did not keep a background training process alive after the
  command returned.
- `systemd-run --user` was tested with a 60-second service and confirmed to
  keep the process alive.
- The service is equivalent for the user's purpose: the run is detached from
  Codex and will continue independently overnight.

Verification before launch:

- `py_compile`: passed
- `pytest`: 19 passed
- `git diff --check`: passed
- MONAI GPU smoke:
  - env: `cardiac-diffusion`
  - samples: 24 train / 12 eval
  - epoch: 1
  - result: completed and wrote checkpoints

Live status check after launch:

- `systemctl --user status cathaction-monai-overnight --no-pager`
  reported `Active: active (running)`.
- `nvidia-smi` reported GPU activity, about 2,538 MiB used out of 49,140 MiB
  at the time of the check.
- Training log showed epoch 1 progressing normally.

Tomorrow's inspection commands:

```bash
systemctl --user status cathaction-monai-overnight --no-pager
tail -80 quality_reports/logs/task1_monai_unet_dicece_full_512_overnight.log
ls -lh outputs/task1/monai_unet_dicece_full_512_overnight
cat outputs/task1/monai_unet_dicece_full_512_overnight/metrics.json
```

Stop command if needed:

```bash
systemctl --user stop cathaction-monai-overnight
```

---

## Completion Record

Completed on 2026-05-21 at about 04:01 CEST.

The transient service was no longer present after completion, and the output
directory contained:

- `best_checkpoint.pt`
- `checkpoint.pt`
- `metrics.json`
- `resolved_config.json`

Training monitor result, capped 512-sample eval:

- best epoch: 98
- best capped DSC: 0.6148080685840516
- best capped IoU/mIoU: 0.4968428031080951
- best capped per-class DSC:
  - `label_1`: 0.6930938037567017
  - `label_2`: 0.5365223334114013
- final epoch: 100
- final capped DSC: 0.6057077205690814
- final capped per-class DSC:
  - `label_1`: 0.6540682187782509
  - `label_2`: 0.5573472223599119

Best checkpoint, full released eval:

- checkpoint:
  `outputs/task1/monai_unet_dicece_full_512_overnight/best_checkpoint.pt`
- eval JSON:
  `outputs/task1/monai_unet_dicece_full_512_overnight/eval_best_full.json`
- eval samples: 4,691
- DSC: 0.5303603258369163
- IoU/mIoU: 0.40486750742246314
- per-class DSC:
  - `label_1`: 0.5917695853589093
  - `label_2`: 0.4689510663149234
- per-class IoU:
  - `label_1`: 0.4324125091702051
  - `label_2`: 0.3773225056747212
- pixel accuracy: 0.9934357240799265

Final checkpoint, full released eval:

- checkpoint:
  `outputs/task1/monai_unet_dicece_full_512_overnight/checkpoint.pt`
- eval JSON:
  `outputs/task1/monai_unet_dicece_full_512_overnight/eval_final_full.json`
- eval samples: 4,691
- DSC: 0.5276875655622489
- IoU/mIoU: 0.4023158708736349
- per-class DSC:
  - `label_1`: 0.57903575568973
  - `label_2`: 0.4763393754347679
- per-class IoU:
  - `label_1`: 0.41942266690672625
  - `label_2`: 0.38520907484054356
- pixel accuracy: 0.993190148165932

Saved prediction subset from the best checkpoint:

- predictions: 64
- matched predictions: 64
- missing sample IDs: 0
- eval JSON:
  `outputs/task1/monai_unet_dicece_full_512_overnight/eval_best_saved_predictions.json`
- DSC: 0.5916743104494404
- IoU/mIoU: 0.47456814962269517
- per-class DSC:
  - `label_1`: 0.6879701028549572
  - `label_2`: 0.49537851804392374
- pixel accuracy: 0.9944639205932617

Qualitative outputs:

- predictions:
  `outputs/task1/monai_unet_dicece_full_512_overnight/best_predictions/`
- overlays:
  `outputs/task1/monai_unet_dicece_full_512_overnight/best_overlays/`
- contact sheet:
  `outputs/task1/monai_unet_dicece_full_512_overnight/best_overlays/contact_sheet.png`

Comparison to previous full released baseline:

- Previous TinyUNet best full eval DSC: 0.4437008138855879
- MONAI UNet best full eval DSC: 0.5303603258369163
- Absolute DSC gain: about 0.0867

Interpretation:

- The library-based MONAI baseline is clearly stronger than the TinyUNet
  reference.
- Best checkpoint should be used for comparisons, although the final checkpoint
  is close on full released eval.
- The released eval score remains an engineering benchmark only, not hidden-test
  or challenge leaderboard performance.
