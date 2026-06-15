# Task 1 Weekend Wrap-Up Plan

Date: 2026-05-29

## Goal

Finish the Task 1 segmentation work in a submission-oriented state before
switching primary effort to Task 2 next week.

Current local champion:

- `ConvNeXt-Tiny 640`: `0.325`
- `EfficientNet-B3 512`: `0.325`
- `ConvNeXtV2-Base 512`: `0.117`
- `ConvNeXt-Small 512`: `0.117`
- `ConvNeXt-Small 640`: `0.116`
- full released original-mask-space Dice: `0.6516511688167698`

## Scope

This is a wrap-up pass, not another broad architecture search.

## Steps

1. **Inference feasibility test**
   - Run the fixed champion pipeline on a small released subset.
   - Measure approximate wall time, GPU memory, and output generation.
   - Confirm the method is fully automated and suitable for Docker packaging.

2. **Error-analysis overlays**
   - Generate worst-case examples for the current champion.
   - Inspect label/domain failure modes:
     - `label_1`;
     - `label_2`;
     - animal;
     - phantom.

3. **One overnight training run**
   - Choose only one experiment after the error-analysis signal:
     - if failures are mainly thin-line gaps/fragmentation: run a clDice or
       thin-structure-loss ConvNeXt variant;
     - if failures look like model-capacity/domain complementarity: run
       `ConvNeXtV2-Base 640 aspect-pad`;
     - if runtime is already too high: skip overnight and focus on submission
       inference.

4. **Submission notes**
   - Freeze the champion weights/config.
   - Record ensemble compliance assumptions:
     - fixed automated inference;
     - no hidden-test tuning;
     - no manual interaction;
     - public pretrained weights only.

## Decision Rule

The overnight run must have a clear reason and a single success criterion:

- it should either improve the current ensemble after one integration test, or
  provide a strong fallback model with better speed/stability.

If it does neither, keep the current champion and move to Task 2.

## Progress Update: 2026-05-29 16:45

Completed:

- Full champion prediction smoke was expanded to the full released eval split.
- Generated `4691` original-image-space PNG predictions with the fixed
  five-model ensemble at
  `outputs/task1/stage5_submission_smoke/predict_full`.
- Generated per-sample metrics and worst-case overlays at
  `outputs/task1/stage5_error_analysis/champion_refined`.
- Verified the submission-style prediction path reproduces the current
  champion score exactly:
  - mean Dice: `0.6516511688167698`;
  - label_1 Dice: `0.6305547443921731`;
  - label_2 Dice: `0.6727475932413666`;
  - animal Dice: `0.7373800564302811`;
  - phantom Dice: `0.6282472988197025`.

Error-analysis signal:

- Worst animal failures are often correct-line/wrong-class cases, especially
  target label_1 predicted as label_2.
- Worst phantom failures combine class confusion, long-line distractors, and
  label_2 misses.
- This points more toward class-disambiguation and thin-structure loss than a
  blind architecture increase.

Overnight choice:

- Config:
  `configs/task1/smp_fpn_convnext_small_640_cldice_stage6.yaml`
- Runner:
  `scripts/task1/run_stage6_convnext_small_640_cldice.sh`
- Rationale:
  keep the current best ConvNeXt-640 family, reduce the label_2-heavy class
  weighting from the previous setup, and add class-wise soft-clDice after a
  warmup.
- Smoke result:
  `--epochs 1 --max-train-samples 32 --max-eval-samples 16` completed on GPU
  without OOM or config errors.

Launch:

- Plain `nohup` again exited early in this environment, leaving only a log
  header. The persistent run was launched through user systemd.
- Service:
  `cathaction-task1-stage6-convnext-small640-cldice.service`
- Master log:
  `quality_reports/logs/task1_stage6_convnext_small_640_cldice_master_systemd.log`
- Training log:
  `quality_reports/logs/task1_smp_fpn_convnext_small_640_cldice_stage6.log`
- Output directory:
  `outputs/task1/smp_fpn_convnext_small_640_cldice_stage6`
- Final full-eval JSON when training completes:
  `outputs/task1/stage6_convnext_small_640_cldice_eval/convnext_small_640_cldice_hflip_original_full.json`

Monitoring:

```bash
systemctl --user status cathaction-task1-stage6-convnext-small640-cldice.service --no-pager -l
tail -f quality_reports/logs/task1_smp_fpn_convnext_small_640_cldice_stage6.log
nvidia-smi
```

## Progress Update: 2026-05-31

Stage 6 completed:

- Service completed and was collected by user systemd.
- Training finished on 2026-05-30 16:04 CEST.
- Full released eval finished on 2026-05-30 16:07 CEST.
- Result JSON:
  `outputs/task1/stage6_convnext_small_640_cldice_eval/convnext_small_640_cldice_hflip_original_full.json`

Single-model result:

- mean Dice: `0.633660055514658`
- label_1 Dice: `0.612316012945537`
- label_2 Dice: `0.655004098083779`
- animal Dice: `0.7258733496359073`
- phantom Dice: `0.6084859513393589`

Comparison against the previous `ConvNeXt-Small 640` single model:

- previous mean Dice: `0.6373111721520722`
- previous label_1 Dice: `0.6161710981745443`
- previous label_2 Dice: `0.6584512461296002`
- previous animal Dice: `0.7281588395844082`
- previous phantom Dice: `0.612509882209893`

Decision:

- As a standalone model, Stage 6 is worse than the previous
  `ConvNeXt-Small 640`.
- A six-model ensemble probe still showed small complementarity:
  `add_cldice_010` reached mean Dice `0.6523720412288835`, compared with the
  previous champion `0.6516511688167698`.
- The gain is real on this released eval but very small (`+0.0007208724121137`);
  keep the Stage 6 checkpoint only as a low-weight ensemble tail, not as a new
  main direction.

Six-model probe:

- JSON:
  `outputs/task1/stage6_convnext_small_640_cldice_eval/six_model_cldice_weight_probe_full.json`
- Best candidate:
  `add_cldice_010`
- Weights:
  - `ConvNeXt-Tiny 640`: `0.2925`
  - `EfficientNet-B3 512`: `0.2925`
  - `ConvNeXtV2-Base 512`: `0.1053`
  - `ConvNeXt-Small 512`: `0.1053`
  - `ConvNeXt-Small 640`: `0.1044`
  - `ConvNeXt-Small 640 clDice`: `0.10`

Next implication:

- Do not spend another broad run on full-frame clDice/weight tuning.
- Move to the stronger method track: toolness auxiliary head and ROI/patch
  refinement.
