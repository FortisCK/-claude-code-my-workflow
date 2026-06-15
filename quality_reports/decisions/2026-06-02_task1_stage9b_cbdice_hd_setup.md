# Decision: Start Stage 9B cbDice + Hausdorff Alignment-Aware Training

Date: 2026-06-02

## Context

The current local Task 1 fallback champion is:

- seven-model `add_both_010_010` ensemble;
- fixed `remove_small_min32` postprocess;
- released-eval mean Dice: `0.6551964758686204`.

The diagnostic package showed that strict Dice is much lower than tolerant Dice:

- exact Dice: about `0.65`;
- 1px tolerant Dice: about `0.80`;
- 2px tolerant Dice: about `0.88`;
- 3px tolerant Dice: about `0.92`;
- oracle class-swap gain: about `+0.0014`.

The main bottleneck is therefore thin-line exactness: centerline offset,
line-width mismatch, endpoint error, small gaps, and small false-positive
fragments.

## Research Update

The user-provided literature review ranked cbDice above plain clDice for this
failure mode. The key reason is that cbDice extends clDice with
boundary/radius-aware terms and is explicitly designed to improve translation,
deformation, and diameter-balance sensitivity.

Decision:

- do not start the earlier `toolness + clDice + HD` variant;
- start a cbDice-first variant instead.

## Implementation

Added combined loss in:

`src/cathaction/training/task1_baseline.py`

New loss name:

`monai_dice_ce_toolness_cbdice_hd`

Loss structure:

```text
DiceCE(3-class)
+ toolness auxiliary loss
+ cbDice(foreground union)
+ MONAI HausdorffDTLoss(foreground union)
```

Notes:

- DiceCE supervises the three actual labels.
- Toolness supervises binary tool-vs-background through the fourth channel.
- cbDice collapses label_1/label_2 into a foreground union using the official
  cbDice differentiable-binarization pattern.
- cbDice uses morphological skeletonization and distance-transform weighting.
- Distance transform is computed on CPU because `cupy`/`cucim` are not installed
  in the local environment.
- HausdorffDTLoss uses MONAI `HausdorffDTLoss` on the foreground union.
- cbDice and HD are both warm-started by epoch.

Added config:

`configs/task1/smp_fpn_convnext_small_640_toolness_cbdice_hd_stage9b.yaml`

Added run script:

`scripts/task1/run_stage9b_convnext_small_640_toolness_cbdice_hd.sh`

## Smoke Test

Command:

```bash
scripts/task1/run_stage2_shootout.sh \
  configs/task1/smp_fpn_convnext_small_640_toolness_cbdice_hd_stage9b.yaml \
  quality_reports/logs/task1_stage9b_cbdice_hd_smoke.log \
  --output-dir outputs/task1/smp_fpn_convnext_small_640_toolness_cbdice_hd_stage9b_smoke \
  --max-train-samples 16 \
  --max-eval-samples 8 \
  --epochs 16 \
  --device cuda
```

Result:

- status: passed;
- `exit_status=0`;
- train samples: `16`;
- eval samples: `8`;
- best epoch: `5`;
- best smoke Dice: `0.808522342311324`;
- final epoch: `16`;
- final train loss: `0.1631128266453743`;
- final eval loss: `1.0918902158737183`;
- final eval Dice: `0.7480807373863305`.

Interpretation:

- Stage 7 warm-start checkpoint loads successfully.
- cbDice activates after warmup and runs without NaN.
- HausdorffDTLoss activates at epoch 16 and runs without NaN.
- The higher final eval loss reflects the additional HD term scale, not a smoke
  failure.

## Next Step

Start full Stage 9B training using:

```bash
scripts/task1/run_stage9b_convnext_small_640_toolness_cbdice_hd.sh
```

After training, compare:

- standalone raw hflip original-space Dice;
- standalone raw PNG eval;
- standalone fixed `remove_small_min32` eval;
- ensemble probe against the current seven-model + postprocess champion.

## Full Training Launch

Initial attempts to launch the wrapper script with `nohup` exited early before
the CUDA check produced output. The smoke run itself had succeeded, so full
training was launched directly with the `cardiac-diffusion` environment Python
and `setsid`, avoiding `nohup + conda run`.

Command:

```bash
setsid env \
  MPLCONFIGDIR=/tmp/cathaction-matplotlib \
  PYTHONUNBUFFERED=1 \
  PYTHONPATH=/home/mingzhang/cathaction/src:/home/mingzhang/cathaction \
  /home/mingzhang/miniconda3/envs/cardiac-diffusion/bin/python -u \
  scripts/task1/train_baseline.py \
  --config configs/task1/smp_fpn_convnext_small_640_toolness_cbdice_hd_stage9b.yaml \
  > quality_reports/logs/task1_smp_fpn_convnext_small_640_toolness_cbdice_hd_stage9b.log \
  2>&1 < /dev/null &
```

Running state at launch:

- main training PID: `847275`;
- PID file:
  `quality_reports/logs/task1_smp_fpn_convnext_small_640_toolness_cbdice_hd_stage9b.pid`;
- log:
  `quality_reports/logs/task1_smp_fpn_convnext_small_640_toolness_cbdice_hd_stage9b.log`;
- GPU: `NVIDIA RTX 6000 Ada Generation`;
- observed memory: about `14.4 GB`;
- observed utilization: `100%`.

