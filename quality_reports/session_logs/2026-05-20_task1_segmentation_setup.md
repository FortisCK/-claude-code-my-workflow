# Session Log: Task 1 Segmentation Setup

**Date:** 2026-05-20
**Status:** Full released baseline complete

---

## Goal

Start CATHACTION Task 1 segmentation work by approving the data-pipeline/baseline plan and creating a reproducible Conda environment.

## Plan

- `quality_reports/plans/2026-05-20_task1-segmentation-data-pipeline-baseline.md`
- Status changed from `DRAFT` to `APPROVED` after user approval.

## Decisions

- Use PyTorch as the default first baseline stack.
- Treat released `animal_test` and `phantom_test` folders as validation/evaluation-style released data, not the MICCAI hidden test set.
- Keep `human_train` separate as a human-domain holdout/analysis domain until label semantics and split strategy are fully confirmed.
- Start with a CPU-capable PyTorch environment because `nvidia-smi` currently cannot communicate with the NVIDIA driver in this session.

## Environment

- Created Conda environment: `cathaction-task1`
- Environment spec: `environment-task1.yml`
- Installed the repo as an editable package in `cathaction-task1` with:
  `conda run -n cathaction-task1 python -m pip install -e . --no-build-isolation`
- Verified imports:
  - Python `3.11.15`
  - NumPy `2.4.6`
  - pandas `3.0.3`
  - Pillow `12.2.0`
  - OpenCV `4.13.0`
  - PyTorch `2.4.0`
  - torchvision `0.19.0`
  - pytest `9.0.3`
- `torch.cuda.is_available()` returned `False`.

## Data Setup

- Dataset archives were previously extracted under `datasets/`.
- Zip archives were removed after extraction.
- `.gitignore` now ignores `datasets/*`, while allowing optional tracked `datasets/README.md` or `.gitkeep`.

## Next Actions

1. Start the minimal baseline scaffold after metrics and schema tests are clean.
2. Define resize/crop/pad strategy for heterogeneous animal/phantom image sizes.
3. Keep label `1/2` semantics unresolved until confirmed by official docs or visual review.

## Phase 1 Progress

- Added `scripts/task1/inspect_task1_dataset.py`.
- Ran the inventory script with `--sample-size 64`, `--full-shapes`, and `--fail-on-pairing-errors`.
- Wrote JSON inventory to `quality_reports/decisions/2026-05-20_task1_dataset_inventory.json`.
- Recorded schema decision in `quality_reports/decisions/2026-05-20_task1_dataset_schema.md`.

Inventory result:

- Total images: 28,732
- Total masks: 28,732
- Total paired samples: 28,732
- Pairing issues: 0

Important findings:

- Animal/phantom masks are `uint8` `.npy` arrays with observed values `0,1,2`.
- Human masks are `uint8` grayscale PNGs with observed values `0,255`.
- Animal/phantom image sizes are heterogeneous; human images are consistently `512x512`.
- Exact semantic mapping for animal/phantom labels `1` and `2` remains unresolved.

## Phase 2 Progress

- Added installable `src` package metadata in `pyproject.toml`.
- Added `src/cathaction/data/task1.py`:
  - released collection specs;
  - image/mask pairing;
  - `Task1Sample` dataclass;
  - human case ID inference from filename prefix before `_img-`;
  - mask loading and binary foreground conversion.
- Added `src/cathaction/metrics/segmentation.py`:
  - per-class and mean DSC;
  - per-class and mean IoU/mIoU;
  - pixel accuracy;
  - binary foreground helpers.
- Added tests:
  - `tests/test_task1_dataset_index.py`
  - `tests/test_task1_segmentation_metrics.py`

Verification:

- `conda run -n cathaction-task1 python -m py_compile scripts/task1/inspect_task1_dataset.py src/cathaction/data/task1.py src/cathaction/metrics/segmentation.py`
- `conda run -n cathaction-task1 pytest`
- Result: 9 tests passed.
- Real-data smoke check indexed 28,732 samples:
  - `animal_train`: 4,021
  - `animal_test`: 1,006
  - `phantom_train`: 14,737
  - `phantom_test`: 3,685
  - `human_train`: 5,283

## Phase 3 Progress

- Added `scripts/task1/write_task1_split_manifests.py`.
- Added `configs/task1/splits/README.md`.
- Added `quality_reports/decisions/2026-05-20_task1_split_policy.md`.
- Generated:
  - `configs/task1/splits/released_train.csv`
  - `configs/task1/splits/released_eval.csv`
  - `configs/task1/splits/human_holdout.csv`
  - `configs/task1/splits/summary.json`
- Added tests in `tests/test_task1_split_manifests.py`.

Split policy:

- `released_train`: `animal_train` + `phantom_train` = 18,758 samples.
- `released_eval`: `animal_test` + `phantom_test` = 4,691 samples.
- `human_holdout`: `human_train` = 5,283 samples.
- Do not call `released_eval` hidden-test performance.
- Do not create frame-random internal validation splits for animal/phantom without a verified case map.

Verification:

- `conda run -n cathaction-task1 python scripts/task1/write_task1_split_manifests.py --data-root datasets --output-dir configs/task1/splits`
- `conda run -n cathaction-task1 pytest`
- Result: 11 tests passed.
- CSV manifests use LF line endings and contain expected sample counts plus header.


---
**Context compaction (auto) at 17:43**
Check git log and quality_reports/plans/ for current state.

## Phase 4 Progress

- Added `configs/task1/baseline_unet.yaml`.
- Added a compact PyTorch U-Net baseline:
  - `src/cathaction/models/unet.py`
  - `src/cathaction/training/task1_baseline.py`
- Added script entrypoints:
  - `scripts/task1/train_baseline.py`
  - `scripts/task1/evaluate_baseline.py`
  - `scripts/task1/predict_baseline.py`
- Added tests:
  - `tests/test_task1_baseline_components.py`
  - `tests/test_task1_saved_prediction_evaluation.py`
- Added `outputs/` to `.gitignore` for generated checkpoints, metrics, and
  prediction masks.
- Recorded baseline policy in
  `quality_reports/decisions/2026-05-20_task1_baseline_smoke_policy.md`.

Baseline policy:

- First baseline uses `binary_foreground` because label values `1/2` in
  animal/phantom masks are still semantically unresolved.
- Default config is a CPU smoke run, not a performance experiment:
  - 16 train samples
  - 8 eval samples
  - 1 epoch
  - `128x128` direct resize
  - `base_channels=8`
- `released_eval` metrics are released-folder metrics only, not hidden-test
  metrics.

Smoke outputs:

- `outputs/task1/baseline_unet_smoke/checkpoint.pt`
- `outputs/task1/baseline_unet_smoke/metrics.json`
- `outputs/task1/baseline_unet_smoke/eval_smoke.json`
- `outputs/task1/baseline_unet_smoke/predictions/predictions.csv`
- `outputs/task1/baseline_unet_smoke/eval_saved_predictions_smoke.json`

Smoke metrics from checkpoint evaluation:

- eval samples: 8
- loss: 0.6379348784685135
- DSC: 0.0
- IoU: 0.0
- mIoU: 0.0
- pixel accuracy: 0.9923019409179688

Smoke metrics from saved prediction PNG evaluation:

- predictions: 2
- matched predictions: 2
- missing sample IDs: 0
- DSC: 0.0
- IoU: 0.0
- mIoU: 0.0
- pixel accuracy: 0.991119384765625

Interpretation:

- The zero DSC/IoU is expected for this tiny untrained smoke baseline and should
  not be interpreted as a model result.
- The useful result is that manifest loading, image/mask resizing, training,
  checkpoint save/load, metric JSON writing, prediction PNG writing, and saved
  prediction re-evaluation all run end-to-end.

Verification:

- `conda run -n cathaction-task1 python -m py_compile scripts/task1/train_baseline.py scripts/task1/evaluate_baseline.py scripts/task1/predict_baseline.py src/cathaction/models/unet.py src/cathaction/training/task1_baseline.py`
- `conda run -n cathaction-task1 pytest`
- Result: 14 tests passed.
- `git diff --check`
- `conda run -n cathaction-task1 python scripts/task1/train_baseline.py --config configs/task1/baseline_unet.yaml`
- `conda run -n cathaction-task1 python scripts/task1/evaluate_baseline.py --config configs/task1/baseline_unet.yaml --output-json outputs/task1/baseline_unet_smoke/eval_smoke.json`
- `conda run -n cathaction-task1 python scripts/task1/predict_baseline.py --config configs/task1/baseline_unet.yaml --max-samples 2`
- `conda run -n cathaction-task1 python scripts/task1/evaluate_baseline.py --config configs/task1/baseline_unet.yaml --predictions-csv outputs/task1/baseline_unet_smoke/predictions/predictions.csv --output-json outputs/task1/baseline_unet_smoke/eval_saved_predictions_smoke.json`

## Phase 5 Progress

- Re-checked Task 1 label semantics using:
  - local MICCAI 2026 PDF in `datasets/`;
  - CATHACTION public segmentation page;
  - Hugging Face dataset README;
  - local extracted animal/phantom masks.
- Recorded label decision in
  `quality_reports/decisions/2026-05-20_task1_label_semantics.md`.
- Added multiclass smoke config:
  - `configs/task1/baseline_unet_multiclass_smoke.yaml`
- Extended `src/cathaction/training/task1_baseline.py` to support:
  - `binary_foreground`: 1 output channel, BCEWithLogitsLoss, thresholded PNG
    predictions;
  - `multiclass_012`: 3 output channels, CrossEntropyLoss, argmax PNG
    predictions with values `0/1/2`;
  - per-class metrics as `label_1` and `label_2`;
  - saved prediction re-evaluation for both binary and multiclass outputs.
- Added tests for multiclass dataset loading and saved-prediction evaluation.

Label semantics decision:

- MICCAI 2026 PDF confirms Task 1 masks distinguish catheter and guidewire as
  separate classes.
- No checked source currently maps numeric `.npy` values `1/2` to the names
  catheter/guidewire.
- Use `multiclass_012` for animal/phantom class-preserving experiments, but
  report class-specific metrics as `label_1`/`label_2` until the mapping is
  confirmed.

Full animal/phantom `.npy` pixel-count scan:

- `animal_train`: 4,021 files; label 1 = 4,926,859 px; label 2 = 1,492,425 px;
  label 1 foreground fraction = 0.7675.
- `animal_test`: 1,006 files; label 1 = 1,217,178 px; label 2 = 362,067 px;
  label 1 foreground fraction = 0.7707.
- `phantom_train`: 14,737 files; label 1 = 52,465,596 px; label 2 = 6,854,851
  px; label 1 foreground fraction = 0.8844.
- `phantom_test`: 3,685 files; label 1 = 12,907,528 px; label 2 = 1,718,795
  px; label 1 foreground fraction = 0.8825.

Multiclass smoke metrics from checkpoint evaluation:

- eval samples: 8
- loss: 1.1993911862373352
- DSC: 0.2560235254864348
- IoU/mIoU: 0.2530364990234375
- per-class DSC:
  - `label_1`: 0.012047050972869545
  - `label_2`: 0.5
- per-class IoU:
  - `label_1`: 0.006072998046875
  - `label_2`: 0.5
- pixel accuracy: 0.006072998046875

Multiclass smoke metrics from saved prediction PNG evaluation:

- predictions: 2
- matched predictions: 2
- missing sample IDs: 0
- DSC: 0.0066164764997074955
- IoU/mIoU: 0.0033416748046875
- per-class DSC:
  - `label_1`: 0.013232952999414991
  - `label_2`: 0.0
- per-class IoU:
  - `label_1`: 0.006683349609375
  - `label_2`: 0.0
- pixel accuracy: 0.006683349609375

Binary smoke was re-run after the multiclass changes and still passed.

Verification:

- `conda run -n cathaction-task1 python -m py_compile scripts/task1/train_baseline.py scripts/task1/evaluate_baseline.py scripts/task1/predict_baseline.py src/cathaction/models/unet.py src/cathaction/training/task1_baseline.py`
- `conda run -n cathaction-task1 pytest`
- Result: 16 tests passed.
- `git diff --check`
- `conda run -n cathaction-task1 python scripts/task1/train_baseline.py --config configs/task1/baseline_unet_multiclass_smoke.yaml`
- `conda run -n cathaction-task1 python scripts/task1/evaluate_baseline.py --config configs/task1/baseline_unet_multiclass_smoke.yaml --output-json outputs/task1/baseline_unet_multiclass_smoke/eval_smoke.json`
- `conda run -n cathaction-task1 python scripts/task1/predict_baseline.py --config configs/task1/baseline_unet_multiclass_smoke.yaml --max-samples 2`
- `conda run -n cathaction-task1 python scripts/task1/evaluate_baseline.py --config configs/task1/baseline_unet_multiclass_smoke.yaml --predictions-csv outputs/task1/baseline_unet_multiclass_smoke/predictions/predictions.csv --output-json outputs/task1/baseline_unet_multiclass_smoke/eval_saved_predictions_smoke.json`

## Phase 6 Progress

- User flagged that the workstation GPU should be available.
- Confirmed the issue was not hardware:
  - outside the Codex sandbox, `nvidia-smi` reports NVIDIA RTX 6000 Ada
    Generation;
  - driver version: 570.195.03;
  - CUDA shown by `nvidia-smi`: 12.8.
- Confirmed the dedicated `cathaction-task1` env is CPU-only:
  - `pytorch 2.4.0 py3.11_cpu_0`;
  - `torchvision py311_cpu`;
  - `cpuonly`;
  - `torch.version.cuda is None`.
- Confirmed existing GPU envs can access the GPU:
  - `pointdet`: PyTorch 2.5.1, CUDA 12.1, CUDA available;
  - `pointdet_mamba`: PyTorch 2.5.1+cu121, CUDA available;
  - `cardiac-diffusion`: PyTorch 2.11.0+cu128, CUDA available.
- Added clean future GPU env spec:
  - `environment-task1-gpu.yml`
- Added GPU pilot config:
  - `configs/task1/baseline_unet_multiclass_gpu_pilot.yaml`
- Used `pointdet` to run the first GPU multiclass baseline pilot.

GPU pilot config:

- label mode: `multiclass_012`
- train samples: 2,048
- eval samples: 512
- epochs: 5
- batch size: 16
- image size: `256x256`
- device: `cuda`
- output directory: `outputs/task1/baseline_unet_multiclass_gpu_pilot/`

GPU pilot final checkpoint metrics:

- train loss: 0.023291283825528808
- eval loss: 0.02270361187402159
- eval DSC: 0.4720236794117548
- eval IoU/mIoU: 0.4060698406421759
- per-class DSC:
  - `label_1`: 0.5690473588235097
  - `label_2`: 0.375
- per-class IoU:
  - `label_1`: 0.4371396812843519
  - `label_2`: 0.375
- pixel accuracy: 0.9952917397022247

Generated GPU pilot outputs:

- `outputs/task1/baseline_unet_multiclass_gpu_pilot/checkpoint.pt`
- `outputs/task1/baseline_unet_multiclass_gpu_pilot/metrics.json`
- `outputs/task1/baseline_unet_multiclass_gpu_pilot/eval_smoke.json`
- `outputs/task1/baseline_unet_multiclass_gpu_pilot/predictions/predictions.csv`
- `outputs/task1/baseline_unet_multiclass_gpu_pilot/eval_saved_predictions_smoke.json`
- `outputs/task1/baseline_unet_multiclass_gpu_pilot/overlays/`

Saved prediction evaluation over 32 exported PNGs:

- predictions: 32
- matched predictions: 32
- missing sample IDs: 0
- DSC: 0.534831876281967
- IoU/mIoU: 0.4669615834834998
- per-class DSC:
  - `label_1`: 0.6321637525639341
  - `label_2`: 0.4375
- per-class IoU:
  - `label_1`: 0.4964231669669996
  - `label_2`: 0.4375
- pixel accuracy: 0.9956531524658203

Prediction class distribution over the 32 exported PNGs:

- predicted pixels:
  - label 0: 2,088,115
  - label 1: 9,037
  - label 2: 0
- resized GT pixels:
  - label 0: 2,083,026
  - label 1: 11,595
  - label 2: 2,531

Interpretation:

- The baseline is now actually training on GPU.
- The current pilot learns/predicts `label_1` foreground but does not yet predict
  `label_2`.
- `label_2` scores are inflated by empty-frame behavior in some samples; use the
  pixel distribution and overlays to judge the qualitative result.
- Next model improvement should address class imbalance, likely with weighted CE
  and/or Dice loss, before scaling to the full released train split.

GPU pilot commands:

- `conda run -n pointdet python -c "import sys; sys.path[:0]=['src','.']; from cathaction.training.task1_baseline import main_train; raise SystemExit(main_train(['--config','configs/task1/baseline_unet_multiclass_gpu_pilot.yaml']))"`
- `conda run -n pointdet python -c "import sys; sys.path[:0]=['src','.']; from cathaction.training.task1_baseline import main_evaluate; raise SystemExit(main_evaluate(['--config','configs/task1/baseline_unet_multiclass_gpu_pilot.yaml','--output-json','outputs/task1/baseline_unet_multiclass_gpu_pilot/eval_smoke.json']))"`
- `conda run -n pointdet python -c "import sys; sys.path[:0]=['src','.']; from cathaction.training.task1_baseline import main_predict; raise SystemExit(main_predict(['--config','configs/task1/baseline_unet_multiclass_gpu_pilot.yaml','--max-samples','32']))"`
- `conda run -n pointdet python -c "import sys; sys.path[:0]=['src','.']; from cathaction.training.task1_baseline import main_evaluate; raise SystemExit(main_evaluate(['--config','configs/task1/baseline_unet_multiclass_gpu_pilot.yaml','--predictions-csv','outputs/task1/baseline_unet_multiclass_gpu_pilot/predictions/predictions.csv','--output-json','outputs/task1/baseline_unet_multiclass_gpu_pilot/eval_saved_predictions_smoke.json']))"`
- `conda run -n cathaction-task1 python scripts/task1/make_prediction_overlays.py --eval-manifest configs/task1/splits/released_eval.csv --predictions-csv outputs/task1/baseline_unet_multiclass_gpu_pilot/predictions/predictions.csv --output-dir outputs/task1/baseline_unet_multiclass_gpu_pilot/overlays --max-samples 32`

## Phase 7 Progress

- Added explicit class-weight support for multiclass cross entropy in
  `src/cathaction/training/task1_baseline.py`.
- Added weighted pilot config:
  - `configs/task1/baseline_unet_multiclass_weighted_gpu_pilot.yaml`
- Added weighted-loss plan:
  - `quality_reports/plans/2026-05-20_task1-weighted-loss-pilot.md`
- Added test coverage for configured class weights.

Class weights:

- Based on the first 2,048 released-train masks resized to `256x256`.
- Pixel counts:
  - label 0: 133,331,142
  - label 1: 677,236
  - label 2: 209,350
- Mean-normalized inverse-frequency weights:
  - label 0: 0.003593860466557044
  - label 1: 0.7075428952310618
  - label 2: 2.288863244302381

Verification before training:

- `conda run -n cathaction-task1 python -m py_compile scripts/task1/train_baseline.py scripts/task1/evaluate_baseline.py scripts/task1/predict_baseline.py scripts/task1/make_prediction_overlays.py src/cathaction/training/task1_baseline.py`
- `conda run -n cathaction-task1 pytest`
- Result: 18 tests passed.
- `git diff --check`

Weighted GPU pilot:

- config: `configs/task1/baseline_unet_multiclass_weighted_gpu_pilot.yaml`
- conda env: `pointdet`
- train samples: 2,048
- eval samples: 512
- epochs: 8
- batch size: 16
- image size: `256x256`
- device: `cuda`
- output directory:
  `outputs/task1/baseline_unet_multiclass_weighted_gpu_pilot/`

Final checkpoint metrics:

- train loss: 0.21976511809043586
- eval loss: 0.3966479431837797
- eval DSC: 0.2515822948807326
- eval IoU/mIoU: 0.16684544688099584
- per-class DSC:
  - `label_1`: 0.4214537409553657
  - `label_2`: 0.08171084880609952
- per-class IoU:
  - `label_1`: 0.28878716438291563
  - `label_2`: 0.044903729379076066
- pixel accuracy: 0.9690150618553162

Best epoch by eval DSC during this run:

- epoch 6
- eval DSC: 0.27327491757102007
- eval IoU/mIoU: 0.1785187652007422
- per-class DSC:
  - `label_1`: 0.4226768114301553
  - `label_2`: 0.12387302371188477
- per-class IoU:
  - `label_1`: 0.28546346369312314
  - `label_2`: 0.0715740667083612

Saved prediction evaluation over 32 exported PNGs from the final checkpoint:

- predictions: 32
- matched predictions: 32
- missing sample IDs: 0
- DSC: 0.2451207286065397
- IoU/mIoU: 0.1567709304987303
- per-class DSC:
  - `label_1`: 0.4194138854696924
  - `label_2`: 0.07082757174338694
- per-class IoU:
  - `label_1`: 0.27468783595898383
  - `label_2`: 0.03885402503847686
- pixel accuracy: 0.9696927070617676

Weighted prediction class distribution over 32 exported PNGs:

- predicted pixels:
  - label 0: 2,023,466
  - label 1: 23,561
  - label 2: 50,125
- resized GT pixels:
  - label 0: 2,083,026
  - label 1: 11,595
  - label 2: 2,531

Interpretation:

- Weighted CE fixed the prior "no `label_2` predictions" failure.
- The chosen inverse-frequency weights are too aggressive: `label_2` is now
  heavily over-predicted on the 32 exported eval samples.
- Overall DSC/IoU is worse than the unweighted pilot because the model trades
  broad false-positive `label_2` regions for small-class recall.
- Next implementation step should save best checkpoints during training and
  test a milder class-weight schedule or CE+Dice loss.


---
**Context compaction (auto) at 00:06**
Check git log and quality_reports/plans/ for current state.

## Phase 8 Progress

- Added best-checkpoint tracking to `src/cathaction/training/task1_baseline.py`.
- Added mild weighted CE pilot config:
  - `configs/task1/baseline_unet_multiclass_mild_weighted_gpu_pilot.yaml`
- Added mild weighted/best-checkpoint plan:
  - `quality_reports/plans/2026-05-21_task1-mild-weighted-best-checkpoint-pilot.md`
- Added test coverage for monitor-metric best checkpoint selection.

Verification before training:

- `conda run -n cathaction-task1 python -m py_compile scripts/task1/train_baseline.py scripts/task1/evaluate_baseline.py scripts/task1/predict_baseline.py scripts/task1/make_prediction_overlays.py src/cathaction/training/task1_baseline.py`
- `conda run -n cathaction-task1 pytest`
- Result: 19 tests passed.
- `git diff --check`

Mild weighted GPU pilot:

- config: `configs/task1/baseline_unet_multiclass_mild_weighted_gpu_pilot.yaml`
- conda env: `pointdet`
- class weights: `[0.02, 1.0, 2.0]`
- train samples: 2,048
- eval samples: 512
- epochs: 8
- batch size: 16
- image size: `256x256`
- device: `cuda`
- monitor metric: `eval.dice`
- output directory:
  `outputs/task1/baseline_unet_multiclass_mild_weighted_gpu_pilot/`

Best checkpoint metrics:

- checkpoint:
  `outputs/task1/baseline_unet_multiclass_mild_weighted_gpu_pilot/best_checkpoint.pt`
- best epoch: 3
- train loss: 0.2004014056874439
- eval loss: 0.33980269357562065
- eval DSC: 0.46236699014996263
- eval IoU/mIoU: 0.36690955949437576
- per-class DSC:
  - `label_1`: 0.5641163915148246
  - `label_2`: 0.36061758878510064
- per-class IoU:
  - `label_1`: 0.42054360492099896
  - `label_2`: 0.31327551406775256
- pixel accuracy: 0.993168979883194

Final checkpoint metrics:

- checkpoint:
  `outputs/task1/baseline_unet_multiclass_mild_weighted_gpu_pilot/checkpoint.pt`
- final epoch: 8
- train loss: 0.1195320330443792
- eval loss: 0.13376865559257567
- eval DSC: 0.35443269816979944
- eval IoU/mIoU: 0.24306551146930994
- per-class DSC:
  - `label_1`: 0.5214893140298007
  - `label_2`: 0.18737608230979824
- per-class IoU:
  - `label_1`: 0.3726631775776683
  - `label_2`: 0.11346784536095156
- pixel accuracy: 0.9868965148925781

Saved prediction evaluation over 32 exported PNGs from the best checkpoint:

- predictions: 32
- matched predictions: 32
- missing sample IDs: 0
- DSC: 0.5282670680660289
- IoU/mIoU: 0.42706114496120234
- per-class DSC:
  - `label_1`: 0.605359586343144
  - `label_2`: 0.4511745497889139
- per-class IoU:
  - `label_1`: 0.454140093666454
  - `label_2`: 0.39998219625595044
- pixel accuracy: 0.9934463500976562

Mild weighted best-prediction class distribution over 32 exported PNGs:

- predicted pixels:
  - label 0: 2,075,035
  - label 1: 20,465
  - label 2: 1,652
- resized GT pixels:
  - label 0: 2,083,026
  - label 1: 11,595
  - label 2: 2,531

Outputs:

- `outputs/task1/baseline_unet_multiclass_mild_weighted_gpu_pilot/best_checkpoint.pt`
- `outputs/task1/baseline_unet_multiclass_mild_weighted_gpu_pilot/checkpoint.pt`
- `outputs/task1/baseline_unet_multiclass_mild_weighted_gpu_pilot/metrics.json`
- `outputs/task1/baseline_unet_multiclass_mild_weighted_gpu_pilot/eval_best.json`
- `outputs/task1/baseline_unet_multiclass_mild_weighted_gpu_pilot/best_predictions/predictions.csv`
- `outputs/task1/baseline_unet_multiclass_mild_weighted_gpu_pilot/eval_best_saved_predictions.json`
- `outputs/task1/baseline_unet_multiclass_mild_weighted_gpu_pilot/best_overlays/`

Interpretation:

- Best-checkpoint saving now works and prevents losing the better epoch.
- Mild weights are a better tradeoff than inverse-frequency weights:
  - unweighted pilot predicted no `label_2` pixels in the 32 exported samples;
  - inverse-frequency weighted pilot predicted 50,125 `label_2` pixels against
    2,531 GT pixels;
  - mild weighted best checkpoint predicted 1,652 `label_2` pixels against
    2,531 GT pixels.
- Full 512-sample eval DSC for the mild best checkpoint is slightly lower than
  the unweighted pilot (0.4624 vs. 0.4720), but it actually models both labels.
- The next useful step is qualitative overlay review plus a less toy baseline:
  either Dice+CE loss or a pretrained segmentation backbone, still preserving
  case/procedure-level split discipline.


---
**Context compaction (auto) at 01:18**
Check git log and quality_reports/plans/ for current state.

## Phase 9 Progress

- Added and completed full released-split baseline plan:
  `quality_reports/plans/2026-05-21_task1-full-released-baseline.md`.
- Added full-run config:
  `configs/task1/baseline_unet_multiclass_mild_weighted_full_512.yaml`.
- Ran the first complete Task 1 multiclass baseline on all released
  animal/phantom training samples.

Runtime configuration:

- conda env: `pointdet`
- command style: `conda run --no-capture-output`
- model: `TinyUNet`
- label mode: `multiclass_012`
- class weights: `[0.02, 1.0, 2.0]`
- image size: `512x512`
- train samples: 18,758
- capped monitor eval samples during training: 512
- final full eval samples after training: 4,691
- epochs: 15
- batch size: 64
- workers: 8
- device: CUDA
- output directory:
  `outputs/task1/baseline_unet_multiclass_mild_weighted_full_512/`

Runtime notes:

- Initial full-run launch at batch size 8 used only about 4.0 GB GPU memory, so
  it was stopped before completing epoch 1.
- Batch size was raised to 64. During training, `nvidia-smi` reported about
  29,506 MiB used out of 49,140 MiB, with 100% GPU utilization.
- This is still a resized `512x512` baseline. True original-resolution training
  remains future work because animal/phantom image sizes are heterogeneous and
  require padding or bucketing before batching.

Training monitor result:

- best checkpoint by capped `eval.dice`: epoch 14
- best capped eval DSC: 0.512032602422059
- best capped per-class DSC:
  - `label_1`: 0.5406980488567339
  - `label_2`: 0.48336715598738395
- final epoch: 15
- final capped eval DSC: 0.3114492851385282
- final capped per-class DSC:
  - `label_1`: 0.4408843031222964
  - `label_2`: 0.18201426715476005

Best checkpoint full released eval:

- checkpoint:
  `outputs/task1/baseline_unet_multiclass_mild_weighted_full_512/best_checkpoint.pt`
- eval JSON:
  `outputs/task1/baseline_unet_multiclass_mild_weighted_full_512/eval_best_full.json`
- eval samples: 4,691
- DSC: 0.4437008138855879
- IoU/mIoU: 0.3399198392117686
- per-class DSC:
  - `label_1`: 0.442940108746356
  - `label_2`: 0.4444615190248199
- per-class IoU:
  - `label_1`: 0.29273946702334275
  - `label_2`: 0.3871002114001944
- pixel accuracy: 0.9904373834138739

Final checkpoint full released eval:

- checkpoint:
  `outputs/task1/baseline_unet_multiclass_mild_weighted_full_512/checkpoint.pt`
- eval JSON:
  `outputs/task1/baseline_unet_multiclass_mild_weighted_full_512/eval_final_full.json`
- eval samples: 4,691
- DSC: 0.271676945781372
- IoU/mIoU: 0.17325436992533275
- per-class DSC:
  - `label_1`: 0.3905182702943531
  - `label_2`: 0.15283562126839095
- per-class IoU:
  - `label_1`: 0.2497812967416261
  - `label_2`: 0.09672744310903937
- pixel accuracy: 0.9850121024218583

Saved-prediction subset from best checkpoint:

- exported predictions: 64
- matched predictions: 64
- missing sample IDs: 0
- eval JSON:
  `outputs/task1/baseline_unet_multiclass_mild_weighted_full_512/eval_best_saved_predictions.json`
- DSC: 0.5000091709365408
- IoU/mIoU: 0.3914642801116009
- per-class DSC:
  - `label_1`: 0.5299399969097738
  - `label_2`: 0.4700783449633079
- pixel accuracy: 0.9919760227203369

Prediction-vs-GT pixel distribution over the 64 exported predictions at
prediction resolution:

- predicted pixels:
  - label 0: 16,581,789
  - label 1: 165,449
  - label 2: 29,978
- resized GT pixels:
  - label 0: 16,660,928
  - label 1: 93,908
  - label 2: 22,380

Qualitative outputs:

- predictions:
  `outputs/task1/baseline_unet_multiclass_mild_weighted_full_512/best_predictions/`
- overlays:
  `outputs/task1/baseline_unet_multiclass_mild_weighted_full_512/best_overlays/`
- contact sheet:
  `outputs/task1/baseline_unet_multiclass_mild_weighted_full_512/best_overlays/contact_sheet.png`

Visual interpretation from the contact sheet:

- The model usually finds the main tool trajectory.
- `label_1` is often thick and over-predicted relative to the resized GT pixel
  count.
- `label_2` appears in shorter segments and is no longer the all-zero failure
  mode from the unweighted pilot.
- Some frames still show small disconnected false positives.

Verification:

- `conda run --no-capture-output -n pointdet python -u -c "... main_train(['--config','configs/task1/baseline_unet_multiclass_mild_weighted_full_512.yaml']) ..."`
- `conda run --no-capture-output -n pointdet python -u -c "... main_evaluate([... '--checkpoint','outputs/task1/baseline_unet_multiclass_mild_weighted_full_512/best_checkpoint.pt', '--max-samples','4691']) ..."`
- `conda run --no-capture-output -n pointdet python -u -c "... main_evaluate([... '--checkpoint','outputs/task1/baseline_unet_multiclass_mild_weighted_full_512/checkpoint.pt', '--max-samples','4691']) ..."`
- `conda run --no-capture-output -n pointdet python -u -c "... main_predict([... '--checkpoint','outputs/task1/baseline_unet_multiclass_mild_weighted_full_512/best_checkpoint.pt', '--max-samples','64']) ..."`
- `conda run -n cathaction-task1 python -c "... main_evaluate([... '--predictions-csv','outputs/task1/baseline_unet_multiclass_mild_weighted_full_512/best_predictions/predictions.csv']) ..."`
- `conda run -n cathaction-task1 python scripts/task1/make_prediction_overlays.py --eval-manifest configs/task1/splits/released_eval.csv --predictions-csv outputs/task1/baseline_unet_multiclass_mild_weighted_full_512/best_predictions/predictions.csv --output-dir outputs/task1/baseline_unet_multiclass_mild_weighted_full_512/best_overlays --max-samples 64`

Interpretation:

- This is the first real Task 1 baseline result for this repo.
- Use the best checkpoint, not the final checkpoint, for comparisons.
- The released eval DSC of 0.4437 is an engineering benchmark only and must not
  be described as hidden-test or challenge leaderboard performance.
- Next useful model step: CE+Dice loss and/or a pretrained encoder segmentation
  backbone, still using released_train/released_eval discipline.

## Phase 10 Progress

- User asked for a stronger library-based overnight baseline rather than
  continuing with the hand-written `TinyUNet`.
- Checked installed environments:
  - `pointdet`: no MONAI
  - `pointdet_mamba`: no MONAI
  - `cathaction-task1`: no MONAI
  - `cardiac-diffusion`: MONAI 1.5.2, timm, CUDA available with external
    execution permission
- Added plan:
  `quality_reports/plans/2026-05-21_task1-monai-overnight-strong-baseline.md`
- Added MONAI config:
  `configs/task1/monai_unet_dicece_full_512_overnight.yaml`
- Added launcher:
  `scripts/task1/run_monai_overnight.sh`

Strong baseline configuration:

- environment: `cardiac-diffusion`
- model: `monai.networks.nets.UNet`
- spatial dims: 2
- input channels: 3
- output channels: 3
- channels: `[32, 64, 128, 256, 512]`
- strides: `[2, 2, 2, 2]`
- residual units: 2
- loss: `monai.losses.DiceCELoss`
- class weights: `[0.02, 1.0, 2.0]`
- lambda dice: 1.0
- lambda CE: 0.5
- AMP: enabled
- image size: `512x512`
- train samples: all 18,758 released train samples
- training eval cap: 512 released eval samples
- epochs: 100
- batch size: 12
- output directory:
  `outputs/task1/monai_unet_dicece_full_512_overnight/`

Launcher decision:

- Plain `nohup ... &`, `setsid ... &`, and `screen -dmS ...` did not keep a
  background process alive in the Codex execution environment.
- `systemd-run --user` was tested and confirmed to keep a background service
  active.
- The overnight run was launched as:
  `cathaction-monai-overnight.service`.

Live run state after launch:

- launch time: 2026-05-21 01:46 CEST
- service: `cathaction-monai-overnight.service`
- log:
  `quality_reports/logs/task1_monai_unet_dicece_full_512_overnight.log`
- service status: active/running
- GPU check after launch: about 2,538 MiB used out of 49,140 MiB
- log showed epoch 1 progressing normally.

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

## Phase 11 Progress

- Checked the overnight MONAI run after completion.
- `cathaction-monai-overnight.service` was no longer present because the
  transient user service had finished.
- Output directory contained:
  - `best_checkpoint.pt`
  - `checkpoint.pt`
  - `metrics.json`
  - `resolved_config.json`
- Ran full released eval for both the best and final checkpoints.
- Exported 64 best-checkpoint predictions, evaluated the saved predictions, and
  generated overlays plus a contact sheet.

Training monitor result, capped 512-sample eval:

- best checkpoint: epoch 98
- best capped eval DSC: 0.6148080685840516
- best capped per-class DSC:
  - `label_1`: 0.6930938037567017
  - `label_2`: 0.5365223334114013
- final checkpoint: epoch 100
- final capped eval DSC: 0.6057077205690814
- final capped per-class DSC:
  - `label_1`: 0.6540682187782509
  - `label_2`: 0.5573472223599119

Best checkpoint full released eval:

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

Final checkpoint full released eval:

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

Saved-prediction subset from best checkpoint:

- exported predictions: 64
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

Comparison:

- TinyUNet full 512 best released eval DSC: 0.4437008138855879
- MONAI UNet DiceCE full 512 best released eval DSC: 0.5303603258369163
- Absolute DSC gain: about 0.0867

Interpretation:

- MONAI UNet + DiceCE is now the stronger Task 1 baseline.
- Use `best_checkpoint.pt` for comparisons, though final checkpoint is close on
  full released eval.
- These are released-folder engineering results only, not hidden-test or
  leaderboard results.


---
**Context compaction (auto) at 12:51**
Check git log and quality_reports/plans/ for current state.


---
**Context compaction (auto) at 16:01**
Check git log and quality_reports/plans/ for current state.


---
**Context compaction (auto) at 16:42**
Check git log and quality_reports/plans/ for current state.


---
**Context compaction (auto) at 17:03**
Check git log and quality_reports/plans/ for current state.


---
**Context compaction (auto) at 22:36**
Check git log and quality_reports/plans/ for current state.


---
**Context compaction (auto) at 10:47**
Check git log and quality_reports/plans/ for current state.


---
**Context compaction (auto) at 13:01**
Check git log and quality_reports/plans/ for current state.


---
**Context compaction (auto) at 16:25**
Check git log and quality_reports/plans/ for current state.


---
**Context compaction (auto) at 02:56**
Check git log and quality_reports/plans/ for current state.


---
**Context compaction (auto) at 14:18**
Check git log and quality_reports/plans/ for current state.


---
**Context compaction (auto) at 13:10**
Check git log and quality_reports/plans/ for current state.


---
**Context compaction (auto) at 16:19**
Check git log and quality_reports/plans/ for current state.


---
**Context compaction (auto) at 12:39**
Check git log and quality_reports/plans/ for current state.


---
**Context compaction (auto) at 13:11**
Check git log and quality_reports/plans/ for current state.


---
**Context compaction (auto) at 13:54**
Check git log and quality_reports/plans/ for current state.


---
**Context compaction (auto) at 10:20**
Check git log and quality_reports/plans/ for current state.


---
**Context compaction (auto) at 15:22**
Check git log and quality_reports/plans/ for current state.


---
**Context compaction (auto) at 16:21**
Check git log and quality_reports/plans/ for current state.


---
**Context compaction (auto) at 10:41**
Check git log and quality_reports/plans/ for current state.


---
**Context compaction (auto) at 10:39**
Check git log and quality_reports/plans/ for current state.


---
**Context compaction (auto) at 12:15**
Check git log and quality_reports/plans/ for current state.


---
**Context compaction (auto) at 15:23**
Check git log and quality_reports/plans/ for current state.


---
**Context compaction (auto) at 01:31**
Check git log and quality_reports/plans/ for current state.


---
**Context compaction (auto) at 13:38**
Check git log and quality_reports/plans/ for current state.


---
**Context compaction (auto) at 14:24**
Check git log and quality_reports/plans/ for current state.


---
**Context compaction (auto) at 11:57**
Check git log and quality_reports/plans/ for current state.


---
**Context compaction (auto) at 14:56**
Check git log and quality_reports/plans/ for current state.


---
**Context compaction (auto) at 15:38**
Check git log and quality_reports/plans/ for current state.


---
**Context compaction (auto) at 16:42**
Check git log and quality_reports/plans/ for current state.


---
**Context compaction (auto) at 20:09**
Check git log and quality_reports/plans/ for current state.


---
**Context compaction (auto) at 14:16**
Check git log and quality_reports/plans/ for current state.


---
**Context compaction (auto) at 15:35**
Check git log and quality_reports/plans/ for current state.


---
**Context compaction (auto) at 19:07**
Check git log and quality_reports/plans/ for current state.


---
**Context compaction (auto) at 21:26**
Check git log and quality_reports/plans/ for current state.


---
**Context compaction (auto) at 23:38**
Check git log and quality_reports/plans/ for current state.


---
**Context compaction (auto) at 00:53**
Check git log and quality_reports/plans/ for current state.


---
**Context compaction (auto) at 03:36**
Check git log and quality_reports/plans/ for current state.


---
**Context compaction (auto) at 04:15**
Check git log and quality_reports/plans/ for current state.


---
**Context compaction (auto) at 05:02**
Check git log and quality_reports/plans/ for current state.


---
**Context compaction (auto) at 05:36**
Check git log and quality_reports/plans/ for current state.


---
**Context compaction (auto) at 06:10**
Check git log and quality_reports/plans/ for current state.
