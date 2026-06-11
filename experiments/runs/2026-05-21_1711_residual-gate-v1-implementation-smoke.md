# Run: residual_gate_v1 implementation smoke

Date: 2026-05-21

Purpose: Implement and smoke-test the first reliability-gated posterior residual
diffusion pipeline:

```text
x_u = U-Net(corrupted)
mu_r, sigma_r = residual diffusion posterior mean/std
g = GateNet(corrupted, x_u, corrupted - x_u, mu_r, sigma_r, |grad x_u|)
x_final = x_u + g * mu_r
```

New files:

- `code/models/residual_gate.py`
- `code/training/configs/residual_gate_v1.yaml`
- `code/training/train_residual_gate.py`
- `scripts/python/cache_residual_diffusion_features.py`
- `scripts/python/evaluate_residual_gate_full_volume.py`

Verification:

1. Static compile:

   ```bash
   /home/mingzhang/miniconda3/envs/cardiac-diffusion/bin/python -m py_compile \
     code/models/residual_gate.py \
     code/training/train_residual_gate.py \
     scripts/python/cache_residual_diffusion_features.py \
     scripts/python/evaluate_residual_gate_full_volume.py
   ```

2. CPU synthetic GateNet training smoke:

   ```bash
   /home/mingzhang/miniconda3/envs/cardiac-diffusion/bin/python \
     -m code.training.train_residual_gate \
     --config code/training/configs/residual_gate_v1.yaml \
     --smoke \
     --no-wandb
   ```

   Output checkpoint:

   ```text
   experiments/checkpoints/residual_gate_v1/smoke/epoch_000.pt
   ```

   Initial gate behavior was conservative, with gate mean about `0.005`.

3. Synthetic cache eval smoke:

   Output:

   ```text
   experiments/runs/_tmp/residual_gate_eval_smoke_out/
   ```

4. Real-model feature-cache smoke:

   ```bash
   /home/mingzhang/miniconda3/envs/cardiac-diffusion/bin/python \
     scripts/python/cache_residual_diffusion_features.py \
     --ckpt experiments/checkpoints/diffusion_v2_residual/pilot5/epoch_005.pt \
     --case-ids 21 \
     --manual-split smoke \
     --num-steps 2 \
     --n-samples 2 \
     --dtype float16 \
     --out-dir experiments/runs/_tmp/residual_diffusion_feature_cache_smoke \
     --overwrite
   ```

   Output:

   ```text
   experiments/runs/_tmp/residual_diffusion_feature_cache_smoke/smoke/case_21.npz
   ```

5. Real-cache one-batch GateNet training smoke:

   ```bash
   /home/mingzhang/miniconda3/envs/cardiac-diffusion/bin/python \
     -m code.training.train_residual_gate \
     --config code/training/configs/residual_gate_v1.yaml \
     --cache-dir experiments/runs/_tmp/residual_diffusion_feature_cache_smoke \
     --train-splits smoke \
     --epochs 1 \
     --limit-train-batches 1 \
     --run-slug residual_gate_cache_smoke \
     --ckpt-dir experiments/checkpoints/residual_gate_v1/cache_smoke \
     --num-workers 0 \
     --no-pin-memory \
     --no-wandb
   ```

   Output checkpoint:

   ```text
   experiments/checkpoints/residual_gate_v1/cache_smoke/epoch_001.pt
   ```

6. Real-cache learned-gate eval smoke:

   ```bash
   /home/mingzhang/miniconda3/envs/cardiac-diffusion/bin/python \
     scripts/python/evaluate_residual_gate_full_volume.py \
     --ckpt experiments/checkpoints/residual_gate_v1/cache_smoke/epoch_001.pt \
     --cache-dir experiments/runs/_tmp/residual_diffusion_feature_cache_smoke \
     --splits smoke \
     --device cuda \
     --roi-size 128 \
     --figure-cases 0 \
     --no-print-summary \
     --out-dir experiments/runs/_tmp/residual_gate_eval_real_cache_smoke
   ```

   Output:

   ```text
   experiments/runs/_tmp/residual_gate_eval_real_cache_smoke/
   ```

Notes:

- A first real-cache training smoke with default `num_workers=4` hung under the
  sandbox because PyTorch worker IPC requires socket operations. The training
  script now supports `--num-workers 0 --no-pin-memory` for local smoke tests.
  Formal GPU jobs can still use the YAML worker settings.
- The real-model cache smoke used `2` denoising steps and `2` samples only to
  validate the pipeline. It is not numerically meaningful.

Next command for a useful small cache:

```bash
/home/mingzhang/miniconda3/envs/cardiac-diffusion/bin/python \
  scripts/python/cache_residual_diffusion_features.py \
  --ckpt experiments/checkpoints/diffusion_v2_residual/pilot5/epoch_005.pt \
  --train-cases 100 \
  --val-cases 20 \
  --test-cases 5 \
  --num-steps 16 \
  --n-samples 4 \
  --dtype float16 \
  --out-dir experiments/cache/residual_diffusion_features/v1_train100_val20_test5
```

Then train:

```bash
/home/mingzhang/miniconda3/envs/cardiac-diffusion/bin/python \
  -m code.training.train_residual_gate \
  --config code/training/configs/residual_gate_v1.yaml \
  --cache-dir experiments/cache/residual_diffusion_features/v1_train100_val20_test5 \
  --run-slug residual_gate_v1_train100 \
  --ckpt-dir experiments/checkpoints/residual_gate_v1/train100 \
  --epochs 20 \
  --no-wandb
```

Then evaluate:

```bash
/home/mingzhang/miniconda3/envs/cardiac-diffusion/bin/python \
  scripts/python/evaluate_residual_gate_full_volume.py \
  --ckpt experiments/checkpoints/residual_gate_v1/train100/epoch_020.pt \
  --cache-dir experiments/cache/residual_diffusion_features/v1_train100_val20_test5 \
  --splits test \
  --max-cases 5 \
  --out-dir experiments/runs/residual_gate_v1/eval_train100_test5
```
