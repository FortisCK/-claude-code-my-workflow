# Session log — 2026-05-01 — Diffusion-model code skeleton (Steps 1–8)

**Date:** 2026-05-01
**Author:** CMZ + Claude (Opus 4.7, 1M context)
**Plan:** [`../plans/atomic-munching-naur.md`](../plans/atomic-munching-naur.md) (COMPLETED)

## 高层目标
GPU 被占,所以这个 window 用来把整套 3D conditional latent-diffusion 的 code skeleton
搭起来,并通过 CPU smoke test。等 GPU 释放后翻一下 flag 就能直接进入真训练。
完全不动数据生成的 motion synth(那一块上一个 session 已经做完)。

## 完成项

12 个新代码文件 + 2 个 YAML config,全部跑过 CPU smoke:

- **数据层**:`preprocessing.py` (Step 1),`imagecas_dataset.py` + `transforms.py` (Step 2)
- **模型层**:`vae.py` (Step 3),`conditional_denoiser.py` + `edm.py` (Step 5,EDM 数学从
  HM-EDM/Karras 2022 移植,移除 lucidrains 依赖)
- **训练层**:`train_vae.py` + `vae_v1.yaml` (Step 4),`train_diffusion.py` +
  `diffusion_v1.yaml` (Step 6)
- **推理 / eval**:`posterior_sample.py` (Step 7),`metrics.py` + `run_eval.py` (Step 8)

Smoke 完整跑通:`run_eval --smoke` 端到端 → V_corrupted → 编码 z_cond → N=2 chains × 4
Heun steps → 解码 → PSNR/SSIM/NRMSE/dice + 不确定性图 → CSV + PNG。指标本身是 noise(用的是
未训练的 tiny model + 4 步采样),但 plumbing verified。

## Key 设计决定 / Gotchas (写进了 plan 的 Completion log)

1. **MONAI AutoencoderKL / DiffusionModelUNet** 要求所有 `channels` 是 `norm_num_groups`
   (default 32) 的整数倍。Smoke 配置降到 `[32]*4`(必须 ≥ 32,不能再小)。
2. **DiffusionModelUNet 的 middle block 总是带 attention**(无视 `attention_levels` 标志),
   所以 `num_head_channels[-1]` 必须能整除 `channels[-1]`。Smoke 路径用 `[32]*4`。
3. **Checkpoint 嵌入 model_cfg / denoiser_cfg**:smoke 训出来的 ckpt 用 tiny arch,
   real ckpt 用 full arch。Loader 优先读 ckpt 内嵌的 cfg,所以两种 ckpt 可以共用同一个
   `load_frozen_*` 函数。
4. **Smoke pool 包含 train+val**:只有 1 个 case 在 disk 上时,默认 5% val_split 会让
   train_ids 变空。Smoke 路径合并 train+val 后做 padding,保证 batch_size×4 个样本可用。
5. **EDM concat conditioning** 只对 z_t 做 c_in 缩放,z_cond 全幅拼接 —— 这个是 Karras
   preconditioning 唯一对 noisy input 生效的位置,把 condition 塞进网络但不缩放。

## 下一步(等 GPU 释放)

按 `atomic-munching-naur.md` 完成 log 的 5 步:
1. 生成至少 1 个 precomputed pair / case(或把 GPU motion_synth 接成 online factory)
2. 真训练 VAE Stage 1
3. 用 VAE-encoded train data 重新 fit `sigma_data`,更新 `diffusion_v1.yaml`
4. 真训练 EDM Stage 2
5. 跑 `run_eval` 在 test split

## 当前 GPU 占用情况
mcastro 的 nnUNet 还在跑(放假前占的)。无法预估释放时间。本次 session 完全 CPU,
未启动任何 GPU 任务。

## 文件清单
```
code/data/preprocessing.py        ~250 LoC
code/data/imagecas_dataset.py     ~190 LoC
code/data/transforms.py            ~80 LoC
code/models/vae.py                 ~80 LoC
code/models/conditional_denoiser.py ~80 LoC
code/models/edm.py                ~250 LoC (port from HM-EDM)
code/training/train_vae.py        ~330 LoC
code/training/configs/vae_v1.yaml
code/training/train_diffusion.py  ~440 LoC
code/training/configs/diffusion_v1.yaml
code/inference/posterior_sample.py ~80 LoC
code/evaluation/metrics.py         ~80 LoC
code/evaluation/run_eval.py       ~250 LoC
scripts/python/smoke_dataset.py   ~110 LoC (Step 2 smoke harness)
```

总计 ~2200 LoC + 2 YAML(plan 估的是 ~1380 LoC,实际多出来主要是 train_diffusion
的 dataloader / VAE-loader / EMA / config-resolution 这几块比预期细)。
