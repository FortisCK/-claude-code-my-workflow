# Diffusion v1 Long Run

**Date:** 2026-05-17  
**Status:** running  
**Config:** `code/training/configs/diffusion_v1.yaml`  
**Warm start:** `experiments/checkpoints/diffusion_v1_pilot/epoch_005.pt`

## Launch

Detached screen session:

```bash
screen -dmS diffusion_v1_long bash -lc 'cd /home/mingzhang/cardiac-artifacts && WANDB_MODE=offline /home/mingzhang/miniconda3/envs/cardiac-diffusion/bin/python -m code.training.train_diffusion --config code/training/configs/diffusion_v1.yaml --ckpt experiments/checkpoints/diffusion_v1_pilot/epoch_005.pt 2>&1 | tee experiments/runs/diffusion_v1/train_from_pilot_2026-05-17.log'
```

Monitoring:

```bash
screen -ls
tail -f experiments/runs/diffusion_v1/train_from_pilot_2026-05-17.log
nvidia-smi
```

## Run Parameters

- Engine: EDM
- Frozen VAE: `experiments/checkpoints/vae_v2/best_val.pt`
- VAE config: `code/training/configs/vae_v2_128.yaml`
- Train split: 800 paired cases from `data/imagecas/splits/v1.json`
- Patch size: 128³
- Latent shape: 4 × 32³
- Batch size: 4
- Target epoch: 200
- Checkpoint cadence: every 10 epochs
- Checkpoint directory: `experiments/checkpoints/diffusion_v1`
- WandB: offline mode
- Log: `experiments/runs/diffusion_v1/train_from_pilot_2026-05-17.log`

## Initial Monitor

The first attempt with `nohup ... &` exited immediately with an empty log, so
the run was relaunched under detached `screen`.

Confirmed startup:

- Screen session: `diffusion_v1_long`
- Resume checkpoint loaded at epoch 5
- Training resumed at epoch 6
- First logged loss: epoch 6 step 0, `edm=0.3473`
- Later early loss: epoch 6 step 50, `edm=0.3749`
- GPU utilization during early training: about 100%
- GPU memory during early training: about 48.1 GiB / 49.1 GiB

## Next Checkpoint

Because this run resumes from epoch 5 and `ckpt_every_epochs=10`, the first
formal long-run checkpoint should be:

- `experiments/checkpoints/diffusion_v1/epoch_010.pt`
