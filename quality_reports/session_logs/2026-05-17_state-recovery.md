# Session log - 2026-05-17 - state recovery before VAE verification

## Scope

Recovered context after moving from Claude Code to this Codex session. This was
a read-only audit first, followed by recording the actual VAE state before any
new verification or training work.

## Claude context

Project-local `.claude/` contains workflow rules, hooks, agents, skills, and
settings. It is not the prior conversation transcript store.

Prior Claude Code transcripts are under:

`/home/mingzhang/.claude/projects/-home-mingzhang-cardiac-artifacts/`

Main recovered transcript:

`b57a20c2-932b-48ec-b1d6-a1459c5f38ec.jsonl`

The last Claude Code state was a pause at 128^3 VAE finetune around epoch 310,
then a failed API response after the user requested continuing training. The
local repository logs show that the resume did happen afterward and completed.

## Current repository state

- Branch: `cardiac-artifacts`
- Upstream status before this log: behind `origin/cardiac-artifacts` by 1 commit
- Worktree before this log: dirty, with prior uncommitted code/config/script/run
  artifacts from the VAE v2 and Stage-2 preparation work.

Important stale docs found:

- `CLAUDE.md` current-state text is older than the code and run artifacts.
- `code/README.md` still says the code is skeleton-only.
- The original VAE v2 run card was left as PLANNED even though VAE v2 and 128^3
  finetuning have completed.

## Data state

- Raw ImageCAS cache: 1000 image NIfTI files and 1000 label NIfTI files.
- Processed ImageCAS cache: 1000 clean `.npz` volumes plus 2000 motion pair
  `.npz` files.
- Motion pair distribution: exactly 2 variants per case for all 1000 cases.
- `experiments/runs/generate_motion_pairs/failed.txt`: 0 failures.
- `experiments/runs/preprocess_all/failed.txt`: 0 failures.
- Current filesystem capacity at audit: about 800 GB available on the project
  filesystem.

The older pipeline log contains a no-space failure from a 4-variant attempt, but
the current pair cache and generation log show the usable state is complete for
2 variants per case.

## VAE state

The usable Stage-1 candidate is VAE v2 128^3 finetune:

- Config: `code/training/configs/vae_v2_128.yaml`
- Checkpoint directory: `experiments/checkpoints/vae_v2/`
- Resume log: `experiments/runs/vae_v2/train_128_v2_resume.log`
- Resume point: epoch 310
- Completed: epoch 500, 2026-05-16 03:17 CEST
- Wall time: 169582.1 s (47.1 h)
- Best checkpoint: `experiments/checkpoints/vae_v2/best_val.pt`
- Best epoch: 445
- Best `val/recon`: 0.011378614380955696
- Approximate HU L1: 23.3 HU, using 2048 HU per normalized unit
- Epoch 500 `val/recon`: 0.01175710055977106
- WandB run: `fortis/cardiac-vae/nqa0ot6v`

Decision recorded today:

Use `best_val.pt` rather than `epoch_500.pt` as the candidate frozen encoder
until visual/downstream verification says otherwise.

## Stage-2 state

Stage-2 code exists for both EDM and flow matching:

- `code/training/train_diffusion.py`
- `code/models/edm.py`
- `code/models/stochastic_interpolant.py`
- `code/models/conditional_denoiser.py`

But real Stage-2 should not be launched yet because both Stage-2 configs still
contain stale VAE v1 placeholders:

- `code/training/configs/diffusion_v1.yaml`
- `code/training/configs/flow_v1.yaml`

Before a real Stage-2 run:

1. Verify VAE reconstructions numerically and visually.
2. Estimate latent `sigma_data` from the VAE v2 train split.
3. Point Stage-2 configs to `experiments/checkpoints/vae_v2/best_val.pt` and
   `code/training/configs/vae_v2_128.yaml`.
4. Create a fresh Stage-2 run card.

## Immediate next task

Start VAE verification from the committed state: fixed validation/test cases,
numeric metrics, and visual slices comparing clean input to VAE reconstruction.
