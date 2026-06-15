# Session Log — 2026-06-15 — State recovery + v3a close-out

## Goal

Resume the cardiac-artifacts project after a gap. Two user asks:
1. Retrieve and read the prior Codex session ("查找 Claude Code 会话").
2. Map the full current project state, then **close open loops before discussing
   next steps** (user-approved sequence + scope: experiment + docs together).

## What was done

**Codex session retrieval.** Found thread `019e365d` (`~/.codex/sessions/2026/05/17/
rollout-...019e365d...jsonl`, 43 MB, 2026-05-17 → 06-12) — actually the main
cardiac-artifacts work session. Exported a clean transcript to
`codex_session_查找ClaudeCode会话.md` (untracked scratch). Note: the old Claude
Code transcripts it referenced (`~/.claude/projects/.../b57a20c2...jsonl`) are
gone; only the current session remains.

**State map (6-reader workflow).** Confirmed the project is far past CLAUDE.md's
"Not started" table. Active line = **Reliability-Gated Posterior Residual
Diffusion** (frozen U-Net initializer → residual-EDM posterior mean+std → learned
GateNet, `x_final = x_u + g·μ_r`, g ≤ 0.25). Key results:
- U-Net v1 test100 MAE **38.07 HU** beats latent diffusion v1 (**72.94 HU**).
- v1 gate e060 test100: **-0.0762 HU** global vs U-Net, 100/100 cases; calibrated
  but globally diluted (gains concentrate in heart +0.65 / boundary +0.72 HU).
- v2 dense/sparse oracle gates DISCARDed (regress below v1).

**Open loop being closed: v3a.** `residual_gate_v3a` (9ch: + heart_mask,
boundary_band, residual_snr) was trained to epoch60 on 2026-06-13 but never
evaluated (no run card, no verdict). Launched val20 + test100 eval (reuses v1
posterior caches → ~1.5h, no diffusion re-sampling). Run card:
`experiments/runs/2026-06-15_1511_residual-gate-v3a-e060-test100.md` (RUNNING).
2-case smoke confirmed the 9ch eval pipeline works (global delta -0.072 HU).

**Docs close-out (verdict-independent parts done).**
- `CLAUDE.md` "Current project state" table rewritten to reflect reality.
- `MEMORY.md` +5 `[LEARN]` entries filling the 2026-05-01 → 06-13 gap.
- `aaai-first-cvpr-fallback-roadmap.md` Progress Log extended through 2026-06-15.

## Pending

- v3a val20 + test100 eval result → fill run card outcome + KEEP/DISCARD/ITERATE,
  patch the v3a verdict line in MEMORY.md (if material) and the roadmap Progress Log.
- commit the close-out (experiment + docs).
- THEN discuss next steps with the user.

## Open questions for next-step discussion (deferred)

- Venue call (CVPR vs MICCAI) still open; gate win judged too small to stand alone.
- Novelty #2 (posterior-sampling UQ) has no classical calibration metrics yet
  (ECE/coverage/reliability diagram) — needed to defend the UQ claim.
- Novelty #3 (downstream lumen eval) — `dice_lumen` still a STUB, not TAVI-calibrated.
- Whether to build v3b (v3a + weak sparse high-confidence oracle aux loss).
