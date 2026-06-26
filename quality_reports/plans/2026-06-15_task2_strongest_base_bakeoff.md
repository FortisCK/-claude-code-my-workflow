# Plan — Task 2 Strongest Feasible Base Detector + Bake-off

**Status:** PROPOSED (supersedes the YOLOV-first framing in `2026-06-15_task2_base_detector_pivot.md`)
**Date:** 2026-06-15 · Deadline 2026-08-23 · Official evaluator 2026-07-10 · GPU `cardiac-diffusion` on RTX 6000 Ada 49GB
**Basis:** multi-agent research — 5 detector families, 30 candidates, 23 adversarially verified.

## Verdict

Picking the strongest base once is the right call (detector-agnostic scaffold makes a later swap cheap; short runway). But two honest corrections:

1. **YOLOV is the BAR, not a horse.** It already lost to EFF (14.11 vs 14.88) and we sit below both (phantom mAP50 ~0.10–0.13). Don't adopt it — beat it. (Re-proving YOLOV is wasted motion; the honest baseline already gives the floor.)
2. **A stronger base ALONE will not clear EFF.** The gap is on phantom = tiny, thin, low-contrast tips. The evidenced lever is the **tiny-object graft**: **NWD (Normalized Wasserstein Distance) label-assignment/loss/NMS + a P2 small-object head + 1024px**, grafted onto an env-safe base. NWD has real X-ray evidence (+1.6 mAP, Fine-YOLO PMC11175173; +1–4 AP on YOLOv5/7/8; +6.7 on AI-TOD). The base choice is mostly about **feasibility + license**, not raw tiny-object magic.

**Temporal note:** the research again confirms temporal is the *cheap bonus* axis, not the lever (YOWO/STEP/HIT lost; YOLOV temporal lost to per-frame EFF). So the primary base is **non-temporal**; a temporal head can be added later if it pays. (This deprioritizes the "switch to temporal" instinct in favor of tiny-object — by evidence, twice.)

## Recommended base

| Rank | Detector | Why | License | Feasible |
|---|---|---|---|---|
| **PRIMARY** | **RT-DETRv2-R18/R50** (HF `RtDetrV2ForObjectDetection`) + P2 + NWD graft | Only candidate that is a real trainable detector AND torch-2.11/cu128-robust AND Apache-2.0 (sidesteps AGPL-Ultralytics) AND emits per-frame boxes natively. Maximizes capability×feasibility×license. | Apache-2.0 | **yes** |
| #2 (CNN hedge) | **RTMDet-s + custom P2** | Light/fast CNN, structurally different from a DETR (insurance vs transformer overfitting the thin val) | Apache-2.0 | maybe (needs pinned cu128 sub-env) |
| #3 (ceiling) | **Co-DETR (R50/Swin-T) + P2** | High ceiling if RT-DETRv2 plateaus and env budget allows | MIT | maybe (memory-heavy, 8×A100 recipe) |
| probe | **DQ-DETR** | Genuine tiny-object specialist (AI-TOD-v2 SOTA, 86% objects <16px) — run only if issue looks like tiny-localization | Apache lineage (add LICENSE) | maybe (dead repo, CUDA patch) |
| **component** | **NWD graft** (not a base) | Highest-leverage piece (priority 68); grafted onto whichever base wins | Apache-2.0 | yes (few-line) |

**Rejected:** YOLOV (the bar, lost to EFF); **AttWire** (highest domain fit but license-blocked / TF2.10 / dead repo → architecture template only: rotated center-point head + DoG wire-attention); CFINet/SimD (mmcv-1.x/torch-1.x, won't build on cu128; aerial-only); YOLOFT/StreamYOLO (temporal=cheap axis, AGPL/CUDA-port/stale).

## Reality check (honest)

Will any candidate plausibly clear EFF 14.88 on *this* data? **Generic SOTA bases will NOT on their own** — RT-DETRv2's paper concedes small-object inferiority; DINO is the AI-TOD baseline specialists beat; Co-DETR/RTMDet have zero tiny/fluoroscopy evidence. **The realistic EFF-beating path is the NWD/P2/high-res graft, not the base swap.** The ~0.24 combined headline is animal-inflated (one 107-frame video); the real fight is phantom. Ceiling is gated by tiny-tip localization + low contrast + phantom→human shift — a base swap alone doesn't fix it.

## Bake-off design

**Shared harness (identical for all):** train on video-disjoint 35k (`configs/task2/splits/train_clean_*`); export 14-col CSV; eval via `scripts/task2/evaluate_candidate_honest_baseline.py` → **per-case phantom mAP50** primary (per-domain + global reported, never the combined headline). Round 1 = bare base, no NWD/P2, smallest backbone, to measure base capability vs the EFF/YOLOV floor without conflating graft gains.

**Order (cheapest/most-likely first):** 1) RT-DETRv2-R18 (floor probe = primary bet, same run) → 2) RTMDet-tiny+P2 (CNN diversity) → 3) DQ-DETR or Co-DETR-R50 (only if 1+2 stall after graft).

**Gates:** A — bare base must reach ≥~0.13 phantom per-case (not collapse below floor) to advance. B — winner + NWD+P2+1024px must show measurable phantom lift over its own bare number. C — anything not cleanly trainable + CSV-exporting ≥4 weeks before 08-23 is dropped (kills CUDA-port/legacy-env traps).

**Anti-overfit (thin 2-video val):** select on phantom per-case (not combined); prioritize the multi-video phantom retrain before locking a winner; tie-break on phantom+animal both holding (penalize single-video spikes); fix seeds + identical aug/res + frozen eval GT.

## Recommendation

**(b) Do NOT run a full bake-off first.** Run **RT-DETRv2-R18 as the cheapest floor probe — which IS the primary-base bet (same run)** — then graft NWD/P2/1024px, escalate only if it stalls. A simultaneous bake-off burns the short runway on env-fighting candidates before we know if the cheap horse + NWD already clears EFF.

**First concrete step:** RT-DETRv2-R18 via HF Trainer on video-disjoint 35k, 2-class, tip-scale res → 14-col CSV → phantom per-case mAP50 vs frozen GT (bare base, Gate A). Then NWD+P2+1024px on the same pipeline (Gate B) before touching #2.

## Risks
1. **Structural ceiling (highest):** gap may be inherent to tiny low-contrast tips; bet rides on NWD/P2 delivering its evidenced X-ray +1–4 AP (fluoroscopy transfer unproven).
2. **Validation thinness (2 videos):** any winner could be a single-video artifact → multi-video retrain before locking.
3. **NWD must be re-implemented** on the chosen base (MMDet repos won't build on cu128); graft gain likely more modest than the +6.7 headline.
4. **Env/CUDA-op risk** for escalation candidates (Co-DETR/DINO/DQ-DETR/RTMDet on cu128) — Gate C kills stalls.
5. **License:** avoid AGPL host; confirm DQ-DETR LICENSE; AttWire hard-blocked (template only).
6. **Domain shift (phantom→human hidden test):** no candidate has evidenced robustness — largest unhedged unknown.
