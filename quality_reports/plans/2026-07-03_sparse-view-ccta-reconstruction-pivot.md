# Plan — Research Pivot: Trustworthy Ultra-Sparse-View CCTA Reconstruction

**Status:** DRAFT (awaiting user/advisor evaluation)
**Date:** 2026-07-03
**Branch:** `claude/project-review-ia9d9w`
**Supersedes as active line:** the single-phase motion-artifact-correction spine (declared a structural dead end — see §1).
**Evidence base:** novelty scan `wf_f29d0fcb-575` (11 agents, web) + full-text confirmation of the 3 closest competitors (PSDM, CDPIR, De Paepe — PDFs read 2026-07-03).

---

## 0. 执行摘要 (Chinese exec summary for the user)

我们把主线从**单相位运动伪影校正**（已被三重锁结构性证死）转到**极稀疏视角增强 CCTA 重建**。根本区别：运动那条线的前向算子是"运动 DVF"——假的、真实数据上不可表示（R0 已量化）；稀疏视角的算子是 **cone-beam Radon——真实物理**，于是测量一致性 (DPS) 名正言顺，全视角重建即 ground truth，信息在投影里只是欠采样。三把锁（MMSE / 算子不可表示 / 无真值）在新问题上**全绿**。

但"稀疏视角 CT 重建"本身已饱和，不是论文。**我们的贡献活在三个限定词里**：(1) 增强 CCTA 冠脉管腔（竞品只做钙化/体模/腹部）；(2) 下游冠脉几何 Dice 评判（竞品全用 PSNR/SSIM）；(3) 标定不确定性 + 幻觉 guard（领域在喊、无人做的诚实性层）。三篇最近竞品（PSDM、CDPIR、De Paepe）全文确认**没有一篇占这个格子**。技术门槛已明确：必须在极稀疏区、下游冠脉指标上**打赢 ASD-POCS**（CDPIR 证明 naive 扩散在真实 OOD 下会输给它）。

三个待拍板点见 §6。此文档供评估，非最终定稿。

---

## 1. Why the previous line is closed (one paragraph, for continuity)

Single-volume cardiac motion-artifact correction is triple-locked: (a) **MMSE law** — an L1/L2 U-Net approximates the conditional mean and no point estimator beats it on distortion; the residual-diffusion long run confirmed the posterior mean converges *toward* the U-Net and the voxel-oracle headroom *shrinks* with convergence (`2026-06-15_1705`). (b) **Operator non-representability (R0)** — real native cardiac motion cannot be explained by any parametric cone-beam+DVF operator (real +0.6 % vs synthetic +21.2 % residual reduction, disjoint bootstrap CIs, n=20; `2026-06-21_r0-cohort-operator-fidelity`), so the DPS wins were inverse-crime artifacts of synthetic evaluation. (c) **No real paired ground truth** — a motion-free reference for a single cardiac phase does not physically exist. These three are one disease: an information-destroying, ground-truth-less inverse problem. Assets built (VAE, DPS, tomosipo/ASTRA forward projector, coronary-downstream eval harness, UQ calibration suite) are **not** sunk — they are the toolbox for the new line.

---

## 2. The new problem (precise statement)

> **Reconstruct diagnostically-usable coronary anatomy from ultra-sparse-view contrast CCTA projections, using a cardiac-native 3D latent diffusion posterior sampler under a real cone-beam operator, and prove the reconstruction is trustworthy on the downstream coronary task — beating classical iterative (ASD-POCS) and 2D-diffusion baselines on coronary-geometry Dice, while flagging where it cannot be trusted via calibrated uncertainty and a reference-free hallucination guard.**

**Why the three locks turn green here:**

| Axis | Motion line (dead) | Sparse-view line (live) |
|---|---|---|
| Forward operator | motion DVF — fabricated, non-representable | cone-beam Radon — **real physics** |
| Ground truth | none (no motion-free ref) | **yes** — full-view reconstruction is GT |
| Information | destroyed at acquisition | **present** in projections, merely undersampled |

Measurement-consistency (DPS) is legitimate here precisely because `A` is the true acquisition model. Inverse-crime is still a risk and is controlled by matched-but-not-identical geometry/noise between the synthesis and evaluation forward models (lesson carried over from the motion dead-end).

---

## 3. Novelty — confirmed against full text of the 3 closest works

"Ultra-sparse-view CT reconstruction" alone is **SATURATED** (dozens/yr on AAPM abdomen, PSNR-chasing). Our defensible cell is the intersection of three load-bearing qualifiers. Each was checked against the nearest competitor's full text:

| Competitor (full text read) | What it is | Why it does **not** occupy our cell |
|---|---|---|
| **PSDM** (Han et al., IEEE TMI 44(9):3629, 2025) | **Limited-angle** (120°/90°, *not* sparse-view), 2D-slice NCSNPP, PC+PDHG | Data = XCAT phantom + COCA *calcium* (non-contrast) + 1 GE case; metrics = PSNR/SSIM/HC/LBP only; **no coronary Dice, no downstream, no UQ**; LAD qualitative; DPS is their *future work* |
| **CDPIR** (Li/Han/…/Yu, IEEE TMI 2025, arXiv 2509.13576) | Sparse-view (55/984; 123-view clinical), SiT-transformer diffusion + ASD-POCS, **2D-slice, image-domain (no latent)** | Focus = cross-domain OOD robustness; data = AAPM/COCA/XCAT/1 GE/1 wrist; metrics = **PSNR/SSIM/LPIPS only**; no contrast-CCTA coronary lumen, no coronary Dice, no UQ, no cardiac-native prior |
| **De Paepe et al.** (LaTIM Brest, arXiv 2501.12249) | **Respiratory** 4DCT (radiotherapy), sparse-view + blind motion, wavelet-DPS | XCAT **phantom only** (they state real-data adaptation is future work); PSNR/SSIM only; not cardiac coronary, not contrast, no downstream |

**Three differentiators survive cleanly (no counterexample in any full text):**

1. **Contrast CCTA coronary *lumen*** as the target — all three use calcium / phantom / abdomen. Lumen is where stenosis assessment happens; a strictly more valuable and unoccupied target. Our ImageCAS (with coronary labels) is the enabling asset.
2. **Downstream coronary-geometry Dice** (Dice-RCA / lumen Dice) as the evaluation axis — all three report only pixel/perceptual metrics. Nobody ties sparse-view diffusion reconstruction to coronary structural correctness.
3. **Trustworthiness layer** — all three *name* hallucination as the danger (PSDM "false structures"; CDPIR "hallucinations—plausible yet incorrect"; De Paepe motion artifacts) but **none quantify it** with calibrated UQ or a downstream guard. Our sharpness-trap + reference-free hallucination guard + calibrated posterior-std UQ (ECE/coverage/AUSE + temperature scaling) is the rigorous answer the field is explicitly gesturing at.

Secondary methodological differentiator: **genuine 3D latent prior** — all three competitors are 2D-slice, image-domain.

**Strongest framing:** not "another sparse-view reconstructor," but *"the first to quantify whether ultra-sparse-view diffusion CCTA reconstruction is coronary-trustworthy, and how to know without ground truth."*

---

## 4. Reuse map — what we already have vs what is new

| Asset | Status | Role in new line |
|---|---|---|
| tomosipo + ASTRA cone-beam forward/back projector | **have** | swap operator from motion-DVF to sparse-view Radon; the crown jewel, now on a *real* operator |
| DPS / measurement-consistency sampler (`dps_*`) | **have** | core reconstruction engine; likelihood term becomes `‖A_sparse x − y‖` |
| 3D KL-VAE (`vae_v2`) + latent diffusion infra | **have** | candidate cardiac-native 3D latent prior (see open decision §6.2) |
| Coronary eval harness (Dice-RCA, lumen Dice, severity strata) | **have** | the downstream win axis — directly transfers |
| UQ calibration suite (ECE/coverage/AUSE + temp scaling) | **have** | trustworthiness leg — directly transfers |
| Hallucination guard (voxel-ratio, two-segmenter gap; ROC-AUC 0.80) | **have** | reference-free trust flag on real data |
| ImageCAS (contrast CCTA + coronary labels) | **have** | the differentiating dataset |
| **Unconditional/clean cardiac 3D prior for DPS** | **new** | may need training a clean-volume prior (current diffusion is a *residual* model) |
| **Sparse-view sinogram simulation pipeline** | **partly new** | forward-project clean CCTA → subsample views → add realistic noise; matched-not-identical to eval operator |
| **ASD-POCS / SART / TV baselines** | **new** | mandatory iterative baselines (must beat) |

---

## 5. Execution plan (staged, go/no-go gated — least compute first)

Every stage opens a run card *before* execution (experiments-protocol). GPU is shared (single A6000, ~22 GB always occupied by another job) — serialize, never two GPU jobs.

**Stage 0 — Sinogram sim + FBP/SART sanity (~0.5 GPU-day).**
Forward-project clean ImageCAS CCTA through tomosipo cone-beam, subsample to the target ultra-sparse view count, reconstruct with FBP + SART. *Gate:* artifacts are severe and coronary lumen is visibly destroyed at the chosen view count (confirms the regime is genuinely hard and worth a prior). Fix the view count here (§6.3).

**Stage 1 — DPS reconstruction, decisive first signal (~1–2 GPU-days).**
Plug the sparse-view operator into the existing DPS sampler with the current 3D prior. Reconstruct test5 → test20. Evaluate on the full harness: coronary Dice-RCA, lumen Dice, PSNR/SSIM/LPIPS, heart-MAE, **vs FBP, SART, ASD-POCS**. *Decisive gate:* does DPS beat **ASD-POCS** on coronary Dice at the ultra-sparse operating point? (CDPIR shows this is *not* automatic on real OOD data.)
- WIN → Stage 2.
- TIE/LOSE → diagnose: prior mismatch (→ §6.2, train clean cardiac prior) or view-count off-regime (→ revisit §6.3) before scaling.

**Stage 2 — Scale + trustworthiness layer (~2–3 GPU-days).**
Scale to test100. Add the two trust legs: (a) posterior-std UQ calibration (ECE/coverage/AUSE + temperature scaling) tied to coronary error; (b) reference-free hallucination guard sweep (sharpness vs structural Dice; guard ROC). Build the main comparison table + the killer figure (corrupted / FBP / ASD-POCS / DDS / DiffusionMBIR / ours / GT + error + uncertainty + guard overlays).

**Stage 3 — External validity + (optional) motion stress (~2 GPU-days).**
Second coronary-labeled dataset for cross-dataset generalization (§ data ask below). If the motion axis is kept (§6.1): stress-test reconstruction robustness under residual cardiac motion on real anatomy — framed as *characterization*, not operator-estimation correction (that path is dead per R0).

**Required baselines (reviewer-mandatory ladder):**
- Analytical: FBP/FDK.
- Iterative (must beat): SART, **ASD-POCS / TV-min**, PWLS.
- Supervised/unrolled: FBPConvNet, our own U-Net; ideally one unrolled/INR recon.
- Diffusion: score-SDE (Song 2022), **DPS** (own, ablation), DiffusionMBIR, **DDS** (fast comparator), ≥1 recent SOTA (CvG-Diff), and **PSDM / CDPIR** re-run or differentiated in the sparse-view regime.

---

## 6. Open decisions (need user/advisor call before finalizing)

1. **Motion axis — keep or cut?** Keep = deeper moat + uses prior work as an asset, but collides with LaTIM Brest and is harder. Cut = sharper, faster. *Recommendation:* keep, but only as a **robustness stress-test** (not operator-estimation correction); our own R0 finding de-risks the LaTIM scoop because their respiratory-DVF approach would hit the same non-representability wall extending to real cardiac motion.
2. **3D latent prior — retrain a clean-volume prior?** This is the biggest methodological differentiator vs the three 2D competitors, but costs GPU. Current diffusion is a *residual* model; DPS wants a clean-volume prior `p(x)`. *Recommendation:* decide after Stage 1 — if DPS with an adapted current prior already beats ASD-POCS, defer; if not, training a clean cardiac 3D prior is the fix and the differentiator.
3. **Ultra-sparse view count.** Load-bearing: diffusion's advantage over classical priors is regime-dependent and plateaus after ~10–15 projections (Cheung et al., ISBI 2025). Must operate in the extreme-sparsity zone (≈ ≤15–20 views) or the whole rationale weakens. *Recommendation:* fix in Stage 0; report the sparsity sweep so the operating point is defensible.

---

## 7. Risks & competitive watch

| Risk | Mitigation |
|---|---|
| **ASD-POCS beats naive diffusion on real OOD** (CDPIR-documented) | Win must be *earned* on downstream Dice at ultra-sparsity; if DPS alone ties, hybridize with iterative data-consistency (as CDPIR does) — but keep the contrast/coronary/UQ differentiators |
| **LaTIM Brest (same region) extends motion+sparse-view to cardiac** | Move fast; frame motion as stress-test; leverage R0 (their DVF approach hits the non-representability wall on real cardiac) |
| **Yu lab (UMass Lowell)** — owns PSDM + CDPIR, most active player | Our cell (contrast CCTA + coronary-lumen downstream + UQ + 3D latent) is orthogonal to theirs (calcium/phantom, image-quality, OOD, 2D); monitor arXiv |
| **A 2026 preprint closes the reconstruction cell** | Fallback = pure evaluation/benchmark contribution ("first coronary-geometry + calibrated-UQ + severity-stratified downstream *protocol* for undersampled cardiac reconstruction"), orthogonal to DM4CT (no cardiac scope) — survives even if the method is scooped |
| Inverse-crime creeps back in | Matched-not-identical synthesis vs eval forward operator; report the operator-fidelity budget as we learned from R0 |

**Target venues:** IEEE TMI (home) / MICCAI 2027 eval-analysis track. **Not** a CVPR SOTA slot — the contribution is trustworthiness + real-data + downstream, not a new sampler.

---

## 8. Data ask (raw projection confirmed unobtainable → adjusted)

- **Dropped:** real raw cardiac projection data (user confirmed basically unobtainable). Sparse-view will therefore be *simulated* from DICOM via forward projection — acceptable and standard (all 3 competitors do the same), but state it plainly and control inverse-crime.
- **Highest-value gettable:** a **second coronary-labeled CCTA dataset** (ASOCA / orCaScore / CAT08) for cross-dataset external validity (Stage 3). Public, cheap, addresses the "single dataset" referee critique.
- **Optional/high-effort:** CCTA with an independent clinical endpoint (ICA / FFR / CAD-RADS) — would upgrade the downstream trust story from "geometry Dice" to "clinical decision," but needs clinical collaboration.

---

## 9. Immediate next actions (once approved)

1. Open run card `experiments/runs/<ts>_sparse-view-sim-fbp-sart.md`; build the sinogram-sim + FBP/SART Stage-0 sanity.
2. Fix the ultra-sparse view count from the Stage-0 sweep.
3. Wire the sparse-view operator into the DPS sampler; run the Stage-1 decisive gate vs ASD-POCS.
4. Report back before scaling — the ASD-POCS gate decides whether the prior needs retraining.

**Verification (this plan):** the three competitor claims in §3 are grounded in full-text reads committed to the session record; the reuse map in §4 references assets that exist in `code/` and `experiments/`. No code changed by this document — it is a direction artifact for evaluation.
