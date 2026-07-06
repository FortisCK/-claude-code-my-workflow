# Plan — Controllable Annulus-Geometry CT Synthesis for Exact-GT TAVI Sizing (Pilot)

**Status:** DRAFT (pilot spec, ready to execute at LTSI)
**Date:** 2026-07-06
**Branch:** `claude/project-review-ia9d9w`
**Direction:** generation-as-star, *synthesis* (not recovery) — the honest home for the group's diffusion toolbox.
**Independence:** self-contained; does NOT build on the first paper (TRUST) — no landmark outputs, no uncertainty machinery reused. Same aortic-root data is fine.
**Provenance:** distilled from the generative-synthesis problem hunt (`wf_f0170847-5e5`, runner-up "TAVI-SizeBench") + the session's recoverability / MMSE / FID-trap filters.

---

## 0. 中文执行摘要

**做什么**:用你冻结的 VAE + latent 扩散,做一个**瓣环几何可控**的主动脉根 CT 生成器——你钦定瓣环周长/面积/偏心率,它渲染出贴合的 CT。**真值 = 你钦定的几何(精确、零观测误差),不是从图像量出来的。**

**为什么**:临床上验证一个自动 TAVI sizing 工具没有干净真值(专家间差 2–3mm)。合成"钦定尺寸"的主动脉根,就得到零观测误差的 GT,可以(a)当基准给 sizing 工具打分,(b)扫难例(高偏心、钙化 blooming、卡型号分界线)找工具在哪失效。

**命门**:生成器能不能把 CT 渲染到**亚毫米贴合**你钦定的瓣环。这是整条线的生死点,而它**几天就能测出来**。**先用现有 ~100 例瓣环轮廓做 pilot;gate 过了,再让医生补标扩规模。**

**诚实风险**:100 例配对偏小,亚毫米贴合更可能失败——所以先跑 gate、不先投入。**即使 gate 失败也有科学价值**(表征了这个数据规模下可控扩散合成的贴合极限)。

---

## 1. Problem & motivation

TAVI prosthesis sizing is read from the aortic annulus (perimeter/area) on pre-procedural CT. Automated sizing tools exist (3mensio, HeartNavigator, academic pipelines), but **validating them cleanly is impossible on real data**: there is no observer-free ground truth (inter-observer annulus variability ~2–3 mm), and you cannot order up specific/rare/borderline anatomies.

**The idea:** generate synthetic aortic-root CT conditioned on a *dictated* annulus geometry, so the sizing ground truth is **exact by construction**. Then:
- **(a) Exact-GT benchmark** — score a sizing tool's error in mm against the geometry you authored, with zero observer noise.
- **(b) Systematic stress-test** — sweep hard/rare/borderline geometries (eccentricity, asymmetric calcium blooming, prosthesis-threshold sizes) and map *where* a tool systematically fails or flips to the wrong prosthesis size — impossible with real data.

This is an **in-silico validation / device-testing** contribution (regulatory theme: FDA in-silico trials, ASME V&V40). Least-crowded lane found in the hunt: existing TAVI virtual cohorts are mesh/shape-only with no imaging appearance or calcium (e.g., HUG-VAS, arXiv:2507.11474); no direct competitor does **intensity-domain prescribed-annulus CT rendering** + a sizing-tool stress-test. Re-verify **CORA (arXiv:2603.24847, 2026)** full text before writing related work — it is the load-bearing coronary-synthesis scoop flagged by the hunt (near/after knowledge cutoff, medium confidence).

**Why it escapes the traps:** it is *synthesis* (no per-patient truth image to betray, unlike the dead de-artifacting/recon lines), and the win is **controllability faithfulness in mm + a real-tool failure pattern that reproduces on real cases** — NOT FID/realism. A pretty-but-uncontrollable root fails the faithfulness metric.

---

## 2. Approach — geometry-first, then appearance (resolves the "how do I know the true annulus" circularity)

Do NOT generate a CT and then wonder its annulus. Reverse the direction:

1. **Author the geometry** — the annulus is exact because you construct it as a shape (a contour), not a scalar. Truth lives in the authored geometry.
2. **Render the CT appearance** conditioned on that geometry (mask → image). The generator paints a plausible root that *conforms* to the dictated annulus.
3. **Verify conformance** — measure the annulus back from the rendered CT; the residual (authored vs measured-from-render) is the faithfulness metric AND the go/no-go gate.

Crucially, geometry authoring is **pure code from the existing `.pf` annulus contours — zero manual work in Slicer/MITK**.

---

## 3. Pilot plan (reuses existing assets; ~100 `.pf` contours + 620 unlabeled + VAE/EDM)

### Stage 1 — Geometry pipeline (pure code, ~1–2 days)
- **Parse** the ~100 MITK `.pf` PlanarPolygon files → annulus contour vertices + the stored geometry transform + spacing.
- **Measure** each contour: perimeter (edge-length sum), area (shoelace), max/min diameter + eccentricity (ellipse fit) → the *real* annulus-geometry distribution.
- **Perturb** to author controllable geometries: center a real contour, apply affine scale (→ target perimeter) and axis stretch (→ target eccentricity); extrapolate beyond the real range for rare/borderline sizes.
- **Rasterize** each authored contour to a binary mask on the volume grid via the stored transform + spacing.
- **Self-check (must pass first):** rasterize a real contour → measure the mask back → confirm it matches the source polygon to sub-voxel. This proves the geometry→mask→measure loop is itself exact before any generation.

### Stage 2 — Mask-conditioned generator (pilot scale, reuse VAE + latent EDM)
- Train a **mask → aortic-root CT** conditional generator.
- **Data split of labor:** appearance realism from the ~620 UNLABELED roots (unconditional pretrain of the prior); annulus conformance from the ~100 (annulus-mask, real-CT) pairs (conditional fine-tune).
- **Conditioning:** encode the annulus mask to latent resolution; inject via channel-concat or a ControlNet-style branch on the existing latent EDM.
- **Pilot ethos:** smallest model that can be *measured* for faithfulness — not SOTA realism. Even a rough renderer answers the gate.

### Stage 3 — Faithfulness gate (the decisive test)
- Build a **sweep** of dictated annulus geometries: real + perturbed across a grid of (perimeter/area × eccentricity), **including extrapolated rare/borderline sizes** (the cases the eventual paper needs).
- Generate a CT for each.
- **Measure the annulus back** at the KNOWN annular plane (known because you placed the mask there): segment the contrast lumen in-plane (threshold + connected component / active contour) → measure perimeter/area. This verifier is **independent** of the conditioning mask and of TRUST.
- **Residual** = authored vs measured-from-render, per case, across the sweep.

**Go / No-Go:**
- **GO** if mean |residual| ≤ ~1 mm AND no strong size-dependent bias across the sweep (faithfulness holds at rare/borderline sizes, not just near the real distribution). → the exact-GT premise is real; proceed to scale (clinician annotation for more pairs) + add calcium-blooming control + the sizing-tool stress-test.
- **NO-GO** if mean |residual| ~2–3 mm (≈ inter-observer variability it must beat) OR conformance collapses on extrapolated sizes. → the premise is unmet at this data scale; document the faithfulness limit (still a publishable negative characterization) and fall back (Option 2 below) or stop.

---

## 4. Honest risks

| Risk | Note |
|---|---|
| **~100 pairs is small for sub-mm conformance** | The binding risk. Appearance can lean on 620 unlabeled, but conformance learns only from ~100 pairs. Raises the odds the gate fails — which is exactly why the gate runs FIRST and cheap. |
| **Calcium blooming** | The clinically-relevant failure mode AND the hardest to render faithfully; deferred to post-gate (pilot can start calcium-free, add it only if the gate passes). |
| **Conformance drift on rare/borderline sizes** | The paper's value lives in extrapolated hard cases; conformance must hold there, not just in-distribution. The sweep tests this explicitly. |
| **Novelty / scoop** | Re-verify CORA (2603.24847) and the mesh-only virtual-cohort neighbors before writing; the wedge is intensity-domain + calcium + tool-under-test transfer. |
| **Value ceiling** | This is a benchmark/tooling contribution (clinically-adjacent, not a clinical home-run). Honest MICCAI/TMI/MELBA scope, not CVPR-star. |

**Fallback (Option 2) if the gate fails:** downscope to *controllable augmentation* — use the perturbed-geometry generations to improve a sizing/landmark model's robustness to annulus size/eccentricity/calcium. This does NOT need sub-mm GT (only plausible controllable variation that lifts a downstream real-test metric), so it survives the small-data regime, at lower novelty.

---

## 5. Open design decision (needs a call before Stage 3)

**How to measure the annulus back from the generated CT (the verifier).** Proposed: in-plane contrast-lumen segmentation at the known annular plane (threshold + connected component), independent of the conditioning mask and of TRUST. Alternative: an existing annulus-measurement routine if one is on hand. Decide before locking the gate; keep it independent to avoid circularity.

---

## 6. After the pilot (only if GO)
1. Clinician annotation to expand the (contour, CT) pairs beyond ~100 → stronger conformance.
2. Add controllable **calcium + blooming** rendering (the clinical failure mode).
3. Run the **sizing-tool stress-test** (external published tool as device-under-test) → failure-envelope curves.
4. **Transfer-to-real** exclusionary test: the failure pattern found in-silico must reproduce on the real Rennes TAVI cohort — the credibility anchor.

---

## 7. Assets reused
Frozen VAE + latent EDM; DPS/sampling code; ~100 `.pf` annulus contours; ~620 unlabeled aortic-root volumes; the group's geometric-measurement / exclusionary-test skill. New code: the `.pf`→controllable-mask geometry pipeline; the mask-conditioning on the latent EDM; the in-plane annulus verifier.

## 8. Immediate next actions
1. Open a run card `experiments/runs/<ts>_annulus-geometry-pipeline.md`; build + self-check Stage 1 (pure code, no GPU).
2. Pilot-train Stage 2 mask-conditioned generator (calcium-free first).
3. Run the Stage 3 faithfulness sweep → report mean |residual| + size-dependent bias → GO/NO-GO.
4. Report back before scaling or adding calcium.

**Verification (this plan):** Stage-1 self-check gates the whole pipeline on an exact geometry→mask→measure loop before any generation; the faithfulness gate is a direct mm truth metric, not realism. No code changed by this document — it is an executable spec.
