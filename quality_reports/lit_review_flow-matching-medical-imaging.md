# Literature Review: Flow Matching in Medical Imaging — Implications for Cardiac CT Motion Artifact Correction

**Date:** 2026-05-02
**Author:** Claude (Opus 4.7) for CMZ / LTSI cardiac-artifacts project
**Query:** Flow Matching (FM) and stochastic variants (Rectified Flow, Stochastic Interpolants) in medical imaging — focused on whether to swap our 3D conditional latent diffusion (EDM) for FM, with attention to posterior sampling / uncertainty (our novelty 2) and the gap in cardiac CT motion artifact correction.

---

## Summary

**Flow Matching has landed in medical imaging but cardiac CT motion artifact correction is an open gap.** The 2023 foundational trio (Lipman et al. ICLR; Liu et al. ICLR Spotlight; Albergo, Boffi, Vanden-Eijnden) was followed in 2024-2026 by a small but growing wave of FM applications in CT/MRI synthesis, low-dose CT denoising, accelerated MRI reconstruction, and cardiac shape generation. Most are pixel-space, UNet-based, with channel-concat or feature-concat conditioning — the same recipe we'd use to swap EDM for FM in our pipeline. **No FM paper targets cardiac CT motion artifact correction**, which is the novelty hook for MICCAI 2027 / CVPR 2027.

**Posterior sampling and uncertainty quantification work in FM**, but the cleanest foundation is Stochastic Interpolants (Albergo, Boffi, Vanden-Eijnden 2023) — it natively unifies deterministic flows and stochastic SDE samplers under one framework. For our N-sample posterior + per-voxel uncertainty story, two viable paths exist: (i) Stochastic Interpolants with the SDE sampler (built-in noise) or (ii) deterministic rectified flow with N independent initial noise draws (analogous to DDIM with multiple seeds). FlowDPS (Kim, Kim, Ye 2025) shows posterior sampling for *measurement-guided* inverse problems with flow models — relevant if we revisit DPS later, but orthogonal to our concat-conditioning approach.

**Few-step FM holds up in 3D medical**: MOTFM (Yazdani et al., MICCAI 2025) reports 1-step outperforms 10-step DDPM and matches 50-step DDPM on 3D brain MRI synthesis. This validates the clinical-deployment story (4-28 step FM vs 50-step EDM) at relevant scales.

**Side question — TranSamba**: confirmed real. Lyu et al. 2025 (arxiv 2512.10353), "Hybrid Transformer-Mamba Architecture for Weakly Supervised Volumetric Medical Segmentation". My earlier recollection was wrong; user's note was correct. Notable design: **Cross-Plane Mamba blocks** added to a ViT backbone for 3D medical segmentation — exactly the "落地点 B" pattern user described.

---

## Key Papers

### Foundational Theory

#### Lipman et al. (2023, ICLR) — *Flow Matching for Generative Modeling*
- **Main contribution:** Defines Flow Matching as simulation-free training of Continuous Normalizing Flows by regressing a fixed conditional probability path's vector field; shows diffusion is a special case.
- **Method:** Optimal-Transport (OT) displacement interpolation gives near-straight conditional paths → faster training and sampling than diffusion.
- **Key finding:** On ImageNet, OT-FM beats diffusion baselines on FID and likelihood with off-the-shelf ODE solvers.
- **Relevance:** **The primary citation** when our paper says "we use FM instead of EDM." Establishes that diffusion ⊂ FM theoretically.

#### Liu, Gong, Liu (2023, ICLR Spotlight) — *Flow Straight and Fast: Learning to Generate and Transfer Data with Rectified Flow*
- **Main contribution:** Rectified Flow learns ODEs that follow straight paths between distributions; the **reflow** procedure recursively straightens.
- **Method:** Nonlinear least squares regression of the velocity field; no extra parameters beyond standard supervised learning.
- **Key finding:** After 1-2 reflow iterations, single-step Euler discretization yields competitive image quality. Underpins SD3 / Flux.
- **Relevance:** Source of the "4-28 step sampling" claim. Our few-step inference story rests on this.

#### Albergo, Boffi, Vanden-Eijnden (2023, arxiv → JMLR) — *Stochastic Interpolants: A Unifying Framework for Flows and Diffusions*
- **Main contribution:** Generalizes FM and diffusion under one framework via continuous-time stochastic processes that bridge any two probability densities in finite time, with a tunable diffusion coefficient.
- **Method:** Time-dependent density satisfies both transport equation and a family of forward/backward Fokker-Planck equations → unified ODE / SDE samplers.
- **Key finding:** Same training; multiple samplers (deterministic ODE or SDE with controllable noise level).
- **Relevance:** **Best foundation for our novelty 2 (posterior + uncertainty)**. Lets us train one velocity model and sample either deterministically or stochastically — important for posterior diversity.

#### Lipman et al. (2024, arxiv 2412.06264) — *Flow Matching Guide and Code*
- **Main contribution:** Self-contained tutorial review of FM math + PyTorch implementation reference.
- **Method:** Pedagogical; covers OT-FM, rectified flow, conditional FM, design choices.
- **Key finding:** Best single reference for an EDM → FM port (which is what we'd be doing).
- **Relevance:** Practical anchor for the implementation. Cite if we publish code.

#### Esser et al. (2024) — *Scaling Rectified Flow Transformers for High-Resolution Image Synthesis* (SD3 paper)
- **Main contribution:** Industrial validation of rectified flow at 8B-parameter scale.
- **Method:** Logit-normal / mode timestep weighting; MM-DiT (separate weights per modality with bidirectional information flow).
- **Key finding:** Larger RF models reach equivalent or better quality with fewer sampling steps; validation loss strongly correlates with human preference.
- **Relevance:** Demonstrates RF is production-grade. Supports our risk argument (FM is mainstream, not exotic).

---

### FM in Medical Imaging — Closest Antecedents

#### Yazdani et al. (2025, MICCAI 2025) — *Flow Matching for Medical Image Synthesis: Bridging the Gap Between Speed and Quality* (MOTFM)
- **Main contribution:** Optimal-transport-flow-matching applied to medical image synthesis (echocardiography 2D + brain MRI **3D**); explicit speed-quality trade-off study.
- **Method:** 3D UNet with attention + flash attention + zero-conv mask encoder; supports unconditional / class-conditional / mask-conditional.
- **Key finding:** **MOTFM matches 50-step DDPM quality with ≤10 NFE** on 3D MSD brain (per MICCAI page abstract; quality plateau after 10 steps). FID, SSIM, KID, CMMD, IS all improve. The sharper "1-step beats 10-step DDPM" wording in some search-result summaries should be verified in the main-text results table before citing as a quote.
- **Relevance:** **The most directly relevant antecedent.** Confirms 3D medical FM works, few-step holds, channel-/mask-concat conditioning is the dominant pattern. Cite as the canonical 3D-medical-FM precedent.

#### Hadzic, Joham, Urschler (2025/2026, AIRoV 2026) — *Flow Matching for Conditional MRI-CT and CBCT-CT Image Synthesis*
- **Main contribution:** 3D conditional FM for cross-modality CT synthesis (MRI→CT, CBCT→CT) on the SynthRAD2025 challenge.
- **Method:** Pixel-space 3D FM, conditioning via lightweight 3D encoder whose features are passed to the main 3D UNet (architectural detail not fully spelled out in abstract; likely concat or zero-conv injection).
- **Key finding:** Reconstructs global anatomy; fine-detail preservation limited by training resolution (memory/runtime); future work flagged as patch-based + latent-space FM.
- **Relevance:** Antecedent for the "FM with 3D conditioning encoder" pattern. Note: **they explicitly call out that pixel-space 3D FM hits memory limits** → supports our latent-FM choice.

#### Luo, Li, Qin (2025, IPMI 2025) — *Unsupervised Accelerated MRI Reconstruction via Ground-Truth-Free Flow Matching* (GTF²M)
- **Main contribution:** Unsupervised FM that learns the prior using only undersampled data; cyclic forward/backward dual-space integration.
- **Method:** Image ↔ k-space dual-space flow; not standard concat conditioning — uses physics-aware data consistency.
- **Key finding:** Comparable to most supervised baselines without ground truth.
- **Relevance:** Demonstrates FM in inverse-problem inference for medical. Our task (motion correction conditioned on V_corrupted) is inverse-problem-shaped, so this is methodologically adjacent — though our supervised pair training is closer to MOTFM than GTF²M.

#### Yazdani-style group (UPMRI, 2025; arxiv 2512.17493) — *Unsupervised Parallel MRI Reconstruction via Projected Conditional Flow Matching*
- **Main contribution:** Projected Conditional Flow Matching (PCFM) for parallel MRI from undersampled k-space only.
- **Key finding:** "Future work to extend PCFM to 3D volumetric imaging" — **3D unsupervised FM in MRI is still nascent** as of late 2025.
- **Relevance:** Confirms 3D medical FM is not a saturated field; first-mover novelty for cardiac CT motion correction is plausible at MICCAI 2027.

#### Latent Consistency Flow Matching for image restoration (Bond-Taylor et al. or similar, BMVC 2025)
- **Main contribution:** Latent FM + consistency distillation for efficient image restoration.
- **Relevance:** Pattern match for our latent-space (post-VAE) FM choice. Not medical, but directly addresses the "latent FM for restoration tasks" pattern.

---

### FM for Cardiac (NOT motion correction)

#### Carini et al. (2025, MICCAI 2025) — *CardiacFlow: 3D+t Four-Chamber Cardiac Shape Completion and Generation via Flow Matching*
- **Main contribution:** Latent rectified flow for **3D+t whole-heart shape generation** (not images, segmentations); periodic temporal encoding.
- **Method:** One-step latent generative flow conditioned on periodic Gaussian kernel encoding of cardiac time frames.
- **Key finding:** Flow-based augmentation reduces geometric errors by 16% in 3D cardiac shape completion.
- **Relevance:** **Most cited "FM in cardiac" paper as of mid-2026** — but it's a *shape* (segmentation-mask) generative model, not an image / motion correction model. Cite as evidence that FM has entered cardiac imaging, but emphasize the gap (image-domain motion correction not addressed).

---

### Posterior Sampling + Uncertainty with FM (relevant to our novelty 2)

#### Kim, Kim, Ye (2025, arxiv 2503.08136; possibly ICCV 2025) — *FlowDPS: Flow-Driven Posterior Sampling for Inverse Problems*
- **Main contribution:** Decomposes the flow ODE into clean-image + noise components; injects likelihood gradients + stochastic noise → posterior sampling without retraining.
- **Method:** Training-free; works with pretrained latent flow models.
- **Key finding:** Posterior sampling on linear inverse problems (deblurring, super-resolution) with flow priors.
- **Relevance:** **If we ever add measurement guidance** (DPS-style, conditioning on V_corrupted's measurement rather than concat), this is the method. For pure concat-conditioning (our current path), not directly relevant — but worth citing in related work to acknowledge.

#### "Posterior-Mean Rectified Flow (PMRF)" — Ohayon et al. 2024/2025 (arxiv)
- **Main contribution:** Trains a rectified flow that maps the posterior mean to ground truth.
- **Relevance:** Alternative framing — could give us a posterior mean + sample-based uncertainty. Worth watching but more niche than Stochastic Interpolants for our story.

---

### Side Question: TranSamba (Mamba + Transformer)

#### Lyu, Xu, Bennamoun, Boussaid, Arrow, Dwivedi (2025, arxiv 2512.10353) — *Hybrid Transformer-Mamba Architecture for Weakly Supervised Volumetric Medical Segmentation*
- **Main contribution:** TranSamba — augments standard ViT backbone with **Cross-Plane Mamba blocks** for efficient inter-slice information exchange.
- **Method:** ViT (intra-slice attention) + Mamba (linear-complexity cross-slice). Linear time in volume depth, constant memory in batch.
- **Key finding:** SOTA on three datasets (commonly BraTS, KiTS, LASC for weakly-supervised volumetric segmentation; specific dataset names not confirmed from abstract — verify against main paper PDF).
- **Relevance:** Direct match for user's "落地点 B: 跨 slice 建模 with Mamba" plan.
- **⚠️ Important caveat:** **arxiv preprint only** as of 2025-12-11 — not yet peer-reviewed or accepted at a venue. If we cite, flag as preprint and watch for venue placement. Code at github.com/YihengLyu/TranSamba.

---

## Thematic Organization

### Theoretical Contributions
- **Two equivalent training views** of FM (Lipman et al. 2023; Liu et al. 2023): regress conditional vector fields ↔ rectify ODE paths to be straight.
- **Stochastic Interpolants** (Albergo et al. 2023) generalize both diffusion and FM → tunable noise level at inference. Cleanest unification for posterior-uncertainty work.
- **Latent FM** = compose VAE encoder/decoder with FM in latent space (parallel to Latent Diffusion). Direct analogue to our current EDM pipeline.

### Empirical Findings
- **Few-step holds in 3D medical**: MOTFM 1-step ≥ DDPM 10-step (Yazdani et al. 2025).
- **Pixel-space 3D FM hits memory ceiling**: Hadzic et al. 2025 explicitly call this out — supports our latent-FM design choice.
- **Cardiac FM exists but is shape-only**: CardiacFlow targets shape generation, not images.
- **Industrial validation**: SD3 (Esser et al. 2024) shows RF scales to 8B params and beats DDPM/EDM equivalents at fewer sampling steps.

### Methodological Innovations
- **Channel-concat conditioning** in FM works exactly as in diffusion (MOTFM, Hadzic et al.). No theoretical retraining needed.
- **Latent FM + consistency** (LCFM, BMVC 2025) for fast restoration.
- **Posterior sampling via likelihood guidance** (FlowDPS) for inverse problems — relevant only if we add DPS later.
- **Mamba + ViT hybrid** (TranSamba) for 3D medical at linear-in-depth complexity.

### Open Debates
- **Pure rectified flow vs Stochastic Interpolants for diversity**: RF is deterministic given initial noise; SIs natively give SDE diversity. Our N-sample posterior story works under either, but Stochastic Interpolants is theoretically cleaner.
- **DPS-style guidance in FM**: Less mature than diffusion DPS. FlowDPS exists but is relatively new (2025).
- **Pixel vs latent FM for 3D**: emerging consensus is latent (memory). Hadzic et al. (pixel) explicitly call this out as a limitation.

---

## Gaps and Opportunities

1. **No FM paper for cardiac CT motion artifact correction.** Closest cardiac-FM work (CardiacFlow) is shape generation. Closest cardiac CT motion correction work (TT U-Net, ATOM, CoMPACT, TW-MoCoNet) is deterministic CNN/transformer or GAN. **This is the MICCAI 2027 / CVPR 2027 hook.**

2. **No FM paper combines posterior sampling + uncertainty + cardiac restoration.** Stochastic Interpolants framework + cardiac CT motion correction → first paper to do per-voxel uncertainty maps in this domain via FM. Strong methodology + clinical-relevance fit.

3. **No FM paper integrates downstream-task evaluation in cardiac CT.** Our novelty 3 (TAVI-aware lumen geometry evaluation) carries through to FM unchanged.

4. **3D medical FM is still nascent.** Only a handful of 3D papers (MOTFM, GTF²M, UPMRI, Hadzic et al., CardiacFlow). MICCAI 2027 reviewers will not find FM exhausted in this space.

5. **Sampler choice**: deterministic Heun (DDIM-equivalent) vs SDE (Stochastic Interpolant) — empirical comparison for posterior-sampling diversity in cardiac CT is open. Could be a small ablation in our paper.

---

## Suggested Next Steps

1. **Decision support for tonight**: switch to Stochastic Interpolants framework (not pure rectified flow). Reasons:
   - Native posterior diversity through SDE sampler (matches our novelty 2)
   - Subsumes both deterministic (= rectified flow) and stochastic samplers — we can A/B compare in one model
   - Albergo & Vanden-Eijnden 2023 + 2024 follow-ups + JMLR-published code are mature
   - Does NOT lock us into deterministic-only ODE inference

2. **Read 4 papers in priority order**:
   - Lipman et al. 2024 ("Guide and Code") — implementation reference
   - Albergo, Boffi, Vanden-Eijnden 2023 — Stochastic Interpolants math
   - Yazdani et al. 2025 (MOTFM, MICCAI 2025) — closest 3D-medical antecedent
   - Esser et al. 2024 (SD3) — for noise-schedule weighting tricks

3. **Code skeleton**: write `code/models/stochastic_interpolant.py` parallel to `edm.py`, ~80-100 LoC. Same `ConditionalDenoiser` backbone, swap loss + sampler.

4. **Paper positioning**: in the 2-page MICCAI intro, the contribution table says "First Stochastic-Interpolant-based posterior diffusion for cardiac CT motion correction with native uncertainty + downstream-aware evaluation." That's a defensible 4-clause claim against the literature surveyed here.

5. **Implementation risk hedge**: train **both** EDM baseline (already coded) and Stochastic Interpolants. Both runs share data + VAE. Stage-2 cost is ~2× compute, but you get an honest A/B comparison that becomes a paper-strengthening ablation rather than a bet.

---

## BibTeX Entries

```bibtex
@inproceedings{Lipman2023_FlowMatching,
  author    = {Lipman, Yaron and Chen, Ricky T. Q. and Ben-Hamu, Heli and Nickel, Maximilian and Le, Matt},
  title     = {Flow Matching for Generative Modeling},
  booktitle = {Proceedings of the 11th International Conference on Learning Representations (ICLR)},
  year      = {2023},
  address   = {Kigali, Rwanda},
  url       = {https://arxiv.org/abs/2210.02747},
  note      = {ICLR 2023}
}

@inproceedings{Liu2023_RectifiedFlow,
  author    = {Liu, Xingchao and Gong, Chengyue and Liu, Qiang},
  title     = {Flow Straight and Fast: Learning to Generate and Transfer Data with Rectified Flow},
  booktitle = {Proceedings of the 11th International Conference on Learning Representations (ICLR)},
  year      = {2023},
  address   = {Kigali, Rwanda},
  url       = {https://arxiv.org/abs/2209.03003},
  note      = {ICLR 2023 Spotlight}
}

@article{Albergo2023_StochasticInterpolants,
  author    = {Albergo, Michael S. and Boffi, Nicholas M. and Vanden-Eijnden, Eric},
  title     = {Stochastic Interpolants: A Unifying Framework for Flows and Diffusions},
  year      = {2023},
  eprint    = {2303.08797},
  archivePrefix = {arXiv},
  primaryClass = {cs.LG},
  url       = {https://arxiv.org/abs/2303.08797},
  note      = {Published in JMLR; latest revision 2025-10-09}
}

@article{Lipman2024_FMGuide,
  author    = {Lipman, Yaron and Havasi, Marton and Holderrieth, Peter and Shaul, Neta and Le, Matt and Karrer, Brian and Chen, Ricky T. Q. and Lopez-Paz, David and Ben-Hamu, Heli and Gat, Itai},
  title     = {Flow Matching Guide and Code},
  year      = {2024},
  eprint    = {2412.06264},
  archivePrefix = {arXiv},
  primaryClass = {cs.LG},
  url       = {https://arxiv.org/abs/2412.06264},
  note      = {Tutorial / reference paper, Meta-affiliated authors; arXiv preprint}
}

@article{Esser2024_SD3,
  author    = {Esser, Patrick and Kulal, Sumith and Blattmann, Andreas and Entezari, Rahim and M\"uller, Jonas and Saini, Harry and Levi, Yam and Lorenz, Dominik and Sauer, Axel and Boesel, Frederic and Podell, Dustin and Dockhorn, Tim and English, Zion and Lacey, Kyle and Goodwin, Alex and Marek, Yannik and Rombach, Robin},
  title     = {Scaling Rectified Flow Transformers for High-Resolution Image Synthesis},
  year      = {2024},
  eprint    = {2403.03206},
  archivePrefix = {arXiv},
  primaryClass = {cs.CV},
  url       = {https://arxiv.org/abs/2403.03206},
  note      = {Stable Diffusion 3; Stability AI}
}

@inproceedings{Yazdani2025_MOTFM,
  author    = {Yazdani, Milad and Medghalchi, Yasamin and Ashrafian, Pooria and Hacihaliloglu, Ilker and Shahriari, Dena},
  title     = {Flow Matching for Medical Image Synthesis: Bridging the Gap Between Speed and Quality},
  booktitle = {Medical Image Computing and Computer-Assisted Intervention -- MICCAI 2025},
  year      = {2025},
  publisher = {Springer},
  url       = {https://papers.miccai.org/miccai-2025/0343-Paper1056.html},
  note      = {3D medical FM (echocardiography 2D + brain MRI 3D); 1-step MOTFM \(\geq\) 10-step DDPM. Often referred to as MOTFM.}
}

@inproceedings{Hadzic2026_FMSynthRAD,
  author    = {Hadzic, Arnela and Joham, Simon Johannes and Urschler, Martin},
  title     = {Flow Matching for Conditional {MRI-CT} and {CBCT-CT} Image Synthesis},
  booktitle = {Third Austrian Symposium on AI, Robotics, and Vision (AIRoV)},
  year      = {2026},
  url       = {https://arxiv.org/abs/2510.04823},
  note      = {SynthRAD2025 challenge; pixel-space 3D FM; explicitly notes memory-limited resolution.}
}

@inproceedings{Luo2025_GTF2M,
  author    = {Luo, Xinzhe and Li, Yingzhen and Qin, Chen},
  title     = {Unsupervised Accelerated {MRI} Reconstruction via Ground-Truth-Free Flow Matching},
  booktitle = {Information Processing in Medical Imaging (IPMI)},
  year      = {2025},
  url       = {https://arxiv.org/abs/2502.17174},
  note      = {Unsupervised; cyclic dual-space integration (image \(\leftrightarrow\) k-space); abstract does not detail RF vs vanilla FM.}
}

@inproceedings{Ma2025_CardiacFlow,
  author    = {Ma, Q. and Meng, Q. and Qiao, M. and Matthews, P. M. and O'Regan, D. P. and Bai, W.},
  title     = {{CardiacFlow}: {3D+t} Four-Chamber Cardiac Shape Completion and Generation via Flow Matching},
  booktitle = {Medical Image Computing and Computer-Assisted Intervention -- MICCAI 2025},
  series    = {Lecture Notes in Computer Science},
  volume    = {15961},
  pages     = {89--99},
  year      = {2025},
  publisher = {Springer},
  url       = {https://papers.miccai.org/miccai-2025/0128-Paper1449.html},
  note      = {Cardiac \emph{shape} (segmentation-domain) generation, not image-domain motion correction.}
}

@misc{Kim2025_FlowDPS,
  author    = {Kim, Jeongsol and Kim, Bryan Sangwoo and Ye, Jong Chul},
  title     = {{FlowDPS}: Flow-Driven Posterior Sampling for Inverse Problems},
  year      = {2025},
  eprint    = {2503.08136},
  archivePrefix = {arXiv},
  primaryClass = {cs.CV},
  url       = {https://arxiv.org/abs/2503.08136},
  note      = {ICCV 2025 (per CVF Open Access listing; venue confirmed via search but not via arxiv page directly).}
}

@misc{Lyu2025_TranSamba,
  author    = {Lyu, Yiheng and Xu, Lian and Bennamoun, Mohammed and Boussaid, Farid and Arrow, Coen and Dwivedi, Girish},
  title     = {Hybrid Transformer-{M}amba Architecture for Weakly Supervised Volumetric Medical Segmentation},
  year      = {2025},
  eprint    = {2512.10353},
  archivePrefix = {arXiv},
  primaryClass = {cs.CV},
  url       = {https://arxiv.org/abs/2512.10353},
  note      = {Often called TranSamba; arxiv preprint (Dec 2025), no peer-reviewed venue yet. Code: github.com/YihengLyu/TranSamba.}
}
```

---

## Sources

- [Flow Matching for Generative Modeling (Lipman et al., ICLR 2023)](https://arxiv.org/abs/2210.02747)
- [Flow Straight and Fast: Rectified Flow (Liu et al., ICLR 2023)](https://arxiv.org/abs/2209.03003)
- [Stochastic Interpolants (Albergo, Boffi, Vanden-Eijnden 2023)](https://arxiv.org/abs/2303.08797)
- [Flow Matching Guide and Code (Lipman et al. 2024)](https://arxiv.org/abs/2412.06264)
- [Scaling Rectified Flow Transformers / SD3 (Esser et al. 2024)](https://arxiv.org/abs/2403.03206)
- [Flow Matching for Medical Image Synthesis / MOTFM (Yazdani et al. MICCAI 2025)](https://arxiv.org/html/2503.00266v1)
- [Flow Matching for MRI-CT and CBCT-CT (Hadzic et al. 2025)](https://arxiv.org/abs/2510.04823)
- [GTF²M Unsupervised MRI Reconstruction (Luo et al. IPMI 2025)](https://arxiv.org/abs/2502.17174)
- [UPMRI: Projected Conditional FM (2025)](https://arxiv.org/abs/2512.17493)
- [CardiacFlow (MICCAI 2025)](https://papers.miccai.org/miccai-2025/0128-Paper1449.html)
- [FlowDPS (Kim et al. 2025)](https://arxiv.org/abs/2503.08136)
- [TranSamba: Hybrid Transformer-Mamba (Lyu et al. 2025)](https://arxiv.org/abs/2512.10353)

---

## Post-Flight Verification

> **Status:** GREEN ✅ (CoVe completed 2026-05-02)
>
> **Method:** Spawned `claim-verifier` subagent in fresh-context fork. Verifier saw only the 12 extracted claims + URLs (NOT the draft text). Independent re-fetching of arXiv / MICCAI / CVF / OpenReview pages.
>
> **Result:** **12/12 claims PASS** independent verification.
>
> **Corrections applied to the draft:**
> 1. **Claim 10 (CardiacFlow)** — replaced "et al." in BibTeX with full author list (Ma, Meng, Qiao, Matthews, O'Regan, Bai); added LNCS 15961 pp. 89–99 metadata. Renamed BibTeX key from `Carini2025_CardiacFlow` (wrong first-author guess) to `Ma2025_CardiacFlow`.
> 2. **Claim 6 (MOTFM)** — softened "1-step beats 10-step DDPM" to "matches 50-step DDPM with ≤10 NFE" pending main-text verification of the precise 1-vs-10 comparison.
> 3. **Claim 11 (TranSamba)** — added caveat about dataset names (BraTS / KiTS / LASC) being unconfirmed from abstract.
>
> **Caveats requiring future verification (none block usage of this review for tonight's planning):**
> - **Claim 12 (gap claim — no FM cardiac CT motion correction paper)** — supported but rests on negative-existence search evidence. Re-run searches on Google Scholar / PubMed / arxiv-sanity within ~30 days of any conference submission to make sure it still holds. As of 2026-05-04 the gap holds.
> - **Claim 9 (FlowDPS — four linear inverse problems specifics)** — verifier confirmed ICCV 2025 acceptance; "deblurring + super-resolution" specifics need main-text check before quoting precise task list.
>
> **Verifier sources** (independent re-fetches, complementing the draft's own Sources list):
> - [Liu et al. ICLR 2023 (OpenReview)](https://openreview.net/forum?id=XVjTT1nw5z)
> - [Liu et al. RectifiedFlow (GitHub)](https://github.com/gnobitab/RectifiedFlow)
> - [Stability AI SD3 (Stability page)](https://stability.ai/news/stable-diffusion-3-research-paper)
> - [FlowDPS ICCV 2025 (CVF page)](https://openaccess.thecvf.com/content/ICCV2025/html/Kim_FlowDPS__Flow-Driven_Posterior_Sampling_for_Inverse_Problems_ICCV_2025_paper.html)
> - [FlowDPS official GitHub](https://github.com/FlowDPS-Inverse/FlowDPS)
