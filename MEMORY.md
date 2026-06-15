# Project memory — Cardiac CT Motion-Artifact Correction

Cross-session learnings (corrections, validated approaches, locked-in
decisions). When something is corrected or a non-obvious approach is
validated, append a `[LEARN:category]` entry below.

The upstream **template** development log (v1.5.x → v1.8.0 cycle lessons)
lives at [`.claude/archive/MEMORY-template.md`](.claude/archive/MEMORY-template.md).
It's reference material only — not load-bearing for this project.

> **Bilingual note.** Entries may be in English or 中文 + English terms.
> Pick whichever expresses the lesson more precisely. Anything that ships
> externally stays in English; learnings are internal.

---

<!-- Append new entries below. Most recent at the bottom. -->

## Project framing & strategic constraints

[LEARN:project] **单 A6000 算力预算 (48 GB)。** Latent-space diffusion 是默认路径
(VAE 把 256³ volume 压到 ~32×32×16);pixel-space cropped ROI (128×128×96)
is the explicit fallback if VAE reconstruction fidelity fails the HU-preservation
sanity check. Why: `simple_plan` §08 risk row 1 — VAE reconstruction error
exceeding motion-artifact magnitude collapses the whole latent route.
How to apply: every architecture-level decision should preserve the option
to retreat to pixel-space ROI; never lock in VAE-specific assumptions
(e.g., specific latent shapes hard-coded into downstream losses).

[LEARN:framing] **Three orthogonal novelty claims, each independently
defensible.** (1) gated-CCTA + latent-LDM + lumen-aware loss; (2) theoretically
grounded posterior-sampling UQ; (3) downstream-task-aware evaluation
(stenosis grading, TAVI landmark accuracy). Why: a single novelty claim is
fragile against one skeptical reviewer; three orthogonal claims create
redundancy. How to apply: when ablating or simplifying, never collapse two
novelty claims into one — the redundancy IS the contribution.

[LEARN:scope] **C2F-MC (Wang & Tamir 2025) is Path B; we are Path A.** They
exploit k-space linear forward operator (`y = M·F·S·Φ·x`) for non-rigid
cardiac MRI motion correction via DPS. CT image-domain has no equivalent
linear handle; ImageCAS provides only reconstructed DICOM. Why: this is the
elegant scope-cut paragraph that protects the manuscript intro from
"why don't you do posterior sampling on raw measurements?" reviews.
How to apply: cite C2F-MC explicitly in the intro; offer Path-B-lite
(image-domain pseudo-forward) as an ablation subsection; never claim our
image-domain method is theoretically optimal — only that it's the
realistic CT image-domain approach for this dataset class.

[LEARN:scope] **HM-EDM is a baseline template, not an unbeatable SOTA.**
Brain SSIM 0.17 → 0.51 with 100 cases and a 5-case reader study —
workshop-quality proof-of-concept. Their 128×128×50 patch + EDM σ-schedule
+ channel-concat conditioning runs on A100 32 GB; our A6000 48 GB has
headroom. Why: avoids over-emphasising HM-EDM as a generalisation gate.
How to apply: reproduce HM-EDM in cardiac as a method-level baseline
(showing it fails on cardiac geometry); do not credit it with cardiac
generalisation we haven't seen.

[LEARN:scope] **ProDM (Gong et al., Dec 2025) does NOT overlap our scope.**
They do non-contrast COCA Agatston scoring; their motion engine *only moves
calcium points* (inpaint → reinsert per projection angle → Radon). We do
contrast CCTA + lumen/stenosis with PAD whole-heart motion. Why: ProDM
appearing in arXiv created scooping panic — full-text read showed
methodological non-overlap. How to apply: cite ProDM, borrow their
task-driven differentiable-loss philosophy (Agatston-surrogate → our
lumen-geometry-aware loss), but do not concede scope overlap.

## Workflow conventions

[LEARN:workflow] **Bilingual policy.** English in `.claude/`, `CLAUDE.md`,
`manuscript/`, code docstrings, and external artefacts. 中文 + English terms
welcome in user-authored specs / plans / session logs / `[LEARN]` entries
/ run-card intent paragraphs / decision records. Why: matches the user's
natural register (per `simple_plan.html`) without breaking the workflow's
portability. How to apply: when authoring an internal artefact, default
to whichever language expresses the idea most precisely.

[LEARN:reproducibility] **No run card → not a run.** Every training or
evaluation invocation produces a Markdown run card under
`experiments/runs/YYYY-MM-DD_HHMM_<slug>.md`, written *before* training
starts (intent), updated *after* (outcome). Why: forces falsifiable
expectations to be locked in before seeing the loss curve — same logic as
preregistration. How to apply: see
[`.claude/rules/experiments-protocol.md`](.claude/rules/experiments-protocol.md);
`python-reviewer` flags training scripts whose docstrings don't reference a
run-card path.

[LEARN:workflow] **Plan-first for non-trivial tasks.** Multi-file changes,
ambiguous requirements, anything > 1 hour → enter plan mode → save plan to
`quality_reports/plans/YYYY-MM-DD_<short>.md` → wait for approval → execute.
Why: catches mid-plan pivots before they cost real time. How to apply:
covered in [`.claude/rules/plan-first-workflow.md`](.claude/rules/plan-first-workflow.md);
the very first session of this project (workflow adaptation) followed it.

## Tooling expectations

[LEARN:tools] **Active surface is ~16 skills + 7 agents.** Lecture-/R-flavoured
template surface lives under `.claude/archive/`. Restoring an archived
skill / agent / rule is a single `git mv`; see
[`.claude/archive/README.md`](.claude/archive/README.md).
How to apply: if a missing capability surfaces during a task, check
`.claude/archive/` before authoring something new from scratch.

## Mac ↔ LTSI 双 session 工作流(2026-04-30 锁定)

[LEARN:workflow] **Mac session(此处)= 战略层 / spec 层;LTSI session = 执行层 / 代码层。**

**Why**:用户在 LTSI A6000 服务器上会另开一个独立 Claude Code session 专门写代码 / 跑训练 / 填 run card。Mac 端 session 的角色是 PI / 思考伙伴,不写代码。这种分工最大化两端的发挥:Mac 端 context 长 / 文献丰富 / 战略思考清晰;LTSI 端贴近代码与 GPU,可以快速迭代。

**How to apply**:

| 此 session(Mac)做 | 此 session(Mac)不做 |
| --- | --- |
| Spec 维护(用户从 LTSI 反馈实验结果后,update spec ASSUMED 项 / Risk Register) | 写 `.py` / `.yaml` / `pyproject.toml` 等代码或配置 |
| Plan 起草(Week 3-4, Week 5-8,...各阶段 plan) | 实际跑训练 / 调参 / debug stack trace |
| 决策记录(重大 framing pivot 时新增 v3 / v4 decision-record) | 填 run card 的 outcome 字段(LTSI 端跑完才有数据) |
| 概念扫盲 + literature review(像今天 "lumen 是什么"+"为什么单相位"那一轮) | 写 dataloader / model / training loop |
| 论文草稿评审(`/review-paper`, `/seven-pass-review`, `/verify-claims`) | 跑 `kaggle datasets download` / `pip install` / `nvidia-smi` |
| 与 PI / Pascal / Carlos 讨论的话术准备(elevator pitch / FAQ) | 跑 `tmux` 长任务 |
| MEMORY.md 维护([LEARN] 条目积累) | LTSI 上的 conda env 操作 |
| Coordination:用户在两端之间传递信息时帮忙 reconcile | 任何需要 SSH 到 LTSI 才能做的事 |

**已经在 Mac 端 commit 的代码**(`pyproject.toml`, `code/data/paths.py`, `code/data/imagecas_loader.py`, `scripts/python/check_env.py`, `scripts/python/imagecas_inspect.py`, 6 个 `__init__.py`, run-card 模板)— **保留作为 LTSI Claude session 的 scaffolding**,不 rollback。LTSI 端 session 可以自由 keep / 重写 / 替换。从今天起 Mac 端不再写新代码。

**LTSI session 启动时的 hand-off package**(用户带过去的内容):
- spec v1.0 → [`quality_reports/specs/2026-04-29_master-spec.md`](quality_reports/specs/2026-04-29_master-spec.md)
- Week 1-2 plan → [`quality_reports/plans/wondrous-honking-gray.md`](quality_reports/plans/wondrous-honking-gray.md)
- V1 + V2 decision records → [`quality_reports/decisions/2026-04-29_research-direction-v{1,2}.{md,html,pdf}`](quality_reports/decisions/)
- CLAUDE.md + MEMORY.md(自动加载)
- supporting papers 全文 → [`master_supporting_docs/supporting_papers/`](master_supporting_docs/supporting_papers/)

LTSI session 不需要重新 onboard — 上面这些文档 self-contained 描述了项目 state。

## Motion synthesis pipeline 的实际选择(2026-05-01 LTSI session 验证)

[LEARN:methodology] **走 Path D(parametric DVF)而非 PAD(XCAT-based 4D-SSM)**。理由:**TT U-Net 的 4D-SSM 训练代码 NOT released**(只 release 5 个 PAD MATLAB demo 文件)— 即使付 $1000 买 XCAT license 也无法完整复现 PAD;只能在他们的 demo 输出上做 inference。这把 v1.1 spec 里的 "PAD pipeline 复现" path 实质性地 invalidate 了。Why: lit review 2026-05-01 从 TT U-Net §III-B 全文 + GitHub repo 检查得出。How to apply: 论文 framing 改为 "license-free parametric alternative to PAD"(positive framing,不是 fallback);spec MUST "PAD pipeline 单相位简化版" 改为 "parametric DVF pipeline";Operational AI #1(XCAT license ask)**RESCINDED**(任何 path 都不需要 XCAT)。

[LEARN:antecedent] **Lossau 2019 (CoMoFACT, MedIA 52:68-79) is closest published antecedent**,not Deng 2023 PAD。CoMoFACT 是 coronary-segment patch 上的参数化 forward model,我们做 whole-heart volume + paired-data-for-LDM。论文 related work 必须把 Lossau 2019 / CoMPACT 2019 作为 main antecedent,Hahn 2017 / Maier 2021/2025 作为 vessel-centerline 平行 lineage。Why: 这是 pre-Lossau / post-Lossau 的 framing 轴 — reviewer 问 "why didn't you just use CoMoFACT" 的标准回答是 "patch-level vs whole-volume + 2D motion vector vs 4-component DVF + classifier vs paired LDM training"。

[LEARN:lit-anchor] **CoVe 验证 fact-of-existence 强,但 numeric range claims 必须回 PDF 看 Fig/Table 自己确认**。Stöhr 2016 案例:CoVe 第一遍说 LV twist normal ~7-8° ± 3°(跨人群均值),但 PDF Fig 1C 显示 resting healthy peak twist ~15°(单人 peak,不是均值)。两者都对,但语义完全不同。Why: 数字论证(default 12° vs 15°)依赖于这个区分。How to apply: 任何 numeric anchor 进 spec / paper 之前,**必须看一次原 PDF 的 Fig/Table**,不依赖纯 CoVe(CoVe 只验"这篇文献存在 + DOI 对",不验 Claude 的语义解读)。

[LEARN:tooling] **LEAP install 失败,fallback tomosipo + ASTRA 工作**。LEAP 没 PyPI wheel,readthedocs 文档失效,源码 build 需要 CUDA dev tools。tomosipo (MIT) + ASTRA (GPL,只动态链接 wheel,不影响 paper 代码 license) 在 RTX 6000 Ada (CC 8.9, CUDA 12.8) 上 smoke test 通过。Why: motion synth 的 cone-beam 投影 + Parker FBP 都需要 differentiable / non-equispaced angle 的 backend。How to apply: spec §架构 forward projection 的 ASSUMED → CLEAR (tomosipo+ASTRA);LEAP 留作 "considered alternative" 在论文 method 里一句话提一下。

[LEARN:debug] **HU calibration scale 异常 ∝ 1/n_views,是 FDK normalization,不是 bug**。motion-day6 demo v3 看到 scale=0.0185 @ 1000 views,v5 看到 0.0370 @ 500 views,严格线性。Why: FDK 的 backprojection 把所有 view 的 contribution 累加,所以重建强度自然 scale with N_views。How to apply: synth pipeline 的 calibration step (V_clean 也走一次 forward+FBP no motion 当 baseline) 自动处理;不需要 special case。

[LEARN:totalsegmentator] TotalSegmentator API 两个 trap:(1) 新版(>=2.x) `total` task 把心脏合并成 `heart` 单 ROI;chamber-level 需要 `task="heartchambers_highres"`。(2) `ml=True` 时 `output` 参数当**文件路径**(`.nii.gz` 后缀);`ml=False` 时当**目录路径**。两种模式 API 不一致。How to apply: motion-day6 demo 已 patch 这两点;未来 update 版本时 watch 这俩 trap。

[LEARN:resource] **GPU contention 是真实约束**。`mcastro` 的 nnUNet 占了 39 GB / 48 GB(放假前留的,无人值守),只剩 2.8 GB,无法启动任何 training。LTSI session 2026-05-01 整个 windows 全 CPU smoke,等 GPU 释放。Why: A6000 是 shared resource,不是 dedicated。How to apply: spec timeline 不能假设 GPU always available;Week 5/6 训 VAE / LDM 时给 1-2 周 buffer 应对 contention;CPU smoke skeleton 已准备好,GPU 一释放 flip flag 即可启动真训练。

## 架构具体落点(LTSI 2026-05-01 实现)

[LEARN:arch] **Latent shape 实际是 192³ → 24³ × 4ch**(8x spatial compression),不是 spec v1.1 说的 256³ → 32×32×16。差别:(a) volume crop 到 192³ 围绕 heart bbox,不是全 256³;(b) latent 8x per-axis compression 是 MONAI default;(c) 4 channel 不是 16。Why: 192³ 是 heart bbox + 30mm pad 的实际大小;4ch latent 在 MONAI AutoencoderKL default,够用且参数少。How to apply: spec §计算预算 & 架构 update;Success Criteria 数字 target 不变(VAE 重建 RMSE 等指标在 192³ 上同样适用)。

[LEARN:framework] **EDM (Karras 2022) over DDPM**。从 HM-EDM `conditional_EDM_3D.py` ported,移除 lucidrains UNet 依赖,wrap MONAI `DiffusionModelUNet`。50-step Heun sampling 比 DDPM 1000-step 快 30 倍 → N=16 posterior 采样可行。Why: spec v1.1 没显式 lock framework choice;LTSI 实际选 EDM 是合理的(precedent + speed)。How to apply: spec §三个 Novelty 主张 #1 措辞精化加 "EDM-based",论文方法节明确写 EDM motivation。

## Stage-2 结果与路线转折 (2026-05-15 → 06-13 LTSI session)

> 这批条目补回 MEMORY.md 此前 6 周的缺口(2026-05-01 之后无 [LEARN])。详见 `experiments/runs/` 各 run card 与 `quality_reports/plans/reliability-gated-posterior-residual-diffusion.md`。

[LEARN:result] **监督 U-Net 作为点估计器全面击败 conditional latent diffusion**。test100 整卷:U-Net v1 (epoch200 EMA) MAE **38.07 HU** vs latent diffusion v1 (det 50-step) **72.94 HU**(约 1.9× 误差),heart/boundary/PSNR/SSIM 全面更优(`2026-05-19_1511`, `2026-05-19_1600`)。Why: 配对 HU-preserving restoration 里 supervised U-Net ≈ conditional-mean estimator,这是 latent-diffusion 在该类任务的经典困境。How to apply: 论文 diffusion 卖点必须重定位到 **posterior-sampling 不确定性 + reliability gating**,**不能**主张「diffusion 点估计更准」;`research-direction-v2` §07 novelty #1 措辞需相应收敛。

[LEARN:method] **路线转折:Reliability-Gated Posterior Residual Diffusion**(三层冻结级联)。`x_u = U-Net(corrupted)` → 冻结 residual-EDM 采 K 个残差 → `μ_r,σ_r` → 学习 GateNet `g` → `x_final = x_u + g·μ_r`(`g ≤ g_max=0.25`, `init_bias=-4.0` 初始保守)。Why: 直接把 posterior mean 残差加回去会过校正(test100 反而恶化到 49.55 HU);有用的修正信号高度局部,需要逐体素 reliability gate(`2026-05-21_1651` voxel-oracle 上限仅 -3.34 HU,block/scalar gate 几乎无效)。How to apply: 这是当前活跃主线;新实验都在 gate 之上做,不要重训 ungated 直加残差的版本。

[LEARN:result] **gate 校准良好,但全局增益被心脏外体素稀释**。v1 gate e060 test100:全局 MAE -0.076 HU 但 **100/100 例全改善**;heart -0.65 HU,boundary -0.72 HU,增益随伪影严重度上升(`2026-06-11_1058`)。calibration 显示高 gate 值→高改善单调成立(boundary gate≥0.2 → +2.97 HU),但 ~55% 体素(心脏外)gate<0.001 被抑制,non-heart 区仅 +0.031 HU(`2026-06-11_1615`)。How to apply: v3a 因此新增 `heart_mask + boundary_band + residual_snr=|μ_r|/σ_r` 三通道输入(9ch),把容量集中到心脏/边界;run card 自承「全局增益太小,不足以单独支撑论文级 performance story」→ 需并行做更强 deterministic 基线或把心脏区增益做大。

[LEARN:method] **oracle 监督 gate 反而更差,v1 的保守 final-image loss 是锚**。dense per-voxel oracle BCE → test100 +0.014 HU(比 U-Net 还差);sparse oracle → -0.007 HU(比 v1 的 -0.038 差)(`2026-05-22_1301/1322`)。Why: per-voxel oracle 匹配会过校正、在 sliding-window 推理下不鲁棒。How to apply: 不要用 dense/sparse oracle BCE 作主损失;只在 v1 conservative loss 之上加「弱、稀疏、高置信」释放校准(即拟议的 v3b)。

[LEARN:workflow] **gate 评估复用缓存的 posterior residual features,迭代极廉价**。`scripts/python/cache_residual_diffusion_features.py` 把 clean/corrupted/initial/μ_r/σ_r/heart_mask 落盘(`experiments/cache/residual_diffusion_features/`);`evaluate_residual_gate_full_volume.py` 从 config 的 `feature_names` 自动派生 boundary_band/residual_snr,无需改代码即可评估不同 gate 版本。Why: 那 ~4h 的 diffusion 采样只需做一次。How to apply: 评估新 gate(如 v3a/v3b)直接复用 `v1_test100`(test)与 `v1_train100_val20_test5`(val) cache,只花 ~1h gate 推理,不重跑采样。

[LEARN:result] **v3a「加结构通道」假设被证伪——v1 仍是最好的 gate**。residual_gate_v3a(v1 + heart_mask + boundary_band + residual_snr,9ch,同 v1 保守损失,epoch60)在 test100 上 100/100 例全改善,但**没跑赢 v1**:全局 MAE Δ -0.066 vs v1 -0.076、heart Δ -0.562 vs v1 -0.654(都在噪声内但一致偏弱),boundary 打平,只有 boundary-gradient L1 明显更好(-0.327 vs -0.228);gate mean 反而更低(0.0067 vs 0.0076)。Why: 保守 v1 损失从不奖励「用这些新通道」,gate 没有信号去 exploit residual_snr/mask,只多了输入维度。How to apply: **不要再靠堆输入通道改善 gate**;下一步应是 v3b = v3a 输入 + 弱稀疏高置信 oracle 辅助损失(给 gate 一个用这些通道的理由),而非继续加输入。Run card `2026-06-15_1511_residual-gate-v3a-e060-test100.md`,Decision DISCARD。
