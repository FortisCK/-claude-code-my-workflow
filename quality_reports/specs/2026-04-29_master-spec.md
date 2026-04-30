# Requirements Specification — Cardiac CT Motion-Artifact Correction (Master Spec)

**Date:** 2026-04-29(原始)/ 2026-04-30(v1.1 升级)
**Status:** APPROVED v1.1 (2026-04-30,CMZ 批准)
**Revision history:**
- v0.1 (2026-04-29): Initial spec from interview
- v0.2 (2026-04-29): TAVI to MAY-deferred / TT U-Net architecture clarification / reader study deferred to rebuttal / venue page-flexible
- v0.3 (2026-04-29): "为什么单相位" 专节加入 IN scope,基于 TT U-Net §III-B 全文 ground truth + Risk Register 加 reviewer 应对
- v0.4 (2026-04-29): 重写"为什么单相位"专节,诚实承认多相位四条优势,把研究论证升级为可验证科学 hypothesis(diffusion prior 能否补偿 temporal redundancy)
- v0.5 (2026-04-29): 加"伪影来源 framing"专节(PAD 是 physics-based synthesis)+ "真实测试数据 4-tier 策略"专节
- **v1.0 (2026-04-29): 用户批准**(MICCAI / TMI hybrid 路线)
- **v1.1 (2026-04-30): MICCAI / CVPR / AAAI 多 venue cascade + clinical entanglement 完全 prune**。用户决策:不和临床扯关系(避免和医生协调浪费时间),公开数据 only,demo 程度后再讨论导师。三档 venue cascade 形成自然 timeline。下列章节大改:venue 状态 / 下游评估 prune / reader study 完全删除 / Operational AI 缩到 1 项 / 真实测试数据 4-tier → 2-tier。新加 §Venue strategy / §Reproducibility commitments / §Baseline comparison plan / §Framing dual-track。撤回 v0.5 推过的 clinical decision support 升级(MICCAI / CVPR / AAAI 都不需要)。
**Source documents:**
- V1 (landscape底盘): [`../decisions/2026-04-29_research-direction-v1.md`](../decisions/2026-04-29_research-direction-v1.md) → [`v1.pdf`](../decisions/2026-04-29_research-direction-v1.pdf)
- V2 (focus层): [`../decisions/2026-04-29_research-direction-v2.md`](../decisions/2026-04-29_research-direction-v2.md) → [`v2.html`](../decisions/2026-04-29_research-direction-v2.html)
- Interview record: 本文档底部 §Interview Trace
**Author:** CMZ + Claude (via /interview)

---

## Objective(一句话)

在公开 ImageCAS 1000 例 contrast CCTA + PAD 仿真 数据上,训一个 **3D 条件 latent diffusion** 心脏 CT 运动伪影校正模型,提供 **后验采样不确定性** 与 **冠脉 lumen 下游评估**,投 **CVPR 2027 / MICCAI 2027 / AAAI 2028 三档 cascade**(根据 demo 表现 + 时间余量决定优先档)。**纯方法学论文,不涉及医生 / 临床 RCT / vendor 比较。**

---

## Venue Strategy(v1.1 新增)

| Venue | 截止 | 投递时机 | Reviewer mode | Framing emphasis |
| --- | --- | --- | --- | --- |
| **CVPR 2027** | ~2026 年 11 月 | 第 1 档(最高 bar 先冲) | CV 通用:methodology novelty 优先;医学细节不在乎;diffusion + 逆问题 + UQ 是熟悉地盘 | Hypothesis-driven(diffusion prior vs temporal redundancy);ablation rigor;general inverse problem framing |
| **MICCAI 2027** | ~2027 年 3 月 | 第 2 档(CVPR 拒后投) | Medical imaging:技术创新 + sound evaluation + reproducibility;无 RCT 要求 | 三个 novelty claims 平衡呈现;ImageCAS Dice / FOR-LIRS-MAS / UQ calibration 平铺直叙 |
| **AAAI 2028** | ~2027 年 8 月 | 第 3 档(MICCAI 拒后投) | ML 通用:Bayesian 逆问题 / calibrated UQ angle | 强调 BIPSDA-style UQ + selective prediction methodology;medical 应用作为 plus |

**Cascade 不增加工作量**:同一篇 paper draft,根据投哪里调整 abstract / intro 的 emphasis。Spec 的下游评估 / 方法学不区分 venue。

**导师讨论时机**:**2026 年 7-8 月**(demo presentable 后,LDM 出第一组 baseline + 可视化 before/after)。届时把 demo 给 Pascal/Carlos 看,问他们对 venue 选择是否有 LTSI 政治偏好。当前阶段(spec lock + Week 1-20)不主动联系导师。

---

## Scope

### IN(必须做)
- gated **contrast CCTA**(冠脉成像协议;ImageCAS 是这个分布)
- **单相位 input → 单相位 output**(见 §"为什么单相位"专节)
- **3D 条件 latent diffusion**(VAE-KL → latent 上的 conditional UNet denoiser),conditioning 通过 channel-concat 或 cross-attention(Week 5-8 ablation 决定)
- **非刚性全心运动**仿真(原版 PAD pipeline 起步,改进留 6 月底决定)
- **后验采样 UQ** + reliability diagram + ECE + coverage 校准(BIPSDA-style)
- **冠脉 lumen 自动评估**(Dice-RCA + 中心线 Dice + stenosis 分级 inter-method 一致性)— **全部 ImageCAS 公开 GT label / auto grader,不涉及医生**
- **Lumen-aware differentiable loss** as auxiliary training signal

### 为什么单相位(显式 framing 决策)

#### Ground truth(TT U-Net §I-§III-B 全文)
- 训练数据 = **MM-WHS 43 例单相位 clinical CT** + 4D SSM 合成 → PAD 48-frame pseudo all-phase
- 真实测试 = **uCT 960+ 5 例 retrospective ECG-gated scan**,48 frames per case
- 部署要求:**真实多相位 retrospective scan**(临床 5-15%,高剂量协议)

#### 多相位的客观优势(诚实承认)

TT U-Net §I + §II-A + §V(ablation)表明,多相位有四条**真实**优势:

1. **信息增益**:Table VI 显示 w/o TTL 时 SSIM 从 84% 掉到 73-81%,Dice 从 81% 掉到 73-76% — 时间维度提供 5-10 pp 真实信息增益
2. **临床应用唯一性**:**Functional cardiac CT**(心肌灌注、TMVR 动态规划)**只能多相位**
3. **First-principles correctness**:运动伪影根源是 4D 动态采集 → 4D 建模 first-principles correct
4. **Quasi-quiescent reference**:多相位里准静止相位是天然 clean reference

**所以 TT U-Net 不是"多此一举",他们解决的是另一个问题。**

#### 不同战场,不是竞争(简化版)

| 维度 | TT U-Net(多相位)| 我们(单相位)|
| --- | --- | --- |
| 数据来源 | 多相位 retrospective scan(获取受限) | ImageCAS 单相位 prospective gating(公开,任意获得) |
| 信息补偿机制 | Temporal redundancy | Diffusion prior |
| 不可逾越关系 | 我们没有 retrospective 数据/设备 | 他们的架构需要时间维度,无法降级 |

#### 我们的核心研究论证(可验证科学命题,venue-agnostic)

> **Hypothesis: A diffusion-based learned image prior can substantially compensate for the information loss that multi-phase methods exploit via temporal redundancy.**

可量化验证的预测(落入 Success Criteria):

| 我们的 single-phase SSIM 与 TT U-Net multi-phase 84% 的差距 | 解读 |
| --- | --- |
| **< 5 pp**(SSIM ≥ 80%)| Diffusion prior 几乎完全补偿了时间维度 — diffusion 在医学逆问题上的强证据 |
| **5-10 pp**(SSIM 76-79%)| 部分补偿;单相位场景下我们仍是该 niche SOTA |
| **> 10 pp**(SSIM < 76%)| Prior 不够强,需要重新思考 |

这个 hypothesis 就是 CVPR / AAAI reviewer 期待的"science question",同时也是给所有 venue reviewer 的"why not multi-phase" 可验证回答。

### 伪影来源 framing(基于 TT U-Net §II-B Fig 1-2)

**关键澄清**:PAD 是 **physics-based synthesis** —— 从干净图出发模拟 cone-beam CT 扫描物理过程(投影来自不同心动相位 → short-scan FDK 重建 → 时空混合 → motion 伪影自然涌现)。

**ImageCAS "几乎无伪影" 是 feature 不是 bug**:扮演 ground-truth clean reference 角色(等同于 TT U-Net 用 MM-WHS)。我们用单相位简化 PAD 直接合成 (clean ImageCAS, corrupted PAD-synth) paired 数据。

**Sim-to-Real Gap**(领域共同限制,显式承认):
- PAD 心脏运动模型来自 XCAT,不一定覆盖所有真实病人模式
- 真实扫描还有呼吸 / 心律不齐 / 钙化模糊等 PAD 未建模
- V1 §3.8 #3 + V2 §08 都承认这是领域共同限制 — **不是 spec showstopper**

### 真实测试数据 2-Tier 策略(v1.1 简化)

物理上不存在 (motion-corrupted, motion-free) paired 真实数据。所以测试两条线:**PAD-synthetic for 量化(有 GT),真实带伪影 for 视觉 + auto motion-metrics(无 GT)**。后者来源精简到 2 tier(撤回 v0.5 的 LTSI PACS 与 Deng 实验室合作 — 那两路涉及 operational 协调,与"不和临床扯关系"原则不符):

| Tier | 来源 | 预期数量 | 工作量 | 阻塞? |
| --- | --- | --- | --- | --- |
| **1**(主路径)| **ImageCAS 自然带伪影子集** — 从 1000 例按 HR > 70 / arrhythmia label / FOR-LIRS-MAS auto-score 筛选 | 30-100 例 | 低(Week 5-6 开发筛选 pipeline)| 否 |
| **2**(fallback)| 公开数据集:ASOCA / MM-WHS / TCIA CCTA | 10-30 例 | 低 | 否 |

**Tier 1 主路径足够**;Tier 2 是补充。

### ADJACENT(写入论文但非主线)
- DPS-style guidance 在 reverse SDE 注入(可选 ablation 子节)
- Image-domain 伪 forward DPS 作为 Path B-lite ablation
- HM-EDM 在心脏几何上复现作为 baseline
- TT U-Net 单帧版作为 baseline 下界

### OUT(明确排除)
- k-space / 投影域 posterior sampling(C2F-MC 的领地)
- non-contrast COCA Agatston 评分(ProDM 的领地)
- 多相位 video deblurring framing(TT U-Net 的领地)
- 头部刚性运动(HM-EDM 的领地)
- 非冠脉的瓣膜评估
- **Reader study / 医生评分**(全部 venue 均不必需,撤回 v0.5 中的 fallback 路径)
- **临床 outcome metrics**(FFR-CT 精度 / TAVI annulus 测量 / CAD-RADS reproducibility — 都需要 clinical context,我们走纯方法学路线)
- **Vendor 算法比较**(SSF2 等闭源,无法对比;reviewer 不会强求)
- **TAVI landmark 下游评估**(TRUST 论文有结果后再考虑;不进入当前 5–7 月 plan)

---

## 三个 Novelty 主张(必须保持正交)

| # | Claim | 实化形式 | Spec 状态 |
| --- | --- | --- | --- |
| **#1** | gated CCTA 上的第一个 **conditional latent diffusion** for motion correction with calibrated UQ and downstream lumen evaluation | 3D Latent LDM(基于 MONAI GenerativeModels 的 AutoencoderKL + DiffusionModelUNet) | CLEAR |
| **#2** | 心脏运动校正领域第一个 **理论严格的后验采样 UQ**(BIPSDA-style calibration:reliability diagram + ECE + coverage probability) | N=8–16 次 reverse sampling → 像素级方差;对比 MC-Dropout / TTA / ensemble | CLEAR(协议已锁,具体 N 由 Week 9-12 决定)|
| **#3** | 第一个 **下游任务感知 + lumen-aware differentiable loss** 评估协议 | (a) 评估侧:Dice-RCA + centerline Dice + stenosis 分级 inter-method 一致性(全 auto,公开 GT);(b) 训练侧:lumen-aware auxiliary loss(默认 centerline Dice via frozen pre-trained nnU-Net) | ASSUMED(Lumen loss 具体函数 Week 5-8 LDM 跑通后调) |

⚠️ **重要 framing 警觉**:不再使用"first diffusion-based cardiac CT motion correction"这种宽泛措辞 — 这个标题已被 ProDM (non-gated COCA) 部分占据。我们的精确 framing 必须包含 **gated + contrast + latent + UQ + downstream** 五个限定词同时出现。

### Framing dual-track(根据投哪里调 emphasis,v1.1 新增)

| Venue | Abstract / intro emphasis |
| --- | --- |
| **CVPR** | "We test whether a learned diffusion prior can compensate for temporal redundancy that multi-phase methods exploit, in the context of cardiac CT motion correction. We provide rigorous calibrated uncertainty (BIPSDA-style) and downstream-task evaluation as a benchmark protocol." |
| **MICCAI** | "We propose a 3D conditional latent diffusion model for cardiac CT motion correction with native posterior-sampling uncertainty and lumen-aware downstream evaluation. First method to combine these three elements on the public ImageCAS benchmark." |
| **AAAI** | "We address Bayesian inverse-problem solving via diffusion priors in the cardiac CT setting. Our calibrated UQ approach addresses the BIPSDA-flagged failure mode of DPS-class methods, with selective-prediction-style downstream task evaluation." |

主体 paper 内容相同,只调 abstract / intro 三段开头与若干 keyword。

---

## Requirements

### MUST Have(非协商)

#### 数据
- [ ] **ImageCAS 自然带伪影子集筛选 pipeline**(基于 HR / arrhythmia label / FOR-LIRS-MAS auto-score),Week 5-6 同步开发;预期产出 30-100 例 real-motion test set
- [ ] **PAD pipeline 单相位简化版**:从 ImageCAS 干净 → 单相位 corrupted 合成,physics-based(从 TT U-Net repo 改),不做多相位扩展
- [ ] **ImageCAS 1000 例**作为主训练数据
- [ ] **训/验/测 split** 锁定一次,文件提交到 `data/imagecas/splits/`,**禁止后期改 split**
- [ ] 每个训练 run 有对应 run-card(per [`.claude/rules/experiments-protocol.md`](../../.claude/rules/experiments-protocol.md))

#### 方法
- [ ] **VAE-KL 在 ImageCAS 上预训**,HU-preserving reconstruction loss(L1 + perceptual)
- [ ] **HU 保真度 go/no-go gate**:VAE 重建 RMSE < motion artifact magnitude(Week 5 sanity check;失败则 fallback 到 2D-axial + z-TV)
- [ ] **Conditional latent diffusion** 在 PAD-paired (clean, motion-corrupted) 数据上训
- [ ] **后验采样推断**:N≥8 次 reverse sampling,像素级方差作为 UQ map
- [ ] **Reliability diagram + ECE + coverage probability** 用于 UQ calibration

#### 评估
- [ ] **Dice-RCA**(右冠脉分割 Dice)— 该领域 headline metric,直接可比 TT U-Net
- [ ] **冠脉中心线 Dice**(via ImageCAS GT 训练的 nnU-Net frozen extractor)
- [ ] **PSNR / SSIM-Local / RMSE-Local** 在 PAD-synthetic test split — 与 TT U-Net 直接可比
- [ ] **FOR / LIRS / MAS** 在 ImageCAS 自然带伪影子集 — TT U-Net 真实临床数据用此
- [ ] **UQ calibration metrics**(reliability + ECE + coverage)
- [ ] **Stenosis 分级 inter-method 一致性**:对比 corrupted vs corrected 在同一 auto-grader 下(全自动,无需 ICA GT)

#### 工程
- [ ] 代码 + 模型 + 仿真 pipeline 全部开源(GitHub release 同 arXiv)
- [ ] 单卡 A6000(48 GB)可复现训练 — `pyproject.toml` 锁版本

### SHOULD Have(优先考虑)

- [ ] **额外 baseline**:Pix2pix(Zhang 2023) + I2SB(Path C)— 至少其一,加固 baseline 全集
- [ ] **Path B-lite ablation 子节**:image-domain 伪 forward DPS,展示考虑过更激进路线
- [ ] **Lumen segmentation Dice**(全冠脉,不只 RCA)
- [ ] **Lumen-aware loss ablation**:有 vs 没有该 loss,Dice-RCA 改善 ≥ 1 pp(novelty #3 数据)

### MAY Have(选做)

- [ ] **PAD-Pro** 改进(V1 §3.4):patient-specific cyclic motion atlases + 心率多样性 + 呼吸/床移 — 6 月底原版 PAD baseline plateau 时再投
- [ ] **3D consistency loss**(z-TV 风格)若 latent 在 z 方向出现切片间不一致
- [ ] **DPS-style guidance** 在 reverse SDE 注入,作为可选推断模式
- [ ] **Conformal prediction / ensemble distillation** 对 UQ 二次校准 — follow-up paper 候选
- [ ] **TAVI landmark 下游评估** — 触发条件:TRUST 论文已投出且有结果

---

## Reproducibility Commitments(v1.1 新增)

MICCAI / CVPR / AAAI reviewer 都查 reproducibility,所以这些是 paper-level commitments:

- [ ] `pyproject.toml` 锁主要库版本;`environment.yml` 提供 conda 等价 env
- [ ] 全部 random seed 固定,在 run card 与 paper supplementary 都记录
- [ ] 每个 figure / table / metric 都有可复现 script + 命令行参数
- [ ] 模型权重在 paper acceptance 后 release(目标:Zenodo / HuggingFace,有 DOI)
- [ ] PAD pipeline 仿真代码完全开源(从 TT U-Net repo 改起,开源同等 license)
- [ ] ImageCAS train/val/test split JSON 文件作为 supplementary
- [ ] (可选)Dockerfile 提供完整 reproducible env

---

## Baseline Comparison Plan(v1.1 新增)

MICCAI / CVPR 都要求至少 2-3 个合理 baseline,公平对比。MUST 全套:

| Baseline | 复现策略 | 单 / 多相位 | 预期作用 |
| --- | --- | --- | --- |
| **TT U-Net w/o TTL**(单相位 ablation 版) | 从他们 repo 训练,仅去掉 TTL 模块,用我们 ImageCAS 数据 | 单相位 | 单相位下界(SSIM 73-81%, Dice 73-76% 来自他们 §V Table VI)|
| **HM-EDM cardiac-reproduction** | 从他们 repo 改适应 ImageCAS;EDM 框架不变,数据替换 | 单相位(他们原本头部) | 同类 diffusion baseline(pixel-space EDM 对比 latent LDM)|
| **简单 U-Net**(Pix2pix 风格) | 直接训 paired (corrupted, clean) U-Net | 单相位 | 非生成式简单 baseline,做 sanity floor |

SHOULD(加分):
- Pix2pix GAN(Zhang 2023)
- I2SB Bridge(Liu 2023)— Path C 嵌入

每个 baseline 用相同 PAD-augmented training set + 相同 train/val/test split + 相同 metrics protocol。报告:per-method (PSNR, SSIM, Dice-RCA, centerline Dice, FOR/LIRS/MAS) 全套表。

---

## 计算预算 & 架构

| 项 | 决策 |
| --- | --- |
| GPU | 单卡 A6000 48 GB |
| 起点架构 | **3D 256³ latent**(VAE 压缩到 32×32×16);若 VAE 保真度失败,fallback 到 **2D-axial 256² + z-TV cross-slice consistency**(DiffusionMBIR 风格)|
| Patch 策略 | 训练:全 latent 体积 batch=2–4;推断:`SlidingWindowInferer` 全 256³ 体素 |
| 精度 | FP16 + AMP;VAE 训练用 FP32 sanity baseline 一次 |
| Seeds | 单一 `set_determinism()` per script,记录在 run card |
| 训练时长预算 | VAE 3–5 天,LDM 5–10 天;总训练 GPU-hour ≤ 500h |

---

## 时间线(v1.1 venue-cascade aware)

| 月 | 里程碑 | 关键输出 |
| --- | --- | --- |
| 2026-04(now) | Spec lock + Week 1-2 plan | 此 spec 批准 + (低优先级)向 Pascal 问 XCAT |
| 2026-05 | TT U-Net 复现 + ImageCAS 准备 + VAE 预训 + **HU 保真度 go/no-go gate** + **MUST 阈值 review** | VAE checkpoint + 第一个 run card |
| 2026-06 | Conditional LDM v0.1 + condition injection ablation + **架构是否升级 / PAD 是否改进 决策点** | LDM checkpoint v0.1,baseline PSNR/SSIM 接近 TT U-Net |
| 2026-07 | LDM v1.0 训练完成 + **Lumen-aware loss 函数确定** + **demo presentable** | Final LDM checkpoint;before/after 可视化;**导师讨论 #1**(把 demo 给 Pascal/Carlos 看,问 venue 偏好) |
| 2026-08 | 后验采样 + UQ calibration | reliability diagram;UQ vs MC-Dropout 对比表 |
| 2026-09 | 下游任务评估 + 全 baseline 复现 | Dice-RCA + centerline Dice + stenosis 一致性表;TT U-Net w/o TTL + HM-EDM cardiac + Pix2pix 全部跑完 |
| 2026-10 | **CVPR draft + ablation 完整 + 第二轮 baseline pass** | Paper draft v1;arXiv 占坑 readiness |
| 2026-11 | **CVPR 2027 投递** + arXiv 挂出 | First submission |
| 2027-Q1 | (若 CVPR 拒)**MICCAI 2027 投递**(3 月) | Re-frame abstract/intro;否则 CVPR rebuttal |
| 2027-Q2-Q3 | (若 MICCAI 拒)**AAAI 2028 投递**(8 月) | Final venue;若三档都拒 → 走 MIDL / WACV / Med Image Anal 期刊 |

---

## Clarity Status

| 维度 | 状态 | 备注 |
| --- | --- | --- |
| Project goal & scope | **CLEAR** | V1 + V2 + interview + v1.1 多 venue cascade reconciled |
| 起点架构(3D latent) | **CLEAR** | 用户选择,接受 VAE 保真度 go/no-go;5 月底 sanity check |
| 架构 fallback 路径 | **CLEAR** | 2D-axial 256² + z-TV(DiffusionMBIR 风格)|
| Conditioning 机制(channel-concat vs cross-attn)| **ASSUMED** | Week 5-8 ablation 决定;默认从 channel-concat 起步 |
| 下游评估范围(冠脉为主,纯 auto)| **CLEAR** | 用户选择 lumen 优先,纯 ImageCAS GT / auto grader,不涉及医生 |
| TAVI landmark 是否进入 paper | **MAY-DEFERRED** | 推迟到 TRUST 论文有结果后再考虑 |
| Lumen-aware loss 函数形式 | **ASSUMED** | 默认 centerline Dice via frozen nnU-Net,Week 5-8 决定 |
| Reader study | **OUT-OF-SCOPE** | v1.1 完全删除;MICCAI/CVPR/AAAI 都不必需;TT U-Net 在 TMI 也无 |
| 临床 outcome metrics(FFR/TAVI/CAD-RADS reader)| **OUT-OF-SCOPE** | v1.1 完全删除;走纯方法学路线 |
| Vendor 算法比较 | **OUT-OF-SCOPE** | 闭源无法对比 |
| XCAT phantom license | **BLOCKED-low** | 仅 PAD-Pro(MAY)需要,async 问 Pascal,Week 6 之前 |
| PAD 改进深度 | **ASSUMED** | 6 月底基于 baseline plateau 决定;原版起步 |
| Venue 选择 | **CLEAR (cascade)** | CVPR 2027 第 1 档 → MICCAI 2027 第 2 档 → AAAI 2028 第 3 档;7 月 demo 后导师讨论调 emphasis |
| arXiv 占坑节奏 | **CLEAR** | 11 月与 CVPR 投递同步挂 |
| UQ N-sample 数量 | **ASSUMED** | N=8–16 起步;Week 9-12 实验决定 |
| MUST 量化阈值合理性 | **ASSUMED** | 基于 TT U-Net §IV ablation 数据(SSIM 73-81% w/o TTL);Week 5 第一组 baseline 后调 |

---

## Success Criteria(分三档)

### MUST(论文能投)
- 在 ImageCAS PAD-synthetic test split 上,**SSIM-Local ≥ 80%、Dice-RCA ≥ 75%**(超越 TT U-Net 单帧版下界)
- 后验采样 UQ + reliability diagram + ECE + coverage 报告(无论 calibration 好坏,有报)
- 至少 2 个 novelty claim 有定量证据(#1 + #2 必须;#3 lumen-aware loss ablation 显示正贡献)
- 全 baseline 完整复现(TT U-Net w/o TTL + HM-EDM cardiac + 简单 U-Net)
- 全部代码 + 模型 + 仿真 pipeline 开源

### SHOULD(论文有竞争力,CVPR / MICCAI 高 acceptance 概率)
- 在 ImageCAS test split 上,**SSIM-Local ≥ 84%、Dice-RCA ≥ 81%**(close gap to TT U-Net 多相位版 SOTA;<5pp gap = diffusion prior 强证据)
- UQ calibration 显著优于 MC-Dropout 和 TTA(ECE 或 coverage 上 p < 0.05)
- Lumen-aware loss ablation 显示 Δ Dice-RCA ≥ 1 pp 改善
- Stenosis 分级 inter-method 一致性显示显著改善
- 至少 5 个 baseline 全部跑通

### NICE(award territory)
- 在 ImageCAS 自然带伪影子集上,FOR/LIRS/MAS 显著优于 baselines
- 跨数据集 zero-shot 泛化(ImageCAS → ASOCA / TCIA)
- BIPSDA-style 严格 UQ 偏差诊断报告(follow-up paper 雏形)
- (撤回 v1.0 中提到的 reader study award criterion — v1.1 不做)

---

## Verification(怎么知道 spec 满足)

1. **MUST 校验**:每条 MUST 在最终 paper 里都能指向具体 figure/table/section
2. **Cross-artifact 校验**:每个 MUST 量化指标都对应一个 `experiments/runs/<...>.md` 卡 + paper `% source:` 注释
3. **Reproducibility 校验**:他人 clone repo 后 `pip install -e . && python -m code.training.train_ldm --config <pinned>` 能复现 paper headline 指标(±tolerance,见 [`replication-protocol.md`](../../.claude/rules/replication-protocol.md))
4. **三 novelty claim 独立性校验**:每条 claim 即使另外两条被 reviewer 否决,本身仍站得住
5. **Venue cascade readiness**:Paper draft 在 Oct 2026 完成,abstract/intro 三种 framing emphasis 都准备好,投递时 1 小时切换

---

## Risk Register

| 风险 | 严重度 | 已规划 fallback |
| --- | --- | --- |
| VAE 保真度不够,latent 路线崩 | 高 | 2D-axial + z-TV(DiffusionMBIR 风格)|
| HM-EDM 团队抢发 cardiac 扩展 | 中 | 不激进策略;若发生,扩展 novelty 到 UQ + downstream + open benchmark 防御 |
| PAD pipeline 复现遇 MATLAB 阻塞 | 中 | Octave / 简化 DVF-warp + forward projection |
| ProDM 已经填补 "first diffusion + cardiac CT motion" | 已发生 | framing 精化为 "first conditional latent + gated + UQ + downstream",ProDM 在 related-work scope-cut |
| Reviewer 问 "vs TT U-Net 为何不用多相位 input" | 中 | 引用 TT U-Net §III-B(临床测试只 5 例 retrospective);单相位是临床主流;diffusion prior 补时间冗余信息缺失 |
| BIPSDA 报告 DPS 类方法 posterior variance 偏差可达 50% | 高 | reliability diagram 必报;若 calibration 差,加 ensemble distillation 或 conformal prediction(MAY)|
| Sim-to-real 泛化差 | 中(领域共同问题)| 这是领域共同限制;未来工作扩 |
| **CVPR reviewer 觉得 "incremental application"** | **中-高(v1.1 新增)** | (a) 强调 hypothesis-driven framing(diffusion vs temporal redundancy 是 testable scientific question);(b) BIPSDA-style UQ 在医学 inverse problem 上是首次;(c) MICCAI/AAAI fallback 接住 |
| **没有 clinical impact 论证 → reviewer 觉得论文 "academic exercise"** | **低(v1.1 新增)** | 三档 venue 都不要求 clinical RCT;methodological + offline metrics + reproducibility 足够;ImageCAS GT 上 Dice 直接对应 stenosis 评估精度,有 implicit 临床相关性 |

---

## Operational Action Items(v1.1 简化到 1 项)

仅 1 项,async 低优先级,不阻塞任何代码工作:

- [ ] **#1 — 问 Pascal/Carlos**:LTSI 是否已经有 XCAT phantom license(若无,联系 Duke Segars 组要 academic license,~6-8 周)— 仅 PAD-Pro 改进(MAY)需要。**Week 6 之前问就行**。

**已撤回的 operational items**(v1.1):
- ~~#2 LTSI / CHU Rennes PACS retrospective pull~~ → 不需要(临床数据 OUT-OF-SCOPE)
- ~~#3 给 Deng 实验室发邮件求合作~~ → 不需要(数据自给自足)
- ~~Pascal/Carlos 询问 reader study 医生~~ → 不需要(reader study OUT-OF-SCOPE)
- ~~TRUST detector zero-shot self-check~~ → TAVI MAY-DEFERRED

**导师讨论时机**(7-8 月 demo presentable 后)— 不算 operational AI(不需要现在做),记录在 §Venue Strategy。

---

## Interview Trace(决策记录,用于 audit)

### Round 1 (scope-shaping)
| 问题 | 用户选择 |
| --- | --- |
| 下游评估范围 | "先冠脉, TAVI landmark 作 stretch goal" |
| 起点架构 | "3D 256³ latent (V2 野心)" |
| 论文野心 | "看我们跑的结果和时间来定吧,两者都可以" → v1.0 hybrid → v1.1 三档 venue cascade |
| arXiv 占坑紧迫性 | "真被抢先了我们就再拓展新的,没必要很激进" |

### Round 2 (technical/operational)
| 问题 | 用户选择 / 反应 |
| --- | --- |
| Lumen-aware loss 函数 | "我并不太了解这个" → spec 标 ASSUMED + 默认 centerline Dice |
| Reader study 医生来源 | "需要医生做什么,这个不一定能找到医生" → v0 BLOCKED → v1.1 完全删除 |
| PAD 改进深度 | "先原玩, 6 月底根据 baseline 表现决定" |

### Concept clarification + framing iterations (2026-04-29 全天)
- "Lumen / 冠脉 / 为什么不全心" — 通过读 TT U-Net 全文,确认 Dice-RCA 是字段 standard
- "TT U-Net 是 3D 起步吗" — 它是 2D-spatial + 时间相位,不是 volumetric 3D
- "为什么我们不要时间维度" — 客观承认多相位 4 条优势,升级为 diffusion-vs-temporal-redundancy hypothesis
- "TT U-Net 的伪影是 PAD 生成还是单相位自带" — 它是 PAD 从干净 MM-WHS 通过 physics-based scan simulation 自然涌现
- "怎么找真实带伪影测试数据" — Tier 1 ImageCAS 自然子集为主路径

### Spec v0 review feedback (2026-04-29 post-draft)
- TAVI landmark → MAY-DEFERRED(等 TRUST 出结果)
- Reader study → 审稿要求时再做("带圣旨")
- Venue → 不限页数先写完
- 量化阈值 → ASSUMED Week 5 review

### v1.0 → v1.1 升级触发(2026-04-30)
**用户决策 1**:不和临床扯关系,公开数据 only,demo 程度后再讨论导师
**用户决策 2**:不只 MICCAI,CVPR / AAAI 都开

**v1.1 改动汇总**:
| # | 改动 | 影响章节 |
| --- | --- | --- |
| 1 | 加 §Venue strategy(三档 cascade) | 新章节 |
| 2 | 加 §Framing dual-track | §Novelty 内 |
| 3 | 加 §Reproducibility commitments | 新章节 |
| 4 | 加 §Baseline comparison plan | 新章节 |
| 5 | OUT 加 reader study / 临床 outcome / vendor / TAVI 全部明确排除 | §Scope/OUT |
| 6 | 真实测试数据 4-tier → 2-tier | §IN 内 |
| 7 | 撤回 v0.5 clinical decision support 升级(过度纠正) | §Novelty,§Risk Register |
| 8 | Operational AI 3 项 → 1 项 | §Operational AI |
| 9 | Risk Register 加 CVPR-incremental + 删 reader-related rows | §Risk Register |
| 10 | 时间线加 venue cascade 节点 + 导师讨论 7-8 月 | §时间线 |
| 11 | NICE Success Criteria 撤回 reader study mention | §Success Criteria |
| 12 | Objective 一句话改写,删 MICCAI single-target | §Objective |

---

## Approval

- [x] **CMZ 批准 v1.0:2026-04-29**
- [x] **CMZ 批准 v1.1:2026-04-30**(MICCAI / CVPR / AAAI cascade + clinical entanglement prune)
- [ ] 后续 ASSUMED 解锁触发后(Week 5 阈值 review / Week 5-8 conditioning + lumen loss / Week 9-12 N-sample / Week 17+ PAD-Pro),本 spec 升级到 v1.2 / v1.3

修订机制:**spec 不直接编辑覆盖**,需重大修订时新增 v1.2 / v1.3 版本并记录变更原因。
