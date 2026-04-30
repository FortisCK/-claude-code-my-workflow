# Decision Record — Research direction (V1, 2026-04-29)

**Status:** ACTIVE — landscape底盘
**Companion document:** [`2026-04-29_research-direction-v1.pdf`](2026-04-29_research-direction-v1.pdf)
   *(原 `simple_research.pdf` at the repo root, 17 pages)*
**Successor (focus layer):** [`2026-04-29_research-direction-v2.md`](2026-04-29_research-direction-v2.md)
**Author:** CMZ
**Audience:** CMZ (LTSI Rennes)

---

## V1 是什么 / 不是什么

V1 = **综合调研与决策报告** —— 这是 V2 战略简报的 **底盘 (landscape layer)**,
不是过时版本。V1 和 V2 是 **互补的两层**,不是替代关系:

- **V1**: 详尽的方法学盘点、4-path 横向比较、20-周可执行计划、15-篇按优先级 reading list、所有具体技术锚点(BIPSDA / DiffusionMBIR / DAPS / I3SB / LEAP / CTorch …)
- **V2**: 在 V1 基础上做 focus 层 —— 三个正交 novelty 主张、TAVI 临床闭环、5–7 月时间线、面向投稿的 elevator-pitch 化论证

写 paper 的 related-work / method 节时,V1 的具体细节直接 cite;
做项目级决策时,V2 的 focused argument 先行,V1 的细节 fall back。

---

## V1 提供的、V2 没有的关键资产

1. **§1.1 完整 landscape 矩阵** — 心脏 CT motion correction 2023-2026 全部代表工作 + non-cardiac 但相关 CT 场景
2. **§3.1 Gap 矩阵 (8 类)** — 每一类被多大程度填补 + 还剩什么
3. **§3.2 Path A/B/C/D 横向对比表** — 7 个维度 (数据需求 / A6000 成本 / 推断成本 / 技术风险 / Novelty 防御 / 时间线 / 审稿弹性 / 与 LTSI 资源契合度)
4. **§3.3 Path B 的精确工程量化** — 不是"难",是 1–2 月 + jointly motion field estimation,**博士论文级工程**,作为代表作不推荐
5. **§3.7 20-周可执行计划** — 周 1-2 复现 TT U-Net + clone HM-EDM,周 3-4 训 VAE,周 5-8 LDM v0.1,周 9-12 N-sample UQ,周 13-16 下游任务,周 17-20 baseline 完整跑 + manuscript 一稿
6. **§3.8 七条诚实的不利发现** — 包括 ProDM 已让 "first diffusion for cardiac CT motion" 标题不再成立、HM-EDM 团队同时是 I3SB 作者(说明他们极可能正在做 cardiac 版)、A6000 256³ 3D 直接训不动、ImageCAS 没有 ICA stenosis GT 因此 reader study 必备
7. **§3.6 Reading list (15 篇分三档)** — 本周必读 5 / 本月读 5 / 第三优先 5,直接驱动接下来的 lit-review 流程
8. **具体 UQ 锚点**: BIPSDA = arXiv 2503.03007, 2025 = *Can Diffusion Models Provide Rigorous Uncertainty Quantification for Bayesian Inverse Problems?* (DPS 类方法 posterior variance 在 nonlinear 上偏差可达 50%)

---

## V1 → V2 的故意演化(不是 bug,是用户的 focus 选择)

| 维度 | V1 起点 | V2 选择 |
| --- | --- | --- |
| 下游 killer figure | 冠脉 lumen Dice + stenosis CAD-RADS + reader study | **加上**了 TAVI landmark(和你 TRUST 论文形成故事闭环) |
| 架构野心 | 2D-axial 256² + z-TV 跨切面一致性(保守、A6000 友好) | 3D 256³ latent(更野心) |
| 论文 framing | enabler / benchmark paper(5 个 baseline 统一仿真 + 全部开源) | method paper(三个 novelty 主张主导) |
| Novelty 第 1 条措辞 | "first **conditional latent** diffusion for **gated CCTA** + UQ + downstream" | 偶尔回到不那么精确的 "first diffusion-based cardiac CT motion" — 需警觉,被 ProDM 占了 |

这些迁移会在 master spec(interview 产物)里被逐项 reconciled。

---

## 引用方式

- 写 spec / plan / 论文时,引用 V1 的具体节用 §1.1 / §3.2 / §3.7 / §3.8 等节号
- 引用 V2 用文件路径 `quality_reports/decisions/2026-04-29_research-direction-v2.md`

## 修正机制

V1/V2 都不直接编辑。如果方向变化,新增 V3 record 与之并列。
