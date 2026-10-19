# Research Ideation: Post-TRUST 研究方向探索

> 按照 research-ideation 技能的 5 步工作流生成。
> 日期：2026-03-17

---

## Step 1: 长期研究目标

**Ultimate Form**: 实现结构性心脏病介入手术的全自动术前规划与风险评估系统——在标注稀缺条件下，从 CT 影像中可靠提取所有临床决策所需的解剖信息，并给出带置信度的风险预判。

**如果完全实现**：
- 医生拿到 CT → 系统自动输出瓣膜尺寸推荐、冠脉阻塞风险、传导阻滞风险、入路评估
- 每个输出附带置信度，低置信度自动提示人工复核
- 不需要大量标注即可适配新中心/新设备/新瓣膜类型

**你的技术栈**：半监督学习 + 不确定性估计 + 拓扑/图推理 + 频域特征传播 + 3D 标志点检测

---

## Step 2: 文献树

### 2A. Novelty Tree（贡献类型分类）

#### Milestone Task 1: 主动脉根部分割
| 类型 | 工作 | 贡献 |
|------|------|------|
| Type 2 | Lalys et al., 2019, MITAT | 首个自动主动脉根分割 + 标志点 pipeline for TAVI |
| Type 2 | Wang et al., 2023, eBioMedicine | 全自动 DL 算法做 pre-TAVI CT 评估（多中心验证） |
| Type 4 | 各种 U-Net 变体 | 在标准分割框架上的增量改进 |

**空白**: ❌ 无半监督分割工作专门针对主动脉根部。❌ 无联合分割+标志点 end-to-end 学习。

#### Milestone Task 2: 主动脉瓣标志点检测
| 类型 | 工作 | 贡献 |
|------|------|------|
| Type 1 | Zheng et al., 2015, MICCAI | 定义了 3D 体数据标志点检测任务的 DL 范式 |
| Type 2 | Ma et al., 2023, SPIE | 两阶段 CNN pipeline 直接检测 8 个主动脉瓣标志点 (MRE 2.23mm) |
| Type 2 | **TRUST (Ours)** | 首个半监督 + 不确定性 + 拓扑推理框架，扩展到 10 标志点 |
| Type 3 | Payer et al., 2019, MedIA | 空间配置网络（SCN）捕获标志点间关系 |
| Type 3 | Li et al., 2020 | 拓扑自适应深度图学习做结构化标志点检测 |
| Type 3 | Lang et al., 2020, MICCAI | 局部注意力 GCN 做颅面标志点 |

**空白**: ❌ 无 active learning 工作。❌ 无 foundation model 适配。❌ 无 diffusion-based 检测。❌ BAV 的标志点定义和检测未被探索。

#### Milestone Task 3: 冠脉安全性评估
| 类型 | 工作 | 贡献 |
|------|------|------|
| Type 1 | Ribeiro et al., 2013, JACC-CI | 系统综述定义冠脉阻塞风险因素 |
| Type 1 | Blanke et al., 2019, JACC-Imaging | SCCT 专家共识：CT 在 TAVI 中的角色（含冠脉高度测量标准） |
| Type 2 | **TRUST (Ours)** | 首个通过自动标志点检测实现冠脉高度测量的 AI 系统 |

**空白**: ❌ 无 AI 驱动的冠脉阻塞风险预测模型。❌ 无从标志点→临床决策的端到端系统。❌ Valve-in-Valve 的冠脉风险评估 AI 完全空白。

#### Milestone Task 4: TAVI 术后结果预测
| 类型 | 工作 | 贡献 |
|------|------|------|
| Type 1 | 临床统计学研究（logistics regression 为主） | 基于手工测量的风险因子分析 |

**空白**: ❌❌ 几乎无 DL 端到端从 CT → 术后结果预测。这是一个**巨大的空白**。

#### Milestone Task 5: 术中引导
| 类型 | 工作 | 贡献 |
|------|------|------|
| Type 2 | 传统 3D-2D registration 方法 | 基于优化的 CT-fluoroscopy 配准 |

**空白**: ❌ 无 DL 驱动的从术前标志点到术中引导的迁移。

#### Milestone Task 6: 半监督标志点检测（跨领域）
| 类型 | 工作 | 贡献 |
|------|------|------|
| Type 1 | Honari et al., 2018, CVPR | 首次将 SSL 引入标志点检测（人脸） |
| Type 2 | Dong & Yang, 2019, ICCV | Teacher-student 框架处理部分标注人脸标志点 |
| Type 2 | Ren et al., 2024, Neural Networks | G2LCPS: 端到端 CPS 做标志点预测 |
| Type 3 | Kim et al., 2023 | HybridMatch: 混合 heatmap+坐标表示做半监督人脸标志点 |
| Type 2 | **TRUST (Ours)** | 首个将 SSL 标志点检测应用于 3D 医学影像 + 统一不确定性机制 |

**空白**: ❌ SSL 标志点检测几乎全在 2D 人脸上，**3D 医学影像上只有 TRUST**。❌ 无 active learning + SSL 组合。

### 2B. Challenge-Insight Tree（技术挑战→已知解法）

```
Challenge A: 伪标签噪声在标志点检测中的传播
├── Insight A1: 不确定性加权 CPS loss (TRUST)
├── Insight A2: 置信度阈值筛选 (FixMatch, Sohn 2020)
├── Insight A3: 不可靠伪标签做负学习 (U2PL, Wang 2022)
└── ❌ 缺失: 基于 diffusion 的伪标签去噪

Challenge B: 标志点间的结构/拓扑约束
├── Insight B1: GCN 图推理 (Li 2020, Lang 2020, TRUST)
├── Insight B2: 空间配置网络 SCN (Payer 2019)
├── Insight B3: 解剖先验损失函数
└── ❌ 缺失: 可变拓扑（BAV vs TAV 的图结构自适应）

Challenge C: 钙化/伪影导致的特征退化
├── Insight C1: UG-HCO 频域自适应扩散 (TRUST)
├── Insight C2: Transformer 全局注意力
├── Insight C3: 多尺度特征融合 (HRNet, FPN)
└── ❌ 缺失: 钙化区域的显式建模（钙化分割→条件特征增强）

Challenge D: 标注效率
├── Insight D1: 半监督学习 (TRUST, Mean Teacher, CPS)
├── Insight D2: 自监督预训练 (MAE, contrastive learning)
├── ❌ 缺失: Active learning for landmarks
├── ❌ 缺失: Foundation model 适配 (SAM-Med3D → landmark head)
└── ❌ 缺失: 合成数据生成

Challenge E: 不确定性作为临床输出
├── Insight E1: MC Dropout 不确定性 (Gal 2016)
├── Insight E2: TTA 方差 (TRUST)
├── Insight E3: Deep Ensembles (Lakshminarayanan 2017)
├── ❌ 缺失: 校准不确定性→临床决策支持（风险分层）
└── ❌ 缺失: 不确定性传播到下游测量值（冠脉高度的置信区间）

Challenge F: 跨域泛化
├── ❌ 缺失: 多中心/多设备适配
├── ❌ 缺失: CT → 术中透视的模态迁移
└── ❌ 缺失: BAV vs TAV 的解剖变异适配
```

---

## Step 3: 问题选择（Well-Established Solution Check）

基于文献树中的空白，筛选出 **6 个候选问题**，用 4 级检查评估：

### 候选问题 A: Active Learning + SSL for 3D 医学标志点检测
- **Check Level**: **Level 4**（无领域有专门解法）
  - Active learning 在医学图像**分割**上有工作，但在**标志点检测**上几乎没有
  - SSL + Active Learning 的组合在标志点上完全空白
- **评估**: ✅ **强力推荐**。技术核心未被探索，且直接复用 TRUST 的不确定性估计
- **风险**: 低。不需要新数据，不需要新架构

### 候选问题 B: 从标志点到临床决策（冠脉阻塞风险预测）
- **Check Level**: **Level 4**（DL 端到端从 CT→风险预测几乎没有）
  - 临床上用手工测量+阈值规则(冠脉高度<10mm)
  - 无人将自动检测的标志点+不确定性→概率化风险预测
- **评估**: ✅ **强力推荐**。临床影响力最大，且有独特角度（不确定性传播）
- **风险**: 中。需要术后结果数据（需临床合作）

### 候选问题 C: 解剖自适应标志点检测（BAV + TAV 统一框架）
- **Check Level**: **Level 4**（BAV 的 AI 标志点检测完全空白）
  - BAV 拓扑与 TAV 完全不同（2 hinge vs 3 hinge）
  - 无人做过可变拓扑的 GCN 来同时处理 BAV 和 TAV
- **评估**: ✅ **推荐**。临床需求迫切（BAV 是 TAVI 增长最快的群体）
- **风险**: 中-高。需要 BAV CT 数据

### 候选问题 D: 联合分割 + 标志点检测
- **Check Level**: **Level 2-3**
  - 分割和标志点的 multi-task 学习在其他领域有不少工作
  - 但在主动脉根部上无人做过 end-to-end joint learning
- **评估**: ⚠️ **可做但不是最优先**。技术新意有限（multi-task 本身不新），但临床故事好
- **风险**: 低。但贡献可能被认为是 incremental

### 候选问题 E: UG-HCO 推广为通用模块
- **Check Level**: **Level 3**
  - HCO (Wang 2025) 是新工具，uncertainty-conditioned 版本只有 TRUST 做过
  - 但"方法推广到多个数据集"类论文贡献感偏弱
- **评估**: ⚠️ **可做但需要亮点**。需要在推广过程中发现新 insight，不能只是"在更多数据集上验证"
- **风险**: 低。公开数据集即可

### 候选问题 F: Foundation Model + 标志点检测
- **Check Level**: **Level 4**
  - SAM-Med3D 等 3D foundation model 用于标志点检测完全未探索
  - 现有 FM 主要做分割，landmark head 的适配是开放问题
- **评估**: ✅ **高风险高回报**。如果 few-shot 效果好，影响力巨大
- **风险**: 高。FM 对精细定位任务的表现未知

---

## Step 4: 解决方案设计

### 方案 A: Uncertainty-Guided Active Learning for 3D Landmark Detection（⭐ 最推荐）

**问题分解**：
1. **子问题 1**: 如何用不确定性选择最有价值的未标注样本？
   - 跨域迁移源：Active learning for medical segmentation (Nath et al., 2020; Gaillochet et al., 2023)
   - 适配：把 voxel-level uncertainty 换成 per-landmark uncertainty（TRUST 已有）
   - 设计：sample-level 不确定性 = 该样本上所有 10 个标志点不确定性的某种聚合（max? mean? 加权？）

2. **子问题 2**: 如何在 SSL 框架内嵌入 active learning loop？
   - 跨域迁移源：CEAL (Cost-Effective Active Learning, Wang et al., 2017)，将 SSL 的高置信伪标签和 AL 的低置信选择统一
   - 设计：每 N epoch 暂停 → 用 teacher 的 TTA 不确定性排序所有未标注样本 → 选 top-K 请求标注 → 移入标注池 → 继续训练

3. **子问题 3**: 如何评估 AL 的效果？
   - 标准：annotation efficiency curve（横轴=标注量，纵轴=MRE）
   - 对比：random / uncertainty / diversity / hybrid sampling

**创新性验证**：
- Active Learning + SSL + 3D 医学标志点检测 = 此组合在文献中未出现
- 不确定性同时驱动 SSL（TRUST 已验证）和 AL（新贡献）= "estimate once, consume four times"
- 不是简单拼接：AL 选择影响 SSL 的数据分布，SSL 的不确定性估计影响 AL 选择 → 存在耦合交互

**实验设计草案**：
```
数据：150L + 600U（现有）
模拟 AL 场景：初始只用 30L → 每轮从 600U 中选 20 例标注 → 观察 6 轮后 = 30+120 = 150L
对比：
  - Random selection + 全监督
  - Random selection + SSL (TRUST)
  - Uncertainty selection + 全监督  
  - Uncertainty selection + SSL (TRUST) ← 我们的
  - Diversity selection + SSL
评估：MRE@各标注量、SDR@各标注量、per-landmark 分析（P7 和冠脉口是否优先被选中？）
```

---

### 方案 B: Uncertainty-Propagated Clinical Decision Support（⭐ 临床影响力最大）

**核心思路**：TRUST 输出标志点坐标 + 每个标志点的不确定性 → 自动计算临床参数（冠脉高度等）→ 将标志点不确定性传播到临床参数的置信区间 → 基于置信区间做风险分层

**问题分解**：
1. **子问题 1**: 从标志点到临床参数的确定性计算
   - 冠脉高度 = dist(P8/P9, plane(P0, P1, P2))
   - 瓣环直径 = 基于 P0-P2 的几何计算
   - 已有现成几何公式

2. **子问题 2**: 不确定性传播（关键创新）
   - TRUST 给出每个 $p_k$ 的协方差矩阵（来自 TTA 的多次预测）
   - 通过 error propagation（一阶 Taylor 展开或 Monte Carlo propagation）→ 冠脉高度的置信区间
   - 跨域迁移源：测量不确定性传播（GUM, Guide to the Expression of Uncertainty in Measurement）——物理/工程中的标准方法，从未被用于 AI 驱动的临床测量

3. **子问题 3**: 风险分层与临床验证
   - 传统规则：冠脉高度 < 10mm → 高风险
   - 我们的改进：冠脉高度 = 12mm ± 3mm（95% CI 含 <10mm）→ 仍标记为"需关注"
   - 验证需要术后结果数据

**创新性验证**：
- "不确定性从像素级到标志点级到临床决策级"的完整传播链 = 文献中未出现
- 将工程测量学的不确定性传播引入 AI 辅助医学测量 = 跨领域迁移（Level 3）

---

### 方案 C: Anatomy-Adaptive Landmark Detection（BAV + TAV）

**核心思路**：设计一个统一框架，自动检测瓣膜类型（BAV vs TAV）并切换标志点定义和图拓扑

**问题分解**：
1. **子问题 1**: BAV vs TAV 分类
   - 相对简单的二分类任务，可以用 backbone 的 global feature
2. **子问题 2**: 可变标志点数量的检测
   - TAV: 10 landmarks (3 hinge + 3 comm + center + MS + 2 ostia)
   - BAV: 8 landmarks (2 hinge + 2 comm + center + MS + 2 ostia)？（需和医生确认）
   - 设计：最大集合标志点 + 存在性预测头（类似 DETR 的 "no object" class）
3. **子问题 3**: 可变拓扑的 GCN
   - 跨域迁移源：Dynamic Graph Networks (DGCNN, Wang et al., 2019) — 根据输入动态构建图
   - 适配：根据 BAV/TAV 分类结果切换邻接矩阵模板

---

## Step 5: 研究方向总结

### 最终推荐排序

| 优先级 | 方向 | 核心问题 | 新颖性等级 | 可行性 | 影响力 |
|--------|------|---------|-----------|--------|-------|
| 🥇 | **Active Learning + SSL** | 不确定性驱动的高效标注 | Level 4 (无人做过) | ⭐⭐⭐⭐⭐ | ⭐⭐⭐⭐ |
| 🥈 | **不确定性传播→临床决策** | 从标志点不确定性到临床风险置信区间 | Level 3-4 | ⭐⭐⭐ | ⭐⭐⭐⭐⭐ |
| 🥉 | **BAV 自适应标志点检测** | 可变拓扑的解剖自适应框架 | Level 4 | ⭐⭐⭐ | ⭐⭐⭐⭐ |

### 推荐研究叙事线

> **"Uncertainty as a First-Class Clinical Signal"**
> 
> TRUST 证明了不确定性可以驱动半监督学习的三个模块。后续工作将这个思路推向两个新维度：
> 1. **标注维度**（Active Learning）：不确定性不仅指导学习，还指导*数据采集*
> 2. **临床维度**（Decision Support）：不确定性不仅改善模型，还直接成为*临床输出*
> 
> 这形成一条清晰的研究主线：**不确定性作为贯穿数据采集→模型训练→临床决策的统一信号**。

---

## 附：后续步骤

1. **如果选定方向 A（Active Learning）**：→ 进入 idea-tournament，生成具体方法候选并排名
2. **如果选定方向 B（临床决策）**：→ 需要先确认是否有术后结果数据，再进入 paper-planning
3. **如果选定方向 C（BAV）**：→ 需要先确认 BAV 数据来源，再进入 experiment-pipeline

> ⚠️ 注意：由于 API 过载，本文档的文献调研未能通过 web search 补充 2024-2026 最新论文。建议在 API 恢复后用 research-agent 做一轮补充搜索，重点检查：
> - Active learning + landmark detection 是否有 2024-2026 新工作
> - Diffusion model 用于标志点检测是否有新突破
> - Foundation model 适配标志点检测是否有新进展
