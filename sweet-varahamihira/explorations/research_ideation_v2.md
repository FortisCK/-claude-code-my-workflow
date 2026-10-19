# Research Ideation v2: Post-TRUST 博士研究方向规划

> research-ideation 5 步工作流 | 2026-03-17
> 
> **背景**：博士在读 @ Université de Rennes，还有 2-3 年，需要一条连贯研究主线
> **已有数据**：TAVI 主动脉根部 CT (150L + 600U) + AAA-30 腹主动脉瘤 NCCT (30 例)
> **已有技术**：TRUST (半监督 + 不确定性 + Topo-GCN + UG-HCO + 标志点检测)

---

## Step 1: 长期研究目标

> **"在标注稀缺条件下，实现面向主动脉介入手术（TAVI、EVAR 等）的自动化 CT 解剖分析与风险评估——让同一套技术框架能泛化到不同的主动脉病变和介入场景。"**

---

## Step 2: 文献树

### 2A. Novelty Tree

#### 📌 Milestone Task A: 主动脉根部标志点检测 (TAVI)

| 类型 | 工作 | 年份/会议 | 贡献 |
|------|------|----------|------|
| Type 1 | Zheng et al. | 2015 MICCAI | 定义了 3D 体数据 DL 标志点检测范式 |
| Type 2 | Ma et al. | 2023 SPIE | 两阶段 CNN，8 点，MRE 2.23mm，全监督 |
| Type 2 | **TRUST (Ours)** | 2026 | 首个半监督框架，10 点（含冠脉口），统一不确定性 |
| Type 2 | TAVI-PREP (Santaló-Corcoy et al.) | 2023 Diagnostics | MeshDeformNet + 3D ResUNet，22 个测量值，**但冠脉高度相关性仅 r=0.72-0.80**（关键弱点） |
| Type 3 | Payer et al. (SCN) | 2019 MedIA | 空间配置网络捕获标志点间关系 |
| Type 3 | Li et al. | 2020 | 拓扑自适应深度图学习 |

**🔴 空白**：
- ❌ 冠脉高度自动测量的精度仍不足临床标准（TAVI-PREP: r=0.72-0.80）
- ❌ 无 active learning 工作
- ❌ BAV 标志点检测完全空白
- ❌ 无 foundation model 适配
- ❌ 无术后结果预测

#### 📌 Milestone Task B: 腹主动脉瘤 (AAA) 分析

| 类型 | 工作 | 年份/会议 | 贡献 |
|------|------|----------|------|
| Type 1 | Lu et al. (DeepAAA) | 2019 MICCAI | 首个用 DL 做 AAA 检测和径径测量 |
| Type 2 | Roby et al. | 2025 IEEE Access | Mixture-of-Experts U-Net，CTA 分割 Dice 0.96 |
| Type 3 | Chandrashekar et al. | ~2023 | Attention U-Net 用于主动脉分割 |
| Type 4 | 各种 U-Net 变体 | 2020-2025 | 增量改进 |

**🔴 空白**：
- ❌ **非造影 CT (NCCT) 分割比 CTA 差 ~5% DSC**——你的 AAA-30 正好是 NCCT！
- ❌ 血栓/钙化子分割 under-explored
- ❌ 自动 EVAR 支架选型几乎没有 AI 工作
- ❌ 破裂风险预测超越直径的 DL 方法极少
- ❌ 无大型公开 AAA 基准——**你的 AAA-30 数据集有竞争力**
- ❌ 无标志点检测用于 AAA/EVAR 规划

#### 📌 Milestone Task C: 半监督标志点检测 (跨解剖)

| 类型 | 工作 | 年份/会议 | 贡献 |
|------|------|----------|------|
| Type 2 | Honari et al. | 2018 CVPR | 首次 SSL + 人脸标志点 |
| Type 2 | Ren et al. (G2LCPS) | 2024 Neural Networks | CPS 用于端到端标志点预测 |
| Type 3 | Chen et al. | 2021 Neurocomputing | Shape-regulated self-training（PCA 形状先验） |
| Type 3 | Kim et al. (HybridMatch) | 2023 | 混合 heatmap+坐标表示 |
| Type 3 | **Di Via et al.** | **2025 WACV** | **⭐ Diffusion model 预训练用于 few-shot 标志点检测 (2D X-ray)** |

**🔴 关键发现**：
- ❌❌ **所有工作都是 2D。3D 半监督标志点检测完全空白（只有 TRUST）**
- ❌ SAM 适配标志点检测（而非分割）是开放问题
- ⭐ Di Via et al. (WACV 2025) 的 diffusion 预训练方向可以迁移到 3D

#### 📌 Milestone Task D: 新兴 3D 医学影像工具

| 类型 | 工作 | 年份/会议 | 贡献 |
|------|------|----------|------|
| Type 2 | Swin-UMamba (Liu et al.) | 2024 IEEE TMI | Mamba 架构用于医学分割，线性复杂度 |
| Type 2 | BiomedCLIP / LLaVA-Med | 2023-2024 | 视觉-语言模型做医学影像理解 |
| Type 2 | STU-Net / TotalSegmentator | 2023-2024 | 跨解剖预训练分割模型 |

**🔴 空白**：
- ❌ Mamba 未用于标志点检测
- ❌ VLM 无法做精确测量
- ❌ 跨解剖预训练未在精细主动脉子结构上评估

---

### 2B. Challenge-Insight Tree

```
Challenge A: 标注稀缺
├── ✅ SSL (Mean Teacher, CPS, TRUST)
├── ✅ Shape-regulated self-training (Chen 2021)
├── ⭐ Diffusion 预训练 for few-shot (Di Via 2025) → 仅限 2D
├── ❌ 缺失: Active learning for 3D landmarks
├── ❌ 缺失: Foundation model → landmark head
└── ❌ 缺失: 跨解剖结构的预训练（主动脉根→腹主动脉迁移）

Challenge B: 钙化/伪影/低对比度
├── ✅ UG-HCO 频域自适应扩散 (TRUST)
├── ✅ Attention 机制
├── ❌ 缺失: NCCT vs CTA 的对比度差距（AAA-30 正是 NCCT）
└── ❌ 缺失: 钙化区域的显式建模

Challenge C: 解剖变异
├── ✅ 固定图拓扑 GCN (TRUST)
├── ❌ 缺失: BAV vs TAV 可变拓扑
├── ❌ 缺失: 正常 vs 瘤变主动脉的形态适配
└── ❌ 缺失: 统一框架处理不同主动脉病变

Challenge D: 临床测量的不确定性传播
├── ✅ Per-landmark uncertainty (TRUST)
├── ❌ 缺失: uncertainty → 冠脉高度置信区间
├── ❌ 缺失: 校准不确定性用于临床决策
└── ❌ 缺失: 不确定性作为临床可信度指标

Challenge E: 跨域泛化
├── ⭐ TotalSegmentator 跨解剖预训练
├── ❌ 缺失: NCCT ↔ CTA 域适配
├── ❌ 缺失: 主动脉根 ↔ 腹主动脉迁移
└── ❌ 缺失: 多中心泛化
```

---

## Step 3: 问题选择 (Well-Established Solution Check)

从文献树空白中提取 **6 个候选问题**：

### ✅ 候选 1: Diffusion 预训练用于 3D 少样本标志点检测
- **Check**: Level 3 — Di Via et al. (WACV 2025) 在 2D X-ray 上做了 diffusion 预训练→few-shot 标志点，但**无人在 3D 上做过**
- **判断**: **✅ 强力推荐**（跨维度迁移，有明确的 2D 先例）
- **数据**: 现有 TAVI 数据即可，无需额外采集
- **与 TRUST 的关系**: 可作为 TRUST 的 backbone 升级，用 diffusion 预训练替代 ImageNet 预训练

### ✅ 候选 2: 跨主动脉解剖的迁移学习 (主动脉根 ↔ 腹主动脉)
- **Check**: Level 4 — 无人做过主动脉不同段之间的迁移学习
- **判断**: **✅ 强力推荐**（你有两份数据！这是你的独特优势）
- **数据**: TAVI CT + AAA-30 NCCT，完美匹配
- **故事**: "同一条主动脉的不同段、不同病变、不同成像条件——同一个框架能否适配？"

### ✅ 候选 3: NCCT 腹主动脉瘤分析（填补 AAA-30 的空白）
- **Check**: Level 3 — CTA 分割已接近成熟(Dice 0.96)，但 NCCT 差 5%，且无公开 NCCT 基准
- **判断**: **✅ 推荐**（你的 AAA-30 是 NCCT，直接对标这个 gap）
- **数据**: AAA-30 直接可用
- **额外价值**: 发布 AAA-30 作为 benchmark = 社区贡献

### ⚠️ 候选 4: Active Learning + SSL for landmarks
- **Check**: Level 4 — 无人做过
- **判断**: **⚠️ 推荐但优先级降低**——技术新意有限（AL 本身不新），除非能找到标志点检测特有的 AL 策略
- **数据**: 现有即可

### ⚠️ 候选 5: 不确定性传播→临床决策支持
- **Check**: Level 4
- **判断**: **⚠️ 有价值但需要临床验证数据**——你目前没有术后结果数据
- **适合**: 作为某篇论文的附加贡献，而非独立论文

### ⚠️ 候选 6: BAV 自适应检测
- **Check**: Level 4
- **判断**: **⚠️ 好方向但需要 BAV 数据**——你目前没有

---

## Step 4: 解决方案设计

基于 Step 3 的排序，为 Top 3 候选设计具体方案：

---

### 🥇 方案 1: DiffLand3D — Diffusion 预训练用于 3D 少样本标志点检测

**问题**: 3D 医学标志点检测依赖大量标注。能否用 diffusion model 的去噪预训练学习 3D 解剖结构表征，使下游标志点检测在少量标注下就能达到好效果？

**跨域迁移源**: Di Via et al. (WACV 2025) — 在 2D X-ray 上用 diffusion 预训练特征做 few-shot 标志点检测。

**方案设计**:
1. **预训练阶段**: 在 600 例无标注 TAVI CT 上训练一个 3D Denoising Diffusion Model（学习主动脉根部的分布）
2. **特征提取**: 用训练好的 diffusion U-Net 的中间层特征作为 landmark detector 的 backbone
3. **微调阶段**: 用 10/30/50/150 例标注数据微调标志点检测 head
4. **对比**: vs ImageNet pretrain → finetune, vs TRUST (SSL), vs from scratch
5. **额外实验**: diffusion 预训练 + TRUST SSL 组合 → 是否进一步提升？

**创新点**:
- 首个 3D 医学标志点检测的 diffusion 预训练
- Diffusion 特征天然包含多尺度解剖结构信息
- 可以和 TRUST 的 SSL 互补（预训练解决 representation 问题，SSL 解决 label 问题）

**实验草案**:
```
Few-shot learning curve:
  x 轴: 标注数量 [10, 30, 50, 100, 150]
  y 轴: MRE (mm)
  曲线: (1) Scratch  (2) ImageNet pretrain  (3) Diffusion pretrain  
        (4) TRUST SSL  (5) Diffusion pretrain + TRUST SSL
预期: (5) > (4) > (3) > (2) > (1)
额外: per-landmark 分析，看 diffusion 预训练对 P7/P8/P9 等难点的提升
```

**发表目标**: MICCAI 2027 或 MedIA

---

### 🥈 方案 2: AortaTransfer — 跨主动脉解剖的标注高效迁移框架

**问题**: 主动脉不同段（胸主动脉根 vs 腹主动脉）有共享的管状解剖特征，但也有各自特异的病变（钙化瓣膜 vs 动脉瘤）。能否利用一个解剖区域的标注数据帮助另一个区域的分析？

**这是你的独特优势**: 你同时有 TAVI 数据（150L+600U, 含标志点标注）和 AAA-30（30 例，含分割标注）——全世界可能只有你的组同时有这两份数据。

**方案设计**:
1. **共享 encoder 预训练**: 在两个数据集上联合自监督预训练（MAE 或 contrastive），学习"通用主动脉特征"
2. **任务特异 head**: 主动脉根 → 标志点检测 head; 腹主动脉 → 分割 head
3. **跨域增益实验**: 
   - 只用 AAA-30 训练 vs 加入 TAVI 特征预训练后训练（AAA 分割提升多少？）
   - 只用少量 TAVI 标注 vs 加入 AAA 预训练后训练（标志点检测在 few-shot 下提升多少？）
4. **NCCT 域适配**: AAA-30 是 NCCT，TAVI 是造影 CT → 训练过程天然包含域适配

**创新点**:
- 首个跨主动脉区域的迁移学习框架
- 利用"同一条血管不同段"的解剖连续性
- NCCT ↔ CTA 的隐式域适配
- 同时产出两个任务的结果（标志点 + 分割）

**实验草案**:
```
实验 A (AAA 分割): 
  Baseline: U-Net from scratch on AAA-30 (30 例)
  Ours: 共享 encoder 预训练 (TAVI+AAA) → finetune AAA-30
  指标: Dice, HD95

实验 B (TAVI 标志点 few-shot):
  Baseline: TRUST with 30 labels
  Ours: 共享 encoder 预训练 (TAVI+AAA) → TRUST with 30 labels
  指标: MRE, SDR

实验 C (消融):
  预训练只用 TAVI vs 只用 AAA vs 两者联合
```

**发表目标**: MedIA 或 TMI（因为跨两个临床应用）

---

### 🥉 方案 3: AAA-NCCT-Bench — 非造影 CT 腹主动脉瘤分析基准与方法

**问题**: AAA 的 NCCT 分割比 CTA 差 ~5% DSC，且无公开 NCCT 基准。你的 AAA-30 可以填补这个空白。

**方案设计**:
1. **发布 AAA-30 作为 benchmark**: 标注质量描述、train/val/test 划分、评估协议
2. **Baseline 方法**: U-Net, Attention U-Net, nnU-Net, Swin-UMamba 等在 AAA-30 上的 benchmark 结果
3. **半监督方法**: 将 TRUST 的 SSL 思路迁移到分割（Mean Teacher + CPS for segmentation on NCCT）
4. **不确定性量化**: 哪些区域/患者分割最不确定？与钙化/血栓的关系？
5. **临床价值**: 从分割结果自动提取 AAA 径径、形态参数

**创新点**:
- 首个 NCCT AAA 公开基准
- SSL 用于 NCCT 条件下的小样本 AAA 分割
- TRUST 的不确定性机制迁移到分割任务

**发表目标**: MICCAI (数据集 + 方法) 或 IEEE JBHI

---

## Step 5: 研究主线总结

### 推荐的 PhD 路线图 (2026-2029)

```
Year 1 (2026): TRUST 论文发表
              ↓
Year 1-2:     方案 3 (AAA-NCCT-Bench) ← 快速出成果，发布数据集建立影响力
              ↓
Year 2:       方案 1 (DiffLand3D) ← 方法论创新，diffusion 预训练
              ↓
Year 2-3:     方案 2 (AortaTransfer) ← 整合两个数据集，跨域迁移
              ↓
Year 3:       博士论文整合
```

### 统一叙事线

> **"Label-Efficient Aortic Anatomy Understanding: From Semi-Supervised Learning to Cross-Anatomy Transfer"**
>
> - Paper 1 (TRUST): 半监督 + 不确定性 + 拓扑推理 → 主动脉根部标志点
> - Paper 2 (AAA-NCCT-Bench): 将 SSL 和不确定性迁移到 AAA 分割 + 公开数据集
> - Paper 3 (DiffLand3D): Diffusion 预训练 → 更强的少样本标志点检测
> - Paper 4 (AortaTransfer): 跨主动脉区域的统一框架 → PhD 总结篇
>
> 贯穿主题：**在标注稀缺的现实条件下，让 AI 理解主动脉的全段解剖**。

### 每篇论文与已有资源的关系

| 论文 | 使用数据 | 复用 TRUST 技术 | 额外需求 |
|------|---------|----------------|---------|
| AAA-NCCT-Bench | AAA-30 | SSL + 不确定性 | 补充 baseline 实现 |
| DiffLand3D | TAVI 150L+600U | Topo-GCN, 评估框架 | 实现 3D diffusion 预训练 |
| AortaTransfer | TAVI + AAA-30 | 全部 | 实现跨域预训练 |

---

## 附：需要后续补充的信息

1. **AAA-30 的标注内容**：是 lumen/wall/thrombus 子分割？还是只有外轮廓？标注格式？
2. **是否有可能扩大 AAA 数据量**：雷恩大学医院是否有更多 AAA CT 可以获取？
3. **导师/师兄对 AAA 方向的态度**：是否支持你同时做 TAVI + AAA？
4. **GPU 资源**：3D diffusion 预训练对 GPU 要求较高（建议 A100 40GB+）
5. **Di Via et al. (WACV 2025) 原文**：建议精读，了解 2D diffusion 预训练的具体实现细节

> 下一步：选定一个方向后，可以进入 **idea-tournament**（生成具体方法候选排名）或 **paper-planning**（规划实验和论文结构）。
