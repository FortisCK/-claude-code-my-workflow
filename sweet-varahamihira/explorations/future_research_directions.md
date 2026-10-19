# TRUST 之后的研究方向规划

> 基于 TRUST 论文（半监督主动脉根部标志点检测 + 冠脉安全性评估）的技术积累和数据资源，以下是后续可延伸的研究方向，按**可行性**和**发表潜力**排序。

---

## 🔥 Tier 1：直接延伸，最快出成果（6–12 个月）

### 方向 1：不确定性引导的主动学习（Active Learning）
**一句话**：TRUST 已经有了不确定性估计 → 用它来自动挑选「最值得标注的样本」，让医生标更少的数据达到更好的效果。

- **核心思路**：600 例无标签数据中，TRUST 的不确定性最高的样本 = 模型最"不确定"的 = 标注收益最大的。设计一个 active learning loop：SSL 训练 → 用不确定性排序无标签数据 → 选 top-K 让医生标 → 重新训练。
- **卖点**：
  - 目前 SSL + Active Learning 在 **landmark detection** 上几乎没有工作，你们是第一个
  - 直接复用 TRUST 的 TTA 不确定性机制，不需要重新设计模型
  - 临床故事很好讲：「用最少的医生时间达到最优精度」
- **实验设计**：
  - 对比策略：random sampling / uncertainty sampling / diversity sampling / hybrid
  - 曲线：每轮标注 10/20/30 例后的 MRE 变化
  - 展示：标注 50 例 + Active Learning ≈ 标注 150 例全量
- **投稿目标**：MICCAI / MedIA / TMI
- **所需资源**：现有 150L + 600U 数据即可，不需要额外采集

---

### 方向 2：联合分割 + 标志点检测（Multi-task）
**一句话**：同时输出主动脉根部分割 mask 和 10 个标志点坐标，分割为标志点提供几何约束，标志点为分割提供解剖锚点。

- **核心思路**：
  - 在 TRUST 的 backbone 上加一个分割 decoder head
  - 分割结果（主动脉根轮廓、窦部、瓣环平面）为标志点提供隐式约束：标志点必须在分割出的结构上/附近
  - 反过来，标志点坐标可以引导分割关注关键区域
- **卖点**：
  - 临床上医生需要的本来就不只是"点"，还需要主动脉根部的形态测量（周长、面积、体积）
  - Multi-task 可以进一步利用无标签数据（分割和检测互相提供伪监督）
  - Lalys et al. 和 Wang et al. 做过分割→标志点的 pipeline，但**没人做过 end-to-end joint learning**
- **数据需求**：需要对部分数据补标分割 mask（或用 SAM-Med3D 等工具半自动生成）
- **投稿目标**：IEEE JBHI / MedIA

---

### 方向 3：UG-HCO 作为通用模块推广
**一句话**：把 UG-HCO 从「TAVI 专用」提炼成一个通用的 frequency-domain uncertainty-aware feature propagation 模块，证明它在多个任务上有效。

- **核心思路**：
  - UG-HCO 的设计理念（不确定性 → 自适应扩散系数 → 频域特征传播）是任务无关的
  - 在其他任务上验证：脊柱标志点检测（公开数据集）、头影测量标志点（公开数据集）、关节标志点
  - 写成一个 "plug-and-play module" 的故事
- **卖点**：
  - 原始 HCO (Wang et al., 2025) 是在分割/分类上做的，**你们把它扩展到了 landmark detection + uncertainty conditioning**
  - 如果在 2-3 个公开数据集上都有提升，就是一个很强的 contribution
- **公开数据集**：ISBI 2015 Cephalometric / AASCE Spine / WFLW Face
- **投稿目标**：ECCV / CVPR / MICCAI（偏方法论的会议）

---

## 🌟 Tier 2：需要一定拓展，但影响力更大（12–18 个月）

### 方向 4：TAVI 术后结果预测
**一句话**：用术前 CT 检测到的标志点坐标 + 衍生测量值（冠脉高度、瓣环尺寸等），预测术后并发症风险。

- **核心思路**：
  - TRUST 输出 10 个标志点 → 自动计算临床参数（瓣环直径、冠脉高度、窦部高度、MSL 等）
  - 用这些参数 + 患者基线数据 → 训练预测模型 → 预测：
    - 冠脉阻塞风险（二分类）
    - 瓣周漏（paravalvular leak）严重程度
    - 传导阻滞（need for pacemaker）
    - 瓣膜型号推荐
  - 从"检测工具"升级为"决策支持系统"
- **卖点**：
  - 极强的临床转化价值，医生很感兴趣
  - 不确定性估计可以直接用：「当模型对某个标志点不确定时，下游预测也应该反映这个不确定性」→ 不确定性传播（uncertainty propagation）
  - 这条路线可以发一篇方法论（MICCAI/TMI）+ 一篇临床验证（EHJ/JACC 的影像子刊）
- **数据需求**：需要临床结果数据（术后随访），需要和临床团队合作
- **投稿目标**：TMI / MedIA / European Heart Journal - Imaging

---

### 方向 5：跨模态迁移——CT → 术中透视（Fluoroscopy）
**一句话**：把术前 CT 上学到的标志点知识迁移到术中 2D 透视图像上，实现实时手术导航。

- **核心思路**：
  - 术前有 3D CT 的精确标志点（TRUST 检测）
  - 术中只有 2D X-ray / fluoroscopy，且没有标注
  - 做 CT → Fluoroscopy 的 domain adaptation：
    - 方案 A：用 DRR（Digitally Reconstructed Radiograph）从 CT 生成模拟透视图，带自动标注 → 训练 2D 检测器
    - 方案 B：用 CT 标志点做 3D-2D registration，训练自监督的跨模态网络
- **卖点**：
  - 术中实时指导是 TAVI 领域的圣杯问题之一
  - 目前没有从「landmark detection on CT → intra-operative guidance」的端到端工作
  - 不确定性在术中更关键：不确定就提醒医生注意
- **风险**：需要术中透视数据，技术难度较高
- **投稿目标**：MICCAI / IEEE TMI / Medical Physics

---

### 方向 6：二叶式主动脉瓣（BAV）的专项检测
**一句话**：BAV 患者的解剖结构与三叶瓣完全不同（2 个 hinge 而非 3 个），需要自适应的标志点模型。

- **核心思路**：
  - BAV 约占人口 1-2%，是 TAVI 增长最快的适应症群体
  - BAV 只有 2 个 hinge points + 2 个 commissures，标志点数量和拓扑关系都不同
  - 设计一个 **anatomy-adaptive** 框架：
    - 先分类 BAV vs TAV（三叶瓣）
    - 根据分类结果切换标志点定义和图拓扑
    - 或者：设计一个统一的 GCN，通过 learned adjacency 自适应拓扑
- **卖点**：
  - BAV + TAVI + AI 几乎没有工作，临床需求迫切
  - Topo-GCN 的 graph structure 天然支持拓扑变化
  - 可以展示「同一框架处理不同解剖变异」的 generalization 能力
- **数据需求**：需要 BAV CT 数据（可能需要单独采集或合作获取）
- **投稿目标**：EHJ - Imaging / JACC Imaging / MedIA

---

## 💎 Tier 3：高风险高回报，可以是博士论文级别的工作

### 方向 7：基于 Foundation Model 的通用心脏标志点检测
**一句话**：用 3D 医学影像 Foundation Model（如 SAM-Med3D / SuPreM / Universal Model）的 encoder 替代你的 backbone，做 few-shot / zero-shot 标志点检测。

- **核心思路**：
  - 用大规模预训练的 3D 医学影像 encoder 提取特征
  - 在上面接 TRUST 的 UG-HCO + Topo-GCN head
  - 验证：只用 10/20/50 例标注能否达到 TRUST 用 150 例标注的效果
  - 进一步：用 prompt（如粗略的标志点位置作为 point prompt）做 interactive detection
- **卖点**：
  - Foundation Model + 标志点检测的结合目前是空白
  - 如果 few-shot 效果好，对临床推广意义巨大
  - 可以同时在多个解剖结构上验证（主动脉根 + 脊柱 + 颅面）
- **风险**：现有 3D foundation model 对精细定位任务的表现未知
- **投稿目标**：Nature Machine Intelligence / MICCAI / NeurIPS

---

### 方向 8：Diffusion Model 用于标志点检测
**一句话**：把标志点检测建模为条件去噪过程——从随机噪声坐标逐步 denoise 到精确的标志点位置。

- **核心思路**：
  - 近年来 DiffusionDet（目标检测）和 LDMLandmark（人脸标志点）已经证明 diffusion 可以做检测
  - 将 TRUST 的问题重新 formulate：输入 CT + 随机初始化的 10 个坐标 → 通过 diffusion reverse process 逐步精炼到正确位置
  - 天然的不确定性：diffusion 的多次采样本身就给出预测分布
  - 可以和 Topo-GCN 结合：每一步 denoise 后用 GCN 做拓扑约束
- **卖点**：
  - Diffusion + 3D 医学标志点检测 = 全新方向
  - 不需要额外的不确定性估计模块（diffusion 本身就是概率模型）
  - 多次采样 → 每个标志点的预测分布 → 天然的临床置信度输出
- **风险**：3D diffusion 的计算成本高，收敛速度慢
- **投稿目标**：CVPR / ICCV / MICCAI

---

### 方向 9：联邦学习用于多中心 TAVI 标志点检测
**一句话**：多家医院的数据不出院，在本地训练 → 聚合模型，解决数据隐私和分布差异问题。

- **核心思路**：
  - 不同医院的 CT 扫描协议、设备、患者群体不同 → 模型 generalization 是难题
  - Federated Learning：每个中心用本地数据训练 TRUST，只共享模型参数
  - 结合 TRUST 的不确定性：不确定性高的中心 → 降低其聚合权重
  - 可以进一步做 personalized FL：每个中心有自己的 adapter
- **卖点**：
  - 多中心验证本身就是临床可信度的金标准
  - FL + 标志点检测几乎没有工作
  - 隐私保护是欧洲（GDPR）特别关注的话题，Rennes 做这个很合适
- **数据需求**：需要 2-3 家合作医院（即使只有少量数据也可以做 proof-of-concept）
- **投稿目标**：MedIA / MICCAI / npj Digital Medicine

---

## 各方向对比总结

| 方向 | 新数据需求 | 新技术复杂度 | 发表周期 | 临床影响力 | 与 TRUST 的关联度 |
|------|-----------|-------------|---------|-----------|-----------------|
| 1. Active Learning | ❌ 无需 | ⭐⭐ 低 | 6–9 月 | ⭐⭐⭐ | ⭐⭐⭐⭐⭐ 直接复用 |
| 2. 联合分割+检测 | ⚠️ 需补标 mask | ⭐⭐⭐ 中 | 9–12 月 | ⭐⭐⭐⭐ | ⭐⭐⭐⭐ |
| 3. UG-HCO 推广 | ❌ 用公开数据 | ⭐⭐ 低 | 6–9 月 | ⭐⭐ | ⭐⭐⭐⭐⭐ 核心模块 |
| 4. 术后结果预测 | ⚠️ 需临床数据 | ⭐⭐⭐ 中 | 12–18 月 | ⭐⭐⭐⭐⭐ | ⭐⭐⭐⭐ |
| 5. CT→术中透视 | ⚠️ 需透视数据 | ⭐⭐⭐⭐ 高 | 12–18 月 | ⭐⭐⭐⭐⭐ | ⭐⭐⭐ |
| 6. BAV 专项 | ⚠️ 需 BAV 数据 | ⭐⭐⭐ 中 | 12–15 月 | ⭐⭐⭐⭐ | ⭐⭐⭐⭐ |
| 7. Foundation Model | ❌ 用公开数据 | ⭐⭐⭐⭐ 高 | 12–18 月 | ⭐⭐⭐⭐ | ⭐⭐⭐ |
| 8. Diffusion Model | ❌ 现有数据 | ⭐⭐⭐⭐⭐ 很高 | 12–18 月 | ⭐⭐⭐ | ⭐⭐ |
| 9. 联邦学习 | ⚠️ 需多中心合作 | ⭐⭐⭐ 中 | 12–18 月 | ⭐⭐⭐⭐ | ⭐⭐⭐ |

---

## 推荐的优先组合策略

### 如果你是硕士/短期目标（1 篇快速论文）：
→ **方向 1（Active Learning）** 或 **方向 3（UG-HCO 推广）**
- 不需要新数据，直接复用 TRUST 的代码和实验
- 6–9 个月可以完成

### 如果你是博士/中期规划（2–3 篇论文）：
→ **方向 1 → 方向 4 → 方向 6** 的递进路线
1. 先发 Active Learning（快速出成果，建立 reputation）
2. 再做术后预测（临床合作，高影响力）
3. 拓展到 BAV（展示 generalization）

### 如果你想做一个系统性的研究方向：
→ **「AI-assisted TAVI planning: from annotation to decision support」**
- TRUST（标志点检测）→ Active Learning（高效标注）→ Multi-task（完整分析）→ 结果预测（决策支持）→ 术中导航（临床闭环）
- 这是一个完整的 PhD thesis 级别的研究线路
