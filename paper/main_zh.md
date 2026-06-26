# TRUST：面向 TAVI 关键点检测与冠脉安全评估的拓扑推理与不确定性感知半监督教学

> **English source of truth:** `paper/main.tex`  
> **中文稿类型：** 逐句审稿镜像稿  
> **同步日期：** 2026-05-18  
> 本文件尽量按 `paper/main.tex` 的段落顺序和句子顺序翻译，用于直接中文审稿。公式、表格数值、图注信息、限定语和主要引用键尽量保留。英文 LaTeX 仍是唯一投稿源文件。

---

## 题目与作者占位

**英文题目：** TRUST: Topological Reasoning and Uncertainty-aware Semi-supervised Teaching for TAVI Landmark Detection and Coronary Safety Assessment

**中文题目：** TRUST：面向 TAVI 关键点检测与冠脉安全评估的拓扑推理与不确定性感知半监督教学

**作者信息：** 英文主文当前仍为 IEEE 模板占位，包括 `First A. Author`、`Second B. Author`、`Third C. Author Jr.`、`\thanks` 中的资助/单位示例，以及 biography 占位。作者、单位、资助和致谢需要后续与导师确认后统一替换。

---

## 摘要

经导管主动脉瓣置入术（TAVI）规划需要从术前计算机断层扫描（CT）中准确定位主动脉根部关键点，以评估传导、冠脉、尺寸选择和释放相关风险。

现有方法只在全监督设定下检测部分关键点集合，使冠脉阻塞风险评估仍处于自动化流程之外，同时还要求对三维（3D）CT 体进行完整标注。

我们提出 TRUST（Topological Reasoning and Uncertainty-aware Semi-supervised Teaching），一个两阶段半监督框架，用于检测 10 个主动脉根部关键点，其中包括用于冠脉安全评估的冠脉口。

TRUST 从跨学生测试时增强（TTA）池中估计统一的逐关键点不确定性，并通过全方差定律将其分解为学生内部 TTA 扰动方差和跨学生分歧。

不同于将不确定性用作单一目的启发式，TRUST 将这一训练期可靠性信号路由到三个协同消费者：交叉伪监督（CPS）损失加权、不确定性引导热传导算子（UG-HCO）扩散率条件化，以及拓扑图卷积网络（Topo-GCN）消息传递门控。

在包含 150 个有标签和 620 个无标签主动脉根部 CT 体的数据集上（100 训练、20 验证、30 测试），TRUST 达到 2.19 mm 的平均径向误差（MRE），相对最强监督基线降低 21.5%。

消融研究识别出统一不确定性是最大的单步贡献。

跨六项 TAVI 规划测量的下游分析表明，关键点精度的提升可以转化为对 TAVI 规划有用的测量层面一致性。

仍需进一步验证的目标包括瓣环面积近似以及冠脉高度筛查的外部验证。

**关键词：** 主动脉根部；冠脉安全；关键点检测；半监督学习；TAVI；拓扑结构。

---

## 1. 引言

**图 1 图注翻译。** 主动脉根部以及本文目标 10 个关键点的解剖概览。左：心脏横截面视图，显示主动脉根部位于左心室和升主动脉交界处。右：主动脉瓣放大俯视示意图，包含三个瓣叶：无冠瓣（NCC）、左冠瓣（LCC）和右冠瓣（RCC）。10 个关键点包括：每个瓣叶最低点处的铰链点（$P_0$：NCC，$P_1$：RCC，$P_2$：LCC），它们定义用于瓣膜尺寸选择的瓣环平面；相邻瓣叶交界处的连合点（$P_3$：NCC--RCC，$P_4$：RCC--LCC，$P_5$：LCC--NCC）；瓣膜几何中心（$P_6$）；膜部室间隔点（$P_7$），位于靠近 NCC--RCC 交界的室间隔基底部，用于膜部室间隔长度（MSL）测量和起搏器风险预测；以及左、右冠脉口（$P_8$：左冠脉口，LCO；$P_9$：右冠脉口，RCO），冠状动脉从主动脉窦起源于此，用于冠脉阻塞风险评估。

经导管主动脉瓣置入术（TAVI）已经成为所有外科风险类别中重度主动脉瓣狭窄患者的标准治疗。

术前计算机断层扫描（CT）规划需要精确识别主动脉根部解剖关键点，以服务两个互补的临床目的：预判操作相关并发症，以及指导器械选择~\cite{Blanke2019_scct_tavi_ct,Ma2023_aortic_landmarks}。

虽然器械尺寸选择依赖由铰链点关键点推导的瓣环平面测量，但关键点更具安全意义的作用在于并发症预判。

两个主要并发症说明了这种依赖关系。

第一，新发传导异常（NOCD）是 TAVI 后起搏器植入的主要原因之一，可通过测量膜部室间隔长度（MSL）来预测；MSL 定义为膜部室间隔点到瓣环平面的距离~\cite{Hamdan2015_msl_avblock,Maeno2017_pacemaker_tavr}。

第二，冠脉阻塞虽然罕见（发生率约为 0.66%），但 30 天死亡率最高可达 40.9%~\cite{Blanke2019_scct_tavi_ct,Ribeiro2013_coronary_obstruction}；专家共识指南将冠脉口距瓣环平面高度低于 12 mm 识别为风险升高指标~\cite{Blanke2019_scct_tavi_ct}。

这两项测量都预设相关关键点能够被准确定位（图 1）：MSL 需要铰链点和膜部室间隔点，冠脉高度需要铰链点和冠脉口。

因此，一个能够联合定位所有临床相关主动脉根部关键点的可靠自动系统，可以支持术前并发症评估所需的测量输入。

现有主动脉根部分析深度学习方法只能部分满足这些临床需求。

Lalys 等~\cite{Lalys2019_tavi_segmentation} 将主动脉根部分割与 TAVI 规划关键点识别结合，Wang 等~\cite{Wang2023_pretavr_dl} 开发了用于综合 pre-TAVI CT 评估的分割型算法。

Ma 等~\cite{Ma2023_aortic_landmarks} 提出两阶段卷积神经网络（CNN）流水线，用于检测 8 个关键点，其中包括用于 NOCD 风险预测的膜部室间隔点，并达到 2.23 mm 的平均径向误差（MRE）。

然而，这些系统都没有对包含冠脉口在内的全部感兴趣关键点进行综合检测。

它们覆盖了器械尺寸选择和部分并发症预测（例如通过 MSL 预测 NOCD），但遗漏冠脉口，使冠脉阻塞风险评估仍处在自动化流程之外。

此外，某些关键点仍难以可靠检测，尤其是膜部室间隔点（$P_7$），它位于易受钙化伪影影响的软组织边界处。

第二个共同限制是，这些方法都在全监督设定下运行，并要求训练数据完整标注。

这是一个实际瓶颈，因为三维（3D）CT 体的专家标注耗时且需要专门临床知识，导致绝大多数可用扫描仍然未标注。

标注瓶颈促使我们从全监督方法转向能够利用大量未标注扫描的半监督方法。

半监督学习（SSL）已经在医学影像分割中取得显著成功；在该领域，一致性正则~\cite{Tarvainen2017_mean_teacher,Laine2017_temporal_ensembling}、交叉伪监督（CPS）~\cite{Chen2021_cps} 和基于置信度的筛选~\cite{Sohn2020_fixmatch} 能够有效利用未标注数据。

然而，用于关键点检测的 SSL 仍相对缺乏探索~\cite{Dong2019_teacher_students,Honari2018_ssl_landmarks,Kim2023_hybridmatch}。

根本原因是错误容忍度在性质上不同：在分割任务中，一个误标体素会被周围正确预测吸收；而在关键点检测中，一个错误点可以直接破坏下游临床测量。

例如，冠脉口定位 3 mm 的误差就可能改变测得冠脉高度是否低于临床相关参考阈值。

Ren 等~\cite{Ren2024_g2lcps} 近期将 CPS 引入关键点预测，但其方法没有纳入解剖结构约束，也没有处理 3D 主动脉根部 CT 的特定成像挑战，例如钙化导致的特征退化。

主动脉根部 CT 成像为关键点检测带来三个相互关联的技术挑战。

第一，主动脉瓣及周围结构的重度钙化会破坏局部图像特征，并不成比例地影响邻近钙化组织的关键点，例如膜部室间隔点 $P_7$。

第二，10 个主动脉根部关键点具有严格的解剖空间关系：铰链点定义瓣环平面，连合点沿根部圆周与铰链点交替排列，冠脉口与瓣环保持特征性距离；这些关系不是单关键点检测器能够强制保证的。

第三，不同关键点的检测难度差异很大：像 $P_6$ 这样的中心点天然较容易，而像 $P_7$ 这样的模糊软组织关键点更难。

图神经网络（GNN）~\cite{Scarselli2009_gnn,Kipf2017_gcn} 和拓扑感知方法~\cite{Li2020_structured_landmark,Lang2020_gcn_landmark} 可以编码关键点之间的结构关系，但在半监督设定中，不可靠伪标签可能通过图消息传递机制传播错误。

认知不确定性估计~\cite{Gal2016_dropout_uncertainty,Kendall2017_uncertainty} 可以识别不可靠预测，而不确定性引导 SSL 方法~\cite{Yu2019_uncertainty_aware,Wang2022_u2pl} 已在分割中显示潜力。

然而，现有方法通常将不确定性用于单一目的：要么加权伪标签，要么调制特征，要么辅助事后精修；它们并没有把不确定性作为一个同时协调多个学习模块的一致控制信号。

本文提出 **T**opological **R**easoning and **U**ncertainty-aware **S**emi-supervised **T**eaching（**TRUST**），一个用于 10 个主动脉根部关键点检测的半监督框架。

该框架将既有 TAVI 关键点流水线扩展到冠脉口，并通过不确定性感知拓扑推理在标注有限条件下提升定位可靠性。

TRUST 旨在回应上述两个缺口：冠脉安全评估的临床覆盖不完整，以及解剖约束关键点检测中的不可靠伪监督问题。

具体架构、不确定性公式和训练目标在第 3 节中介绍。

本文主要贡献如下。

1. 提出一个 10 关键点 TAVI 规划框架，将既有主动脉根部关键点检测扩展到冠脉口，从而在标签稀缺条件下同时支持 NOCD 风险评估和冠脉阻塞筛查。

2. 提出 TRUST 不确定性感知半监督框架，通过统一逐关键点可靠性信号协调伪监督、自适应特征传播和解剖图精修。

3. 在 150 个有标签和 620 个无标签 CT 体上进行综合评估，包括逐关键点准确性、消融分析，以及六项具有临床意义的下游 TAVI 规划测量。

本文其余部分组织如下。

第 2 节回顾相关工作。

第 3 节详细介绍 TRUST 方法。

第 4 节和第 5 节分别呈现实验结果和讨论。

第 6 节总结全文。

---

## 2. 相关工作

### 2.1 TAVI 规划与主动脉根部关键点

主动脉根部自动分析方法主要采用基于分割或直接关键点检测的流水线。

Lalys 等~\cite{Lalys2019_tavi_segmentation} 提出主动脉根部自动分割，并随后识别 TAVI 规划关键点；Wang 等~\cite{Wang2023_pretavr_dl} 开发了用于综合 pre-TAVI CT 评估的全自动深度学习算法，其中包括风险因素检测。

采用直接检测策略的 Ma 等~\cite{Ma2023_aortic_landmarks} 提出两阶段 CNN 流水线，检测 8 个主动脉瓣关键点，包括用于 NOCD 风险预测的膜部室间隔点，并达到 2.23 mm 的 MRE。

这些系统支持器械尺寸选择和部分并发症预测，但既有关键点流水线没有联合纳入冠脉口用于冠脉阻塞筛查，并且仍依赖全监督训练。

主动脉根部关键点还支持操作指导：由铰链点推导的瓣环平面方向决定瓣膜释放的最优 C 臂透视角度，可以减少对比剂使用和辐射暴露~\cite{Gurvitch2010_tavi_angulation,Achenbach2013_annular_plane,Arnold2012_carm_projection}。

因此，本文将关键点集合扩展到 10 个，纳入冠脉口，并在半监督条件下处理标注稀缺。

### 2.2 解剖关键点检测

深度学习解剖关键点检测方法大体遵循两种策略：热图回归和直接坐标回归。

在热图回归中，CNN 为每个关键点预测一个空间概率图，峰值表示预测位置。

Payer 等~\cite{Payer2019_integrating_spatial} 将空间配置网络整合到热图回归中，使用第二阶段网络捕捉关键点之间的空间关系。

Newell 等~\cite{Newell2016_stacked_hourglass} 提出 stacked hourglass 网络，通过自底向上和自顶向下路径反复处理特征，从而实现多尺度推理。

高分辨率表示网络（HRNet）~\cite{Sun2019_hrnet} 在整个网络中保持高分辨率特征，避免下采样造成的信息损失。

在直接坐标回归中，注意力引导深度回归等方法~\cite{Zhong2019_attention_landmark} 通过全连接层预测关键点坐标，有时还增强了注意力机制。

三维医学关键点检测引入额外挑战，包括巨大的体数据搜索空间和各向异性体素分辨率。

Zheng 等~\cite{Zheng2015_3d_landmark} 提出较早的 3D 深度学习体数据关键点检测流水线之一，采用粗到细策略实现高效检测。

Ghesu 等~\cite{Ghesu2017_multiscale_landmark} 开发了鲁棒多尺度关键点检测方法，通过基于强化学习的搜索处理不完整 3D CT 数据。

这些粗到细范式启发了本文的两阶段设计：Stage 1 先定位主动脉根部感兴趣区域（ROI），Stage 2 再在裁剪体内进行精细关键点检测。

最近，nnLandmark~\cite{nnLandmark2025} 提出基于 nnU-Net 训练引擎的自配置 3D 关键点检测框架，为通用体数据关键点任务自动化超参数选择和预处理。

上述方法的共同限制是，它们将每个关键点独立处理，或仅施加有限成对约束。

当关键点具有强解剖相互依赖关系时，例如主动脉根部中铰链点、连合点和冠脉口保持固定拓扑关系，独立检测可能产生解剖上不合理的配置。

这促使本文整合基于图的结构推理（第 2.4 节）。

### 2.3 医学影像中的半监督学习

半监督学习旨在利用未标注数据和少量有标签数据。

在专家标注昂贵的医学影像中，SSL 已经取得显著收益，尤其是在分割任务中。

核心范式包括一致性正则，即鼓励输入或模型扰动下预测保持不变~\cite{Laine2017_temporal_ensembling,Tarvainen2017_mean_teacher}；以及伪标签，即使用高置信模型预测作为替代标注~\cite{Sohn2020_fixmatch}。

CPS~\cite{Chen2021_cps} 将伪标签扩展到双网络架构：两个不同初始化的网络互相提供伪标签，从而减少自训练固有的确认偏差。

Perturbed Mean Teacher~\cite{Liu2022_perturbed_mean_teacher} 和 unreliable pseudo-label exploitation~\cite{Wang2022_u2pl} 等变体进一步提升鲁棒性。

尽管已有这些进展，用于关键点检测的 SSL 仍不如分割任务成熟。

既有研究已经通过辅助属性、teacher-student 监督、语义表示和热图-坐标混合表示探索了半监督人脸或通用关键点定位~\cite{Honari2018_ssl_landmarks,Dong2019_teacher_students,Moskvyak2021_ssl_keypoints,Kim2023_hybridmatch}。

与本文设定更接近的是，Ren 等~\cite{Ren2024_g2lcps} 提出 G2LCPS，将 CPS 引入具有 global-to-local 架构的端到端关键点预测；Tang 等~\cite{Tang2025_cbct_ssl} 将 SSL 应用于正颌手术规划中的 3D CBCT 关键点检测。

然而，3D 医学关键点检测中的 SSL 研究仍然稀少，现有方法没有同时处理主动脉根部特异拓扑、钙化退化 CT 外观和下游 TAVI 测量这三类要求。

此外，分割和关键点检测之间的基本错误传播不对称性仍很少被关注：一个伪标签错误即可直接破坏下游临床测量。

### 2.4 面向关键点的图结构推理

图神经网络（GNN）~\cite{Scarselli2009_gnn,Kipf2017_gcn} 为编码关键点之间的结构关系提供了自然机制。

Battaglia 等~\cite{Battaglia2018_relational_inductive} 形式化了关系归纳偏置在深度学习中的作用，并指出显式图结构能使模型更高效地学习几何配置。

在关键点检测中，Li 等~\cite{Li2020_structured_landmark} 提出 topology-adapting deep graph learning，其中 GCN 通过学习邻接结构上的推理精修关键点预测。

Lang 等~\cite{Lang2020_gcn_landmark} 将局部注意力图卷积应用于颅颌面 CBCT 关键点定位，通过邻近关键点之间的注意力加权消息传递提升精度。

这些方法表明，对关键点施加结构一致性可以提高检测准确性，尤其适用于单独观察时模糊但受解剖邻居约束的关键点。

然而，它们工作在全监督设定中，图输入来自真值或高质量初始预测。

在半监督设定中，图接收来自教师伪标签坐标的含噪输入，标准消息传递有将不可靠关键点错误传播到邻居的风险。

TRUST 通过不确定性门控消息传递解决这一问题：同一逐关键点可靠性权重既过滤伪标签，也在 GCN 更新中衰减不确定关键点的贡献，防止错误沿解剖图传播。

### 2.5 深度学习中的不确定性估计

深度学习中的不确定性估计区分偶然不确定性和认知不确定性~\cite{Kendall2017_uncertainty}。

偶然不确定性对应固有数据噪声，认知不确定性对应有限训练数据导致的模型无知。

认知不确定性会随着更多数据而降低，因此在标签稀缺的半监督设定中尤其相关。

常见估计技术包括 Monte Carlo dropout~\cite{Gal2016_dropout_uncertainty}、deep ensembles~\cite{Lakshminarayanan2017_deep_ensembles} 和 TTA，它们都通过随机扰动下的预测方差来衡量不确定性。

在半监督医学影像分析中，不确定性主要用于提升伪标签质量。

Yu 等~\cite{Yu2019_uncertainty_aware} 提出用于 3D 左心房分割的不确定性感知自集成模型，使用 MC dropout 不确定性加权一致性损失并排除高不确定体素。

Wang 等~\cite{Wang2022_u2pl} 进一步表明，由不确定性判定为不可靠的伪标签可以被重新用于负学习信号，而不是直接丢弃。

这些工作将不确定性应用于单一机制（损失加权或伪标签选择），并且处理的是密集分割任务。

在 TRUST 中，我们离开这种单一用途范式。

我们从跨学生 TTA 池中推导一个单一逐关键点不确定性估计，结合学生内部 TTA 扰动方差和跨学生分歧；后者作为模型分歧代理，而不是 Kendall-Gal 严格意义下的认知不确定性估计。

随后，我们将其作为协调的训练期控制信号，同时路由到三个消费者：CPS 损失加权、HCO 扩散率条件化和 GCN 消息传递门控。

与其把不确定性视作某个模块中的孤立启发式，这种统一设计允许同一个可靠性信号共同塑造特征提取、过滤伪监督并调节结构精修。

**图 2 图注翻译。** TRUST 框架概览。**Stage 1**（顶部）：粗检测器将完整 CT 体下采样并产生感兴趣区域（ROI）中心 $p_c$；ROI 被裁剪并重采样到 $128^3$。**Stage 2**（中部）：两个结构相同的 student-teacher 对在 ROI 空间中通过交叉伪监督（CPS）运行（交叉箭头）。跨学生测试时增强（TTA）池聚合两个指数移动平均（EMA）教师的增强预测，产生统一逐关键点不确定性 $u_k$ 和可靠性权重 $w_k$（绿色模块）。在每个学生内部（蓝色模块），特征经过带不确定性引导热传导算子（UG-HCO）层的 2 级级联 hourglass backbone（HG1、HG2），随后进入热图头和 offset 头，二者输出结合用于坐标提取（$\hat{p}_k$）。拓扑图卷积网络（Topo-GCN）将坐标精修为 $p_k^{\mathrm{ref}}$，两个学生的预测被融合为最终输出 $p_k^{\mathrm{final}}$。**损失函数**（底部）：总损失包含监督热图和坐标回归（$\mathcal{L}_{\mathrm{sup}}$）、不确定性加权 CPS（$\mathcal{L}_{\mathrm{cps}}$）以及拓扑精修（$\mathcal{L}_{\mathrm{topo}}$）。

---

## 3. 方法

### 3.1 问题定义

给定一个 3D CT 体 $\mathbf{x} \in \mathbb{R}^{D \times H \times W}$，我们的目标是定位主动脉根部的 $K=10$ 个解剖关键点：

$$
\mathcal{Y} = \{p_1, p_2, \ldots, p_K\}, \qquad p_k \in \mathbb{R}^3 .
$$

其中 $p_k$ 表示第 $k$ 个关键点的 3D 坐标；全文中我们用小写 $p_k$ 表示坐标，用大写 $P_k$ 表示关键点标识。

与既有专注于 8 个瓣膜尺寸相关关键点的流水线~\cite{Ma2023_aortic_landmarks} 相比，我们额外纳入左、右冠脉口（$P_8$、$P_9$），以便为冠脉安全评估测量冠脉高度。

我们将关键点定位表述为热图回归问题。

对于每个关键点 $p_k$，目标热图 $H_k^*$ 由以 $p_k$ 为中心、标准差为 $\sigma$ 的 3D Gaussian 生成：

$$
H_k^{*}(v) = \exp\left(-\frac{\|v - p_k\|^2}{2\sigma^2}\right),
$$

其中 $v$ 索引热图空间中的体素。

推理过程中，关键点坐标由预测热图通过峰值定位获得，例如 $\arg\max$。

我们在半监督设定下处理该任务，数据包括有标签子集 $\mathcal{D}_L = \{(\mathbf{x}_i, \mathcal{Y}_i)\}_{i=1}^{N_L}$ 和一个大的无标签子集 $\mathcal{D}_U = \{\mathbf{x}_i\}_{i=1}^{N_U}$，其中 $N_U \gg N_L$。

我们的目标是利用 $\mathcal{D}_U$ 在有限标注下提升 10 关键点定位泛化能力，尤其是视觉模糊关键点和新加入的冠脉口。

### 3.2 框架总览

我们提出 TRUST，即 Topological Reasoning and Uncertainty-aware Semi-supervised Teaching，一个用于半监督主动脉根部关键点检测的两阶段框架。

图 2 展示了整体架构。

**Stage 1：ROI 提议。**

如图 2 顶部所示，Stage 1 原样复用 Ma 等~\cite{Ma2023_aortic_landmarks} 的级联 hourglass 粗检测器 $g_\psi$，不改变架构、训练协议或超参数。

该检测器将完整 CT 体按各向同性间距下采样到 $128\times128\times128$，并预测所有 $K=10$ 个关键点的近似坐标。

ROI 中心 $p_c$ 取为这些粗预测的质心。

随后以 $p_c$ 为中心裁剪并重采样 ROI 到固定大小 $\mathbf{x}_{\mathrm{roi}} \in \mathbb{R}^{D' \times H' \times W'}$，所有后续处理都在该 ROI 内进行。

ROI 窗口尺寸被设置为在典型主动脉根部范围周围提供安全边界，以在 Stage 2 开始前吸收 Stage 1 的粗检测误差。

**Stage 2：TRUST。**

Stage 2 完全在 ROI 坐标空间中运行。

两个结构相同的学生网络各自配有一个指数移动平均（EMA）教师，并通过 CPS~\cite{Ren2024_g2lcps} 训练：每个教师的热图预测作为对方学生的伪标签（第 3.3 节）。

在每个学生内部，不确定性引导热传导算子（UG-HCO）通过为预测不可靠的关键点增大扩散率来自适应特征传播，从而补偿钙化导致的特征退化（第 3.4 节）。

随后，轻量拓扑图卷积网络（Topo-GCN）通过主动脉根部解剖关键点图上的推理来精修预测坐标，以强制结构上合理的配置（第 3.5 节）。

训练期驱动这些模块的逐关键点不确定性只估计一次，来自聚合两个教师增强预测的跨学生 TTA 池，并通过全方差定律分解为学生内部 TTA 扰动分量和跨学生分歧分量。

得到的单一可靠性信号同时加权 CPS 损失、条件化 UG-HCO 扩散率并门控 GCN 消息传递（第 3.6 节）。

### 3.3 带不确定性估计的交叉伪监督

**双网络架构。**

如图 2 左侧所示，Stage 2 使用两个结构相同的学生网络 $f_{\theta_1}$ 和 $f_{\theta_2}$。

它们共享同一架构（stacked hourglass、UG-HCO 层和 Topo-GCN 精修头），但使用不同随机权重初始化。

每个学生都有一个对应的 EMA 教师~\cite{Tarvainen2017_mean_teacher}：

$$
\theta'_{j,n} = \alpha\,\theta'_{j,n-1} + (1-\alpha)\,\theta_{j,n}, \quad j \in \{1,2\},
$$

其中 $\alpha$ 是 EMA 衰减率。

四个网络都在 Stage 1 产生的同一个 ROI 裁剪 $\mathbf{x}_{\mathrm{roi}}$ 上运行。

**交叉伪监督。**

遵循 Chen 等~\cite{Chen2021_cps}，并采用 Ren 等~\cite{Ren2024_g2lcps} 对关键点检测的适配，我们使用 CPS 来利用无标签数据。

如图 2 中交叉箭头所示，在每个无标签体上，Teacher 1 的热图预测作为 Student 2 的伪标签，反之亦然：

$$
\hat{H}^{(T_1)} = \mathrm{sg}\!\bigl(f_{\theta'_1}(\mathbf{x}_{\mathrm{roi}})\bigr) \to f_{\theta_2},
$$

$$
\hat{H}^{(T_2)} = \mathrm{sg}\!\bigl(f_{\theta'_2}(\mathbf{x}_{\mathrm{roi}})\bigr) \to f_{\theta_1}.
$$

其中 $\hat{H}^{(T_j)}$ 表示来自 Teacher $j$ 的 stop-gradient 热图预测，$\mathrm{sg}(\cdot)$ 表示 stop-gradient 操作。

由于两个学生初始化不同，它们的错误部分去相关；与自训练相比，交叉指导因而减少确认偏差。

TRUST 中一个有意的设计选择是将 CPS 完全限制在 Stage 2 的 ROI 坐标空间内。

原因是 CPS 依赖热图到热图的均方误差（MSE），两个网络必须在相同空间分辨率和视野下运行，而 ROI 裁剪内自然满足这一点。

**通过跨学生 TTA 估计认知不确定性。**

为了防止不可靠伪标签污染训练，尤其是重度钙化下 $P_7$ 这类视觉模糊关键点，我们从跨学生测试时增强（TTA）池估计逐关键点不确定性，如图 2 绿色模块所示。

对于每个无标签体，两个 EMA 教师在随机增强 $\{\tau_1,\ldots,\tau_T\}$ 下各自产生 $T$ 个热图预测：

$$
\hat{H}_k^{(j,m)} = f_{\theta'_j}\!\bigl(\tau_m(\mathbf{x}_{\mathrm{roi}})\bigr),
\quad j\in\{1,2\},\quad m=1,\ldots,T.
$$

这些热图产生坐标 $\hat{p}_k^{(j,m)} = \mathrm{softargmax}(\hat{H}_k^{(j,m)})$。

我们池化全部 $2T$ 个坐标，并将逐关键点不确定性定义为池内空间方差：

$$
u_k = \frac{1}{2T}\sum_{j=1}^{2}\sum_{m=1}^{T}
\bigl\|\hat{p}_k^{(j,m)} - \bar{p}_k\bigr\|^2,
\quad
\bar{p}_k = \frac{1}{2T}\sum_{j,m}\hat{p}_k^{(j,m)}.
$$

根据全方差定律，这个池化量可以原则性地分解为两个互补部分：

$$
u_k =
\underbrace{\tfrac{1}{2}\!\bigl(u_k^{(1)} + u_k^{(2)}\bigr)}_{\text{学生内部 TTA 方差}}
+
\underbrace{\tfrac{1}{4}\!\bigl\|\bar{p}_k^{(1)} - \bar{p}_k^{(2)}\bigr\|^2}_{\text{跨学生分歧}}.
$$

其中 $u_k^{(j)} = \tfrac{1}{T}\sum_m\|\hat{p}_k^{(j,m)}-\bar{p}_k^{(j)}\|^2$ 是教师内部 TTA 方差，$\bar{p}_k^{(j)}=\tfrac{1}{T}\sum_m\hat{p}_k^{(j,m)}$ 是教师 $j$ 的平均坐标。

第一项捕捉每个教师预测对输入扰动的敏感性。

第二项捕捉两个教师中心预测之间的分歧；这种分歧会持续存在，因为两个学生由不同初始化训练，因此在不同子空间中编码残余定位噪声。

我们有意避免在严格 Kendall-Gal 意义上将这些项标记为“偶然”或“认知”不确定性~\cite{Kendall2017_uncertainty}。

跨学生分歧反映不同初始化下的模型分歧，而不是经过校准的模型无知估计；我们将其视为有用的经验可靠性代理，而不是形式化的认知不确定性估计。

这种分解在实践中很重要：在收敛的深度热图回归模型中，soft-argmax 输出由单个高 logit 体素主导，导致学生内部 TTA 项饱和，使纯 TTA 不确定性几乎无信息~\cite{Yu2019_uncertainty_aware}。

跨学生分歧项保持非平凡，从而保留可靠性信号，使框架对过度自信预测更鲁棒。

我们将统一不确定性转换为逐关键点可靠性权重：

$$
w_k = \exp\!\bigl(-\alpha_u \cdot u_k\bigr),
$$

其中 $\alpha_u>0$ 是灵敏度超参数，单位为 mm$^{-2}$，使指数无量纲。

当 $u_k \to 0$ 时，$w_k \to 1$（完全信任）；当 $u_k$ 增大时，$w_k \to 0$（抑制）。

如图 2 中从不确定性模块出发的彩色箭头所示，同一个统一可靠性信号被三个下游模块消费：CPS 损失中的伪监督加权、UG-HCO 扩散率条件化，以及 Topo-GCN 消息传递门控。

### 3.4 不确定性引导热传导算子

主动脉根部 CT 图像经常表现为低对比和钙化伪影，这会降低受影响关键点的局部特征质量。

标准卷积 backbone 在固定感受野尺度上处理特征，无法在退化区域选择性增强全局上下文，同时在其他位置保留细节。

我们通过 UG-HCO 处理这一问题；它被集成到两个 Stage 2 学生网络中，如图 2 的学生路径所示，并因此也被其 EMA 教师继承。

UG-HCO 改造近期提出的基于热传导的特征传播~\cite{wang2025building}，用第 3.3 节估计的统一逐关键点不确定性对扩散率进行条件化。

**热传导预备知识。**

热方程描述标量场 $\varphi(\mathbf{r},t)$ 在空间坐标 $\mathbf{r}$ 和时间 $t$ 上的扩散：

$$
\frac{\partial \varphi}{\partial t} = \kappa\,\nabla^2 \varphi,
$$

其中 $\kappa>0$ 是热扩散率。

应用 Fourier 变换得到频域常微分方程：

$$
\frac{\mathrm{d}\tilde{\varphi}(\boldsymbol{\omega},t)}{\mathrm{d}t}
=
-\kappa\,\|\boldsymbol{\omega}\|^2\,\tilde{\varphi}(\boldsymbol{\omega},t),
$$

其闭式解为：

$$
\tilde{\varphi}(\boldsymbol{\omega},t)
=
\tilde{\varphi}_0(\boldsymbol{\omega})
\exp\!\bigl(-\kappa\,\|\boldsymbol{\omega}\|^2\,t\bigr),
$$

其中 $\tilde{\varphi}_0(\boldsymbol{\omega})$ 是频域初始条件。

直观上，高频衰减更快，形成一个自适应低通滤波，其强度由乘积 $\kappa t$ 控制。

Wang 等~\cite{wang2025building} 表明，该操作可以在 Neumann 边界条件下通过离散余弦变换（DCT）及其逆变换（IDCT）高效实现：

$$
\mathbf{U}_t =
\mathrm{IDCT}\!\left(
\mathrm{DCT}(\mathbf{U}_0)
\odot
\exp\!\bigl(-\kappa(\boldsymbol{\omega})\,\|\boldsymbol{\omega}\|^2\,t\bigr)
\right),
$$

其中 $\mathbf{U}_0$ 是输入特征图，$\kappa(\boldsymbol{\omega})$ 是从频率值嵌入（FVE）预测得到的、可学习且频率自适应的扩散率。

**不确定性条件化扩散率。**

在 vanilla HCO 中，$\kappa(\boldsymbol{\omega})$ 只依赖频率，并在所有空间位置共享。

我们扩展这一点，使扩散率以逐关键点不确定性 $\{u_k\}_{k=1}^K$ 为条件。

具体来说，通过将每个归一化不确定性 $u_k/\max_{k'}u_{k'}$ 放置在对应关键点位置，并通过三线性插值到特征分辨率，构造归一化不确定性图 $\bar{\mathcal{U}}$。

不确定性条件化扩散率为：

$$
\kappa(\boldsymbol{\omega},\mathcal{U})
=
\kappa_{\mathrm{base}}(\boldsymbol{\omega})\cdot
\bigl(1+\beta\,\bar{\mathcal{U}}\bigr),
$$

其中 $\kappa_{\mathrm{base}}(\boldsymbol{\omega})$ 是标准 FVE 预测扩散率，$\bar{\mathcal{U}}$ 是归一化且空间广播的不确定性图，$\beta$ 是可学习尺度参数。

该效果是直观的：在教师不确定的区域（高 $\bar{\mathcal{U}}$，例如 $P_7$ 周围钙化区域），扩散率增大，从而促进更强全局上下文传播，以补偿退化局部特征。

在置信区域（低 $\bar{\mathcal{U}}$），扩散率保持接近基线，从而保留细粒度局部细节。

UG-HCO 层替代 backbone 内的标准卷积块，并端到端训练：

$$
\mathbf{U}_t =
\mathrm{IDCT}\!\left(
\mathrm{DCT}(\mathbf{U}_0)
\odot
\exp\!\bigl(-\kappa(\boldsymbol{\omega},\mathcal{U})\,\|\boldsymbol{\omega}\|^2\,t\bigr)
\right).
$$

对于 3D CT 体，我们使用三维 DCT，将该公式自然扩展到体数据。

### 3.5 通过图精修进行拓扑推理

backbone 的初始关键点预测可能违反解剖合理性，例如由于钙化伪影将 $P_7$ 预测到主动脉根部边界之外。

如图 2 中每个学生路径末端所示，两个 Stage 2 学生都包含轻量 Topo-GCN 精修头，通过在解剖关键点图上推理来强制结构一致性~\cite{Li2020_structured_landmark}。

重要的是，CPS 监督应用在热图层面（GCN 之前），因此 GCN 精修坐标时不会把图结构耦合进伪标签对齐。

**图构建。**

我们定义固定图 $\mathcal{G}=(\mathcal{V},\mathcal{E})$，其中 $|\mathcal{V}|=K=10$ 个节点。

边 $\mathcal{E}$ 通过二值邻接矩阵 $A$ 编码解剖连接：主动脉中心 $P_6$ 连接所有其他关键点；铰链点 $P_0$--$P_2$ 和连合点 $P_3$--$P_5$ 形成瓣环；每个冠脉口（$P_8$、$P_9$）连接最近的铰链和连合对；膜部室间隔点 $P_7$ 连接解剖相邻关键点（$P_0$、$P_1$、$P_3$、$P_6$）。

每个节点通过多层感知机（MLP）接收一个初始特征：

$$
\mathbf{h}_k^{(0)}
=
\mathrm{MLP}_{\mathrm{in}}\!\bigl(\hat{p}_k\bigr)
\in \mathbb{R}^{d_h},
$$

其中 $\hat{p}_k\in\mathbb{R}^3$ 是通过可微 soft-argmax 从 backbone 热图获得的坐标，$d_h=64$ 是隐藏维度。

**不确定性门控消息传递。**

GCN 由 $L=4$ 层组成，参数量约为 32K。

在每一层中，节点特征更新为：

$$
\mathbf{h}_k^{(l+1)}
=
\sigma_{\mathrm{act}}\!\left(
\sum_{i\in\mathcal{N}(k)}
\tilde{A}_{ki}\,w_i\,
\mathbf{W}^{(l)}\mathbf{h}_i^{(l)}
+
\mathbf{W}_e^{(l)}\mathbf{e}_{ki}
\right).
$$

其中 $\tilde{A}_{ki}$ 是带自环的度归一化邻接矩阵第 $(k,i)$ 项，$w_i$ 是来自可靠性权重公式、并在两个学生 GCN 更新中共享的统一可靠性权重。

$\mathbf{W}^{(l)}$ 和 $\mathbf{W}_e^{(l)}$ 是可学习权重矩阵，$\mathbf{e}_{ki}=\mathrm{MLP}_e(\hat{p}_k-\hat{p}_i)$ 编码相连关键点之间的相对位置，$\sigma_{\mathrm{act}}(\cdot)$ 是 ReLU 激活。

可靠性门控确保高不确定性关键点（例如钙化下的 $P_7$）对邻居更新贡献更少，而准确检测的关键点，如 $P_6$（中心，通常误差低）和 $P_0$--$P_2$（铰链点），可以作为可靠锚点。

**门控残差坐标修正。**

经过 $L$ 层 GCN 后，我们用一个学习到的门预测残差坐标偏移：

$$
\Delta p_k = \mathrm{MLP}_{\mathrm{out}}\!\bigl(\mathbf{h}_k^{(L)}\bigr),
\qquad
\gamma_k = \sigma_{\mathrm{gate}}\!\bigl(\mathbf{h}_k^{(L)}\bigr),
$$

$$
p_k^{\mathrm{ref}} = \hat{p}_k + \gamma_k \odot \Delta p_k.
$$

其中 $\gamma_k\in(0,1)^3$ 是逐坐标门，$p_k^{\mathrm{ref}}$ 是精修坐标。

残差形式确保 GCN 学习小幅修正而不是绝对位置，而门控允许网络按坐标轴选择性应用或抑制精修。

整条流水线，包括 soft-argmax、GCN 和残差修正，都是可微的，使端到端训练中梯度能够回传到 backbone。

### 3.6 损失函数与训练

**监督损失。**

在有标签数据 $\mathcal{D}_L$ 上，每个学生 $j\in\{1,2\}$ 最小化热图回归和坐标回归损失的组合：

$$
\mathcal{L}_{\mathrm{sup}}^{(j)}
=
\frac{1}{K}\sum_{k=1}^{K}
\left(
\|H_k^{(j)}-H_k^*\|^2
+
\lambda_p\|\hat{p}_k^{(j)}-p_k\|_1
\right),
$$

其中 $H_k^{(j)}$ 是 Student $j$ 的预测热图，$H_k^*$ 是由高斯热图公式生成的真值热图，$\hat{p}_k^{(j)}$ 是通过 soft-argmax 得到的预测坐标，$p_k$ 是真值坐标。

**不确定性加权 CPS 损失。**

在无标签数据 $\mathcal{D}_U$ 上，CPS 强制每个学生与对方教师之间的热图一致性，并由伪标签生成教师的可靠性进行加权：

$$
\mathcal{L}_{\mathrm{cps}}
=
\frac{1}{K}\sum_{k=1}^{K}w_k
\left(
\|H_k^{(2)}-\hat{H}_k^{(T_1)}\|^2
+
\|H_k^{(1)}-\hat{H}_k^{(T_2)}\|^2
\right),
$$

其中 $\hat{H}_k^{(T_j)}$ 是 Teacher $j$ 的 stop-gradient 热图预测，$w_k$ 是统一可靠性权重。

CPS 在 Topo-GCN 之前的热图层面计算，以保证稳定的同空间监督。

不确定性加权降低了整体池化方差较高关键点的噪声伪标签影响。

**拓扑精修损失。**

对于有标签样本，我们监督两个学生的 GCN 精修坐标：

$$
\mathcal{L}_{\mathrm{topo}}^{(j)}
=
\frac{1}{K}\sum_{k=1}^{K}
\|p_k^{\mathrm{ref},(j)}-p_k\|_1,
\quad j\in\{1,2\}.
$$

**总损失。**

Stage 2 总训练目标组合所有项：

$$
\mathcal{L}_{\mathrm{total}}
=
\sum_{j=1}^{2}\mathcal{L}_{\mathrm{sup}}^{(j)}
+
\lambda_c\rho(n)\mathcal{L}_{\mathrm{cps}}
+
\lambda_t\sum_{j=1}^{2}\mathcal{L}_{\mathrm{topo}}^{(j)}.
$$

其中 $\lambda_c$ 和 $\lambda_t$ 是平衡系数，$\rho(n)=\exp(-5(1-n/n_{\max})^2)$ 是 Gaussian ramp-up 函数，用于在训练 epoch $n$ 中逐渐增加 CPS 权重~\cite{Tarvainen2017_mean_teacher}。

**训练过程。**

训练分三阶段进行。

第一，Stage 1 预训练：ROI 提议网络 $g_\psi$ 在 $\mathcal{D}_L$ 上用监督关键点损失训练。

第二，Stage 2 warm-up：两个学生在 $\mathcal{D}_L$ 上用 $\mathcal{L}_{\mathrm{sup}}^{(j)}+\lambda_t\mathcal{L}_{\mathrm{topo}}^{(j)}$ 训练 $N_{\mathrm{warm}}$ 个 epoch。

EMA 教师初始化为各自学生的副本。

第三，Stage 2 主阶段：每次迭代从 $\mathcal{D}_L$ 采样一个有标签 mini-batch，并从 $\mathcal{D}_U$ 采样一个无标签 mini-batch。

每个教师产生 $T$ 个随机预测用于不确定性估计；总损失中的所有项共同参与学生更新；两个教师通过 EMA 更新。

推理时，Stage 1 产生 ROI 裁剪。

两个 Stage 2 学生独立处理 ROI，并各自通过 Topo-GCN 产生精修坐标 $p_k^{\mathrm{ref},(j)}$。

跨学生 TTA 池同时提供池级坐标均值 $\bar{p}_k$ 以及全方差分解。

如图 2 右侧所示，我们使用来自学生内部分量的权重融合两个学生预测：

$$
p_k^{\mathrm{final}}
=
\frac{
\tilde{w}_k^{(1)}p_k^{\mathrm{ref},(1)}
+
\tilde{w}_k^{(2)}p_k^{\mathrm{ref},(2)}
}{
\tilde{w}_k^{(1)}+\tilde{w}_k^{(2)}
},
\qquad
\tilde{w}_k^{(j)}=\exp(-\alpha_u u_k^{(j)}).
$$

因此，每个学生预测按其自身教师内部 TTA 方差的反比加权。

融合坐标被重投影到原始物理坐标空间，用于临床测量，例如冠脉高度和 MSL。

---

## 4. 实验

### 4.1 数据集与评价指标

**数据集。**

我们在 Rennes University Hospital 采集的 770 例术前心脏 CT 体数据上评估 TRUST。

所有患者都接受了心电门控增强 CT，作为术前 TAVI 评估的一部分。

每个体数据由 $512\times512$ 像素切片组成，面内像素间距范围为 0.225 到 0.492 mm，层厚为 0.500 到 0.750 mm。

770 个体数据中，150 个由受训临床医师标注了 $K=10$ 个关键点坐标，并构成有标签子集 $\mathcal{D}_L$。

原始 8 个关键点（$P_0$--$P_7$）遵循 Ma 等~\cite{Ma2023_aortic_landmarks} 的标注协议；我们将其扩展到包括左、右冠脉口（$P_8$、$P_9$），从而支持自动冠脉高度测量。

剩余 620 个未标注体数据构成无标签子集 $\mathcal{D}_U$，并且在同一成像协议下采集。

在 $\mathcal{D}_L$ 内，我们采用与 Ma 等~\cite{Ma2023_aortic_landmarks} 相同的划分：100 个体数据用于训练，20 个用于验证，30 个用于测试。

全部 620 个无标签体数据都在 Stage 2 训练中使用。

为将模型表现置于人工标注变异的背景中，同一 30 例测试集还按照原始 8 关键点协议（$P_0$--$P_7$）接受了独立的额外临床标注。

因此，这个同测试集多标注者分析仅限于 Ma 等原始关键点子集，并不评估本文新增的冠脉口。

**评价指标。**

我们采用两个关键点检测标准指标。

MRE（Mean Radial Error）是预测坐标和真值坐标之间的欧氏距离（mm），按关键点报告，并在完整关键点集合上求平均。

SDR（Success Detection Rate）是定位误差低于距离阈值 $r$ 的关键点比例。

遵循 Ma 等~\cite{Ma2023_aortic_landmarks}，我们报告 $r\in\{2.0,2.5,3.0,4.0\}$ mm 的 SDR。

除这些标准检测指标外，我们还通过评估真值关键点和预测关键点导出的下游测量一致性，分析预测的临床相关性。

对于多标注者分析，我们在共享的 8 关键点子集上计算相同的关键点级指标，比较 Reader 1 与 Reader 2，并将其与 TRUST 分别相对于两位 reader 的表现进行对照。

### 4.2 实现细节

所有实验用 PyTorch 实现，并在 NVIDIA RTX A6000 图形处理器（GPU；48 GB）上进行。

两个检测阶段都使用 2 级级联 hourglass backbone~\cite{Newell2016_stacked_hourglass}，包含 3D--2D fusion 卷积层（初始 16 个卷积核，instance normalization）。

网络输入为 $128\times128\times128$ 体数据，输出 $32\times32\times32$ 热图，下采样因子为 4。

Gaussian 热图标准差为 $\sigma=\lfloor\log_2 32\rfloor+1=6$ 体素。

强度值被裁剪到 [0.5, 99.5] 百分位范围，并归一化为零均值和单位标准差。

**Stage 1（ROI 提议）。**

遵循 Ma 等~\cite{Ma2023_aortic_landmarks}，原始体数据通过三次样条插值重采样到各向同性间距（数据集中位体素间距），然后下采样到 $128\times128\times128$。

粗检测器用 Adam 训练 500 epoch，学习率为 $10^{-2}$，每 80 epoch 乘以 0.9。

在 30 个测试体数据上，Stage 1 达到总体 MRE $6.74\pm5.12$ mm，与 Ma 等在原始 8 关键点任务中报告的 $6.19\pm3.52$ mm 相当~\cite{Ma2023_aortic_landmarks}。

裁剪 ROI 保留了 300 个真值关键点中的 298 个（99.3%）；剩余 2 个关键点由于 Stage 1 严重离群而被移出裁剪窗口之外，这些残余病例在讨论部分处理。

**Stage 2（TRUST）。**

ROI 从各向同性体数据中围绕 Stage 1 预测中心裁剪为 $128\times128\times128$，训练时增强策略为 80% 随机裁剪和 20% 中心裁剪。

两个学生使用不同随机种子初始化。

我们首先只在 $\mathcal{D}_L$ 上 warm-up $N_{\mathrm{warm}}=50$ 个 epoch，然后使用完整损失再训练 950 个 epoch。

优化器为 Adam，初始学习率为 $10^{-3}$，并采用 step-decay 调度（每 200 epoch 乘以 0.9）。

EMA 衰减率为 $\alpha=0.999$。

每个训练迭代采样一个有标签 mini-batch 和一个无标签 mini-batch。

对于基于 TTA 的不确定性估计，我们使用 $T=5$ 个随机增强，包括随机仿射变换和强度抖动。

损失系数为 $\lambda_c=1.5$ 和 $\lambda_t=0.5$；不确定性灵敏度参数为 $\alpha_u=1.0$ mm$^{-2}$。

在线数据增强包括随机体素强度抖动和 Gaussian blur，二者均以 50% 概率应用。

Topo-GCN 使用 $L=4$ 层，隐藏维度 $d_h=64$。

**模型复杂度。**

每个学生网络包含 69.6M 参数，其中 UG-HCO 和 Topo-GCN 分别只增加 13.5K 和 26.3K 参数。

训练时，TRUST 维护两个 student-teacher 对，即四个网络共 278.4M 参数，其中 139.2M 为可训练参数（两个学生）。

两个 EMA 教师共享同一架构，但不计算梯度。

推理时，两个学生及其 EMA 教师共同参与产生融合预测：每个学生执行一次干净 Stage 2 前向，得到 $p_k^{\mathrm{ref},(j)}$；每个 EMA 教师执行 $T=5$ 个增强前向，得到学生内部 TTA 方差 $u_k^{(j)}$。

将每个教师的 TTA 增强批处理后，完整流水线在 NVIDIA RTX A6000 上处理一个体数据约需 250 ms，约 4 frames per second（FPS）。

### 4.3 与最先进方法比较

我们将 TRUST 与四个全监督关键点检测方法比较，以证明利用未标注数据的半监督学习能够达到或超过全监督表现。

第一，3D U-Net~\cite{Cicek2016_3dunet}：用于热图回归的标准体数据 encoder-decoder 架构。

第二，SCN~\cite{Payer2019_integrating_spatial}：SpatialConfiguration-Net，在热图回归上加入第二阶段空间配置模块，以捕捉关键点间关系。

第三，nnLandmark~\cite{nnLandmark2025}：近期基于 nnU-Net 训练引擎的自配置 3D 关键点检测框架。

第四，Ma et al.~\cite{Ma2023_aortic_landmarks}：原本用于 8 个主动脉瓣关键点的两阶段全监督流水线；我们使用相同架构和训练协议在自己的 10 关键点数据集上重新训练该模型。

所有四个基线都作为 Stage 2 backbone 替代整合到我们的两阶段流水线中，并且只在 $\mathcal{D}_L$ 上训练，从而确保与 TRUST 的唯一差异是检测架构以及是否使用未标注数据。

与半监督基线的比较，即 Mean Teacher~\cite{Tarvainen2017_mean_teacher} 和 CPS~\cite{Chen2021_cps,Ren2024_g2lcps}，在第 4.4 节累计消融的第 2 行和第 3 行报告。

在那里，两个 SSL 方法与 TRUST 共享相同 backbone 和训练流水线，以避免将本文不确定性感知设计的贡献与架构差异混淆。

**表 1 图注翻译。** 在测试集（30 个体数据）上与全监督方法比较。上部为 10 关键点评估；下部为 8 关键点评估（排除冠脉口 $P_8$、$P_9$）。最佳结果以粗体表示。$^\dagger$ 表示来自 Ma 等~\cite{Ma2023_aortic_landmarks} 的结果。所有 SDR 单元格均带有 95% Clopper--Pearson 二项置信区间（将每个病例-关键点试验视作独立计算），半宽最多为 ±5.8 个百分点（10 关键点评估每格 $n=300$ 次试验；8 关键点评估每格 $n=240$ 次试验）；正文引用了部分不重叠区间。

| Method | MRE (mm)↓ | SDR$_{2.0}$ (%) | SDR$_{2.5}$ (%) | SDR$_{3.0}$ (%) | SDR$_{4.0}$ (%) |
|---|---:|---:|---:|---:|---:|
| nnLandmark | 8.62±5.13 | 5.00 | 5.67 | 8.00 | 18.00 |
| 3D U-Net | 4.44±2.91 | 21.00 | 29.33 | 33.67 | 54.00 |
| SCN | 3.12±2.05 | 33.33 | 47.67 | 58.67 | 75.00 |
| Ma et al. | 2.79±1.68 | 39.00 | 51.00 | 64.00 | 80.33 |
| **TRUST (ours)** | **2.19±1.22** | **50.67** | **67.00** | **77.33** | **92.33** |
|  |  |  |  |  |  |
| *8-landmark evaluation (excluding $P_8$, $P_9$)* |  |  |  |  |  |
| Ma et al.$^\dagger$ (8 pts) | 2.23±1.35 | 47.90 | 65.00 | 77.50 | 89.17 |
| **TRUST (ours, 8 pts)** | **2.07±0.98** | **57.08** | **68.33** | **82.92** | **96.25** |

表 1 总结了结果。

在我们 10 关键点任务上训练的全监督基线中，SCN 表现最好（MRE 3.12±2.05 mm），这得益于其捕捉关键点间关系的第二阶段空间配置模块。

缺乏任何结构推理的 3D U-Net 产生显著更高的 MRE，为 4.44 mm。

nnLandmark 在该任务上表现较差（MRE 8.62 mm），可能因为其自配置流水线为通用关键点检测设计，不能很好适应主动脉根部这种所有 10 个关键点都位于小空间范围内的紧凑密集解剖。

TRUST 达到 2.19±1.22 mm 的 MRE 和 50.67% 的 SDR$_{2.0}$（95% CI [44.9, 56.5]），尽管使用相同有标签训练集，仍优于所有全监督基线。

与 Ma 等~\cite{Ma2023_aortic_landmarks} 相比，TRUST 将 MRE 降低 21.5%（2.79→2.19 mm），并将 SDR$_{2.0}$ 从 39.00%（95% CI [33.4, 44.8]）提高到 50.67%（+11.67 个百分点；区间不重叠）。

Ma 等是最强监督基线，且与 TRUST 共享相同 backbone 架构和训练协议，因此该结果证明通过不确定性感知半监督学习利用 620 个无标签体数据，相比纯监督方法带来了显著收益。

相比最强替代架构 SCN，改进更大：MRE 降低 29.8%（3.12→2.19 mm），SDR$_{2.0}$ 增加 17.34 个百分点。

较低标准差（1.22 vs Ma et al. 的 1.68 mm）进一步说明 TRUST 在不同关键点和患者上产生更一致的预测。

需要说明的是，21.5% 降低是相对未利用无标签池的全监督基线计算的。

与相同 backbone、相同数据预算的 vanilla CPS 配置相比（表 2 第 3 行，MRE 2.55 mm），对应增益为 14.1%；这将本文不确定性感知设计的贡献从 backbone 选择和 SSL 范式本身中分离出来。

为了与 Ma 等~\cite{Ma2023_aortic_landmarks} 原始报告结果直接比较，我们还在原始 8 关键点子集上评估 TRUST，即排除冠脉口 $P_8$ 和 $P_9$。

如表 1 下部所示，TRUST 在 8 个关键点上达到 2.07±0.98 mm 的 MRE，相对 Ma 等原始结果（2.23±1.35 mm）降低 7.2%，同时将 SDR$_{2.0}$ 从 47.90%（95% CI [41.4, 54.4]）提高到 57.08%（95% CI [50.6, 63.4]），并将 SDR$_{4.0}$ 从 89.17%（95% CI [84.5, 92.8]）提高到 96.25%（95% CI [93.0, 98.3]）。

由于该评估使用与 Ma 等相同的 8 关键点定义和测试集划分，因此该比较可以直接解释；不过 SDR$_{2.0}$ 区间略有重叠，说明最有力的证据来自更低的 MRE、更高的 SDR$_{4.0}$ 以及更低标准差所共同呈现的趋势。

较低标准差（0.98 vs 1.35 mm）进一步说明 TRUST 在不同关键点和患者上产生更一致的预测。

### 4.4 消融研究

为隔离每个组件贡献，我们从 supervised-only 基线出发，进行累计消融，逐步添加模块。

该顺序旨在先引入 vanilla 架构组件（HCO、Topo-GCN），然后再激活条件化所有三个模块的统一不确定性机制，从而把“不确定性作为协调控制信号”的贡献与底层架构块本身的贡献分离开。

表 3 中的消费者子消融进一步拆解了不确定性对三个消费者的贡献。

所有变体使用相同 backbone 和训练计划；只有半监督组件不同。

表 2 报告测试集结果。

第一，Supervised-only：单学生网络只在 $\mathcal{D}_L$ 上训练，不使用任何无标签数据，建立下界。

第二，+ Mean Teacher：单 student-teacher 对在 $\mathcal{D}_U$ 上使用基于 EMA 的一致性正则~\cite{Tarvainen2017_mean_teacher}。

第三，+ CPS：双 student-teacher 对在热图上使用 CPS~\cite{Chen2021_cps,Ren2024_g2lcps}，替代单网络 Mean Teacher。

第四，+ HCO（vanilla）：将热传导特征传播~\cite{wang2025building} 集成到两个学生中，扩散率只依赖频率，不进行不确定性条件化。

第五，+ Topo-GCN（vanilla）：在上方加入解剖图精修头，消息传递是均匀的，不使用不确定性门控。

第六，+ Unified Uncertainty（完整 TRUST）：同一个基于 TTA 的不确定性估计同时加权 CPS 损失、条件化 HCO 扩散率（UG-HCO），并门控 Topo-GCN 消息传递。

**表 2 图注翻译。** 测试集上的累计消融研究（10 个关键点）。每一行在前一配置基础上增加一个组件。$\Delta$MRE 是相对前一行的增量改善。

| Configuration | MRE (mm)↓ | SDR$_{2.0}$ | SDR$_{4.0}$ | $\Delta$MRE |
|---|---:|---:|---:|---:|
| Supervised-only | 2.79±1.68 | 39.00 | 80.33 | --- |
| + Mean Teacher | 2.65±1.58 | 43.00 | 83.00 | -0.14 |
| + CPS | 2.55±1.50 | 45.67 | 84.67 | -0.10 |
| + HCO (vanilla) | 2.49±1.54 | 47.33 | 85.67 | -0.06 |
| + Topo-GCN (vanilla) | 2.46±1.58 | 48.00 | 86.00 | -0.03 |
| **+ Unified Uncertainty** | **2.19±1.22** | **50.67** | **92.33** | **-0.27** |

表 2 揭示两个关键观察。

第一，SSL 组件（Mean Teacher 和 CPS）在早期阶段贡献最大的个体收益，分别为 -0.14 mm 和 -0.10 mm；这确认了在强监督基线已经存在时，利用无标签数据是改进的主要驱动。

第二，不带不确定性的 vanilla 架构添加项，即 HCO 和 Topo-GCN，只带来边际增量改善（-0.06 mm 和 -0.03 mm），甚至各自使标准差略微增加（+0.04 mm）。

这提示，仅仅堆叠复杂的特征传播和图精修模块是不够的：没有可靠性信号时，这些模块无法选择性关注可信证据，甚至可能放大不可靠伪标签的噪声。

决定性跳跃来自最后一行激活统一不确定性机制（-0.27 mm），同时将标准差减少 0.36 mm，这是所有配置中最大的方差降低。

这一单一添加项的 MRE 贡献超过 vanilla HCO 和 vanilla Topo-GCN 合计贡献的三倍，并逆转了它们对预测方差的不利影响。

该结果确认了 TRUST 的中心设计原则：同一不确定性信号必须协同地条件化特征提取、门控图消息传递并加权伪监督，才能充分释放底层架构组件的潜力。

**不确定性最重要的位置在哪里？**

表 2 中的主消融显示，跨全部三个消费者统一使用不确定性产生最大的单步改善，但它并不揭示这种收益是否主要由一个消费者驱动，还是由协调使用驱动。

为分离每个不确定性消费者的贡献，我们从表 2 中的“Topo-GCN (vanilla)”配置出发，即 HCO、Topo-GCN 和 CPS 都存在但没有任何组件接收不确定性信号。

随后，我们在两个互补子集中逐步激活不确定性消费者。

表 3 报告结果。

**表 3 图注翻译。** 不确定性消费者子消融。从 no-uncertainty 配置（表 2 第 5 行）开始，我们选择性地在不同消费者子集中激活不确定性信号：CPS 损失加权、UG-HCO 扩散率和 Topo-GCN 消息传递门控，或三者同时激活。Unc. 表示 uncertainty，std 表示 standard deviation；$\Delta$MRE 和 $\Delta$std 都相对于 no-uncertainty 配置计算。

| Configuration | MRE (mm)↓ | SDR$_{2.0}$ | SDR$_{4.0}$ | $\Delta$MRE | $\Delta$std |
|---|---:|---:|---:|---:|---:|
| No uncertainty | 2.46±1.58 | 48.00 | 86.00 | --- | --- |
| Unc.→CPS only | 2.41±1.52 | 49.33 | 87.00 | -0.05 | -0.06 |
| Unc.→HCO+GCN | 2.35±1.55 | 51.00 | 88.00 | -0.11 | -0.03 |
| **Unc.→All three** | **2.19±1.22** | **50.67** | **92.33** | **-0.27** | **-0.36** |

两个发现很突出。

第一，在任何消费者子集中激活不确定性都已经优于 no-uncertainty 基线，但幅度差异明显：只加权 CPS 损失产生较小的 -0.05 mm 收益，而将同一不确定性输入 UG-HCO 和 Topo-GCN 产生超过两倍的收益（-0.11 mm）。

这种不对称说明，在我们的框架中，不确定性作为特征层控制信号更有价值；它重塑每个学生看到的内容以及聚合结构上下文的方式，而不仅仅是作为伪标签的监督层过滤器。

第二，完整配置（-0.27 mm）显著超过两个子集之和（-0.05 + -0.11 = -0.16 mm），并同时取得最大方差降低（标准差 -0.36 mm）。

三个消费者之间的互补效应以及同步方差压缩共同支持“估计一次、消费三次”的设计：同一个不确定性估计必须流经全部三个消费者，框架才能达到完整潜力。

**不确定性来源分解。**

为验证全方差分解公式，我们训练三个 TRUST 变体，它们唯一差异是由 $u_k$ 的哪个分量驱动三个消费者：学生内部 TTA 方差、学生间分歧、或池化总方差（默认）。

表 4 在报告检测性能的同时，报告每个不确定性来源与定位误差之间的逐病例 Pearson 相关。

**表 4 图注翻译。** 不确定性来源子消融。每一行使用全方差分解中的不同分量作为三个消费者的不确定性信号。$r$：不确定性与定位误差之间的逐病例 Pearson 相关（$N=300$ 个病例-关键点对）。$\bar{u}$：测试集上的平均不确定性值。

| Source | MRE (mm)↓ | SDR$_{2.0}$ | SDR$_{4.0}$ | $r$ | $\bar{u}$ |
|---|---:|---:|---:|---:|---:|
| (a) Within only | 2.36±1.62 | 51.6 | 87.9 | 0.154 | 0.006 |
| (b) Between only | 2.34±1.48 | 49.5 | 91.1 | 0.278 | 1.343 |
| **(c) Pooled** | **2.19±1.22** | **50.7** | **92.3** | 0.243 | 0.356 |

出现三个发现。

第一，学生内部 TTA 方差产生接近零的不确定性值（$\bar{u}=0.006$）和较弱逐病例相关（$r=0.154$），确认了第 3.3 节描述的 soft-argmax 饱和现象：收敛热图 logits 足够大，使 TTA 扰动几乎无信息。

第二，学生间分歧提供明显更强的信号（$\bar{u}=1.343$，$r=0.278$），验证其作为池化分解中主导不确定性来源的作用。

第三，尽管池化不确定性与误差的逐病例相关低于 between-only 变体（$r=0.243$ vs 0.278），它仍取得最佳检测性能（MRE 2.19 mm）和最低标准差（1.22 mm）。

这说明学生内部分量虽然作为独立信号几乎无信息，但在与学生间项结合时提供互补正则化效果，使训练期间预测更稳定。

**标签效率。**

为了评估 TRUST 的收益如何随可用有标签数据量缩放，我们使用 $|\mathcal{D}_L|\in\{20,50,100\}$ 个体数据的有标签子集，重新训练监督基线（Ma et al.~\cite{Ma2023_aortic_landmarks}）和完整 TRUST 框架，同时保持 TRUST 的无标签池固定为 $|\mathcal{D}_U|=620$。

表 5 报告结果。

**表 5 图注翻译。** 标签效率。TRUST 在三个有标签体数据数量下与监督基线（Ma et al.）比较，无标签池固定为 620 个体数据。$\Delta$MRE 是相对相同 $|\mathcal{D}_L|$ 下监督基线的绝对 MRE 降低。

| Method | $|\mathcal{D}_L|$ | MRE (mm)↓ | SDR$_{2.0}$ | SDR$_{4.0}$ | $\Delta$MRE |
|---|---:|---:|---:|---:|---:|
| Supervised | 20 | 4.16 | 13.33 | 55.00 | --- |
| TRUST | 20 | 3.86 | 18.33 | 62.33 | -0.30 |
| Supervised | 50 | 3.55 | 23.67 | 66.33 | --- |
| TRUST | 50 | 3.34 | 28.67 | 66.33 | -0.21 |
| Supervised | 100 | 2.79 | 39.00 | 80.33 | --- |
| **TRUST** | **100** | **2.19** | **50.67** | **92.33** | **-0.60** |

TRUST 在每个测试标签量下都优于监督基线，确认了框架的一般收益。

然而，相对收益并非单调：$|\mathcal{D}_L|=20$ 时为 -0.30 mm，$|\mathcal{D}_L|=50$ 时为 -0.21 mm，$|\mathcal{D}_L|=100$ 时为 -0.60 mm。

这与传统半监督分割叙事形成对比；后者通常认为相对收益在最低标签量时最高~\cite{Yu2019_uncertainty_aware,Wang2022_u2pl}。

我们将这一模式归因于两个范式的不同行为缩放。

从 20 到 50 个标签时，监督性能快速提升，因为模型脱离严重数据饥饿状态，从而压缩了 SSL 的相对提升空间。

从 50 到 100 个标签时，监督收益减弱，而 TRUST 的不确定性引导模块需要最低限度的伪标签质量才能有效，因此完全激活并产生最大绝对收益。

一个促成因素是，统一不确定性中的跨学生分歧分量需要足够模型多样性才能提供有意义的可靠性信号。

在极低标签数量下，两个学生会收敛到相似且较差的解，从而降低跨学生分解的有效性。

这些观察表明，TRUST 的多组件设计在中等标注监督预算下最有利，因为此时有足够标注来引导可靠伪标签和不确定性估计。

未来面向极低标签 regime 的工作应探索单教师替代方案，或在 CPS 之外加入辅助的标签高效先验。

**无标签池规模。**

为了直接评估半监督收益是否随可用无标签池规模变化，我们固定有标签集为 $|\mathcal{D}_L|=100$，并改变 $|\mathcal{D}_U|\in\{0,100,300,620\}$。

其中 $|\mathcal{D}_U|=0$ 对应不使用任何无标签训练的监督 counterpart。

表 6 显示，随着使用更多无标签体数据，性能单调改善：100 个无标签体数据时收益较小（MRE 2.79→2.74 mm），300 个体数据时改善更明显（2.55 mm），620 个体数据时达到完整 TRUST 结果（2.19 mm，相比 $|\mathcal{D}_U|=0$ 降低 21.5%）。

SDR$_{2.0}$ 和 SDR$_{4.0}$ 呈现相同趋势；从 0 到 620 个无标签体数据分别提高 11.67 和 12.00 个百分点。

这些结果支持无标签池本身，而不仅仅是架构组件，对 TRUST 的最终性能具有实质贡献。

**表 6 图注翻译。** 固定有标签集为 $|\mathcal{D}_L|=100$ 时，无标签池规模的影响。$\Delta$MRE 是相对 $|\mathcal{D}_U|=0$ 的监督 counterpart 的绝对 MRE 变化。

| $|\mathcal{D}_U|$ | MRE (mm)↓ | SDR$_{2.0}$ | SDR$_{4.0}$ | $\Delta$MRE |
|---:|---:|---:|---:|---:|
| 0 | 2.79 | 39.00 | 80.33 | --- |
| 100 | 2.74 | 40.33 | 81.67 | -0.05 |
| 300 | 2.55 | 44.67 | 86.00 | -0.24 |
| **620** | **2.19** | **50.67** | **92.33** | **-0.60** |

### 4.5 逐关键点与临床分析

**逐关键点检测精度。**

表 7 报告 TRUST 和 supervised-only 基线的逐关键点 MRE，并按照图 1 已定义的临床功能进行分组：瓣环铰链点、连合点、中心点、膜部室间隔点和冠脉口。

**表 7 图注翻译。** 按临床功能分组的逐关键点 MRE（mm）。NCC、LCC 和 RCC 分别表示无冠瓣、左冠瓣和右冠瓣；LCO 和 RCO 分别表示左、右冠脉口。

| Group | Landmark | Supervised | TRUST |
|---|---|---:|---:|
| Annular | $P_0$ (NCC Hinge) | 2.95±1.61 | 2.41±1.24 |
| Annular | $P_1$ (RCC Hinge) | 2.61±1.45 | 2.47±1.14 |
| Annular | $P_2$ (LCC Hinge) | 2.20±1.15 | 1.87±0.86 |
| Commissure | $P_3$ (NCC--RCC) | 2.74±1.43 | 2.02±1.10 |
| Commissure | $P_4$ (RCC--LCC) | 2.62±1.84 | 2.19±0.94 |
| Commissure | $P_5$ (LCC--NCC) | 2.91±1.52 | 1.84±0.89 |
| Center | $P_6$ (Central) | 1.91±0.81 | 1.58±0.64 |
| Septum | $P_7$ (Membranous septum) | 3.64±1.66 | 3.29±1.88 |
| Coronary | $P_8$ (LCO) | 2.73±1.71 | 2.17±1.29 |
| Coronary | $P_9$ (RCO) | 3.57±2.37 | 2.10±0.94 |
| **Overall** |  | 2.79±1.68 | 2.19±1.22 |

**图 3 图注翻译。** supervised baseline 和 TRUST 的逐关键点定位误差（MRE ± standard deviation）。虚线水平线表示每种方法的总体 MRE。关键点按临床功能分组（阴影背景）。相对 supervised baseline，TRUST 下全部 10 个关键点均改善。

表 7、图 3 和图 4 的定性例子显示若干模式。

第一，TRUST 在恰好激发其设计动机的关键点上取得最大收益：膜部室间隔点 $P_7$ 改善 9.6%（3.64→3.29 mm），右冠脉口 $P_9$ 改善 41.2%（3.57→2.10 mm）。

连合点（$P_3$--$P_5$）均匀降低 16% 到 37%，反映 Topo-GCN 有效利用了瓣环中强铰链-连合交替结构。

中心点 $P_6$ 在监督下已经是最容易的关键点，进一步改善到 1.58 mm，并仍然是该集合中误差最低的关键点。

第二，本文引入且在 Ma 等~\cite{Ma2023_aortic_landmarks} 等既有监督流水线中缺失的两个冠脉口，在 TRUST 中均低于 2.20 mm。

这种逐关键点精度是在 SCCT 推荐的 12 mm 参考截断值上进行冠脉高度筛查的必要前提。

不过阈值层面分析（表 9）显示，当前队列只包含少量低于阈值的冠脉口（LCO 有 8 例，RCO 只有 1 例），因此决定性的逐冠脉口灵敏度评估需要低冠脉高度解剖富集的外部队列。

因此，我们将从 8 个到 10 个关键点的扩展表述为朝向自动化冠脉高度测量的一步，而不是已经完成临床冠脉风险分层证明。

第三，相对监督基线，全部 10 个关键点在 TRUST 下均改善，逐关键点收益范围从 -5.4%（$P_1$ RCC 铰链，最小）到 -41.2%（$P_9$ RCO，最大）。

这种均匀改善，加上总体标准差从 1.68 降低到 1.22 mm，说明多组件设计在完整关键点集合上产生一致收益，而不是用某一子集换取另一子集的表现。

**图 4 图注翻译。** 三个代表性测试病例的定性关键点检测结果，按总体 MRE 递增排列。绿色圆圈：真值关键点；红色方块：TRUST 预测；星号：膜部室间隔点 $P_7$；菱形：冠脉口 $P_8$/$P_9$。黄色虚线表示 $P_7$ 误差向量。每个病例均显示于瓣环平面水平的轴位视图。Case A（MRE 1.46 mm，$P_7$ 误差 1.5 mm）显示所有关键点几乎完美重叠。Case B（MRE 2.13 mm，$P_7$ 误差 2.5 mm）接近队列均值，并表现出轻度残余偏差。Case C（MRE 2.59 mm，$P_7$ 误差 3.8 mm）是测试集中 MRE 最高的病例；误差分布于多个关键点，而不是集中在单个关键点上。

**临床下游验证。**

除逐关键点几何精度外，我们评估预测关键点是否保留六项有临床意义的测量，这些测量按照三个主要 TAVI 规划支柱组织。

支柱 I，即传导异常风险预测，包括膜部室间隔长度（MSL）和逐瓣叶 device-landing-zone（DLZ）钙化体积；这一组织方式对应 Lemarchand 等~\cite{Lemarchand2023_nocd,Lemarchand2026_cusp_overlap} 提出的术前 CT 框架，该框架将患者特异性 MSL 和主动脉瓣复合体钙化识别为 TAVI 后传导结局的重要决定因素。

支柱 II，即冠脉安全评估，包括冠脉口高度这一用于冠脉阻塞评估的筛查性测量。

支柱 III，即器械选择和操作释放，包括 3 瓣叶瓣环直径、用于假体尺寸选择的瓣环横截面积，以及用于术中瓣膜释放的 C 臂透视角度。

对每项测量，我们从相应关键点子集分别计算真值和预测值，并报告两层一致性。

第一层是连续一致性，包括 MAE、Pearson 相关、ICC 和 Bland-Altman limits（表 8）。

第二层是在有文献报告阈值时进行阈值不一致分析；这些阈值在本文中只作为模型评价探针，而不是临床决策规则（表 9）。

**膜部室间隔长度（MSL）。**

MSL 是膜部室间隔点 $P_7$ 到通过三个铰链点 $P_0$、$P_1$、$P_2$ 拟合的瓣环平面的垂直距离：

$$
\text{MSL} = |(p_7-p_0)\cdot\hat{n}|,
\quad
\hat{n} =
\frac{(p_1-p_0)\times(p_2-p_0)}
{\|(p_1-p_0)\times(p_2-p_0)\|}.
$$

较短 MSL 是 TAVI 后新发传导异常的独立预测因子~\cite{Hamdan2015_msl_avblock,Maeno2017_pacemaker_tavr}。

根据 Jilaihawi 等~\cite{Jilaihawi2019_msl} 报告、并由 Lemarchand 等~\cite{Lemarchand2023_nocd} 讨论的截断点，我们将 2 mm 和 5 mm 阈值用作分析性的 MSL 层级，而不是独立的临床风险类别：短 MSL（MSL $\leq2$ mm）、中间 MSL（$2<\text{MSL}\leq5$ mm）和长 MSL（MSL $>5$ mm）。

由于中间层级仅跨越 3 mm，且与典型关键点检测误差相当，我们额外报告极端层级偏移的比例，即短 MSL 与长 MSL 之间的相互变化；这用于评价定位误差是否会让阈值解释跨越超过一个相邻层级。

**逐瓣叶 device-landing-zone 钙化。**

作为第二个主要术前 CT 衍生 NOCD 预测因子~\cite{Lemarchand2023_nocd}，我们量化每个瓣叶 device-landing zone（DLZ）内的钙化体积。

DLZ 定义为沿瓣环平面反向法向 $-\hat{n}$ 向下延伸 $d_{\text{DLZ}}=5$ mm 的 slab，并由从瓣环中心 $C_{\text{ann}}=(P_0+P_1+P_2)/3$ 出发的径向边界 $R_{\max}=1.2R_{\triangle}$ 限定。

其中 $R_{\triangle}$ 表示铰链三角形 $P_0P_1P_2$ 的外接圆半径。

在该 slab 内，连合点 $P_3$、$P_4$、$P_5$ 的平面内投影将瓣环分割为三个角向扇区，每个扇区恰好包含一个铰链点，并对应一个瓣叶：NCC（包含 $P_0$ 的扇区）、RCC（包含 $P_1$ 的扇区）和 LCC（包含 $P_2$ 的扇区）。

逐瓣叶钙化体积定义为：

$$
V_k =
\sum_{\mathbf{p}\in\Omega_k}
\mathbb{I}\{\rho(\mathbf{p})>\tau_{\mathrm{HU}}\}\,v_{\mathrm{vox}},
\quad
k\in\{\text{NCC, RCC, LCC}\}.
$$

其中 $\Omega_k$ 是瓣叶 $k$ 的 DLZ 楔形区域，$\rho(\mathbf{p})$ 是体素 $\mathbf{p}$ 的 Hounsfield unit（HU）值，$\tau_{\mathrm{HU}}=800$ HU 是增强计算机断层血管成像（CTA）的标准钙化阈值，$v_{\mathrm{vox}}$ 是体素体积。

该分解遵循 Maeno 等~\cite{Maeno2017_pacemaker_tavr}，他们确定 NCC-DLZ 钙化是起搏器植入的独立预测因子；也遵循 Mauri 等~\cite{Mauri2016_lvot_calcification}，他们识别左、右冠瓣下方左室流出道钙化是 NOCD 的独立预测因子。

由于 Mauri 等报告的截断值（LCC > 13.7 mm$^3$，RCC > 4.8 mm$^3$）来自不同分割约定，不能直接转移到我们的协议，因此我们只报告连续一致性。

**冠脉口高度。**

冠脉高度是每个冠脉口（$P_8$ 为 LCO，$P_9$ 为 RCO）到瓣环平面的垂直距离，沿 MSL 公式中的平面法向 $\hat{n}$ 计算：

$$
h_{\text{LCO}} = |(p_8-p_0)\cdot\hat{n}|,
\quad
h_{\text{RCO}} = |(p_9-p_0)\cdot\hat{n}|.
$$

根据 Society of Cardiovascular Computed Tomography（SCCT）2019 专家共识~\cite{Blanke2019_scct_tavi_ct}，冠脉口高度低于 12 mm（同时 sinus of Valsalva 直径低于 30 mm）提示冠脉阻塞风险升高；冠脉阻塞罕见（约 0.66%），但常常致命，报告 30 天死亡率最高可达 40.9%。

由于冠脉阻塞风险依赖多个解剖因素，且没有任何绝对高度阈值会禁忌手术~\cite{Blanke2019_scct_tavi_ct}，我们只将 12 mm 截断值作为模型评价的参考阈值，将每个冠脉口标记为低于阈值（$h<12$ mm）或高于阈值（$h\geq12$ mm）。

由于假阴性阈值跨越，即真值低于阈值但预测高于阈值，对筛查性测量并不理想，我们报告对低于阈值冠脉口的 sensitivity。

**用于假体尺寸选择的瓣环直径。**

临床假体尺寸选择通常依赖通过完整瓣环轮廓描绘获得的瓣环面积或周长~\cite{Blanke2019_scct_tavi_ct}，这需要沿瓣环环周 8 到 12 个点，不能仅由三个铰链点推导。

我们将瓣环直径近似为三个铰链点外接圆的直径：

$$
D_{\text{circ}} = 2R_{\triangle},
$$

其中 $R_{\triangle}$ 是前文引入的铰链三角形外接圆半径。

在圆形瓣环的简化假设下，$D_{\text{circ}}$ 等价于两种主流 TAVI 尺寸选择约定所用的面积推导直径（$2\sqrt{A/\pi}$）和周长推导直径（$C/\pi$）：Edwards SAPIEN（基于面积）和 Medtronic Evolut（基于周长）。

外接圆公式还使 $D_{\text{circ}}$ 位于临床典型瓣环直径范围（20--30 mm）内，从而可以与假体尺寸分桶进行有意义比较。

我们将 $D_{\text{circ}}$ 与两家制造商共同的 23、26 和 29 mm 尺寸分桶比较，并报告 TRUST 导出和真值导出的 $D_{\text{circ}}$ 落入不同分桶的病例比例。

另一种使用铰链点平均成对距离的公式~\cite{Lalys2019_tavi_segmentation} 作为方法比较在讨论部分提及。

**C 臂透视角度。**

在与瓣环平面平行的一族 C 臂角度中（使三个铰链点投影到同一直线上），TAVI 释放需要一个特殊角度，使 LCC 和 RCC 铰链进一步相互重叠，即 cusp-overlap 投影。

该投影最初由 Tang 等~\cite{Tang2018_cusp_overlap_intro} 描述用于自膨式经导管主动脉瓣置换；随后在当代临床队列中得到评估~\cite{Pascual2022_cusp_overlap,Lemarchand2026_cusp_overlap}，并可帮助相对于患者特异性 MSL 实现更高位释放。

因此，本文将 cusp-overlap 角度视为操作规划测量，而不是传导结局的直接预测因子。

cusp-overlap 射线方向必须满足两个条件：一是位于瓣环平面内，二是垂直于 LCC--RCC 轴 $P_2-P_1$：

$$
\hat{d} =
\frac{\hat{n}\times(P_2-P_1)}
{\|\hat{n}\times(P_2-P_1)\|},
$$

并定向为投影指向前方。

遵循标准 CT 到透视映射~\cite{Achenbach2013_annular_plane,Arnold2012_carm_projection}，对应角度为：

$$
\theta_{\text{LAO/RAO}}
=
\operatorname{atan2}(\hat{d}_x,\hat{d}_y),
$$

$$
\theta_{\text{CRA/CAU}}
=
\operatorname{atan2}\left(\hat{d}_z,\sqrt{\hat{d}_x^2+\hat{d}_y^2}\right).
$$

其中 LAO/RAO 表示左/右前斜角度，CRA/CAU 表示头侧/足侧角度；正的 $\theta_{\text{LAO/RAO}}$ 表示 LAO，正的 $\theta_{\text{CRA/CAU}}$ 表示头侧角。

术前准确预测 cusp-overlap 角度可以减少术中对比剂使用和主动脉造影调整，这对肾功能受损患者有重要临床价值~\cite{Gurvitch2010_tavi_angulation}。

由于 C 臂角度没有已建立的离散临床阈值，我们只报告两个角度的连续一致性。

**表 8 图注翻译。** TRUST 预测和真值导出的临床测量之间的连续一致性。MAE：mean absolute error；std：standard deviation；$r$：Pearson 相关；ICC(2,1)：intraclass correlation coefficient（双向随机、单次测量）；95% LoA：Bland--Altman limits of agreement。

| Measurement | MAE±std | $r$ | ICC | 95% LoA |
|---|---:|---:|---:|---|
| MSL (mm) | 1.72±1.54 | 0.33 | 0.32 | [-4.70, +4.41] |
| DLZ calc. NCC (mm$^3$) | 5.5±18.1 | 0.999 | 0.986 | [-31.8, +40.5] |
| DLZ calc. LCC (mm$^3$) | 7.8±21.0 | 0.998 | 0.988 | [-36.6, +48.1] |
| DLZ calc. RCC (mm$^3$) | 9.9±20.7 | 0.986 | 0.986 | [-43.9, +46.2] |
| Coronary h. LCO (mm) | 1.71±1.55 | 0.52 | 0.48 | [-5.08, +3.43] |
| Coronary h. RCO (mm) | 1.67±1.25 | 0.71 | 0.71 | [-4.11, +4.15] |
| Annular $D_{\text{circ}}$ (mm) | 0.83±0.55 | 0.83 | 0.83 | [-1.70, +2.15] |
| Annular area approx. (mm$^2$) | 43.1±28.0 | 0.87 | 0.71 | [-102.2, +20.3] |
| C-arm LAO/RAO (°) | 6.7±5.0 | 0.79 | 0.71 | [-8.3, +18.3] |
| C-arm CRA/CAU (°) | 4.8±3.2 | 0.72 | 0.68 | [-10.1, +12.4] |

**表 9 图注翻译。** 文献报告截断值下的阈值评价；这些截断值在本文中作为模型评价探针，而不是临床决策规则。对于瓣环 circumdiameter，分桶对应标准 TAVI 假体尺寸（Edwards SAPIEN / Medtronic Evolut）。Sens. 表示对低于阈值病例的 sensitivity；C 臂角度缺乏离散临床阈值，因此省略。

| Measurement | Threshold(s) | Discordance / key metric |
|---|---|---|
| MSL | 2 / 5 mm strata | 46.7%; 0% extreme |
| Coronary h. LCO | 12 mm | 23.3%; Sens. 75.0% |
| Coronary h. RCO | 12 mm | 6.7%; Sens. 0.0% (n$_+$=1) |
| Annular $D_{\text{circ}}$ | 23 / 26 / 29 mm bins | 23.3%; 0 cases $|\Delta|\geq2$ |

**MSL 结果与解释。**

预测 MSL 和真值 MSL 之间的连续一致性（表 8）显示低平均误差但弱排序相关：MAE 为 1.72±1.54 mm，Pearson $r=0.33$，ICC(2,1)=0.32，95% Bland--Altman limits 为 [-4.70, +4.41] mm，并且系统偏差接近零，为 -0.14 mm。

在 Jilaihawi/Lemarchand 的 2 mm 和 5 mm MSL 截断点~\cite{Jilaihawi2019_msl,Lemarchand2023_nocd} 下，46.7% 病例进入与真值不同的 MSL 阈值层级（表 9，95% Clopper--Pearson CI [28.3, 65.7]）。

短 MSL 与长 MSL 两个极端层级之间出现零极端层级偏移（0/30，95% CI [0.0, 11.6]）。

全部 14 个不一致病例都是单层级偏移；方向大致平衡：8 例向更短 MSL 层级偏移，其中 7 例为 long→intermediate，1 例为 intermediate→short；6 例向更长 MSL 层级偏移，其中 5 例为 intermediate→long，1 例为 short→intermediate。

这一点与接近零的系统偏差（-0.14 mm）一致。

中间层级跨度为 3 mm，与典型关键点定位误差相当，因此靠近任一阈值边界的病例在小扰动下可能移动到相邻层级。

相反，短 MSL 与长 MSL 之间的极端跳变需要 MSL 误差超过 3 mm，而 30 个测试病例中没有观察到。

**逐瓣叶 DLZ 钙化结果与解释。**

逐瓣叶 DLZ 钙化体积表现出本文所有下游测量中最强的一致性（表 8）。

三瓣叶 Pearson 相关均超过 0.98：NCC $r=0.999$，LCC $r=0.998$，RCC $r=0.986$，全部 $p<0.001$。

对应 ICC(2,1) 分别为 0.986、0.988 和 0.986。

MAE 范围从 NCC 的 5.5 mm$^3$ 到 RCC 的 9.9 mm$^3$，相对于宽广的真值体积分布较小：NCC 范围 0--478 mm$^3$，LCC 0--643 mm$^3$，RCC 0--679 mm$^3$，总量范围 0--1638 mm$^3$。

这种鲁棒性反映了主动脉根部钙化的局灶特征：小的连合定位误差会转化为瓣叶楔形区域的小角度边界移动，但由于钙化核在空间上稀疏且高度集中，多数边界扰动扫过非钙化区域，使积分体积几乎不变。

三个瓣叶共享同一偏差方向：NCC +4.34 mm$^3$，LCC +5.76 mm$^3$，RCC +1.16 mm$^3$，均为正值。

这提示瓣叶楔形体积有轻度过预测；跨测量偏差特征在讨论部分联合分析。

结合 MSL 分析，逐瓣叶 DLZ 钙化定量提供了 Lemarchand NOCD 风险框架~\cite{Lemarchand2023_nocd} 的完整术前 CT 实例化，并且两个支柱都显示 TRUST 导出的关键点支持标准术前 NOCD 风险评估流程。

**冠脉口高度结果与解释。**

表 8 显示 LCO 和 RCO 高度具有可比的连续一致性，MAE 分别为 1.71 mm 和 1.67 mm，二者都接近表 7 中 $P_8$ 和 $P_9$ 的底层逐关键点误差。

二者 ICC 分别为 0.48 和 0.71。

值得注意的是，RCO 表现出近零系统偏差（+0.02 mm）和对称的 95% LoA，说明由 $P_0$--$P_2$ 拟合的瓣环平面相对于 RCO 一侧配准良好。

LCO 表现出轻度系统性低估（-0.83 mm）；对于筛查性测量而言，这意味着 TRUST 倾向于将 LCO 放置得比真值更接近瓣环平面。

这种偏差会使阈值筛查偏向提高警惕，而不是漏掉低于阈值的测量。

在 SCCT 推荐的 12 mm 参考截断值~\cite{Blanke2019_scct_tavi_ct} 下，两个冠脉口由于低于阈值的病例数不同，表现出不同统计 regime。

测试集中 LCO 包含 8 个低于阈值病例（真值高度 < 12 mm），其中 TRUST 正确标记 6 个，灵敏度为 75.0%，95% Clopper--Pearson CI 为 [34.9, 96.8]；特异度为 77.3%，CI 为 [54.6, 92.2]；总体不一致率为 23.3%，CI 为 [9.9, 42.3]。

当前队列中 RCO 只有 1 个低于阈值病例，TRUST 漏检该病例；灵敏度为 0.0%，CI 为 [0.0, 97.5]；特异度为 96.6%，CI 为 [82.2, 99.9]；总体不一致率为 6.7%，CI 为 [0.8, 22.1]。

单阳性病例约束使 RCO 灵敏度估计在统计上无信息（CI 实质上为 [0,1]），因此该队列中逐冠脉口灵敏度比较没有意义；我们仅为完整性报告它。

两侧冠脉共观察到 3 个假阴性阈值跨越，全部病例均位于 12 mm 截断值 1 mm 以内。

LCO 两例为真值 11.66 mm、预测 12.65 mm，以及真值 11.27 mm、预测 14.83 mm；RCO 一例为真值 11.98 mm、预测 14.56 mm。

因此，所有漏检的低于阈值病例都反映关键点分辨率受限的边界跨越现象，而不是真正检测失败。

Bland--Altman 95% limits（LCO 约 ±5 mm，RCO 约 ±4 mm）意味着任何冠脉高度位于 12 mm 截断值 ±5 mm 内的病例都应谨慎解释。

需要更大的、低于阈值病例更多的外部队列，才能确立两侧冠脉口的决定性阈值灵敏度。

**瓣环直径结果与解释。**

在所有下游测量中，瓣环 circumdiameter $D_{\text{circ}}$ 显示最强连续一致性：MAE 0.83±0.55 mm，Pearson $r=0.83$，ICC(2,1)=0.83，95% LoA 为 [-1.70, +2.15] mm。

这是符合预期的：$D_{\text{circ}}$ 只依赖三个铰链点 $P_0$--$P_2$，它们在 supervised baseline 和 TRUST 中都是定位最好的关键点之一（表 7）。

此外，单个铰链坐标误差在聚合为单一标量直径时会部分抵消。

系统偏差可忽略（+0.23 mm），因此 $D_{\text{circ}}$ 不存在某些其他距离型测量中观察到的低估问题。

在 TAVI 假体尺寸分桶（23、26、29 mm）上，23.3% 病例落入与真值不同的分桶（95% Clopper--Pearson CI [9.9, 42.3]）。

没有病例出现超过相邻分桶的偏移（$|\Delta\text{bin}|\geq2$；0/30，95% CI [0.0, 11.6]）。

全部 7 个不一致病例都只是相邻尺寸之间的单分桶偏移，类似于 MSL 和冠脉高度中观察到的关键点分辨率受限边界跨越现象。

在当前队列中，全部 30 个真值直径都落入两个最小分桶，19 例 <23 mm，11 例在 23--26 mm，因此该分析没有覆盖上方尺寸范围。

需要覆盖完整假体尺寸谱的更广泛外部队列，才能完成完整验证。

鉴于系统偏差很小（+0.23 mm），这些阈值型下游分析给出的实践含义是一致的：TRUST 导出的测量在参考阈值下显示出有用的一致性，但靠近阈值或尺寸分桶边界的病例仍需要专家复核或术中确认。

**瓣环横截面积。**

除 3 瓣叶 circumdiameter 之外，当前 SCCT 指南~\cite{Blanke2019_scct_tavi_ct} 推荐使用由人工瓣环轮廓导出的瓣环横截面积进行面积型尺寸选择，因为它比通过三个铰链点的外接圆更忠实捕捉原生瓣环的椭圆和非对称几何。

我们只在具有手工轮廓标注的子集中评估这一临床重要测量。

参考面积计算为 Medical Imaging Interaction Toolkit（MITK）瓣环多边形在其自身轮廓平面内的二维 shoelace 面积。

因为 TRUST 预测 10 个关键点而不是稠密瓣环轮廓，预测侧面积定义为圆形三铰链近似 $A_{\text{circ}}^{\text{pred}} = \pi R_{\triangle,\text{pred}}^2$，其中 $R_{\triangle,\text{pred}}$ 是 TRUST 预测铰链三角形（$P_0$--$P_2$）的外接圆半径。

因此，该分析检验现有关键点集合能否近似轮廓导出的 sizing area，而不是检验 TRUST 是否直接预测瓣环轮廓。

在具有可用瓣环轮廓多边形的 14 个测试病例中，一致性为中等：MAE 43.1±28.0 mm$^2$，Pearson $r=0.87$，ICC(2,1)=0.71，95% Bland--Altman limits 为 [-102.2, +20.3] mm$^2$（表 8）。

不同于 $D_{\text{circ}}$ 的近零偏差，点导出的面积近似表现出明显系统低估：在人工轮廓均值 435 mm$^2$ 上偏差为 -41.0 mm$^2$，相对低估约 9.4%。

机制上，这种方向性偏差应理解为用圆形三点近似表示椭圆/非对称瓣环，以及残余铰链点定位误差的共同结果，而不是平面投影效应。

临床上，如果直接使用这种 landmark-only 面积估计，会倾向于推荐小于最优的瓣膜尺寸，因此在 TRUST 扩展到直接瓣环轮廓预测之前，面积型尺寸选择仍应依赖人工轮廓或术中确认。

由于当前 TAVI 工作流中的面积型尺寸选择并不使用单一二分阈值，我们只报告该测量的连续一致性。

**C 臂角度结果与解释。**

两个 cusp-overlap 角度表现出同数量级误差：LAO/RAO 的 MAE 为 6.7±5.0°（$r=0.79$，ICC(2,1)=0.71），CRA/CAU 的 MAE 为 4.8±3.2°（$r=0.72$，ICC=0.68）；见表 8。

预测角度覆盖临床预期范围：LAO/RAO 真值为 [-22°, +38°]，CRA/CAU 为 [+13°, +52°]，与既往 cusp-overlap 透视系列一致~\cite{Pascual2022_cusp_overlap}。

其一致性弱于瓣环 $D_{\text{circ}}$ 测量，因为 cusp-overlap 公式只依赖两个关键点（$P_1$ 和 $P_2$），没有通过第三个铰链点平均，因此逐关键点定位噪声直接传播到射线方向估计。

这与 5 个离群病例一致；这些病例的 $|\Delta\theta_{\text{LAO}}|$ 位于 [10°, 15°]，并对应较高的 $P_1$/$P_2$ 残差。

两个分量都显示正偏差：LAO/RAO 为 +5.04°，CRA/CAU 为 +1.15°。

尽管存在这些限制，两个角度约 6° 的绝对 MAE 仍落在 Arnold 等~\cite{Arnold2012_carm_projection} 报告的 first-shot CT-derived TAVI projection 临床可用 ±10° 工作范围内；该投影在其队列中成功率为 84%。

因此，TRUST 导出的 cusp-overlap 角度可以作为起始投影，并随后在术中精修。

**单中心多标注者分析。**

为将 8 关键点结果置于人工标注变异的背景中，我们进一步在同一 30 例测试集上，按照原始 $P_0$--$P_7$ 协议比较 Reader 1、Reader 2 和 TRUST（表 10）。

在 240 个病例-关键点对上，reader-reader 分歧为 1.87±1.80 mm。

TRUST 相对于两位 reader 仍处于同一数量级（相对于 Reader 1 为 2.20±1.09 mm；相对于 Reader 2 为 2.47±1.71 mm），但并未完全达到观察者间变异水平。

主要差异来源是膜部室间隔关键点 $P_7$：$P_7$ 的 reader-reader 分歧达到 4.69±3.36 mm，而 $P_0$--$P_6$ 的平均分歧仅为 1.47 mm。

这说明限制模型表现的同一个软组织关键点，也是独立人工标注下最不可重复的关键点。

**表 10 图注翻译。** 同一 30 例测试集上的单中心多标注者分析，使用原始 8 关键点协议（$P_0$--$P_7$）。Landmark MRE 在 240 个病例-关键点对上计算。MSL 和 $D_{\text{circ}}$ 条目报告 MAE，并在适用时附带阈值不一致；极端 MSL shift 指最短和最长 MSL 层级之间的变化。C-arm 条目为 LAO/RAO 和 CRA/CAU MAE。DLZ total 报告 MAE，并在括号中给出 Pearson $r$。

| Comparison | 8-pt MRE (mm) | MSL MAE; extreme | $D_{\text{circ}}$ MAE; bin shift | C-arm MAE (°) | DLZ total MAE; $r$ |
|---|---:|---:|---:|---:|---:|
| Reader 1 vs. Reader 2 | 1.87±1.80 | 2.26; 2/30 | 0.81; 4/30 | 4.17 / 2.85 | 18.2; 0.9998 |
| TRUST vs. Reader 1 | 2.20±1.09 | 1.51; 0/30 | 1.03; 9/30 | 6.47 / 4.67 | 25.8; 0.9990 |
| TRUST vs. Reader 2 | 2.47±1.71 | 2.25; 0/30 | 1.25; 9/30 | 6.53 / 3.73 | 11.3; 0.9991 |

下游测量显示出相同模式。

Reader 1 和 Reader 2 在 13/30 例中给出不同的 MSL 阈值层级，其中包括 2 例短 MSL 与长 MSL 之间的极端变化；TRUST 相对于两位 reader 的相邻层级不一致比例相近（15/30），但没有极端 MSL shift。

对于瓣环 $D_{\text{circ}}$，reader-reader 尺寸分桶不一致为 4/30，而 TRUST 相对于每位 reader 均为 9/30；这一结果仍未评估完整轮廓导出的假体尺寸选择。

C-arm 角度从 reader-reader 变异（4.17°/2.85°）到 TRUST-reader 变异（约 6.5° LAO/RAO 和 3.7--4.7° CRA/CAU）有轻度增加。

DLZ 钙化在所有比较中仍保持高度相关（total DLZ volume 的 $r>0.999$），支持这种体积积分终点对小幅关键点扰动的鲁棒性。

由于重标注遵循原始 8 关键点协议，这一分析可用于说明主要铰链点、连合点、中心点和 MSL 相关测量的人工标注变异背景，但不能评估新增冠脉口的观察者间变异。

---

## 5. 讨论

消融研究（表 2--表 3）显示，统一不确定性机制是 TRUST 中单一影响最大的组件。

在累计消融中，在所有架构模块之上激活不确定性带来 -0.27 mm 的 MRE 降低，这比 vanilla HCO 和 Topo-GCN 的合计贡献（-0.06 和 -0.03 mm）大三倍。

更重要的是，不确定性消费者子消融（表 3）显示，在已经存在 HCO 和 Topo-GCN 模块时激活不确定性信号，额外带来 -0.11 mm 降低（2.46→2.35 mm）。

这超过了这些模块作为主消融中的 vanilla 组件时自身的独立贡献 -0.09 mm（2.55→2.46 mm）。

这说明架构模块提供改进容量，即频域特征传播和基于图的结构推理；但这种容量只有在逐关键点可靠性信号指导下区分可信预测和噪声预测时，才会充分实现。

相比之下，现有半监督方法通常只把不确定性用于单一机制：Yu 等~\cite{Yu2019_uncertainty_aware} 用它进行损失加权，Wang 等~\cite{Wang2022_u2pl} 用它进行伪标签选择。

我们的结果表明，多消费者设计，即同一不确定性估计同时塑造特征提取、过滤伪监督并调节结构精修，在平均 MRE 上产生超过部件之和的收益，同时提供消融中观察到的最强方差降低。

临床下游结果说明，TRUST 主要产生边界级别的阈值差异，而不是系统性测量失败：铰链点导出的距离偏差较小（MSL -0.14 mm，LCO 高度 -0.83 mm，RCO 高度 +0.02 mm，瓣环 $D_{\text{circ}}$ +0.23 mm），MSL 没有跨越最短和最长两个极端层级的不一致，$D_{\text{circ}}$ 也没有超过一个尺寸分桶的偏移。

多标注者分析为这些相邻边界偏移提供了必要背景：独立 reader 之间本身在 13/30 例中给出不同 MSL 层级，其中包括 2 例极端层级偏移，并且在 $D_{\text{circ}}$ 上有 4/30 的尺寸分桶不一致。

它也确认 $P_7$ 是读者依赖的瓶颈；与此同时，reader-reader 和 model-reader 比较中 DLZ total 的高度相关（$r>0.999$）说明体积积分型钙化终点对小幅关键点扰动相对不敏感。

### 局限

若干局限需要讨论。

第一，当前测试集仍然来自单中心。

新增的多标注者分析在同一 30 例上说明了原始 8 关键点子集（$P_0$--$P_7$）的人工标注变异背景，但完整 10 关键点参考，包括本文新增的冠脉口 $P_8$ 和 $P_9$，仍缺乏独立多标注者裁决；因此，冠脉高度结果应被理解为相对于一个专家参考的一致性，而不是多读者共识性能。

当前队列在 SCCT 12 mm 冠脉高度参考阈值下也包含很少低于阈值的冠脉口（LCO 8 例，RCO 只有 1 例；表 9），因此逐冠脉口阈值灵敏度估计具有很宽二项置信区间。

尤其是 RCO 只有一个阳性病例，使其 sensitivity 在统计上无信息。

两侧冠脉共观察到的三个假阴性阈值跨越全部位于 12 mm 截断值 1 mm 内，因此反映关键点分辨率受限的边界跨越，而不是真正检测失败。

要对关键点精度和冠脉高度筛查能力给出决定性评估，仍需要在低冠脉高度以及其他冠脉阻塞风险特征富集的队列中进行多中心、多读者 10 关键点外部验证。

第二，虽然不确定性机制在训练期有效，我们观察到 TTA 池化不确定性在推理期会由于收敛模型高置信而饱和，限制其作为推理期质量保证工具的用途。

研究互补的不确定性估计器，例如热图熵或 evidential deep learning，可能解决这一缺口。

第三，级联设计使 Stage 2 暴露于 Stage 1 残余误差。

在测试集 300 个关键点预测中，有 2 个（0.7%）因 Stage 1 严重离群而将 ROI 中心移位到足以使真值关键点落在裁剪体之外的位置；在这种情况下，Stage 2 无法恢复。

为被裁出关键点设计专门恢复机制，仍是本文未处理的方法学方向。

最后，TRUST 预测的是稀疏的 10 关键点表示，而不是完整瓣环轮廓。

因此，瓣环直径 $D_{\text{circ}}$ 和点导出的面积估计仍然是三铰链圆形近似，而当前假体尺寸选择依赖由轮廓导出的瓣环面积和周长。

未来工作应将 TRUST 扩展到稠密瓣环环周建模或直接瓣环轮廓预测，并随后针对具体器械进行假体尺寸选择验证。

---

## 6. 结论

本文提出 TRUST，一个用于从 3D CT 体检测 10 个主动脉根部关键点、支持 TAVI 规划的半监督框架。

通过用统一逐关键点不确定性信号协调 CPS 损失加权、UG-HCO 特征传播和 Topo-GCN 精修，TRUST 在 10 个关键点上达到 2.19 mm 的累计 MRE，在利用 620 个无标签体的同时，相对最强全监督基线降低误差 21.5%。

跨六项 TAVI 规划测量的下游评估显示，TRUST 导出的关键点在阈值型评价下保持临床相关的测量一致性，包括 MSL 无极端阈值层级偏移、假体尺寸无超过相邻分桶的偏移，以及逐瓣叶钙化一致性 $r>0.98$。

同一测试集多标注者分析进一步将原始 8 关键点结果置于人工标注变异的背景中，并识别 $P_7$ 为主要的读者依赖关键点。

主要后续工作包括在低冠脉高度及其他高风险解剖特征富集队列中进行多中心、多读者验证、校准推理期不确定性，以及发展直接瓣环轮廓预测以替代当前三点圆形面积近似。

---

## 致谢

英文主文当前为 `\section*{Acknowledgment}` 后接 `% TODO`。

这表示致谢、资助和贡献说明尚未填写，需要后续与导师确认。

## 参考文献

英文主文使用 `\bibliographystyle{IEEEtran}` 和 `\bibliography{Bibliography_base}`。

因此参考文献由 `Bibliography_base.bib` 提供，并按 IEEE 样式排版。

本中文稿保留主要引用键，方便你从中文版回查英文原文与 bibliography。

## 作者 biography

英文主文包含三个 biography 占位：`First A. Author`、`Second B. Author` 和 `Third C. Author Jr.`。

这些内容目前都是模板占位，正式投稿前需要与作者名单、单位和期刊/会议格式一起处理。
