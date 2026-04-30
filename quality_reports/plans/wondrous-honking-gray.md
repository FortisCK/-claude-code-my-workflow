# Plan — Week 1-2 Foundation (env setup + ImageCAS + PAD pipeline + TT U-Net / HM-EDM 入手)

**Status:** DRAFT
**Date:** 2026-04-29
**Branch:** `cardiac-artifacts`
**Spec:** [`../specs/2026-04-29_master-spec.md`](../specs/2026-04-29_master-spec.md) v1.0 APPROVED
**Plan slot note:** 此文件之前是 workflow-adaptation 计划(已 COMPLETED,在 git 历史里;Session log [`../session_logs/2026-04-29_workflow-adaptation.md`](../session_logs/2026-04-29_workflow-adaptation.md) 完整记录了那一轮)。系统单 plan slot 限制,本文件现在用于 Week 1-2 plan。

---

## Context

Spec v1.0 APPROVED。Master spec 时间线 5 月里程碑 = "TT U-Net 复现 + ImageCAS 准备 + VAE 预训 + HU 保真度 go/no-go gate"。Week 1-2 是这条路径开局,聚焦 **environment + data + PAD pipeline + 第三方 repo 入手**。VAE 预训放到 Week 3-4。

**用户 compute setup(刚 confirmed)**:
- LTSI / Rennes 集群 A6000 节点(SSH access)
- macOS 本地开发(此 repo 所在),远程 LTSI 训练
- **不想用 MATLAB**(影响 PAD pipeline 路线选择)
- Kaggle 账户已有,ImageCAS 未下载

**Mac 本地状态(刚 inspect)**:
- 没 NVIDIA GPU,没 MATLAB
- Python 3.11.5(够用,但远程 LTSI 上会建 conda env 用 3.10 — 主流 ML libs 兼容性最好)
- 没现有 venv / pyproject.toml / 已 clone repos / ImageCAS 数据

完全从 0 起步,这个 plan 必须把 env / data / repo / dual-machine workflow 全部 stand up。

---

## Goals(Week 1-2 deliverables)

| # | Goal | Verification |
| --- | --- | --- |
| 1 | **Mac ↔ LTSI dual-machine workflow established** | git push 从 Mac → LTSI git pull 工作;数据/checkpoints 不进 git |
| 2 | **LTSI conda env `cardiac-diffusion`** with PyTorch + MONAI + monai-generative pinned in `pyproject.toml` | `python -c "import torch, monai, generative; print(torch.cuda.is_available())"` 返回 True |
| 3 | **ImageCAS 1000 例下载到 LTSI 大磁盘** | `data/imagecas/raw/` 下文件计数对得上;1 例 NIfTI volume 加载 + metadata 读取成功 |
| 4 | **PAD pipeline first-run**(走非 MATLAB 路径)生成 ≥ 5 对 (clean, corrupted) | 视觉对比 PNG slices in `explorations/figures/`;crescent / tail / horn 形态可见 |
| 5 | **TT U-Net 仓库 cloned + readthrough notes** + small-batch sanity training | training script forward + backward 跑通,不要求 metrics 收敛 |
| 6 | **HM-EDM 仓库 cloned + EDM-3D-patch readthrough notes** | notes 文件 in `explorations/notes/`,记录 σ schedule / patch loader / conditioning 关键细节 |
| 7 | **3 张 run card** + 1 张过 python-reviewer | `experiments/runs/` 下 3 个 .md;reviewer 报告 in `quality_reports/` |

**显式不做**(留给 Week 3+):
- VAE 预训(Week 3-4)
- 完整 TT U-Net training reproduction(Week 3,Week 1-2 只 sanity)
- LDM 训练(Week 5+)
- 任何 model 评估 / metric 报数(Week 5+)
- nnU-Net 冠脉 segmenter 训练(Week 5-8 lumen-loss 准备时)

---

## Critical decisions(此 plan 内 settled)

### 1. MATLAB 避开策略 — Ladder L1 → L2 → L3 → fallback

TT U-Net 的 PAD 流程拆分(per §II-B):

| Step | 性质 | MATLAB 依赖? |
| --- | --- | --- |
| 1-2 单相位分割 + Shape Model 转换 | 几何处理 | Python OK |
| **3 4D SSM via PCA on XCAT** | **MATLAB 原版** | **是** |
| 4 ShapeMorph dense deformation field | PyTorch | Python OK |
| 5 Cone-beam scan + FDK reconstruction | 数值仿真 | Python OK(LEAP / CTorch / DeepDRR 可选) |

替代 Step 3 的 4-级 ladder(按工作量从低到高):

- **L1**: 检查 TT U-Net repo 是否 release pretrained 4D SSM(`.npy` / `.mat` 文件 → scipy.io 读取 OK)→ 不需要重建,直接用。**Day 5 决定**
- **L2**: 用 GNU Octave 跑他们的 MATLAB 代码 unchanged(免费 + 大部分 PCA 类 MATLAB 代码兼容 Octave)。L1 失败时,**Day 6 试 L2**
- **L3**: 纯 Python 重写 PCA(`scipy.linalg.svd` / `sklearn.decomposition.PCA` on shape vectors)— 工作量 +2-3 天。L2 也失败时,**Week 3 转 L3**(超出本 plan 范围)
- **Fallback**: 整体放弃 SSM 路线,走 **DVF-warp + forward projection** 简化版(V2 §08 risk row 3)— Week 1-2 内可跑通,但仿真真实度低于 SSM-driven PAD。**只在 L1+L2 都失败时启用**

Plan 默认 L1 → L2 顺序尝试。Week 1-2 内不投资 L3。

### 2. 远程开发 workflow

**模式**:Mac 草拟代码 / 文档 → git push → LTSI pull → LTSI 跑代码 / 数据。

**约定**:
- 代码 / 配置 / spec / plan / 文档:进 git(Mac ↔ LTSI sync via GitHub)
- 数据(ImageCAS, generated PAD pairs, model checkpoints, run outputs):**只在 LTSI**,不进 git(`data/.gitignore` 已配置)
- LTSI 路径配置在 `data/.paths.local`(gitignored)
- 每天 work session 结束:Mac 端 git push;LTSI 端 git pull next morning

**具体 commands(reference)**:
```bash
# Mac:
git add ... && git commit -m "..." && git push origin cardiac-artifacts

# LTSI(初次):
git clone https://github.com/FortisCK/-claude-code-my-workflow.git cardiac-artifacts
cd cardiac-artifacts && git checkout cardiac-artifacts

# LTSI(每天):
git pull
conda activate cardiac-diffusion
python -m code.training.xxx --config ...
```

**SSH 多路复用**:推荐 LTSI ssh 配置加 `ControlMaster auto` + `ControlPersist 10m`(可选,加快后续连接)。

### 3. Conda env 构建(LTSI 端)

| 组件 | 版本 |
| --- | --- |
| Python | 3.10(主流 ML libs 兼容性最好) |
| PyTorch | 2.4+(CUDA 12.1+ 匹配 A6000 driver) |
| MONAI | 1.4+ + monai-generative 0.2+ |
| WandB | latest |
| 医学影像 IO | nibabel + pydicom + SimpleITK |
| 数值 / 图像 | numpy, scipy, scikit-image |

**不在 Day 1-2 装**:
- nnU-Net(Week 5-8 lumen-loss 准备时再装)
- Octave(L2 fallback 时才装)
- LEAP / CTorch(Path B 探索时才装,Week 17+)

`pyproject.toml` 锁版本,`pip install -e .` 让 `code/` 可作为 package 导入。

---

## Implementation steps(10 个工作日)

### Day 1 — Baseline 同步

- [ ] Mac:运行 `/commit` 把所有 spec / plan / decision-record / 上一轮 workflow-adaptation 提交到 `cardiac-artifacts` 分支,push 到 GitHub
- [ ] LTSI:SSH 进去,验证 `nvidia-smi` 显示 A6000 + CUDA 12.x + free disk ≥ 200 GB(为 ImageCAS + checkpoints + outputs)
- [ ] LTSI:`git clone https://github.com/FortisCK/-claude-code-my-workflow.git cardiac-artifacts && cd cardiac-artifacts && git checkout cardiac-artifacts && git pull`
- [ ] LTSI:`conda create -n cardiac-diffusion python=3.10 -y`(或 mamba 加速)

**Verification**:LTSI `nvidia-smi` 显示 A6000;`conda env list` 含 `cardiac-diffusion`;`git status` 显示 cardiac-artifacts 分支。

### Day 2 — Python ML 栈安装 + smoke test

- [ ] LTSI: 写 `pyproject.toml`(此 repo 第一个),pin 主要库版本
- [ ] LTSI:`conda activate cardiac-diffusion && pip install torch ...`(参考 PyTorch 官网,匹配 LTSI CUDA driver 版本)
- [ ] LTSI:`pip install monai monai-generative wandb nibabel pydicom SimpleITK numpy scipy scikit-image`
- [ ] LTSI:`pip install -e .`(editable install)
- [ ] LTSI:smoke test 脚本 `scripts/python/check_env.py` —— 加载 torch 验证 CUDA 可见,加载 monai/AutoencoderKL 实例化,记录所有库版本到 stdout
- [ ] WandB:`wandb login`(用户提供 token)

**Run card #1**: `experiments/runs/2026-04-XX_HHMM_env-setup.md`
- 不是真"训练 run",但记录 env state for reproducibility
- 模板:python 版本 / torch 版本 / monai 版本 / GPU 型号 / driver / CUDA / 系统

### Day 3-4 — ImageCAS 下载 + 初次 inspect

- [ ] LTSI:`pip install kaggle`,配置 `~/.kaggle/kaggle.json`(用户的 Kaggle credential)
- [ ] LTSI:`kaggle datasets download xiaoweixu/imagecas`(确认数据集 slug;下载到 `/data` 或 `/scratch` 大磁盘)
- [ ] LTSI:解压到 `data/imagecas/raw/`,`du -sh` 看实际大小,`ls | wc -l` 看文件数
- [ ] LTSI:写 `data/.paths.local`(LTSI 路径)
- [ ] LTSI:写 `code/data/imagecas_loader.py` skeleton —— 1 个函数 `load_volume(case_id) -> SimpleITK.Image`
- [ ] LTSI:写 `scripts/python/imagecas_inspect.py` —— 加载 1 例,打印 shape / voxel size / HU range / metadata
- [ ] LTSI:跑 inspect on 5 例,确认 1000 例都加载成功
- [ ] LTSI:写 `data/imagecas/SUMMARY.md` —— 1000 例的元数据汇总(HR 分布、phase 分布、是否带分割标签)

**Verification**:`python scripts/python/imagecas_inspect.py --case-id <id>` 在 LTSI 上跑通,输出合理 metadata。

⚠️ **风险点**:LTSI 网络速度未知。若下载 > 36h,联系 LTSI sysadmin 是否有镜像 / 优化路径。

### Day 5 — TT U-Net 仓库 clone + 阅读

- [ ] LTSI:`git clone https://github.com/ivy9092111111/TT-U-Net`(per TT U-Net abstract)在 LTSI scratch 区
- [ ] LTSI:阅读 README, requirements.txt, 主要 entry points
- [ ] **重点检查**:是否 release pretrained 4D SSM(`.npy` / `.mat`)→ 决定 PAD ladder L1/L2
- [ ] **重点检查**:training script 期望的 data loader 格式(NIfTI? DICOM? video clip vs single volume?)
- [ ] **重点检查**:PAD pipeline 哪些是 MATLAB,哪些是 Python
- [ ] 写 `explorations/notes/tt-u-net-readthrough.md`(在 explorations/ 因为是探索性 read-only 笔记):
  - 仓库结构概要
  - PAD 流程的 MATLAB / Python 分界
  - 4D SSM 是否 release(L1 决策依据)
  - data loader 期望
  - 训练入口 + 关键超参
  - 与我们 single-phase + non-rigid + ImageCAS 的迁移点

**Decision point at end of Day 5**:
- 若 release 4D SSM → L1 路径,Day 6 直接用 release artifact
- 若没 release → Day 6 装 Octave 试 L2

### Day 6-7 — PAD 第一次跑通

**Day 6**:基于 Day 5 决策,实施 SSM 替代:
- L1 路径:加载 TT U-Net release 的 4D SSM,跑 ShapeMorph(Python)+ cone-beam scan + FDK,生成 1 例 (clean, corrupted)
- L2 路径:在 LTSI 装 Octave,跑 TT U-Net 的 MATLAB 代码 unchanged 构建 4D SSM,然后跑 ShapeMorph(Python)等下游

**Day 7**:扩展 PAD 到 5 例 ImageCAS,确认 pipeline 稳定:
- 写 `code/data/pad_pipeline.py`(把 generation 流程封装)
- 视觉对比:`explorations/figures/pad-first-run/` 下放 5 例 PNG slices(干净 vs 伪影 side-by-side)
- 检查 corrupted 是否有 crescent / tail / horn 形态(对照 TT U-Net Fig 2-3)
- 若伪影 looks wrong → 调试 motion model;若合理 → 进 Day 8

**Run card #2**: `experiments/runs/2026-04-XX_HHMM_pad-first-run.md`
- 记录:几例输入,SSM 路线(L1 / L2 / fallback),生成时间 per case,输出质量(视觉判定)

⚠️ **风险点**:L1 + L2 都失败 → Day 7 切 fallback(DVF-warp + forward projection 简化版,无 SSM)。这条路用 ImageCAS 的 GT 冠脉 mask + 随机 dense DF 直接 warp,然后用 LEAP/Tomosipo / DeepDRR 做投影 + FBP 重建,得到伪影。真实度低但 Week 1-2 内能跑通。Spec MAY-pad-pro 允许 Week 3+ 升级。

### Day 8 — HM-EDM 仓库 clone + EDM-3D-patch 阅读

- [ ] LTSI:`git clone github.com/zhennongchen/Diffusion_for_CT_motion`(per V1 §3.6)
- [ ] 阅读 README, training script
- [ ] **重点看**:EDM σ schedule, σ-dependent preconditioning, 128×128×50 patch loader, channel-concat conditioning
- [ ] 写 `explorations/notes/hm-edm-readthrough.md`:
  - 仓库结构概要
  - EDM 训练核心(σ sampling + denoiser + loss)
  - patch loader / batch size / batch 内 patch 数
  - conditioning 实现细节
  - 思考:他们的 head CT rigid motion 框架如何迁移到我们 cardiac non-rigid + 单相位 + latent
  - 我们应当从他们这里继承什么 / 改什么

### Day 9 — TT U-Net 训练 attempt(小规模 sanity only)

- [ ] 用 Day 6-7 PAD 生成的 5-20 对作为 mini-training set
- [ ] 配置 TT U-Net training(原 hyperparam / 原 architecture)
- [ ] 训 100-500 step **看 loss 下降合理**;不期望 metrics 收敛
- [ ] 重点验证:dataloader OK / forward 不爆显存 / backward 不爆 / loss 不 NaN

**Run card #3**: `experiments/runs/2026-04-XX_HHMM_tt-unet-sanity.md`
- 不是 reproduction,只是 plumbing 验证
- 完整 reproduction 移到 Week 3-4

### Day 10 — 总结 + 回顾 + Week 3-4 plan 调整

- [ ] 写 Week 1-2 session log `quality_reports/session_logs/2026-05-XX_week-1-2.md`
- [ ] 把 3 张 run card 给 `python-reviewer` agent(via `/review-python`)跑一遍
- [ ] **评估 Week 3-4 plan 是否需要调整**:
  - PAD pipeline 顺利吗?(若不顺利,Week 3 多分配 1-2 天)
  - TT U-Net sanity 跑通了吗?(若没,Week 3-4 重做)
  - 4D SSM 替代 ladder 落在哪一级?(L1 / L2 / fallback 各对应不同 Week 3 plan)
- [ ] update spec(若 ASSUMED 项有了新数据)
- [ ] **给 Pascal/Carlos 发邮件**:
  - XCAT phantom license(operational AI #1)
  - LTSI PACS retrospective pull(operational AI #2,Week 8-10 才用)
- [ ] **给 Deng 实验室发邮件**(operational AI #3,stretch)

---

## Files this plan will create / modify

### New
- `pyproject.toml`(repo 第一个)
- `code/data/imagecas_loader.py`
- `code/data/pad_pipeline.py`
- `scripts/python/check_env.py`
- `scripts/python/imagecas_inspect.py`
- `data/imagecas/SUMMARY.md`
- `data/.paths.local`(在 LTSI 上,gitignored)
- `experiments/runs/2026-04-XX_HHMM_env-setup.md`(run card #1)
- `experiments/runs/2026-04-XX_HHMM_pad-first-run.md`(run card #2)
- `experiments/runs/2026-04-XX_HHMM_tt-unet-sanity.md`(run card #3)
- `explorations/notes/tt-u-net-readthrough.md`
- `explorations/notes/hm-edm-readthrough.md`
- `explorations/figures/pad-first-run/*.png`
- `quality_reports/session_logs/2026-05-XX_week-1-2.md`

### Cloned (in LTSI scratch, NOT in this repo)
- `TT-U-Net/` repo
- `Diffusion_for_CT_motion/` repo (HM-EDM)

### Modified
- (无 — 此 plan 不动现有 spec / decision-records / archived 内容)

---

## Risk Register(Week 1-2 specific)

| 风险 | 严重度 | 应对 |
| --- | --- | --- |
| ImageCAS 下载慢(50-100 GB on LTSI 网络) | 中 | 多线程 download,用 `aria2` 或 `kaggle-cli` resume;若 > 2 天 → 联系 LTSI sysadmin 优化 |
| TT U-Net repo 不 release pretrained 4D SSM | 中 | L2 Octave fallback;Day 7 再失败则 DVF-warp 简化(V2 §08 fallback) |
| LTSI 上 conda env 安装冲突 | 中 | mamba 替代 conda;若深度问题 → docker container |
| LTSI A6000 PyTorch CUDA 不兼容 | 低-中 | nvidia-smi 看 driver 版本,装匹配 PyTorch wheel |
| ImageCAS license 限制 | 低 | 假定 Apache 2.0 OK;若发现 restriction → Week 3 处理 |
| LTSI scratch 满了 / disk quota 不够 | 低 | Day 1 先 `df -h` 验证,确保 ≥ 200 GB free;不够则联系 sysadmin |
| 你 SSH 到 LTSI 中断丢失任务 | 中 | 所有长任务跑在 `tmux` / `screen` 下;不直接前台跑 |
| Mac 与 LTSI git 同步混乱 | 低 | 严格"Mac edit, LTSI run-only";不在 LTSI 编辑 .py 然后忘 push |

---

## Out of scope(显式排除)

- VAE 预训(Week 3-4)
- 完整 TT U-Net training reproduction(Week 3,Week 1-2 只 sanity)
- LDM 设计 / 实现 / 训练(Week 5+)
- 任何 evaluation pipeline / metric 报数(Week 5+)
- nnU-Net 冠脉 segmenter 训练(Week 5-8)
- Lumen-aware loss 函数实现(Week 7+)
- Path C(I2SB)/ DAPS / 其他 baseline(Week 17+)

---

## Verification(end of Week 2,以下都要 PASS)

1. **环境**:LTSI `python -c "import torch, monai, generative; print(torch.cuda.is_available())"` 输出 True + 显示 A6000
2. **数据**:LTSI 1000 例 ImageCAS volumes 可加载,SUMMARY.md 完整
3. **PAD**:≥ 5 对 (clean, corrupted) 已生成,视觉合理(crescent / tail / horn 可见)
4. **代码 + 仓库**:TT U-Net 和 HM-EDM repos 都 clone 完,readthrough notes 写完
5. **TT U-Net sanity**:training script 在小 batch 上 forward + backward 跑通(不要求 metrics 收敛)
6. **Run cards**:3 张已写 + 1 张过 python-reviewer
7. **Operational outreach**:3 封 email 发出(Pascal × 2,Deng × 1)

如果 (1)-(7) 都 PASS,Week 3-4 plan 可以基于"VAE 预训 + HU 保真度 go/no-go gate"那个 spec milestone 起步。如果某项 FAIL,Week 3 第一周补齐 + 后续节奏顺延。

---

## Estimated effort

10 个工作日,~60-80 小时。每天预留 1 小时 buffer(SSH 调试 / 包冲突 / 等等)。

| Phase | 工作量分布 |
| --- | --- |
| Day 1-2 env setup | 12-16 h |
| Day 3-4 ImageCAS | 8-12 h(大部分是下载等待 + inspect 脚本) |
| Day 5 TT U-Net read | 6-8 h |
| Day 6-7 PAD first-run | 12-16 h(L1/L2 决策 + debug) |
| Day 8 HM-EDM read | 4-6 h |
| Day 9 TT U-Net sanity | 6-10 h |
| Day 10 总结 | 4-6 h |

如果 Day 6-7 的 L1/L2 都失败要切 fallback,可能挤压 Day 9-10。最坏情况:TT U-Net sanity 推到 Week 3 Day 1,不影响 spec 时间线(Week 3-4 milestone 原本就有 buffer)。
