# Spec — 参数化 Motion Synthesis Pipeline (Week 1-2)

**Date:** 2026-04-30
**Status:** APPROVED v1.0 (2026-04-30, CMZ 批准) — IMPLEMENTING
**Spec parent:** [`2026-04-29_master-spec.md`](2026-04-29_master-spec.md) v1.1
**Decision parent:** Mac↔LTSI 2026-04-30 session 锁定的 motion source 决策 — **Path D(参数化 DVF + cone-beam + FBP)是 Week 1-2 主线**,B(cine MR)/ A(XCAT)stretch 平行
**Related:** Web search 2024-2026 evidence summary(此 conversation 上文)

---

## 一句话目标

从一例 ImageCAS 静态 contrast CCTA volume(`V_clean`),通过参数化 DVF + cone-beam scan + FBP/FDK 重建,生成一对 `(V_clean, V_corrupted)`,作为 latent diffusion 的训练对。**单例 ImageCAS → 多对 corrupted 版本**(随机化 motion 参数)。

## Scope

### IN(Week 1-2 必须)

- 参数化 DVF 生成器(closed-form, 无 learning):LV/RV contractile + apex twist + sinusoidal time profile
- Whole-heart segmentation 自动化(用于限定 motion 范围)
- Cone-beam forward projection + Parker short-scan weighted FBP
- 端到端 `synthesize(V_clean) → V_corrupted` 函数 + per-case 多变体
- 1 例视觉 sanity(伪影 morphology vs PAD demo case 40/50 side-by-side)

### OUT(Week 3+ 或不做)

- 4D-SSM 重造(取消)
- Cine MR DVF 提取(B 路线,Week 9+ stretch)
- Lumen-aware loss(Week 5+)
- Forward projection 的 detector noise / scatter(Week 5+)
- Differentiable end-to-end (gradient through scan)(Week 7+ 如有需要)

---

## Pipeline architecture

```
ImageCAS V_clean (Z, Y, X) HU
    │
    ├─[1] TotalSegmentator → whole_heart_mask (Z, Y, X) {0, 1}
    │
    ├─[2] 参数化 DVF 生成 — 给定 motion_params + heart_mask,生成 N_phases=48
    │        个 dense displacement field DVF(t)(每个 (Z, Y, X, 3))
    │
    ├─[3] 动态 volume — V_dyn(t) = warp(V_clean, DVF(t)),48 个相位
    │
    ├─[4] Cone-beam scan — 给定 N_views ≈ 1000 个角度,每个角度对应一个
    │        scan_time(由 rotation_period 决定),scan_time 决定看到哪个
    │        cardiac phase。Forward project V_dyn(phase(angle_i)) → 
    │        sinogram[i]
    │
    ├─[5] Parker short-scan weighting(避免冗余 view 的 weight 不一致问题)
    │
    └─[6] FBP/FDK 重建 → V_corrupted (Z, Y, X) HU
```

**关键洞察**:每一个 projection view 看到一个**冻结的某相位 volume**,但**不同 view 看到不同相位**。这种 "views see different snapshots of a moving heart" 正是 motion artifact 的物理来源。

---

## 组件决策

### 1. Forward projection library: **tomosipo + ASTRA**(实际落点)

> **2026-04-30 update**:LEAP **没有 PyPI wheel**,readthedocs 安装文档失效 + GitHub wiki 加载错误,源码 build 需要 CUDA dev tools。按 spec 允许的 fallback 切换到 **tomosipo + ASTRA**。同等满足 cone-beam + non-equispaced angles + PyTorch tensor + GPU 的需求,smoke test 通过(32³ phantom 8 angles 在 RTX 6000 Ada 上跑通)。
>
> 实际安装:
> - `pip install astra-toolbox`(2.4.1,PyPI 直接 wheel)
> - `pip install git+https://github.com/ahendriksen/tomosipo.git`(0.7.0,纯 Python wrapper)
>
> 跟 LEAP 比的 trade-off:tomosipo API 更轻量;ASTRA 是 GPL,但只动态链接(pip wheel 形式),不影响我们 paper 代码 license。
>
> ### 1.1(原决策)Forward projection library: **LEAP** (LLNL)

**选 LEAP 不选 tomosipo / TIGRE / DeepDRR 的理由**:

| 特性 | LEAP | tomosipo | 我们要的 |
|---|---|---|---|
| Native PyTorch tensor | ✅ `leaptorch.Projector` is `nn.Module` | ✅ via ASTRA | ✅ |
| Cone-beam geometry | ✅ `set_coneBeam(phis=...)` | ✅ | ✅ |
| **Non-equispaced angles** | ✅ explicit support | ✅ | ✅ critical(每个 view 角度 + 时间映射不同 phase) |
| **每 view 不同 volume** | ⚠️ 需要 N_views 次 forward call(可批) | ⚠️ 同 | OK(n_phases=48 batch) |
| Parker short-scan FBP | ✅ 内置 | ⚠️ 需要自己实现 weighting | ✅ |
| GPU multi-device | ✅ | ✅ | OK,A6000 单卡 |
| License | BSD-3 (per LLNL convention) | MIT (tomosipo) + GPL (ASTRA backend) | LEAP 更干净 |
| 安装复杂度 | `pip install leapct`(预编译 wheel) | `pip install tomosipo` + ASTRA conda | LEAP 简单 |
| AI/ML focus | ✅ designed for AI/ML CT | 通用 tomography | LEAP 更对口 |

**Fallback**:如果 LEAP 安装失败 / 性能不达标,**改用 tomosipo + ASTRA**(成熟稳定,科研标配)。这是 spec 里允许的 ASSUMED → CLEAR 切换路径。

### 2. Whole-heart segmentation: **TotalSegmentator**

**选 TotalSegmentator(`pip install totalsegmentator`)的理由**:
- 公开权重(Apache-2.0),开箱即用
- 支持 117 个解剖结构,**心脏 4 chamber + 大血管全有**(LV/RV/LA/RA + 升主动脉 + 主肺动脉)
- `pip install` 直接装,**无 license 障碍**
- 推理速度:~30s per ImageCAS volume on A6000(单 GPU)
- 跟我们的 ImageCAS 同 modality(contrast CT)

**不选 nnU-Net pretrained**:大部分 cardiac pretrained nnU-Net 是 MMWHS-trained,只 segment 7 类心脏结构,体外大血管缺失;TotalSegmentator 全。

**不选 MONAI Bundle 心脏模型**:MONAI bundle 的 `wholeBody_ct_segmentation` 实际就是 TotalSegmentator 重新打包,不如直接用 upstream。

### 3. 参数化 motion 模型: **closed-form,4 个分量叠加**

```python
DVF(x, t) = D_translation(t)          # 全心平移(可选,小)
          + D_contraction(x, t)        # LV/RV 径向收缩
          + D_twist(x, t)              # apex-base 扭转
          + D_long_axis(x, t)          # 长轴缩短
```

**主导分量是 contraction(径向收缩)**,~5-15 mm at apex,decay to 0 at boundary。其它三项是 small additive。Per-component 数学形式见 §Appendix A。

**为什么不用 random smooth field**:伪影 morphology 跟运动方向强相关。径向收缩产生典型的"crescent / horn",随机平滑场产生不自然的 blur,reviewer 一看就出。

**为什么不用 ML 训出来的 motion**:violation of 我们的 contribution(diffusion prior),且 4D 训练数据获取本身就是难题。

#### Antecedents(2026-05-01 lit-review 后补)

**最直接 antecedent**: Lossau et al. 2019(MedIA 52:68-79)— CoMoFACT。Coronary-segment patch 上的参数化 forward model + CT scan simulation。我们和他们的差别:

| 维度 | Lossau 2019 (CoMoFACT) | 本工作 |
|------|------------------------|--------|
| 作用域 | Coronary 2.5D patch | Whole-heart 3D volume |
| 输出 | 2D motion vector label | Paired (clean, motion-corrupted) volume |
| 用途 | 训 CNN 识别 / 量化伪影 | 训 3D latent diffusion 去伪影 |

**其他 lineage**: Hahn 2017 PAMoCo + Maier 2021/2025 Deep PAMoCo —— vessel-centerline polynomial motion,不是 whole-volume DVF。**XCAT-based**: Deng 2023 TT U-Net(license-bound,4D-SSM 训练代码未开源)。

**为什么我们和这些不一样**:license-free + whole-heart 3D + 直接给 diffusion model 配套 paired data。

**详细**: `quality_reports/lit_review_parametric-cardiac-CT-motion-simulation.md`(含 BibTeX,2026-05-01 已 CoVe 验证)。

#### 参数物理 anchors(2026-05-01 CoVe 修正)

| 参数 | 默认值(median) | 训练采样区间(augmentation) | 文献 anchor |
|------|---------|---------|---------|
| `contraction_amp_mm` | 10 | U(7, 14) | LV global radial strain ~47%(Truong 2024 meta) |
| `twist_amp_deg` | **10**(从 15 调下来) | U(5, 20) | LV twist normal ~7-8° ± 3°(Sengupta 2008 / Stöhr 2016) |
| `long_axis_amp_mm` | 10 | U(8, 14) | AV-plane displacement 12-15 mm 健康人 |
| `cardiac_period_ms` | 800 | U(600, 1000) | 60-100 bpm normal HR |

**注**:训练时 augmentation 分布上限可以略超人群 mean,**让 diffusion 模型见到一些"高 motion"案例,提升对真实临床重伪影的去噪能力**。这是用户选项 1(保守 default + augmentation 覆盖上 tail)的核心 rationale。

### 4. Cone-beam scan geometry: cardiac CT 典型参数

| 参数 | 默认 | 物理意义 / 来源 |
|---|---|---|
| SID(source-isocenter) | 540 mm | Siemens / GE 临床机器典型 |
| SDD(source-detector) | 950 mm | 同上 |
| 探测器 row pitch | 0.625 mm | ImageCAS slice thickness 同量级 |
| 探测器 row 数 | 320 | 16 cm collimation,涵盖整心 |
| 探测器 col pitch | 1.0 mm | 默认 |
| 探测器 col 数 | 800 | 800 mm 视野 |
| Rotation period | 250 ms | 临床机器典型 gantry rotation |
| **N_views per rotation** | **1000** | 临床机器 angular sampling |
| Scan range(angles) | 180° + 2 × fan_angle | Parker short-scan |
| Heart rate | 60-100 bpm | 训练时 random per case |
| Reconstruction phase | 75% R-R | mid-diastole(临床实践 standard) |

随机化 motion variant 的参数:`heart_rate`、`initial_scan_angle`、`reconstruction_phase`、`contraction_amplitude`。每例 ImageCAS → 5-10 个 corrupted 版本。

### 5. 重建 — Parker short-scan FBP / FDK

LEAP 内置 `LEAP.FBP()` 接受 weighted sinogram。Parker weighting 计算:

$$w(\beta, \alpha) = \begin{cases} \sin^2\left(\frac{\pi}{4} \cdot \frac{\beta}{\delta - \alpha}\right) & 0 \le \beta < 2(\delta - \alpha) \\ 1 & 2(\delta - \alpha) \le \beta \le \pi - 2\alpha \\ \sin^2\left(\frac{\pi}{4} \cdot \frac{\pi + 2\delta - \beta}{\delta + \alpha}\right) & \pi - 2\alpha < \beta \le \pi + 2\delta \end{cases}$$

其中 $\beta$ 是 view angle,$\alpha$ 是 detector fan angle,$\delta$ = max($\alpha$)。

跟 TT U-Net 的 `getshortscanweighted.m` 完全一致,可以直接交叉验证。

---

## 实现 stages

### Stage 1 — Day 6(单元跑通)

文件:`code/data/motion_synth.py`

```python
def synthesize_motion_artifact(
    V_clean: torch.Tensor,         # (Z, Y, X), HU
    heart_mask: torch.Tensor,      # (Z, Y, X), {0, 1}
    motion_params: MotionParams,
    scan_params: ScanParams,
    seed: int,
) -> torch.Tensor:                 # (Z, Y, X), HU corrupted
```

输出 1 例 motion-corrupted volume。先跑 1 例 ImageCAS,视觉 sanity:

- 在 `explorations/figures/motion-synth-day6/` 输出 mid-axial / coronal / sagittal slice 的 PNG(`V_clean` vs `V_corrupted` side-by-side)
- 期望看到 crescent / horn / tail 形态在 RCA / LCA branch 区域

### Stage 2 — Day 7

- 多 motion variant 随机化(heart_rate / scan_angle / recon_phase / contraction_amp 4 维 random)
- 5 例 ImageCAS × 5 variants = 25 对(可以放进 mini-training)
- 视觉 sanity vs PAD demo case 40/50(从 GDrive 下载,如果时间允许)

### Stage 3 — Day 8-10(Week 1-2 收尾)

- 扩到 ImageCAS 200 例 × 5 variants = 1000 对(用 batch processing)
- 写 Run card #2(`experiments/runs/<...>_pad-first-run.md`)
- 输出 `data/imagecas/synth-motion/v1/` 目录,后续 Week 3-4 直接喂 VAE 训练

---

## 验证 plan

### V1: 视觉 — Day 6

1 例 ImageCAS 生成的 `V_corrupted` 切片,**人工对比** TT U-Net 论文 Fig 2-3 的伪影形态。Pass criterion:**主观判断"长得像"**(cardiac CT motion artifact 该有的特征)。

### V2: PAD demo 对照 — Day 7-10(stretch)

下载 PAD GDrive 那 2 个 demo cases(case 40 + 50),用我们 pipeline 生成 `V_corrupted_ours`,跟原作者 `V_corrupted_pad` 视觉并排。Pass criterion:**形态一致**(crescent 位置、严重程度量级)。

### V3: 量化 — Week 5+

训第一版 diffusion prior,看在 ImageCAS 自己的 held-out test set 上 PSNR / SSIM 收敛性能。Week 1-2 不验。

---

## Open questions(等数据来 + Day 6 后回填)

| Q | 等什么决定 |
|---|---|
| ImageCAS volume 实际 shape / spacing | 等下载完跑 `imagecas_inspect.py` |
| TotalSegmentator 在 ImageCAS 上分割效果 | 等 Day 5 用 5 例验证 |
| LEAP 在 RTX 6000 Ada (CC 8.9, CUDA 12.8) 安装 + 运行 OK 吗 | 等 `pip install leapct` 试 |
| heart_rate 随机化 distribution | Day 7 调试时定 |
| Per-case variant 数 | Day 8 量化定 |

## Risks

| Risk | Severity | Mitigation |
|---|---|---|
| LEAP 安装失败(预编译 wheel 不匹配) | Medium | Fallback to tomosipo |
| TotalSegmentator 在某些 ImageCAS 例上 fail(异常解剖) | Low | 跳过失败例,记录 case_id;100 例失 5 例可接受 |
| 参数化 motion 跟真实临床 morphology 差距大 | Medium | V2 PAD demo 对照;Week 5+ 量化看 generalization;最差走 B 路线(cine MR) |
| Cone-beam projection 速度太慢(48 phases × 200 cases × 5 variants = 48000 forward call) | Medium | LEAP multi-GPU + batch;若仍慢,降 N_phases=24 或 batch view 数 |
| 重建生成的 `V_corrupted` HU 范围漂移(reconstruction artifact 引入额外 HU shift) | Medium | 校准:V_clean 也走一次 forward+FBP(no motion)看 HU 漂移,作为基线减掉 |

---

## Appendix A — Parametric motion model 数学

### A.1 LV/RV contraction(主分量)

设 LV centroid 为 $c_{LV}$(从 heart_mask 提的 LV chamber centroid),contraction amplitude $a_c$,周期 $T$。在某像素 $x$:

$$D_{contraction}(x, t) = -a_c \cdot s(t) \cdot \mathbf{1}_{x \in \text{heart}} \cdot \text{decay}(\|x - c_{LV}\|) \cdot \frac{x - c_{LV}}{\|x - c_{LV}\|}$$

- $s(t)$:asymmetric sin 时间 profile,systole 0.4T 快,diastole 0.6T 慢,见 A.4
- $\text{decay}(d)$:Gaussian decay $\exp(-d^2 / 2\sigma^2)$,$\sigma$ 设为 LV radius
- 负号表示**收缩**(向 centroid 方向)

### A.2 Apex twist

设 apex centroid $c_{apex}$,base centroid $c_{base}$,长轴 $\hat{e}_z = (c_{base} - c_{apex}) / \|\cdot\|$。在某像素 $x$:

$$D_{twist}(x, t) = a_t \cdot s(t) \cdot \frac{(x - c_{LV})_\perp \cdot d_z}{H} \times \hat{e}_z$$

- $(x - c_{LV})_\perp$:相对 LV centroid 的向量,投影到垂直 $\hat{e}_z$ 的平面
- $d_z$:相对 base 的 z 距离
- $H$:apex-base 距离
- $\times \hat{e}_z$:旋转方向叉积

幅度 $a_t \approx 15°$ 在 apex,衰减到 0 在 base。

### A.3 Long-axis shortening

$$D_{long}(x, t) = a_l \cdot s(t) \cdot \frac{d_z}{H} \cdot \hat{e}_z$$

apex 朝 base 移动,~10 mm。

### A.4 时间 profile

$$s(t) = \begin{cases} \sin\left(\frac{\pi t}{0.4 T}\right) & 0 \le t < 0.4T \quad \text{(systole, fast)} \\ \cos\left(\frac{\pi (t - 0.4T)}{2 \cdot 0.6 T}\right) & 0.4T \le t < T \quad \text{(diastole, slow)} \end{cases}$$

$s(t=0) = 0$, $s(t = 0.4T) = 1$, $s(t = T) = 0$。

### A.5 随机化(per case)

| 参数 | 默认 | 随机范围 |
|---|---|---|
| $a_c$(contraction amp) | 10 mm | $\mathcal{U}[6, 14]$ mm |
| $a_t$(twist amp) | 15° | $\mathcal{U}[10, 20]$° |
| $a_l$(long-axis amp) | 10 mm | $\mathcal{U}[6, 14]$ mm |
| Heart rate | 75 bpm → $T = 800$ ms | $\mathcal{U}[60, 100]$ bpm |
| Reconstruction phase | 75% R-R | $\{40\%, 75\%\}$(systolic / diastolic) |
| Initial scan angle | 0° | $\mathcal{U}[0°, 360°]$ |

---

## Appendix B — 文件输出 layout

```
data/imagecas/
├── raw/                                # 原始下载 + 解压
│   ├── case_001/img.nii.gz
│   ├── case_001/label.nii.gz
│   └── ...
├── synth-motion/v1/
│   ├── case_001/
│   │   ├── clean.nii.gz                # = raw/case_001/img.nii.gz 拷贝
│   │   ├── corrupted_seed0.nii.gz      # variant 0
│   │   ├── corrupted_seed1.nii.gz
│   │   ├── ...
│   │   ├── heart_mask.nii.gz           # TotalSegmentator 输出
│   │   └── meta.yaml                   # motion_params + scan_params per variant
│   └── ...
└── SUMMARY.md                          # imagecas_inspect.py 自动生成
```

**meta.yaml** schema(per variant):

```yaml
case_id: "001"
seed: 0
motion_params:
  contraction_amp_mm: 12.3
  twist_amp_deg: 14.2
  long_axis_amp_mm: 9.8
  heart_rate_bpm: 78.5
  recon_phase_pct: 75.0
scan_params:
  initial_scan_angle_deg: 132.7
  rotation_period_ms: 250.0
  n_views: 1000
  sid_mm: 540.0
  sdd_mm: 950.0
git_sha: "<at synth time>"
generated_at: "2026-04-30T..."
```

每例 corrupted variant 都是 reproducible:`case_id + seed + git_sha` 唯一确定。

---

## Approval gate

**This spec needs user approval before Stage 1 coding starts**:

- [ ] Forward projection lib 选 LEAP(vs tomosipo fallback)— 同意?
- [ ] Whole-heart segmentation 用 TotalSegmentator — 同意?
- [ ] 4 分量 closed-form motion 模型(contraction + twist + long-axis + translation)— 同意?
- [ ] Per case 5-10 个 variants — 同意?
- [ ] Cone-beam scan defaults(SID=540, SDD=950, 1000 views, 250ms rotation)— 同意?
- [ ] V2 验证用 PAD demo case 40/50 视觉对照 — 同意?

approve 后 → spec → IMPLEMENTING,Day 6 coding 开始(Stage 1)。
