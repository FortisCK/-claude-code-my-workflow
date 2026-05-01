# SESSION_LOG: motion-day6

## 2026-04-30 16:30 — 启动 Day 6 demo

**前置**: ImageCAS 1000 例已全部解压（5 batch × 200 case，89 GB）；磁盘 274 GB 剩。

**目标**: 在真实 case 1 上首次跑通 `code/data/motion_synth.py` 的端到端管线，目测伪影质量。

## 16:41 — Demo v1 fail

`scripts/python/synth_motion_demo.py --case-id 1` 第一次尝试。

**错误**: `KeyError: 'heart_myocardium'` in `totalsegmentator/python_api.py`

**原因**: 新版 TotalSegmentator (>=2.x) 的 `total` task 把整个心脏合并成单个 `heart` ROI，原来的子结构（heart_myocardium / heart_atrium_left/right / heart_ventricle_left/right）挪到了独立的 `heartchambers_highres` task。

**修复**: `synth_motion_demo.py` 的 `roi_subset` 改成 `["heart"]`。对 demo 来说我们只要 bbox crop，单个 heart ROI 足够。

[LEARN:totalsegmentator] `total` task 现在心脏只暴露 `heart` 单个 ROI；要 chamber-level 需要 `task="heartchambers_highres"`。

## 16:43 — Demo v2 fail

`KeyError` 修了，但找不到分割输出文件 (`FileNotFoundError: No segmentation output in .../seg_out`)。

**原因**: `ml=True` 时 TotalSegmentator 把 `output` 参数当作**文件路径**而非目录路径。我们传了 `output=str(out_dir)` 期望它是目录，结果它把多标签 NIfTI 写到了 `_seg_out.nii`（不带 `.gz`）作为文件名，旁边的同名 `_seg_out/` 目录是空的。

**修复**: 改为 `output=str(seg_path)` 其中 `seg_path = work_dir / "seg_total.nii.gz"`，然后直接 `sitk.ReadImage(seg_path)`。

[LEARN:totalsegmentator] `ml=True` → `output` 参数必须是文件路径（推荐 `.nii.gz` 后缀）；`ml=False` → `output` 必须是目录。两种模式下 API 不一致。

## 16:43 — Demo v3 ✅ 端到端跑通

```
Loading case 1 ...                    0.7s
  shape=(275, 512, 512)  voxel=(0.5, 0.377, 0.377) mm  HU=[-1024, 3071]
TotalSegmentator (fast, gpu) ...     32.0s
  heart mask: 8.7M voxels (12.12% of volume)
Crop to heart bbox + 30mm pad ...
  cropped: (275, 512, 512) -> (275, 468, 512)
synthesize_motion_artifact() ...      4.5s
  HU calibration: scale=0.0185 offset=8.44   ⚠️ scale 异常
  V_corrupted HU: [-1718, 3090] mean=-167   (vs clean mean=-213)
Saved PNG + corrupted volume
```

**总耗时**: ~38s for 1 case（其中 32s 是 TotalSegmentator）。

## 16:44 — 视觉评估 + 三个待排查问题

PNG 三视图（axial/coronal/sagittal × clean/corrupted/diff）显示：

✅ Pipeline 工作；diff 红蓝结构集中在心肌壁/血管壁（高梯度边界处运动模糊最强 —— 物理正确）。

⚠️ **3 个让人不安的现象**：
1. 伪影偏弱（更像低通软化，没有真正运动伪影该有的条纹/双轮廓）
2. HU calibration scale=0.018（应在 [0.5, 2] 区间）
3. Sagittal diff 在肺野边缘有强条带（疑似 FBP 侧向截断）

**待诊断**: 见 `README.md` "待排查的 3 个候选 root cause" 表。

## 16:49 — 归档当前状态

PNG + corrupted + heart_mask 已移到 `output/`。
README + SESSION_LOG 写完。
等用户决定下一步:跑诊断 A/B/C,还是先看 PNG 拍板再说。

---

## 2026-05-01 — Lit-review + code update + v5 rerun

### 14:00-14:40 文献调研

读了 5 篇 paper 的 Methods 章节(`master_supporting_docs/supporting_papers/`):
- **Lossau 2019 CoMoFACT**:`d̃ = s · m_c̃(v) · δ̃(t)`,centerline 15mm dilation + 12.4mm uniform filter mask,5-point piecewise linear time profile
- **Maier 2021 Deep PAMoCo**:polyharmonic spline 3 control points,`w(r)→0` at patch boundary,**axial 70 mm/s / z 25 mm/s 速度上限**,confirmed "100k samples / 25 reconstructions"
- **Stöhr 2016 Fig 1C**:resting healthy LV peak twist **~15°**,apex rotation ~12°,base ~−5°(关键修正:之前 CoVe 给的"~7-8° ± 3°"是跨人群均值,不是单人 peak。我们的 15° 实际上没问题)

[LEARN:lit-anchor] CoVe 验证 fact-of-existence 强,但 numeric range claims 要回 PDF 看 Fig/Table 自己确认。

### 14:40 — 代码更新(Lossau/Maier alignment)

`code/data/motion_synth.py` 改三处:
1. `MotionParams.twist_amp_deg` 默认 10° → **12°**(对应 Stöhr peak 的保守值)
2. 加 `motion_strength: float = 1.0` 全局标量(Lossau s∈[0,10] convention)
3. 加 `mask_smooth_sigma_mm: float = 5.0` + `compute_motion_weight()` helper(Gaussian smooth heart_mask 边界,防止 boundary discontinuity)

`parametric_dvf` 现在接受 `motion_weight: torch.Tensor` 而非 binary `heart_mask`。

### 14:47 — v5 rerun(boundary smoothing + 新 defaults)

**配置**:CPU,n_phases=24,n_views=500(GPU 被 mcastro 的 nnUNet 占了 39 GB,只剩 2.8 GB free)
**结果**: `output_v2/case_1_motion_synth.png`

发现:

| 假设 | 验证结果 |
|------|----------|
| HU calibration scale=0.018 是 bug | **❌ 否定** — scale ∝ 1/n_views(v3 0.018 @ 1000 views,v5 0.037 @ 500 views,完美线性)。这是 FDK normalization,不是 bug。 |
| Sagittal 肺野条带是 heart_mask 边界跳变造成 | **❌ 否定** — 加了 5mm Gaussian smoothing 后条带依旧。元凶可能是 FBP lateral truncation(crop 到 468 列引入)或 n_views=500 angular aliasing。 |
| 视觉强度变化 | corrupted 比 v3 更糊,但 confound:n_phases/n_views 都减半本身就更糊。不能纯归因于参数变化。 |

[LEARN:debug] Day 6 三个假设(A FDK baseline,B DVF time-loop,C truncation)中 A 已澄清(scale 是 normalization),C 仍未排查(等 GPU 才能 apples-to-apples)。

### Pending(等 GPU 释放)

跑 apples-to-apples v6: GPU + n_phases=48 + n_views=1000 + 新 motion_synth defaults。然后才能干净地比较 boundary smoothing 是否改善了什么。
