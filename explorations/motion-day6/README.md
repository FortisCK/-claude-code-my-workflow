# motion-day6: Parametric DVF + Cone-Beam Motion Synthesis (First Real-Data Run)

## Goal
首次在真实 ImageCAS CCTA 数据上跑通 `code/data/motion_synth.py` 的端到端管线（参数化 DVF + tomosipo cone-beam 投影 + Parker short-scan FBP），定位是否需要调参 / 补 component / 修 bug，再决定是否进入 Week 3-4 KL-VAE 预训练。

## Status
IN PROGRESS（started 2026-04-30）

## Hypotheses to Test (验证的) + 排查清单 (尚未验证)
1. ✅ Pipeline 端到端可跑通（load → segment → crop → synth → save）
2. ✅ TotalSegmentator 心脏分割接入 fast=True 模式可用
3. ✅ 计算预算可接受（一例 ~31s 总，其中合成本身 4.5s）
4. ⚠️ HU 校准 scale=0.018 异常 —— 需诊断 FDK 重建单位
5. ⚠️ 视觉伪影偏弱 —— 需诊断 DVF 是否真的进入 cone-beam 时变循环
6. ⚠️ Sagittal 肺野边缘强 diff —— 需诊断 lateral truncation

## Success Criteria
- 走通端到端 ✅
- 视觉上 corrupted vs clean 有可辨别的运动条纹 / 双轮廓 / 边缘涂抹（**未达成**，目前更像低通软化）
- HU calibration scale ∈ [0.5, 2.0]（**未达成**，0.018）
- diff 集中在心脏 ROI 而非全场（部分达成；axial 达标，sagittal 肺野有强响应）

## Findings (2026-04-30)

### 已验证
- **Case 1 数据**：shape (275, 512, 512)，voxel (0.5, 0.377, 0.377) mm，HU [-1024, 3071]
- **Heart mask**：8.7M voxels = 12.12% 体积（合理）
- **Crop**：(275, 512, 512) → (275, 468, 512)（pad 30mm 后心脏 bbox）
- **Synthesis**：4.5s（48 phases × 1000 views × 275×468×512 = 强于预期）
- **HU 输出范围**：corrupted [-1718, 3090] mean=-167 vs clean [-1024, 3071] mean=-213
  - 负侧拓宽 700 HU 物理上合理（软组织/空气界面运动模糊产生暗带）
  - 但 mean 漂移 +46 HU 暗示有系统偏置

### 视觉评估（PNG: `output/case_1_motion_synth.png`）
- **Axial**：clean 心腔/血管清晰；corrupted 边缘略软；diff 红蓝集中在心肌壁、血管壁 → 物理位置正确
- **Coronal**：clean LV 腔清晰；corrupted 软化；diff 在心脏轮廓有明显结构
- **Sagittal**：⚠️ diff 在肺野左右两侧有显著条带 —— **疑似 FBP lateral truncation artifact**

### 待排查的 3 个候选 root cause

| 假设 | 诊断方法 | 预期信号 |
|------|---------|---------|
| **A**: FDK 单位/HU 不一致（与运动无关的 baseline drift） | `--n-phases 1` 跑一遍（无运动） | 若 diff 仍大 → FDK 自身就和 sitk-loaded HU 对不齐 |
| **B**: DVF 没真正进 cone-beam 时变循环 | 把 contraction_amp 拉到 30mm + twist 拉到 45° | 若伪影几乎不变 → time-varying loop 退化成单 phase |
| **C**: Lateral truncation（FBP 对侧向截断敏感） | 关掉 crop 跑全 512×512 | 若 sagittal 边缘条带消失 → truncation 是元凶 |

## Files in this folder

```
output/
├── case_1_motion_synth.png         # 三视图 clean/corrupted/diff 对比图
├── case_1_corrupted.nii.gz         # 合成的运动伪影 CCTA 体积（224 MB）
└── heart_mask_1.nii.gz             # TotalSegmentator 输出的心脏 mask（265 KB）
scripts/                            # （后续诊断脚本放这里）
README.md                           # 本文件
SESSION_LOG.md                      # 时间线 + 决策记录
```

## Timeline
- 2026-04-30: 启动;ImageCAS 1000 例解压完毕;Day 6 demo v3 跑通;发现 3 个待排查问题
- 2026-05-01: 读 Lossau 2019 / Maier 2021 / Stöhr 2016 PDF;motion_synth.py 加 motion_strength + mask boundary smoothing + twist 默认值改 12°;v5 rerun(CPU 上 reduced sampling,GPU 被占)。发现:HU scale 异常是 FDK normalization 不是 bug;sagittal 条带不是 boundary 跳变(加了 smoothing 仍存在,可能是 FBP truncation)。等 GPU 释放跑 v6 apples-to-apples 验证。

## Updated Findings (2026-05-01)

### 已澄清
- **HU calibration scale 异常 ✅ NOT a bug**:scale ∝ 1/n_views(v3 0.0185 @ 1000 views,v5 0.0370 @ 500 views,严格线性)。FDK normalization 在 calibration step 自动补偿。
- **Twist 参数 anchor 修正**:Stöhr 2016 Fig 1C 显示 resting healthy peak twist **~15°**(我们的原值)。CoVe 之前说的"~7-8°"是跨人群均值,不是 peak。所以 default 12° 偏保守,可调到 15°,training 区间 U(8°, 25°)。

### 仍未澄清
- **Sagittal 肺野条带源头**:不是 heart_mask boundary 跳变(加 5mm smoothing 后仍在)。最可能元凶:FBP lateral truncation(crop 引入)或 angular aliasing。
- **视觉强度评估 (apples-to-apples)**:v5 在 reduced sampling 下跑,n_phases/n_views 减半本身就让 corrupted 更糊,无法干净归因于代码改动。

### 下一步
1. 等 GPU 释放(mcastro nnUNet 训完)
2. 跑 v6:n_phases=48 + n_views=1000 + 新 motion_synth(motion_strength=1.0,boundary smoothing 5mm,twist 12°)→ 和 v3 干净对比
3. 如果 sagittal 条带仍在 → 排查 FBP truncation(扩大 reconstruction FOV / 不 crop / 加 sinogram zero-padding)
