# CATHACTION Task 1 阶段性总结汇报

日期：2026-06-04

## 1. 一句话结论

我们已经完成 CATHACTION Task 1 导管/导丝三类分割的完整实验链路，并冻结了当前最可靠的 submission candidate：**Stage9A 七模型 ensemble + flip TTA + original-space inference + 轻量 connected-component post-processing**。

当前 frozen result 在 released eval 上的官方风格指标为：

| 指标 | 数值 |
| --- | ---: |
| Mean Dice | `0.6551964758686204` |
| Label 1 Dice | `0.6339706899994402` |
| Label 2 Dice | `0.6764222617378007` |
| Animal Dice | `0.742322590929714` |
| Phantom Dice | `0.6314111646741944` |
| Mean IoU | `0.5331671049089795` |
| Pixel accuracy | `0.9948812664975007` |

后续我们尝试了多条更复杂的细结构优化路线，包括 clDice/cbDice/Hausdorff-style loss、ROI patch refinement、MSLNet-style hard post-processing、binary hard-negative refiner 等，但都没有稳定超过 Stage9A。尤其是 hard foreground filtering 虽然可以提高 binary foreground precision，但会降低 recall 并伤害官方三类 Dice。因此 Task 1 当前先冻结，接下来应进入 Task 2 collision detection。

## 2. 任务定义与难点

CATHACTION Task 1 是 X-ray fluoroscopy 中的导管/导丝工具分割任务。输入是一帧透视 X 光图像，输出是三类 mask：

- `0`: background
- `1`: label_1
- `2`: label_2

目前我们按照官方 Task 1 的 multiclass segmentation 口径评估，核心指标是 label_1 和 label_2 的平均 Dice。 released eval 共 `4691` 张：

| Domain | Samples |
| --- | ---: |
| Animal | `1006` |
| Phantom | `3685` |
| Total | `4691` |

这个任务的主要难点不是普通大区域分割，而是极细、稀疏、长线状结构分割：

- 前景像素占比极低，背景占绝大多数，因此 pixel accuracy 不是关键指标。
- 工具标注通常只有几个像素宽，1-2 像素偏移在视觉上看起来接近，但 strict Dice 会明显下降。
- `label_1` 和 `label_2` 在局部形态上容易混淆，三类分割会惩罚类别互换。
- Animal 和 phantom 域差异明显，phantom 中骨边缘、管路边缘和导丝/导管形态容易混淆。
- 一些失败样本不是完全漏检，而是线条局部偏移、断裂、类别混淆或预测线段与 GT 相邻但不重叠。

## 3. 实验路线概览

我们不是只跑了一个 baseline，而是按阶段完成了数据、模型、损失、后处理和诊断的系统探索。

| 阶段 | 目的 | 结论 |
| --- | --- | --- |
| Task1 data pipeline | 建立数据读取、split、mask encoding、metric 计算 | 已完成，可复现 |
| UNet / MONAI baseline | 跑通最小训练和评估 | 可用，但不是最强 |
| Architecture shootout | 比较 EfficientNet、MiT/SegFormer、ConvNeXt 等 | ConvNeXt/FPN 系列最稳定 |
| ConvNeXt rescue/deepening | 调整 ConvNeXt 分辨率和训练配置 | ConvNeXt tiny/small 成为主力 |
| Ensemble / TTA | 利用模型互补性提升 released eval | 形成当前 Stage9A champion |
| Fine-structure losses | 尝试 clDice、cbDice、HausdorffDTLoss 等 | 没有稳定超过 Stage9A |
| ROI / patch refinement | 尝试局部精修细线结构 | 没有稳定收益 |
| MSLNet-style diagnostic | 对齐相关论文的 binary foreground 评价 | 有诊断价值，但不能替代官方三类指标 |
| Hard gate / binary refiner | 尝试提高 precision、减少 false positives | 提高 precision 但降低 recall，官方 Dice 下降 |

当前最重要的经验判断是：**这个任务上最稳定的收益来自 recall-preserving ensemble，而不是 aggressive hard foreground deletion。**

## 4. 当前最佳方案：Stage9A

Stage9A 是当前 frozen Task 1 candidate。

方法概括：

> Stage9A = seven-model multiclass ensemble + horizontal flip TTA + original-space inference + light connected-component filtering.

Stage9A raw ensemble 使用七个模型，权重如下：

| Model name | Config | Weight | Resize mode | Image size |
| --- | --- | ---: | --- | --- |
| `convnext640` | `configs/task1/smp_fpn_convnext_tiny_640_rescue_stable.yaml` | `0.26` | aspect pad | `640 x 640` |
| `efficientnet_b3` | `configs/task1/smp_unet_efficientnet_b3_512_shootout.yaml` | `0.26` | direct | `512 x 512` |
| `convnextv2_base` | `configs/task1/smp_fpn_convnextv2_base_512_stage4a.yaml` | `0.0936` | direct | `512 x 512` |
| `convnext_small512` | `configs/task1/smp_fpn_convnext_small_512_stage4a.yaml` | `0.0936` | direct | `512 x 512` |
| `convnext_small640` | `configs/task1/smp_fpn_convnext_small_640_stage5.yaml` | `0.0928` | aspect pad | `640 x 640` |
| `convnext_small640_cldice` | `configs/task1/smp_fpn_convnext_small_640_cldice_stage6.yaml` | `0.1` | aspect pad | `640 x 640` |
| `convnext_small640_toolness` | `configs/task1/smp_fpn_convnext_small_640_toolness_aux_stage7.yaml` | `0.1` | aspect pad | `640 x 640` |

TTA:

- `none`
- `hflip`

Post-processing:

- Stage9A raw predictions 后接轻量 component filtering。
- 当前使用 `remove_small_min32`，即移除小于 32 像素的小 connected components。
- 不做 aggressive erosion/gating，不改变 Stage9A 的主三类预测逻辑。

最终 prediction manifest：

- `outputs/task1/stage9a_seven_model_add_both_010_010/remove_small_min32/predictions.csv`

最终 eval JSON：

- `outputs/task1/stage9a_seven_model_add_both_010_010/remove_small_min32/eval.json`

## 5. Stage9A 结果

Stage9A 在 released eval 上的官方风格结果：

| 指标 | 数值 |
| --- | ---: |
| Mean Dice | `0.6551964758686204` |
| Mean IoU | `0.5331671049089795` |
| Pixel accuracy | `0.9948812664975007` |
| Label 1 Dice | `0.6339706899994402` |
| Label 2 Dice | `0.6764222617378007` |
| Label 1 IoU | `0.47901747544174483` |
| Label 2 IoU | `0.587316734376214` |

按 domain 分：

| Domain | Dice | IoU | Label 1 Dice | Label 2 Dice | Samples |
| --- | ---: | ---: | ---: | ---: | ---: |
| Animal | `0.742322590929714` | `0.631325393449197` | `0.7561582529077036` | `0.7284869289517244` | `1006` |
| Phantom | `0.6314111646741944` | `0.5063700253237802` | `0.600613651115936` | `0.6622086782324527` | `3685` |

主要观察：

- Animal 域明显强于 phantom 域。
- Phantom 是当前主要瓶颈，尤其 `label_1` Dice 较低。
- `label_2` 总体略强于 `label_1`，但 worst-case 中仍存在 `label_2` 完全错失或类别互换。
- Pixel accuracy 很高，但该任务中背景极多，不能作为主要判断依据。

## 6. 结果图与误差分析

我们已经生成了 Stage9A 的最终可视化结果，供导师汇报和后续 method report 使用。

代表性 overlay：

- `outputs/task1/stage9a_final_figures/representative_overlays/`

Worst-case error analysis：

- `outputs/task1/stage9a_final_figures/error_analysis/`

主要 contact sheets：

- `outputs/task1/stage9a_final_figures/error_analysis/worst_overall/contact_sheet.png`
- `outputs/task1/stage9a_final_figures/error_analysis/worst_label_1/contact_sheet.png`
- `outputs/task1/stage9a_final_figures/error_analysis/worst_label_2/contact_sheet.png`
- `outputs/task1/stage9a_final_figures/error_analysis/worst_animal/contact_sheet.png`
- `outputs/task1/stage9a_final_figures/error_analysis/worst_phantom/contact_sheet.png`

这些图每个样本包含四列：

- 原图
- GT overlay
- prediction overlay
- error overlay

Worst-case 中最典型的错误包括：

- 预测线和 GT 线相邻但没有严格重叠。
- 局部方向基本正确，但有 1-3 像素偏移。
- 红蓝类别互换或局部错分。
- Phantom 中出现与骨边缘、器械边缘或其他线状结构混淆。
- 少数样本存在较大漏检，导致 single-sample Dice 接近 0。

这些错误解释了为什么肉眼看起来“还可以”的预测，在 strict pixel Dice 下分数仍然不高。

## 7. MSLNet 对比：有诊断价值，但不是最终优化目标

我们读了 MSLNet 论文，并实现了 MSLNet-style diagnostic metrics，用于分析工具整体 foreground 的定位和 tolerance-based F1。

但需要明确：**MSLNet 的主要 CathAction 对比更接近 binary foreground segmentation，而官方 Task 1 是 label_1 / label_2 multiclass segmentation。**

两者差异：

| 评价口径 | 关注点 | 是否惩罚 label_1 / label_2 互换 |
| --- | --- | --- |
| Official Task 1 style | label_1 和 label_2 的 multiclass Dice | 是 |
| MSLNet-style diagnostic | foreground vs background，以及 r=3 容忍 F1 | 否，或惩罚较弱 |

因此，MSLNet-style metric 可以帮助诊断 “工具整体是否定位到”，但不能替代官方 ranking metric。

我们的关键对比：

| System | Official multiclass Dice | MSLNet-style F1 r=3 | 结论 |
| --- | ---: | ---: | --- |
| Stage9A | `0.655196` | `0.911578` | 当前 official champion |
| Stage9D hard gate | `0.645414` | `0.916587` | binary F1 小幅上升，但 official Dice 下降 |
| MSLNet paper reference | binary task | `0.9305` | 不可直接等价比较 |

Stage9D 的经验尤其重要：

- Precision r=3 从 `0.866395` 提高到 `0.903981`。
- Recall r=3 从 `0.967729` 降到 `0.932747`。
- Official Dice 从 `0.655196` 降到 `0.645414`。

这说明 hard gate / hard foreground deletion 会删掉真实细线，尤其伤害 `label_2` 和 phantom 域。最终我们决定不继续沿这个方向优化。

## 8. 为什么当前冻结 Stage9A

冻结 Stage9A 不是因为它已经理论最优，而是因为目前继续盲目探索 Task 1 的边际收益和风险不成比例。

我们已经验证的事实：

- 单模型架构替换没有明显超越 ensemble。
- 更复杂的 transformer/SegFormer 类路线没有直接秒杀 ConvNeXt/FPN。
- 细结构 loss 在这个数据设置下没有稳定带来正收益。
- ROI patch refinement 没有稳定改善。
- Hard post-processing 和 binary refiner 会牺牲 recall，进而伤害官方 Dice。

因此当前更合理的策略是：

- Stage9A 作为 Task 1 frozen candidate。
- 不再做大规模 Task 1 架构探索。
- Task 1 只保留低风险返工窗口，例如 Docker packaging、inference entrypoint、submission format、报告图表和 method text。
- 主要研发精力转到 Task 2。

## 9. 当前保留的复现链路

Stage9A final predictions：

- `outputs/task1/stage9a_seven_model_add_both_010_010/remove_small_min32/predictions.csv`

Stage9A final eval：

- `outputs/task1/stage9a_seven_model_add_both_010_010/remove_small_min32/eval.json`

Stage9A raw ensemble summary：

- `outputs/task1/stage9a_seven_model_add_both_010_010/raw_predictions/summary.json`

Stage9A post-processing summary：

- `outputs/task1/stage9a_seven_model_add_both_010_010/remove_small_min32/summary.json`

七个 `best_checkpoint.pt`：

- `outputs/task1/smp_fpn_convnext_tiny_640_rescue_stable/best_checkpoint.pt`
- `outputs/task1/smp_unet_efficientnet_b3_512_shootout/best_checkpoint.pt`
- `outputs/task1/smp_fpn_convnextv2_base_512_stage4a/best_checkpoint.pt`
- `outputs/task1/smp_fpn_convnext_small_512_stage4a/best_checkpoint.pt`
- `outputs/task1/smp_fpn_convnext_small_640_stage5/best_checkpoint.pt`
- `outputs/task1/smp_fpn_convnext_small_640_cldice_stage6/best_checkpoint.pt`
- `outputs/task1/smp_fpn_convnext_small_640_toolness_aux_stage7/best_checkpoint.pt`

最终图：

- `outputs/task1/stage9a_final_figures/`

Task 1 final wrap-up：

- `quality_reports/decisions/2026-06-04_task1_final_wrapup.md`

清理记录：

- `quality_reports/decisions/2026-06-04_task1_cleanup_candidates.md`

## 10. 清理与当前状态

我们已经执行了保守 cleanup，删除失败路线、smoke/debug、旧 baseline、Stage8/9B/9C/9D 输出等。

清理前后：

| Path | Size |
| --- | ---: |
| `outputs/task1` before cleanup | `9.9G` |
| `outputs/task1` after cleanup | `2.7G` |

保留内容：

- Stage9A predictions
- Stage9A final figures
- Stage9A seven champion model directories
- diagnostics
- morphology-grid records
- all datasets

未执行：

- 没有删除 `datasets/video_action_understanding/`。
- 没有删除 retained champion 目录里的 `checkpoint.pt`。

当前数据占用：

| Path | Size |
| --- | ---: |
| `outputs/task1` | `2.7G` |
| `datasets/segmentation` | `31G` |
| `datasets/collision_detection` | `4.4G` |
| `datasets/video_action_understanding` | `54G` |

## 11. 可以给导师直接讲的口头版本

> 我们已经把 Task 1 跑到了一个稳定可复现的阶段。这个任务是 X-ray fluoroscopy 中的导管/导丝三类分割，目标很细，前景稀疏，所以 strict Dice 对像素偏移非常敏感。我们系统比较了 UNet、EfficientNet、SegFormer/MiT、ConvNeXt/FPN、ConvNeXtV2，以及 clDice、cbDice、Hausdorff-style loss、ROI refinement 和 MSLNet-style hard post-processing。当前最好的结果是 Stage9A，一个七模型 ensemble，使用 flip TTA、original-space inference 和轻量 connected-component filtering。在 released eval 上 mean Dice 是 0.6552，animal Dice 是 0.7423，phantom Dice 是 0.6314。
>
> 后续更复杂的细结构 loss 和 patch refinement 没有稳定超过 Stage9A。我们还对齐了 MSLNet-style binary diagnostic，发现 hard gate 可以提升 binary foreground precision，但会明显降低 recall，并使官方三类 Dice 从 0.6552 降到 0.6454。因此我们决定不追不完全一致的 binary metric，而是以官方 multiclass Dice 为准，冻结 Stage9A 作为当前 Task 1 candidate。
>
> 现在 Task 1 已经完成了预测、评估、结果图、error analysis 和保守清理。下一步建议切到 Task 2 collision detection，同时保留 Task 1 的 Docker/inference packaging 和 method report 工作。

## 12. 导师可能会问的问题

### Q1. `0.655` 的 Dice 是不是低？

不能只按普通大区域分割理解。这里是 3-5 像素宽的细线结构，1-2 像素偏移会显著伤害 strict Dice。很多 worst-case 图显示预测和 GT 在视觉上相邻，但像素不重叠，Dice 会很低。另外这是三类 Dice，label_1/label_2 互换会被惩罚。

### Q2. 为什么不继续调 Task 1？

我们已经尝试过主要方向：架构替换、ensemble、TTA、细结构 loss、ROI refinement、MSLNet-style postprocess 和 binary hard-negative refiner。当前只有 ensemble 和轻量 component filtering 稳定有效。继续盲目探索会占用 Task 2 时间，且存在伤害 frozen candidate 的风险。

### Q3. 为什么 ConvNeXt/FPN 表现最好？

ConvNeXt/FPN 在这个任务上提供了较强的局部纹理建模和多尺度特征融合，同时比纯 transformer 类模型更稳定。细线结构同时需要低层边缘信息和高层上下文，FPN-style decoder 与 ConvNeXt encoder 的组合在我们的 architecture shootout 中最可靠。

### Q4. MSLNet 和我们谁更好？

不能直接横比。MSLNet 的主要 CathAction 数字更接近 binary foreground segmentation 和 radius-tolerant F1，而官方 Task 1 是 label_1/label_2 multiclass Dice。我们实现了 MSLNet-style diagnostic，发现我们的 recall 很高但 precision/grouping 不如 MSLNet。尝试 hard gate 后 binary F1 小幅上升，但 official Dice 下降，所以没有采用。

### Q5. 目前 Task 1 还缺什么？

主要不是继续训练，而是 submission engineering：

- stable inference entrypoint
- Docker-compatible prediction flow
- final method report text
- final submission format validation

## 13. Task 2 过渡计划

Task 1 冻结后，下一阶段进入 Task 2 collision detection。建议按以下顺序推进：

1. 数据结构检查：
   - inventory `datasets/collision_detection/`
   - 明确 frame/video/case 标识
   - 明确 label 格式和类别定义

2. 指标对齐：
   - 官方 Task 2 primary metric 是 mAP
   - 先写本地 evaluation script 或确认已有脚本
   - 明确 AP/mAP 的 IoU/temporal/object matching 口径

3. Baseline：
   - 先跑一个可靠、简单、可复现的 baseline
   - 优先保证 split discipline 和 metric fidelity

4. 模型路线：
   - 如果标签是 frame-level collision detection，先做 image/frame classification baseline
   - 如果需要定位或时序信息，再扩展到 video/temporal model
   - 逐步考虑 optical flow、clip-level transformer、temporal smoothing 或 case-level consistency

5. 汇报节奏：
   - 先给出 Task 2 dataset schema + first baseline
   - 再决定是否需要复杂时序模型

## 14. 最终状态

Task 1 当前状态：

- Stage9A frozen。
- Released eval metrics confirmed。
- Result figures generated。
- Failed routes cleaned。
- Reproducibility chain preserved。
- Ready to move to Task 2.

