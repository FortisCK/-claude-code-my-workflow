# CATHACTION 项目状态合并报告 — Task 1 与 Task 2

## 1. 文档元信息

- **用途 (Purpose):** 这是一份**内部权威参考文档**,合并覆盖 CATHACTION Task 1(分割)与 Task 2(碰撞检测)的全部当前状态、方法演进、最佳结果、可复现性与提交打包情况。用户将基于本文件撰写正式的 MICCAI method report,因此本文档要求**每一个数字都可溯源**到具体的 file / checkpoint / decision record。
- **日期 (Date):** 2026-06-26
- **反映的提交 (Commit):** HEAD = `4fd7bad`("Task 2 base-detector exploration + EFF-tier calibration (post-Tier-A)"),其前一提交为 `a630bc2`("Tier A pre-validation hardening + version-control the participant codebase")。当前分支 `task2-tier-a-prevalidation-hardening`。
- **Source of truth 声明:** 以 **2026-04-22 CATHACTION MICCAI challenge PDF** 为权威来源(INV-1),直到官方 2026 challenge 网站/平台正式取代它为止。公开网站 (airvlab.github.io/cathaction) 仍显示旧计数(~500k frames / ~25k masks,更新于 2024-11-28),**不得**与 PDF 计数混用。
- **取代声明:** 本文档**取代过时的 2026-06-14 Task 2 状态报告**(`quality_reports/reports/2026-06-14_task2_current_status*.md`),后者是 pre-calibration 视角,其绝对 mAP 数字已被 2026-06-15 honest-baseline 与 2026-06-17/18 的 EFF-gap calibration / 多视频面板 / YOLOV-reranker-swap 工作重新定性。

---

## 2. 执行摘要 (Executive Summary)

**Task 1(分割,DSC primary):** 当前 frozen 冠军是 **Stage9A 七模型 softmax ensemble + horizontal-flip TTA + 原图尺度推理 + `remove_small_min32` 连通域后处理**,在内部 released-eval(frame-level,N=4691,仅 animal+phantom,**非** hidden test)上达到 **mean Dice = 0.6552**(精确值 0.6551964758686204;label_1/catheter 0.6340,label_2/guidewire 0.6764,mIoU 0.5332,pixel accuracy 0.9949)。分域:animal 0.7423 / phantom 0.6314,phantom + label_1 是主要瓶颈。所有数字均标记为 **PROVISIONAL**,等待官方 validation package(预计 2026-07-10)。

**Task 2(碰撞检测,mAP primary):** 经过 honest-baseline 与 EFF-gap calibration 校准后,当前**最佳已验证配置**是 **raw YOLOV-S two-class (mv split, 576px) + 既有 Stage2U reranker (Arm A,OOD diverse-trained checkpoint,无重训)= per-case mAP50 = 0.278 ± 0.225**(global 0.256,per-case mAP50-95 0.103;raw YOLOV baseline 0.252),在 6-video full-frame-GT phantom panel(6489 GT)上测得。核心结论 (**EFF-tier conclusion**):此前"我们落后 EFF 约 2x"是一个 **eval-ruler 测量伪影**——通过在同一把尺子上复现 YOLOV(0.131 ≈ champion Stage2AQ 0.130),换算公式 `EFF_ours = 1.0546 × Y` 显示 champion 在 mAP50 上基本**与 EFF 同档**、在 mAP50-95 上甚至高出约 22%。所有探索过的方法(Stage2AQ、reproduced YOLOV、收敛后的 RT-DETRv2、YOLOV+reranker)都**收敛到同一个 "EFF tier"**;要真正突破需要域数据或研究级范式转变,而非继续调参。

---

## 3. 挑战赛背景与约束

| 项目 | 工作事实 | 溯源 |
| --- | --- | --- |
| 主办 | MICCAI 2026,Université de Rennes | README.md / CLAUDE.md |
| Task 1 | X-ray fluoroscopy 中 catheter + guidewire 分割 | README L17;INV-7 |
| Task 1 metrics | **DSC** primary;IoU/Jaccard、mIoU、pixel accuracy secondary | CLAUDE.md Facts;INV-7 |
| Task 2 | endovascular intervention 碰撞检测 | README L18;INV-8 |
| Task 2 metrics | **mAP** primary;AP secondary | CLAUDE.md Facts;INV-8 |
| 数据规模 | **650** videos/cases;约 **40k** segmentation frames;约 **600k** collision-label frames | README L22-25;CLAUDE.md Facts |
| 三个域 (Domains) | silicon vascular phantom、preclinical animal X-ray、real human X-ray | README L23 |
| Split | **70% train / 15% validation / 15% hidden test**,在 **procedure/case level** 划分 | README L26;INV-2 |
| 提交形式 | **Docker container + predefined result file + 简短 method description/report**;推理全自动、无人工交互 | README L27;INV-6 |
| 数据政策 | 仅允许在 challenge launch 时**公开可得**的外部数据/标注/预训练模型/prompts;禁止 private/proprietary 临床数据(INV-4);禁止提交 PII/凭证/私有下载链接(INV-5);hidden-test labels 仅供评测,不得 hand-tuning/逆向(INV-3) | content-invariants.md |
| **关键日期** | 官方 **validation set + evaluator 发布:2026-07-10**(binding revisit trigger);**提交截止:2026-08-23** | prevalidation_roadmap.md L1/L18;README L28 |

> **注意:** prompt 中提到的命名 license `CC-BY-NC-SA` 在仓库版本控制文件中**未逐字出现**;仓库仅以 INV-4(public-only external data)+ INV-5(no PII)表达数据政策。该 license 归属需对照 2026-04-22 PDF 核实(本轮未定位该 PDF 原文)。另注:Task 1 method-report draft(`2026-06-15_task1_method_report_draft.md` L54)已将 `CC-BY-NC-SA challenge data` 当作既定事实陈述——该 draft 表述偏强,对外报告前须与 PDF 对齐,避免两份工件 drift。

---

## 4. 仓库与可复现性状态

- **分支/提交/remote:** 当前分支 `task2-tier-a-prevalidation-hardening`,HEAD = `4fd7bad`(post-Tier-A);前序 `a630bc2`(Tier A)、`eb5862c`(workflow 配置)。remote `origin = https://github.com/FortisCK/-claude-code-my-workflow.git`(这是上游 workflow-template 仓库,CATHACTION 工作位于其分支上)。分支**已推送**(origin 上 `refs/heads/task2-tier-a-prevalidation-hardening` SHA = 4fd7bad,与本地 HEAD 一致),但本地分支**未配置 upstream tracking ref**。实时 working tree 为 **clean**(session-start 快照中列出的 modified/untracked 文件已并入 4fd7bad,该快照已过时)。

- **INV-2 leak 修复故事:** Tier A 多 agent pre-validation sweep(41 agents,8 维度)发现 Task 2 split 存在 **INV-2(case/procedure-level split integrity)泄漏**。`scripts/task2/prepare_yolo_splits.py`(注意实际文件名是 `prepare_yolo_splits.py`,**非** `prepare_yolov_splits.py`)原先按 **frame name** 去重 train-vs-valid,而非按 **video id**。由于 `valid_phantom.txt` 恰好是单一视频 `video_0`,而 `train_phantom.txt` 也含 `video_0`,导致 **5 个 phantom video_0 帧**(frames 64/66/176/189/237)残留在 train。量级 **5 / 35,788 = 0.01%**,**未实质性抬高结果**(真正的 Task 2 弱点——单视频 phantom validation、selection-on-eval、animal-dominated headline——是独立的 Tier B 问题)。修复:dedup 从 frame-level 改为 **video-level**(凡其整段视频在 validation 中的 train 帧全部丢弃),并加入硬 `AssertionError`(若 `train_clean ∩ valid` video ids ≠ ∅),在 `summary.json` 写入 `video_level_disjointness` 块,并新增可复用审计器 `scripts/task2/audit_split_disjointness.py`(检测到重叠则非零退出,docstring 引用 INV-2)。所有派生 split 已重生成并审计 PASS。

- **per-case mAP evaluator:** `compute_detection_map_per_case`(`src/cathaction/metrics/detection.py:215`)在每个 case(video/procedure)内部计算 mAP 再对 case 等权 macro-average,匹配 PDF 措辞 "averaging AP scores across all test cases";与之对比的 `compute_detection_map` 是全局 box pooling(frame-count weighted)。两种读法都保留至官方 evaluator 发布;divergence 单测见 `tests/test_task2_per_case_map.py`。无 GT 的 case 被跳过。

- **Tier A 的 9 项(全部 2026-06-15 完成并验证):** (1) video-level split guard + assert;(2) disjointness auditor + tests;(3) per-case AP evaluator + divergence test;(4) metric-interpretation decision + tie-break tree;(5) Task1 no-GT entrypoint + preflight + weight manifest;(6) symmetric candidate freeze(Stage2V/AQ/AI);(7) Task2 deps env + license manifest;(8) Dockerfile + .dockerignore + entrypoint(**BUILD UNVERIFIED**);(9) Task1 method-report draft。来源:`quality_reports/plans/2026-06-15_prevalidation_roadmap.md` L5-14。

- **YOLOV reproducibility pin:** 官方 YOLOV 仓库位于 `external_repos/YOLOV`(**gitignored**,vendored 第三方代码)。仓库仅版本控制我方修改于 `external_repo_patches/YOLOV/`:README 锁定上游 commit `fe777cbeeff92d3340d0424a83a6a0e906b97f17`(`fe777cb`,Apache-2.0)、`cathaction_yolov.patch`(97 insertions / 13 deletions,跨 4 个上游文件)、以及 `exps/cathaction/` 下 4 个 experiment configs。复现需重新 clone pinned commit 并 apply patch + 复制 exp configs(无上游代码的 committed 副本)。

- **测试状态:** 文档化的全套结果为 **152 passed, 1 skipped, 1 pre-existing failure**;唯一失败 `test_task1_mslnet_style_metrics`(`evaluate_mslnet_style.py:190` 的 `f1_r2` KeyError,日期 2026-06-03,与 Tier A 无关)。本轮(2026-06-26)由本助手亲自实跑确认 Tier A split/per-case 测试:**13 passed in 4.14s**(`test_task2_split_disjointness.py`、`test_task2_dataset.py`、`test_task2_per_case_map.py`,env `cathaction-task1`)。Task 2 子套件历史增长 105 → 113 passing。

---

## 5. Task 1 — 分割

### 目标与数据

X-ray fluoroscopy 中 catheter 与 guidewire 的多类分割,主指标 DSC。所有 Task 1 数字均为**内部 released-eval、frame-level、N=4691(仅 animal+phantom)**,**非** hidden test。released-eval 不含 **human** 域(human masks 是 binary PNG 0/255,无法直接进入 3-class evaluator),因此 human 性能从未在 champion 上评分。catheter=label_1 / guidewire=label_2 映射为 **PROVISIONAL**(由形态推断,无官方数值 label map;见 `2026-05-20_task1_label_semantics.md`,但 method report draft 已将其当作事实陈述,对外应软化为 provisional)。

### 方法管线

- **Architecture:** ConvNeXt / EfficientNet encoders 配 FPN decoder(EffB3 成员用 Unet),全部为 `segmentation_models_pytorch` + timm ImageNet-pretrained encoders,输出 raw logits。Stage-2 architecture shootout 选定 ConvNeXt/FPN 最稳、EfficientNet-B3/Unet 互补;SegFormer/MiT-B2 与 ResNet50 较弱。
- **Loss:** MONAI DiceCE(3-class 0/1/2)为主干,外加两个互补 auxiliary 成员:soft-clDice 与 binary "toolness" auxiliary head。
- **Postprocess:** 仅小连通域移除(`remove_small_min32`,丢弃 <32 px 连通域);**无** dilation/erosion/thinning/gating(硬形态学与 gating 经测试因损害 recall 与 official multiclass Dice 被否决)。
- **Inference:** 逐模型 loader(model 2 跳过 ImageNet normalization),softmax probs 映射回原图尺寸 → 加权求和 → TTA = mean{identity, hflip} → argmax → 丢弃 <32 px 连通域。

### 阶段演进

| Stage | 新增内容 | released-eval mean Dice | 状态 |
| --- | --- | --- | --- |
| anchor | MONAI UNet | ≈0.530 | superseded |
| Stage2 | EfficientNet-B3/Unet 单模型 | 0.6270 | superseded |
| Stage2 | ConvNeXt-Tiny/FPN "rescue"(ImageNet norm, lr 1e-4, wd 0.05, AMP off, grad-clip 1.0,修复 NaN 0.4802) | 0.6311 | superseded |
| Stage3A | 双模型 ConvNeXt512+EffB3 0.5/0.5 + hflip | 0.6419 | superseded |
| Stage3B | ConvNeXt-Tiny **640** + EffB3 0.5/0.5 hflip(two-model champion) | 0.6439 | superseded |
| Stage5 | four-model → five-model(+ConvNeXt-Small 640)→ weight-refined five | 0.6480 → 0.6501 → 0.6517 | superseded |
| Stage6 | + clDice 成员(standalone 0.6337,作尾部 +0.0007) | 0.6524(six-model) | superseded |
| Stage7 | + toolness-aux 成员(standalone 0.6380);**seven-model `add_both_010_010` raw** | 0.6537(精确 0.6536723455148790) | superseded(raw) |
| **Stage9A** | morphology grid:`remove_small_min32` 提升 raw +0.0015 | **0.6552(精确 0.6551964758686204)** | **FINAL CHAMPION** |

**失败/否决路线(均未超过 Stage9A,已记录):** Stage8(warm-started ROI/patch refiner,standalone ~0.565,subset ROI 仅 ~+0.002);Stage9B(cbDice + HausdorffDTLoss,仅 setup/smoke,无稳定胜出,provenance 偏弱);Stage9C(MSLNet-style threshold/erosion/prob-gate,F1 r3 ~0.9175 但 Dice 跌至 0.6167,低于 MSLNet 0.9305);Stage9D(binary hard-negative refiner gate t=0.65,precision r3 0.8664→0.9040 但 recall 0.9677→0.9327,official Dice **0.6552→0.6454**)。诊断结论:0.65 regime 是 **thin-line exactness** 问题(exact 0.6517 → tol r1 0.7969 / r2 0.8777 / r3 0.9201,oracle class-swap 仅 +0.0014);增益来自 recall-preserving ensembling + 轻量连通域过滤,而非激进的前景删除。

### 最终冠军与指标

**Stage9A 七模型 softmax ensemble + hflip TTA + 原图尺度推理 + `remove_small_min32`。** 七个成员(权重 re-normalize 至和为 1,即 Stage7 `add_both_010_010` 权重向量):

| # | 成员 | 权重 |
| --- | --- | --- |
| 1 | FPN/convnext_tiny v1, 640 aspect_pad, imagenet-norm | 0.260 |
| 2 | Unet/efficientnet-b3, 512 direct,**跳过 imagenet-norm** | 0.260 |
| 3 | FPN/convnextv2_base, 512 direct | 0.0936 |
| 4 | FPN/convnext_small v1, 512 direct | 0.0936 |
| 5 | FPN/convnext_small, 640 aspect_pad | 0.0928 |
| 6 | FPN/convnext_small 640 + clDice | 0.10 |
| 7 | FPN/convnext_small 640 + toolness-aux(softmax 前切掉第 4 通道) | 0.10 |

| metric | value | split | source |
| --- | --- | --- | --- |
| mean Dice | **0.6551964758686204** | released-eval N=4691 | `outputs/task1/stage9a_seven_model_add_both_010_010/remove_small_min32/eval.json`(已直接读盘核对) |
| label_1 (catheter) Dice | 0.6339706899994402 | 同上 | eval.json |
| label_2 (guidewire) Dice | 0.6764222617378007 | 同上 | eval.json |
| mIoU / IoU(foreground macro,bg 排除) | 0.5331671049089795 | 同上 | eval.json |
| label_1 IoU | 0.47901747544174483 | 同上 | eval.json |
| label_2 IoU | 0.587316734376214 | 同上 | eval.json |
| pixel accuracy | 0.9948812664975007 | 同上 | eval.json |
| animal-domain Dice | 0.742322590929714(IoU 0.631, n=1006) | released-eval animal | eval.json by_domain |
| phantom-domain Dice | 0.6314111646741944(IoU 0.506, n=3685) | released-eval phantom | eval.json by_domain |

> 该 evaluator 的 "mIoU"(0.5332)= "IoU",是对 labels 1,2 的 foreground macro IoU(**background 排除**),并非含 background 的 3-class mIoU。
>
> MSLNet-style binary 评测(非官方 3-class,不可直接比较):binary Dice 0.6216 / IoU 0.4605 / AHD 1.371 / F1 r3 0.9116 / precision r3 0.8664 / recall r3 0.9677,vs MSLNet 论文 Dice 0.6251 / F1 r3 0.9305(`2026-06-03_task1_mslnet_metric_alignment_result.md`)。

**Frozen 工件:** `outputs/task1/stage9a_seven_model_add_both_010_010/remove_small_min32/{predictions.csv,eval.json,summary.json}`;raw ensemble `.../raw_predictions/`;七个 `best_checkpoint.pt`(~1.27 GB,114M/51M/343M/197M/197M/197M/197M)位于 `outputs/task1/{smp_fpn_convnext_tiny_640_rescue_stable, smp_unet_efficientnet_b3_512_shootout, smp_fpn_convnextv2_base_512_stage4a, smp_fpn_convnext_small_512_stage4a, smp_fpn_convnext_small_640_stage5, smp_fpn_convnext_small_640_cldice_stage6, smp_fpn_convnext_small_640_toolness_aux_stage7}/`。冠军定义脚本 `scripts/task1/run_stage9a_seven_model_postprocess.sh`。

### 提交打包状态

提交路径为三文件链,根为 `scripts/task1/run_task1_submission_inference.py`。该 entrypoint 定义唯一 source of truth `CHAMPION_MODELS`(frozen Stage9A 七模型),并派生 `REQUIRED_WEIGHT_FILES`(7 checkpoints)与 `REQUIRED_FILES`(代码 + 7 configs);preflight(`check_task1_submission_package.py`)与 checksum manifest writer(`write_task1_weight_manifest.py`)**import** 这些常量而非重复声明,无 drift 风险。已 byte-for-byte 核对 `CHAMPION_MODELS` 与 `run_stage9a_seven_model_postprocess.sh` 一致(同 7 configs/checkpoints、权重 0.2600/0.2600/0.0936/0.0936/0.0928/0.1000/0.1000、`--tta hflip`、`remove_small_min32`)。

- **No-GT 路径(hidden-test/Docker):** `Task1SegmentationDataset.__getitem__`(`src/cathaction/training/task1_baseline.py` L127-138)在 `mask_path` 为空或 `mask_encoding ∈ ("", "none")` 时合成全 background dummy target(仅供 collate,从不被读取;predict 脚本只消费 `batch["image"]`)。entrypoint 自合成 no-GT manifest(`write_nogt_manifest`)。INV-6 合规(全自动、无交互;`docker_entrypoint.sh` 用 `set -euo pipefail` 经 `TASK` env 派发)。
- **已验证:** no-GT dataset 路径单测 `tests/test_task1_nogt_inference.py` PASS(1 passed,刻意不 import smp);preflight `--strict --checksum-manifest` 退出 0(failure_count 0,7/7 weights sha256+size match);entrypoint dry-run 在合成图像上产出正确 manifest(7 models、`--tta hflip`、`remove_small_min32`、missing_assets []、executed false);5 个提交脚本 py_compile clean;Task 1 子套件 30 passed / 1 skipped / 1 pre-existing fail。
- **被阻塞 (blocked):** docker 未安装(image 从未 build/smoke;Dockerfile 自记 BUILD UNVERIFIED);CPU dev host 上 `segmentation-models-pytorch` (smp) 缺失且 torch CPU-only(cuda=False),完整 `--execute` ensemble 推理需 GPU env(`environment-task1-gpu.yml`)。
- **覆盖不对称:** Task 1 **无** preflight/entrypoint 专用单测(仅测 no-GT dataset 路径),而 Task 2 有 `test_task2_submission_package_preflight.py` 与 `test_task2_submission_inference.py`——建议补齐。

### 局限与后续

- hidden-test DSC 未知,官方 validation package(~2026-07-10)未到,**所有数字均为 proxy**。
- **human 域未评测**;Task 1 split 为 frame-level、leak-ASSUMED(非 case-level 已验证),与 INV-2 存在张力,须在外部报告中明确披露,不得宣称 case-level 严谨。
- catheter/guidewire 数值 label 映射未经组织方确认,method report 应软化为 provisional。
- champion 成员 3 的 ConvNeXtV2 ImageNet 权重为 **CC-BY-NC**(非商用),其余 encoders 为 MIT/Apache——与 INV-4 外部资产政策相关,可能为 submission 阻塞项。
- method report 仍为 DRAFT,尚未经 `/review-paper` 或 `/verify-claims`,目标 ≥90 质量门。

---

## 6. Task 2 — 碰撞检测

### 目标与指标解读

Task 2 自第 1 天起即被框定为 **tiny-object tip detection**(每帧一个 box,classes 0=normal / 1=collision),而非帧分类;本地 46,506 images/labels,median normalized box width ~0.033,class IDs 0=normal(28,369)/1=collision(18,137,稀有事件)——class 映射为**工作假设**,待官方确认。

**指标口径(关键):** primary metric = **mAP,per-case macro-averaged**(在每个 case 内算 mAP 再等权平均),匹配 PDF "averaging AP scores across all test cases";与 global pool(frame-count weighted)不同,且 macro 会显著抬高 headline。**IoU 轴(AP50 vs COCO mAP50-95)仍真正模糊**,可能反转结论,因此保留 Stage2AQ(loose/AP50-favoring)+ Stage2AI(strict/high-IoU-favoring)两个 hedge,直至官方 evaluator(2026-07-10)。来源:`quality_reports/decisions/2026-06-15_task2_metric_interpretation.md`。

### "我们的方法"管线

**multi-source candidate proposals → Stage2U ConvNeXt ROI quality reranker → score fusion → ranked detections。**

- **Proposals(候选生成):** 主源 = YOLO11s class-agnostic `tool_roi` detector(及早期 YOLO11s/Stage2L);互补源 = **Task1-segmentation geometry**(skeleton-endpoint 矩形 box,用于恢复 animal/class0——YOLO 在 animal class0 的 recall 为 0,几何 proposal 把 union animal R@0.50 从 0.0000 提到 0.6667);可选 sequence-tip / YOLOV。
- **Stage2U reranker:** ConvNeXt ROI 模型,**6 个输出** = 3 class logits(background/normal/collision)+ 1 IoU regression(pred_iou,sigmoid)+ 1 P(IoU≥0.50) logit + 1 P(IoU≥0.75) logit。必须从 Stage2O checkpoint **warm-start**(from-scratch 弱;load 180 tensors,仅跳过 final head)。代码 `scripts/task2/train_stage2u_quality_ranker.py`(num_classes=6 at L352)。
- **Score fusion:** final score = quality × base score;最佳 quality mode `blend` = 0.50·pred_iou + 0.25·P(IoU≥0.50) + 0.25·P(IoU≥0.75),乘以 Stage2O `rank_decay_roi`(带 per-source rank decay 的 class prob)→ mode `blend_rank_decay_roi`。哲学转变:Stage2O 问 "这是什么类?",Stage2U 问 "这个 crop 既类别正确又 box 紧致吗?"。
- **Oracle recall(候选池上限,Stage2O 4-source union):** R@0.50 = combined 0.5789 / phantom 0.5316 / animal 0.9533;R@0.75 = combined 0.2771 / phantom 0.3131 / **animal 0.0000**(没有任何 proposal/refiner 能产出紧致的 animal box——这是结构性瓶颈)。
- **关键诊断(Stage2S):** oracle top-1 selector 可达 combined mAP50-95 ~0.1410(vs learned 0.0388),证明候选池有 headroom,缺的是 IoU-aligned per-frame selector 而非更多 pruning。

### 老冠军(pre-calibration 视角)

> **重要警示:以下数字是 Stage2U/V ranker pipeline 的内部 public-validation 选择指标,属 PRE-calibration 视角,已被 2026-06-15 honest-baseline 与 2026-06-17/18 工作重新定性,不得作为最终上报结果引用。**

| config | valid_combined mAP50 / mAP50-95 | 说明 | source |
| --- | --- | --- | --- |
| Stage2V(class0=prob_iou75_source_rank_decay_roi, class1=roi) | 0.20097 / 0.04924 | conservative champion(无 domain tuning,fallback) | `2026-06-14_task2_stage2v_submission_hardening_result.md` |
| Stage2AB(per class+domain score policy) | 0.21128 / 0.05063 | 03:34 报告称 public-valid best | `2026-06-14_task2_stage2ab_domain_policy_result.md` |
| Stage2AD(animal-only box calib, strength 1.0) | 0.21128 / 0.06251 | animal mAP50-95 0.05001→0.12699 | `2026-06-14_task2_stage2ad_animal_only_calibration_result.md` |
| Stage2AE(animal-only box calib, strength 1.25) | 0.21128 / 0.06327 | balanced champion;animal mAP50-95→0.14317 | `2026-06-14_task2_stage2ae_calibration_strength_sweep_result.md` |
| Stage2AI(calib boxes + 重选 score policy) | 0.18440 / 0.08395 | high-IoU/COCO hedge;animal mAP50-95 0.23877 | `2026-06-14_task2_stage2ai_calibrated_policy_sweep_result.md` |
| Stage2AN(global class/domain score scaling) | 0.20037 / 0.06479 | minor backup | `2026-06-14_task2_stage2an_global_score_calibration_result.md` |
| **Stage2AQ**(phantom-class1-only multisource replacement, rank_decay_roi, top-k 5, scale 0.75) | **0.23661 / 0.07024** | 16:09 报告称 best balanced;phantom class1 AP50 0.0457→0.0961 | `2026-06-14_task2_stage2aq_multisource_phantom_class1_result.md` |
| Stage2AK ORACLE-IoU 上界(在 Stage2AE pool 上) | 0.45357 / 0.25871 | 证明 ranking/scoring 是主要可恢复差距 | `2026-06-14_task2_stage2ak_bottleneck_attribution_result.md` |

**瓶颈归因:** 同一 Stage2AE pool 上把 score 换成 oracle IoU,mAP50 0.2113→0.4536、mAP50-95 0.0633→0.2587——**ranking/scoring(非 candidate coverage)是主导可恢复差距**,phantom 最差且主导 combined(39,338 phantom vs 1,632 animal 行)。Stage2AP 隔离 phantom class1:oracle IoU50 recall 仅 0.2743、top-score IoU50 recall 0.1262(coverage 与 ranking 双瓶颈)。五个直接修 ranking 的尝试(Stage2X/Y/Z/AA/AL/AM)均在 full-validation 失败,仅 scope-controlled 的 Stage2AQ 产出平衡增益。

> **两份同日报告分歧(均在 pre-calibration 窗口):** 03:34 `..._champion_report.md` 称 Stage2AB best / Stage2V fallback;16:09 `..._current_status.md` 称 Stage2AQ best balanced、Stage2AI high-IoU 备选。另注:champion_report 表格 L19-22 将 Stage2V 仅列 combined,随后紧跟 Stage2AB 的 phantom/animal,易被误读为 Stage2V 的分域值;Stage2V **实际** phantom 0.09604/0.04086、animal 0.49984/0.05000。
>
> **同一 Stage2AQ 两个数值不矛盾:** 本表的 **0.23661**(06-14 内部 public-validation ruler)与后文 EFF 校准段的 **0.130**(2026-06-15 honest-baseline phantom proxy ruler)是**同一配置在两把不同尺子上**的测量,并非矛盾。

### 后 Tier-A 探索(2026-06-15 .. 06-18)

- **Step 1 — Honest baseline(2026-06-15):** Tier-A combined headline(**global ~0.18-0.24 / per-case ~0.29-0.31** mAP50;此前易被笼统说成 "~0.24-0.31",但 0.24 实为 combined **global** 值 Stage2AQ 0.2372)被揭为 **animal-inflated**。validation 实质是**两个单视频**(phantom video_0,824 GT;animal video_2_animal,107 GT;num_cases=2),GT-from-candidates 抬高 recall,per-case macro 进一步抬高 headline。bulk phantom 上 mAP50 仅 ~0.10-0.13,**处于/低于** paper baselines(YOLOV 0.1589 / EFF 0.1691 AP-mean)——"我们击败 EFF/YOLOV" 被**撤回**为 animal-inflated 伪影。保留三候选(Stage2AQ operational / Stage2AI COCO hedge / Stage2V fallback)。
- **Step 2 — RT-DETRv2 floor + ceiling:** bare RT-DETRv2-R18@640(6 epochs,broken OneCycle,无 EMA)**FAIL Gate A**:phantom per-case mAP50 = 0.0276(animal 0.000,combined 0.0138),诊断为真实的 resolution + score-ranking-collapse 失败(非坐标 bug)。换 research-panel recipe(warmup-flat LR + EMA + 3 param-groups + num_queries=60 + light aug + in-loop eval),R18@640 从 0.028 → epoch 20 峰值 **0.103**(~3.7x)后过拟合。结论:原失败主要是 recipe 而非 architecture;但收敛 ceiling ~0.103 仅匹配 champion 低端、**不**超越,且远低于 EFF 0.169,残差为结构性(stride-8 features vs ~13px tips)。
- **Step 3 — EFF-gap calibration(2026-06-17,枢纽性重构):** 忠实复现 YOLOV-S two-class(24 epochs,leakage-safe video-disjoint train_clean = **35,079** 帧),在**同样的 824 phantom + 107 animal 帧、同一 evaluator** 上得 phantom per-case mAP50 = **0.131**(vs champion Stage2AQ 0.130,RT-DETR 0.103;paper YOLOV/EFF official 0.141/0.149)。因 YOLOV 是唯一能在两把尺子上都测量的方法,eval-transfer ratio:**`EFF_ours = 0.1488 × (Y/0.1411) = 1.0546 × Y`**,**cross-over:champion 0.13 击败 EFF 当且仅当 Y < 0.1233**。在 Y=0.131 时投影 EFF_ours ~0.138,故 champion 0.130 仅低约 6%(mAP50 上基本 AT EFF level)、在 mAP50-95 上高约 22%。"2x to EFF" 是把我方薄单视频 proxy 与 paper 官方测试数跨两把尺子比较的**测量幻象**;ratio 抵消了共享的 GT-from-candidates 乐观,因此**ratio(而非绝对 Y)才是 deliverable**。

### 评测口径校准

核心公式与结论(`quality_reports/decisions/2026-06-17_task2_eff_gap_calibration_result.md`):

- `EFF_ours = 1.0546 × Y`(Y = 在我方 ruler 上测得的方法分数)。
- crossover:`Y < 0.123`(更精确 0.1233)时 champion 0.13 超过投影 EFF。
- **"apparent 2x gap is a ruler artifact":** 表面 2x 差距源于跨两把不同尺子比较;在共同尺子上,champion 与收敛后的 YOLOV 同档。该校准为 phantom-only 陈述(animal=0.0,human 未测)。

### 可信多视频面板

单视频 proxy(video_0)被 **6-video full-frame-GT phantom panel(6489 GT)**取代。raw YOLOV 得 per-case macro mAP50 = **0.2523 ± 0.216**(global 0.2091,per-case mAP50-95 0.0979),揭示 video_0 既高方差又偏悲观——**真实 phantom 水平 ~0.21-0.25**,per-video 跨度极大(video_15 0.6455 至 video_11 0.0313)。来源 `quality_reports/decisions/2026-06-17_task2_multivideo_panel_yolov.json`。这说明此前单视频 video_0(0.13)误导了判断。

### 当前最佳配置

**raw YOLOV-S two-class (mv split, 576px) + 既有 Stage2U reranker(Arm A,OOD diverse-trained / YOLO11-1024px-trained checkpoint,无重训)。**

| config | per-case mAP50 | global mAP50 | per-case mAP50-95 | split | source |
| --- | --- | --- | --- | --- | --- |
| **YOLOV + Stage2U reranker (Arm A) — BEST** | **0.278 ± 0.225** | 0.256 | 0.103 | 6-video phantom panel | `2026-06-18_task2_yolov_reranker_swap_armA.md` |
| raw YOLOV (mv 576) | 0.2523 ± 0.216 | 0.2091 | 0.0979 | 同上 | `2026-06-17_task2_multivideo_panel_yolov.json` |
| YOLOV + Stage2U reranker (Arm B, 重训,OVERFIT) | 0.267 | 0.221 | 0.103 | 同上 | `2026-06-18_task2_yolov_reranker_swap_armA.md` |
| raw YOLOV @768(resolution probe,**NEGATIVE**) | 0.248 | 0.220 | 0.100 | 同上 | `2026-06-18_task2_yolov_resolution_negative.md` |

- **Arm A vs Arm B:** Arm A(diverse-trained 既有 checkpoint,无重训)= 0.278,改善 4/6 视频,较 raw YOLOV +10% per-case / +22% global。Arm B(在 YOLOV-only pool 上重训 Stage2U)**OVERFIT**:epoch 1 峰值后降至 0.267,低于 Arm A——diverse-trained 既有 checkpoint 胜出。
- **768px 为 NEGATIVE:** 0.248 vs 576 的 0.252(mAP50-95 flat),输入分辨率**不是** localization lever;feature **STRIDE**(P2/stride-4 head)才是——~13px tip 在 stride-8 maps 上仅 ~2px,低于 head 可定位下限。
- **⚠️ best ≠ shipped(关键区分):** 这个 **0.278 是当前最佳研究结果,但尚未接入提交 entrypoint**。`scripts/task2/run_task2_submission_inference.py` 当前跑的仍是 **Stage2AQ pipeline**(default ship candidate = Stage2V/AQ,在更早的 2-video / pre-calibration ruler 上选定)。撰写报告时务必区分:0.278 = 6-video 面板的研究结论;Stage2V/AQ = 当前默认上线配置。把 0.278 接入提交链是 2026-07-10 后的待办。

**精确路径(full provenance):**
- YOLOV base config:`external_repo_patches/YOLOV/exps/cathaction/cathaction_yolov_s_two_class_mv.py`(上游 pin `fe777cb`,patch `external_repo_patches/YOLOV/cathaction_yolov.patch`)。
- YOLOV base checkpoint:`outputs/task2/yolov_two_class_mv/cathaction_yolov_s_two_class_mv/best_ckpt.pth`。
- Stage2U reranker Arm A checkpoint:`outputs/task2/stage2u_quality_ranker/stage2u_warm_stage2o_fulltrain_valid512_e6/checkpoints/best.pt`(best score mode `prob_iou75_sqrt_source_roi`)。
- YOLOV raw predictions:`outputs/task2/yolov_mv_eval/run1/valid_{phantom,animal,combined}_predictions.csv`。
- Scripts:export `scripts/task2/export_yolov_clean_predictions.py`;panel eval `scripts/task2/evaluate_clean_csv_vs_coco_gt.py`;honest-baseline single-video eval `scripts/task2/evaluate_candidate_honest_baseline.py`。
- Candidate pools:`outputs/task2/stage2o_candidate_pool/yolov_mv_{train,valid}_pool`。
- **不要用** Arm B retrain(`outputs/task2/stage2u_quality_ranker/yolov_mv_armB`,0.267,overfit)。
- 768px config:`external_repo_patches/YOLOV/exps/cathaction/cathaction_yolov_s_two_class_mv_768.py`;输出 `outputs/task2/yolov_two_class_mv_768/`。

### 与 EFF 的真实关系

通过 `EFF_ours = 1.0546 × Y`:在 phantom proxy 上,reproduced YOLOV(0.131)≈ champion Stage2AQ(0.130),投影 EFF_ours ~0.138。结论:champion 在 mAP50 上**基本与 EFF 同档**(低约 6%),在 mAP50-95 上**高出约 22%**。这是 phantom-domain 陈述(animal=0.0,human 未测)。所有探索方法收敛到同一 "EFF tier":薄 proxy ~0.10-0.13,可信 panel ~0.21-0.28。

### 提交打包状态

hidden-test/no-GT 推理链已端到端 wired 并 CPU-smoke 通过,但**尚非完成的官方提交**(PDF 未定义 Task2 result-file schema;docker 未安装)。

- **管线:** raw images → proposals → no-GT candidate pool → Stage2U reranker → Stage2AE/Stage2AQ domain-policy export → internal CSV。entrypoint `scripts/task2/run_task2_submission_inference.py` 调 `run_stage2aq_hidden_pipeline.py --generate-proposals`,链:`export_yolo_nogt_proposals.py` + `export_sequence_tip_nogt_proposals.py` → `build_nogt_candidate_pool.py` → `infer_stage2u_quality_ranker.py` → `run_stage2_champion_pipeline.py`(Stage2AE baseline + Stage2AQ phantom-class1 replacement)→ `{split}_domain_policy_predictions.csv` → copy 到 `task2_predictions_internal.csv`。
- **Result schema(internal,14 列,无 GT,hidden-safe):** `sample_id, video_id, frame_index, domain, class_id, score, x1, y1, x2, y2, source, source_rank, score_mode, policy_name`(`export_stage2ab_domain_policy_predictions.py` L319-336)。**明确不是**官方 schema;`scripts/task2/format_task2_official_submission.py` 尚未编写,待 schema 发布。
- **三个 pinned weights(size + sha256,manifest `quality_reports/decisions/2026-06-14_task2_weight_manifest.json`,preflight failure_count=0,均已 stat 核对存在):** YOLO Stage2L = 19,255,962 bytes `b21e7b48...`;Stage2X class1 tip = 358,019,609 bytes `c6490dfc...`;Stage2U ranker = 334,124,917 bytes `138b3826...`。对应路径见 Provenance 索引表。
- **Docker:** Dockerfile(base `pytorch/pytorch:2.4.0-cuda12.1-cudnn9-runtime`)+ allowlist `.dockerignore` + `scripts/docker_entrypoint.sh`(TASK 派发,INPUT_DIR=/input,OUTPUT_DIR=/output,DEVICE=cuda)。**BUILD UNVERIFIED**(docker 未安装);所有 smoke 仅 CPU(sandbox 无 NVIDIA driver);full-scale hidden run 从未执行;GPU/Docker 需 `environment-task2-gpu.yml`(Stage2U ckpt 用 timm-style ConvNeXt keys,需 timm)。
- **冻结策略:** 按 roadmap,**官方 2026-07-10 前不锁定单一 champion**,对称冻结 Stage2AQ / Stage2AI / Stage2V,默认 domain-agnostic Stage2V;reproduced YOLOV(0.131,Apache-2.0)作为 license-clean 第四候选。Stage2V conservative champion 位于 `outputs/task2/stage2u_quality_ranker/stage2v_class0_yolo_iou75_class1_yolo_roi_export`。

### 局限、为什么难、与突破方向

- **所有 Task 2 自建排名为 directional,非 decisional**(validation 历史薄/曾泄漏)。
- **IoU 轴(AP50 vs COCO mAP50-95)真正模糊**,可反转结论,仅官方 evaluator(2026-07-10)裁定——故保留 Stage2AQ(loose)+ Stage2AI(strict)。
- **animal = 0.0(所有方法)**,291 animal train frames,域偏斜;**human 完全未测**——hidden test 最大未对冲风险。
- **reranker lift 有上限:** 0.278 vs oracle ceiling 0.66;locR75=0.18 将 mAP50-95 封顶在 ~0.10,接近 ~15px tips 的 annotation-precision floor。进一步 reranker/resolution 调参为 low-EV。
- **为什么难:** ~13px 的极小 tip 在 stride-8 特征上仅 ~2px,低于 head 定位能力;真正的 localization lever 是 **P2/stride-4 head**(`YOLOPAFPN_P2` 已在 YOLOX-inside-YOLOV 实现但**未** wire 进 exp,principled fix 未运行,诚实预期为增量而非翻倍)。
- **突破方向:** 不靠继续调本 stack,而需 **域数据**(animal/human)或 **research-grade 范式转变**(如 vessel-geometry collision modeling / multi-task joint segmentation+detection)。近期价值在可信多视频/human-domain validation 与 submission hardening。

---

## 7. 跨任务:提交打包(Docker)汇总

| 项目 | Task 1 | Task 2 |
| --- | --- | --- |
| Entrypoint | `scripts/task1/run_task1_submission_inference.py` | `scripts/task2/run_task2_submission_inference.py` |
| No-GT/hidden 路径 | dataset dummy-target(已单测 PASS) | 完整 proposal→pool→reranker→export 链(CPU smoke PASS,26 行) |
| Preflight | `check_task1_submission_package.py` `--strict` failure_count 0(7/7 weights) | `check_task2_submission_package.py` `--strict` failure_count 0(3/3 weights) |
| Result schema | 原图 PNG masks(0=bg,1=catheter,2=guidewire)+ predictions.csv(**官方 schema 未定义**) | 14-col internal CSV(**官方 schema 未定义**) |
| 专用单测 | 仅 no-GT dataset(无 preflight/entrypoint 测试) | 有 entrypoint + preflight 单测 |
| Docker | 共用 root `Dockerfile` + `.dockerignore`(allowlist)+ `scripts/docker_entrypoint.sh`(TASK 派发) | 同左 |
| Docker 状态 | **BUILD UNVERIFIED**(docker 未安装) | **BUILD UNVERIFIED**(docker 未安装) |
| GPU 依赖 | `environment-task1-gpu.yml`(smp 缺失于 CPU env) | `environment-task2-gpu.yml`(timm 需求) |

**共同阻塞:** docker 未安装于 dev host,image 从未 build/in-container smoke,须在 submission machine 或 CI(`apt install docker.io`)执行;两任务官方 result-file schema 均未在 2026 PDF 定义。

**Licenses(external-asset manifest `quality_reports/decisions/2026-06-15_external_asset_license_manifest.md`):**
- **Ultralytics YOLO11 = AGPL-3.0(HIGH,submission-blocking for public code release)**——proposal stage 实际 `import ultralytics.YOLO`。
- **ConvNeXtV2 pretrained weights = CC-BY-NC 4.0(MEDIUM)**——用于 Task 1 champion ensemble 成员 3,须披露、禁商用;Task 2 tip localizer 用 permissive ConvNeXt-v1。
- **YOLOV repo + YOLOX-S = Apache-2.0(LOW)**——license-clean fallback。
- 三个 no-license 仓库(FGA-Net、Swin-Unet、WT-CMUNeXt)HIGH-risk,但确认两条推理链均不可达、已排除出 build context。

---

## 8. 已知风险与诚实局限

- **Selection-on-eval 历史:** 大量 Task 2 score-mode/domain/strength/scope 选择(Stage2AB、AE strength 1.25、AI policy、AQ scope)在 public validation 上调优,带过拟合风险;decisions 已显式标注并保留低风险 fallback。
- **单视频 validation 陷阱(现已修复方向):** Task 2 早期 headline 依赖单一 phantom video_0(824 GT,GT-from-candidates recall 乐观);已用 6-video panel 揭示其高方差且偏悲观,但 champion Stage2AQ 仍部分基于单视频 proxy。
- **Animal-dominated headlines:** Task 2 早期 combined ~0.24-0.31 由 animal(仅 ~107/1,632 行)抬高;animal class0 AP ≈ 0,animal mAP ~0.50 几乎全来自 class1。
- **Hidden-test domain shift:** Task 2 animal = 0.0(所有方法),human 完全未测;Task 1 human 域未评测且 mask 格式(binary PNG)需转换——两任务的 hidden-test 域泛化均未对冲。
- **Docker 未验证:** image 从未 build/smoke;GPU 推理路径未验证(sandbox 无 driver)。
- **Task 1 split 为 frame-level、leak-assumed**(非 case-level 已验证),与 INV-2 存在张力。
- **Metric 轴未定:** Task 2 AP50 vs COCO mAP50-95、per-case vs global 均未由 PDF 定死,结论可翻转,两读法均保留至 2026-07-10。
- **AGPL-3.0 风险:** Ultralytics YOLO11 对 public code release 为 submission-blocking,需决定 ship-AGPL-compliant 还是换 detector(已 gated,pre-validation 期间不换)。

---

## 9. 时间线与后续计划

**当下 → 2026-07-10(validation+evaluator 发布前):**
- 保持 Task 2 三候选(Stage2AQ / Stage2AI / Stage2V)+ YOLOV 第四候选对称冻结,不锁单一 champion。
- 补齐 Task 1 preflight/entrypoint 专用单测(消除 vs Task 2 的不对称)。
- 修复 pre-existing `test_task1_mslnet_style_metrics`(`f1_r2` KeyError)使测试套件全绿。
- 在 submission machine 或 CI 上 `apt install docker.io`,build image 并跑 in-container CPU smoke + preflight(两任务)。
- 评估 license 决策:Task 1 ConvNeXtV2(CC-BY-NC)、Task 2 Ultralytics(AGPL)——决定披露/合规/替换。
- 若可行,构建 Task 1 human holdout 诊断与 Task 2 human/animal 域验证。

**在 validation 发布时(2026-07-10,binding revisit):**
- 将本地 evaluator 对齐官方代码;在官方 validation set 上重排所有候选;裁定 AP50 vs COCO mAP50-95 / per-case vs global 口径。
- 按官方 result-file schema 编写 `scripts/task2/format_task2_official_submission.py`(及 Task 1 schema adapter);填入 Docker entrypoint。
- 选定每任务的最终 submission champion。

**面向 2026-08-23(提交截止):**
- 完成 full-scale hidden run(非 limit)+ Docker 容器端到端验证。
- 完成 method report(经 `/review-paper` + `/verify-claims`,目标 ≥90 质量门),软化所有 provisional 陈述(label 映射、域泛化、leaderboard 定位)。
- 封板提交:Docker container + official result file + method description。

---

## 10. Provenance 索引表

| Headline result | config | script | checkpoint / output | decision record |
| --- | --- | --- | --- | --- |
| Task1 mean Dice **0.6552** | Stage9A 7-model ensemble + hflip TTA + remove_small_min32 | `scripts/task1/run_stage9a_seven_model_postprocess.sh`;entrypoint `scripts/task1/run_task1_submission_inference.py`;postproc `scripts/task1/apply_morphology_postprocess.py` | `outputs/task1/stage9a_seven_model_add_both_010_010/remove_small_min32/eval.json`;7×`best_checkpoint.pt`(见 §5) | `2026-06-04_task1_final_wrapup.md`;`2026-06-02_task1_stage9a_morphology_grid_result.md` |
| Task1 raw 7-model 0.6537 | Stage7 add_both_010_010 | `run_stage9a_seven_model_postprocess.sh` | `.../raw_predictions/` | `2026-06-01_task1_stage6_stage7_results.md`(superseded) |
| Task1 七成员组成 | — | — | — | `2026-06-01_task1_stage6_stage7_results.md` |
| Task1 weight pin(7/7 sha256) | — | `write_task1_weight_manifest.py`;`check_task1_submission_package.py` | — | `2026-06-15_task1_weight_manifest.json`;`2026-06-15_task1_submission_package_preflight.json`(failure_count 0) |
| Task1 thin-line 诊断 | — | — | `outputs/task1/diagnostics/stage5_full_prediction_quality/summary.json` | `2026-06-02_task1_dice_diagnostic_results.md` |
| Task1 Stage9D 回归(0.6454) | binary hard-neg gate t=0.65(REJECTED) | — | — | `2026-06-04_task1_stage9d_binary_refiner_result.md` |
| Task1 MSLNet-style binary | — | `scripts/task1/evaluate_mslnet_style.py` | `outputs/task1/diagnostics/stage9a_seven_model_remove_small_min32_mslnet_style/eval_mslnet_style.json` | `2026-06-03_task1_mslnet_metric_alignment_result.md` |
| Task1 method report | DRAFT | — | — | `quality_reports/reports/2026-06-15_task1_method_report_draft.md` |
| Task2 first baseline 0.1156/0.0421 | YOLO11s @1024 clean combined(ep9) | — | — | `2026-06-04_task2_yolo11s_baseline_started.md` |
| Task2 collision learnable(AP 0.9067) | Stage2A GT-ROI ConvNeXt-Tiny | — | — | `2026-06-08_task2_stage2a_gt_roi_classifier_result.md` |
| Task2 candidate-pool oracle recall | Stage2O 4-source union | — | `outputs/task2/stage2o_candidate_pool/...` | `2026-06-11_task2_stage2o_candidate_pool_audit_result.md` |
| Task2 Stage2O ranker 0.1887/0.0388 | rank_decay_roi | `scripts/task2/train_stage2u_quality_ranker.py`(Stage2O 系) | — | `2026-06-11_task2_stage2o_ranker_pilot_result.md` |
| Task2 Stage2U(early) 0.1875/0.0414 | blend_rank_decay_roi(YOLO+geom pool) | `scripts/task2/train_stage2u_quality_ranker.py` | `outputs/task2/stage2u_quality_ranker/stage2u_warm_stage2o_fulltrain_valid512_e6/checkpoints/best.pt`;eval `.../stage2u_warm_stage2o_fulltrain_fullvalid_yologeom_eval` | `2026-06-12_task2_stage2u_image_aware_quality_ranker_result.md` |
| Task2 oracle top-1 上界 0.4261/0.1410 | Stage2S | — | — | `2026-06-12_task2_stage2s_topk_output_control_result.md` |
| Task2 OLD Stage2V 0.20097/0.04924 | class0=prob_iou75_source_rank_decay_roi, class1=roi | `scripts/task2/verify_stage2v_champion_export.py` | `outputs/task2/stage2u_quality_ranker/stage2v_class0_yolo_iou75_class1_yolo_roi_export` | `2026-06-14_task2_stage2v_submission_hardening_result.md` |
| Task2 OLD Stage2AB 0.21128/0.05063 | per class+domain policy | `scripts/task2/verify_stage2ab_domain_policy.py` | `outputs/task2/stage2v_domain_policy/yolo_only_domain_policy_sweep.json` | `2026-06-14_task2_stage2ab_domain_policy_result.md` |
| Task2 OLD Stage2AE 0.21128/0.06327 | animal-only box calib strength 1.25 | `scripts/task2/run_stage2_champion_pipeline.py` | — | `2026-06-14_task2_stage2ae_calibration_strength_sweep_result.md` |
| Task2 OLD Stage2AI 0.18440/0.08395 | calib boxes + 重选 policy(COCO hedge) | `run_stage2_champion_pipeline.py` | — | `2026-06-14_task2_stage2ai_calibrated_policy_sweep_result.md` |
| Task2 OLD Stage2AQ 0.23661/0.07024 | phantom-class1-only multisource | `run_stage2_champion_pipeline.py` | `outputs/task2/stage2ar_champion_pipeline_with_aq/public_valid_ab_ad_ae_ai_an_aq/stage2aq` | `2026-06-14_task2_stage2aq_multisource_phantom_class1_result.md` |
| Task2 bottleneck(oracle 0.4536/0.2587) | Stage2AK | — | — | `2026-06-14_task2_stage2ak_bottleneck_attribution_result.md` |
| Task2 honest baseline(retract) | Stage2AQ/V/AI 2-video | `scripts/task2/evaluate_candidate_honest_baseline.py` | `2026-06-15_task2_honest_baseline_stage2{aq,v,ai}.json` | `2026-06-15_task2_honest_baseline_result.md` |
| Task2 metric 解读 | per-case macro;AP50/COCO hedge | `src/cathaction/metrics/detection.py`(per_case L215) | `tests/test_task2_per_case_map.py` | `2026-06-15_task2_metric_interpretation.md` |
| Task2 RT-DETRv2 floor 0.0276 | bare R18@640 broken recipe | `scripts/task2/train_rtdetrv2_task2.py` | — | `2026-06-15_task2_rtdetrv2_r18_bare_baseline.json`;`2026-06-15_task2_rtdetrv2_r18_bare_result.md` |
| Task2 RT-DETRv2 ceiling 0.103 | R18@640 fixed recipe(peak ep20) | `scripts/task2/train_rtdetrv2_task2.py`;export `scripts/task2/export_rtdetrv2_predictions.py` | `outputs/task2/rtdetrv2_r18_recipe_v2/`、`.../rtdetrv2_r18_recipe_v2_cont/` | `2026-06-16_task2_rtdetrv2_ceiling_result.md` |
| Task2 EFF-gap(YOLOV 0.131≈champion 0.130;`EFF_ours=1.0546·Y`) | reproduced YOLOV-S 24ep | `scripts/task2/evaluate_candidate_honest_baseline.py` | — | `2026-06-17_task2_eff_gap_calibration_result.md`;plan `2026-06-17_task2_yolov_reproduction_calibration.md` |
| Task2 raw YOLOV panel 0.2523 | YOLOV-S two-class mv 576 | `scripts/task2/export_yolov_clean_predictions.py`;`scripts/task2/evaluate_clean_csv_vs_coco_gt.py` | `outputs/task2/yolov_mv_eval/run1/valid_*_predictions.csv` | `2026-06-17_task2_multivideo_panel_yolov.json` |
| **Task2 BEST 0.278** | YOLOV-S two-class mv 576 + Stage2U Arm A | `evaluate_clean_csv_vs_coco_gt.py`;config `external_repo_patches/YOLOV/exps/cathaction/cathaction_yolov_s_two_class_mv.py`(pin `fe777cb`) | YOLOV `outputs/task2/yolov_two_class_mv/cathaction_yolov_s_two_class_mv/best_ckpt.pth`;reranker `outputs/task2/stage2u_quality_ranker/stage2u_warm_stage2o_fulltrain_valid512_e6/checkpoints/best.pt` | `2026-06-18_task2_yolov_reranker_swap_armA.md`;`external_repo_patches/YOLOV/README.md` |
| Task2 Arm B 0.267(overfit,勿用) | retrain Stage2U on YOLOV pool | — | `outputs/task2/stage2u_quality_ranker/yolov_mv_armB` | `2026-06-18_task2_yolov_reranker_swap_armA.md` |
| Task2 768px NEGATIVE 0.248 | YOLOV-S mv 768 | — | `outputs/task2/yolov_two_class_mv_768/`;config `..._two_class_mv_768.py` | `2026-06-18_task2_yolov_resolution_negative.md` |
| Task2 submission weights pin(3/3) | — | `check_task2_submission_package.py` | YOLO Stage2L `outputs/task2/yolo_stage2l_proposal/yolo11s_1024_agnostic_train_v0_v1_val_v2_combined_bal_e20/weights/best.pt`(`b21e7b48...`);Stage2X `outputs/task2/sequence_tip_localizer/stage2x_class1_tip384_convnext_centernet_coord20_e8/checkpoints/best.pt`(`c6490dfc...`);Stage2U `.../stage2u_warm_stage2o_fulltrain_valid512_e6/checkpoints/best.pt`(`138b3826...`) | `2026-06-14_task2_weight_manifest.json`;`2026-06-14_task2_stage2bd_weight_checksum_manifest_result.md` |
| Task2 submission entrypoint(CPU smoke 26 行) | Stage2AQ pipeline | `scripts/task2/run_task2_submission_inference.py` → `run_stage2aq_hidden_pipeline.py` → `run_stage2_champion_pipeline.py` | `task2_predictions_internal.csv` | `2026-06-14_task2_stage2ba_submission_entrypoint_result.md`;`2026-06-14_task2_stage2bb_submission_entrypoint_execute_smoke_result.md` |
| INV-2 leak(5/35,788=0.01%) | video-level dedup 修复 | `scripts/task2/prepare_yolo_splits.py`;`scripts/task2/audit_split_disjointness.py` | `summary.json` video_level_disjointness 块 | `2026-06-15_task2_split_integrity_tier_a.md`(session log);`2026-06-15_prevalidation_roadmap.md` |
| Tier A 9 items | — | — | — | `2026-06-15_prevalidation_roadmap.md` |
| External-asset licenses | AGPL YOLO11 / CC-BY-NC ConvNeXtV2 / Apache YOLOV+YOLOX | — | — | `2026-06-15_external_asset_license_manifest.md` |

> **数字状态标记:** Task1 仅 **0.6552** 为 frozen champion(0.6439/0.6517/0.6537、诊断用 0.6517 均 superseded)。Task2 §6"老冠军"全部为 **PRE-calibration**,§6"当前最佳"的 **0.278** 为当前最佳已验证配置(6-video panel),但所有 Task2 数字均为内部 proxy,以 2026-07-10 官方 evaluator 为准。
