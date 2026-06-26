#!/usr/bin/env python3
"""Fine-tune RT-DETRv2 (HF) on CATHACTION Task 2 (2-class tip detection).

Recipe per quality_reports/decisions/2026-06-16_task2_rtdetrv2_recipe (research
panel): warmup-then-FLAT LR, EMA (eval EMA weights), 3 param-groups (backbone
0.1x LR, no weight-decay on norms/biases), num_queries 60, light scale-preserving
augmentation, in-loop eval every K epochs on phantom per-case mAP50 with
best-on-val checkpointing, an epoch-12 falsifiable gate, and patience early-stop.
"""

from __future__ import annotations

import argparse
import copy
import json
import random
from collections import defaultdict
from pathlib import Path

import torch
from PIL import Image, ImageEnhance
from torch.utils.data import DataLoader, Dataset
from transformers import AutoImageProcessor, RTDetrV2ForObjectDetection

import sys
REPO_ROOT = Path(__file__).resolve().parents[2]
for p in (REPO_ROOT, REPO_ROOT / "src"):
    if p.as_posix() not in sys.path:
        sys.path.insert(0, p.as_posix())
from cathaction.metrics.detection import (  # noqa: E402
    DetectionGroundTruth, DetectionPrediction, compute_detection_map_per_case,
)


def label_to_image_path(label_path: Path) -> Path:
    return Path(str(label_path).replace("/labels/", "/images/")).with_suffix(".jpg")


def read_yolo_boxes(label_path: Path):
    out = []
    for line in label_path.read_text(encoding="utf-8").splitlines():
        parts = line.split()
        if len(parts) == 5:
            out.append((int(float(parts[0])), *map(float, parts[1:])))
    return out


class Task2DetDataset(Dataset):
    def __init__(self, list_file: Path, limit=None, aug="none"):
        lines = [l.strip() for l in Path(list_file).read_text().splitlines() if l.strip()]
        self.label_paths = [Path(l) for l in (lines[:limit] if limit else lines)]
        self.aug = aug

    def __len__(self):
        return len(self.label_paths)

    def __getitem__(self, idx):
        lp = self.label_paths[idx]
        img = Image.open(label_to_image_path(lp)).convert("RGB")
        boxes = read_yolo_boxes(lp)  # (cls, cx, cy, w, h) normalized
        if self.aug == "light":
            if random.random() < 0.5:  # horizontal flip (class-invariant for collision/normal)
                img = img.transpose(Image.FLIP_LEFT_RIGHT)
                boxes = [(c, 1.0 - cx, cy, w, h) for (c, cx, cy, w, h) in boxes]
            if random.random() < 0.5:  # photometric, scale-preserving
                img = ImageEnhance.Brightness(img).enhance(random.uniform(0.8, 1.2))
                img = ImageEnhance.Contrast(img).enhance(random.uniform(0.8, 1.2))
        W, H = img.size
        anns = []
        for c, cx, cy, w, h in boxes:
            bw, bh = w * W, h * H
            anns.append({"bbox": [(cx - w / 2) * W, (cy - h / 2) * H, bw, bh],
                         "category_id": c, "area": bw * bh, "iscrowd": 0})
        return {"image": img, "annotations": {"image_id": idx, "annotations": anns}}


def make_collate(processor):
    def collate(batch):
        return processor(images=[b["image"] for b in batch],
                         annotations=[b["annotations"] for b in batch], return_tensors="pt")
    return collate


class EMA:
    def __init__(self, model, decay=0.9999):
        self.decay = decay
        self.shadow = {k: v.detach().clone().float() for k, v in model.state_dict().items()}

    @torch.no_grad()
    def update(self, model, step):
        d = min(self.decay, (1 + step) / (10 + step))  # warmup
        for k, v in model.state_dict().items():
            s = self.shadow[k]
            if v.dtype.is_floating_point:
                s.mul_(d).add_(v.detach().float(), alpha=1 - d)
            else:
                s.copy_(v)

    def copy_to(self, model):
        model.load_state_dict({k: v.to(next(model.parameters()).dtype) if v.dtype.is_floating_point else v
                               for k, v in self.shadow.items()})


def build_param_groups(model, head_lr, backbone_lr, wd):
    groups = {"bb_decay": [], "bb_nodecay": [], "head_decay": [], "head_nodecay": []}
    for name, p in model.named_parameters():
        if not p.requires_grad:
            continue
        is_bb = "backbone" in name
        no_decay = p.ndim <= 1  # biases + norm weights
        key = ("bb" if is_bb else "head") + ("_nodecay" if no_decay else "_decay")
        groups[key].append(p)
    return [
        {"params": groups["bb_decay"], "lr": backbone_lr, "weight_decay": wd},
        {"params": groups["bb_nodecay"], "lr": backbone_lr, "weight_decay": 0.0},
        {"params": groups["head_decay"], "lr": head_lr, "weight_decay": wd},
        {"params": groups["head_nodecay"], "lr": head_lr, "weight_decay": 0.0},
    ]


def iou(a, b):
    x1, y1 = max(a[0], b[0]), max(a[1], b[1])
    x2, y2 = min(a[2], b[2]), min(a[3], b[3])
    inter = max(0, x2 - x1) * max(0, y2 - y1)
    ua = (a[2] - a[0]) * (a[3] - a[1]) + (b[2] - b[0]) * (b[3] - b[1]) - inter
    return inter / (ua + 1e-9)


def load_eval_frames(gt_run_dir: Path, split="valid_phantom"):
    import csv
    path = gt_run_dir / f"{split}_candidates_used.csv"
    seen, frames, gts = set(), [], {}
    for r in csv.DictReader(path.open(encoding="utf-8")):
        sid = r["sample_id"]
        if sid in seen:
            continue
        seen.add(sid)
        frames.append((sid, r["image_path"]))
        gts[sid] = (int(r["gt_class"]),
                    (float(r["gt_x1"]), float(r["gt_y1"]), float(r["gt_x2"]), float(r["gt_y2"])))
    return frames, gts


@torch.no_grad()
def evaluate(model, processor, device, frames, gts, *, batch_size=16, topk=100):
    model.eval()
    preds_by_sid = defaultdict(list)
    confs, top1_iou = [], []
    for start in range(0, len(frames), batch_size):
        chunk = frames[start:start + batch_size]
        imgs, sizes = [], []
        for sid, ip in chunk:
            ip = ip if Path(ip).is_absolute() else str(REPO_ROOT / ip)
            im = Image.open(ip).convert("RGB"); imgs.append(im); sizes.append((im.size[1], im.size[0]))
        inputs = processor(images=imgs, return_tensors="pt").to(device)
        outputs = model(**inputs)
        res = processor.post_process_object_detection(
            outputs, target_sizes=torch.tensor(sizes, device=device), threshold=0.001)
        for (sid, _), r in zip(chunk, res):
            sc = r["scores"].tolist(); lb = r["labels"].tolist(); bx = r["boxes"].tolist()
            order = sorted(range(len(sc)), key=lambda i: sc[i], reverse=True)[:topk]
            for i in order:
                preds_by_sid[sid].append(DetectionPrediction(sid, int(lb[i]), float(sc[i]), tuple(bx[i])))
            if order:
                confs.append(sc[order[0]]); top1_iou.append(iou(bx[order[0]], gts[sid][1]))
    gt_list = [DetectionGroundTruth(sid, gc, gb) for sid, (gc, gb) in gts.items()]
    pred_list = [p for ps in preds_by_sid.values() for p in ps]
    m = compute_detection_map_per_case(gt_list, pred_list, class_ids=(0, 1))
    import statistics as st
    return {
        "phantom_per_case_mAP50": float(m["mAP50"]),
        "phantom_per_case_mAP50_95": float(m["mAP50-95"]),
        "mean_top1_conf": float(st.mean(confs)) if confs else 0.0,
        "median_top1_iou": float(st.median(top1_iou)) if top1_iou else 0.0,
        "frac_top1_iou_ge_0.5": float(sum(1 for x in top1_iou if x >= 0.5) / len(top1_iou)) if top1_iou else 0.0,
    }


def parse_args():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--train-list", type=Path, default=REPO_ROOT / "configs/task2/splits/train_clean_labels.txt")
    p.add_argument("--gt-run-dir", type=Path, required=True, help="Holds valid_phantom_candidates_used.csv for in-loop eval.")
    p.add_argument("--checkpoint", default="PekingU/rtdetr_v2_r18vd")
    p.add_argument("--output-dir", type=Path, required=True)
    p.add_argument("--epochs", type=int, default=40)
    p.add_argument("--batch-size", type=int, default=16)
    p.add_argument("--lr", type=float, default=1e-4)
    p.add_argument("--backbone-lr", type=float, default=1e-5)
    p.add_argument("--weight-decay", type=float, default=1e-4)
    p.add_argument("--warmup-steps", type=int, default=2000)
    p.add_argument("--ema-decay", type=float, default=0.9999)
    p.add_argument("--num-queries", type=int, default=60)
    p.add_argument("--image-size", type=int, default=640)
    p.add_argument("--aug", default="light", choices=["none", "light"])
    p.add_argument("--workers", type=int, default=8)
    p.add_argument("--limit", type=int, default=None)
    p.add_argument("--max-steps", type=int, default=None)
    p.add_argument("--log-interval", type=int, default=100)
    p.add_argument("--eval-every", type=int, default=4)
    p.add_argument("--gate-epoch", type=int, default=12)
    p.add_argument("--gate-min-map50", type=float, default=0.04,
                   help="At gate-epoch, fail if phantom per-case mAP50 < this (trajectory gate, NOT confidence).")
    p.add_argument("--patience", type=int, default=12)
    p.add_argument("--device", default="cuda")
    return p.parse_args()


def main() -> int:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    device = torch.device(args.device)

    processor = AutoImageProcessor.from_pretrained(args.checkpoint)
    if args.image_size:
        processor.size = {"height": args.image_size, "width": args.image_size}
    model = RTDetrV2ForObjectDetection.from_pretrained(
        args.checkpoint, num_labels=2, num_queries=args.num_queries, ignore_mismatched_sizes=True,
        id2label={0: "normal", 1: "collision"}, label2id={"normal": 0, "collision": 1},
    ).to(device)

    ds = Task2DetDataset(args.train_list, limit=args.limit, aug=args.aug)
    loader = DataLoader(ds, batch_size=args.batch_size, shuffle=True, num_workers=args.workers,
                        collate_fn=make_collate(processor), pin_memory=True, drop_last=True)
    optim = torch.optim.AdamW(build_param_groups(model, args.lr, args.backbone_lr, args.weight_decay))
    base_lrs = [g["lr"] for g in optim.param_groups]

    def lr_at(step):
        return min(1.0, step / max(1, args.warmup_steps))  # warmup then flat

    gt_frames, gts = load_eval_frames(args.gt_run_dir)
    ema = EMA(model, decay=args.ema_decay)
    print(f"train={len(ds)} steps/epoch={len(loader)} eval_frames={len(gt_frames)} "
          f"queries={args.num_queries} bb_lr={args.backbone_lr} head_lr={args.lr}", flush=True)

    history, best_map, best_epoch, no_improve = [], -1.0, -1, 0
    step = 0
    train_weights = None
    for epoch in range(args.epochs):
        model.train()
        for batch in loader:
            step += 1
            for g, blr in zip(optim.param_groups, base_lrs):
                g["lr"] = blr * lr_at(step)
            batch = {k: (v.to(device) if torch.is_tensor(v) else
                         [{kk: vv.to(device) for kk, vv in d.items()} for d in v]) for k, v in batch.items()}
            loss = model(**batch).loss
            optim.zero_grad(); loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 0.1)
            optim.step(); ema.update(model, step)
            if step % args.log_interval == 0:
                print(f"e{epoch} s{step} loss={loss.item():.3f} lr_head={optim.param_groups[2]['lr']:.2e}", flush=True)
            if args.max_steps and step >= args.max_steps:
                break

        # eval EMA weights every eval_every epochs (and last epoch)
        if (epoch + 1) % args.eval_every == 0 or epoch == args.epochs - 1 or args.max_steps:
            train_weights = copy.deepcopy(model.state_dict())
            ema.copy_to(model)
            try:
                metrics = evaluate(model, processor, device, gt_frames, gts, batch_size=args.batch_size)
            except Exception as e:  # eval must never kill training
                metrics = {"error": str(e)}
            metrics.update({"epoch": epoch + 1, "step": step})
            history.append(metrics)
            print(f"[EVAL e{epoch+1}] {json.dumps({k: (round(v,4) if isinstance(v,float) else v) for k,v in metrics.items()})}", flush=True)
            m50 = metrics.get("phantom_per_case_mAP50", -1.0)
            if m50 > best_map:
                best_map, best_epoch, no_improve = m50, epoch + 1, 0
                model.save_pretrained(args.output_dir)  # saves EMA weights (current state)
                processor.save_pretrained(args.output_dir)
                print(f"  [best EMA checkpoint @ e{epoch+1} mAP50={m50:.4f}]", flush=True)
            else:
                no_improve += 1
            model.load_state_dict(train_weights)  # restore training weights
            # falsifiable epoch-12 gate — on mAP50 TRAJECTORY (absolute confidence is
            # a red herring under VFL + 1-box/frame; AP only needs ranking).
            if epoch + 1 == args.gate_epoch:
                m50 = metrics.get("phantom_per_case_mAP50", 0.0)
                if m50 < args.gate_min_map50:
                    print(f"GATE FAIL @e{args.gate_epoch}: mAP50={m50:.3f}<{args.gate_min_map50} -> not learning, stopping.", flush=True)
                    break
            if no_improve >= args.patience // args.eval_every:
                print(f"EARLY STOP @e{epoch+1}: no mAP50 improvement (best {best_map:.4f}@e{best_epoch})", flush=True)
                break
        if args.max_steps:
            break

    (args.output_dir / "train_history.json").write_text(json.dumps({
        "checkpoint": args.checkpoint, "epochs": args.epochs, "batch_size": args.batch_size,
        "head_lr": args.lr, "backbone_lr": args.backbone_lr, "num_queries": args.num_queries,
        "image_size": args.image_size, "aug": args.aug, "best_mAP50": best_map,
        "best_epoch": best_epoch, "history": history,
    }, indent=2) + "\n", encoding="utf-8")
    print(f"DONE best phantom per-case mAP50={best_map:.4f} @e{best_epoch} -> {args.output_dir}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
