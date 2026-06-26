#!/usr/bin/env python3
"""Evaluate official YOLOV checkpoints as Task 2 proposal detectors."""

from __future__ import annotations

import argparse
import json
import math
import random
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import torch

REPO_ROOT = Path(__file__).resolve().parents[2]
SRC_ROOT = REPO_ROOT / "src"
YOLOV_ROOT = REPO_ROOT / "external_repos" / "YOLOV"
for path in (REPO_ROOT, SRC_ROOT, YOLOV_ROOT):
    if path.as_posix() not in sys.path:
        sys.path.insert(0, path.as_posix())

from cathaction.data.task2 import Task2Sample, build_task2_index, repo_relative  # noqa: E402
from cathaction.metrics.detection import iou_xyxy  # noqa: E402
from scripts.task2.evaluate_yolo_proposals import (  # noqa: E402
    format_summary,
    parse_top_k_values,
    summarize_proposals,
    write_rows_csv,
)


@dataclass(frozen=True)
class SplitSpec:
    name: str
    annotation: str


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-root", type=Path, default=Path("datasets/collision_detection"))
    parser.add_argument("--yolov-root", type=Path, default=Path("external_repos/YOLOV"))
    parser.add_argument(
        "--exp-file",
        type=Path,
        default=Path("external_repos/YOLOV/exps/cathaction/cathaction_yolov_s_agnostic.py"),
    )
    parser.add_argument(
        "--data-dir",
        type=Path,
        default=Path("datasets/collision_detection_yolov_agnostic"),
        help="Converted YOLOV dataset used for evaluation.",
    )
    parser.add_argument(
        "--checkpoint",
        type=Path,
        default=Path("outputs/task2/yolov_pilot/cathaction_yolov_s_agnostic/best_ckpt.pth"),
    )
    parser.add_argument("--output-dir", type=Path, default=Path("outputs/task2/yolov_proposal_eval"))
    parser.add_argument("--name", default="yolov_s_576_agnostic_pilot_e3")
    parser.add_argument("--top-k", default="1,3,5,10,20,50")
    parser.add_argument("--max-proposals", type=int, default=50)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--half", action="store_true")
    parser.add_argument("--data-num-workers", type=int, default=0)
    parser.add_argument("--seed", type=int, default=2026)
    parser.add_argument(
        "--splits",
        nargs="+",
        default=None,
        help="Override split specs as name=annotation.json (e.g. train=cathaction_train.json). "
        "Defaults to the three valid panels.",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    set_seed(args.seed)

    data_root = resolve_path(args.data_root)
    data_dir = resolve_path(args.data_dir)
    checkpoint = resolve_path(args.checkpoint)
    exp_file = resolve_path(args.exp_file)
    run_dir = resolve_path(args.output_dir) / args.name
    run_dir.mkdir(parents=True, exist_ok=True)

    device = choose_device(args.device)
    top_k_values = parse_top_k_values(args.top_k)
    sample_by_id = {sample.sample_id: sample for sample in build_task2_index(data_root)}

    from yolox.exp import get_exp

    exp = get_exp(exp_file.as_posix(), None)
    exp.data_dir = data_dir.as_posix()
    exp.data_num_workers = int(args.data_num_workers)
    model = load_model(exp, checkpoint, device=device, half=bool(args.half))

    args_record = {
        **vars(args),
        "data_root": repo_relative(data_root, REPO_ROOT),
        "data_dir": repo_relative(data_dir, REPO_ROOT),
        "checkpoint": repo_relative(checkpoint, REPO_ROOT),
        "exp_file": repo_relative(exp_file, REPO_ROOT),
        "run_dir": repo_relative(run_dir, REPO_ROOT),
        "device": str(device),
        "top_k_values": top_k_values,
        "test_size": list(exp.test_size),
        "lframe_val": int(exp.lframe_val),
        "gframe_val": int(exp.gframe_val),
    }
    (run_dir / "args.json").write_text(json.dumps(json_ready(args_record), indent=2) + "\n", encoding="utf-8")
    print(json.dumps(json_ready(args_record), indent=2), flush=True)

    if args.splits:
        split_specs = [SplitSpec(*spec.split("=", 1)) for spec in args.splits]
    else:
        split_specs = [
            SplitSpec("valid_combined", "cathaction_valid_combined.json"),
            SplitSpec("valid_phantom", "cathaction_valid_phantom.json"),
            SplitSpec("valid_animal", "cathaction_valid_animal.json"),
        ]
    all_metrics: dict[str, Any] = {}
    for split_spec in split_specs:
        exp.val_name = split_spec.name
        exp.val_ann = split_spec.annotation
        loader = exp.get_eval_loader(
            batch_size=int(exp.lframe_val + exp.gframe_val),
            data_num_workers=int(args.data_num_workers),
        )
        rows, processed_samples = evaluate_split(
            model=model,
            exp=exp,
            loader=loader,
            sample_by_id=sample_by_id,
            split_name=split_spec.name,
            device=device,
            half=bool(args.half),
            max_proposals=int(args.max_proposals),
        )
        metrics = summarize_proposals(processed_samples, rows, top_k_values=top_k_values)
        metrics["covered_samples"] = len(processed_samples)
        metrics["loader_batches"] = len(loader)
        all_metrics[split_spec.name] = metrics
        write_rows_csv(run_dir / f"{split_spec.name}_proposals.csv", rows)
        (run_dir / f"{split_spec.name}_metrics.json").write_text(
            json.dumps(json_ready(metrics), indent=2) + "\n",
            encoding="utf-8",
        )
        print(format_summary(split_spec.name, metrics, top_k_values), flush=True)

    (run_dir / "summary_metrics.json").write_text(
        json.dumps(json_ready(all_metrics), indent=2) + "\n",
        encoding="utf-8",
    )
    print(f"Saved YOLOV proposal evaluation to {run_dir}", flush=True)
    return 0


def load_model(exp: Any, checkpoint: Path, *, device: torch.device, half: bool) -> torch.nn.Module:
    model = exp.get_model()
    payload = torch.load(checkpoint, map_location="cpu", weights_only=False)
    state_dict = payload.get("model", payload)
    model.load_state_dict(state_dict)
    model.to(device)
    model.eval()
    if half:
        model.half()
    return model


def evaluate_split(
    *,
    model: torch.nn.Module,
    exp: Any,
    loader: Any,
    sample_by_id: dict[str, Task2Sample],
    split_name: str,
    device: torch.device,
    half: bool,
    max_proposals: int,
) -> tuple[list[dict[str, Any]], list[Task2Sample]]:
    rows: list[dict[str, Any]] = []
    processed_samples: list[Task2Sample] = []
    tensor_dtype = torch.float16 if half else torch.float32
    with torch.no_grad():
        for imgs, _, info_imgs, _, paths, _ in loader:
            imgs = imgs.to(device=device, dtype=tensor_dtype)
            outputs, _ = model(imgs, lframe=exp.lframe_val, gframe=exp.gframe_val)
            for output, info_img, image_path in zip(outputs, info_imgs, paths):
                sample_id = Path(str(image_path)).stem
                sample = sample_by_id[sample_id]
                image_height, image_width = int(info_img[0]), int(info_img[1])
                gt_box = tuple(float(value) for value in sample.box.xyxy_pixels(image_width, image_height))
                processed_samples.append(sample)
                proposals = output_to_proposals(
                    output,
                    image_width=image_width,
                    image_height=image_height,
                    test_size=tuple(exp.test_size),
                    max_proposals=max_proposals,
                )
                rows.extend(rows_for_sample(sample, split_name, image_width, image_height, gt_box, proposals))
    return rows, processed_samples


def output_to_proposals(
    output: torch.Tensor | None,
    *,
    image_width: int,
    image_height: int,
    test_size: tuple[int, int],
    max_proposals: int,
) -> list[dict[str, Any]]:
    if output is None:
        return []
    output_cpu = output.detach().float().cpu()
    if output_cpu.numel() == 0:
        return []
    scale = min(test_size[0] / float(image_height), test_size[1] / float(image_width))
    boxes = output_cpu[:, 0:4] / scale
    scores = output_cpu[:, 4] * output_cpu[:, 5]
    classes = output_cpu[:, 6]
    order = torch.argsort(scores, descending=True)[:max_proposals]
    proposals: list[dict[str, Any]] = []
    for rank, index in enumerate(order.tolist(), start=1):
        x1, y1, x2, y2 = boxes[index].tolist()
        proposals.append(
            {
                "rank": rank,
                "class_id": int(classes[index].item()),
                "confidence": float(scores[index].item()),
                "xyxy": (
                    max(0.0, min(float(x1), float(image_width))),
                    max(0.0, min(float(y1), float(image_height))),
                    max(0.0, min(float(x2), float(image_width))),
                    max(0.0, min(float(y2), float(image_height))),
                ),
            }
        )
    return proposals


def rows_for_sample(
    sample: Task2Sample,
    split_name: str,
    image_width: int,
    image_height: int,
    gt_box: tuple[float, float, float, float],
    proposals: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    if not proposals:
        return [
            {
                "split": split_name,
                "sample_id": sample.sample_id,
                "gt_class": sample.class_id,
                "image_path": repo_relative(sample.image_path, REPO_ROOT),
                "image_width": image_width,
                "image_height": image_height,
                "gt_x1": gt_box[0],
                "gt_y1": gt_box[1],
                "gt_x2": gt_box[2],
                "gt_y2": gt_box[3],
                "has_proposal": False,
                "proposal_rank": "",
                "proposal_class": "",
                "proposal_conf": 0.0,
                "proposal_x1": "",
                "proposal_y1": "",
                "proposal_x2": "",
                "proposal_y2": "",
                "proposal_gt_iou": 0.0,
            }
        ]
    rows: list[dict[str, Any]] = []
    for proposal in proposals:
        xyxy = proposal["xyxy"]
        rows.append(
            {
                "split": split_name,
                "sample_id": sample.sample_id,
                "gt_class": sample.class_id,
                "image_path": repo_relative(sample.image_path, REPO_ROOT),
                "image_width": image_width,
                "image_height": image_height,
                "gt_x1": gt_box[0],
                "gt_y1": gt_box[1],
                "gt_x2": gt_box[2],
                "gt_y2": gt_box[3],
                "has_proposal": True,
                "proposal_rank": proposal["rank"],
                "proposal_class": proposal["class_id"],
                "proposal_conf": proposal["confidence"],
                "proposal_x1": xyxy[0],
                "proposal_y1": xyxy[1],
                "proposal_x2": xyxy[2],
                "proposal_y2": xyxy[3],
                "proposal_gt_iou": iou_xyxy(xyxy, gt_box),  # type: ignore[arg-type]
            }
        )
    return rows


def choose_device(value: str) -> torch.device:
    if value == "auto":
        return torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    return torch.device(value)


def resolve_path(path: Path) -> Path:
    return path if path.is_absolute() else REPO_ROOT / path


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def json_ready(value: Any) -> Any:
    if isinstance(value, Path):
        return repo_relative(value, REPO_ROOT)
    if isinstance(value, dict):
        return {str(key): json_ready(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [json_ready(item) for item in value]
    if isinstance(value, np.ndarray):
        return json_ready(value.tolist())
    if isinstance(value, (np.floating, np.integer)):
        return json_ready(value.item())
    if isinstance(value, float) and not math.isfinite(value):
        return None
    return value


if __name__ == "__main__":
    raise SystemExit(main())
