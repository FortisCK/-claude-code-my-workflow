#!/usr/bin/env python3
"""Export YOLO proposal CSVs from raw images without GT labels.

This is the hidden-test counterpart to ``evaluate_yolo_proposals.py``. It runs
YOLO inference on an image directory or image-list CSV and writes
``{split}_proposals.csv`` without requiring labels or GT boxes.
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable

REPO_ROOT = Path(__file__).resolve().parents[2]
SRC_ROOT = REPO_ROOT / "src"
if SRC_ROOT.as_posix() not in sys.path:
    sys.path.insert(0, REPO_ROOT.as_posix())
if REPO_ROOT.as_posix() not in sys.path:
    sys.path.insert(0, REPO_ROOT.as_posix())
if SRC_ROOT.as_posix() not in sys.path:
    sys.path.insert(0, SRC_ROOT.as_posix())

from cathaction.data.task2 import repo_relative  # noqa: E402


IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff"}
PROPOSAL_FIELDS = (
    "split",
    "sample_id",
    "video_id",
    "frame_index",
    "domain",
    "image_path",
    "image_width",
    "image_height",
    "has_proposal",
    "proposal_rank",
    "proposal_class",
    "proposal_conf",
    "proposal_x1",
    "proposal_y1",
    "proposal_x2",
    "proposal_y2",
)


@dataclass(frozen=True)
class ImageItem:
    sample_id: str
    image_path: Path
    video_id: str
    frame_index: int
    domain: str


@dataclass(frozen=True)
class Proposal:
    xyxy: tuple[float, float, float, float]
    confidence: float
    class_id: int
    rank: int


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--weights", type=Path, required=True)
    parser.add_argument("--split", default="hidden")
    parser.add_argument("--image-dir", type=Path, default=None)
    parser.add_argument(
        "--image-list-csv",
        type=Path,
        default=None,
        help="CSV with at least image_path; optional sample_id, video_id, frame_index, domain.",
    )
    parser.add_argument("--output-dir", type=Path, default=Path("outputs/task2/yolo_nogt_proposals"))
    parser.add_argument("--name", default="hidden_yolo_nogt")
    parser.add_argument("--imgsz", type=int, default=1024)
    parser.add_argument("--conf", type=float, default=0.001)
    parser.add_argument("--iou", type=float, default=0.7)
    parser.add_argument("--max-det", type=int, default=20)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--source-chunk-size", type=int, default=64)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--fallback-domain", default="unknown")
    parser.add_argument("--limit", type=int, default=None)
    return parser.parse_args()


def resolve_path(path: Path) -> Path:
    return path if path.is_absolute() else REPO_ROOT / path


def choose_device(value: str) -> str:
    if value == "auto":
        try:
            import torch

            return "0" if torch.cuda.is_available() else "cpu"
        except Exception:
            return "cpu"
    return value


def infer_video_frame(sample_id: str, video_id: str = "", frame_index: str = "") -> tuple[str, int]:
    if video_id and frame_index:
        return video_id, int(float(frame_index))
    if "_" in sample_id:
        prefix, suffix = sample_id.rsplit("_", maxsplit=1)
        try:
            return video_id or prefix, int(float(suffix))
        except ValueError:
            return video_id or prefix, 0
    return video_id or sample_id, 0


def infer_domain(sample_id: str, video_id: str, fallback_domain: str) -> str:
    text = f"{sample_id} {video_id}".lower()
    for domain in ("phantom", "animal", "human"):
        if domain in text:
            return domain
    return fallback_domain


def image_item_from_path(path: Path, *, fallback_domain: str) -> ImageItem:
    sample_id = path.stem
    video_id, frame_index = infer_video_frame(sample_id)
    return ImageItem(
        sample_id=sample_id,
        image_path=path,
        video_id=video_id,
        frame_index=frame_index,
        domain=infer_domain(sample_id, video_id, fallback_domain),
    )


def load_images_from_dir(image_dir: Path, *, fallback_domain: str) -> list[ImageItem]:
    root = resolve_path(image_dir)
    images = sorted(path for path in root.rglob("*") if path.suffix.lower() in IMAGE_SUFFIXES)
    return [image_item_from_path(path, fallback_domain=fallback_domain) for path in images]


def first_non_empty(row: dict[str, str], *keys: str, default: str = "") -> str:
    for key in keys:
        value = row.get(key)
        if value not in (None, ""):
            return str(value)
    return default


def load_images_from_csv(path: Path, *, fallback_domain: str) -> list[ImageItem]:
    rows: list[ImageItem] = []
    with resolve_path(path).open(newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        for row in reader:
            image_path = resolve_path(Path(first_non_empty(row, "image_path", "path", "filename")))
            sample_id = first_non_empty(row, "sample_id", default=image_path.stem)
            video_id, frame_index = infer_video_frame(
                sample_id,
                video_id=first_non_empty(row, "video_id", default=""),
                frame_index=first_non_empty(row, "frame_index", default=""),
            )
            domain = first_non_empty(row, "domain", default=infer_domain(sample_id, video_id, fallback_domain))
            rows.append(
                ImageItem(
                    sample_id=sample_id,
                    image_path=image_path,
                    video_id=video_id,
                    frame_index=frame_index,
                    domain=domain,
                )
            )
    return rows


def load_image_items(args: argparse.Namespace) -> list[ImageItem]:
    if args.image_list_csv is None and args.image_dir is None:
        raise ValueError("Provide --image-dir or --image-list-csv")
    if args.image_list_csv is not None and args.image_dir is not None:
        raise ValueError("Provide only one of --image-dir or --image-list-csv")
    if args.image_list_csv is not None:
        items = load_images_from_csv(args.image_list_csv, fallback_domain=str(args.fallback_domain))
    else:
        items = load_images_from_dir(args.image_dir, fallback_domain=str(args.fallback_domain))
    if args.limit is not None:
        items = items[: int(args.limit)]
    return items


def select_proposals(result: Any) -> list[Proposal]:
    boxes = getattr(result, "boxes", None)
    if boxes is None or len(boxes) == 0:
        return []
    xyxy = boxes.xyxy.detach().cpu().numpy()
    conf = boxes.conf.detach().cpu().numpy()
    cls = boxes.cls.detach().cpu().numpy() if getattr(boxes, "cls", None) is not None else [0] * len(xyxy)
    order = sorted(range(len(xyxy)), key=lambda index: float(conf[index]), reverse=True)
    proposals: list[Proposal] = []
    for rank, index in enumerate(order, start=1):
        proposals.append(
            Proposal(
                xyxy=tuple(float(value) for value in xyxy[index]),
                confidence=float(conf[index]),
                class_id=int(float(cls[index])),
                rank=rank,
            )
        )
    return proposals


def rows_for_item(
    item: ImageItem,
    *,
    split: str,
    image_width: int,
    image_height: int,
    proposals: list[Proposal],
) -> list[dict[str, Any]]:
    common = {
        "split": split,
        "sample_id": item.sample_id,
        "video_id": item.video_id,
        "frame_index": item.frame_index,
        "domain": item.domain,
        "image_path": repo_relative(item.image_path, REPO_ROOT),
        "image_width": image_width,
        "image_height": image_height,
    }
    if not proposals:
        return [
            {
                **common,
                "has_proposal": False,
                "proposal_rank": "",
                "proposal_class": "",
                "proposal_conf": 0.0,
                "proposal_x1": "",
                "proposal_y1": "",
                "proposal_x2": "",
                "proposal_y2": "",
            }
        ]
    return [
        {
            **common,
            "has_proposal": True,
            "proposal_rank": proposal.rank,
            "proposal_class": proposal.class_id,
            "proposal_conf": proposal.confidence,
            "proposal_x1": proposal.xyxy[0],
            "proposal_y1": proposal.xyxy[1],
            "proposal_x2": proposal.xyxy[2],
            "proposal_y2": proposal.xyxy[3],
        }
        for proposal in proposals
    ]


def write_rows(path: Path, rows: Iterable[dict[str, Any]]) -> int:
    count = 0
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(PROPOSAL_FIELDS))
        writer.writeheader()
        for row in rows:
            writer.writerow(row)
            count += 1
    return count


def predict_rows(args: argparse.Namespace, items: list[ImageItem]) -> list[dict[str, Any]]:
    from ultralytics import YOLO

    yolo_model = YOLO(resolve_path(args.weights).as_posix())
    device_arg = choose_device(str(args.device))
    rows: list[dict[str, Any]] = []
    for start in range(0, len(items), int(args.source_chunk_size)):
        chunk = items[start : start + int(args.source_chunk_size)]
        results = yolo_model.predict(
            source=[item.image_path.as_posix() for item in chunk],
            imgsz=int(args.imgsz),
            conf=float(args.conf),
            iou=float(args.iou),
            max_det=int(args.max_det),
            batch=int(args.batch_size),
            device=device_arg,
            stream=True,
            verbose=False,
        )
        for item, result in zip(chunk, results):
            image_height, image_width = int(result.orig_shape[0]), int(result.orig_shape[1])
            rows.extend(
                rows_for_item(
                    item,
                    split=str(args.split),
                    image_width=image_width,
                    image_height=image_height,
                    proposals=select_proposals(result),
                )
            )
    return rows


def main() -> int:
    args = parse_args()
    run_dir = resolve_path(args.output_dir) / args.name
    items = load_image_items(args)
    rows = predict_rows(args, items)
    output_csv = run_dir / f"{args.split}_proposals.csv"
    row_count = write_rows(output_csv, rows)
    manifest = {
        "artifact_type": "task2_yolo_nogt_proposals",
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "run_dir": repo_relative(run_dir, REPO_ROOT),
        "weights": repo_relative(resolve_path(args.weights), REPO_ROOT),
        "split": args.split,
        "image_count": len(items),
        "row_count": row_count,
        "output_csv": repo_relative(output_csv, REPO_ROOT),
        "imgsz": int(args.imgsz),
        "conf": float(args.conf),
        "iou": float(args.iou),
        "max_det": int(args.max_det),
    }
    run_dir.mkdir(parents=True, exist_ok=True)
    (run_dir / "args.json").write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(manifest, indent=2, sort_keys=True), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
