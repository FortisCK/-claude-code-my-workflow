#!/usr/bin/env python3
"""Export sequence-tip proposal CSVs from raw image sequences without GT."""

from __future__ import annotations

import argparse
import csv
import json
import sys
from bisect import bisect_left
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable

import torch
from torch.utils.data import DataLoader, Dataset

REPO_ROOT = Path(__file__).resolve().parents[2]
SRC_ROOT = REPO_ROOT / "src"
if REPO_ROOT.as_posix() not in sys.path:
    sys.path.insert(0, REPO_ROOT.as_posix())
if SRC_ROOT.as_posix() not in sys.path:
    sys.path.insert(0, SRC_ROOT.as_posix())

from cathaction.data.task2 import repo_relative  # noqa: E402
from scripts.task2.evaluate_sequence_tip_proposals import (  # noqa: E402
    TemplateSize,
    build_model_from_checkpoint,
    decode_candidates,
    load_checkpoint,
    parse_templates,
)
from scripts.task2.export_yolo_nogt_proposals import (  # noqa: E402
    IMAGE_SUFFIXES,
    first_non_empty,
    infer_domain,
    infer_video_frame,
)
from scripts.task2.train_sequence_tip_localizer import (  # noqa: E402
    LetterboxMeta,
    choose_device,
    load_letterboxed_grayscale,
)


DEFAULT_STAGE2X_CHECKPOINT = Path(
    "outputs/task2/sequence_tip_localizer/stage2x_class1_tip384_convnext_centernet_coord20_e8/checkpoints/best.pt"
)
DEFAULT_STAGE2X_TEMPLATES = (
    "16x16,18x18,20x20,22x22,24x24,26x26,28x28,30x30,34x34,38x38,"
    "44x44,52x52,24x58,28x58,30x64,30x70,34x70,38x70,24x80,30x80,"
    "40x80,58x24,70x30,80x40"
)
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
    "proposal_source",
    "proposal_source_peak_rank",
    "proposal_template",
)


@dataclass(frozen=True)
class ImageFrame:
    sample_id: str
    image_path: Path
    video_id: str
    frame_index: int
    domain: str


class NoGtSequenceTipDataset(Dataset[dict[str, Any]]):
    def __init__(self, frames: list[ImageFrame], *, input_size: int, frame_radius: int) -> None:
        self.frames = frames
        self.input_size = input_size
        self.frame_radius = frame_radius
        self.frame_lookup = {(frame.video_id, frame.frame_index): frame for frame in frames}
        self.video_frames: dict[str, list[ImageFrame]] = {}
        for frame in frames:
            self.video_frames.setdefault(frame.video_id, []).append(frame)
        for video_id in self.video_frames:
            self.video_frames[video_id].sort(key=lambda item: item.frame_index)

    def __len__(self) -> int:
        return len(self.frames)

    def __getitem__(self, index: int) -> dict[str, Any]:
        frame = self.frames[index]
        tensors: list[torch.Tensor] = []
        center_meta: LetterboxMeta | None = None
        for offset in range(-self.frame_radius, self.frame_radius + 1):
            neighbor = self.find_neighbor(frame, offset)
            tensor, meta = load_letterboxed_grayscale(neighbor.image_path, input_size=self.input_size)
            if offset == 0:
                center_meta = meta
            tensors.append(tensor)
        if center_meta is None:
            raise RuntimeError("center frame metadata was not created")
        return {
            "image": torch.cat(tensors, dim=0),
            "meta": center_meta,
            "sample_id": frame.sample_id,
            "video_id": frame.video_id,
            "frame_index": frame.frame_index,
            "domain": frame.domain,
            "image_path": frame.image_path,
        }

    def find_neighbor(self, frame: ImageFrame, offset: int) -> ImageFrame:
        target_index = frame.frame_index + offset
        exact = self.frame_lookup.get((frame.video_id, target_index))
        if exact is not None:
            return exact
        frames = self.video_frames[frame.video_id]
        indices = [item.frame_index for item in frames]
        insert_at = bisect_left(indices, target_index)
        candidates: list[ImageFrame] = []
        if insert_at < len(frames):
            candidates.append(frames[insert_at])
        if insert_at > 0:
            candidates.append(frames[insert_at - 1])
        if not candidates:
            return frame
        return min(candidates, key=lambda item: abs(item.frame_index - target_index))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, default=DEFAULT_STAGE2X_CHECKPOINT)
    parser.add_argument("--split", default="hidden")
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--image-dir", type=Path)
    group.add_argument("--image-list-csv", type=Path)
    parser.add_argument("--output-dir", type=Path, default=Path("outputs/task2/sequence_tip_nogt_proposals"))
    parser.add_argument("--name", default="hidden_stage2x_class1")
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--top-peaks", type=int, default=30)
    parser.add_argument("--max-proposals", type=int, default=100)
    parser.add_argument("--peak-nms-kernel", type=int, default=17)
    parser.add_argument("--min-peak-score", type=float, default=0.0)
    parser.add_argument("--templates", default=DEFAULT_STAGE2X_TEMPLATES)
    parser.add_argument("--include-predicted-size", default=True, action=argparse.BooleanOptionalAction)
    parser.add_argument("--amp", default=False, action=argparse.BooleanOptionalAction)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--fallback-domain", default="unknown")
    parser.add_argument("--limit", type=int, default=None)
    return parser.parse_args()


def resolve_path(path: Path) -> Path:
    return path if path.is_absolute() else REPO_ROOT / path


def image_frame_from_path(path: Path, *, fallback_domain: str) -> ImageFrame:
    sample_id = path.stem
    video_id, frame_index = infer_video_frame(sample_id)
    return ImageFrame(
        sample_id=sample_id,
        image_path=path,
        video_id=video_id,
        frame_index=frame_index,
        domain=infer_domain(sample_id, video_id, fallback_domain),
    )


def load_frames_from_dir(image_dir: Path, *, fallback_domain: str) -> list[ImageFrame]:
    root = resolve_path(image_dir)
    paths = sorted(path for path in root.rglob("*") if path.suffix.lower() in IMAGE_SUFFIXES)
    return [image_frame_from_path(path, fallback_domain=fallback_domain) for path in paths]


def load_frames_from_csv(path: Path, *, fallback_domain: str) -> list[ImageFrame]:
    frames: list[ImageFrame] = []
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
            frames.append(
                ImageFrame(
                    sample_id=sample_id,
                    image_path=image_path,
                    video_id=video_id,
                    frame_index=frame_index,
                    domain=domain,
                )
            )
    return frames


def load_frames(args: argparse.Namespace) -> list[ImageFrame]:
    if args.image_dir is not None:
        frames = load_frames_from_dir(args.image_dir, fallback_domain=str(args.fallback_domain))
    else:
        frames = load_frames_from_csv(args.image_list_csv, fallback_domain=str(args.fallback_domain))
    frames.sort(key=lambda item: (item.video_id, item.frame_index, item.sample_id))
    if args.limit is not None:
        frames = frames[: int(args.limit)]
    return frames


def collate_nogt_sequence_tip(batch: list[dict[str, Any]]) -> dict[str, Any]:
    return {
        "images": torch.stack([item["image"] for item in batch], dim=0),
        "metas": [item["meta"] for item in batch],
        "sample_ids": [item["sample_id"] for item in batch],
        "video_ids": [item["video_id"] for item in batch],
        "frame_indices": [item["frame_index"] for item in batch],
        "domains": [item["domain"] for item in batch],
        "image_paths": [item["image_path"] for item in batch],
    }


def rows_for_sample(
    *,
    split: str,
    sample_id: str,
    video_id: str,
    frame_index: int,
    domain: str,
    image_path: Path,
    meta: LetterboxMeta,
    candidates: list[Any],
) -> list[dict[str, Any]]:
    common = {
        "split": split,
        "sample_id": sample_id,
        "video_id": video_id,
        "frame_index": frame_index,
        "domain": domain,
        "image_path": repo_relative(image_path, REPO_ROOT),
        "image_width": meta.original_width,
        "image_height": meta.original_height,
    }
    if not candidates:
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
                "proposal_source": "",
                "proposal_source_peak_rank": "",
                "proposal_template": "",
            }
        ]
    return [
        {
            **common,
            "has_proposal": True,
            "proposal_rank": rank,
            "proposal_class": 0,
            "proposal_conf": candidate.confidence,
            "proposal_x1": candidate.xyxy[0],
            "proposal_y1": candidate.xyxy[1],
            "proposal_x2": candidate.xyxy[2],
            "proposal_y2": candidate.xyxy[3],
            "proposal_source": candidate.source,
            "proposal_source_peak_rank": candidate.source_peak_rank,
            "proposal_template": candidate.source_template,
        }
        for rank, candidate in enumerate(candidates, start=1)
    ]


@torch.no_grad()
def predict_rows(
    *,
    model: torch.nn.Module,
    loader: DataLoader,
    device: torch.device,
    templates: list[TemplateSize],
    args: argparse.Namespace,
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    model.eval()
    for batch in loader:
        images = batch["images"].to(device, non_blocking=True)
        with torch.amp.autocast("cuda", enabled=bool(args.amp) and device.type == "cuda"):
            output = model(images)
        heatmap_logits = output["heatmap_logits"].float().cpu()
        size_map = torch.sigmoid(output["size_logits"].float()).cpu()
        offset_map = torch.sigmoid(output["offset_logits"].float()).cpu()
        for index, sample_id in enumerate(batch["sample_ids"]):
            meta: LetterboxMeta = batch["metas"][index]
            candidates = decode_candidates(
                heatmap_logits[index],
                size_map[index],
                offset_map[index],
                meta=meta,
                templates=templates,
                top_peaks=int(args.top_peaks),
                max_proposals=int(args.max_proposals),
                peak_nms_kernel=int(args.peak_nms_kernel),
                min_peak_score=float(args.min_peak_score),
                include_predicted_size=bool(args.include_predicted_size),
            )
            rows.extend(
                rows_for_sample(
                    split=str(args.split),
                    sample_id=str(sample_id),
                    video_id=str(batch["video_ids"][index]),
                    frame_index=int(batch["frame_indices"][index]),
                    domain=str(batch["domains"][index]),
                    image_path=batch["image_paths"][index],
                    meta=meta,
                    candidates=candidates,
                )
            )
    return rows


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


def main() -> int:
    args = parse_args()
    checkpoint_path = resolve_path(args.checkpoint)
    checkpoint = load_checkpoint(checkpoint_path)
    model, model_config = build_model_from_checkpoint(checkpoint["args"])
    model.load_state_dict(checkpoint["model_state_dict"], strict=True)
    device = choose_device(str(args.device))
    model.to(device)
    model.eval()
    templates = parse_templates(str(args.templates))
    frames = load_frames(args)
    dataset = NoGtSequenceTipDataset(
        frames,
        input_size=int(model_config["input_size"]),
        frame_radius=int(model_config["frame_radius"]),
    )
    loader = DataLoader(
        dataset,
        batch_size=int(args.batch_size),
        shuffle=False,
        num_workers=int(args.workers),
        pin_memory=device.type == "cuda",
        persistent_workers=int(args.workers) > 0,
        collate_fn=collate_nogt_sequence_tip,
    )
    rows = predict_rows(model=model, loader=loader, device=device, templates=templates, args=args)
    run_dir = resolve_path(args.output_dir) / args.name
    output_csv = run_dir / f"{args.split}_proposals.csv"
    row_count = write_rows(output_csv, rows)
    manifest = {
        "artifact_type": "task2_sequence_tip_nogt_proposals",
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "checkpoint": repo_relative(checkpoint_path, REPO_ROOT),
        "run_dir": repo_relative(run_dir, REPO_ROOT),
        "output_csv": repo_relative(output_csv, REPO_ROOT),
        "split": args.split,
        "image_count": len(frames),
        "row_count": row_count,
        "model_config": model_config,
        "templates": [f"{item.width:g}x{item.height:g}" for item in templates],
        "top_peaks": int(args.top_peaks),
        "max_proposals": int(args.max_proposals),
    }
    run_dir.mkdir(parents=True, exist_ok=True)
    (run_dir / "args.json").write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(manifest, indent=2, sort_keys=True), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
