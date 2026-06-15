#!/usr/bin/env python3
"""Run Stage2U quality-ranker inference on candidate CSVs without GT fields.

This is the hidden-test oriented companion to
``train_stage2u_quality_ranker.py --eval-checkpoint``. It does not compute
metrics and does not require ``gt_*`` or ``candidate_iou`` fields. The output
schema is compatible with the existing Stage2AB/AE/AQ export pipeline:

- ``{split}_candidates_used.csv``
- ``{split}_eval_prediction_rows.csv``
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
from dataclasses import asdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import torch
from PIL import Image
from torch import nn
from torch.utils.data import DataLoader

REPO_ROOT = Path(__file__).resolve().parents[2]
SRC_ROOT = REPO_ROOT / "src"
if REPO_ROOT.as_posix() not in sys.path:
    sys.path.insert(0, REPO_ROOT.as_posix())
if SRC_ROOT.as_posix() not in sys.path:
    sys.path.insert(0, SRC_ROOT.as_posix())

from cathaction.data.task2 import repo_relative  # noqa: E402
from cathaction.data.task2_roi import RoiCropConfig  # noqa: E402
from scripts.task2.train_stage2o_candidate_ranker import (  # noqa: E402
    CandidateExample,
    SCORE_MODES,
    build_transform,
    candidate_score,
    compute_roi_bounds_from_xyxy,
    create_model,
    first_non_empty,
    parse_candidate_xyxy,
    resolve_path,
    write_candidate_csv,
    write_prediction_csv,
)
from scripts.task2.train_stage2u_quality_ranker import (  # noqa: E402
    ALL_DETECTION_MODES,
    SOURCE_METADATA_DIM,
    SourceAwareResidualRanker,
    encode_source_metadata,
    forward_ranker,
    quality_adjusted_score,
    quality_values,
    split_quality_outputs,
)


class InferenceQualityRankerDataset(torch.utils.data.Dataset[tuple[torch.Tensor, torch.Tensor, int]]):
    def __init__(
        self,
        candidates: list[CandidateExample],
        *,
        crop_config: RoiCropConfig,
        input_size: int,
    ) -> None:
        self.candidates = candidates
        self.crop_config = crop_config
        self.transform = build_transform(input_size=input_size, training=False)

    def __len__(self) -> int:
        return len(self.candidates)

    def __getitem__(self, index: int) -> tuple[torch.Tensor, torch.Tensor, int]:
        candidate = self.candidates[index]
        with Image.open(candidate.image_path) as image:
            bounds = compute_roi_bounds_from_xyxy(candidate.xyxy, self.crop_config)
            roi = image.convert("RGB")
            from cathaction.data.task2_roi import crop_roi_with_padding

            roi = crop_roi_with_padding(roi, bounds, fill=self.crop_config.fill)
        return self.transform(roi), encode_source_metadata(candidate), index


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--candidate-csv", action="append", required=True, help="Split candidate CSV as name=path.")
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--backend", choices=("auto", "timm", "torchvision"), default="auto")
    parser.add_argument("--model", default="convnext_tiny")
    parser.add_argument("--pretrained", default=False, action=argparse.BooleanOptionalAction)
    parser.add_argument("--source-aware", default=False, action=argparse.BooleanOptionalAction)
    parser.add_argument("--input-size", type=int, default=224)
    parser.add_argument("--crop-scale", type=float, default=8.0)
    parser.add_argument("--min-crop-size", type=int, default=224)
    parser.add_argument("--max-crop-size", type=int, default=512)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--amp", default=True, action=argparse.BooleanOptionalAction)
    parser.add_argument(
        "--fallback-domain",
        default=None,
        help="Use this domain when a candidate row lacks domain and it cannot be inferred from sample/video id.",
    )
    return parser.parse_args()


def choose_device(value: str) -> torch.device:
    if value == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(value)


def parse_candidate_specs(values: list[str]) -> dict[str, Path]:
    result: dict[str, Path] = {}
    for value in values:
        if "=" not in value:
            raise ValueError(f"candidate-csv must use split=path syntax: {value!r}")
        name, path_text = value.split("=", maxsplit=1)
        name = name.strip()
        path_text = path_text.strip()
        if not name or not path_text:
            raise ValueError(f"candidate-csv must use split=path syntax: {value!r}")
        result[name] = Path(path_text)
    return result


def infer_domain(sample_id: str, video_id: str, fallback_domain: str | None) -> str:
    text = f"{sample_id} {video_id}".lower()
    for domain in ("phantom", "animal", "human"):
        if domain in text:
            return domain
    if fallback_domain is not None:
        return fallback_domain
    return "unknown"


def image_size(path: Path) -> tuple[int, int]:
    with Image.open(path) as image:
        return image.size


def load_inference_candidate_csv(
    path: Path,
    *,
    split: str,
    fallback_domain: str | None = None,
) -> list[CandidateExample]:
    candidates: list[CandidateExample] = []
    with path.open(newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        for row_index, raw in enumerate(reader):
            image_path = resolve_path(Path(first_non_empty(raw, "image_path")))
            width_text = first_non_empty(raw, "image_width", default="")
            height_text = first_non_empty(raw, "image_height", default="")
            if width_text and height_text:
                image_width = int(float(width_text))
                image_height = int(float(height_text))
            else:
                image_width, image_height = image_size(image_path)
            sample_id = first_non_empty(raw, "sample_id")
            video_id = first_non_empty(raw, "video_id", default=sample_id.rsplit("_", 1)[0])
            frame_index = int(float(first_non_empty(raw, "frame_index", default=sample_id.rsplit("_", 1)[-1])))
            domain = first_non_empty(
                raw,
                "domain",
                default=infer_domain(sample_id, video_id, fallback_domain),
            )
            candidates.append(
                CandidateExample(
                    split=split,
                    sample_id=sample_id,
                    video_id=video_id,
                    frame_index=frame_index,
                    domain=domain,
                    image_path=image_path,
                    gt_class=int(float(first_non_empty(raw, "gt_class", default="-1"))),
                    verifier_label=int(float(first_non_empty(raw, "verifier_label", default="-1"))),
                    xyxy=parse_candidate_xyxy(raw),
                    gt_xyxy=(
                        float(first_non_empty(raw, "gt_x1", default="0")),
                        float(first_non_empty(raw, "gt_y1", default="0")),
                        float(first_non_empty(raw, "gt_x2", default="0")),
                        float(first_non_empty(raw, "gt_y2", default="0")),
                    ),
                    gt_iou=float(first_non_empty(raw, "candidate_iou", "gt_iou", default="0")),
                    source_conf=float(first_non_empty(raw, "source_conf", "det_conf", "score", default="0")),
                    source=first_non_empty(raw, "source", default="proposal"),
                    source_priority=int(float(first_non_empty(raw, "source_priority", default="0"))),
                    source_rank=int(float(first_non_empty(raw, "source_rank", "rank", default="9999"))),
                    row_index=row_index,
                    image_width=image_width,
                    image_height=image_height,
                )
            )
    return candidates


def make_prediction_row(
    candidate: CandidateExample,
    *,
    probs: np.ndarray,
    pred_iou_value: float,
    prob_iou50: float,
    prob_iou75: float,
) -> dict[str, Any]:
    p_bg, p_normal, p_collision = (float(value) for value in probs.tolist())
    pred_label = int(np.argmax(probs))
    q_values = quality_values(pred_iou_value, prob_iou50, prob_iou75)
    row: dict[str, Any] = {
        "split": candidate.split,
        "sample_id": candidate.sample_id,
        "gt_class": candidate.gt_class,
        "verifier_label": candidate.verifier_label,
        "pred_label": pred_label,
        "prob_background": p_bg,
        "prob_normal": p_normal,
        "prob_collision": p_collision,
        "source_conf": candidate.source_conf,
        "source": candidate.source,
        "source_rank": candidate.source_rank,
        "gt_iou": candidate.gt_iou,
        "pred_iou": q_values["pred_iou"],
        "prob_iou50": q_values["prob_iou50"],
        "prob_iou75": q_values["prob_iou75"],
        "quality_blend": q_values["blend"],
        "iou_abs_error": "",
    }
    for base_mode in SCORE_MODES:
        score_normal = candidate_score(candidate, p_bg, p_normal, mode=base_mode)
        score_collision = candidate_score(candidate, p_bg, p_collision, mode=base_mode)
        row[f"{base_mode}_score_normal"] = score_normal
        row[f"{base_mode}_score_collision"] = score_collision
        for quality_mode, quality in q_values.items():
            mode = f"{quality_mode}_{base_mode}" if quality_mode != "blend" else f"blend_{base_mode}"
            row[f"{mode}_score_normal"] = quality_adjusted_score(
                candidate,
                p_bg,
                p_normal,
                base_mode=base_mode,
                quality=quality,
            )
            row[f"{mode}_score_collision"] = quality_adjusted_score(
                candidate,
                p_bg,
                p_collision,
                base_mode=base_mode,
                quality=quality,
            )
    return row


@torch.no_grad()
def infer_split(
    *,
    model: nn.Module,
    candidates: list[CandidateExample],
    crop_config: RoiCropConfig,
    input_size: int,
    batch_size: int,
    workers: int,
    device: torch.device,
    amp: bool,
) -> list[dict[str, Any]]:
    loader = DataLoader(
        InferenceQualityRankerDataset(candidates, crop_config=crop_config, input_size=input_size),
        batch_size=batch_size,
        shuffle=False,
        num_workers=workers,
        pin_memory=device.type == "cuda",
        persistent_workers=workers > 0,
    )
    rows: list[dict[str, Any]] = []
    model.eval()
    for images, metadata, indices in loader:
        images = images.to(device, non_blocking=True)
        metadata = metadata.to(device, non_blocking=True)
        with torch.amp.autocast("cuda", enabled=amp and device.type == "cuda"):
            outputs = forward_ranker(model, images, metadata)
            class_logits, pred_iou, iou50_logits, iou75_logits = split_quality_outputs(outputs)
            probabilities = torch.softmax(class_logits, dim=1)
        probabilities_np = probabilities.detach().cpu().numpy()
        pred_iou_np = pred_iou.detach().cpu().numpy()
        prob_iou50_np = torch.sigmoid(iou50_logits).detach().cpu().numpy()
        prob_iou75_np = torch.sigmoid(iou75_logits).detach().cpu().numpy()
        indices_np = indices.numpy()
        for probs, pred_iou_value, prob_iou50, prob_iou75, candidate_index in zip(
            probabilities_np,
            pred_iou_np,
            prob_iou50_np,
            prob_iou75_np,
            indices_np,
        ):
            rows.append(
                make_prediction_row(
                    candidates[int(candidate_index)],
                    probs=probs,
                    pred_iou_value=float(pred_iou_value),
                    prob_iou50=float(prob_iou50),
                    prob_iou75=float(prob_iou75),
                )
            )
    return rows


def create_quality_model(args: argparse.Namespace, device: torch.device) -> nn.Module:
    payload = torch.load(resolve_path(args.checkpoint), map_location="cpu", weights_only=False)
    state_dict = payload.get("model_state_dict", payload)
    backend = choose_backend_for_checkpoint(str(args.backend), state_dict)
    image_model = create_model(backend, args.model, pretrained=args.pretrained, num_classes=6)
    model: nn.Module
    if args.source_aware:
        model = SourceAwareResidualRanker(image_model, metadata_dim=SOURCE_METADATA_DIM, output_dim=6)
    else:
        model = image_model
    model.load_state_dict(state_dict, strict=True)
    return model.to(device)


def choose_backend_for_checkpoint(backend: str, state_dict: dict[str, Any]) -> str:
    if backend != "auto":
        return backend
    keys = tuple(str(key) for key in state_dict)
    if any(key.startswith("stages.") or key.startswith("stem.") or key.startswith("head.") for key in keys):
        return "timm"
    if any(key.startswith("features.") or key.startswith("classifier.") for key in keys):
        return "torchvision"
    return "auto"


def main() -> int:
    args = parse_args()
    device = choose_device(str(args.device))
    crop_config = RoiCropConfig(
        crop_scale=float(args.crop_scale),
        min_crop_size=int(args.min_crop_size),
        max_crop_size=int(args.max_crop_size),
    )
    candidate_specs = parse_candidate_specs(args.candidate_csv)
    output_dir = resolve_path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    model = create_quality_model(args, device)

    split_summaries: dict[str, Any] = {}
    for split, csv_path in candidate_specs.items():
        candidates = load_inference_candidate_csv(
            resolve_path(csv_path),
            split=split,
            fallback_domain=args.fallback_domain,
        )
        write_candidate_csv(output_dir / f"{split}_candidates_used.csv", candidates)
        rows = infer_split(
            model=model,
            candidates=candidates,
            crop_config=crop_config,
            input_size=int(args.input_size),
            batch_size=int(args.batch_size),
            workers=int(args.workers),
            device=device,
            amp=bool(args.amp),
        )
        write_prediction_csv(output_dir / f"{split}_eval_prediction_rows.csv", rows)
        split_summaries[split] = {
            "candidate_csv": resolve_path(csv_path).as_posix(),
            "candidate_rows": len(candidates),
            "prediction_rows": len(rows),
        }

    manifest = {
        "artifact_type": "task2_stage2u_quality_ranker_nogt_inference",
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "checkpoint": repo_relative(resolve_path(args.checkpoint), REPO_ROOT),
        "output_dir": repo_relative(output_dir, REPO_ROOT),
        "crop_config": asdict(crop_config),
        "input_size": int(args.input_size),
        "device": str(device),
        "amp": bool(args.amp),
        "detection_modes": ALL_DETECTION_MODES,
        "splits": split_summaries,
    }
    (output_dir / "inference_manifest.json").write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(manifest, indent=2, sort_keys=True), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
