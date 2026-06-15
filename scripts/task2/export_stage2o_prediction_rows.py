#!/usr/bin/env python3
"""Export Stage2O ranker prediction rows for an existing candidate CSV."""

from __future__ import annotations

import argparse
import importlib.machinery
import json
import os
import sys
import types
from dataclasses import asdict
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[2]
SRC_ROOT = REPO_ROOT / "src"
if REPO_ROOT.as_posix() not in sys.path:
    sys.path.insert(0, REPO_ROOT.as_posix())
if SRC_ROOT.as_posix() not in sys.path:
    sys.path.insert(0, SRC_ROOT.as_posix())
os.environ.setdefault("TORCHINDUCTOR_CACHE_DIR", (REPO_ROOT / "quality_reports" / "logs").as_posix())


def install_wandb_stub() -> None:
    """Prevent timm import from importing wandb, which needs a writable tempdir."""
    if "wandb" in sys.modules:
        return
    module = types.ModuleType("wandb")
    module.__spec__ = importlib.machinery.ModuleSpec("wandb", loader=None)
    module.run = None
    module.init = lambda *args, **kwargs: None
    module.log = lambda *args, **kwargs: None
    sys.modules["wandb"] = module


install_wandb_stub()

import torch  # noqa: E402
from torch.utils.data import DataLoader  # noqa: E402

from cathaction.data.task2 import repo_relative  # noqa: E402
from cathaction.data.task2_roi import RoiCropConfig, load_task2_samples_from_split  # noqa: E402
from scripts.task2.train_stage2o_candidate_ranker import (  # noqa: E402
    CandidateRoiDataset,
    apply_candidate_budget,
    choose_device,
    create_model,
    evaluate_ranker,
    load_candidate_csv,
    resolve_path,
    write_candidate_csv,
    write_prediction_csv,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--candidate-csv",
        type=Path,
        default=Path(
            "outputs/task2/stage2o_ranker/"
            "convnext_tiny_stage2o_yolo_geometry_train_subset_full_eval/train_candidates_used.csv"
        ),
    )
    parser.add_argument(
        "--split-labels",
        type=Path,
        default=Path("configs/task2/splits_stage2o_ranker/train_v0_v1_val_v2_animal_all_phantom1000pc_labels.txt"),
    )
    parser.add_argument("--data-root", type=Path, default=Path("datasets/collision_detection"))
    parser.add_argument(
        "--checkpoint",
        type=Path,
        default=Path(
            "outputs/task2/stage2o_ranker/"
            "convnext_tiny_stage2o_yolo_geometry_train_subset_valid256_e4/checkpoints/best.pt"
        ),
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("outputs/task2/stage2o_ranker/convnext_tiny_stage2o_yolo_geometry_train_subset_full_eval"),
    )
    parser.add_argument("--output-prefix", default="train_eval")
    parser.add_argument("--backend", choices=("auto", "timm", "torchvision"), default="timm")
    parser.add_argument("--model", default="convnext_tiny")
    parser.add_argument("--pretrained", default=False, action=argparse.BooleanOptionalAction)
    parser.add_argument("--input-size", type=int, default=224)
    parser.add_argument("--crop-scale", type=float, default=8.0)
    parser.add_argument("--min-crop-size", type=int, default=224)
    parser.add_argument("--max-crop-size", type=int, default=512)
    parser.add_argument("--batch-size", type=int, default=512)
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--amp", default=True, action=argparse.BooleanOptionalAction)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--max-candidates-per-sample", type=int, default=100)
    parser.add_argument(
        "--candidate-budget-sort",
        choices=("source_order", "raw_conf"),
        default="source_order",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    candidate_csv = resolve_path(args.candidate_csv)
    split_labels = resolve_path(args.split_labels)
    data_root = resolve_path(args.data_root)
    checkpoint = resolve_path(args.checkpoint)
    output_dir = resolve_path(args.output_dir)
    if not output_dir.exists():
        raise FileNotFoundError(
            f"Output directory must already exist in this sandboxed run: {repo_relative(output_dir, REPO_ROOT)}"
        )

    crop_config = RoiCropConfig(
        crop_scale=float(args.crop_scale),
        min_crop_size=int(args.min_crop_size),
        max_crop_size=int(args.max_crop_size),
    )
    candidates = apply_candidate_budget(
        load_candidate_csv(candidate_csv, split_override=args.output_prefix, repo_root=REPO_ROOT),
        max_candidates_per_sample=int(args.max_candidates_per_sample),
        sort_mode=str(args.candidate_budget_sort),
    )
    samples = load_task2_samples_from_split(data_root, split_labels)
    device = choose_device(args.device)
    model = create_model(args.backend, args.model, pretrained=args.pretrained, num_classes=3).to(device)
    payload = torch.load(checkpoint, map_location="cpu", weights_only=False)
    state_dict = payload.get("model_state_dict", payload)
    model.load_state_dict(state_dict, strict=True)

    loader = DataLoader(
        CandidateRoiDataset(
            candidates,
            crop_config=crop_config,
            input_size=int(args.input_size),
            training=False,
        ),
        batch_size=int(args.batch_size),
        shuffle=False,
        num_workers=int(args.workers),
        pin_memory=device.type == "cuda",
        persistent_workers=int(args.workers) > 0,
    )
    metrics, rows = evaluate_ranker(
        model=model,
        loader=loader,
        candidates=candidates,
        samples=samples,
        device=device,
        use_amp=bool(args.amp),
    )

    write_candidate_csv(output_dir / f"{args.output_prefix}_candidates_used.csv", candidates)
    write_prediction_csv(output_dir / f"{args.output_prefix}_prediction_rows.csv", rows)
    (output_dir / f"{args.output_prefix}_metrics.json").write_text(
        json.dumps(
            json_ready(
                {
                    "args": vars(args),
                    "crop_config": asdict(crop_config),
                    "candidate_rows": len(candidates),
                    "sample_rows": len(samples),
                    "checkpoint": checkpoint,
                    "metrics": metrics,
                }
            ),
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )
    print(
        f"Exported {len(rows)} prediction rows to "
        f"{repo_relative(output_dir / f'{args.output_prefix}_prediction_rows.csv', REPO_ROOT)}",
        flush=True,
    )
    return 0


def json_ready(value: Any) -> Any:
    if isinstance(value, Path):
        return repo_relative(value, REPO_ROOT)
    if isinstance(value, dict):
        return {str(key): json_ready(item) for key, item in value.items()}
    if isinstance(value, list):
        return [json_ready(item) for item in value]
    if isinstance(value, tuple):
        return [json_ready(item) for item in value]
    if hasattr(value, "item"):
        try:
            return json_ready(value.item())
        except Exception:
            return str(value)
    return value


if __name__ == "__main__":
    raise SystemExit(main())
