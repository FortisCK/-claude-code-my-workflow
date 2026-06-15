#!/usr/bin/env python3
"""Write Task 1 predictions from TTA/ensemble checkpoints."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Any

import numpy as np
import torch
from PIL import Image
from torch.utils.data import DataLoader
from tqdm import tqdm

from cathaction.training.task1_baseline import (
    _path_for_json,
    _resolve_path,
    _safe_filename,
    select_device,
    write_json,
)
from scripts.task1.evaluate_tta_ensemble import (
    ModelSpec,
    _build_loader,
    _load_model_spec,
    _normalize_tta_modes,
    _normalized_weights,
    _predict_model_probability,
    _validate_matching_batches,
    _validate_model_specs,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Write PNG predictions from Task 1 checkpoints with flip TTA and ensembling."
    )
    parser.add_argument(
        "--model",
        action="append",
        nargs=4,
        metavar=("NAME", "CONFIG", "CHECKPOINT", "WEIGHT"),
        required=True,
        help="Model spec. Repeat for ensembles.",
    )
    parser.add_argument(
        "--tta",
        choices=("none", "hflip", "vflip", "hvflip"),
        action="append",
        default=None,
        help="TTA transform to include. Repeatable. Defaults to none.",
    )
    parser.add_argument("--manifest", type=Path, default=None)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--max-samples", type=int, default=None)
    parser.add_argument("--batch-size", type=int, default=None)
    parser.add_argument("--num-workers", type=int, default=None)
    parser.add_argument("--device", type=str, default=None)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    device = select_device(args.device or "auto")
    model_specs = [_load_model_spec(raw, device=device) for raw in args.model]
    if not model_specs:
        raise ValueError("At least one --model spec is required.")

    _validate_model_specs(model_specs)
    label_mode = str(model_specs[0].config["data"]["label_mode"])
    if label_mode != "multiclass_012":
        raise ValueError(f"Only multiclass_012 is supported for soft ensembling, got {label_mode!r}.")

    tta_modes = _normalize_tta_modes(args.tta)
    weights = _normalized_weights(model_specs)
    repo_root = model_specs[0].repo_root
    manifest = (
        _resolve_path(args.manifest, repo_root)
        if args.manifest is not None
        else _resolve_path(model_specs[0].config["paths"]["eval_manifest"], repo_root)
    )
    output_dir = _resolve_path(args.output_dir, repo_root)
    output_dir.mkdir(parents=True, exist_ok=True)

    batch_size = args.batch_size or min(
        int(spec.config["training"]["batch_size"]) for spec in model_specs
    )
    num_workers = (
        args.num_workers
        if args.num_workers is not None
        else min(int(spec.config["training"].get("num_workers", 0)) for spec in model_specs)
    )
    loaders = [
        _build_loader(
            spec,
            eval_manifest=manifest,
            max_samples=args.max_samples,
            batch_size=batch_size,
            num_workers=num_workers,
        )
        for spec in model_specs
    ]

    rows = _write_predictions(
        model_specs,
        loaders,
        weights=weights,
        tta_modes=tta_modes,
        output_dir=output_dir,
        device=device,
    )
    manifest_path = output_dir / "predictions.csv"
    with manifest_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=["sample_id", "collection", "domain", "prediction_path"],
            lineterminator="\n",
        )
        writer.writeheader()
        writer.writerows(rows)

    result = {
        "models": [
            {
                "name": spec.name,
                "config_path": _path_for_json(spec.config_path, repo_root),
                "checkpoint_path": _path_for_json(spec.checkpoint_path, repo_root),
                "weight": spec.weight,
                "normalized_weight": weights[index],
            }
            for index, spec in enumerate(model_specs)
        ],
        "tta_modes": tta_modes,
        "manifest": _path_for_json(manifest, repo_root),
        "prediction_dir": _path_for_json(output_dir, repo_root),
        "prediction_manifest": _path_for_json(manifest_path, repo_root),
        "predictions": len(rows),
        "batch_size": batch_size,
        "num_workers": num_workers,
        "device": str(device),
    }
    write_json(output_dir / "summary.json", result)
    print(json.dumps(result, indent=2))
    return 0


def _write_predictions(
    model_specs: list[ModelSpec],
    loaders: list[DataLoader[dict[str, Any]]],
    *,
    weights: list[float],
    tta_modes: list[str],
    output_dir: Path,
    device: torch.device,
) -> list[dict[str, str]]:
    repo_root = model_specs[0].repo_root
    rows: list[dict[str, str]] = []
    with torch.no_grad():
        for batches in tqdm(zip(*loaders), total=len(loaders[0]), desc="predict", leave=False):
            reference_batch = batches[0]
            _validate_matching_batches(batches)
            target_shape = tuple(int(value) for value in reference_batch["image"].shape[-2:])
            probability_sum: torch.Tensor | None = None

            for spec, batch, weight in zip(model_specs, batches, weights):
                probability = _predict_model_probability(
                    spec,
                    batch["image"].to(device),
                    tta_modes=tta_modes,
                    target_shape=target_shape,
                    device=device,
                )
                weighted_probability = probability * float(weight)
                probability_sum = (
                    weighted_probability
                    if probability_sum is None
                    else probability_sum + weighted_probability
                )

            if probability_sum is None:
                raise RuntimeError("No model probabilities were produced.")
            prediction = torch.argmax(probability_sum, dim=1).cpu().numpy().astype(np.uint8)
            for index, sample_id in enumerate(reference_batch["sample_id"]):
                filename = f"{_safe_filename(str(sample_id))}.png"
                prediction_path = output_dir / filename
                _save_multiclass_prediction(prediction_path, prediction[index])
                rows.append(
                    {
                        "sample_id": str(sample_id),
                        "collection": str(reference_batch["collection"][index]),
                        "domain": str(reference_batch["domain"][index]),
                        "prediction_path": _path_for_json(prediction_path, repo_root),
                    }
                )
    return rows


def _save_multiclass_prediction(path: Path, prediction: np.ndarray) -> None:
    Image.fromarray(prediction.astype(np.uint8)).save(path)


if __name__ == "__main__":
    raise SystemExit(main())
