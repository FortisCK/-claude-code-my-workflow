#!/usr/bin/env python3
"""Task 1 (segmentation) submission inference entrypoint.

Runs the frozen Stage9A 7-model ensemble + hflip TTA + remove_small_min32
postprocess on a directory of hidden-test images (NO ground-truth masks), and
writes original-space prediction masks. Mirrors the Task 2 submission wrapper:
fully automated (INV-6), repo-relative paths (INV-18), and `--python` defaults to
`sys.executable` instead of a hardcoded conda interpreter.

Dry-run by default (writes the no-GT manifest + prints the commands); pass
`--execute` to actually run inference.
"""

from __future__ import annotations

import argparse
import csv
import json
import subprocess
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]

IMAGE_SUFFIXES = (".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff")

# Single source of truth for the frozen champion ensemble (Stage9A).
# Authoritative: scripts/task1/run_stage9a_seven_model_postprocess.sh.
CHAMPION_MODELS: tuple[dict[str, object], ...] = (
    {"name": "convnext640", "config": "configs/task1/smp_fpn_convnext_tiny_640_rescue_stable.yaml",
     "checkpoint": "outputs/task1/smp_fpn_convnext_tiny_640_rescue_stable/best_checkpoint.pt", "weight": 0.2600},
    {"name": "efficientnet_b3", "config": "configs/task1/smp_unet_efficientnet_b3_512_shootout.yaml",
     "checkpoint": "outputs/task1/smp_unet_efficientnet_b3_512_shootout/best_checkpoint.pt", "weight": 0.2600},
    {"name": "convnextv2_base", "config": "configs/task1/smp_fpn_convnextv2_base_512_stage4a.yaml",
     "checkpoint": "outputs/task1/smp_fpn_convnextv2_base_512_stage4a/best_checkpoint.pt", "weight": 0.0936},
    {"name": "convnext_small512", "config": "configs/task1/smp_fpn_convnext_small_512_stage4a.yaml",
     "checkpoint": "outputs/task1/smp_fpn_convnext_small_512_stage4a/best_checkpoint.pt", "weight": 0.0936},
    {"name": "convnext_small640", "config": "configs/task1/smp_fpn_convnext_small_640_stage5.yaml",
     "checkpoint": "outputs/task1/smp_fpn_convnext_small_640_stage5/best_checkpoint.pt", "weight": 0.0928},
    {"name": "convnext_small640_cldice", "config": "configs/task1/smp_fpn_convnext_small_640_cldice_stage6.yaml",
     "checkpoint": "outputs/task1/smp_fpn_convnext_small_640_cldice_stage6/best_checkpoint.pt", "weight": 0.1000},
    {"name": "convnext_small640_toolness", "config": "configs/task1/smp_fpn_convnext_small_640_toolness_aux_stage7.yaml",
     "checkpoint": "outputs/task1/smp_fpn_convnext_small_640_toolness_aux_stage7/best_checkpoint.pt", "weight": 0.1000},
)

# Weight checkpoints whose checksums are pinned in the Task 1 weight manifest.
REQUIRED_WEIGHT_FILES: tuple[str, ...] = tuple(str(m["checkpoint"]) for m in CHAMPION_MODELS)

# Code/config files the submission package must contain.
REQUIRED_FILES: tuple[str, ...] = (
    "scripts/task1/run_task1_submission_inference.py",
    "scripts/task1/predict_tta_ensemble_original_space.py",
    "scripts/task1/apply_morphology_postprocess.py",
    *(str(m["config"]) for m in CHAMPION_MODELS),
)


def repo_path(value: str | Path) -> Path:
    path = Path(value)
    return path if path.is_absolute() else (REPO_ROOT / path)


def display_path(value: str | Path) -> str:
    path = Path(value).resolve()
    try:
        return path.relative_to(REPO_ROOT).as_posix()
    except ValueError:
        return path.as_posix()


def discover_images(image_dir: Path) -> list[Path]:
    return sorted(
        p for p in image_dir.rglob("*") if p.is_file() and p.suffix.lower() in IMAGE_SUFFIXES
    )


def write_nogt_manifest(images: list[Path], manifest_path: Path, *, domain: str) -> None:
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    with manifest_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=["image_path", "mask_path", "mask_encoding", "sample_id", "collection", "domain"],
            lineterminator="\n",
        )
        writer.writeheader()
        for image in images:
            writer.writerow(
                {
                    "image_path": display_path(image),
                    "mask_path": "",          # no-GT: hidden test ships no masks
                    "mask_encoding": "none",  # dataset returns a dummy target
                    "sample_id": image.stem,
                    "collection": "hidden",
                    "domain": domain,
                }
            )


def build_predict_command(python: Path, manifest: Path, raw_dir: Path, args) -> list[str]:
    command = [str(python), "scripts/task1/predict_tta_ensemble_original_space.py"]
    for model in CHAMPION_MODELS:
        command += ["--model", str(model["name"]), str(model["config"]),
                    str(model["checkpoint"]), f"{float(model['weight']):.4f}"]
    command += ["--tta", "hflip", "--manifest", display_path(manifest),
                "--output-dir", display_path(raw_dir), "--device", args.device]
    if args.batch_size is not None:
        command += ["--batch-size", str(args.batch_size)]
    if args.num_workers is not None:
        command += ["--num-workers", str(args.num_workers)]
    if args.max_samples is not None:
        command += ["--max-samples", str(args.max_samples)]
    return command


def build_postprocess_command(python: Path, raw_dir: Path, post_dir: Path) -> list[str]:
    return [
        str(python), "scripts/task1/apply_morphology_postprocess.py",
        "--predictions-csv", display_path(raw_dir / "predictions.csv"),
        "--output-dir", display_path(post_dir),
    ]


def run(command: list[str]) -> None:
    env_pythonpath = f"{(REPO_ROOT / 'src')}:{REPO_ROOT}"
    import os

    env = dict(os.environ)
    existing = env.get("PYTHONPATH", "")
    env["PYTHONPATH"] = f"{env_pythonpath}:{existing}" if existing else env_pythonpath
    subprocess.run(command, cwd=REPO_ROOT, env=env, check=True)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--image-dir", type=Path, required=True, help="Directory of hidden-test images (no masks).")
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--python", type=Path, default=Path(sys.executable),
                        help="Interpreter for the inference subprocesses (default: sys.executable).")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--num-workers", type=int, default=8)
    parser.add_argument("--max-samples", type=int, default=None)
    parser.add_argument("--fallback-domain", default="unknown",
                        help="Domain tag for synthesized manifest rows (hidden test domain is unknown).")
    parser.add_argument("--work-subdir", default="task1_submission_work")
    parser.add_argument("--execute", action="store_true", help="Actually run inference (default: dry-run).")
    parser.add_argument("--strict", action="store_true", help="Fail if no images or a champion asset is missing.")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    image_dir = repo_path(args.image_dir)
    output_dir = repo_path(args.output_dir)
    work_dir = output_dir / args.work_subdir
    raw_dir = work_dir / "raw_predictions"
    post_dir = output_dir / "predictions"
    manifest_path = work_dir / "nogt_manifest.csv"

    missing_assets = [
        display_path(repo_path(p))
        for p in (*REQUIRED_WEIGHT_FILES, *(str(m["config"]) for m in CHAMPION_MODELS))
        if not repo_path(p).is_file()
    ]

    if not image_dir.is_dir():
        print(f"ERROR: --image-dir not found: {image_dir}", file=sys.stderr)
        return 2
    images = discover_images(image_dir)

    work_dir.mkdir(parents=True, exist_ok=True)
    if images:
        write_nogt_manifest(images, manifest_path, domain=args.fallback_domain)

    python = repo_path(args.python) if not Path(args.python).is_absolute() else Path(args.python)
    predict_command = build_predict_command(python, manifest_path, raw_dir, args)
    postprocess_command = build_postprocess_command(python, raw_dir, post_dir)

    submission_manifest = {
        "artifact_type": "task1_submission_inference_manifest",
        "champion": "stage9a_seven_model_ensemble_hflip_remove_small_min32",
        "image_dir": display_path(image_dir),
        "num_images": len(images),
        "nogt_manifest": display_path(manifest_path),
        "raw_prediction_dir": display_path(raw_dir),
        "prediction_dir": display_path(post_dir),
        "models": [
            {"name": m["name"], "config": str(m["config"]),
             "checkpoint": str(m["checkpoint"]), "weight": m["weight"]}
            for m in CHAMPION_MODELS
        ],
        "predict_command": predict_command,
        "postprocess_command": postprocess_command,
        "missing_assets": missing_assets,
        "executed": bool(args.execute),
        "note": "Official Task 1 result schema is undefined in the 2026 PDF; "
                "output is original-space PNG masks (0=bg,1=catheter,2=guidewire) + predictions.csv.",
    }
    output_dir.mkdir(parents=True, exist_ok=True)
    manifest_json = output_dir / "task1_submission_inference_manifest.json"
    manifest_json.write_text(json.dumps(submission_manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(submission_manifest, indent=2, sort_keys=True))

    if args.strict and not images:
        print("ERROR (--strict): no images found in --image-dir", file=sys.stderr)
        return 1
    if args.strict and missing_assets:
        print(f"ERROR (--strict): missing champion assets: {missing_assets}", file=sys.stderr)
        return 1

    if not args.execute:
        print("\nDRY RUN — re-run with --execute to run inference. Commands above.", file=sys.stderr)
        return 0

    if missing_assets:
        print(f"ERROR: cannot execute, missing champion assets: {missing_assets}", file=sys.stderr)
        return 1
    run(predict_command)
    run(postprocess_command)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
