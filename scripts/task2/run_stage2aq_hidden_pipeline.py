#!/usr/bin/env python3
"""Dry-run-first hidden-test wrapper for the Stage2AQ Task2 pipeline.

This wrapper can either start from precomputed proposal CSVs or generate the
hidden-ready yolo/stage2x proposal CSVs from raw images first. It does not train
and does not require GT. It connects:

1. yolo-only no-GT candidate pool;
2. multi-source no-GT candidate pool;
3. Stage2U no-GT inference for both branches;
4. Stage2AE/Stage2AQ clean prediction export.
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_YOLO_SOURCE = "yolo_stage2l=outputs/task2/yolo_proposal_eval/yolo11s_1024_stage2l_combined_bal_e20_stage2l_panels"
DEFAULT_MULTISOURCE = (
    DEFAULT_YOLO_SOURCE,
    "stage2x_class1=outputs/task2/stage2ax_proposal_generation/hidden_stage2x_class1",
)
DEFAULT_CHECKPOINT = Path(
    "outputs/task2/stage2u_quality_ranker/stage2u_warm_stage2o_fulltrain_valid512_e6/checkpoints/best.pt"
)
DEFAULT_YOLO_WEIGHTS = Path(
    "outputs/task2/yolo_stage2l_proposal/"
    "yolo11s_1024_agnostic_train_v0_v1_val_v2_combined_bal_e20/weights/best.pt"
)
DEFAULT_STAGE2X_CHECKPOINT = Path(
    "outputs/task2/sequence_tip_localizer/stage2x_class1_tip384_convnext_centernet_coord20_e8/checkpoints/best.pt"
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--python", type=Path, default=Path(sys.executable))
    parser.add_argument("--split", action="append", default=None, help="Split prefix. Defaults to hidden.")
    parser.add_argument(
        "--generate-proposals",
        action="store_true",
        help="Prepend raw-image yolo_stage2l and stage2x_class1 proposal generation commands.",
    )
    image_group = parser.add_mutually_exclusive_group()
    image_group.add_argument("--raw-image-dir", type=Path, default=None)
    image_group.add_argument("--raw-image-list-csv", type=Path, default=None)
    parser.add_argument("--yolo-weights", type=Path, default=DEFAULT_YOLO_WEIGHTS)
    parser.add_argument("--stage2x-checkpoint", type=Path, default=DEFAULT_STAGE2X_CHECKPOINT)
    parser.add_argument("--proposal-output-name-yolo", default="hidden_yolo_stage2l")
    parser.add_argument("--proposal-output-name-stage2x", default="hidden_stage2x_class1")
    parser.add_argument(
        "--proposal-limit",
        type=int,
        default=None,
        help="Optional raw-image proposal export limit for smoke tests. Defaults to full input.",
    )
    parser.add_argument("--yolo-source", default=DEFAULT_YOLO_SOURCE, help="Yolo-only source as name=directory.")
    parser.add_argument(
        "--source",
        action="append",
        default=None,
        help=(
            "Multisource proposal source as name=directory. Defaults to the "
            "hidden-ready Stage2AQ policy sources: yolo_stage2l + stage2x_class1."
        ),
    )
    parser.add_argument("--source-top-k", type=int, default=50)
    parser.add_argument("--image-dir", type=Path, default=Path("datasets/collision_detection/images"))
    parser.add_argument("--fallback-domain", default=None)
    parser.add_argument("--checkpoint", type=Path, default=DEFAULT_CHECKPOINT)
    parser.add_argument(
        "--work-dir",
        type=Path,
        default=Path("outputs/task2/stage2aw_hidden_pipeline"),
        help="Root for candidate pools, ranker rows, predictions, and manifest.",
    )
    parser.add_argument("--baseline-candidate-name", default="hidden_yolo_only_top50")
    parser.add_argument("--multisource-candidate-name", default="hidden_five_source_top50")
    parser.add_argument("--baseline-ranker-name", default="hidden_yolo_only_stage2u")
    parser.add_argument("--multisource-ranker-name", default="hidden_multisource_stage2u")
    parser.add_argument("--prediction-name", default="hidden_stage2ae_stage2aq_predictions")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--manifest", type=Path, default=None)
    parser.add_argument("--execute", action="store_true")
    parser.add_argument("--strict", action="store_true", help="Fail in dry-run if required proposal files are missing.")
    return parser.parse_args()


def repo_path(path: Path) -> Path:
    return path if path.is_absolute() else REPO_ROOT / path


def display_path(path: Path) -> str:
    try:
        return path.relative_to(REPO_ROOT).as_posix()
    except ValueError:
        return path.as_posix()


def source_name(value: str) -> str:
    if "=" not in value:
        raise ValueError(f"source must use name=directory syntax: {value!r}")
    name, _ = value.split("=", maxsplit=1)
    return name.strip()


def source_dir(value: str) -> Path:
    if "=" not in value:
        raise ValueError(f"source must use name=directory syntax: {value!r}")
    _, path_text = value.split("=", maxsplit=1)
    return Path(path_text.strip())


def command_record(name: str, argv: list[str], outputs: list[Path]) -> dict[str, Any]:
    return {
        "name": name,
        "argv": argv,
        "outputs": [display_path(repo_path(path)) for path in outputs],
    }


def candidate_csvs(candidate_root: Path, candidate_name: str, splits: list[str]) -> list[Path]:
    return [candidate_root / candidate_name / f"{split}_candidates.csv" for split in splits]


def generated_yolo_source(args: argparse.Namespace) -> str:
    return f"yolo_stage2l={(args.work_dir / 'proposal_generation' / args.proposal_output_name_yolo).as_posix()}"


def generated_stage2x_source(args: argparse.Namespace) -> str:
    return f"stage2x_class1={(args.work_dir / 'proposal_generation' / args.proposal_output_name_stage2x).as_posix()}"


def proposal_input_args(args: argparse.Namespace) -> list[str]:
    if args.raw_image_dir is not None:
        return ["--image-dir", args.raw_image_dir.as_posix()]
    if args.raw_image_list_csv is not None:
        return ["--image-list-csv", args.raw_image_list_csv.as_posix()]
    raise ValueError("--generate-proposals requires --raw-image-dir or --raw-image-list-csv")


def candidate_pool_image_dir(args: argparse.Namespace) -> Path:
    if bool(getattr(args, "generate_proposals", False)) and args.raw_image_dir is not None:
        return args.raw_image_dir
    return args.image_dir


def build_proposal_generation_commands(args: argparse.Namespace, splits: list[str]) -> list[dict[str, Any]]:
    proposal_root = args.work_dir / "proposal_generation"
    commands: list[dict[str, Any]] = []
    for split in splits:
        common_input = proposal_input_args(args)
        yolo_cmd = [
            args.python.as_posix(),
            "scripts/task2/export_yolo_nogt_proposals.py",
            "--weights",
            args.yolo_weights.as_posix(),
            "--split",
            split,
            "--output-dir",
            proposal_root.as_posix(),
            "--name",
            args.proposal_output_name_yolo,
            "--device",
            str(args.device),
            "--batch-size",
            str(args.batch_size),
            "--fallback-domain",
            str(args.fallback_domain or "unknown"),
            *common_input,
        ]
        stage2x_cmd = [
            args.python.as_posix(),
            "scripts/task2/export_sequence_tip_nogt_proposals.py",
            "--checkpoint",
            args.stage2x_checkpoint.as_posix(),
            "--split",
            split,
            "--output-dir",
            proposal_root.as_posix(),
            "--name",
            args.proposal_output_name_stage2x,
            "--device",
            str(args.device),
            "--batch-size",
            str(args.batch_size),
            "--workers",
            str(args.workers),
            "--fallback-domain",
            str(args.fallback_domain or "unknown"),
            *common_input,
        ]
        if args.proposal_limit is not None:
            yolo_cmd.extend(["--limit", str(args.proposal_limit)])
            stage2x_cmd.extend(["--limit", str(args.proposal_limit)])
        commands.append(
            command_record(
                f"export_yolo_stage2l_nogt_proposals_{split}",
                yolo_cmd,
                [proposal_root / args.proposal_output_name_yolo / f"{split}_proposals.csv"],
            )
        )
        commands.append(
            command_record(
                f"export_stage2x_class1_nogt_proposals_{split}",
                stage2x_cmd,
                [proposal_root / args.proposal_output_name_stage2x / f"{split}_proposals.csv"],
            )
        )
    return commands


def build_commands(args: argparse.Namespace, splits: list[str], multisources: list[str]) -> list[dict[str, Any]]:
    work_dir = repo_path(args.work_dir)
    candidate_root = args.work_dir / "candidate_pool"
    image_dir = candidate_pool_image_dir(args)
    ranker_root = args.work_dir / "stage2u_inference"
    prediction_dir = args.work_dir / args.prediction_name
    baseline_candidate_dir = candidate_root / args.baseline_candidate_name
    multisource_candidate_dir = candidate_root / args.multisource_candidate_name
    baseline_ranker_dir = ranker_root / args.baseline_ranker_name
    multisource_ranker_dir = ranker_root / args.multisource_ranker_name

    baseline_pool_cmd = [
        args.python.as_posix(),
        "scripts/task2/build_nogt_candidate_pool.py",
        "--source",
        args.yolo_source,
        "--source-top-k",
        str(args.source_top_k),
        "--image-dir",
        image_dir.as_posix(),
        "--output-dir",
        candidate_root.as_posix(),
        "--name",
        args.baseline_candidate_name,
    ]
    multisource_pool_cmd = [
        args.python.as_posix(),
        "scripts/task2/build_nogt_candidate_pool.py",
        "--source-top-k",
        str(args.source_top_k),
        "--image-dir",
        image_dir.as_posix(),
        "--output-dir",
        candidate_root.as_posix(),
        "--name",
        args.multisource_candidate_name,
    ]
    for split in splits:
        baseline_pool_cmd.extend(["--split", split])
        multisource_pool_cmd.extend(["--split", split])
    if args.fallback_domain is not None:
        baseline_pool_cmd.extend(["--fallback-domain", str(args.fallback_domain)])
        multisource_pool_cmd.extend(["--fallback-domain", str(args.fallback_domain)])
    for source in multisources:
        multisource_pool_cmd.extend(["--source", source])

    baseline_infer_cmd = [
        args.python.as_posix(),
        "scripts/task2/infer_stage2u_quality_ranker.py",
        "--checkpoint",
        args.checkpoint.as_posix(),
        "--output-dir",
        baseline_ranker_dir.as_posix(),
        "--device",
        str(args.device),
        "--batch-size",
        str(args.batch_size),
        "--workers",
        str(args.workers),
    ]
    multisource_infer_cmd = [
        args.python.as_posix(),
        "scripts/task2/infer_stage2u_quality_ranker.py",
        "--checkpoint",
        args.checkpoint.as_posix(),
        "--output-dir",
        multisource_ranker_dir.as_posix(),
        "--device",
        str(args.device),
        "--batch-size",
        str(args.batch_size),
        "--workers",
        str(args.workers),
    ]
    for split in splits:
        baseline_infer_cmd.extend(
            ["--candidate-csv", f"{split}={(baseline_candidate_dir / f'{split}_candidates.csv').as_posix()}"]
        )
        multisource_infer_cmd.extend(
            ["--candidate-csv", f"{split}={(multisource_candidate_dir / f'{split}_candidates.csv').as_posix()}"]
        )

    champion_cmd = [
        args.python.as_posix(),
        "scripts/task2/run_stage2_champion_pipeline.py",
        "--run-dir",
        baseline_ranker_dir.as_posix(),
        "--output-dir",
        prediction_dir.as_posix(),
        "--variant",
        "stage2ae",
        "--variant",
        "stage2aq",
        "--stage2aq-multisource-run-dir",
        multisource_ranker_dir.as_posix(),
        "--skip-metrics",
        "--splits",
        *splits,
    ]
    if args.fallback_domain is not None:
        champion_cmd.extend(["--fallback-domain", str(args.fallback_domain)])

    commands = []
    if bool(getattr(args, "generate_proposals", False)):
        commands.extend(build_proposal_generation_commands(args, splits))
    commands.extend([
        command_record("build_yolo_only_nogt_candidate_pool", baseline_pool_cmd, candidate_csvs(candidate_root, args.baseline_candidate_name, splits)),
        command_record("build_multisource_nogt_candidate_pool", multisource_pool_cmd, candidate_csvs(candidate_root, args.multisource_candidate_name, splits)),
        command_record(
            "infer_yolo_only_stage2u",
            baseline_infer_cmd,
            [baseline_ranker_dir / f"{split}_eval_prediction_rows.csv" for split in splits],
        ),
        command_record(
            "infer_multisource_stage2u",
            multisource_infer_cmd,
            [multisource_ranker_dir / f"{split}_eval_prediction_rows.csv" for split in splits],
        ),
        command_record(
            "export_stage2ae_stage2aq_hidden_predictions",
            champion_cmd,
            [prediction_dir / "stage2aq" / f"{split}_domain_policy_predictions.csv" for split in splits],
        ),
    ])
    return commands


def preflight(args: argparse.Namespace, splits: list[str], multisources: list[str]) -> list[dict[str, str]]:
    checks: list[dict[str, str]] = []

    def add(kind: str, path: Path) -> None:
        full = repo_path(path)
        checks.append({"kind": kind, "path": display_path(full), "exists": str(full.exists()).lower()})

    add("checkpoint", args.checkpoint)
    generated_sources = set()
    if bool(getattr(args, "generate_proposals", False)):
        add("yolo_weights", args.yolo_weights)
        add("stage2x_checkpoint", args.stage2x_checkpoint)
        generated_sources = {generated_yolo_source(args), generated_stage2x_source(args)}
        if args.raw_image_dir is not None:
            add("raw_image_dir", args.raw_image_dir)
        elif args.raw_image_list_csv is not None:
            add("raw_image_list_csv", args.raw_image_list_csv)
    else:
        add("image_dir", args.image_dir)
    for source in [args.yolo_source, *multisources]:
        if source in generated_sources:
            continue
        name = source_name(source)
        directory = source_dir(source)
        add(f"source_dir:{name}", directory)
        for split in splits:
            add(f"source_proposals:{name}:{split}", directory / f"{split}_proposals.csv")
    return checks


def missing_checks(checks: list[dict[str, str]]) -> list[dict[str, str]]:
    return [check for check in checks if check["exists"] != "true"]


def run_commands(commands: list[dict[str, Any]]) -> None:
    for command in commands:
        print(f"[Stage2AW] running {command['name']}", flush=True)
        subprocess.run(command["argv"], cwd=REPO_ROOT, check=True)


def main() -> int:
    args = parse_args()
    splits = args.split if args.split is not None else ["hidden"]
    if args.generate_proposals:
        if args.yolo_source == DEFAULT_YOLO_SOURCE:
            args.yolo_source = generated_yolo_source(args)
        if args.source is None:
            multisources = [generated_yolo_source(args), generated_stage2x_source(args)]
        else:
            multisources = args.source
    else:
        multisources = args.source if args.source is not None else list(DEFAULT_MULTISOURCE)
    commands = build_commands(args, splits, multisources)
    checks = preflight(args, splits, multisources)
    missing = missing_checks(checks)
    if (args.strict or args.execute) and missing:
        text = "\n".join(f"- {item['kind']}: {item['path']}" for item in missing)
        raise FileNotFoundError(f"Stage2AW missing required inputs:\n{text}")

    if args.generate_proposals:
        note = (
            "This wrapper generates hidden-ready yolo_stage2l and stage2x_class1 "
            "proposal CSVs from raw images, then runs no-GT candidate pooling, "
            "Stage2U ranking, and Stage2AE/Stage2AQ clean prediction export. "
            "Final challenge-specific result-file conversion remains separate."
        )
    else:
        note = (
            "This wrapper starts from precomputed proposal CSVs, then runs no-GT "
            "candidate pooling, Stage2U ranking, and Stage2AE/Stage2AQ clean "
            "prediction export. Final challenge-specific result-file conversion "
            "remains separate."
        )
    manifest = {
        "artifact_type": "task2_stage2aw_hidden_pipeline",
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "repo_root": REPO_ROOT.as_posix(),
        "mode": "execute" if args.execute else "dry_run",
        "splits": list(splits),
        "yolo_source": args.yolo_source,
        "multisources": list(multisources),
        "preflight_checks": checks,
        "missing_required_inputs": missing,
        "commands": commands,
        "note": note,
    }
    manifest_path = repo_path(args.manifest) if args.manifest is not None else repo_path(args.work_dir) / "stage2aw_manifest.json"
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    manifest_path.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(manifest, indent=2, sort_keys=True), flush=True)

    if args.execute:
        run_commands(commands)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
