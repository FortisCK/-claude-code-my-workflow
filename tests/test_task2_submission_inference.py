from __future__ import annotations

import json
from pathlib import Path

from scripts.task2.run_task2_submission_inference import (
    build_hidden_pipeline_command,
    final_internal_prediction_csv,
    main,
    result_csv_path,
    wrapper_manifest_path,
)


def test_stage2ba_build_hidden_pipeline_command_from_image_dir(tmp_path: Path) -> None:
    class Args:
        python = Path("/usr/bin/python")
        image_dir = tmp_path / "images"
        image_list_csv = None
        output_dir = tmp_path / "out"
        split = "hidden"
        work_subdir = "work"
        result_filename = "task2_predictions_internal.csv"
        fallback_domain = "phantom"
        device = "cpu"
        batch_size = 2
        workers = 0
        proposal_limit = 3
        execute = False
        strict = True

    command = build_hidden_pipeline_command(Args)

    assert command[:3] == [
        "/usr/bin/python",
        "scripts/task2/run_stage2aq_hidden_pipeline.py",
        "--generate-proposals",
    ]
    assert command[command.index("--raw-image-dir") + 1] == Args.image_dir.as_posix()
    assert command[command.index("--work-dir") + 1] == (Args.output_dir / Args.work_subdir).as_posix()
    assert command[command.index("--fallback-domain") + 1] == "phantom"
    assert command[command.index("--proposal-limit") + 1] == "3"
    assert "--execute" not in command
    assert "--strict" in command


def test_stage2ba_paths_are_stable(tmp_path: Path) -> None:
    output_dir = tmp_path / "out"

    assert final_internal_prediction_csv(output_dir, "work", "hidden") == (
        output_dir / "work" / "hidden_stage2ae_stage2aq_predictions" / "stage2aq" / "hidden_domain_policy_predictions.csv"
    )
    assert result_csv_path(output_dir, "task2_predictions_internal.csv") == (
        output_dir / "task2_predictions_internal.csv"
    )
    assert wrapper_manifest_path(output_dir, "work") == (
        output_dir / "work" / "stage2aq_hidden_pipeline_manifest.json"
    )


def test_stage2ba_main_writes_dry_run_manifest(tmp_path: Path, monkeypatch) -> None:
    image_dir = tmp_path / "images"
    output_dir = tmp_path / "out"
    image_dir.mkdir()
    monkeypatch.setattr(
        "sys.argv",
        [
            "run_task2_submission_inference.py",
            "--image-dir",
            image_dir.as_posix(),
            "--output-dir",
            output_dir.as_posix(),
            "--split",
            "hidden",
            "--device",
            "cpu",
            "--batch-size",
            "2",
            "--workers",
            "0",
            "--proposal-limit",
            "2",
        ],
    )

    assert main() == 0
    manifest_path = output_dir / "task2_submission_inference_manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))

    assert manifest["mode"] == "dry_run"
    assert manifest["split"] == "hidden"
    assert manifest["fallback_domain"] == "phantom"
    assert manifest["stable_internal_result_csv"].endswith("task2_predictions_internal.csv")
    assert "official validation package" in manifest["official_format_status"]
    assert "--execute" not in manifest["command"]

