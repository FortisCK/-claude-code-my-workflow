from __future__ import annotations

import json
from pathlib import Path

from scripts.task2.plan_stage2ax_proposal_generation import SOURCE_STATUS, main, stage2x_command, yolo_command


def test_stage2ax_source_status_marks_only_yolo_hidden_ready() -> None:
    assert SOURCE_STATUS["yolo_stage2l"]["status"] == "hidden_ready"
    assert SOURCE_STATUS["stage2x_class1"]["status"] == "hidden_ready"
    assert SOURCE_STATUS["task1_geometry_rect"]["status"] == "needs_nogt_adapter"


def test_stage2ax_yolo_command_uses_nogt_exporter(tmp_path: Path) -> None:
    class Args:
        python = Path("/usr/bin/python")
        yolo_weights = Path("best.pt")
        stage2x_checkpoint = Path("stage2x.pt")
        split = "hidden"
        output_dir = tmp_path / "out"
        name = "yolo"
        device = "cpu"
        imgsz = 1024
        conf = 0.001
        iou = 0.7
        max_det = 20
        batch_size = 4
        source_chunk_size = 8
        fallback_domain = "human"
        image_dir = tmp_path / "images"
        image_list_csv = None

    cmd = yolo_command(Args)

    assert "scripts/task2/export_yolo_nogt_proposals.py" in cmd
    assert "--image-dir" in cmd
    assert "--weights" in cmd
    assert "--fallback-domain" in cmd

    stage2x_cmd = stage2x_command(Args)
    assert "scripts/task2/export_sequence_tip_nogt_proposals.py" in stage2x_cmd
    assert "--checkpoint" in stage2x_cmd
    assert "--image-dir" in stage2x_cmd


def test_stage2ax_main_writes_manifest(tmp_path: Path, monkeypatch) -> None:
    image_dir = tmp_path / "images"
    image_dir.mkdir()
    manifest_path = tmp_path / "manifest.json"
    monkeypatch.setattr(
        "sys.argv",
        [
            "plan_stage2ax_proposal_generation.py",
            "--image-dir",
            image_dir.as_posix(),
            "--manifest",
            manifest_path.as_posix(),
            "--yolo-weights",
            (tmp_path / "missing.pt").as_posix(),
            "--stage2x-checkpoint",
            (tmp_path / "missing_stage2x.pt").as_posix(),
        ],
    )

    assert main() == 0
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))

    assert manifest["artifact_type"] == "task2_stage2ax_proposal_generation_plan"
    assert [command["name"] for command in manifest["commands"]] == [
        "export_yolo_stage2l_nogt_proposals",
        "export_stage2x_class1_nogt_proposals",
    ]
    assert "stage2x_class1" not in manifest["remaining_adapters"]
