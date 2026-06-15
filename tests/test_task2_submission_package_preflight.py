from __future__ import annotations

import hashlib
import json
from pathlib import Path

from scripts.task2 import check_task2_submission_package as package_preflight
from scripts.task2.check_task2_submission_package import build_manifest, main


def test_stage2bc_build_manifest_has_required_checks(tmp_path: Path) -> None:
    class Args:
        image_dir = None
        image_list_csv = None
        output_dir = tmp_path / "out"
        manifest = tmp_path / "manifest.json"
        strict = False

    manifest = build_manifest(Args)

    kinds = [item["kind"] for item in manifest["checks"]]
    paths = [item["path"] for item in manifest["checks"]]
    assert "required_file" in kinds
    assert "output_dir_creatable" in kinds
    assert "scripts/task2/run_task2_submission_inference.py" in paths
    assert any(path.endswith("stage2u_warm_stage2o_fulltrain_valid512_e6/checkpoints/best.pt") for path in paths)
    assert "Official Task2 result-file schema" in manifest["official_format_status"]


def test_stage2bc_main_passes_with_current_package(tmp_path: Path, monkeypatch) -> None:
    manifest_path = tmp_path / "preflight.json"
    monkeypatch.setattr(
        "sys.argv",
        [
            "check_task2_submission_package.py",
            "--image-dir",
            "datasets/collision_detection/images",
            "--output-dir",
            (tmp_path / "out").as_posix(),
            "--manifest",
            manifest_path.as_posix(),
            "--strict",
        ],
    )

    assert main() == 0
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    assert manifest["failure_count"] == 0


def test_stage2bc_main_strict_fails_for_missing_input(tmp_path: Path, monkeypatch) -> None:
    manifest_path = tmp_path / "preflight.json"
    monkeypatch.setattr(
        "sys.argv",
        [
            "check_task2_submission_package.py",
            "--image-dir",
            (tmp_path / "missing_images").as_posix(),
            "--manifest",
            manifest_path.as_posix(),
            "--strict",
        ],
    )

    assert main() == 1
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    assert manifest["failure_count"] >= 1
    assert any(item["kind"] == "input_image_dir" for item in manifest["failures"])


def test_stage2bd_checksum_manifest_passes_and_fails(tmp_path: Path) -> None:
    weight = tmp_path / "weight.pt"
    weight.write_bytes(b"checkpoint")
    digest = hashlib.sha256(b"checkpoint").hexdigest()

    original_weights = package_preflight.REQUIRED_WEIGHT_FILES
    package_preflight.REQUIRED_WEIGHT_FILES = (weight.as_posix(),)
    try:
        good_manifest = tmp_path / "good.json"
        good_manifest.write_text(
            json.dumps({"weights": [{"path": weight.as_posix(), "size_bytes": 10, "sha256": digest}]}) + "\n",
            encoding="utf-8",
        )
        bad_manifest = tmp_path / "bad.json"
        bad_manifest.write_text(
            json.dumps({"weights": [{"path": weight.as_posix(), "size_bytes": 10, "sha256": "0" * 64}]}) + "\n",
            encoding="utf-8",
        )

        class GoodArgs:
            image_dir = None
            image_list_csv = None
            output_dir = None
            checksum_manifest = good_manifest
            manifest = tmp_path / "manifest.json"
            strict = False

        class BadArgs:
            image_dir = None
            image_list_csv = None
            output_dir = None
            checksum_manifest = bad_manifest
            manifest = tmp_path / "manifest.json"
            strict = False

        good = package_preflight.build_manifest(GoodArgs)
        bad = package_preflight.build_manifest(BadArgs)
    finally:
        package_preflight.REQUIRED_WEIGHT_FILES = original_weights

    assert good["failure_count"] == 0
    assert any(item["kind"] == "weight_checksum" and item["valid"] for item in good["checks"])
    assert bad["failure_count"] == 1
    assert any(item["kind"] == "weight_checksum" and not item["valid"] for item in bad["failures"])
