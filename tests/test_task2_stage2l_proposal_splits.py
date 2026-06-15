from __future__ import annotations

from pathlib import Path

from scripts.task2.prepare_stage2l_proposal_splits import prepare_stage2l_proposal_splits


def test_stage2l_proposal_splits_include_animal_normal_without_val_overlap(tmp_path: Path) -> None:
    repo_root = Path.cwd()

    summary = prepare_stage2l_proposal_splits(
        source_data_root=repo_root / "datasets/collision_detection",
        agnostic_data_root=repo_root / "datasets/collision_detection_agnostic",
        base_split_dir=repo_root / "configs/task2/splits",
        stage2j_split_dir=repo_root / "configs/task2/splits_stage2j_balanced",
        output_dir=tmp_path / "splits_stage2l_proposal",
        config_dir=tmp_path / "configs",
        repo_root=repo_root,
        class_name="tool_roi",
        seed=2026,
    )

    assert summary["animal_train_videos"] == ["video_0_animal", "video_1_animal"]
    assert summary["animal_val_videos"] == ["video_2_animal"]

    train_panel = summary["panels"]["train_v0_v1_val_v2_train"]
    valid_panel = summary["panels"]["train_v0_v1_val_v2_valid_animal"]
    combined_panel = summary["panels"]["train_v0_v1_val_v2_valid_combined_balanced"]
    assert train_panel["domain_counts"]["animal"] == 291
    assert train_panel["original_class_counts"]["0"] >= 108
    assert valid_panel["original_class_counts"] == {"0": 15, "1": 92}
    assert combined_panel["samples"] == 931
    assert combined_panel["original_class_counts"] == {"0": 427, "1": 504}
    assert combined_panel["domain_counts"] == {"animal": 107, "phantom": 824}

    lists = summary["lists"]
    train_label_list = repo_root / lists["train_v0_v1_val_v2_train"]["labels"]
    train_image_list = repo_root / lists["train_v0_v1_val_v2_train"]["images"]
    valid_label_list = repo_root / lists["train_v0_v1_val_v2_valid_animal"]["labels"]
    combined_label_list = repo_root / lists["train_v0_v1_val_v2_valid_combined_balanced"]["labels"]
    combined_image_list = repo_root / lists["train_v0_v1_val_v2_valid_combined_balanced"]["images"]
    train_labels = read_names(train_label_list)
    valid_labels = read_names(valid_label_list)
    combined_labels = read_names(combined_label_list)
    assert train_labels.isdisjoint(valid_labels)
    assert train_labels.isdisjoint(combined_labels)
    assert any(name.startswith("video_0_animal_") for name in train_labels)
    assert any(name.startswith("video_1_animal_") for name in train_labels)
    assert all(name.startswith("video_2_animal_") for name in valid_labels)
    assert any(name.startswith("video_2_animal_") for name in combined_labels)
    assert any(name.startswith("video_0_") for name in combined_labels)

    for label_path in read_paths(train_label_list)[:10]:
        assert label_path.read_text(encoding="utf-8").split()[0] == "0"
    for image_path in read_paths(train_image_list)[:10]:
        assert "collision_detection_agnostic/images" in image_path.as_posix()
    for image_path in read_paths(combined_image_list)[:10]:
        assert "collision_detection_agnostic/images" in image_path.as_posix()

    yaml_paths = summary["yolo_yaml"]
    assert set(yaml_paths) == {
        "animal",
        "combined_balanced",
        "valid_phantom_balanced_small",
        "valid_phantom_full",
    }
    for yaml_path in yaml_paths.values():
        assert (repo_root / yaml_path).is_file() or Path(yaml_path).is_file()


def read_names(path: Path) -> set[str]:
    return {Path(line.strip()).name for line in path.read_text(encoding="utf-8").splitlines() if line.strip()}


def read_paths(path: Path) -> list[Path]:
    return [Path(line.strip()) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]
