import csv
from pathlib import Path

import numpy as np
import pytest
import torch
from PIL import Image

from cathaction.models import TinyUNet
from cathaction.training.task1_baseline import (
    DEFAULT_CONFIG,
    Task1SegmentationDataset,
    build_loss,
    build_model,
    build_train_sampler,
    merge_config,
    predict_logits,
    primary_logits_for_label_mode,
    _is_better_metric,
    _metric_from_path,
    _binary_soft_dice_loss,
    _resize_multiclass_012_mask,
    _soft_cldice_loss,
)


def test_tiny_unet_preserves_input_shape() -> None:
    model = TinyUNet(in_channels=3, out_channels=3, base_channels=4)
    output = model(torch.zeros((2, 3, 31, 29), dtype=torch.float32))

    assert tuple(output.shape) == (2, 3, 31, 29)


def test_smp_unet_branch_preserves_input_shape() -> None:
    pytest.importorskip("segmentation_models_pytorch")
    config = merge_config(
        DEFAULT_CONFIG,
        {
            "data": {"label_mode": "multiclass_012"},
            "model": {
                "name": "smp",
                "architecture": "Unet",
                "encoder_name": "resnet18",
                "encoder_weights": "none",
                "in_channels": 3,
                "out_channels": 3,
            },
        },
    )

    model = build_model(config).eval()
    with torch.no_grad():
        output = model(torch.zeros((1, 3, 64, 64), dtype=torch.float32))

    assert tuple(output.shape) == (1, 3, 64, 64)


def test_toolness_auxiliary_model_uses_four_output_channels() -> None:
    config = merge_config(
        DEFAULT_CONFIG,
        {
            "data": {"label_mode": "multiclass_012"},
            "model": {
                "name": "tiny_unet",
                "in_channels": 3,
                "out_channels": 4,
                "base_channels": 4,
                "toolness_auxiliary": True,
            },
            "loss": {"name": "monai_dice_ce_toolness_aux"},
        },
    )

    model = build_model(config).eval()
    with torch.no_grad():
        output = model(torch.zeros((1, 3, 32, 32), dtype=torch.float32))

    assert tuple(output.shape) == (1, 4, 32, 32)


def test_manifest_dataset_resizes_and_binarizes_masks(tmp_path: Path) -> None:
    image_path = tmp_path / "image.png"
    mask_path = tmp_path / "mask.npy"
    manifest_path = tmp_path / "manifest.csv"

    Image.fromarray(np.zeros((3, 4, 3), dtype=np.uint8)).save(image_path)
    np.save(mask_path, np.array([[0, 1, 2, 0], [2, 0, 0, 1], [0, 0, 0, 0]], dtype=np.uint8))
    with manifest_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=[
                "split_role",
                "collection",
                "domain",
                "released_split",
                "sample_id",
                "case_id",
                "case_id_source",
                "image_path",
                "mask_path",
                "mask_encoding",
            ],
        )
        writer.writeheader()
        writer.writerow(
            {
                "split_role": "released_train",
                "collection": "animal_train",
                "domain": "animal",
                "released_split": "train",
                "sample_id": "ani_00000",
                "case_id": "",
                "case_id_source": "unavailable_from_filename",
                "image_path": "image.png",
                "mask_path": "mask.npy",
                "mask_encoding": "npy_multiclass",
            }
        )

    dataset = Task1SegmentationDataset(
        manifest_csv=manifest_path,
        repo_root=tmp_path,
        image_size=(8, 8),
        label_mode="binary_foreground",
    )
    sample = dataset[0]

    assert tuple(sample["image"].shape) == (3, 8, 8)
    assert tuple(sample["mask"].shape) == (1, 8, 8)
    assert set(torch.unique(sample["mask"]).tolist()) <= {0.0, 1.0}
    assert sample["sample_id"] == "ani_00000"


def test_manifest_dataset_resizes_multiclass_masks(tmp_path: Path) -> None:
    image_path = tmp_path / "image.png"
    mask_path = tmp_path / "mask.npy"
    manifest_path = tmp_path / "manifest.csv"

    Image.fromarray(np.zeros((3, 4, 3), dtype=np.uint8)).save(image_path)
    np.save(mask_path, np.array([[0, 1, 2, 0], [2, 0, 0, 1], [0, 0, 0, 0]], dtype=np.uint8))
    _write_manifest(manifest_path)

    dataset = Task1SegmentationDataset(
        manifest_csv=manifest_path,
        repo_root=tmp_path,
        image_size=(8, 8),
        label_mode="multiclass_012",
    )
    sample = dataset[0]

    assert tuple(sample["image"].shape) == (3, 8, 8)
    assert tuple(sample["mask"].shape) == (8, 8)
    assert sample["mask"].dtype == torch.long
    assert set(torch.unique(sample["mask"]).tolist()) <= {0, 1, 2}


def test_manifest_dataset_applies_configured_image_normalization(tmp_path: Path) -> None:
    image_path = tmp_path / "image.png"
    mask_path = tmp_path / "mask.npy"
    manifest_path = tmp_path / "manifest.csv"

    Image.fromarray(np.full((3, 4, 3), 255, dtype=np.uint8)).save(image_path)
    np.save(mask_path, np.zeros((3, 4), dtype=np.uint8))
    _write_manifest(manifest_path)

    dataset = Task1SegmentationDataset(
        manifest_csv=manifest_path,
        repo_root=tmp_path,
        image_size=(3, 4),
        label_mode="multiclass_012",
        normalization={
            "enabled": True,
            "mean": [1.0, 1.0, 1.0],
            "std": [1.0, 1.0, 1.0],
        },
    )
    sample = dataset[0]

    assert torch.allclose(sample["image"], torch.zeros_like(sample["image"]))


def test_aspect_pad_resize_preserves_mask_classes() -> None:
    mask = np.full((2, 4), 2, dtype=np.uint8)

    resized = _resize_multiclass_012_mask(mask, (8, 8), resize_mode="aspect_pad")

    assert resized.shape == (8, 8)
    assert np.all(resized[:2, :] == 0)
    assert np.all(resized[2:6, :] == 2)
    assert np.all(resized[6:, :] == 0)
    assert set(np.unique(resized).tolist()) == {0, 2}


def test_domain_balanced_sampler_weights_inverse_domain_frequency(tmp_path: Path) -> None:
    manifest_path = tmp_path / "manifest.csv"
    _write_manifest_rows(
        manifest_path,
        [
            {"sample_id": "animal_0", "domain": "animal", "collection": "animal_train"},
            {"sample_id": "phantom_0", "domain": "phantom", "collection": "phantom_train"},
            {"sample_id": "phantom_1", "domain": "phantom", "collection": "phantom_train"},
            {"sample_id": "phantom_2", "domain": "phantom", "collection": "phantom_train"},
        ],
    )
    dataset = Task1SegmentationDataset(
        manifest_csv=manifest_path,
        repo_root=tmp_path,
        image_size=(8, 8),
        label_mode="multiclass_012",
    )

    sampler = build_train_sampler(dataset, {"sampling": {"domain_balanced": True}})

    assert sampler is not None
    assert sampler.num_samples == 4
    assert sampler.weights.tolist() == pytest.approx([1.0, 1 / 3, 1 / 3, 1 / 3])


def test_soft_cldice_loss_rewards_line_overlap() -> None:
    target = torch.zeros((1, 5, 5), dtype=torch.long)
    target[0, 2, 1:4] = 1
    logits_good = torch.full((1, 3, 5, 5), -4.0)
    logits_good[:, 0] = 4.0
    logits_good[:, 1, 2, 1:4] = 6.0
    logits_good[:, 0, 2, 1:4] = -4.0
    logits_bad = torch.zeros_like(logits_good)

    good_loss = _soft_cldice_loss(
        logits_good,
        target,
        labels=[1],
        class_weights=torch.tensor([1.0]),
        iterations=3,
    )
    bad_loss = _soft_cldice_loss(
        logits_bad,
        target,
        labels=[1],
        class_weights=torch.tensor([1.0]),
        iterations=3,
    )

    assert good_loss.item() < bad_loss.item()


def test_primary_logits_excludes_auxiliary_channels() -> None:
    logits = torch.zeros((2, 4, 8, 8), dtype=torch.float32)

    primary = primary_logits_for_label_mode(logits, "multiclass_012")

    assert tuple(primary.shape) == (2, 3, 8, 8)


def test_predict_logits_ignores_toolness_auxiliary_channel() -> None:
    logits = torch.zeros((1, 4, 2, 2), dtype=torch.float32)
    logits[:, 1] = 5.0
    logits[:, 3] = 100.0

    prediction = predict_logits(logits, label_mode="multiclass_012", threshold=0.5)

    assert prediction.tolist() == [[[1, 1], [1, 1]]]


def test_binary_soft_dice_loss_rewards_foreground_overlap() -> None:
    target = torch.zeros((1, 1, 4, 4), dtype=torch.float32)
    target[:, :, 1:3, 1:3] = 1.0
    logits_good = torch.full((1, 1, 4, 4), -5.0)
    logits_good[:, :, 1:3, 1:3] = 5.0
    logits_bad = torch.zeros_like(logits_good)

    assert _binary_soft_dice_loss(logits_good, target) < _binary_soft_dice_loss(logits_bad, target)


def test_weighted_cross_entropy_uses_configured_class_weights() -> None:
    config = merge_config(
        DEFAULT_CONFIG,
        {
            "data": {"label_mode": "multiclass_012"},
            "model": {"out_channels": 3},
            "loss": {
                "name": "weighted_cross_entropy",
                "class_weights": [0.1, 1.0, 3.0],
            },
        },
    )

    loss = build_loss(config, device=torch.device("cpu"))

    assert loss.weight is not None
    assert loss.weight.tolist() == pytest.approx([0.1, 1.0, 3.0])


def test_metric_monitor_helpers() -> None:
    payload = {"epoch": 3, "eval": {"dice": 0.42, "loss": 0.12}}

    assert _metric_from_path(payload, "eval.dice") == pytest.approx(0.42)
    assert _is_better_metric(0.43, 0.42, "max")
    assert not _is_better_metric(0.41, 0.42, "max")
    assert _is_better_metric(0.11, 0.12, "min")
    assert not _is_better_metric(0.13, 0.12, "min")


def _write_manifest(path: Path) -> None:
    _write_manifest_rows(
        path,
        [
            {
                "sample_id": "ani_00000",
                "domain": "animal",
                "collection": "animal_train",
            }
        ],
    )


def _write_manifest_rows(path: Path, rows: list[dict[str, str]]) -> None:
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=[
                "split_role",
                "collection",
                "domain",
                "released_split",
                "sample_id",
                "case_id",
                "case_id_source",
                "image_path",
                "mask_path",
                "mask_encoding",
            ],
        )
        writer.writeheader()
        for row in rows:
            writer.writerow(
                {
                    "split_role": "released_train",
                    "collection": row["collection"],
                    "domain": row["domain"],
                    "released_split": "train",
                    "sample_id": row["sample_id"],
                    "case_id": "",
                    "case_id_source": "unavailable_from_filename",
                    "image_path": "image.png",
                    "mask_path": "mask.npy",
                    "mask_encoding": "npy_multiclass",
                }
            )
