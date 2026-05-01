"""imagecas_dataset.py — PyTorch Datasets for KL-VAE and diffusion training.

Two datasets:

    ImageCASCleanDataset
        Yields (volume, heart_mask, case_id) — single clean volume per item.
        Used for KL-VAE Stage-1 training (no motion synth needed).

    ImageCASPairedDataset
        Yields (volume, corrupted, heart_mask, case_id, motion_params).
        Used for conditional diffusion Stage-2 training.
        Supports two modes:
            - "precomputed": pairs pre-generated on disk (production path)
            - "online":      synthesize corrupted on-the-fly per batch (slow,
                             only for testing without precomputed cache)

Cache directory layout:
    data/imagecas/processed/v1/
        case_<id>.npz                    # clean volume (from preprocessing.py)
        case_<id>__pair_<vid>.npz        # paired clean + corrupted (vid = variant id)

Per `.claude/rules/python-code-conventions.md`:
    - pathlib.Path everywhere
    - paths via `code.data.paths`
    - DataLoader pin_memory=True / persistent_workers=True downstream
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Callable, Optional

import numpy as np
import torch
from monai.data import Dataset
from monai.transforms import Compose

from . import paths

log = logging.getLogger(__name__)


# ============================================================
# Clean-only dataset (Stage 1 VAE training)
# ============================================================


class ImageCASCleanDataset(Dataset):
    """Yields clean preprocessed volumes from data/imagecas/processed/v1/case_<id>.npz.

    Each item is a dict (MONAI convention):
        {
            "volume":     torch.FloatTensor (1, Z, Y, X), normalized to [-1, 1]
            "heart_mask": torch.ByteTensor  (1, Z, Y, X)
            "case_id":    str
        }

    A `transform` Compose may be applied to the dict (see transforms.py).
    """

    def __init__(
        self,
        case_ids: list[str],
        processed_dir: Optional[Path] = None,
        transform: Optional[Compose] = None,
    ):
        self.case_ids = list(case_ids)
        self.processed_dir = processed_dir or paths.get("IMAGECAS_PROCESSED")
        # Build MONAI Dataset's `data` list (one dict per item). The
        # base Dataset class applies `transform` to each dict.
        data_list = [{"case_id": cid} for cid in self.case_ids]
        super().__init__(data=data_list, transform=transform)
        self._verify_cache_exists()

    def _verify_cache_exists(self) -> None:
        missing = [
            cid
            for cid in self.case_ids
            if not (self.processed_dir / f"case_{cid}.npz").exists()
        ]
        if missing:
            raise FileNotFoundError(
                f"{len(missing)} case(s) not in processed cache "
                f"{self.processed_dir}. First few: {missing[:5]}. "
                f"Run code.data.preprocessing first."
            )

    def _load_case(self, case_id: str) -> dict:
        npz_path = self.processed_dir / f"case_{case_id}.npz"
        with np.load(npz_path, allow_pickle=True) as data:
            volume = data["volume"]                 # float32 (Z, Y, X)
            heart_mask = data["heart_mask"]         # uint8   (Z, Y, X)
        return {
            "volume": torch.from_numpy(volume).unsqueeze(0).float(),
            "heart_mask": torch.from_numpy(heart_mask).unsqueeze(0).byte(),
            "case_id": case_id,
        }

    def __getitem__(self, index: int) -> dict:
        # MONAI Dataset.__getitem__ would call _transform on data[i]. We override
        # to load from disk on demand instead of holding 1000 volumes in memory.
        item = self._load_case(self.case_ids[index])
        if self.transform is not None:
            item = self.transform(item)
        return item

    def __len__(self) -> int:
        return len(self.case_ids)


# ============================================================
# Paired (clean, corrupted) dataset (Stage 2 diffusion training)
# ============================================================


class ImageCASPairedDataset(Dataset):
    """Yields paired (clean, motion-corrupted) volumes for diffusion training.

    Each item:
        {
            "volume":        torch.FloatTensor (1, Z, Y, X)  # clean / target
            "corrupted":     torch.FloatTensor (1, Z, Y, X)  # condition
            "heart_mask":    torch.ByteTensor (1, Z, Y, X)
            "case_id":       str
            "motion_params": dict (the MotionParams used; for run-card / debug)
        }

    Modes:
        "precomputed" — read pre-generated pair files
                        `case_<id>__pair_<variant>.npz`. variant is sampled
                        uniformly from the available variants for that case.
        "online"      — call code.data.motion_synth.synthesize_motion_artifact()
                        every __getitem__. Slow; for smoke tests / no batch
                        cache available.

    A `transform` Compose may augment both volumes jointly (see
    transforms.diffusion_train_transforms).
    """

    def __init__(
        self,
        case_ids: list[str],
        processed_dir: Optional[Path] = None,
        transform: Optional[Compose] = None,
        mode: str = "precomputed",
        online_motion_factory: Optional[Callable[[str], dict]] = None,
    ):
        if mode not in ("precomputed", "online"):
            raise ValueError(f"mode must be 'precomputed' or 'online', got {mode!r}")
        self.case_ids = list(case_ids)
        self.processed_dir = processed_dir or paths.get("IMAGECAS_PROCESSED")
        self.mode = mode
        self.online_motion_factory = online_motion_factory
        super().__init__(
            data=[{"case_id": cid} for cid in self.case_ids],
            transform=transform,
        )
        # Discover variants per case (precomputed mode only).
        self._variants_per_case: dict[str, list[Path]] = {}
        if self.mode == "precomputed":
            self._discover_variants()

    def _discover_variants(self) -> None:
        for cid in self.case_ids:
            pattern = f"case_{cid}__pair_*.npz"
            self._variants_per_case[cid] = sorted(self.processed_dir.glob(pattern))
        missing = [cid for cid, v in self._variants_per_case.items() if not v]
        if missing:
            raise FileNotFoundError(
                f"{len(missing)} case(s) have no precomputed pairs in "
                f"{self.processed_dir}. First few: {missing[:5]}. "
                f"Run scripts/python/generate_motion_pairs.py first, "
                f"or use mode='online'."
            )

    def _load_pair_precomputed(self, case_id: str) -> dict:
        variants = self._variants_per_case[case_id]
        # Random uniform pick (each call may yield a different variant for the
        # same case — augmentation diversity comes from this).
        idx = int(torch.randint(0, len(variants), (1,)).item())
        npz_path = variants[idx]
        with np.load(npz_path, allow_pickle=True) as data:
            volume = data["volume"]
            corrupted = data["corrupted"]
            heart_mask = data["heart_mask"]
            metadata = data["metadata"].item() if "metadata" in data.files else {}
        return {
            "volume": torch.from_numpy(volume).unsqueeze(0).float(),
            "corrupted": torch.from_numpy(corrupted).unsqueeze(0).float(),
            "heart_mask": torch.from_numpy(heart_mask).unsqueeze(0).byte(),
            "case_id": case_id,
            "motion_params": metadata.get("motion_params", {}),
        }

    def _load_pair_online(self, case_id: str) -> dict:
        if self.online_motion_factory is None:
            raise RuntimeError(
                "mode='online' requires `online_motion_factory` callable."
            )
        return self.online_motion_factory(case_id)

    def __getitem__(self, index: int) -> dict:
        case_id = self.case_ids[index]
        if self.mode == "precomputed":
            item = self._load_pair_precomputed(case_id)
        else:
            item = self._load_pair_online(case_id)
        if self.transform is not None:
            item = self.transform(item)
        return item

    def __len__(self) -> int:
        return len(self.case_ids)
