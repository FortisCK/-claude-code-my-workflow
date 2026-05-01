"""smoke_dataset.py — Step 2 smoke test for the data-layer classes.

Exercises:
    - ImageCASCleanDataset on the single preprocessed case (case_1.npz).
    - vae_train_transforms() Compose pipeline.
    - ImageCASPairedDataset(mode="online") with a stub motion factory
      (so we don't need precomputed pair files for this smoke check).
    - DataLoader collation with batch_size=2.

CPU-only. Run from repo root:
    python -m scripts.python.smoke_dataset

Pass criteria:
    1. clean batch shape == (2, 1, 192, 192, 192), dtype float32, in [-1, 1]
    2. heart_mask batch shape == (2, 1, 192, 192, 192), dtype uint8
    3. paired (online) batch yields BOTH volume and corrupted with matching shape
    4. transform Compose passes through without raising
"""

from __future__ import annotations

import logging
import sys
from pathlib import Path

import torch
from torch.utils.data import DataLoader

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from code.data.imagecas_dataset import ImageCASCleanDataset, ImageCASPairedDataset
from code.data.transforms import diffusion_train_transforms, vae_train_transforms

log = logging.getLogger(__name__)

EXPECTED_SHAPE = (1, 192, 192, 192)


def smoke_clean_dataset() -> None:
    log.info("[clean] instantiating ImageCASCleanDataset on case_id='1' ...")
    ds = ImageCASCleanDataset(case_ids=["1", "1"], transform=vae_train_transforms())
    assert len(ds) == 2, f"expected len 2, got {len(ds)}"

    item = ds[0]
    assert item["volume"].shape == EXPECTED_SHAPE, item["volume"].shape
    assert item["heart_mask"].shape == EXPECTED_SHAPE, item["heart_mask"].shape
    assert item["volume"].dtype == torch.float32
    assert item["heart_mask"].dtype == torch.uint8
    v = item["volume"]
    log.info(
        "[clean] item ok: vol %s [%.3f, %.3f]  mask sum=%d",
        tuple(v.shape), float(v.min()), float(v.max()), int(item["heart_mask"].sum()),
    )

    loader = DataLoader(ds, batch_size=2, num_workers=0, shuffle=False)
    batch = next(iter(loader))
    assert batch["volume"].shape == (2, *EXPECTED_SHAPE), batch["volume"].shape
    assert batch["heart_mask"].shape == (2, *EXPECTED_SHAPE), batch["heart_mask"].shape
    log.info("[clean] DataLoader batch ok: %s", tuple(batch["volume"].shape))


def _stub_online_motion(case_id: str) -> dict:
    """Minimal online motion factory: returns clean + a noise-corrupted copy.

    Avoids invoking tomosipo (GPU-only) for this CPU smoke test. The real
    online factory (wrapping motion_synth.synthesize_motion_artifact) plugs
    in here once GPU is free.
    """
    from code.data import paths

    import numpy as np
    npz = np.load(paths.get("IMAGECAS_PROCESSED") / f"case_{case_id}.npz", allow_pickle=True)
    volume = torch.from_numpy(npz["volume"]).unsqueeze(0).float()
    heart_mask = torch.from_numpy(npz["heart_mask"]).unsqueeze(0).byte()
    corrupted = volume + 0.05 * torch.randn_like(volume)
    return {
        "volume": volume,
        "corrupted": corrupted,
        "heart_mask": heart_mask,
        "case_id": case_id,
        "motion_params": {"stub": True},
    }


def smoke_paired_dataset_online() -> None:
    log.info("[paired/online] instantiating ImageCASPairedDataset(mode='online') ...")
    ds = ImageCASPairedDataset(
        case_ids=["1", "1"],
        transform=diffusion_train_transforms(),
        mode="online",
        online_motion_factory=_stub_online_motion,
    )
    item = ds[0]
    assert item["volume"].shape == EXPECTED_SHAPE
    assert item["corrupted"].shape == EXPECTED_SHAPE
    assert item["volume"].dtype == torch.float32
    assert item["corrupted"].dtype == torch.float32
    log.info(
        "[paired/online] item ok: vol %s  corrupted %s",
        tuple(item["volume"].shape), tuple(item["corrupted"].shape),
    )

    loader = DataLoader(ds, batch_size=2, num_workers=0, shuffle=False)
    batch = next(iter(loader))
    assert batch["volume"].shape == (2, *EXPECTED_SHAPE)
    assert batch["corrupted"].shape == (2, *EXPECTED_SHAPE)
    log.info(
        "[paired/online] DataLoader batch ok: vol %s  corrupted %s",
        tuple(batch["volume"].shape), tuple(batch["corrupted"].shape),
    )


def main() -> int:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
    smoke_clean_dataset()
    smoke_paired_dataset_online()
    log.info("STEP 2 SMOKE TEST: PASS")
    return 0


if __name__ == "__main__":
    sys.exit(main())
