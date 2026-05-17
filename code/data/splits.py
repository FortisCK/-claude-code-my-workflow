"""splits.py — load fixed train/val/test splits from JSON.

A versioned JSON file (e.g. `data/imagecas/splits/v1.json`) is the single
source of truth for which case IDs go into train / val / test. It is
generated **once** with a deterministic seed and committed to git so all
training and evaluation runs use the same partition.

The schema is:

    {
        "_meta": { ... documentation ... },
        "train": ["1", "2", ...],
        "val":   ["9", "16", ...],
        "test":  ["21", "32", ...]
    }

Per `.claude/rules/python-code-conventions.md`:
    - relative paths only (no /home/... in code)
    - type hints on public functions
"""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass
from pathlib import Path

log = logging.getLogger(__name__)


@dataclass(frozen=True)
class CaseSplit:
    """Train/val/test partition by case id."""

    train: list[str]
    val: list[str]
    test: list[str]
    meta: dict

    @property
    def all_ids(self) -> list[str]:
        return list(self.train) + list(self.val) + list(self.test)


def load_split(split_file: Path | str) -> CaseSplit:
    """Read the split JSON and return a `CaseSplit`. Disjoint check enforced."""
    p = Path(split_file)
    if not p.exists():
        raise FileNotFoundError(f"split file not found: {p}")
    payload = json.loads(p.read_text())
    train = list(payload["train"])
    val = list(payload["val"])
    test = list(payload["test"])
    meta = dict(payload.get("_meta", {}))

    s_train, s_val, s_test = set(train), set(val), set(test)
    if s_train & s_val or s_val & s_test or s_train & s_test:
        raise ValueError(f"split {p} has overlapping ids — corrupt split file")

    log.info(
        "[split] %s: train=%d val=%d test=%d (version=%s)",
        p, len(train), len(val), len(test), meta.get("version", "?"),
    )
    return CaseSplit(train=train, val=val, test=test, meta=meta)
