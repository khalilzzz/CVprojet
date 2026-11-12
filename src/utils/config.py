"""Config loading and seeding helpers (reproducibility rule, CLAUDE.md section 1.7)."""
from __future__ import annotations

import random
from pathlib import Path

import numpy as np
import yaml

REPO_ROOT = Path(__file__).resolve().parents[2]


def load_config(path: str | Path) -> dict:
    """Load a YAML config. Relative paths resolve against the repo root."""
    path = Path(path)
    if not path.is_absolute():
        path = REPO_ROOT / path
    with open(path, encoding="utf-8") as f:
        return yaml.safe_load(f)


def set_seed(seed: int) -> np.random.Generator:
    """Seed Python and NumPy global RNGs and return a dedicated Generator."""
    random.seed(seed)
    np.random.seed(seed)
    return np.random.default_rng(seed)
