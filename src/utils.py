"""Small helpers shared by all scripts: config, paths and seed."""
import random
from pathlib import Path

import numpy as np
import yaml

ROOT = Path(__file__).resolve().parents[1]


def load_config(path: Path = ROOT / "config.yaml") -> dict:
    """Read config.yaml and return it as a dict."""
    with open(path, encoding="utf-8") as f:
        return yaml.safe_load(f)


def get_dir(cfg: dict, key: str, create: bool = False) -> Path:
    """Return a folder from the config. Relative paths start at the repo root."""
    path = Path(cfg[key])
    if not path.is_absolute():
        path = ROOT / path
    if create:
        path.mkdir(parents=True, exist_ok=True)
    return path


def set_seed(seed: int) -> None:
    """Fix the random seed for Python and NumPy (torch is fixed in train_cnn.py)."""
    random.seed(seed)
    np.random.seed(seed)
