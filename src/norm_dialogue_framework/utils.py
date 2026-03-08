"""
Shared utility functions for the Norm-Constrained Dialogue Framework.
"""

from __future__ import annotations

import json
import logging
import os
import random
from pathlib import Path
from typing import Any

import numpy as np


# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------


def get_logger(name: str, level: str = "INFO") -> logging.Logger:
    """Return a consistently formatted logger."""
    logger = logging.getLogger(name)
    if not logger.handlers:
        handler = logging.StreamHandler()
        formatter = logging.Formatter(
            "%(asctime)s | %(levelname)-8s | %(name)s | %(message)s",
            datefmt="%Y-%m-%d %H:%M:%S",
        )
        handler.setFormatter(formatter)
        logger.addHandler(handler)
    logger.setLevel(getattr(logging, level.upper(), logging.INFO))
    return logger


# ---------------------------------------------------------------------------
# Reproducibility
# ---------------------------------------------------------------------------


def set_seed(seed: int = 42) -> None:
    """Set random seeds for reproducibility across random, numpy, and (optionally) torch."""
    random.seed(seed)
    np.random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    try:
        import torch

        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
    except ImportError:
        pass


# ---------------------------------------------------------------------------
# File I/O
# ---------------------------------------------------------------------------


def ensure_dir(path: str | Path) -> Path:
    """Create directory (and parents) if it does not exist."""
    p = Path(path)
    p.mkdir(parents=True, exist_ok=True)
    return p


def save_json(data: Any, path: str | Path, indent: int = 2) -> None:
    """Serialise *data* to JSON and write to *path*."""
    path = Path(path)
    ensure_dir(path.parent)
    with path.open("w", encoding="utf-8") as fh:
        json.dump(data, fh, indent=indent, default=_json_default)


def load_json(path: str | Path) -> Any:
    """Load JSON from *path* and return as Python object."""
    with Path(path).open("r", encoding="utf-8") as fh:
        return json.load(fh)


def _json_default(obj: Any) -> Any:
    """JSON serialiser for types not natively supported."""
    if hasattr(obj, "isoformat"):
        return obj.isoformat()
    if hasattr(obj, "model_dump"):
        return obj.model_dump()
    if isinstance(obj, np.integer):
        return int(obj)
    if isinstance(obj, np.floating):
        return float(obj)
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    raise TypeError(f"Object of type {type(obj)} is not JSON serialisable")


# ---------------------------------------------------------------------------
# Text helpers
# ---------------------------------------------------------------------------


def truncate(text: str, max_chars: int = 120) -> str:
    """Truncate *text* to *max_chars* characters with an ellipsis."""
    return text if len(text) <= max_chars else text[:max_chars].rstrip() + "…"


def clamp(value: float, lo: float = 0.0, hi: float = 1.0) -> float:
    """Clamp *value* to the interval [*lo*, *hi*]."""
    return max(lo, min(hi, value))


def softmax(values: list[float]) -> list[float]:
    """Compute softmax over a list of floats."""
    arr = np.array(values, dtype=float)
    arr -= arr.max()
    exp = np.exp(arr)
    return (exp / exp.sum()).tolist()


def weighted_average(scores: dict[str, float], weights: dict[str, float]) -> float:
    """Compute a weighted average of *scores* using *weights*.

    Only keys present in both dicts are included.  Weights are
    normalised to sum to 1.0.
    """
    common = set(scores) & set(weights)
    if not common:
        return 0.0
    total_w = sum(weights[k] for k in common)
    if total_w == 0.0:
        return 0.0
    return sum(scores[k] * weights[k] for k in common) / total_w


# ---------------------------------------------------------------------------
# Sampling helpers
# ---------------------------------------------------------------------------


def sample_uniform(lo: float, hi: float, rng: random.Random | None = None) -> float:
    """Sample a float uniformly from [*lo*, *hi*]."""
    r = rng or random
    return r.uniform(lo, hi)


def sample_choice(options: list[Any], rng: random.Random | None = None) -> Any:
    """Sample one item from *options*."""
    r = rng or random
    return r.choice(options)
