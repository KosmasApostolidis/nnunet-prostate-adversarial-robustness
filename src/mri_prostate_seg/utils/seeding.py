"""Deterministic seeding helpers shared by experiment runners."""

from __future__ import annotations

import os
import random
from typing import Any

import numpy as np


def set_seed(seed: int | None) -> None:
    """Seed Python ``random``, NumPy, and torch (if available) deterministically.

    Passing ``None`` is a no-op so callers can wire ``--seed`` flags directly
    without an extra branch.
    """
    if seed is None:
        return

    random.seed(seed)
    np.random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)

    try:
        import torch  # local import keeps utils/ free of torch at import time
    except ModuleNotFoundError:
        return

    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def seeded_rng(seed: int | None) -> np.random.Generator:
    """Return a NumPy ``Generator`` seeded with ``seed`` (``None`` ⇒ entropy)."""
    return np.random.default_rng(seed)


def torch_seeded_generator(seed: int | None) -> Any:
    """Return a ``torch.Generator`` seeded with ``seed`` (``None`` ⇒ entropy).

    Returns ``None`` if torch is not importable (callers can fall back to NumPy).
    """
    try:
        import torch
    except ModuleNotFoundError:
        return None

    g = torch.Generator()
    if seed is not None:
        g.manual_seed(int(seed))
    return g
