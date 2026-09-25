"""Shell rankings, cumulative sets, K->cycles/mm, a-priori band controls (SSEUA)."""

from __future__ import annotations

import math
from collections.abc import Sequence

import numpy as np

from .ranking import rank_order

RANKINGS = ("frequency_low_first", "frequency_high_first", "utility", "energy", "ig")
_NAN = float("nan")


def ranking_orders(
    necessity: np.ndarray, energy: np.ndarray, ig: np.ndarray, empty: np.ndarray
) -> dict[str, np.ndarray]:
    """0-based shell orders per ranking; empty shells are dropped from every order."""

    ids = np.nonzero(~np.asarray(empty, dtype=bool))[0]

    def by(values: np.ndarray) -> np.ndarray:
        return ids[rank_order(np.asarray(values, dtype=np.float64)[ids])]

    return {
        "frequency_low_first": ids.copy(),
        "frequency_high_first": ids[::-1].copy(),
        "utility": by(necessity),
        "energy": by(energy),
        "ig": by(ig),
    }


def cumulative_masks(masks: np.ndarray, order: np.ndarray) -> np.ndarray:
    """Running union of ``masks[order[:k]]`` for k = 1..len(order)."""

    return np.logical_or.accumulate(
        np.asarray(masks, dtype=bool)[np.asarray(order)], axis=0
    )


def k_to_cpm(order: np.ndarray, k: float, edges: np.ndarray) -> float:
    """Upper edge (cycles/mm) of the highest shell among the first ``k`` of ``order``."""

    if not math.isfinite(float(k)):
        return _NAN
    k = int(k)
    if k < 1 or k > len(order):
        return _NAN
    highest = int(np.max(np.asarray(order)[:k]))
    return float(np.asarray(edges)[highest + 1])


def band_indices(band: str, n_shells: int) -> list[int]:
    width = n_shells // 3
    start = {"LF": 0, "MF": width, "HF": 2 * width}[band]
    end = n_shells if band == "HF" else start + width
    return list(range(start, end))


def count_matched_draws(
    band_idx: Sequence[int], n_shells: int, trials: int, rng: np.random.Generator
) -> list[list[int]]:
    """Distinct random complement subsets with the band's shell count."""

    excluded = set(band_idx)
    complement = np.array([i for i in range(n_shells) if i not in excluded])
    size = len(band_idx)
    limit = math.comb(len(complement), size)
    seen: set[frozenset[int]] = set()
    draws: list[list[int]] = []
    while len(draws) < min(trials, limit):
        pick = sorted(int(v) for v in rng.choice(complement, size=size, replace=False))
        key = frozenset(pick)
        if key in seen:
            continue
        seen.add(key)
        draws.append(pick)
    return draws


def energy_matched_draws(
    band_idx: Sequence[int],
    energies: np.ndarray,
    trials: int,
    rng: np.random.Generator,
    *,
    tol: float = 0.10,
    max_tries: int = 2000,
) -> tuple[list[list[int]], int]:
    """Random complement subsets whose energy is within ``tol`` of the band's.

    Returns the admissible draws and the number of tries spent; an empty list
    after ``max_tries`` means the band could not be energy-matched from the
    rest of the spectrum, which is itself a reportable outcome.
    """

    e = np.asarray(energies, dtype=np.float64)
    excluded = set(band_idx)
    target = float(e[list(band_idx)].sum())
    complement = np.array([i for i in range(e.size) if i not in excluded])
    draws: list[list[int]] = []
    seen: set[frozenset[int]] = set()
    tries = 0
    while len(draws) < trials and tries < max_tries:
        tries += 1
        size = int(rng.integers(1, complement.size + 1))
        pick = sorted(int(v) for v in rng.choice(complement, size=size, replace=False))
        key = frozenset(pick)
        if key in seen:
            continue
        if abs(float(e[pick].sum()) - target) <= tol * target:
            seen.add(key)
            draws.append(pick)
    return draws, tries


def permutation_p(observed: Sequence[float], control: Sequence[float]) -> float:
    """(#{control >= observed} + 1) / (n + 1), paired trial-wise; NaN if no pairs."""

    pairs = [
        (o, c)
        for o, c in zip(observed, control)
        if math.isfinite(o) and math.isfinite(c)
    ]
    if not pairs:
        return _NAN
    return float((sum(1 for o, c in pairs if c >= o) + 1) / (len(pairs) + 1))


__all__ = [
    "RANKINGS",
    "band_indices",
    "count_matched_draws",
    "cumulative_masks",
    "energy_matched_draws",
    "k_to_cpm",
    "permutation_p",
    "ranking_orders",
]
