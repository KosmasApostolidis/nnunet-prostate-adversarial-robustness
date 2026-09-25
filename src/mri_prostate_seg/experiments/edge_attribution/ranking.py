"""Rankings, cumulative subset curves, screening, subcohorts (plan §21–§24, §26)."""

from __future__ import annotations

import math

import numpy as np
from scipy.stats import spearmanr

_NAN = float("nan")


def rank_order(values: np.ndarray, *, descending: bool = True) -> np.ndarray:
    v = np.asarray(values, dtype=np.float64)
    key = -v if descending else v
    key = np.where(np.isnan(v), np.inf, key)
    return np.argsort(key, kind="stable")


def cumulative_curves(
    order: np.ndarray,
    *,
    energy: np.ndarray,
    voxels: np.ndarray,
    e_total: float,
    e_edges: float,
    v_roi: int,
    damage_removed: np.ndarray | None,
    damage_kept: np.ndarray | None,
    d_full: float,
) -> list[dict[str, float | int]]:
    e = np.asarray(energy, dtype=np.float64)[order]
    n = np.asarray(voxels, dtype=np.float64)[order]
    k_max = int(order.size)
    cum_e = np.cumsum(e)
    cum_v = np.cumsum(n)
    damaging = math.isfinite(d_full) and d_full > 0
    rows: list[dict[str, float | int]] = []
    for k in range(1, k_max + 1):
        removed = _NAN
        kept = _NAN
        if damaging and damage_removed is not None:
            removed = float((d_full - damage_removed[k - 1]) / d_full)
        if damaging and damage_kept is not None:
            kept = float(damage_kept[k - 1] / d_full)
        rows.append(
            {
                "k": k,
                "patch_fraction": k / k_max,
                "volume_fraction": float(cum_v[k - 1] / v_roi) if v_roi else _NAN,
                "total_energy_fraction": float(cum_e[k - 1] / e_total)
                if e_total > 0
                else _NAN,
                "edge_energy_fraction": float(cum_e[k - 1] / e_edges)
                if e_edges > 0
                else _NAN,
                "damage_removed_fraction": removed,
                "damage_kept_fraction": kept,
            }
        )
    return rows


def k_alpha(fractions: np.ndarray, alpha: float) -> float:
    f = np.asarray(fractions, dtype=np.float64)
    hits = np.nonzero(f >= alpha)[0]
    return float(hits[0] + 1) if hits.size else _NAN


def top_fraction_ids(order: np.ndarray, ids: np.ndarray, fraction: float) -> set[int]:
    count = max(1, int(math.ceil(fraction * order.size)))
    return {int(v) for v in np.asarray(ids)[order[:count]]}


def _spearman(a: np.ndarray, b: np.ndarray) -> float:
    keep = np.isfinite(a) & np.isfinite(b)
    if keep.sum() < 3:
        return _NAN
    rho = spearmanr(a[keep], b[keep]).correlation
    return float(rho) if rho is not None and math.isfinite(rho) else _NAN


def _jaccard(a: set[int], b: set[int]) -> float:
    union = a | b
    return float(len(a & b) / len(union)) if union else _NAN


def _finite_pairs(a: np.ndarray, b: np.ndarray) -> int:
    return int((np.isfinite(a) & np.isfinite(b)).sum())


def _jaccard_top(
    a: np.ndarray, b: np.ndarray, ids: np.ndarray, top_fraction: float
) -> float:
    """Top-set overlap, NaN below 3 finite pairs (same guard as _spearman)."""
    if _finite_pairs(a, b) < 3:
        return _NAN
    return _jaccard(
        top_fraction_ids(rank_order(a), ids, top_fraction),
        top_fraction_ids(rank_order(b), ids, top_fraction),
    )


def rank_agreement(
    strength: np.ndarray,
    energy_fold: np.ndarray,
    utility: np.ndarray,
    *,
    top_fraction: float = 0.10,
) -> dict[str, float]:
    s = np.asarray(strength, dtype=np.float64)
    e = np.asarray(energy_fold, dtype=np.float64)
    u = np.asarray(utility, dtype=np.float64)
    ids = np.arange(s.size)
    if s.size < 3:
        keys = ("strength_energy", "energy_utility", "strength_utility")
        return {f"spearman_{k}": _NAN for k in keys} | {
            f"jaccard_{k}": _NAN for k in keys
        }
    return {
        "spearman_strength_energy": _spearman(s, e),
        "spearman_energy_utility": _spearman(e, u),
        "spearman_strength_utility": _spearman(s, u),
        "jaccard_strength_energy": _jaccard_top(s, e, ids, top_fraction),
        "jaccard_strength_utility": _jaccard_top(s, u, ids, top_fraction),
        "jaccard_energy_utility": _jaccard_top(e, u, ids, top_fraction),
    }


def screen_patches(
    ids: np.ndarray,
    *,
    ig_positive: np.ndarray,
    energy_fold: np.ndarray,
    uncertainty: np.ndarray,
    rng: np.random.Generator,
    all_if_at_most: int = 100,
    top_ig: int = 50,
    top_energy: int = 50,
    top_uncertainty: int = 20,
    random_controls: int = 20,
    use_energy: bool = True,
) -> tuple[np.ndarray, dict[int, str]]:
    """Plan §21.3 screening; returns sorted ids and the first-matching reason."""

    ids = np.asarray(ids)
    if ids.size <= all_if_at_most:
        return ids.copy(), {int(i): "all" for i in ids}
    reason: dict[int, str] = {}

    def take(values: np.ndarray, count: int, label: str) -> None:
        for i in ids[rank_order(values)[:count]]:
            reason.setdefault(int(i), label)

    take(np.asarray(ig_positive, dtype=np.float64), top_ig, "ig")
    if use_energy:
        take(np.asarray(energy_fold, dtype=np.float64), top_energy, "energy")
    take(np.asarray(uncertainty, dtype=np.float64), top_uncertainty, "uncertainty")
    remaining = np.array([i for i in ids if int(i) not in reason])
    if random_controls > 0 and remaining.size:
        for i in rng.choice(
            remaining, size=min(random_controls, remaining.size), replace=False
        ):
            reason[int(i)] = "random"
    selected = np.array(sorted(reason), dtype=ids.dtype)
    return selected, reason


def subcohort_case_ids(case_ids: list[str], n: int) -> list[str]:
    if n >= len(case_ids):
        return list(case_ids)
    positions = np.unique(np.round(np.linspace(0, len(case_ids) - 1, n)).astype(int))
    return [case_ids[int(p)] for p in positions]


__all__ = [
    "cumulative_curves",
    "k_alpha",
    "rank_agreement",
    "rank_order",
    "screen_patches",
    "subcohort_case_ids",
    "top_fraction_ids",
]
