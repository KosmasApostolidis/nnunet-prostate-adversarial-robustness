"""Pure statistics for paired, fold-structured architecture contrasts.

Used by ``experiments/analyze_resenc_revision_evidence.py``. Every case is
scored by exactly one of five independently trained fold models, so
resampling respects that structure: the stratified bootstrap resamples cases
within each fold (fold sizes fixed), and the hierarchical bootstrap resamples
whole folds first and then cases within each drawn fold. All intervals are
percentile intervals. NumPy/SciPy only.
"""

from __future__ import annotations

import numpy as np
from scipy import stats


def stratified_bootstrap_indices(
    strata: np.ndarray, n_resamples: int = 2000, seed: int = 42
) -> np.ndarray:
    """``(n_resamples, n)`` case indices, resampled with replacement within strata.

    Each resample keeps every stratum's size, so a fold of 280 cases always
    contributes 280 draws from its own 280 cases.
    """
    strata = np.asarray(strata)
    rng = np.random.default_rng(seed)
    out = np.empty((n_resamples, len(strata)), dtype=np.int64)
    col = 0
    for level in np.unique(strata):
        members = np.flatnonzero(strata == level)
        draws = rng.integers(0, len(members), size=(n_resamples, len(members)))
        out[:, col : col + len(members)] = members[draws]
        col += len(members)
    return out


def hierarchical_bootstrap_indices(
    strata: np.ndarray, n_resamples: int = 2000, seed: int = 42
) -> list[np.ndarray]:
    """Two-stage resamples: draw strata with replacement, then cases within each.

    Returns one index array per resample; lengths vary with the drawn strata.
    """
    strata = np.asarray(strata)
    rng = np.random.default_rng(seed)
    levels = np.unique(strata)
    members = [np.flatnonzero(strata == lv) for lv in levels]
    resamples: list[np.ndarray] = []
    for _ in range(n_resamples):
        drawn = rng.integers(0, len(levels), size=len(levels))
        parts = [
            members[k][rng.integers(0, len(members[k]), len(members[k]))] for k in drawn
        ]
        resamples.append(np.concatenate(parts))
    return resamples


def resampled_means(
    values: np.ndarray, indices: np.ndarray | list[np.ndarray]
) -> np.ndarray:
    """Mean of ``values`` under each resample (a 2-D index array or a list)."""
    values = np.asarray(values, dtype=float)
    if isinstance(indices, np.ndarray):
        return values[indices].mean(axis=1)
    return np.array([values[idx].mean() for idx in indices])


def percentile_ci(samples: np.ndarray, alpha: float = 0.05) -> tuple[float, float]:
    """Percentile interval of a bootstrap distribution (non-finite draws dropped)."""
    finite = samples[np.isfinite(samples)]
    if finite.size == 0:
        return float("nan"), float("nan")
    lo, hi = np.percentile(finite, [100 * alpha / 2, 100 * (1 - alpha / 2)])
    return float(lo), float(hi)


def ratio_of_means(numerator: np.ndarray, denominator: np.ndarray) -> float:
    """``mean(numerator) / mean(denominator)``; NaN when the denominator mean is 0."""
    den = float(np.mean(denominator))
    return float("nan") if den == 0 else float(np.mean(numerator)) / den


def additive_decomposition(
    base: np.ndarray, middle: np.ndarray, top: np.ndarray
) -> dict[str, float]:
    """Mean-based split of ``top - base`` into ``(middle - base) + (top - middle)``.

    Means are linear, so the two steps sum to the total exactly and the two
    shares sum to 1 (unlike ratios of Hodges-Lehmann shifts).
    """
    first = np.asarray(middle, float) - np.asarray(base, float)
    second = np.asarray(top, float) - np.asarray(middle, float)
    total = first + second
    return {
        "mean_first_step": float(first.mean()),
        "mean_second_step": float(second.mean()),
        "mean_total": float(total.mean()),
        "share_first_step": ratio_of_means(first, total),
        "share_second_step": ratio_of_means(second, total),
    }


def tost_paired(
    diff: np.ndarray, margin: float, alpha: float = 0.05
) -> dict[str, float]:
    """Paired two one-sided t-tests for equivalence within ``[-margin, +margin]``.

    ``p_tost`` is the larger of the two one-sided p-values; equivalence is
    declared when it is below ``alpha``, equivalently when the ``1 - 2 alpha``
    CI of the mean difference lies inside the margin.
    """
    d = np.asarray(diff, dtype=float)
    d = d[np.isfinite(d)]
    n = len(d)
    mean = float(d.mean())
    se = float(d.std(ddof=1) / np.sqrt(n))
    df = n - 1
    if se == 0:
        inside = abs(mean) < margin
        p = 0.0 if inside else 1.0
        return {
            "mean_diff": mean,
            "ci90_lo": mean,
            "ci90_hi": mean,
            "p_lower": p,
            "p_upper": p,
            "p_tost": p,
        }
    p_lower = float(stats.t.sf((mean + margin) / se, df))  # H0: mean <= -margin
    p_upper = float(stats.t.cdf((mean - margin) / se, df))  # H0: mean >= +margin
    t_crit = float(stats.t.isf(alpha, df))
    return {
        "mean_diff": mean,
        "ci90_lo": mean - t_crit * se,
        "ci90_hi": mean + t_crit * se,
        "p_lower": p_lower,
        "p_upper": p_upper,
        "p_tost": max(p_lower, p_upper),
    }
