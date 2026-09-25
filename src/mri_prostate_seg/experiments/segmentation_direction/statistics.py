"""Cohort statistics with case as the unit (spec §27-§29).

Percentile bands across cases describe heterogeneity; the bootstrap interval
is the only thing here that may be called a confidence interval (spec §28).
"""

from __future__ import annotations

import numpy as np

_NAN = float("nan")


def _finite(values: np.ndarray) -> np.ndarray:
    array = np.asarray(values, dtype=float).ravel()
    return array[np.isfinite(array)]


def describe(values: np.ndarray) -> dict[str, float]:
    array = _finite(values)
    if array.size == 0:
        return {
            k: _NAN for k in ("n", "median", "q25", "q75", "p05", "p95", "mean")
        } | {"n": 0.0}
    q = np.quantile(array, [0.05, 0.25, 0.5, 0.75, 0.95])
    return {
        "n": float(array.size),
        "median": float(q[2]),
        "q25": float(q[1]),
        "q75": float(q[3]),
        "p05": float(q[0]),
        "p95": float(q[4]),
        "mean": float(array.mean()),
    }


def bootstrap_median_ci(
    values: np.ndarray, *, resamples: int = 10_000, seed: int = 42
) -> tuple[float, float]:
    array = _finite(values)
    if array.size < 2:
        return (_NAN, _NAN)
    rng = np.random.default_rng(seed)
    draws = rng.integers(0, array.size, size=(resamples, array.size))
    medians = np.median(array[draws], axis=1)
    return (float(np.quantile(medians, 0.025)), float(np.quantile(medians, 0.975)))


def paired_median_difference(
    a: np.ndarray, b: np.ndarray, *, resamples: int = 10_000, seed: int = 42
) -> dict[str, float]:
    x = np.asarray(a, dtype=float).ravel()
    y = np.asarray(b, dtype=float).ravel()
    if x.shape != y.shape:
        raise ValueError("paired arrays must align")
    keep = np.isfinite(x) & np.isfinite(y)
    diff = x[keep] - y[keep]
    low, high = bootstrap_median_ci(diff, resamples=resamples, seed=seed)
    return {
        "n": float(diff.size),
        "median_diff": float(np.median(diff)) if diff.size else _NAN,
        "ci_low": low,
        "ci_high": high,
    }


__all__ = ["bootstrap_median_ci", "describe", "paired_median_difference"]
