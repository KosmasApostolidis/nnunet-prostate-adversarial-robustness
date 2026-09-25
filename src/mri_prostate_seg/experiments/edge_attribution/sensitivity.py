"""Phase B: how much the primary settings move the answer.

The spec fixes the primary settings for STABILITY, never for the size of the
effect, so nothing here ranks a setting by damage, enrichment or recovery.
"""

from __future__ import annotations

import itertools
import math
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field

import numpy as np
import pandas as pd


@dataclass(frozen=True)
class SweepPoint:
    """One sweep setting's results, keyed by case id."""

    name: str
    parameter: str
    value: float
    k50: Mapping[str, float] = field(default_factory=dict)
    top_sets: Mapping[str, set[int]] = field(default_factory=dict)


def top_set_jaccard(left: set[int], right: set[int]) -> float:
    """Jaccard of two top-fraction patch-id sets; NaN when both are empty."""

    union = left | right
    if not union:
        return math.nan
    return len(left & right) / len(union)


def k_stability(per_setting: Mapping[str, Sequence[float]]) -> dict[str, float]:
    """Spread of K50 across settings, pooled over cases."""

    medians = {
        name: float(np.median([v for v in values if np.isfinite(v)]))
        for name, values in per_setting.items()
        if any(np.isfinite(v) for v in values)
    }
    if not medians:
        return {"median_k50": math.nan, "max_abs_deviation": math.nan}
    centre = float(np.median(list(medians.values())))
    return {
        "median_k50": centre,
        "max_abs_deviation": float(max(abs(v - centre) for v in medians.values())),
    }


def stability_table(points: Sequence[SweepPoint]) -> pd.DataFrame:
    """Pairwise agreement between settings of the same parameter."""

    out: list[dict[str, object]] = []
    by_parameter: dict[str, list[SweepPoint]] = {}
    for point in points:
        by_parameter.setdefault(point.parameter, []).append(point)
    for parameter, group in by_parameter.items():
        for left, right in itertools.combinations(group, 2):
            shared = sorted(set(left.top_sets) & set(right.top_sets))
            jaccards = [
                top_set_jaccard(left.top_sets[c], right.top_sets[c]) for c in shared
            ]
            finite_j = [j for j in jaccards if np.isfinite(j)]
            diffs = [
                abs(left.k50[c] - right.k50[c])
                for c in sorted(set(left.k50) & set(right.k50))
                if np.isfinite(left.k50[c]) and np.isfinite(right.k50[c])
            ]
            out.append(
                {
                    "parameter": parameter,
                    "setting_a": left.name,
                    "setting_b": right.name,
                    "n_cases": len(shared),
                    "jaccard_median": float(np.median(finite_j))
                    if finite_j
                    else math.nan,
                    "k50_abs_difference_median": float(np.median(diffs))
                    if diffs
                    else math.nan,
                }
            )
    return pd.DataFrame(out)


def choose_primary(
    table: pd.DataFrame, parameter: str, *, min_jaccard: float = 0.6
) -> str | None:
    """The setting that agrees most with its neighbours, or None if none does.

    Returning None is a real outcome, not a failure: it means the result depends
    on the setting and the report must say so instead of quietly picking one.
    """

    rows = table[table["parameter"] == parameter]
    rows = rows[rows["jaccard_median"] >= min_jaccard]
    if rows.empty:
        return None
    scores: dict[str, list[float]] = {}
    for _, row in rows.iterrows():
        for side in ("setting_a", "setting_b"):
            scores.setdefault(str(row[side]), []).append(float(row["jaccard_median"]))
    return max(scores, key=lambda name: (np.mean(scores[name]), -len(name)))


__all__ = [
    "SweepPoint",
    "choose_primary",
    "k_stability",
    "stability_table",
    "top_set_jaccard",
]
