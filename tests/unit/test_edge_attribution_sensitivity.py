"""Phase B stability metrics."""

from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd
import pytest

REPO_ROOT = Path(__file__).resolve().parents[2]
for path in (REPO_ROOT, REPO_ROOT / "src"):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

from mri_prostate_seg.experiments.edge_attribution.sensitivity import (  # noqa: E402
    SweepPoint,
    choose_primary,
    k_stability,
    stability_table,
    top_set_jaccard,
)


def test_top_set_jaccard_is_one_for_identical_sets() -> None:
    assert top_set_jaccard({1, 2, 3}, {1, 2, 3}) == pytest.approx(1.0)


def test_top_set_jaccard_is_zero_for_disjoint_sets() -> None:
    assert top_set_jaccard({1, 2}, {3, 4}) == pytest.approx(0.0)


def test_top_set_jaccard_handles_partial_overlap() -> None:
    assert top_set_jaccard({1, 2, 3}, {2, 3, 4}) == pytest.approx(0.5)


def test_top_set_jaccard_of_two_empty_sets_is_nan_not_one() -> None:
    # Two settings that both selected nothing agree vacuously; scoring that 1.0
    # would let a degenerate setting win the stability ranking.
    import math

    assert math.isnan(top_set_jaccard(set(), set()))


def test_k_stability_is_the_spread_of_k50_across_settings() -> None:
    per_setting = {"a": [10.0, 12.0], "b": [11.0, 13.0], "c": [50.0, 60.0]}
    out = k_stability(per_setting)
    # 'c' moves K50 far from the others, so the spread is large
    assert out["max_abs_deviation"] > 30.0
    assert out["median_k50"] == pytest.approx(
        12.0
    )  # median of the per-setting medians 11, 12, 55


def test_stability_table_reports_one_row_per_setting_pair() -> None:
    points = [
        SweepPoint(
            name="q0.90",
            parameter="edge_quantile",
            value=0.90,
            k50={"case1": 10.0},
            top_sets={"case1": {1, 2, 3}},
        ),
        SweepPoint(
            name="q0.95",
            parameter="edge_quantile",
            value=0.95,
            k50={"case1": 11.0},
            top_sets={"case1": {2, 3, 4}},
        ),
    ]
    out = stability_table(points)
    assert len(out) == 1
    row = out.iloc[0]
    assert row["parameter"] == "edge_quantile"
    assert row["jaccard_median"] == pytest.approx(0.5)
    assert row["k50_abs_difference_median"] == pytest.approx(1.0)


def test_choose_primary_picks_the_most_stable_not_the_largest_effect() -> None:
    # 'b' has the highest Jaccard agreement with its neighbours; 'c' would win on
    # any effect-size criterion, which is exactly what must NOT decide this.
    table = pd.DataFrame(
        {
            "parameter": ["patch_extent_mm"] * 3,
            "setting_a": ["a", "b", "c"],
            "setting_b": ["b", "c", "a"],
            "jaccard_median": [0.9, 0.4, 0.4],
            "k50_abs_difference_median": [1.0, 9.0, 9.0],
        }
    )
    assert choose_primary(table, "patch_extent_mm") in {"a", "b"}


def test_choose_primary_returns_none_when_no_pair_is_stable() -> None:
    table = pd.DataFrame(
        {
            "parameter": ["ig_steps"] * 1,
            "setting_a": ["16"],
            "setting_b": ["32"],
            "jaccard_median": [0.05],
            "k50_abs_difference_median": [80.0],
        }
    )
    assert choose_primary(table, "ig_steps", min_jaccard=0.6) is None
