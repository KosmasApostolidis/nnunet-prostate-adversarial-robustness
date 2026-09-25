from __future__ import annotations

import numpy as np
import pytest

from mri_prostate_seg.experiments.segmentation_direction.statistics import (
    bootstrap_median_ci,
    describe,
    paired_median_difference,
)


def test_describe_drops_nan_and_reports_quantiles() -> None:
    d = describe(np.asarray([1.0, 2.0, 3.0, 4.0, np.nan]))
    assert d["n"] == 4
    assert d["median"] == pytest.approx(2.5)
    assert d["q25"] == pytest.approx(1.75) and d["q75"] == pytest.approx(3.25)
    assert d["p05"] < d["median"] < d["p95"]
    assert np.isnan(describe(np.asarray([np.nan]))["median"])


def test_bootstrap_median_ci_brackets_the_median_and_is_deterministic() -> None:
    rng = np.random.default_rng(0)
    values = rng.normal(loc=3.0, scale=1.0, size=200)
    low, high = bootstrap_median_ci(values, resamples=2000)
    assert low < np.median(values) < high
    assert high - low < 0.6
    assert bootstrap_median_ci(values, resamples=2000) == (low, high)
    assert all(np.isnan(v) for v in bootstrap_median_ci(np.asarray([1.0])))


def test_paired_difference_is_a_minus_b_and_drops_incomplete_pairs() -> None:
    a = np.asarray([2.0, 3.0, 4.0, np.nan])
    b = np.asarray([1.0, 1.0, 1.0, 1.0])
    out = paired_median_difference(a, b, resamples=500)
    assert out["n"] == 3
    assert out["median_diff"] == pytest.approx(2.0)
    assert out["ci_low"] <= 2.0 <= out["ci_high"]
