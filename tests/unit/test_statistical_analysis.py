"""Unit tests for pure statistics helpers in ``experiments.statistical_analysis``.

CPU-only deterministic helpers: epsilon selection, uniqueness assertions,
Benjamini-Hochberg FDR, bootstrap CIs, paired Cohen's d, and Wilcoxon.

Relocated from the former ``tests/test_helpers.py`` when its
``experiments.fgsm_adversarial_evaluation`` half was removed in the
experiment-script reorganisation.
"""

from __future__ import annotations

import math

import numpy as np
import pandas as pd
import pytest

import experiments.statistical_analysis as sa


def test_mask_eps_matches_close_floats():
    # rtol=1e-9: 0.02 + 1e-12 well within tolerance; 0.020001 not.
    series = pd.Series([0.02, 0.02 + 1e-12, 0.04, 0.0, 0.020001])
    mask = sa._mask_eps(series, 0.02)
    assert mask.tolist() == [True, True, False, False, False]


def test_epsilons_for_plots_uses_cli_when_provided():
    df = pd.DataFrame({"epsilon": [0.0, 0.02, 0.04]})
    out = sa._epsilons_for_plots(df, [0.05, 0.01])
    assert out == [0.01, 0.05]  # sorted, CLI takes precedence


def test_epsilons_for_plots_falls_back_to_data_excluding_zero():
    df = pd.DataFrame({"epsilon": [0.0, 0.02, 0.04, 0.0]})
    out = sa._epsilons_for_plots(df, None)
    assert out == [0.02, 0.04]


def test_assert_unique_case_class_eps_passes_on_unique():
    df = pd.DataFrame(
        {
            "case_id": ["a", "a", "b"],
            "epsilon": [0.0, 0.02, 0.0],
            "class": [1, 1, 1],
        }
    )
    sa._assert_unique_case_class_eps(df, "wg")  # no raise


def test_assert_unique_case_class_eps_raises_on_dup():
    df = pd.DataFrame(
        {
            "case_id": ["a", "a"],
            "epsilon": [0.0, 0.0],
            "class": [1, 1],
        }
    )
    with pytest.raises(ValueError):
        sa._assert_unique_case_class_eps(df, "wg")


def test_fdr_bh_monotone_and_clipped():
    pvals = np.array([0.01, 0.02, 0.03, 0.04, 0.5])
    adj = sa._fdr_bh(pvals)
    assert (adj <= 1.0).all()
    assert (adj >= 0.0).all()
    # BH is monotone non-decreasing in sorted order
    sorted_adj = adj[np.argsort(pvals)]
    assert all(
        sorted_adj[i] <= sorted_adj[i + 1] + 1e-12 for i in range(len(sorted_adj) - 1)
    )


def test_fdr_bh_handles_nans():
    pvals = np.array([0.01, np.nan, 0.5])
    adj = sa._fdr_bh(pvals)
    assert math.isnan(adj[1])
    assert np.isfinite(adj[0]) and np.isfinite(adj[2])


def test_fdr_bh_all_nan_returns_all_nan():
    pvals = np.array([np.nan, np.nan])
    adj = sa._fdr_bh(pvals)
    assert np.isnan(adj).all()


def test_bootstrap_ci_returns_finite_interval():
    rng = np.random.RandomState(0)
    data = rng.normal(loc=10.0, scale=2.0, size=200)
    lo, hi = sa._bootstrap_ci(data, n_boot=500, ci=0.95, seed=0)
    assert math.isfinite(lo) and math.isfinite(hi)
    assert lo < 10.0 < hi
    assert lo < hi


def test_bootstrap_ci_too_few_samples_returns_nan():
    lo, hi = sa._bootstrap_ci(np.array([1.0]), n_boot=10)
    assert math.isnan(lo) and math.isnan(hi)


def test_cohens_d_paired_zero_difference_returns_zero():
    a = np.array([1.0, 2.0, 3.0])
    b = a.copy()
    assert sa._cohens_d_paired(a, b) == 0.0


def test_cohens_d_paired_positive_when_a_greater():
    # Differences must vary so SD > 0 — constant-diff returns NaN by design
    a = np.array([2.0, 3.5, 4.0, 5.5, 6.0])
    b = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
    d = sa._cohens_d_paired(a, b)
    assert math.isfinite(d) and d > 0


def test_cohens_d_paired_too_few_returns_nan():
    assert math.isnan(sa._cohens_d_paired(np.array([1.0]), np.array([2.0])))


def test_wilcoxon_paired_returns_pvalue():
    a = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
    b = np.array([1.1, 2.1, 3.1, 4.1, 5.1])
    res = sa._wilcoxon_paired(a, b)
    assert hasattr(res, "pvalue")
    assert 0.0 <= float(res.pvalue) <= 1.0
