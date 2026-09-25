"""Shell rankings, K->cycles/mm, a-priori band controls (SSEUA spec)."""

from __future__ import annotations

import numpy as np
import pytest

from mri_prostate_seg.experiments.edge_attribution.spectral_analysis import (
    RANKINGS,
    band_indices,
    count_matched_draws,
    cumulative_masks,
    energy_matched_draws,
    k_to_cpm,
    permutation_p,
    ranking_orders,
)


def test_ranking_orders_cover_all_shells_and_exclude_empty_ones() -> None:
    n = 6
    nec = np.array([0.1, 0.5, 0.2, np.nan, 0.4, 0.3])
    energy = np.array([6, 5, 4, 3, 2, 1.0])
    ig = np.array([0, 0, 1, 0, 2, 3.0])
    empty = np.array([False, False, False, True, False, False])
    orders = ranking_orders(nec, energy, ig, empty)
    assert set(orders) == set(RANKINGS)
    assert orders["frequency_low_first"].tolist() == [0, 1, 2, 4, 5]
    assert orders["frequency_high_first"].tolist() == [5, 4, 2, 1, 0]
    assert orders["utility"].tolist() == [1, 4, 5, 2, 0]
    assert orders["energy"].tolist() == [0, 1, 2, 4, 5]
    assert orders["ig"].tolist() == [5, 4, 2, 0, 1]
    for o in orders.values():
        assert 3 not in o and len(o) == n - 1


def test_cumulative_masks_are_running_unions() -> None:
    masks = np.zeros((3, 2, 2, 2), dtype=bool)
    masks[0, 0], masks[1, 1, 0], masks[2, 1, 1] = True, True, True
    cum = cumulative_masks(masks, np.array([2, 0, 1]))
    assert cum.shape == (3, 2, 2, 2)
    assert cum[0].sum() == masks[2].sum()
    assert cum[1].sum() == masks[2].sum() + masks[0].sum()
    assert cum[2].all()


def test_k_to_cpm_is_the_highest_edge_among_the_first_k() -> None:
    edges = np.array([0.0, 0.1, 0.2, 0.3, np.inf])
    assert k_to_cpm(np.array([0, 1, 2, 3]), 2, edges) == pytest.approx(0.2)
    assert k_to_cpm(np.array([2, 0, 1, 3]), 1, edges) == pytest.approx(0.3)
    assert np.isinf(k_to_cpm(np.array([3, 0]), 1, edges))
    assert np.isnan(k_to_cpm(np.array([0, 1]), float("nan"), edges))
    assert k_to_cpm(np.array([0, 1]), 2.0, edges) == pytest.approx(0.2)


def test_band_indices() -> None:
    assert band_indices("LF", 24) == list(range(0, 8))
    assert band_indices("MF", 24) == list(range(8, 16))
    assert band_indices("HF", 24) == list(range(16, 24))


def test_count_matched_draws_are_distinct_from_the_complement() -> None:
    rng = np.random.default_rng(0)
    band = band_indices("LF", 24)
    draws = count_matched_draws(band, 24, 20, rng)
    assert len(draws) == 20
    assert len({frozenset(d) for d in draws}) == 20
    for d in draws:
        assert len(d) == 8 and not set(d) & set(band)


def test_energy_matched_draws_respect_tolerance_and_report_undrawable() -> None:
    rng = np.random.default_rng(0)
    energies = np.ones(24)
    energies[:8] = 3.0  # LF holds 24 of 40 units; complement holds 16 -> undrawable
    draws, tries = energy_matched_draws(band_indices("LF", 24), energies, 10, rng)
    assert draws == [] and tries == 2000
    energies[:8] = 1.0  # now LF energy 8, complement 16: drawable
    draws, tries = energy_matched_draws(band_indices("LF", 24), energies, 10, rng)
    assert len(draws) == 10
    for d in draws:
        assert abs(energies[d].sum() - 8.0) <= 0.8 + 1e-9
        assert not set(d) & set(range(8))


def test_permutation_p_floor_and_direction() -> None:
    assert permutation_p([0.5] * 20, [0.1] * 20) == pytest.approx(1 / 21)
    assert permutation_p([0.1] * 20, [0.5] * 20) == pytest.approx(1.0)
    assert np.isnan(permutation_p([], []))
