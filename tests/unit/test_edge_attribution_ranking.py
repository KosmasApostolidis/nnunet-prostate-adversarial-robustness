"""Ranking, curve, screening, and subcohort phantoms (plan §21–§24, §26)."""

from __future__ import annotations

import math

import numpy as np
import pytest

from mri_prostate_seg.experiments.edge_attribution.ranking import (
    cumulative_curves,
    k_alpha,
    rank_agreement,
    rank_order,
    screen_patches,
    subcohort_case_ids,
    top_fraction_ids,
)


def test_rank_order_is_stable_descending_with_nans_last() -> None:
    order = rank_order(np.array([0.5, np.nan, 0.9, 0.5]))
    assert order.tolist() == [2, 0, 3, 1]


def test_cumulative_curves_shapes_and_monotone_volume() -> None:
    order = np.array([1, 0, 2])
    energy = np.array([1.0, 3.0, 6.0])
    voxels = np.array([10, 20, 30])
    removed = np.array([0.4, 0.2, 0.0])  # damage left after restoring top-k
    kept = np.array([0.3, 0.5, 0.6])
    rows = cumulative_curves(
        order,
        energy=energy,
        voxels=voxels,
        e_total=20.0,
        e_edges=10.0,
        v_roi=600,
        damage_removed=removed,
        damage_kept=kept,
        d_full=0.6,
    )
    assert [r["k"] for r in rows] == [1, 2, 3]
    assert rows[0]["total_energy_fraction"] == pytest.approx(3 / 20)
    assert rows[1]["edge_energy_fraction"] == pytest.approx(4 / 10)
    assert rows[2]["volume_fraction"] == pytest.approx(60 / 600)
    assert rows[0]["damage_removed_fraction"] == pytest.approx((0.6 - 0.4) / 0.6)
    assert rows[2]["damage_kept_fraction"] == pytest.approx(1.0)
    assert rows[1]["patch_fraction"] == pytest.approx(2 / 3)


def test_cumulative_curves_nan_when_non_damaging_or_missing() -> None:
    rows = cumulative_curves(
        np.array([0]),
        energy=np.array([1.0]),
        voxels=np.array([1]),
        e_total=1.0,
        e_edges=1.0,
        v_roi=10,
        damage_removed=None,
        damage_kept=np.array([0.1]),
        d_full=0.0,
    )
    assert math.isnan(rows[0]["damage_removed_fraction"])
    assert math.isnan(rows[0]["damage_kept_fraction"])


def test_k_alpha_and_top_fraction() -> None:
    fr = np.array([0.2, 0.55, 0.7, 0.9])
    assert k_alpha(fr, 0.5) == 2 and k_alpha(fr, 0.8) == 4
    assert math.isnan(k_alpha(fr, 0.95))
    assert top_fraction_ids(np.array([2, 0, 1]), np.array([10, 11, 12]), 0.1) == {12}
    assert top_fraction_ids(np.array([2, 0, 1]), np.array([10, 11, 12]), 0.5) == {
        12,
        10,
    }


def test_rank_agreement_perfect_and_inverse() -> None:
    s = np.arange(10, dtype=float)
    out = rank_agreement(s, s, s[::-1], top_fraction=0.2)
    assert out["spearman_strength_energy"] == pytest.approx(1.0)
    assert out["spearman_strength_utility"] == pytest.approx(-1.0)
    assert out["jaccard_strength_energy"] == pytest.approx(1.0)
    assert out["jaccard_strength_utility"] == pytest.approx(0.0)
    assert math.isnan(rank_agreement(s[:2], s[:2], s[:2])["spearman_strength_energy"])


def test_screen_returns_all_when_small_and_union_when_large() -> None:
    ids = np.arange(1, 51)
    sel, reason = screen_patches(
        ids,
        ig_positive=ids.astype(float),
        energy_fold=ids.astype(float),
        uncertainty=ids.astype(float),
        rng=np.random.default_rng(0),
    )
    assert sel.tolist() == ids.tolist() and set(reason.values()) == {"all"}
    ids = np.arange(1, 301)
    rng = np.random.default_rng(0)
    sel, reason = screen_patches(
        ids,
        ig_positive=ids.astype(float),
        energy_fold=ids[::-1].astype(float),
        uncertainty=np.zeros(300),
        rng=rng,
        top_ig=5,
        top_energy=5,
        top_uncertainty=2,
        random_controls=3,
    )
    assert 296 in sel and 5 in sel  # top by ig and top by energy
    assert reason[300] == "ig" and reason[1] == "energy"
    assert list(reason.values()).count("random") == 3
    assert sel.tolist() == sorted(sel.tolist())
    no_energy, reason2 = screen_patches(
        ids,
        ig_positive=ids.astype(float),
        energy_fold=ids[::-1].astype(float),
        uncertainty=np.zeros(300),
        rng=np.random.default_rng(0),
        top_ig=5,
        top_energy=5,
        top_uncertainty=0,
        random_controls=0,
        use_energy=False,
    )
    assert "energy" not in reason2.values() and len(no_energy) == 5


def test_rank_agreement_jaccard_is_nan_below_three_finite_pairs() -> None:
    s = np.arange(10, dtype=float)
    sparse = np.full(10, np.nan)
    sparse[:2] = [1.0, 2.0]  # only 2 finite entries
    out = rank_agreement(s, sparse, s, top_fraction=0.2)
    assert math.isnan(out["spearman_strength_energy"])
    assert math.isnan(out["jaccard_strength_energy"])
    assert math.isnan(out["jaccard_energy_utility"])
    assert out["jaccard_strength_utility"] == pytest.approx(1.0)
    assert out["spearman_strength_utility"] == pytest.approx(1.0)


def test_subcohort_is_deterministic_evenly_spaced_and_nested() -> None:
    cases = [f"c{i:03d}" for i in range(1000)]
    causal = subcohort_case_ids(cases, 100)
    interaction = subcohort_case_ids(causal, 30)
    assert len(causal) == 100 and causal[0] == "c000" and causal[-1] == "c999"
    assert causal == subcohort_case_ids(cases, 100)
    assert set(interaction) <= set(causal) and len(interaction) == 30
    assert subcohort_case_ids(cases[:5], 100) == cases[:5]
