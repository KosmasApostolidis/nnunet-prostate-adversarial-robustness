"""Energy accounting phantoms (plan §16, §25, §37)."""

from __future__ import annotations

import numpy as np
import pytest

from mri_prostate_seg.experiments.edge_attribution.energy import (
    concentration_indices,
    conservation_error,
    energy_map,
    gini,
    patch_energy_table,
)

SHAPE = (6, 32, 32)


def _tubes() -> np.ndarray:
    tubes = np.zeros(SHAPE, dtype=np.int32)
    tubes[2, 8:12, 8:12] = 1
    tubes[2, 8:12, 20:24] = 2
    tubes[3, 8:16, 8:12] = 3
    return tubes


def test_energy_map_handles_single_and_multi_channel() -> None:
    d = np.ones(SHAPE)
    assert energy_map(d).sum() == pytest.approx(d.size)
    assert energy_map(np.stack([d, 2 * d])).sum() == pytest.approx(5 * d.size)


def test_uniform_energy_gives_unit_fold_enrichment() -> None:
    energy = np.ones(SHAPE)
    roi = np.ones(SHAPE, dtype=bool)
    rows = patch_energy_table(energy, _tubes(), roi=roi, fov=roi)
    assert [r["patch_id"] for r in rows] == [1, 2, 3]
    for r in rows:
        assert r["fold_enrichment"] == pytest.approx(1.0)
        assert r["percentage_point_enrichment"] == pytest.approx(0.0)
    assert rows[2]["voxel_count"] == 32


def test_single_energised_patch_takes_all_edge_energy() -> None:
    energy = np.zeros(SHAPE)
    energy[_tubes() == 2] = 5.0
    roi = np.ones(SHAPE, dtype=bool)
    rows = {
        r["patch_id"]: r for r in patch_energy_table(energy, _tubes(), roi=roi, fov=roi)
    }
    assert rows[2]["edge_energy_share"] == pytest.approx(1.0)
    assert rows[2]["energy_share_roi"] == pytest.approx(1.0)
    assert rows[1]["energy"] == 0.0 and rows[3]["edge_energy_share"] == 0.0


def test_roi_and_fov_denominators_differ() -> None:
    energy = np.ones(SHAPE)
    fov = np.ones(SHAPE, dtype=bool)
    roi = np.zeros(SHAPE, dtype=bool)
    roi[2:4, 4:20, 4:28] = True
    rows = patch_energy_table(energy, _tubes(), roi=roi, fov=fov)
    r = rows[0]
    assert r["energy_share_fov"] < r["energy_share_roi"]
    assert r["energy_share_roi"] == pytest.approx(16 / roi.sum())


def test_conservation_holds_exactly_with_a_valid_mask() -> None:
    rng = np.random.default_rng(3)
    energy = rng.random(SHAPE)
    domain = rng.random(SHAPE) > 0.3
    assert conservation_error(energy, _tubes(), domain) < 1e-12


def test_energy_outside_domain_contributes_nothing() -> None:
    energy = np.zeros(SHAPE)
    energy[0] = 100.0
    domain = np.ones(SHAPE, dtype=bool)
    domain[0] = False
    rows = patch_energy_table(energy, _tubes(), roi=domain, fov=domain)
    assert all(r["energy_share_roi"] == 0.0 for r in rows)


def test_concentration_indices_extremes() -> None:
    flat = concentration_indices(np.full(4, 0.25))
    assert flat["n_eff"] == pytest.approx(4.0) and flat["n_eff_norm"] == pytest.approx(
        1.0
    )
    assert flat["entropy_concentration"] == pytest.approx(0.0, abs=1e-12)
    assert flat["gini"] == pytest.approx(0.0, abs=1e-12)
    peaked = concentration_indices(np.array([1.0, 0.0, 0.0, 0.0]))
    assert peaked["n_eff"] == pytest.approx(1.0)
    assert peaked["entropy_concentration"] == pytest.approx(1.0)
    assert peaked["gini"] == pytest.approx(0.75)
    assert np.isnan(concentration_indices(np.zeros(3))["n_eff"])
    assert gini(np.array([2.0, 2.0])) == pytest.approx(0.0)
