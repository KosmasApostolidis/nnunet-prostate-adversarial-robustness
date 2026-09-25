"""Radial-shell partition of the perturbation spectrum (SSEUA spec, Partition)."""

from __future__ import annotations

import numpy as np
import pytest

from mri_prostate_seg.experiments.edge_attribution.spectral import (
    BAND_OF_SHELL,
    N_SHELLS,
    REFERENCE_SHAPE,
    REFERENCE_SPACING,
    assert_hermitian_symmetric,
    equal_count_edges,
    hermitian_weights,
    parseval_error,
    project,
    reference_shell_edges,
    shell_attribution,
    shell_energy,
    shell_masks,
    split_delta,
)

SMALL = (15, 8, 9)
SPACING = (3.0, 0.5, 0.5)


def test_reference_edges_reproduce_bandats_terciles() -> None:
    edges = reference_shell_edges(N_SHELLS)
    assert edges.shape == (N_SHELLS + 1,)
    assert edges[0] == 0.0 and np.isinf(edges[-1])
    assert np.all(np.diff(edges[:-1]) > 0)
    assert abs(edges[8] - 0.6585) < 0.005
    assert abs(edges[16] - 0.9263) < 0.005
    assert REFERENCE_SHAPE == (24, 256, 256) and REFERENCE_SPACING == SPACING


def test_bands_are_shell_terciles() -> None:
    assert [BAND_OF_SHELL(k) for k in (1, 8, 9, 16, 17, 24)] == [
        "LF", "LF", "MF", "MF", "HF", "HF",
    ]


def test_hermitian_weights_sum_to_width() -> None:
    assert hermitian_weights(9).sum() == 9 and hermitian_weights(8).sum() == 8
    assert hermitian_weights(8)[-1] == 1.0 and hermitian_weights(9)[-1] == 2.0


def test_masks_are_disjoint_cover_non_dc_and_hermitian_symmetric() -> None:
    edges = reference_shell_edges(3)
    part = shell_masks(SMALL, SPACING, edges)
    assert part.masks.shape == (3, 15, 8, 5)
    assert part.masks.sum(axis=0).max() == 1
    covered = part.masks.any(axis=0)
    assert not covered[0, 0, 0]  # DC excluded
    assert covered.sum() == 15 * 8 * 5 - 1
    for mask in part.masks:
        assert_hermitian_symmetric(mask)
    assert abs(part.counts.sum() - (15 * 8 * 9 - 1)) < 1e-9


def test_bandat_port_regression_counts_on_its_grid() -> None:
    # Edges taken from THIS grid (equal-count terciles) reproduce bandat's
    # verified weighted counts 360 / 364 / 355 on 15x8x9 at (3.0, 0.5, 0.5).
    edges = equal_count_edges(SMALL, SPACING, 3)
    part = shell_masks(SMALL, SPACING, edges)
    assert [int(round(c)) for c in part.counts] == [360, 364, 355]


def test_small_crop_marks_empty_top_shell() -> None:
    edges = reference_shell_edges(N_SHELLS)
    part = shell_masks((4, 6, 6), (3.0, 0.5, 0.5), edges)
    assert part.empty.any()
    assert part.counts[part.empty].sum() == 0


def test_parseval_and_split_delta() -> None:
    rng = np.random.default_rng(0)
    delta = rng.normal(size=SMALL)
    valid = np.ones(SMALL, dtype=bool)
    valid[:, :2, :] = False
    part = shell_masks(SMALL, SPACING, reference_shell_edges(N_SHELLS))
    sd = split_delta(delta, valid)
    assert np.allclose(sd.delta_out, delta * ~valid)
    assert np.isclose(sd.c0, (delta * valid).mean())
    assert sd.F[0, 0, 0] == 0
    energies = shell_energy(sd.F, part)
    assert energies.shape == (N_SHELLS,)
    total = float(((delta * valid) ** 2).sum())
    assert np.isclose(energies.sum() + sd.energy_dc, total, rtol=1e-9)
    assert parseval_error(sd.F, part, delta * valid) < 1e-9
    assert np.isclose(sd.energy_out, float(((delta * ~valid) ** 2).sum()))


def test_projection_of_all_shells_recovers_delta_minus_dc() -> None:
    rng = np.random.default_rng(1)
    delta = rng.normal(size=SMALL)
    valid = np.ones(SMALL, dtype=bool)
    part = shell_masks(SMALL, SPACING, reference_shell_edges(N_SHELLS))
    sd = split_delta(delta, valid)
    recon = project(sd.F, part.masks.any(axis=0), SMALL)
    assert np.allclose(recon, delta - delta.mean(), atol=1e-10)


def test_shell_attribution_equals_spatial_inner_product() -> None:
    rng = np.random.default_rng(2)
    delta = rng.normal(size=SMALL)
    grad = rng.normal(size=SMALL)
    valid = np.ones(SMALL, dtype=bool)
    part = shell_masks(SMALL, SPACING, reference_shell_edges(N_SHELLS))
    sd = split_delta(delta, valid)
    attr = shell_attribution(sd.F, grad, part)
    for k in range(N_SHELLS):
        spatial = float((project(sd.F, part.masks[k], SMALL) * grad).sum())
        assert np.isclose(attr[k], spatial, atol=1e-6)
    assert np.isclose(
        attr.sum(), float(((delta - delta.mean()) * grad).sum()), atol=1e-6
    )


def test_hermitian_assertion_rejects_asymmetric_mask() -> None:
    mask = np.zeros((4, 4, 3), dtype=bool)
    mask[1, 0, 0] = True  # its conjugate (-1 -> index 3, 0, 0) is not set
    with pytest.raises(AssertionError):
        assert_hermitian_symmetric(mask)
