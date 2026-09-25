"""Signed-distance damage profile (spec §7)."""

from __future__ import annotations

import numpy as np
import pytest

from mri_prostate_seg.experiments.perturbation_structure import (
    SIGNED_BAND_NAMES,
    signed_distance_mm,
)
from mri_prostate_seg.experiments.segmentation_direction.profile import (
    SIGNED_BAND_EDGES_MM,
    directional_distance_profile,
)
from mri_prostate_seg.experiments.segmentation_direction.transitions import (
    binary_transition_masks,
)

SPACING = (3.0, 0.5, 0.5)
SHAPE = (12, 64, 64)


def _ball(radius_mm: float) -> np.ndarray:
    grids = np.meshgrid(
        *[np.arange(n) * s for n, s in zip(SHAPE, SPACING)], indexing="ij"
    )
    r2 = sum((g - c) ** 2 for g, c in zip(grids, (18.0, 16.0, 16.0)))
    return r2 <= radius_mm**2


def test_edges_cover_the_line_in_band_order() -> None:
    assert tuple(SIGNED_BAND_EDGES_MM) == SIGNED_BAND_NAMES
    edges = list(SIGNED_BAND_EDGES_MM.values())
    assert edges[0][0] == -np.inf and edges[-1][1] == np.inf
    for (_, right), (left, _) in zip(edges[:-1], edges[1:]):
        assert right == left


def test_expansion_by_1mm_is_an_fp_rate_in_the_0_2mm_band_only() -> None:
    gt = _ball(8.0)
    adv = _ball(9.0)
    valid = np.ones(SHAPE, dtype=bool)
    tr = binary_transition_masks(gt, gt, adv, valid=valid)
    rows = directional_distance_profile(tr, gt, gt, valid=valid, spacing=SPACING)
    by_band = {row["band"]: row for row in rows}
    assert [row["band"] for row in rows] == list(SIGNED_BAND_NAMES)
    assert by_band["0_2mm"]["induced_fp_rate"] > 0.3
    # The 3 mm slices add one cap voxel per pole at r = 9 mm; those sit 3 mm
    # from the nearest GT voxel and land in 2_5mm, so only the outer bands
    # are asserted empty.
    assert by_band["2_5mm"]["induced_fp_rate"] < 0.01
    for band in ("5_10mm", "beyond_10mm"):
        assert by_band[band]["induced_fp_rate"] == 0.0
    for band in SIGNED_BAND_NAMES:
        assert by_band[band]["induced_fn_rate"] in (0.0,) or np.isnan(
            by_band[band]["induced_fn_rate"]
        )
    assert by_band["0_2mm"]["net_fg_change_per_100_voxels"] > 0


def test_rates_use_at_risk_denominators() -> None:
    gt = _ball(8.0)
    clean = _ball(9.0)  # clean already over-segments the 0-2 mm shell
    adv = _ball(9.0)  # attack changes nothing
    valid = np.ones(SHAPE, dtype=bool)
    tr = binary_transition_masks(gt, clean, adv, valid=valid)
    rows = directional_distance_profile(tr, gt, clean, valid=valid, spacing=SPACING)
    shell = next(r for r in rows if r["band"] == "0_2mm")
    # At-risk FP voxels are (G=0, C=0); the over-segmented ones are excluded.
    assert shell["at_risk_fp_voxels"] < shell["n_voxels"]
    assert shell["induced_fp_rate"] == 0.0


def test_invalid_voxels_are_outside_every_band() -> None:
    gt = _ball(8.0)
    valid = np.ones(SHAPE, dtype=bool)
    valid[:, :8, :] = False
    tr = binary_transition_masks(gt, gt, gt, valid=valid)
    rows = directional_distance_profile(tr, gt, gt, valid=valid, spacing=SPACING)
    assert sum(int(r["n_voxels"]) for r in rows) == int(valid.sum())


def test_precomputed_distance_matches() -> None:
    gt = _ball(8.0)
    adv = _ball(10.0)
    valid = np.ones(SHAPE, dtype=bool)
    tr = binary_transition_masks(gt, gt, adv, valid=valid)
    phi = signed_distance_mm(gt, SPACING)
    a = directional_distance_profile(tr, gt, gt, valid=valid, spacing=SPACING)
    b = directional_distance_profile(
        tr, gt, gt, valid=valid, spacing=SPACING, distance=phi
    )
    np.testing.assert_equal(a, b)  # NaN-aware: empty bands carry NaN rates


def test_empty_ground_truth_raises() -> None:
    empty = np.zeros(SHAPE, dtype=bool)
    tr = binary_transition_masks(empty, empty, empty, valid=np.ones(SHAPE, bool))
    with pytest.raises(ValueError, match="empty"):
        directional_distance_profile(
            tr, empty, empty, valid=np.ones(SHAPE, bool), spacing=SPACING
        )
