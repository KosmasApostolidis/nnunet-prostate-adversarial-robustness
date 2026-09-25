"""Edge-strength phantoms (spec: anisotropy, plan §10)."""

from __future__ import annotations

import numpy as np
import pytest

from mri_prostate_seg.experiments.edge_attribution.gradients import (
    multiscale_edge_strength,
    robust_normalize,
    roi_mask,
)

SPACING = (3.0, 0.5, 0.5)
SHAPE = (12, 64, 64)


def _step_image(edge_x: int = 32, high: float = 2.0) -> np.ndarray:
    image = np.zeros(SHAPE, dtype=np.float32)
    image[:, :, edge_x:] = high
    return image


def test_roi_mask_expands_in_plane_by_margin_and_one_slice() -> None:
    fg = np.zeros(SHAPE, dtype=bool)
    fg[5:7, 30:34, 30:34] = True
    valid = np.ones(SHAPE, dtype=bool)
    roi = roi_mask(fg, valid, SPACING, margin_mm_inplane=4.0, margin_slices=1)
    z, y, x = np.nonzero(roi)
    assert z.min() == 4 and z.max() == 7  # one slice each side
    assert y.min() == 22 and y.max() == 41  # 4 mm = 8 voxels each side
    assert x.min() == 22 and x.max() == 41


def test_roi_mask_is_clipped_to_valid() -> None:
    fg = np.zeros(SHAPE, dtype=bool)
    fg[5:7, 30:34, 30:34] = True
    valid = np.zeros(SHAPE, dtype=bool)
    valid[:, :, :32] = True
    roi = roi_mask(fg, valid, SPACING, margin_mm_inplane=4.0)
    assert roi.any()
    assert not roi[:, :, 32:].any()


def test_roi_mask_rejects_empty_foreground() -> None:
    with pytest.raises(ValueError):
        roi_mask(np.zeros(SHAPE, bool), np.ones(SHAPE, bool), SPACING)


def test_robust_normalize_maps_median_to_zero_and_p95_to_one() -> None:
    rng = np.random.default_rng(0)
    values = rng.random(SHAPE)
    domain = np.ones(SHAPE, dtype=bool)
    out = robust_normalize(values, domain)
    inside = out[domain]
    assert np.median(inside) == pytest.approx(0.0, abs=1e-9)
    assert np.quantile(inside, 0.95) == pytest.approx(1.0, abs=1e-9)


def test_edge_strength_peaks_on_the_step_edge() -> None:
    image = _step_image()
    domain = np.ones(SHAPE, dtype=bool)
    strength = multiscale_edge_strength(
        image, SPACING, sigmas_mm=[0.5, 1.0, 2.0], domain=domain
    )
    assert strength.shape == SHAPE
    column = strength[6, 32, :]
    assert int(np.argmax(column)) in (31, 32)
    assert column[31:33].min() > column[:20].max()


def test_edge_strength_smooths_in_plane_only() -> None:
    # A one-slice bright plane has a z-gradient; in-plane smoothing must not
    # spread it into neighbouring slices.
    image = np.zeros(SHAPE, dtype=np.float32)
    image[6] = 1.0
    domain = np.ones(SHAPE, dtype=bool)
    strength = multiscale_edge_strength(image, SPACING, sigmas_mm=[2.0], domain=domain)
    assert strength[3].max() == pytest.approx(strength[0].max())  # untouched far slices


def test_edge_strength_is_physical_across_spacings() -> None:
    # Same physical step (2 units over one in-plane voxel) at two spacings gives
    # the same raw physical gradient magnitude before normalisation.
    from mri_prostate_seg.experiments.perturbation_structure import (
        gradient_magnitude_mm,
    )

    a = gradient_magnitude_mm(_step_image(), (3.0, 0.5, 0.5))[6, 32, 32]
    b = gradient_magnitude_mm(_step_image(), (1.0, 0.5, 0.5))[6, 32, 32]
    assert a == pytest.approx(b)
