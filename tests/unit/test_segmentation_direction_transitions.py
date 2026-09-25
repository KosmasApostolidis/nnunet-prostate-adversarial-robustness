"""Phantom tests for ground-truth-aware binary transitions (spec §6, §43, §44)."""

from __future__ import annotations

import numpy as np
import pytest

from mri_prostate_seg.experiments.segmentation_direction.transitions import (
    binary_transition_masks,
    centroid_shift_norm_mm,
    physical_measure,
    summarize_binary_transitions,
)

SPACING = (3.0, 0.5, 0.5)
SHAPE = (12, 64, 64)


def _ball(
    radius_mm: float, center_mm: tuple[float, float, float] = (18.0, 16.0, 16.0)
) -> np.ndarray:
    grids = np.meshgrid(
        *[np.arange(n) * s for n, s in zip(SHAPE, SPACING)], indexing="ij"
    )
    r2 = sum((g - c) ** 2 for g, c in zip(grids, center_mm))
    return r2 <= radius_mm**2


def _valid() -> np.ndarray:
    return np.ones(SHAPE, dtype=bool)


def test_partition_is_exact_for_expansion() -> None:
    gt = _ball(8.0)
    clean = gt.copy()
    adv = _ball(10.0)
    tr = binary_transition_masks(gt, clean, adv, valid=_valid())
    changed = tr.gained_fg | tr.lost_fg
    categories = (
        tr.induced_fp.astype(int) + tr.induced_fn + tr.corrected_fp + tr.corrected_fn
    )
    assert np.array_equal(categories > 0, changed)
    assert categories.max() == 1
    assert tr.induced_fp.sum() == (adv & ~gt).sum()
    assert not tr.induced_fn.any() and not tr.corrected_fp.any()


def test_expansion_has_change_bias_plus_one_and_positive_damage_bias() -> None:
    gt = _ball(8.0)
    clean = gt.copy()
    adv = _ball(10.0)
    tr = binary_transition_masks(gt, clean, adv, valid=_valid())
    s = summarize_binary_transitions(
        tr, gt, clean, adv, valid=_valid(), spacing=SPACING
    )
    assert s["change_bias"] == pytest.approx(1.0)
    assert s["damage_bias"] == pytest.approx(1.0)
    assert s["volume_change_mm3"] > 0
    assert s["volume_change_mm3"] == pytest.approx(
        s["gained_fg_mm3"] - s["lost_fg_mm3"]
    )
    assert s["harm_volume_mm3"] == pytest.approx(s["induced_fp_mm3"])
    assert s["corrected_volume_mm3"] == 0.0
    assert s["centroid_shift_norm_mm"] == pytest.approx(0.0, abs=0.3)


def test_contraction_has_change_bias_minus_one() -> None:
    gt = _ball(8.0)
    adv = _ball(6.0)
    tr = binary_transition_masks(gt, gt, adv, valid=_valid())
    s = summarize_binary_transitions(tr, gt, gt, adv, valid=_valid(), spacing=SPACING)
    assert s["change_bias"] == pytest.approx(-1.0)
    assert s["damage_bias"] == pytest.approx(-1.0)
    assert s["volume_change_mm3"] < 0


def test_correction_is_not_counted_as_harm() -> None:
    gt = _ball(8.0)
    clean = _ball(6.0)  # clean under-segments
    adv = gt.copy()  # attack "fixes" it
    tr = binary_transition_masks(gt, clean, adv, valid=_valid())
    s = summarize_binary_transitions(
        tr, gt, clean, adv, valid=_valid(), spacing=SPACING
    )
    assert s["harm_volume_mm3"] == 0.0
    assert s["corrected_fn_mm3"] > 0
    assert s["change_bias"] == pytest.approx(1.0)
    assert np.isnan(s["damage_bias"]) or s["damage_bias"] == pytest.approx(0.0)


def test_translation_has_zero_net_volume_but_nonzero_centroid_shift() -> None:
    gt = _ball(8.0)
    adv = np.roll(gt, 4, axis=2)  # +2 mm along x
    tr = binary_transition_masks(gt, gt, adv, valid=_valid())
    s = summarize_binary_transitions(tr, gt, gt, adv, valid=_valid(), spacing=SPACING)
    assert s["volume_change_mm3"] == pytest.approx(0.0)
    assert abs(s["change_bias"]) < 1e-9
    assert s["centroid_shift_norm_mm"] == pytest.approx(2.0, abs=1e-6)
    assert s["gained_fg_mm3"] > 0 and s["lost_fg_mm3"] > 0


def test_ignore_label_border_contributes_nothing() -> None:
    gt = _ball(8.0).astype(np.int16)
    valid = np.ones(SHAPE, dtype=bool)
    valid[:, :4, :] = False  # a zeroed border, gt == -1 there
    gt[~valid] = -1
    clean = _ball(8.0)
    adv = clean.copy()
    adv[:, :4, :] = True  # the attack "predicts" foreground in the invalid band
    tr = binary_transition_masks(gt == 1, clean, adv, valid=valid)
    assert not (tr.gained_fg | tr.lost_fg).any()
    s = summarize_binary_transitions(
        tr, gt == 1, clean, adv, valid=valid, spacing=SPACING
    )
    assert s["harm_volume_mm3"] == 0.0
    assert s["changed_voxels_outside_valid"] == int((~valid).sum())


def test_physical_measure_uses_spacing() -> None:
    mask = np.zeros(SHAPE, dtype=bool)
    mask[0, 0, :10] = True
    assert physical_measure(mask, SPACING) == pytest.approx(10 * 3.0 * 0.5 * 0.5)
    with pytest.raises(ValueError):
        physical_measure(mask, (1.0, 1.0))


def test_empty_masks_give_nan_centroid_but_finite_volumes() -> None:
    gt = _ball(8.0)
    empty = np.zeros(SHAPE, dtype=bool)
    tr = binary_transition_masks(gt, gt, empty, valid=_valid())
    s = summarize_binary_transitions(tr, gt, gt, empty, valid=_valid(), spacing=SPACING)
    assert s["adv_volume_mm3"] == 0.0
    assert s["change_bias"] == pytest.approx(-1.0)
    assert np.isnan(centroid_shift_norm_mm(gt, empty, SPACING))


def test_volume_change_pct_is_nan_on_empty_clean_mask() -> None:
    gt = _ball(8.0)
    empty = np.zeros(SHAPE, dtype=bool)
    adv = _ball(6.0)
    tr = binary_transition_masks(gt, empty, adv, valid=_valid())
    s = summarize_binary_transitions(
        tr, gt, empty, adv, valid=_valid(), spacing=SPACING
    )
    assert s["clean_volume_mm3"] == 0.0
    assert np.isnan(s["volume_change_pct"])
    assert np.isfinite(s["clean_volume_mm3"])
    assert np.isfinite(s["adv_volume_mm3"])
    assert np.isfinite(s["volume_change_mm3"])


def test_shape_mismatch_raises() -> None:
    gt = _ball(8.0)
    with pytest.raises(ValueError):
        binary_transition_masks(gt, gt[:-1], gt, valid=_valid())
