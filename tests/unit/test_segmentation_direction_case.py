from __future__ import annotations

import numpy as np
import pytest

from mri_prostate_seg.experiments.perturbation_structure import SIGNED_BAND_NAMES
from mri_prostate_seg.experiments.segmentation_direction import (
    analyze_binary_target,
    attack_success,
    target_geometry,
)

SPACING = (3.0, 0.5, 0.5)
SHAPE = (12, 64, 64)


def _ball(radius_mm: float) -> np.ndarray:
    grids = np.meshgrid(
        *[np.arange(n) * s for n, s in zip(SHAPE, SPACING)], indexing="ij"
    )
    r2 = sum((g - c) ** 2 for g, c in zip(grids, (18.0, 16.0, 16.0)))
    return r2 <= radius_mm**2


def test_summary_and_profile_shapes() -> None:
    gt, adv = _ball(8.0), _ball(10.0)
    valid = np.ones(SHAPE, dtype=bool)
    summary, profile = analyze_binary_target(
        gt, gt, adv, valid=valid, spacing=SPACING, tolerance_mm=0.5
    )
    assert len(profile) == len(SIGNED_BAND_NAMES)
    for key in (
        "damage_bias",
        "median_surface_motion_mm",
        "clean_dice",
        "adv_dice",
        "delta_dice",
        "surface_failure_reason",
    ):
        assert key in summary
    assert summary["clean_dice"] == pytest.approx(1.0)
    assert summary["delta_dice"] < 0
    assert summary["damage_bias"] == pytest.approx(1.0)
    assert summary["median_surface_motion_mm"] == pytest.approx(2.0, abs=0.5)


def test_geometry_reuse_changes_nothing() -> None:
    gt, adv = _ball(8.0), _ball(6.0)
    valid = np.ones(SHAPE, dtype=bool)
    direct = analyze_binary_target(
        gt, gt, adv, valid=valid, spacing=SPACING, tolerance_mm=0.5
    )
    cached = analyze_binary_target(
        gt,
        gt,
        adv,
        valid=valid,
        spacing=SPACING,
        tolerance_mm=0.5,
        geometry=target_geometry(gt, gt, SPACING),
    )
    np.testing.assert_equal(direct[0], cached[0])  # NaN-aware dict comparison
    np.testing.assert_equal(direct[1], cached[1])


def test_empty_adversarial_keeps_volumes_and_flags_surface() -> None:
    gt = _ball(8.0)
    valid = np.ones(SHAPE, dtype=bool)
    summary, profile = analyze_binary_target(
        gt, gt, np.zeros(SHAPE, bool), valid=valid, spacing=SPACING, tolerance_mm=0.5
    )
    assert summary["adv_volume_mm3"] == 0.0
    assert summary["surface_metric_valid"] == 0.0
    assert summary["surface_failure_reason"] == "empty_adv"
    inner = next(r for r in profile if r["band"] == "inside_0_2mm")
    assert inner["induced_fn_rate"] == pytest.approx(1.0)


def test_empty_ground_truth_raises() -> None:
    with pytest.raises(ValueError):
        analyze_binary_target(
            np.zeros(SHAPE, bool),
            _ball(8.0),
            _ball(8.0),
            valid=np.ones(SHAPE, bool),
            spacing=SPACING,
            tolerance_mm=0.5,
        )


def test_attack_success_threshold() -> None:
    assert attack_success(-0.01)
    assert attack_success(-0.5)
    assert not attack_success(-0.009)
    assert not attack_success(float("nan"))
