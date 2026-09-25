"""Damage-sign and metric phantoms (plan §18)."""

from __future__ import annotations

import math

import numpy as np
import pytest

from mri_prostate_seg.experiments.edge_attribution.damage import (
    HIGHER_IS_BETTER,
    damage,
    fractional,
    primary_dice,
    surface_metrics,
)

SHAPE = (4, 20, 20)


def _block(
    z0: int, z1: int, y0: int, y1: int, x0: int, x1: int, value: int = 1
) -> np.ndarray:
    out = np.zeros(SHAPE, dtype=np.int16)
    out[z0:z1, y0:y1, x0:x1] = value
    return out


def test_damage_is_positive_when_dice_falls_and_when_hd95_rises() -> None:
    assert damage("dice", 0.9, 0.6) == pytest.approx(0.3)
    assert damage("hd95", 2.0, 5.0) == pytest.approx(3.0)
    assert damage("loss", 0.1, 0.4) == pytest.approx(0.3)
    assert math.isnan(damage("hd95", 2.0, math.inf))
    assert set(HIGHER_IS_BETTER) == {
        "dice",
        "dice_tz",
        "dice_pz",
        "hd95",
        "asd",
        "loss",
    }


def test_fractional_is_nan_for_non_damaging_or_undefined() -> None:
    assert fractional(0.1, 0.2) == pytest.approx(0.5)
    assert math.isnan(fractional(0.1, 0.0))
    assert math.isnan(fractional(0.1, -0.2))
    assert math.isnan(fractional(math.nan, 0.2))
    assert fractional(-0.1, 0.2) == pytest.approx(-0.5)  # not clipped (plan §19.2)


def test_primary_dice_wg_and_zones_macro() -> None:
    gt = _block(1, 3, 5, 15, 5, 15)
    assert primary_dice(gt, gt, dataset_key="wg") == {"dice": 1.0}
    zones_gt = gt.copy()
    zones_gt[1:3, 5:15, 10:15] = 2
    pred = zones_gt.copy()
    pred[1:3, 5:15, 10:15] = 1  # PZ lost entirely to TZ
    out = primary_dice(pred, zones_gt, dataset_key="zones")
    assert out["dice_pz"] == pytest.approx(0.0)
    assert out["dice_tz"] == pytest.approx(2 * 100 / (200 + 100))
    assert out["dice"] == pytest.approx((out["dice_tz"] + out["dice_pz"]) / 2)


def test_surface_metrics_return_inf_for_empty_prediction() -> None:
    gt = _block(1, 3, 5, 15, 5, 15)
    out = surface_metrics(
        np.zeros(SHAPE, np.int16), gt, dataset_key="wg", spacing=(3.0, 0.5, 0.5)
    )
    assert math.isinf(out["hd95"]) and math.isinf(out["asd"])
    same = surface_metrics(gt, gt, dataset_key="wg", spacing=(3.0, 0.5, 0.5))
    assert same["hd95"] == 0.0 and same["asd"] == 0.0
