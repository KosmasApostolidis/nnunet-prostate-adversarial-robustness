"""Positive-is-worse damage and the metrics it is built from (plan §18)."""

from __future__ import annotations

import math

import numpy as np

from mri_prostate_seg.metrics.segmentation import (
    average_surface_distance,
    dice_score,
    hausdorff_distance_95,
)

HIGHER_IS_BETTER: dict[str, bool] = {
    "dice": True,
    "dice_tz": True,
    "dice_pz": True,
    "hd95": False,
    "asd": False,
    "loss": False,
}
_NAN = float("nan")


def damage(metric: str, clean_value: float, value: float) -> float:
    if metric not in HIGHER_IS_BETTER:
        raise KeyError(f"unknown metric {metric!r}")
    if not (math.isfinite(clean_value) and math.isfinite(value)):
        return _NAN
    if HIGHER_IS_BETTER[metric]:
        return float(clean_value - value)
    return float(value - clean_value)


def fractional(numerator: float, d_full: float) -> float:
    if not (math.isfinite(numerator) and math.isfinite(d_full)) or d_full <= 0:
        return _NAN
    return float(numerator / d_full)


def class_dice(pred: np.ndarray, target: np.ndarray, label: int) -> float:
    return float(dice_score(np.asarray(pred) == label, np.asarray(target) == label))


def primary_dice(
    pred: np.ndarray, target: np.ndarray, *, dataset_key: str
) -> dict[str, float]:
    if dataset_key == "wg":
        return {"dice": class_dice(pred, target, 1)}
    if dataset_key == "zones":
        tz = class_dice(pred, target, 1)
        pz = class_dice(pred, target, 2)
        return {"dice": (tz + pz) / 2.0, "dice_tz": tz, "dice_pz": pz}
    raise ValueError(f"unknown dataset {dataset_key!r}")


def surface_metrics(
    pred: np.ndarray,
    target: np.ndarray,
    *,
    dataset_key: str,
    spacing: tuple[float, float, float],
) -> dict[str, float]:
    if dataset_key not in ("wg", "zones"):
        raise ValueError(f"unknown dataset {dataset_key!r}")
    p = np.asarray(pred) > 0
    t = np.asarray(target) > 0
    return {
        "hd95": hausdorff_distance_95(p, t, voxel_spacing=spacing),
        "asd": average_surface_distance(p, t, voxel_spacing=spacing),
    }


__all__ = [
    "HIGHER_IS_BETTER",
    "class_dice",
    "damage",
    "fractional",
    "primary_dice",
    "surface_metrics",
]
