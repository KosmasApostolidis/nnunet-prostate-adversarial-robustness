"""Segmentation metrics — pure-Python wrappers around medpy and NumPy.

Functions here intentionally take NumPy arrays (not torch tensors) so unit
tests can exercise them without GPU or torch installed. The full
``compute_metrics`` adapter that bridges torch tensor outputs into these
primitives lives in ``mri_prostate_seg.eval.runner``.
"""

from __future__ import annotations

import logging
from typing import Any, Tuple

import numpy as np

logger = logging.getLogger(__name__)

# Additive smoothing constant for Dice numerator/denominator. Keeps the score
# well-defined when both masks are empty (returns ~1.0) and matches the legacy
# adversarial pipeline.
DICE_SMOOTH: float = 1e-6


def dice_score(pred: np.ndarray, target: np.ndarray) -> float:
    """Sørensen–Dice coefficient between two binary arrays of the same shape.

    Both inputs are coerced to boolean; an empty union returns 1.0 to match
    the convention used in the legacy adversarial evaluation pipeline (two
    correctly-empty masks score perfect overlap).
    """
    if pred.shape != target.shape:
        raise ValueError(
            f"dice_score: shape mismatch pred={pred.shape} vs target={target.shape}"
        )

    a = pred.astype(bool)
    b = target.astype(bool)
    intersection = np.logical_and(a, b).sum()
    denom = a.sum() + b.sum()
    if denom == 0:
        return 1.0
    return float(2.0 * intersection / denom)


def iou_score(pred: np.ndarray, target: np.ndarray) -> float:
    """Intersection-over-Union between two binary arrays of the same shape."""
    if pred.shape != target.shape:
        raise ValueError(
            f"iou_score: shape mismatch pred={pred.shape} vs target={target.shape}"
        )

    a = pred.astype(bool)
    b = target.astype(bool)
    intersection = np.logical_and(a, b).sum()
    union = np.logical_or(a, b).sum()
    if union == 0:
        return 1.0
    return float(intersection / union)


def hausdorff_distance_95(
    pred: np.ndarray,
    target: np.ndarray,
    voxel_spacing: tuple[float, ...] | None = None,
) -> float:
    """95th-percentile Hausdorff distance via medpy.

    Returns ``float('inf')`` when either mask is empty — matching legacy
    behaviour where empty predictions are scored as worst-case.
    """
    from medpy.metric.binary import hd95

    a = pred.astype(bool)
    b = target.astype(bool)
    if not a.any() or not b.any():
        return float("inf")
    return float(hd95(a, b, voxelspacing=voxel_spacing))


def average_surface_distance(
    pred: np.ndarray,
    target: np.ndarray,
    voxel_spacing: tuple[float, ...] | None = None,
) -> float:
    """Average surface distance via medpy. Returns ``inf`` for empty masks."""
    from medpy.metric.binary import asd

    a = pred.astype(bool)
    b = target.astype(bool)
    if not a.any() or not b.any():
        return float("inf")
    return float(asd(a, b, voxelspacing=voxel_spacing))


def pad_to_shape(
    array: np.ndarray,
    target_shape: tuple[int, ...],
    *,
    constant_value: float = 0.0,
) -> np.ndarray:
    """Symmetrically zero-pad ``array`` to ``target_shape``.

    Each axis must be at most as large as the corresponding target dim;
    odd extra padding lands on the trailing edge.
    """
    if len(array.shape) != len(target_shape):
        raise ValueError(
            f"pad_to_shape: rank mismatch array={array.shape} vs target={target_shape}"
        )

    pad_width: list[tuple[int, int]] = []
    for cur, tgt in zip(array.shape, target_shape):
        if cur > tgt:
            raise ValueError(
                f"pad_to_shape: axis {cur} larger than target {tgt}; would require cropping"
            )
        diff = tgt - cur
        before = diff // 2
        after = diff - before
        pad_width.append((before, after))

    return np.pad(array, pad_width, mode="constant", constant_values=constant_value)


def crop_to_shape(array: np.ndarray, target_shape: tuple[int, ...]) -> np.ndarray:
    """Center-crop ``array`` to ``target_shape``.

    Each axis must be at least as large as the target dim.
    """
    if len(array.shape) != len(target_shape):
        raise ValueError(
            f"crop_to_shape: rank mismatch array={array.shape} vs target={target_shape}"
        )

    slices: list[slice] = []
    for cur, tgt in zip(array.shape, target_shape):
        if cur < tgt:
            raise ValueError(
                f"crop_to_shape: axis {cur} smaller than target {tgt}; would require padding"
            )
        start = (cur - tgt) // 2
        slices.append(slice(start, start + tgt))

    return array[tuple(slices)]


def per_class_dice(
    pred: np.ndarray,
    target: np.ndarray,
    *,
    num_classes: int,
    ignore_background: bool = True,
) -> dict[int, float]:
    """Compute per-class Dice for an integer-labelled prediction / target."""
    if pred.shape != target.shape:
        raise ValueError(
            f"per_class_dice: shape mismatch pred={pred.shape} vs target={target.shape}"
        )

    out: dict[int, float] = {}
    start = 1 if ignore_background else 0
    for c in range(start, num_classes):
        out[c] = dice_score(
            (pred == c).astype(np.uint8), (target == c).astype(np.uint8)
        )
    return out


def class_metrics_triple(
    pred_bin: np.ndarray,
    gt_bin: np.ndarray,
    voxel_spacing: tuple[float, ...] | None = None,
) -> Tuple[float, float, float]:
    """Return (Dice, HD95, ASD) for one binary class.

    HD95 and ASD are undefined when either mask is empty (no foreground voxels).
    medpy raises in that case; we catch the documented exceptions, log them at
    INFO so downstream NaN rows in result CSVs are auditable, and return NaN
    instead of silently coercing the geometry.

    Dice uses additive smoothing so two empty masks score ~1.0. HD95 and ASD
    share the prediction-to-target surface distances, preserving MedPy's exact
    definitions while avoiding a redundant distance transform.
    """
    inter = (pred_bin * gt_bin).sum()
    denom = pred_bin.sum() + gt_bin.sum()
    dice_val = float((2.0 * inter + DICE_SMOOTH) / (denom + DICE_SMOOTH))

    kwargs: dict[str, Any] = {}
    if voxel_spacing is not None:
        kwargs["voxelspacing"] = voxel_spacing

    pred_count = int(pred_bin.sum())
    gt_count = int(gt_bin.sum())

    from medpy.metric.binary import __surface_distances

    try:
        prediction_to_target = __surface_distances(pred_bin, gt_bin, **kwargs)
    except (RuntimeError, ValueError, TypeError) as e:
        logger.info(
            "boundary metrics undefined (pred=%d, gt=%d voxels): %s",
            pred_count,
            gt_count,
            e,
        )
        return dice_val, float("nan"), float("nan")

    asd_val = float(prediction_to_target.mean())
    try:
        target_to_prediction = __surface_distances(gt_bin, pred_bin, **kwargs)
        hd95_val = float(
            np.percentile(
                np.hstack((prediction_to_target, target_to_prediction)),
                95,
            )
        )
    except (RuntimeError, ValueError, TypeError) as e:
        logger.info(
            "hd95 undefined (pred=%d, gt=%d voxels): %s",
            pred_count,
            gt_count,
            e,
        )
        hd95_val = float("nan")
    return dice_val, hd95_val, asd_val


__all__ = [
    "DICE_SMOOTH",
    "dice_score",
    "iou_score",
    "hausdorff_distance_95",
    "average_surface_distance",
    "pad_to_shape",
    "crop_to_shape",
    "per_class_dice",
    "class_metrics_triple",
]
