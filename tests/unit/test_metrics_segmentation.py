"""Unit tests for ``mri_prostate_seg.metrics.segmentation``.

Pure NumPy assertions — no torch, no sitk, no GPU.
"""

from __future__ import annotations

import numpy as np
import pytest

from mri_prostate_seg.metrics.segmentation import (
    average_surface_distance,
    class_metrics_triple,
    crop_to_shape,
    dice_score,
    hausdorff_distance_95,
    iou_score,
    pad_to_shape,
    per_class_dice,
)


# ---------------------------------------------------------------------------
# dice_score
# ---------------------------------------------------------------------------


def test_dice_perfect_overlap_returns_one() -> None:
    a = np.ones((4, 4), dtype=np.uint8)
    assert dice_score(a, a) == pytest.approx(1.0)


def test_dice_disjoint_masks_returns_zero() -> None:
    a = np.zeros((4, 4), dtype=np.uint8)
    a[:2] = 1
    b = np.zeros((4, 4), dtype=np.uint8)
    b[2:] = 1
    assert dice_score(a, b) == pytest.approx(0.0)


def test_dice_two_empty_masks_returns_one() -> None:
    a = np.zeros((4, 4), dtype=np.uint8)
    assert dice_score(a, a) == 1.0


def test_dice_partial_overlap_value() -> None:
    a = np.array([[1, 1, 0, 0]], dtype=np.uint8)
    b = np.array([[1, 0, 1, 0]], dtype=np.uint8)
    # |A∩B| = 1, |A| = 2, |B| = 2, dice = 2*1 / (2+2) = 0.5
    assert dice_score(a, b) == pytest.approx(0.5)


def test_dice_coerces_to_bool_for_continuous_arrays() -> None:
    a = np.array([0.7, 0.0, 0.3], dtype=np.float32)
    b = np.array([1.0, 0.0, 1.0], dtype=np.float32)
    assert dice_score(a, b) == pytest.approx(2 * 2 / (2 + 2))


def test_dice_shape_mismatch_raises() -> None:
    a = np.zeros((4, 4), dtype=np.uint8)
    b = np.zeros((4, 5), dtype=np.uint8)
    with pytest.raises(ValueError, match="shape mismatch"):
        dice_score(a, b)


# ---------------------------------------------------------------------------
# iou_score
# ---------------------------------------------------------------------------


def test_iou_perfect_overlap_returns_one() -> None:
    a = np.ones((3, 3), dtype=np.uint8)
    assert iou_score(a, a) == 1.0


def test_iou_disjoint_returns_zero() -> None:
    a = np.array([1, 1, 0, 0], dtype=np.uint8)
    b = np.array([0, 0, 1, 1], dtype=np.uint8)
    assert iou_score(a, b) == 0.0


def test_iou_partial_overlap_value() -> None:
    a = np.array([1, 1, 1, 0], dtype=np.uint8)
    b = np.array([1, 1, 0, 0], dtype=np.uint8)
    # ∩ = 2, ∪ = 3, IoU = 2/3
    assert iou_score(a, b) == pytest.approx(2 / 3)


def test_iou_two_empty_masks_returns_one() -> None:
    a = np.zeros((3, 3), dtype=np.uint8)
    assert iou_score(a, a) == 1.0


def test_iou_shape_mismatch_raises() -> None:
    with pytest.raises(ValueError, match="shape mismatch"):
        iou_score(np.zeros((2, 2)), np.zeros((2, 3)))


# ---------------------------------------------------------------------------
# hausdorff + asd  (medpy-backed; only the empty-mask short-circuit is tested)
# ---------------------------------------------------------------------------


def test_hd95_returns_inf_when_pred_empty() -> None:
    a = np.zeros((4, 4), dtype=np.uint8)
    b = np.ones((4, 4), dtype=np.uint8)
    assert hausdorff_distance_95(a, b) == float("inf")


def test_hd95_returns_inf_when_target_empty() -> None:
    a = np.ones((4, 4), dtype=np.uint8)
    b = np.zeros((4, 4), dtype=np.uint8)
    assert hausdorff_distance_95(a, b) == float("inf")


def test_asd_returns_inf_when_either_empty() -> None:
    a = np.zeros((4, 4), dtype=np.uint8)
    b = np.ones((4, 4), dtype=np.uint8)
    assert average_surface_distance(a, b) == float("inf")
    assert average_surface_distance(b, a) == float("inf")


def test_hd95_real_distance_for_nonempty_masks() -> None:
    # Single voxels at distance 3 in a 5x5 grid
    a = np.zeros((5, 5), dtype=np.uint8)
    a[0, 0] = 1
    b = np.zeros((5, 5), dtype=np.uint8)
    b[0, 3] = 1
    d = hausdorff_distance_95(a, b)
    assert d == pytest.approx(3.0)


def test_asd_zero_for_identical_masks() -> None:
    a = np.zeros((5, 5), dtype=np.uint8)
    a[1:4, 1:4] = 1
    assert average_surface_distance(a, a) == 0.0


def test_class_metrics_triple_matches_medpy_boundary_metrics() -> None:
    from medpy.metric.binary import asd, hd95

    target = np.zeros((4, 7, 7), dtype=np.uint8)
    target[1:3, 2:5, 2:5] = 1
    prediction = np.roll(target, shift=1, axis=2)
    spacing = (3.0, 0.5, 0.5)

    _dice, combined_hd95, combined_asd = class_metrics_triple(
        prediction,
        target,
        voxel_spacing=spacing,
    )

    assert combined_hd95 == pytest.approx(
        hd95(prediction, target, voxelspacing=spacing)
    )
    assert combined_asd == pytest.approx(asd(prediction, target, voxelspacing=spacing))


# ---------------------------------------------------------------------------
# pad_to_shape
# ---------------------------------------------------------------------------


def test_pad_already_target_shape_returns_unchanged_array() -> None:
    a = np.arange(6).reshape(2, 3)
    out = pad_to_shape(a, (2, 3))
    assert out.shape == (2, 3)
    np.testing.assert_array_equal(out, a)


def test_pad_symmetric_even_difference() -> None:
    a = np.array([[1, 2]], dtype=np.int32)  # shape (1, 2)
    out = pad_to_shape(a, (3, 4))
    assert out.shape == (3, 4)
    assert out[1, 1] == 1
    assert out[1, 2] == 2


def test_pad_odd_difference_padding_lands_on_trailing_edge() -> None:
    a = np.array([[5]], dtype=np.int32)  # (1, 1)
    out = pad_to_shape(a, (1, 4))
    assert out.shape == (1, 4)
    # diff=3, before=1, after=2 → value at index 1
    assert out[0, 1] == 5
    assert out[0, 2] == 0
    assert out[0, 3] == 0


def test_pad_constant_value_propagated() -> None:
    a = np.zeros((1, 1), dtype=np.float32)
    out = pad_to_shape(a, (3, 3), constant_value=-1.0)
    assert out[0, 0] == -1.0
    assert out[1, 1] == 0.0


def test_pad_axis_already_too_large_raises() -> None:
    a = np.zeros((5, 5))
    with pytest.raises(ValueError, match="larger than target"):
        pad_to_shape(a, (3, 5))


def test_pad_rank_mismatch_raises() -> None:
    a = np.zeros((2, 2))
    with pytest.raises(ValueError, match="rank mismatch"):
        pad_to_shape(a, (2, 2, 2))


# ---------------------------------------------------------------------------
# crop_to_shape
# ---------------------------------------------------------------------------


def test_crop_returns_center_slice() -> None:
    a = np.arange(25).reshape(5, 5)
    out = crop_to_shape(a, (3, 3))
    assert out.shape == (3, 3)
    np.testing.assert_array_equal(out, a[1:4, 1:4])


def test_crop_no_op_for_target_equal_shape() -> None:
    a = np.arange(9).reshape(3, 3)
    np.testing.assert_array_equal(crop_to_shape(a, (3, 3)), a)


def test_crop_axis_smaller_than_target_raises() -> None:
    a = np.zeros((3, 3))
    with pytest.raises(ValueError, match="smaller than target"):
        crop_to_shape(a, (5, 3))


def test_crop_rank_mismatch_raises() -> None:
    with pytest.raises(ValueError, match="rank mismatch"):
        crop_to_shape(np.zeros((2, 2)), (2,))


# ---------------------------------------------------------------------------
# per_class_dice
# ---------------------------------------------------------------------------


def test_per_class_dice_skips_background_by_default() -> None:
    pred = np.array([0, 1, 1, 2, 2], dtype=np.int32)
    target = np.array([0, 1, 0, 2, 2], dtype=np.int32)
    out = per_class_dice(pred, target, num_classes=3)
    assert 0 not in out
    assert out[1] == pytest.approx(2 * 1 / (2 + 1))
    assert out[2] == pytest.approx(1.0)


def test_per_class_dice_includes_background_when_requested() -> None:
    pred = np.array([0, 1, 1], dtype=np.int32)
    target = np.array([0, 1, 0], dtype=np.int32)
    out = per_class_dice(pred, target, num_classes=2, ignore_background=False)
    assert 0 in out
    assert 1 in out


def test_per_class_dice_shape_mismatch_raises() -> None:
    with pytest.raises(ValueError, match="shape mismatch"):
        per_class_dice(np.zeros(5), np.zeros(6), num_classes=2)
