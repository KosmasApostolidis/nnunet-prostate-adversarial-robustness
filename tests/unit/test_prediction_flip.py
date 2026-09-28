from __future__ import annotations

import numpy as np
import pytest

from mri_prostate_seg.experiments.slice_vulnerability.prediction_flip import (
    prediction_flip_by_slice,
)


def _volume(depth: int = 3, size: int = 6) -> np.ndarray:
    return np.zeros((depth, size, size), dtype=np.int64)


def test_unchanged_prediction_has_flip_dice_one():
    gt = _volume()
    gt[:, 1:4, 1:4] = 1
    clean = gt.copy()
    rows = prediction_flip_by_slice(clean, clean.copy(), gt, class_id=1)
    assert [r["flip_dice"] for r in rows] == pytest.approx([1.0, 1.0, 1.0])


def test_flip_dice_compares_adversarial_with_clean_prediction_not_ground_truth():
    gt = _volume(depth=1)
    gt[0, 0:4, 0:4] = 1  # 16 px
    clean = _volume(depth=1)
    clean[0, 0:2, 0:4] = 1  # 8 px, half of the GT
    adversarial = clean.copy()  # identical to clean, still wrong vs GT
    (row,) = prediction_flip_by_slice(clean, adversarial, gt, class_id=1)
    assert row["flip_dice"] == pytest.approx(1.0)
    assert row["gt_area"] == 16


def test_partial_change_gives_dice_between_clean_and_adversarial_masks():
    gt = _volume(depth=1)
    gt[0, 0:4, 0:4] = 1
    clean = gt.copy()
    adversarial = _volume(depth=1)
    adversarial[0, 0:2, 0:4] = 1  # keeps 8 of 16 px
    (row,) = prediction_flip_by_slice(clean, adversarial, gt, class_id=1)
    assert row["flip_dice"] == pytest.approx(2 * 8 / (16 + 8), abs=1e-4)


def test_slices_without_ground_truth_are_skipped():
    gt = _volume()
    gt[1, 1:4, 1:4] = 1
    rows = prediction_flip_by_slice(gt.copy(), gt.copy(), gt, class_id=1)
    assert [r["slice_idx"] for r in rows] == [1]


def test_both_empty_predictions_count_as_unchanged():
    gt = _volume(depth=1)
    gt[0, 1:3, 1:3] = 1
    empty = _volume(depth=1)
    (row,) = prediction_flip_by_slice(empty, empty.copy(), gt, class_id=1)
    assert row["flip_dice"] == pytest.approx(1.0)


def test_ignore_label_voxels_are_excluded():
    gt = _volume(depth=1)
    gt[0, 0:2, 0:2] = 1
    gt[0, 4:, 4:] = -1
    clean = gt.clip(min=0)
    adversarial = clean.copy()
    adversarial[0, 4:, 4:] = 1  # change only inside the ignore region
    (row,) = prediction_flip_by_slice(clean, adversarial, gt, class_id=1)
    assert row["flip_dice"] == pytest.approx(1.0)
