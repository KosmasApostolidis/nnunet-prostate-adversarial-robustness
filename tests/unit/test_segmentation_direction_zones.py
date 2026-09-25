"""Zone transition matrices (spec §19, §43.8)."""

from __future__ import annotations

import numpy as np
import pytest

from mri_prostate_seg.experiments.segmentation_direction.zones import (
    TRANSITION_TYPES,
    multiclass_transition_tables,
    transition_rows,
)

SPACING = (3.0, 0.5, 0.5)
SHAPE = (8, 40, 40)
CLASS_IDS = [0, 1, 2]
CLASS_NAMES = ["background", "TZ+CZ", "PZ"]


def _two_zones() -> np.ndarray:
    """TZ+CZ (1) on the left half of a block, PZ (2) on the right."""

    gt = np.zeros(SHAPE, dtype=np.int16)
    gt[2:6, 10:30, 10:20] = 1
    gt[2:6, 10:30, 20:30] = 2
    return gt


def test_interface_shift_is_one_directional_harm() -> None:
    gt = _two_zones()
    adv = gt.copy()
    adv[2:6, 10:30, 18:20] = 2  # PZ invades 1 mm of TZ
    valid = np.ones(SHAPE, dtype=bool)
    tables = multiclass_transition_tables(gt, gt, adv, valid=valid, class_ids=CLASS_IDS)
    assert set(tables) == set(TRANSITION_TYPES)
    harm = tables["new_harm"]
    assert harm[1, 2] == 4 * 20 * 2
    assert harm[2, 1] == 0 and harm[1, 0] == 0 and harm[0, 1] == 0
    assert tables["raw"][1, 2] == harm[1, 2]
    assert tables["raw"].sum() == valid.sum()
    for name in ("new_harm", "correction", "wrong_to_wrong"):
        assert np.all(np.diag(tables[name]) == 0)


def test_correction_and_wrong_to_wrong_are_separated_from_harm() -> None:
    gt = _two_zones()
    clean = gt.copy()
    clean[2:6, 10:30, 18:20] = 2  # clean already wrong: PZ where TZ should be
    adv = gt.copy()
    adv[2:6, 10:30, 18:19] = 1  # attack fixes one column ...
    adv[2:6, 10:30, 19:20] = 0  # ... and turns the other into background (still wrong)
    valid = np.ones(SHAPE, dtype=bool)
    tables = multiclass_transition_tables(
        gt, clean, adv, valid=valid, class_ids=CLASS_IDS
    )
    assert tables["correction"][2, 1] == 4 * 20
    assert tables["wrong_to_wrong"][2, 0] == 4 * 20
    assert tables["new_harm"].sum() == 0


def test_rows_normalise_harm_by_clean_correct_voxels() -> None:
    gt = _two_zones()
    adv = gt.copy()
    adv[2:6, 10:30, 18:20] = 2
    valid = np.ones(SHAPE, dtype=bool)
    tables = multiclass_transition_tables(gt, gt, adv, valid=valid, class_ids=CLASS_IDS)
    rows = transition_rows(
        tables,
        class_ids=CLASS_IDS,
        class_names=CLASS_NAMES,
        ground_truth=gt,
        clean_pred=gt,
        valid=valid,
        spacing=SPACING,
    )
    assert len(rows) == 4 * 9
    harm = next(
        r
        for r in rows
        if r["transition_type"] == "new_harm"
        and r["source_class"] == "TZ+CZ"
        and r["target_class"] == "PZ"
    )
    assert harm["voxel_count"] == 160
    assert harm["row_normalized_rate"] == pytest.approx(160 / (4 * 20 * 10))
    assert harm["physical_volume_mm3"] == pytest.approx(160 * 0.75)


def test_invalid_voxels_are_excluded() -> None:
    gt = _two_zones()
    valid = np.ones(SHAPE, dtype=bool)
    valid[:, :5, :] = False
    gt[~valid] = -1
    adv = gt.copy()
    adv[:, :5, :] = 2
    tables = multiclass_transition_tables(gt, gt, adv, valid=valid, class_ids=CLASS_IDS)
    assert tables["raw"].sum() == valid.sum()
    assert tables["new_harm"].sum() == 0
