"""Driver helpers that need no GPU."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from experiments.evaluate_segmentation_direction_all_cases import (
    PROFILE_FIELDNAMES,
    SUMMARY_FIELDNAMES,
    TRANSITION_FIELDNAMES,
    completed_keys,
    direction_rows,
    expected_targets,
    head_target,
    mask_path,
    targets_for,
)
from mri_prostate_seg.experiments.segmentation_direction import target_geometry

SPACING = (3.0, 0.5, 0.5)
SHAPE = (10, 48, 48)


def _zones_case() -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    gt = np.zeros(SHAPE, dtype=np.int16)
    gt[2:8, 8:40, 8:24] = 1
    gt[2:8, 8:40, 24:40] = 2
    gt[:, :2, :] = -1
    clean = np.where(gt < 0, 0, gt).astype(np.int16)
    adv = clean.copy()
    adv[2:8, 8:40, 22:24] = 2
    return gt, clean, adv


def test_targets_for_each_dataset() -> None:
    gt, clean, adv = _zones_case()
    names = [t[0] for t in targets_for("zones", gt, clean, adv)]
    assert names == ["union", "TZ+CZ", "PZ"]
    union = targets_for("zones", gt, clean, adv)[0][1]
    assert union.dtype == bool and not union[0, 0, 0]
    ones = np.ones(SHAPE, np.int16)
    assert [t[0] for t in targets_for("wg", ones, ones, ones)] == ["WG"]
    assert expected_targets("zones") == {"union", "TZ+CZ", "PZ"}
    assert head_target("wg") == "WG" and head_target("zones") == "union"


def test_completed_keys_needs_every_target(tmp_path) -> None:
    csv = tmp_path / "s.csv"
    rows = [
        {
            "dataset": "zones",
            "case_id": "a",
            "epsilon_n": 2,
            "attack": "PGD-BCE",
            "class_or_union": t,
        }
        for t in ("union", "TZ+CZ", "PZ")
    ]
    rows.append(
        {
            "dataset": "zones",
            "case_id": "b",
            "epsilon_n": 2,
            "attack": "PGD-BCE",
            "class_or_union": "union",
        }
    )
    pd.DataFrame(rows).to_csv(csv, index=False)
    assert completed_keys(csv) == {("zones", "a", 2, "PGD-BCE")}
    assert completed_keys(tmp_path / "missing.csv") == set()


def test_mask_path_layout(tmp_path) -> None:
    assert (
        mask_path(tmp_path, "wg", "c1")
        == tmp_path / "masks" / "wg" / "c1" / "clean.npz"
    )
    assert (
        mask_path(tmp_path, "wg", "c1", "APGD-BCE", 16)
        == tmp_path / "masks" / "wg" / "c1" / "APGD-BCE_eps16.npz"
    )


def test_direction_rows_cover_the_three_tables() -> None:
    gt, clean, adv = _zones_case()
    valid = gt >= 0
    geometries = {
        name: target_geometry(g, c, SPACING)
        for name, g, c, _ in targets_for("zones", gt, clean, adv)
    }
    summary, profile, transitions = direction_rows(
        "zones",
        "case",
        16,
        "APGD-BCE",
        7,
        SPACING,
        gt,
        clean,
        adv,
        valid=valid,
        tolerance_mm=0.5,
        geometries=geometries,
    )
    assert [r["class_or_union"] for r in summary] == ["union", "TZ+CZ", "PZ"]
    assert all(set(r) == set(SUMMARY_FIELDNAMES) for r in summary)
    assert len(profile) == 3 * 8 and all(
        set(r) == set(PROFILE_FIELDNAMES) for r in profile
    )
    assert len(transitions) == 36 and all(
        set(r) == set(TRANSITION_FIELDNAMES) for r in transitions
    )
    union = summary[0]
    assert union["volume_change_mm3"] == pytest.approx(
        0.0
    )  # PZ ate TZ; the union is unchanged
    tz = summary[1]
    assert tz["change_bias"] == pytest.approx(-1.0)
    assert all(r["attack_success"] == union["attack_success"] for r in summary)
    harm = next(
        r
        for r in transitions
        if r["transition_type"] == "new_harm"
        and r["source_class"] == "TZ+CZ"
        and r["target_class"] == "PZ"
    )
    assert harm["voxel_count"] == 6 * 32 * 2


def test_direction_rows_marks_an_empty_target_invalid() -> None:
    gt, clean, adv = _zones_case()
    gt[gt == 2] = 1  # no PZ in the truth
    clean = np.where(gt < 0, 0, gt).astype(np.int16)
    valid = gt >= 0
    geometries = {}
    for name, g, c, _ in targets_for("zones", gt, clean, clean):
        geometries[name] = target_geometry(g, c, SPACING) if g.any() else None
    summary, profile, _ = direction_rows(
        "zones",
        "case",
        2,
        "FGSM-BCE",
        -1,
        SPACING,
        gt,
        clean,
        clean,
        valid=valid,
        tolerance_mm=0.5,
        geometries=geometries,
    )
    pz = next(r for r in summary if r["class_or_union"] == "PZ")
    assert pz["target_valid"] == 0 and np.isnan(pz["damage_bias"])
    assert not any(r["class_or_union"] == "PZ" for r in profile)
