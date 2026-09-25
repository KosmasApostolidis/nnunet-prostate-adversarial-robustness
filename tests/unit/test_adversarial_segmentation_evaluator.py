from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from experiments.evaluate_adversarial_segmentation_all_cases import (
    FIELDNAMES,
    _class_metrics,
    _class_metrics_batch,
    _completed_keys,
    _seed_completed_rows,
    _segmentation_rows,
)


def test_completed_keys_requires_every_expected_class(tmp_path) -> None:
    output = tmp_path / "segmentation.csv"
    pd.DataFrame(
        [
            {
                "dataset": "wg",
                "case_id": "complete",
                "epsilon_n": 2,
                "attack": "FGSM-BCE",
                "class": "WG",
            },
            {
                "dataset": "wg",
                "case_id": "complete",
                "epsilon_n": 2,
                "attack": "FGSM-BCE",
                "class": "macro",
            },
            {
                "dataset": "zones",
                "case_id": "partial",
                "epsilon_n": 2,
                "attack": "PGD-BCE",
                "class": "TZ+CZ",
            },
            {
                "dataset": "zones",
                "case_id": "partial",
                "epsilon_n": 2,
                "attack": "PGD-BCE",
                "class": "macro",
            },
        ]
    ).to_csv(output, index=False)

    assert _completed_keys(output) == {("wg", "complete", 2, "FGSM-BCE")}


def test_seed_completed_rows_reuses_only_complete_requested_keys(tmp_path) -> None:
    seed_csv = tmp_path / "seed.csv"
    output_csv = tmp_path / "output.csv"

    def row(case_id: str, class_name: str) -> dict[str, object]:
        values = dict.fromkeys(FIELDNAMES, np.nan)
        values.update(
            {
                "dataset": "wg",
                "case_id": case_id,
                "epsilon_n": 2,
                "attack": "PGD-BCE",
                "class": class_name,
                "dice_drop": 0.25,
            }
        )
        return values

    pd.DataFrame(
        [row("complete", "WG"), row("complete", "macro"), row("partial", "WG")]
    ).to_csv(seed_csv, index=False)
    requested = {
        ("wg", "complete", 2, "PGD-BCE"),
        ("wg", "partial", 2, "PGD-BCE"),
    }

    assert _seed_completed_rows(output_csv, seed_csv, requested) == 1
    assert _completed_keys(output_csv) == {("wg", "complete", 2, "PGD-BCE")}
    assert len(pd.read_csv(output_csv)) == 2
    assert _seed_completed_rows(output_csv, seed_csv, requested) == 0

    conflicting = pd.read_csv(seed_csv)
    conflicting.loc[conflicting["class"] == "WG", "dice_drop"] = 0.5
    conflicting.to_csv(seed_csv, index=False)
    with pytest.raises(ValueError, match="conflicts.*dice_drop"):
        _seed_completed_rows(output_csv, seed_csv, requested)


def test_class_metrics_includes_foreground_classes_and_macro() -> None:
    target = np.array([[[0, 1], [2, 2]]])
    prediction = np.array([[[0, 1], [0, 2]]])

    rows = _class_metrics(
        prediction,
        target,
        (1.0, 1.0, 1.0),
        ["TZ+CZ", "PZ"],
        boundary_metrics=False,
    )

    assert [row["class"] for row in rows] == ["TZ+CZ", "PZ", "macro"]
    assert rows[0]["dice"] == 1.0
    assert np.isclose(rows[1]["dice"], 2.0 / 3.0)
    assert np.isclose(rows[2]["dice"], 5.0 / 6.0)
    assert np.isnan(rows[2]["hd95_mm"])


def test_parallel_class_metrics_match_sequential_results() -> None:
    target = np.zeros((3, 8, 8), dtype=np.uint8)
    target[:, 2:6, 2:6] = 1
    predictions = [target.copy(), np.roll(target, shift=1, axis=2)]
    targets = [target, target]
    spacings = [(3.0, 0.5, 0.5), (3.0, 0.5, 0.5)]

    sequential = _class_metrics_batch(
        predictions,
        targets,
        spacings,
        ["WG"],
        boundary_metrics=True,
        workers=1,
    )
    parallel = _class_metrics_batch(
        predictions,
        targets,
        spacings,
        ["WG"],
        boundary_metrics=True,
        workers=2,
    )

    assert parallel == sequential


def test_segmentation_rows_use_positive_damage_direction() -> None:
    clean = [
        {
            "class": "WG",
            "class_label": 1,
            "dice": 0.9,
            "iou": 0.82,
            "hd95_mm": 2.0,
            "asd_mm": 0.5,
            "target_foreground_voxels": 100,
            "prediction_foreground_voxels": 105,
        },
        {
            "class": "macro",
            "class_label": 0,
            "dice": 0.9,
            "iou": 0.82,
            "hd95_mm": 2.0,
            "asd_mm": 0.5,
            "target_foreground_voxels": 100,
            "prediction_foreground_voxels": 105,
        },
    ]
    adversarial = [
        {
            "class": "WG",
            "class_label": 1,
            "dice": 0.7,
            "iou": 0.55,
            "hd95_mm": 4.0,
            "asd_mm": 1.5,
            "target_foreground_voxels": 100,
            "prediction_foreground_voxels": 80,
        },
        {
            "class": "macro",
            "class_label": 0,
            "dice": 0.7,
            "iou": 0.55,
            "hd95_mm": 4.0,
            "asd_mm": 1.5,
            "target_foreground_voxels": 100,
            "prediction_foreground_voxels": 80,
        },
    ]

    rows = _segmentation_rows(
        dataset_key="wg",
        case_id="case",
        epsilon_n=8,
        attack="APGD-BCE",
        attack_steps=20,
        attack_precision="fp32",
        attack_seed=7,
        spacing=(3.0, 1.0, 1.0),
        linf_norm=8 / 255,
        rms_norm=0.01,
        quality_metrics={
            "linf_norm": 8 / 255,
            "rms_norm": 0.01,
            "rms_percent_case_sigma": 1.0,
            "mae_norm": 0.008,
            "robust_signal_range_norm": 4.0,
            "psnr_db_robust_range": 52.0,
            "ssim_axial_prostate": 0.99,
            "epsilon_saturation_fraction": 0.2,
        },
        max_ssim_slices=5,
        clean_metrics=clean,
        adversarial_metrics=adversarial,
    )

    assert len(rows) == 2
    assert np.isclose(rows[0]["dice_drop"], 0.2)
    assert np.isclose(rows[0]["hd95_increase_mm"], 2.0)
    assert np.isclose(rows[0]["asd_increase_mm"], 1.0)
    assert np.isclose(rows[0]["clean_volume_error_percent"], 5.0)
    assert np.isclose(rows[0]["adversarial_volume_error_percent"], -20.0)
