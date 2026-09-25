from __future__ import annotations

import numpy as np
import pandas as pd

from experiments.analyze_attack_quality_damage import (
    ATTACKS,
    _add_damage_characterization_columns,
    _strict_macro_boundary_rows,
    dense_paired_front_with_uncertainty,
    failure_thresholds,
    interpolate_rms_matched,
    mark_pareto_points,
    measured_summary_with_clean,
)


def _synthetic_paired() -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    for attack_index, attack in enumerate(ATTACKS):
        for epsilon_n, rms_percent, dice_drop, ssim in (
            (2, 1.0, 0.05 + attack_index * 0.01, 0.99),
            (4, 2.0, 0.15 + attack_index * 0.01, 0.97),
        ):
            rows.append(
                {
                    "dataset": "wg",
                    "case_id": "case",
                    "attack": attack,
                    "epsilon_n": epsilon_n,
                    "epsilon_norm": epsilon_n / 255,
                    "rms_norm": rms_percent / 100,
                    "rms_percent_case_sigma": rms_percent,
                    "robust_signal_range_norm": 4.0,
                    "psnr_db_robust_range": 40.0,
                    "ssim_axial_prostate": ssim,
                    "clean_dice": 0.9,
                    "adversarial_dice": 0.9 - dice_drop,
                    "dice_drop": dice_drop,
                    "clean_iou": 0.82,
                    "adversarial_iou": 0.82 - dice_drop,
                    "iou_drop": dice_drop,
                    "clean_hd95_mm": 2.0,
                    "adversarial_hd95_mm": 2.0 + 10 * dice_drop,
                    "hd95_increase_mm": 10 * dice_drop,
                    "clean_asd_mm": 0.5,
                    "adversarial_asd_mm": 0.5 + 2 * dice_drop,
                    "asd_increase_mm": 2 * dice_drop,
                    "clean_volume_error_percent": 0.0,
                    "adversarial_volume_error_percent": -10 * dice_drop,
                    "target_foreground_voxels": 1000,
                    "clean_prediction_foreground_voxels": 1000,
                    "adversarial_prediction_foreground_voxels": int(
                        1000 * (1.0 - dice_drop)
                    ),
                }
            )
    return _add_damage_characterization_columns(pd.DataFrame(rows))


def test_rms_matching_is_paired_and_interpolates_patient_trajectories() -> None:
    available, matched = interpolate_rms_matched(_synthetic_paired(), [0.0, 1.5])

    assert len(available) == 6
    assert len(matched) == 6
    assert matched["paired_complete"].all()
    fgsm = matched[
        (matched["attack"] == "FGSM-BCE")
        & (matched["target_rms_percent_case_sigma"] == 1.5)
    ].iloc[0]
    assert np.isclose(fgsm["epsilon_n"], 3.0)
    assert np.isclose(fgsm["dice_drop"], 0.10)
    assert np.isclose(fgsm["ssim_axial_prostate"], 0.98)
    assert np.isclose(fgsm["rms_norm"], 0.015)


def test_failure_threshold_interpolates_first_crossing() -> None:
    result = failure_thresholds(_synthetic_paired(), [0.10])

    fgsm = result[result["attack"] == "FGSM-BCE"].iloc[0]
    assert bool(fgsm["attained"])
    assert np.isclose(fgsm["estimated_epsilon_n"], 3.0)
    assert np.isclose(fgsm["rms_percent_case_sigma"], 1.5)
    assert bool(fgsm["paired_attained_all_attacks"])


def test_pareto_marking_rejects_more_distortion_with_less_damage() -> None:
    summary = pd.DataFrame(
        [
            {
                "dataset": "wg",
                "attack": "FGSM-BCE",
                "epsilon_n": 2,
                "rms_percent_case_sigma_median": 1.0,
                "dice_drop_median": 0.10,
            },
            {
                "dataset": "wg",
                "attack": "PGD-BCE",
                "epsilon_n": 2,
                "rms_percent_case_sigma_median": 2.0,
                "dice_drop_median": 0.08,
            },
            {
                "dataset": "wg",
                "attack": "APGD-BCE",
                "epsilon_n": 2,
                "rms_percent_case_sigma_median": 2.0,
                "dice_drop_median": 0.20,
            },
        ]
    )

    marked = mark_pareto_points(summary)

    assert bool(marked.iloc[0]["pareto_efficient"])
    assert not bool(marked.iloc[1]["pareto_efficient"])
    assert bool(marked.iloc[2]["pareto_efficient"])


def test_measured_summary_allows_uncomputed_boundary_metrics() -> None:
    paired = _synthetic_paired()
    paired[["hd95_increase_mm", "asd_increase_mm"]] = np.nan

    summary = measured_summary_with_clean(paired)

    attacked = summary[summary["epsilon_n"] > 0]
    assert attacked["hd95_increase_mm_median"].isna().all()
    assert attacked["asd_increase_mm_median"].isna().all()
    assert np.isfinite(attacked["dice_drop_median"]).all()


def test_damage_characterization_adds_absolute_volume_worsening() -> None:
    paired = _synthetic_paired()

    assert np.allclose(
        paired["absolute_volume_error_increase_percent"],
        paired["adversarial_volume_error_percent"].abs(),
    )
    assert not paired["adversarial_empty_prediction"].any()
    assert paired["boundary_metrics_defined"].all()


def test_macro_boundary_metrics_require_every_foreground_class() -> None:
    rows = []
    for class_name, adversarial_hd95 in (
        ("TZ+CZ", 4.0),
        ("PZ", float("nan")),
        ("macro", 4.0),
    ):
        rows.append(
            {
                "dataset": "zones",
                "case_id": "case",
                "attack": "APGD-BCE",
                "epsilon_n": 8,
                "class": class_name,
                "clean_hd95_mm": 2.0,
                "adversarial_hd95_mm": adversarial_hd95,
                "hd95_increase_mm": adversarial_hd95 - 2.0,
                "clean_asd_mm": 0.5,
                "adversarial_asd_mm": (
                    adversarial_hd95 / 4.0 if np.isfinite(adversarial_hd95) else np.nan
                ),
                "asd_increase_mm": (
                    adversarial_hd95 / 4.0 - 0.5
                    if np.isfinite(adversarial_hd95)
                    else np.nan
                ),
            }
        )

    strict = _strict_macro_boundary_rows(pd.DataFrame(rows)).iloc[0]

    assert np.isclose(strict["clean_hd95_mm"], 2.0)
    assert np.isnan(strict["adversarial_hd95_mm"])
    assert np.isnan(strict["hd95_increase_mm"])
    assert np.isnan(strict["asd_increase_mm"])


def test_dense_front_uses_one_fixed_paired_cohort_and_bootstrap_cis() -> None:
    case_one = _synthetic_paired()
    case_two = _synthetic_paired().copy()
    case_two["case_id"] = "case_two"
    case_two["dice_drop"] += 0.02
    case_two["adversarial_dice"] -= 0.02
    case_two["iou_drop"] += 0.02
    case_two["adversarial_iou"] -= 0.02
    case_three = _synthetic_paired().copy()
    case_three["case_id"] = "incomplete_case"
    case_three = case_three[
        ~((case_three["attack"] == "APGD-BCE") & (case_three["epsilon_n"] == 4))
    ]
    paired = pd.concat([case_one, case_two, case_three], ignore_index=True)
    targets = [0.0, 0.5, 1.0, 1.5, 2.0]
    _available, matched = interpolate_rms_matched(paired, targets)

    common, summary, differences = dense_paired_front_with_uncertainty(
        matched,
        seed=7,
        n_bootstrap=100,
    )

    assert set(common["case_id"]) == {"case", "case_two"}
    assert len(common) == 2 * len(ATTACKS) * len(targets)
    assert summary["n_cases"].eq(2).all()
    assert summary["pareto_selection_probability"].between(0.0, 1.0).all()
    assert summary["dice_drop_ci95_low"].le(summary["dice_drop_median"]).all()
    assert summary["dice_drop_ci95_high"].ge(summary["dice_drop_median"]).all()
    assert differences["n_cases"].eq(2).all()


def test_dense_front_preserves_anatomical_class_strata() -> None:
    first = _synthetic_paired()
    first["class"] = "TZ+CZ"
    second = _synthetic_paired().copy()
    second["class"] = "PZ"
    second["dice_drop"] += 0.10
    second["adversarial_dice"] -= 0.10
    paired = pd.concat([first, second], ignore_index=True)
    targets = [0.0, 1.0, 2.0]
    _available, matched = interpolate_rms_matched(
        paired,
        targets,
        strata=("class",),
    )

    _common, summary, _differences = dense_paired_front_with_uncertainty(
        matched,
        seed=11,
        n_bootstrap=20,
        strata=("class",),
    )

    assert set(summary["class"]) == {"TZ+CZ", "PZ"}
    pz = summary[
        (summary["class"] == "PZ")
        & (summary["attack"] == "FGSM-BCE")
        & (summary["target_rms_percent_case_sigma"] == 2.0)
    ].iloc[0]
    tz = summary[
        (summary["class"] == "TZ+CZ")
        & (summary["attack"] == "FGSM-BCE")
        & (summary["target_rms_percent_case_sigma"] == 2.0)
    ].iloc[0]
    assert np.isclose(pz["dice_drop_median"] - tz["dice_drop_median"], 0.10)
