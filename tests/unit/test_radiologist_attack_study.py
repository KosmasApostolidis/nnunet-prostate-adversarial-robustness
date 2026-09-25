from __future__ import annotations

import numpy as np
import pandas as pd

from experiments.analyze_radiologist_attack_study import (
    load_adequacy_scores,
    load_visibility_scores,
    summarize_adequacy,
    summarize_visibility,
)
from experiments.prepare_radiologist_attack_study import (
    ATTACKS,
    _rescale_delta_to_target_rms,
    build_study_manifests,
    select_conditions,
)


def _matched_frame() -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    for dataset in ("wg", "zones"):
        for target in (1.0, 4.0):
            for case_index in range(6):
                case_id = f"{dataset}_target{target:g}_case{case_index}"
                for attack_index, attack in enumerate(ATTACKS):
                    rows.append(
                        {
                            "dataset": dataset,
                            "case_id": case_id,
                            "attack": attack,
                            "target_rms_percent_case_sigma": target,
                            "epsilon_n": target * (2 + attack_index),
                            "rms_norm": target / 100,
                            "ssim_axial_prostate": 1 - target / 100,
                            "dice_drop": (case_index / 100 + attack_index / 1000),
                            "paired_complete": True,
                        }
                    )
    return pd.DataFrame(rows)


def test_reader_manifests_keep_truth_out_of_public_tables() -> None:
    selected = select_conditions(
        _matched_frame(),
        rms_targets=[1.0, 4.0],
        cases_per_dataset_target=2,
    )
    manifests = build_study_manifests(
        selected,
        reader_blocks=3,
        seed=7,
        repeat_fraction=0.25,
    )

    assert len(selected) == 2 * 2 * 2 * 3
    assert {"case_id", "attack", "altered_side"}.isdisjoint(
        manifests["visibility_public"].columns
    )
    assert "is_reliability_repeat" not in manifests["visibility_public"].columns
    assert {"case_id", "attack", "altered_side"}.issubset(
        manifests["visibility_truth"].columns
    )
    assert manifests["visibility_public"]["pair_id"].is_unique
    assert set(manifests["visibility_public"]["reader_block"]) <= {1, 2, 3}
    assert manifests["visibility_truth"]["is_reliability_repeat"].astype(bool).any()


def test_residual_rescaling_matches_target_rms() -> None:
    clean = np.zeros((2, 3, 4), dtype=np.float32)
    altered = np.ones_like(clean) * 0.2
    valid = np.ones_like(clean, dtype=bool)

    calibrated, rms = _rescale_delta_to_target_rms(
        clean,
        altered,
        valid_mask=valid,
        target_rms=0.05,
        lower=-1.0,
        upper=1.0,
    )

    assert np.isclose(rms, 0.05, atol=1e-8)
    assert np.isclose(np.sqrt(np.mean(calibrated**2)), 0.05, atol=1e-7)


def test_reader_score_analysis_joins_private_truth(tmp_path) -> None:
    visibility_truth = pd.DataFrame(
        [
            {
                "pair_id": "p1",
                "altered_side": "B",
                "dataset": "wg",
                "case_id": "case",
                "attack": "FGSM-BCE",
                "target_rms_percent_case_sigma": 1.0,
                "is_reliability_repeat": False,
                "source_pair_id": "",
            }
        ]
    )
    visibility_scores = pd.DataFrame(
        [
            {
                "reader_id": "R1",
                "pair_id": "p1",
                "selected_altered_image": "B",
                "difference_visible": "yes",
                "confidence_1_to_5": 4,
            }
        ]
    )
    adequacy_truth = pd.DataFrame(
        [
            {
                "adequacy_item_id": "a1",
                "dataset": "wg",
                "case_id": "case",
                "condition": "attacked",
                "attack": "FGSM-BCE",
                "target_rms_percent_case_sigma": 1.0,
            }
        ]
    )
    adequacy_scores = pd.DataFrame(
        [
            {
                "reader_id": "R1",
                "adequacy_item_id": "a1",
                "capsule_visibility_1_to_5": 4,
                "zonal_anatomy_visibility_1_to_5": "NA",
                "adequate_for_segmentation_correction_yes_no": "yes",
                "overall_quality_1_to_3": 2,
                "confidence_1_to_5": 4,
            }
        ]
    )
    visibility_truth_path = tmp_path / "visibility_truth.csv"
    visibility_scores_path = tmp_path / "visibility_scores.csv"
    adequacy_truth_path = tmp_path / "adequacy_truth.csv"
    adequacy_scores_path = tmp_path / "adequacy_scores.csv"
    visibility_truth.to_csv(visibility_truth_path, index=False)
    visibility_scores.to_csv(visibility_scores_path, index=False)
    adequacy_truth.to_csv(adequacy_truth_path, index=False)
    adequacy_scores.to_csv(adequacy_scores_path, index=False)

    visibility = load_visibility_scores([visibility_scores_path], visibility_truth_path)
    adequacy = load_adequacy_scores([adequacy_scores_path], adequacy_truth_path)
    visibility_summary = summarize_visibility(visibility)
    adequacy_summary = summarize_adequacy(adequacy)

    assert bool(visibility.iloc[0]["correct"])
    assert visibility_summary.iloc[0]["detection_accuracy"] == 1.0
    assert adequacy_summary.iloc[0]["adequate_fraction"] == 1.0
    assert np.isnan(adequacy_summary.iloc[0]["zonal_visibility_mean"])
