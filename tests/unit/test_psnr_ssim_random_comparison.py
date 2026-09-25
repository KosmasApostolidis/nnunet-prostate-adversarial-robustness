"""Tests for paired APGD versus random-noise image-quality summaries."""

from __future__ import annotations

import pandas as pd
import pytest

from experiments.plot_adversarial_vs_random_psnr_ssim import (
    _case_average_controls,
    _paired_differences,
    _summarize_paired,
)


def _apgd_frame() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "dataset": ["wg", "wg"],
            "case_id": ["a", "b"],
            "epsilon_n": [16, 16],
            "epsilon_norm": [16 / 255, 16 / 255],
            "rms_norm": [0.05, 0.06],
            "psnr_db_robust_range": [38.0, 37.0],
            "ssim_axial_prostate": [0.90, 0.80],
        }
    )


def _random_trials() -> pd.DataFrame:
    rows = []
    for case_id, rms, psnr, ssim_values in (
        ("a", 0.05, 38.0, [0.94, 0.96]),
        ("b", 0.06, 37.0, [0.87, 0.89]),
    ):
        for trial, ssim in enumerate(ssim_values):
            rows.append(
                {
                    "dataset": "wg",
                    "case_id": case_id,
                    "epsilon_n": 16,
                    "epsilon_norm": 16 / 255,
                    "perturbation": "gaussian_rms_matched",
                    "trial": trial,
                    "rms_norm": rms,
                    "rms_percent_case_sigma": 100 * rms,
                    "psnr_db_robust_range": psnr,
                    "ssim_axial_prostate": ssim,
                    "rms_match_relative_error": 0.0,
                }
            )
    return pd.DataFrame(rows)


def test_random_trials_are_averaged_before_pairing() -> None:
    controls = _case_average_controls(_random_trials())
    assert len(controls) == 2
    assert controls.set_index("case_id").loc["a", "ssim_axial_prostate"] == pytest.approx(
        0.95
    )
    assert controls.set_index("case_id").loc["b", "ssim_axial_prostate"] == pytest.approx(
        0.88
    )


def test_paired_summary_reports_random_minus_apgd() -> None:
    controls = _case_average_controls(_random_trials())
    paired = _paired_differences(_apgd_frame(), controls)
    assert paired.set_index("case_id").loc["a", "ssim_random_minus_apgd"] == pytest.approx(
        0.05
    )
    assert paired.set_index("case_id").loc["b", "ssim_random_minus_apgd"] == pytest.approx(
        0.08
    )
    assert paired["psnr_random_minus_apgd"].abs().max() == 0.0

    summary = _summarize_paired(paired).iloc[0]
    assert summary["ssim_random_minus_apgd_median"] == pytest.approx(0.065)
    assert summary["fraction_random_ssim_higher"] == 1.0
    assert summary["n_cases"] == 2
