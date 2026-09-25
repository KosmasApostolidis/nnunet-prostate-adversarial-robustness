"""Unit tests for normalized-space epsilon calibration helpers."""

from __future__ import annotations

import numpy as np
import pytest

from mri_prostate_seg.experiments.epsilon_calibration import (
    matched_gaussian_noise,
    matched_rician_proxy_noise,
    normalized_epsilon_mapping,
    perturbation_quality,
    stable_seed,
)


def test_normalized_epsilon_mapping_uses_case_sigma_units() -> None:
    row = normalized_epsilon_mapping(16, cohort_foreground_std_au=400.0)
    assert row["epsilon_norm"] == pytest.approx(16 / 255)
    assert row["epsilon_percent_within_case_sigma"] == pytest.approx(100 * 16 / 255)
    assert row["cohort_raw_linf_proxy_au"] == pytest.approx(400 * 16 / 255)
    assert "epsilon_norm=0.06275" in str(row["recommended_label"])
    assert "legacy 16/255" in str(row["recommended_label"])


def test_stable_seed_is_repeatable_and_keyed() -> None:
    assert stable_seed("wg", "case-1", 16, base_seed=7) == stable_seed(
        "wg", "case-1", 16, base_seed=7
    )
    assert stable_seed("wg", "case-1", 16, base_seed=7) != stable_seed(
        "wg", "case-2", 16, base_seed=7
    )


def test_perturbation_quality_known_constant_delta() -> None:
    clean = np.linspace(-1.0, 1.0, 9 * 9 * 7, dtype=np.float32).reshape(7, 9, 9)
    altered = clean + 0.05
    metrics = perturbation_quality(
        clean,
        altered,
        epsilon=0.1,
        high_frequency_noise_sigma=0.02,
        max_ssim_slices=3,
    )
    assert metrics["linf_norm"] == pytest.approx(0.05, abs=1e-6)
    assert metrics["rms_norm"] == pytest.approx(0.05, abs=1e-6)
    assert metrics["mae_norm"] == pytest.approx(0.05, abs=1e-6)
    assert metrics["rms_over_hf_noise"] == pytest.approx(2.5, abs=1e-5)
    assert metrics["epsilon_saturation_fraction"] == 0.0
    assert np.isfinite(metrics["psnr_db_robust_range"])
    assert np.isfinite(metrics["ssim_axial_prostate"])
    assert -1.0 <= metrics["ssim_axial_prostate"] <= 1.0


@pytest.mark.parametrize("kind", ["gaussian", "rician"])
def test_matched_noise_respects_budget_and_matches_rms(kind: str) -> None:
    rng = np.random.default_rng(123)
    clean = np.linspace(-1.5, 2.5, 13 * 15 * 9, dtype=np.float32).reshape(9, 13, 15)
    mask = np.ones_like(clean, dtype=bool)
    epsilon = 0.08
    target_rms = 0.025

    if kind == "gaussian":
        altered = matched_gaussian_noise(
            clean,
            valid_mask=mask,
            epsilon=epsilon,
            target_rms=target_rms,
            rng=rng,
        )
    else:
        altered = matched_rician_proxy_noise(
            clean,
            valid_mask=mask,
            epsilon=epsilon,
            target_rms=target_rms,
            rng=rng,
            raw_zero_normalized=-0.4,
        )

    delta = altered - clean
    assert np.max(np.abs(delta)) <= epsilon + 1e-6
    assert float(np.sqrt(np.mean(delta**2))) == pytest.approx(target_rms, rel=2e-3)


def test_noise_leaves_invalid_support_unchanged() -> None:
    rng = np.random.default_rng(22)
    clean = np.zeros((7, 9, 9), dtype=np.float32)
    clean[:, 2:7, 2:7] = 1.0
    mask = np.zeros_like(clean, dtype=bool)
    mask[:, 2:7, 2:7] = True
    altered = matched_gaussian_noise(
        clean,
        valid_mask=mask,
        epsilon=0.1,
        target_rms=0.03,
        rng=rng,
    )
    np.testing.assert_array_equal(altered[~mask], clean[~mask])


def test_matched_noise_rejects_nonpositive_match_iterations() -> None:
    clean = np.zeros((7, 9, 9), dtype=np.float32)
    mask = np.ones_like(clean, dtype=bool)
    with pytest.raises(ValueError, match="iterations must be at least 1"):
        matched_gaussian_noise(
            clean,
            valid_mask=mask,
            epsilon=0.1,
            target_rms=0.03,
            rng=np.random.default_rng(5),
            match_iterations=0,
        )
