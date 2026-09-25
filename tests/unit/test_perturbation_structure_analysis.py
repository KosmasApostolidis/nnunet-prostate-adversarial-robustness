"""Tests for perturbation structure analysis helpers."""

from __future__ import annotations

import pandas as pd
import pytest

from experiments.plot_perturbation_structure import (
    has_signed_bands,
    join_quality,
    paired_vs_control,
    require_full_spectra_coverage,
    require_reference_epsilon,
    signed_band_contrasts,
    summarize,
    write_report,
)


def _structure() -> pd.DataFrame:
    rows = []
    for case_id, apgd, gauss in (
        ("a", 3.0, 1.0),
        ("b", 5.0, 1.0),
        ("c", 0.5, 1.0),
    ):
        rows.append(
            {
                "dataset": "wg",
                "case_id": case_id,
                "epsilon_n": 16,
                "condition": "APGD-BCE",
                "foreground_energy_enrichment": apgd,
            }
        )
        rows.append(
            {
                "dataset": "wg",
                "case_id": case_id,
                "epsilon_n": 16,
                "condition": "gaussian_rms_matched",
                "foreground_energy_enrichment": gauss,
            }
        )
    return pd.DataFrame(rows)


def test_reference_epsilon_must_be_present_in_every_table_it_is_read_from() -> None:
    """The spectra table is generated separately from scalars, so it can lag.

    When it lacks the reference epsilon, ``plot_radial_spectra`` selects nothing
    and matplotlib dies on ``subplots(1, 0)`` — a stack trace that names neither
    the epsilon nor the table.
    """

    frame = pd.DataFrame({"epsilon_n": [2, 16]})

    require_reference_epsilon(frame, 16, label="spectra data")
    with pytest.raises(ValueError, match="32 is absent from spectra data"):
        require_reference_epsilon(frame, 32, label="spectra data")


def test_full_spectra_coverage_requires_every_case_condition_and_bin() -> None:
    structure = _structure()
    spectra = pd.DataFrame(
        [
            {
                "dataset": row.dataset,
                "case_id": row.case_id,
                "epsilon_n": row.epsilon_n,
                "condition": row.condition,
                "freq_bin": freq_bin,
                "f_r": 0.25 + 0.5 * freq_bin,
                "power": 1.0,
            }
            for row in structure.itertuples(index=False)
            for freq_bin in (0, 1)
        ]
    )

    require_full_spectra_coverage(structure, spectra, reference_epsilon=16)

    subset = spectra[spectra["case_id"] != "c"]
    with pytest.raises(ValueError, match="2 missing"):
        require_full_spectra_coverage(structure, subset, reference_epsilon=16)

    duplicate = pd.concat([spectra, spectra.iloc[[0]]], ignore_index=True)
    with pytest.raises(ValueError, match="1 duplicate"):
        require_full_spectra_coverage(structure, duplicate, reference_epsilon=16)


def test_join_quality_attaches_psnr_ssim_to_attack_rows_only() -> None:
    quality = pd.DataFrame(
        [
            {
                "dataset": "wg",
                "case_id": "a",
                "epsilon_n": 16,
                "attack": "APGD-BCE",
                "psnr_db_robust_range": 37.2,
                "ssim_axial_prostate": 0.97,
                "epsilon_saturation_fraction": 0.83,
            }
        ]
    )
    joined = join_quality(_structure(), quality)

    attack = joined[(joined.case_id == "a") & (joined.condition == "APGD-BCE")].iloc[0]
    assert attack["psnr_db_robust_range"] == pytest.approx(37.2)
    assert attack["ssim_axial_prostate"] == pytest.approx(0.97)

    control = joined[joined.condition == "gaussian_rms_matched"].iloc[0]
    assert pd.isna(control["psnr_db_robust_range"])
    assert len(joined) == len(_structure())


def test_join_quality_carries_saturation_for_family_c() -> None:
    """Family C is blind on saturated rows, so every attack row must carry it.

    Both alignment descriptors are functions of |δ|; at 100% saturation |δ| is
    the constant ε, ``spearmanr`` returns NaN and ``edge_energy_enrichment`` is
    exactly 1.0 — indistinguishable from the uniform-noise reference. Without
    this column the report cannot separate "unstructured" from "unmeasurable".
    """
    quality = pd.DataFrame(
        [
            {
                "dataset": "wg",
                "case_id": "a",
                "epsilon_n": 16,
                "attack": "APGD-BCE",
                "psnr_db_robust_range": 37.2,
                "ssim_axial_prostate": 0.97,
                "epsilon_saturation_fraction": 0.83,
            }
        ]
    )
    joined = join_quality(_structure(), quality)

    attack = joined[(joined.case_id == "a") & (joined.condition == "APGD-BCE")].iloc[0]
    assert attack["epsilon_saturation_fraction"] == pytest.approx(0.83)
    control = joined[joined.condition == "gaussian_rms_matched"].iloc[0]
    assert pd.isna(control["epsilon_saturation_fraction"])


def test_join_quality_rejects_a_quality_table_without_saturation() -> None:
    quality = pd.DataFrame(
        [
            {
                "dataset": "wg",
                "case_id": "a",
                "epsilon_n": 16,
                "attack": "APGD-BCE",
                "psnr_db_robust_range": 37.2,
                "ssim_axial_prostate": 0.97,
            }
        ]
    )
    with pytest.raises(ValueError, match="epsilon_saturation_fraction"):
        join_quality(_structure(), quality)


def test_summarize_reports_quantiles_per_condition() -> None:
    out = summarize(_structure(), ["foreground_energy_enrichment"])
    apgd = out[out.condition == "APGD-BCE"].iloc[0]
    assert apgd["n_cases"] == 3
    assert apgd["foreground_energy_enrichment_median"] == pytest.approx(3.0)
    assert apgd["foreground_energy_enrichment_mean"] == pytest.approx(8.5 / 3)


def test_paired_vs_control_pairs_within_case() -> None:
    out = paired_vs_control(_structure(), "foreground_energy_enrichment")
    row = out[out.condition == "APGD-BCE"].iloc[0]
    assert row["n_pairs"] == 3
    assert row["median_difference"] == pytest.approx(2.0)
    assert row["fraction_above_control"] == pytest.approx(2 / 3)


def test_paired_vs_control_excludes_the_control_from_its_own_comparison() -> None:
    out = paired_vs_control(_structure(), "foreground_energy_enrichment")
    assert "gaussian_rms_matched" not in set(out["condition"])


def test_report_tables_label_controls_and_expose_family_c_saturation(tmp_path) -> None:
    """The analytic-reference caveat and the saturation caveat must be in-table.

    Raw condition strings in the markdown tables drop "(analytic reference)",
    which is the framing that stops a Gaussian enrichment of 1.0 being read as a
    finding; the saturation column is what stops a Family C value on the
    reference line being read as "unstructured" rather than "unmeasurable".
    """
    structure = _structure()
    summary = summarize(structure, ["foreground_energy_enrichment"])
    summary["spearman_absdelta_gradmag_median"] = 0.4
    summary["edge_energy_enrichment_median"] = 1.0
    saturation = pd.DataFrame(
        [
            {
                "dataset": "wg",
                "condition": "APGD-BCE",
                "epsilon_n": 16,
                "epsilon_saturation_fraction_median": 0.999,
            }
        ]
    )

    path = write_report(
        structure,
        summary,
        pd.DataFrame(),
        pd.DataFrame(),
        saturation,
        tmp_path,
        reference_epsilon=16,
    )
    text = path.read_text(encoding="utf-8")

    assert "Gaussian (analytic reference)" in text
    assert "| gaussian_rms_matched |" not in text
    assert "epsilon_saturation_fraction_median" in text
    assert "0.999" in text


def test_random_trials_are_averaged_before_pairing() -> None:
    frame = pd.DataFrame(
        [
            {
                "dataset": "wg",
                "case_id": "a",
                "epsilon_n": 16,
                "condition": "APGD-BCE",
                "metric": 3.0,
            },
            {
                "dataset": "wg",
                "case_id": "a",
                "epsilon_n": 16,
                "condition": "gaussian_rms_matched",
                "metric": 1.0,
            },
            {
                "dataset": "wg",
                "case_id": "a",
                "epsilon_n": 16,
                "condition": "gaussian_rms_matched",
                "metric": 3.0,
            },
        ]
    )

    row = paired_vs_control(frame, "metric").iloc[0]
    assert row["n_pairs"] == 1
    assert row["median_difference"] == pytest.approx(1.0)


def _signed_frame(boundary: float, deep: float, exterior: float) -> pd.DataFrame:
    """One condition whose signed profile is fixed by three band values."""

    rows = []
    for case_id in ("a", "b", "c"):
        row = {
            "dataset": "wg",
            "case_id": case_id,
            "epsilon_n": 16,
            "condition": "APGD-BCE",
            "enrichment_band_inside_0_2mm": boundary,
            "enrichment_band_inside_2_5mm": 1.0,
            "enrichment_band_inside_5_10mm": 1.0,
            "enrichment_band_inside_beyond_10mm": deep,
            "enrichment_band_0_2mm": exterior,
            "enrichment_band_2_5mm": 1.0,
            "enrichment_band_5_10mm": 1.0,
            "enrichment_band_beyond_10mm": 1.0,
        }
        rows.append(row)
    return pd.DataFrame(rows)


def test_has_signed_bands_rejects_a_table_that_predates_the_interior_split() -> None:
    assert has_signed_bands(_signed_frame(1.06, 1.01, 1.06))
    assert not has_signed_bands(
        _signed_frame(1.06, 1.01, 1.06).drop(
            columns=["enrichment_band_inside_beyond_10mm"]
        )
    )


def test_contrasts_separate_boundary_hugging_from_deep_targeting() -> None:
    """The two profiles the pooled `inside` band reports identically."""

    hugging = signed_band_contrasts(_signed_frame(1.10, 1.00, 1.10))
    deep = signed_band_contrasts(_signed_frame(1.00, 1.10, 1.00))

    def value(frame: pd.DataFrame, contrast: str) -> float:
        return float(frame.loc[frame["contrast"] == contrast, "mean"].iloc[0])

    assert value(hugging, "boundary_vs_deep_interior") == pytest.approx(0.10)
    assert value(deep, "boundary_vs_deep_interior") == pytest.approx(-0.10)
    # Both are symmetric about the mask edge, which is a separate question.
    assert value(hugging, "interior_vs_exterior_shell") == pytest.approx(0.0)
    assert value(deep, "deep_interior") == pytest.approx(0.10)


def test_contrast_ci_brackets_the_mean_and_reports_the_case_count() -> None:
    frame = _signed_frame(1.10, 1.00, 1.04)
    frame.loc[0, "enrichment_band_inside_0_2mm"] = 1.20
    row = signed_band_contrasts(frame)
    row = row[row["contrast"] == "boundary_vs_deep_interior"].iloc[0]
    assert row["n_cases"] == 3
    assert row["ci_low"] <= row["mean"] <= row["ci_high"]
    assert row["ci_low"] < row["ci_high"]
