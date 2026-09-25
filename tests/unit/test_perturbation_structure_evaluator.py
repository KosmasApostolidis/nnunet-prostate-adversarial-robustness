"""Tests for the perturbation structure evaluator's pure helpers."""

from __future__ import annotations

import time
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import torch
from scipy.ndimage import gaussian_filter

from experiments import evaluate_perturbation_structure_all_cases as evaluator
from experiments.evaluate_perturbation_structure_all_cases import (
    CONDITIONS,
    build_row,
    check_reproduction,
    completed_keys,
    reproduction_error,
    spectra_case_subset,
)


SETTINGS = {
    "base_seed": 20260721,
    "attack_steps": 20,
    "spectral_bins": 32,
    "spectral_margin": 16,
    "spectral_min_size": 32,
    "alignment_voxels": 200_000,
    "max_gmsd_slices": 5,
    "spectra_cases": 50,
}


def test_condition_labels_match_the_existing_csv_vocabulary() -> None:
    assert CONDITIONS == {
        "fgsm": "FGSM-BCE",
        "pgd": "PGD-BCE",
        "apgd": "APGD-BCE",
        "gaussian": "gaussian_rms_matched",
        "rician": "rician_proxy_rms_matched",
    }


def test_completed_keys_round_trips_written_rows(tmp_path) -> None:
    path = tmp_path / "structure.csv"
    frame = pd.DataFrame(
        [
            {
                "dataset": "wg",
                "case_id": "a",
                "epsilon_n": 16,
                "condition": "APGD-BCE",
                **SETTINGS,
            },
            {
                "dataset": "wg",
                "case_id": "b",
                "epsilon_n": 8,
                "condition": "PGD-BCE",
                **SETTINGS,
            },
        ]
    )
    frame.to_csv(path, index=False)
    assert completed_keys(path, settings=SETTINGS) == {
        ("wg", "a", 16, "APGD-BCE"),
        ("wg", "b", 8, "PGD-BCE"),
    }


def test_completed_keys_is_empty_for_a_missing_file(tmp_path) -> None:
    assert completed_keys(tmp_path / "absent.csv", settings=SETTINGS) == set()


def test_completed_keys_refuses_to_resume_across_different_settings(tmp_path) -> None:
    path = tmp_path / "structure.csv"
    row = {
        "dataset": "wg",
        "case_id": "a",
        "epsilon_n": 16,
        "condition": "APGD-BCE",
        **SETTINGS,
    }
    row["spectral_bins"] = 64
    pd.DataFrame([row]).to_csv(path, index=False)
    with pytest.raises(ValueError, match="spectral_bins"):
        completed_keys(path, settings=SETTINGS)


def test_settings_record_every_parameter_that_changes_the_numbers() -> None:
    """The resume guard can only check what ``settings`` records.

    ``attack_steps`` is the load-bearing one: resuming with a different value
    would silently mix 20-step and n-step rows *and* invalidate the RMS
    reproduction audit, with no trace in the artifact. ``spectra_cases`` changes
    no scalar but selects which cases reach the spectra table, and completed
    cases are never revisited, so resuming under a different value would leave
    that table covering an inconsistent case set. Every settings key must also
    reach the CSV, or the guard would reject its own output on resume.
    """
    assert set(SETTINGS) == {
        "base_seed",
        "attack_steps",
        "spectral_bins",
        "spectral_margin",
        "spectral_min_size",
        "alignment_voxels",
        "max_gmsd_slices",
        "spectra_cases",
    }
    assert set(SETTINGS) <= set(evaluator.FIELDNAMES)


def test_completed_keys_treats_a_header_only_file_as_empty(tmp_path) -> None:
    """A crash between the header write and the first row must stay resumable.

    With no rows there is nothing for the settings guard to disagree with, but
    it read the absent values as ``[]``, refused the resume, and told the
    operator to choose a different output directory.
    """

    path = tmp_path / "structure.csv"
    path.write_text(",".join(evaluator.FIELDNAMES) + "\n", encoding="utf-8")

    assert completed_keys(path, settings=SETTINGS) == set()


def test_completed_keys_refuses_to_resume_across_different_attack_steps(
    tmp_path,
) -> None:
    path = tmp_path / "structure.csv"
    row = {
        "dataset": "wg",
        "case_id": "a",
        "epsilon_n": 16,
        "condition": "APGD-BCE",
        **SETTINGS,
    }
    row["attack_steps"] = 40
    pd.DataFrame([row]).to_csv(path, index=False)
    with pytest.raises(ValueError, match="attack_steps"):
        completed_keys(path, settings=SETTINGS)


def test_check_control_rms_match_raises_past_the_published_tolerance() -> None:
    """Controls never run ``check_reproduction``, so this is their only gate."""

    assert evaluator.CONTROL_RMS_TOLERANCE == 2e-3
    assert evaluator.check_control_rms_match(
        0.05, 0.05, label="wg/a/16/gaussian_rms_matched"
    ) == pytest.approx(0.0)
    # 1e-3 relative: inside the tolerance, as the measured 1e-7..4e-4 drift is.
    assert evaluator.check_control_rms_match(
        0.05005, 0.05, label="wg/a/16/gaussian_rms_matched"
    ) == pytest.approx(1e-3)
    with pytest.raises(RuntimeError, match="RMS match failed"):
        evaluator.check_control_rms_match(
            0.055, 0.05, label="wg/a/16/gaussian_rms_matched"
        )


def test_spectra_subset_is_deterministic_evenly_spaced_and_bounded() -> None:
    cases = [f"case_{i:03d}" for i in range(100)]
    subset = spectra_case_subset(cases, count=10)
    assert len(subset) == 10
    assert subset == spectra_case_subset(cases, count=10)
    assert subset[0] == "case_000"
    assert subset[-1] == "case_099"
    assert subset == sorted(set(subset))
    assert spectra_case_subset(cases[:5], count=10) == cases[:5]


def test_reproduction_error_is_relative() -> None:
    assert reproduction_error(0.0501, 0.05) == pytest.approx(0.002)
    assert reproduction_error(0.05, 0.05) == pytest.approx(0.0)


def test_check_reproduction_records_drift_and_raises_only_on_misconfiguration(
    recwarn,
) -> None:
    """Ordinary GPU drift must not abort a resumable 8-hour run.

    A failed row never enters ``completed``, so raising on it made the failure
    unrecoverable: every resume regenerated the same case and died in the same
    place. Sign-based attacks compounding over 20 steps drift far more than the
    old 1e-2 ceiling allowed — the production run measures APGD at 4.9e-2 max
    over its first 53 cases — so that ceiling sat well inside the expected
    distribution of ~15k iterative-attack rows. Only drift far beyond GPU
    nondeterminism — a wrong seed, step count, or precision — still raises.
    """

    assert check_reproduction(0.05, 0.05, label="wg/a/16/APGD-BCE") == pytest.approx(
        0.0
    )
    # 5% drift: the worst APGD row measured so far. Recorded, not warned, not fatal.
    assert check_reproduction(0.0525, 0.05, label="wg/a/16/APGD-BCE") == pytest.approx(
        0.05
    )
    assert len(recwarn) == 0

    with pytest.raises(RuntimeError, match="misconfigured"):
        check_reproduction(0.11, 0.05, label="wg/a/16/APGD-BCE")


def test_reproduction_ceilings_match_how_each_attack_actually_reproduces() -> None:
    """One ceiling cannot serve all three attacks.

    FGSM is a single deterministic step with no random start, so it reproduces
    to ~1e-5 and is the sharp fingerprint of the configuration: a wrong
    checkpoint, epsilon, loss, or precision moves it immediately. PGD and APGD
    take sign(gradient) over 20 steps from a random start; measured over the
    first 53 cases of the production run, APGD drift is median 1.5e-3 and max
    4.9e-2, and a tail fit projects ~6e-1 over the full 9,995 APGD rows. A
    shared 1e-1 ceiling would abort a healthy run, unrecoverably.
    """

    assert evaluator.REPRO_FAIL_TOLERANCE["FGSM-BCE"] == 1e-3
    assert evaluator.REPRO_FAIL_TOLERANCE["PGD-BCE"] == 1.0
    assert evaluator.REPRO_FAIL_TOLERANCE["APGD-BCE"] == 1.0
    assert set(evaluator.REPRO_FAIL_TOLERANCE) == {
        evaluator.CONDITIONS[key] for key in evaluator.ATTACK_KEYS
    }

    # 1% drift on FGSM is a broken configuration, not GPU nondeterminism.
    with pytest.raises(RuntimeError, match="misconfigured"):
        check_reproduction(
            0.0505,
            0.05,
            label="wg/a/16/FGSM-BCE",
            fail=evaluator.REPRO_FAIL_TOLERANCE["FGSM-BCE"],
        )
    # The same drift on APGD is expected and must pass.
    assert check_reproduction(
        0.0505,
        0.05,
        label="wg/a/16/APGD-BCE",
        fail=evaluator.REPRO_FAIL_TOLERANCE["APGD-BCE"],
    ) == pytest.approx(0.01)


def test_drift_summary_reports_counts_instead_of_one_warning_per_row() -> None:
    """The per-row warnings were unique strings, so nothing deduplicated them.

    ~14k attack rows each emitting a distinct ``UserWarning`` buried every other
    warning the run produced; one tallied line per condition replaces them.
    """

    lines = evaluator.format_drift_summary(
        "wg", {"APGD-BCE": [1e-4, 4e-3, 5e-3], "FGSM-BCE": [1e-9]}
    )

    assert len(lines) == 2
    assert "n=3" in lines[0] and "2 rows above" in lines[0]
    assert "n=1" in lines[1] and "0 rows above" in lines[1]
    assert evaluator.format_drift_summary("wg", {}) == []


def test_build_row_rejects_a_perturbation_outside_the_linf_budget() -> None:
    clean = np.zeros((2, 4, 4), dtype=np.float32)
    altered = clean.copy()
    altered[0, 0, 0] = 0.1

    with pytest.raises(RuntimeError, match="L-inf violation"):
        build_row(
            "wg",
            "a",
            2,
            "fgsm",
            clean,
            altered,
            np.zeros_like(clean, dtype=np.int16),
            valid=np.ones_like(clean, dtype=bool),
            spacing=(3.0, 0.5, 0.5),
            published_rms=None,
            base_seed=20260721,
            settings=SETTINGS,
            options={
                "spectral_bins": 32,
                "spectral_margin": 16,
                "spectral_min_size": 2,
                "alignment_voxels": 100,
                "max_gmsd_slices": 1,
            },
        )


def _synthetic_case() -> tuple[np.ndarray, np.ndarray]:
    """Same fixture shape as test_perturbation_structure.py's ``_case`` helper."""

    clean = gaussian_filter(
        np.random.default_rng(7).standard_normal((8, 64, 64)), sigma=(0.0, 2.0, 2.0)
    ).astype(np.float32)
    segmentation = np.zeros((8, 64, 64), dtype=np.int16)
    segmentation[2:6, 20:44, 20:44] = 1
    return clean, segmentation


CONTROL_TARGET_RMS = 0.05


def _rms_matched_field(clean_arr: np.ndarray, target_rms: float) -> np.ndarray:
    """A ±``target_rms`` sign field: RMS exactly on target, L-inf below ε.

    Deliberately not a constant offset -- the spectral family removes each crop's
    mean, so a constant delta would carry no spectral energy at all and abort the
    case before the control gate under test is ever reached.
    """

    signs = np.where(
        np.indices(np.shape(clean_arr)).sum(axis=0) % 2 == 0, 1.0, -1.0
    ).astype(np.float32)
    return (
        np.asarray(clean_arr, dtype=np.float32) + np.float32(target_rms) * signs
    ).astype(np.float32)


def _drive_controls(tmp_path, monkeypatch, fake_gaussian, fake_rician) -> Path:
    """Run the real control call sites in ``evaluate_dataset`` with I/O faked."""

    clean, segmentation = _synthetic_case()

    monkeypatch.setattr(
        evaluator,
        "_dataset_paths",
        lambda dataset_key: {
            "data": tmp_path,
            "plans": tmp_path / "plans.json",
            "checkpoint": tmp_path / "checkpoint.pth",
        },
    )
    monkeypatch.setattr(evaluator, "_available_case_ids", lambda data_dir: ["case_a"])
    monkeypatch.setattr(evaluator, "_use_mask_for_norm", lambda plans_path: False)
    monkeypatch.setattr(
        evaluator,
        "load_sample",
        lambda data_dir, case_id: (
            clean[np.newaxis],
            segmentation,
            (3.0, 0.5, 0.5),
        ),
    )

    monkeypatch.setattr(evaluator, "matched_gaussian_noise", fake_gaussian)
    monkeypatch.setattr(evaluator, "matched_rician_proxy_noise", fake_rician)

    quality = pd.DataFrame(
        [
            {
                "dataset": "wg",
                "case_id": "case_a",
                "epsilon_n": 16,
                "attack": "APGD-BCE",
                "rms_norm": CONTROL_TARGET_RMS,
            }
        ]
    ).set_index(["dataset", "case_id", "epsilon_n", "attack"])
    normalization = pd.DataFrame(
        [
            {
                "dataset": "wg",
                "case_id": "case_a",
                "raw_norm_mean_au": 300.0,
                "raw_norm_std_au": 200.0,
            }
        ]
    ).set_index(["dataset", "case_id"])

    output_csv = tmp_path / "structure.csv"
    spectra_csv = tmp_path / "spectra.csv"
    options = {
        "spectral_bins": 8,
        "spectral_margin": 4,
        "spectral_min_size": 16,
        "alignment_voxels": 1000,
        "max_gmsd_slices": 2,
    }
    settings = {
        "base_seed": 20260721,
        "attack_steps": 20,
        "spectra_cases": 0,
        **options,
    }

    evaluator.evaluate_dataset(
        "wg",
        conditions=["gaussian", "rician"],
        epsilon_n_values=[16],
        attack_steps=20,
        batch_size=1,
        max_cases=None,
        base_seed=20260721,
        settings=settings,
        options=options,
        output_csv=output_csv,
        spectra_csv=spectra_csv,
        quality=quality,
        normalization=normalization,
        completed=set(),
        spectra_cases=set(),
        device=torch.device("cpu"),
        start_time=time.monotonic(),
        total_rows=2,
    )
    return output_csv


def test_controls_pin_the_published_run_match_iterations(tmp_path, monkeypatch) -> None:
    """Both matched-noise call sites must pin ``match_iterations=12``.

    The published random-noise control run
    (``all_cases_random_noise_quality/random_noise_quality_all_cases.csv``) used
    ``match_iterations=12`` throughout (59,970 rows, one unique value); the
    library default in ``epsilon_calibration.py`` is 32. ``_match_parameter_by_rms``
    returns whichever candidate its bisection lands on at the final iteration, so
    a different depth returns a materially different field -- and since controls
    pass ``published_rms=None``, ``check_reproduction`` never runs to catch the
    drift. This test drives the real ``evaluate_dataset`` call sites (with I/O
    boundaries faked) and captures the kwargs the two noise functions actually
    receive.
    """

    assert evaluator.PUBLISHED_MATCH_ITERATIONS == 12

    captured: dict[str, object] = {}

    def fake_gaussian(clean_arr, *, valid_mask, epsilon, target_rms, rng, **kwargs):
        captured["gaussian"] = kwargs.get("match_iterations")
        return _rms_matched_field(clean_arr, target_rms)

    def fake_rician(clean_arr, *, valid_mask, epsilon, target_rms, rng, **kwargs):
        captured["rician"] = kwargs.get("match_iterations")
        return _rms_matched_field(clean_arr, target_rms)

    output_csv = _drive_controls(tmp_path, monkeypatch, fake_gaussian, fake_rician)

    assert captured == {"gaussian": 12, "rician": 12}
    written = pd.read_csv(output_csv)
    assert set(written["condition"]) == {
        "gaussian_rms_matched",
        "rician_proxy_rms_matched",
    }
    assert written["rms_norm"].to_numpy() == pytest.approx(
        CONTROL_TARGET_RMS, rel=evaluator.CONTROL_RMS_TOLERANCE
    )


def test_case_geometry_is_built_once_per_case_not_once_per_row(
    tmp_path, monkeypatch
) -> None:
    """Case-invariant work must not be repeated for every condition.

    The spectral crops, boundary bands, gradient magnitude, and intensity bounds
    are functions of the clean image and segmentation alone, so they are the
    same for all 25 (condition x epsilon) rows of a case. Recomputing them per
    row cost 0.30 s of the 0.37 s a whole-gland row takes, most of it the signed
    distance transform. This drives two conditions through one case and requires
    a single geometry build, reused by both.
    """

    builds: list[object] = []
    real_geometry = evaluator.case_geometry

    def counting_geometry(*args, **kwargs):
        geometry = real_geometry(*args, **kwargs)
        builds.append(geometry)
        return geometry

    used: list[object] = []
    real_structure = evaluator.perturbation_structure

    def spying_structure(*args, geometry=None, **kwargs):
        used.append(geometry)
        return real_structure(*args, geometry=geometry, **kwargs)

    monkeypatch.setattr(evaluator, "case_geometry", counting_geometry)
    monkeypatch.setattr(evaluator, "perturbation_structure", spying_structure)

    def fake_noise(clean_arr, *, valid_mask, epsilon, target_rms, rng, **kwargs):
        return _rms_matched_field(clean_arr, target_rms)

    output_csv = _drive_controls(tmp_path, monkeypatch, fake_noise, fake_noise)

    assert len(pd.read_csv(output_csv)) == 2
    assert len(builds) == 1
    assert used == [builds[0], builds[0]]


def test_a_control_that_misses_its_rms_target_raises(tmp_path, monkeypatch) -> None:
    """The design spec requires raising on "RMS match failure for controls".

    Controls pass ``published_rms=None``, so ``check_reproduction`` never runs on
    them and nothing else would notice. The whole paired comparison rests on the
    control carrying the attack's magnitude: if the match silently degraded,
    every paired difference would compare different magnitudes with no signal
    anywhere in the output.
    """

    def fake_gaussian(clean_arr, *, valid_mask, epsilon, target_rms, rng, **kwargs):
        # 20% low -- far outside the 1e-7..4e-4 drift real controls exhibit.
        return _rms_matched_field(clean_arr, target_rms * 0.8)

    with pytest.raises(RuntimeError, match="RMS match failed"):
        _drive_controls(tmp_path, monkeypatch, fake_gaussian, fake_gaussian)


def test_completed_keys_reads_a_table_without_attack_valid_only_as_unconfined(
    tmp_path,
) -> None:
    """Tables written before the flag existed were unconfined runs."""

    path = tmp_path / "structure.csv"
    pd.DataFrame(
        [{"dataset": "zones", "case_id": "a", "epsilon_n": 16, "condition": "PGD-BCE", **SETTINGS}]
    ).to_csv(path, index=False)

    assert completed_keys(path, settings={**SETTINGS, "attack_valid_only": 0}) == {
        ("zones", "a", 16, "PGD-BCE")
    }
    with pytest.raises(ValueError, match="attack_valid_only"):
        completed_keys(path, settings={**SETTINGS, "attack_valid_only": 1})


def _drive_fgsm(tmp_path, monkeypatch, *, attack_valid_only: int) -> torch.Tensor | None:
    """Run the real FGSM call site with the model and attack faked; return the mask it got."""

    clean, segmentation = _synthetic_case()
    segmentation[:, :12, :] = evaluator.IGNORE_LABEL  # zones-style zero-filled margin
    received: dict[str, object] = {}

    def fake_fgsm(model, x, y, epsilon, num_classes, *, perturbation_mask=None):
        received["mask"] = perturbation_mask
        signs = torch.where(
            torch.arange(x.numel(), device=x.device).reshape(x.shape) % 2 == 0, 1.0, -1.0
        )
        adv = x + epsilon * signs
        return adv if perturbation_mask is None else torch.where(perturbation_mask, adv, x)

    monkeypatch.setattr(
        evaluator,
        "_dataset_paths",
        lambda dataset_key: {
            "data": tmp_path,
            "plans": tmp_path / "plans.json",
            "checkpoint": tmp_path / "checkpoint.pth",
        },
    )
    monkeypatch.setattr(evaluator, "_available_case_ids", lambda data_dir: ["case_a"])
    monkeypatch.setattr(evaluator, "_use_mask_for_norm", lambda plans_path: True)
    monkeypatch.setattr(
        evaluator,
        "load_sample",
        lambda data_dir, case_id: (clean[np.newaxis], segmentation[np.newaxis], (3.0, 0.5, 0.5)),
    )
    monkeypatch.setattr(
        evaluator, "build_model", lambda path, device: (torch.nn.Identity(), [1, 1, 1], 2)
    )
    monkeypatch.setattr(
        evaluator, "_plan_batches", lambda data_dir, case_ids, div_factors, batch_size: [list(case_ids)]
    )
    monkeypatch.setattr(evaluator, "fgsm_bce_independent_batch", fake_fgsm)

    quality = pd.DataFrame(
        [{"dataset": "zones", "case_id": "case_a", "epsilon_n": 16, "attack": "FGSM-BCE", "rms_norm": 16 / 255.0}]
    ).set_index(["dataset", "case_id", "epsilon_n", "attack"])
    normalization = pd.DataFrame(
        [{"dataset": "zones", "case_id": "case_a", "raw_norm_mean_au": 300.0, "raw_norm_std_au": 200.0}]
    ).set_index(["dataset", "case_id"])
    options = {
        "spectral_bins": 8,
        "spectral_margin": 4,
        "spectral_min_size": 16,
        "alignment_voxels": 1000,
        "max_gmsd_slices": 2,
    }
    settings = {
        "base_seed": 20260721,
        "attack_steps": 20,
        "spectra_cases": 0,
        "attack_valid_only": attack_valid_only,
        **options,
    }
    output_csv = tmp_path / "structure.csv"
    evaluator.evaluate_dataset(
        "zones",
        conditions=["fgsm"],
        epsilon_n_values=[16],
        attack_steps=20,
        batch_size=1,
        max_cases=None,
        base_seed=20260721,
        settings=settings,
        options=options,
        output_csv=output_csv,
        spectra_csv=tmp_path / "spectra.csv",
        quality=quality,
        normalization=normalization,
        completed=set(),
        spectra_cases=set(),
        device=torch.device("cpu"),
        start_time=time.monotonic(),
        total_rows=1,
    )
    row = pd.read_csv(output_csv).iloc[0]
    assert int(row["attack_valid_only"]) == attack_valid_only
    if attack_valid_only:
        assert np.isnan(row["rms_published_norm"])  # unconfined reference not a reproduction
    else:
        assert row["rms_published_norm"] == pytest.approx(16 / 255.0)
    return received["mask"]  # type: ignore[return-value]


def test_attack_valid_only_confines_the_attack_to_non_ignore_voxels(
    tmp_path, monkeypatch
) -> None:
    mask = _drive_fgsm(tmp_path, monkeypatch, attack_valid_only=1)
    assert mask is not None
    assert mask.shape == (1, 1, 8, 64, 64)
    assert not mask[0, 0, :, :12, :].any()
    assert mask[0, 0, :, 12:, :].all()


def test_attack_valid_only_off_keeps_the_unconfined_call(tmp_path, monkeypatch) -> None:
    assert _drive_fgsm(tmp_path, monkeypatch, attack_valid_only=0) is None
