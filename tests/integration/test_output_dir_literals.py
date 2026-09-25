"""Output-dir literal regression.

The 13+ output directory names in ``mri_prostate_seg.paths`` are paper
artifacts: changing them invalidates downstream analysis runs. This test
locks each canonical literal to its expected value, and verifies the
``--help`` snapshots still reference the literal as the ``--output-dir``
default.
"""

from __future__ import annotations

import pytest

from mri_prostate_seg import paths

# Canonical literals — must NEVER drift. Pulled from pre-restructure scripts.
EXPECTED_LITERALS: dict[str, str] = {
    "FGSM_RESULTS_DIR": "fgsm_results",
    "FGSM_RESULTS_ROBUST_DIR": "fgsm_results_robust",
    "FGSM_FOLD_ENSEMBLE_RESULTS_DIR": "fgsm_fold_ensemble_results",
    "PGD_RESULTS_DIR": "pgd_results",
    "NOISE_RESULTS_DIR": "noise_results",
    "UNCERTAINTY_RESULTS_DIR": "uncertainty_results",
    "SENSITIVITY_RESULTS_DIR": "sensitivity_results",
    "FAILURE_RESULTS_DIR": "results/failure_results",
    "SLICE_VULNERABILITY_RESULTS_DIR": "results/slice_vulnerability_results",
    "SLICE_BOUNDARY_DECOMPOSITION_RESULTS_DIR": "slice_boundary_decomposition_results",
    "STATISTICAL_RESULTS_DIR": "statistical_results",
    "EXPLAINABILITY_RESULTS_DIR": "explainability_results",
    "TRANSFER_RESULTS_DIR": "transfer_results",
    "PAPER_FIGURES_DIR": "paper_figures",
    "DATASET016_FGSM_PER_FOLD_OUTPUT_DIR": "dataset016_fgsm_per_fold_output",
    "DICOM_OUTPUTS_DIR": "dicom_outputs",
    "INPUT_VOLUME": "Pats",
    "OUTPUT_VOLUME": "Outputs",
}


@pytest.mark.parametrize(("name", "expected"), sorted(EXPECTED_LITERALS.items()))
def test_canonical_literal_unchanged(name: str, expected: str) -> None:
    assert getattr(paths, name) == expected, (
        f"paths.{name} drifted from canonical value {expected!r}"
    )


EXPECTED_TUPLE_NAMES: frozenset[str] = frozenset(
    {
        "FGSM_RESULTS_DIR",
        "FGSM_RESULTS_ROBUST_DIR",
        "FGSM_FOLD_ENSEMBLE_RESULTS_DIR",
        "PGD_RESULTS_DIR",
        "NOISE_RESULTS_DIR",
        "UNCERTAINTY_RESULTS_DIR",
        "SENSITIVITY_RESULTS_DIR",
        "FAILURE_RESULTS_DIR",
        "SLICE_VULNERABILITY_RESULTS_DIR",
        "SLICE_BOUNDARY_DECOMPOSITION_RESULTS_DIR",
        "STATISTICAL_RESULTS_DIR",
        "EXPLAINABILITY_RESULTS_DIR",
        "TRANSFER_RESULTS_DIR",
        "PAPER_FIGURES_DIR",
        "DATASET016_FGSM_PER_FOLD_OUTPUT_DIR",
    }
)


def test_all_output_dir_literals_tuple_complete() -> None:
    """Every output-dir constant is exposed in ``ALL_OUTPUT_DIR_LITERALS``."""
    expected = {EXPECTED_LITERALS[name] for name in EXPECTED_TUPLE_NAMES}
    assert set(paths.ALL_OUTPUT_DIR_LITERALS) == expected


def test_no_output_dir_literal_collisions() -> None:
    """Each canonical output-dir literal value is unique."""
    values = list(paths.ALL_OUTPUT_DIR_LITERALS)
    assert len(values) == len(set(values)), (
        "duplicate dir literal in ALL_OUTPUT_DIR_LITERALS — paper artifacts would collide"
    )
