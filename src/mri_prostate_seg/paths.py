"""Path resolution + canonical output-dir literals.

This module centralises the absolute path of ``nnUnet_paths`` and the 13
output-directory string literals used as ``--output-dir`` defaults across
the experiment scripts. Importing the constants from a single source lets
new CLI modules reuse the same values without risking drift, while the
values themselves remain byte-identical to the legacy literals.
"""

from __future__ import annotations

import os

# Repo root = parent of ``src/``.
PACKAGE_ROOT: str = os.path.dirname(os.path.abspath(__file__))
SRC_ROOT: str = os.path.dirname(PACKAGE_ROOT)
REPO_ROOT: str = os.path.dirname(SRC_ROOT)

# nnU-Net checkpoints / preprocessed data live next to the repo root.
NNUNET_PATHS: str = os.path.join(REPO_ROOT, "nnUnet_paths")

# ---------------------------------------------------------------------------
# Output-directory defaults, byte-identical to legacy literals.
# Cross-script consumers must use these constants so that any future rename
# stays consistent across all callers.
# ---------------------------------------------------------------------------

FGSM_RESULTS_DIR: str = "fgsm_results"
FGSM_RESULTS_ROBUST_DIR: str = "fgsm_results_robust"
FGSM_FOLD_ENSEMBLE_RESULTS_DIR: str = "fgsm_fold_ensemble_results"
PGD_RESULTS_DIR: str = "pgd_results"
NOISE_RESULTS_DIR: str = "noise_results"
UNCERTAINTY_RESULTS_DIR: str = "uncertainty_results"
SENSITIVITY_RESULTS_DIR: str = "sensitivity_results"
FAILURE_RESULTS_DIR: str = "results/failure_results"
SLICE_VULNERABILITY_RESULTS_DIR: str = "results/slice_vulnerability_results"
SLICE_BOUNDARY_DECOMPOSITION_RESULTS_DIR: str = "slice_boundary_decomposition_results"
STATISTICAL_RESULTS_DIR: str = "statistical_results"
EXPLAINABILITY_RESULTS_DIR: str = "explainability_results"
TRANSFER_RESULTS_DIR: str = "transfer_results"
PAPER_FIGURES_DIR: str = "paper_figures"
DATASET016_FGSM_PER_FOLD_OUTPUT_DIR: str = "dataset016_fgsm_per_fold_output"

# Production volumes — used by mri-segmentor (was __main__.py).
INPUT_VOLUME: str = "Pats"
OUTPUT_VOLUME: str = "Outputs"
DICOM_OUTPUTS_DIR: str = "dicom_outputs"

ALL_OUTPUT_DIR_LITERALS: tuple[str, ...] = (
    FGSM_RESULTS_DIR,
    FGSM_RESULTS_ROBUST_DIR,
    FGSM_FOLD_ENSEMBLE_RESULTS_DIR,
    PGD_RESULTS_DIR,
    NOISE_RESULTS_DIR,
    UNCERTAINTY_RESULTS_DIR,
    SENSITIVITY_RESULTS_DIR,
    FAILURE_RESULTS_DIR,
    SLICE_VULNERABILITY_RESULTS_DIR,
    SLICE_BOUNDARY_DECOMPOSITION_RESULTS_DIR,
    STATISTICAL_RESULTS_DIR,
    EXPLAINABILITY_RESULTS_DIR,
    TRANSFER_RESULTS_DIR,
    PAPER_FIGURES_DIR,
    DATASET016_FGSM_PER_FOLD_OUTPUT_DIR,
)


__all__ = [
    "PACKAGE_ROOT",
    "SRC_ROOT",
    "REPO_ROOT",
    "NNUNET_PATHS",
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
    "INPUT_VOLUME",
    "OUTPUT_VOLUME",
    "DICOM_OUTPUTS_DIR",
    "ALL_OUTPUT_DIR_LITERALS",
]
