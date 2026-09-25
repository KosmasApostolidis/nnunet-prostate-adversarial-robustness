"""Compare robust-trained vs standard models.

Console-script wrapper — see :mod:`mri_prostate_seg.cli._runpy_shim` for the
delegation strategy. The legacy script remains the source of truth so that
``python experiments/compare_robust_vs_standard.py --help`` and ``mri-compare-robust-vs-standard`` behave identically.
"""

from __future__ import annotations

from mri_prostate_seg.cli._runpy_shim import run_legacy

LEGACY_PATH = "experiments/compare_robust_vs_standard.py"


def main() -> None:
    """Delegate to the legacy script with byte-identical argparse output."""
    run_legacy(LEGACY_PATH)
