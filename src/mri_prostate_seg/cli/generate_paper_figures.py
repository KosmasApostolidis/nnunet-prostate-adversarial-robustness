"""Aggregate paper figure generation.

Console-script wrapper — see :mod:`mri_prostate_seg.cli._runpy_shim` for the
delegation strategy. The legacy script remains the source of truth so that
``python experiments/generate_paper_figures.py --help`` and ``mri-generate-paper-figures`` behave identically.
"""

from __future__ import annotations

from mri_prostate_seg.cli._runpy_shim import run_legacy

LEGACY_PATH = "experiments/generate_paper_figures.py"


def main() -> None:
    """Delegate to the legacy script with byte-identical argparse output."""
    run_legacy(LEGACY_PATH)
