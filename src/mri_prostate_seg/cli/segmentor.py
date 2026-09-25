"""Production segmentation entry point (was __main__.py).

Console-script wrapper — see :mod:`mri_prostate_seg.cli._runpy_shim` for the
delegation strategy. The legacy script remains the source of truth so that
``python __main__.py --help`` and ``mri-segmentor`` behave identically.
"""

from __future__ import annotations

from mri_prostate_seg.cli._runpy_shim import run_legacy

LEGACY_PATH = "__main__.py"


def main() -> None:
    """Delegate to the legacy script with byte-identical argparse output."""
    run_legacy(LEGACY_PATH)
