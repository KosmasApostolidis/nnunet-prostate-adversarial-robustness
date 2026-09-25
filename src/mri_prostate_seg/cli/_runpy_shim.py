"""Helper that runs a legacy script as ``__main__`` while preserving argparse byte-identity.

Console-script entry points wired through ``[project.scripts]`` invoke
``main()`` of a small wrapper module. To keep the experiment scripts'
``--help`` output identical to the legacy ``python experiments/foo.py
--help`` form, we rewrite ``sys.argv[0]`` to the legacy script path before
delegating via :mod:`runpy`. argparse derives its program name from
``os.path.basename(sys.argv[0])``, so the usage line then matches the
golden snapshots captured in ``tests/golden/cli_help/``.
"""

from __future__ import annotations

import os
import runpy
import sys
from typing import NoReturn

from mri_prostate_seg.paths import REPO_ROOT


def _resolve_legacy(rel_path: str) -> str:
    abs_path = os.path.join(REPO_ROOT, rel_path)
    if not os.path.isfile(abs_path):
        raise FileNotFoundError(
            f"Legacy script not found at {abs_path}. The repository layout "
            f"may have changed since the console-script wrappers were generated."
        )
    return abs_path


def run_legacy(rel_path: str) -> NoReturn:
    """Run the legacy script at ``rel_path`` (relative to repo root) as ``__main__``.

    Raises ``SystemExit`` on completion (argparse's ``--help`` exits 0; other
    runs exit with whatever code the legacy script chose). Never returns.
    """
    abs_path = _resolve_legacy(rel_path)
    sys.argv[0] = abs_path
    runpy.run_path(abs_path, run_name="__main__")
    raise SystemExit(0)
