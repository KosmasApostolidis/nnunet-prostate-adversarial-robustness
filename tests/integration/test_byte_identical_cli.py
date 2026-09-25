"""Byte-identical CLI gate.

Each legacy entry point (``experiments/*.py`` + ``__main__.py``) must emit
the exact ``--help`` text captured pre-restructure into
``tests/golden/cli_help/*.txt``. Drift = test fails.
"""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[2]
GOLDEN_DIR = REPO_ROOT / "tests" / "golden" / "cli_help"


def _entrypoints() -> list[Path]:
    out: list[Path] = []
    main = REPO_ROOT / "__main__.py"
    if main.exists() and (GOLDEN_DIR / "__main__.txt").exists():
        out.append(main)
    for script in sorted((REPO_ROOT / "experiments").glob("*.py")):
        if script.name == "__init__.py":
            continue
        if (GOLDEN_DIR / f"{script.stem}.txt").exists():
            out.append(script)
    return out


@pytest.mark.parametrize("script", _entrypoints(), ids=lambda p: p.stem)
def test_help_byte_identical(script: Path) -> None:
    golden = GOLDEN_DIR / f"{script.stem}.txt"
    assert golden.exists(), f"missing golden snapshot for {script.name}"
    result = subprocess.run(
        [sys.executable, str(script), "--help"],
        capture_output=True,
        text=True,
        cwd=REPO_ROOT,
    )
    got = result.stdout
    assert got == golden.read_text(), (
        f"--help drift in {script.relative_to(REPO_ROOT)}\n"
        f"diff golden vs current — re-capture or fix CLI"
    )
