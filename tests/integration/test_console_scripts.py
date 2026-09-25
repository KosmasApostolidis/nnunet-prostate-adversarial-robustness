"""Console-script entry points must produce ``--help`` output equivalent
to the legacy ``python experiments/<name>.py --help`` invocation.

Each console script is wired in ``pyproject.toml`` ``[project.scripts]``
and resolves to ``mri_prostate_seg.cli.<name>:main``. The cli wrapper
delegates via ``runpy`` with ``sys.argv[0]`` rewritten to the legacy
script path, so argparse ``prog=`` matches and ``--help`` is identical.
"""

from __future__ import annotations

import shutil
import subprocess
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[2]
GOLDEN_DIR = REPO_ROOT / "tests" / "golden" / "cli_help"

# (console-script, golden-snapshot-stem)
CONSOLE_SCRIPTS: list[tuple[str, str]] = [
    ("mri-segmentor", "__main__"),
    ("mri-pgd-eval", "pgd_adversarial_evaluation"),
    ("mri-slice-vulnerability", "slice_vulnerability_analysis"),
    ("mri-uncertainty-analysis", "uncertainty_analysis"),
    ("mri-failure-analysis", "failure_analysis"),
    ("mri-transferability-analysis", "transferability_analysis"),
    ("mri-sensitivity-analysis", "sensitivity_analysis"),
    ("mri-statistical-analysis", "statistical_analysis"),
    ("mri-compare-robust-vs-standard", "compare_robust_vs_standard"),
    ("mri-generate-paper-figures", "generate_paper_figures"),
]


@pytest.mark.parametrize(
    ("script", "stem"), CONSOLE_SCRIPTS, ids=[s[0] for s in CONSOLE_SCRIPTS]
)
def test_console_script_help_matches_golden(script: str, stem: str) -> None:
    if shutil.which(script) is None:
        pytest.skip(f"{script} not on PATH (run `pip install -e .`)")
    golden = GOLDEN_DIR / f"{stem}.txt"
    if not golden.exists():
        pytest.skip(f"no golden snapshot for {stem}")
    result = subprocess.run(
        [script, "--help"], capture_output=True, text=True, cwd=REPO_ROOT
    )
    assert result.stdout == golden.read_text(), (
        f"{script} --help drift vs {golden.name}"
    )
