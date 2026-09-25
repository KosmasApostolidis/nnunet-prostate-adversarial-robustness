"""Refactor invariants — protect light-cleanup outcomes from regression.

Covers:
  * Subpackages (`experiments/`, `training/`) are importable.
  * Hardcoded absolute paths replaced by `__file__`-relative resolution.
  * `Utils/__init__.py` star imports removed without breaking submodule access.
  * `--output-dir` defaults preserved (paper-figure reproducibility).
  * Sibling-import shim makes the surviving experiment scripts importable from any CWD.
  * No `/home/<user>/...` absolute paths leak back into source.
"""

from __future__ import annotations

import importlib
import re
import subprocess
import sys
from pathlib import Path

import pytest

PROJECT_ROOT = Path(__file__).resolve().parent.parent


# --------------------------------------------------------------------------- #
# Package importability
# --------------------------------------------------------------------------- #


@pytest.mark.parametrize("pkg", ["experiments", "training", "Utils"])
def test_subpackage_importable(pkg):
    mod = importlib.import_module(pkg)
    assert mod.__file__ is not None or hasattr(mod, "__path__")


def test_experiments_init_is_empty_marker():
    p = PROJECT_ROOT / "experiments" / "__init__.py"
    assert p.exists()
    # Must exist for package recognition; content is intentionally empty.
    assert p.read_text() == ""


def test_training_init_is_empty_marker():
    p = PROJECT_ROOT / "training" / "__init__.py"
    assert p.exists()
    assert p.read_text() == ""


def test_utils_init_emptied():
    p = PROJECT_ROOT / "Utils" / "__init__.py"
    content = p.read_text()
    assert "import *" not in content, "star imports must be removed"


# --------------------------------------------------------------------------- #
# Path portability
# --------------------------------------------------------------------------- #


def test_no_user_home_absolute_paths_in_python_sources():
    """Refactor removed all `/home/<user>/...` literals from source. Guard against re-introduction."""
    pattern = re.compile(r"/home/medadmin/")
    offenders = []
    for sub in ("Utils", "experiments", "training"):
        for path in (PROJECT_ROOT / sub).rglob("*.py"):
            text = path.read_text()
            for lineno, line in enumerate(text.splitlines(), 1):
                if pattern.search(line) and not line.lstrip().startswith("#"):
                    offenders.append(
                        f"{path.relative_to(PROJECT_ROOT)}:{lineno}: {line.strip()}"
                    )
    assert not offenders, "Hardcoded user-home paths reintroduced:\n" + "\n".join(
        offenders
    )


# --------------------------------------------------------------------------- #
# Output-dir reproducibility (paper artefacts depend on these)
# --------------------------------------------------------------------------- #

EXPECTED_OUTPUT_DIR_DEFAULTS = {
    ("slice_vulnerability_analysis.py", "results/slice_vulnerability_results"),
    ("uncertainty_analysis.py", "uncertainty_results"),
    ("pgd_adversarial_evaluation.py", "pgd_results"),
    ("failure_analysis.py", "results/failure_results"),
}


@pytest.mark.parametrize(
    "filename,expected_default", sorted(EXPECTED_OUTPUT_DIR_DEFAULTS)
)
def test_output_dir_default_string_present(filename, expected_default):
    """Each script's `--output-dir` default literal must remain unchanged."""
    text = (PROJECT_ROOT / "experiments" / filename).read_text()
    needle = f'"{expected_default}"'
    assert needle in text, f"{filename} no longer contains default '{expected_default}'"


# --------------------------------------------------------------------------- #
# Sibling-import shim — scripts must import from any CWD
# --------------------------------------------------------------------------- #

SHIM_SCRIPTS = [
    "slice_vulnerability_analysis",
    "uncertainty_analysis",
    "transferability_analysis",
    "pgd_adversarial_evaluation",
    "failure_analysis",
    "sensitivity_analysis",
]


@pytest.mark.parametrize("script", SHIM_SCRIPTS)
def test_shim_present_in_script(script):
    text = (PROJECT_ROOT / "experiments" / f"{script}.py").read_text()
    assert "sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))" in text, (
        f"{script}.py missing sys.path shim above sibling import"
    )


@pytest.mark.parametrize(
    "script",
    SHIM_SCRIPTS
    + [
        "compare_robust_vs_standard",
        "statistical_analysis",
        "generate_paper_figures",
    ],
)
def test_experiment_module_importable(script):
    """Smoke import — surfaces any syntax/import regression introduced by Phase 2-3 edits."""
    importlib.import_module(f"experiments.{script}")


def test_sibling_import_works_from_arbitrary_cwd(tmp_path):
    """Run a one-liner that imports a shim'd script from /tmp — proves shim is CWD-independent."""
    snippet = (
        "import sys; "
        f"sys.path.insert(0, {str(PROJECT_ROOT)!r}); "
        "import experiments.failure_analysis as m; "
        "print('ok')"
    )
    proc = subprocess.run(
        [sys.executable, "-c", snippet],
        cwd=str(tmp_path),
        capture_output=True,
        text=True,
        timeout=60,
    )
    assert proc.returncode == 0, f"stderr:\n{proc.stderr}"
    assert proc.stdout.strip() == "ok"


# --------------------------------------------------------------------------- #
# Main entry point — production inference imports
# --------------------------------------------------------------------------- #


def test_main_entry_imports_resolve():
    """`__main__.py` imports must resolve after `Utils/__init__.py` was emptied."""
    # Import each submodule reference from `__main__.py` directly.
    importlib.import_module("Utils.helpers")
    importlib.import_module("Utils.InputCheck")
    # Utils.nnUnet_call is covered by test_nnunet_call_shim_imports_under_nnunetv2_26
    # (the perform_everything_on_gpu drift was fixed when the wrapper moved to
    # nnunetv2 2.6); Utils.segmentor_pipeline needs MedProIO, absent in some envs.


def test_nnunet_call_shim_imports_under_nnunetv2_26():
    """The legacy shim re-exports the wrappers now built for nnunetv2 2.6.

    Until the model-version registry landed, this import raised
    ``TypeError: ... perform_everything_on_gpu`` (kwarg removed upstream); the
    wrapper now passes ``perform_everything_on_device`` and imports cleanly.
    """
    shim = importlib.import_module("Utils.nnUnet_call")
    assert shim.WGNNUnet.__module__ == "mri_prostate_seg.models.nnunet_call"
    assert shim.ZonesNNUnet.__module__ == "mri_prostate_seg.models.nnunet_call"
