"""The driver must expose auto_pgd and must not confuse the two APGD modules."""

from __future__ import annotations

import inspect
import subprocess
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]


def test_cli_offers_auto_pgd() -> None:
    out = subprocess.run(
        [
            sys.executable,
            str(REPO_ROOT / "experiments" / "pgd_adversarial_evaluation.py"),
            "--help",
        ],
        capture_output=True,
        text=True,
        check=True,
        cwd=REPO_ROOT,
    ).stdout
    assert "auto_pgd" in out


def test_auto_pgd_wrapper_matches_the_apgd_signature() -> None:
    from experiments.pgd_adversarial_evaluation import (
        a_pgd_with_restarts,
        auto_pgd_with_restarts,
    )

    assert list(inspect.signature(auto_pgd_with_restarts).parameters) == list(
        inspect.signature(a_pgd_with_restarts).parameters
    )


def test_apgd_comparator_is_the_campaign_module() -> None:
    """Guards the mistake that produced the wrong Fig. 3 panel."""
    from experiments import pgd_adversarial_evaluation as driver

    source = inspect.getsource(driver.a_pgd_with_restarts)
    assert "a_pgd_campaign" in source
    assert "from mri_prostate_seg.attacks.a_pgd import" not in source

    auto_source = inspect.getsource(driver.auto_pgd_with_restarts)
    assert "attacks.auto_pgd" in auto_source
    assert "a_pgd_campaign" not in auto_source
