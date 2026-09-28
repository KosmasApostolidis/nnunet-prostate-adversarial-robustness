"""Regenerate Figure 3 with the camera-ready attacks.

Same patient, folds and epsilon as ``generate_single_case_predictions`` (the
shared case closest to the Table I cohort means), but the adversarial examples
come from the re-derived attacks in ``slice_vulnerability.perturbations``:
the random +/-eps noise control, FGSM, PGD (random start, step 2.5*eps/steps)
and Croce and Hein's Auto-PGD
(``auto_pgd.py``), labelled APGD. The
grid is drawn by ``create_worst_case_grid_raw`` without its super-title, which
the caption already carries, and copied into ``paper/figures``.
"""

from __future__ import annotations

import os
import shutil
import sys

import matplotlib.figure
import torch

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(REPO_ROOT, "experiments"))
sys.path.insert(0, os.path.join(REPO_ROOT, "src"))

import create_worst_case_grid_raw as grid  # noqa: E402
import generate_single_case_predictions as single  # noqa: E402
import generate_worst_case_predictions as base  # noqa: E402
from mri_prostate_seg.experiments.slice_vulnerability.perturbations import (  # noqa: E402
    ATTACKS,
)

OUTPUT_DIR = os.path.join(
    REPO_ROOT, "results", "camera_ready_rederived", "single_case_predictions"
)
PAPER_FIGURE = os.path.join(REPO_ROOT, "paper", "figures", "worst_case_3x3_grid.png")
# File stems keep the old attack keys so the grid script reads them unchanged.
ARM_FOR_KEY = {"noise": "noise", "fgsm": "fgsm", "pgd": "pgd", "a_pgd": "auto_pgd"}
N_STEPS = 20


def run_attack(
    network, x_padded, y_padded, eps, num_classes, attack_type, x_bounds, device
):
    with torch.enable_grad():
        return ATTACKS[ARM_FOR_KEY[attack_type]](
            network,
            x_padded,
            y_padded,
            eps,
            num_classes,
            x_bounds,
            n_steps=N_STEPS,
            generator=None,
        ).detach()


def main() -> None:
    base.run_attack = run_attack
    noise_cases = [{**c, "attack": "noise"} for c in single.CASES if c["attack"] == "fgsm"]
    base.CASES = noise_cases + single.CASES
    base.OUTPUT_DIR = OUTPUT_DIR
    base.main()

    grid.PRED_DIR = OUTPUT_DIR
    grid.ATTACK_LABELS["a_pgd"] = "APGD"
    grid.ATTACK_LABELS["noise"] = "Noise"
    grid.ATTACKS = ["noise", "fgsm", "pgd", "a_pgd"]
    grid.reported_dice = lambda *args: float("nan")  # no evaluation-run reference
    matplotlib.figure.Figure.suptitle = lambda self, *args, **kwargs: None
    grid.main()
    shutil.copyfile(
        os.path.join(OUTPUT_DIR, "worst_case_3x3_grid_raw.png"), PAPER_FIGURE
    )
    print(f"Copied to {PAPER_FIGURE}")


if __name__ == "__main__":
    main()
