"""Generate the Figure 3 predictions for a single representative patient.

Reuses the attack driver from ``generate_worst_case_predictions`` but points it
at one patient that appears in both cohorts, so every panel of the 3x3 grid
shows the same anatomy under the three attacks.

The patient was chosen as the shared case whose nine epsilon=0.1 endpoints in
the paper's own per-fold ``*_per_sample.csv`` results sit closest to the
Table I cohort means.
"""

from __future__ import annotations

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import generate_worst_case_predictions as base  # noqa: E402

HASH = "60391143992449728674423394462840605622"
WG_CASE = f"ProstateWG_{HASH}"
ZONES_CASE = f"ProstateZonesFilteredLessDilated_ProstateZones_{HASH}"
WG_FOLD, ZONES_FOLD = 2, 3

# target_dice values are this patient's own results from the paper's
# per-sample CSVs at epsilon=0.1, not cohort means.
CASES = [
    {
        "case_id": WG_CASE,
        "fold": WG_FOLD,
        "attack": "fgsm",
        "model": "wg",
        "class_name": "WG",
        "target_dice": 0.819,
    },
    {
        "case_id": WG_CASE,
        "fold": WG_FOLD,
        "attack": "pgd",
        "model": "wg",
        "class_name": "WG",
        "target_dice": 0.591,
    },
    {
        "case_id": WG_CASE,
        "fold": WG_FOLD,
        "attack": "a_pgd",
        "model": "wg",
        "class_name": "WG",
        "target_dice": 0.467,
    },
    {
        "case_id": ZONES_CASE,
        "fold": ZONES_FOLD,
        "attack": "fgsm",
        "model": "zones",
        "class_name": "TZ+CZ",
        "target_dice": 0.883,
    },
    {
        "case_id": ZONES_CASE,
        "fold": ZONES_FOLD,
        "attack": "pgd",
        "model": "zones",
        "class_name": "TZ+CZ",
        "target_dice": 0.698,
    },
    {
        "case_id": ZONES_CASE,
        "fold": ZONES_FOLD,
        "attack": "a_pgd",
        "model": "zones",
        "class_name": "TZ+CZ",
        "target_dice": 0.091,
    },
    {
        "case_id": ZONES_CASE,
        "fold": ZONES_FOLD,
        "attack": "fgsm",
        "model": "zones",
        "class_name": "PZ",
        "target_dice": 0.746,
    },
    {
        "case_id": ZONES_CASE,
        "fold": ZONES_FOLD,
        "attack": "pgd",
        "model": "zones",
        "class_name": "PZ",
        "target_dice": 0.412,
    },
    {
        "case_id": ZONES_CASE,
        "fold": ZONES_FOLD,
        "attack": "a_pgd",
        "model": "zones",
        "class_name": "PZ",
        "target_dice": 0.002,
    },
]

OUTPUT_DIR = os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
    "results",
    "wg_zones_adversarial_robustness_results",
    "single_case_predictions",
)


def main() -> None:
    # The zonal model emits TZ+CZ and PZ in one forward pass, so the two zonal
    # entries per attack repeat the same run. They are kept anyway: the output
    # files are named per class, and seeding makes the repeat identical.
    base.CASES = CASES
    base.OUTPUT_DIR = OUTPUT_DIR
    base.main()


if __name__ == "__main__":
    main()
