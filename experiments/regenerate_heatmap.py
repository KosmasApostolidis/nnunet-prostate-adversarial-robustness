"""Regenerate combined_wg_zones_slice_vulnerability_heatmap.png from existing CSVs.

Reads pre-computed per-slice FGSM data and re-plots only the Dice Drop
positional heatmap — no GPU or model loading required.
"""

import csv
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from mri_prostate_seg.experiments.slice_vulnerability.metrics import group_records_by_eps_class
from mri_prostate_seg.experiments.slice_vulnerability.combined import _draw_combined_heatmap

RESULTS_DIR = os.path.join(
    os.path.dirname(os.path.abspath(__file__)),
    "..", "results", "slice_vulnerability_results",
)


_NUMERIC_FIELDS = [
    "slice_idx", "total_slices", "epsilon", "gt_area",
    "dice_clean", "dice_adv", "dice_drop",
    "hd95_clean", "hd95_adv", "hd95_change",
    "asd_clean", "asd_adv", "asd_change",
]


def _read_csv(path: str) -> list[dict]:
    records = []
    with open(path, newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            for k in _NUMERIC_FIELDS:
                raw = row[k]
                row[k] = float(raw) if raw.strip() != "" else float("nan")
            records.append(row)
    return records


def main():
    wg_path = os.path.join(RESULTS_DIR, "wg_per_slice_fgsm.csv")
    zones_path = os.path.join(RESULTS_DIR, "zones_per_slice_fgsm.csv")

    for p in (wg_path, zones_path):
        if not os.path.exists(p):
            print(f"Missing: {p}")
            sys.exit(1)

    print("Reading CSV data...")
    wg_records = _read_csv(wg_path)
    zones_records = _read_csv(zones_path)

    eps_list = sorted(set(float(r["epsilon"]) for r in wg_records))
    print(f"Epsilons: {eps_list}")

    grouped_wg = group_records_by_eps_class(wg_records)
    grouped_z = group_records_by_eps_class(zones_records)

    print("Drawing combined Dice Drop heatmap...")
    _draw_combined_heatmap(
        grouped_wg=grouped_wg,
        grouped_z=grouped_z,
        eps_list=eps_list,
        metric_key="dice_drop",
        metric_short="Dice Drop",
        fname_tag="slice_vulnerability_heatmap",
        cmap="YlOrRd",
        cbar="Mean Dice Drop",
        output_dir=RESULTS_DIR,
    )
    print("Done.")


if __name__ == "__main__":
    main()
