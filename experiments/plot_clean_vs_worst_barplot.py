"""Grouped bar charts: clean vs. worst-case Dice, before vs. after PGD-AT.

Reads adv_rob_eval_results/combined_summary_table.csv (written by
generate_combined_summary_table.py) and renders, per dataset, a bar chart
with one group of 4 bars (Clean-model/Clean-data, Clean-model/Worst-case,
PGD-AT-model/Clean-data, PGD-AT-model/Worst-case) per architecture.

Worst-case = APGD @ epsilon=16/255 Dice (the largest epsilon retained in the sweep).

Usage::

    python experiments/plot_clean_vs_worst_barplot.py
"""

from __future__ import annotations

import csv
import os

import matplotlib.pyplot as plt
import numpy as np

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
IN_CSV = os.path.join(REPO_ROOT, "results", "adv_rob_eval_results", "combined_summary_table.csv")
OUT_DIR = os.path.join(REPO_ROOT, "results", "adv_rob_eval_results", "figures")

DATASET_LABELS = {"WG": "Whole Gland (WG)", "Zones": "Prostate Zones"}
ARCHS = ["UNet", "ResEnc-M", "ResEnc-L", "ResEnc-XL"]

# (legend label, variant, dice column, sem column, bar color)
BAR_SPECS = [
    ("Clean model — clean data", "Clean", "clean_dice", "clean_dice_sem", "#4C72B0"),
    (
        "Clean model — worst-case (APGD@16)",
        "Clean",
        "apgd_worst_dice",
        "apgd_worst_dice_sem",
        "#A6C8E0",
    ),
    ("PGD-AT model — clean data", "PGD-AT", "clean_dice", "clean_dice_sem", "#C44E52"),
    (
        "PGD-AT model — worst-case (APGD@16)",
        "PGD-AT",
        "apgd_worst_dice",
        "apgd_worst_dice_sem",
        "#E8A6A9",
    ),
]


def load_rows(path: str) -> list[dict]:
    with open(path, newline="") as f:
        return list(csv.DictReader(f))


def plot_dataset(rows: list[dict], ds: str) -> None:
    block = {(r["architecture"], r["variant"]): r for r in rows if r["dataset"] == ds}

    n_arch = len(ARCHS)
    n_bars = len(BAR_SPECS)
    x = np.arange(n_arch)
    width = 0.8 / n_bars

    fig, ax = plt.subplots(figsize=(9, 5.5))

    for i, (label, variant, col, sem_col, color) in enumerate(BAR_SPECS):
        heights = [float(block[(a, variant)][col]) for a in ARCHS]
        errs = [float(block[(a, variant)][sem_col]) for a in ARCHS]
        offset = (i - (n_bars - 1) / 2) * width
        bars = ax.bar(
            x + offset,
            heights,
            width,
            yerr=errs,
            capsize=2,
            label=label,
            color=color,
            edgecolor="black",
            linewidth=0.5,
        )
        for bar, h in zip(bars, heights):
            ax.text(
                bar.get_x() + bar.get_width() / 2,
                h + 0.02,
                f"{h:.2f}",
                ha="center",
                va="bottom",
                fontsize=7,
                rotation=90,
            )

    ax.set_xticks(x)
    ax.set_xticklabels(ARCHS)
    ax.set_ylabel("Dice")
    ax.set_ylim(0, 1.15)
    ax.set_title(
        f"{DATASET_LABELS[ds]} — clean vs. worst-case Dice, before/after PGD-AT",
        fontsize=12,
        fontweight="bold",
    )
    ax.legend(
        loc="upper center",
        bbox_to_anchor=(0.5, -0.12),
        ncol=2,
        fontsize=9,
        frameon=False,
    )
    ax.yaxis.grid(True, linestyle=":", alpha=0.5)
    ax.set_axisbelow(True)

    out_path = os.path.join(OUT_DIR, f"{ds.lower()}_clean_vs_worst_barplot.png")
    fig.savefig(
        out_path, dpi=300, bbox_inches="tight", facecolor="white", edgecolor="none"
    )
    plt.close(fig)
    print(f"Saved: {out_path}")


def main() -> None:
    plt.rcParams.update(
        {
            "font.family": "sans-serif",
            "font.sans-serif": ["DejaVu Sans"],
            "figure.dpi": 150,
        }
    )
    os.makedirs(OUT_DIR, exist_ok=True)
    rows = load_rows(IN_CSV)
    for ds in DATASET_LABELS:
        plot_dataset(rows, ds)


if __name__ == "__main__":
    main()
