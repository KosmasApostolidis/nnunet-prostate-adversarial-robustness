"""Adversarial robustness — clean vs. PGD-AT finetuned comparison figures.

Reads the four CSVs produced by ``adv_rob_eval_clean_vs_advft.py`` and builds
one figure per dataset (WG: ResEnc-L, Zones: UNet) overlaying the clean and
PGD-AT finetuned model: a 3×3 grid of metric (rows: Dice/HD95/ASD) ×
attack (columns: FGSM/PGD/APGD) panels vs epsilon.

Usage::

    python experiments/plot_clean_vs_advft.py
"""

from __future__ import annotations

import csv
import os
from collections import defaultdict
from typing import Dict, List, Tuple

import matplotlib
import numpy as np

matplotlib.use("Agg")
import matplotlib.pyplot as plt

# ---------------------------------------------------------------------------
# Publication style (matches plot_model_comparison.py)
# ---------------------------------------------------------------------------
matplotlib.rcParams.update(
    {
        "font.family": "serif",
        "font.serif": ["Times New Roman", "DejaVu Serif"],
        "font.size": 11,
        "axes.titlesize": 13,
        "axes.labelsize": 11.5,
        "xtick.labelsize": 9.5,
        "ytick.labelsize": 9.5,
        "legend.fontsize": 12,
        "figure.dpi": 150,
        "savefig.dpi": 300,
        "savefig.bbox": "tight",
        "savefig.facecolor": "white",
    }
)

# ---------------------------------------------------------------------------
# Paths & constants
# ---------------------------------------------------------------------------
REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RESULTS_DIR = os.path.join(REPO_ROOT, "results", "adv_rob_eval_results", "clean_vs_advft")

EPSILONS: List[float] = [0.0, 2 / 255, 4 / 255, 8 / 255, 16 / 255, 32 / 255]

# (dataset_key, arch_key, dataset_label, arch_label)
COMPARISONS = [
    ("wg", "unet", "WG", "UNet"),
    ("wg", "resenc_m", "WG", "ResEnc-M"),
    ("wg", "resenc_l", "WG", "ResEnc-L"),
    ("wg", "resenc_xl", "WG", "ResEnc-XL"),
    ("zones", "unet", "Zones", "UNet"),
    ("zones", "resenc_m", "Zones", "ResEnc-M"),
    ("zones", "resenc_l", "Zones", "ResEnc-L"),
]

VARIANT_STYLE = {
    "clean": {"color": "#0072B2", "marker": "o", "label": "Clean"},
    "advft": {"color": "#D55E00", "marker": "s", "label": "PGD-AT"},
}

METRICS = ["dice", "hd95", "asd"]
METRIC_LABELS = {"dice": "Dice", "hd95": "HD95 (mm)", "asd": "ASD (mm)"}

ATTACK_ORDER = ["FGSM", "PGD", "APGD-CE", "AutoAttack"]


# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------
def load_csv(path: str) -> Dict[str, Dict[float, Dict[str, Tuple[float, float]]]]:
    """Return data[attack][epsilon][metric] = (mean, sem)."""
    data: Dict[str, Dict[float, Dict[str, Tuple[float, float]]]] = defaultdict(
        lambda: defaultdict(dict)
    )
    with open(path, newline="") as f:
        for row in csv.DictReader(f):
            eps = float(row["epsilon"])
            for m in METRICS:
                data[row["attack"]][eps][m] = (
                    float(row[f"{m}_mean"]),
                    float(row[f"{m}_sem"]),
                )
    return data


# ---------------------------------------------------------------------------
# Figure construction
# ---------------------------------------------------------------------------
def plot_comparison(
    dataset_key: str, arch_key: str, ds_label: str, arch_label: str
) -> None:
    csv_paths = {
        "clean": os.path.join(RESULTS_DIR, f"{dataset_key}_{arch_key}_adv_rob.csv"),
        "advft": os.path.join(
            RESULTS_DIR, f"{dataset_key}_{arch_key}_advft_adv_rob.csv"
        ),
    }
    per_variant = {}
    for variant, path in csv_paths.items():
        if not os.path.isfile(path):
            print(f"  [SKIP] {path} not found")
            continue
        per_variant[variant] = load_csv(path)

    if len(per_variant) < 2:
        print(f"  [WARN] Missing data for {dataset_key}/{arch_key}")
        return

    # Attack columns actually present in the CSVs, in preferred order — so the
    # figure matches whatever attack set the eval was run with (e.g. AutoAttack
    # only) instead of assuming all of ATTACK_ORDER.
    present = set()
    for pv in per_variant.values():
        present.update(pv.keys())
    cols = [a for a in ATTACK_ORDER if a in present]
    if not cols:
        print(f"  [WARN] No known attacks in {dataset_key}/{arch_key} CSVs")
        return

    fig, axes = plt.subplots(
        len(METRICS),
        len(cols),
        figsize=(4.3 * len(cols) + 1.0, 10.5),
        sharex="col",
        squeeze=False,
    )
    fig.suptitle(
        f"Clean vs. PGD-AT Finetuned — {ds_label} ({arch_label})",
        fontsize=16,
        fontweight="bold",
        y=1.03,
    )

    for row, metric in enumerate(METRICS):
        for col, atk in enumerate(cols):
            ax = axes[row][col]
            for variant, data in per_variant.items():
                means, sems = [], []
                for eps in EPSILONS:
                    mean, sem = (
                        data.get(atk, {}).get(eps, {}).get(metric, (np.nan, 0.0))
                    )
                    means.append(mean)
                    sems.append(sem)
                means_arr = np.array(means)
                sems_arr = np.array(sems)
                style = VARIANT_STYLE[variant]

                ax.fill_between(
                    EPSILONS,
                    means_arr - sems_arr,
                    means_arr + sems_arr,
                    color=style["color"],
                    alpha=0.15,
                    lw=0,
                    zorder=1,
                )
                ax.plot(
                    EPSILONS,
                    means_arr,
                    marker=style["marker"],
                    color=style["color"],
                    linestyle="-",
                    linewidth=2.0,
                    markersize=6,
                    markerfacecolor="white",
                    markeredgewidth=1.3,
                    alpha=0.95,
                    zorder=3,
                    label=style["label"],
                )

            if row == 0:
                ax.set_title(atk, fontweight="bold", pad=10)
            if row == len(METRICS) - 1:
                ax.set_xlabel("ε (·/255, L∞)")
            if col == 0:
                ax.set_ylabel(METRIC_LABELS[metric])
            ax.set_xticks(EPSILONS)
            ax.set_xticklabels([str(round(e * 255)) for e in EPSILONS])
            ax.spines["top"].set_visible(False)
            ax.spines["right"].set_visible(False)
            ax.yaxis.grid(True, alpha=0.3, lw=0.6)
            ax.set_axisbelow(True)

    axes[0][0].set_ylim(0.0, 1.02)

    handles, labels = axes[0][0].get_legend_handles_labels()
    fig.legend(
        handles,
        labels,
        loc="upper center",
        ncol=2,
        fontsize=13,
        frameon=False,
        bbox_to_anchor=(0.5, 0.995),
    )

    fig.subplots_adjust(
        left=0.07, right=0.98, bottom=0.05, top=0.90, wspace=0.15, hspace=0.22
    )

    out_path = os.path.join(RESULTS_DIR, f"{dataset_key}_{arch_key}_clean_vs_advft.png")
    plt.savefig(out_path, bbox_inches="tight")
    plt.close(fig)
    print(f"  → {out_path}")


def main() -> None:
    for dataset_key, arch_key, ds_label, arch_label in COMPARISONS:
        print(f"Building clean-vs-advft figure for {ds_label} ({arch_label}) …")
        plot_comparison(dataset_key, arch_key, ds_label, arch_label)
    print("Done.")


if __name__ == "__main__":
    main()
