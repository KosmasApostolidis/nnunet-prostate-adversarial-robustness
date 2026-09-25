"""Adversarial robustness — cross-model comparison figures.

Reads per-(dataset, trainer) CSVs produced by
``adv_rob_eval_nnunet_nnunetrecenc.py`` and builds one figure per dataset
(WG, Zones) combining all four architecture variants: a 3×3 grid of
metric (rows: Dice/HD95/ASD) × attack (columns: FGSM/PGD-BCE/APGD-BCE)
panels, each showing macro mean ± SEM across folds vs epsilon (shaded
band), one line per model.

Usage::

    python experiments/plot_model_comparison.py
"""

from __future__ import annotations

import csv
import os
from collections import defaultdict
from typing import Dict, List

import matplotlib
import numpy as np

matplotlib.use("Agg")
import matplotlib.pyplot as plt

# ---------------------------------------------------------------------------
# Publication style
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
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
RESULTS_DIR = os.path.join(SCRIPT_DIR, "adv_rob_eval_results")

EPSILONS: List[float] = [0.0, 0.02, 0.04, 0.06, 0.08, 0.10]

DATASETS: Dict[str, str] = {"wg": "WG", "zones": "Zones"}

TRAINER_DISPLAY: Dict[str, str] = {
    "unet": "UNet",
    "resenc_l": "ResEnc-L",
    "resenc_m": "ResEnc-M",
    "resenc_xl": "ResEnc-XL",
}

# Okabe-Ito colorblind-safe palette (consistent with plot_adversarial_robustness.py)
MODEL_STYLE: Dict[str, dict] = {
    "unet": {"color": "#0072B2", "marker": "o"},
    "resenc_l": {"color": "#D55E00", "marker": "s"},
    "resenc_m": {"color": "#009E73", "marker": "^"},
    "resenc_xl": {"color": "#CC79A7", "marker": "D"},
}

METRICS = ["dice", "hd95", "asd"]
METRIC_LABELS = {"dice": "Dice", "hd95": "HD95 (mm)", "asd": "ASD (mm)"}

ATTACK_ORDER = ["FGSM", "PGD", "APGD"]
ATTACK_LABELS = {"FGSM": "FGSM", "PGD": "PGD-BCE", "APGD": "APGD-BCE"}


# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------
def load_trainer_csv(path: str) -> Dict[str, Dict[float, Dict[str, List[float]]]]:
    """Return data[attack][epsilon][metric] = list of per-fold mean values."""
    data: Dict[str, Dict[float, Dict[str, List[float]]]] = defaultdict(
        lambda: defaultdict(lambda: defaultdict(list))
    )
    with open(path, newline="") as f:
        for row in csv.DictReader(f):
            eps = float(row["epsilon"])
            for m in METRICS:
                val = float(row[f"{m}_mean"])
                if np.isfinite(val):
                    data[row["attack"]][eps][m].append(val)
    return data


# ---------------------------------------------------------------------------
# Figure construction
# ---------------------------------------------------------------------------
def plot_dataset_comparison(dataset_key: str) -> None:
    ds_label = DATASETS[dataset_key]
    per_trainer: Dict[str, dict] = {}

    for trainer_key in TRAINER_DISPLAY:
        csv_path = os.path.join(RESULTS_DIR, f"{dataset_key}_{trainer_key}_adv_rob.csv")
        if not os.path.isfile(csv_path):
            print(f"  [SKIP] {csv_path} not found")
            continue
        per_trainer[trainer_key] = load_trainer_csv(csv_path)

    if not per_trainer:
        print(f"  [WARN] No CSVs found for dataset '{dataset_key}'")
        return

    fig, axes = plt.subplots(
        len(METRICS),
        len(ATTACK_ORDER),
        figsize=(15, 11.6),
        sharex="col",
        sharey="row",
    )
    fig.suptitle(
        f"Adversarial Robustness — {ds_label}", fontsize=18, fontweight="bold", y=1.05
    )
    fig.text(
        0.5,
        1.01,
        "All metrics vs. perturbation budget, by attack",
        ha="center",
        fontsize=12,
        style="italic",
        color="0.35",
    )

    for row, metric in enumerate(METRICS):
        for col, atk in enumerate(ATTACK_ORDER):
            ax = axes[row][col]
            for trainer_key, data in per_trainer.items():
                means, sems = [], []
                for eps in EPSILONS:
                    arr = np.asarray(
                        data.get(atk, {}).get(eps, {}).get(metric, []), dtype=np.float64
                    )
                    if arr.size == 0:
                        means.append(np.nan)
                        sems.append(0.0)
                        continue
                    means.append(float(arr.mean()))
                    sems.append(
                        float(arr.std(ddof=1) / np.sqrt(arr.size))
                        if arr.size > 1
                        else 0.0
                    )
                means_arr = np.array(means)
                sems_arr = np.array(sems)
                style = MODEL_STYLE[trainer_key]

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
                    label=TRAINER_DISPLAY[trainer_key],
                )

            if row == 0:
                ax.set_title(ATTACK_LABELS[atk], fontweight="bold", pad=10)
            if row == len(METRICS) - 1:
                ax.set_xlabel("ε (L∞)")
            if col == 0:
                ax.set_ylabel(METRIC_LABELS[metric])
            ax.set_xticks(EPSILONS)
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
        ncol=len(per_trainer),
        fontsize=13,
        frameon=False,
        bbox_to_anchor=(0.5, 0.995),
    )

    fig.subplots_adjust(
        left=0.06, right=0.98, bottom=0.05, top=0.90, wspace=0.12, hspace=0.22
    )

    out_path = os.path.join(RESULTS_DIR, f"{dataset_key}_model_comparison.png")
    plt.savefig(out_path, bbox_inches="tight")
    plt.close(fig)
    print(f"  → {out_path}")


def main() -> None:
    os.makedirs(RESULTS_DIR, exist_ok=True)
    for dataset_key in DATASETS:
        print(f"Building model-comparison figure for dataset '{dataset_key}' …")
        plot_dataset_comparison(dataset_key)
    print("Done.")


if __name__ == "__main__":
    main()
