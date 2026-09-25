"""Plot adversarial robustness — per-attack model comparison.

Reads summary CSVs from ``wg_zones_adversarial_robustness_results/`` and
generates a 3×3 panel figure comparing WG / TZ+CZ / PZ performance under
each attack (FGSM / PGD / A-PGD) across Dice, HD95, and ASD.

Usage::

    python experiments/plot_adversarial_robustness.py
"""

from __future__ import annotations

import csv
import os
import sys
from typing import NamedTuple

import matplotlib
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import numpy as np

# ---------------------------------------------------------------------------
# Publication style
# ---------------------------------------------------------------------------

matplotlib.rcParams.update({
    "font.family": "serif",
    "font.serif": ["Times New Roman", "DejaVu Serif"],
    "font.size": 9.5,
    "axes.titlesize": 10.5,
    "axes.labelsize": 9.5,
    "xtick.labelsize": 8,
    "ytick.labelsize": 8,
    "legend.fontsize": 8.5,
    "figure.dpi": 150,
    "savefig.dpi": 300,
    "savefig.bbox": "tight",
    "savefig.facecolor": "white",
    "errorbar.capsize": 2.5,
})

# Colorblind-friendly palette for the three anatomical targets
C_WG    = "#D55E00"   # vermillion / dark orange
C_TZCZ  = "#0072B2"   # deep blue
C_PZ    = "#CC79A7"   # reddish purple

MODEL_STYLES: dict[str, dict] = {
    "WG":    {"color": C_WG,   "marker": "o", "ls": "-",  "lw": 1.8, "ms": 5.5, "mew": 0.5, "mfc": "white"},
    "TZ+CZ": {"color": C_TZCZ, "marker": "s", "ls": "-",  "lw": 1.8, "ms": 5.5, "mew": 0.5, "mfc": "white"},
    "PZ":    {"color": C_PZ,   "marker": "D", "ls": "-",  "lw": 1.8, "ms": 5.0, "mew": 0.5, "mfc": "white"},
}

# ---------------------------------------------------------------------------
# Paths & constants
# ---------------------------------------------------------------------------

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RESULTS_DIR = os.path.join(REPO_ROOT, "results", "wg_zones_adversarial_robustness_results")
FIGURES_DIR = os.path.join(RESULTS_DIR, "figures")

CLASSES = ["WG", "TZ+CZ", "PZ"]
CLASS_DIR = {"WG": "WG", "TZ+CZ": "Zones", "PZ": "Zones"}

ATTACKS = ["fgsm", "pgd", "a_pgd"]
ATTACK_TITLES = {"fgsm": "FGSM", "pgd": "PGD", "a_pgd": "A-PGD"}

METRICS = ["dice", "hd95", "asd"]
METRIC_YLABEL = {
    "dice": "Dice Score  ↑",
    "hd95": "HD95  ↓  (mm)",
    "asd":  "ASD  ↓  (mm)",
}
Y_LIMITS: dict[str, tuple[float | None, float | None]] = {
    "dice": (0.0, 1.02),
    "hd95": (-0.5, None),
    "asd":  (-0.3, None),
}


# ---------------------------------------------------------------------------
# Data
# ---------------------------------------------------------------------------

class CurveRow(NamedTuple):
    epsilon: float
    mean_val: float
    std_val: float


def load_class_data(
    class_name: str, attack: str,
) -> dict[str, list[CurveRow]]:
    """Return {metric: [CurveRow, ...]} for one (class, attack) pair."""
    model_dir = CLASS_DIR[class_name]
    path = os.path.join(RESULTS_DIR, model_dir, f"{attack}_summary.csv")
    out: dict[str, list[CurveRow]] = {m: [] for m in METRICS}
    if not os.path.isfile(path):
        print(f"  [WARN] Missing: {path}", file=sys.stderr)
        return out

    with open(path, newline="") as f:
        for entry in csv.DictReader(f):
            if entry["class"] != class_name:
                continue
            eps = float(entry["epsilon"])
            for m in METRICS:
                out[m].append(CurveRow(
                    epsilon=eps,
                    mean_val=float(entry[f"mean_{m}"]),
                    std_val=float(entry[f"std_{m}"]),
                ))
    return out


# ---------------------------------------------------------------------------
# Plotting
# ---------------------------------------------------------------------------

def _style_panel(ax, metric: str, attack_title: str):
    """Apply consistent axis styling to one panel."""
    ax.set_ylabel(METRIC_YLABEL[metric])
    ax.set_xlabel("Epsilon (ε)")
    ax.set_title(attack_title, fontweight="bold", loc="left", fontsize=10)

    ylo, yhi = Y_LIMITS[metric]
    ax.set_ylim(ylo, yhi)
    ax.set_xlim(-0.005, 0.105)
    ax.xaxis.set_major_locator(mticker.MultipleLocator(0.02))
    ax.grid(True, alpha=0.22, lw=0.35)

    # Subtle reference line at clean (eps=0) value from WG
    for line in ax.get_lines():
        if line.get_label() == "WG" and len(line.get_xdata()) > 0:
            xd, yd = line.get_xdata(), line.get_ydata()
            if xd[0] == 0.0:
                ax.axhline(yd[0], color="gray", lw=0.5, ls=":", alpha=0.35,
                           zorder=0)
            break


def plot_one_panel(ax, attack: str, metric: str):
    """Draw 3 curves (WG / TZ+CZ / PZ) for one attack × metric combination."""
    for class_name in CLASSES:
        data = load_class_data(class_name, attack)
        curves = data.get(metric, [])
        if not curves:
            continue
        curves.sort(key=lambda r: r.epsilon)
        eps = np.array([r.epsilon for r in curves])
        means = np.array([r.mean_val for r in curves])
        stds = np.array([r.std_val for r in curves])

        style = MODEL_STYLES[class_name]
        ax.errorbar(
            eps, means, yerr=stds,
            marker=style["marker"], markersize=style["ms"],
            markerfacecolor=style["mfc"], markeredgewidth=style["mew"],
            markeredgecolor=style["color"],
            color=style["color"], linestyle=style["ls"],
            linewidth=style["lw"],
            capsize=2.5, capthick=0.6, elinewidth=0.7,
            alpha=0.92,
            label=class_name,
            zorder=3,
        )

    # Attack strength rank annotation (top-right corner)
    _add_rank_badge(ax, attack, metric)


def _add_rank_badge(ax, attack: str, metric: str):
    """Annotate which attack is strongest for this metric at max epsilon."""
    # Determine which model drops most at eps=0.1 vs eps=0
    best_name, best_drop = "", -1.0
    for class_name in CLASSES:
        data = load_class_data(class_name, attack)
        curves = data.get(metric, [])
        if len(curves) < 2:
            continue
        curves.sort(key=lambda r: r.epsilon)
        if metric == "dice":
            drop = curves[0].mean_val - curves[-1].mean_val
        else:
            drop = curves[-1].mean_val - curves[0].mean_val
        if drop > best_drop:
            best_drop = drop
            best_name = class_name

    if best_name and best_drop > 0:
        ax.text(
            0.97, 0.97,
            f"Most affected: {best_name}",
            transform=ax.transAxes,
            ha="right", va="top",
            fontsize=6.5, fontstyle="italic",
            color="0.35",
            bbox=dict(boxstyle="round,pad=0.25", fc="white", ec="0.8",
                      alpha=0.75, lw=0.4),
        )


# ---------------------------------------------------------------------------
# Figure construction
# ---------------------------------------------------------------------------

def build_figure() -> plt.Figure:
    """3×3 grid: rows = metrics, columns = attacks."""
    n_rows, n_cols = 3, 3
    fig, axes = plt.subplots(
        n_rows, n_cols,
        figsize=(12.5, 9.8),
        sharex="col",
    )

    for col, attack in enumerate(ATTACKS):
        for row, metric in enumerate(METRICS):
            ax = axes[row, col]
            plot_one_panel(ax, attack, metric)
            _style_panel(ax, metric, ATTACK_TITLES[attack])

    # Metric group labels on right edge
    for row, metric in enumerate(METRICS):
        label = METRIC_YLABEL[metric]
        axes[row, -1].yaxis.set_label_position("right")
        axes[row, -1].set_ylabel(label, fontweight="bold", fontsize=10,
                                 labelpad=8)
        for col in range(n_cols - 1):
            axes[row, col].set_ylabel("")

    # Shared legend at top
    handles, labels = axes[0, 0].get_legend_handles_labels()
    fig.legend(
        handles, labels,
        loc="upper center",
        ncol=3,
        fontsize=9.5,
        frameon=True,
        fancybox=False,
        edgecolor="0.65",
        bbox_to_anchor=(0.5, 1.025),
        handlelength=1.8,
        handletextpad=0.6,
        columnspacing=1.5,
    )

    fig.subplots_adjust(
        left=0.07, right=0.93, bottom=0.07, top=0.92,
        wspace=0.30, hspace=0.34,
    )

    return fig


def build_single_metric_figure(metric: str) -> plt.Figure:
    """1×3 figure: one metric row, three attack columns."""
    n_rows, n_cols = 1, 3
    fig, axes = plt.subplots(
        n_rows, n_cols,
        figsize=(12.5, 4.6),
        sharey=True,
    )

    for col, attack in enumerate(ATTACKS):
        ax = axes[col]
        plot_one_panel(ax, attack, metric)
        _style_panel(ax, metric, ATTACK_TITLES[attack])

    # Shared legend at top
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(
        handles, labels,
        loc="upper center",
        ncol=3,
        fontsize=10,
        frameon=True,
        fancybox=False,
        edgecolor="0.65",
        bbox_to_anchor=(0.5, 1.04),
        handlelength=1.8,
        handletextpad=0.6,
        columnspacing=1.5,
    )

    fig.subplots_adjust(
        left=0.08, right=0.97, bottom=0.14, top=0.86,
        wspace=0.12,
    )

    return fig


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def main():
    os.makedirs(FIGURES_DIR, exist_ok=True)
    print("Generating adversarial robustness figures...")

    # Combined 3×3
    fig_all = build_figure()
    for ext in ("png", "pdf"):
        path = os.path.join(FIGURES_DIR, f"adversarial_robustness_summary.{ext}")
        fig_all.savefig(path)
        print(f"  {path}")
    plt.close(fig_all)

    # Per-metric 1×3 figures
    for metric in METRICS:
        fig = build_single_metric_figure(metric)
        for ext in ("png", "pdf"):
            path = os.path.join(FIGURES_DIR, f"adversarial_robustness_{metric}.{ext}")
            fig.savefig(path)
            print(f"  {path}")
        plt.close(fig)

    print("Done.")


if __name__ == "__main__":
    main()
