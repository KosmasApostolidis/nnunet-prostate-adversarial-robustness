"""Two figures from the budget-averaged median tables.

``analyze_main_comparison_stats.py`` writes two CSVs describing the same
quantity -- each case's metric averaged over the attack budgets, then
summarised across cases. This script turns them into the two figures they
support, reading only the CSVs so the plots can be redrawn without repeating
the analysis.

  1. **Effect figure** (``*_median_effects.*``), from the medians table.
     A dot for each arm's median with its exact 95% confidence interval and
     its interquartile range, ordered most- to least-damaging. Answers "how
     much damage, and how certain".

  2. **Contrast figure** (``*_median_contrasts.*``), from the pairwise table.
     A matrix of Hodges-Lehmann median shifts, every arm against every other,
     with non-significant cells struck out. Answers "which arms actually
     differ, and by how much".

Usage::

    PYTHONPATH=src python experiments/plot_median_comparison_figures.py \
        --medians results/blade_attack/whole_gland/native_per_eps/main_comparison_wg_budget_avg_medians.csv
"""

from __future__ import annotations

import argparse
import os

import numpy as np
import pandas as pd

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import TwoSlopeNorm

METRIC_LABELS = {"dice": "Dice", "hd95": "HD95 (mm)", "asd": "ASD (mm)"}
# Dice falls under attack, the distances rise. Ordering on the median times
# this sign puts the strongest attack first for every metric.
METRIC_DIRECTION = {"dice": -1, "hd95": +1, "asd": +1}
LOG_METRICS = ("hd95", "asd")


def damage_order(medians: pd.DataFrame, metric: str) -> list[str]:
    """Arms most- to least-damaging by median, for one metric."""
    block = medians[medians.metric == metric]
    key = METRIC_DIRECTION[metric] * block["median"]
    return list(block.assign(_k=key).sort_values("_k", ascending=False)["arm"])


def plot_effects(medians: pd.DataFrame, output_dir: str, stem: str) -> str:
    """Median, 95% CI and IQR per arm -- one panel per metric."""
    metrics = [m for m in METRIC_LABELS if m in set(medians.metric)]
    fig, axes = plt.subplots(
        1, len(metrics), figsize=(5.6 * len(metrics), 6.4), squeeze=False
    )

    for ax, metric in zip(axes[0], metrics):
        block = medians[medians.metric == metric].set_index("arm")
        order = damage_order(medians, metric)
        ys = np.arange(len(order))[::-1]  # strongest at the top

        for y, arm in zip(ys, order):
            row = block.loc[arm]
            # IQR first, as the wide pale bar the CI sits inside.
            ax.plot(
                [row.q1, row.q3], [y, y], color="#BBBBBB", linewidth=7,
                solid_capstyle="butt", zorder=1,
            )
            ax.plot(
                [row.ci95_low, row.ci95_high], [y, y], color="#333333",
                linewidth=2.4, solid_capstyle="butt", zorder=2,
            )
            ax.plot(
                row["median"], y, "o", color="#CC3311", markersize=7,
                markeredgecolor="black", markeredgewidth=0.7, zorder=3,
            )

        ax.set_yticks(ys)
        ax.set_yticklabels([block.loc[a, "label"] for a in order], fontsize=10)
        ax.set_xlabel(METRIC_LABELS[metric], fontsize=12)
        ax.set_ylim(-0.8, len(order) - 0.2)
        ax.grid(axis="x", alpha=0.3, linewidth=0.6)
        if metric in LOG_METRICS:
            ax.set_xscale("log")

    handles = [
        plt.Line2D([], [], color="#CC3311", marker="o", linestyle="none",
                   markeredgecolor="black", markersize=7, label="median"),
        plt.Line2D([], [], color="#333333", linewidth=2.4, label="95% CI (order statistics)"),
        plt.Line2D([], [], color="#BBBBBB", linewidth=7, label="interquartile range"),
    ]
    fig.legend(handles=handles, loc="lower center", ncol=3, fontsize=11, frameon=False)

    n_cases = int(medians.n_cases.iloc[0])
    fig.suptitle(
        "Attack strength on the budget-averaged score\n"
        f"whole gland, {n_cases:,} cases; per case the metric is averaged over "
        "the five attack budgets, then summarised across cases",
        fontsize=14,
    )
    fig.tight_layout(rect=(0, 0.06, 1, 0.93))

    base = os.path.join(output_dir, f"{stem}_median_effects")
    fig.savefig(f"{base}.png", dpi=200, bbox_inches="tight")
    fig.savefig(f"{base}.pdf", bbox_inches="tight")
    plt.close(fig)
    return f"{base}.png"


def _shift_matrix(
    pairs: pd.DataFrame, metric: str, order: list[str]
) -> tuple[np.ndarray, np.ndarray]:
    """Antisymmetric HL-shift matrix and its significance mask.

    ``hl_shift`` is signed so a positive value means ``arm_a`` did more damage;
    reading the pair the other way flips it.
    """
    index = {arm: i for i, arm in enumerate(order)}
    n = len(order)
    shift = np.full((n, n), np.nan)
    sig = np.zeros((n, n), dtype=bool)
    for _, row in pairs[pairs.metric == metric].iterrows():
        i, j = index[row.arm_a], index[row.arm_b]
        shift[i, j], shift[j, i] = row.hl_shift, -row.hl_shift
        sig[i, j] = sig[j, i] = bool(row.significant)
    return shift, sig


def plot_contrasts(
    medians: pd.DataFrame, pairs: pd.DataFrame, output_dir: str, stem: str
) -> str:
    """Every pairwise median shift as a matrix -- one panel per metric."""
    metrics = [m for m in METRIC_LABELS if m in set(pairs.metric)]
    fig, axes = plt.subplots(
        1, len(metrics), figsize=(7.4 * len(metrics), 8.0), squeeze=False
    )

    for ax, metric in zip(axes[0], metrics):
        order = damage_order(medians, metric)
        labels = list(
            medians[medians.metric == metric].set_index("arm").loc[order, "label"]
        )
        shift, sig = _shift_matrix(pairs, metric, order)

        limit = float(np.nanmax(np.abs(shift)))
        image = ax.imshow(
            shift,
            cmap="RdBu_r",
            norm=TwoSlopeNorm(vmin=-limit, vcenter=0.0, vmax=limit),
        )
        for i in range(len(order)):
            for j in range(len(order)):
                if i == j:
                    ax.add_patch(
                        plt.Rectangle(
                            (j - 0.5, i - 0.5), 1, 1, facecolor="#EEEEEE",
                            edgecolor="white",
                        )
                    )
                    continue
                value = shift[i, j]
                text = f"{value:.3g}" if abs(value) >= 0.01 else f"{value:.0e}"
                # A struck-through cell is a pair the test could not separate.
                ax.text(
                    j, i, text, ha="center", va="center", fontsize=6.6,
                    color="black" if abs(value) < 0.55 * limit else "white",
                )
                if not sig[i, j]:
                    ax.plot(
                        [j - 0.42, j + 0.42], [i, i], color="black",
                        linewidth=1.1, zorder=4,
                    )

        ax.set_xticks(range(len(order)))
        ax.set_yticks(range(len(order)))
        ax.set_xticklabels(labels, rotation=90, fontsize=9)
        ax.set_yticklabels(labels, fontsize=9)
        ax.set_title(METRIC_LABELS[metric], fontsize=13)
        ax.set_xticks(np.arange(-0.5, len(order), 1), minor=True)
        ax.set_yticks(np.arange(-0.5, len(order), 1), minor=True)
        ax.grid(which="minor", color="white", linewidth=1.0)
        ax.tick_params(which="minor", length=0)
        bar = fig.colorbar(image, ax=ax, fraction=0.046, pad=0.03)
        bar.set_label("row minus column, extra damage", fontsize=10)

    n_pairs = len(pairs[pairs.metric == metrics[0]])
    n_ns = int((~pairs.significant).sum())
    verdict = (
        "every pair separated"
        if n_ns == 0
        else f"{n_ns} pairs did not separate and are struck through"
    )
    fig.suptitle(
        "Pairwise median shift between attacks (Hodges-Lehmann estimator)\n"
        f"{n_pairs} pairs per metric, paired Wilcoxon with Benjamini-Hochberg "
        f"correction -- {verdict}. Arms are ordered most- to least-damaging by "
        "median, so a blue cell above the diagonal is a pair the pairing "
        "reverses.",
        fontsize=13,
    )
    fig.tight_layout(rect=(0, 0, 1, 0.92))

    base = os.path.join(output_dir, f"{stem}_median_contrasts")
    fig.savefig(f"{base}.png", dpi=200, bbox_inches="tight")
    fig.savefig(f"{base}.pdf", bbox_inches="tight")
    plt.close(fig)
    return f"{base}.png"


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Draw the effect and contrast figures from the budget-averaged "
            "median tables written by analyze_main_comparison_stats.py."
        )
    )
    parser.add_argument(
        "--medians",
        default=os.path.join(
            "results",
            "blade_attack",
            "whole_gland",
            "native_per_eps",
            "main_comparison_wg_budget_avg_medians.csv",
        ),
        help="Per-arm median table.",
    )
    parser.add_argument(
        "--pairwise",
        default=None,
        help="Pairwise table (default: the medians path with _medians -> _pairwise).",
    )
    parser.add_argument(
        "--output-dir",
        default=None,
        help="Where to write the figures (default: the medians file's directory).",
    )
    args = parser.parse_args()

    pairwise_path = args.pairwise or args.medians.replace(
        "_medians.csv", "_pairwise.csv"
    )
    output_dir = args.output_dir or os.path.dirname(os.path.abspath(args.medians))
    stem = os.path.basename(args.medians).replace("_budget_avg_medians.csv", "")

    medians = pd.read_csv(args.medians)
    pairs = pd.read_csv(pairwise_path)

    print(f"medians  : {args.medians} ({len(medians)} rows)")
    print(f"pairwise : {pairwise_path} ({len(pairs)} rows)")
    print(plot_effects(medians, output_dir, stem))
    print(plot_contrasts(medians, pairs, output_dir, stem))


if __name__ == "__main__":
    main()
