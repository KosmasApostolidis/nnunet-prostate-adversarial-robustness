"""Grouped bar charts of the budget-averaged tables: one figure per metric.

Reads ``attack_comparison_budget_avg_{wg,zones}.csv`` (written by
``build_attack_budget_avg_tables.py``): one group per column (model, or
model x class for the zones), one bar per attack in the table's row order,
with the clean value drawn as a black dash across the group.

Usage::

    PYTHONPATH=src python experiments/plot_attack_budget_avg.py \
        --input-dir results/blade_attack/attack_comparison
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parent))
from plot_attack_comparison import ARM_COLOUR  # noqa: E402
from run_twelve_arm_wg_campaign import ARM_LABELS, METRIC_LABELS, METRICS  # noqa: E402

LABEL_TO_ARM = {v: k for k, v in ARM_LABELS.items()}
TITLES = {"wg": "Whole gland", "zones": "Prostate zones"}


def plot_metric(table: pd.DataFrame, metric: str, name: str, target: Path) -> None:
    block = table[table["metric"] == metric].set_index("attack").drop(columns="metric")
    clean = block.loc["clean"]
    arms = block.drop(index="clean")
    groups = list(block.columns)
    n_arms = len(arms)
    width = 0.8 / n_arms
    x = np.arange(len(groups))
    fig, ax = plt.subplots(figsize=(max(7.2, 1.3 * len(groups) + 2), 4.8), layout="constrained")
    for i, (label, row) in enumerate(arms.iterrows()):
        colour = ARM_COLOUR.get(LABEL_TO_ARM.get(label, ""), "#777777")
        ax.bar(x - 0.4 + width * (i + 0.5), row.values, width, color=colour, label=label)
    for xi, g in zip(x, groups):
        ax.hlines(clean[g], xi - 0.4, xi + 0.4, colors="black", linestyles="--", linewidth=1.2)
    ax.hlines([], [], [], colors="black", linestyles="--", label="clean")
    ax.set_xticks(x, groups, rotation=30 if len(groups) > 4 else 0, ha="right" if len(groups) > 4 else "center")
    ax.set_ylabel(METRIC_LABELS[metric])
    ax.grid(axis="y", alpha=0.3)
    if metric == "dice":
        ax.set_ylim(0, 1.0)
        ax.legend(fontsize=7, loc="lower left", ncol=3)
    else:
        ax.set_yscale("log")
        ax.legend(fontsize=7, loc="upper left", ncol=3)
    ax.set_title(
        f"{TITLES[name]} -- {METRIC_LABELS[metric]} per attack\n"
        r"(mean over folds and attack budgets $\varepsilon$ = 0.02--0.1)",
        fontsize=11,
        fontweight="bold",
    )
    for suffix in (".png", ".pdf"):
        fig.savefig(target.with_suffix(suffix), dpi=600, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved {target.with_suffix('.png')} / .pdf")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument(
        "--input-dir", type=Path, default=Path("results/blade_attack/attack_comparison")
    )
    args = parser.parse_args()
    for name in ("wg", "zones"):
        csv = args.input_dir / f"attack_comparison_budget_avg_{name}.csv"
        if not csv.is_file():
            sys.exit(f"table not found: {csv}")
        table = pd.read_csv(csv)
        for metric in METRICS:
            plot_metric(table, metric, name, csv.with_name(f"{csv.stem}_{metric}"))


if __name__ == "__main__":
    main()
