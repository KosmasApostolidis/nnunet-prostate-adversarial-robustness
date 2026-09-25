"""Per-architecture metric-vs-epsilon figures for the arms stronger than the PGD family.

Reads the same per-tree summary CSVs as ``plot_architecture_comparison.py``
(one ``<arch>=<path>`` pair per architecture) and writes, for the chosen
class, one figure per architecture and metric (Dice, HD95, ASD) with one line
per arm in ``BEYOND_PGD``, mean +/- SEM across folds against epsilon.

``BEYOND_PGD`` is every arm whose budget-averaged Dice lies below the PGD
family (PGD, SegPGD, CosPGD) in
``results/blade_attack/attack_comparison/attack_comparison_budget_avg_{wg,zones}.csv``.
The ranking is the same for all four architectures and all three classes, so
the set is fixed here rather than recomputed per figure.

Usage::

    PYTHONPATH=src python experiments/plot_beyond_pgd_per_model.py \
        --summary unet=results/blade_attack/whole_gland/native_per_eps/main_comparison_wg_summary.csv \
        --summary resenc_m=... --summary resenc_l=... --summary resenc_xl=... \
        --output-dir results/blade_attack/attack_comparison/beyond_pgd/WG

Writes ``<output-dir>/<arch>_<metric>_beyond_pgd.png``.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt  # noqa: E402
import pandas as pd  # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parent))
from plot_architecture_comparison import ARCH_STYLE, parse_summaries  # noqa: E402
from plot_attack_comparison import ARM_COLOUR  # noqa: E402
from run_twelve_arm_wg_campaign import (  # noqa: E402
    ARM_LABELS,
    METRIC_LABELS,
    METRICS,
)

# Strongest first (lowest budget-averaged whole-gland Dice on nnU-Net).
BEYOND_PGD = (
    "blade",
    "blade_mm",
    "blade_boundary",
    "blade1",
    "auto_pgd_r3",
    "auto_pgd",
    "blade_frontier",
    "sea",
)
MARKERS = ("o", "s", "^", "D", "v", "P", "X", "*")


def plot_model_metric(
    table: pd.DataFrame, arch: str, metric: str, target: Path, label: str
) -> None:
    fig, ax = plt.subplots(figsize=(5.4, 4.0), layout="constrained")
    for arm, marker in zip(BEYOND_PGD, MARKERS, strict=True):
        rows = table[(table["attack"] == arm) & (table["metric"] == metric)]
        if rows.empty:
            sys.exit(f"{arch}: no {metric} rows for arm {arm!r}")
        rows = rows.sort_values("epsilon")
        ax.errorbar(
            rows["epsilon"],
            rows["mean"],
            yerr=rows["sem"],
            color=ARM_COLOUR[arm],
            marker=marker,
            markersize=4,
            capsize=2.5,
            linewidth=1.4,
            label=ARM_LABELS[arm],
        )
    ax.grid(alpha=0.3)
    if metric == "dice":
        ax.set_ylim(-0.02, 1.0)
    else:
        ax.set_yscale("log")
    ax.set_xlabel(r"$\varepsilon$ ($L_\infty$, after z-score normalization)")
    ax.set_ylabel(METRIC_LABELS[metric])
    ax.legend(fontsize=7, loc="best", ncol=2)
    subject = "Whole-gland" if label == "WG" else label
    ax.set_title(
        f"{ARCH_STYLE[arch][0]}: {subject} {METRIC_LABELS[metric]}, "
        "arms stronger than the PGD family\n(mean +/- SEM across folds)",
        fontsize=10,
        fontweight="bold",
    )
    fig.savefig(target, dpi=300, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument(
        "--summary",
        action="append",
        required=True,
        metavar="ARCH=PATH",
        help="architecture name and its summary CSV; repeatable",
    )
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument(
        "--label",
        default="WG",
        help="foreground class as named in the summary's class column "
        "(WG, TZ+CZ or PZ)",
    )
    args = parser.parse_args()
    summaries = parse_summaries(args.summary, args.label)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    for arch, table in summaries.items():
        for metric in METRICS:
            target = args.output_dir / f"{arch}_{metric}_beyond_pgd.png"
            plot_model_metric(table, arch, metric, target, args.label)
    print(f"wrote {len(summaries) * len(METRICS)} figures to {args.output_dir}")


if __name__ == "__main__":
    main()
