"""One figure per attack arm and metric, one line per architecture.

Companion to ``plot_architecture_comparison.py``, which puts every arm on one
sheet. This writes the same curves as standalone figures, one per (arm, metric):
Dice, HD95 or ASD against epsilon for vanilla nnU-Net and the three ResEnc
variants, mean +/- SEM across folds. It reads the same per-tree summary CSVs
and reuses that script's parsing, so the two cannot disagree.

Usage::

    PYTHONPATH=src python experiments/plot_attack_architecture_figures.py \
        --summary unet=results/blade_attack/whole_gland/native_per_eps/main_comparison_wg_summary.csv \
        --summary resenc_m=results/blade_attack/whole_gland/native_per_eps_resenc_m/thirteen_arm_wg_summary.csv \
        --summary resenc_l=... --summary resenc_xl=... \
        --output-dir results/blade_attack/attack_comparison/per_attack/WG

Writes ``<output-dir>/<arm>/<metric>/<arm>_<metric>_nnunet_resenc_m_l_xl.png`` for every arm
present in all summaries and every metric. ``--label`` picks the foreground
class (``WG``, ``TZ+CZ`` or ``PZ``); run once per class.
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
from plot_architecture_comparison import (  # noqa: E402
    ARCH_STYLE,
    common_arms,
    parse_summaries,
)
from run_twelve_arm_wg_campaign import (  # noqa: E402
    ARM_LABELS,
    METRIC_LABELS,
    METRICS,
)

FILE_SUFFIX = "nnunet_resenc_m_l_xl"


def plot_arm_metric(
    summaries: dict[str, pd.DataFrame],
    arm: str,
    metric: str,
    target: Path,
    label: str,
) -> None:
    fig, ax = plt.subplots(figsize=(5.0, 3.8), layout="constrained")
    for arch, table in summaries.items():
        rows = table[(table["attack"] == arm) & (table["metric"] == metric)]
        rows = rows.sort_values("epsilon")
        name, colour, marker = ARCH_STYLE[arch]
        ax.errorbar(
            rows["epsilon"],
            rows["mean"],
            yerr=rows["sem"],
            color=colour,
            marker=marker,
            markersize=4,
            capsize=2.5,
            linewidth=1.4,
            label=name,
        )
    ax.grid(alpha=0.3)
    if metric == "dice":
        ax.set_ylim(-0.02, 1.0)
    else:
        ax.set_yscale("log")
    ax.set_xlabel(r"$\varepsilon$ ($L_\infty$, after z-score normalization)")
    ax.set_ylabel(METRIC_LABELS[metric])
    ax.legend(fontsize=8, loc="best")
    subject = "Whole-gland" if label == "WG" else label
    ax.set_title(
        f"{ARM_LABELS[arm]}: {subject} {METRIC_LABELS[metric]} by architecture\n"
        "(mean +/- SEM across folds)",
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
    parser.add_argument(
        "--output-dir",
        type=Path,
        required=True,
        help="directory receiving <arm>/<metric>/<arm>_<metric>_nnunet_resenc_m_l_xl.png",
    )
    parser.add_argument(
        "--label",
        default="WG",
        help="foreground class to plot, as named in the summary's class column "
        "(WG, TZ+CZ or PZ)",
    )
    args = parser.parse_args()
    summaries = parse_summaries(args.summary, args.label)
    arms = common_arms(summaries)
    if not arms:
        sys.exit("no attack arm is present in every summary")
    args.output_dir.mkdir(parents=True, exist_ok=True)
    for arm in arms:
        for metric in METRICS:
            target = args.output_dir / arm / metric / f"{arm}_{metric}_{FILE_SUFFIX}.png"
            target.parent.mkdir(parents=True, exist_ok=True)
            plot_arm_metric(summaries, arm, metric, target, args.label)
    print(f"wrote {len(arms) * len(METRICS)} figures to {args.output_dir}")


if __name__ == "__main__":
    main()
