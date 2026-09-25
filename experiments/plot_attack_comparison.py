"""One figure per metric per foreground class, all attack arms on one axis.

Reads the campaign summary CSVs (``attack,class,metric,epsilon,mean,sd_folds,
sem,n_folds``) and draws, for every (class, metric) pair, the mean +/- SEM
across folds against epsilon with one line per arm -- the same styling and the
same numbers as the per-tree summary figure, just one panel per file so each
metric/class can be placed on its own.

Usage::

    PYTHONPATH=src python experiments/plot_attack_comparison.py \
        --summary results/blade_attack/whole_gland/native_per_eps/main_comparison_wg_summary.csv \
        --summary results/blade_attack/zones_confined_mc/campaign/thirteen_arm_zones_summary.csv \
        --output-dir results/blade_attack/attack_comparison

Every class found in the summaries is plotted (WG from the first, TZ+CZ and
PZ from the second); ``--classes`` restricts them. Files are named
``attack_comparison_<class>_<metric>.png/.pdf`` with ``+`` dropped from the
class name.
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
from run_twelve_arm_wg_campaign import (  # noqa: E402
    ARM_LABELS,
    ARM_STYLE,
    ARMS,
    METRIC_LABELS,
    METRICS,
)

# ARM_STYLE reuses Okabe-Ito colours across arms (e.g. Auto-PGD, SEA and
# Auto-PGD x3 all green, told apart by line style). With 13 lines on one axis
# that is hard to read, so this figure gives every arm its own colour and keeps
# ARM_STYLE's marker and line style.
ARM_COLOUR = {
    "fgsm": "#999999",
    "pgd": "#0072B2",
    "auto_pgd": "#009E73",
    "segpgd": "#E69F00",
    "cospgd": "#56B4E9",
    "dag": "#D55E00",
    "sea": "#882255",
    "blade1": "#CC79A7",
    "blade": "#332288",
    "blade_mm": "#000000",
    "auto_pgd_r3": "#8B4513",
    "blade_boundary": "#AA4499",
    "blade_frontier": "#DDCC77",
}

CLASS_TITLES = {
    "WG": "Whole gland",
    "TZ+CZ": "Transition + central zone",
    "PZ": "Peripheral zone",
}


def load_summaries(paths: list[str]) -> pd.DataFrame:
    frames = []
    for path in paths:
        if not Path(path).is_file():
            sys.exit(f"summary not found: {path}")
        frames.append(pd.read_csv(path))
    return pd.concat(frames, ignore_index=True)


def plot_one(
    table: pd.DataFrame, class_name: str, metric: str, target: Path, arch: str
) -> None:
    rows_all = table[(table["class"] == class_name) & (table["metric"] == metric)]
    arms = [a for a in ARMS if a in set(rows_all["attack"])]
    n_folds = int(rows_all["n_folds"].max())
    fig, ax = plt.subplots(figsize=(7.2, 4.8), layout="constrained")
    for arm in arms:
        rows = rows_all[rows_all["attack"] == arm].sort_values("epsilon")
        _, marker, linestyle = ARM_STYLE[arm]
        colour = ARM_COLOUR.get(arm, ARM_STYLE[arm][0])
        ax.errorbar(
            rows["epsilon"],
            rows["mean"],
            yerr=rows["sem"],
            color=colour,
            marker=marker,
            linestyle=linestyle,
            markersize=5,
            capsize=3,
            linewidth=1.6,
            label=ARM_LABELS[arm],
        )
    ax.set_xlabel(r"$\varepsilon$ ($L_\infty$, after z-score normalization)")
    ax.set_ylabel(METRIC_LABELS[metric])
    ax.grid(alpha=0.3)
    if metric == "dice":
        ax.set_ylim(-0.02, 1.0)
        ax.legend(fontsize=8, loc="lower left", ncol=2)
    else:
        ax.set_yscale("log")
        ax.legend(fontsize=8, loc="upper left", ncol=2)
    ax.set_title(
        f"{CLASS_TITLES.get(class_name, class_name)} -- {METRIC_LABELS[metric]} under "
        f"{len(arms)} adversarial attacks\n({arch}, mean +/- SEM across {n_folds} folds)",
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
        "--summary", action="append", required=True, help="summary CSV; repeatable"
    )
    parser.add_argument(
        "--classes",
        nargs="+",
        default=None,
        help="classes to plot (default: all found)",
    )
    parser.add_argument(
        "--arch", default="nnU-Net", help="architecture name for the titles"
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("results/blade_attack/attack_comparison"),
    )
    args = parser.parse_args()
    table = load_summaries(args.summary)
    classes = args.classes or [
        c for c in ("WG", "TZ+CZ", "PZ") if c in set(table["class"])
    ]
    missing = [c for c in classes if c not in set(table["class"])]
    if missing:
        sys.exit(f"classes not present in the summaries: {missing}")
    args.output_dir.mkdir(parents=True, exist_ok=True)
    for class_name in classes:
        for metric in METRICS:
            stem = f"attack_comparison_{class_name.replace('+', '')}_{metric}"
            plot_one(table, class_name, metric, args.output_dir / stem, args.arch)


if __name__ == "__main__":
    main()
