"""Cross-architecture overlay of the 13-arm whole-gland campaign.

One figure per metric (Dice, HD95, ASD); one panel per attack arm; one line
per architecture (vanilla nnU-Net and the three ResEnc variants), mean +/- SEM
across folds against epsilon. Reads only the per-tree summary CSVs written by
``run_twelve_arm_wg_campaign.py`` (``main_comparison_wg_summary.csv`` for the
vanilla tree, ``thirteen_arm_wg_summary.csv`` for the ResEnc trees), which
share one schema, so the figure cannot disagree with the tables.

Usage::

    PYTHONPATH=src python experiments/plot_architecture_comparison.py \
        --summary unet=results/blade_attack/whole_gland/native_per_eps/main_comparison_wg_summary.csv \
        --summary resenc_m=results/blade_attack/whole_gland/native_per_eps_resenc_m/thirteen_arm_wg_summary.csv \
        --output results/blade_attack/whole_gland/architecture_comparison_wg

Writes ``<output>_<metric>.png`` and ``.pdf`` for each metric. Every summary
named on the command line must exist; a missing one is an error, not a skip.
``--label`` picks the foreground class (default ``WG``; ``TZ+CZ`` or ``PZ``
for the zones summaries, one figure set per call).
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
    ARMS,
    METRIC_LABELS,
    METRICS,
)

# Okabe-Ito; the vanilla baseline is black so the three ResEnc variants read as
# departures from it.
ARCH_STYLE = {
    "unet": ("nnU-Net", "#000000", "o"),
    "resenc_m": ("ResEnc-M", "#0072B2", "s"),
    "resenc_l": ("ResEnc-L", "#E69F00", "^"),
    "resenc_xl": ("ResEnc-XL", "#009E73", "D"),
}
N_COLS = 5


def parse_summaries(specs: list[str], label: str) -> dict[str, pd.DataFrame]:
    """``name=path`` pairs -> {name: rows of that summary for ``label``}."""
    out: dict[str, pd.DataFrame] = {}
    for spec in specs:
        name, _, path = spec.partition("=")
        if name not in ARCH_STYLE or not path:
            sys.exit(
                f"--summary expects <arch>=<path> with arch in {sorted(ARCH_STYLE)}: {spec!r}"
            )
        if not Path(path).is_file():
            sys.exit(f"summary not found: {path}")
        table = pd.read_csv(path)
        rows = table[table["class"] == label]
        if rows.empty:
            sys.exit(f"no rows with class {label!r} in {path}")
        out[name] = rows
    if not out:
        sys.exit("at least one --summary is required")
    return out


def common_arms(summaries: dict[str, pd.DataFrame]) -> list[str]:
    """Arms present in every summary, in campaign order."""
    present = [set(t["attack"]) for t in summaries.values()]
    return [a for a in ARMS if all(a in s for s in present)]


def plot_metric(
    summaries: dict[str, pd.DataFrame],
    arms: list[str],
    metric: str,
    target: Path,
    label: str = "WG",
) -> None:
    n_rows = -(-len(arms) // N_COLS)
    fig, axes = plt.subplots(
        n_rows,
        N_COLS,
        figsize=(3.2 * N_COLS, 2.9 * n_rows),
        sharex=True,
        sharey=True,
        layout="constrained",
    )
    flat = axes.ravel()
    for ax, arm in zip(flat, arms):
        for arch, table in summaries.items():
            rows = table[(table["attack"] == arm) & (table["metric"] == metric)]
            rows = rows.sort_values("epsilon")
            label, colour, marker = ARCH_STYLE[arch]
            ax.errorbar(
                rows["epsilon"],
                rows["mean"],
                yerr=rows["sem"],
                color=colour,
                marker=marker,
                markersize=4,
                capsize=2.5,
                linewidth=1.4,
                label=label,
            )
        ax.set_title(ARM_LABELS[arm], fontsize=10, fontweight="bold")
        ax.grid(alpha=0.3)
        if metric != "dice":
            ax.set_yscale("log")
    for ax in flat[len(arms) :]:
        ax.set_visible(False)
    # A column whose bottom panel is hidden gets its tick labels back on the
    # lowest visible panel, which sharex would otherwise leave bare.
    for i in range(len(arms), n_rows * N_COLS):
        above = flat[i - N_COLS]
        above.tick_params(labelbottom=True)
        above.set_xlabel(r"$\varepsilon$ ($L_\infty$, after z-score normalization)")
    if metric == "dice":
        flat[0].set_ylim(-0.02, 1.0)
    for ax in axes[-1] if n_rows > 1 else [flat[0]]:
        ax.set_xlabel(r"$\varepsilon$ ($L_\infty$, after z-score normalization)")
    for row in axes if n_rows > 1 else [axes]:
        row[0].set_ylabel(METRIC_LABELS[metric])
    flat[0].legend(fontsize=8, loc="best")
    subject = "Whole-gland" if label == "WG" else label
    fig.suptitle(
        f"{subject} {METRIC_LABELS[metric]} by architecture under {len(arms)} "
        "adversarial attacks (mean +/- SEM across folds)",
        fontsize=13,
        fontweight="bold",
    )
    for suffix in (".png", ".pdf"):
        fig.savefig(target.with_suffix(suffix), dpi=600, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved {target.with_suffix('.png')} / .pdf")


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
        "--output",
        type=Path,
        default=Path("results/blade_attack/whole_gland/architecture_comparison_wg"),
        help="output stem; <stem>_<metric>.png/.pdf are written",
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
    print(f"architectures: {list(summaries)}; arms: {arms}")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    for metric in METRICS:
        plot_metric(
            summaries, arms, metric, Path(f"{args.output}_{metric}"), args.label
        )


if __name__ == "__main__":
    main()
