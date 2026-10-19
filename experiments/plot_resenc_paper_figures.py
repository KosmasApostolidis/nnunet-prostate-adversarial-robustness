"""Generate Fig. 1 (Dice vs. epsilon) and Fig. 2 (WG forest plot) for paper_resenc.

Both figures are built strictly from the evidence CSVs under
``results/blade_attack/attack_comparison/stats_no_blade/``:

- Fig. 1 (``fig1_dice_vs_eps.pdf``): mean Dice vs. attack budget epsilon for
  nnU-Net and ResEnc-M/L/XL under Auto-PGD (solid) and SEA-20 (dashed), one
  stacked panel per region (WG, TZ+CZ, PZ), with 95% case-bootstrap CI bands.
  The eps = 0 (clean) point comes from ``evidence_clean_dice.csv``.
- Fig. 2 (``fig2_forest_shifts.pdf``): forest plot of the WG paired
  Hodges-Lehmann Dice shifts (encoder step, two batch steps, total step)
  with 95% CIs, one row group per attack. With ``--fig2-revised`` the steps
  are labelled "regime" steps, the right margin shows the additive mean-based
  M->XL share with its fold-stratified bootstrap 95% CI from
  ``revision/E02_additive_decomposition.csv``, and the DAG row is greyed out.
- ``--fig2-steps`` writes only the replacement Fig. 2 (``fig2_case_steps.pdf``):
  per-case Auto-PGD architecture and regime steps per region, from
  ``revision/E33_*`` (``analyze_resenc_round3_evidence.py``).

``--compact`` shrinks both figures for the EMBC page limit (Fig. 1 legend on
top, y ticks 0/0.5/1; Fig. 2 tighter rows, one-line share header); without it
the original sizes are reproduced.

No numbers are computed here beyond reading and re-shaping the CSV columns.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib
import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
from matplotlib.figure import Figure
from matplotlib.lines import Line2D

matplotlib.use("Agg")

# ---------------------------------------------------------------------------
# Shared constants: architectures, attacks, colour-blind-safe palette.
# ---------------------------------------------------------------------------

ARCH_ORDER: tuple[str, ...] = ("unet", "resenc_m", "resenc_l", "resenc_xl")
ARCH_LABELS: dict[str, str] = {
    "unet": "nnU-Net",
    "resenc_m": "ResEnc-M",
    "resenc_l": "ResEnc-L",
    "resenc_xl": "ResEnc-XL",
}
# Okabe-Ito colour-blind-safe subset, validated with dataviz's
# validate_palette.js (all checks pass at categorical light-mode settings).
ARCH_COLORS: dict[str, str] = {
    "unet": "#0072B2",
    "resenc_m": "#E69F00",
    "resenc_l": "#009E73",
    "resenc_xl": "#D55E00",
}
ARCH_MARKERS: dict[str, str] = {
    "unet": "o",
    "resenc_m": "s",
    "resenc_l": "^",
    "resenc_xl": "D",
}

CLASS_ORDER: tuple[str, ...] = ("WG", "TZ+CZ", "PZ")

ATTACK_ORDER: tuple[str, ...] = (
    "fgsm",
    "pgd",
    "segpgd",
    "cospgd",
    "dag",
    "auto_pgd",
    "auto_pgd_r3",
    "sea",
)
ATTACK_LABELS: dict[str, str] = {
    "fgsm": "FGSM",
    "pgd": "PGD",
    "segpgd": "SegPGD",
    "cospgd": "CosPGD",
    "dag": "DAG",
    "auto_pgd": "Auto-PGD",
    "auto_pgd_r3": "Auto-PGD" + "×" + "3",
    "sea": "SEA",
}

FONT_SIZE_BASE = 8
FONT_SIZE_LABEL = 9

RC_PARAMS: dict[str, object] = {
    "font.family": "sans-serif",
    "font.sans-serif": ["Arial", "Helvetica", "DejaVu Sans"],
    "font.size": FONT_SIZE_BASE,
    "axes.labelsize": FONT_SIZE_LABEL,
    "axes.titlesize": FONT_SIZE_LABEL,
    "xtick.labelsize": FONT_SIZE_BASE,
    "ytick.labelsize": FONT_SIZE_BASE,
    "legend.fontsize": FONT_SIZE_BASE,
    "figure.dpi": 300,
    "savefig.dpi": 300,
    "savefig.bbox": "tight",
    "axes.spines.top": False,
    "axes.spines.right": False,
    "pdf.fonttype": 42,
}

SINGLE_COLUMN_WIDTH_IN = 3.5
# --compact heights (inches). Fig. 1 prints at 0.9 columnwidth (scale ~0.93),
# so its compact fonts stay at >= 8 pt to print at >= 7 pt.
FIG1_COMPACT_HEIGHT_IN = 3.85
FIG2_COMPACT_HEIGHT_IN = 3.3


def _evidence_dir(repo_root: Path) -> Path:
    return (
        repo_root / "results" / "blade_attack" / "attack_comparison" / "stats_no_blade"
    )


def _save(fig: Figure, output_dir: Path, stem: str) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_dir / f"{stem}.pdf", format="pdf")
    fig.savefig(output_dir / f"{stem}.png", format="png")


def _arch_legend_handles(with_linestyle: bool = False) -> list[Line2D]:
    handles: list[Line2D] = []
    for arch in ARCH_ORDER:
        handles.append(
            Line2D(
                [0],
                [0],
                color=ARCH_COLORS[arch],
                marker=ARCH_MARKERS[arch],
                markersize=5,
                linestyle="-" if with_linestyle else "None",
                linewidth=1.4,
                label=ARCH_LABELS[arch],
            )
        )
    return handles


# ---------------------------------------------------------------------------
# Fig. 1: mean Dice vs. epsilon, per region, Auto-PGD (solid) vs. SEA-20 (dashed).
# ---------------------------------------------------------------------------


def load_fig1_data(evidence_dir: Path) -> tuple[pd.DataFrame, pd.DataFrame]:
    per_eps = pd.read_csv(evidence_dir / "evidence_per_epsilon_ci.csv")
    per_eps["epsilon"] = per_eps["epsilon"].astype(float)
    clean = pd.read_csv(evidence_dir / "evidence_clean_dice.csv")
    return per_eps, clean


def plot_fig1(
    per_eps: pd.DataFrame, clean: pd.DataFrame, output_dir: Path, compact: bool = False
) -> None:
    height = FIG1_COMPACT_HEIGHT_IN if compact else 5.4
    with plt.rc_context(RC_PARAMS):
        fig, axes = plt.subplots(
            nrows=3,
            ncols=1,
            figsize=(SINGLE_COLUMN_WIDTH_IN, height),
            sharex=True,
        )
        attack_style = {"auto_pgd": "-", "sea": "--"}

        for ax, class_name in zip(axes, CLASS_ORDER, strict=True):
            for arch in ARCH_ORDER:
                clean_row = clean[
                    (clean["class_name"] == class_name) & (clean["arch"] == arch)
                ]
                clean_dice = float(clean_row["mean_dice"].iloc[0])
                color = ARCH_COLORS[arch]
                marker = ARCH_MARKERS[arch]

                for arm in ("auto_pgd", "sea"):
                    sub = per_eps[
                        (per_eps["class_name"] == class_name)
                        & (per_eps["arch"] == arch)
                        & (per_eps["arm"] == arm)
                    ].sort_values("epsilon")

                    x = np.concatenate(([0.0], sub["epsilon"].to_numpy()))
                    y = np.concatenate(([clean_dice], sub["mean_dice"].to_numpy()))
                    ax.plot(
                        x,
                        y,
                        linestyle=attack_style[arm],
                        color=color,
                        marker=marker,
                        markersize=3.5,
                        linewidth=1.3,
                        markeredgewidth=0.0,
                    )

                    # 95% case-bootstrap band, available for eps > 0 only.
                    lo = sub["ci_lo"].to_numpy()
                    hi = sub["ci_hi"].to_numpy()
                    ax.fill_between(
                        sub["epsilon"].to_numpy(),
                        lo,
                        hi,
                        color=color,
                        alpha=0.15,
                        linewidth=0.0,
                    )

            ax.set_ylabel("Mean Dice")
            ax.set_ylim(-0.02, 1.02)
            ax.set_xlim(-0.003, 0.103)
            if compact:
                ax.set_yticks([0.0, 0.5, 1.0])
            ax.text(
                0.02,
                0.06,
                class_name,
                transform=ax.transAxes,
                fontsize=FONT_SIZE_LABEL,
                fontweight="bold",
                va="bottom",
                ha="left",
            )

        axes[-1].set_xlabel(r"Attack budget $\varepsilon$ (normalized units)")
        axes[-1].set_xticks([0.0, 0.02, 0.04, 0.06, 0.08, 0.10])

        arch_handles = _arch_legend_handles(with_linestyle=False)
        style_handles = [
            Line2D(
                [0], [0], color="black", linestyle="-", linewidth=1.3, label="Auto-PGD"
            ),
            Line2D([0], [0], color="black", linestyle="--", linewidth=1.3, label="SEA-20"),
        ]
        if compact:
            fig.legend(
                handles=arch_handles + style_handles,
                loc="upper center",
                ncol=3,
                frameon=False,
                bbox_to_anchor=(0.55, 1.0),
                handletextpad=0.4,
                columnspacing=1.0,
                labelspacing=0.25,
                borderaxespad=0.0,
            )
            fig.tight_layout(rect=(0, 0, 1, 0.925), h_pad=0.3)
        else:
            fig.legend(
                handles=arch_handles + style_handles,
                loc="lower center",
                ncol=3,
                frameon=False,
                bbox_to_anchor=(0.5, -0.06),
                handletextpad=0.5,
                columnspacing=1.0,
            )

            fig.subplots_adjust(hspace=0.1)
            fig.tight_layout(rect=(0, 0.075, 1, 1), h_pad=0.5)
        _save(fig, output_dir, "fig1_dice_vs_eps")
        plt.close(fig)


# ---------------------------------------------------------------------------
# Fig. 2: WG encoder-vs-batch forest plot.
# ---------------------------------------------------------------------------

STEP_ORDER: tuple[str, ...] = ("encoder", "batch_l", "batch_xl", "total")
STEP_LABELS: dict[str, str] = {
    "encoder": r"Encoder (nnU-Net$\to$M)",
    "batch_l": r"Batch (M$\to$L)",
    "batch_xl": r"Batch (M$\to$XL)",
    "total": r"Total (nnU-Net$\to$XL)",
}
# Colour = the architecture the step arrives at; marker = the step's own shape.
STEP_COLOR_ARCH: dict[str, str] = {
    "encoder": "resenc_m",
    "batch_l": "resenc_l",
    "batch_xl": "resenc_xl",
    "total": "resenc_xl",
}
STEP_MARKERS: dict[str, str] = {
    "encoder": "s",
    "batch_l": "^",
    "batch_xl": "D",
    "total": "P",
}
STEP_Y_OFFSET: dict[str, float] = {
    "encoder": 0.27,
    "batch_l": 0.09,
    "batch_xl": -0.09,
    "total": -0.27,
}


REGIME_STEP_LABELS: dict[str, str] = {
    **STEP_LABELS,
    # The nnU-Net->M step also changes depth, parameters and software stack.
    "encoder": r"Architecture (nnU-Net$\to$M)",
    "batch_l": r"Regime (M$\to$L)",
    "batch_xl": r"Regime (M$\to$XL)",
}
# Attacks whose share is not reported (total step near zero): greyed out, "n/a".
FIG2_NA_ATTACKS: frozenset[str] = frozenset({"dag"})
GREYED_COLOR = "#BBBBBB"


def load_fig2_additive_shares(
    evidence_dir: Path, epsilon: float | None = None
) -> dict[str, str]:
    """Return {attack: "82 [80, 84]"} WG additive M->XL shares, in percent.

    Budget-averaged (revision/E02) by default; at one ``epsilon`` (revision/E27)
    when given, e.g. 0.02, where no model is near the Dice floor.
    """
    if epsilon is None:
        e02 = pd.read_csv(evidence_dir / "revision" / "E02_additive_decomposition.csv")
    else:
        e02 = pd.read_csv(evidence_dir / "revision" / "E27_E28_per_epsilon_steps.csv")
        e02 = e02[np.isclose(e02["epsilon"], epsilon)]
    e02 = e02[e02["class_name"] == "WG"].set_index("arm", verify_integrity=True)
    shares: dict[str, str] = {}
    for attack in ATTACK_ORDER:
        if attack in FIG2_NA_ATTACKS:
            shares[attack] = "n/a"
            continue
        row = e02.loc[attack]
        shares[attack] = (
            f"{row['share_batch_xl'] * 100:.0f} "
            f"[{row['share_batch_xl_ci_lo'] * 100:.0f}, "
            f"{row['share_batch_xl_ci_hi'] * 100:.0f}]"
        )
    return shares


def load_fig2_data(evidence_dir: Path) -> tuple[pd.DataFrame, pd.DataFrame]:
    wilcoxon = pd.read_csv(evidence_dir / "cross_arch_wilcoxon_WG.csv")
    wilcoxon = wilcoxon[
        (wilcoxon["metric"] == "dice") & (wilcoxon["epsilon"] == "budget_avg")
    ]
    decomposition = pd.read_csv(evidence_dir / "evidence_wg_batch_decomposition.csv")
    decomposition = decomposition[decomposition["metric"] == "dice"]
    return wilcoxon, decomposition


def _step_estimates(
    attack: str, wilcoxon: pd.DataFrame, decomposition: pd.DataFrame
) -> dict[str, tuple[float, float, float]]:
    """Return {step: (hl_shift, ci_lo, ci_hi)} for one attack, WG, Dice."""
    dec_row = decomposition[decomposition["arm"] == attack].iloc[0]
    encoder_row = wilcoxon[
        (wilcoxon["arm"] == attack) & (wilcoxon["comparator"] == "resenc_m")
    ].iloc[0]
    total_row = wilcoxon[
        (wilcoxon["arm"] == attack) & (wilcoxon["comparator"] == "resenc_xl")
    ].iloc[0]

    return {
        "encoder": (
            float(encoder_row["hl_shift"]),
            float(encoder_row["hl_ci95_low"]),
            float(encoder_row["hl_ci95_high"]),
        ),
        "batch_l": (
            float(dec_row["hl_batch_l_minus_m"]),
            float(dec_row["hl_batch_l_minus_m_ci_lo"]),
            float(dec_row["hl_batch_l_minus_m_ci_hi"]),
        ),
        "batch_xl": (
            float(dec_row["hl_batch_xl_minus_m"]),
            float(dec_row["hl_batch_xl_minus_m_ci_lo"]),
            float(dec_row["hl_batch_xl_minus_m_ci_hi"]),
        ),
        "total": (
            float(total_row["hl_shift"]),
            float(total_row["hl_ci95_low"]),
            float(total_row["hl_ci95_high"]),
        ),
    }


def plot_fig2(
    wilcoxon: pd.DataFrame,
    decomposition: pd.DataFrame,
    output_dir: Path,
    additive_shares: dict[str, str] | None = None,
    compact: bool = False,
    share_eps: float | None = None,
) -> None:
    """Draw Fig. 2; ``additive_shares`` switches on the revised layout."""
    revised = additive_shares is not None
    step_labels = REGIME_STEP_LABELS if revised else STEP_LABELS
    greyed = FIG2_NA_ATTACKS if revised else frozenset()
    n_attacks = len(ATTACK_ORDER)
    with plt.rc_context(RC_PARAMS):
        height = FIG2_COMPACT_HEIGHT_IN if compact else 5.6
        fig, ax = plt.subplots(figsize=(SINGLE_COLUMN_WIDTH_IN, height))

        y_positions = np.arange(n_attacks)[::-1]  # top row = first attack
        share_x = 0.275 if revised else 0.315

        for attack, y_center in zip(ATTACK_ORDER, y_positions, strict=True):
            estimates = _step_estimates(attack, wilcoxon, decomposition)
            for step in STEP_ORDER:
                shift, lo, hi = estimates[step]
                y = y_center + STEP_Y_OFFSET[step]
                arch = STEP_COLOR_ARCH[step]
                color = GREYED_COLOR if attack in greyed else ARCH_COLORS[arch]
                ax.errorbar(
                    shift,
                    y,
                    xerr=[[shift - lo], [hi - shift]],
                    fmt=STEP_MARKERS[step],
                    color=color,
                    ecolor=color,
                    markersize=4.5,
                    markeredgecolor="grey" if attack in greyed else "black",
                    markeredgewidth=0.3,
                    elinewidth=1.0,
                    capsize=2.0,
                )

            # Annotate the batch-XL share of the total shift at the right margin.
            if additive_shares is not None:
                share_text = additive_shares[attack]
            else:
                share = float(
                    decomposition[decomposition["arm"] == attack][
                        "share_batch_xl_of_total"
                    ].iloc[0]
                )
                share_text = f"{share * 100:.0f}%"
            ax.text(
                share_x,
                y_center,
                share_text,
                fontsize=FONT_SIZE_BASE - 1,
                va="center",
                ha="left",
                color=GREYED_COLOR if attack in greyed else "dimgrey",
            )

        ax.axvline(x=0.0, color="grey", linestyle=":", linewidth=0.8, zorder=0)
        ax.set_yticks(list(y_positions))
        ax.set_yticklabels([ATTACK_LABELS[a] for a in ATTACK_ORDER])
        for tick, attack in zip(ax.get_yticklabels(), ATTACK_ORDER, strict=True):
            if attack in greyed:
                tick.set_color(GREYED_COLOR)
        ax.set_xlabel("Paired HL shift in budget-averaged Dice")
        ax.set_xlim(-0.02, 0.36)
        if compact:
            ax.set_ylim(-0.45, n_attacks - 0.55)
            header = "regime share,\n% [95% CI]" if revised else "batch\nshare"
            if revised and share_eps is not None:
                header = f"regime share at\n$\\varepsilon$={share_eps:g}, % [CI]"
            header_y = n_attacks - 0.5
            # Pull the header left so it does not widen the figure past 252 pt.
            header_x = share_x - 0.02
        else:
            ax.set_ylim(-0.6, n_attacks - 0.4)
            header = "regime\nshare, %\n[95% CI]" if revised else "batch\nshare"
            header_y = n_attacks - 0.15
            header_x = share_x
        ax.text(
            header_x,
            header_y,
            header,
            fontsize=FONT_SIZE_BASE - 1,
            va="bottom",
            ha="left",
            color="dimgrey",
        )

        legend_handles = [
            Line2D(
                [0],
                [0],
                color=ARCH_COLORS[STEP_COLOR_ARCH[step]],
                marker=STEP_MARKERS[step],
                markeredgecolor="black",
                markeredgewidth=0.3,
                linestyle="None",
                markersize=5,
                label=step_labels[step],
            )
            for step in STEP_ORDER
        ]
        if compact:
            fig.legend(
                handles=legend_handles,
                loc="lower center",
                bbox_to_anchor=(0.5, 0.0),
                ncol=2,
                frameon=False,
                fontsize=FONT_SIZE_BASE - 1,
                handletextpad=0.3,
                columnspacing=0.8,
                labelspacing=0.3,
                borderaxespad=0.0,
            )
            fig.tight_layout(rect=(0, 0.09, 1, 1))
        else:
            ax.legend(
                handles=legend_handles,
                loc="lower center",
                bbox_to_anchor=(0.42, -0.30),
                ncol=2,
                frameon=False,
                fontsize=FONT_SIZE_BASE - 1,
                handletextpad=0.4,
                columnspacing=1.0,
            )

            fig.tight_layout(rect=(0, 0.14, 1, 1))
        _save(fig, output_dir, "fig2_forest_shifts")
        plt.close(fig)


# ---------------------------------------------------------------------------
# Fig. 2 (steps): per-case architecture vs. regime steps, per region.
# ---------------------------------------------------------------------------

STEP_COLORS: dict[str, str] = {"architecture": "#0072B2", "regime": "#D55E00"}
STEP_LABELS: dict[str, str] = {
    "architecture": "Architecture step",
    "regime": "Regime step",
}


def load_fig2_steps_data(evidence_dir: Path) -> tuple[pd.DataFrame, pd.DataFrame]:
    rev = evidence_dir / "revision"
    return (
        pd.read_csv(rev / "E33_per_case_steps_auto_pgd.csv"),
        pd.read_csv(rev / "E33_step_summary_auto_pgd.csv").set_index("class_name"),
    )


def _p_label(p: float) -> str:
    return "p < 0.001" if p < 0.001 else f"p = {p:.3f}"


def plot_fig2_steps(
    steps: pd.DataFrame, summary: pd.DataFrame, output_dir: Path
) -> None:
    """Boxes: per-case quartiles, whiskers 2.5-97.5 percentiles; diamonds: mean
    with its fold-stratified bootstrap 95% CI; text: paired Wilcoxon test of
    regime minus architecture (all from revision/E33)."""
    with plt.rc_context(RC_PARAMS):
        fig, ax = plt.subplots(figsize=(SINGLE_COLUMN_WIDTH_IN, 1.75))
        ax.axhline(0.0, color="0.6", linewidth=0.6, zorder=0)
        offsets = {"architecture": -0.19, "regime": 0.19}
        for i, region in enumerate(CLASS_ORDER):
            g = steps[steps["class_name"] == region]
            s = summary.loc[region]
            for step, dx in offsets.items():
                pos = i + dx
                ax.boxplot(
                    g[step].to_numpy(),
                    positions=[pos],
                    widths=0.3,
                    whis=(2.5, 97.5),
                    showfliers=False,
                    patch_artist=True,
                    boxprops={
                        "facecolor": STEP_COLORS[step],
                        "alpha": 0.35,
                        "edgecolor": STEP_COLORS[step],
                    },
                    medianprops={"color": STEP_COLORS[step], "linewidth": 1.2},
                    whiskerprops={"color": STEP_COLORS[step]},
                    capprops={"color": STEP_COLORS[step]},
                )
                ax.errorbar(
                    pos,
                    s[step],
                    yerr=[
                        [s[step] - s[f"{step}_ci_lo"]],
                        [s[f"{step}_ci_hi"] - s[step]],
                    ],
                    fmt="D",
                    color="black",
                    markersize=3,
                    capsize=0,
                    linewidth=1.0,
                    zorder=3,
                )
            top = float(g[["architecture", "regime"]].quantile(0.975).max())
            ax.text(
                i,
                top + 0.03,
                _p_label(float(s["wilcoxon_p_regime_vs_architecture"])),
                ha="center",
                va="bottom",
                fontsize=FONT_SIZE_BASE - 1,
            )
        ax.set_xticks(range(len(CLASS_ORDER)))
        ax.set_xticklabels(
            [f"{r}\n(n = {int(summary.loc[r, 'n_cases']):,})" for r in CLASS_ORDER]
        )
        ax.set_xlim(-0.6, len(CLASS_ORDER) - 0.4)
        ax.set_ylim(top=ax.get_ylim()[1] + 0.05)
        ax.set_ylabel("Per-case step in Dice")
        handles = [
            plt.Rectangle(
                (0, 0),
                1,
                1,
                facecolor=STEP_COLORS[k],
                alpha=0.35,
                edgecolor=STEP_COLORS[k],
                label=STEP_LABELS[k],
            )
            for k in offsets
        ] + [
            Line2D(
                [0],
                [0],
                marker="D",
                color="black",
                linestyle="None",
                markersize=3,
                label="Mean, 95% CI",
            )
        ]
        ax.legend(
            handles=handles,
            loc="lower center",
            bbox_to_anchor=(0.5, 1.02),
            frameon=False,
            ncol=3,
            handlelength=1.2,
            columnspacing=0.8,
            fontsize=FONT_SIZE_BASE - 1,
        )
        _save(fig, output_dir, "fig2_case_steps")
        plt.close(fig)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    repo_root = Path(__file__).resolve().parent.parent
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=repo_root / "paper_resenc" / "figures",
        help="Directory to write fig1_dice_vs_eps.{pdf,png} and fig2_forest_shifts.{pdf,png}.",
    )
    parser.add_argument(
        "--evidence-dir",
        type=Path,
        default=_evidence_dir(repo_root),
        help="Directory holding the stats_no_blade evidence CSVs.",
    )
    parser.add_argument(
        "--fig2-revised",
        action="store_true",
        help=(
            "Fig. 2 with regime-step labels, the additive mean-based M->XL share "
            "and its 95%% CI (revision/E02), and DAG greyed out as n/a."
        ),
    )
    parser.add_argument(
        "--fig2-share-eps",
        type=float,
        default=None,
        help="With --fig2-revised: show the share at this epsilon (revision/E27) "
        "instead of the budget-averaged one.",
    )
    parser.add_argument(
        "--compact",
        action="store_true",
        help="Shorter Fig. 1 and Fig. 2 for the EMBC page limit.",
    )
    parser.add_argument(
        "--fig2-steps",
        action="store_true",
        help="Only write Fig. 2 as per-case Auto-PGD step boxplots "
        "(fig2_case_steps.{pdf,png}, from revision/E33); Fig. 1 is left untouched.",
    )
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> None:
    args = parse_args(argv)
    if args.fig2_steps:
        plot_fig2_steps(*load_fig2_steps_data(args.evidence_dir), args.output_dir)
        print(f"Wrote fig2_case_steps to {args.output_dir}")
        return

    per_eps, clean = load_fig1_data(args.evidence_dir)
    plot_fig1(per_eps, clean, args.output_dir, compact=args.compact)

    wilcoxon, decomposition = load_fig2_data(args.evidence_dir)
    additive_shares = (
        load_fig2_additive_shares(args.evidence_dir, args.fig2_share_eps)
        if args.fig2_revised
        else None
    )
    plot_fig2(
        wilcoxon,
        decomposition,
        args.output_dir,
        additive_shares,
        compact=args.compact,
        share_eps=args.fig2_share_eps,
    )

    print(f"Wrote figures to {args.output_dir}")


if __name__ == "__main__":
    main()
