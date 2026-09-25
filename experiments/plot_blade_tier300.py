"""Figures for the tier-300 BLADE statistics (``blade_tier300_*.csv``).

Reads the CSVs written by ``analyze_blade_tier300.py`` from
``results/blade_attack/cross_dataset/`` and writes, next to them,
``blade_tier300_per_epsilon_rank_biserial.{png,pdf}`` and
``blade_tier300_contrast_forest.{png,pdf}``.

Usage::

    PYTHONPATH=src python experiments/plot_blade_tier300.py
"""

from __future__ import annotations

import os

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import pandas as pd  # noqa: E402
from matplotlib.figure import Figure  # noqa: E402
from matplotlib.lines import Line2D  # noqa: E402

DIR = os.path.join("results", "blade_attack", "cross_dataset")
STEM = "blade_tier300"

DATASETS = ["Whole gland", "TZ+CZ", "PZ"]  # anatomical order
METRICS = ["Dice", "HD95 (mm)", "ASD (mm)"]
BLADES = ["BLADE-3", "BLADE-MM"]
OPPS = ["SEA", "Auto-PGD x3"]
CONTRASTS = [(b, o) for b in BLADES for o in OPPS]

# Reference palette slots 1 and 2 (dataviz skill, validated earlier).
OPP_COLOUR = {"SEA": "#2a78d6", "Auto-PGD x3": "#eb6834"}
BLADE_STYLE = {"BLADE-3": "-", "BLADE-MM": "--"}
EST_COLOUR = {"hl": "#000000", "lmm": "#2a78d6", "boot": "#eb6834"}


def load(name: str) -> pd.DataFrame:
    return pd.read_csv(os.path.join(DIR, f"{STEM}_{name}.csv"))


def save(fig: Figure, base: str) -> str:
    path = os.path.join(DIR, f"{STEM}_{base}")
    fig.savefig(f"{path}.png", dpi=200, bbox_inches="tight")
    fig.savefig(f"{path}.pdf", bbox_inches="tight")
    plt.close(fig)
    return f"{path}.png"


def fig_per_epsilon(peps: pd.DataFrame, lrt: pd.DataFrame) -> str:
    """3 datasets x 3 metrics; rank-biserial against epsilon, four contrasts."""
    fig, axes = plt.subplots(3, 3, figsize=(15.5, 12.0), squeeze=False)
    for r, ds in enumerate(DATASETS):
        for c, met in enumerate(METRICS):
            ax = axes[r][c]
            ax.axhline(0.0, color="black", linewidth=1.0, zorder=0)
            for blade, opp in CONTRASTS:
                sub = peps[
                    (peps.dataset == ds)
                    & (peps.metric == met)
                    & (peps.blade == blade)
                    & (peps.opponent == opp)
                ].sort_values("epsilon")
                colour = OPP_COLOUR[opp]
                ax.plot(
                    sub.epsilon,
                    sub.rank_biserial,
                    BLADE_STYLE[blade],
                    color=colour,
                    linewidth=1.6,
                    zorder=2,
                )
                sig = sub[sub.significant]
                ns = sub[~sub.significant]
                ax.plot(
                    sig.epsilon,
                    sig.rank_biserial,
                    "o",
                    color=colour,
                    markersize=7,
                    markeredgecolor=colour,
                    zorder=3,
                )
                ax.plot(
                    ns.epsilon,
                    ns.rank_biserial,
                    "o",
                    color="white",
                    markersize=7,
                    markeredgecolor=colour,
                    markeredgewidth=1.6,
                    zorder=3,
                )
            icc = lrt[(lrt.dataset == ds) & (lrt.metric == met)].icc.iloc[0]
            ax.set_title(f"{ds} — {met}   (case ICC = {icc:.2f})", fontsize=12)
            ax.set_ylim(-1.05, 1.05)
            ax.set_xticks([0.02, 0.04, 0.06, 0.08, 0.10])
            ax.grid(alpha=0.3, linewidth=0.6)
            if c == 0:
                ax.set_ylabel(
                    "rank-biserial $r$\n(> 0: BLADE damages more)", fontsize=11
                )
            if r == 2:
                ax.set_xlabel(r"attack budget $\epsilon$", fontsize=11)

    handles = [
        Line2D(
            [],
            [],
            color=OPP_COLOUR[o],
            linestyle=BLADE_STYLE[b],
            marker="o",
            markersize=7,
            linewidth=1.6,
            label=f"{b} vs {o}",
        )
        for b, o in CONTRASTS
    ]
    handles += [
        Line2D(
            [],
            [],
            color="grey",
            marker="o",
            linestyle="none",
            markersize=7,
            label="Holm-significant (m = 20 per panel)",
        ),
        Line2D(
            [],
            [],
            color="white",
            marker="o",
            linestyle="none",
            markersize=7,
            markeredgecolor="grey",
            markeredgewidth=1.6,
            label="not significant",
        ),
    ]
    fig.legend(
        handles=handles,
        loc="lower center",
        ncol=3,
        fontsize=11,
        frameon=False,
        bbox_to_anchor=(0.5, -0.035),
    )
    fig.suptitle(
        "Budget-matched tier-300 comparison: paired Wilcoxon effect size at "
        "every attack budget\n"
        "nnU-Net, 5-fold pooled, zones confined; Holm correction within each "
        "dataset x metric panel (20 tests); $\\epsilon$ = 0 excluded",
        fontsize=14,
    )
    fig.tight_layout(rect=(0, 0.01, 1, 0.94))
    return save(fig, "per_epsilon_rank_biserial")


def fig_forest(holm: pd.DataFrame, lmm: pd.DataFrame, boot: pd.DataFrame) -> str:
    """3 datasets x 3 metrics; the four contrasts under three estimators.

    One panel per dataset x metric because the effect scales differ by an
    order of magnitude (whole-gland HD95 shifts ~25 mm, zones ~1 mm); a
    shared axis would flatten the zonal panels onto the zero line.
    """
    key = ["dataset", "metric", "blade", "opponent"]
    merged = (
        holm[key + ["hl_shift", "sig_within"]]
        .merge(lmm[key + ["lmm_estimate", "lmm_ci_low", "lmm_ci_high"]], on=key)
        .merge(boot[key + ["auc_median_diff", "auc_ci_low", "auc_ci_high"]], on=key)
    )
    est = [
        (
            "Hodges-Lehmann shift (median, Holm-tested)",
            "hl_shift",
            None,
            None,
            EST_COLOUR["hl"],
            "s",
            0.26,
        ),
        (
            "mixed model contrast (mean, 95% CI)",
            "lmm_estimate",
            "lmm_ci_low",
            "lmm_ci_high",
            EST_COLOUR["lmm"],
            "o",
            0.0,
        ),
        (
            "case-clustered bootstrap of the AUC difference (median, 95% CI)",
            "auc_median_diff",
            "auc_ci_low",
            "auc_ci_high",
            EST_COLOUR["boot"],
            "D",
            -0.26,
        ),
    ]

    fig, axes = plt.subplots(3, 3, figsize=(16.5, 11.0), squeeze=False)
    for r, ds in enumerate(DATASETS):
        for c, met in enumerate(METRICS):
            ax = axes[r][c]
            ax.axvline(0.0, color="black", linewidth=1.0, zorder=0)
            for i, (b, o) in enumerate(CONTRASTS):
                y = len(CONTRASTS) - 1 - i
                m = merged[
                    (merged.dataset == ds)
                    & (merged.metric == met)
                    & (merged.blade == b)
                    & (merged.opponent == o)
                ]
                if m.empty:
                    continue
                m = m.iloc[0]
                for _, col, lo, hi, colour, marker, off in est:
                    if lo is None:
                        # Hollow when the Holm test did not reject.
                        face = colour if m["sig_within"] else "white"
                    else:
                        ax.plot(
                            [m[lo], m[hi]],
                            [y + off] * 2,
                            color=colour,
                            linewidth=1.7,
                            zorder=2,
                        )
                        face = "white" if m[lo] <= 0.0 <= m[hi] else colour
                    ax.plot(
                        m[col],
                        y + off,
                        marker,
                        color=face,
                        markeredgecolor=colour,
                        markeredgewidth=1.4,
                        markersize=6.5,
                        zorder=3,
                    )
            ax.set_yticks(range(len(CONTRASTS)))
            ax.set_yticklabels(
                [f"{b} vs {o}" for b, o in reversed(CONTRASTS)], fontsize=10
            )
            ax.set_ylim(-0.75, len(CONTRASTS) - 0.25)
            ax.set_title(f"{ds} — {met}", fontsize=12)
            ax.grid(axis="x", alpha=0.3, linewidth=0.6)
            if r == 2:
                ax.set_xlabel(
                    "BLADE − opponent, sign-corrected\n(> 0: BLADE damages more)",
                    fontsize=10,
                )

    handles = [
        Line2D(
            [],
            [],
            color=colour,
            marker=marker,
            markersize=6.5,
            linewidth=1.7,
            label=label,
        )
        for label, _, _, _, colour, marker, _ in est
    ]
    handles.append(
        Line2D(
            [],
            [],
            color="grey",
            marker="o",
            linestyle="none",
            markersize=6.5,
            markerfacecolor="white",
            markeredgewidth=1.4,
            label="hollow: 95% CI covers zero (bootstrap, LMM) "
            "or Holm did not reject (HL)",
        )
    )
    fig.legend(
        handles=handles,
        loc="lower center",
        ncol=2,
        fontsize=11,
        frameon=False,
        bbox_to_anchor=(0.5, -0.055),
    )
    fig.suptitle(
        "Three estimators of the same twelve contrasts\n"
        "Hodges-Lehmann and the bootstrap summarise medians, the mixed model "
        "summarises means; where they disagree the paired differences are skewed",
        fontsize=14,
    )
    fig.tight_layout(rect=(0, 0.02, 1, 0.93))
    return save(fig, "contrast_forest")


def main() -> None:
    peps = load("per_epsilon_wilcoxon")
    lrt = load("mixedlm_interaction_lrt")
    holm = load("holm_corrected")
    lmm = load("mixedlm_contrasts")
    boot = load("clustered_bootstrap_ci")
    print(fig_per_epsilon(peps, lrt))
    print(fig_forest(holm, lmm, boot))


if __name__ == "__main__":
    main()
