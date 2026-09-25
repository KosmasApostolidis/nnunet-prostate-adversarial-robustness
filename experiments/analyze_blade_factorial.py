"""Where BLADE's advantage comes from: a factorial attribution on existing runs.

No new attacks. The campaign already contains the cells of a design that
separates BLADE's two ingredients from the extra compute it spends:

==============  ==========  ======================  =========  =====================
arm             eps-ladder  objectives              gradients  selection (WG / zones)
==============  ==========  ======================  =========  =====================
Auto-PGD        no          Dice+CE                 105        surrogate
Auto-PGD x3     no          Dice+CE, 3 restarts     315        surrogate
SEA             no          mCE, balanced mCE, JS   315        Dice across losses
BLADE-DiceCE    yes         Dice+CE                 105        surrogate / Dice, every iterate
BLADE-3         yes         Dice+CE, boundary,      315        Dice across losses /
                            frontier                           Dice, every iterate
==============  ==========  ======================  =========  =====================

(gradients per case at 20 steps, measured in ``computational_cost/``.) On the
whole gland the campaign runs the legacy binary BLADE, whose rungs keep Auto-PGD's
final iterate, so there the "ladder" ingredient is the ascending-epsilon warm
start alone. On the zones it runs BLADE-MC, which also keeps the lowest-Dice
iterate seen on each rung (the clean image included); there the two are not
separable with these arms.

Contrasts, each a paired per-case difference signed so that positive means the
first arm damaged more (Dice lower, HD95/ASD higher):

* ``ladder_single``   BLADE-DiceCE vs Auto-PGD. Same loss, same 105 gradients:
  the clean ladder effect.
* ``ladder_ensemble`` BLADE-3 vs SEA. Both 315 gradients and both an objective
  ensemble, but the loss sets differ, so this is ladder plus BLADE's losses.
* ``compute``         Auto-PGD x3 vs Auto-PGD. Three restarts instead of one.
* ``ensemble``        SEA vs Auto-PGD x3. 315 gradients each: an objective
  ensemble against spending the same gradients on restarts.
* ``ensemble_ladder`` BLADE-3 vs BLADE-DiceCE. Objectives added on top of the
  ladder, with 3x the gradients (not budget-matched).
* ``interaction``     (BLADE-3 - SEA) - (BLADE-DiceCE - Auto-PGD): does the
  ladder help more when the objectives are an ensemble?
* ``total``           BLADE-3 vs Auto-PGD, and ``total_matched`` BLADE-3 vs
  Auto-PGD x3.

``total`` = ``compute`` + ``ensemble`` + ``ladder_ensemble`` exactly for the
per-case means, so the mean shares of that chain are reported alongside the
Hodges-Lehmann shifts (which do not add).

Usage::

    PYTHONPATH=src python experiments/analyze_blade_factorial.py
"""

from __future__ import annotations

import argparse
import os
import sys

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.patches import Patch  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from scipy import stats  # noqa: E402

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from analyze_attack_architecture_stats import (  # noqa: E402
    ALPHA,
    ARCHS,
    CLASS_TAG,
    EPSILONS,
    METRIC_DIRECTION,
    RESULTS,
    TREES,
    _cohens_dz,
    _fdr_bh,
    _hodges_lehmann,
    _rank_biserial,
    build_matrix,
    load_per_sample,
    mean_over_epsilons,
)
from plot_architecture_comparison import ARCH_STYLE  # noqa: E402
from run_twelve_arm_wg_campaign import METRIC_LABELS, METRICS  # noqa: E402

ARMS = ("auto_pgd", "auto_pgd_r3", "sea", "blade1", "blade")
# name -> (first arm, second arm, budget-matched, description)
CONTRASTS = {
    "ladder_single": (
        "blade1",
        "auto_pgd",
        True,
        "ladder, same loss (BLADE-DiceCE vs Auto-PGD)",
    ),
    "ladder_ensemble": ("blade", "sea", True, "ladder + BLADE losses (BLADE-3 vs SEA)"),
    "compute": (
        "auto_pgd_r3",
        "auto_pgd",
        False,
        "3x restarts (Auto-PGD x3 vs Auto-PGD)",
    ),
    "ensemble": (
        "sea",
        "auto_pgd_r3",
        True,
        "loss ensemble vs restarts (SEA vs Auto-PGD x3)",
    ),
    "ensemble_ladder": (
        "blade",
        "blade1",
        False,
        "objectives on the ladder (BLADE-3 vs BLADE-DiceCE)",
    ),
    "total": ("blade", "auto_pgd", False, "total (BLADE-3 vs Auto-PGD)"),
    "total_matched": (
        "blade",
        "auto_pgd_r3",
        True,
        "total, matched (BLADE-3 vs Auto-PGD x3)",
    ),
}
CHAIN = ("compute", "ensemble", "ladder_ensemble")
OUT_DIR = os.path.join(RESULTS, "attack_comparison", "factorial")


def _paired_row(diff: np.ndarray) -> dict[str, float]:
    nz = diff[diff != 0]
    w_stat, w_p = (float("nan"), float("nan")) if nz.size == 0 else stats.wilcoxon(nz)
    hl, lo, hi = _hodges_lehmann(diff)
    return {
        "n_cases": int(diff.size),
        "mean_diff": float(diff.mean()),
        "hl_shift": hl,
        "hl_ci95_low": lo,
        "hl_ci95_high": hi,
        "wilcoxon_stat": w_stat,
        "p_value": w_p,
        "rank_biserial": _rank_biserial(diff),
        "cohens_dz": _cohens_dz(diff),
    }


def factorial_tests(
    frames: dict[str, pd.DataFrame], class_name: str, arch: str
) -> pd.DataFrame:
    rows = []
    for metric in METRICS:
        sign = METRIC_DIRECTION[metric]
        for eps in [None, *EPSILONS]:
            if eps is None:
                mat = mean_over_epsilons(
                    frames, list(ARMS), metric, EPSILONS, class_name
                )
            else:
                mat = build_matrix(frames, metric, eps, list(ARMS), class_name)
            eff = {
                name: sign * (mat[a] - mat[b]).to_numpy()
                for name, (a, b, _, _) in CONTRASTS.items()
            }
            eff["interaction"] = eff["ladder_ensemble"] - eff["ladder_single"]
            family = []
            for name, diff in eff.items():
                a, b, matched, desc = CONTRASTS.get(
                    name, ("", "", False, "ladder gain with ensemble minus without")
                )
                family.append(
                    {
                        "class": class_name,
                        "arch": arch,
                        "metric": metric,
                        "epsilon": "budget_avg" if eps is None else eps,
                        "contrast": name,
                        "arm_a": a,
                        "arm_b": b,
                        "budget_matched": matched,
                        "description": desc,
                        **_paired_row(diff),
                    }
                )
            fam = pd.DataFrame(family)
            fam["p_adj_bh"] = _fdr_bh(fam["p_value"].to_numpy())
            fam["significant"] = fam["p_adj_bh"] < ALPHA
            total = fam.loc[fam["contrast"] == "total", "mean_diff"].iloc[0]
            fam["share_of_total_mean"] = np.where(
                fam["contrast"].isin(CHAIN), fam["mean_diff"] / total, np.nan
            )
            rows.append(fam)
    return pd.concat(rows, ignore_index=True)


def _fmt_p(p: float) -> str:
    if not np.isfinite(p):
        return "-"
    return "<0.001" if p < 1e-3 else f"{p:.3f}"


def render_md(table: pd.DataFrame) -> str:
    lines = [
        "# Factorial attribution of BLADE's gain",
        "",
        "Budget-averaged over epsilon 0.02-0.1, paired by case. Each cell: "
        "Hodges-Lehmann shift (positive = first arm more damaging) and "
        "BH-adjusted p within the (class, architecture, metric) family. "
        "`=` marks budget-matched contrasts. Chain shares split the mean "
        "BLADE-3 minus Auto-PGD gain into compute, loss ensemble and ladder.",
        "",
    ]
    order = [*CONTRASTS, "interaction"]
    for metric in METRICS:
        lines += [f"## {metric}", ""]
        for class_name in CLASS_TAG:
            sub = table[
                (table["class"] == class_name)
                & (table["metric"] == metric)
                & (table["epsilon"] == "budget_avg")
            ]
            lines += [
                f"### {class_name}",
                "",
                "| Contrast | " + " | ".join(ARCH_STYLE[a][0] for a in ARCHS) + " |",
                "|---|" + "---|" * len(ARCHS),
            ]
            for name in order:
                cells = []
                for arch in ARCHS:
                    r = sub[(sub["arch"] == arch) & (sub["contrast"] == name)].iloc[0]
                    star = "*" if r["significant"] else ""
                    share = (
                        f", {100 * r['share_of_total_mean']:.0f}% of total"
                        if np.isfinite(r["share_of_total_mean"])
                        else ""
                    )
                    cells.append(
                        f"{r['hl_shift']:+.3f} ({_fmt_p(r['p_adj_bh'])}){star}{share}"
                    )
                desc = sub[sub["contrast"] == name].iloc[0]["description"]
                eq = (
                    " ="
                    if sub[sub["contrast"] == name].iloc[0]["budget_matched"]
                    else ""
                )
                lines.append(f"| {desc}{eq} | " + " | ".join(cells) + " |")
            lines.append("")
    return "\n".join(lines)


# Categorical slots 1-3 of the dataviz reference palette (validated: CVD
# separation and normal-vision floor pass). The total is a hollow bar in text
# ink so it never reads as a fourth series.
CHAIN_STYLE = {
    "compute": ("#1baf7a", "3x restarts (Auto-PGD x3 vs Auto-PGD)"),
    "ensemble": ("#eb6834", "loss ensemble instead of restarts (SEA vs Auto-PGD x3)"),
    "ladder_ensemble": ("#2a78d6", "ladder + BLADE losses (BLADE-3 vs SEA)"),
}
INK = "#1a1a1a"
MUTED = "#6b6b6b"


def _mean_ci(row: pd.Series) -> tuple[float, float]:
    """95% CI of the mean paired difference, from its Cohen's d_z and n."""
    mean, dz, n = row["mean_diff"], row["cohens_dz"], row["n_cases"]
    if not np.isfinite(dz) or dz == 0 or n < 2:
        return float("nan"), float("nan")
    half = 1.96 * abs(mean / dz) / np.sqrt(n)
    return mean - half, mean + half


def plot_chain(table: pd.DataFrame, out_dir: str) -> list[str]:
    """Grouped bars: the three chain components and their sum, per architecture.

    The components are consecutive paired contrasts along Auto-PGD -> Auto-PGD
    x3 -> SEA -> BLADE-3, so their per-case means add up exactly to the total
    (BLADE-3 vs Auto-PGD). Error bars are 95% CIs of the mean difference; a
    hatched bar is not significant after BH. The label over each ladder bar is
    its share of the total.
    """
    parts = [*CHAIN, "total"]
    width = 0.19
    offsets = (np.arange(len(parts)) - (len(parts) - 1) / 2) * (width + 0.02)
    written = []
    for metric in METRICS:
        fig, axes = plt.subplots(
            1, len(CLASS_TAG), figsize=(4.6 * len(CLASS_TAG), 4.4), layout="constrained"
        )
        for ax, class_name in zip(axes, CLASS_TAG):
            sub = table[
                (table["class"] == class_name)
                & (table["metric"] == metric)
                & (table["epsilon"] == "budget_avg")
            ].set_index(["arch", "contrast"])
            x = np.arange(len(ARCHS))
            top = 0.0
            labels_at: list[tuple[float, float, float]] = []
            for off, name in zip(offsets, parts):
                rows = [sub.loc[(a, name)] for a in ARCHS]
                vals = np.array([r["mean_diff"] for r in rows])
                cis = np.array([_mean_ci(r) for r in rows])
                sig = np.array([bool(r["significant"]) for r in rows])
                err = np.vstack([vals - cis[:, 0], cis[:, 1] - vals])
                if name == "total":
                    face, edge = "white", INK
                else:
                    face, edge = CHAIN_STYLE[name][0], "white"
                for xi, v, s in zip(x + off, vals, sig):
                    ax.bar(
                        xi, v, width=width, color=face, edgecolor=edge,
                        linewidth=1.2 if name == "total" else 0.8,
                        hatch=None if s else "////", zorder=2,
                    )
                ax.errorbar(x + off, vals, yerr=err, fmt="none", ecolor=INK,
                            elinewidth=0.9, capsize=2, zorder=3)
                top = max(top, float(np.nanmax(cis[:, 1])))
                if name == "ladder_ensemble":
                    shares = [r["share_of_total_mean"] for r in rows]
                    labels_at.extend(zip(x + off, cis[:, 1], shares))
            ax.set_ylim(top=top * 1.14 if top > 0 else None)
            pad = 0.025 * (ax.get_ylim()[1] - ax.get_ylim()[0])
            for xi, hi, sh in labels_at:
                ax.text(xi, hi + pad, f"{100 * sh:.0f}%", ha="center", va="bottom",
                        fontsize=7, color=INK)
            ax.axhline(0, color=MUTED, linewidth=0.8, zorder=1)
            ax.set_xticks(x)
            ax.set_xticklabels([ARCH_STYLE[a][0] for a in ARCHS], fontsize=9)
            ax.set_title(CLASS_TITLE_FACTORIAL[class_name], fontsize=11, fontweight="bold")
            ax.grid(axis="y", alpha=0.25, zorder=0)
            ax.spines[["top", "right"]].set_visible(False)
        unit = "Dice" if metric == "dice" else f"{metric.upper()}, mm"
        axes[0].set_ylabel(f"extra damage over Auto-PGD\n(mean paired difference, {unit})")
        handles = [Patch(facecolor=CHAIN_STYLE[n][0], edgecolor="white") for n in CHAIN]
        labels = [CHAIN_STYLE[n][1] for n in CHAIN]
        handles += [
            Patch(facecolor="white", edgecolor=INK, linewidth=1.2),
            Patch(facecolor="white", edgecolor=MUTED, hatch="////"),
        ]
        labels += ["total (BLADE-3 vs Auto-PGD)", "not significant (BH, alpha 0.05)"]
        fig.legend(handles, labels, loc="outside lower center", ncol=3, fontsize=8,
                   frameon=False)
        fig.suptitle(
            f"Where BLADE-3's gain over Auto-PGD comes from: {METRIC_LABELS[metric]}, "
            "budget-averaged, 95% CI\n"
            "components add up to the total; % = share carried by the ladder + BLADE losses",
            fontsize=11,
            fontweight="bold",
        )
        out = os.path.join(out_dir, f"factorial_chain_{metric}.png")
        fig.savefig(out, dpi=300, bbox_inches="tight")
        plt.close(fig)
        written.append(out)
    return written


CLASS_TITLE_FACTORIAL = {"WG": "Whole gland", "TZ+CZ": "TZ+CZ", "PZ": "PZ"}


# Eight distinct hues (Paul Tol muted + dark), one per (contrast, architecture):
# cool colours for the ladder contrast, warm ones for the loss ensemble.
SERIES_COLOUR = {
    ("ladder_single", "unet"): "#332288",
    ("ladder_single", "resenc_m"): "#117733",
    ("ladder_single", "resenc_l"): "#44AA99",
    ("ladder_single", "resenc_xl"): "#88CCEE",
    ("ensemble", "unet"): "#882255",
    ("ensemble", "resenc_m"): "#CC6677",
    ("ensemble", "resenc_l"): "#AA4499",
    ("ensemble", "resenc_xl"): "#DDAA33",
}


def plot_ladder_by_eps(table: pd.DataFrame, out_dir: str) -> str:
    """Clean ladder effect (BLADE-DiceCE vs Auto-PGD) against epsilon, Dice."""
    fig, axes = plt.subplots(
        1,
        len(CLASS_TAG),
        figsize=(4.2 * len(CLASS_TAG), 3.6),
        sharey=False,
        layout="constrained",
    )
    for ax, class_name in zip(axes, CLASS_TAG):
        for name, style in (("ladder_single", "-"), ("ensemble", "--")):
            for arch in ARCHS:
                label, _, marker = ARCH_STYLE[arch]
                colour = SERIES_COLOUR[(name, arch)]
                sub = table[
                    (table["class"] == class_name)
                    & (table["metric"] == "dice")
                    & (table["arch"] == arch)
                    & (table["contrast"] == name)
                    & (table["epsilon"] != "budget_avg")
                ].copy()
                sub["epsilon"] = sub["epsilon"].astype(float)
                sub = sub.sort_values("epsilon")
                err = np.vstack(
                    [
                        sub["hl_shift"] - sub["hl_ci95_low"],
                        sub["hl_ci95_high"] - sub["hl_shift"],
                    ]
                )
                ax.errorbar(
                    sub["epsilon"],
                    sub["hl_shift"],
                    yerr=err,
                    color=colour,
                    marker=marker,
                    linestyle=style,
                    markersize=4,
                    capsize=2,
                    linewidth=1.2,
                    label=f"{label}, {'ladder' if name == 'ladder_single' else 'ensemble'}",
                )
        ax.axhline(0, color="0.4", linewidth=0.8)
        ax.set_xlabel(r"$\varepsilon$")
        ax.set_title(class_name, fontsize=11, fontweight="bold")
        ax.grid(alpha=0.3)
    axes[0].set_ylabel("HL Dice shift (positive = more damage)")
    axes[-1].legend(fontsize=6, loc="best", ncol=2)
    fig.suptitle(
        "Budget-matched effects per epsilon: ladder (BLADE-DiceCE vs Auto-PGD, solid) "
        "and loss ensemble (SEA vs Auto-PGD x3, dashed)",
        fontsize=11,
        fontweight="bold",
    )
    out = os.path.join(out_dir, "factorial_matched_effects_by_eps_dice.png")
    fig.savefig(out, dpi=300, bbox_inches="tight")
    plt.close(fig)
    return out


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--output-dir", default=OUT_DIR)
    args = parser.parse_args()
    os.makedirs(args.output_dir, exist_ok=True)
    parts = []
    for class_name, (trees, _) in TREES.items():
        for arch in ARCHS:
            frames = {
                arm: load_per_sample(os.path.join(RESULTS, trees[arch]), arm)
                for arm in ARMS
            }
            parts.append(factorial_tests(frames, class_name, arch))
            print(f"{class_name} {arch}: done", flush=True)
    table = pd.concat(parts, ignore_index=True)
    table.to_csv(os.path.join(args.output_dir, "blade_factorial.csv"), index=False)
    with open(os.path.join(args.output_dir, "blade_factorial.md"), "w") as fh:
        fh.write(render_md(table))
    for path in [
        *plot_chain(table, args.output_dir),
        plot_ladder_by_eps(table, args.output_dir),
    ]:
        print("wrote", path)


if __name__ == "__main__":
    main()
