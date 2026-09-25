"""Figures for the cross-architecture attack statistics.

Reads the CSVs written by ``analyze_attack_architecture_stats.py`` and draws,
into the same ``stats/`` directory:

* ``forest_<metric>.png`` -- one panel per class; per attack the budget-averaged
  Hodges-Lehmann shift (ResEnc minus nnU-Net) with its 95% CI for each ResEnc
  variant. Hollow markers are contrasts that are not significant after BH.
* ``heatmap_<class>_<metric>.png`` -- one panel per ResEnc variant; attack x
  epsilon grid of the Hodges-Lehmann shift, so the growth of the gap with the
  budget is visible.
* ``rank_bump_<class>.png`` -- one panel per metric; each attack's rank (1 =
  most damaging, by budget-averaged median) across the four architectures.
  A filled marker means the arm is significantly weaker than that
  architecture's strongest arm; hollow means statistically tied with it.

Usage::

    PYTHONPATH=src python experiments/plot_attack_architecture_stats.py
"""

from __future__ import annotations

import argparse
import os
import sys

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from analyze_attack_architecture_stats import (  # noqa: E402
    ALPHA,
    ARCHS,
    ARMS,
    BASELINES,
    BLADE_ARMS,
    CLASS_TAG,
    EPSILONS,
    METRIC_DIRECTION,
    _budget_matched,
)
from plot_architecture_comparison import ARCH_STYLE  # noqa: E402
from run_twelve_arm_wg_campaign import ARM_LABELS, METRIC_LABELS, METRICS  # noqa: E402

RESENC = [a for a in ARCHS if a != "unet"]
CLASS_TITLE = {"WG": "Whole gland", "TZ+CZ": "TZ+CZ", "PZ": "PZ"}
STATS_DIR = os.path.join("results", "blade_attack", "attack_comparison", "stats")


def load(stats_dir: str, kind: str, class_name: str) -> pd.DataFrame:
    return pd.read_csv(os.path.join(stats_dir, f"{kind}_{CLASS_TAG[class_name]}.csv"))


def plot_forest(stats_dir: str, metric: str) -> str:
    classes = list(CLASS_TAG)
    fig, axes = plt.subplots(
        1,
        len(classes),
        figsize=(4.6 * len(classes), 5.2),
        sharey=True,
        layout="constrained",
    )
    y = np.arange(len(ARMS))
    offsets = np.linspace(-0.27, 0.27, len(RESENC))
    for ax, class_name in zip(axes, classes):
        w = load(stats_dir, "cross_arch_wilcoxon", class_name)
        w = w[(w["epsilon"] == "budget_avg") & (w["metric"] == metric)]
        for off, arch in zip(offsets, RESENC):
            label, colour, marker = ARCH_STYLE[arch]
            rows = w[w["comparator"] == arch].set_index("arm").loc[ARMS]
            sig = rows["significant"].to_numpy(dtype=bool)
            x = rows["hl_shift"].to_numpy()
            err = np.vstack([x - rows["hl_ci95_low"], rows["hl_ci95_high"] - x])
            ax.errorbar(
                x,
                y + off,
                xerr=err,
                fmt="none",
                ecolor=colour,
                elinewidth=1.2,
                capsize=2,
            )
            ax.scatter(
                x[sig],
                (y + off)[sig],
                marker=marker,
                color=colour,
                s=28,
                label=label,
                zorder=3,
            )
            ax.scatter(
                x[~sig],
                (y + off)[~sig],
                marker=marker,
                facecolors="white",
                edgecolors=colour,
                s=28,
                zorder=3,
            )
        ax.axvline(0, color="0.4", linewidth=0.8)
        ax.set_yticks(y)
        ax.set_yticklabels([ARM_LABELS[a] for a in ARMS])
        ax.invert_yaxis()
        ax.grid(axis="x", alpha=0.3)
        ax.set_title(CLASS_TITLE[class_name], fontsize=11, fontweight="bold")
        direction = (
            "ResEnc more robust"
            if METRIC_DIRECTION[metric] < 0
            else "nnU-Net more robust"
        )
        ax.set_xlabel(
            f"HL shift in {METRIC_LABELS[metric]} (ResEnc - nnU-Net)\n-> {direction}"
        )
        if metric != "dice":
            ax.set_xscale("symlog", linthresh=1.0)
    axes[0].legend(fontsize=8, loc="upper right")
    fig.suptitle(
        f"Architecture effect per attack, budget-averaged {METRIC_LABELS[metric]} "
        f"(95% CI; hollow = not significant at BH alpha={ALPHA})",
        fontsize=12,
        fontweight="bold",
    )
    out = os.path.join(stats_dir, f"forest_{metric}.png")
    fig.savefig(out, dpi=300, bbox_inches="tight")
    plt.close(fig)
    return out


def plot_heatmap(stats_dir: str, class_name: str, metric: str) -> str:
    w = load(stats_dir, "cross_arch_wilcoxon", class_name)
    w = w[(w["epsilon"] != "budget_avg") & (w["metric"] == metric)].copy()
    w["epsilon"] = w["epsilon"].astype(float)
    grids = {
        arch: w[w["comparator"] == arch]
        .pivot(index="arm", columns="epsilon", values="hl_shift")
        .loc[ARMS, EPSILONS]
        for arch in RESENC
    }
    vmax = max(np.nanmax(np.abs(g.to_numpy())) for g in grids.values())
    fig, axes = plt.subplots(
        1,
        len(RESENC),
        figsize=(3.6 * len(RESENC), 5.0),
        sharey=True,
        layout="constrained",
    )
    for ax, arch in zip(axes, RESENC):
        g = grids[arch]
        im = ax.imshow(
            g.to_numpy(),
            cmap="RdBu" if METRIC_DIRECTION[metric] < 0 else "RdBu_r",
            vmin=-vmax,
            vmax=vmax,
            aspect="auto",
        )
        ax.set_xticks(range(len(EPSILONS)))
        ax.set_xticklabels([f"{e:g}" for e in EPSILONS])
        ax.set_yticks(range(len(ARMS)))
        ax.set_yticklabels([ARM_LABELS[a] for a in ARMS])
        ax.set_xlabel(r"$\varepsilon$")
        ax.set_title(ARCH_STYLE[arch][0], fontsize=11, fontweight="bold")
        sig = (
            w[w["comparator"] == arch]
            .pivot(index="arm", columns="epsilon", values="significant")
            .loc[ARMS, EPSILONS]
            .to_numpy(dtype=bool)
        )
        for i in range(len(ARMS)):
            for j in range(len(EPSILONS)):
                v = g.iat[i, j]
                ax.text(
                    j,
                    i,
                    f"{v:+.2f}" if abs(v) < 10 else f"{v:+.0f}",
                    ha="center",
                    va="center",
                    fontsize=6.5,
                    color="black" if abs(v) < 0.6 * vmax else "white",
                    fontweight="bold" if sig[i, j] else "normal",
                )
    cbar = fig.colorbar(im, ax=axes, shrink=0.8)
    cbar.set_label(
        f"HL shift in {METRIC_LABELS[metric]} (ResEnc - nnU-Net); blue = ResEnc more robust"
    )
    fig.suptitle(
        f"{CLASS_TITLE[class_name]}: architecture effect per attack and budget "
        "(bold = significant after BH)",
        fontsize=12,
        fontweight="bold",
    )
    out = os.path.join(stats_dir, f"heatmap_{CLASS_TAG[class_name]}_{metric}.png")
    fig.savefig(out, dpi=300, bbox_inches="tight")
    plt.close(fig)
    return out


def plot_rank_bump(stats_dir: str, class_name: str) -> str:
    r = load(stats_dir, "attack_ranking_by_arch", class_name)
    fig, axes = plt.subplots(
        1,
        len(METRICS),
        figsize=(4.8 * len(METRICS), 5.4),
        sharey=True,
        layout="constrained",
    )
    cmap = plt.get_cmap("tab20")
    x = np.arange(len(ARCHS))
    for ax, metric in zip(axes, METRICS):
        sub = r[r["metric"] == metric].set_index("arm")
        for k, arm in enumerate(ARMS):
            ranks = []
            for a in ARCHS:
                col = sub[f"{a}_median"] * METRIC_DIRECTION[metric]
                ranks.append(int((col.rank(ascending=False)).loc[arm]))
            ranks_arr = np.array(ranks)
            p = np.array(
                [sub.loc[arm, f"{a}_p_adj_vs_strongest"] for a in ARCHS], dtype=float
            )
            tied = np.isnan(p) | (p >= ALPHA)
            colour = cmap(k % 20)
            ax.plot(x, ranks_arr, color=colour, linewidth=1.6, zorder=2)
            ax.scatter(x[~tied], ranks_arr[~tied], color=colour, s=36, zorder=3)
            ax.scatter(
                x[tied],
                ranks_arr[tied],
                facecolors="white",
                edgecolors=colour,
                s=36,
                zorder=3,
            )
            ax.text(
                x[-1] + 0.12,
                ranks_arr[-1],
                ARM_LABELS[arm],
                fontsize=7.5,
                va="center",
                color=colour,
            )
        ax.set_xticks(x)
        ax.set_xticklabels([ARCH_STYLE[a][0] for a in ARCHS], fontsize=8)
        ax.set_xlim(-0.3, len(ARCHS) - 0.3 + 1.0)
        ax.set_yticks(range(1, len(ARMS) + 1))
        ax.invert_yaxis()
        ax.grid(axis="y", alpha=0.3)
        ax.set_title(METRIC_LABELS[metric], fontsize=11, fontweight="bold")
    axes[0].set_ylabel("rank by budget-averaged median (1 = most damaging)")
    fig.suptitle(
        f"{CLASS_TITLE[class_name]}: attack ranking across architectures "
        "(hollow = tied with that architecture's strongest arm)",
        fontsize=12,
        fontweight="bold",
    )
    out = os.path.join(stats_dir, f"rank_bump_{CLASS_TAG[class_name]}.png")
    fig.savefig(out, dpi=300, bbox_inches="tight")
    plt.close(fig)
    return out


def plot_blade_vs_baselines(stats_dir: str, metric: str) -> str:
    """Forest of every BLADE arm against Auto-PGD / Auto-PGD x3 / SEA, per class."""
    classes = list(CLASS_TAG)
    pairs = [(b, base) for b in BLADE_ARMS for base in BASELINES]
    fig, axes = plt.subplots(
        1,
        len(classes),
        figsize=(4.8 * len(classes), 6.2),
        sharey=True,
        layout="constrained",
    )
    y = np.arange(len(pairs))
    offsets = np.linspace(-0.3, 0.3, len(ARCHS))
    for ax, class_name in zip(axes, classes):
        d = load(stats_dir, "blade_vs_apgd_sea", class_name)
        d = d[(d["epsilon"] == "budget_avg") & (d["metric"] == metric)]
        for off, arch in zip(offsets, ARCHS):
            label, colour, marker = ARCH_STYLE[arch]
            rows = d[d["arch"] == arch].set_index(["blade_arm", "baseline"]).loc[pairs]
            sig = rows["significant"].to_numpy(dtype=bool)
            x = rows["hl_shift"].to_numpy()
            err = np.vstack([x - rows["hl_ci95_low"], rows["hl_ci95_high"] - x])
            ax.errorbar(x, y + off, xerr=err, fmt="none", ecolor=colour, elinewidth=1.1, capsize=2)
            ax.scatter(x[sig], (y + off)[sig], marker=marker, color=colour, s=26, label=label, zorder=3)
            ax.scatter(
                x[~sig], (y + off)[~sig], marker=marker, facecolors="white",
                edgecolors=colour, s=26, zorder=3,
            )
        ax.axvline(0, color="0.4", linewidth=0.8)
        ax.set_yticks(y)
        ax.set_yticklabels(
            [
                f"{ARM_LABELS[b]} vs {ARM_LABELS[base]}"
                + (" =" if _budget_matched(b, base) else "")
                for b, base in pairs
            ],
            fontsize=8,
        )
        ax.invert_yaxis()
        ax.grid(axis="x", alpha=0.3)
        ax.set_title(CLASS_TITLE[class_name], fontsize=11, fontweight="bold")
        ax.set_xlabel(f"HL shift in {METRIC_LABELS[metric]}\n-> BLADE arm more damaging")
        if metric != "dice":
            ax.set_xscale("symlog", linthresh=1.0)
    axes[0].legend(fontsize=8, loc="lower right")
    fig.suptitle(
        f"BLADE family vs Auto-PGD and SEA, budget-averaged {METRIC_LABELS[metric]} "
        f"(95% CI; hollow = not significant at BH alpha={ALPHA}; = budget-matched)",
        fontsize=12,
        fontweight="bold",
    )
    out = os.path.join(stats_dir, f"blade_vs_apgd_sea_{metric}.png")
    fig.savefig(out, dpi=300, bbox_inches="tight")
    plt.close(fig)
    return out


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--stats-dir", default=STATS_DIR)
    args = parser.parse_args()
    written = [plot_forest(args.stats_dir, m) for m in METRICS]
    written += [plot_blade_vs_baselines(args.stats_dir, m) for m in METRICS]
    for class_name in CLASS_TAG:
        written += [plot_heatmap(args.stats_dir, class_name, m) for m in METRICS]
        written.append(plot_rank_bump(args.stats_dir, class_name))
    for w in written:
        print("wrote", w)


if __name__ == "__main__":
    main()
