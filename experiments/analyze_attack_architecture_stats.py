"""Statistics on attack performance across the four architectures.

Two layers, both written to ``results/blade_attack/attack_comparison/stats/``:

1. **Cross-architecture, per attack** (new numbers). For every class, arm and
   metric a case x architecture matrix is built from the four trees' per-sample
   CSVs (all four score the same validation cases), on the per-case metric
   averaged over the attack budgets (epsilon > 0) and at each epsilon. Friedman
   across the four architectures, then paired Wilcoxon of each ResEnc variant
   against vanilla nnU-Net, Benjamini-Hochberg corrected within that
   three-comparison family, with the Hodges-Lehmann shift and its CI,
   rank-biserial correlation and Cohen's d_z. The shift is ResEnc minus
   nnU-Net; ``resenc_more_robust`` reads it with the metric's direction.

2. **Within-architecture ranking, consolidated** (existing numbers). The four
   trees' ``*_budget_avg_medians.csv`` and ``*_budget_avg_pairwise.csv`` are
   joined into one table per class and metric: arms in order of damage, and
   per architecture the median, the BH-adjusted p and Hodges-Lehmann shift of
   the arm against that architecture's strongest arm, plus whether the
   strongest arm is the same in all four.

ResEnc-M, -L and -XL are the same network on these datasets and differ only in
training batch size, so the architecture contrasts are nnU-Net versus ResEnc
plus batch-size variants, not a capacity sweep.

All paired machinery is imported from ``analyze_main_comparison_stats.py`` so
the two share provenance.

Usage::

    PYTHONPATH=src python experiments/analyze_attack_architecture_stats.py
"""

from __future__ import annotations

import argparse
import os
import sys

import numpy as np
import pandas as pd
from scipy import stats

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from analyze_main_comparison_stats import (  # noqa: E402
    ALPHA,
    BUDGET_GROUPS,
    METRIC_DIRECTION,
    _cohens_dz,
    _fdr_bh,
    _hodges_lehmann,
    _rank_biserial,
    load_per_sample,
    mean_over_epsilons,
    build_matrix,
)
from plot_architecture_comparison import ARCH_STYLE  # noqa: E402
from run_twelve_arm_wg_campaign import ARM_LABELS, METRICS  # noqa: E402

RESULTS = os.path.join("results", "blade_attack")
ARCHS = ("unet", "resenc_m", "resenc_l", "resenc_xl")
REFERENCE_ARCH = "unet"
# class -> (tree per architecture, stem of that tree's stats files)
TREES = {
    "WG": (
        {
            "unet": "whole_gland/native_per_eps",
            "resenc_m": "whole_gland/native_per_eps_resenc_m",
            "resenc_l": "whole_gland/native_per_eps_resenc_l",
            "resenc_xl": "whole_gland/native_per_eps_resenc_xl",
        },
        "main_comparison_wg",
    ),
    "TZ+CZ": (
        {
            "unet": "zones_confined_mc/campaign",
            "resenc_m": "zones_confined_mc/campaign_resenc_m",
            "resenc_l": "zones_confined_mc/campaign_resenc_l",
            "resenc_xl": "zones_confined_mc/campaign_resenc_xl",
        },
        "main_comparison_zones_TZCZ",
    ),
    "PZ": (
        {
            "unet": "zones_confined_mc/campaign",
            "resenc_m": "zones_confined_mc/campaign_resenc_m",
            "resenc_l": "zones_confined_mc/campaign_resenc_l",
            "resenc_xl": "zones_confined_mc/campaign_resenc_xl",
        },
        "main_comparison_zones_PZ",
    ),
}
ARMS = [a for a in ARM_LABELS if a not in ("a_pgd", "apgd_updated", "auto_pgd_plus")]
EPSILONS = [0.02, 0.04, 0.06, 0.08, 0.1]
CLASS_TAG = {"WG": "WG", "TZ+CZ": "TZCZ", "PZ": "PZ"}


def arch_matrix(
    frames: dict[str, dict[str, pd.DataFrame]],
    arm: str,
    metric: str,
    class_name: str,
    eps: float | None,
) -> pd.DataFrame:
    """Case x architecture matrix of one arm; ``eps=None`` averages the budgets."""
    cols = {}
    for arch in ARCHS:
        one = {arm: frames[arch][arm]}
        if eps is None:
            m = mean_over_epsilons(one, [arm], metric, EPSILONS, class_name)
        else:
            m = build_matrix(one, metric, eps, [arm], class_name)
        cols[arch] = m[arm]
    return pd.DataFrame(cols).dropna(axis=0, how="any")


def cross_arch_tests(
    frames: dict[str, dict[str, pd.DataFrame]], class_name: str
) -> tuple[pd.DataFrame, pd.DataFrame]:
    omnibus, pairwise = [], []
    for arm in ARMS:
        for metric in METRICS:
            for eps in [None, *EPSILONS]:
                mat = arch_matrix(frames, arm, metric, class_name, eps)
                eps_label = "budget_avg" if eps is None else eps
                chi2, p = stats.friedmanchisquare(*[mat[a].to_numpy() for a in ARCHS])
                n, k = mat.shape
                omnibus.append(
                    {
                        "class": class_name,
                        "arm": arm,
                        "metric": metric,
                        "epsilon": eps_label,
                        "n_cases": n,
                        "n_archs": k,
                        "friedman_chi2": chi2,
                        "p_value": p,
                        "kendall_w": chi2 / (n * (k - 1)),
                    }
                )
                rows = []
                ref = mat[REFERENCE_ARCH].to_numpy()
                for arch in ARCHS:
                    if arch == REFERENCE_ARCH:
                        continue
                    diff = mat[arch].to_numpy() - ref
                    nz = diff[diff != 0]
                    if nz.size == 0:
                        w_stat, w_p = float("nan"), float("nan")
                    else:
                        w_stat, w_p = stats.wilcoxon(nz)
                    hl, lo, hi = _hodges_lehmann(diff)
                    rows.append(
                        {
                            "class": class_name,
                            "arm": arm,
                            "metric": metric,
                            "epsilon": eps_label,
                            "reference": REFERENCE_ARCH,
                            "comparator": arch,
                            "n_cases": n,
                            "n_nonzero": int(nz.size),
                            "median_reference": float(np.median(ref)),
                            "median_comparator": float(np.median(mat[arch])),
                            "hl_shift": hl,
                            "hl_ci95_low": lo,
                            "hl_ci95_high": hi,
                            "wilcoxon_stat": w_stat,
                            "p_value": w_p,
                            "rank_biserial": _rank_biserial(diff),
                            "cohens_dz": _cohens_dz(diff),
                            "resenc_more_robust": bool(
                                np.sign(hl) == -METRIC_DIRECTION[metric]
                            ),
                        }
                    )
                fam = pd.DataFrame(rows)
                fam["p_adj_bh"] = _fdr_bh(fam["p_value"].to_numpy())
                fam["significant"] = fam["p_adj_bh"] < ALPHA
                pairwise.append(fam)
    return pd.DataFrame(omnibus), pd.concat(pairwise, ignore_index=True)


BLADE_ARMS = ("blade", "blade_mm", "blade_boundary", "blade1", "blade_frontier")
BASELINES = ("auto_pgd", "auto_pgd_r3", "sea")


def _budget_matched(a: str, b: str) -> bool:
    return any(a in m and b in m for m in BUDGET_GROUPS.values())


def blade_vs_baselines(
    frames: dict[str, dict[str, pd.DataFrame]], class_name: str
) -> pd.DataFrame:
    """Every BLADE arm against Auto-PGD, Auto-PGD x3 and SEA, per architecture.

    Paired by case, budget-averaged and per epsilon. The difference is signed
    with the metric direction so a positive Hodges-Lehmann shift always means
    the BLADE arm attacked harder. BH within each (architecture, metric,
    epsilon) family of 15 contrasts. ``budget_matched`` marks the pairs
    PROTOCOL.md certifies as spending the same gradient budget.
    """
    arms = list(BLADE_ARMS) + list(BASELINES)
    out = []
    for arch in ARCHS:
        for metric in METRICS:
            for eps in [None, *EPSILONS]:
                if eps is None:
                    mat = mean_over_epsilons(frames[arch], arms, metric, EPSILONS, class_name)
                else:
                    mat = build_matrix(frames[arch], metric, eps, arms, class_name)
                sign = METRIC_DIRECTION[metric]
                rows = []
                for blade_arm in BLADE_ARMS:
                    for base in BASELINES:
                        diff = sign * (mat[blade_arm] - mat[base]).to_numpy()
                        nz = diff[diff != 0]
                        w_stat, w_p = (
                            (float("nan"), float("nan"))
                            if nz.size == 0
                            else stats.wilcoxon(nz)
                        )
                        hl, lo, hi = _hodges_lehmann(diff)
                        rows.append(
                            {
                                "class": class_name,
                                "arch": arch,
                                "metric": metric,
                                "epsilon": "budget_avg" if eps is None else eps,
                                "blade_arm": blade_arm,
                                "baseline": base,
                                "budget_matched": _budget_matched(blade_arm, base),
                                "n_cases": len(mat),
                                "n_nonzero": int(nz.size),
                                "median_blade": float(mat[blade_arm].median()),
                                "median_baseline": float(mat[base].median()),
                                "hl_shift": hl,
                                "hl_ci95_low": lo,
                                "hl_ci95_high": hi,
                                "wilcoxon_stat": w_stat,
                                "p_value": w_p,
                                "rank_biserial": _rank_biserial(diff),
                                "cohens_dz": _cohens_dz(diff),
                                "blade_more_damaging": bool(hl > 0),
                            }
                        )
                fam = pd.DataFrame(rows)
                fam["p_adj_bh"] = _fdr_bh(fam["p_value"].to_numpy())
                fam["significant"] = fam["p_adj_bh"] < ALPHA
                out.append(fam)
    return pd.concat(out, ignore_index=True)


def consolidated_ranking(class_name: str) -> pd.DataFrame:
    trees, stem = TREES[class_name]
    medians = {
        a: pd.read_csv(os.path.join(RESULTS, t, f"{stem}_budget_avg_medians.csv"))
        for a, t in trees.items()
    }
    pairwise = {
        a: pd.read_csv(os.path.join(RESULTS, t, f"{stem}_budget_avg_pairwise.csv"))
        for a, t in trees.items()
    }
    out = []
    for metric in METRICS:
        strongest = {}
        for arch in ARCHS:
            med = medians[arch][medians[arch]["metric"] == metric].set_index("arm")
            # Median ties at 0 Dice are broken by the mean, as in the box-plot order.
            # Direction -1 (Dice): lowest median is strongest; +1: highest.
            key = med["median"] * METRIC_DIRECTION[metric] + 1e-9 * (
                med["mean"] * METRIC_DIRECTION[metric]
            )
            strongest[arch] = str(key.idxmax())
        agree = len(set(strongest.values())) == 1
        ref_med = medians[REFERENCE_ARCH]
        ref_med = ref_med[ref_med["metric"] == metric].set_index("arm")
        order = sorted(
            ARMS,
            key=lambda a: -ref_med.loc[a, "median"] * METRIC_DIRECTION[metric],
        )
        for rank, arm in enumerate(order, start=1):
            row = {
                "class": class_name,
                "metric": metric,
                "rank_nnunet": rank,
                "arm": arm,
                "label": ARM_LABELS[arm],
                "strongest_agrees": agree,
            }
            for arch in ARCHS:
                med = medians[arch][medians[arch]["metric"] == metric].set_index("arm")
                row[f"{arch}_median"] = med.loc[arm, "median"]
                row[f"{arch}_strongest"] = strongest[arch]
                if arm == strongest[arch]:
                    row[f"{arch}_p_adj_vs_strongest"] = np.nan
                    row[f"{arch}_hl_vs_strongest"] = 0.0
                    continue
                pw = pairwise[arch]
                pw = pw[pw["metric"] == metric]
                hit = pw[
                    ((pw["arm_a"] == strongest[arch]) & (pw["arm_b"] == arm))
                    | ((pw["arm_b"] == strongest[arch]) & (pw["arm_a"] == arm))
                ]
                if len(hit) != 1:
                    raise ValueError(
                        f"{class_name}/{arch}/{metric}: expected one pair "
                        f"({strongest[arch]}, {arm}), found {len(hit)}"
                    )
                hit = hit.iloc[0]
                sign = 1.0 if hit["arm_a"] == strongest[arch] else -1.0
                row[f"{arch}_p_adj_vs_strongest"] = hit["p_adj_bh"]
                row[f"{arch}_hl_vs_strongest"] = sign * hit["hl_shift"]
            out.append(row)
    return pd.DataFrame(out)


def _fmt_p(p: float) -> str:
    if not np.isfinite(p):
        return "-"
    return "<0.001" if p < 1e-3 else f"{p:.3f}"


def render_md(
    class_name: str,
    omnibus: pd.DataFrame,
    pairwise: pd.DataFrame,
    ranking: pd.DataFrame,
    blade: pd.DataFrame,
) -> str:
    names = {a: ARCH_STYLE[a][0] for a in ARCHS}
    lines = [
        f"# Attack performance statistics across architectures: {class_name}",
        "",
        "Paired by validation case across vanilla nnU-Net and the three ResEnc "
        "variants (same network, different training batch size). Budget-averaged "
        "rows use each case's metric averaged over epsilon 0.02-0.1; epsilon 0 is "
        "the shared clean segmentation and is excluded. Wilcoxon signed-rank, "
        "two-sided, Benjamini-Hochberg within each three-comparison family, "
        f"alpha = {ALPHA}. HL = Hodges-Lehmann shift (ResEnc minus nnU-Net).",
        "",
        "## 1. Does the architecture change how much an attack hurts? (budget-averaged)",
        "",
    ]
    for metric in METRICS:
        lines += [
            f"### {metric}",
            "",
            "| Attack | Friedman W | "
            + " | ".join(
                f"{names[a]} HL [CI] (p_adj)" for a in ARCHS if a != REFERENCE_ARCH
            )
            + " |",
            "|---|---|" + "---|" * (len(ARCHS) - 1),
        ]
        for arm in ARMS:
            om = omnibus[
                (omnibus["arm"] == arm)
                & (omnibus["metric"] == metric)
                & (omnibus["epsilon"] == "budget_avg")
            ].iloc[0]
            cells = []
            for arch in ARCHS:
                if arch == REFERENCE_ARCH:
                    continue
                r = pairwise[
                    (pairwise["arm"] == arm)
                    & (pairwise["metric"] == metric)
                    & (pairwise["epsilon"] == "budget_avg")
                    & (pairwise["comparator"] == arch)
                ].iloc[0]
                star = "*" if r["significant"] else ""
                cells.append(
                    f"{r['hl_shift']:+.3f} [{r['hl_ci95_low']:+.3f}, "
                    f"{r['hl_ci95_high']:+.3f}] ({_fmt_p(r['p_adj_bh'])}){star}"
                )
            lines.append(
                f"| {ARM_LABELS[arm]} | {om['kendall_w']:.3f} | "
                + " | ".join(cells)
                + " |"
            )
        lines.append("")
    lines += [
        "## 2. Within-architecture attack ranking (budget-averaged medians)",
        "",
        "Per architecture: median, then BH-adjusted p of the arm against that "
        "architecture's strongest arm (all-pairs Wilcoxon family from the tree's "
        "own stats). Rank is by the nnU-Net median.",
        "",
    ]
    for metric in METRICS:
        sub = ranking[ranking["metric"] == metric]
        strongest = {a: sub.iloc[0][f"{a}_strongest"] for a in ARCHS}
        lines += [
            f"### {metric}",
            "",
            "Strongest arm: "
            + ", ".join(f"{names[a]} = {ARM_LABELS[strongest[a]]}" for a in ARCHS)
            + (" (agree)" if sub.iloc[0]["strongest_agrees"] else " (disagree)"),
            "",
            "| # | Attack | "
            + " | ".join(f"{names[a]} median (p_adj)" for a in ARCHS)
            + " |",
            "|---|---|" + "---|" * len(ARCHS),
        ]
        for _, r in sub.iterrows():
            cells = [
                f"{r[f'{a}_median']:.3f} ({_fmt_p(r[f'{a}_p_adj_vs_strongest'])})"
                for a in ARCHS
            ]
            lines.append(
                f"| {r['rank_nnunet']} | {r['label']} | " + " | ".join(cells) + " |"
            )
        lines.append("")
    lines += [
        "## 3. Does the BLADE family out-attack Auto-PGD and SEA? (budget-averaged)",
        "",
        "Per architecture: Hodges-Lehmann shift signed so positive = BLADE arm more "
        "damaging, with BH-adjusted p (family = the 15 BLADE x baseline contrasts of "
        "that architecture and metric). `=` marks budget-matched pairs.",
        "",
    ]
    for metric in METRICS:
        sub = blade[(blade["metric"] == metric) & (blade["epsilon"] == "budget_avg")]
        lines += [
            f"### {metric}",
            "",
            "| BLADE arm | vs | " + " | ".join(f"{names[a]} HL (p_adj)" for a in ARCHS) + " |",
            "|---|---|" + "---|" * len(ARCHS),
        ]
        for blade_arm in BLADE_ARMS:
            for base in BASELINES:
                cells = []
                for arch in ARCHS:
                    r = sub[
                        (sub["arch"] == arch)
                        & (sub["blade_arm"] == blade_arm)
                        & (sub["baseline"] == base)
                    ].iloc[0]
                    star = "*" if r["significant"] else ""
                    cells.append(f"{r['hl_shift']:+.3f} ({_fmt_p(r['p_adj_bh'])}){star}")
                eq = " =" if _budget_matched(blade_arm, base) else ""
                lines.append(
                    f"| {ARM_LABELS[blade_arm]} | {ARM_LABELS[base]}{eq} | "
                    + " | ".join(cells)
                    + " |"
                )
        lines.append("")
    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument(
        "--output-dir",
        default=os.path.join(RESULTS, "attack_comparison", "stats"),
    )
    parser.add_argument("--classes", nargs="+", default=list(TREES))
    args = parser.parse_args()
    os.makedirs(args.output_dir, exist_ok=True)
    for class_name in args.classes:
        trees, _ = TREES[class_name]
        frames = {
            arch: {arm: load_per_sample(os.path.join(RESULTS, t), arm) for arm in ARMS}
            for arch, t in trees.items()
        }
        omnibus, pairwise = cross_arch_tests(frames, class_name)
        ranking = consolidated_ranking(class_name)
        blade = blade_vs_baselines(frames, class_name)
        tag = CLASS_TAG[class_name]
        blade.to_csv(
            os.path.join(args.output_dir, f"blade_vs_apgd_sea_{tag}.csv"), index=False
        )
        omnibus.to_csv(
            os.path.join(args.output_dir, f"cross_arch_friedman_{tag}.csv"), index=False
        )
        pairwise.to_csv(
            os.path.join(args.output_dir, f"cross_arch_wilcoxon_{tag}.csv"), index=False
        )
        ranking.to_csv(
            os.path.join(args.output_dir, f"attack_ranking_by_arch_{tag}.csv"),
            index=False,
        )
        with open(
            os.path.join(args.output_dir, f"attack_architecture_stats_{tag}.md"), "w"
        ) as fh:
            fh.write(render_md(class_name, omnibus, pairwise, ranking, blade))
        n = omnibus["n_cases"].unique().tolist()
        print(
            f"{class_name}: n_cases={n}, {len(omnibus)} omnibus rows, {len(pairwise)} pairwise rows"
        )


if __name__ == "__main__":
    main()
