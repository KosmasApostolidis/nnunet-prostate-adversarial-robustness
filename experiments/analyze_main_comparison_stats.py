"""Paired statistics and box-plots for the twelve-arm whole-gland comparison.

``main_comparison_wg_summary.csv`` holds fold-level aggregates -- one mean, SD
and SEM per (arm, metric, epsilon) over five folds. Box-plots need
distributions and paired tests need per-case pairing, so this script reads the
summary only as a *roster* (which arms, epsilons and metrics belong to the
comparison) and takes its numbers from the per-sample CSVs that sit beside it.
Every arm scores the same 1,396 validation cases at the same six epsilons, so
the design is fully paired and non-parametric paired tests apply directly.

Per (metric, epsilon > 0):

  * Friedman omnibus across all roster arms, blocked on case.
  * Post-hoc paired Wilcoxon of the strongest arm against each other arm,
    Benjamini-Hochberg corrected within that family.
  * Rank-biserial correlation and Cohen's d_z as paired effect sizes.

It then repeats the comparison on the quantity the figure plots -- each case's
metric averaged over the attack budgets -- with a per-arm median and its exact
order-statistic CI, and an all-pairs paired Wilcoxon carrying the
Hodges-Lehmann median shift. All pairs rather than one reference, because a
ranked box-plot invites the question of whether two neighbouring boxes differ.

Plus the budget-matched contrasts ``PROTOCOL.md`` names as the only fair
cross-arm comparisons, and a single box-plot figure: one subplot per metric,
one box per attack, each case averaged over the five attack budgets.

Epsilon 0 is the same clean segmentation for every arm, so it is drawn once as
a reference line and excluded from every test.

Usage::

    PYTHONPATH=src python experiments/analyze_main_comparison_stats.py \
        --summary results/blade_attack/whole_gland/native_per_eps/main_comparison_wg_summary.csv
"""

from __future__ import annotations

import argparse
import glob
import os
import sys

import numpy as np
import pandas as pd
from scipy import stats

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from run_twelve_arm_wg_campaign import (  # noqa: E402
    ARM_LABELS,
    ARM_STYLE,
    METRIC_LABELS,
)

# Higher is a stronger attack for the distance metrics; lower is stronger for
# Dice. Used to pick the strongest arm and to sign every effect size so that
# positive always means "the reference arm attacked harder".
METRIC_DIRECTION = {"dice": -1, "hd95": +1, "asd": +1}

# The comparisons PROTOCOL.md certifies as budget-matched: arms in a group
# spend the same number of gradient evaluations over the whole epsilon curve.
BUDGET_GROUPS = {
    "300-315 gradients": ["blade", "sea", "auto_pgd_r3"],
    "100 gradients": [
        "blade1",
        "blade_boundary",
        "blade_frontier",
        "pgd",
        "auto_pgd",
        "segpgd",
        "cospgd",
        "dag",
    ],
}

ALPHA = 0.05

# ARM_STYLE shares a hue across each attack family and separates members by
# marker and linestyle. A box-plot has neither channel, so four hues there map
# to two or three arms each. These twelve are Paul Tol's qualitative scheme,
# one per arm, chosen for separability rather than family resemblance.
BOX_COLOURS = {
    "fgsm": "#DDDDDD",
    "pgd": "#88CCEE",
    "auto_pgd": "#44AA99",
    "segpgd": "#DDCC77",
    "cospgd": "#6699CC",
    "dag": "#999933",
    "sea": "#117733",
    "blade1": "#CC6677",
    "blade": "#882255",
    "auto_pgd_r3": "#332288",
    "blade_boundary": "#AA4499",
    "blade_frontier": "#661100",
    "blade_mm": "#EE8866",
}



def _fdr_bh(pvals: np.ndarray) -> np.ndarray:
    """Benjamini-Hochberg adjusted p-values (two-sided tests)."""
    p = np.asarray(pvals, dtype=float)
    out = np.full(len(p), np.nan, dtype=float)
    finite_idx = np.where(np.isfinite(p))[0]
    if finite_idx.size == 0:
        return out
    pv = p[finite_idx]
    m = len(pv)
    order = np.argsort(pv)
    adj = pv[order] * m / np.arange(1, m + 1, dtype=float)
    adj = np.clip(np.minimum.accumulate(adj[::-1])[::-1], 0.0, 1.0)
    result = np.empty(m)
    result[order] = adj
    out[finite_idx] = result
    return out


def _cohens_dz(diff: np.ndarray) -> float:
    """Paired effect size: mean(difference) / SD(differences)."""
    d = diff[np.isfinite(diff)]
    if len(d) < 2:
        return float("nan")
    sd = d.std(ddof=1)
    return float(d.mean() / sd) if sd > 0 else float("nan")


def _rank_biserial(diff: np.ndarray) -> float:
    """Matched-pairs rank-biserial correlation, the Wilcoxon effect size.

    (sum of positive ranks - sum of negative ranks) / total rank sum, computed
    over non-zero differences. Ranges [-1, 1]; ties contribute nothing.
    """
    d = diff[np.isfinite(diff)]
    d = d[d != 0]
    if len(d) == 0:
        return float("nan")
    ranks = stats.rankdata(np.abs(d))
    total = ranks.sum()
    return float((ranks[d > 0].sum() - ranks[d < 0].sum()) / total)


def load_per_sample(campaign_dir: str, arm: str) -> pd.DataFrame:
    """Concatenate an arm's five per-fold per-sample CSVs.

    The filename carries the model key and the *attack* name -- the attack
    differs from the arm name for compute-matched controls (``auto_pgd_r3``
    runs ``auto_pgd``) -- so the glob fixes neither. ``wg_blade_fold0`` and
    ``zones_blade1_fold0`` both match.
    """
    pattern = os.path.join(campaign_dir, arm, "*_fold*_per_sample.csv")
    files = sorted(glob.glob(pattern))
    if not files:
        raise FileNotFoundError(f"no per-sample CSVs for arm {arm!r}: {pattern}")
    frame = pd.concat([pd.read_csv(f) for f in files], ignore_index=True)
    frame["arm"] = arm
    return frame


def build_matrix(
    frames: dict[str, pd.DataFrame],
    metric: str,
    eps: float,
    arms: list[str],
    class_name: str,
) -> pd.DataFrame:
    """Case x arm matrix of one metric at one epsilon and class.

    The class filter is what makes a multi-class model tractable: its
    per-sample CSVs carry one row per foreground class, so selecting on
    epsilon alone would leave two rows per case and silently misalign the
    arms against each other.
    """
    columns = {}
    for arm in arms:
        f = frames[arm]
        sub = f[
            np.isclose(f["epsilon"].astype(float), eps)
            & (f["class"] == class_name)
        ]
        if sub["case_id"].duplicated().any():
            raise ValueError(
                f"{arm}: duplicate rows for class {class_name!r} at eps={eps}; "
                "the per-sample CSVs mix runs and cannot be paired."
            )
        columns[arm] = sub.set_index("case_id")[metric].astype(float)
    matrix = pd.DataFrame(columns)
    return matrix.dropna(axis=0, how="any")


def strongest_arm(matrix: pd.DataFrame, metric: str) -> str:
    """The arm with the lowest mean per-case rank, where rank 1 = most damaging.

    Selecting on the median fails at large epsilon: once an attack drives most
    cases to Dice 0 several arms share a median of exactly 0 and the winner is
    decided by dictionary order. Mean rank is the quantity Friedman already
    blocks on, and it separates arms that a tied median cannot.
    """
    sign = METRIC_DIRECTION[metric]
    ranks = (-sign * matrix).rank(axis=1, method="average")
    return str(ranks.mean().idxmin())


def run_tests(
    frames: dict[str, pd.DataFrame],
    arms: list[str],
    metrics: list[str],
    epsilons: list[float],
    class_name: str,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Friedman omnibus plus post-hoc Wilcoxon against the strongest arm."""
    omnibus_rows: list[dict] = []
    posthoc_rows: list[dict] = []

    for metric in metrics:
        sign = METRIC_DIRECTION[metric]
        for eps in epsilons:
            matrix = build_matrix(frames, metric, eps, arms, class_name)
            n = len(matrix)

            chi2, p_omni = stats.friedmanchisquare(
                *[matrix[a].to_numpy() for a in arms]
            )
            # Kendall's W: Friedman chi-square normalised by cases and arms.
            kendall_w = float(chi2 / (n * (len(arms) - 1)))
            omnibus_rows.append(
                dict(
                    metric=metric,
                    epsilon=eps,
                    n_cases=n,
                    n_arms=len(arms),
                    friedman_chi2=float(chi2),
                    p_value=float(p_omni),
                    kendall_w=kendall_w,
                )
            )

            reference = strongest_arm(matrix, metric)
            block: list[dict] = []
            for arm in arms:
                if arm == reference:
                    continue
                # Signed so a positive difference always means the reference
                # arm did more damage than the comparator.
                diff = sign * (matrix[reference] - matrix[arm]).to_numpy()
                nonzero = int((diff != 0).sum())
                if nonzero == 0:
                    stat, p = float("nan"), float("nan")
                else:
                    stat, p = stats.wilcoxon(
                        diff, zero_method="wilcox", alternative="two-sided"
                    )
                block.append(
                    dict(
                        metric=metric,
                        epsilon=eps,
                        reference=reference,
                        comparator=arm,
                        n_cases=n,
                        n_nonzero=nonzero,
                        median_reference=float(matrix[reference].median()),
                        median_comparator=float(matrix[arm].median()),
                        median_diff=float(np.median(diff)),
                        wilcoxon_stat=float(stat),
                        p_value=float(p),
                        rank_biserial=_rank_biserial(diff),
                        cohens_dz=_cohens_dz(diff),
                    )
                )
            adjusted = _fdr_bh(np.array([row["p_value"] for row in block]))
            for row, q in zip(block, adjusted):
                row["p_adj_bh"] = float(q)
                sig = bool(np.isfinite(q) and q < ALPHA)
                row["significant"] = sig
                row["reference_wins"] = bool(sig and row["rank_biserial"] > 0)
                row["reference_loses"] = bool(sig and row["rank_biserial"] < 0)
            posthoc_rows.extend(block)

    return pd.DataFrame(omnibus_rows), pd.DataFrame(posthoc_rows)


def _hodges_lehmann(diff: np.ndarray, alpha: float = ALPHA) -> tuple[float, float, float]:
    """Median shift of paired differences with a distribution-free CI.

    The Hodges-Lehmann estimator is the median of the Walsh averages
    ``(d_i + d_j) / 2`` over ``i <= j`` -- the location the Wilcoxon signed-rank
    test is actually testing, so estimate and p-value always agree. The
    interval takes the order statistics at the normal-approximation critical
    rank, exact enough at n in the thousands.
    """
    d = diff[np.isfinite(diff)]
    n = len(d)
    if n < 2:
        return float("nan"), float("nan"), float("nan")
    walsh = np.add.outer(d, d)[np.triu_indices(n)] / 2.0
    walsh.sort()
    m = len(walsh)
    estimate = float(np.median(walsh))
    z = stats.norm.isf(alpha / 2.0)
    spread = z * np.sqrt(n * (n + 1) * (2 * n + 1) / 24.0)
    k = int(np.floor(n * (n + 1) / 4.0 - spread))
    if k < 0:
        return estimate, float("nan"), float("nan")
    return estimate, float(walsh[k]), float(walsh[m - 1 - k])


def median_summary(
    frames: dict[str, pd.DataFrame],
    arms: list[str],
    metrics: list[str],
    epsilons: list[float],
    class_name: str,
) -> pd.DataFrame:
    """Per-arm median of the budget-averaged score, with an exact CI.

    The quantity is a composite: for each case the metric is first averaged
    over the attack budgets, then the median is taken across cases. It is the
    line inside each box of the figure, not a median of raw metric values.

    The interval uses order statistics -- the distribution-free CI for a
    median, from the binomial distribution of how many observations fall below
    it -- so it makes no assumption about the shape of the scores.
    """
    rows: list[dict] = []
    for metric in metrics:
        matrix = mean_over_epsilons(frames, arms, metric, epsilons, class_name)
        n = len(matrix)
        lo_rank = int(stats.binom.ppf(ALPHA / 2.0, n, 0.5))
        hi_rank = n - lo_rank - 1
        for arm in arms:
            values = np.sort(matrix[arm].to_numpy())
            rows.append(
                dict(
                    metric=metric,
                    arm=arm,
                    label=ARM_LABELS[arm],
                    n_cases=n,
                    median=float(np.median(values)),
                    ci95_low=float(values[lo_rank]),
                    ci95_high=float(values[hi_rank]),
                    q1=float(np.quantile(values, 0.25)),
                    q3=float(np.quantile(values, 0.75)),
                    mean=float(values.mean()),
                )
            )
    return pd.DataFrame(rows)


def pairwise_median_tests(
    frames: dict[str, pd.DataFrame],
    arms: list[str],
    metrics: list[str],
    epsilons: list[float],
    class_name: str,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """All-pairs paired Wilcoxon on the budget-averaged score.

    Every pair is tested rather than each arm against the strongest, because a
    ranked box-plot invites the question the strongest-versus-rest design
    cannot answer: whether two arms sitting next to each other differ at all.
    Sixty-six pairs per metric is a small family; BH corrects within a metric.
    """
    omnibus_rows: list[dict] = []
    pair_rows: list[dict] = []

    for metric in metrics:
        sign = METRIC_DIRECTION[metric]
        matrix = mean_over_epsilons(frames, arms, metric, epsilons, class_name)
        n = len(matrix)

        chi2, p_omni = stats.friedmanchisquare(*[matrix[a].to_numpy() for a in arms])
        omnibus_rows.append(
            dict(
                metric=metric,
                n_cases=n,
                n_arms=len(arms),
                friedman_chi2=float(chi2),
                p_value=float(p_omni),
                kendall_w=float(chi2 / (n * (len(arms) - 1))),
            )
        )

        block: list[dict] = []
        for i, arm_a in enumerate(arms):
            for arm_b in arms[i + 1 :]:
                # Signed so a positive difference always means arm_a did more
                # damage, for both metric directions.
                diff = sign * (matrix[arm_a] - matrix[arm_b]).to_numpy()
                if (diff != 0).sum() == 0:
                    stat, p = float("nan"), float("nan")
                else:
                    stat, p = stats.wilcoxon(diff, alternative="two-sided")
                shift, shift_lo, shift_hi = _hodges_lehmann(diff)
                block.append(
                    dict(
                        metric=metric,
                        arm_a=arm_a,
                        arm_b=arm_b,
                        label_a=ARM_LABELS[arm_a],
                        label_b=ARM_LABELS[arm_b],
                        n_cases=n,
                        median_a=float(matrix[arm_a].median()),
                        median_b=float(matrix[arm_b].median()),
                        hl_shift=shift,
                        hl_ci95_low=shift_lo,
                        hl_ci95_high=shift_hi,
                        wilcoxon_stat=float(stat),
                        p_value=float(p),
                        rank_biserial=_rank_biserial(diff),
                        cohens_dz=_cohens_dz(diff),
                    )
                )
        adjusted = _fdr_bh(np.array([r["p_value"] for r in block]))
        for row, q in zip(block, adjusted):
            row["p_adj_bh"] = float(q)
            sig = bool(np.isfinite(q) and q < ALPHA)
            row["significant"] = sig
            row["a_more_damaging"] = bool(sig and row["rank_biserial"] > 0)
        pair_rows.extend(block)

    return pd.DataFrame(omnibus_rows), pd.DataFrame(pair_rows)


def ranking_chain(
    medians: pd.DataFrame, pairs: pd.DataFrame, metric: str
) -> tuple[str, list[str]]:
    """Median-sorted arms joined by the paired test's verdict on each neighbour.

    The order follows the figure's boxes, sorted by median. The joiner reports
    what the paired Wilcoxon found for that adjacent pair, which is not always
    what the median order implies: the median is a marginal summary while the
    test is paired, so an arm with the higher median can still lose case by
    case. Those inversions are marked '<' and returned separately rather than
    smoothed over.
    """
    block = medians[medians.metric == metric]
    sign = METRIC_DIRECTION[metric]
    ordered = block.assign(key=sign * block["median"]).sort_values(
        "key", ascending=False
    )
    names = list(ordered["arm"])
    out = [ARM_LABELS[names[0]]]
    inversions: list[str] = []
    for left, right in zip(names, names[1:]):
        row = pairs[
            (pairs.metric == metric)
            & (
                ((pairs.arm_a == left) & (pairs.arm_b == right))
                | ((pairs.arm_a == right) & (pairs.arm_b == left))
            )
        ].iloc[0]
        if not bool(row.significant):
            joiner = " ~ "
        else:
            # rank_biserial is signed so positive means arm_a did more damage.
            left_wins = (row.rank_biserial > 0) == (row.arm_a == left)
            joiner = " > " if left_wins else " < "
            if not left_wins:
                inversions.append(
                    f"{ARM_LABELS[right]} beats {ARM_LABELS[left]} case by case "
                    f"(HL shift {row.hl_shift:+.3g}, q={row.p_adj_bh:.2g}) "
                    f"despite the lower median"
                )
        out.append(joiner + ARM_LABELS[right])
    return "".join(out), inversions


def budget_matched_tests(
    frames: dict[str, pd.DataFrame],
    arms: list[str],
    metrics: list[str],
    epsilons: list[float],
    class_name: str,
) -> pd.DataFrame:
    """Within each budget group, test its strongest arm against the others."""
    rows: list[dict] = []
    for group_name, members in BUDGET_GROUPS.items():
        present = [a for a in members if a in arms]
        if len(present) < 2:
            continue
        for metric in metrics:
            sign = METRIC_DIRECTION[metric]
            for eps in epsilons:
                matrix = build_matrix(frames, metric, eps, present, class_name)
                reference = strongest_arm(matrix, metric)
                block: list[dict] = []
                for arm in present:
                    if arm == reference:
                        continue
                    diff = sign * (matrix[reference] - matrix[arm]).to_numpy()
                    if (diff != 0).sum() == 0:
                        stat, p = float("nan"), float("nan")
                    else:
                        stat, p = stats.wilcoxon(diff, alternative="two-sided")
                    block.append(
                        dict(
                            budget_group=group_name,
                            metric=metric,
                            epsilon=eps,
                            reference=reference,
                            comparator=arm,
                            n_cases=len(matrix),
                            median_reference=float(matrix[reference].median()),
                            median_comparator=float(matrix[arm].median()),
                            wilcoxon_stat=float(stat),
                            p_value=float(p),
                            rank_biserial=_rank_biserial(diff),
                            cohens_dz=_cohens_dz(diff),
                        )
                    )
                adjusted = _fdr_bh(np.array([r["p_value"] for r in block]))
                for row, q in zip(block, adjusted):
                    row["p_adj_bh"] = float(q)
                    sig = bool(np.isfinite(q) and q < ALPHA)
                    row["significant"] = sig
                    row["reference_wins"] = bool(sig and row["rank_biserial"] > 0)
                    row["reference_loses"] = bool(sig and row["rank_biserial"] < 0)
                rows.extend(block)
    return pd.DataFrame(rows)


def order_by_damage(matrix: pd.DataFrame, metric: str) -> list[str]:
    """Arms sorted most- to least-damaging by median for this metric.

    Dice falls under attack while HD95 and ASD rise, so sorting on the raw
    median would reverse the reading between subplots. Sorting on the median
    times the metric's damage direction puts the strongest attack leftmost in
    every subplot.
    """
    sign = METRIC_DIRECTION[metric]
    medians = sign * matrix.median()
    return list(medians.sort_values(ascending=False).index)


def mean_over_epsilons(
    frames: dict[str, pd.DataFrame],
    arms: list[str],
    metric: str,
    epsilons: list[float],
    class_name: str,
) -> pd.DataFrame:
    """Case x arm matrix, each entry averaged over the attack budgets.

    Averaging happens per case before any aggregation across cases, so one
    value summarises how an arm treated that case over the whole epsilon
    sweep. Epsilon 0 is excluded: it is the same clean segmentation for every
    arm and would only pull all twelve columns toward a shared constant.
    """
    per_eps = [build_matrix(frames, metric, e, arms, class_name) for e in epsilons]
    stacked = pd.concat(per_eps)
    return stacked.groupby(stacked.index).mean()


def plot_boxes(
    frames: dict[str, pd.DataFrame],
    arms: list[str],
    metrics: list[str],
    epsilons: list[float],
    output_dir: str,
    stem: str,
    class_name: str,
    subject: str,
) -> str:
    """One figure, one subplot per metric, one box per attack.

    Each box holds one value per case: that case's metric averaged over the
    five attack budgets. Arms are sorted most- to least-damaging by median, so
    the ranking reads left to right.
    """
    fig, axes = plt.subplots(
        1, len(metrics), figsize=(6.2 * len(metrics), 7.4), squeeze=False
    )
    axes = list(axes[0])

    for ax, metric in zip(axes, metrics):
        matrix = mean_over_epsilons(frames, arms, metric, epsilons, class_name)
        ordered = order_by_damage(matrix, metric)
        clean_median = float(build_matrix(frames, metric, 0.0, arms, class_name)[arms[0]].median())

        # Outliers hidden: 1,396 points per box would bury the boxes.
        bp = ax.boxplot(
            [matrix[a].to_numpy() for a in ordered],
            showfliers=False,
            widths=0.66,
            patch_artist=True,
            medianprops=dict(color="black", linewidth=1.4),
        )
        for patch, arm in zip(bp["boxes"], ordered):
            patch.set_facecolor(BOX_COLOURS[arm])
            patch.set_edgecolor("black")
            patch.set_linewidth(0.7)

        ax.axhline(
            clean_median,
            color="black",
            linestyle=":",
            linewidth=1.2,
            zorder=0,
            label=f"clean median = {clean_median:.3g}",
        )
        ax.set_xticks(range(1, len(ordered) + 1))
        ax.set_xticklabels(
            [ARM_LABELS[a] for a in ordered], rotation=90, fontsize=10
        )
        ax.set_ylabel(METRIC_LABELS[metric], fontsize=13)
        ax.grid(axis="y", alpha=0.3, linewidth=0.6)
        ax.legend(loc="best", fontsize=9, framealpha=0.9)
        if metric in ("hd95", "asd"):
            # Plain log, not symlog: every distance in this campaign is
            # strictly positive (HD95 bottoms out at 1.0 mm, ASD at 0.12 mm),
            # and symlog's linear band around zero would spend most of the
            # panel on values that cannot occur.
            ax.set_yscale("log")

    n_cases = len(mean_over_epsilons(frames, arms, metrics[0], epsilons, class_name))
    budgets = ", ".join(f"{e:g}" for e in epsilons)
    fig.suptitle(
        f"Per-case metrics averaged over the attack budgets "
        f"($\\epsilon$ = {budgets})\n"
        f"{subject}, {n_cases:,} cases, 5-fold pooled; "
        "attacks ordered most- to least-damaging by median",
        fontsize=14,
    )
    fig.tight_layout(rect=(0, 0, 1, 0.93))

    base = os.path.join(output_dir, f"{stem}_boxplots")
    fig.savefig(f"{base}.png", dpi=200, bbox_inches="tight")
    fig.savefig(f"{base}.pdf", bbox_inches="tight")
    plt.close(fig)
    return f"{base}.png"


def audit_metric_validity(frames: dict[str, pd.DataFrame], arms: list[str]) -> None:
    """Report non-finite distance metrics.

    ``metrics/segmentation.py`` returns ``inf`` when either mask is empty, so
    any ``inf`` here would be an undefined distance masquerading as a very
    strong attack. Counting them is what licenses using HD95/ASD untrimmed.
    """
    bad = 0
    for arm in arms:
        f = frames[arm]
        for metric in ("hd95", "asd"):
            v = f[metric].astype(float)
            n = int((~np.isfinite(v)).sum())
            if n:
                bad += n
                print(f"  WARNING {arm}/{metric}: {n} non-finite values")
    if bad == 0:
        print("  all HD95/ASD values finite -- no empty predictions, no trimming")


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Paired statistics and box-plots for the twelve-arm whole-gland "
            "comparison. The summary CSV supplies the roster; the numbers come "
            "from the per-sample CSVs beside it."
        )
    )
    parser.add_argument(
        "--summary",
        default=os.path.join(
            "results",
            "blade_attack",
            "whole_gland",
            "native_per_eps",
            "main_comparison_wg_summary.csv",
        ),
        help="Summary CSV naming the arms, metrics and epsilons to analyse.",
    )
    parser.add_argument(
        "--campaign-dir",
        default=None,
        help="Directory holding the per-arm subdirectories (default: summary's parent).",
    )
    parser.add_argument(
        "--output-dir",
        default=None,
        help="Where to write tables and figures (default: summary's parent).",
    )
    parser.add_argument(
        "--stem",
        default=None,
        help="Output filename stem (default: derived from the summary filename).",
    )
    args = parser.parse_args()

    summary = pd.read_csv(args.summary)
    campaign_dir = args.campaign_dir or os.path.dirname(
        os.path.abspath(args.summary)
    )
    output_dir = args.output_dir or campaign_dir
    stem = args.stem or os.path.basename(args.summary).replace("_summary.csv", "")
    os.makedirs(output_dir, exist_ok=True)

    arms = [a for a in ARM_LABELS if a in set(summary["attack"])]
    metrics = [m for m in ("dice", "hd95", "asd") if m in set(summary["metric"])]
    epsilons = sorted(
        float(e) for e in summary["epsilon"].unique() if float(e) > 0
    )
    # A summary written before the class column existed describes the binary
    # model, whose per-sample CSVs label their single foreground class "WG".
    classes = (
        sorted(summary["class"].unique()) if "class" in summary.columns else ["WG"]
    )

    print(f"Roster from {args.summary}")
    print(f"  arms     : {', '.join(ARM_LABELS[a] for a in arms)}")
    print(f"  metrics  : {', '.join(metrics)}")
    print(f"  classes  : {', '.join(classes)}")
    print(f"  epsilons : {epsilons} (epsilon 0 is shared, excluded from tests)")

    frames = {arm: load_per_sample(campaign_dir, arm) for arm in arms}
    case_sets = {arm: set(f["case_id"]) for arm, f in frames.items()}
    reference_cases = case_sets[arms[0]]
    if not all(s == reference_cases for s in case_sets.values()):
        raise SystemExit("arms do not share an identical case set -- not paired")
    print(f"  paired on {len(reference_cases):,} cases, identical across arms")
    audit_metric_validity(frames, arms)

    for class_name in classes:
        # One self-contained analysis per foreground class. TZ+CZ and PZ fail
        # differently under the same attack, so pooling them would average away
        # the asymmetry a zonal evaluation exists to measure.
        suffix = "" if len(classes) == 1 else f"_{class_name.replace('+', '')}"
        class_stem = f"{stem}{suffix}"
        if len(classes) > 1:
            print(f"\n{'=' * 70}\n=== CLASS {class_name}\n{'=' * 70}")

        analyse_one_class(
            frames, arms, metrics, epsilons, class_name,
            output_dir, class_stem,
        )


def analyse_one_class(
    frames: dict[str, pd.DataFrame],
    arms: list[str],
    metrics: list[str],
    epsilons: list[float],
    class_name: str,
    output_dir: str,
    stem: str,
) -> None:
    """Every table and figure for one foreground class."""
    omnibus, posthoc = run_tests(frames, arms, metrics, epsilons, class_name)
    budget = budget_matched_tests(frames, arms, metrics, epsilons, class_name)

    paths = {
        "friedman": os.path.join(output_dir, f"{stem}_friedman.csv"),
        "posthoc": os.path.join(output_dir, f"{stem}_posthoc_wilcoxon.csv"),
        "budget": os.path.join(output_dir, f"{stem}_budget_matched.csv"),
    }
    omnibus.to_csv(paths["friedman"], index=False)
    posthoc.to_csv(paths["posthoc"], index=False)
    budget.to_csv(paths["budget"], index=False)

    print("\n=== Friedman omnibus (all arms, blocked on case) ===")
    show = omnibus.copy()
    show["p_value"] = show["p_value"].map(lambda v: f"{v:.3g}")
    show["friedman_chi2"] = show["friedman_chi2"].round(1)
    show["kendall_w"] = show["kendall_w"].round(3)
    print(show.to_string(index=False))

    print("\n=== Strongest arm per metric and epsilon ===")
    for metric in metrics:
        for eps in epsilons:
            block = posthoc[(posthoc.metric == metric) & (posthoc.epsilon == eps)]
            if block.empty:
                continue
            ref = block.reference.iloc[0]
            wins = int(block.reference_wins.sum())
            losses = block[block.reference_loses]
            tail = ""
            if not losses.empty:
                tail = "; LOSES to " + ", ".join(
                    ARM_LABELS[a] for a in losses.comparator
                )
            print(
                f"  {metric:5s} eps={eps:<5g} {ARM_LABELS[ref]:<15s} "
                f"median={block.median_reference.iloc[0]:9.4f}  "
                f"beats {wins}/{len(block)} (BH q<{ALPHA}){tail}"
            )

    print("\n=== Budget-matched contrasts ===")
    for group in budget.budget_group.unique():
        print(f"\n  -- {group} --")
        sub = budget[budget.budget_group == group]
        for metric in metrics:
            for eps in epsilons:
                block = sub[(sub.metric == metric) & (sub.epsilon == eps)]
                if block.empty:
                    continue
                ref = block.reference.iloc[0]
                ns = block[~block.significant]
                lost = block[block.reference_loses]
                parts = []
                if not ns.empty:
                    parts.append(
                        "n.s. vs " + ", ".join(ARM_LABELS[a] for a in ns.comparator)
                    )
                if not lost.empty:
                    parts.append(
                        "LOSES to " + ", ".join(ARM_LABELS[a] for a in lost.comparator)
                    )
                tail = "; ".join(parts) if parts else "beats all"
                print(f"     {metric:5s} eps={eps:<5g} {ARM_LABELS[ref]:<15s} {tail}")

    medians = median_summary(frames, arms, metrics, epsilons, class_name)
    avg_omnibus, avg_pairs = pairwise_median_tests(
        frames, arms, metrics, epsilons, class_name
    )
    median_path = os.path.join(output_dir, f"{stem}_budget_avg_medians.csv")
    pairs_path = os.path.join(output_dir, f"{stem}_budget_avg_pairwise.csv")
    medians.to_csv(median_path, index=False)
    avg_pairs.to_csv(pairs_path, index=False)
    paths["budget_avg_medians"] = median_path
    paths["budget_avg_pairwise"] = pairs_path

    print("\n=== Budget-averaged score: median per arm (95% CI, order statistics) ===")
    print("    per case, the metric is averaged over the attack budgets first;")
    print("    the median is then taken across cases -- the line in each box.")
    for metric in metrics:
        print(f"\n  -- {METRIC_LABELS[metric]} --")
        block = medians[medians.metric == metric]
        sign = METRIC_DIRECTION[metric]
        block = block.assign(key=sign * block["median"]).sort_values(
            "key", ascending=False
        )
        for _, r in block.iterrows():
            print(
                f"     {r.label:<15s} {r['median']:9.4f}  "
                f"[{r.ci95_low:8.4f}, {r.ci95_high:8.4f}]  "
                f"IQR {r.q1:8.4f}-{r.q3:8.4f}"
            )

    print("\n=== Friedman on the budget-averaged score ===")
    show = avg_omnibus.copy()
    show["p_value"] = show["p_value"].map(lambda v: f"{v:.3g}")
    show["friedman_chi2"] = show["friedman_chi2"].round(1)
    show["kendall_w"] = show["kendall_w"].round(3)
    print(show.to_string(index=False))

    n_pairs = len(arms) * (len(arms) - 1) // 2
    print(
        f"\n=== Median ranking, all {n_pairs} pairs per metric, BH-corrected ==="
    )
    print(
        f"    '>' left more damaging, '<' right more damaging "
        f"(inverts the median order), '~' not separated; all at q<{ALPHA}"
    )
    for metric in metrics:
        block = avg_pairs[avg_pairs.metric == metric]
        n_sig = int(block.significant.sum())
        chain, inversions = ranking_chain(medians, avg_pairs, metric)
        print(f"\n  {METRIC_LABELS[metric]}  ({n_sig}/{len(block)} pairs separated)")
        print("    " + chain)
        for note in inversions:
            print(f"    NOTE  {note}")

    print("\n=== Figures ===")
    print(
        "  "
        + plot_boxes(
            frames, arms, metrics, epsilons, output_dir, stem, class_name,
            subject="whole gland" if class_name == "WG" else class_name,
        )
    )
    for name, path in paths.items():
        print(f"  {path}")


if __name__ == "__main__":
    main()
