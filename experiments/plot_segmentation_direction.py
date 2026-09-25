"""Cohort summaries, paired comparisons and figures for the direction tables."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import matplotlib
import numpy as np
import pandas as pd

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.patches import Patch  # noqa: E402

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))
SRC_ROOT = REPO_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from experiments.evaluate_segmentation_direction_all_cases import (  # noqa: E402
    DEFAULT_OUTPUT_DIR,
    DEFAULT_RESULTS,
    PROFILE_CSV,
    SUMMARY_CSV,
    TRANSITION_CSV,
    head_target,
)
from mri_prostate_seg.experiments.perturbation_structure import (  # noqa: E402
    SIGNED_BAND_NAMES,
)
from mri_prostate_seg.experiments.segmentation_direction import (  # noqa: E402
    bootstrap_median_ci,
    describe,
    paired_median_difference,
)

# Controller ruling (2026-09-02): the brief points at
# ``all_cases_perturbation_structure/perturbation_structure_all_cases.csv``, but
# that live table predates the signed-distance bands and has no
# ``enrichment_band_*`` columns. ``perturbation_structure_signed/`` is the
# staged table that carries them, so it is the default here instead.
DEFAULT_STRUCTURE_CSV = (
    DEFAULT_RESULTS
    / "perturbation_structure_signed"
    / "perturbation_structure_all_cases.csv"
)
ATTACK_ORDER = ("FGSM-BCE", "PGD-BCE", "APGD-BCE")
PRIMARY_METRICS = (
    "damage_bias",
    "median_surface_motion_mm",
    "worsened_outward_surface_fraction",
    "worsened_inward_surface_fraction",
    "centroid_shift_norm_mm",
)
SECONDARY_METRICS = (
    "induced_fp_mm3",
    "induced_fn_mm3",
    "corrected_volume_mm3",
    "change_bias",
    "outward_surface_fraction",
    "inward_surface_fraction",
    "corrected_surface_fraction",
    "volume_change_pct",
)
SURFACE_METRICS = frozenset(
    m for m in PRIMARY_METRICS + SECONDARY_METRICS if "surface" in m
)
# Figure A row 3 (spec §32.1) needs the stacked outward/inward/stable surface
# fractions; ``stable_surface_fraction`` is neither a primary nor a secondary
# endpoint but is exactly as surface-metric-shaped as the other two, so it is
# folded into the exclusion set here rather than touching the two constants
# above (those are the brief's exact interface).
SURFACE_METRICS = SURFACE_METRICS | {"stable_surface_fraction"}
ALL_METRICS = (*PRIMARY_METRICS, *SECONDARY_METRICS, "stable_surface_fraction")
GROUP = ["dataset", "class_or_union", "attack", "epsilon_n"]

ATTACK_COLORS = {
    "FGSM-BCE": "#E69F00",
    "PGD-BCE": "#D55E00",
    "APGD-BCE": "#AA3377",
}
DATASET_LABELS = {"wg": "Whole gland (WG)", "zones": "Prostate zones"}
BAND_LABELS = {
    "inside_beyond_10mm": "< -10",
    "inside_5_10mm": "-10…-5",
    "inside_2_5mm": "-5…-2",
    "inside_0_2mm": "-2…0",
    "0_2mm": "0…2",
    "2_5mm": "2…5",
    "5_10mm": "5…10",
    "beyond_10mm": "> 10",
}
INTERIOR_BAND_COUNT = (
    4  # the four negative (inside-the-gland) bands, shaded in Figure B
)
REPORT_NAME = "SEGMENTATION_DIRECTION_REPORT.md"


def _strata(frame: pd.DataFrame) -> list[tuple[str, pd.DataFrame]]:
    return [("all", frame), ("successful", frame[frame["attack_success"] == 1])]


def summarize(summary: pd.DataFrame, *, metrics: tuple[str, ...]) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    for key, group in summary.groupby(GROUP, sort=True):
        for stratum, subset in _strata(group):
            for metric in metrics:
                if metric in SURFACE_METRICS:
                    usable = subset[subset["surface_metric_valid"] == 1.0]
                    reasons = subset.loc[
                        subset["surface_metric_valid"] != 1.0, "surface_failure_reason"
                    ]
                else:
                    usable = subset[subset["target_valid"] == 1]
                    reasons = pd.Series(
                        ["empty_gt"] * int((subset["target_valid"] != 1).sum())
                    )
                values = usable[metric].to_numpy(dtype=float)
                stats = describe(values)
                low, high = bootstrap_median_ci(values)
                counts = reasons.astype(str).value_counts().to_dict()
                rows.append(
                    {
                        **dict(zip(GROUP, key)),
                        "stratum": stratum,
                        "metric": metric,
                        **stats,
                        "ci_low": low,
                        "ci_high": high,
                        "n_excluded": int(sum(counts.values())),
                        "exclusion_reasons": ";".join(
                            f"{k}={v}" for k, v in sorted(counts.items())
                        ),
                    }
                )
    return pd.DataFrame(rows)


def _pivot(summary: pd.DataFrame, metric: str) -> pd.DataFrame:
    return summary.pivot_table(
        index=["dataset", "class_or_union", "case_id"],
        columns=["attack", "epsilon_n"],
        values=metric,
        aggfunc="first",
    )


def paired(summary: pd.DataFrame, *, metrics: tuple[str, ...]) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    for metric in metrics:
        table = _pivot(summary, metric)
        for (dataset, target), block in table.groupby(
            level=["dataset", "class_or_union"]
        ):
            for eps in sorted({e for _, e in block.columns}):
                for a, b in (("APGD-BCE", "PGD-BCE"), ("PGD-BCE", "FGSM-BCE")):
                    if (a, eps) in block.columns and (b, eps) in block.columns:
                        out = paired_median_difference(
                            block[(a, eps)].to_numpy(), block[(b, eps)].to_numpy()
                        )
                        rows.append(
                            {
                                "dataset": dataset,
                                "class_or_union": target,
                                "metric": metric,
                                "comparison": f"{a} - {b}",
                                "attack": "",
                                "epsilon_n": eps,
                                **out,
                            }
                        )
            for attack in ATTACK_ORDER:
                if (attack, 32) in block.columns and (attack, 16) in block.columns:
                    out = paired_median_difference(
                        block[(attack, 32)].to_numpy(), block[(attack, 16)].to_numpy()
                    )
                    rows.append(
                        {
                            "dataset": dataset,
                            "class_or_union": target,
                            "metric": metric,
                            "comparison": "eps32 - eps16",
                            "attack": attack,
                            "epsilon_n": 32,
                            **out,
                        }
                    )
    return pd.DataFrame(rows)


def profile_summary(profile: pd.DataFrame) -> pd.DataFrame:
    metrics = ["induced_fp_rate", "induced_fn_rate", "net_fg_change_per_100_voxels"]
    grouped = profile.groupby([*GROUP, "band"], sort=False)[metrics]
    out = grouped.median().add_suffix("_median")
    out = out.join(grouped.quantile(0.25).add_suffix("_q25")).join(
        grouped.quantile(0.75).add_suffix("_q75")
    )
    out["n_cases"] = grouped.size()
    return out.reset_index()


def join_energy(
    profile_summary_frame: pd.DataFrame, structure: pd.DataFrame
) -> pd.DataFrame:
    long = structure.melt(
        id_vars=["dataset", "case_id", "epsilon_n", "condition"],
        value_vars=[f"enrichment_band_{b}" for b in SIGNED_BAND_NAMES],
        var_name="band",
        value_name="enrichment",
    )
    long["band"] = long["band"].str.replace("enrichment_band_", "", regex=False)
    energy = (
        long.groupby(["dataset", "condition", "epsilon_n", "band"])["enrichment"]
        .median()
        .rename("energy_enrichment_median")
        .reset_index()
        .rename(columns={"condition": "attack"})
    )
    joined = profile_summary_frame.merge(
        energy, on=["dataset", "attack", "epsilon_n", "band"], how="left"
    )
    # The structure table's bands are distances to the outer gland; only the
    # head target's profile shares that reference, so the zones' per-class rows
    # (whose bands are to their own GT) get no energy value.
    is_head = joined["class_or_union"] == joined["dataset"].map(head_target)
    joined.loc[~is_head, "energy_enrichment_median"] = np.nan
    return joined


def transition_summary(transitions: pd.DataFrame) -> pd.DataFrame:
    keys = [
        "dataset",
        "attack",
        "epsilon_n",
        "transition_type",
        "source_class",
        "target_class",
    ]
    grouped = transitions.groupby(keys, sort=True)["row_normalized_rate"]
    out = grouped.median().rename("row_normalized_rate_median").to_frame()
    out["row_normalized_rate_q25"] = grouped.quantile(0.25)
    out["row_normalized_rate_q75"] = grouped.quantile(0.75)
    out["n_cases"] = grouped.size()
    return out.reset_index()


def _figure_a(stats: pd.DataFrame, output: Path) -> None:
    """Damage direction across epsilon budgets (spec §32.1–§32.3).

    Row 2's twin axis and row 3's stacked, log-spaced bars are controller-
    ruled fixes (2026-09-02 review) to the brief's originally-given code:
    the brief's prose (spec §32.2–§32.3) is the authority, not the given
    snippet, which was missing the surface-motion twin axis and mispositioned
    the row-3 bars (``sharex="col"`` applies row 0's log2 x-scale to every row
    in the column, so the bars' own ``np.log2(eps) + offset`` was transformed
    a second time, and ``offset`` never reached the x expression at all).
    """
    fraction_order = (
        ("outward_surface_fraction", "outward", ""),
        ("inward_surface_fraction", "inward", "//"),
        ("stable_surface_fraction", "stable", ".."),
    )
    fig, axes = plt.subplots(3, 2, figsize=(11, 10), sharex="col")
    for column, dataset in enumerate(("wg", "zones")):
        target = head_target(dataset)
        block = stats[
            (stats.dataset == dataset)
            & (stats.class_or_union == target)
            & (stats.stratum == "all")
        ]
        motion_twin = axes[1, column].twinx()
        for attack in ATTACK_ORDER:
            color = ATTACK_COLORS[attack]
            index = ATTACK_ORDER.index(attack)
            sub = block[block.attack == attack].sort_values("epsilon_n")
            fp = sub[sub.metric == "induced_fp_mm3"]
            fn = sub[sub.metric == "induced_fn_mm3"]
            axes[0, column].plot(
                fp.epsilon_n,
                fp["median"],
                "-o",
                color=color,
                label=f"{attack} induced FP",
            )
            axes[0, column].plot(
                fn.epsilon_n,
                fn["median"],
                "--s",
                color=color,
                label=f"{attack} induced FN",
            )
            bias = sub[sub.metric == "damage_bias"]
            axes[1, column].plot(
                bias.epsilon_n, bias["median"], "-o", color=color, label=attack
            )
            axes[1, column].fill_between(
                bias.epsilon_n, bias.ci_low, bias.ci_high, color=color, alpha=0.2
            )
            motion = sub[sub.metric == "median_surface_motion_mm"].sort_values(
                "epsilon_n"
            )
            motion_twin.plot(motion.epsilon_n, motion["median"], "--", color=color)

            fractions = {
                metric: sub[sub.metric == metric].set_index("epsilon_n")["median"]
                for metric, _label, _hatch in fraction_order
            }
            eps_values = fractions["outward_surface_fraction"].index.to_numpy(
                dtype=float
            )
            x = eps_values * 2.0 ** (0.3 * (index - 1))
            width = 0.25 * eps_values
            bottom = np.zeros_like(eps_values)
            for metric, _label, hatch in fraction_order:
                height = fractions[metric].to_numpy(dtype=float)
                axes[2, column].bar(
                    x,
                    height,
                    width=width,
                    bottom=bottom,
                    color=color,
                    hatch=hatch,
                    alpha=0.7,
                )
                bottom = bottom + height
        axes[1, column].axhline(0.0, color="k", lw=0.8)
        axes[0, column].set_title(f"{dataset}: {target}")
        axes[0, column].set_ylabel("median volume (mm³)")
        axes[1, column].set_ylabel("damage bias (+ over, − under)")
        motion_twin.set_ylabel("median surface motion (mm, + outward)")
        axes[2, column].set_ylabel("surface fraction")
        axes[2, column].set_xlabel("ε (n/255)")
        axes[0, column].set_xscale("log", base=2)
        axes[0, column].set_xticks([2, 4, 8, 16, 32])
        axes[0, column].set_xticklabels([f"{n}/255" for n in (2, 4, 8, 16, 32)])
        if column == 0:
            motion_twin.plot(
                [], [], "--", color="0.5", label="surface motion (right axis)"
            )
            motion_twin.legend(fontsize=7, loc="lower right")
    axes[0, 0].legend(fontsize=7)
    axes[1, 0].legend(fontsize=7)
    fraction_handles = [
        Patch(facecolor="0.6", hatch=hatch, alpha=0.7, label=label)
        for _metric, label, hatch in fraction_order
    ] + [
        Patch(facecolor=ATTACK_COLORS[attack], alpha=0.7, label=attack)
        for attack in ATTACK_ORDER
    ]
    axes[2, 0].legend(handles=fraction_handles, fontsize=6, ncol=3)
    fig.tight_layout()
    fig.savefig(output, dpi=200)
    plt.close(fig)


def _figure_b(joined: pd.DataFrame, output: Path, *, reference_epsilon: int) -> None:
    """Signed-distance profile of energy enrichment, induced FP and FN rates.

    Spec §62 minus the logit-margin panel (Stage 3).
    """
    band_order = list(SIGNED_BAND_NAMES)
    positions = {band: i for i, band in enumerate(band_order)}
    fig, axes = plt.subplots(3, 2, figsize=(11, 10), sharex="col")
    for column, dataset in enumerate(("wg", "zones")):
        target = head_target(dataset)
        block = joined[
            (joined.dataset == dataset)
            & (joined.class_or_union == target)
            & (joined.epsilon_n == reference_epsilon)
        ]
        for row in range(3):
            axes[row, column].axvspan(
                -0.5, INTERIOR_BAND_COUNT - 0.5, color="0.9", zorder=0
            )
        for attack in ATTACK_ORDER:
            color = ATTACK_COLORS[attack]
            sub = block[block.attack == attack].copy()
            sub["x"] = sub["band"].map(positions)
            sub = sub.sort_values("x")
            axes[0, column].plot(
                sub.x, sub.energy_enrichment_median, "-o", color=color, label=attack
            )
            axes[1, column].fill_between(
                sub.x,
                sub.induced_fp_rate_q25,
                sub.induced_fp_rate_q75,
                color=color,
                alpha=0.2,
            )
            axes[1, column].plot(sub.x, sub.induced_fp_rate_median, "-o", color=color)
            axes[2, column].fill_between(
                sub.x,
                sub.induced_fn_rate_q25,
                sub.induced_fn_rate_q75,
                color=color,
                alpha=0.2,
            )
            axes[2, column].plot(sub.x, sub.induced_fn_rate_median, "-o", color=color)
        axes[0, column].axhline(1.0, color="k", lw=0.8)
        axes[0, column].set_title(f"{dataset}: {target}")
        axes[0, column].set_ylabel("energy enrichment")
        axes[1, column].set_ylabel("induced FP rate")
        axes[2, column].set_ylabel("induced FN rate")
        axes[2, column].set_xlabel("signed distance to boundary (mm)")
        axes[2, column].set_xticks(range(len(band_order)))
        axes[2, column].set_xticklabels(
            [BAND_LABELS[b] for b in band_order], rotation=45, ha="right"
        )
    axes[0, 0].legend(fontsize=7)
    fig.tight_layout()
    fig.savefig(output, dpi=200)
    plt.close(fig)


def _new_harm_matrix(block: pd.DataFrame, classes: tuple[str, ...]) -> np.ndarray:
    matrix = np.zeros((len(classes), len(classes)))
    for i, source in enumerate(classes):
        for j, target in enumerate(classes):
            hit = block[(block.source_class == source) & (block.target_class == target)]
            if not hit.empty:
                matrix[i, j] = float(hit["row_normalized_rate_median"].iloc[0])
    return matrix


def _figure_c(zone_transitions: pd.DataFrame, output: Path) -> None:
    classes = ("background", "TZ+CZ", "PZ")
    harm = zone_transitions[zone_transitions.transition_type == "new_harm"]
    epsilons = sorted(harm.epsilon_n.unique())
    if not epsilons:
        # Nothing to plot (e.g. a WG-only smoke run); still leave a placeholder.
        fig, ax = plt.subplots(figsize=(4, 3))
        ax.text(0.5, 0.5, "no zone transitions available", ha="center", va="center")
        ax.axis("off")
        fig.savefig(output, dpi=200)
        plt.close(fig)
        return
    vmax = float(harm["row_normalized_rate_median"].max()) if not harm.empty else 1.0
    fig, axes = plt.subplots(
        len(ATTACK_ORDER),
        len(epsilons),
        figsize=(3.0 * len(epsilons), 3.2 * len(ATTACK_ORDER)),
    )
    axes = np.atleast_2d(axes)
    for row, attack in enumerate(ATTACK_ORDER):
        for col, eps in enumerate(epsilons):
            ax = axes[row, col]
            block = harm[(harm.attack == attack) & (harm.epsilon_n == eps)]
            matrix = _new_harm_matrix(block, classes)
            im = ax.imshow(matrix, vmin=0.0, vmax=vmax, cmap="Reds")
            for i in range(len(classes)):
                for j in range(len(classes)):
                    ax.text(
                        j,
                        i,
                        f"{matrix[i, j]:.3f}",
                        ha="center",
                        va="center",
                        fontsize=7,
                    )
            ax.set_xticks(range(len(classes)))
            ax.set_yticks(range(len(classes)))
            if row == len(ATTACK_ORDER) - 1:
                ax.set_xticklabels(classes, rotation=45, ha="right", fontsize=7)
            else:
                ax.set_xticklabels([])
            if col == 0:
                ax.set_yticklabels(classes, fontsize=7)
                ax.set_ylabel(attack, fontsize=8)
            else:
                ax.set_yticklabels([])
            if row == 0:
                ax.set_title(f"ε={eps}/255", fontsize=8)
    fig.colorbar(im, ax=axes, shrink=0.6, label="row-normalized new-harm rate (median)")
    fig.suptitle(
        "New-harm transition matrices (rows: source class, columns: target class)"
    )
    fig.savefig(output, dpi=200)
    plt.close(fig)


def _markdown_table(frame: pd.DataFrame) -> str:
    if frame.empty:
        return "_No rows available._"
    display = frame.copy()
    for column in display.select_dtypes(include=[np.number]).columns:
        display[column] = display[column].map(
            lambda value: "—" if not np.isfinite(value) else f"{value:.4g}"
        )
    return display.to_markdown(index=False)


def _phenotype_sentence(
    direction_summary: pd.DataFrame, dataset: str, *, reference_epsilon: int
) -> str:
    target = head_target(dataset)
    rows = direction_summary[
        (direction_summary.dataset == dataset)
        & (direction_summary.class_or_union == target)
        & (direction_summary.attack == "APGD-BCE")
        & (direction_summary.epsilon_n == reference_epsilon)
        & (direction_summary.stratum == "all")
    ]
    bias = rows[rows.metric == "damage_bias"]
    motion = rows[rows.metric == "median_surface_motion_mm"]
    if bias.empty or motion.empty or bias["n"].iloc[0] == 0 or motion["n"].iloc[0] == 0:
        return (
            f"- **{DATASET_LABELS.get(dataset, dataset)}**: no APGD-BCE rows at "
            f"ε={reference_epsilon}/255 to classify (see Table 2)."
        )
    bias_median = float(bias["median"].iloc[0])
    bias_ci_low = float(bias["ci_low"].iloc[0])
    bias_ci_high = float(bias["ci_high"].iloc[0])
    motion_median = float(motion["median"].iloc[0])
    motion_ci_low = float(motion["ci_low"].iloc[0])
    motion_ci_high = float(motion["ci_high"].iloc[0])
    if bias_median > 0 and bias_ci_low > 0 and motion_median > 0 and motion_ci_low > 0:
        phenotype = "predominant expansion (spec §45.1)"
    elif (
        bias_median < 0
        and bias_ci_high < 0
        and motion_median < 0
        and motion_ci_high < 0
    ):
        phenotype = "predominant contraction (spec §45.2)"
    else:
        phenotype = (
            "mixed deformation (spec §45.4); the sign of the median damage bias "
            "and surface motion do not both clear zero, or disagree"
        )
    return (
        f"- **{DATASET_LABELS.get(dataset, dataset)}**: APGD-BCE at "
        f"ε={reference_epsilon}/255 matches {phenotype}, per the damage-bias and "
        'median-surface-motion rows of Table 2 (the "all" stratum, head target).'
    )


def write_report(
    *,
    direction_summary: pd.DataFrame,
    direction_paired: pd.DataFrame,
    direction_profile_summary: pd.DataFrame,
    zone_transition_summary: pd.DataFrame,
    output_dir: Path,
    reference_epsilon: int,
) -> Path:
    report_endpoints = {8, 16, 32}
    lines: list[str] = [
        "# Segmentation Damage Direction Report (Stage 1)",
        "",
        "## 1. Scope and conventions",
        "",
        "- Predictions are raw argmax masks (`_predict_batch`); no post-processing is applied "
        "(spec §4.6).",
        "- Every count, denominator, and band uses `valid = gt >= 0`; `IGNORE_LABEL = -1` voxels "
        "contribute to nothing.",
        "- The displacement tolerance τ_d defaults to 0.5 mm (the in-plane voxel size); the "
        "through-plane spacing is 3.0 mm, so an axial displacement below 3 mm can still be within "
        "one through-plane voxel and is reported, not concealed.",
        "- Bootstrap confidence intervals are case-level (one case per patient assumed; no "
        "patient-to-case mapping exists in the raw dataset.json files) and use 10,000 resamples, "
        "seed 42, percentile CI of the median (spec §28).",
        "- Attack success (spec §31): `delta_dice <= -0.01` on the head class, evaluated in the "
        "same pass as the perturbation (Dice is never joined from another table).",
        "- Zone bands measure distance to the outer gland boundary; the PZ/TZ interface is not "
        "resolved here (spec §21, Stage 2).",
        "- The band profile uses the plain EDT field (Figure B, matching the structure table's "
        "enrichment bands); the surface metrics (Figure A, Tables 2-3) use the half-voxel-corrected "
        "`iso_signed_distance_mm` field instead, because a plain EDT overstates every surface "
        "offset by half a voxel along the nearest-boundary direction.",
        "- Tables 2-4 restrict `class_or_union` to each dataset's head target (WG for `wg`, the "
        "union for `zones`); the per-zone (`TZ+CZ`, `PZ`) rows are in the CSVs but not tabulated "
        "here, except where Section 5 compares them directly.",
        "",
        "## 2. Primary endpoints (spec §26.1)",
        "",
    ]

    primary_all = direction_summary[
        (direction_summary.stratum == "all")
        & (direction_summary.epsilon_n.isin(report_endpoints))
        & (direction_summary.metric.isin(PRIMARY_METRICS))
    ]
    primary_successful = direction_summary[
        (direction_summary.stratum == "successful")
        & (direction_summary.epsilon_n.isin(report_endpoints))
        & (direction_summary.metric.isin(PRIMARY_METRICS))
    ]
    endpoint_columns = [
        "dataset",
        "metric",
        "attack",
        "epsilon_n",
        "n",
        "median",
        "ci_low",
        "ci_high",
        "q25",
        "q75",
        "p05",
        "p95",
        "n_excluded",
    ]
    for dataset in ("wg", "zones"):
        target = head_target(dataset)
        lines.append(
            f"### {DATASET_LABELS.get(dataset, dataset)} ({target}), all cases"
        )
        lines.append("")
        table = primary_all[
            (primary_all.dataset == dataset) & (primary_all.class_or_union == target)
        ][endpoint_columns].sort_values(["metric", "attack", "epsilon_n"])
        lines.append(_markdown_table(table))
        lines.append("")
        lines.append(
            f"### {DATASET_LABELS.get(dataset, dataset)} ({target}), successful attacks only"
        )
        lines.append("")
        table = primary_successful[
            (primary_successful.dataset == dataset)
            & (primary_successful.class_or_union == target)
        ][endpoint_columns].sort_values(["metric", "attack", "epsilon_n"])
        lines.append(_markdown_table(table))
        lines.append("")

    lines += [
        "## 3. Paired comparisons",
        "",
        "This table reports median differences with percentile-bootstrap "
        "confidence intervals and no p-values, so no multiplicity correction "
        "(e.g. Holm, FDR) is applied.",
        "",
    ]
    paired_columns = [
        "dataset",
        "metric",
        "comparison",
        "attack",
        "epsilon_n",
        "n",
        "median_diff",
        "ci_low",
        "ci_high",
    ]
    for dataset in ("wg", "zones"):
        target = head_target(dataset)
        lines.append(f"### {DATASET_LABELS.get(dataset, dataset)} ({target})")
        lines.append("")
        table = direction_paired[
            (direction_paired.dataset == dataset)
            & (direction_paired.class_or_union == target)
        ][paired_columns].sort_values(["metric", "comparison", "epsilon_n"])
        lines.append(_markdown_table(table))
        lines.append("")

    lines += [
        f"## 4. Signed-distance profile (ε = {reference_epsilon}/255)",
        "",
    ]
    profile_columns = [
        "dataset",
        "band",
        "energy_enrichment_median",
        "induced_fp_rate_median",
        "induced_fn_rate_median",
        "net_fg_change_per_100_voxels_median",
        "attack",
        "n_cases",
    ]
    band_rank = {b: i for i, b in enumerate(SIGNED_BAND_NAMES)}
    for dataset in ("wg", "zones"):
        target = head_target(dataset)
        lines.append(f"### {DATASET_LABELS.get(dataset, dataset)} ({target})")
        lines.append("")
        table = direction_profile_summary[
            (direction_profile_summary.dataset == dataset)
            & (direction_profile_summary.class_or_union == target)
            & (direction_profile_summary.epsilon_n == reference_epsilon)
        ][profile_columns].copy()
        table["_rank"] = table["band"].map(band_rank)
        table = table.sort_values(["attack", "_rank"]).drop(columns="_rank")
        lines.append(_markdown_table(table))
        lines.append("")

    lines += ["## 5. Zone transitions", ""]
    harm = zone_transition_summary[
        (zone_transition_summary.transition_type == "new_harm")
        & (zone_transition_summary.epsilon_n.isin(report_endpoints))
    ][
        [
            "attack",
            "epsilon_n",
            "source_class",
            "target_class",
            "row_normalized_rate_median",
            "row_normalized_rate_q25",
            "row_normalized_rate_q75",
            "n_cases",
        ]
    ].sort_values(["attack", "epsilon_n", "source_class", "target_class"])
    lines.append("### `new_harm` medians")
    lines.append("")
    lines.append(_markdown_table(harm))
    lines.append("")
    union_vs_zone = direction_summary[
        (direction_summary.dataset == "zones")
        & (direction_summary.stratum == "all")
        & (direction_summary.metric == "damage_bias")
        & (direction_summary.epsilon_n.isin(report_endpoints))
    ][
        ["class_or_union", "attack", "epsilon_n", "n", "median", "ci_low", "ci_high"]
    ].sort_values(["class_or_union", "attack", "epsilon_n"])
    lines.append("### Union vs. per-zone damage bias")
    lines.append("")
    lines.append(_markdown_table(union_vs_zone))
    lines.append("")

    lines += ["## 6. Exclusions", ""]
    # class_or_union stays in the table: for ``zones`` the surface validity is
    # computed separately per target (union, TZ+CZ, PZ), so without this
    # column three genuinely different rows would print as duplicates.
    exclusions = direction_summary[
        (direction_summary.stratum == "all")
        & (direction_summary.metric == "median_surface_motion_mm")
    ][
        [
            "dataset",
            "class_or_union",
            "attack",
            "epsilon_n",
            "n_excluded",
            "exclusion_reasons",
        ]
    ].sort_values(["dataset", "class_or_union", "attack", "epsilon_n"])
    lines.append(_markdown_table(exclusions))
    lines.append("")

    lines += [
        "## 7. Reading guide",
        "",
        "Case-level damage phenotypes (spec §45); Stage 1 has no RAS centroid components or "
        "topology classification, so only the phenotypes that follow from damage bias, surface "
        "motion, and surface fractions can be read off the tables above:",
        "",
        "- **Predominant expansion (§45.1)**: positive damage bias, positive median surface "
        "displacement, outward surface fraction larger than inward, positive induced-FP volume, "
        "positive volume change.",
        "- **Predominant contraction (§45.2)**: the mirror image — negative damage bias, "
        "negative median displacement, inward fraction larger than outward, positive induced-FN "
        "volume, negative volume change.",
        "- **Translation (§45.3)**: change bias near zero, volume change near zero, nonzero "
        "centroid shift, opposing inward/outward motion on opposite sides — not distinguishable "
        "from mixed deformation here without the RAS centroid components (Stage 2).",
        "- **Mixed deformation (§45.4)**: substantial outward and inward fractions together, "
        "median displacement near zero, large displacement tails.",
        "- **Fragmentation or topology damage (§45.5)**: changed component count, remote "
        "islands or holes — out of scope for Stage 1 (spec §58).",
        "- **Internal zone substitution (§45.6)**: substantial PZ→TZ+CZ or TZ+CZ→PZ "
        "new-harm transitions with limited union motion (Section 5).",
        "",
        "Per-dataset read, derived only from the sign of the median and whether the bootstrap CI "
        'crosses zero (Table 2, "all" stratum, APGD-BCE at the reference ε):',
        "",
        _phenotype_sentence(
            direction_summary, "wg", reference_epsilon=reference_epsilon
        ),
        _phenotype_sentence(
            direction_summary, "zones", reference_epsilon=reference_epsilon
        ),
        "",
    ]

    report_path = output_dir / REPORT_NAME
    report_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return report_path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Summarise, pair and plot the segmentation damage direction tables"
    )
    parser.add_argument("--input-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--structure-csv", type=Path, default=DEFAULT_STRUCTURE_CSV)
    parser.add_argument("--reference-epsilon", type=int, default=16)
    parser.add_argument("--output-dir", type=Path, default=None)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.reference_epsilon <= 0:
        raise ValueError("--reference-epsilon must be positive")
    input_dir = args.input_dir.resolve()
    output_dir = (
        args.output_dir if args.output_dir is not None else input_dir
    ).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    summary_path = input_dir / SUMMARY_CSV
    profile_path = input_dir / PROFILE_CSV
    transition_path = input_dir / TRANSITION_CSV
    structure_path = args.structure_csv.resolve()
    for path in (summary_path, profile_path, transition_path, structure_path):
        if not path.is_file():
            raise FileNotFoundError(f"missing input table: {path}")

    summary = pd.read_csv(summary_path)
    profile = pd.read_csv(profile_path)
    transitions = pd.read_csv(transition_path)
    structure = pd.read_csv(structure_path)

    direction_summary = summarize(summary, metrics=ALL_METRICS)
    direction_paired = paired(summary, metrics=PRIMARY_METRICS)
    direction_profile_summary = join_energy(profile_summary(profile), structure)
    zone_transition_summary = transition_summary(transitions)

    summary_csv_path = output_dir / "direction_summary.csv"
    paired_csv_path = output_dir / "direction_paired.csv"
    profile_summary_csv_path = output_dir / "direction_profile_summary.csv"
    transition_summary_csv_path = output_dir / "zone_transition_summary.csv"
    direction_summary.to_csv(summary_csv_path, index=False)
    direction_paired.to_csv(paired_csv_path, index=False)
    direction_profile_summary.to_csv(profile_summary_csv_path, index=False)
    zone_transition_summary.to_csv(transition_summary_csv_path, index=False)

    figure_a_path = output_dir / "01_direction_across_budgets.png"
    figure_b_path = output_dir / "02_energy_and_damage_by_signed_distance.png"
    figure_c_path = output_dir / "03_zone_transition_heatmaps.png"
    _figure_a(direction_summary, figure_a_path)
    _figure_b(
        direction_profile_summary,
        figure_b_path,
        reference_epsilon=args.reference_epsilon,
    )
    _figure_c(zone_transition_summary, figure_c_path)

    report_path = write_report(
        direction_summary=direction_summary,
        direction_paired=direction_paired,
        direction_profile_summary=direction_profile_summary,
        zone_transition_summary=zone_transition_summary,
        output_dir=output_dir,
        reference_epsilon=args.reference_epsilon,
    )

    for path in (
        summary_csv_path,
        paired_csv_path,
        profile_summary_csv_path,
        transition_summary_csv_path,
        figure_a_path,
        figure_b_path,
        figure_c_path,
        report_path,
    ):
        print(path)


if __name__ == "__main__":
    main()
