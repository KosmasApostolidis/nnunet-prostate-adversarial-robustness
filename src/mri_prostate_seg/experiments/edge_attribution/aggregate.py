"""Cohort aggregation for the edge-attribution tables.

Pure pandas: no I/O, no plotting, no torch.  The unit of inference is the case,
so every public function here reduces repeated measures to one value per case
before anything is pooled across cases.
"""

from __future__ import annotations

from collections.abc import Sequence

import numpy as np
import pandas as pd

from ..segmentation_direction.statistics import bootstrap_median_ci

# Columns whose absence means the table predates the selection-on-outcome fix in
# the joint matched control.  Such a table's ``control_permutation_p`` is skewed
# small by an amount nobody measured, so it must never reach a report.
REQUIRED_SUMMARY_COLUMNS: tuple[str, ...] = (
    "observed_pool_size",
    "control_pool_size",
    "n_trials_skipped_undersized",
    "observed_matched_recovery",
    # The symmetric-M discriminator.  A table can carry all four columns above
    # and still predate the fix that made the observed and control arms select
    # from the same pool: before it, the observed arm took the top q of the
    # whole screened set while the control arm was restricted to the patches
    # whose matched draw happened to succeed -- a size-truncated set, because
    # draw success falls with patch volume.  ``control_drawable_fraction`` is
    # written only by the fixed driver, so its absence is exactly the condition
    # under which control_permutation_p was asymmetric.
    "control_drawable_fraction",
)

BOOTSTRAP_RESAMPLES = 2000
BOOTSTRAP_SEED = 20260721


class SchemaError(ValueError):
    """A table is missing columns that a current driver always writes."""


def check_summary_schema(frame: pd.DataFrame, source: str) -> None:
    """Raise if ``frame`` predates the current driver.

    Named rather than boolean so the message can carry the offending path: a
    stale shard in a merged directory is otherwise very hard to find.
    """

    missing = [c for c in REQUIRED_SUMMARY_COLUMNS if c not in frame.columns]
    if missing:
        raise SchemaError(
            f"{source} is missing {missing}; it was written before the "
            "selection-on-outcome fix and its control_permutation_p is biased. "
            "Re-run the driver for these shards rather than aggregating them."
        )


def case_level(
    frame: pd.DataFrame, value: str, *, group: Sequence[str]
) -> pd.DataFrame:
    """One row per case: the median of ``value`` within each case.

    Patches, epsilons and attacks inside a case are repeated measures, so pooling
    their rows directly would weight a case with 300 patches 3x a case with 100.
    """

    cols = [*group, "case_id", value]
    kept = frame.loc[np.isfinite(frame[value]), cols]
    if kept.empty:
        return kept.copy()
    return (
        kept.groupby([*group, "case_id"], observed=True)[value].median().reset_index()
    )


def median_ci(
    values: Sequence[float],
    *,
    resamples: int = BOOTSTRAP_RESAMPLES,
    seed: int = BOOTSTRAP_SEED,
) -> tuple[float, float, float]:
    """Case-clustered bootstrap median with a 95% CI, as ``(low, median, high)``."""

    finite = np.asarray([float(v) for v in values if np.isfinite(v)], dtype=np.float64)
    if finite.size == 0:
        return (float("nan"), float("nan"), float("nan"))
    if finite.size == 1:
        only = float(finite[0])
        return (only, only, only)
    low, high = bootstrap_median_ci(finite, resamples=resamples, seed=seed)
    return (float(low), float(np.median(finite)), float(high))


CONDITION = ("dataset", "epsilon_n", "attack", "atlas_type")

# The concentration story of the experiment: how unevenly perturbation energy and
# causal utility are distributed over patches, and how much of the ROI the useful
# subset occupies.
CONCENTRATION_METRICS: tuple[str, ...] = (
    "energy_gini",
    "n_eff_norm",
    "entropy_concentration",
    "edge_energy_fraction_roi",
    "energy_top_10pct",
    "utility_gini",
    "damage_removed_top_10pct",
    "damage_kept_top_10pct",
    "k50_remove",
    "k80_remove",
    "useful_subset_volume_fraction",
)

CURVE_VALUES: tuple[str, ...] = (
    "damage_removed_fraction",
    "damage_kept_fraction",
    "volume_fraction",
    "total_energy_fraction",
)


def cohort_concentration(summary: pd.DataFrame) -> pd.DataFrame:
    """Case-clustered median and CI for each concentration metric."""

    check_summary_schema(summary, "case_summary")
    out: list[dict[str, object]] = []
    for metric in CONCENTRATION_METRICS:
        if metric not in summary.columns:
            continue
        per_case = case_level(summary, metric, group=list(CONDITION))
        if per_case.empty:
            continue
        for keys, group in per_case.groupby(list(CONDITION), observed=True):
            low, med, high = median_ci(group[metric].tolist())
            out.append(
                {
                    **dict(zip(CONDITION, keys)),
                    "metric": metric,
                    "n_cases": int(group["case_id"].nunique()),
                    "median": med,
                    "ci_low": low,
                    "ci_high": high,
                }
            )
    return pd.DataFrame(out)


def cohort_curves(curves: pd.DataFrame) -> pd.DataFrame:
    """Pool the per-case subset curves at each k, within ranking type."""

    evaluated = curves[curves["evaluated"].astype(bool)]
    group_cols = [*CONDITION, "ranking_type", "k"]
    out: list[dict[str, object]] = []
    for keys, group in evaluated.groupby(group_cols, observed=True):
        row: dict[str, object] = dict(zip(group_cols, keys))
        row["n_cases"] = int(group["case_id"].nunique())
        for value in CURVE_VALUES:
            if value not in group.columns:
                continue
            per_case = (
                group.loc[np.isfinite(group[value])]
                .groupby("case_id", observed=True)[value]
                .median()
            )
            low, med, high = median_ci(per_case.tolist())
            row[f"{value}_median"] = med
            row[f"{value}_ci_low"] = low
            row[f"{value}_ci_high"] = high
        out.append(row)
    return pd.DataFrame(out).sort_values(group_cols).reset_index(drop=True)


CATEGORY_VALUE = "fractional_recovery_dice"

RANK_PAIRS: tuple[str, ...] = (
    "strength_energy",
    "energy_utility",
    "strength_utility",
)


def anatomical_category_summary(patches: pd.DataFrame) -> pd.DataFrame:
    """Per-category causal utility, case-clustered.

    Only screened patches carry a measured necessity; unscreened rows have no
    intervention behind them and would dilute the median with structural zeros.
    """

    screened = patches[patches["screened"].astype(bool)]
    group_cols = [*CONDITION, "anatomical_category"]
    out: list[dict[str, object]] = []
    for keys, group in screened.groupby(group_cols, observed=True):
        finite = group.loc[np.isfinite(group[CATEGORY_VALUE])]
        if finite.empty:
            continue
        per_case = finite.groupby("case_id", observed=True)[CATEGORY_VALUE].median()
        low, med, high = median_ci(per_case.tolist())
        out.append(
            {
                **dict(zip(group_cols, keys)),
                "n_cases": int(finite["case_id"].nunique()),
                "n_patches": int(len(finite)),
                "median": med,
                "ci_low": low,
                "ci_high": high,
            }
        )
    return pd.DataFrame(out)


def cohort_controls(summary: pd.DataFrame, controls: pd.DataFrame) -> pd.DataFrame:
    """Observed vs matched-random joint recovery, with the pool-symmetry flag.

    ``pool_size_matched`` is the honest caveat column: the control arm only
    cancels the observed arm's max-of-N selection advantage when both arms select
    from pools of the same size.  A capped control pool leaves a smaller version
    of the original bias, and the report must say so rather than bury it.
    """

    check_summary_schema(summary, "case_summary")
    joint = controls[controls["control_type"].astype(str).str.endswith("_joint")]
    out: list[dict[str, object]] = []
    for keys, group in summary.groupby(list(CONDITION), observed=True):
        condition = dict(zip(CONDITION, keys))
        mask = np.ones(len(joint), dtype=bool)
        for column, value in condition.items():
            mask &= joint[column].to_numpy() == value
        control_rows = joint[mask]
        observed = case_level(group, "observed_matched_recovery", group=list(CONDITION))
        o_low, o_med, o_high = median_ci(observed["observed_matched_recovery"].tolist())
        per_case_control = (
            control_rows.loc[np.isfinite(control_rows["recovery_fraction"])]
            .groupby("case_id", observed=True)["recovery_fraction"]
            .median()
        )
        c_low, c_med, c_high = median_ci(per_case_control.tolist())
        p_low, p_med, p_high = median_ci(
            group["control_permutation_p"].tolist()
            if "control_permutation_p" in group.columns
            else []
        )
        pools = group[["observed_pool_size", "control_pool_size"]].to_numpy(float)
        finite_pools = pools[np.isfinite(pools).all(axis=1)]
        out.append(
            {
                **condition,
                # Cases outside the interaction subcohort carry NaN here and
                # are not evidence; count only those with an observed result.
                "n_cases": int(observed["case_id"].nunique()) if not observed.empty else 0,
                "observed_median": o_med,
                "observed_ci_low": o_low,
                "observed_ci_high": o_high,
                "control_median": c_med,
                "control_ci_low": c_low,
                "control_ci_high": c_high,
                "permutation_p_median": p_med,
                "permutation_p_ci_low": p_low,
                "permutation_p_ci_high": p_high,
                "pool_size_matched": bool(
                    finite_pools.size
                    and np.allclose(finite_pools[:, 0], finite_pools[:, 1])
                ),
                "observed_pool_median": float(np.median(finite_pools[:, 0]))
                if finite_pools.size
                else float("nan"),
                "control_pool_median": float(np.median(finite_pools[:, 1]))
                if finite_pools.size
                else float("nan"),
                "trials_skipped_total": int(
                    np.nansum(
                        group.get(
                            "n_trials_skipped_undersized", pd.Series(dtype=float)
                        ).to_numpy(float)
                    )
                ),
            }
        )
    return pd.DataFrame(out)


def rank_agreement_summary(summary: pd.DataFrame) -> pd.DataFrame:
    """Spearman agreement between the strength, energy and utility rankings."""

    check_summary_schema(summary, "case_summary")
    out: list[dict[str, object]] = []
    for pair in RANK_PAIRS:
        column = f"spearman_{pair}"
        if column not in summary.columns:
            continue
        per_case = case_level(summary, column, group=list(CONDITION))
        if per_case.empty:
            continue
        for keys, group in per_case.groupby(list(CONDITION), observed=True):
            low, med, high = median_ci(group[column].tolist())
            out.append(
                {
                    **dict(zip(CONDITION, keys)),
                    "pair": pair,
                    "n_cases": int(group["case_id"].nunique()),
                    "median": med,
                    "ci_low": low,
                    "ci_high": high,
                }
            )
    return pd.DataFrame(out)


# --- boundary-metric re-test -------------------------------------------------
#
# The driver scored HD95 and ASD on every intervention it ran but the cohort
# tables above only pool Dice.  ``subset_curves`` stores the *raw* post-
# intervention value (``hd95_removed`` is the HD95 of the prediction after the
# top-k patches were removed), so recovery is rebuilt here from
# ``clean_<m>`` and ``full_<m>_damage`` in case_summary.  ``patch_metrics``
# already stores a *difference* (``necessity_hd95`` = full damage minus damage
# after removal), so it is only normalised.

BOUNDARY_METRICS: tuple[str, ...] = ("hd95", "asd")
# Below 1 mm of boundary damage a recovery *fraction* has no meaning (one
# in-plane voxel on whole gland, two on zones) and is suppressed; the
# millimetre recovery is reported for every case regardless.
BOUNDARY_DAMAGE_FLOOR_MM = 1.0
BOUNDARY_TOP_FRACTIONS: tuple[float, ...] = (0.01, 0.05, 0.10, 0.20)


def _top_fractions(k: int, n_screened: int) -> list[float]:
    """Every TOP_FRACTIONS label an evaluated k stands for.

    The driver evaluates the *set* of ks, so on a case with few screened
    patches one k can serve several fractions (10 patches: k = 1 is 1 %, 5 %
    and 10 %).  Such a row is counted under each label it satisfies.
    """

    return [
        fraction
        for fraction in BOUNDARY_TOP_FRACTIONS
        if k == max(1, int(np.ceil(fraction * n_screened)))
    ]


def _case_condition_columns(summary: pd.DataFrame) -> pd.DataFrame:
    cols = [*CONDITION, "case_id", "n_patches_screened", "full_dice_damage"]
    for metric in BOUNDARY_METRICS:
        cols += [f"clean_{metric}", f"full_{metric}_damage"]
    return summary[cols].drop_duplicates([*CONDITION, "case_id"])


def cohort_boundary_recovery(summary: pd.DataFrame, curves: pd.DataFrame) -> pd.DataFrame:
    """HD95 / ASD recovery at each evaluated top fraction, case-clustered.

    One row per condition x ranking_type x top_fraction with, for each boundary
    metric, the median recovery fraction (cases above the damage floor), the
    median recovery in mm (all cases), the fraction of damage retained when
    only those patches are kept, and the Dice recovery fraction on the same row.
    """

    evaluated = curves[curves["evaluated"].astype(bool)]
    joined = evaluated.merge(
        _case_condition_columns(summary), on=[*CONDITION, "case_id"], how="inner"
    )
    joined = joined.assign(
        top_fraction=[
            _top_fractions(int(k), int(n))
            for k, n in zip(joined["k"], joined["n_patches_screened"])
        ]
    ).explode("top_fraction")
    joined = joined[joined["top_fraction"].notna()].astype({"top_fraction": float})
    for metric in BOUNDARY_METRICS:
        d_full = joined[f"full_{metric}_damage"]
        after_removal = joined[f"{metric}_removed"] - joined[f"clean_{metric}"]
        after_keep = joined[f"{metric}_kept"] - joined[f"clean_{metric}"]
        joined[f"{metric}_recovery_mm"] = d_full - after_removal
        above_floor = d_full >= BOUNDARY_DAMAGE_FLOOR_MM
        joined[f"{metric}_recovery_fraction"] = np.where(
            above_floor, (d_full - after_removal) / d_full, np.nan
        )
        joined[f"{metric}_kept_fraction"] = np.where(
            above_floor, after_keep / d_full, np.nan
        )
    joined["dice_recovery_fraction"] = joined["damage_removed_fraction"]

    group_cols = [*CONDITION, "ranking_type", "top_fraction"]
    values = ["dice_recovery_fraction"]
    for metric in BOUNDARY_METRICS:
        values += [
            f"{metric}_recovery_fraction",
            f"{metric}_recovery_mm",
            f"{metric}_kept_fraction",
        ]
    out: list[dict[str, object]] = []
    for keys, group in joined.groupby(group_cols, observed=True):
        row: dict[str, object] = dict(zip(group_cols, keys))
        row["n_cases"] = int(group["case_id"].nunique())
        for value in values:
            finite = group.loc[np.isfinite(group[value])]
            per_case = finite.groupby("case_id", observed=True)[value].median()
            low, med, high = median_ci(per_case.tolist())
            row[f"{value}_median"] = med
            row[f"{value}_ci_low"] = low
            row[f"{value}_ci_high"] = high
            row[f"n_cases_{value}"] = int(per_case.size)
        out.append(row)
    return pd.DataFrame(out).sort_values(group_cols).reset_index(drop=True)


def patch_boundary_necessity(summary: pd.DataFrame, patches: pd.DataFrame) -> pd.DataFrame:
    """Per-patch boundary necessity for the screened patches, case-clustered.

    ``necessity_<m>`` is the boundary damage a single patch's removal undoes;
    it is normalised by the case's full boundary damage (floor applied) and the
    within-case Spearman between Dice necessity and boundary necessity says
    whether the patches that matter for overlap are the ones that matter for
    the contour.
    """

    from scipy.stats import spearmanr

    # ``screened`` is NaN on energy-stage rows, which never had an intervention.
    screened = patches[patches["screened"].fillna(False).astype(bool)]
    # The driver scores boundary metrics only for its SURFACE_TOP patches per
    # case, so the rows that carry a value are the top-Dice-utility patches.
    measured = screened[
        np.isfinite(screened[[f"necessity_{m}" for m in BOUNDARY_METRICS]]).any(axis=1)
    ]
    joined = measured.merge(
        _case_condition_columns(summary), on=[*CONDITION, "case_id"], how="inner"
    )
    for metric in BOUNDARY_METRICS:
        d_full = joined[f"full_{metric}_damage"]
        joined[f"{metric}_necessity_fraction"] = np.where(
            d_full >= BOUNDARY_DAMAGE_FLOOR_MM, joined[f"necessity_{metric}"] / d_full, np.nan
        )

    out: list[dict[str, object]] = []
    for keys, group in joined.groupby(list(CONDITION), observed=True):
        row: dict[str, object] = dict(zip(CONDITION, keys))
        row["n_cases"] = int(group["case_id"].nunique())
        row["n_patches_per_case"] = float(group.groupby("case_id", observed=True).size().median())
        for metric in BOUNDARY_METRICS:
            value = f"{metric}_necessity_fraction"
            per_case = (
                group.loc[np.isfinite(group[value])]
                .groupby("case_id", observed=True)[value]
                .median()
            )
            low, med, high = median_ci(per_case.tolist())
            row[f"{value}_median"] = med
            row[f"{value}_ci_low"] = low
            row[f"{value}_ci_high"] = high
            row[f"n_cases_{value}"] = int(per_case.size)
            rhos = []
            for _, case in group.groupby("case_id", observed=True):
                pair = case[["necessity_dice", f"necessity_{metric}"]]
                pair = pair[np.isfinite(pair).all(axis=1)]
                if len(pair) >= 3 and pair.nunique().min() > 1:
                    rhos.append(float(spearmanr(pair.iloc[:, 0], pair.iloc[:, 1])[0]))
            low, med, high = median_ci(rhos)
            row[f"spearman_dice_{metric}_median"] = med
            row[f"spearman_dice_{metric}_ci_low"] = low
            row[f"spearman_dice_{metric}_ci_high"] = high
            # Cases where every top patch has identical boundary necessity
            # (usually all zero) have no rank to agree on and are not counted.
            row[f"n_cases_spearman_dice_{metric}"] = len(rhos)
        out.append(row)
    return pd.DataFrame(out)


__all__ = [
    "BOOTSTRAP_RESAMPLES",
    "BOOTSTRAP_SEED",
    "BOUNDARY_DAMAGE_FLOOR_MM",
    "BOUNDARY_METRICS",
    "CONCENTRATION_METRICS",
    "CURVE_VALUES",
    "REQUIRED_SUMMARY_COLUMNS",
    "SchemaError",
    "anatomical_category_summary",
    "case_level",
    "check_summary_schema",
    "cohort_concentration",
    "cohort_boundary_recovery",
    "cohort_controls",
    "cohort_curves",
    "median_ci",
    "patch_boundary_necessity",
    "rank_agreement_summary",
]
