"""Cohort aggregation primitives for the edge-attribution tables."""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

REPO_ROOT = Path(__file__).resolve().parents[2]
for path in (REPO_ROOT, REPO_ROOT / "src"):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

from experiments.evaluate_edge_attribution_all_cases import (  # noqa: E402
    SUMMARY_FIELDNAMES,
)
from mri_prostate_seg.experiments.edge_attribution.aggregate import (  # noqa: E402
    CONCENTRATION_METRICS,
    REQUIRED_SUMMARY_COLUMNS,
    SchemaError,
    anatomical_category_summary,
    case_level,
    check_summary_schema,
    cohort_concentration,
    cohort_controls,
    cohort_curves,
    median_ci,
    rank_agreement_summary,
)


def _summary_frame(rows: int = 4) -> pd.DataFrame:
    """A synthetic case_summary built from the driver's own field list."""
    frame = pd.DataFrame({name: np.zeros(rows) for name in SUMMARY_FIELDNAMES})
    frame["dataset"] = "wg"
    frame["case_id"] = [f"ProstateWG_1000{i}" for i in range(rows)]
    frame["epsilon_n"] = 16
    frame["attack"] = "APGD-BCE"
    frame["atlas_type"] = "native_edges"
    frame["stage"] = "causal"
    return frame


def test_schema_guard_accepts_a_current_table() -> None:
    check_summary_schema(_summary_frame(), "synthetic")


def test_schema_guard_rejects_a_pre_fix_table_and_names_the_columns() -> None:
    # A table written before the selection-on-outcome fix has no pool columns, so
    # its control_permutation_p is biased small. Aggregating it would carry that
    # bias into the report, so the loader must refuse rather than fill or drop.
    stale = _summary_frame().drop(columns=["observed_pool_size", "control_pool_size"])
    with pytest.raises(SchemaError) as excinfo:
        check_summary_schema(stale, "merged/case_summary.csv")
    message = str(excinfo.value)
    assert "merged/case_summary.csv" in message
    assert "observed_pool_size" in message
    assert "control_pool_size" in message


def test_required_columns_are_a_subset_of_the_drivers_fields() -> None:
    # Guards against the plan naming a column the driver never writes.
    assert set(REQUIRED_SUMMARY_COLUMNS) <= set(SUMMARY_FIELDNAMES)


def test_case_level_reduces_repeated_measures_to_one_row_per_case() -> None:
    frame = _summary_frame(2)
    frame = pd.concat([frame, frame], ignore_index=True)
    frame["energy_gini"] = [0.2, 0.4, 0.6, 0.8]
    out = case_level(
        frame, "energy_gini", group=["dataset", "epsilon_n", "attack", "atlas_type"]
    )
    assert len(out) == 2  # two distinct case_ids, not four rows
    # per-case medians; approx because 0.6 isn't exactly representable as the
    # mean of 0.4 and 0.8 in IEEE-754.
    assert sorted(out["energy_gini"].tolist()) == pytest.approx([0.4, 0.6])


def test_case_level_drops_non_finite_values_before_reducing() -> None:
    frame = _summary_frame(2)
    frame["energy_gini"] = [np.nan, 0.5]
    out = case_level(frame, "energy_gini", group=["dataset"])
    assert len(out) == 1
    assert out["energy_gini"].iloc[0] == 0.5


def test_median_ci_brackets_the_median_and_is_seed_stable() -> None:
    values = list(np.linspace(0.0, 1.0, 51))
    low, med, high = median_ci(values)
    assert low <= med <= high
    assert med == pytest.approx(0.5, abs=1e-9)
    assert median_ci(values) == (low, med, high)  # deterministic


def test_median_ci_on_a_single_value_returns_it_three_times() -> None:
    assert median_ci([0.25]) == (0.25, 0.25, 0.25)


def test_median_ci_on_empty_input_is_all_nan() -> None:
    assert all(np.isnan(v) for v in median_ci([]))


def test_cohort_concentration_is_one_row_per_metric_and_condition() -> None:
    frame = _summary_frame(6)
    frame["energy_gini"] = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6]
    frame["n_eff_norm"] = 0.5
    out = cohort_concentration(frame)
    assert set(out["metric"]) == set(CONCENTRATION_METRICS)
    one = out[out["metric"] == "energy_gini"].iloc[0]
    assert one["n_cases"] == 6
    assert one["median"] == pytest.approx(0.35)
    assert one["ci_low"] <= one["median"] <= one["ci_high"]


def test_cohort_concentration_separates_conditions() -> None:
    frame = _summary_frame(4)
    frame["epsilon_n"] = [2, 2, 16, 16]
    frame["energy_gini"] = [0.1, 0.2, 0.8, 0.9]
    out = cohort_concentration(frame)
    gini = out[out["metric"] == "energy_gini"].set_index("epsilon_n")
    assert gini.loc[2, "median"] == pytest.approx(0.15)
    assert gini.loc[16, "median"] == pytest.approx(0.85)


def test_cohort_concentration_rejects_a_pre_fix_table() -> None:
    stale = _summary_frame().drop(columns=["control_pool_size"])
    with pytest.raises(SchemaError):
        cohort_concentration(stale)


def test_rank_agreement_summary_rejects_a_pre_fix_table() -> None:
    stale = _summary_frame().drop(columns=["control_pool_size"])
    with pytest.raises(SchemaError):
        rank_agreement_summary(stale)


def _curve_frame() -> pd.DataFrame:
    from experiments.evaluate_edge_attribution_all_cases import CURVES_FIELDNAMES

    rows = []
    for case in ("a", "b"):
        for k in (1, 2, 3):
            row = {name: 0.0 for name in CURVES_FIELDNAMES}
            row.update(
                dataset="wg",
                case_id=case,
                epsilon_n=16,
                attack="APGD-BCE",
                atlas_type="native_edges",
                stage="causal",
                ranking_type="utility",
                k=k,
                evaluated=True,
                damage_removed_fraction=0.1 * k * (1 if case == "a" else 2),
                volume_fraction=0.05 * k,
            )
            rows.append(row)
    return pd.DataFrame(rows)


def test_cohort_curves_pools_across_cases_at_each_k() -> None:
    out = cohort_curves(_curve_frame())
    assert len(out) == 3  # one row per k, not per (case, k)
    at_k2 = out[out["k"] == 2].iloc[0]
    assert at_k2["n_cases"] == 2
    # per-case values at k=2 are 0.2 and 0.4
    assert at_k2["damage_removed_fraction_median"] == pytest.approx(0.3)
    assert (
        at_k2["damage_removed_fraction_ci_low"]
        <= at_k2["damage_removed_fraction_median"]
        <= at_k2["damage_removed_fraction_ci_high"]
    )


def test_cohort_curves_ignores_unevaluated_k() -> None:
    frame = _curve_frame()
    frame.loc[frame["k"] == 3, "evaluated"] = False
    out = cohort_curves(frame)
    assert sorted(out["k"]) == [1, 2]


def test_cohort_curves_keeps_ranking_types_separate() -> None:
    frame = _curve_frame()
    other = frame.copy()
    other["ranking_type"] = "energy"
    other["damage_removed_fraction"] = 0.9
    out = cohort_curves(pd.concat([frame, other], ignore_index=True))
    assert set(out["ranking_type"]) == {"utility", "energy"}


def _patch_frame(categories: list[str]) -> pd.DataFrame:
    from experiments.evaluate_edge_attribution_all_cases import PATCH_FIELDNAMES

    rows = []
    for index, category in enumerate(categories):
        row = {name: 0.0 for name in PATCH_FIELDNAMES}
        row.update(
            dataset="wg",
            case_id="a" if index % 2 == 0 else "b",
            epsilon_n=16,
            attack="APGD-BCE",
            atlas_type="native_edges",
            stage="causal",
            patch_id=index + 1,
            anatomical_category=category,
            screened=True,
            energy_share_roi=0.1,
            fractional_recovery_dice=0.2 * (index + 1),
            volume_mm3=100.0,
        )
        rows.append(row)
    return pd.DataFrame(rows)


def test_anatomical_category_summary_is_one_row_per_category() -> None:
    out = anatomical_category_summary(_patch_frame(["apex", "apex", "mid", "mid"]))
    assert sorted(out["anatomical_category"]) == ["apex", "mid"]
    assert set(["n_cases", "n_patches", "median", "ci_low", "ci_high"]) <= set(
        out.columns
    )


def test_anatomical_category_summary_counts_cases_not_patches() -> None:
    # Four patches spread over two cases must report n_cases 2, n_patches 4:
    # weighting by patch count would let one large case dominate the cohort.
    out = anatomical_category_summary(_patch_frame(["apex"] * 4))
    row = out.iloc[0]
    assert row["n_cases"] == 2
    assert row["n_patches"] == 4


def test_anatomical_category_summary_ignores_unscreened_patches() -> None:
    frame = _patch_frame(["apex", "apex", "mid", "mid"])
    frame.loc[frame["anatomical_category"] == "mid", "screened"] = False
    out = anatomical_category_summary(frame)
    assert out["anatomical_category"].tolist() == ["apex"]


def test_cohort_controls_reports_observed_and_control_side_by_side() -> None:
    summary = _summary_frame(4)
    summary["observed_matched_recovery"] = [0.5, 0.6, 0.7, 0.8]
    summary["control_permutation_p"] = [0.05, 0.2, 0.5, 1.0]
    summary["observed_pool_size"] = 110
    summary["control_pool_size"] = 110
    controls = pd.DataFrame(
        {
            "dataset": "wg",
            "case_id": summary["case_id"],
            "epsilon_n": 16,
            "attack": "APGD-BCE",
            "atlas_type": "native_edges",
            "control_type": "volume_matched_joint",
            "recovery_fraction": [0.1, 0.2, 0.3, 0.4],
        }
    )
    out = cohort_controls(summary, controls).iloc[0]
    assert out["n_cases"] == 4
    assert out["observed_median"] == pytest.approx(0.65)
    assert out["control_median"] == pytest.approx(0.25)
    assert out["pool_size_matched"]  # both pools equal -> symmetric selection


def test_cohort_controls_flags_an_asymmetric_pool() -> None:
    summary = _summary_frame(2)
    summary["observed_pool_size"] = 110
    summary["control_pool_size"] = 20  # capped below the observed pool
    controls = pd.DataFrame(
        {
            "dataset": "wg",
            "case_id": summary["case_id"],
            "epsilon_n": 16,
            "attack": "APGD-BCE",
            "atlas_type": "native_edges",
            "control_type": "volume_matched_joint",
            "recovery_fraction": [0.1, 0.2],
        }
    )
    out = cohort_controls(summary, controls).iloc[0]
    assert not out["pool_size_matched"]


def test_rank_agreement_summary_covers_the_three_pairs() -> None:
    frame = _summary_frame(3)
    frame["spearman_strength_energy"] = [0.1, 0.2, 0.3]
    frame["spearman_energy_utility"] = [0.4, 0.5, 0.6]
    frame["spearman_strength_utility"] = [0.7, 0.8, 0.9]
    out = rank_agreement_summary(frame)
    assert set(out["pair"]) == {
        "strength_energy",
        "energy_utility",
        "strength_utility",
    }
    assert out[out["pair"] == "energy_utility"]["median"].iloc[0] == pytest.approx(0.5)


def test_schema_guard_rejects_a_pre_symmetric_m_table() -> None:
    """A table with the old pool columns but no drawable fraction is still biased.

    Before the symmetric-M fix the observed arm selected the top q of the whole
    screened set while the control arm could only draw from the patches whose
    matched region existed -- a volume-truncated subset.  Such a table has every
    column the earlier guard checked, so `control_drawable_fraction` is what
    tells the two apart.
    """

    stale = _summary_frame().drop(columns=["control_drawable_fraction"])
    with pytest.raises(SchemaError) as excinfo:
        check_summary_schema(stale, "merged/case_summary.csv")
    assert "control_drawable_fraction" in str(excinfo.value)
