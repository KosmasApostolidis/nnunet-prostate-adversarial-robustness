"""Tests for the Phase B sensitivity sweep aggregator driver.

Every fixture is built from the driver's own ``SUMMARY_FIELDNAMES`` /
``PATCH_FIELDNAMES`` field lists and written to a ``tmp_path`` sweep root, never
hand-copied from ``results/`` -- so a schema drift in the evaluator breaks
these tests too, same as every other test in this plan.
"""

from __future__ import annotations

import itertools
import math
import sys
from pathlib import Path

import pandas as pd
import pytest

REPO_ROOT = Path(__file__).resolve().parents[2]
for path in (REPO_ROOT, REPO_ROOT / "src"):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

from experiments.aggregate_edge_attribution_sensitivity import (  # noqa: E402
    BASELINE_VALUES,
    PARAMETERS,
    discover_settings,
    load_case_summary_k50,
    load_patch_top_sets,
    main,
    render_report,
)
from experiments.evaluate_edge_attribution_all_cases import (  # noqa: E402
    PATCH_CSV,
    PATCH_FIELDNAMES,
    SUMMARY_CSV,
    SUMMARY_FIELDNAMES,
)
from mri_prostate_seg.experiments.edge_attribution import (  # noqa: E402
    choose_primary,
    stability_table,
)


def _summary_frame(overrides: list[dict[str, object]]) -> pd.DataFrame:
    """A synthetic case_summary.csv frame built from the driver's own field list."""

    rows = []
    for override in overrides:
        row: dict[str, object] = {name: 0.0 for name in SUMMARY_FIELDNAMES}
        row.update(attack="APGD-BCE", epsilon_n=16, stage="causal")
        row.update(override)
        rows.append(row)
    return pd.DataFrame(rows, columns=SUMMARY_FIELDNAMES)


def _patch_frame(overrides: list[dict[str, object]]) -> pd.DataFrame:
    """A synthetic patch_metrics.csv frame built from the driver's own field list."""

    rows = []
    for override in overrides:
        row: dict[str, object] = {name: 0.0 for name in PATCH_FIELDNAMES}
        row.update(
            attack="APGD-BCE",
            epsilon_n=16,
            stage="causal",
            anatomical_category="apex",
            screen_reason="",
            screened=True,
        )
        row.update(override)
        rows.append(row)
    return pd.DataFrame(rows, columns=PATCH_FIELDNAMES)


def _write_setting(
    directory: Path,
    summary_overrides: list[dict[str, object]],
    patch_overrides: list[dict[str, object]],
) -> None:
    directory.mkdir(parents=True, exist_ok=True)
    _summary_frame(summary_overrides).to_csv(directory / SUMMARY_CSV, index=False)
    _patch_frame(patch_overrides).to_csv(directory / PATCH_CSV, index=False)


def _expected_name(parameter: str, value: float) -> str:
    return f"{parameter}={value:g}"


def _one_case(dataset: str, case_id: str, k50: float) -> list[dict[str, object]]:
    return [
        {
            "dataset": dataset,
            "case_id": case_id,
            "atlas_type": "native_edges",
            "k50_remove": k50,
        }
    ]


def _one_patch(dataset: str, case_id: str, patch_id: int) -> list[dict[str, object]]:
    return [
        {
            "dataset": dataset,
            "case_id": case_id,
            "atlas_type": "native_edges",
            "patch_id": patch_id,
            "utility_rank": 1.0,
        }
    ]


# --- 1. discover_settings ---------------------------------------------------


def test_discover_settings_finds_parameter_dirs_and_baseline_and_ignores_others(
    tmp_path: Path,
) -> None:
    sweep_root = tmp_path / "sweep"
    _write_setting(
        sweep_root / "baseline", _one_case("wg", "c1", 5.0), _one_patch("wg", "c1", 1)
    )
    _write_setting(
        sweep_root / "edge_quantile_0.8",
        _one_case("wg", "c1", 6.0),
        _one_patch("wg", "c1", 1),
    )
    # Unrelated directory and file must be skipped without being read as
    # setting directories -- they carry no case_summary.csv/patch_metrics.csv.
    (sweep_root / "scratch_notes").mkdir(parents=True)
    (sweep_root / "README.txt").write_text("not a setting", encoding="utf-8")

    points = discover_settings(sweep_root)

    names = {p.name for p in points}
    expected = {_expected_name(p, v) for p, v in BASELINE_VALUES.items()}
    expected.add("edge_quantile=0.8")
    assert names == expected
    assert len(points) == len(expected)  # no duplicate/extra points slipped in


# --- 2. baseline fan-out -----------------------------------------------------


def test_baseline_fans_out_to_exactly_the_pairs_a_four_fold_rerun_would_produce(
    tmp_path: Path,
) -> None:
    sweep_root = tmp_path / "sweep"
    _write_setting(
        sweep_root / "baseline", _one_case("wg", "c1", 10.0), _one_patch("wg", "c1", 1)
    )
    _write_setting(
        sweep_root / "edge_quantile_0.8",
        _one_case("wg", "c1", 11.0),
        _one_patch("wg", "c1", 2),
    )
    _write_setting(
        sweep_root / "edge_quantile_0.95",
        _one_case("wg", "c1", 12.0),
        _one_patch("wg", "c1", 3),
    )
    _write_setting(
        sweep_root / "patch_extent_mm_4",
        _one_case("wg", "c1", 13.0),
        _one_patch("wg", "c1", 4),
    )
    _write_setting(
        sweep_root / "tube_radius_mm_1",
        _one_case("wg", "c1", 14.0),
        _one_patch("wg", "c1", 5),
    )
    _write_setting(
        sweep_root / "ig_steps_16",
        _one_case("wg", "c1", 15.0),
        _one_patch("wg", "c1", 6),
    )

    points = discover_settings(sweep_root)
    table = stability_table(points)

    # edge_quantile has 3 points (baseline, 0.8, 0.95) -> C(3,2) = 3 pairs;
    # every other parameter has 2 points (baseline, one real setting) -> 1 pair.
    assert len(table) == 3 + 1 + 1 + 1

    eq_names = {"edge_quantile=0.9", "edge_quantile=0.8", "edge_quantile=0.95"}
    eq_rows = table[table["parameter"] == "edge_quantile"]
    observed_pairs = {
        frozenset((row["setting_a"], row["setting_b"])) for _, row in eq_rows.iterrows()
    }
    expected_pairs = {
        frozenset(pair) for pair in itertools.combinations(sorted(eq_names), 2)
    }
    assert observed_pairs == expected_pairs

    for parameter, baseline_name, other_name in (
        ("patch_extent_mm", "patch_extent_mm=6", "patch_extent_mm=4"),
        ("tube_radius_mm", "tube_radius_mm=2", "tube_radius_mm=1"),
        ("ig_steps", "ig_steps=32", "ig_steps=16"),
    ):
        rows = table[table["parameter"] == parameter]
        assert len(rows) == 1  # baseline pairs with this setting and nothing else
        row = rows.iloc[0]
        assert {row["setting_a"], row["setting_b"]} == {baseline_name, other_name}


# --- 3. case keying is dataset/case_id --------------------------------------


def test_case_keying_uses_dataset_and_case_id_not_case_id_alone(
    tmp_path: Path,
) -> None:
    directory = tmp_path / "setting"
    _write_setting(
        directory,
        [
            {
                "dataset": "wg",
                "case_id": "case1",
                "atlas_type": "native_edges",
                "k50_remove": 5.0,
            },
            {
                "dataset": "zones",
                "case_id": "case1",
                "atlas_type": "native_edges",
                "k50_remove": 9.0,
            },
        ],
        [
            {
                "dataset": "wg",
                "case_id": "case1",
                "atlas_type": "native_edges",
                "patch_id": 1,
                "utility_rank": 1.0,
            },
            {
                "dataset": "zones",
                "case_id": "case1",
                "atlas_type": "native_edges",
                "patch_id": 2,
                "utility_rank": 1.0,
            },
        ],
    )

    k50 = load_case_summary_k50(directory)
    assert k50 == {"wg/case1": pytest.approx(5.0), "zones/case1": pytest.approx(9.0)}

    top_sets = load_patch_top_sets(directory)
    assert top_sets == {"wg/case1": {1}, "zones/case1": {2}}


# --- 4. top-10% extraction ---------------------------------------------------


def test_top_set_takes_ceil_10pct_smallest_utility_rank_among_screened(
    tmp_path: Path,
) -> None:
    directory = tmp_path / "setting"
    screened_rows = [
        {
            "dataset": "wg",
            "case_id": "c1",
            "atlas_type": "native_edges",
            "patch_id": 100 + rank,
            "utility_rank": float(rank),
            "screened": True,
        }
        for rank in range(1, 24)  # 23 screened patches, ranks 1..23
    ]
    unscreened_rows = [
        {
            "dataset": "wg",
            "case_id": "c1",
            "atlas_type": "native_edges",
            "patch_id": patch_id,
            "utility_rank": 0.0,  # would win on rank alone if not excluded
            "screened": False,
        }
        for patch_id in range(1, 6)
    ]
    _write_setting(
        directory,
        _one_case("wg", "c1", 1.0),
        screened_rows + unscreened_rows,
    )

    top_sets = load_patch_top_sets(directory)

    # ceil(0.10 * 23) == 3 -> smallest 3 utility_rank among SCREENED rows only
    assert top_sets["wg/c1"] == {101, 102, 103}


# --- 5. native_edges-only filtering ------------------------------------------


def test_native_edges_only_filtering_applies_to_both_csvs(tmp_path: Path) -> None:
    directory = tmp_path / "setting"
    summary_overrides = [
        {
            "dataset": "wg",
            "case_id": "c1",
            "atlas_type": "native_edges",
            "k50_remove": 5.0,
        },
        {
            "dataset": "wg",
            "case_id": "c1",
            "atlas_type": "outer_boundary_gt",
            "k50_remove": 999.0,
        },
    ]
    patch_overrides = [
        {
            "dataset": "wg",
            "case_id": "c1",
            "atlas_type": "native_edges",
            "patch_id": 1,
            "utility_rank": 5.0,
        },
        {
            "dataset": "wg",
            "case_id": "c1",
            "atlas_type": "outer_boundary_gt",
            "patch_id": 2,
            "utility_rank": 0.0,  # would win the top set if not excluded
        },
    ]
    _write_setting(directory, summary_overrides, patch_overrides)

    k50 = load_case_summary_k50(directory)
    assert k50 == {"wg/c1": pytest.approx(5.0)}

    top_sets = load_patch_top_sets(directory)
    assert top_sets == {"wg/c1": {1}}


# --- 6. choose_primary -> None is reported, never a silent fallback --------


def test_report_states_parameter_unstable_and_names_no_chosen_setting(
    tmp_path: Path,
) -> None:
    sweep_root = tmp_path / "sweep"
    _write_setting(
        sweep_root / "baseline",
        _one_case("wg", "c1", 10.0),
        [
            {
                "dataset": "wg",
                "case_id": "c1",
                "atlas_type": "native_edges",
                "patch_id": pid,
                "utility_rank": float(pid),
            }
            for pid in range(1, 4)
        ],
    )
    _write_setting(
        sweep_root / "ig_steps_16",
        _one_case("wg", "c1", 90.0),
        [
            {
                "dataset": "wg",
                "case_id": "c1",
                "atlas_type": "native_edges",
                "patch_id": pid,
                "utility_rank": float(pid),
            }
            for pid in range(101, 104)  # disjoint patch ids -> jaccard 0
        ],
    )

    points = discover_settings(sweep_root)
    table = stability_table(points)
    report = render_report(points, table, min_jaccard=0.6)

    ig_section = report.split("## ig_steps")[1].split("## ")[0]
    assert "unstable" in ig_section
    assert "Chosen setting" not in ig_section


# --- 7. the report states the criterion is stability, not effect size -------


def test_report_states_the_criterion_is_stability_not_effect_size(
    tmp_path: Path,
) -> None:
    sweep_root = tmp_path / "sweep"
    _write_setting(
        sweep_root / "baseline", _one_case("wg", "c1", 10.0), _one_patch("wg", "c1", 1)
    )

    points = discover_settings(sweep_root)
    table = stability_table(points)
    report = render_report(points, table, min_jaccard=0.6)

    assert "Selection criterion: stability." in report
    assert "never because it makes edges look more or less important" in report


# --- 8. degenerate inputs -----------------------------------------------------


def test_main_exits_nonzero_with_a_clear_message_for_an_empty_sweep_root(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    empty_root = tmp_path / "empty"
    empty_root.mkdir()
    monkeypatch.setattr(sys, "argv", ["prog", "--sweep-root", str(empty_root)])

    with pytest.raises(SystemExit) as excinfo:
        main()

    assert str(empty_root) in str(excinfo.value)


def test_report_renders_without_crashing_when_a_parameter_has_one_point(
    tmp_path: Path,
) -> None:
    sweep_root = tmp_path / "sweep"
    _write_setting(
        sweep_root / "baseline", _one_case("wg", "c1", 10.0), _one_patch("wg", "c1", 1)
    )
    # No other setting directories: every parameter has exactly its baseline
    # fan-out point, i.e. one SweepPoint each -- no pair exists yet.
    points = discover_settings(sweep_root)
    table = stability_table(points)

    report = render_report(points, table, min_jaccard=0.6)  # must not raise

    for parameter in PARAMETERS:
        assert f"## {parameter}" in report


def test_a_setting_with_zero_cases_propagates_nan_and_is_excluded_by_the_filter(
    tmp_path: Path,
) -> None:
    sweep_root = tmp_path / "sweep"
    _write_setting(
        sweep_root / "baseline", _one_case("wg", "c1", 10.0), _one_patch("wg", "c1", 1)
    )
    # Only the anatomical atlas is present here -- native_edges filtering
    # leaves zero rows for this setting, i.e. zero cases.
    _write_setting(
        sweep_root / "tube_radius_mm_1",
        [
            {
                "dataset": "wg",
                "case_id": "c1",
                "atlas_type": "outer_boundary_gt",
                "k50_remove": 20.0,
            }
        ],
        [
            {
                "dataset": "wg",
                "case_id": "c1",
                "atlas_type": "outer_boundary_gt",
                "patch_id": 1,
                "utility_rank": 1.0,
            }
        ],
    )

    points = discover_settings(sweep_root)
    zero_case_point = next(p for p in points if p.name == "tube_radius_mm=1")
    assert zero_case_point.k50 == {}
    assert zero_case_point.top_sets == {}

    table = stability_table(points)
    row = table[table["parameter"] == "tube_radius_mm"].iloc[0]
    assert row["n_cases"] == 0
    assert math.isnan(row["jaccard_median"])

    assert choose_primary(table, "tube_radius_mm", min_jaccard=0.6) is None
