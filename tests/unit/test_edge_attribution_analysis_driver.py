"""Loading, schema guard and cohort-CSV writing for the analysis driver."""

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

from experiments.evaluate_edge_attribution_all_cases import atlas_path  # noqa: E402
from experiments.plot_edge_attribution import (  # noqa: E402
    LIMITATIONS,
    LOW_N_CASES,
    _category_bar_spec,
    _curves_panels,
    _representative_case_atlases,
    _scatter_sample_cases,
    figure_01_atlas_qc_panel,
    figure_02_patch_energy_maps,
    figure_03_patch_utility_maps,
    figure_04_cumulative_curves,
    figure_05_strength_energy_utility,
    figure_06_anatomical_categories,
    figure_07_attack_budget_comparison,
    load_tables,
    write_cohort_csvs,
    write_figures,
    write_report,
)
from mri_prostate_seg.experiments.edge_attribution import SchemaError  # noqa: E402


def _identity_row(*, case_id: str, stage: str = "causal") -> dict[str, object]:
    return {
        "dataset": "wg",
        "case_id": case_id,
        "epsilon_n": 16,
        "attack": "APGD-BCE",
        "atlas_type": "native_edges",
        "stage": stage,
    }


def _summary_frame() -> pd.DataFrame:
    from experiments.evaluate_edge_attribution_all_cases import SUMMARY_FIELDNAMES

    rows = []
    for i in range(4):
        row = {name: 0.0 for name in SUMMARY_FIELDNAMES}
        row.update(_identity_row(case_id=f"ProstateWG_1000{i}"))
        row["energy_gini"] = 0.1 * (i + 1)
        row["observed_matched_recovery"] = 0.5
        row["observed_pool_size"] = 100
        row["control_pool_size"] = 100
        row["control_permutation_p"] = 0.1
        row["spearman_strength_energy"] = 0.5
        rows.append(row)
    return pd.DataFrame(rows)


def _patch_frame() -> pd.DataFrame:
    from experiments.evaluate_edge_attribution_all_cases import PATCH_FIELDNAMES

    rows = []
    for i in range(4):
        row = {name: 0.0 for name in PATCH_FIELDNAMES}
        row.update(_identity_row(case_id=f"ProstateWG_1000{i % 2}"))
        row.update(
            patch_id=i + 1,
            anatomical_category="apex",
            screened=True,
            energy=1.0,
            fractional_recovery_dice=0.2 * (i + 1),
            volume_mm3=100.0,
        )
        rows.append(row)
    return pd.DataFrame(rows)


def _curves_frame() -> pd.DataFrame:
    from experiments.evaluate_edge_attribution_all_cases import CURVES_FIELDNAMES

    rows = []
    for case_id in ("ProstateWG_10000", "ProstateWG_10001"):
        for k in (1, 2):
            row = {name: 0.0 for name in CURVES_FIELDNAMES}
            row.update(_identity_row(case_id=case_id))
            row.update(
                ranking_type="utility",
                k=k,
                evaluated=True,
                damage_removed_fraction=0.1 * k,
                volume_fraction=0.05 * k,
            )
            rows.append(row)
    return pd.DataFrame(rows)


def _control_frame() -> pd.DataFrame:
    from experiments.evaluate_edge_attribution_all_cases import CONTROL_FIELDNAMES

    rows = []
    for case_id in ("ProstateWG_10000", "ProstateWG_10001"):
        row = {name: 0.0 for name in CONTROL_FIELDNAMES}
        row.update(_identity_row(case_id=case_id))
        row.update(control_type="volume_matched_joint", recovery_fraction=0.3)
        rows.append(row)
    return pd.DataFrame(rows)


def _qc_frame() -> pd.DataFrame:
    from experiments.evaluate_edge_attribution_all_cases import QC_FIELDNAMES

    rows = []
    for case_id in ("ProstateWG_10000", "ProstateWG_10001"):
        row = {name: 0.0 for name in QC_FIELDNAMES}
        row.update(_identity_row(case_id=case_id))
        row.update(n_flags=0, flags="")
        rows.append(row)
    return pd.DataFrame(rows)


def _write_merged(tmp_path: Path, *, drop: list[str] | None = None) -> Path:
    """A minimal but schema-complete merged directory."""

    summary = _summary_frame().drop(columns=drop or [])
    summary.to_csv(tmp_path / "case_summary.csv", index=False)
    _patch_frame().to_csv(tmp_path / "patch_metrics.csv", index=False)
    _curves_frame().to_csv(tmp_path / "subset_curves.csv", index=False)
    _control_frame().to_csv(tmp_path / "control_summary.csv", index=False)
    _qc_frame().to_csv(tmp_path / "qc.csv", index=False)
    return tmp_path


def _write_atlas_npz(
    merged_dir: Path,
    *,
    dataset: str = "wg",
    case_id: str = "ProstateWG_10000",
    names: tuple[str, ...] = ("native_edges", "outer_boundary_gt"),
) -> Path:
    """A synthetic atlas cache at the real ``atlas_path`` location.

    Patch labels 1-4 cover the tube arrays so they overlap the patch_id
    values ``_patch_frame`` assigns (1-4). ``roi`` is a bounding box that
    covers every slice with an IDENTICAL voxel count (a real ``roi.sum()``
    tie), while the tube voxels are deliberately unbalanced across slices
    (0 on slice 0, 4 on slice 1, 16 on slice 2) so a test can tell the two
    slice-picking rules apart: ``roi.sum()`` argmax would return slice 0
    (the first of a three-way tie, at the box edge where the atlas is
    empty), while the boundary-tube-count rule this driver must use picks
    slice 2, where the atlas is actually visible.
    """

    shape = (3, 8, 8)
    roi = np.zeros(shape, dtype=np.uint8)
    roi[:, 1:7, 1:7] = 1
    strength = np.random.default_rng(0).random(shape).astype(np.float32)
    arrays: dict[str, np.ndarray] = {
        "names": np.array(list(names), dtype=str),
        "roi": roi,
        "strength": strength,
    }
    for name in names:
        tubes = np.zeros(shape, dtype=np.int32)
        tubes[1, 1:3, 1:3] = 1  # slice 1: 4 tube voxels
        tubes[2, 1:3, 1:3] = 2  # slice 2: 16 tube voxels (most)
        tubes[2, 1:3, 3:5] = 3
        tubes[2, 3:5, 1:3] = 4
        arrays[f"cores_{name}"] = tubes.copy()
        arrays[f"tubes_{name}"] = tubes
    path = atlas_path(merged_dir, dataset, case_id)
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez(path, **arrays)
    return path


def test_load_tables_reads_every_table(tmp_path: Path) -> None:
    tables = load_tables(_write_merged(tmp_path))
    assert not tables.summary.empty
    assert not tables.patches.empty
    assert not tables.curves.empty


def test_load_tables_refuses_a_pre_fix_directory(tmp_path: Path) -> None:
    merged = _write_merged(tmp_path, drop=["observed_pool_size"])
    with pytest.raises(SchemaError) as excinfo:
        load_tables(merged)
    assert "case_summary.csv" in str(excinfo.value)


def test_load_tables_prefers_parquet_over_a_stale_csv(tmp_path: Path) -> None:
    pytest.importorskip("pyarrow")
    merged = _write_merged(tmp_path)
    fresh = pd.read_csv(merged / "patch_metrics.csv")
    fresh["energy"] = 999.0
    fresh.to_parquet(merged / "patch_metrics.parquet")
    tables = load_tables(merged)
    assert (tables.patches["energy"] == 999.0).all()


def test_write_cohort_csvs_writes_the_four_named_files(tmp_path: Path) -> None:
    tables = load_tables(_write_merged(tmp_path))
    out = tmp_path / "analysis"
    written = write_cohort_csvs(tables, out)
    assert set(written) == {
        "concentration",
        "curves",
        "categories",
        "controls",
        "rank_agreement",
    }
    for path in written.values():
        assert path.is_file() and path.stat().st_size > 0


# ---------------------------------------------------------------------------
# Figures 01-04 (Task 6)
# ---------------------------------------------------------------------------


def test_representative_case_atlases_is_empty_when_the_atlas_cache_is_absent(
    tmp_path: Path,
) -> None:
    """The placeholder-trigger condition for figures 01-03, tested directly.

    No ``atlases/`` cache is written by ``_write_merged`` alone, so every
    figure that spatially renders per-patch data must see nothing here and
    fall back to a placeholder rather than crash or draw empty axes.
    """

    tables = load_tables(_write_merged(tmp_path))
    assert _representative_case_atlases(tables) == {}


def test_representative_case_atlases_finds_a_written_atlas_cache(
    tmp_path: Path,
) -> None:
    merged = _write_merged(tmp_path)
    _write_atlas_npz(merged)
    tables = load_tables(merged)
    panels = _representative_case_atlases(tables)
    assert set(panels) == {"wg"}
    case_id, data = panels["wg"]
    assert case_id == "ProstateWG_10000"
    assert "tubes_outer_boundary_gt" in data


def test_figure_01_placeholder_when_atlas_cache_absent(tmp_path: Path) -> None:
    tables = load_tables(_write_merged(tmp_path))
    out = figure_01_atlas_qc_panel(tables, tmp_path / "figures")
    assert out.is_file() and out.stat().st_size > 0


def test_slice_picker_prefers_tube_voxel_count_over_a_tied_roi_sum(
    tmp_path: Path,
) -> None:
    """The exact regression this driver must not repeat.

    ``_write_atlas_npz`` gives every slice the same ``roi.sum()`` (a real
    three-way tie) but concentrates tube voxels on slice 2. An ``argmax``
    over ``roi.sum()`` would return slice 0 -- the first tied slice, at the
    box edge, where no tubes exist at all. Verified by breaking the code:
    swapping the picker to ``int(np.argmax(roi.sum(axis=(1, 2))))`` makes
    this assertion fail (it returns 0, not 2).
    """

    from experiments.plot_edge_attribution import _slice_with_most_tube_voxels

    merged = _write_merged(tmp_path)
    npz_path = _write_atlas_npz(merged)
    with np.load(npz_path) as data:
        roi = data["roi"]
        tubes = data["tubes_outer_boundary_gt"]
    # The roi.sum()-per-slice tie this fixture is built to exercise.
    assert (roi.sum(axis=(1, 2)) == roi.sum(axis=(1, 2))[0]).all()
    assert _slice_with_most_tube_voxels(tubes) == 2


def test_figure_01_atlas_qc_panel_smoke(tmp_path: Path) -> None:
    merged = _write_merged(tmp_path)
    _write_atlas_npz(merged)
    tables = load_tables(merged)
    out = figure_01_atlas_qc_panel(tables, tmp_path / "figures")
    assert out.name == "01_atlas_qc_panel.png"
    assert out.is_file() and out.stat().st_size > 0


def test_figure_02_patch_energy_maps_smoke(tmp_path: Path) -> None:
    merged = _write_merged(tmp_path)
    _write_atlas_npz(merged)
    tables = load_tables(merged)
    out = figure_02_patch_energy_maps(tables, tmp_path / "figures")
    assert out.is_file() and out.stat().st_size > 0


def test_figure_02_placeholder_when_atlas_cache_absent(tmp_path: Path) -> None:
    tables = load_tables(_write_merged(tmp_path))
    out = figure_02_patch_energy_maps(tables, tmp_path / "figures")
    assert out.is_file() and out.stat().st_size > 0


def test_figure_03_patch_utility_maps_smoke(tmp_path: Path) -> None:
    merged = _write_merged(tmp_path)
    _write_atlas_npz(merged)
    tables = load_tables(merged)
    out = figure_03_patch_utility_maps(tables, tmp_path / "figures")
    assert out.is_file() and out.stat().st_size > 0


def test_curves_panels_still_reports_a_single_ranking_type(tmp_path: Path) -> None:
    """``_curves_frame`` only ever writes ``ranking_type="utility"``.

    Direct behavioural check that figure 04's panel/ranking selection does
    not silently need more than one ranking type to produce a result.
    """

    tables = load_tables(_write_merged(tmp_path))
    from experiments.plot_edge_attribution import cohort_curves

    curves = cohort_curves(tables.curves)
    _block, panels, ranking_types = _curves_panels(curves, 16)
    assert ranking_types == ["utility"]
    assert panels == [("wg", "native_edges")]


def test_figure_04_cumulative_curves_renders_with_a_single_ranking_type(
    tmp_path: Path,
) -> None:
    tables = load_tables(_write_merged(tmp_path))
    out = figure_04_cumulative_curves(tables, tmp_path / "figures")
    assert out.name == "04_cumulative_curves.png"
    assert out.is_file() and out.stat().st_size > 0


def test_figure_04_cumulative_curves_placeholder_when_curves_empty(
    tmp_path: Path,
) -> None:
    from experiments.evaluate_edge_attribution_all_cases import CURVES_FIELDNAMES

    merged = _write_merged(tmp_path)
    pd.DataFrame(columns=CURVES_FIELDNAMES).to_csv(
        merged / "subset_curves.csv", index=False
    )
    tables = load_tables(merged)
    out = figure_04_cumulative_curves(tables, tmp_path / "figures")
    assert out.is_file() and out.stat().st_size > 0


# --- Task 7: figures 05-07 -------------------------------------------------


def test_figure_05_strength_energy_utility_smoke(tmp_path: Path) -> None:
    tables = load_tables(_write_merged(tmp_path))
    out = figure_05_strength_energy_utility(tables, tmp_path / "figs")
    assert out.is_file() and out.stat().st_size > 0


def test_figure_05_placeholder_when_no_patch_is_screened(tmp_path: Path) -> None:
    merged = _write_merged(tmp_path)
    patches = pd.read_csv(merged / "patch_metrics.csv")
    patches["screened"] = False
    patches.to_csv(merged / "patch_metrics.csv", index=False)
    tables = load_tables(merged)
    out = figure_05_strength_energy_utility(tables, tmp_path / "figs")
    assert out.is_file() and out.stat().st_size > 0


def test_scatter_sample_is_bounded_and_deterministic() -> None:
    """The pooled scatters must not silently include the whole cohort."""

    many = pd.DataFrame({"case_id": [f"c{i:03d}" for i in range(200)]})
    sample = _scatter_sample_cases(many, limit=12)
    assert len(sample) <= 12
    assert sample == _scatter_sample_cases(many, limit=12)
    few = pd.DataFrame({"case_id": ["a", "b"]})
    assert _scatter_sample_cases(few, limit=12) == ["a", "b"]


def test_figure_06_anatomical_categories_smoke(tmp_path: Path) -> None:
    tables = load_tables(_write_merged(tmp_path))
    out = figure_06_anatomical_categories(tables, tmp_path / "figs")
    assert out.is_file() and out.stat().st_size > 0


def test_figure_06_keeps_and_labels_a_thin_category(tmp_path: Path) -> None:
    """A category with n_cases < 3 gets a bar, a grey colour and an "n < 3" label.

    Asserted on the bar spec, not the PNG: a figure that silently dropped the
    thin category would still write a perfectly plausible file, so a
    size-only smoke test could never catch it.
    """

    from mri_prostate_seg.experiments.edge_attribution import (
        anatomical_category_summary,
    )

    merged = _write_merged(tmp_path)
    patches = pd.read_csv(merged / "patch_metrics.csv")
    # "thin" is backed by one case; "thick" by the rest.
    patches["anatomical_category"] = ["thin"] + ["thick"] * (len(patches) - 1)
    patches["case_id"] = ["only_case"] + [f"c{i}" for i in range(len(patches) - 1)]
    patches["screened"] = True
    patches.to_csv(merged / "patch_metrics.csv", index=False)
    tables = load_tables(merged)

    summary = anatomical_category_summary(tables.patches)
    dataset = str(summary["dataset"].iloc[0])
    block, colors, labels = _category_bar_spec(summary, dataset)

    categories = list(block["anatomical_category"])
    assert "thin" in categories, "the thin category was dropped instead of flagged"
    index = categories.index("thin")
    assert int(block["n_cases"].iloc[index]) < LOW_N_CASES
    assert colors[index] == "0.7", "thin category should be greyed"
    assert f"n < {LOW_N_CASES}" in labels[index]
    # And a well-populated category must NOT be flagged, or the label is noise.
    if "thick" in categories:
        thick = categories.index("thick")
        assert colors[thick] == "tab:blue"
        assert f"n < {LOW_N_CASES}" not in labels[thick]

    out = figure_06_anatomical_categories(tables, tmp_path / "figs")
    assert out.is_file() and out.stat().st_size > 0


def test_figure_07_attack_budget_comparison_smoke(tmp_path: Path) -> None:
    tables = load_tables(_write_merged(tmp_path))
    out = figure_07_attack_budget_comparison(tables, tmp_path / "figs")
    assert out.is_file() and out.stat().st_size > 0


def test_figure_07_placeholder_when_no_concentration_metric_present(
    tmp_path: Path,
) -> None:
    merged = _write_merged(tmp_path)
    summary = pd.read_csv(merged / "case_summary.csv")
    for metric in ("energy_gini", "k50_remove", "damage_removed_top_10pct",
                   "useful_subset_volume_fraction"):
        summary[metric] = float("nan")
    summary.to_csv(merged / "case_summary.csv", index=False)
    tables = load_tables(merged)
    out = figure_07_attack_budget_comparison(tables, tmp_path / "figs")
    assert out.is_file() and out.stat().st_size > 0


# --- Task 8: the generated report ------------------------------------------


def test_report_includes_every_limitation(tmp_path: Path) -> None:
    """A limitation dropped by an edit is a limitation the reader never sees."""

    tables = load_tables(_write_merged(tmp_path))
    text = write_report(tables, tmp_path / "analysis", {}).read_text()
    for limitation in LIMITATIONS:
        first_sentence = limitation.split(".")[0]
        assert first_sentence in text


def test_report_reports_the_rms_drift_rate_not_just_the_maximum(
    tmp_path: Path,
) -> None:
    """Half the rows over the flag must read as a rate, not a lone worst case."""

    merged = _write_merged(tmp_path)
    summary = pd.read_csv(merged / "case_summary.csv")
    drift = [3.7e-2, 1e-4] * len(summary)
    summary["rms_reproduction_rel_error"] = drift[: len(summary)]
    summary.to_csv(merged / "case_summary.csv", index=False)
    tables = load_tables(merged)
    text = write_report(tables, tmp_path / "analysis", {}).read_text()
    assert "%" in text
    assert "rate, not a maximum" in text


def test_report_frames_a_small_drawable_pool_as_conditioning_not_asymmetry(
    tmp_path: Path,
) -> None:
    """M < N is the drawability caveat, NOT a residual selection bias.

    After the symmetric-M fix both arms select the top q of the same M
    drawable patches, so an unequal N and M no longer means the comparison is
    biased -- it means the endpoint is conditional on drawability. A report
    that called this "residual asymmetry" would tell every reader their
    controls are broken when they are not.
    """

    merged = _write_merged(tmp_path)
    summary = pd.read_csv(merged / "case_summary.csv")
    summary["observed_pool_size"] = 110
    summary["control_pool_size"] = 20
    summary["control_drawable_fraction"] = 20 / 110
    summary["stage"] = "interaction"
    summary.to_csv(merged / "case_summary.csv", index=False)
    tables = load_tables(merged)
    text = write_report(tables, tmp_path / "analysis", {}).read_text()

    assert "cancels by construction" in text
    assert "conditional on drawability" in text
    assert "residual asymmetry" not in text.replace(
        "is not a residual asymmetry", ""
    )
    # 20/110 is under a quarter, so the thin-pool warning must fire.  The
    # marker is distinctive: "structurally degenerate" also appears in the
    # standing Limitations section, so asserting on that phrase would pass
    # whether or not the warning fired.
    assert "Thin drawable pool:" in text


def test_report_does_not_warn_when_most_patches_are_drawable(
    tmp_path: Path,
) -> None:
    """The degenerate-atlas warning must not fire on a healthy pool."""

    merged = _write_merged(tmp_path)
    summary = pd.read_csv(merged / "case_summary.csv")
    summary["observed_pool_size"] = 110
    summary["control_pool_size"] = 90
    summary["control_drawable_fraction"] = 90 / 110
    summary["stage"] = "interaction"
    summary.to_csv(merged / "case_summary.csv", index=False)
    tables = load_tables(merged)
    text = write_report(tables, tmp_path / "analysis", {}).read_text()
    assert "Thin drawable pool:" not in text


def test_report_is_generated_from_the_tables_not_hardcoded(tmp_path: Path) -> None:
    """Change a table, and the provenance section must change with it."""

    merged = _write_merged(tmp_path)
    # Rename the attack in EVERY table: the anatomy section is built from
    # patch_metrics, so changing only case_summary would leave the old label
    # legitimately present and the assertion would be testing the fixture,
    # not the report.
    for name in ("case_summary.csv", "patch_metrics.csv", "subset_curves.csv",
                 "control_summary.csv", "qc.csv"):
        frame = pd.read_csv(merged / name)
        if "attack" in frame.columns:
            frame["attack"] = "FGSM-BCE"
            frame.to_csv(merged / name, index=False)
    tables = load_tables(merged)
    text = write_report(tables, tmp_path / "analysis", {}).read_text()
    assert "FGSM-BCE" in text
    assert "APGD-BCE" not in text


def test_end_to_end_writes_every_csv_figure_and_the_report(tmp_path: Path) -> None:
    """The whole driver on a synthetic merged directory."""

    merged = _write_merged(tmp_path)
    _write_atlas_npz(merged)
    out = tmp_path / "analysis"
    tables = load_tables(merged)
    written = write_cohort_csvs(tables, out)
    figures = write_figures(tables, out)
    report = write_report(tables, out, figures, merged_dir=merged)

    assert len(written) == 5
    assert len(figures) == 7
    for path in list(written.values()) + list(figures.values()) + [report]:
        assert path.is_file() and path.stat().st_size > 0
    names = sorted(p.name for p in figures.values())
    assert names == [
        "01_atlas_qc_panel.png",
        "02_patch_energy_maps.png",
        "03_patch_utility_maps.png",
        "04_cumulative_curves.png",
        "05_strength_energy_utility.png",
        "06_anatomical_categories.png",
        "07_attack_budget_comparison.png",
    ]
