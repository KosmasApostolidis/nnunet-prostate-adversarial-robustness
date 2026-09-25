"""Cohort summaries, figures and report for the edge-attribution tables.

Reads a merged ``all_cases_edge_attribution`` directory (written by
``merge_edge_attribution_shards.py``), refuses a ``case_summary.csv`` written
before the selection-on-outcome fix, and writes the cohort-level CSVs that
Tasks 6-8 turn into figures and a generated report.
"""

from __future__ import annotations

import argparse
import json
import sys
import textwrap
from dataclasses import dataclass
from pathlib import Path

import matplotlib
import numpy as np
import pandas as pd

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))
SRC_ROOT = REPO_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from experiments.evaluate_edge_attribution_all_cases import (  # noqa: E402
    DEFAULT_RESULTS,
    atlas_path,
)
from mri_prostate_seg.experiments.edge_attribution import (  # noqa: E402
    anatomical_category_summary,
    check_summary_schema,
    cohort_concentration,
    cohort_controls,
    cohort_curves,
    rank_agreement_summary,
)

DEFAULT_MERGED_DIR = DEFAULT_RESULTS / "all_cases_edge_attribution"

TABLE_FILES: dict[str, tuple[str, ...]] = {
    "summary": ("case_summary.csv",),
    "patches": ("patch_metrics.parquet", "patch_metrics.csv"),
    "curves": ("subset_curves.parquet", "subset_curves.csv"),
    "controls": ("control_summary.csv",),
    "qc": ("qc.csv",),
}

# The cohort CSVs written by ``write_cohort_csvs``, keyed the same way as the
# dict it returns.
COHORT_CSV_NAMES: dict[str, str] = {
    "concentration": "cohort_concentration.csv",
    "curves": "cohort_curves.csv",
    "categories": "anatomical_category_summary.csv",
    "controls": "cohort_controls.csv",
    "rank_agreement": "rank_agreement.csv",
}


@dataclass(frozen=True)
class MergedTables:
    summary: pd.DataFrame
    patches: pd.DataFrame
    curves: pd.DataFrame
    controls: pd.DataFrame
    qc: pd.DataFrame
    # The directory the tables were read from. Figures 01-03 need it to find
    # the per-case atlas cache (``atlases/<dataset>/<case_id>.npz``, written
    # unconditionally by ``load_or_build_atlases`` in the evaluator) -- the
    # DataFrames alone carry no spatial information.
    merged_dir: Path


def _read_first(merged_dir: Path, names: tuple[str, ...]) -> tuple[pd.DataFrame, Path]:
    """Read the first of ``names`` that exists, preferring an earlier name.

    ``TABLE_FILES`` lists the parquet name before the csv name for the tables
    the merge step writes as parquet, so a stale csv left beside a fresh
    parquet is never picked up.
    """

    for name in names:
        path = merged_dir / name
        if path.is_file():
            frame = (
                pd.read_parquet(path)
                if path.suffix == ".parquet"
                else pd.read_csv(path)
            )
            return frame, path
    raise FileNotFoundError(f"none of {names} under {merged_dir}")


def load_tables(merged_dir: Path) -> MergedTables:
    """Read every merged table, refusing a summary written before the fix.

    ``check_summary_schema`` is called immediately after the summary is read
    so a pre-fix directory fails before any figure is drawn or any cohort
    statistic (all of which depend on ``control_permutation_p``) is computed.
    """

    summary, summary_path = _read_first(merged_dir, TABLE_FILES["summary"])
    check_summary_schema(summary, str(summary_path))
    patches, _ = _read_first(merged_dir, TABLE_FILES["patches"])
    curves, _ = _read_first(merged_dir, TABLE_FILES["curves"])
    controls, _ = _read_first(merged_dir, TABLE_FILES["controls"])
    qc, _ = _read_first(merged_dir, TABLE_FILES["qc"])
    return MergedTables(
        summary=summary,
        patches=patches,
        curves=curves,
        controls=controls,
        qc=qc,
        merged_dir=merged_dir,
    )


def write_cohort_csvs(tables: MergedTables, out_dir: Path) -> dict[str, Path]:
    """Compute the five cohort tables (Tasks 2-3) and write them to ``out_dir``."""

    out_dir.mkdir(parents=True, exist_ok=True)
    frames = {
        "concentration": cohort_concentration(tables.summary),
        "curves": cohort_curves(tables.curves),
        "categories": anatomical_category_summary(tables.patches),
        "controls": cohort_controls(tables.summary, tables.controls),
        "rank_agreement": rank_agreement_summary(tables.summary),
    }
    written: dict[str, Path] = {}
    for key, frame in frames.items():
        path = out_dir / COHORT_CSV_NAMES[key]
        frame.to_csv(path, index=False)
        written[key] = path
    return written


# ---------------------------------------------------------------------------
# Figures
# ---------------------------------------------------------------------------
# The evaluator's own docstring for ``causal_stage`` and ``load_or_build_atlases``
# is the source of truth here, not the task brief's literal
# ``maps/<dataset>/<case>/atlases.npz`` path, which does not exist anywhere in
# the codebase (verified by grep). Two distinct npz caches exist under a
# merged directory: ``atlases/<dataset>/<case_id>.npz`` (``atlas_path``), the
# per-case ROI/cores/tubes geometry, written unconditionally; and
# ``maps/<dataset>/<case_id>/<attack>_eps<n>.npz`` (``map_path``), per-voxel
# energy/necessity/sufficiency arrays for one attack/epsilon, written only
# when ``--save-maps`` is set. Figure 01 needs tube geometry, which only the
# first cache holds, so it reads ``atlas_path`` and calls the placeholder
# "maps were not saved for this run" (echoing the brief's wording) when that
# per-case cache is absent -- the actual failure mode the placeholder guards
# against, since a ``--save-maps`` run still writes ``atlases/`` on its own.
FIGURE_DPI = 200
PRIMARY_EPSILON = 16
REPRESENTATIVE_ATLAS_TYPE = "native_edges"
ENERGY_EPSILON_GRID = (2, 8, 32)
# Pooled medians at a k that only a handful of cases reached are noise: on the
# zones cohort the utility curve past k~40 swung between 0.05 and 0.35 because
# 1-5 cases had that many patches.  Only the Fibonacci grid {1,2,3,5,8,...} is
# scored on every case; other k are case-specific.  A curve is drawn where at
# least this many cases contributed -- or every case, when the cohort is
# smaller than this, so a two-case fixture still exercises the real figure.
MIN_CURVE_CASES = 10
MAPS_NOT_SAVED_MESSAGE = (
    "No atlas maps were saved for this run\n(atlases/<dataset>/<case_id>.npz "
    "not found under the merged directory)."
)


def _suptitle(fig: "matplotlib.figure.Figure", text: str) -> None:
    """Suptitle wrapped to the figure width.

    Several titles run to 150+ characters; ``tight_layout`` does not wrap a
    suptitle, so on the 6-inch category figure and the 15-inch scatter figure
    they were cut off at both margins.  ~8 characters per inch at the default
    suptitle size, floored so a narrow figure still gets a readable line.
    """

    width = max(40, int(fig.get_figwidth() * 8))
    fig.suptitle(textwrap.fill(text, width))


def _placeholder(out_path: Path, message: str) -> Path:
    """Write a labelled placeholder panel instead of an empty/blank figure.

    A blank axes is the worst failure a QC figure can have -- this pipeline
    already shipped one (a healthy 9-patch boundary atlas rendered blank
    because the slice was chosen by ``argmax`` over a tied count). Every
    figure function below reaches this instead of leaving axes empty.
    """

    fig, ax = plt.subplots(figsize=(6, 4))
    ax.text(0.5, 0.5, message, ha="center", va="center", wrap=True, fontsize=10)
    ax.axis("off")
    fig.savefig(out_path, dpi=FIGURE_DPI, bbox_inches="tight")
    plt.close(fig)
    return out_path


def _representative_cases(summary: pd.DataFrame) -> dict[str, str]:
    """One case_id per dataset, chosen deterministically (lowest sorted id)."""

    out: dict[str, str] = {}
    for dataset, group in summary.groupby("dataset", observed=True):
        case_ids = sorted(group["case_id"].astype(str).unique())
        if case_ids:
            out[str(dataset)] = case_ids[0]
    return out


def _load_atlas_npz(
    merged_dir: Path, dataset: str, case_id: str
) -> dict[str, np.ndarray] | None:
    """Read the per-case atlas cache, or ``None`` if it was never written."""

    path = atlas_path(merged_dir, dataset, case_id)
    if not path.is_file():
        return None
    with np.load(path) as data:
        return {key: data[key] for key in data.files}


def _representative_case_atlases(
    tables: MergedTables,
) -> dict[str, tuple[str, dict[str, np.ndarray]]]:
    """``{dataset: (case_id, atlas_npz)}`` for every dataset with a cached atlas.

    A dataset is silently absent from the result when its representative
    case has no atlas cache -- callers must handle an empty (or partial)
    result with a placeholder, never by indexing into it blindly.
    """

    out: dict[str, tuple[str, dict[str, np.ndarray]]] = {}
    for dataset, case_id in _representative_cases(tables.summary).items():
        data = _load_atlas_npz(tables.merged_dir, dataset, case_id)
        if data is not None:
            out[dataset] = (case_id, data)
    return out


def _slice_with_most_tube_voxels(tubes: np.ndarray, axis: int = 0) -> int:
    """The slice index along ``axis`` with the most nonzero tube voxels.

    Never choose the slice by ``roi.sum()``: the ROI is a bounding box, so
    many slices tie on that count and ``argmax`` returns the first tied
    slice, which sits at the box edge where the gland barely appears -- the
    exact failure this pipeline already shipped once. Counting tube voxels
    (real patch geometry) picks a slice where the atlas is actually visible.
    """

    reduce_axes = tuple(a for a in range(tubes.ndim) if a != axis)
    counts = (tubes > 0).sum(axis=reduce_axes)
    return int(np.argmax(counts))


def figure_01_atlas_qc_panel(tables: MergedTables, out_dir: Path) -> Path:
    """ROI outline and tube geometry for one representative case per dataset."""

    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / "01_atlas_qc_panel.png"
    panels = _representative_case_atlases(tables)
    if not panels:
        return _placeholder(out_path, MAPS_NOT_SAVED_MESSAGE)

    columns = ("native_edges", "outer_boundary_gt", "zone_interface_gt")
    rows = sorted(panels)
    fig, axes = plt.subplots(
        len(rows),
        1 + len(columns),
        figsize=(4 * (1 + len(columns)), 4 * len(rows)),
        squeeze=False,
    )
    for row, dataset in enumerate(rows):
        case_id, data = panels[dataset]
        names = [str(n) for n in data["names"]]
        roi = data["roi"].astype(bool)
        slice_source = "outer_boundary_gt" if "outer_boundary_gt" in names else names[0]
        z = _slice_with_most_tube_voxels(data[f"tubes_{slice_source}"])

        ax = axes[row, 0]
        ax.imshow(roi[z], cmap="gray")
        if roi[z].any():
            ax.contour(roi[z], colors="red", linewidths=1)
        # The case id is ~50 characters and collided with the neighbouring
        # panel's title on one line; it goes on its own smaller line instead.
        ax.set_title(f"{dataset}: ROI (z={z})\n{case_id}", fontsize=8)
        ax.axis("off")

        for col, name in enumerate(columns, start=1):
            ax = axes[row, col]
            if name in names:
                tube_slice = data[f"tubes_{name}"][z]
                ax.imshow(roi[z], cmap="gray", alpha=0.3)
                ax.imshow(np.ma.masked_equal(tube_slice, 0), cmap="tab20")
                ax.set_title(f"{name} tubes")
            else:
                ax.text(
                    0.5, 0.5, f"{name}\nn/a for {dataset}", ha="center", va="center"
                )
            ax.axis("off")
    _suptitle(
        fig,
        "Atlas QC: ROI and tube geometry, slice chosen by boundary-tube voxel count",
    )
    fig.tight_layout()
    fig.savefig(out_path, dpi=FIGURE_DPI, bbox_inches="tight")
    plt.close(fig)
    return out_path


def _paint_patch_values(
    tubes: np.ndarray, patch_ids: pd.Series, values: pd.Series
) -> np.ndarray:
    """Colour every patch's tube region by its already-measured value.

    Renders the DataFrame's own per-patch column spatially -- it does not
    recompute anything.
    """

    out = np.full(tubes.shape, np.nan, dtype=np.float32)
    for pid, value in zip(patch_ids, values, strict=True):
        if np.isfinite(value):
            out[tubes == int(pid)] = float(value)
    return out


def _patch_value_figure(
    tables: MergedTables,
    out_path: Path,
    *,
    value_column: str,
    diverging: bool,
    cbar_label: str,
    title: str,
) -> Path:
    """Shared per-patch spatial-map figure used by figures 02 and 03."""

    panels = _representative_case_atlases(tables)
    if not panels or tables.patches.empty:
        return _placeholder(out_path, MAPS_NOT_SAVED_MESSAGE)

    epsilons = sorted(int(e) for e in tables.patches["epsilon_n"].unique())
    grid = [e for e in ENERGY_EPSILON_GRID if e in epsilons] or epsilons[:3]
    if not grid:
        return _placeholder(out_path, "No epsilon values present in patch_metrics.")

    datasets = sorted(panels)
    fig, axes = plt.subplots(
        len(datasets),
        len(grid),
        figsize=(4 * len(grid), 4 * len(datasets)),
        squeeze=False,
    )
    image = None
    for row, dataset in enumerate(datasets):
        case_id, data = panels[dataset]
        names = [str(n) for n in data["names"]]
        if REPRESENTATIVE_ATLAS_TYPE not in names:
            for col in range(len(grid)):
                ax = axes[row, col]
                ax.text(
                    0.5,
                    0.5,
                    f"{REPRESENTATIVE_ATLAS_TYPE}\nn/a for {dataset}",
                    ha="center",
                    va="center",
                )
                ax.axis("off")
            continue
        tubes = data[f"tubes_{REPRESENTATIVE_ATLAS_TYPE}"]
        z = _slice_with_most_tube_voxels(tubes)
        for col, eps in enumerate(grid):
            ax = axes[row, col]
            subset = tables.patches[
                (tables.patches["dataset"] == dataset)
                & (tables.patches["case_id"].astype(str) == case_id)
                & (tables.patches["epsilon_n"] == eps)
                & (tables.patches["atlas_type"] == REPRESENTATIVE_ATLAS_TYPE)
            ]
            if subset.empty:
                ax.text(0.5, 0.5, "no patches\nat this ε", ha="center", va="center")
                ax.axis("off")
                continue
            value_map = _paint_patch_values(
                tubes[z], subset["patch_id"], subset[value_column]
            )
            if diverging:
                finite = value_map[np.isfinite(value_map)]
                vmax = float(np.abs(finite).max()) if finite.size else 1.0
                vmax = vmax if vmax > 0 else 1.0
                image = ax.imshow(value_map, cmap="RdBu_r", vmin=-vmax, vmax=vmax)
            else:
                image = ax.imshow(value_map, cmap="viridis")
            ax.set_title(f"{dataset} ε={eps}/255")
            ax.axis("off")
    if image is not None:
        fig.colorbar(image, ax=axes, shrink=0.6, label=cbar_label)
    _suptitle(fig, title)
    fig.savefig(out_path, dpi=FIGURE_DPI, bbox_inches="tight")
    plt.close(fig)
    return out_path


def figure_02_patch_energy_maps(tables: MergedTables, out_dir: Path) -> Path:
    """Per-patch share of ROI perturbation energy, one column per ε."""

    out_dir.mkdir(parents=True, exist_ok=True)
    return _patch_value_figure(
        tables,
        out_dir / "02_patch_energy_maps.png",
        value_column="energy_share_roi",
        diverging=False,
        cbar_label="share of ROI perturbation energy",
        title=(
            f"Per-patch perturbation-energy share (atlas_type="
            f"{REPRESENTATIVE_ATLAS_TYPE})"
        ),
    )


def figure_03_patch_utility_maps(tables: MergedTables, out_dir: Path) -> Path:
    """Per-patch ``fractional_recovery_dice``, diverging colormap centred at zero.

    Necessity can be negative -- removing a patch can make the segmentation
    worse -- so a sequential colormap would hide the sign of the effect.
    """

    out_dir.mkdir(parents=True, exist_ok=True)
    return _patch_value_figure(
        tables,
        out_dir / "03_patch_utility_maps.png",
        value_column="fractional_recovery_dice",
        diverging=True,
        # The colourbar sits low on the 4-inch canvas, so a vertical label
        # past ~30 characters runs off the bottom edge where
        # bbox_inches="tight" cannot recover it; the qualifiers go in the title.
        cbar_label="fractional recovery dice",
        title=(
            f"Per-patch causal utility (atlas_type={REPRESENTATIVE_ATLAS_TYPE}): "
            "necessity; negative = harmful to remove"
        ),
    )


def _curves_panels(
    curves: pd.DataFrame, epsilon: int, min_cases: int = MIN_CURVE_CASES
) -> tuple[pd.DataFrame, list[tuple[str, str]], list[str]]:
    """The block of ``cohort_curves`` at ``epsilon``, its panels and ranking types.

    Split out from :func:`figure_04_cumulative_curves` so the "single
    ranking_type still renders" and "sparse k are dropped" behaviours are
    directly testable rather than only inferable from a PNG's byte size.
    Rows with fewer than ``min(min_cases, cohort size)`` contributing cases
    are dropped.
    """

    block = curves[curves["epsilon_n"] == epsilon]
    if not block.empty:
        floor = min(min_cases, int(block["n_cases"].max()))
        block = block[block["n_cases"] >= floor]
    panels = sorted(
        {(str(d), str(a)) for d, a in zip(block["dataset"], block["atlas_type"])}
    )
    ranking_types = sorted(str(r) for r in block["ranking_type"].unique())
    return block, panels, ranking_types


def figure_04_cumulative_curves(tables: MergedTables, out_dir: Path) -> Path:
    """The primary figure: does the utility ranking remove damage faster than energy?

    One panel per (dataset, atlas_type) at the primary ε; one line per
    ranking_type actually present in the data (the evaluator only ever
    writes ``{utility, energy, strength, ig}`` -- never "greedy", which has
    no per-k cumulative curve, only a joint-necessity search reported in
    ``control_summary.csv``); K50/K80 (utility-ranking) marked with vertical
    rules.
    """

    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / "04_cumulative_curves.png"
    # Guard before calling cohort_curves: with zero evaluated rows it builds
    # its output from an empty list of dicts, which drops every column
    # (including the ones its own .sort_values(group_cols) needs) and raises
    # KeyError rather than returning an empty frame. aggregate.py is off
    # limits to edit, so the empty case is caught here instead.
    if tables.curves.empty or not tables.curves["evaluated"].astype(bool).any():
        return _placeholder(out_path, "No evaluated subset curves for this cohort.")
    curves = cohort_curves(tables.curves)
    if curves.empty:
        return _placeholder(out_path, "No evaluated subset curves for this cohort.")

    epsilons = sorted(int(e) for e in curves["epsilon_n"].unique())
    eps = PRIMARY_EPSILON if PRIMARY_EPSILON in epsilons else epsilons[0]
    block, panels, ranking_types = _curves_panels(curves, eps)
    if not panels:
        return _placeholder(out_path, f"No subset curves at epsilon_n={eps}.")

    concentration = cohort_concentration(tables.summary)
    k_marks = concentration[
        concentration["metric"].isin(["k50_remove", "k80_remove"])
        & (concentration["epsilon_n"] == eps)
    ]

    attacks = sorted(str(a) for a in block["attack"].unique())
    line_styles = ["-", "--", ":", "-."]
    attack_styles = {
        a: line_styles[i % len(line_styles)] for i, a in enumerate(attacks)
    }
    cmap = plt.get_cmap("tab10")
    ranking_colors = {r: cmap(i) for i, r in enumerate(ranking_types)}

    fig, axes = plt.subplots(
        1, len(panels), figsize=(5 * len(panels), 4.5), squeeze=False
    )
    axes_row = axes[0]
    for ax, (dataset, atlas_type) in zip(axes_row, panels, strict=True):
        panel = block[
            (block["dataset"] == dataset) & (block["atlas_type"] == atlas_type)
        ]
        n_cases = int(panel["n_cases"].max()) if not panel.empty else 0
        for ranking in ranking_types:
            for attack in attacks:
                sub = panel[
                    (panel["ranking_type"] == ranking) & (panel["attack"] == attack)
                ].sort_values("k")
                if sub.empty:
                    continue
                ax.plot(
                    sub["k"],
                    sub["damage_removed_fraction_median"],
                    color=ranking_colors[ranking],
                    linestyle=attack_styles[attack],
                    label=f"{ranking} / {attack}",
                )
                ax.fill_between(
                    sub["k"],
                    sub["damage_removed_fraction_ci_low"],
                    sub["damage_removed_fraction_ci_high"],
                    color=ranking_colors[ranking],
                    alpha=0.12,
                )
        marks = k_marks[
            (k_marks["dataset"] == dataset) & (k_marks["atlas_type"] == atlas_type)
        ]
        for _, row in marks.iterrows():
            style = "--" if row["metric"] == "k50_remove" else ":"
            ax.axvline(
                float(row["median"]), color="k", linestyle=style, linewidth=1, alpha=0.6
            )
        ax.set_xlabel("k (patches removed; K50/K80 marks are utility-order)")
        ax.set_title(f"{dataset}/{atlas_type} (ε={eps}/255, n_cases={n_cases})")
    # One y-label, on the left panel: the 52-character version on every panel
    # was taller than the 4.5-inch axis and ran into the title and x-label.
    axes_row[0].set_ylabel("fraction of adversarial damage removed")
    handles, labels = axes_row[0].get_legend_handles_labels()
    if handles:
        fig.legend(
            handles, labels, fontsize=7, loc="lower center", ncol=min(len(labels), 5)
        )
    _suptitle(
        fig,
        "Cumulative damage removed by ranking (dashed=K50, dotted=K80, utility "
        f"order); a curve is drawn only at k reached by at least {MIN_CURVE_CASES} "
        "cases",
    )
    fig.tight_layout(rect=(0.0, 0.1, 1.0, 1.0))
    fig.savefig(out_path, dpi=FIGURE_DPI, bbox_inches="tight")
    plt.close(fig)
    return out_path


# --- Task 7: figures 05-07 -------------------------------------------------

# Pooled per-patch scatters are dominated by whichever cases have the most
# patches, so they are drawn from a bounded, deterministic sample of cases and
# the count is stated on the figure rather than left implicit.
SCATTER_CASE_SAMPLE = 12

# A category backed by fewer cases than this is drawn, but greyed and labelled:
# dropping it silently would hide exactly the anatomy the cohort is thin on.
LOW_N_CASES = 3

# The concentration metrics whose behaviour across the epsilon ladder is the
# subject of figure 07.
BUDGET_METRICS: tuple[str, ...] = (
    "energy_gini",
    "k50_remove",
    "damage_removed_top_10pct",
    "useful_subset_volume_fraction",
)


def _scatter_sample_cases(
    patches: pd.DataFrame, limit: int = SCATTER_CASE_SAMPLE
) -> list[str]:
    """A deterministic, evenly-spread sample of case ids for the scatters.

    Returned rather than inlined so a test can pin the sampling instead of
    inferring it from a PNG.
    """

    cases = sorted(str(c) for c in patches["case_id"].unique())
    if len(cases) <= limit:
        return cases
    positions = np.unique(np.round(np.linspace(0, len(cases) - 1, limit)).astype(int))
    return [cases[int(p)] for p in positions]


def figure_05_strength_energy_utility(tables: MergedTables, out_dir: Path) -> Path:
    """Edge strength vs energy vs causal utility -- the "energy is not utility" panel.

    Left and middle are pooled per-patch scatters over a bounded case sample;
    right is the case-clustered Spearman agreement between the three rankings,
    which is the cohort-level version of the same question.
    """

    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / "05_strength_energy_utility.png"
    patches = tables.patches
    if patches.empty or "screened" not in patches.columns:
        return _placeholder(out_path, "No patch rows for this cohort.")
    screened = patches[patches["screened"].astype(bool)]
    if screened.empty:
        return _placeholder(
            out_path, "No screened patches: no measured utility to compare against."
        )
    sample = _scatter_sample_cases(screened)
    block = screened[screened["case_id"].isin(sample)]

    fig, axes = plt.subplots(1, 3, figsize=(15, 4.5))

    axes[0].scatter(
        block["edge_strength_median"], block["energy_share_roi"], s=6, alpha=0.35
    )
    axes[0].set_xlabel("edge strength (median, robust-normalised)")
    axes[0].set_ylabel("energy share of ROI")
    axes[0].set_title("strength vs energy")

    axes[1].scatter(
        block["energy_share_roi"],
        block["fractional_recovery_dice"],
        s=6,
        alpha=0.35,
        color="tab:orange",
    )
    axes[1].axhline(0.0, color="k", linewidth=0.8, alpha=0.5)
    axes[1].set_xlabel("energy share of ROI")
    axes[1].set_ylabel("fractional recovery dice (causal utility)")
    axes[1].set_title("energy vs utility")

    agreement = rank_agreement_summary(tables.summary)
    if agreement.empty:
        axes[2].text(
            0.5,
            0.5,
            "No Spearman agreement rows",
            ha="center",
            va="center",
            transform=axes[2].transAxes,
        )
        axes[2].set_axis_off()
    else:
        pooled = (
            agreement.groupby("pair", observed=True)[["median", "ci_low", "ci_high"]]
            .median()
            .reset_index()
        )
        y = np.arange(len(pooled))
        lower = (pooled["median"] - pooled["ci_low"]).clip(lower=0).to_numpy()
        upper = (pooled["ci_high"] - pooled["median"]).clip(lower=0).to_numpy()
        axes[2].barh(
            y,
            pooled["median"],
            xerr=np.vstack([lower, upper]),
            color="tab:green",
            alpha=0.8,
        )
        axes[2].set_yticks(y)
        axes[2].set_yticklabels(pooled["pair"])
        axes[2].axvline(0.0, color="k", linewidth=0.8, alpha=0.5)
        axes[2].set_xlabel("Spearman rho (case-clustered median, 95% CI)")
        axes[2].set_title("ranking agreement")

    _suptitle(
        fig,
        "Edge strength, perturbation energy and causal utility "
        f"(scatters pooled over {len(sample)} sampled cases; "
        "pooled scatters over the full cohort would be dominated "
        "by the cases with the most patches)",
    )
    fig.tight_layout()
    fig.savefig(out_path, dpi=FIGURE_DPI, bbox_inches="tight")
    plt.close(fig)
    return out_path


def _category_bar_spec(
    summary: pd.DataFrame, dataset: str
) -> tuple[pd.DataFrame, list[str], list[str]]:
    """Bars, colours and annotations for one dataset's category panel.

    Split out so the "a thin category is kept and flagged" rule is asserted
    directly rather than inferred from a PNG's byte size: a figure that
    silently dropped the category would still produce a plausible file.
    """

    block = (
        summary[summary["dataset"] == dataset]
        .groupby("anatomical_category", observed=True)[
            ["n_cases", "n_patches", "median", "ci_low", "ci_high"]
        ]
        .median()
        .reset_index()
        .sort_values("anatomical_category")
        .reset_index(drop=True)
    )
    colors: list[str] = []
    labels: list[str] = []
    for _, row in block.iterrows():
        thin = float(row["n_cases"]) < LOW_N_CASES
        colors.append("0.7" if thin else "tab:blue")
        label = f"n={int(row['n_cases'])}c/{int(row['n_patches'])}p"
        if thin:
            label += f"  n < {LOW_N_CASES}"
        labels.append(label)
    return block, colors, labels


def figure_06_anatomical_categories(tables: MergedTables, out_dir: Path) -> Path:
    """Causal utility per anatomical category, with thin categories kept and flagged."""

    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / "06_anatomical_categories.png"
    if tables.patches.empty:
        return _placeholder(out_path, "No patch rows for this cohort.")
    summary = anatomical_category_summary(tables.patches)
    if summary.empty:
        return _placeholder(
            out_path, "No screened patches with a finite utility per category."
        )

    datasets = sorted(str(d) for d in summary["dataset"].unique())
    fig, axes = plt.subplots(
        1, len(datasets), figsize=(6 * len(datasets), 4.5), squeeze=False
    )
    for ax, dataset in zip(axes[0], datasets, strict=True):
        block, colors, labels = _category_bar_spec(summary, dataset)
        x = np.arange(len(block))
        lower = (block["median"] - block["ci_low"]).clip(lower=0).to_numpy()
        upper = (block["ci_high"] - block["median"]).clip(lower=0).to_numpy()
        ax.bar(
            x,
            block["median"].to_numpy(),
            yerr=np.vstack([lower, upper]),
            color=colors,
            alpha=0.9,
        )
        ax.set_xticks(x)
        ax.set_xticklabels(block["anatomical_category"], rotation=30, ha="right")
        ax.axhline(0.0, color="k", linewidth=0.8, alpha=0.5)
        # Anchored above the CI cap: a vertical label rising from the bar's
        # foot ran straight through the error bar drawn at the same x.
        tops = np.maximum(block["ci_high"].to_numpy(), block["median"].to_numpy())
        for xi, top, label in zip(x, tops, labels):
            ax.annotate(
                label,
                (float(xi), float(top)),
                textcoords="offset points",
                xytext=(0, 3),
                ha="center",
                va="bottom",
                fontsize=6,
            )
        ax.set_ylabel("fractional recovery dice (median, 95% CI)")
        ax.set_title(f"{dataset}")
    _suptitle(
        fig,
        "Causal utility by anatomical category "
        f"(grey bars: fewer than {LOW_N_CASES} cases, kept and labelled)",
    )
    fig.tight_layout()
    fig.savefig(out_path, dpi=FIGURE_DPI, bbox_inches="tight")
    plt.close(fig)
    return out_path


def figure_07_attack_budget_comparison(tables: MergedTables, out_dir: Path) -> Path:
    """Concentration and utility across the epsilon ladder, one line per attack.

    At large epsilon the attack saturates and the energy distribution flattens;
    this figure is where that shows up, so the metrics are plotted against the
    full ladder rather than at the primary epsilon alone.
    """

    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / "07_attack_budget_comparison.png"
    concentration = cohort_concentration(tables.summary)
    if concentration.empty:
        return _placeholder(out_path, "No concentration rows for this cohort.")
    present = [m for m in BUDGET_METRICS if m in set(concentration["metric"])]
    if not present:
        return _placeholder(
            out_path, "None of the attack-budget metrics are present in the tables."
        )

    datasets = sorted(str(d) for d in concentration["dataset"].unique())
    fig, axes = plt.subplots(
        len(datasets),
        len(present),
        figsize=(4.2 * len(present), 3.4 * len(datasets)),
        squeeze=False,
    )
    attacks = sorted(str(a) for a in concentration["attack"].unique())
    cmap = plt.get_cmap("tab10")
    attack_colors = {a: cmap(i % 10) for i, a in enumerate(attacks)}

    for row, dataset in enumerate(datasets):
        for col, metric in enumerate(present):
            ax = axes[row][col]
            block = concentration[
                (concentration["dataset"] == dataset)
                & (concentration["metric"] == metric)
            ]
            for attack in attacks:
                sub = (
                    block[block["attack"] == attack]
                    .groupby("epsilon_n", observed=True)[
                        ["median", "ci_low", "ci_high"]
                    ]
                    .median()
                    .reset_index()
                    .sort_values("epsilon_n")
                )
                if sub.empty:
                    continue
                ax.plot(
                    sub["epsilon_n"],
                    sub["median"],
                    marker="o",
                    color=attack_colors[attack],
                    label=attack,
                )
                ax.fill_between(
                    sub["epsilon_n"],
                    sub["ci_low"],
                    sub["ci_high"],
                    color=attack_colors[attack],
                    alpha=0.12,
                )
            ax.set_xlabel("epsilon_n (/255)")
            ax.set_ylabel(metric)
            if row == 0:
                ax.set_title(metric)
            if col == 0:
                ax.set_ylabel(f"{dataset}\n{metric}")
    handles, labels = axes[0][0].get_legend_handles_labels()
    if handles:
        fig.legend(handles, labels, loc="lower center", ncol=min(len(labels), 5))
    _suptitle(
        fig,
        "Concentration and utility across the attack budget "
        "(flattening at large epsilon is saturation, not a null result)",
    )
    fig.tight_layout(rect=(0.0, 0.06, 1.0, 1.0))
    fig.savefig(out_path, dpi=FIGURE_DPI, bbox_inches="tight")
    plt.close(fig)
    return out_path


# --- Task 8: the generated report ------------------------------------------

# Carried in a module-level constant so an edit to the report body cannot
# quietly drop one: every limitation below is asserted present by a test.
LIMITATIONS: tuple[str, ...] = (
    "The unit of inference is the case. Patches, epsilon values and attacks "
    "within a case are repeated measures, so every cohort number here is a "
    "case-clustered bootstrap median, never a pooled per-patch average.",
    "RMS reproduction drift exceeded the 1e-2 flag on a minority of cases, "
    "concentrated in whole-gland. The rate is reported below rather than a "
    "reassuring maximum; APGD's sign() steps make a ~1e-7 reduction-order "
    "difference flip a voxel by a full 2*alpha and compound over 20 steps.",
    "The matched-control test is conditional on drawability. A control region "
    "must match its patch in volume, energy and signed-distance band, and "
    "draw success falls with patch volume, so the endpoint speaks to the "
    "drawable matched set rather than to every patch in the atlas. "
    "control_drawable_fraction records the narrowing per condition.",
    "The outer_boundary_gt atlas is structurally degenerate for the matched "
    "control: its patches are the signed-distance band, so a random region "
    "at the boundary that is not the boundary has no referent. It yields NaN "
    "by design and the band was deliberately not widened to make a number "
    "appear.",
    "Out of scope by the spec: the campaign attacks beyond FGSM/PGD/APGD, "
    "the fold-0-4 checkpoints, Shapley approximation, the random-label model "
    "control, edge-normal manipulation, surface Dice and the mixed model.",
)

RMS_FLAG = 1e-2


def _fmt_ci(row: pd.Series, value: str = "median") -> str:
    """``median [low, high]`` with a consistent precision."""

    return (
        f"{float(row[value]):.4g} "
        f"[{float(row['ci_low']):.4g}, {float(row['ci_high']):.4g}]"
    )


def _report_provenance(tables: MergedTables, merged_dir: Path | None) -> list[str]:
    """What was run: read from the tables and manifest, never hardcoded."""

    summary = tables.summary
    lines = ["## What was run", ""]
    lines.append(
        f"- Cases: {summary['case_id'].nunique()} across "
        f"{summary['dataset'].nunique()} dataset(s) "
        f"({', '.join(sorted(str(d) for d in summary['dataset'].unique()))})"
    )
    lines.append(
        f"- Attacks: {', '.join(sorted(str(a) for a in summary['attack'].unique()))}"
    )
    lines.append(
        "- Epsilon grid (/255): "
        f"{', '.join(str(int(e)) for e in sorted(summary['epsilon_n'].unique()))}"
    )
    lines.append(
        f"- Atlases: {', '.join(sorted(str(a) for a in summary['atlas_type'].unique()))}"
    )
    for stage, block in summary.groupby("stage", observed=True):
        lines.append(f"- Stage `{stage}`: {block['case_id'].nunique()} case(s)")
    manifest = (merged_dir / "run_manifest.json") if merged_dir else None
    if manifest is not None and manifest.is_file():
        settings = json.loads(manifest.read_text())
        for key in ("seed", "attack_steps", "ig_steps", "atlas_config"):
            if key in settings:
                lines.append(f"- `{key}`: {settings[key]}")
    lines.append("")
    return lines


def _report_sanity(tables: MergedTables) -> list[str]:
    """Conservation, IG completeness, RMS drift RATE, and the qc flag counts."""

    summary = tables.summary
    lines = ["## Sanity", ""]
    for column, budget in (
        ("conservation_error", "1e-6"),
        ("ig_completeness_error", "5e-2"),
    ):
        if column in summary.columns:
            values = summary[column].dropna()
            if not values.empty:
                lines.append(
                    f"- `{column}`: {values.min():.2e} to {values.max():.2e} "
                    f"(budget {budget})"
                )
    if "rms_reproduction_rel_error" in summary.columns:
        drift = summary["rms_reproduction_rel_error"].dropna()
        if not drift.empty:
            flagged = int((drift > RMS_FLAG).sum())
            pct = 100.0 * flagged / len(drift)
            lines.append(
                f"- RMS reproduction drift above the {RMS_FLAG:.0e} flag: "
                f"{flagged}/{len(drift)} rows ({pct:.1f}%), max {drift.max():.3e}. "
                "Reported as a rate, not a maximum: a single worst case would "
                "understate how often this fires."
            )
    if not tables.qc.empty and "flags" in tables.qc.columns:
        counts: dict[str, int] = {}
        for entry in tables.qc["flags"].astype(str):
            for flag in entry.split(";"):
                if flag and flag != "nan":
                    counts[flag] = counts.get(flag, 0) + 1
        if counts:
            lines.append("- QC flags:")
            for flag, count in sorted(counts.items(), key=lambda kv: -kv[1]):
                lines.append(f"    - `{flag}`: {count}")
    lines.append("")
    return lines


def _drawable_ratio(row: pd.Series) -> str:
    """``M/N`` as a fraction, or ``n/a`` when either pool size is missing."""

    observed = float(row["observed_pool_median"])
    control = float(row["control_pool_median"])
    if not np.isfinite(observed) or not np.isfinite(control) or observed <= 0:
        return "n/a"
    return f"{control / observed:.2f}"


def _report_controls(tables: MergedTables) -> list[str]:
    """Observed vs matched-random recovery, with the pool-symmetry caveat."""

    lines = ["## Matched controls", ""]
    controls = cohort_controls(tables.summary, tables.controls)
    if controls.empty:
        lines.append("No interaction-stage conditions in these tables.")
        lines.append("")
        return lines
    lines.append(
        "| dataset | eps | attack | atlas | n | observed | control | p (median) | "
        "screened N | drawable M | M/N |"
    )
    lines.append("| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |")
    for _, row in controls.iterrows():
        lines.append(
            f"| {row['dataset']} | {int(row['epsilon_n'])} | {row['attack']} | "
            f"{row['atlas_type']} | {int(row['n_cases'])} | "
            f"{float(row['observed_median']):.4g} | "
            f"{float(row['control_median']):.4g} | "
            f"{float(row['permutation_p_median']):.3g} | "
            f"{float(row['observed_pool_median']):.0f} | "
            f"{float(row['control_pool_median']):.0f} | "
            f"{_drawable_ratio(row)} |"
        )
    lines.append("")
    lines.append(
        "Both arms select the top q of the SAME M drawable patches -- the "
        "observed arm from the patches themselves, the control arm from their "
        "volume-, energy- and band-matched twins -- so the max-of-noise "
        "selection advantage cancels by construction and M < N is not a "
        "residual asymmetry."
    )
    lines.append("")
    lines.append(
        "M/N is the CONDITIONING caveat instead: a control region must fit its "
        "patch's signed-distance band, and draw success falls with patch "
        "volume, so the endpoint speaks to the drawable matched set rather "
        "than to every patch in the atlas. Read every p above as conditional "
        "on drawability."
    )
    thin = controls[
        controls["control_pool_median"].astype(float)
        < 0.25 * controls["observed_pool_median"].astype(float)
    ]
    if not thin.empty:
        lines.append("")
        lines.append(
            f"**Thin drawable pool: {len(thin)} condition(s) draw fewer than a "
            "quarter of their patches.** An atlas whose patches ARE the "
            "signed-distance band "
            "(outer_boundary_gt) has almost no candidate space left after the "
            "tubes are excluded: a random region at the boundary that is not "
            "the boundary has no referent. Those conditions are structurally "
            "degenerate for this test and their p, where one exists at all, "
            "rests on a handful of draws."
        )
    lines.append("")
    return lines


def write_report(
    tables: MergedTables,
    out_dir: Path,
    figures: dict[str, Path],
    *,
    merged_dir: Path | None = None,
) -> Path:
    """Generate ``EDGE_ATTRIBUTION_REPORT.md`` from the tables.

    Generated rather than hand-written so it cannot drift from the numbers it
    describes.
    """

    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / "EDGE_ATTRIBUTION_REPORT.md"
    lines: list[str] = [
        "# Edge-specific energy-utility attribution",
        "",
        "Generated by `experiments/plot_edge_attribution.py` from the merged "
        "tables. Do not edit by hand: rerun the script.",
        "",
    ]
    lines += _report_provenance(tables, merged_dir)
    lines += _report_sanity(tables)

    concentration = cohort_concentration(tables.summary)
    lines += ["## Concentration", ""]
    if concentration.empty:
        lines.append("No concentration metrics available.")
    else:
        lines.append("| dataset | eps | attack | atlas | metric | median [95% CI] | n |")
        lines.append("| --- | --- | --- | --- | --- | --- | --- |")
        for _, row in concentration.iterrows():
            lines.append(
                f"| {row['dataset']} | {int(row['epsilon_n'])} | {row['attack']} | "
                f"{row['atlas_type']} | `{row['metric']}` | {_fmt_ci(row)} | "
                f"{int(row['n_cases'])} |"
            )
    lines.append("")

    lines += ["## Energy is not utility", ""]
    k50 = concentration[concentration["metric"] == "k50_remove"]
    if k50.empty:
        lines.append(
            "`k50_remove` is not present in these tables, so the primary "
            "comparison cannot be made."
        )
    else:
        lines.append(
            "`k50_remove` is the number of patches whose removal recovers half "
            "the adversarial damage under the utility ranking. Compare it "
            "against the energy ranking in `04_cumulative_curves.png`: if the "
            "utility curve rises faster, perturbation energy and causal "
            "utility are different quantities."
        )
        lines.append("")
        for _, row in k50.iterrows():
            lines.append(
                f"- {row['dataset']} / {row['atlas_type']} / {row['attack']} "
                f"eps={int(row['epsilon_n'])}: K50 = {_fmt_ci(row)} "
                f"(n={int(row['n_cases'])} cases)"
            )
    lines.append("")

    categories = anatomical_category_summary(tables.patches)
    lines += ["## Anatomy", ""]
    if categories.empty:
        lines.append("No screened patches with a finite utility per category.")
    else:
        lines.append(
            "| dataset | eps | attack | atlas | category | utility [95% CI] | "
            "cases | patches |"
        )
        lines.append("| --- | --- | --- | --- | --- | --- | --- | --- |")
        for _, row in categories.iterrows():
            lines.append(
                f"| {row['dataset']} | {int(row['epsilon_n'])} | {row['attack']} | "
                f"{row['atlas_type']} | {row['anatomical_category']} | "
                f"{_fmt_ci(row)} | {int(row['n_cases'])} | {int(row['n_patches'])} |"
            )
    lines.append("")

    lines += _report_controls(tables)

    if figures:
        lines += ["## Figures", ""]
        for name in sorted(figures):
            lines.append(f"- `{figures[name].name}` ({name})")
        lines.append("")

    lines += ["## Limitations", ""]
    for limitation in LIMITATIONS:
        lines.append(f"- {limitation}")
    lines.append("")

    out_path.write_text("\n".join(lines), encoding="utf-8")
    return out_path


FIGURE_BUILDERS = {
    "atlas_qc": figure_01_atlas_qc_panel,
    "patch_energy": figure_02_patch_energy_maps,
    "patch_utility": figure_03_patch_utility_maps,
    "cumulative_curves": figure_04_cumulative_curves,
    "strength_energy_utility": figure_05_strength_energy_utility,
    "anatomical_categories": figure_06_anatomical_categories,
    "attack_budget": figure_07_attack_budget_comparison,
}


def write_figures(tables: MergedTables, out_dir: Path) -> dict[str, Path]:
    """Render every figure, returning ``{name: path}``."""

    return {name: fn(tables, out_dir) for name, fn in FIGURE_BUILDERS.items()}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Cohort summaries, figures and report for the edge-attribution tables"
        )
    )
    parser.add_argument("--merged-dir", type=Path, default=DEFAULT_MERGED_DIR)
    parser.add_argument("--output-dir", type=Path, default=None)
    parser.add_argument(
        "--figures", action=argparse.BooleanOptionalAction, default=True
    )
    parser.add_argument("--report", action=argparse.BooleanOptionalAction, default=True)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    merged_dir = args.merged_dir.resolve()
    output_dir = (
        args.output_dir if args.output_dir is not None else merged_dir / "analysis"
    ).resolve()

    tables = load_tables(merged_dir)
    written = write_cohort_csvs(tables, output_dir)
    figures: dict[str, Path] = {}
    if args.figures:
        figures = write_figures(tables, output_dir)
    if args.report:
        written["report"] = write_report(
            tables, output_dir, figures, merged_dir=merged_dir
        )

    for path in list(written.values()) + list(figures.values()):
        print(path)


if __name__ == "__main__":
    main()
