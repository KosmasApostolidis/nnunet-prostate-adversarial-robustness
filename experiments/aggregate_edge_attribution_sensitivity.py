"""Phase B: aggregate the sensitivity sweep into a stability report.

Reads every ``<parameter>_<value>/`` directory under the sweep root (plus the
shared ``baseline/`` directory, which stands in for the baseline value of all
four parameters at once -- see ``scripts/run_edge_attribution_sensitivity.sh``),
builds one ``SweepPoint`` per (parameter, value) and reports how much ``K50``
and the top-10% utility patch set move across settings.

Only the ``native_edges`` atlas is used: ``outer_boundary_gt`` and
``zone_interface_gt`` are ground-truth-derived and do not depend on
``edge_quantile``, ``patch_extent_mm`` or ``tube_radius_mm`` at all, and their
patch numbering has a different count by construction, so mixing them in would
confound the comparison this sweep exists to make.

The choice of primary setting is made on stability -- does K50 stop moving,
does the top-10% patch set stop changing -- never on the size of any damage,
enrichment or recovery quantity. This module must not rank settings by any
such quantity.
"""

from __future__ import annotations

import argparse
import math
import sys
from pathlib import Path

import numpy as np
import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))
SRC_ROOT = REPO_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from experiments.evaluate_edge_attribution_all_cases import (  # noqa: E402
    atlas_path,
    DEFAULT_RESULTS,
    PATCH_CSV,
    SUMMARY_CSV,
)
from mri_prostate_seg.experiments.edge_attribution import (  # noqa: E402
    SweepPoint,
    check_summary_schema,
    choose_primary,
    k_stability,
    median_ci,
    stability_table,
)

DEFAULT_SWEEP_ROOT = DEFAULT_RESULTS / "all_cases_edge_attribution_sensitivity"

# native_edges is the only atlas that these four settings actually construct;
# the anatomical atlases (outer_boundary_gt, zone_interface_gt) are fixed by
# the ground truth and would dilute the comparison with an unrelated, constant
# patch set.
NATIVE_EDGES_ATLAS = "native_edges"
TOP_FRACTION = 0.10

# The pilot's fixed baseline configuration (matches the driver's own defaults).
BASELINE_VALUES: dict[str, float] = {
    "edge_quantile": 0.90,
    "patch_extent_mm": 6.0,
    "tube_radius_mm": 2.0,
    "ig_steps": 32.0,
}
PARAMETERS: tuple[str, ...] = tuple(BASELINE_VALUES)
BASELINE_DIR_NAME = "baseline"


def _format_value(value: float) -> str:
    return f"{value:g}"


def _setting_name(parameter: str, value: float) -> str:
    return f"{parameter}={_format_value(value)}"


def _case_key(dataset: object, case_id: object) -> str:
    """A case key unique across datasets: the sweep runs wg and zones together.

    ``case_id`` alone is not guaranteed unique across datasets, and a
    collision would silently merge two different cases' K50 values or two
    different patch populations (whose ``patch_id``s can coincide) under one
    key. Keying on ``(dataset, case_id)`` removes the question rather than
    relying on unverified ID uniqueness.
    """

    return f"{dataset}/{case_id}"


def load_case_summary_k50(directory: Path) -> dict[str, float]:
    """``k50_remove`` per case, ``native_edges`` atlas only."""

    path = directory / SUMMARY_CSV
    frame = pd.read_csv(path)
    check_summary_schema(frame, str(path))
    native = frame[frame["atlas_type"] == NATIVE_EDGES_ATLAS]
    native = native.loc[np.isfinite(native["k50_remove"])]
    return {
        _case_key(dataset, case_id): float(value)
        for dataset, case_id, value in zip(
            native["dataset"], native["case_id"], native["k50_remove"]
        )
    }


def load_patch_top_sets(directory: Path) -> dict[str, set[int]]:
    """Top-10% utility patch-id set per case, ``native_edges`` atlas only.

    Taken as the patches with the smallest ``utility_rank`` among ``screened``
    rows, ``ceil(0.10 * n_screened)`` of them.
    """

    path = directory / PATCH_CSV
    frame = pd.read_csv(path)
    native = frame[
        (frame["atlas_type"] == NATIVE_EDGES_ATLAS) & frame["screened"].astype(bool)
    ]
    out: dict[str, set[int]] = {}
    for (dataset, case_id), group in native.groupby(["dataset", "case_id"]):
        n_top = math.ceil(TOP_FRACTION * len(group))
        top = group.nsmallest(n_top, "utility_rank")
        out[_case_key(dataset, case_id)] = {int(pid) for pid in top["patch_id"]}
    return out


# Two of the four swept parameters re-partition the atlas, so a patch id in one
# setting names a different physical region than the same id in another and an
# id-set Jaccard is structurally near zero no matter how stable the answer is.
# Measured on the real sweep: edge_quantile takes a case from 261 to 441/142
# patches and patch_extent_mm from 261 to 558/192, and both scored 0.00-0.05,
# while tube_radius_mm and ig_steps (which leave the partition alone at 261
# patches) scored 0.57-0.72 and 1.000.  For the re-partitioning parameters the
# comparison must therefore be made in VOXEL space, where it is
# partition-independent.
REPARTITIONING_PARAMETERS: frozenset[str] = frozenset(
    {"edge_quantile", "patch_extent_mm"}
)


def load_patch_top_voxels(directory: Path) -> dict[str, set[int]]:
    """Flat voxel indices covered by the top-10% utility tubes, per case.

    Reads the atlas cache each sweep directory already wrote, so this is a
    read rather than a re-run.  Returns flat indices so the existing
    set-Jaccard applies unchanged.
    """

    top_ids = load_patch_top_sets(directory)
    out: dict[str, set[int]] = {}
    for key, ids in top_ids.items():
        dataset, case_id = key.split("/", 1)
        path = atlas_path(directory, dataset, case_id)
        if not path.is_file() or not ids:
            continue
        with np.load(path) as data:
            name = f"tubes_{NATIVE_EDGES_ATLAS}"
            if name not in data:
                continue
            tubes = data[name]
        mask = np.isin(tubes, np.fromiter(ids, dtype=tubes.dtype, count=len(ids)))
        out[key] = {int(i) for i in np.flatnonzero(mask.reshape(-1))}
    return out


def build_sweep_point(directory: Path, parameter: str, value: float) -> SweepPoint:
    return SweepPoint(
        name=_setting_name(parameter, value),
        parameter=parameter,
        value=value,
        k50=load_case_summary_k50(directory),
        top_sets=(
            load_patch_top_voxels(directory)
            if parameter in REPARTITIONING_PARAMETERS
            else load_patch_top_sets(directory)
        ),
    )


def discover_settings(sweep_root: Path) -> list[SweepPoint]:
    """One ``SweepPoint`` per (parameter, value) sweep entry under the root.

    The shared ``baseline/`` directory is read once and turned into four
    points -- one per parameter, at that parameter's baseline value -- because
    it is the single run that all four sweeps hold in common.
    """

    points: list[SweepPoint] = []
    baseline_dir = sweep_root / BASELINE_DIR_NAME
    if baseline_dir.is_dir():
        for parameter, value in BASELINE_VALUES.items():
            points.append(build_sweep_point(baseline_dir, parameter, value))
    for directory in sorted(p for p in sweep_root.iterdir() if p.is_dir()):
        if directory.name == BASELINE_DIR_NAME:
            continue
        for parameter in PARAMETERS:
            prefix = f"{parameter}_"
            if directory.name.startswith(prefix):
                raw_value = directory.name[len(prefix) :]
                points.append(build_sweep_point(directory, parameter, float(raw_value)))
                break
    return points


def _neighbour_jaccards(
    table: pd.DataFrame, parameter: str, setting: str
) -> list[float]:
    """Jaccard of ``setting`` against every other setting of ``parameter``."""

    rows = table[table["parameter"] == parameter]
    return [
        float(row["jaccard_median"])
        for _, row in rows.iterrows()
        if setting in (row["setting_a"], row["setting_b"])
    ]


def render_report(
    points: list[SweepPoint], table: pd.DataFrame, *, min_jaccard: float
) -> str:
    lines = [
        "# Phase B Sensitivity Report",
        "",
        "**Selection criterion: stability.** A setting is chosen because K50 and "
        "the top-10% utility patch set stop moving across neighbouring settings "
        "of the same parameter -- never because it makes edges look more or less "
        f"important. A pair of settings counts as agreeing when their median "
        f"top-set Jaccard is at least {min_jaccard:g}.",
        "",
    ]
    for parameter in PARAMETERS:
        param_points = [p for p in points if p.parameter == parameter]
        # `table` has no columns at all when stability_table found zero pairs
        # anywhere (e.g. only baseline/ has run so far); guard the lookup
        # rather than let a bare column-index raise KeyError.
        param_table = (
            table[table["parameter"] == parameter]
            if "parameter" in table.columns
            else table.iloc[0:0]
        )
        lines.append(f"## {parameter}")
        lines.append("")
        if parameter in REPARTITIONING_PARAMETERS:
            lines.append(
                "*This parameter re-partitions the atlas, so patch ids are not "
                "comparable between its settings. Agreement is measured in "
                "VOXEL space -- the overlap of the union of the top-10% "
                "utility tubes -- which is partition-independent.*"
            )
        else:
            lines.append(
                "*This parameter leaves the patch partition unchanged, so "
                "agreement is measured directly on top-10% utility patch-id "
                "sets.*"
            )
        lines.append("")
        if param_table.empty:
            lines.append("No sweep data found for this parameter.")
            lines.append("")
            continue
        lines.append(
            "| setting_a | setting_b | n_cases | jaccard_median | k50_abs_difference_median |"
        )
        lines.append("|---|---|---|---|---|")
        for _, row in param_table.sort_values(["setting_a", "setting_b"]).iterrows():
            lines.append(
                f"| {row['setting_a']} | {row['setting_b']} | {int(row['n_cases'])} "
                f"| {row['jaccard_median']:.3f} | {row['k50_abs_difference_median']:.3f} |"
            )
        lines.append("")
        spread = k_stability({p.name: list(p.k50.values()) for p in param_points})
        lines.append(
            f"K50 spread across settings: median_k50={spread['median_k50']:.3f}, "
            f"max_abs_deviation={spread['max_abs_deviation']:.3f}."
        )
        chosen = choose_primary(table, parameter, min_jaccard=min_jaccard)
        if chosen is None:
            best_jaccard = (
                float(param_table["jaccard_median"].max())
                if not param_table["jaccard_median"].isna().all()
                else float("nan")
            )
            space = (
                "voxel" if parameter in REPARTITIONING_PARAMETERS else "patch-id"
            )
            lines.append(
                f"**No setting reached the minimum Jaccard agreement of "
                f"{min_jaccard:g}; {parameter} is unstable across the pilot "
                f"cohort** (best observed pairwise jaccard_median={best_jaccard:.3f} "
                f"in {space} space, K50 "
                f"max_abs_deviation={spread['max_abs_deviation']:.3f}). No "
                "primary value is selected by the stability criterion for this "
                "parameter; it falls back to the spec baseline, and that "
                "baseline is a default rather than a finding."
            )
        else:
            chosen_point = next(p for p in param_points if p.name == chosen)
            neighbour_jaccards = _neighbour_jaccards(table, parameter, chosen)
            finite_neighbours = [v for v in neighbour_jaccards if np.isfinite(v)]
            neighbour_jaccard = (
                float(np.median(finite_neighbours))
                if finite_neighbours
                else float("nan")
            )
            n_agreeing = sum(1 for v in finite_neighbours if v >= min_jaccard)
            low, med, high = median_ci(list(chosen_point.k50.values()))
            lines.append(
                f"**Chosen setting: `{chosen}`** -- agrees (jaccard_median >= "
                f"{min_jaccard:g}) with {n_agreeing} of {len(finite_neighbours)} "
                f"neighbours (median Jaccard against all neighbours = "
                f"{neighbour_jaccard:.3f}; K50 median = {med:.3f} "
                f"[{low:.3f}, {high:.3f}] 95% CI, n={len(chosen_point.k50)} cases)."
            )
        lines.append("")
    return "\n".join(lines) + "\n"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--sweep-root", type=Path, default=DEFAULT_SWEEP_ROOT)
    parser.add_argument("--output-dir", type=Path, default=None)
    parser.add_argument("--min-jaccard", type=float, default=0.6)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    sweep_root: Path = args.sweep_root
    output_dir: Path = args.output_dir if args.output_dir is not None else sweep_root

    points = discover_settings(sweep_root)
    if not points:
        raise SystemExit(f"No sweep settings found under {sweep_root}")

    table = stability_table(points)
    output_dir.mkdir(parents=True, exist_ok=True)
    summary_path = output_dir / "sensitivity_summary.csv"
    table.to_csv(summary_path, index=False)

    report = render_report(points, table, min_jaccard=args.min_jaccard)
    report_path = output_dir / "SENSITIVITY_REPORT.md"
    report_path.write_text(report)

    print(f"wrote {summary_path}")
    print(f"wrote {report_path}")


if __name__ == "__main__":
    main()
