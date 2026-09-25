"""Re-test the edge hypothesis on boundary metrics (HD95, ASD).

The Phase E cohort tables pool Dice only.  Dice is volume-weighted, so an edge
perturbation that displaces the contour by a millimetre or two barely moves it
while dominating HD95 / ASD.  The driver scored both on every intervention it
ran; this script re-scores those stored values -- no re-inference -- and
writes the result beside the Phase E ``analysis/`` directory.

Reads ``case_summary.csv``, ``subset_curves.parquet`` and
``patch_metrics.parquet`` from the merged directory; writes
``analysis_boundary/{boundary_recovery.csv, boundary_patch_necessity.csv,
BOUNDARY_RETEST_REPORT.md}``.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[1]
for root in (REPO_ROOT, REPO_ROOT / "src"):
    if str(root) not in sys.path:
        sys.path.insert(0, str(root))

from experiments.plot_edge_attribution import (  # noqa: E402
    DEFAULT_MERGED_DIR,
    load_tables,
)
from mri_prostate_seg.experiments.edge_attribution.aggregate import (  # noqa: E402
    BOUNDARY_DAMAGE_FLOOR_MM,
    cohort_boundary_recovery,
    patch_boundary_necessity,
)

REPORT_EPS = (16, 32)
REPORT_FRACTION = 0.10
# Pre-registered escalation rule (agreed before the tables were computed):
# re-inference with an HD95-ranked oracle and matched controls is warranted
# only if boundary recovery exceeds Dice recovery by more than this, in at
# least this many of the eps 16/32 conditions, or if the model-free strength
# ranking beats the IG ranking on HD95.
ESCALATION_EXCESS = 0.10
ESCALATION_MIN_CONDITIONS = 2


def _fmt(value: float, digits: int = 3) -> str:
    return "—" if not np.isfinite(value) else f"{value:.{digits}f}"


def _at_report_point(recovery: pd.DataFrame, ranking: str) -> pd.DataFrame:
    return recovery[
        (recovery["ranking_type"] == ranking)
        & (recovery["top_fraction"] == REPORT_FRACTION)
        & recovery["epsilon_n"].isin(REPORT_EPS)
    ]


def _recovery_table(recovery: pd.DataFrame, ranking: str) -> list[str]:
    sel = _at_report_point(recovery, ranking)
    lines = [
        "| Dataset | Atlas | Attack | ε | Dice rec. (n) | HD95 rec. (n) | HD95 mm (n) | HD95 kept (n) | ASD rec. (n) | ASD mm (n) |",
        "| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |",
    ]
    for _, r in sel.iterrows():
        lines.append(
            f"| {r.dataset} | {r.atlas_type} | {r.attack} | {int(r.epsilon_n)} "
            f"| {_fmt(r.dice_recovery_fraction_median)} ({int(r.n_cases_dice_recovery_fraction)}) "
            f"| {_fmt(r.hd95_recovery_fraction_median)} ({int(r.n_cases_hd95_recovery_fraction)}) "
            f"| {_fmt(r.hd95_recovery_mm_median, 2)} ({int(r.n_cases_hd95_recovery_mm)}) "
            f"| {_fmt(r.hd95_kept_fraction_median)} ({int(r.n_cases_hd95_kept_fraction)}) "
            f"| {_fmt(r.asd_recovery_fraction_median)} ({int(r.n_cases_asd_recovery_fraction)}) "
            f"| {_fmt(r.asd_recovery_mm_median, 2)} ({int(r.n_cases_asd_recovery_mm)}) |"
        )
    return lines


def _condition(r: pd.Series) -> str:
    return f"{r.dataset} {r.atlas_type} {r.attack} ε{int(r.epsilon_n)}"


def verdict(recovery: pd.DataFrame) -> list[str]:
    """Evaluate the pre-registered escalation rule from the table itself."""

    lines: list[str] = []
    triggered = False
    for ranking in ("utility", "ig"):
        sel = _at_report_point(recovery, ranking).copy()
        sel["excess"] = sel["hd95_recovery_fraction_median"] - sel["dice_recovery_fraction_median"]
        sel = sel[np.isfinite(sel["excess"])]
        hits = sel[sel["excess"] > ESCALATION_EXCESS]
        worst = sel.loc[sel["excess"].idxmax()]
        lines.append(
            f"- {ranking}: HD95 recovery exceeds Dice recovery by > {ESCALATION_EXCESS:.0%} "
            f"in **{len(hits)} of {len(sel)}** conditions; largest excess "
            f"{worst.excess:+.3f} at {_condition(worst)} "
            f"(n = {int(worst.n_cases_hd95_recovery_fraction)}, "
            f"{worst.hd95_recovery_mm_median:.2f} mm)."
        )
        triggered |= len(hits) >= ESCALATION_MIN_CONDITIONS
    keys = ["dataset", "atlas_type", "attack", "epsilon_n"]
    strength = _at_report_point(recovery, "strength").set_index(keys)
    ig = _at_report_point(recovery, "ig").set_index(keys)
    diff = (
        strength["hd95_recovery_fraction_median"] - ig["hd95_recovery_fraction_median"]
    ).dropna()
    beats = diff[diff > 0]
    lines.append(
        f"- strength vs IG on HD95: strength-ranked removal recovers more in "
        f"**{len(beats)} of {len(diff)}** conditions "
        f"(largest margin {diff.max():+.3f})."
    )
    triggered |= len(beats) > 0
    mm = _at_report_point(recovery, "utility")
    lines.append(
        f"- absolute: oracle top-{REPORT_FRACTION:.0%} removal returns a median "
        f"{mm.hd95_recovery_mm_median.min():.2f}–{mm.hd95_recovery_mm_median.max():.2f} mm "
        f"of HD95 and {mm.asd_recovery_mm_median.min():.2f}–"
        f"{mm.asd_recovery_mm_median.max():.2f} mm of ASD."
    )
    lines.append("")
    lines.append(
        "**Escalation rule "
        + (
            "TRIGGERED — re-inference with an HD95-ranked oracle and matched "
            "controls is warranted.**"
            if triggered
            else "NOT triggered — the edge hypothesis does not survive on "
            "boundary metrics either; no re-inference.**"
        )
    )
    return lines


def _necessity_table(necessity: pd.DataFrame) -> list[str]:
    sel = necessity[necessity["epsilon_n"].isin(REPORT_EPS)]
    lines = [
        "| Dataset | Atlas | Attack | ε | patches/case | HD95 necessity / damage (n) | ρ(Dice, HD95) (n) | ASD necessity / damage (n) | ρ(Dice, ASD) (n) |",
        "| --- | --- | --- | --- | --- | --- | --- | --- | --- |",
    ]
    for _, r in sel.iterrows():
        lines.append(
            f"| {r.dataset} | {r.atlas_type} | {r.attack} | {int(r.epsilon_n)} | {r.n_patches_per_case:g} "
            f"| {_fmt(r.hd95_necessity_fraction_median)} ({int(r.n_cases_hd95_necessity_fraction)}) "
            f"| {_fmt(r.spearman_dice_hd95_median, 2)} ({int(r.n_cases_spearman_dice_hd95)}) "
            f"| {_fmt(r.asd_necessity_fraction_median)} ({int(r.n_cases_asd_necessity_fraction)}) "
            f"| {_fmt(r.spearman_dice_asd_median, 2)} ({int(r.n_cases_spearman_dice_asd)}) |"
        )
    return lines


def write_report(recovery: pd.DataFrame, necessity: pd.DataFrame, path: Path) -> None:
    lines = [
        "# Boundary-metric re-test of the edge hypothesis",
        "",
        "Re-scored from the HD95 / ASD values the driver stored for every evaluated "
        "intervention (`subset_curves.hd95_removed`, `hd95_kept`, `asd_*`; "
        "`patch_metrics.necessity_hd95`, `necessity_asd`).  No re-inference.  "
        "Recovery fraction = (full damage − damage after removal) / full damage; "
        "kept fraction = damage retained when only those patches keep their "
        "perturbation.  Fractions are reported only for cases with "
        f"full boundary damage ≥ {BOUNDARY_DAMAGE_FLOOR_MM:g} mm; the "
        "mm columns include every case, so where the two n differ (FGSM, whose "
        "boundary damage is often sub-millimetre) the fraction and the mm median "
        "describe different subsets of the 100 causal cases.  Every cell carries "
        "its n.  Case-clustered bootstrap medians, 2,000 resamples.  Zones "
        "HD95/ASD are on the foreground union (outer gland contour), so the "
        "TZ/PZ interface is not measured.  The energy ranking is in "
        "`boundary_recovery.csv` but not tabulated here (it is undefined for FGSM).",
        "",
        "## Verdict",
        "",
        f"Pre-registered rule: escalate to re-inference if HD95 recovery exceeds Dice "
        f"recovery by > {ESCALATION_EXCESS:.0%} at the top {REPORT_FRACTION:.0%} in ≥ "
        f"{ESCALATION_MIN_CONDITIONS} ε16/32 conditions (utility or IG ranking), or if the "
        "strength ranking beats IG on HD95 anywhere.",
        "",
        *verdict(recovery),
        "",
        f"## Top {REPORT_FRACTION:.0%} of edge patches by *image edge strength* (model-free)",
        "",
        "The hypothesis in its purest form: remove the perturbation on the "
        "strongest anatomical edges and ask whether the contour comes back.",
        "",
        *_recovery_table(recovery, "strength"),
        "",
        f"## Top {REPORT_FRACTION:.0%} by *measured Dice utility* (oracle)",
        "",
        *_recovery_table(recovery, "utility"),
        "",
        f"## Top {REPORT_FRACTION:.0%} by *Integrated Gradients*",
        "",
        *_recovery_table(recovery, "ig"),
        "",
        "## Single-patch boundary necessity (screened patches)",
        "",
        "Median per-patch share of the full boundary damage undone by removing "
        "that patch alone, and the within-case Spearman between a patch's Dice "
        "necessity and its boundary necessity.  The driver scored boundary "
        "metrics only for the 10 highest-Dice-utility patches per case "
        "(`SURFACE_TOP`), so ρ is a rank agreement within that top 10.",
        "",
        *_necessity_table(necessity),
        "",
        "## Limitations",
        "",
        "- No `k = every edge patch` point: boundary metrics were scored only at "
        "the top 1/5/10/20 % of screened patches.",
        "- Matched controls, greedy and pairwise stages have no boundary scores; "
        "a boundary-metric null needs re-inference.",
        "- The utility and IG rankings were built on Dice; the strength ranking is "
        "the only one independent of it.",
        f"- The {BOUNDARY_DAMAGE_FLOOR_MM:g} mm floor is strict for ASD (a mean distance): "
        "FGSM ASD fractions rest on few cases.  Slightly negative ASD medians there "
        "are sub-voxel noise, not a real worsening.",
    ]
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--merged-dir", type=Path, default=DEFAULT_MERGED_DIR)
    parser.add_argument("--output-dir", type=Path, default=None)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    merged_dir = args.merged_dir
    out_dir = args.output_dir or merged_dir / "analysis_boundary"
    out_dir.mkdir(parents=True, exist_ok=True)
    tables = load_tables(merged_dir)
    recovery = cohort_boundary_recovery(tables.summary, tables.curves)
    necessity = patch_boundary_necessity(tables.summary, tables.patches)
    recovery.to_csv(out_dir / "boundary_recovery.csv", index=False)
    necessity.to_csv(out_dir / "boundary_patch_necessity.csv", index=False)
    write_report(recovery, necessity, out_dir / "BOUNDARY_RETEST_REPORT.md")
    print(f"wrote {len(recovery)} recovery rows, {len(necessity)} necessity rows -> {out_dir}")


if __name__ == "__main__":
    main()
