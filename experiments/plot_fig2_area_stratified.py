"""Regenerate Figure 2 with an area-stratified anatomical panel.

Panels 1-3 are unchanged (foreground area vs Dice drop per class). Panel 4
replaces the GEE adjusted-effects plot with a non-parametric stratified
comparison: slices are split into within-class foreground-area quartiles and
the Base-Mid and Apex-Mid differences in mean Dice drop are computed inside
each quartile. Comparing regions at matched slice size separates a positional
effect from the intrinsic size sensitivity of Dice without fitting a model.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))

from mri_prostate_seg.experiments.slice_vulnerability.figures import (  # noqa: E402
    add_combined_wg_tz_pz_axes,
    plot_fg_size_vs_vulnerability_on_ax,
)

CLASS_COLORS = {"WG": "#1f77b4", "TZ+CZ": "#ff7f0e", "PZ": "#2ca02c"}
CLASSES = ("WG", "TZ+CZ", "PZ")
REGIONS = ("Base", "Mid", "Apex")
N_QUARTILES = 4
# Both cohorts are resampled to a uniform 0.5 x 0.5 mm in-plane grid (Sec. II-A),
# so one pixel is 0.25 mm^2. Panels 1-3 are plotted in mm^2 to match panel 4,
# which stratifies on gt_area_mm2.
MM2_PER_PIXEL = 0.5 * 0.5
AREA_UNIT = "mm$^2$"
N_BOOTSTRAP = 2000
EPSILON = 0.1
SEED = 42


def _cell_means(frame: pd.DataFrame) -> tuple[np.ndarray, list[str]]:
    """Per-patient mean Dice drop as [patient, quartile, region], NaN if empty."""
    patients = sorted(frame["patient_id"].unique())
    patient_index = {name: i for i, name in enumerate(patients)}
    totals = np.zeros((len(patients), N_QUARTILES, len(REGIONS)))
    counts = np.zeros_like(totals)
    for patient, quartile, region, drop in zip(
        frame["patient_id"],
        frame["area_quartile"],
        frame["anatomical_region"],
        frame["dice_drop"],
        strict=True,
    ):
        i = patient_index[patient]
        j = int(quartile)
        k = REGIONS.index(region)
        totals[i, j, k] += float(drop)
        counts[i, j, k] += 1.0
    with np.errstate(invalid="ignore"):
        means = np.where(counts > 0, totals / np.maximum(counts, 1.0), np.nan)
    return means, patients


def _contrasts(means: np.ndarray) -> np.ndarray:
    """Paired Base-Mid and Apex-Mid differences per quartile.

    The difference is taken inside each patient before averaging, so both arms
    of a contrast come from the same patients. Averaging the two regions
    independently and subtracting would compare a base mean over one set of
    patients against a mid mean over another, which within the smallest area
    quartile are very different sets.
    """
    mid = means[:, :, REGIONS.index("Mid")]
    with np.errstate(invalid="ignore"):
        base_mid = np.nanmean(means[:, :, REGIONS.index("Base")] - mid, axis=0)
        apex_mid = np.nanmean(means[:, :, REGIONS.index("Apex")] - mid, axis=0)
    return np.stack([base_mid, apex_mid])


def _pair_counts(means: np.ndarray) -> np.ndarray:
    """Patients contributing to each contrast, i.e. having both regions."""
    mid = means[:, :, REGIONS.index("Mid")]
    return np.stack(
        [
            np.sum(np.isfinite(means[:, :, REGIONS.index(region)] - mid), axis=0)
            for region in ("Base", "Apex")
        ]
    )


def _stratified_contrasts(frame: pd.DataFrame, rng: np.random.Generator) -> dict:
    means, patients = _cell_means(frame)
    point = _contrasts(means)
    draws = np.empty((N_BOOTSTRAP, *point.shape))
    for b in range(N_BOOTSTRAP):
        sample = rng.integers(0, len(patients), len(patients))
        draws[b] = _contrasts(means[sample])
    low = np.nanpercentile(draws, 2.5, axis=0)
    high = np.nanpercentile(draws, 97.5, axis=0)
    return {
        "point": point,
        "low": low,
        "high": high,
        "n_patients": _pair_counts(means),
    }


def _plot_stratified(ax: plt.Axes, results: dict[str, dict]) -> None:
    offsets = {"WG": -0.12, "TZ+CZ": 0.0, "PZ": 0.12}
    x = np.arange(N_QUARTILES)
    for class_name in CLASSES:
        stats = results[class_name]
        colour = CLASS_COLORS[class_name]
        for row, (label, style, marker) in enumerate(
            [("Base--Mid", "-", "o"), ("Apex--Mid", "--", "s")]
        ):
            point = stats["point"][row]
            yerr = np.stack(
                [point - stats["low"][row], stats["high"][row] - point]
            )
            ax.errorbar(
                x + offsets[class_name],
                point,
                yerr=yerr,
                color=colour,
                linestyle=style,
                marker=marker,
                markersize=5,
                capsize=3,
                linewidth=1.6,
                label=f"{class_name} {label.replace('--', '-')}",
            )
    ax.axhline(0.0, color="black", linewidth=1.0, linestyle=":")
    ax.set_xticks(x)
    ax.set_xticklabels([f"Q{i + 1}" for i in range(N_QUARTILES)])
    ax.set_xlabel("Foreground area quartile (within class, smallest to largest)")
    ax.set_ylabel(r"Difference in mean $\Delta_{Dice}$ vs mid-gland")
    ax.set_title(
        "Anatomical effect at matched foreground area", fontsize=12, fontweight="bold"
    )
    ax.legend(fontsize=7, ncol=2, loc="upper right", framealpha=0.9)
    ax.grid(alpha=0.3)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument(
        "--input-dir",
        type=Path,
        default=REPO_ROOT / "rebuttal_paper2" / "slice_anatomy_corrected",
        help="directory holding all_folds_per_slice_with_shape.csv",
    )
    parser.add_argument(
        "--output-dir", type=Path, default=None, help="default: --input-dir"
    )
    parser.add_argument(
        "--paper-figures",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="also write the figure into paper/figures/ (the manuscript copy)",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    source = args.input_dir / "all_folds_per_slice_with_shape.csv"
    frame = pd.read_csv(source)
    frame = frame[np.isclose(frame["epsilon"], EPSILON)]
    frame = frame[frame["gt_area_mm2"] > 0]

    rng = np.random.default_rng(SEED)
    results: dict[str, dict] = {}
    rows: list[dict] = []
    for class_name in CLASSES:
        subset = frame[frame["class"] == class_name].copy()
        subset["area_quartile"] = pd.qcut(
            subset["gt_area_mm2"], N_QUARTILES, labels=False
        )
        stats = _stratified_contrasts(subset, rng)
        results[class_name] = stats
        for quartile in range(N_QUARTILES):
            for row, term in enumerate(["Base_vs_Mid", "Apex_vs_Mid"]):
                rows.append(
                    {
                        "class": class_name,
                        "area_quartile": f"Q{quartile + 1}",
                        "term": term,
                        "difference": stats["point"][row, quartile],
                        "ci95_low": stats["low"][row, quartile],
                        "ci95_high": stats["high"][row, quartile],
                        "n_patients": int(stats["n_patients"][row, quartile]),
                    }
                )

    output_dir = args.output_dir or args.input_dir
    output_dir.mkdir(parents=True, exist_ok=True)
    table = pd.DataFrame(rows)
    table.to_csv(output_dir / "area_stratified_region_effects.csv", index=False)
    print(table.to_string(index=False))

    endpoint = frame
    wg_records = endpoint[endpoint["task"] == "wg"].to_dict("records")
    zones_records = endpoint[endpoint["task"] == "zones"].to_dict("records")

    fig = plt.figure(figsize=(14, 10), layout="constrained")
    ax_wg, ax_tz, ax_pz, ax_stratified = add_combined_wg_tz_pz_axes(
        fig, hspace=0.14, wspace=0.16, bottom_inner_wspace=0.3
    )
    scale = {"area_scale": MM2_PER_PIXEL, "area_unit": AREA_UNIT}
    plot_fg_size_vs_vulnerability_on_ax(
        ax_wg, wg_records, "WG", EPSILON, "Whole Gland", CLASS_COLORS["WG"], **scale
    )
    plot_fg_size_vs_vulnerability_on_ax(
        ax_tz, zones_records, "TZ+CZ", EPSILON, "Prostate Zones",
        CLASS_COLORS["TZ+CZ"], **scale
    )
    plot_fg_size_vs_vulnerability_on_ax(
        ax_pz, zones_records, "PZ", EPSILON, "Prostate Zones",
        CLASS_COLORS["PZ"], **scale
    )
    _plot_stratified(ax_stratified, results)
    fig.suptitle(
        "Foreground Area, Dice Drop, and Anatomical Position at Matched "
        "Foreground Area under FGSM",
        fontsize=14,
        fontweight="bold",
    )
    targets = [output_dir / "combined_wg_zones_fg_size_vs_vulnerability"]
    if args.paper_figures:
        targets.append(
            REPO_ROOT / "paper" / "figures" / "combined_wg_zones_fg_size_vs_vulnerability"
        )
    for target in targets:
        fig.savefig(
            target.with_suffix(".png"), dpi=600, bbox_inches="tight", pad_inches=0.08
        )
        fig.savefig(target.with_suffix(".pdf"), bbox_inches="tight", pad_inches=0.08)
        print(f"  Saved {target}.png / .pdf")
    plt.close(fig)


if __name__ == "__main__":
    main()
