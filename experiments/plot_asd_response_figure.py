"""Standalone ASD position figure in millimetres.

The runner's generic ASD plot reports pixel distances and reuses a Dice-drop
summary panel, so it misstates both the units and the metric. This script
redraws the figure from the same on-disk slice tables with every panel in
millimetres and the summary panel showing ASD rather than Dice.

The figure is reported in neither the manuscript nor the Reviewer 2 response
letter. See analyze_asd_base_effect for why, and keep it that way unless a
later review round asks for a size-insensitive metric.
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

REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPOSITORY_ROOT / "src"))
sys.path.insert(0, str(REPOSITORY_ROOT / "experiments"))

from analyze_asd_base_effect import (  # noqa: E402
    CLASS_NAMES,
    ENDPOINT_EPSILON,
    load_asd_slice_table,
    merge_shape_covariates,
)
from mri_prostate_seg.experiments.slice_vulnerability.adjusted import (  # noqa: E402
    patient_bootstrap_mean_ci,
)

CLASS_COLORS = {"WG": "#1976D2", "TZ+CZ": "#D32F2F", "PZ": "#388E3C"}
REGION_ORDER = ["Base", "Mid", "Apex"]
N_BINS = 10


def to_millimetres(merged: pd.DataFrame) -> pd.DataFrame:
    """Scale the pixel ASD change by the in-plane spacing."""
    spacing = merged[["row_spacing_mm", "column_spacing_mm"]].to_numpy(dtype=float)
    if not np.allclose(spacing[:, 0], spacing[:, 1]):
        raise RuntimeError("in-plane spacing is anisotropic; per-axis ASD required")
    frame = merged.copy()
    frame["asd_change_mm"] = frame["asd_change"].to_numpy(dtype=float) * spacing[:, 0]
    return frame[np.isfinite(frame["asd_change_mm"])]


def positional_heatmap(frame: pd.DataFrame, class_name: str) -> tuple[np.ndarray, list]:
    """Patient-weighted mean ASD change per position bin and epsilon."""
    subset = frame[frame["class"] == class_name].copy()
    position = subset["prostate_relative_position"].to_numpy(dtype=float)
    subset = subset[(position >= 0.0) & (position <= 1.0)]
    subset["bin_index"] = np.minimum(
        (subset["prostate_relative_position"].to_numpy(dtype=float) * N_BINS).astype(
            int
        ),
        N_BINS - 1,
    )
    epsilons = sorted(subset["epsilon"].unique())
    # Average within patient-and-bin first, then across patients.
    per_patient = (
        subset.groupby(["epsilon", "bin_index", "patient_id"], observed=True)[
            "asd_change_mm"
        ]
        .mean()
        .reset_index()
    )
    cell = (
        per_patient.groupby(["epsilon", "bin_index"], observed=True)["asd_change_mm"]
        .mean()
        .reset_index()
    )
    grid = np.full((N_BINS, len(epsilons)), np.nan)
    for _, row in cell.iterrows():
        grid[int(row["bin_index"]), epsilons.index(row["epsilon"])] = row[
            "asd_change_mm"
        ]
    return grid, epsilons


def regional_summary(frame: pd.DataFrame) -> pd.DataFrame:
    """Patient-weighted regional mean ASD change with bootstrap intervals."""
    endpoint = frame[np.isclose(frame["epsilon"], ENDPOINT_EPSILON)]
    per_patient = (
        endpoint.groupby(["class", "anatomical_region", "patient_id"], observed=True)[
            "asd_change_mm"
        ]
        .mean()
        .reset_index()
    )
    rows: list[dict[str, object]] = []
    for class_name in CLASS_NAMES:
        for region in REGION_ORDER:
            values = per_patient[
                (per_patient["class"] == class_name)
                & (per_patient["anatomical_region"] == region)
            ]["asd_change_mm"].to_numpy(dtype=float)
            low, high = patient_bootstrap_mean_ci(values, n_resamples=2_000, seed=42)
            rows.append(
                {
                    "class": class_name,
                    "region": region,
                    "mean_asd_change_mm": float(np.mean(values)),
                    "ci95_low": low,
                    "ci95_high": high,
                    "n_patients": int(values.size),
                }
            )
    return pd.DataFrame(rows)


def draw_figure(frame: pd.DataFrame, summary: pd.DataFrame, output_base: Path) -> None:
    figure, axes = plt.subplots(2, 2, figsize=(14, 11))
    grids = {name: positional_heatmap(frame, name) for name in CLASS_NAMES}
    finite = np.concatenate([g[np.isfinite(g)].ravel() for g, _ in grids.values()])
    vmax = float(np.percentile(finite, 99))

    for axis, class_name in zip(axes.ravel()[:3], CLASS_NAMES):
        grid, epsilons = grids[class_name]
        image = axis.imshow(
            grid,
            aspect="auto",
            origin="lower",
            cmap="YlOrRd",
            vmin=0.0,
            vmax=vmax,
            extent=(-0.5, len(epsilons) - 0.5, 0.0, 1.0),
        )
        axis.set_title(
            f"{class_name}\nMean $\\Delta_{{ASD}}$ (mm) by position and $\\varepsilon$",
            fontweight="bold",
        )
        axis.set_xlabel("FGSM $\\varepsilon$")
        axis.set_ylabel("Prostate-relative position (0 = base, 1 = apex)")
        axis.set_xticks(range(len(epsilons)))
        axis.set_xticklabels([f"{e:.2f}" for e in epsilons])
        figure.colorbar(image, ax=axis, label="Mean $\\Delta_{ASD}$ (mm)")

    summary_axis = axes.ravel()[3]
    width = 0.26
    offsets = np.arange(len(REGION_ORDER))
    for index, class_name in enumerate(CLASS_NAMES):
        rows = summary[summary["class"] == class_name].set_index("region")
        means = np.array([rows.loc[r, "mean_asd_change_mm"] for r in REGION_ORDER])
        lows = np.array([rows.loc[r, "ci95_low"] for r in REGION_ORDER])
        highs = np.array([rows.loc[r, "ci95_high"] for r in REGION_ORDER])
        summary_axis.bar(
            offsets + (index - 1) * width,
            means,
            width,
            label=class_name,
            color=CLASS_COLORS[class_name],
            yerr=[means - lows, highs - means],
            capsize=4,
        )
    summary_axis.set_title(
        f"Patient-weighted $\\Delta_{{ASD}}$ at $\\varepsilon={ENDPOINT_EPSILON:.2f}$"
        "\n(95% patient-bootstrap CI)",
        fontweight="bold",
    )
    summary_axis.set_xticks(offsets)
    summary_axis.set_xticklabels(REGION_ORDER)
    summary_axis.set_xlabel("Anatomical region")
    summary_axis.set_ylabel("Mean $\\Delta_{ASD}$ (mm)")
    summary_axis.legend()
    summary_axis.grid(axis="y", alpha=0.3)

    figure.suptitle(
        "Adversarial boundary displacement by prostate-relative position "
        "(FGSM, size-insensitive metric)",
        fontsize=15,
        fontweight="bold",
    )
    figure.tight_layout(rect=(0, 0, 1, 0.97))
    for suffix in (".png", ".pdf"):
        figure.savefig(output_base.with_suffix(suffix), dpi=200, bbox_inches="tight")
        print(f"  Saved {output_base.with_suffix(suffix)}")
    plt.close(figure)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--asd-dir",
        type=Path,
        default=REPOSITORY_ROOT / "results" / "slice_anatomy_asd",
    )
    parser.add_argument(
        "--enriched-csv",
        type=Path,
        default=REPOSITORY_ROOT
        / "rebuttal_paper2"
        / "slice_anatomy_corrected"
        / "all_folds_per_slice_with_shape.csv",
    )
    arguments = parser.parse_args()

    print("Loading ASD slice tables")
    merged = merge_shape_covariates(
        load_asd_slice_table(arguments.asd_dir), arguments.enriched_csv
    )
    frame = to_millimetres(merged)
    print(f"  analysable rows: {len(frame):,}")

    summary = regional_summary(frame)
    summary_path = arguments.asd_dir / "asd_response_regional_summary_mm.csv"
    summary.to_csv(summary_path, index=False)
    print(summary.to_string(index=False))

    draw_figure(frame, summary, arguments.asd_dir / "response_asd_position_heatmap_mm")
    print(f"  Saved {summary_path}")


if __name__ == "__main__":
    main()
