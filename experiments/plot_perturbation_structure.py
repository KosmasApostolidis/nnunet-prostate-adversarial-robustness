"""Analyze and plot full-cohort perturbation-structure descriptors."""

from __future__ import annotations

import argparse
import sys
import textwrap
from pathlib import Path

import matplotlib
import matplotlib.patheffects as path_effects
import numpy as np
import pandas as pd

matplotlib.use("Agg")
import matplotlib.pyplot as plt


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))
SRC_ROOT = REPO_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from experiments.evaluate_perturbation_structure_all_cases import (  # noqa: E402
    DEFAULT_OUTPUT_DIR,
    DEFAULT_QUALITY_CSV,
)
from mri_prostate_seg.experiments.perturbation_structure import (  # noqa: E402
    BAND_NAMES,
    SIGNED_BAND_NAMES,
)


STRUCTURE_CSV = "perturbation_structure_all_cases.csv"
SPECTRA_CSV = "radial_spectra_all_cases_epsilon16.csv"
SUBSAMPLE_SPECTRA_CSV = "radial_spectra_subsample.csv"
SUMMARY_CSV = "perturbation_structure_summary.csv"
SIGNED_BAND_CONTRAST_CSV = "signed_band_contrasts.csv"
PAIRED_CSV = "paired_vs_gaussian_rms_matched.csv"
REPRODUCTION_CSV = "rms_reproduction_summary.csv"
REPORT_NAME = "PERTURBATION_STRUCTURE_REPORT.md"

ATTACK_CONDITIONS = ("FGSM-BCE", "PGD-BCE", "APGD-BCE")
CONTROL_CONDITIONS = (
    "gaussian_rms_matched",
    "rician_proxy_rms_matched",
)
CONDITION_ORDER = (*ATTACK_CONDITIONS, *CONTROL_CONDITIONS)
DATASET_LABELS = {"wg": "Whole gland (WG)", "zones": "Prostate zones"}
BAND_LABELS = {
    "inside": "inside",
    "0_2mm": "0–2 mm",
    "2_5mm": "2–5 mm",
    "5_10mm": "5–10 mm",
    "beyond_10mm": "> 10 mm",
}
# Signed distance to the boundary, negative inside the gland.  The two bands
# either side of zero are the shells the pooled ``inside`` band cannot separate.
SIGNED_BAND_LABELS = {
    "inside_beyond_10mm": "< −10",
    "inside_5_10mm": "−10 to −5",
    "inside_2_5mm": "−5 to −2",
    "inside_0_2mm": "−2 to 0",
    "0_2mm": "0 to 2",
    "2_5mm": "2 to 5",
    "5_10mm": "5 to 10",
    "beyond_10mm": "> 10",
}
CONDITION_LABELS = {
    "FGSM-BCE": "FGSM-BCE",
    "PGD-BCE": "PGD-BCE",
    "APGD-BCE": "APGD-BCE",
    "gaussian_rms_matched": "Gaussian (analytic reference)",
    "rician_proxy_rms_matched": "Rician proxy (analytic reference)",
}
# Keep figure legends compact enough to sit outside the plotting area. The
# report retains the longer labels above, where the analytic-reference framing
# is stated in full.
PLOT_CONDITION_LABELS = {
    **CONDITION_LABELS,
    "gaussian_rms_matched": "Gaussian reference",
    "rician_proxy_rms_matched": "Rician reference",
}
COLORS = {
    "FGSM-BCE": "#E69F00",
    "PGD-BCE": "#D55E00",
    "APGD-BCE": "#AA3377",
    "gaussian_rms_matched": "#0072B2",
    "rician_proxy_rms_matched": "#009E73",
}
MARKERS = {
    "FGSM-BCE": "o",
    "PGD-BCE": "s",
    "APGD-BCE": "D",
    "gaussian_rms_matched": "^",
    "rician_proxy_rms_matched": "v",
}
LINESTYLES = {
    "FGSM-BCE": "-",
    "PGD-BCE": "--",
    "APGD-BCE": "-.",
    "gaussian_rms_matched": (0, (5, 2)),
    "rician_proxy_rms_matched": (0, (1, 1)),
}

# nnU-Net resamples both datasets to the same plans target spacing, and every
# one of the 1,999 preprocessed cases carries exactly (3.0, 0.5, 0.5) mm, so the
# conversion below is one global constant rather than a per-case one. Spectra
# are computed in-plane only -- see perturbation_structure.radial_power_spectrum
# -- so the 3.0 mm slice thickness never enters the frequency axis.
IN_PLANE_SPACING_MM = 0.5
# Each axis is normalized to its own Nyquist and the radius is divided by
# sqrt(2), so a plotted radius of 1 is the corner of the frequency plane.
RADIAL_TO_CYCLES_PER_MM = np.sqrt(2.0) / (2.0 * IN_PLANE_SPACING_MM)

SPECTRAL_METRICS = [
    "spectral_centroid_norm",
    "hf_mean_power_fraction",
    "spectral_slope",
    "spectral_slices_used",
]
LOCALIZATION_METRICS = [
    "energy_fraction_foreground",
    "volume_fraction_foreground",
    "foreground_energy_enrichment",
    *[
        f"{prefix}_band_{band}"
        for band in BAND_NAMES
        for prefix in ("energy_fraction", "volume_fraction", "enrichment")
    ],
]
ALIGNMENT_METRICS = [
    "spearman_absdelta_gradmag",
    "edge_energy_enrichment",
    "alignment_voxels_used",
]
GMSD_METRICS = ["gmsd_axial_prostate"]
ALL_METRICS = [
    "rms_norm",
    *SPECTRAL_METRICS,
    *LOCALIZATION_METRICS,
    *ALIGNMENT_METRICS,
    *GMSD_METRICS,
]
# Family C's two descriptors are both functions of |δ|.  An ε-ball attack that
# saturates drives |δ| to the constant ε, which destroys them: ``spearmanr``
# returns NaN on a constant input (it warns, it does not raise) and
# ``edge_energy_enrichment`` is then exactly 1.0 -- numerically indistinguishable
# from the uniform-noise reference.  1.4% of WG and 9.3% of zones FGSM rows are
# exactly 100% saturated, so the descriptor is blind, not the attack unstructured.
SATURATION_COLUMN = "epsilon_saturation_fraction"
SATURATION_MEDIAN_COLUMN = f"{SATURATION_COLUMN}_median"

PAIRED_METRICS = [
    "foreground_energy_enrichment",
    *[f"enrichment_band_{band}" for band in BAND_NAMES],
    "spectral_centroid_norm",
    "hf_mean_power_fraction",
    "spectral_slope",
    "spearman_absdelta_gradmag",
    "edge_energy_enrichment",
    "gmsd_axial_prostate",
]


def join_quality(structure: pd.DataFrame, quality: pd.DataFrame) -> pd.DataFrame:
    """Attach published PSNR/SSIM/saturation to attack rows; controls keep NaN.

    ``epsilon_saturation_fraction`` is carried because Family C needs it: both
    alignment descriptors are functions of |δ|, and a saturated ε-ball attack has
    |δ| = ε everywhere, which destroys them (see ``SATURATION_COLUMN``).
    """

    columns = [
        "dataset",
        "case_id",
        "epsilon_n",
        "attack",
        "psnr_db_robust_range",
        "ssim_axial_prostate",
        SATURATION_COLUMN,
    ]
    missing = sorted(set(columns) - set(quality.columns))
    if missing:
        raise ValueError(f"quality table is missing columns: {missing}")
    right = quality[columns].rename(columns={"attack": "condition"})
    keys = ["dataset", "case_id", "epsilon_n", "condition"]
    if right.duplicated(keys, keep=False).any():
        raise ValueError("quality table contains duplicate case/epsilon/attack rows")
    return structure.merge(right, on=keys, how="left", validate="many_to_one")


def summarize(structure: pd.DataFrame, metrics: list[str]) -> pd.DataFrame:
    """Summarize finite metric values by dataset, condition, and epsilon."""

    rows: list[dict[str, object]] = []
    for (dataset, condition, epsilon_n), group in structure.groupby(
        ["dataset", "condition", "epsilon_n"], sort=True
    ):
        row: dict[str, object] = {
            "dataset": dataset,
            "condition": condition,
            "epsilon_n": int(epsilon_n),
            "n_cases": int(group["case_id"].nunique()),
        }
        for metric in metrics:
            values = group[metric].to_numpy(dtype=float)
            values = values[np.isfinite(values)]
            if values.size:
                row[f"{metric}_mean"] = float(np.mean(values))
                row[f"{metric}_median"] = float(np.median(values))
                row[f"{metric}_q05"] = float(np.quantile(values, 0.05))
                row[f"{metric}_q95"] = float(np.quantile(values, 0.95))
            else:
                for statistic in ("mean", "median", "q05", "q95"):
                    row[f"{metric}_{statistic}"] = float("nan")
        rows.append(row)
    return pd.DataFrame(rows)


def paired_vs_control(
    structure: pd.DataFrame,
    metric: str,
    *,
    control: str = "gaussian_rms_matched",
) -> pd.DataFrame:
    """Within-case condition-minus-control differences.

    Repeated control trials are averaged before joining so they cannot multiply
    attack rows and silently inflate ``n_pairs``.
    """

    keys = ["dataset", "case_id", "epsilon_n"]
    baseline = structure[structure["condition"] == control][keys + [metric]]
    baseline = baseline.groupby(keys, as_index=False, sort=True)[metric].mean()
    baseline = baseline.rename(columns={metric: "control_value"})
    other = structure[structure["condition"] != control]
    merged = other.merge(baseline, on=keys, how="inner", validate="many_to_one")
    merged["difference"] = merged[metric] - merged["control_value"]

    rows: list[dict[str, object]] = []
    for (dataset, condition, epsilon_n), group in merged.groupby(
        ["dataset", "condition", "epsilon_n"], sort=True
    ):
        differences = group["difference"].to_numpy(dtype=float)
        differences = differences[np.isfinite(differences)]
        if not differences.size:
            continue
        rows.append(
            {
                "dataset": dataset,
                "condition": condition,
                "epsilon_n": int(epsilon_n),
                "n_pairs": int(len(differences)),
                "median_difference": float(np.median(differences)),
                "mean_difference": float(np.mean(differences)),
                "fraction_above_control": float(np.mean(differences > 0)),
            }
        )
    return pd.DataFrame(rows)


def _configure_style() -> None:
    plt.rcParams.update(
        {
            "font.size": 10,
            "axes.titlesize": 12,
            "axes.labelsize": 11,
            "axes.titleweight": "semibold",
            "legend.fontsize": 9.5,
            "figure.titlesize": 15,
            "figure.titleweight": "semibold",
            "xtick.labelsize": 9.5,
            "ytick.labelsize": 9.5,
            "axes.linewidth": 0.9,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "axes.axisbelow": True,
            "grid.color": "#A7A7A7",
            "grid.linewidth": 0.7,
            "grid.alpha": 0.25,
            "savefig.facecolor": "white",
            "figure.facecolor": "white",
        }
    )


def _datasets(frame: pd.DataFrame) -> list[str]:
    present = set(frame["dataset"].astype(str))
    ordered = [value for value in ("wg", "zones") if value in present]
    return ordered or sorted(present)


def _conditions(frame: pd.DataFrame) -> list[str]:
    present = set(frame["condition"].astype(str))
    ordered = [value for value in CONDITION_ORDER if value in present]
    return [*ordered, *sorted(present - set(ordered))]


def _condition_label(condition: str) -> str:
    return CONDITION_LABELS.get(condition, condition)


def _plot_condition_label(condition: str) -> str:
    return PLOT_CONDITION_LABELS.get(condition, condition)


def _metric_quantiles(frame: pd.DataFrame, metric: str) -> pd.DataFrame:
    rows: list[dict[str, float]] = []
    for epsilon_n, group in frame.groupby("epsilon_n", sort=True):
        values = group[metric].to_numpy(dtype=float)
        values = values[np.isfinite(values)]
        if values.size:
            rows.append(
                {
                    "epsilon_n": float(epsilon_n),
                    "median": float(np.median(values)),
                    "q05": float(np.quantile(values, 0.05)),
                    "q95": float(np.quantile(values, 0.95)),
                }
            )
    return pd.DataFrame(rows)


def _plot_metric_vs_epsilon(
    ax: plt.Axes,
    frame: pd.DataFrame,
    metric: str,
    *,
    title: str,
    ylabel: str,
    reference: float = 0.0,
    scale: float = 1.0,
) -> None:
    for condition in _conditions(frame):
        values = _metric_quantiles(frame[frame["condition"] == condition], metric)
        if values.empty:
            continue
        x = values["epsilon_n"].to_numpy(dtype=float)
        median = (values["median"].to_numpy(dtype=float) - float(reference)) * float(
            scale
        )
        q05 = (values["q05"].to_numpy(dtype=float) - float(reference)) * float(scale)
        q95 = (values["q95"].to_numpy(dtype=float) - float(reference)) * float(scale)
        color = COLORS.get(condition)
        ax.plot(
            x,
            median,
            color=color,
            marker=MARKERS.get(condition, "o"),
            linestyle=LINESTYLES.get(condition, "-"),
            linewidth=2.6,
            markersize=6.2,
            markeredgecolor="white",
            markeredgewidth=0.7,
            label=_plot_condition_label(condition),
            zorder=3,
            path_effects=[
                path_effects.Stroke(linewidth=3.8, foreground="white", alpha=0.8),
                path_effects.Normal(),
            ],
        )
        ax.fill_between(
            x,
            q05,
            q95,
            color=color,
            alpha=0.065,
            linewidth=0,
            zorder=1,
        )
    ax.set_title(title)
    ax.set_xlabel("Perturbation budget ε (n/255)")
    ax.set_ylabel(ylabel)
    ax.set_xticks([2, 4, 8, 16, 32])
    ax.grid()


def _shared_legend(
    fig: plt.Figure,
    axes: np.ndarray | list[plt.Axes],
    *,
    ncol: int = 5,
    anchor_y: float = 0.94,
    title: str | None = None,
) -> None:
    """Place one deduplicated legend above the panels, never over the data."""

    flat_axes = np.asarray(axes, dtype=object).ravel()
    handles: list[object] = []
    labels: list[str] = []
    for ax in flat_axes:
        panel_handles, panel_labels = ax.get_legend_handles_labels()
        for handle, label in zip(panel_handles, panel_labels):
            if label and label not in labels:
                handles.append(handle)
                labels.append(label)
    if handles:
        legend = fig.legend(
            handles,
            labels,
            loc="upper center",
            bbox_to_anchor=(0.5, anchor_y),
            ncol=min(ncol, len(handles)),
            title=title,
            frameon=True,
            facecolor="white",
            edgecolor="#D8D8D8",
            framealpha=0.98,
            fancybox=True,
            borderpad=0.45,
            handlelength=3.2,
            markerscale=1.15,
            handletextpad=0.7,
            columnspacing=1.5,
        )
        if title:
            legend.get_title().set_fontweight("semibold")


def _add_physical_frequency_axis(ax: plt.Axes) -> None:
    """Mirror the normalized radial axis in cycles/mm.

    The normalized axis is what the spectral scalars and the white reference of
    0.5 are expressed in, so it stays primary; this adds the physical reading
    that a fixed 0.5 mm in-plane spacing makes exact.
    """

    secondary = ax.secondary_xaxis(
        "top",
        functions=(
            lambda radius: np.asarray(radius) * RADIAL_TO_CYCLES_PER_MM,
            lambda cycles: np.asarray(cycles) / RADIAL_TO_CYCLES_PER_MM,
        ),
    )
    secondary.set_xlabel("Radial frequency (cycles/mm)", labelpad=4)


def _figure_note(fig: plt.Figure, text: str, *, y: float = 0.012) -> None:
    fig.text(
        0.5,
        y,
        text,
        ha="center",
        va="bottom",
        fontsize=8.5,
        color="#4B4B4B",
    )


def _save(
    fig: plt.Figure,
    path: Path,
    *,
    tight: bool = True,
    rect: tuple[float, float, float, float] | None = None,
) -> None:
    if tight:
        fig.tight_layout(rect=rect)
    fig.savefig(path, dpi=220, bbox_inches="tight")
    plt.close(fig)


def plot_localization(structure: pd.DataFrame, output_dir: Path) -> None:
    datasets = _datasets(structure)
    fig, axes = plt.subplots(
        1, len(datasets), figsize=(6.6 * len(datasets), 4.8), squeeze=False
    )
    for column, dataset in enumerate(datasets):
        ax = axes[0, column]
        _plot_metric_vs_epsilon(
            ax,
            structure[structure["dataset"] == dataset],
            "foreground_energy_enrichment",
            title=DATASET_LABELS.get(dataset, dataset),
            ylabel="Enrichment relative to uniform (%)",
            reference=1.0,
            scale=100.0,
        )
        ax.axhline(0.0, color="#3F3F3F", linestyle=":", linewidth=1.2, zorder=2)
    fig.suptitle("Perturbation energy localized to the prostate", y=0.995)
    _shared_legend(fig, axes, anchor_y=0.92)
    _figure_note(
        fig,
        "Lines show cohort medians; shading spans the 5th–95th percentiles. "
        "Zero is the uniform-energy reference.",
    )
    _save(
        fig,
        output_dir / "11_localization_enrichment_vs_epsilon.png",
        rect=(0.0, 0.07, 1.0, 0.87),
    )


def plot_boundary_bands(
    structure: pd.DataFrame, output_dir: Path, *, reference_epsilon: int
) -> None:
    selected = structure[structure["epsilon_n"] == reference_epsilon]
    datasets = _datasets(selected)
    fig, axes = plt.subplots(
        1, len(datasets), figsize=(6.6 * len(datasets), 4.8), squeeze=False
    )
    x = np.arange(len(BAND_NAMES), dtype=float)
    for column, dataset in enumerate(datasets):
        ax = axes[0, column]
        data = selected[selected["dataset"] == dataset]
        conditions = _conditions(data)
        for condition in conditions:
            group = data[data["condition"] == condition]
            medians = [
                100.0
                * (
                    float(
                        np.nanmedian(
                            group[f"enrichment_band_{band}"].to_numpy(dtype=float)
                        )
                    )
                    - 1.0
                )
                for band in BAND_NAMES
            ]
            ax.plot(
                x,
                medians,
                color=COLORS.get(condition),
                linestyle=LINESTYLES.get(condition, "-"),
                marker=MARKERS.get(condition, "o"),
                linewidth=2.6,
                markersize=6.4,
                markeredgecolor="white",
                markeredgewidth=0.7,
                label=_plot_condition_label(condition),
                zorder=3,
                path_effects=[
                    path_effects.Stroke(linewidth=3.8, foreground="white", alpha=0.8),
                    path_effects.Normal(),
                ],
            )
        ax.axhline(0.0, color="#3F3F3F", linestyle=":", linewidth=1.2, zorder=2)
        ax.set_xticks(x, [BAND_LABELS[name] for name in BAND_NAMES])
        ax.set_ylabel("Median enrichment relative to uniform (%)")
        ax.set_title(DATASET_LABELS.get(dataset, dataset))
        ax.grid(axis="y")
        ax.margins(x=0.04, y=0.14)
    fig.suptitle(
        f"Perturbation energy by distance from the boundary (ε={reference_epsilon}/255)",
        y=0.995,
    )
    _shared_legend(fig, axes, anchor_y=0.92)
    _figure_note(
        fig,
        "Positive values indicate concentration above a spatially uniform field; "
        "negative values indicate depletion.",
    )
    _save(
        fig,
        output_dir / "12_boundary_band_enrichment.png",
        rect=(0.0, 0.07, 1.0, 0.87),
    )


def _band_quantiles(
    frame: pd.DataFrame, bands: tuple[str, ...] = BAND_NAMES
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Return per-band median and 5th/95th percentiles, in percent above uniform."""

    median: list[float] = []
    q05: list[float] = []
    q95: list[float] = []
    for band in bands:
        values = frame[f"enrichment_band_{band}"].to_numpy(dtype=float)
        values = values[np.isfinite(values)]
        if values.size:
            median.append(100.0 * (float(np.median(values)) - 1.0))
            q05.append(100.0 * (float(np.quantile(values, 0.05)) - 1.0))
            q95.append(100.0 * (float(np.quantile(values, 0.95)) - 1.0))
        else:
            median.append(np.nan)
            q05.append(np.nan)
            q95.append(np.nan)
    return (
        np.asarray(median, dtype=float),
        np.asarray(q05, dtype=float),
        np.asarray(q95, dtype=float),
    )


def plot_boundary_bands_by_epsilon(structure: pd.DataFrame, output_dir: Path) -> None:
    """Boundary-band profiles at every budget, with cohort spread.

    Companion to ``12_boundary_band_enrichment.png``, which fixes one budget and
    plots medians alone. Reading across a row shows how the boundary preference
    scales with ε; the shading shows how much of the cohort shares it.
    """

    datasets = _datasets(structure)
    epsilons = sorted(int(value) for value in structure["epsilon_n"].dropna().unique())
    fig, axes = plt.subplots(
        len(datasets),
        len(epsilons),
        figsize=(3.5 * len(epsilons), 3.9 * len(datasets)),
        squeeze=False,
        sharey="row",
    )
    x = np.arange(len(BAND_NAMES), dtype=float)
    for row, dataset in enumerate(datasets):
        dataset_rows = structure[structure["dataset"] == dataset]
        for column, epsilon_n in enumerate(epsilons):
            ax = axes[row, column]
            data = dataset_rows[dataset_rows["epsilon_n"] == epsilon_n]
            for condition in _conditions(data):
                group = data[data["condition"] == condition]
                median, q05, q95 = _band_quantiles(group)
                color = COLORS.get(condition)
                ax.fill_between(
                    x, q05, q95, color=color, alpha=0.065, linewidth=0, zorder=1
                )
                ax.plot(
                    x,
                    median,
                    color=color,
                    linestyle=LINESTYLES.get(condition, "-"),
                    marker=MARKERS.get(condition, "o"),
                    linewidth=2.2,
                    markersize=5.2,
                    markeredgecolor="white",
                    markeredgewidth=0.6,
                    label=_plot_condition_label(condition),
                    zorder=3,
                    path_effects=[
                        path_effects.Stroke(
                            linewidth=3.4, foreground="white", alpha=0.8
                        ),
                        path_effects.Normal(),
                    ],
                )
            ax.axhline(0.0, color="#3F3F3F", linestyle=":", linewidth=1.2, zorder=2)
            ax.set_xticks(x, [BAND_LABELS[name] for name in BAND_NAMES])
            for label in ax.get_xticklabels():
                label.set_rotation(30)
                label.set_horizontalalignment("right")
            ax.grid(axis="y")
            ax.margins(x=0.05, y=0.12)
            if row == 0:
                ax.set_title(f"ε={epsilon_n}/255")
            if column == 0:
                ax.set_ylabel(
                    f"{DATASET_LABELS.get(dataset, dataset)}\n"
                    "enrichment relative to uniform (%)"
                )
    fig.suptitle(
        "Perturbation energy by distance from the boundary, across budgets",
        y=0.995,
    )
    _shared_legend(fig, axes, anchor_y=0.945)
    _figure_note(
        fig,
        "Lines show cohort medians; shading spans the 5th–95th percentiles. "
        "Positive values indicate concentration above a spatially uniform "
        "field; negative values indicate depletion.",
    )
    _save(
        fig,
        output_dir / "17_boundary_band_enrichment_by_epsilon.png",
        rect=(0.0, 0.04, 1.0, 0.91),
    )


def has_signed_bands(structure: pd.DataFrame) -> bool:
    """Whether the table carries the interior bands added in commit 96809eb."""

    return all(
        f"enrichment_band_{band}" in structure.columns for band in SIGNED_BAND_NAMES
    )


def _draw_signed_band_panel(ax: plt.Axes, data: pd.DataFrame) -> None:
    """Draw one dataset x epsilon panel of the signed-distance profile."""

    x = np.arange(len(SIGNED_BAND_NAMES), dtype=float)
    # The boundary sits between the last interior band and the first exterior
    # one, not on a tick.
    boundary_x = float(np.argmax([name == "0_2mm" for name in SIGNED_BAND_NAMES])) - 0.5
    for condition in _conditions(data):
        group = data[data["condition"] == condition]
        median, q05, q95 = _band_quantiles(group, SIGNED_BAND_NAMES)
        color = COLORS.get(condition)
        ax.fill_between(x, q05, q95, color=color, alpha=0.065, linewidth=0, zorder=1)
        ax.plot(
            x,
            median,
            color=color,
            linestyle=LINESTYLES.get(condition, "-"),
            marker=MARKERS.get(condition, "o"),
            linewidth=2.2,
            markersize=5.2,
            markeredgecolor="white",
            markeredgewidth=0.6,
            label=_plot_condition_label(condition),
            zorder=3,
            path_effects=[
                path_effects.Stroke(linewidth=3.4, foreground="white", alpha=0.8),
                path_effects.Normal(),
            ],
        )
    ax.axhline(0.0, color="#3F3F3F", linestyle=":", linewidth=1.2, zorder=2)
    ax.axvline(boundary_x, color="#3F3F3F", linewidth=1.1, alpha=0.55, zorder=2)
    ax.set_xticks(x, [SIGNED_BAND_LABELS[name] for name in SIGNED_BAND_NAMES])
    for label in ax.get_xticklabels():
        label.set_rotation(45)
        label.set_horizontalalignment("right")
    ax.grid(axis="y")
    ax.margins(x=0.05, y=0.12)


SIGNED_BAND_NOTE = (
    "Negative distances are inside the gland; the vertical rule marks the "
    "boundary. Lines show cohort medians, shading the 5th-95th percentiles. "
    "A profile peaking at the boundary and decaying both ways is boundary "
    "localization; one rising monotonically towards < -10 mm places its "
    "energy away from the boundary. The two cohorts need not share a "
    "shape. Distance is to the outer gland boundary only, so on the zones "
    "cohort the TZ/PZ interface is not resolved."
)


def _signed_band_figure(
    structure: pd.DataFrame,
    epsilons: list[int],
    output_dir: Path,
    *,
    filename: str,
    title: str,
    column_titles: bool,
    note_width: int,
    panel_width: float,
    panel_height: float,
    rect_bottom: float,
    ylims: list[tuple[float, float]] | None = None,
) -> list[tuple[float, float]]:
    """One signed-distance figure over the given epsilon columns.

    ``sharey="row"`` only ties the columns of a single figure together, so the
    per-epsilon figures would each rescale to their own budget and stop being
    comparable with one another.  Passing ``ylims`` pins them to the limits the
    pooled figure chose, which is what the shared axis buys in that figure.
    Returns the per-row limits actually used.
    """

    datasets = _datasets(structure)
    fig, axes = plt.subplots(
        len(datasets),
        len(epsilons),
        figsize=(panel_width * len(epsilons), panel_height * len(datasets)),
        squeeze=False,
        sharey="row",
    )
    for row, dataset in enumerate(datasets):
        dataset_rows = structure[structure["dataset"] == dataset]
        for column, epsilon_n in enumerate(epsilons):
            ax = axes[row, column]
            _draw_signed_band_panel(
                ax, dataset_rows[dataset_rows["epsilon_n"] == epsilon_n]
            )
            if row == 0 and column_titles:
                ax.set_title(f"\u03b5={epsilon_n}/255")
            if row == len(datasets) - 1:
                ax.set_xlabel("signed distance to boundary (mm)")
            if column == 0:
                ax.set_ylabel(
                    f"{DATASET_LABELS.get(dataset, dataset)}\n"
                    "enrichment relative to uniform (%)"
                )
        if ylims is not None:
            axes[row, 0].set_ylim(ylims[row])
    used_ylims = [tuple(axes[row, 0].get_ylim()) for row in range(len(datasets))]
    fig.suptitle(title, y=0.995)
    _shared_legend(fig, axes, anchor_y=0.945)
    _figure_note(fig, textwrap.fill(SIGNED_BAND_NOTE, note_width))
    _save(fig, output_dir / filename, rect=(0.0, rect_bottom, 1.0, 0.91))
    return used_ylims


def plot_signed_band_profile(structure: pd.DataFrame, output_dir: Path) -> None:
    """Enrichment against signed distance, deep interior to far exterior.

    ``12`` and ``17`` pool every voxel inside the mask into one ``inside`` band,
    so a positive value there is ambiguous: a perturbation spread evenly through
    the gland and one hugging the inner face of the boundary produce the same
    number.  Splitting that pool by depth separates them.  A profile that peaks
    at the boundary and decays inwards is boundary localization; one that stays
    flat or rises towards ``< -10 mm`` places its energy away from the boundary.

    The foreground is ``segmentation > 0``, so depth is measured to the outer
    gland boundary alone.  On the zones cohort that union covers TZ+CZ and PZ,
    which leaves the TZ/PZ interface unresolved: a deep band there means "far
    from the outer edge", not "in the gland core specifically".

    Writes the pooled figure across every budget, then one figure per budget so
    a single epsilon can be read without the neighbouring columns competing for
    the eye.
    """

    epsilons = sorted(int(value) for value in structure["epsilon_n"].dropna().unique())
    ylims = _signed_band_figure(
        structure,
        epsilons,
        output_dir,
        filename="20_signed_distance_band_profile.png",
        title=(
            "Perturbation energy by signed distance to the boundary, "
            "across budgets"
        ),
        column_titles=True,
        note_width=190,
        panel_width=3.7,
        panel_height=3.9,
        rect_bottom=0.05,
    )
    for epsilon_n in epsilons:
        _signed_band_figure(
            structure,
            [epsilon_n],
            output_dir,
            filename=f"20_signed_distance_band_profile_eps{epsilon_n}.png",
            title=(
                "Perturbation energy by signed distance to the boundary "
                f"(\u03b5={epsilon_n}/255)"
            ),
            column_titles=False,
            note_width=104,
            panel_width=8.6,
            panel_height=4.6,
            rect_bottom=0.10,
            ylims=ylims,
        )


# Each contrast is a per-case difference of two enrichment columns, so the
# geometry that both share -- gland size, shape, slice count -- cancels within
# the case rather than adding spread across the cohort.
SIGNED_BAND_CONTRASTS = (
    ("boundary_vs_deep_interior", "inside_0_2mm", "inside_beyond_10mm"),
    ("interior_vs_exterior_shell", "inside_0_2mm", "0_2mm"),
)


def _bootstrap_ci(
    values: np.ndarray, *, resamples: int = 10000, seed: int = 42
) -> tuple[float, float]:
    """Percentile bootstrap CI of the mean, matching the seed used elsewhere."""

    if values.size < 2:
        return (float("nan"), float("nan"))
    rng = np.random.default_rng(seed)
    draws = rng.integers(0, values.size, size=(resamples, values.size))
    means = values[draws].mean(axis=1)
    return (
        float(np.quantile(means, 0.025)),
        float(np.quantile(means, 0.975)),
    )


def signed_band_contrasts(structure: pd.DataFrame) -> pd.DataFrame:
    """Per-case signed-band contrasts with bootstrap CIs.

    Answers the question the pooled ``inside`` band cannot:
    ``boundary_vs_deep_interior`` is positive when energy concentrates on the
    inner face rather than through the gland; ``interior_vs_exterior_shell`` is
    zero when the boundary peak is symmetric about the mask edge; and the
    ``deep_interior`` row reports the deep band itself, whose reference is the
    RMS-matched control on the same row rather than another band.
    """

    rows: list[dict[str, object]] = []
    for (dataset, condition, epsilon_n), group in structure.groupby(
        ["dataset", "condition", "epsilon_n"], sort=True
    ):
        specs = [
            (name, group[f"enrichment_band_{a}"] - group[f"enrichment_band_{b}"])
            for name, a, b in SIGNED_BAND_CONTRASTS
        ]
        specs.append(
            ("deep_interior", group["enrichment_band_inside_beyond_10mm"] - 1.0)
        )
        for name, series in specs:
            values = series.to_numpy(dtype=float)
            values = values[np.isfinite(values)]
            low, high = _bootstrap_ci(values)
            rows.append(
                {
                    "dataset": dataset,
                    "condition": condition,
                    "epsilon_n": int(epsilon_n),
                    "contrast": name,
                    "n_cases": int(values.size),
                    "mean": float(np.mean(values)) if values.size else float("nan"),
                    "ci_low": low,
                    "ci_high": high,
                }
            )
    return pd.DataFrame(rows)


def plot_radial_spectra(
    spectra: pd.DataFrame, output_dir: Path, *, reference_epsilon: int
) -> None:
    selected = spectra[spectra["epsilon_n"] == reference_epsilon]
    datasets = _datasets(selected)
    cohort_counts = ", ".join(
        f"{DATASET_LABELS.get(dataset, dataset)} n="
        f"{selected.loc[selected['dataset'] == dataset, 'case_id'].nunique():,}"
        for dataset in datasets
    )
    fig, axes = plt.subplots(
        1,
        len(datasets),
        figsize=(6.6 * len(datasets), 4.8),
        squeeze=False,
        sharey=True,
    )
    for column, dataset in enumerate(datasets):
        ax = axes[0, column]
        data = selected[selected["dataset"] == dataset]
        for condition in _conditions(data):
            curve = (
                data[data["condition"] == condition]
                .groupby("f_r", as_index=False, sort=True)["power"]
                .mean()
            )
            ax.plot(
                curve["f_r"],
                np.maximum(curve["power"], np.finfo(float).tiny),
                color=COLORS.get(condition),
                linestyle=LINESTYLES.get(condition, "-"),
                linewidth=2.7,
                label=_plot_condition_label(condition),
                zorder=3,
                path_effects=[
                    path_effects.Stroke(linewidth=3.9, foreground="white", alpha=0.8),
                    path_effects.Normal(),
                ],
            )
        ax.set_yscale("log")
        ax.set_xlabel("Normalized radial frequency")
        ax.set_ylabel("Mean power (log scale)")
        ax.set_title(DATASET_LABELS.get(dataset, dataset), pad=30)
        ax.grid()
        ax.set_xlim(0.0, 1.0)
        _add_physical_frequency_axis(ax)
    fig.suptitle(
        f"Radial perturbation power spectra (ε={reference_epsilon}/255)", y=0.995
    )
    _shared_legend(fig, axes, anchor_y=0.92)
    _figure_note(
        fig,
        f"Curves average every evaluated case ({cohort_counts}). "
        "White-noise references remain approximately flat. Spectra are in-plane "
        "only, at a uniform 0.5 mm in-plane spacing, so the two axes are one "
        "exact relabel of each other.",
    )
    _save(
        fig,
        output_dir / "13_radial_power_spectra.png",
        rect=(0.0, 0.07, 1.0, 0.87),
    )


def _spectrum_quantiles(frame: pd.DataFrame) -> pd.DataFrame:
    """Per-frequency-bin median and 5th/95th percentiles of case power."""

    grouped = frame.groupby("freq_bin", sort=True)
    values = grouped["power"].agg(
        median="median",
        q05=lambda series: series.quantile(0.05),
        q95=lambda series: series.quantile(0.95),
    )
    values["f_r"] = grouped["f_r"].median()
    tiny = np.finfo(float).tiny
    for column in ("median", "q05", "q95"):
        values[column] = values[column].clip(lower=tiny)
    return values.reset_index()


def _plot_spectrum_band(ax: plt.Axes, frame: pd.DataFrame) -> None:
    """Draw one panel of median spectra with their 5th–95th-percentile bands."""

    for condition in _conditions(frame):
        values = _spectrum_quantiles(frame[frame["condition"] == condition])
        if values.empty:
            continue
        f_r = values["f_r"].to_numpy(dtype=float)
        color = COLORS.get(condition)
        ax.fill_between(
            f_r,
            values["q05"].to_numpy(dtype=float),
            values["q95"].to_numpy(dtype=float),
            color=color,
            alpha=0.085,
            linewidth=0,
            zorder=1,
        )
        ax.plot(
            f_r,
            values["median"].to_numpy(dtype=float),
            color=color,
            linestyle=LINESTYLES.get(condition, "-"),
            linewidth=2.4,
            label=_plot_condition_label(condition),
            zorder=3,
            path_effects=[
                path_effects.Stroke(linewidth=3.6, foreground="white", alpha=0.8),
                path_effects.Normal(),
            ],
        )
    ax.set_yscale("log")
    ax.grid()
    ax.set_xlim(0.0, 1.0)


def plot_spectra_variability(
    spectra: pd.DataFrame, output_dir: Path, *, reference_epsilon: int
) -> None:
    """Full-cohort spectra with the spread figure 13 collapses into a mean.

    Figure 13 plots one averaged curve per condition, so it cannot show whether
    a cohort agrees on the spectral shape. Same data, same budget, drawn as
    median and 5th–95th-percentile envelope.
    """

    selected = spectra[spectra["epsilon_n"] == reference_epsilon]
    datasets = _datasets(selected)
    cohort_counts = ", ".join(
        f"{DATASET_LABELS.get(dataset, dataset)} n="
        f"{selected.loc[selected['dataset'] == dataset, 'case_id'].nunique():,}"
        for dataset in datasets
    )
    fig, axes = plt.subplots(
        1, len(datasets), figsize=(6.6 * len(datasets), 4.8), squeeze=False
    )
    for column, dataset in enumerate(datasets):
        ax = axes[0, column]
        _plot_spectrum_band(ax, selected[selected["dataset"] == dataset])
        ax.set_xlabel("Normalized radial frequency")
        ax.set_ylabel("Power (log scale)")
        ax.set_title(DATASET_LABELS.get(dataset, dataset), pad=30)
        _add_physical_frequency_axis(ax)
    fig.suptitle(
        f"Radial perturbation power spectra, cohort spread (ε={reference_epsilon}/255)",
        y=0.995,
    )
    _shared_legend(fig, axes, anchor_y=0.92)
    _figure_note(
        fig,
        f"Lines show cohort medians ({cohort_counts}); shading spans the "
        "5th–95th percentiles across cases. Figure 13 plots the mean of the "
        "same data. Spectra are in-plane only, at a uniform 0.5 mm in-plane "
        "spacing, so the two axes are one exact relabel of each other.",
    )
    _save(
        fig,
        output_dir / "18_radial_power_spectra_variability.png",
        rect=(0.0, 0.07, 1.0, 0.87),
    )


def plot_spectra_by_epsilon(
    subsample: pd.DataFrame, output_dir: Path, *, reference_epsilon: int
) -> None:
    """Spectral shape across every budget, on the retained spectra subsample.

    Full-cohort spectra were backfilled at one budget only, so the ε sweep is
    available only for the deterministic per-cohort subsample. The panel at the
    reference budget overlaps figure 18 and is what licenses reading the rest.
    """

    datasets = _datasets(subsample)
    epsilons = sorted(int(value) for value in subsample["epsilon_n"].dropna().unique())
    case_counts = ", ".join(
        f"{DATASET_LABELS.get(dataset, dataset)} n="
        f"{subsample.loc[subsample['dataset'] == dataset, 'case_id'].nunique():,}"
        for dataset in datasets
    )
    fig, axes = plt.subplots(
        len(datasets),
        len(epsilons),
        figsize=(3.5 * len(epsilons), 3.9 * len(datasets)),
        squeeze=False,
        sharey="row",
    )
    for row, dataset in enumerate(datasets):
        dataset_rows = subsample[subsample["dataset"] == dataset]
        for column, epsilon_n in enumerate(epsilons):
            ax = axes[row, column]
            _plot_spectrum_band(
                ax, dataset_rows[dataset_rows["epsilon_n"] == epsilon_n]
            )
            ax.set_xlabel("Normalized radial frequency")
            if row == 0:
                # Every panel shares this x-axis; ten physical axes would be
                # clutter, so only the top row carries the cycles/mm reading.
                ax.set_title(f"ε={epsilon_n}/255", pad=30)
                _add_physical_frequency_axis(ax)
            if column == 0:
                ax.set_ylabel(
                    f"{DATASET_LABELS.get(dataset, dataset)}\npower (log scale)"
                )
    fig.suptitle("Radial perturbation power spectra, across budgets", y=0.995)
    _shared_legend(fig, axes, anchor_y=0.945)
    _figure_note(
        fig,
        f"Medians and 5th–95th percentiles over the retained spectra subsample "
        f"({case_counts}), so the bands are coarser than figure 18's "
        f"full-cohort ones. At ε={reference_epsilon}/255 the subsample median "
        "tracks the full cohort within 0.07 (WG) and 0.20 (zones) in absolute "
        "log ratio across all 32 bins. The cycles/mm reading on the top row "
        "applies to every panel.",
    )
    _save(
        fig,
        output_dir / "19_radial_power_spectra_by_epsilon.png",
        rect=(0.0, 0.04, 1.0, 0.91),
    )


def plot_spectral_scalars(structure: pd.DataFrame, output_dir: Path) -> None:
    specifications = [
        ("spectral_centroid_norm", "Spectral centroid", "Normalized frequency"),
        (
            "hf_mean_power_fraction",
            "High-frequency mean power",
            "Fraction of mean-power profile",
        ),
        ("spectral_slope", "Log–log spectral slope", "Slope"),
    ]
    datasets = _datasets(structure)
    fig, axes = plt.subplots(
        len(datasets), 3, figsize=(15.6, 4.35 * len(datasets)), squeeze=False
    )
    for row, dataset in enumerate(datasets):
        data = structure[structure["dataset"] == dataset]
        for column, (metric, title, ylabel) in enumerate(specifications):
            _plot_metric_vs_epsilon(
                axes[row, column],
                data,
                metric,
                title=f"{DATASET_LABELS.get(dataset, dataset)} — {title}",
                ylabel=ylabel,
            )
    fig.suptitle("Spectral structure across perturbation budgets", y=0.995)
    _shared_legend(fig, axes, anchor_y=0.957)
    _figure_note(
        fig,
        "Lines show cohort medians; shading spans the 5th–95th percentiles. "
        "White-noise references are centroid=0.5, high-frequency fraction=0.5, slope=0.",
        y=0.006,
    )
    _save(
        fig,
        output_dir / "14_spectral_scalars_vs_epsilon.png",
        rect=(0.0, 0.04, 1.0, 0.92),
    )


def plot_alignment(structure: pd.DataFrame, output_dir: Path) -> None:
    specifications = [
        (
            "spearman_absdelta_gradmag",
            "Rank alignment with image gradient",
            "Spearman ρ",
        ),
        (
            "edge_energy_enrichment",
            "Energy enrichment on top-decile edges",
            "Enrichment",
        ),
    ]
    datasets = _datasets(structure)
    fig, axes = plt.subplots(
        len(datasets), 2, figsize=(12.4, 4.35 * len(datasets)), squeeze=False
    )
    for row, dataset in enumerate(datasets):
        data = structure[structure["dataset"] == dataset]
        for column, (metric, title, ylabel) in enumerate(specifications):
            ax = axes[row, column]
            is_enrichment = metric == "edge_energy_enrichment"
            _plot_metric_vs_epsilon(
                ax,
                data,
                metric,
                title=f"{DATASET_LABELS.get(dataset, dataset)} — {title}",
                ylabel=(
                    "Enrichment relative to uniform (%)" if is_enrichment else ylabel
                ),
                reference=(1.0 if is_enrichment else 0.0),
                scale=(100.0 if is_enrichment else 1.0),
            )
            if is_enrichment:
                ax.axhline(
                    0.0,
                    color="#3F3F3F",
                    linestyle=":",
                    linewidth=1.2,
                    zorder=2,
                )
    fig.suptitle("Perturbation alignment with native image structure", y=0.995)
    _shared_legend(fig, axes, anchor_y=0.957)
    _figure_note(
        fig,
        "Lines show cohort medians; shading spans the 5th–95th percentiles. "
        "Near-saturated FGSM rows limit the interpretability of both alignment metrics.",
        y=0.006,
    )
    _save(
        fig,
        output_dir / "15_structure_alignment.png",
        rect=(0.0, 0.04, 1.0, 0.92),
    )


def plot_gmsd_vs_ssim(joined: pd.DataFrame, output_dir: Path) -> None:
    """Show how cohort-median image change grows with attack strength."""

    datasets = _datasets(joined)
    fig, axes = plt.subplots(
        1, len(datasets), figsize=(6.2 * len(datasets), 4.8), squeeze=False
    )
    attacks_all = joined[joined["condition"].isin(ATTACK_CONDITIONS)]
    epsilon_levels = sorted(attacks_all["epsilon_n"].dropna().astype(int).unique())
    if not epsilon_levels:
        raise ValueError("GMSD-versus-SSIM plot has no attack epsilon values")

    for column, dataset in enumerate(datasets):
        ax = axes[0, column]
        data = joined[joined["dataset"] == dataset]
        attacks = data[data["condition"].isin(ATTACK_CONDITIONS)]
        epsilon_points: dict[int, list[tuple[float, float]]] = {
            epsilon_n: [] for epsilon_n in epsilon_levels
        }
        for condition in ATTACK_CONDITIONS:
            group = attacks[attacks["condition"] == condition]
            trajectory: list[dict[str, float]] = []
            for epsilon_n in epsilon_levels:
                epsilon_group = group[group["epsilon_n"] == epsilon_n]
                finite = np.isfinite(
                    epsilon_group[
                        ["ssim_axial_prostate", "gmsd_axial_prostate"]
                    ].to_numpy(dtype=float)
                ).all(axis=1)
                epsilon_group = epsilon_group[finite]
                if epsilon_group.empty:
                    continue
                ssim = epsilon_group["ssim_axial_prostate"].to_numpy(dtype=float)
                gmsd = epsilon_group["gmsd_axial_prostate"].to_numpy(dtype=float)
                trajectory.append(
                    {
                        "epsilon_n": float(epsilon_n),
                        "ssim": float(np.median(ssim)),
                        "gmsd": float(np.median(gmsd)),
                    }
                )
            if not trajectory:
                continue
            trajectory_frame = pd.DataFrame(trajectory).sort_values("epsilon_n")
            ax.plot(
                trajectory_frame["ssim"],
                trajectory_frame["gmsd"],
                color=COLORS[condition],
                linestyle=LINESTYLES[condition],
                linewidth=2.7,
                marker=MARKERS[condition],
                markersize=7.0,
                markerfacecolor="white",
                markeredgecolor=COLORS[condition],
                markeredgewidth=1.7,
                label=_plot_condition_label(condition),
                zorder=4,
                path_effects=[
                    path_effects.Stroke(linewidth=4.0, foreground="white", alpha=0.85),
                    path_effects.Normal(),
                ],
            )
            last = trajectory_frame.iloc[-1]
            for point in trajectory_frame.itertuples(index=False):
                epsilon_points[int(point.epsilon_n)].append(
                    (float(point.ssim), float(point.gmsd))
                )
            if len(trajectory_frame) >= 2:
                previous = trajectory_frame.iloc[-2]
                ax.annotate(
                    "",
                    xy=(last["ssim"], last["gmsd"]),
                    xytext=(previous["ssim"], previous["gmsd"]),
                    arrowprops={
                        "arrowstyle": "-|>",
                        "color": COLORS[condition],
                        "linewidth": 2.0,
                        "mutation_scale": 11,
                        "shrinkA": 8,
                        "shrinkB": 8,
                    },
                    zorder=6,
                )

        label_offsets = [
            (5, 8),
            (-30, 10),
            (7, 10),
            (7, 10),
            (7, 12),
        ]
        for index, epsilon_n in enumerate(epsilon_levels):
            points = epsilon_points[epsilon_n]
            if not points:
                continue
            center = np.median(np.asarray(points), axis=0)
            offset = label_offsets[min(index, len(label_offsets) - 1)]
            ax.annotate(
                f"ε={epsilon_n}",
                xy=center,
                xytext=offset,
                textcoords="offset points",
                fontsize=8,
                fontweight="bold",
                bbox={
                    "boxstyle": "round,pad=0.12",
                    "facecolor": "white",
                    "edgecolor": "none",
                    "alpha": 0.88,
                },
                arrowprops={
                    "arrowstyle": "-",
                    "color": "#777777",
                    "linewidth": 0.7,
                    "shrinkA": 2,
                    "shrinkB": 3,
                },
                zorder=7,
            )
        ax.set_xlabel("SSIM (higher = more similar)")
        ax.set_ylabel("GMSD (higher = more change)")
        ax.set_title(f"({chr(97 + column)})  {DATASET_LABELS.get(dataset, dataset)}")
        ax.grid()
        left, _right = ax.get_xlim()
        ax.set_xlim(left, 1.02)
    axes[0, 0].plot(
        [],
        [],
        linestyle="none",
        marker=r"$\epsilon$",
        markersize=10,
        color="#555555",
        label="ε label = budget (n/255)",
    )
    axes[0, 0].plot(
        [],
        [],
        color="#555555",
        linewidth=1.6,
        marker=">",
        markersize=6,
        label="Arrow = increasing ε",
    )
    _shared_legend(
        fig,
        axes,
        ncol=5,
        anchor_y=0.93,
        title="How to read the plot",
    )
    fig.suptitle("Stronger attacks cause greater image change", y=0.995)
    _figure_note(
        fig,
        "Markers show full-cohort medians; every attack uses the same five budgets.",
        y=0.018,
    )
    fig.subplots_adjust(top=0.74, bottom=0.18, wspace=0.30)
    _save(fig, output_dir / "16_gmsd_vs_ssim.png", tight=False)


def reproduction_summary(structure: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    attacks = structure[structure["condition"].isin(ATTACK_CONDITIONS)]
    for (dataset, condition), group in attacks.groupby(
        ["dataset", "condition"], sort=True
    ):
        values = group["rms_reproduction_rel_error"].to_numpy(dtype=float)
        values = values[np.isfinite(values)]
        if not values.size:
            continue
        rows.append(
            {
                "dataset": dataset,
                "condition": condition,
                "n_rows": int(values.size),
                "median_relative_error": float(np.median(values)),
                "q95_relative_error": float(np.quantile(values, 0.95)),
                "max_relative_error": float(np.max(values)),
            }
        )
    return pd.DataFrame(rows)


def _markdown_table(frame: pd.DataFrame) -> str:
    if frame.empty:
        return "_No rows available._"
    display = frame.copy()
    for column in display.select_dtypes(include=[np.number]).columns:
        display[column] = display[column].map(
            lambda value: "—" if not np.isfinite(value) else f"{value:.5g}"
        )
    header = "| " + " | ".join(str(column) for column in display.columns) + " |"
    divider = "| " + " | ".join("---" for _ in display.columns) + " |"
    rows = [
        "| " + " | ".join(str(value) for value in record) + " |"
        for record in display.itertuples(index=False, name=None)
    ]
    return "\n".join([header, divider, *rows])


def _family_table(
    summary: pd.DataFrame,
    metrics: list[str],
    *,
    reference_epsilon: int,
) -> pd.DataFrame:
    selected = summary[summary["epsilon_n"] == reference_epsilon]
    columns = ["dataset", "condition", "n_cases"]
    columns.extend(f"{metric}_median" for metric in metrics)
    available = [column for column in columns if column in selected.columns]
    return selected[available].sort_values(["dataset", "condition"])


def _labeled(table: pd.DataFrame) -> pd.DataFrame:
    """Route the condition column through the same map the figure legends use.

    The matched Gaussian control is uniform and white *by construction*, so its
    enrichment of 1.0 and flat spectrum are analytic references rather than
    findings.  That caveat is the single most important framing in this
    deliverable, so it belongs in the written tables and not only in the
    blockquote above them.
    """

    if "condition" not in table.columns:
        return table
    return table.assign(condition=table["condition"].map(_condition_label))


def _with_saturation(
    table: pd.DataFrame, saturation: pd.DataFrame, *, reference_epsilon: int
) -> pd.DataFrame:
    """Attach median ε-ball saturation to the Family C table.

    Reported beside the alignment descriptors so a value sitting on the Gaussian
    reference line is never read as "this attack is as unstructured as noise"
    when the real explanation is that a saturated |δ| carries no structure for
    the descriptor to measure.
    """

    if saturation.empty or SATURATION_MEDIAN_COLUMN not in saturation.columns:
        return table
    right = saturation.loc[
        saturation["epsilon_n"] == reference_epsilon,
        ["dataset", "condition", SATURATION_MEDIAN_COLUMN],
    ]
    return table.merge(right, on=["dataset", "condition"], how="left")


def _signed_contrast_section(
    structure: pd.DataFrame, *, reference_epsilon: int
) -> list[str]:
    """The signed-band contrast table, or nothing for a pre-interior-band table.

    Reported for the attacks only.  The RMS-matched controls are uniform over
    the mask by construction, so their contrasts are zero analytically and
    belong in the CSV as a check rather than in the report as a result.
    """

    if not has_signed_bands(structure):
        return []
    contrasts = signed_band_contrasts(structure)
    contrasts = contrasts[
        (contrasts["epsilon_n"] == reference_epsilon)
        & (~contrasts["condition"].str.contains("matched"))
    ]
    if contrasts.empty:
        return []
    table = _labeled(
        contrasts.drop(columns=["epsilon_n"]).rename(
            columns={"ci_low": "ci_low_95", "ci_high": "ci_high_95"}
        )
    )
    return [
        f"Per-case contrasts at ε={reference_epsilon}/255, mean with a "
        "percentile-bootstrap 95% CI. A positive `boundary_vs_deep_interior` "
        "with a `deep_interior` at zero is a boundary shell; a negative one "
        "with `deep_interior` above zero places the energy away from the "
        "boundary instead. The two cohorts do not have to agree, and the "
        "pooled `inside` band above cannot tell them apart.",
        "",
        "> The foreground is `segmentation > 0`, so the signed distance is "
        "measured to the *outer* gland boundary and to nothing else. On the "
        "zones cohort that union spans TZ+CZ and PZ together, so the TZ/PZ "
        "interface — itself a decision boundary, and one lying a few "
        "millimetres inside the outer edge — is invisible to these bands. "
        "Read a negative `boundary_vs_deep_interior` there as energy placed "
        "away from the outer boundary, not as evidence that the deep gland "
        "core specifically is the target.",
        "",
        _markdown_table(table),
        "",
    ]


def write_report(
    structure: pd.DataFrame,
    summary: pd.DataFrame,
    paired: pd.DataFrame,
    reproduction: pd.DataFrame,
    saturation: pd.DataFrame,
    output_dir: Path,
    *,
    reference_epsilon: int,
    spectra_name: str = SPECTRA_CSV,
) -> Path:
    """Write the descriptive perturbation-structure report."""

    cohort_parts = []
    for dataset in _datasets(structure):
        count = int(structure.loc[structure["dataset"] == dataset, "case_id"].nunique())
        noun = "case" if count == 1 else "cases"
        cohort_parts.append(f"{count:,} {DATASET_LABELS.get(dataset, dataset)} {noun}")
    epsilon_values = sorted(structure["epsilon_n"].astype(int).unique())
    lines = [
        "# Perturbation Structure Characterization",
        "",
        "## Scope and setup",
        "",
        f"This analysis covers {' and '.join(cohort_parts)} at ε numerators "
        f"{epsilon_values} over 255. It characterizes perturbation frequency, "
        "anatomical localization, alignment with image gradients, and axial GMSD.",
        "",
        "> Matched Gaussian noise is uniform over the valid mask and white by construction, so its localization enrichment of 1.0 and its flat spectrum are analytic references, not findings. The informative quantity is the adversarial value measured against them.",
        "",
        "The tables below show medians at "
        f"ε={reference_epsilon}/255; the CSV summaries retain every requested ε.",
        "",
        "### RMS reproduction check",
        "",
        "Attack RMS values were recomputed from regenerated perturbations and compared "
        "with the published image-quality rows. Relative-error distributions are:",
        "",
        _markdown_table(reproduction),
        "",
        "## Results",
        "",
        "### Family A — spectral organization",
        "",
        _markdown_table(
            _labeled(
                _family_table(
                    summary,
                    [
                        "spectral_centroid_norm",
                        "hf_mean_power_fraction",
                        "spectral_slope",
                    ],
                    reference_epsilon=reference_epsilon,
                )
            )
        ),
        "",
        "",
        "Both scalars are computed from the radially binned *per-bin mean* power, "
        "not from raw coefficient energy: outer bins hold most of the "
        "coefficients, so a count-weighted version would track bin population "
        "rather than spectral shape. `hf_mean_power_fraction` is accordingly the "
        "share of the mean-power profile above half Nyquist — 0.5 for a white "
        "field — and not the share of the perturbation's total energy above half "
        "Nyquist. Both are read against the white reference in the same table.",
        "",
        f"Figure 13 averages radial power spectra over every case in each cohort "
        f"at ε={reference_epsilon}/255. See `13_radial_power_spectra.png` and "
        "`14_spectral_scalars_vs_epsilon.png`. "
        "`18_radial_power_spectra_variability.png` redraws the same cohort as "
        "medians with 5th–95th-percentile bands, and "
        "`19_radial_power_spectra_by_epsilon.png` repeats the profile at every "
        "budget on the retained spectra subsample — full-cohort spectra were "
        f"backfilled at ε={reference_epsilon}/255 only. At that budget the "
        "subsample median tracks the full cohort within 0.07 (WG) and 0.20 "
        "(zones) in absolute log ratio across all 32 radial bins.",
        "",
        "> **Frequency units.** Spectra are computed in-plane, per axial slice, "
        "never as a radially averaged 3-D transform: voxels are "
        "(3.0, 0.5, 0.5) mm, so a 3-D radial average would pool frequencies six "
        "times apart. Slice thickness therefore never enters the frequency "
        "axis. nnU-Net resamples both cohorts to that same target spacing and "
        "all 1,999 preprocessed cases carry it exactly, so the normalized axis "
        "and cycles/mm are related by one global constant: a plotted radius r "
        "is r x sqrt(2) cycles/mm, putting axis-Nyquist (1.0 cycles/mm) at "
        "r = 0.707 and the corner of the frequency plane at 1.41 cycles/mm. "
        "Figures 13, 18 and 19 carry both axes. The scalars in the table above "
        "stay normalized, because the white reference of 0.5 is defined there.",
        "",
        "### Family B — anatomical localization",
        "",
        _markdown_table(
            _labeled(
                _family_table(
                    summary,
                    [
                        "foreground_energy_enrichment",
                        "enrichment_band_0_2mm",
                        "enrichment_band_2_5mm",
                    ],
                    reference_epsilon=reference_epsilon,
                )
            )
        ),
        "",
        "See `11_localization_enrichment_vs_epsilon.png` and "
        "`12_boundary_band_enrichment.png`. "
        "`17_boundary_band_enrichment_by_epsilon.png` repeats the band profile "
        "at every budget with 5th–95th-percentile cohort spread.",
        "",
        "Both of those pool every voxel inside the mask into one `inside` band, "
        "so a positive value there cannot separate a perturbation spread through "
        "the gland from one hugging the inner face of the boundary. "
        "`20_signed_distance_band_profile.png` splits that pool by depth and "
        "plots the full signed profile, and `"
        + SIGNED_BAND_CONTRAST_CSV
        + "` reports the per-case contrasts that decide between the two: "
        "`boundary_vs_deep_interior` is positive when energy concentrates on the "
        "inner face, `interior_vs_exterior_shell` is zero when the peak is "
        "symmetric about the mask edge, and `deep_interior` is the deep band "
        "itself against its RMS-matched control. Both appear only for tables "
        "regenerated with the interior bands.",
        "",
        *_signed_contrast_section(structure, reference_epsilon=reference_epsilon),
        "### Family C — alignment with image structure",
        "",
        "> Read this table together with its saturation column. Both descriptors "
        "are functions of |δ|, so an attack that saturates the ε-ball has |δ| = ε "
        "everywhere and offers the descriptor no structure to measure: Spearman ρ "
        "is undefined and edge enrichment is exactly 1.0 — the uniform-noise "
        "value. A near-reference value at high saturation means the descriptor is "
        "blind, not that the attack is unstructured.",
        "",
        _markdown_table(
            _labeled(
                _with_saturation(
                    _family_table(
                        summary,
                        ["spearman_absdelta_gradmag", "edge_energy_enrichment"],
                        reference_epsilon=reference_epsilon,
                    ),
                    saturation,
                    reference_epsilon=reference_epsilon,
                )
            )
        ),
        "",
        "See `15_structure_alignment.png`.",
        "",
        "### Family D — gradient-sensitive perceptual change",
        "",
        _markdown_table(
            _labeled(
                _family_table(
                    summary,
                    ["gmsd_axial_prostate"],
                    reference_epsilon=reference_epsilon,
                )
            )
        ),
        "",
        "See `16_gmsd_vs_ssim.png`. Each trajectory connects the full-cohort "
        "attack medians from ε=2 to ε=32; stronger image change appears as lower "
        "SSIM and higher GMSD.",
        "",
        "## Discussion",
        "",
        "These descriptors are descriptive rather than causal. Paired differences "
        "are computed within case against the RMS-matched Gaussian reference; they "
        "control perturbation magnitude but do not turn the reference construction "
        "into an empirical discovery. Boundary bands use physical millimetres, and "
        "spectra use valid, foreground-centred in-plane crops.",
        "",
        "**Limitation — Family C cannot see structure in a saturated "
        "perturbation.** Both alignment descriptors are functions of |δ|. An "
        "ε-ball attack that saturates drives |δ| to the constant ε, at which "
        "point the rank correlation is undefined (reported as NaN and excluded "
        "from the summaries) and the edge-energy enrichment is exactly 1.0, "
        "numerically identical to the uniform-noise reference. FGSM saturates "
        "almost completely — its published median ε-saturation is 0.9988 on the "
        "whole gland and 0.99997 on the zones — so its Family C values sitting on "
        "the Gaussian reference line in `15_structure_alignment.png` must not be "
        'read as "FGSM is as unstructured as noise". Saturation also falls with '
        "ε (WG FGSM: 70% of rows above 99.9% saturation at ε=2 versus 12% at "
        "ε=32), so part of the apparent ε-trend in this family tracks saturation "
        "rather than structure. The saturation column in the Family C table above "
        "makes the affected rows identifiable; Families A, B and D do not share "
        "this failure mode.",
        "",
        "Spectral scalars are computed after removing each crop's mean. A "
        "constant offset is a global brightness shift that carries no "
        "spatial-organization information, but its energy lands entirely in the "
        "DC coefficient and hence in radial bin 0, which holds few enough "
        "coefficients that it would otherwise dominate both the centroid and the "
        "high-frequency fraction — for a near-saturated field such as FGSM's, "
        "turning them into a function of sign imbalance rather than of structure.",
        "",
        f"The paired machine-readable table contains {len(paired):,} grouped "
        "condition-versus-Gaussian comparisons.",
        "",
        "## Machine-readable outputs",
        "",
        f"- `{STRUCTURE_CSV}` — per-case scalar descriptors",
        f"- `{spectra_name}` — full-cohort long-form radial spectra at "
        f"ε={reference_epsilon}/255",
        f"- `{SUMMARY_CSV}` — mean, median, 5th, and 95th percentiles",
        f"- `{PAIRED_CSV}` — within-case condition-minus-Gaussian comparisons",
        f"- `{REPRODUCTION_CSV}` — RMS reproduction error distributions",
        "",
    ]
    path = output_dir / REPORT_NAME
    path.write_text("\n".join(lines), encoding="utf-8")
    return path


def require_reference_epsilon(
    frame: pd.DataFrame, reference_epsilon: int, *, label: str
) -> None:
    """Every table a reference-epsilon figure reads must contain that epsilon.

    A frame that lacks it selects to nothing, and the per-dataset figures then
    call ``plt.subplots(1, 0)`` and fail inside matplotlib rather than telling
    the operator which table is short.
    """

    if int(reference_epsilon) not in set(frame["epsilon_n"].astype(int)):
        raise ValueError(
            f"reference epsilon {reference_epsilon} is absent from {label}"
        )


def require_full_spectra_coverage(
    structure: pd.DataFrame,
    spectra: pd.DataFrame,
    *,
    reference_epsilon: int,
) -> None:
    """Require one complete radial spectrum for every full-cohort case/condition.

    Figure 13 used to read a deterministic 50-case-per-cohort subsample. Exact
    key equality against the scalar table prevents that silent downsampling from
    recurring and catches partial/resumed spectra files before plotting.
    """

    structure_selected = structure[
        structure["epsilon_n"].astype(int) == int(reference_epsilon)
    ]
    spectra_selected = spectra[
        spectra["epsilon_n"].astype(int) == int(reference_epsilon)
    ]
    case_keys = ["dataset", "case_id", "condition"]
    expected = set(
        structure_selected[case_keys]
        .drop_duplicates()
        .itertuples(index=False, name=None)
    )
    observed = set(
        spectra_selected[case_keys].drop_duplicates().itertuples(index=False, name=None)
    )
    missing = expected - observed
    extra = observed - expected
    if missing or extra:
        raise ValueError(
            "radial spectra do not cover the full scalar cohort at "
            f"epsilon {reference_epsilon}: {len(missing)} missing and "
            f"{len(extra)} extra case-condition keys"
        )

    bin_keys = [*case_keys, "freq_bin"]
    duplicates = int(spectra_selected.duplicated(bin_keys).sum())
    if duplicates:
        raise ValueError(
            f"radial spectra contain {duplicates} duplicate case-condition-bin rows"
        )
    expected_bins = int(spectra_selected["freq_bin"].nunique())
    bin_counts = spectra_selected.groupby(case_keys, sort=False)["freq_bin"].nunique()
    if expected_bins < 2 or bin_counts.empty or not bin_counts.eq(expected_bins).all():
        incomplete = (
            int((~bin_counts.eq(expected_bins)).sum()) if not bin_counts.empty else 0
        )
        raise ValueError(
            "radial spectra contain incomplete frequency grids: "
            f"{incomplete} case-condition keys differ from {expected_bins} bins"
        )


def _required_columns(frame: pd.DataFrame, required: set[str], *, label: str) -> None:
    missing = sorted(required - set(frame.columns))
    if missing:
        raise ValueError(f"{label} is missing required columns: {missing}")
    if frame.empty:
        raise ValueError(f"{label} contains no rows")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Summarize and plot perturbation-structure evaluation outputs"
    )
    parser.add_argument("--structure-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--quality-csv", type=Path, default=DEFAULT_QUALITY_CSV)
    parser.add_argument(
        "--spectra-csv",
        type=Path,
        default=None,
        help=(
            "full-cohort radial spectra table; defaults to "
            f"<structure-dir>/{SPECTRA_CSV}"
        ),
    )
    parser.add_argument("--output-dir", type=Path, default=None)
    parser.add_argument("--reference-epsilon", type=int, default=16)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.reference_epsilon <= 0:
        raise ValueError("--reference-epsilon must be positive")
    structure_dir = args.structure_dir.resolve()
    structure_path = structure_dir / STRUCTURE_CSV
    spectra_path = (
        args.spectra_csv.resolve()
        if args.spectra_csv is not None
        else structure_dir / SPECTRA_CSV
    )
    if not structure_path.is_file():
        raise FileNotFoundError(f"missing perturbation structure CSV: {structure_path}")
    if not spectra_path.is_file():
        raise FileNotFoundError(f"missing radial spectra CSV: {spectra_path}")
    quality_path = args.quality_csv.resolve()
    if not quality_path.is_file():
        raise FileNotFoundError(f"missing adversarial quality CSV: {quality_path}")

    structure = pd.read_csv(structure_path)
    spectra = pd.read_csv(spectra_path)
    quality = pd.read_csv(quality_path)
    # Optional: only the epsilon sweep needs it, and it predates the
    # full-cohort backfill, so a missing file must not fail the run.
    subsample_path = structure_dir / SUBSAMPLE_SPECTRA_CSV
    subsample = pd.read_csv(subsample_path) if subsample_path.is_file() else None
    if subsample is None:
        print(f"skipping epsilon-sweep spectra: {subsample_path} not found", flush=True)
    _required_columns(
        structure,
        {"dataset", "case_id", "epsilon_n", "condition", *ALL_METRICS},
        label=str(structure_path),
    )
    _required_columns(
        spectra,
        {"dataset", "case_id", "epsilon_n", "condition", "freq_bin", "f_r", "power"},
        label=str(spectra_path),
    )
    require_reference_epsilon(structure, args.reference_epsilon, label="structure data")
    require_reference_epsilon(spectra, args.reference_epsilon, label="spectra data")
    require_full_spectra_coverage(
        structure,
        spectra,
        reference_epsilon=args.reference_epsilon,
    )

    output_dir = (
        args.output_dir.resolve() if args.output_dir is not None else structure_dir
    )
    output_dir.mkdir(parents=True, exist_ok=True)
    joined = join_quality(structure, quality)
    summary = summarize(structure, ALL_METRICS).sort_values(
        ["dataset", "condition", "epsilon_n"]
    )
    summary.to_csv(output_dir / SUMMARY_CSV, index=False)

    paired_frames: list[pd.DataFrame] = []
    for metric in PAIRED_METRICS:
        frame = paired_vs_control(structure, metric)
        if not frame.empty:
            frame.insert(0, "metric", metric)
            paired_frames.append(frame)
    paired = (
        pd.concat(paired_frames, ignore_index=True) if paired_frames else pd.DataFrame()
    )
    paired.to_csv(output_dir / PAIRED_CSV, index=False)
    # Only tables regenerated after commit 96809eb carry the interior bands.
    if has_signed_bands(structure):
        signed_band_contrasts(structure).to_csv(
            output_dir / SIGNED_BAND_CONTRAST_CSV, index=False
        )
    else:
        print(
            "skipping signed-distance bands: this table predates them",
            flush=True,
        )
    reproduction = reproduction_summary(structure)
    reproduction.to_csv(output_dir / REPRODUCTION_CSV, index=False)
    # Summarized from ``joined`` rather than ``structure``: saturation is a
    # published quality column, so only the merge above puts it on attack rows.
    saturation = summarize(joined, [SATURATION_COLUMN])

    _configure_style()
    plot_localization(structure, output_dir)
    plot_boundary_bands(structure, output_dir, reference_epsilon=args.reference_epsilon)
    plot_boundary_bands_by_epsilon(structure, output_dir)
    if has_signed_bands(structure):
        plot_signed_band_profile(structure, output_dir)
    plot_radial_spectra(spectra, output_dir, reference_epsilon=args.reference_epsilon)
    plot_spectra_variability(
        spectra, output_dir, reference_epsilon=args.reference_epsilon
    )
    if subsample is not None:
        plot_spectra_by_epsilon(
            subsample, output_dir, reference_epsilon=args.reference_epsilon
        )
    plot_spectral_scalars(structure, output_dir)
    plot_alignment(structure, output_dir)
    plot_gmsd_vs_ssim(joined, output_dir)
    report = write_report(
        structure,
        summary,
        paired,
        reproduction,
        saturation,
        output_dir,
        reference_epsilon=args.reference_epsilon,
        spectra_name=spectra_path.name,
    )
    print(f"Saved analysis outputs to {output_dir}", flush=True)
    print(f"Report: {report}", flush=True)


if __name__ == "__main__":
    main()
