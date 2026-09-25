"""Plot the relationship between APGD magnitude, PSNR, and SSIM.

The full-cohort evaluator output is preferred when it exists; the original
calibration subset remains supported through ``--quality-csv``. This script
does not rerun attacks.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib
import numpy as np
import pandas as pd
from matplotlib.colors import BoundaryNorm, ListedColormap

matplotlib.use("Agg")
import matplotlib.pyplot as plt


REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_RESULTS = (
    REPO_ROOT / "results" / "adv_rob_eval_results" / "image_quality_analysis"
)
DEFAULT_ALL_CASES_CSV = (
    DEFAULT_RESULTS
    / "all_cases_adversarial_quality"
    / "adversarial_quality_all_cases.csv"
)
DEFAULT_SUBSET_CSV = DEFAULT_RESULTS / "perturbation_quality_per_case.csv"
DEFAULT_OUTPUT = DEFAULT_RESULTS / "psnr_ssim_all_cases_figures"
EPSILON_VALUES = [2, 4, 8, 16, 32]
EPSILON_AXIS_VALUES = [0, *EPSILON_VALUES]
EPSILON_POSITIONS = {
    epsilon_n: index for index, epsilon_n in enumerate(EPSILON_AXIS_VALUES)
}
DATASET_LABELS = {"wg": "Whole gland (WG)", "zones": "Prostate zones"}
DATASET_COLORS = {"wg": "#6A4C93", "zones": "#E07A1F"}
EPSILON_COLORS = ["#3B4CC0", "#2C7FB8", "#41AB5D", "#F0A202", "#D1495B"]


def _configure_style() -> None:
    plt.rcParams.update(
        {
            "font.size": 10,
            "axes.titlesize": 12,
            "axes.labelsize": 10,
            "legend.fontsize": 9,
            "figure.titlesize": 15,
            "axes.spines.top": False,
            "axes.spines.right": False,
        }
    )


def _load_apgd(path: Path) -> pd.DataFrame:
    if not path.is_file():
        raise FileNotFoundError(f"missing adversarial quality result: {path}")
    frame = pd.read_csv(path)
    if "attack" in frame:
        frame = frame[frame["attack"].str.casefold() == "apgd-bce"].copy()
    elif "perturbation" in frame:
        frame = frame[frame["perturbation"] == "apgd_bce"].copy()
    else:
        raise ValueError(f"{path} has neither an attack nor perturbation column")
    frame = frame[frame["epsilon_n"].isin(EPSILON_VALUES)].copy()
    if frame.empty:
        raise ValueError("no positive-epsilon APGD records found")
    if "rms_percent_case_sigma" in frame:
        frame["rms_percent_sigma"] = frame["rms_percent_case_sigma"]
    elif "rms_percent_sigma" not in frame:
        frame["rms_percent_sigma"] = 100.0 * frame["rms_norm"]
    frame["epsilon_percent_sigma"] = 100.0 * frame["epsilon_norm"]
    required = {
        "dataset",
        "case_id",
        "epsilon_n",
        "epsilon_norm",
        "rms_norm",
        "rms_percent_sigma",
        "psnr_db_robust_range",
        "ssim_axial_prostate",
    }
    missing = sorted(required - set(frame.columns))
    if missing:
        raise ValueError(f"{path} is missing required columns: {missing}")
    duplicate = frame.duplicated(["dataset", "case_id", "epsilon_n"], keep=False)
    if duplicate.any():
        raise ValueError(f"{path} contains duplicate case/epsilon records")
    incomplete = (
        frame.groupby(["dataset", "case_id"])["epsilon_n"]
        .agg(lambda values: set(values) != set(EPSILON_VALUES))
        .sum()
    )
    if incomplete:
        raise ValueError(
            f"{path} contains {incomplete} cases without all five epsilon values"
        )
    missing_datasets = {"wg", "zones"} - set(frame["dataset"])
    if missing_datasets:
        raise ValueError(f"{path} is missing datasets: {sorted(missing_datasets)}")
    return frame


def _cohort_description(apgd: pd.DataFrame) -> str:
    parts = []
    for dataset in ("wg", "zones"):
        count = apgd.loc[apgd["dataset"] == dataset, "case_id"].nunique()
        if count:
            parts.append(f"{count:,} {DATASET_LABELS[dataset]}")
    return " and ".join(parts)


def _summarize(apgd: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, float | int | str]] = []
    for (dataset, epsilon_n), group in apgd.groupby(["dataset", "epsilon_n"]):
        row: dict[str, float | int | str] = {
            "dataset": str(dataset),
            "epsilon_n": int(epsilon_n),
            "epsilon_norm": float(group["epsilon_norm"].iloc[0]),
            "epsilon_percent_sigma": float(group["epsilon_percent_sigma"].iloc[0]),
            "n_cases": int(group["case_id"].nunique()),
        }
        for source, prefix in (
            ("rms_percent_sigma", "rms_percent_sigma"),
            ("psnr_db_robust_range", "psnr_db"),
            ("ssim_axial_prostate", "ssim"),
        ):
            values = group[source].to_numpy(dtype=float)
            row[f"{prefix}_median"] = float(np.median(values))
            row[f"{prefix}_q05"] = float(np.quantile(values, 0.05))
            row[f"{prefix}_q95"] = float(np.quantile(values, 0.95))
        rows.append(row)
    return pd.DataFrame(rows).sort_values(["dataset", "epsilon_n"])


def _epsilon_labels() -> list[str]:
    return [f"{value}/255" for value in EPSILON_VALUES]


def _configure_epsilon_axis(ax: plt.Axes, *, attack_name: str) -> None:
    ax.set_xticks(
        range(len(EPSILON_AXIS_VALUES)),
        [f"{value}/255" for value in EPSILON_AXIS_VALUES],
    )
    ax.set_xlim(-0.25, len(EPSILON_AXIS_VALUES) - 0.75)
    ax.set_xlabel(
        f"Maximum allowed {attack_name} change to any voxel, ε\n"
        "(0–32/255 = 0–12.55% of clean-image SD)"
    )


def _annotate_clean_psnr(ax: plt.Axes) -> None:
    ax.text(
        EPSILON_POSITIONS[0],
        0.97,
        "Clean\nPSNR = ∞",
        transform=ax.get_xaxis_transform(),
        ha="center",
        va="top",
        fontsize=8,
        bbox={
            "boxstyle": "round,pad=0.25",
            "facecolor": "white",
            "edgecolor": "#777777",
            "alpha": 0.9,
        },
    )


def _annotate_clean_trajectory(ax: plt.Axes) -> None:
    ax.text(
        0.97,
        0.06,
        "Clean (0/255)\nRMS = 0, PSNR = ∞, SSIM = 1",
        transform=ax.transAxes,
        ha="right",
        va="bottom",
        fontsize=8,
        bbox={
            "boxstyle": "round,pad=0.3",
            "facecolor": "white",
            "edgecolor": "#777777",
            "alpha": 0.9,
        },
    )


def _padded_limits(
    values: pd.Series,
    *,
    padding_fraction: float = 0.04,
    lower_bound: float | None = None,
    upper_bound: float | None = None,
) -> tuple[float, float]:
    finite = values[np.isfinite(values.to_numpy(dtype=float))].to_numpy(dtype=float)
    if not len(finite):
        raise ValueError("cannot determine axis limits from non-finite values")
    lower = float(np.min(finite))
    upper = float(np.max(finite))
    padding = max((upper - lower) * padding_fraction, 1e-3)
    lower -= padding
    upper += padding
    if lower_bound is not None:
        lower = max(lower, lower_bound)
    if upper_bound is not None:
        upper = min(upper, upper_bound)
    return lower, upper


def plot_epsilon_relationship(
    apgd: pd.DataFrame, summary: pd.DataFrame, output_dir: Path
) -> Path:
    fig, axes = plt.subplots(1, 2, figsize=(15, 6))
    configurations = (
        ("psnr_db", "PSNR (dB)", (29, 59), ".1f"),
        ("ssim", "SSIM", (0.875, 1.004), ".3f"),
    )
    for ax, (prefix, ylabel, ylim, number_format) in zip(axes, configurations):
        for dataset in ("wg", "zones"):
            group = summary[summary["dataset"] == dataset].sort_values("epsilon_n")
            epsilon_n_values = group["epsilon_n"].to_numpy(dtype=int)
            x = np.asarray(
                [EPSILON_POSITIONS[value] for value in epsilon_n_values], dtype=float
            )
            center = group[f"{prefix}_median"].to_numpy(dtype=float)
            q05 = group[f"{prefix}_q05"].to_numpy(dtype=float)
            q95 = group[f"{prefix}_q95"].to_numpy(dtype=float)
            if prefix == "ssim":
                x = np.insert(x, 0, EPSILON_POSITIONS[0])
                center = np.insert(center, 0, 1.0)
                q05 = np.insert(q05, 0, 1.0)
                q95 = np.insert(q95, 0, 1.0)
            color = DATASET_COLORS[dataset]
            ax.plot(
                x,
                center,
                color=color,
                marker="o",
                markersize=7,
                linewidth=2.4,
                label=DATASET_LABELS[dataset],
            )
            ax.fill_between(x, q05, q95, color=color, alpha=0.14)
            for epsilon_n, x_value, y_value in zip(
                epsilon_n_values,
                x[-len(epsilon_n_values) :],
                center[-len(epsilon_n_values) :],
            ):
                other = summary[
                    (summary["dataset"] != dataset)
                    & (summary["epsilon_n"] == int(epsilon_n))
                ]
                other_value = (
                    float(other[f"{prefix}_median"].iloc[0])
                    if not other.empty
                    else y_value
                )
                ax.annotate(
                    format(y_value, number_format),
                    (x_value, y_value),
                    xytext=(0, 8 if y_value >= other_value else -14),
                    textcoords="offset points",
                    ha="center",
                    color=color,
                    fontsize=8,
                )
        _configure_epsilon_axis(ax, attack_name="APGD")
        if prefix == "psnr_db":
            ax.set_ylabel("PSNR (dB; higher = more similar to clean)")
            _annotate_clean_psnr(ax)
        else:
            ax.set_ylabel("SSIM (higher = more structurally similar to clean)")
        ax.set_ylim(*ylim)
        ax.grid(alpha=0.25)
        ax.legend(frameon=False)
        ax.annotate(
            "larger allowed APGD change →",
            xy=(0.98, 0.04),
            xycoords="axes fraction",
            ha="right",
            color="#444444",
        )
    axes[0].set_title("Overall image similarity (PSNR)")
    axes[1].set_title("Local structural similarity (SSIM)")
    fig.suptitle(
        "Image similarity decreases as the maximum allowed APGD change increases"
    )
    fig.text(
        0.5,
        -0.01,
        f"Points are medians over {_cohort_description(apgd)} cases; "
        "shading is the 5th–95th percentile range. At 0/255, RMS = 0, "
        "PSNR = ∞, and SSIM = 1.",
        ha="center",
        color="#444444",
    )
    fig.tight_layout(rect=(0, 0.03, 1, 0.94))
    path = output_dir / "01_adversarial_epsilon_vs_psnr_ssim.png"
    fig.savefig(path, dpi=240, bbox_inches="tight")
    plt.close(fig)
    return path


def _epsilon_cmap() -> tuple[ListedColormap, BoundaryNorm]:
    cmap = ListedColormap(EPSILON_COLORS)
    boundaries = [1, 3, 6, 12, 24, 40]
    return cmap, BoundaryNorm(boundaries, cmap.N)


def plot_rms_relationship(apgd: pd.DataFrame, output_dir: Path) -> Path:
    metrics = (
        ("psnr_db_robust_range", "PSNR (dB)"),
        ("ssim_axial_prostate", "SSIM"),
    )
    metric_limits = {
        "psnr_db_robust_range": _padded_limits(apgd["psnr_db_robust_range"]),
        "ssim_axial_prostate": _padded_limits(
            apgd["ssim_axial_prostate"], lower_bound=-1.0, upper_bound=1.005
        ),
    }
    cmap, norm = _epsilon_cmap()
    fig, axes = plt.subplots(2, 2, figsize=(15, 11))
    scatter = None
    for row, dataset in enumerate(("wg", "zones")):
        group = apgd[apgd["dataset"] == dataset]
        point_size = 50 if len(group) < 250 else max(7, 2_500 / len(group))
        point_alpha = 0.75 if len(group) < 250 else 0.18
        medians = (
            group.groupby("epsilon_n", as_index=False)[
                ["rms_percent_sigma", "psnr_db_robust_range", "ssim_axial_prostate"]
            ]
            .median()
            .sort_values("epsilon_n")
        )
        for column, (metric, ylabel) in enumerate(metrics):
            ax = axes[row, column]
            scatter = ax.scatter(
                group["rms_percent_sigma"],
                group[metric],
                c=group["epsilon_n"],
                cmap=cmap,
                norm=norm,
                s=point_size,
                alpha=point_alpha,
                edgecolors="white",
                linewidths=0.4,
            )
            ax.plot(
                medians["rms_percent_sigma"],
                medians[metric],
                color="#222222",
                marker="X",
                linewidth=1.7,
                markersize=8,
                label="Median across cases",
                zorder=4,
            )
            rho = float(
                group[["rms_percent_sigma", metric]].corr(method="spearman").iloc[0, 1]
            )
            ax.text(
                0.04,
                0.08,
                f"Spearman ρ = {rho:.2f}",
                transform=ax.transAxes,
                bbox={
                    "boxstyle": "round,pad=0.25",
                    "facecolor": "white",
                    "alpha": 0.85,
                },
            )
            ax.set_xscale("log")
            ax.set_xlim(0.45, 11)
            ax.set_xticks([0.5, 1, 2, 4, 8, 10], ["0.5", "1", "2", "4", "8", "10"])
            ax.set_ylim(*metric_limits[metric])
            ax.set_xlabel(
                "Average APGD perturbation magnitude\n"
                "(% of clean-image SD; logarithmic scale)"
            )
            if metric == "psnr_db_robust_range":
                ax.set_ylabel("PSNR (dB; higher = more similar to clean)")
            else:
                ax.set_ylabel("SSIM (higher = more structurally similar to clean)")
            ax.set_title(f"{DATASET_LABELS[dataset]} — {ylabel}")
            ax.grid(alpha=0.25)
            ax.legend(frameon=False, loc="upper right")
    fig.subplots_adjust(
        left=0.07, right=0.88, bottom=0.09, top=0.91, wspace=0.25, hspace=0.30
    )
    if scatter is not None:
        colorbar = fig.colorbar(scatter, ax=axes, shrink=0.78, pad=0.02)
        colorbar.set_ticks(EPSILON_VALUES)
        colorbar.set_ticklabels([f"{value}/255" for value in EPSILON_VALUES])
        colorbar.set_label("Maximum allowed APGD change to any voxel, ε")
    fig.suptitle(
        "Larger average APGD perturbations correspond to lower PSNR and SSIM",
        y=1.015,
    )
    fig.text(
        0.5,
        0.012,
        "Each dot is one case at one ε; black X markers connect the case medians. "
        "The x-axis is logarithmic. Clean (0/255) has RMS = 0, PSNR = ∞, and "
        "SSIM = 1, so it lies outside these axes.",
        ha="center",
        color="#444444",
    )
    path = output_dir / "02_adversarial_rms_vs_psnr_ssim.png"
    fig.savefig(path, dpi=240, bbox_inches="tight")
    plt.close(fig)
    return path


def plot_joint_trajectory(apgd: pd.DataFrame, output_dir: Path) -> Path:
    cmap, norm = _epsilon_cmap()
    fig, axes = plt.subplots(1, 2, figsize=(15, 6), sharex=True, sharey=True)
    scatter = None
    for ax, dataset in zip(axes, ("wg", "zones")):
        group = apgd[apgd["dataset"] == dataset]
        point_size = 38 if len(group) < 250 else max(5, 1_800 / len(group))
        point_alpha = 0.32 if len(group) < 250 else 0.12
        medians = (
            group.groupby("epsilon_n", as_index=False)[
                ["psnr_db_robust_range", "ssim_axial_prostate"]
            ]
            .median()
            .sort_values("epsilon_n")
        )
        scatter = ax.scatter(
            group["psnr_db_robust_range"],
            group["ssim_axial_prostate"],
            c=group["epsilon_n"],
            cmap=cmap,
            norm=norm,
            s=point_size,
            alpha=point_alpha,
            edgecolors="none",
        )
        ax.plot(
            medians["psnr_db_robust_range"],
            medians["ssim_axial_prostate"],
            color="#222222",
            linewidth=1.8,
            zorder=3,
        )
        for index, record in medians.iterrows():
            epsilon_n = int(record["epsilon_n"])
            color = EPSILON_COLORS[EPSILON_VALUES.index(epsilon_n)]
            ax.scatter(
                [record["psnr_db_robust_range"]],
                [record["ssim_axial_prostate"]],
                color=color,
                marker="X",
                s=130,
                edgecolors="black",
                linewidths=0.6,
                zorder=5,
            )
            ax.annotate(
                f"{epsilon_n}/255",
                (record["psnr_db_robust_range"], record["ssim_axial_prostate"]),
                xytext=(7, -11 if index < 3 else 7),
                textcoords="offset points",
                fontsize=8,
                color=color,
                weight="bold",
            )
        previous = medians.iloc[-2]
        last = medians.iloc[-1]
        ax.annotate(
            "",
            xy=(last["psnr_db_robust_range"], last["ssim_axial_prostate"]),
            xytext=(
                previous["psnr_db_robust_range"],
                previous["ssim_axial_prostate"],
            ),
            arrowprops={"arrowstyle": "->", "color": "#333333", "lw": 1.5},
        )
        ax.set_title(DATASET_LABELS[dataset])
        ax.set_xlabel("PSNR (dB; higher = more similar to clean)")
        ax.grid(alpha=0.25)
        _annotate_clean_trajectory(ax)
    axes[0].set_ylabel("SSIM (higher = more structurally similar to clean)")
    axes[0].set_xlim(*_padded_limits(apgd["psnr_db_robust_range"]))
    axes[0].set_ylim(
        *_padded_limits(
            apgd["ssim_axial_prostate"], lower_bound=-1.0, upper_bound=1.005
        )
    )
    fig.subplots_adjust(left=0.07, right=0.88, bottom=0.12, top=0.86, wspace=0.20)
    if scatter is not None:
        colorbar = fig.colorbar(scatter, ax=axes, shrink=0.78, pad=0.02)
        colorbar.set_ticks(EPSILON_VALUES)
        colorbar.set_ticklabels([f"{value}/255" for value in EPSILON_VALUES])
        colorbar.set_label("Maximum allowed APGD change to any voxel, ε")
    fig.suptitle("Increasing APGD perturbation reduces both PSNR and SSIM")
    fig.text(
        0.5,
        -0.01,
        "Dots are individual cases; X markers are medians at each ε. "
        "Clean (0/255) is annotated because its PSNR is infinite.",
        ha="center",
        color="#444444",
    )
    path = output_dir / "03_psnr_ssim_adversarial_trajectory.png"
    fig.savefig(path, dpi=240, bbox_inches="tight")
    plt.close(fig)
    return path


def _markdown_summary_table(summary: pd.DataFrame) -> list[str]:
    rows = [
        "| Dataset | ε_norm | L∞ budget (% σ) | Realized RMS (% σ) | Median PSNR | Median SSIM |",
        "| --- | ---: | ---: | ---: | ---: | ---: |",
    ]
    for _, record in summary.iterrows():
        rows.append(
            "| "
            + " | ".join(
                [
                    DATASET_LABELS[str(record["dataset"])],
                    f"{int(record['epsilon_n'])}/255",
                    f"{record['epsilon_percent_sigma']:.2f}%",
                    f"{record['rms_percent_sigma_median']:.2f}%",
                    f"{record['psnr_db_median']:.1f} dB",
                    f"{record['ssim_median']:.3f}",
                ]
            )
            + " |"
        )
    return rows


def write_interpretation(
    apgd: pd.DataFrame,
    summary: pd.DataFrame,
    figures: list[Path],
    output_dir: Path,
    source_csv: Path,
) -> Path:
    correlations: dict[tuple[str, str], float] = {}
    for dataset in ("wg", "zones"):
        group = apgd[apgd["dataset"] == dataset]
        correlations[(dataset, "psnr")] = float(
            group[["rms_norm", "psnr_db_robust_range"]]
            .corr(method="spearman")
            .iloc[0, 1]
        )
        correlations[(dataset, "ssim")] = float(
            group[["rms_norm", "ssim_axial_prostate"]]
            .corr(method="spearman")
            .iloc[0, 1]
        )

    lines = [
        "# Adversarial noise, PSNR, and SSIM",
        "",
        f"This report includes {_cohort_description(apgd)} cases. Every case was evaluated at "
        "all five adversarial epsilon values.",
        "",
        f"Source data: `{source_csv}`",
        "",
        "## Main relationship",
        "",
        "As adversarial noise increases, both PSNR and SSIM decrease. Higher PSNR/SSIM means "
        "the attacked image is closer to the clean image, so the downward curves indicate "
        "increasing image change.",
        "",
        "PSNR is directly linked to RMS perturbation magnitude:",
        "",
        "`PSNR = 20 log10(robust image range / perturbation RMS)`",
        "",
        "Therefore, approximately doubling the realized RMS should reduce PSNR by about 6 dB. "
        "That pattern is visible in these results. SSIM has no equivalent fixed formula; it "
        "measures how much local contrast, texture, and structure changed.",
        "",
        *_markdown_summary_table(summary),
        "",
        "## 1. Epsilon budget versus PSNR and SSIM",
        "",
        f"![Epsilon versus PSNR and SSIM]({figures[0].name})",
        "",
        "The x-axis is the maximum allowed adversarial change. Both datasets follow the same "
        "pattern: increasing epsilon lowers PSNR and SSIM. SSIM remains extremely close to 1 at "
        "`2/255` and `4/255`, declines modestly at `8/255`, and drops more clearly at `16/255` "
        "and `32/255`.",
        "",
        "## 2. Realized RMS versus PSNR and SSIM",
        "",
        f"![RMS versus PSNR and SSIM]({figures[1].name})",
        "",
        "Epsilon is an L∞ ceiling, not the average perturbation. Realized RMS is the actual "
        "average perturbation magnitude and therefore has the clearest relationship with PSNR. "
        f"The Spearman correlations with PSNR are {correlations[('wg', 'psnr')]:.2f} for WG "
        f"and {correlations[('zones', 'psnr')]:.2f} for zones; correlations with SSIM are "
        f"{correlations[('wg', 'ssim')]:.2f} and {correlations[('zones', 'ssim')]:.2f}. "
        "Values near -1 mean that larger adversarial perturbations almost always correspond to "
        "lower image similarity.",
        "",
        "## 3. Joint PSNR–SSIM trajectory",
        "",
        f"![PSNR and SSIM trajectory]({figures[2].name})",
        "",
        "Increasing adversarial noise moves the points down and left: down means lower SSIM and "
        "left means lower PSNR. The two metrics therefore tell a consistent image-similarity "
        "story, although they measure different properties.",
        "",
        "## Practical reading",
        "",
        *[
            f"- `{epsilon_n}/255`: median PSNR/SSIM is "
            f"{dataset_rows.iloc[0]['psnr_db_median']:.1f} dB/"
            f"{dataset_rows.iloc[0]['ssim_median']:.3f} for WG and "
            f"{dataset_rows.iloc[1]['psnr_db_median']:.1f} dB/"
            f"{dataset_rows.iloc[1]['ssim_median']:.3f} for zones."
            for epsilon_n in EPSILON_VALUES
            if len(
                dataset_rows := summary[summary["epsilon_n"] == epsilon_n].sort_values(
                    "dataset"
                )
            )
            == 2
        ],
        "",
        "These values quantify similarity but do not establish whether changes are visible to a "
        "radiologist or physically realizable during MRI acquisition. There is no universal "
        "clinical PSNR or SSIM threshold for these data.",
        "The cohorts are the available preprocessed cases used by this experiment; these are "
        "not an independent prospective clinical test cohort.",
        "",
    ]
    comparison_report = output_dir / "RANDOM_NOISE_COMPARISON.md"
    if comparison_report.is_file():
        lines.extend(
            [
                "## RMS-matched random-noise comparison",
                "",
                "The complete paired comparison with Gaussian and Rician-proxy random noise "
                "is reported in [`RANDOM_NOISE_COMPARISON.md`](RANDOM_NOISE_COMPARISON.md). "
                "It separates perturbation energy from spatially targeted adversarial effects.",
                "",
                "![Adversarial versus random-noise PSNR and SSIM]"
                "(04_adversarial_vs_random_psnr_ssim.png)",
                "",
                "![Paired random-noise minus adversarial differences]"
                "(05_paired_random_minus_adversarial.png)",
                "",
                "![Gaussian and Rician-proxy PSNR and SSIM trajectories]"
                "(06_psnr_ssim_random_noise_trajectories.png)",
                "",
                "![Merged APGD, Gaussian, and Rician-proxy trajectories]"
                "(07_psnr_ssim_adversarial_and_random_trajectories.png)",
                "",
            ]
        )
    attack_comparison_report = output_dir / "ADVERSARIAL_ATTACK_COMPARISON.md"
    if attack_comparison_report.is_file():
        lines.extend(
            [
                "## FGSM, PGD, and APGD comparison",
                "",
                "The complete paired attack comparison is reported in "
                "[`ADVERSARIAL_ATTACK_COMPARISON.md`]"
                "(ADVERSARIAL_ATTACK_COMPARISON.md). All attacks use the same cases, "
                "epsilon values, BCE objective, and FP32 evaluation pipeline.",
                "",
                "![FGSM, PGD, and APGD PSNR and SSIM comparison]"
                "(08_fgsm_pgd_apgd_psnr_ssim_comparison.png)",
                "",
                "![FGSM, PGD, and APGD PSNR-SSIM trajectories]"
                "(09_fgsm_pgd_apgd_psnr_ssim_trajectories.png)",
                "",
                "![Paired FGSM and PGD differences relative to APGD]"
                "(10_fgsm_pgd_minus_apgd_psnr_ssim.png)",
                "",
            ]
        )
    path = output_dir / "README.md"
    path.write_text("\n".join(lines), encoding="utf-8")
    return path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Plot adversarial perturbation magnitude against PSNR and SSIM"
    )
    parser.add_argument("--results-dir", type=Path, default=DEFAULT_RESULTS)
    parser.add_argument(
        "--quality-csv",
        type=Path,
        default=None,
        help="APGD quality CSV; defaults to the full-cohort result when available",
    )
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    results_dir = args.results_dir.resolve()
    output_dir = args.output_dir.resolve()
    if args.quality_csv is not None:
        quality_csv = args.quality_csv.resolve()
    else:
        all_cases_csv = (
            results_dir
            / "all_cases_adversarial_quality"
            / "adversarial_quality_all_cases.csv"
        )
        quality_csv = all_cases_csv if all_cases_csv.is_file() else DEFAULT_SUBSET_CSV
    output_dir.mkdir(parents=True, exist_ok=True)
    _configure_style()
    apgd = _load_apgd(quality_csv)
    summary = _summarize(apgd)

    figures = [
        plot_epsilon_relationship(apgd, summary, output_dir),
        plot_rms_relationship(apgd, output_dir),
        plot_joint_trajectory(apgd, output_dir),
    ]
    summary_path = output_dir / "adversarial_psnr_ssim_summary.csv"
    summary.to_csv(summary_path, index=False)
    report = write_interpretation(apgd, summary, figures, output_dir, quality_csv)

    for path in [*figures, summary_path, report]:
        print(f"Saved {path}")


if __name__ == "__main__":
    main()
