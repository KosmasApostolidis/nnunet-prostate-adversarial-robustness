"""Compare full-cohort APGD image quality with RMS-matched random controls."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import matplotlib
import numpy as np
import pandas as pd
from matplotlib.colors import BoundaryNorm, ListedColormap

matplotlib.use("Agg")
import matplotlib.pyplot as plt


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))


DEFAULT_RESULTS = (
    REPO_ROOT / "results" / "adv_rob_eval_results" / "image_quality_analysis"
)
DEFAULT_APGD_CSV = (
    DEFAULT_RESULTS
    / "all_cases_adversarial_quality"
    / "adversarial_quality_all_cases.csv"
)
DEFAULT_RANDOM_CSV = (
    DEFAULT_RESULTS
    / "all_cases_random_noise_quality"
    / "random_noise_quality_all_cases.csv"
)
DEFAULT_OUTPUT = DEFAULT_RESULTS / "psnr_ssim_all_cases_figures"
EPSILON_VALUES = [2, 4, 8, 16, 32]
EPSILON_AXIS_VALUES = [0, *EPSILON_VALUES]
EPSILON_POSITIONS = {
    epsilon_n: index for index, epsilon_n in enumerate(EPSILON_AXIS_VALUES)
}
DATASET_LABELS = {"wg": "Whole gland (WG)", "zones": "Prostate zones"}
PERTURBATION_LABELS = {
    "apgd_bce": "Adversarial attack (APGD-BCE)",
    "gaussian_rms_matched": "Gaussian noise (RMS-matched)",
    "rician_proxy_rms_matched": "Rician-proxy noise (RMS-matched)",
}
COLORS = {
    "apgd_bce": "#D1495B",
    "gaussian_rms_matched": "#0072B2",
    "rician_proxy_rms_matched": "#009E73",
}
MARKERS = {
    "apgd_bce": "o",
    "gaussian_rms_matched": "s",
    "rician_proxy_rms_matched": "^",
}
EPSILON_COLORS = ["#3B4CC0", "#2C7FB8", "#41AB5D", "#F0A202", "#D1495B"]
METRICS = [
    "rms_norm",
    "rms_percent_case_sigma",
    "psnr_db_robust_range",
    "ssim_axial_prostate",
]


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
        raise FileNotFoundError(f"missing APGD quality CSV: {path}")
    frame = pd.read_csv(path)
    if "attack" in frame.columns:
        frame = frame[frame["attack"].str.casefold() == "apgd-bce"].copy()
    elif "perturbation" in frame.columns:
        frame = frame[frame["perturbation"] == "apgd_bce"].copy()
    frame = frame[frame["epsilon_n"].isin(EPSILON_VALUES)].copy()
    frame["perturbation"] = "apgd_bce"
    frame["trial"] = 0
    _validate_quality(frame, path, key_columns=["dataset", "case_id", "epsilon_n"])
    return frame


def _load_random(path: Path) -> pd.DataFrame:
    if not path.is_file():
        raise FileNotFoundError(
            f"missing random-control quality CSV: {path}; run "
            "experiments/evaluate_random_noise_quality_all_cases.py first"
        )
    frame = pd.read_csv(path)
    frame = frame[frame["epsilon_n"].isin(EPSILON_VALUES)].copy()
    allowed = set(PERTURBATION_LABELS) - {"apgd_bce"}
    unknown = sorted(set(frame["perturbation"]) - allowed)
    if unknown:
        raise ValueError(f"{path} contains unknown random controls: {unknown}")
    _validate_quality(
        frame,
        path,
        key_columns=["dataset", "case_id", "epsilon_n", "perturbation", "trial"],
    )
    return frame


def _validate_quality(
    frame: pd.DataFrame, path: Path, *, key_columns: list[str]
) -> None:
    required = {
        "dataset",
        "case_id",
        "epsilon_n",
        "epsilon_norm",
        "perturbation",
        *METRICS,
    }
    missing = sorted(required - set(frame.columns))
    if missing:
        raise ValueError(f"{path} is missing required columns: {missing}")
    if frame.empty:
        raise ValueError(f"{path} contains no requested quality rows")
    duplicate = frame.duplicated(key_columns, keep=False)
    if duplicate.any():
        raise ValueError(f"{path} contains duplicate rows for keys {key_columns}")
    if not np.isfinite(frame[METRICS].to_numpy(dtype=float)).all():
        raise ValueError(f"{path} contains non-finite image-quality values")


def _case_average_controls(random: pd.DataFrame) -> pd.DataFrame:
    """Average repeated random draws so each patient contributes once."""

    aggregations = {metric: "mean" for metric in METRICS}
    if "rms_match_relative_error" in random.columns:
        aggregations["rms_match_relative_error"] = "mean"
    return (
        random.groupby(
            ["dataset", "case_id", "epsilon_n", "epsilon_norm", "perturbation"],
            as_index=False,
        )
        .agg(aggregations)
        .sort_values(["dataset", "case_id", "epsilon_n", "perturbation"])
    )


def _validate_pairing(apgd: pd.DataFrame, controls: pd.DataFrame) -> None:
    apgd_keys = set(
        map(tuple, apgd[["dataset", "case_id", "epsilon_n"]].itertuples(index=False))
    )
    for perturbation, group in controls.groupby("perturbation"):
        control_keys = set(
            map(
                tuple,
                group[["dataset", "case_id", "epsilon_n"]].itertuples(index=False),
            )
        )
        missing = apgd_keys - control_keys
        extra = control_keys - apgd_keys
        if missing or extra:
            raise ValueError(
                f"{perturbation} is not paired one-to-one with APGD: "
                f"{len(missing)} missing, {len(extra)} extra keys"
            )


def _summarize(frame: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    for keys, group in frame.groupby(
        ["dataset", "epsilon_n", "epsilon_norm", "perturbation"], sort=True
    ):
        dataset, epsilon_n, epsilon_norm, perturbation = keys
        row: dict[str, object] = {
            "dataset": dataset,
            "epsilon_n": int(epsilon_n),
            "epsilon_norm": float(epsilon_norm),
            "epsilon_percent_sigma": 100.0 * float(epsilon_norm),
            "perturbation": perturbation,
            "n_cases": int(group["case_id"].nunique()),
        }
        for metric in METRICS:
            values = group[metric].to_numpy(dtype=float)
            row[f"{metric}_median"] = float(np.median(values))
            row[f"{metric}_q05"] = float(np.quantile(values, 0.05))
            row[f"{metric}_q95"] = float(np.quantile(values, 0.95))
        rows.append(row)
    return pd.DataFrame(rows).sort_values(["dataset", "epsilon_n", "perturbation"])


def _paired_differences(apgd: pd.DataFrame, controls: pd.DataFrame) -> pd.DataFrame:
    adversarial = apgd[
        [
            "dataset",
            "case_id",
            "epsilon_n",
            "epsilon_norm",
            "rms_norm",
            "psnr_db_robust_range",
            "ssim_axial_prostate",
        ]
    ].rename(
        columns={
            "rms_norm": "apgd_rms_norm",
            "psnr_db_robust_range": "apgd_psnr",
            "ssim_axial_prostate": "apgd_ssim",
        }
    )
    paired = controls.merge(
        adversarial,
        on=["dataset", "case_id", "epsilon_n", "epsilon_norm"],
        validate="many_to_one",
    )
    paired["rms_relative_error"] = (
        paired["rms_norm"] - paired["apgd_rms_norm"]
    ).abs() / paired["apgd_rms_norm"].clip(lower=1e-12)
    paired["psnr_random_minus_apgd"] = (
        paired["psnr_db_robust_range"] - paired["apgd_psnr"]
    )
    paired["ssim_random_minus_apgd"] = (
        paired["ssim_axial_prostate"] - paired["apgd_ssim"]
    )
    return paired.sort_values(["dataset", "case_id", "epsilon_n", "perturbation"])


def _summarize_paired(paired: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    metrics = [
        "rms_relative_error",
        "psnr_random_minus_apgd",
        "ssim_random_minus_apgd",
        "apgd_ssim",
        "ssim_axial_prostate",
    ]
    for keys, group in paired.groupby(
        ["dataset", "epsilon_n", "epsilon_norm", "perturbation"], sort=True
    ):
        dataset, epsilon_n, epsilon_norm, perturbation = keys
        row: dict[str, object] = {
            "dataset": dataset,
            "epsilon_n": int(epsilon_n),
            "epsilon_norm": float(epsilon_norm),
            "perturbation": perturbation,
            "n_cases": int(group["case_id"].nunique()),
            "fraction_random_ssim_higher": float(
                np.mean(group["ssim_random_minus_apgd"].to_numpy(dtype=float) > 0)
            ),
        }
        for metric in metrics:
            values = group[metric].to_numpy(dtype=float)
            row[f"{metric}_mean"] = float(np.mean(values))
            row[f"{metric}_median"] = float(np.median(values))
            row[f"{metric}_q05"] = float(np.quantile(values, 0.05))
            row[f"{metric}_q95"] = float(np.quantile(values, 0.95))
        rows.append(row)
    return pd.DataFrame(rows).sort_values(["dataset", "epsilon_n", "perturbation"])


def _epsilon_labels() -> list[str]:
    return [f"{value}/255" for value in EPSILON_VALUES]


def _configure_epsilon_axis(ax: plt.Axes) -> None:
    ax.set_xticks(
        range(len(EPSILON_AXIS_VALUES)),
        [f"{value}/255" for value in EPSILON_AXIS_VALUES],
    )
    ax.set_xlim(-0.25, len(EPSILON_AXIS_VALUES) - 0.75)
    ax.set_xlabel(
        "Maximum allowed change to any voxel, ε\n"
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


def _epsilon_cmap() -> tuple[ListedColormap, BoundaryNorm]:
    cmap = ListedColormap(EPSILON_COLORS)
    boundaries = [1, 3, 6, 12, 24, 40]
    return cmap, BoundaryNorm(boundaries, cmap.N)


def _plot_joint_trajectory_grid(
    trajectory_data: pd.DataFrame,
    perturbations: list[str],
    output_dir: Path,
    *,
    title: str,
    caption: str,
    filename: str,
) -> Path:
    cmap, norm = _epsilon_cmap()
    fig, axes = plt.subplots(
        len(perturbations),
        2,
        figsize=(15, 5.2 * len(perturbations)),
        sharex=True,
        sharey=True,
        squeeze=False,
    )
    scatter = None
    for row, perturbation in enumerate(perturbations):
        for column, dataset in enumerate(("wg", "zones")):
            ax = axes[row, column]
            group = trajectory_data[
                (trajectory_data["dataset"] == dataset)
                & (trajectory_data["perturbation"] == perturbation)
            ]
            if group.empty:
                raise ValueError(
                    f"{PERTURBATION_LABELS[perturbation]} has no {dataset} cases"
                )
            point_size = 38 if len(group) < 250 else max(5, 1_800 / len(group))
            point_alpha = 0.32 if len(group) < 250 else 0.12
            medians = (
                group.groupby("epsilon_n", as_index=False)[
                    ["psnr_db_robust_range", "ssim_axial_prostate"]
                ]
                .median()
                .sort_values("epsilon_n")
                .reset_index(drop=True)
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
                    (
                        record["psnr_db_robust_range"],
                        record["ssim_axial_prostate"],
                    ),
                    xytext=(7, -11 if index < 3 else 7),
                    textcoords="offset points",
                    fontsize=8,
                    color=color,
                    weight="bold",
                )
            if len(medians) >= 2:
                previous = medians.iloc[-2]
                last = medians.iloc[-1]
                ax.annotate(
                    "",
                    xy=(
                        last["psnr_db_robust_range"],
                        last["ssim_axial_prostate"],
                    ),
                    xytext=(
                        previous["psnr_db_robust_range"],
                        previous["ssim_axial_prostate"],
                    ),
                    arrowprops={"arrowstyle": "->", "color": "#333333", "lw": 1.5},
                )
            ax.set_title(
                f"{DATASET_LABELS[dataset]} — {PERTURBATION_LABELS[perturbation]}"
            )
            ax.set_xlabel("PSNR (dB; higher = more similar to clean)")
            ax.tick_params(axis="x", labelbottom=True)
            ax.grid(alpha=0.25)
            _annotate_clean_trajectory(ax)
        axes[row, 0].set_ylabel("SSIM (higher = more structurally similar to clean)")

    # Match Figure 3 so every perturbation can be compared directly.
    axes[0, 0].set_xlim(30.0, 60.3)
    axes[0, 0].set_ylim(0.765, 1.005)
    bottom = 0.055 if len(perturbations) >= 3 else 0.08
    fig.subplots_adjust(left=0.07, right=0.88, bottom=bottom, top=0.93, wspace=0.20)
    if scatter is not None:
        colorbar = fig.colorbar(scatter, ax=axes, shrink=0.78, pad=0.02)
        colorbar.set_ticks(EPSILON_VALUES)
        colorbar.set_ticklabels([f"{value}/255" for value in EPSILON_VALUES])
        colorbar.set_label("Maximum allowed change to any voxel, ε")
    fig.suptitle(title)
    fig.text(
        0.5,
        0.012,
        f"{caption} Clean (0/255) is annotated because its PSNR is infinite.",
        ha="center",
        color="#444444",
    )
    path = output_dir / filename
    fig.savefig(path, dpi=240, bbox_inches="tight")
    plt.close(fig)
    return path


def plot_random_joint_trajectories(controls: pd.DataFrame, output_dir: Path) -> Path:
    """Plot case-level PSNR–SSIM trajectories for each random control."""

    perturbations = [
        value
        for value in ("gaussian_rms_matched", "rician_proxy_rms_matched")
        if value in set(controls["perturbation"])
    ]
    if not perturbations:
        raise ValueError("no Gaussian or Rician-proxy controls are available to plot")
    return _plot_joint_trajectory_grid(
        controls,
        perturbations,
        output_dir,
        title=(
            "PSNR and SSIM under Gaussian and Rician-proxy noise "
            "matched to APGD magnitude"
        ),
        caption=(
            "Dots are individual cases averaged across repeated random draws; "
            "X markers are medians at each ε."
        ),
        filename="06_psnr_ssim_random_noise_trajectories.png",
    )


def plot_combined_joint_trajectories(
    apgd: pd.DataFrame, controls: pd.DataFrame, output_dir: Path
) -> Path:
    """Merge APGD, Gaussian, and Rician PSNR–SSIM trajectories into one figure."""

    combined = pd.concat([apgd, controls], ignore_index=True, sort=False)
    perturbations = list(PERTURBATION_LABELS)
    missing = [
        value for value in perturbations if value not in set(combined["perturbation"])
    ]
    if missing:
        raise ValueError(f"cannot create merged trajectory; missing {missing}")
    return _plot_joint_trajectory_grid(
        combined,
        perturbations,
        output_dir,
        title=("PSNR and SSIM across APGD and magnitude-matched random noise"),
        caption=(
            "Dots are individual cases (random controls are trial-averaged); "
            "X markers are medians at each ε."
        ),
        filename="07_psnr_ssim_adversarial_and_random_trajectories.png",
    )


def plot_metric_comparison(summary: pd.DataFrame, output_dir: Path) -> Path:
    fig, axes = plt.subplots(2, 2, figsize=(15, 11))
    configurations = (
        ("psnr_db_robust_range", "PSNR (dB)", (29, 59)),
        ("ssim_axial_prostate", "SSIM", (0.85, 1.004)),
    )
    perturbations = [
        value for value in PERTURBATION_LABELS if value in set(summary["perturbation"])
    ]
    for row, dataset in enumerate(("wg", "zones")):
        for column, (metric, ylabel, ylim) in enumerate(configurations):
            ax = axes[row, column]
            for perturbation in perturbations:
                group = summary[
                    (summary["dataset"] == dataset)
                    & (summary["perturbation"] == perturbation)
                ].sort_values("epsilon_n")
                if group.empty:
                    continue
                x = np.asarray(
                    [EPSILON_POSITIONS[int(value)] for value in group["epsilon_n"]],
                    dtype=float,
                )
                median = group[f"{metric}_median"].to_numpy(dtype=float)
                q05 = group[f"{metric}_q05"].to_numpy(dtype=float)
                q95 = group[f"{metric}_q95"].to_numpy(dtype=float)
                if metric == "ssim_axial_prostate":
                    x = np.insert(x, 0, EPSILON_POSITIONS[0])
                    median = np.insert(median, 0, 1.0)
                    q05 = np.insert(q05, 0, 1.0)
                    q95 = np.insert(q95, 0, 1.0)
                ax.plot(
                    x,
                    median,
                    color=COLORS[perturbation],
                    marker=MARKERS[perturbation],
                    linewidth=2.2,
                    label=PERTURBATION_LABELS[perturbation],
                )
                ax.fill_between(x, q05, q95, color=COLORS[perturbation], alpha=0.08)
            _configure_epsilon_axis(ax)
            ax.set_ylim(*ylim)
            if metric == "psnr_db_robust_range":
                ax.set_ylabel("PSNR (dB; higher = more similar to clean)")
                _annotate_clean_psnr(ax)
            else:
                ax.set_ylabel("SSIM (higher = more structurally similar to clean)")
            ax.set_title(f"{DATASET_LABELS[dataset]} — {ylabel}")
            ax.grid(alpha=0.25)
            ax.legend(frameon=False)
            if metric == "psnr_db_robust_range":
                ax.text(
                    0.03,
                    0.05,
                    "PSNR overlaps because average perturbation\n"
                    "magnitude is matched case by case",
                    transform=ax.transAxes,
                    bbox={
                        "boxstyle": "round,pad=0.25",
                        "facecolor": "white",
                        "alpha": 0.85,
                    },
                )
    fig.suptitle(
        "APGD versus random noise with the same average perturbation magnitude",
        y=1.01,
    )
    fig.text(
        0.5,
        -0.005,
        "Each random control is matched to APGD's RMS (average perturbation magnitude) "
        "within the same case. "
        "Lines are medians; shading is the 5th–95th percentile range. "
        "At 0/255, RMS = 0, PSNR = ∞, and SSIM = 1.",
        ha="center",
        color="#444444",
    )
    fig.tight_layout(rect=(0, 0.02, 1, 0.98))
    path = output_dir / "04_adversarial_vs_random_psnr_ssim.png"
    fig.savefig(path, dpi=240, bbox_inches="tight")
    plt.close(fig)
    return path


def plot_paired_differences(paired_summary: pd.DataFrame, output_dir: Path) -> Path:
    fig, axes = plt.subplots(2, 2, figsize=(15, 10))
    configurations = (
        (
            "psnr_random_minus_apgd",
            "PSNR difference",
            "Random − APGD PSNR difference (dB)",
        ),
        (
            "ssim_random_minus_apgd",
            "SSIM difference",
            "Random − APGD SSIM difference",
        ),
    )
    controls = [
        value
        for value in PERTURBATION_LABELS
        if value != "apgd_bce" and value in set(paired_summary["perturbation"])
    ]
    for row, dataset in enumerate(("wg", "zones")):
        for column, (metric, panel_label, ylabel) in enumerate(configurations):
            ax = axes[row, column]
            for perturbation in controls:
                group = paired_summary[
                    (paired_summary["dataset"] == dataset)
                    & (paired_summary["perturbation"] == perturbation)
                ].sort_values("epsilon_n")
                x = np.asarray(
                    [EPSILON_POSITIONS[int(value)] for value in group["epsilon_n"]],
                    dtype=float,
                )
                median = np.insert(
                    group[f"{metric}_median"].to_numpy(dtype=float), 0, 0.0
                )
                q05 = np.insert(group[f"{metric}_q05"].to_numpy(dtype=float), 0, 0.0)
                q95 = np.insert(group[f"{metric}_q95"].to_numpy(dtype=float), 0, 0.0)
                x = np.insert(x, 0, EPSILON_POSITIONS[0])
                ax.plot(
                    x,
                    median,
                    color=COLORS[perturbation],
                    marker=MARKERS[perturbation],
                    linewidth=2.2,
                    label=PERTURBATION_LABELS[perturbation],
                )
                ax.fill_between(x, q05, q95, color=COLORS[perturbation], alpha=0.12)
            ax.axhline(0.0, color="#555555", linestyle="--", linewidth=1.2)
            _configure_epsilon_axis(ax)
            ax.set_ylabel(ylabel)
            ax.set_title(f"{DATASET_LABELS[dataset]} — {panel_label}")
            ax.grid(alpha=0.25)
            ax.legend(frameon=False)
            if metric == "ssim_random_minus_apgd":
                ax.text(
                    0.03,
                    0.92,
                    "Above 0: random noise preserves more structure\n"
                    "Below 0: APGD preserves more structure",
                    transform=ax.transAxes,
                    va="top",
                    color="#444444",
                )
    fig.suptitle(
        "Within-case image-similarity difference: random noise minus APGD", y=1.01
    )
    fig.text(
        0.5,
        -0.005,
        "Positive values favor random noise; negative values favor APGD. "
        "Lines are medians; shading is the 5th–95th percentile range. "
        "At clean 0/255, both perturbations are identical and the difference is 0.",
        ha="center",
        color="#444444",
    )
    fig.tight_layout(rect=(0, 0.02, 1, 0.98))
    path = output_dir / "05_paired_random_minus_adversarial.png"
    fig.savefig(path, dpi=240, bbox_inches="tight")
    plt.close(fig)
    return path


def _comparison_table(paired_summary: pd.DataFrame) -> list[str]:
    lines = [
        "| Dataset | ε | Random control | RMS match error | Median APGD SSIM | "
        "Median random SSIM | Median ΔSSIM | Cases with random SSIM higher |",
        "| --- | ---: | --- | ---: | ---: | ---: | ---: | ---: |",
    ]
    for record in paired_summary.itertuples(index=False):
        lines.append(
            "| "
            + " | ".join(
                [
                    DATASET_LABELS[str(record.dataset)],
                    f"{int(record.epsilon_n)}/255",
                    PERTURBATION_LABELS[str(record.perturbation)],
                    f"{100 * record.rms_relative_error_median:.4f}%",
                    f"{record.apgd_ssim_median:.5f}",
                    f"{record.ssim_axial_prostate_median:.5f}",
                    f"{record.ssim_random_minus_apgd_median:+.5f}",
                    f"{100 * record.fraction_random_ssim_higher:.1f}%",
                ]
            )
            + " |"
        )
    return lines


def write_report(
    apgd: pd.DataFrame,
    random: pd.DataFrame,
    paired_summary: pd.DataFrame,
    figures: list[Path],
    output_dir: Path,
    apgd_csv: Path,
    random_csv: Path,
) -> Path:
    cohort = ", ".join(
        f"{apgd.loc[apgd['dataset'] == dataset, 'case_id'].nunique():,} "
        f"{DATASET_LABELS[dataset]}"
        for dataset in ("wg", "zones")
    )
    trials = int(random["trial"].nunique()) if "trial" in random.columns else 1
    lines = [
        "# Adversarial versus RMS-matched random noise",
        "",
        f"This paired analysis includes {cohort} cases at all five epsilon values.",
        f"Each random control uses {trials} deterministic trial(s) per case and epsilon.",
        "",
        f"APGD source: `{apgd_csv}`",
        "",
        f"Random-control source: `{random_csv}`",
        "",
        "## Why RMS matching matters",
        "",
        "The Gaussian and optional Rician-proxy controls are matched case by case to the "
        "realized APGD RMS and constrained by the same L∞ budget and clean-image intensity "
        "bounds. This comparison therefore isolates spatial organization from total "
        "perturbation energy.",
        "",
        "PSNR is expected to overlap because it is determined by RMS and the clean-image "
        "range. Its agreement is a matching validation, not evidence that the perturbations "
        "have the same structure. SSIM is the informative comparison.",
        "",
        "## 1. Absolute PSNR and SSIM",
        "",
        f"![Adversarial versus random PSNR and SSIM]({figures[0].name})",
        "",
        "## 2. Paired within-case differences",
        "",
        f"![Paired random minus adversarial differences]({figures[1].name})",
        "",
        "Positive ΔSSIM means that random noise preserved more local structure than APGD at "
        "the same RMS magnitude. The paired analysis avoids confounding from different case "
        "intensity ranges.",
        "",
        "## 3. Joint PSNR–SSIM trajectories for random controls",
        "",
        f"![Gaussian and Rician-proxy PSNR and SSIM trajectories]({figures[2].name})",
        "",
        "Each small point represents one case after averaging its repeated random trials. "
        "The axes and epsilon colors match the adversarial trajectory figure for direct "
        "visual comparison.",
        "",
        "## 4. Merged adversarial and random-control trajectories",
        "",
        f"![Merged APGD, Gaussian, and Rician-proxy trajectories]({figures[3].name})",
        "",
        "This six-panel figure merges Figures 3 and 6: rows identify perturbation type, "
        "columns identify cohort, and all panels use the same PSNR and SSIM scales.",
        "",
        "## Numerical comparison",
        "",
        *_comparison_table(paired_summary),
        "",
        "## Interpretation limits",
        "",
        "- These controls distinguish optimized from random perturbation structure; they do "
        "not make either perturbation a physical MRI acquisition model.",
        "- PSNR and SSIM quantify similarity to the clean volume, not segmentation accuracy, "
        "radiologist visibility, diagnostic adequacy, or patient outcomes.",
        "- Repeated random trials are averaged within case before cohort summaries so cases "
        "receive equal weight.",
        "- The available preprocessed cohorts are not an independent prospective clinical "
        "test cohort.",
        "",
    ]
    path = output_dir / "RANDOM_NOISE_COMPARISON.md"
    path.write_text("\n".join(lines), encoding="utf-8")
    return path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Plot APGD versus RMS-matched random-noise PSNR/SSIM"
    )
    parser.add_argument("--apgd-csv", type=Path, default=DEFAULT_APGD_CSV)
    parser.add_argument("--random-csv", type=Path, default=DEFAULT_RANDOM_CSV)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    apgd_csv = args.apgd_csv.resolve()
    random_csv = args.random_csv.resolve()
    output_dir = args.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    _configure_style()

    apgd = _load_apgd(apgd_csv)
    random = _load_random(random_csv)
    controls = _case_average_controls(random)
    _validate_pairing(apgd, controls)
    combined = pd.concat([apgd, controls], ignore_index=True, sort=False)
    summary = _summarize(combined)
    paired = _paired_differences(apgd, controls)
    paired_summary = _summarize_paired(paired)

    figures = [
        plot_metric_comparison(summary, output_dir),
        plot_paired_differences(paired_summary, output_dir),
        plot_random_joint_trajectories(controls, output_dir),
        plot_combined_joint_trajectories(apgd, controls, output_dir),
    ]
    summary_path = output_dir / "adversarial_vs_random_psnr_ssim_summary.csv"
    paired_path = output_dir / "adversarial_vs_random_paired_differences.csv"
    summary.to_csv(summary_path, index=False)
    paired_summary.to_csv(paired_path, index=False)
    report = write_report(
        apgd,
        random,
        paired_summary,
        figures,
        output_dir,
        apgd_csv,
        random_csv,
    )
    for path in [*figures, summary_path, paired_path, report]:
        print(f"Saved {path}")


if __name__ == "__main__":
    main()
