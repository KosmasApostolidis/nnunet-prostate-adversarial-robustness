"""Create plain-language plots for the physical-epsilon calibration results.

The calibration tables are scientifically useful but difficult to read without
seeing distributions and paired case-level comparisons. This script turns the
existing CSV outputs into a small visual narrative. It does not rerun attacks.

Example::

    python experiments/plot_epsilon_calibration_explained.py
"""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib
import numpy as np
import pandas as pd

matplotlib.use("Agg")
import matplotlib.pyplot as plt


REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_RESULTS = REPO_ROOT / "results" / "adv_rob_eval_results" / "image_quality_analysis"
DEFAULT_OUTPUT = DEFAULT_RESULTS / "explanatory_plots"

PERTURBATIONS = (
    "apgd_bce",
    "gaussian_rms_matched",
    "rician_proxy_rms_matched",
)
LABELS = {
    "clean": "Clean",
    "apgd_bce": "Adversarial (APGD)",
    "gaussian_rms_matched": "Gaussian, same RMS",
    "rician_proxy_rms_matched": "Rician proxy, same RMS",
}
COLORS = {
    "clean": "#555555",
    "apgd_bce": "#D1495B",
    "gaussian_rms_matched": "#0072B2",
    "rician_proxy_rms_matched": "#009E73",
    "wg": "#6A4C93",
    "zones": "#F28E2B",
}
MARKERS = {
    "clean": "D",
    "apgd_bce": "o",
    "gaussian_rms_matched": "s",
    "rician_proxy_rms_matched": "^",
}
DATASET_NAMES = {"wg": "Whole gland (WG)", "zones": "Prostate zones"}
EPSILON_GRID = [0, 2, 4, 8, 16, 32]


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


def _load_results(results_dir: Path) -> dict[str, pd.DataFrame]:
    files = {
        "dice": "segmentation_dice_per_case.csv",
        "quality": "perturbation_quality_per_case.csv",
        "mapping": "epsilon_mapping.csv",
        "normalization": "normalization_per_case.csv",
    }
    frames: dict[str, pd.DataFrame] = {}
    for key, filename in files.items():
        path = results_dir / filename
        if not path.is_file():
            raise FileNotFoundError(f"required calibration result is missing: {path}")
        frames[key] = pd.read_csv(path)
    return frames


def _macro_case_dice(dice: pd.DataFrame) -> pd.DataFrame:
    macro = dice[dice["class"] == "macro"].copy()
    return (
        macro.groupby(
            ["dataset", "case_id", "epsilon_n", "perturbation"],
            as_index=False,
        )["dice"]
        .mean()
        .sort_values(["dataset", "case_id", "epsilon_n", "perturbation"])
    )


def _short_case_id(dataset: str, case_id: str) -> str:
    if dataset == "wg":
        core = case_id.removeprefix("ProstateWG_")
        return f"WG-{core}" if len(core) <= 8 else f"WG-…{core[-7:]}"
    core = case_id.removeprefix("ProstateZonesFilteredLessDilated_ProstateZones_")
    return f"Z-{core}" if len(core) <= 8 else f"Z-…{core[-7:]}"


def _epsilon_tick_labels(values: list[int]) -> list[str]:
    labels = []
    for value in values:
        if value == 0:
            labels.append("0\nclean")
        else:
            percent = 100.0 * value / 255.0
            labels.append(f"{value}/255\n{percent:.2f}% σ")
    return labels


def _curve_statistics(
    case_dice: pd.DataFrame, dataset: str, perturbation: str
) -> pd.DataFrame:
    subset = case_dice[
        (case_dice["dataset"] == dataset) & (case_dice["perturbation"] == perturbation)
    ]
    return (
        subset.groupby("epsilon_n")["dice"]
        .agg(
            median="median",
            q05=lambda values: values.quantile(0.05),
            q95=lambda values: values.quantile(0.95),
        )
        .reset_index()
    )


def plot_dice_vs_epsilon(case_dice: pd.DataFrame, output_dir: Path) -> Path:
    fig, axes = plt.subplots(1, 2, figsize=(15, 6), sharey=True)
    for ax, dataset in zip(axes, ("wg", "zones")):
        clean = _curve_statistics(case_dice, dataset, "clean").iloc[0]
        for perturbation in PERTURBATIONS:
            stats = _curve_statistics(case_dice, dataset, perturbation)
            x = np.r_[0, stats["epsilon_n"].to_numpy(dtype=float)]
            center = np.r_[clean["median"], stats["median"].to_numpy(dtype=float)]
            q05 = np.r_[clean["q05"], stats["q05"].to_numpy(dtype=float)]
            q95 = np.r_[clean["q95"], stats["q95"].to_numpy(dtype=float)]
            ax.plot(
                x,
                center,
                marker=MARKERS[perturbation],
                linewidth=2.3,
                color=COLORS[perturbation],
                label=LABELS[perturbation],
            )
            ax.fill_between(x, q05, q95, color=COLORS[perturbation], alpha=0.11)

        ax.set_title(DATASET_NAMES[dataset])
        ax.set_xticks(EPSILON_GRID, _epsilon_tick_labels(EPSILON_GRID))
        ax.set_xlabel("Normalized-space epsilon (legacy grid)")
        ax.set_ylim(-0.02, 1.03)
        ax.grid(alpha=0.25)
        ax.axhline(float(clean["median"]), color="#777777", linestyle=":", alpha=0.7)
        if dataset == "wg":
            ax.set_ylabel("Median segmentation Dice (higher is better)")

        eps16 = {
            perturbation: float(
                _curve_statistics(case_dice, dataset, perturbation)
                .set_index("epsilon_n")
                .loc[16, "median"]
            )
            for perturbation in PERTURBATIONS
        }
        ax.annotate(
            f"At 16/255:\nAPGD {eps16['apgd_bce']:.2f}\n"
            f"Gaussian {eps16['gaussian_rms_matched']:.2f}",
            xy=(16, eps16["apgd_bce"]),
            xytext=(9.2, 0.18 if dataset == "wg" else 0.24),
            arrowprops={"arrowstyle": "->", "color": COLORS["apgd_bce"]},
            bbox={"boxstyle": "round,pad=0.3", "facecolor": "white", "alpha": 0.9},
        )

    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper center", ncol=3, frameon=False)
    fig.suptitle(
        "Adversarial perturbations reduce Dice; matched random noise does not",
        y=1.02,
    )
    fig.text(
        0.5,
        -0.01,
        "Lines show the case median; shaded areas show the 5th–95th percentile range (12 cases).",
        ha="center",
        color="#444444",
    )
    fig.tight_layout(rect=(0, 0.03, 1, 0.94))
    path = output_dir / "01_dice_vs_epsilon.png"
    fig.savefig(path, dpi=240, bbox_inches="tight")
    plt.close(fig)
    return path


def _apgd_matrix(
    case_dice: pd.DataFrame, dataset: str
) -> tuple[pd.DataFrame, str | None]:
    clean = case_dice[
        (case_dice["dataset"] == dataset) & (case_dice["perturbation"] == "clean")
    ][["case_id", "dice"]].rename(columns={"dice": 0})
    apgd = case_dice[
        (case_dice["dataset"] == dataset) & (case_dice["perturbation"] == "apgd_bce")
    ].pivot(index="case_id", columns="epsilon_n", values="dice")
    matrix = clean.set_index("case_id").join(apgd).reindex(columns=EPSILON_GRID)
    sort_column = 16 if 16 in matrix.columns else max(matrix.columns)
    matrix = matrix.sort_values(sort_column)
    outlier: str | None = None
    if 2 in matrix.columns:
        low_epsilon_drop = matrix[0] - matrix[2]
        candidate = str(low_epsilon_drop.idxmax())
        if float(low_epsilon_drop.max()) > 0.25:
            outlier = candidate
    return matrix, outlier


def plot_case_heatmap(
    case_dice: pd.DataFrame, output_dir: Path
) -> tuple[Path, str | None]:
    fig, axes = plt.subplots(1, 2, figsize=(15, 8), constrained_layout=True)
    image = None
    wg_outlier: str | None = None
    for ax, dataset in zip(axes, ("wg", "zones")):
        matrix, outlier = _apgd_matrix(case_dice, dataset)
        if dataset == "wg":
            wg_outlier = outlier
        values = matrix.to_numpy(dtype=float)
        image = ax.imshow(values, cmap="RdYlGn", vmin=0, vmax=1, aspect="auto")
        aliases = []
        for case_id in matrix.index:
            alias = _short_case_id(dataset, str(case_id))
            if str(case_id) == outlier:
                alias += " *"
            aliases.append(alias)
        ax.set_yticks(np.arange(len(matrix)), aliases)
        ax.set_xticks(np.arange(len(EPSILON_GRID)), _epsilon_tick_labels(EPSILON_GRID))
        ax.set_xlabel("Normalized-space epsilon")
        ax.set_title(f"{DATASET_NAMES[dataset]}: each row is one case")
        for row in range(values.shape[0]):
            for column in range(values.shape[1]):
                value = values[row, column]
                text_color = "white" if value < 0.28 else "black"
                ax.text(
                    column,
                    row,
                    f"{value:.2f}",
                    ha="center",
                    va="center",
                    fontsize=8,
                    color=text_color,
                )
    if image is not None:
        colorbar = fig.colorbar(image, ax=axes, shrink=0.78, pad=0.02)
        colorbar.set_label("Dice: green = good, red = failure")
    fig.suptitle("Adversarial vulnerability is not identical across cases", y=1.035)
    if wg_outlier is not None:
        fig.text(
            0.01,
            -0.025,
            "* The marked WG case collapses at 2/255 and has an unusual 16-bit acquisition "
            "scale; it strongly affects the WG mean at low epsilon.",
            fontsize=9,
            color="#444444",
        )
    path = output_dir / "02_case_vulnerability_heatmap.png"
    fig.savefig(path, dpi=240, bbox_inches="tight")
    plt.close(fig)
    return path, wg_outlier


def plot_paired_controls(
    case_dice: pd.DataFrame, output_dir: Path, *, epsilon_n: int = 16
) -> Path:
    fig, axes = plt.subplots(1, 2, figsize=(15, 8), sharex=True)
    for ax, dataset in zip(axes, ("wg", "zones")):
        clean = case_dice[
            (case_dice["dataset"] == dataset) & (case_dice["perturbation"] == "clean")
        ][["case_id", "dice"]].rename(columns={"dice": "clean"})
        altered = case_dice[
            (case_dice["dataset"] == dataset)
            & (case_dice["epsilon_n"] == epsilon_n)
            & (case_dice["perturbation"].isin(PERTURBATIONS))
        ].pivot(index="case_id", columns="perturbation", values="dice")
        values = clean.set_index("case_id").join(altered)
        values = values.sort_values("apgd_bce")
        y = np.arange(len(values))

        for row, (_, record) in enumerate(values.iterrows()):
            ax.hlines(
                row,
                float(record["apgd_bce"]),
                float(record["clean"]),
                color="#BBBBBB",
                linewidth=1.2,
                zorder=1,
            )
        for perturbation in ("clean", *PERTURBATIONS):
            ax.scatter(
                values[perturbation],
                y,
                color=COLORS[perturbation],
                marker=MARKERS[perturbation],
                s=48,
                label=LABELS[perturbation],
                zorder=3,
                edgecolors="white",
                linewidths=0.5,
            )

        aliases = [_short_case_id(dataset, str(case_id)) for case_id in values.index]
        ax.set_yticks(y, aliases)
        ax.invert_yaxis()
        ax.set_xlim(-0.02, 1.02)
        ax.set_xlabel("Dice (higher is better)")
        ax.grid(axis="x", alpha=0.25)
        worse_than_both = int(
            (
                (values["apgd_bce"] < values["gaussian_rms_matched"])
                & (values["apgd_bce"] < values["rician_proxy_rms_matched"])
            ).sum()
        )
        ax.set_title(
            f"{DATASET_NAMES[dataset]}\n"
            f"APGD worse than both controls in {worse_than_both}/{len(values)} cases"
        )

    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper center", ncol=4, frameon=False)
    fig.suptitle(
        f"Case-by-case comparison at ε_norm={epsilon_n}/255 ({100 * epsilon_n / 255:.2f}% σ)",
        y=1.03,
    )
    fig.tight_layout(rect=(0, 0, 1, 0.93))
    path = output_dir / f"03_paired_controls_epsilon_{epsilon_n}.png"
    fig.savefig(path, dpi=240, bbox_inches="tight")
    plt.close(fig)
    return path


def _quality_dice_join(quality: pd.DataFrame, dice: pd.DataFrame) -> pd.DataFrame:
    macro = dice[dice["class"] == "macro"][
        ["dataset", "case_id", "epsilon_n", "perturbation", "trial", "dice"]
    ]
    joined = quality.merge(
        macro,
        on=["dataset", "case_id", "epsilon_n", "perturbation", "trial"],
        validate="one_to_one",
    )
    return (
        joined.groupby(
            ["dataset", "case_id", "epsilon_n", "perturbation"],
            as_index=False,
        )[
            [
                "dice",
                "psnr_db_robust_range",
                "ssim_axial_prostate",
            ]
        ]
        .mean()
        .sort_values(["dataset", "epsilon_n", "perturbation", "case_id"])
    )


def plot_similarity_vs_dice(
    quality: pd.DataFrame,
    dice: pd.DataFrame,
    output_dir: Path,
    *,
    epsilon_n: int = 16,
) -> Path:
    joined = _quality_dice_join(quality, dice)
    joined = joined[
        (joined["epsilon_n"] == epsilon_n)
        & (joined["perturbation"].isin(PERTURBATIONS))
    ]
    metrics = (
        ("psnr_db_robust_range", "PSNR (dB; higher = numerically closer)"),
        ("ssim_axial_prostate", "SSIM (closer to 1 = structurally closer)"),
    )
    fig, axes = plt.subplots(2, 2, figsize=(15, 11), sharey=True)
    for row, dataset in enumerate(("wg", "zones")):
        dataset_values = joined[joined["dataset"] == dataset]
        for column, (metric, xlabel) in enumerate(metrics):
            ax = axes[row, column]
            for perturbation in PERTURBATIONS:
                group = dataset_values[dataset_values["perturbation"] == perturbation]
                ax.scatter(
                    group[metric],
                    group["dice"],
                    color=COLORS[perturbation],
                    marker=MARKERS[perturbation],
                    s=46,
                    alpha=0.8,
                    label=LABELS[perturbation],
                )
                ax.scatter(
                    [group[metric].mean()],
                    [group["dice"].mean()],
                    color=COLORS[perturbation],
                    marker="X",
                    s=150,
                    edgecolors="black",
                    linewidths=0.7,
                    zorder=5,
                )
            ax.set_ylim(-0.02, 1.03)
            ax.set_xlabel(xlabel)
            ax.grid(alpha=0.25)
            if column == 0:
                ax.set_ylabel(f"{DATASET_NAMES[dataset]}\nSegmentation Dice")
            if metric == "ssim_axial_prostate":
                lower = min(0.90, float(dataset_values[metric].min()) - 0.01)
                ax.set_xlim(lower, 1.003)
            ax.set_title("Each small marker is one case; X is the mean")

    handles, labels = axes[0, 0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper center", ncol=3, frameon=False)
    fig.suptitle(
        f"At ε_norm={epsilon_n}/255, similar-looking images can produce very different Dice",
        y=1.01,
    )
    fig.tight_layout(rect=(0, 0, 1, 0.95))
    path = output_dir / f"04_similarity_vs_dice_epsilon_{epsilon_n}.png"
    fig.savefig(path, dpi=240, bbox_inches="tight")
    plt.close(fig)
    return path


def plot_physical_scale(mapping: pd.DataFrame, output_dir: Path) -> Path:
    fig, axes = plt.subplots(1, 2, figsize=(15, 6))
    for dataset in ("wg", "zones"):
        group = mapping[mapping["dataset"] == dataset].sort_values("epsilon_n")
        x = group["epsilon_n"].to_numpy(dtype=float)
        color = COLORS[dataset]
        axes[0].plot(
            x,
            group["exact_case_raw_linf_median_au"],
            color=color,
            marker="o",
            linewidth=2.3,
            label=DATASET_NAMES[dataset],
        )
        axes[0].fill_between(
            x,
            group["exact_case_raw_linf_q05_au"].to_numpy(dtype=float),
            group["exact_case_raw_linf_q95_au"].to_numpy(dtype=float),
            color=color,
            alpha=0.15,
        )
        axes[1].plot(
            x,
            group["epsilon_over_raw_noise_proxy_median"],
            color=color,
            marker="o",
            linewidth=2.3,
            label=DATASET_NAMES[dataset],
        )
        axes[1].fill_between(
            x,
            group["epsilon_over_raw_noise_proxy_q05"].to_numpy(dtype=float),
            group["epsilon_over_raw_noise_proxy_q95"].to_numpy(dtype=float),
            color=color,
            alpha=0.15,
        )

    axes[0].set_title("Same normalized epsilon gives different raw-AU changes")
    axes[0].set_ylabel("Exact raw-space L∞ budget (AU)\nmedian and 5th–95th percentile")
    axes[1].set_title("Budget relative to the raw-image noise proxy")
    axes[1].set_ylabel("L∞ budget / estimated noise sigma")
    axes[1].axhline(1.0, color="#333333", linestyle="--", linewidth=1.2)
    axes[1].text(1.0, 1.08, "1× estimated noise", color="#333333")
    for ax in axes:
        ax.set_xticks(EPSILON_GRID, _epsilon_tick_labels(EPSILON_GRID))
        ax.set_xlabel("Normalized-space epsilon (legacy grid)")
        ax.grid(alpha=0.25)
        ax.legend(frameon=False)
    fig.suptitle("Physical scale: normalized units are comparable; raw MRI AU are not")
    fig.text(
        0.5,
        -0.02,
        "Noise is an image-derived high-frequency proxy, not a measured scanner-noise value.",
        ha="center",
        color="#444444",
    )
    fig.tight_layout(rect=(0, 0.03, 1, 0.94))
    path = output_dir / "05_physical_scale_and_noise_proxy.png"
    fig.savefig(path, dpi=240, bbox_inches="tight")
    plt.close(fig)
    return path


def _value_at(
    case_dice: pd.DataFrame,
    dataset: str,
    perturbation: str,
    epsilon_n: int,
    statistic: str = "mean",
) -> float:
    values = case_dice[
        (case_dice["dataset"] == dataset)
        & (case_dice["perturbation"] == perturbation)
        & (case_dice["epsilon_n"] == epsilon_n)
    ]["dice"]
    return float(getattr(values, statistic)())


def write_interpretation(
    frames: dict[str, pd.DataFrame],
    output_dir: Path,
    generated: list[Path],
    wg_outlier: str | None,
) -> Path:
    case_dice = _macro_case_dice(frames["dice"])
    quality_joined = _quality_dice_join(frames["quality"], frames["dice"])
    quality_16 = quality_joined[
        (quality_joined["epsilon_n"] == 16)
        & (quality_joined["perturbation"] == "apgd_bce")
    ]
    mapping_16 = frames["mapping"][frames["mapping"]["epsilon_n"] == 16].set_index(
        "dataset"
    )

    rows = []
    for dataset in ("wg", "zones"):
        rows.append(
            [
                DATASET_NAMES[dataset],
                f"{_value_at(case_dice, dataset, 'clean', 0):.3f}",
                f"{_value_at(case_dice, dataset, 'apgd_bce', 16):.3f}",
                f"{_value_at(case_dice, dataset, 'gaussian_rms_matched', 16):.3f}",
                f"{_value_at(case_dice, dataset, 'rician_proxy_rms_matched', 16):.3f}",
                f"{quality_16[quality_16['dataset'] == dataset]['psnr_db_robust_range'].mean():.1f}",
                f"{quality_16[quality_16['dataset'] == dataset]['ssim_axial_prostate'].mean():.3f}",
            ]
        )

    table = [
        "| Dataset | Clean Dice | APGD Dice | Gaussian Dice | Rician Dice | APGD PSNR | APGD SSIM |",
        "| --- | ---: | ---: | ---: | ---: | ---: | ---: |",
        *("| " + " | ".join(row) + " |" for row in rows),
    ]

    outlier_text = ""
    if wg_outlier is not None:
        clean = _value_at(case_dice, "wg", "clean", 0)
        record = case_dice[
            (case_dice["dataset"] == "wg")
            & (case_dice["case_id"] == wg_outlier)
            & (case_dice["perturbation"] == "clean")
        ]
        outlier_clean = float(record["dice"].iloc[0])
        outlier_eps2 = float(
            case_dice[
                (case_dice["dataset"] == "wg")
                & (case_dice["case_id"] == wg_outlier)
                & (case_dice["perturbation"] == "apgd_bce")
                & (case_dice["epsilon_n"] == 2)
            ]["dice"].iloc[0]
        )
        outlier_text = (
            f"One WG case, `{_short_case_id('wg', wg_outlier)}`, has clean Dice "
            f"{outlier_clean:.3f} but APGD Dice {outlier_eps2:.3f} at only `2/255`. "
            "Its raw standard deviation is unusually large because it uses a different 16-bit "
            "intensity scale. The attack panel deliberately spans the noise-proxy range, so this "
            "outlier is included. It explains why the low-epsilon WG mean looks much worse than "
            f"the typical case; the overall clean WG mean is {clean:.3f}."
        )

    lines = [
        "# Visual interpretation of the epsilon calibration",
        "",
        "## Main conclusion",
        "",
        "The segmentation failure is not explained by perturbation magnitude alone. At the same "
        "RMS magnitude, Gaussian and Rician-proxy noise leave Dice almost unchanged, whereas the "
        "adversarial pattern causes large failures. The model is sensitive to *where and in which "
        "direction* intensities change.",
        "",
        *table,
        "",
        "The table uses mean Dice across the 12-case experiment panel at "
        "`epsilon_norm=16/255`. PSNR and SSIM remain high, meaning the adversarial images are "
        "numerically and structurally close to the clean images despite the segmentation damage.",
        "",
        "## 1. Overall response to epsilon",
        "",
        f"![Dice versus epsilon]({generated[0].name})",
        "",
        "The red adversarial curve falls as epsilon grows. The blue and green random-noise "
        "curves remain close to the clean baseline. The shaded ranges show that case-to-case "
        "variation is substantial, especially for WG.",
        "",
        "## 2. Individual cases",
        "",
        f"![Case vulnerability heatmap]({generated[1].name})",
        "",
        "Green means good overlap; red means segmentation failure. WG collapses almost completely "
        "by `32/255`. Zones degrade more gradually, but most cases are below Dice 0.5 by `32/255`.",
        "",
        outlier_text,
        "",
        "## 3. Same magnitude, different segmentation effect",
        "",
        f"![Paired controls]({generated[2].name})",
        "",
        "Each row is one case at `16/255`. The random controls sit near the clean result, while "
        "the adversarial point moves left. This paired design is stronger than comparing only "
        "group averages because every case acts as its own control.",
        "",
        "## 4. Why PSNR and SSIM do not guarantee model stability",
        "",
        f"![Similarity versus Dice]({generated[3].name})",
        "",
        "Moving right means the modified image is more similar to the clean image; moving up means "
        "better segmentation. Adversarial points can remain far to the right while falling far "
        "down. PSNR and SSIM measure image similarity, not whether the segmentation network is "
        "using robust features.",
        "",
        "## 5. Physical interpretation",
        "",
        f"![Physical scale]({generated[4].name})",
        "",
        f"At `16/255`, the exact median L∞ budget is "
        f"{mapping_16.loc['wg', 'exact_case_raw_linf_median_au']:.2f} AU for WG and "
        f"{mapping_16.loc['zones', 'exact_case_raw_linf_median_au']:.2f} AU for zones. "
        "The wide shaded ranges demonstrate why raw MRI AU should not be compared directly "
        "between scans. The right panel compares epsilon with an image-derived noise proxy only; "
        "it is not measured scanner noise.",
        "",
        "## What can and cannot be concluded",
        "",
        "Supported: these models are much more vulnerable to targeted adversarial patterns than "
        "to matched random Gaussian or Rician-proxy perturbations in this 12-case panel.",
        "",
        "Not yet supported: that the adversarial perturbations are physically realizable during "
        "MRI acquisition, or that the observed Dice losses generalize to an external population. "
        "That requires held-out/external cases, repeated attack seeds/restarts, and noise-only or "
        "repeated MRI acquisitions for a true acquisition-noise measurement.",
        "",
    ]
    path = output_dir / "README.md"
    path.write_text("\n".join(lines), encoding="utf-8")
    return path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Generate explanatory plots from physical-epsilon calibration CSV files"
    )
    parser.add_argument("--results-dir", type=Path, default=DEFAULT_RESULTS)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    results_dir = args.results_dir.resolve()
    output_dir = args.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    _configure_style()
    frames = _load_results(results_dir)
    case_dice = _macro_case_dice(frames["dice"])

    generated: list[Path] = []
    generated.append(plot_dice_vs_epsilon(case_dice, output_dir))
    heatmap, wg_outlier = plot_case_heatmap(case_dice, output_dir)
    generated.append(heatmap)
    generated.append(plot_paired_controls(case_dice, output_dir, epsilon_n=16))
    generated.append(
        plot_similarity_vs_dice(
            frames["quality"], frames["dice"], output_dir, epsilon_n=16
        )
    )
    generated.append(plot_physical_scale(frames["mapping"], output_dir))
    report = write_interpretation(frames, output_dir, generated, wg_outlier)

    for path in [*generated, report]:
        print(f"Saved {path}")


if __name__ == "__main__":
    main()
