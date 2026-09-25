"""Compare full-cohort FGSM, PGD, and APGD image quality.

This script consumes the attack-aware output from
``evaluate_adversarial_quality_all_cases.py``. It does not rerun attacks.
Comparisons are paired by dataset, case, and epsilon so differences cannot be
explained by unequal cohorts.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib
import numpy as np
import pandas as pd
from matplotlib.lines import Line2D

matplotlib.use("Agg")
import matplotlib.pyplot as plt


REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_RESULTS = (
    REPO_ROOT / "results" / "adv_rob_eval_results" / "image_quality_analysis"
)
DEFAULT_QUALITY_CSV = (
    DEFAULT_RESULTS
    / "all_cases_adversarial_quality"
    / "adversarial_quality_all_cases.csv"
)
DEFAULT_OUTPUT = DEFAULT_RESULTS / "psnr_ssim_all_cases_figures"

EPSILON_VALUES = [2, 4, 8, 16, 32]
EPSILON_AXIS_VALUES = [0, *EPSILON_VALUES]
EPSILON_POSITIONS = {
    epsilon_n: index for index, epsilon_n in enumerate(EPSILON_AXIS_VALUES)
}
ATTACKS = ["FGSM-BCE", "PGD-BCE", "APGD-BCE"]
ATTACK_LABELS = {
    "FGSM-BCE": "FGSM-BCE (1 step)",
    "PGD-BCE": "PGD-BCE (20 steps)",
    "APGD-BCE": "APGD-BCE (20 adaptive steps)",
}
ATTACK_COLORS = {
    "FGSM-BCE": "#0072B2",
    "PGD-BCE": "#E69F00",
    "APGD-BCE": "#D55E00",
}
ATTACK_MARKERS = {"FGSM-BCE": "o", "PGD-BCE": "s", "APGD-BCE": "^"}
ATTACK_LINESTYLES = {"FGSM-BCE": "--", "PGD-BCE": "-.", "APGD-BCE": "-"}
DATASET_LABELS = {"wg": "Whole gland (WG)", "zones": "Prostate zones"}
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


def _load_quality(path: Path) -> pd.DataFrame:
    if not path.is_file():
        raise FileNotFoundError(f"missing multi-attack quality CSV: {path}")
    frame = pd.read_csv(path)
    required = {
        "dataset",
        "case_id",
        "epsilon_n",
        "epsilon_norm",
        "attack",
        *METRICS,
    }
    missing = sorted(required - set(frame.columns))
    if missing:
        raise ValueError(f"{path} is missing required columns: {missing}")

    frame = frame[
        frame["attack"].isin(ATTACKS) & frame["epsilon_n"].isin(EPSILON_VALUES)
    ].copy()
    missing_attacks = sorted(set(ATTACKS) - set(frame["attack"]))
    if missing_attacks:
        raise ValueError(
            f"{path} is missing {missing_attacks}; run the full-cohort evaluator first"
        )
    missing_datasets = sorted(set(DATASET_LABELS) - set(frame["dataset"]))
    if missing_datasets:
        raise ValueError(f"{path} is missing datasets: {missing_datasets}")
    duplicate = frame.duplicated(
        ["dataset", "case_id", "epsilon_n", "attack"], keep=False
    )
    if duplicate.any():
        raise ValueError(f"{path} contains duplicate case/attack/epsilon rows")
    if not np.isfinite(frame[METRICS].to_numpy(dtype=float)).all():
        raise ValueError(f"{path} contains non-finite image-quality values")

    reference_keys = set(
        map(
            tuple,
            frame.loc[
                frame["attack"] == "APGD-BCE",
                ["dataset", "case_id", "epsilon_n"],
            ].itertuples(index=False),
        )
    )
    for attack in ATTACKS:
        keys = set(
            map(
                tuple,
                frame.loc[
                    frame["attack"] == attack,
                    ["dataset", "case_id", "epsilon_n"],
                ].itertuples(index=False),
            )
        )
        missing_keys = reference_keys - keys
        extra_keys = keys - reference_keys
        if missing_keys or extra_keys:
            raise ValueError(
                f"{attack} is not paired one-to-one with APGD-BCE: "
                f"{len(missing_keys)} missing and {len(extra_keys)} extra rows"
            )
    return frame


def _summarize(frame: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    for keys, group in frame.groupby(
        ["dataset", "attack", "epsilon_n", "epsilon_norm"], sort=True
    ):
        dataset, attack, epsilon_n, epsilon_norm = keys
        row: dict[str, object] = {
            "dataset": dataset,
            "attack": attack,
            "epsilon_n": int(epsilon_n),
            "epsilon_norm": float(epsilon_norm),
            "n_cases": int(group["case_id"].nunique()),
        }
        for metric in METRICS:
            values = group[metric].to_numpy(dtype=float)
            row[f"{metric}_mean"] = float(np.mean(values))
            row[f"{metric}_median"] = float(np.median(values))
            row[f"{metric}_q05"] = float(np.quantile(values, 0.05))
            row[f"{metric}_q95"] = float(np.quantile(values, 0.95))
        rows.append(row)
    return pd.DataFrame(rows).sort_values(["dataset", "attack", "epsilon_n"])


def _paired_against_apgd(frame: pd.DataFrame) -> pd.DataFrame:
    key_columns = ["dataset", "case_id", "epsilon_n", "epsilon_norm"]
    apgd = frame[frame["attack"] == "APGD-BCE"][key_columns + METRICS].rename(
        columns={metric: f"apgd_{metric}" for metric in METRICS}
    )
    comparisons = frame[frame["attack"] != "APGD-BCE"].merge(
        apgd,
        on=key_columns,
        how="inner",
        validate="many_to_one",
    )
    for metric in METRICS:
        comparisons[f"{metric}_minus_apgd"] = (
            comparisons[metric] - comparisons[f"apgd_{metric}"]
        )
    return comparisons.sort_values(["dataset", "attack", "case_id", "epsilon_n"])


def _summarize_paired(paired: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    difference_metrics = [
        "rms_percent_case_sigma_minus_apgd",
        "psnr_db_robust_range_minus_apgd",
        "ssim_axial_prostate_minus_apgd",
    ]
    for keys, group in paired.groupby(
        ["dataset", "attack", "epsilon_n", "epsilon_norm"], sort=True
    ):
        dataset, attack, epsilon_n, epsilon_norm = keys
        row: dict[str, object] = {
            "dataset": dataset,
            "attack": attack,
            "reference_attack": "APGD-BCE",
            "epsilon_n": int(epsilon_n),
            "epsilon_norm": float(epsilon_norm),
            "n_cases": int(group["case_id"].nunique()),
        }
        for metric in difference_metrics:
            values = group[metric].to_numpy(dtype=float)
            row[f"{metric}_median"] = float(np.median(values))
            row[f"{metric}_q05"] = float(np.quantile(values, 0.05))
            row[f"{metric}_q95"] = float(np.quantile(values, 0.95))
            row[f"{metric}_positive_fraction"] = float(np.mean(values > 0))
        rows.append(row)
    return pd.DataFrame(rows).sort_values(["dataset", "attack", "epsilon_n"])


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


def _padded_limits(
    values: pd.Series,
    *,
    lower_bound: float | None = None,
    upper_bound: float | None = None,
) -> tuple[float, float]:
    finite = values[np.isfinite(values.to_numpy(dtype=float))].to_numpy(dtype=float)
    lower = float(np.min(finite))
    upper = float(np.max(finite))
    padding = max(0.04 * (upper - lower), 1e-3)
    lower -= padding
    upper += padding
    if lower_bound is not None:
        lower = max(lower, lower_bound)
    if upper_bound is not None:
        upper = min(upper, upper_bound)
    return lower, upper


def plot_attack_metric_comparison(summary: pd.DataFrame, output_dir: Path) -> Path:
    configurations = (
        (
            "psnr_db_robust_range",
            "PSNR (dB; higher = more similar to clean)",
        ),
        (
            "ssim_axial_prostate",
            "SSIM (higher = more structurally similar to clean)",
        ),
    )
    fig, axes = plt.subplots(2, 2, figsize=(15, 10), sharex=True)
    for row, dataset in enumerate(("wg", "zones")):
        for column, (metric, ylabel) in enumerate(configurations):
            ax = axes[row, column]
            for attack in ATTACKS:
                group = summary[
                    (summary["dataset"] == dataset) & (summary["attack"] == attack)
                ].sort_values("epsilon_n")
                x = np.asarray(
                    [EPSILON_POSITIONS[int(value)] for value in group["epsilon_n"]],
                    dtype=float,
                )
                center = group[f"{metric}_median"].to_numpy(dtype=float)
                q05 = group[f"{metric}_q05"].to_numpy(dtype=float)
                q95 = group[f"{metric}_q95"].to_numpy(dtype=float)
                if metric == "ssim_axial_prostate":
                    x = np.insert(x, 0, EPSILON_POSITIONS[0])
                    center = np.insert(center, 0, 1.0)
                    q05 = np.insert(q05, 0, 1.0)
                    q95 = np.insert(q95, 0, 1.0)
                ax.plot(
                    x,
                    center,
                    color=ATTACK_COLORS[attack],
                    marker=ATTACK_MARKERS[attack],
                    linewidth=2.2,
                    markersize=6,
                    label=ATTACK_LABELS[attack],
                )
                ax.fill_between(x, q05, q95, color=ATTACK_COLORS[attack], alpha=0.09)
            _configure_epsilon_axis(ax)
            ax.set_ylabel(ylabel)
            if metric == "psnr_db_robust_range":
                _annotate_clean_psnr(ax)
            ax.set_title(
                f"{DATASET_LABELS[dataset]} — {'PSNR' if column == 0 else 'SSIM'}"
            )
            ax.grid(alpha=0.25)
    handles, labels = axes[0, 0].get_legend_handles_labels()
    fig.legend(
        handles,
        labels,
        loc="upper center",
        ncol=3,
        frameon=False,
        bbox_to_anchor=(0.5, 0.955),
    )
    fig.suptitle("Image similarity under FGSM, PGD, and APGD at the same ε", y=0.995)
    fig.text(
        0.5,
        0.012,
        "Lines are cohort medians; shading is the 5th–95th percentile across cases. "
        "All attacks maximize the same BCE loss. At 0/255, RMS = 0, PSNR = ∞, "
        "and SSIM = 1.",
        ha="center",
        color="#444444",
    )
    fig.tight_layout(rect=(0, 0.04, 1, 0.91))
    path = output_dir / "08_fgsm_pgd_apgd_psnr_ssim_comparison.png"
    fig.savefig(path, dpi=240, bbox_inches="tight")
    plt.close(fig)
    return path


def plot_attack_joint_trajectories(frame: pd.DataFrame, output_dir: Path) -> Path:
    marker_sizes = {2: 48, 4: 65, 8: 88, 16: 118, 32: 155}
    medians = (
        frame.groupby(["dataset", "attack", "epsilon_n"], as_index=False)[
            ["psnr_db_robust_range", "ssim_axial_prostate"]
        ]
        .median()
        .sort_values(["dataset", "attack", "epsilon_n"])
        .reset_index(drop=True)
    )

    fig, axes = plt.subplots(1, 2, figsize=(15, 6.4), sharex=True, sharey=True)
    for ax, dataset in zip(axes, ("wg", "zones")):
        for attack in ATTACKS:
            trajectory = (
                medians[(medians["attack"] == attack) & (medians["dataset"] == dataset)]
                .sort_values("epsilon_n")
                .reset_index(drop=True)
            )
            ax.plot(
                trajectory["psnr_db_robust_range"],
                trajectory["ssim_axial_prostate"],
                color=ATTACK_COLORS[attack],
                linestyle=ATTACK_LINESTYLES[attack],
                linewidth=2.5,
                zorder=3,
            )
            for _index, record in trajectory.iterrows():
                epsilon_n = int(record["epsilon_n"])
                ax.scatter(
                    [record["psnr_db_robust_range"]],
                    [record["ssim_axial_prostate"]],
                    color=ATTACK_COLORS[attack],
                    marker=ATTACK_MARKERS[attack],
                    s=marker_sizes[epsilon_n],
                    edgecolors="white",
                    linewidths=1.1,
                    zorder=5,
                )
            if len(trajectory) >= 2:
                previous, last = trajectory.iloc[-2], trajectory.iloc[-1]
                ax.annotate(
                    "",
                    xy=(last["psnr_db_robust_range"], last["ssim_axial_prostate"]),
                    xytext=(
                        previous["psnr_db_robust_range"],
                        previous["ssim_axial_prostate"],
                    ),
                    arrowprops={
                        "arrowstyle": "-|>",
                        "color": ATTACK_COLORS[attack],
                        "lw": 1.8,
                        "shrinkA": 9,
                        "shrinkB": 9,
                    },
                )
        ax.set_title(DATASET_LABELS[dataset])
        ax.set_xlabel("PSNR (dB; higher = more similar to clean)")
        ax.grid(alpha=0.25)

    axes[0].set_ylabel("SSIM (higher = more structurally similar to clean)")
    axes[0].set_xlim(*_padded_limits(medians["psnr_db_robust_range"]))
    axes[0].set_ylim(
        *_padded_limits(
            medians["ssim_axial_prostate"], lower_bound=-1.0, upper_bound=1.005
        )
    )
    attack_handles = [
        Line2D(
            [0],
            [0],
            color=ATTACK_COLORS[attack],
            marker=ATTACK_MARKERS[attack],
            markerfacecolor=ATTACK_COLORS[attack],
            markeredgecolor="white",
            linestyle=ATTACK_LINESTYLES[attack],
            linewidth=2.2,
            label=ATTACK_LABELS[attack],
        )
        for attack in ATTACKS
    ]
    fig.legend(
        handles=attack_handles,
        loc="upper center",
        ncol=3,
        frameon=False,
        bbox_to_anchor=(0.5, 0.91),
    )
    epsilon_handles = [
        Line2D(
            [0],
            [0],
            linestyle="none",
            marker="o",
            markersize=np.sqrt(marker_sizes[epsilon_n]),
            markerfacecolor="#777777",
            markeredgecolor="white",
            label=f"{epsilon_n}/255",
        )
        for epsilon_n in EPSILON_VALUES
    ]
    fig.legend(
        handles=epsilon_handles,
        loc="upper center",
        ncol=5,
        frameon=False,
        title="Maximum allowed voxel change, ε (marker size)",
        bbox_to_anchor=(0.5, 0.835),
    )
    fig.subplots_adjust(left=0.07, right=0.98, bottom=0.17, top=0.72, wspace=0.15)
    fig.suptitle("Median PSNR–SSIM trajectories: FGSM vs PGD vs APGD", y=0.985)
    fig.text(
        0.5,
        0.075,
        "Each point is the cohort median; larger markers and arrowheads indicate "
        "stronger perturbations (2 → 32/255).",
        ha="center",
        color="#444444",
    )
    fig.text(
        0.5,
        0.035,
        "Clean reference (0/255): RMS = 0, PSNR = ∞, SSIM = 1 "
        "(not plotted on the finite PSNR axis).",
        ha="center",
        color="#444444",
    )
    path = output_dir / "09_fgsm_pgd_apgd_psnr_ssim_trajectories.png"
    fig.savefig(path, dpi=240, bbox_inches="tight")
    plt.close(fig)
    return path


def plot_paired_differences(paired_summary: pd.DataFrame, output_dir: Path) -> Path:
    configurations = (
        (
            "psnr_db_robust_range_minus_apgd",
            "PSNR difference from APGD (dB)",
            "Above 0: attack image is closer to clean than APGD",
        ),
        (
            "ssim_axial_prostate_minus_apgd",
            "SSIM difference from APGD",
            "Above 0: attack preserves more local structure than APGD",
        ),
    )
    fig, axes = plt.subplots(2, 2, figsize=(15, 10), sharex=True)
    for row, dataset in enumerate(("wg", "zones")):
        for column, (metric, ylabel, note) in enumerate(configurations):
            ax = axes[row, column]
            for attack in ("FGSM-BCE", "PGD-BCE"):
                group = paired_summary[
                    (paired_summary["dataset"] == dataset)
                    & (paired_summary["attack"] == attack)
                ].sort_values("epsilon_n")
                x = np.asarray(
                    [EPSILON_POSITIONS[int(value)] for value in group["epsilon_n"]],
                    dtype=float,
                )
                center = np.insert(
                    group[f"{metric}_median"].to_numpy(dtype=float), 0, 0.0
                )
                q05 = np.insert(group[f"{metric}_q05"].to_numpy(dtype=float), 0, 0.0)
                q95 = np.insert(group[f"{metric}_q95"].to_numpy(dtype=float), 0, 0.0)
                x = np.insert(x, 0, EPSILON_POSITIONS[0])
                ax.plot(
                    x,
                    center,
                    color=ATTACK_COLORS[attack],
                    marker=ATTACK_MARKERS[attack],
                    linewidth=2.2,
                    markersize=6,
                    label=f"{attack} − APGD-BCE",
                )
                ax.fill_between(x, q05, q95, color=ATTACK_COLORS[attack], alpha=0.10)
            ax.axhline(0.0, color="#333333", linewidth=1.0, linestyle="--")
            _configure_epsilon_axis(ax)
            ax.set_ylabel(ylabel)
            ax.set_title(
                f"{DATASET_LABELS[dataset]} — {'PSNR' if column == 0 else 'SSIM'}"
            )
            ax.grid(alpha=0.25)
            ax.text(
                0.03,
                0.05,
                note,
                transform=ax.transAxes,
                fontsize=8,
                color="#444444",
            )
    handles, labels = axes[0, 0].get_legend_handles_labels()
    fig.legend(
        handles,
        labels,
        loc="upper center",
        ncol=2,
        frameon=False,
        bbox_to_anchor=(0.5, 0.955),
    )
    fig.suptitle("Within-case image-similarity difference relative to APGD", y=0.995)
    fig.text(
        0.5,
        0.012,
        "Lines are median paired differences; shading is the 5th–95th percentile. "
        "Every comparison uses the same case and ε. At clean 0/255, all attacks are "
        "identical and the difference is 0.",
        ha="center",
        color="#444444",
    )
    fig.tight_layout(rect=(0, 0.04, 1, 0.91))
    path = output_dir / "10_fgsm_pgd_minus_apgd_psnr_ssim.png"
    fig.savefig(path, dpi=240, bbox_inches="tight")
    plt.close(fig)
    return path


def _markdown_summary_table(summary: pd.DataFrame) -> list[str]:
    lines = [
        "| Cohort | Attack | ε | Cases | Median RMS (% SD) | Median PSNR | Median SSIM |",
        "| --- | --- | ---: | ---: | ---: | ---: | ---: |",
    ]
    for row in summary.itertuples(index=False):
        lines.append(
            f"| {DATASET_LABELS[str(row.dataset)]} | {ATTACK_LABELS[str(row.attack)]} "
            f"| {int(row.epsilon_n)}/255 | {int(row.n_cases):,} "
            f"| {row.rms_percent_case_sigma_median:.2f}% "
            f"| {row.psnr_db_robust_range_median:.2f} dB "
            f"| {row.ssim_axial_prostate_median:.4f} |"
        )
    return lines


def _markdown_paired_table(paired_summary: pd.DataFrame) -> list[str]:
    lines = [
        "| Cohort | Comparison | ε | Median ΔPSNR | Median ΔSSIM | Cases with higher SSIM than APGD |",
        "| --- | --- | ---: | ---: | ---: | ---: |",
    ]
    for row in paired_summary.itertuples(index=False):
        lines.append(
            f"| {DATASET_LABELS[str(row.dataset)]} | {row.attack} − APGD-BCE "
            f"| {int(row.epsilon_n)}/255 "
            f"| {row.psnr_db_robust_range_minus_apgd_median:+.2f} dB "
            f"| {row.ssim_axial_prostate_minus_apgd_median:+.4f} "
            f"| {100 * row.ssim_axial_prostate_minus_apgd_positive_fraction:.1f}% |"
        )
    return lines


def write_report(
    frame: pd.DataFrame,
    summary: pd.DataFrame,
    paired_summary: pd.DataFrame,
    figures: list[Path],
    output_dir: Path,
    source_csv: Path,
) -> Path:
    cohort = " and ".join(
        f"{frame.loc[frame['dataset'] == dataset, 'case_id'].nunique():,} "
        f"{DATASET_LABELS[dataset]}"
        for dataset in ("wg", "zones")
    )
    lines = [
        "# FGSM, PGD, and APGD image-quality comparison",
        "",
        f"This paired analysis contains {cohort} cases at all five epsilon values.",
        "",
        f"Source data: `{source_csv}`",
        "",
        "## Experimental comparison",
        "",
        "All three white-box attacks maximize the same masked BCE segmentation loss against "
        "the same fold-all UNet, use the same normalized input, ground truth, L-infinity "
        "budget, and clean-image intensity bounds. FGSM uses one step; PGD uses 20 fixed "
        "steps from a deterministic random start; APGD uses the existing 20-step adaptive "
        "procedure from a deterministic random start.",
        "",
        "## 1. PSNR and SSIM versus epsilon",
        "",
        f"![Attack comparison]({figures[0].name})",
        "",
        "This figure compares the cohort medians at the same maximum per-voxel budget. "
        "Separation in PSNR primarily reflects a difference in realized RMS perturbation "
        "magnitude. Separation in SSIM additionally reflects a difference in spatial "
        "perturbation structure.",
        "",
        "## 2. Joint PSNR–SSIM trajectories",
        "",
        f"![Attack trajectories]({figures[1].name})",
        "",
        "The two cohort panels share the same axes. Colors, marker shapes, and line "
        "styles identify attacks; marker size identifies epsilon. Only cohort medians "
        "are shown so the three trajectories can be compared without case-cloud "
        "overplotting.",
        "",
        "## 3. Paired differences relative to APGD",
        "",
        f"![Paired differences]({figures[2].name})",
        "",
        "Positive values mean FGSM or PGD produced an image closer to clean than APGD for "
        "the same patient and epsilon. These are image-similarity differences, not direct "
        "measures of attack success against the segmentation.",
        "",
        "## Cohort summaries",
        "",
        *_markdown_summary_table(summary),
        "",
        "## Paired comparisons",
        "",
        *_markdown_paired_table(paired_summary),
        "",
        "## Interpretation limits",
        "",
        "- Equal epsilon does not guarantee equal realized RMS, so PSNR differences cannot "
        "be attributed only to spatial attack structure.",
        "- Higher PSNR or SSIM means closer to the clean MRI; it does not mean a weaker "
        "segmentation attack. Attack effectiveness must be assessed with paired Dice or "
        "boundary metrics.",
        "- PSNR and SSIM do not establish radiologist visibility, diagnostic adequacy, "
        "segmentation safety, or patient harm.",
        "- The attacks use one restart and 20 iterative steps; stronger configurations may "
        "change the attack ranking.",
        "",
    ]
    path = output_dir / "ADVERSARIAL_ATTACK_COMPARISON.md"
    path.write_text("\n".join(lines), encoding="utf-8")
    return path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Plot paired FGSM/PGD/APGD PSNR and SSIM comparisons"
    )
    parser.add_argument("--quality-csv", type=Path, default=DEFAULT_QUALITY_CSV)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    quality_csv = args.quality_csv.resolve()
    output_dir = args.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    _configure_style()

    frame = _load_quality(quality_csv)
    summary = _summarize(frame)
    paired = _paired_against_apgd(frame)
    paired_summary = _summarize_paired(paired)
    figures = [
        plot_attack_metric_comparison(summary, output_dir),
        plot_attack_joint_trajectories(frame, output_dir),
        plot_paired_differences(paired_summary, output_dir),
    ]
    summary_path = output_dir / "adversarial_attack_psnr_ssim_summary.csv"
    paired_path = output_dir / "adversarial_attack_paired_differences.csv"
    summary.to_csv(summary_path, index=False)
    paired_summary.to_csv(paired_path, index=False)
    report = write_report(
        frame,
        summary,
        paired_summary,
        figures,
        output_dir,
        quality_csv,
    )
    for path in [*figures, summary_path, paired_path, report]:
        print(f"Saved {path}")


if __name__ == "__main__":
    main()
