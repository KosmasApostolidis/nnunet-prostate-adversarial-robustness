"""Run RMS-matched and image-quality-versus-segmentation-damage experiments.

This analysis joins the full-cohort image-quality and segmentation passes using
the exact dataset/case/attack/epsilon key. It then:

1. interpolates every patient's measured attack trajectory to common realized
   RMS targets;
2. quantifies segmentation damage as a function of RMS, SSIM, and PSNR;
3. estimates the image distortion required to cross pre-specified Dice-drop
   thresholds; and
4. identifies cohort-level Pareto-efficient attack operating points.

All primary comparisons remain paired at the patient level.
"""

from __future__ import annotations

import argparse
import json
import math
import warnings
from pathlib import Path

import matplotlib
import numpy as np
import pandas as pd

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.figure import Figure
from matplotlib.lines import Line2D


REPO_ROOT = Path(__file__).resolve().parents[1]
CALIBRATION_ROOT = (
    REPO_ROOT / "results" / "adv_rob_eval_results" / "image_quality_analysis"
)
DEFAULT_QUALITY_CSV = (
    CALIBRATION_ROOT
    / "all_cases_adversarial_segmentation_with_boundaries"
    / "adversarial_quality_same_pass.csv"
)
DEFAULT_SEGMENTATION_CSV = (
    CALIBRATION_ROOT
    / "all_cases_adversarial_segmentation_with_boundaries"
    / "adversarial_segmentation_all_cases.csv"
)
DEFAULT_OUTPUT_DIR = CALIBRATION_ROOT / "attack_quality_damage_experiments"

ATTACKS = ["FGSM-BCE", "PGD-BCE", "APGD-BCE"]
ATTACK_LABELS = {
    "FGSM-BCE": "FGSM",
    "PGD-BCE": "PGD",
    "APGD-BCE": "APGD",
}
ATTACK_COLORS = {
    "FGSM-BCE": "#0072B2",
    "PGD-BCE": "#E69F00",
    "APGD-BCE": "#D55E00",
}
ATTACK_MARKERS = {"FGSM-BCE": "o", "PGD-BCE": "s", "APGD-BCE": "^"}
ATTACK_LINESTYLES = {
    "FGSM-BCE": "-",
    "PGD-BCE": "--",
    "APGD-BCE": "-.",
}
DATASET_LABELS = {"wg": "Whole gland (WG)", "zones": "Prostate zones"}
CLASS_LABELS = {
    ("wg", "WG"): "Whole gland",
    ("zones", "TZ+CZ"): "Transition + central zones",
    ("zones", "PZ"): "Peripheral zone",
}
JOIN_KEYS = ["dataset", "case_id", "epsilon_n", "attack"]


def _configure_style() -> None:
    plt.rcParams.update(
        {
            "font.size": 12,
            "axes.titlesize": 15,
            "axes.titleweight": "semibold",
            "axes.labelsize": 13,
            "axes.labelpad": 8,
            "xtick.labelsize": 11,
            "ytick.labelsize": 11,
            "legend.fontsize": 11,
            "figure.titlesize": 18,
            "figure.titleweight": "semibold",
            "axes.linewidth": 1.0,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "xtick.major.size": 5,
            "ytick.major.size": 5,
            "xtick.major.width": 1.0,
            "ytick.major.width": 1.0,
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
            "savefig.facecolor": "white",
        }
    )


def _plot_attack_series(
    ax: plt.Axes,
    x: np.ndarray | pd.Series,
    y: np.ndarray | pd.Series,
    attack: str,
) -> None:
    ax.plot(
        x,
        y,
        color=ATTACK_COLORS[attack],
        linestyle=ATTACK_LINESTYLES[attack],
        marker=ATTACK_MARKERS[attack],
        markersize=7.5,
        markeredgecolor="white",
        markeredgewidth=0.8,
        linewidth=2.6,
        label=ATTACK_LABELS[attack],
        zorder=3,
    )


def _style_axis(ax: plt.Axes) -> None:
    ax.set_axisbelow(True)
    ax.grid(color="#D5D5D5", linewidth=0.8, alpha=0.7)
    ax.tick_params(axis="both", which="major", pad=5)


def _add_panel_labels(axes: np.ndarray) -> None:
    for label, ax in zip("ABCDEFGHIJKLMNOPQRSTUVWXYZ", axes.flat):
        ax.text(
            -0.10,
            1.04,
            label,
            transform=ax.transAxes,
            fontsize=14,
            fontweight="bold",
            ha="left",
            va="bottom",
        )


def _save_figure(fig: Figure, path: Path) -> None:
    fig.savefig(path, dpi=300, bbox_inches="tight", facecolor="white")
    fig.savefig(path.with_suffix(".pdf"), bbox_inches="tight", facecolor="white")


def _require_columns(frame: pd.DataFrame, required: set[str], path: Path) -> None:
    missing = sorted(required - set(frame.columns))
    if missing:
        raise ValueError(f"{path} is missing required columns: {missing}")


def _add_damage_characterization_columns(frame: pd.DataFrame) -> pd.DataFrame:
    """Add overlap-independent volume and boundary-availability outcomes."""

    characterized = frame.copy()
    characterized["clean_absolute_volume_error_percent"] = characterized[
        "clean_volume_error_percent"
    ].abs()
    characterized["adversarial_absolute_volume_error_percent"] = characterized[
        "adversarial_volume_error_percent"
    ].abs()
    characterized["absolute_volume_error_increase_percent"] = (
        characterized["adversarial_absolute_volume_error_percent"]
        - characterized["clean_absolute_volume_error_percent"]
    )
    characterized["adversarial_empty_prediction"] = (
        characterized["adversarial_prediction_foreground_voxels"] <= 0
    )
    characterized["boundary_metrics_defined"] = np.isfinite(
        characterized["hd95_increase_mm"]
    ) & np.isfinite(characterized["asd_increase_mm"])
    return characterized


def load_class_damage_data(segmentation_path: Path) -> pd.DataFrame:
    """Load non-macro rows with the same attack-quality characterization."""

    segmentation = pd.read_csv(segmentation_path)
    required = {
        *JOIN_KEYS,
        "class",
        "class_label",
        "rms_norm_recomputed",
        "rms_percent_case_sigma",
        "robust_signal_range_norm",
        "psnr_db_robust_range",
        "ssim_axial_prostate",
        "clean_dice",
        "adversarial_dice",
        "dice_drop",
        "clean_iou",
        "adversarial_iou",
        "iou_drop",
        "clean_hd95_mm",
        "adversarial_hd95_mm",
        "hd95_increase_mm",
        "clean_asd_mm",
        "adversarial_asd_mm",
        "asd_increase_mm",
        "target_foreground_voxels",
        "clean_prediction_foreground_voxels",
        "adversarial_prediction_foreground_voxels",
        "clean_volume_error_percent",
        "adversarial_volume_error_percent",
    }
    _require_columns(segmentation, required, segmentation_path)
    classes = segmentation[
        (segmentation["class"] != "macro")
        & segmentation["attack"].isin(ATTACKS)
        & segmentation["dataset"].isin(DATASET_LABELS)
    ].copy()
    class_keys = [*JOIN_KEYS, "class"]
    if classes.duplicated(class_keys).any():
        raise ValueError("segmentation CSV contains duplicate per-class join keys")
    return _add_damage_characterization_columns(classes)


def _strict_macro_boundary_rows(segmentation: pd.DataFrame) -> pd.DataFrame:
    """Require every foreground class when aggregating macro boundary metrics."""

    boundary_columns = [
        "clean_hd95_mm",
        "adversarial_hd95_mm",
        "hd95_increase_mm",
        "clean_asd_mm",
        "adversarial_asd_mm",
        "asd_increase_mm",
    ]
    classes = segmentation[segmentation["class"] != "macro"]
    grouped = classes.groupby(JOIN_KEYS, sort=False)[boundary_columns]
    means = grouped.mean()
    finite_counts = grouped.count()
    expected_counts = grouped.size()
    strict = means.where(finite_counts.eq(expected_counts, axis=0)).reset_index()
    macro = segmentation[segmentation["class"] == "macro"].drop(
        columns=boundary_columns
    )
    return macro.merge(strict, on=JOIN_KEYS, how="left", validate="one_to_one")


def load_paired_data(
    quality_path: Path,
    segmentation_path: Path,
    *,
    norm_relative_tolerance: float,
) -> tuple[pd.DataFrame, dict[str, float]]:
    if not quality_path.is_file():
        raise FileNotFoundError(f"missing image-quality CSV: {quality_path}")
    if not segmentation_path.is_file():
        raise FileNotFoundError(
            "missing paired segmentation CSV; run "
            "evaluate_adversarial_segmentation_all_cases.py first: "
            f"{segmentation_path}"
        )
    quality = pd.read_csv(quality_path)
    segmentation = pd.read_csv(segmentation_path)
    _require_columns(
        quality,
        {
            *JOIN_KEYS,
            "epsilon_norm",
            "rms_norm",
            "rms_percent_case_sigma",
            "robust_signal_range_norm",
            "psnr_db_robust_range",
            "ssim_axial_prostate",
        },
        quality_path,
    )
    _require_columns(
        segmentation,
        {
            *JOIN_KEYS,
            "class",
            "rms_norm_recomputed",
            "linf_norm_recomputed",
            "clean_dice",
            "adversarial_dice",
            "dice_drop",
            "clean_iou",
            "adversarial_iou",
            "iou_drop",
            "clean_hd95_mm",
            "adversarial_hd95_mm",
            "hd95_increase_mm",
            "clean_asd_mm",
            "adversarial_asd_mm",
            "asd_increase_mm",
            "clean_volume_error_percent",
            "adversarial_volume_error_percent",
            "target_foreground_voxels",
            "clean_prediction_foreground_voxels",
            "adversarial_prediction_foreground_voxels",
        },
        segmentation_path,
    )
    quality = quality[
        quality["attack"].isin(ATTACKS) & quality["dataset"].isin(DATASET_LABELS)
    ].copy()
    segmentation = segmentation[
        segmentation["attack"].isin(ATTACKS)
        & segmentation["dataset"].isin(DATASET_LABELS)
    ].copy()
    segmentation = _strict_macro_boundary_rows(segmentation)
    if quality.duplicated(JOIN_KEYS).any():
        raise ValueError("image-quality CSV contains duplicate join keys")
    if segmentation.duplicated(JOIN_KEYS).any():
        raise ValueError("macro segmentation CSV contains duplicate join keys")

    quality_keys = set(map(tuple, quality[JOIN_KEYS].itertuples(index=False)))
    segmentation_keys = set(map(tuple, segmentation[JOIN_KEYS].itertuples(index=False)))
    if quality_keys != segmentation_keys:
        raise ValueError(
            "quality/segmentation key mismatch: "
            f"{len(quality_keys - segmentation_keys)} missing segmentation keys, "
            f"{len(segmentation_keys - quality_keys)} missing quality keys"
        )
    paired = quality.merge(
        segmentation,
        on=JOIN_KEYS,
        how="inner",
        validate="one_to_one",
        suffixes=("", "_segmentation"),
    )
    rms_absolute_difference = np.abs(
        paired["rms_norm_recomputed"].to_numpy(dtype=float)
        - paired["rms_norm"].to_numpy(dtype=float)
    )
    rms_relative_difference = rms_absolute_difference / np.maximum(
        paired["rms_norm"].to_numpy(dtype=float), 1e-12
    )
    linf_absolute_difference = np.abs(
        paired["linf_norm_recomputed"].to_numpy(dtype=float)
        - paired["linf_norm"].to_numpy(dtype=float)
    )
    audit = {
        "rows": float(len(paired)),
        "cases": float(paired[["dataset", "case_id"]].drop_duplicates().shape[0]),
        "max_rms_absolute_difference": float(np.max(rms_absolute_difference)),
        "max_rms_relative_difference": float(np.max(rms_relative_difference)),
        "q99_rms_relative_difference": float(
            np.quantile(rms_relative_difference, 0.99)
        ),
        "max_linf_absolute_difference": float(np.max(linf_absolute_difference)),
        "boundary_metrics_available": float(
            np.isfinite(paired["hd95_increase_mm"]).any()
            and np.isfinite(paired["asd_increase_mm"]).any()
        ),
    }
    if audit["max_rms_relative_difference"] > norm_relative_tolerance:
        raise ValueError(
            "regenerated segmentation attacks do not match the image-quality "
            "attacks closely enough: maximum relative RMS difference "
            f"{audit['max_rms_relative_difference']:.3%} exceeds "
            f"{norm_relative_tolerance:.3%}. Use identical full-cohort batching, "
            "precision, seeds, steps, and checkpoints."
        )
    paired["rms_reproduction_relative_difference"] = rms_relative_difference
    return _add_damage_characterization_columns(paired), audit


def _finite_interpolate(x: np.ndarray, y: np.ndarray, target: float) -> float:
    mask = np.isfinite(x) & np.isfinite(y)
    x_valid = x[mask]
    y_valid = y[mask]
    if not len(x_valid):
        return float("nan")
    order = np.argsort(x_valid)
    x_valid = x_valid[order]
    y_valid = y_valid[order]
    unique_x, inverse = np.unique(x_valid, return_inverse=True)
    if len(unique_x) != len(x_valid):
        y_valid = np.asarray(
            [np.mean(y_valid[inverse == index]) for index in range(len(unique_x))]
        )
        x_valid = unique_x
    if target < x_valid[0] - 1e-12 or target > x_valid[-1] + 1e-12:
        return float("nan")
    return float(np.interp(target, x_valid, y_valid))


def _case_scale(group: pd.DataFrame) -> tuple[float, float]:
    rms = group["rms_norm"].to_numpy(dtype=float)
    percent = group["rms_percent_case_sigma"].to_numpy(dtype=float)
    valid = np.isfinite(rms) & np.isfinite(percent) & (percent > 0)
    clean_std = float(np.median(rms[valid] / (percent[valid] / 100.0)))
    robust_range = float(
        np.median(group["robust_signal_range_norm"].to_numpy(dtype=float))
    )
    return clean_std, robust_range


def _curve_with_clean(group: pd.DataFrame) -> pd.DataFrame:
    ordered = group.sort_values("epsilon_n").copy()
    first = ordered.iloc[0]
    clean = {
        column: float("nan")
        for column in ordered.columns
        if column not in {"dataset", "case_id", "attack"}
    }
    clean.update(
        {
            "dataset": first["dataset"],
            "case_id": first["case_id"],
            "attack": first["attack"],
            "epsilon_n": 0.0,
            "epsilon_norm": 0.0,
            "rms_norm": 0.0,
            "rms_percent_case_sigma": 0.0,
            "psnr_db_robust_range": float("inf"),
            "ssim_axial_prostate": 1.0,
            "adversarial_dice": float(first["clean_dice"]),
            "dice_drop": 0.0,
            "adversarial_iou": float(first["clean_iou"]),
            "iou_drop": 0.0,
            "adversarial_hd95_mm": float(first["clean_hd95_mm"]),
            "hd95_increase_mm": 0.0,
            "adversarial_asd_mm": float(first["clean_asd_mm"]),
            "asd_increase_mm": 0.0,
            "adversarial_volume_error_percent": float(
                first["clean_volume_error_percent"]
            ),
            "adversarial_absolute_volume_error_percent": abs(
                float(first["clean_volume_error_percent"])
            ),
            "absolute_volume_error_increase_percent": 0.0,
            "adversarial_prediction_foreground_voxels": float(
                first["clean_prediction_foreground_voxels"]
            ),
            "adversarial_empty_prediction": bool(
                first["clean_prediction_foreground_voxels"] <= 0
            ),
            "boundary_metrics_defined": bool(
                np.isfinite(first["clean_hd95_mm"])
                and np.isfinite(first["clean_asd_mm"])
            ),
        }
    )
    return pd.concat([pd.DataFrame([clean]), ordered], ignore_index=True)


def interpolate_rms_matched(
    paired: pd.DataFrame,
    targets: list[float],
    *,
    strata: tuple[str, ...] = (),
) -> tuple[pd.DataFrame, pd.DataFrame]:
    metrics = (
        "epsilon_n",
        "epsilon_norm",
        "ssim_axial_prostate",
        "adversarial_dice",
        "dice_drop",
        "adversarial_iou",
        "iou_drop",
        "adversarial_hd95_mm",
        "hd95_increase_mm",
        "adversarial_asd_mm",
        "asd_increase_mm",
        "adversarial_volume_error_percent",
        "adversarial_absolute_volume_error_percent",
        "absolute_volume_error_increase_percent",
        "adversarial_prediction_foreground_voxels",
    )
    identity_columns = ["dataset", *strata, "case_id", "attack"]
    pair_columns = [
        "dataset",
        *strata,
        "case_id",
        "target_rms_percent_case_sigma",
    ]
    rows: list[dict[str, object]] = []
    for key, group in paired.groupby(identity_columns, sort=False):
        if not isinstance(key, tuple):
            key = (key,)
        identity = dict(zip(identity_columns, key))
        curve = _curve_with_clean(group)
        x = curve["rms_percent_case_sigma"].to_numpy(dtype=float)
        clean_std, robust_range = _case_scale(group)
        max_rms = float(np.nanmax(x))
        for target in targets:
            if target > max_rms + 1e-12:
                continue
            row: dict[str, object] = {
                **identity,
                "target_rms_percent_case_sigma": target,
                "max_observed_rms_percent_case_sigma": max_rms,
                "interpolation_method": "linear_between_measured_epsilon_levels",
            }
            for metric in metrics:
                row[metric] = _finite_interpolate(
                    x,
                    curve[metric].to_numpy(dtype=float),
                    target,
                )
            target_rms_norm = clean_std * target / 100.0
            row["rms_norm"] = target_rms_norm
            row["robust_signal_range_norm"] = robust_range
            row["psnr_db_robust_range"] = (
                float("inf")
                if target == 0
                else 20.0 * math.log10(robust_range / target_rms_norm)
            )
            rows.append(row)
    available = pd.DataFrame(rows)
    attack_counts = (
        available.groupby(pair_columns)["attack"]
        .nunique()
        .rename("n_attacks_supported")
        .reset_index()
    )
    available = available.merge(
        attack_counts,
        on=pair_columns,
        validate="many_to_one",
    )
    available["paired_complete"] = available["n_attacks_supported"] == len(ATTACKS)
    paired_complete = available[available["paired_complete"]].copy()
    paired_complete = paired_complete.sort_values(
        ["dataset", *strata, "target_rms_percent_case_sigma", "case_id", "attack"]
    ).reset_index(drop=True)
    return available, paired_complete


def _summarize(
    frame: pd.DataFrame,
    group_columns: list[str],
    metrics: tuple[str, ...],
) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    for key, group in frame.groupby(group_columns, sort=True):
        if not isinstance(key, tuple):
            key = (key,)
        row = dict(zip(group_columns, key))
        row["n_cases"] = int(group["case_id"].nunique())
        for metric in metrics:
            values = group[metric].to_numpy(dtype=float)
            values = values[np.isfinite(values)]
            row[f"{metric}_mean"] = (
                float(np.mean(values)) if len(values) else float("nan")
            )
            row[f"{metric}_median"] = (
                float(np.median(values)) if len(values) else float("nan")
            )
            row[f"{metric}_q05"] = (
                float(np.quantile(values, 0.05)) if len(values) else float("nan")
            )
            row[f"{metric}_q95"] = (
                float(np.quantile(values, 0.95)) if len(values) else float("nan")
            )
        rows.append(row)
    return pd.DataFrame(rows)


def _bootstrap_median_ci(
    values: np.ndarray,
    *,
    seed: int,
    n_bootstrap: int = 2000,
) -> tuple[float, float, float]:
    values = values[np.isfinite(values)]
    if not len(values):
        return float("nan"), float("nan"), float("nan")
    rng = np.random.default_rng(seed)
    estimates = np.empty(n_bootstrap, dtype=float)
    for index in range(n_bootstrap):
        sample = rng.choice(values, size=len(values), replace=True)
        estimates[index] = np.median(sample)
    return (
        float(np.median(values)),
        float(np.quantile(estimates, 0.025)),
        float(np.quantile(estimates, 0.975)),
    )


FRONT_DAMAGE_METRICS = (
    "dice_drop",
    "iou_drop",
    "absolute_volume_error_increase_percent",
    "hd95_increase_mm",
    "asd_increase_mm",
)


def _nanmedian(values: np.ndarray, *, axis: int) -> np.ndarray:
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", category=RuntimeWarning)
        return np.nanmedian(values, axis=axis)


def _nanquantile(values: np.ndarray, quantile: float, *, axis: int) -> np.ndarray:
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", category=RuntimeWarning)
        return np.nanquantile(values, quantile, axis=axis)


def _pareto_efficient_mask(x: np.ndarray, damage: np.ndarray) -> np.ndarray:
    """Return non-dominated points for lower RMS and greater damage."""

    efficient = np.zeros(len(x), dtype=bool)
    finite = np.isfinite(x) & np.isfinite(damage)
    for index in np.flatnonzero(finite):
        dominated = np.any(
            finite
            & (x <= x[index] + 1e-12)
            & (damage >= damage[index] - 1e-12)
            & ((x < x[index] - 1e-12) | (damage > damage[index] + 1e-12))
        )
        efficient[index] = not dominated
    return efficient


def dense_paired_front_with_uncertainty(
    matched: pd.DataFrame,
    *,
    seed: int,
    n_bootstrap: int,
    strata: tuple[str, ...] = (),
    metrics: tuple[str, ...] = FRONT_DAMAGE_METRICS,
    bootstrap_chunk_size: int = 32,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Summarize a fixed patient cohort over a dense, paired RMS grid.

    Every bootstrap draw resamples patient IDs and preserves all attacks and
    RMS targets for each selected patient. This provides paired confidence
    bands, paired attack-minus-APGD differences, and uncertainty in Pareto-front
    membership without treating interpolated rows as independent observations.
    """

    if n_bootstrap < 1:
        raise ValueError("n_bootstrap must be positive")
    if bootstrap_chunk_size < 1:
        raise ValueError("bootstrap_chunk_size must be positive")
    missing = sorted(
        {*metrics, "adversarial_prediction_foreground_voxels"} - set(matched.columns)
    )
    if missing:
        raise ValueError(f"dense paired front is missing metrics: {missing}")

    group_columns = ["dataset", *strata]
    summary_rows: list[dict[str, object]] = []
    difference_rows: list[dict[str, object]] = []
    common_frames: list[pd.DataFrame] = []
    for group_index, (raw_key, raw_group) in enumerate(
        matched.groupby(group_columns, sort=True)
    ):
        key = raw_key if isinstance(raw_key, tuple) else (raw_key,)
        identity = dict(zip(group_columns, key))
        targets = np.asarray(
            sorted(raw_group["target_rms_percent_case_sigma"].unique()),
            dtype=float,
        )
        expected_rows = len(ATTACKS) * len(targets)
        unique_columns = [
            "case_id",
            "attack",
            "target_rms_percent_case_sigma",
        ]
        if raw_group.duplicated(unique_columns).any():
            raise ValueError(
                f"dense paired front has duplicate rows for stratum {identity}"
            )
        case_counts = raw_group.groupby("case_id").size()
        case_ids = sorted(case_counts[case_counts == expected_rows].index.astype(str))
        if not case_ids:
            raise ValueError(f"no complete dense-front patients for stratum {identity}")
        group = raw_group[raw_group["case_id"].astype(str).isin(case_ids)].copy()
        group["case_id"] = group["case_id"].astype(str)
        common_frames.append(group)

        ordered_index = pd.MultiIndex.from_product(
            [case_ids, ATTACKS, targets],
            names=["case_id", "attack", "target_rms_percent_case_sigma"],
        )
        indexed = group.set_index(unique_columns).reindex(ordered_index)
        n_cases = len(case_ids)
        shape = (n_cases, len(ATTACKS), len(targets))
        arrays = {
            metric: indexed[metric].to_numpy(dtype=float).reshape(shape)
            for metric in metrics
        }
        foreground = (
            indexed["adversarial_prediction_foreground_voxels"]
            .to_numpy(dtype=float)
            .reshape(shape)
        )

        bootstrap_medians = {
            metric: np.empty((n_bootstrap, len(ATTACKS), len(targets)), dtype=float)
            for metric in metrics
        }
        bootstrap_finite_fractions = {
            metric: np.empty((n_bootstrap, len(ATTACKS), len(targets)), dtype=float)
            for metric in metrics
        }
        comparison_attacks = ("FGSM-BCE", "PGD-BCE")
        bootstrap_differences = {
            metric: np.empty(
                (n_bootstrap, len(comparison_attacks), len(targets)), dtype=float
            )
            for metric in metrics
        }
        bootstrap_empty_fractions = np.empty(
            (n_bootstrap, len(ATTACKS), len(targets)), dtype=float
        )
        rng = np.random.default_rng(seed + group_index * 100_003)
        reference_index = ATTACKS.index("APGD-BCE")
        comparison_indices = [ATTACKS.index(attack) for attack in comparison_attacks]
        for start in range(0, n_bootstrap, bootstrap_chunk_size):
            stop = min(start + bootstrap_chunk_size, n_bootstrap)
            sample_indices = rng.integers(
                0,
                n_cases,
                size=(stop - start, n_cases),
            )
            for metric, values in arrays.items():
                sampled = values[sample_indices]
                bootstrap_medians[metric][start:stop] = _nanmedian(
                    sampled,
                    axis=1,
                )
                bootstrap_finite_fractions[metric][start:stop] = np.mean(
                    np.isfinite(sampled),
                    axis=1,
                )
                for comparison_index, attack_index in enumerate(comparison_indices):
                    bootstrap_differences[metric][start:stop, comparison_index] = (
                        _nanmedian(
                            sampled[:, :, attack_index, :]
                            - sampled[:, :, reference_index, :],
                            axis=1,
                        )
                    )
            bootstrap_empty_fractions[start:stop] = np.mean(
                foreground[sample_indices] <= 0.5,
                axis=1,
            )

        point_medians = {
            metric: _nanmedian(values, axis=0) for metric, values in arrays.items()
        }
        dice_bootstrap = bootstrap_medians["dice_drop"]
        x_grid = np.broadcast_to(targets, (len(ATTACKS), len(targets)))
        median_pareto = _pareto_efficient_mask(
            x_grid.ravel(),
            point_medians["dice_drop"].ravel(),
        ).reshape(x_grid.shape)
        bootstrap_pareto = np.empty_like(dice_bootstrap, dtype=bool)
        for bootstrap_index in range(n_bootstrap):
            bootstrap_pareto[bootstrap_index] = _pareto_efficient_mask(
                x_grid.ravel(),
                dice_bootstrap[bootstrap_index].ravel(),
            ).reshape(x_grid.shape)
        pareto_probability = bootstrap_pareto.mean(axis=0)
        upper_envelope = np.nanmax(dice_bootstrap, axis=1, keepdims=True)
        upper_envelope_probability = np.mean(
            np.isclose(dice_bootstrap, upper_envelope, rtol=0.0, atol=1e-12),
            axis=0,
        )
        empty_fraction = np.mean(foreground <= 0.5, axis=0)
        empty_ci_low = np.quantile(bootstrap_empty_fractions, 0.025, axis=0)
        empty_ci_high = np.quantile(bootstrap_empty_fractions, 0.975, axis=0)

        for attack_index, attack in enumerate(ATTACKS):
            for target_index, target in enumerate(targets):
                row: dict[str, object] = {
                    **identity,
                    "attack": attack,
                    "target_rms_percent_case_sigma": float(target),
                    "n_cases": n_cases,
                    "pareto_efficient_at_sample_median": bool(
                        median_pareto[attack_index, target_index]
                    ),
                    "pareto_selection_probability": float(
                        pareto_probability[attack_index, target_index]
                    ),
                    "same_rms_upper_envelope_probability": float(
                        upper_envelope_probability[attack_index, target_index]
                    ),
                    "empty_prediction_fraction": float(
                        empty_fraction[attack_index, target_index]
                    ),
                    "empty_prediction_fraction_ci95_low": float(
                        empty_ci_low[attack_index, target_index]
                    ),
                    "empty_prediction_fraction_ci95_high": float(
                        empty_ci_high[attack_index, target_index]
                    ),
                }
                for metric, point_values in point_medians.items():
                    bootstrap_values = bootstrap_medians[metric][
                        :, attack_index, target_index
                    ]
                    bootstrap_finite = bootstrap_finite_fractions[metric][
                        :, attack_index, target_index
                    ]
                    finite_values = np.isfinite(
                        arrays[metric][:, attack_index, target_index]
                    )
                    row[f"{metric}_median"] = float(
                        point_values[attack_index, target_index]
                    )
                    row[f"{metric}_ci95_low"] = float(
                        _nanquantile(bootstrap_values, 0.025, axis=0)
                    )
                    row[f"{metric}_ci95_high"] = float(
                        _nanquantile(bootstrap_values, 0.975, axis=0)
                    )
                    row[f"{metric}_n_finite"] = int(finite_values.sum())
                    row[f"{metric}_finite_fraction"] = float(finite_values.mean())
                    row[f"{metric}_finite_fraction_ci95_low"] = float(
                        np.quantile(bootstrap_finite, 0.025)
                    )
                    row[f"{metric}_finite_fraction_ci95_high"] = float(
                        np.quantile(bootstrap_finite, 0.975)
                    )
                summary_rows.append(row)

        for comparison_index, attack in enumerate(comparison_attacks):
            attack_index = ATTACKS.index(attack)
            for target_index, target in enumerate(targets):
                row = {
                    **identity,
                    "attack": attack,
                    "reference_attack": "APGD-BCE",
                    "target_rms_percent_case_sigma": float(target),
                    "n_cases": n_cases,
                }
                for metric, values in arrays.items():
                    differences = (
                        values[:, attack_index, target_index]
                        - values[:, reference_index, target_index]
                    )
                    bootstrap_values = bootstrap_differences[metric][
                        :, comparison_index, target_index
                    ]
                    finite = differences[np.isfinite(differences)]
                    row[f"{metric}_difference_median"] = (
                        float(np.median(finite)) if len(finite) else float("nan")
                    )
                    row[f"{metric}_difference_ci95_low"] = float(
                        _nanquantile(bootstrap_values, 0.025, axis=0)
                    )
                    row[f"{metric}_difference_ci95_high"] = float(
                        _nanquantile(bootstrap_values, 0.975, axis=0)
                    )
                    row[f"{metric}_difference_n_finite"] = int(len(finite))
                difference_rows.append(row)

    common = pd.concat(common_frames, ignore_index=True).sort_values(
        ["dataset", *strata, "target_rms_percent_case_sigma", "case_id", "attack"]
    )
    return (
        common.reset_index(drop=True),
        pd.DataFrame(summary_rows),
        pd.DataFrame(difference_rows),
    )


def paired_rms_differences(matched: pd.DataFrame, *, seed: int) -> pd.DataFrame:
    metrics = (
        "epsilon_n",
        "ssim_axial_prostate",
        "dice_drop",
        "iou_drop",
        "absolute_volume_error_increase_percent",
        "hd95_increase_mm",
        "asd_increase_mm",
    )
    rows: list[dict[str, object]] = []
    for (dataset, target), group in matched.groupby(
        ["dataset", "target_rms_percent_case_sigma"]
    ):
        for attack in ("FGSM-BCE", "PGD-BCE"):
            attack_frame = group[group["attack"] == attack].set_index("case_id")
            reference = group[group["attack"] == "APGD-BCE"].set_index("case_id")
            common = attack_frame.index.intersection(reference.index)
            row: dict[str, object] = {
                "dataset": dataset,
                "target_rms_percent_case_sigma": target,
                "attack": attack,
                "reference_attack": "APGD-BCE",
                "n_cases": int(len(common)),
            }
            for metric_index, metric in enumerate(metrics):
                differences = attack_frame.loc[common, metric].to_numpy(
                    dtype=float
                ) - reference.loc[common, metric].to_numpy(dtype=float)
                median, lower, upper = _bootstrap_median_ci(
                    differences,
                    seed=seed
                    + metric_index
                    + int(round(float(target) * 100))
                    + (0 if attack == "FGSM-BCE" else 10_000)
                    + (0 if dataset == "wg" else 20_000),
                )
                row[f"{metric}_difference_median"] = median
                row[f"{metric}_difference_ci95_low"] = lower
                row[f"{metric}_difference_ci95_high"] = upper
            rows.append(row)
    return pd.DataFrame(rows)


def failure_thresholds(paired: pd.DataFrame, thresholds: list[float]) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    for (dataset, case_id, attack), group in paired.groupby(
        ["dataset", "case_id", "attack"], sort=False
    ):
        curve = _curve_with_clean(group).sort_values("epsilon_n").reset_index(drop=True)
        clean_std, robust_range = _case_scale(group)
        damage = curve["dice_drop"].to_numpy(dtype=float)
        for threshold in thresholds:
            crossing_index = next(
                (
                    index
                    for index in range(1, len(curve))
                    if np.isfinite(damage[index]) and damage[index] >= threshold
                ),
                None,
            )
            row: dict[str, object] = {
                "dataset": dataset,
                "case_id": case_id,
                "attack": attack,
                "dice_drop_threshold": threshold,
                "attained": crossing_index is not None,
                "max_observed_dice_drop": float(np.nanmax(damage)),
                "censored_at_epsilon_n": float(curve["epsilon_n"].iloc[-1]),
            }
            if crossing_index is None:
                for metric in (
                    "estimated_epsilon_n",
                    "rms_percent_case_sigma",
                    "rms_norm",
                    "psnr_db_robust_range",
                    "ssim_axial_prostate",
                    "iou_drop",
                    "absolute_volume_error_increase_percent",
                    "hd95_increase_mm",
                    "asd_increase_mm",
                ):
                    row[metric] = float("nan")
                rows.append(row)
                continue

            previous = curve.iloc[crossing_index - 1]
            current = curve.iloc[crossing_index]
            previous_damage = float(previous["dice_drop"])
            current_damage = float(current["dice_drop"])
            fraction = (
                0.0
                if current_damage <= previous_damage
                else (threshold - previous_damage) / (current_damage - previous_damage)
            )
            fraction = float(np.clip(fraction, 0.0, 1.0))

            def between(column: str) -> float:
                low = float(previous[column])
                high = float(current[column])
                if not np.isfinite(low) or not np.isfinite(high):
                    return float("nan")
                return low + fraction * (high - low)

            rms_percent = between("rms_percent_case_sigma")
            rms_norm = clean_std * rms_percent / 100.0
            row.update(
                {
                    "estimated_epsilon_n": between("epsilon_n"),
                    "rms_percent_case_sigma": rms_percent,
                    "rms_norm": rms_norm,
                    "psnr_db_robust_range": 20.0 * math.log10(robust_range / rms_norm),
                    "ssim_axial_prostate": between("ssim_axial_prostate"),
                    "iou_drop": between("iou_drop"),
                    "absolute_volume_error_increase_percent": between(
                        "absolute_volume_error_increase_percent"
                    ),
                    "hd95_increase_mm": between("hd95_increase_mm"),
                    "asd_increase_mm": between("asd_increase_mm"),
                }
            )
            rows.append(row)
    result = pd.DataFrame(rows)
    complete_counts = (
        result.groupby(["dataset", "case_id", "dice_drop_threshold"])["attained"]
        .sum()
        .rename("n_attacks_attaining_threshold")
        .reset_index()
    )
    result = result.merge(
        complete_counts,
        on=["dataset", "case_id", "dice_drop_threshold"],
        validate="many_to_one",
    )
    result["paired_attained_all_attacks"] = result[
        "n_attacks_attaining_threshold"
    ] == len(ATTACKS)
    return result.sort_values(
        ["dataset", "dice_drop_threshold", "case_id", "attack"]
    ).reset_index(drop=True)


def summarize_failure_thresholds(thresholds: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    metrics = (
        "estimated_epsilon_n",
        "rms_percent_case_sigma",
        "psnr_db_robust_range",
        "ssim_axial_prostate",
        "iou_drop",
        "absolute_volume_error_increase_percent",
        "hd95_increase_mm",
        "asd_increase_mm",
    )
    for (dataset, attack, threshold), group in thresholds.groupby(
        ["dataset", "attack", "dice_drop_threshold"]
    ):
        attained = group[group["attained"]]
        paired_attained = group[group["paired_attained_all_attacks"]]
        row: dict[str, object] = {
            "dataset": dataset,
            "attack": attack,
            "dice_drop_threshold": threshold,
            "n_cases": int(group["case_id"].nunique()),
            "n_attained": int(attained["case_id"].nunique()),
            "attained_fraction": float(group["attained"].mean()),
            "n_paired_attained_all_attacks": int(paired_attained["case_id"].nunique()),
        }
        for metric in metrics:
            values = attained[metric].to_numpy(dtype=float)
            values = values[np.isfinite(values)]
            row[f"{metric}_attained_median"] = (
                float(np.median(values)) if len(values) else float("nan")
            )
            paired_values = paired_attained[metric].to_numpy(dtype=float)
            paired_values = paired_values[np.isfinite(paired_values)]
            row[f"{metric}_paired_median"] = (
                float(np.median(paired_values)) if len(paired_values) else float("nan")
            )
        rows.append(row)
    return pd.DataFrame(rows)


def damage_auc(matched: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    auc_metrics = {
        "normalized_damage_auc": "dice_drop",
        "normalized_iou_damage_auc": "iou_drop",
        "normalized_absolute_volume_error_increase_auc": (
            "absolute_volume_error_increase_percent"
        ),
        "normalized_hd95_increase_auc": "hd95_increase_mm",
        "normalized_asd_increase_auc": "asd_increase_mm",
    }
    common_maximum = float(matched["target_rms_percent_case_sigma"].max())
    supported_maximum = (
        matched.groupby(["dataset", "case_id"])["target_rms_percent_case_sigma"]
        .max()
        .rename("case_maximum")
        .reset_index()
    )
    eligible = supported_maximum[
        np.isclose(
            supported_maximum["case_maximum"].to_numpy(dtype=float),
            common_maximum,
        )
    ][["dataset", "case_id"]]
    matched = matched.merge(
        eligible,
        on=["dataset", "case_id"],
        how="inner",
        validate="many_to_one",
    )
    rows: list[dict[str, object]] = []
    for (dataset, case_id, attack), group in matched.groupby(
        ["dataset", "case_id", "attack"]
    ):
        group = group.sort_values("target_rms_percent_case_sigma")
        x = group["target_rms_percent_case_sigma"].to_numpy(dtype=float)
        if len(x) < 2 or x[-1] <= x[0]:
            continue
        row: dict[str, object] = {
            "dataset": dataset,
            "case_id": case_id,
            "attack": attack,
            "max_target_rms_percent_case_sigma": common_maximum,
        }
        for output_column, metric in auc_metrics.items():
            y = group[metric].to_numpy(dtype=float)
            row[output_column] = (
                float(np.trapz(y, x) / (x[-1] - x[0]))
                if np.isfinite(y).all()
                else float("nan")
            )
        rows.append(row)
    per_case = pd.DataFrame(rows)
    summary = _summarize(
        per_case,
        ["dataset", "attack"],
        tuple(auc_metrics),
    )
    return per_case, summary


def measured_summary_with_clean(paired: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    for (dataset, attack, epsilon_n), group in paired.groupby(
        ["dataset", "attack", "epsilon_n"]
    ):
        row: dict[str, object] = {
            "dataset": dataset,
            "attack": attack,
            "epsilon_n": int(epsilon_n),
            "n_cases": int(group["case_id"].nunique()),
        }
        for metric in (
            "rms_percent_case_sigma",
            "psnr_db_robust_range",
            "ssim_axial_prostate",
            "dice_drop",
            "iou_drop",
            "absolute_volume_error_increase_percent",
            "hd95_increase_mm",
            "asd_increase_mm",
        ):
            values = group[metric].to_numpy(dtype=float)
            values = values[np.isfinite(values)]
            if values.size == 0:
                row[f"{metric}_median"] = float("nan")
                row[f"{metric}_q05"] = float("nan")
                row[f"{metric}_q95"] = float("nan")
                continue
            row[f"{metric}_median"] = float(np.median(values))
            row[f"{metric}_q05"] = float(np.quantile(values, 0.05))
            row[f"{metric}_q95"] = float(np.quantile(values, 0.95))
        rows.append(row)
    summary = pd.DataFrame(rows)
    clean_rows: list[dict[str, object]] = []
    for dataset in DATASET_LABELS:
        n_cases = int(paired.loc[paired["dataset"] == dataset, "case_id"].nunique())
        for attack in ATTACKS:
            clean_rows.append(
                {
                    "dataset": dataset,
                    "attack": attack,
                    "epsilon_n": 0,
                    "n_cases": n_cases,
                    "rms_percent_case_sigma_median": 0.0,
                    "rms_percent_case_sigma_q05": 0.0,
                    "rms_percent_case_sigma_q95": 0.0,
                    "psnr_db_robust_range_median": float("inf"),
                    "psnr_db_robust_range_q05": float("inf"),
                    "psnr_db_robust_range_q95": float("inf"),
                    "ssim_axial_prostate_median": 1.0,
                    "ssim_axial_prostate_q05": 1.0,
                    "ssim_axial_prostate_q95": 1.0,
                    "dice_drop_median": 0.0,
                    "dice_drop_q05": 0.0,
                    "dice_drop_q95": 0.0,
                    "iou_drop_median": 0.0,
                    "iou_drop_q05": 0.0,
                    "iou_drop_q95": 0.0,
                    "absolute_volume_error_increase_percent_median": 0.0,
                    "absolute_volume_error_increase_percent_q05": 0.0,
                    "absolute_volume_error_increase_percent_q95": 0.0,
                    "hd95_increase_mm_median": 0.0,
                    "hd95_increase_mm_q05": 0.0,
                    "hd95_increase_mm_q95": 0.0,
                    "asd_increase_mm_median": 0.0,
                    "asd_increase_mm_q05": 0.0,
                    "asd_increase_mm_q95": 0.0,
                }
            )
    return (
        pd.concat([pd.DataFrame(clean_rows), summary], ignore_index=True)
        .sort_values(["dataset", "attack", "epsilon_n"])
        .reset_index(drop=True)
    )


def mark_pareto_points(summary: pd.DataFrame) -> pd.DataFrame:
    marked = summary.copy()
    marked["pareto_efficient"] = False
    for dataset, group in marked.groupby("dataset"):
        indices = list(group.index)
        rms = group["rms_percent_case_sigma_median"].to_numpy(dtype=float)
        damage = group["dice_drop_median"].to_numpy(dtype=float)
        for local_index, global_index in enumerate(indices):
            dominated = np.any(
                (rms <= rms[local_index] + 1e-12)
                & (damage >= damage[local_index] - 1e-12)
                & (
                    (rms < rms[local_index] - 1e-12)
                    | (damage > damage[local_index] + 1e-12)
                )
            )
            marked.loc[global_index, "pareto_efficient"] = not dominated
    return marked


def plot_rms_matched(summary: pd.DataFrame, output_dir: Path) -> Path:
    fig, axes = plt.subplots(
        2,
        2,
        figsize=(15.5, 10.5),
        sharex="col",
        sharey="row",
    )
    configurations = (
        ("ssim_axial_prostate", "Median SSIM"),
        ("dice_drop", "Median macro Dice decrease"),
    )
    for column, dataset in enumerate(("wg", "zones")):
        for row, (metric, ylabel) in enumerate(configurations):
            ax = axes[row, column]
            for attack in ATTACKS:
                group = summary[
                    (summary["dataset"] == dataset) & (summary["attack"] == attack)
                ].sort_values("target_rms_percent_case_sigma")
                x = group["target_rms_percent_case_sigma"].to_numpy(dtype=float)
                median = group[f"{metric}_median"].to_numpy(dtype=float)
                q05 = group[f"{metric}_q05"].to_numpy(dtype=float)
                q95 = group[f"{metric}_q95"].to_numpy(dtype=float)
                ax.fill_between(
                    x,
                    q05,
                    q95,
                    color=ATTACK_COLORS[attack],
                    alpha=0.07,
                    linewidth=0,
                    zorder=1,
                )
                _plot_attack_series(ax, x, median, attack)
            if column == 0:
                ax.set_ylabel(ylabel)
            _style_axis(ax)
            if row == 0:
                ax.set_title(DATASET_LABELS[dataset])
            if row == 1:
                ax.set_xlabel("Perturbation RMS (% of clean-image SD)")
    rms_ticks = sorted(summary["target_rms_percent_case_sigma"].unique())
    for ax in axes[1, :]:
        ax.set_xticks(rms_ticks)
        ax.set_xticklabels([f"{value:g}" for value in rms_ticks])
    axes[0, 0].set_ylim(0.89, 1.004)
    axes[1, 0].set_ylim(-0.015, 1.01)
    _add_panel_labels(axes)
    handles, labels = axes[0, 0].get_legend_handles_labels()
    fig.legend(
        handles,
        labels,
        loc="upper center",
        ncol=3,
        frameon=False,
        bbox_to_anchor=(0.5, 0.935),
        handlelength=3.0,
        columnspacing=2.2,
    )
    fig.suptitle(
        "RMS-matched attacks: similar image distortion, different model damage",
        y=0.985,
    )
    fig.text(
        0.5,
        0.018,
        "Lines show paired patient medians; light bands show the 5th–95th "
        "percentiles. Clean (0/255): RMS = 0, SSIM = 1, Dice decrease = 0.",
        ha="center",
        color="#444444",
        fontsize=10.5,
    )
    fig.tight_layout(rect=(0.025, 0.055, 1, 0.875), h_pad=2.2, w_pad=2.0)
    path = output_dir / "01_rms_matched_attack_comparison.png"
    _save_figure(fig, path)
    plt.close(fig)
    return path


def plot_quality_damage(summary: pd.DataFrame, output_dir: Path) -> Path:
    fig, axes = plt.subplots(
        3,
        2,
        figsize=(16, 14.5),
        sharey="row",
    )
    for column, dataset in enumerate(("wg", "zones")):
        for attack in ATTACKS:
            group = summary[
                (summary["dataset"] == dataset) & (summary["attack"] == attack)
            ].sort_values("epsilon_n")
            damage = group["dice_drop_median"].to_numpy(dtype=float)
            psnr = group["psnr_db_robust_range_median"].to_numpy(dtype=float)
            finite_psnr = np.isfinite(psnr)
            _plot_attack_series(
                axes[0, column],
                group["rms_percent_case_sigma_median"],
                damage,
                attack,
            )
            _plot_attack_series(
                axes[1, column],
                psnr[finite_psnr],
                damage[finite_psnr],
                attack,
            )
            _plot_attack_series(
                axes[2, column],
                group["ssim_axial_prostate_median"],
                damage,
                attack,
            )
        axes[0, column].set_title(DATASET_LABELS[dataset])
        axes[0, column].set_xlabel("Perturbation RMS (% of clean-image SD)")
        axes[1, column].set_xlabel("PSNR (dB; higher is cleaner)")
        axes[2, column].set_xlabel("SSIM (higher is cleaner)")
        axes[1, column].text(
            0.96,
            0.08,
            "Clean (0/255): PSNR → ∞",
            transform=axes[1, column].transAxes,
            ha="right",
            va="bottom",
            fontsize=10.5,
            color="#444444",
        )
        for row in range(3):
            if column == 0:
                axes[row, column].set_ylabel("Median macro Dice decrease")
            axes[row, column].set_ylim(-0.02, 1.02)
            axes[row, column].set_yticks(np.arange(0.0, 1.01, 0.2))
            _style_axis(axes[row, column])
        axes[0, column].set_xlim(-0.3, 13.0)
        axes[1, column].set_xlim(28.5, 59.5)
        axes[2, column].set_xlim(0.82, 1.005)
    _add_panel_labels(axes)
    handles, labels = axes[0, 0].get_legend_handles_labels()
    fig.legend(
        handles,
        labels,
        loc="upper center",
        ncol=3,
        frameon=False,
        bbox_to_anchor=(0.5, 0.945),
        handlelength=3.0,
        columnspacing=2.2,
    )
    fig.suptitle(
        "Segmentation damage can diverge despite similar image-quality scores",
        y=0.988,
    )
    fig.text(
        0.5,
        0.018,
        "Markers represent ε = 0, 2, 4, 8, 16, and 32/255. Clean: RMS = 0, "
        "PSNR = ∞, SSIM = 1, Dice decrease = 0.",
        ha="center",
        color="#444444",
        fontsize=10.5,
    )
    fig.tight_layout(rect=(0.025, 0.05, 1, 0.895), h_pad=2.0, w_pad=2.0)
    path = output_dir / "02_image_quality_vs_segmentation_damage.png"
    _save_figure(fig, path)
    plt.close(fig)
    return path


def plot_failure_thresholds(summary: pd.DataFrame, output_dir: Path) -> Path:
    fig, axes = plt.subplots(
        2,
        2,
        figsize=(15.5, 10.5),
        sharex="col",
        sharey="row",
    )
    for column, dataset in enumerate(("wg", "zones")):
        for attack in ATTACKS:
            group = summary[
                (summary["dataset"] == dataset) & (summary["attack"] == attack)
            ].sort_values("dice_drop_threshold")
            x = 100.0 * group["dice_drop_threshold"].to_numpy(dtype=float)
            _plot_attack_series(
                axes[0, column],
                x,
                group["rms_percent_case_sigma_paired_median"],
                attack,
            )
            _plot_attack_series(
                axes[1, column],
                x,
                100.0 * group["attained_fraction"].to_numpy(dtype=float),
                attack,
            )
        axes[0, column].set_title(DATASET_LABELS[dataset])
        if column == 0:
            axes[0, column].set_ylabel(
                "Median RMS required\n(% of clean-image SD; log scale)"
            )
            axes[1, column].set_ylabel("Cases reaching threshold by 32/255 (%)")
        axes[1, column].set_xlabel("Macro Dice decrease threshold (%)")
        axes[0, column].set_yscale("log")
        axes[0, column].set_ylim(0.75, 13.0)
        axes[0, column].set_yticks([0.8, 1, 2, 4, 8, 12])
        axes[0, column].set_yticklabels(["0.8", "1", "2", "4", "8", "12"])
        axes[1, column].set_ylim(-4, 104)
        axes[1, column].set_yticks([0, 25, 50, 75, 100])
        axes[1, column].axhline(
            100,
            color="#777777",
            linewidth=1.0,
            linestyle=":",
            zorder=1,
        )
        for row in range(2):
            axes[row, column].set_xticks([5, 10, 20])
            _style_axis(axes[row, column])
    _add_panel_labels(axes)
    handles, labels = axes[0, 0].get_legend_handles_labels()
    fig.legend(
        handles,
        labels,
        loc="upper center",
        ncol=3,
        frameon=False,
        bbox_to_anchor=(0.5, 0.935),
        handlelength=3.0,
        columnspacing=2.2,
    )
    fig.suptitle(
        "Perturbation required to cause 5%, 10%, or 20% Dice loss",
        y=0.985,
    )
    fig.text(
        0.5,
        0.018,
        "Top: paired medians include only cases where all three attacks reached "
        "the threshold. Bottom: full-cohort attainment rate.",
        ha="center",
        color="#444444",
        fontsize=10.5,
    )
    fig.tight_layout(rect=(0.025, 0.055, 1, 0.875), h_pad=2.2, w_pad=2.0)
    path = output_dir / "03_segmentation_failure_thresholds.png"
    _save_figure(fig, path)
    plt.close(fig)
    return path


def plot_pareto(summary: pd.DataFrame, output_dir: Path) -> Path:
    fig, axes = plt.subplots(1, 2, figsize=(15.5, 6.8), sharex=True, sharey=True)
    for ax, dataset in zip(axes, ("wg", "zones")):
        for attack in ATTACKS:
            group = summary[
                (summary["dataset"] == dataset) & (summary["attack"] == attack)
            ].sort_values("epsilon_n")
            _plot_attack_series(
                ax,
                group["rms_percent_case_sigma_median"],
                group["dice_drop_median"],
                attack,
            )
            pareto = group[group["pareto_efficient"]]
            ax.scatter(
                pareto["rms_percent_case_sigma_median"],
                pareto["dice_drop_median"],
                s=150,
                facecolors="none",
                edgecolors="#222222",
                linewidths=1.5,
                zorder=4,
            )
        ax.set_title(DATASET_LABELS[dataset])
        ax.set_xlabel("Perturbation RMS (% of clean-image SD)")
        ax.set_xlim(-0.35, 13.0)
        ax.set_ylim(-0.03, 1.02)
        ax.set_xticks(np.arange(0, 13, 2))
        ax.set_yticks(np.arange(0.0, 1.01, 0.2))
        ax.annotate(
            "Clean\n0/255",
            xy=(0, 0),
            xytext=(1.0, 0.13),
            arrowprops={"arrowstyle": "->", "color": "#555555", "lw": 1.0},
            color="#444444",
            fontsize=10,
            ha="center",
        )
        _style_axis(ax)
    axes[0].set_ylabel("Median macro Dice decrease")
    _add_panel_labels(axes)
    handles, labels = axes[0].get_legend_handles_labels()
    handles.append(
        Line2D(
            [0],
            [0],
            marker="o",
            color="none",
            markerfacecolor="none",
            markeredgecolor="#222222",
            markeredgewidth=1.5,
            markersize=10,
            label="Pareto-efficient",
        )
    )
    labels.append("Pareto-efficient")
    fig.legend(
        handles,
        labels,
        loc="upper center",
        ncol=4,
        frameon=False,
        bbox_to_anchor=(0.5, 0.89),
        handlelength=3.0,
        columnspacing=1.8,
    )
    fig.suptitle(
        "Attack efficiency: segmentation damage achieved per unit RMS",
        y=0.98,
    )
    fig.text(
        0.5,
        0.025,
        "A ring marks a measured point for which no alternative has both lower "
        "RMS and equal-or-greater Dice damage.",
        ha="center",
        color="#444444",
        fontsize=10.5,
    )
    fig.tight_layout(rect=(0.025, 0.07, 1, 0.82), w_pad=2.0)
    path = output_dir / "04_attack_damage_pareto_front.png"
    _save_figure(fig, path)
    plt.close(fig)
    return path


def _plot_dense_metric(
    ax: plt.Axes,
    group: pd.DataFrame,
    metric: str,
    attack: str,
) -> None:
    ordered = group.sort_values("target_rms_percent_case_sigma")
    x = ordered["target_rms_percent_case_sigma"].to_numpy(dtype=float)
    median = ordered[f"{metric}_median"].to_numpy(dtype=float)
    low = ordered[f"{metric}_ci95_low"].to_numpy(dtype=float)
    high = ordered[f"{metric}_ci95_high"].to_numpy(dtype=float)
    finite_band = np.isfinite(x) & np.isfinite(low) & np.isfinite(high)
    ax.fill_between(
        x,
        low,
        high,
        where=finite_band,
        color=ATTACK_COLORS[attack],
        alpha=0.13,
        linewidth=0,
        interpolate=True,
        zorder=1,
    )
    markevery = max(1, len(x) // 7)
    ax.plot(
        x,
        median,
        color=ATTACK_COLORS[attack],
        linestyle=ATTACK_LINESTYLES[attack],
        marker=ATTACK_MARKERS[attack],
        markevery=markevery,
        markersize=6.5,
        markeredgecolor="white",
        markeredgewidth=0.7,
        linewidth=2.5,
        label=ATTACK_LABELS[attack],
        zorder=3,
    )


def plot_dense_paired_front(summary: pd.DataFrame, output_dir: Path) -> Path:
    fig, axes = plt.subplots(
        2,
        2,
        figsize=(15.5, 10.5),
        sharex="col",
        sharey="row",
    )
    for column, dataset in enumerate(("wg", "zones")):
        for attack in ATTACKS:
            group = summary[
                (summary["dataset"] == dataset) & (summary["attack"] == attack)
            ]
            _plot_dense_metric(axes[0, column], group, "dice_drop", attack)
            ordered = group.sort_values("target_rms_percent_case_sigma")
            axes[1, column].plot(
                ordered["target_rms_percent_case_sigma"],
                100.0 * ordered["pareto_selection_probability"],
                color=ATTACK_COLORS[attack],
                linestyle=ATTACK_LINESTYLES[attack],
                linewidth=2.5,
                label=ATTACK_LABELS[attack],
            )
        axes[0, column].set_title(DATASET_LABELS[dataset])
        axes[1, column].set_xlabel("Perturbation RMS (% of clean-image SD)")
        axes[0, column].set_ylim(-0.03, 1.02)
        axes[0, column].set_yticks(np.arange(0.0, 1.01, 0.2))
        axes[1, column].set_ylim(-4, 104)
        axes[1, column].set_yticks([0, 25, 50, 75, 100])
        for row in range(2):
            axes[row, column].set_xlim(
                -0.15,
                float(summary["target_rms_percent_case_sigma"].max()) + 0.15,
            )
            _style_axis(axes[row, column])
    axes[0, 0].set_ylabel("Median macro Dice decrease")
    axes[1, 0].set_ylabel("Bootstrap Pareto-front selection (%)")
    _add_panel_labels(axes)
    handles, labels = axes[0, 0].get_legend_handles_labels()
    fig.legend(
        handles,
        labels,
        loc="upper center",
        ncol=3,
        frameon=False,
        bbox_to_anchor=(0.5, 0.935),
        handlelength=3.0,
        columnspacing=2.2,
    )
    cohort_sizes = summary.groupby("dataset")["n_cases"].first().to_dict()
    grid = sorted(summary["target_rms_percent_case_sigma"].unique())
    step = float(np.median(np.diff(grid))) if len(grid) > 1 else 0.0
    fig.suptitle(
        "Dense patient-paired attack-damage frontier with uncertainty",
        y=0.985,
    )
    fig.text(
        0.5,
        0.018,
        f"Fixed cohorts: WG n={int(cohort_sizes.get('wg', 0))}, zones "
        f"n={int(cohort_sizes.get('zones', 0))}; RMS grid step={step:g} percentage "
        "points. Bands are paired patient-bootstrap 95% CIs. Pareto selection "
        "is recomputed in every bootstrap draw.",
        ha="center",
        color="#444444",
        fontsize=10.5,
    )
    fig.tight_layout(rect=(0.025, 0.065, 1, 0.875), h_pad=2.2, w_pad=2.0)
    path = output_dir / "04_attack_damage_pareto_front.png"
    _save_figure(fig, path)
    plt.close(fig)
    return path


def plot_overlap_volume_damage(summary: pd.DataFrame, output_dir: Path) -> Path:
    configurations = (
        ("dice_drop", "Median macro Dice decrease"),
        ("iou_drop", "Median macro IoU decrease"),
        (
            "absolute_volume_error_increase_percent",
            "Median absolute volume-error\nincrease (percentage points)",
        ),
    )
    fig, axes = plt.subplots(3, 2, figsize=(16, 14.5), sharex="col", sharey="row")
    for column, dataset in enumerate(("wg", "zones")):
        for row, (metric, ylabel) in enumerate(configurations):
            ax = axes[row, column]
            for attack in ATTACKS:
                group = summary[
                    (summary["dataset"] == dataset) & (summary["attack"] == attack)
                ]
                _plot_dense_metric(ax, group, metric, attack)
            if column == 0:
                ax.set_ylabel(ylabel)
            if row == 0:
                ax.set_title(DATASET_LABELS[dataset])
            if row == 2:
                ax.set_xlabel("Perturbation RMS (% of clean-image SD)")
            ax.axhline(0.0, color="#777777", linewidth=0.9, linestyle=":", zorder=1)
            _style_axis(ax)
        axes[0, column].set_ylim(-0.03, 1.02)
        axes[1, column].set_ylim(-0.03, 1.02)
    _add_panel_labels(axes)
    handles, labels = axes[0, 0].get_legend_handles_labels()
    fig.legend(
        handles,
        labels,
        loc="upper center",
        ncol=3,
        frameon=False,
        bbox_to_anchor=(0.5, 0.945),
        handlelength=3.0,
        columnspacing=2.2,
    )
    fig.suptitle("RMS-matched overlap and volume damage", y=0.988)
    fig.text(
        0.5,
        0.018,
        "All curves use the same fixed patient cohorts and paired bootstrap "
        "draws. Positive values indicate worse segmentation than the clean model.",
        ha="center",
        color="#444444",
        fontsize=10.5,
    )
    fig.tight_layout(rect=(0.025, 0.05, 1, 0.895), h_pad=2.0, w_pad=2.0)
    path = output_dir / "05_overlap_volume_damage.png"
    _save_figure(fig, path)
    plt.close(fig)
    return path


def plot_boundary_damage(summary: pd.DataFrame, output_dir: Path) -> Path:
    fig, axes = plt.subplots(3, 2, figsize=(16, 14.5), sharex="col", sharey="row")
    for column, dataset in enumerate(("wg", "zones")):
        for attack in ATTACKS:
            group = summary[
                (summary["dataset"] == dataset) & (summary["attack"] == attack)
            ]
            _plot_dense_metric(axes[0, column], group, "hd95_increase_mm", attack)
            _plot_dense_metric(axes[1, column], group, "asd_increase_mm", attack)
            ordered = group.sort_values("target_rms_percent_case_sigma")
            axes[2, column].plot(
                ordered["target_rms_percent_case_sigma"],
                100.0 * ordered["hd95_increase_mm_finite_fraction"],
                color=ATTACK_COLORS[attack],
                linestyle=ATTACK_LINESTYLES[attack],
                linewidth=2.5,
                label=ATTACK_LABELS[attack],
            )
        axes[0, column].set_title(DATASET_LABELS[dataset])
        axes[2, column].set_xlabel("Perturbation RMS (% of clean-image SD)")
        axes[2, column].set_ylim(-4, 104)
        axes[2, column].set_yticks([0, 25, 50, 75, 100])
        for row in range(3):
            _style_axis(axes[row, column])
    axes[0, 0].set_ylabel("Median HD95 increase (mm)")
    axes[1, 0].set_ylabel("Median ASD increase (mm)")
    axes[2, 0].set_ylabel("Cases with defined\nboundary metrics (%)")
    _add_panel_labels(axes)
    handles, labels = axes[0, 0].get_legend_handles_labels()
    fig.legend(
        handles,
        labels,
        loc="upper center",
        ncol=3,
        frameon=False,
        bbox_to_anchor=(0.5, 0.945),
        handlelength=3.0,
        columnspacing=2.2,
    )
    fig.suptitle("RMS-matched boundary displacement and metric availability", y=0.988)
    fig.text(
        0.5,
        0.018,
        "Bands are paired patient-bootstrap 95% CIs. HD95/ASD are undefined "
        "when a foreground mask is empty; macro availability requires every class.",
        ha="center",
        color="#444444",
        fontsize=10.5,
    )
    fig.tight_layout(rect=(0.025, 0.05, 1, 0.895), h_pad=2.0, w_pad=2.0)
    path = output_dir / "06_boundary_damage.png"
    _save_figure(fig, path)
    plt.close(fig)
    return path


def plot_per_class_damage(summary: pd.DataFrame, output_dir: Path) -> Path:
    panels = (
        ("wg", "WG"),
        ("zones", "TZ+CZ"),
        ("zones", "PZ"),
    )
    fig, axes = plt.subplots(1, 3, figsize=(19, 6.8), sharex=True, sharey=True)
    for ax, (dataset, class_name) in zip(axes, panels):
        for attack in ATTACKS:
            group = summary[
                (summary["dataset"] == dataset)
                & (summary["class"] == class_name)
                & (summary["attack"] == attack)
            ]
            _plot_dense_metric(ax, group, "dice_drop", attack)
        ax.set_title(CLASS_LABELS[(dataset, class_name)])
        ax.set_xlabel("Perturbation RMS (% of clean-image SD)")
        ax.set_ylim(-0.03, 1.02)
        ax.set_yticks(np.arange(0.0, 1.01, 0.2))
        _style_axis(ax)
    axes[0].set_ylabel("Median class Dice decrease")
    _add_panel_labels(np.asarray([axes]))
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(
        handles,
        labels,
        loc="upper center",
        ncol=3,
        frameon=False,
        bbox_to_anchor=(0.5, 0.91),
        handlelength=3.0,
        columnspacing=2.2,
    )
    fig.suptitle("Dense RMS-matched damage by anatomical class", y=0.985)
    fig.text(
        0.5,
        0.025,
        "Lines are fixed-cohort patient medians; bands are paired patient-bootstrap 95% CIs.",
        ha="center",
        color="#444444",
        fontsize=10.5,
    )
    fig.tight_layout(rect=(0.025, 0.07, 1, 0.84), w_pad=2.0)
    path = output_dir / "07_per_class_damage.png"
    _save_figure(fig, path)
    plt.close(fig)
    return path


def _markdown_table(
    frame: pd.DataFrame,
    columns: list[tuple[str, str]],
    formatters: dict[str, str],
) -> str:
    headers = [label for _column, label in columns]
    lines = [
        "| " + " | ".join(headers) + " |",
        "| " + " | ".join("---" for _ in headers) + " |",
    ]
    for row in frame.itertuples(index=False):
        mapping = row._asdict()
        values: list[str] = []
        for column, _label in columns:
            value = mapping[column]
            if column in formatters and pd.notna(value):
                values.append(format(value, formatters[column]))
            else:
                values.append(str(value))
        lines.append("| " + " | ".join(values) + " |")
    return "\n".join(lines)


def write_report(
    output_dir: Path,
    *,
    audit: dict[str, float],
    rms_summary: pd.DataFrame,
    failure_summary: pd.DataFrame,
    auc_summary: pd.DataFrame,
    dense_summary: pd.DataFrame,
    class_dense_summary: pd.DataFrame,
    bootstrap_replicates: int,
    figures: list[Path],
) -> Path:
    def ci_text(row: pd.Series, metric: str, digits: int) -> str:
        values = [
            row[f"{metric}_median"],
            row[f"{metric}_ci95_low"],
            row[f"{metric}_ci95_high"],
        ]
        if not np.isfinite(np.asarray(values, dtype=float)).all():
            return "NA"
        return (
            f"{values[0]:.{digits}f} [{values[1]:.{digits}f}, {values[2]:.{digits}f}]"
        )

    highest_target = float(rms_summary["target_rms_percent_case_sigma"].max())
    rms_extract = rms_summary[
        rms_summary["target_rms_percent_case_sigma"] == highest_target
    ].copy()
    rms_extract["cohort"] = rms_extract["dataset"].map(DATASET_LABELS)
    rms_extract["attack_label"] = rms_extract["attack"].map(ATTACK_LABELS)
    failure_extract = failure_summary.copy()
    failure_extract["cohort"] = failure_extract["dataset"].map(DATASET_LABELS)
    failure_extract["attack_label"] = failure_extract["attack"].map(ATTACK_LABELS)
    auc_extract = auc_summary.copy()
    auc_extract["cohort"] = auc_extract["dataset"].map(DATASET_LABELS)
    auc_extract["attack_label"] = auc_extract["attack"].map(ATTACK_LABELS)
    dense_target = float(dense_summary["target_rms_percent_case_sigma"].max())
    dense_extract = dense_summary[
        dense_summary["target_rms_percent_case_sigma"] == dense_target
    ].copy()
    dense_extract["cohort"] = dense_extract["dataset"].map(DATASET_LABELS)
    dense_extract["attack_label"] = dense_extract["attack"].map(ATTACK_LABELS)
    dense_extract["dice_ci"] = dense_extract.apply(
        ci_text, axis=1, metric="dice_drop", digits=3
    )
    dense_extract["iou_ci"] = dense_extract.apply(
        ci_text, axis=1, metric="iou_drop", digits=3
    )
    dense_extract["volume_ci"] = dense_extract.apply(
        ci_text,
        axis=1,
        metric="absolute_volume_error_increase_percent",
        digits=1,
    )
    dense_extract["hd95_ci"] = dense_extract.apply(
        ci_text, axis=1, metric="hd95_increase_mm", digits=2
    )
    dense_extract["asd_ci"] = dense_extract.apply(
        ci_text, axis=1, metric="asd_increase_mm", digits=2
    )
    dense_extract["boundary_defined_percent"] = (
        100.0 * dense_extract["hd95_increase_mm_finite_fraction"]
    )
    dense_extract["empty_prediction_percent"] = (
        100.0 * dense_extract["empty_prediction_fraction"]
    )

    class_extract = class_dense_summary[
        class_dense_summary["target_rms_percent_case_sigma"] == dense_target
    ].copy()
    class_extract["anatomical_class"] = class_extract.apply(
        lambda row: CLASS_LABELS[(str(row["dataset"]), str(row["class"]))],
        axis=1,
    )
    class_extract["attack_label"] = class_extract["attack"].map(ATTACK_LABELS)
    class_extract["dice_ci"] = class_extract.apply(
        ci_text, axis=1, metric="dice_drop", digits=3
    )

    grid = sorted(dense_summary["target_rms_percent_case_sigma"].unique())
    dense_step = float(np.median(np.diff(grid))) if len(grid) > 1 else 0.0
    fixed_cohorts = dense_summary.groupby("dataset")["n_cases"].first().to_dict()
    if audit["boundary_metrics_available"]:
        boundary_scope = (
            "Dice, IoU, volume error, HD95, and ASD were computed for the "
            "segmentation analysis."
        )
        damage_direction = (
            "Positive Dice decrease, HD95 increase, and ASD increase mean worse "
            "segmentation. Curves start at the clean 0/255 reference."
        )
    else:
        boundary_scope = (
            "Dice, IoU, and volume error are available, but HD95 and ASD were "
            "not computed in the supplied segmentation source. Boundary plots "
            "therefore report zero availability rather than silently dropping cases."
        )
        damage_direction = (
            "Positive Dice decrease means worse segmentation. Curves start at "
            "the clean 0/255 reference."
        )
    report = [
        "# RMS-matched image quality and segmentation-damage experiments",
        "",
        "## Scope and pairing",
        "",
        f"The analysis contains {int(audit['cases']):,} cases and "
        f"{int(audit['rows']):,} measured case–attack–epsilon rows. Image-quality "
        "and segmentation records were joined one-to-one by dataset, case, attack, "
        "and epsilon.",
        "",
        "The same-pass attack audit found a maximum relative RMS difference of "
        f"{audit['max_rms_relative_difference']:.4%} and a 99th-percentile "
        f"difference of {audit['q99_rms_relative_difference']:.4%}.",
        "",
        boundary_scope,
        "",
        "## Experiment 1: RMS-matched attacks",
        "",
        "Each patient's five measured epsilon points were linearly interpolated "
        "to common realized-RMS targets. Only patients supported by all three "
        "attacks at a target enter the primary paired comparison. PSNR was "
        "recomputed analytically from the target RMSE and that patient's robust "
        "signal range.",
        "",
        f"### Results at {highest_target:g}% of clean case SD",
        "",
        _markdown_table(
            rms_extract,
            [
                ("cohort", "Cohort"),
                ("attack_label", "Attack"),
                ("n_cases", "Paired cases"),
                ("ssim_axial_prostate_median", "Median SSIM"),
                ("dice_drop_median", "Median Dice decrease"),
                ("epsilon_n_median", "Estimated median ε (/255)"),
            ],
            {
                "ssim_axial_prostate_median": ".4f",
                "dice_drop_median": ".4f",
                "epsilon_n_median": ".2f",
            },
        ),
        "",
        f"![RMS-matched comparison]({figures[0].name})",
        "",
        "## Experiment 2: image quality versus segmentation damage",
        "",
        damage_direction,
        "",
        f"![Quality versus damage]({figures[1].name})",
        "",
        "### Dice-failure thresholds",
        "",
        _markdown_table(
            failure_extract,
            [
                ("cohort", "Cohort"),
                ("attack_label", "Attack"),
                ("dice_drop_threshold", "Dice decrease"),
                ("attained_fraction", "Attainment fraction"),
                ("n_paired_attained_all_attacks", "Paired n"),
                (
                    "rms_percent_case_sigma_paired_median",
                    "Paired median RMS (% SD)",
                ),
                (
                    "ssim_axial_prostate_paired_median",
                    "Paired median SSIM",
                ),
            ],
            {
                "dice_drop_threshold": ".2f",
                "attained_fraction": ".1%",
                "rms_percent_case_sigma_paired_median": ".3f",
                "ssim_axial_prostate_paired_median": ".4f",
            },
        ),
        "",
        f"![Failure thresholds]({figures[2].name})",
        "",
        "The paired threshold sample size changes with the threshold because all "
        "three attacks must attain it; the table reports that selected-cohort size "
        "explicitly.",
        "",
        "### Normalized damage AUC over the common RMS range",
        "",
        _markdown_table(
            auc_extract,
            [
                ("cohort", "Cohort"),
                ("attack_label", "Attack"),
                ("n_cases", "Paired cases"),
                ("normalized_damage_auc_median", "Median damage AUC"),
            ],
            {"normalized_damage_auc_median": ".4f"},
        ),
        "",
        "## Dense patient-paired RMS frontier",
        "",
        f"The dense analysis uses a {dense_step:g}-percentage-point RMS grid from "
        f"0 to {dense_target:g}% of clean case SD. It fixes one cohort across the "
        f"entire grid (WG n={int(fixed_cohorts.get('wg', 0))}; zones "
        f"n={int(fixed_cohorts.get('zones', 0))}) and uses "
        f"{bootstrap_replicates:,} patient-level bootstrap draws. Each draw keeps "
        "all attacks and RMS targets for a sampled patient together. Pareto-front "
        "membership is recomputed in every draw.",
        "",
        f"![Dense paired frontier]({figures[3].name})",
        "",
        f"### Overlap and volume damage at {dense_target:g}% RMS",
        "",
        "Absolute volume-error increase is defined as |adversarial volume bias| "
        "minus |clean volume bias| in percentage points; positive values mean the "
        "attack worsened volumetric calibration.",
        "",
        _markdown_table(
            dense_extract,
            [
                ("cohort", "Cohort"),
                ("attack_label", "Attack"),
                ("n_cases", "Paired n"),
                ("dice_ci", "Dice decrease [95% CI]"),
                ("iou_ci", "IoU decrease [95% CI]"),
                ("volume_ci", "Absolute volume-error increase, pp [95% CI]"),
                ("empty_prediction_percent", "Empty prediction (%)"),
            ],
            {"empty_prediction_percent": ".1f"},
        ),
        "",
        f"![Overlap and volume damage]({figures[4].name})",
        "",
        "### Boundary damage",
        "",
        _markdown_table(
            dense_extract,
            [
                ("cohort", "Cohort"),
                ("attack_label", "Attack"),
                ("hd95_ci", "HD95 increase, mm [95% CI]"),
                ("asd_ci", "ASD increase, mm [95% CI]"),
                ("boundary_defined_percent", "Boundary metrics defined (%)"),
            ],
            {"boundary_defined_percent": ".1f"},
        ),
        "",
        "HD95 and ASD are undefined when either foreground mask is empty. Their "
        "medians and confidence intervals therefore describe defined cases only; "
        "the macro result requires every foreground class to be defined. The "
        "availability percentage and empty-prediction rate must accompany them.",
        "",
        f"![Boundary damage]({figures[5].name})",
        "",
        f"### Anatomical-class damage at {dense_target:g}% RMS",
        "",
        _markdown_table(
            class_extract,
            [
                ("anatomical_class", "Class"),
                ("attack_label", "Attack"),
                ("n_cases", "Paired n"),
                ("dice_ci", "Class Dice decrease [95% CI]"),
            ],
            {},
        ),
        "",
        f"![Per-class damage]({figures[6].name})",
        "",
        "## Machine-readable outputs",
        "",
        "- `dense_rms_paired_front_summary.csv`: macro-metric medians, paired "
        "bootstrap CIs, finite-case fractions, empty-prediction rates, and "
        "Pareto-selection probabilities.",
        "- `dense_rms_paired_front_differences.csv`: paired FGSM/APGD and "
        "PGD/APGD differences with bootstrap CIs.",
        "- `dense_rms_paired_per_class_summary.csv`: the same uncertainty-aware "
        "summary for WG, TZ+CZ, and PZ.",
        "- `dense_rms_matched_paired_per_case.csv` and "
        "`dense_rms_matched_paired_per_class.csv`: fixed-cohort patient-level "
        "interpolated records.",
        "- `analysis_manifest.json`: source paths, RMS grid, pairing unit, "
        "bootstrap count, interpolation method, and seed.",
        "",
        "## Interpretation limits",
        "",
        "- RMS-matched points between measured epsilon levels are interpolated "
        "estimates, not newly generated images.",
        "- A larger Dice decrease at the same RMS indicates a more distortion-"
        "efficient attack against this model; it does not establish human "
        "visibility or patient harm.",
        "- Failure-threshold medians use patients for whom all three attacks "
        "reached the threshold; attainment fractions use the complete cohort and "
        "must be reported alongside those conditional medians.",
        "- HD95/ASD summaries are conditional on non-empty masks; their finite-case "
        "fraction is part of the result, not a technical nuisance.",
        "- The attacked checkpoints are fold-all models evaluated on their full "
        "preprocessed cohorts, so these are in-sample stress-test results rather "
        "than held-out generalization estimates.",
        "- APGD denotes the repository's custom deterministic 20-step APGD-BCE "
        "variant, not an independent reference-APGD certification.",
        "- Radiologist conclusions require the separate blinded reader study and "
        "cannot be inferred from these automated metrics.",
        "",
    ]
    path = output_dir / "README.md"
    path.write_text("\n".join(report), encoding="utf-8")
    return path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Run paired RMS-matched and image-quality-versus-segmentation-damage "
            "analyses for FGSM, PGD, and APGD"
        )
    )
    parser.add_argument("--quality-csv", type=Path, default=DEFAULT_QUALITY_CSV)
    parser.add_argument(
        "--segmentation-csv", type=Path, default=DEFAULT_SEGMENTATION_CSV
    )
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument(
        "--rms-targets",
        nargs="+",
        type=float,
        default=[0.0, 0.5, 1.0, 2.0, 4.0, 7.0],
        help="target realized RMS values expressed as percent of clean case SD",
    )
    parser.add_argument(
        "--dice-drop-thresholds",
        nargs="+",
        type=float,
        default=[0.05, 0.10, 0.20],
    )
    parser.add_argument(
        "--dense-rms-step",
        type=float,
        default=0.25,
        help="RMS-grid spacing in percentage points of clean case SD",
    )
    parser.add_argument(
        "--dense-rms-maximum",
        type=float,
        default=7.0,
        help="maximum RMS target for the fixed-cohort dense paired front",
    )
    parser.add_argument(
        "--bootstrap-replicates",
        type=int,
        default=2000,
        help="patient-paired bootstrap draws for dense-front uncertainty",
    )
    parser.add_argument("--norm-relative-tolerance", type=float, default=0.02)
    parser.add_argument("--seed", type=int, default=20260722)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    rms_targets = sorted(set(float(value) for value in args.rms_targets))
    thresholds = sorted(set(float(value) for value in args.dice_drop_thresholds))
    if args.dense_rms_step <= 0:
        raise ValueError("--dense-rms-step must be positive")
    if args.dense_rms_maximum <= 0:
        raise ValueError("--dense-rms-maximum must be positive")
    if args.bootstrap_replicates < 1:
        raise ValueError("--bootstrap-replicates must be positive")
    dense_targets = np.arange(
        0.0,
        args.dense_rms_maximum + args.dense_rms_step / 2.0,
        args.dense_rms_step,
    )
    dense_targets = sorted(
        set(float(value) for value in np.round(dense_targets, decimals=10))
    )
    if not rms_targets or rms_targets[0] < 0:
        raise ValueError("--rms-targets must be non-negative")
    if not thresholds or thresholds[0] <= 0:
        raise ValueError("--dice-drop-thresholds must be positive")
    if args.norm_relative_tolerance < 0:
        raise ValueError("--norm-relative-tolerance must be non-negative")

    output_dir = args.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    analysis_manifest = {
        "status": "running",
        "quality_csv": str(args.quality_csv.resolve()),
        "segmentation_csv": str(args.segmentation_csv.resolve()),
        "output_dir": str(output_dir),
        "rms_targets_percent_case_sigma": rms_targets,
        "dense_rms_targets_percent_case_sigma": dense_targets,
        "dense_rms_step": float(args.dense_rms_step),
        "dense_rms_maximum": float(args.dense_rms_maximum),
        "dice_drop_thresholds": thresholds,
        "bootstrap_replicates": int(args.bootstrap_replicates),
        "bootstrap_unit": "patient_id_with_all_attacks_and_rms_targets_paired",
        "interpolation": "linear_between_measured_epsilon_levels_per_patient",
        "norm_relative_tolerance": float(args.norm_relative_tolerance),
        "seed": int(args.seed),
    }
    (output_dir / "analysis_manifest.json").write_text(
        json.dumps(analysis_manifest, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    _configure_style()
    paired, audit = load_paired_data(
        args.quality_csv.resolve(),
        args.segmentation_csv.resolve(),
        norm_relative_tolerance=args.norm_relative_tolerance,
    )
    paired.to_csv(output_dir / "quality_damage_per_case.csv", index=False)
    class_paired = load_class_damage_data(args.segmentation_csv.resolve())
    class_paired.to_csv(output_dir / "quality_damage_per_class.csv", index=False)

    available, rms_matched = interpolate_rms_matched(paired, rms_targets)
    available.to_csv(output_dir / "rms_matched_available_per_case.csv", index=False)
    rms_matched.to_csv(output_dir / "rms_matched_paired_per_case.csv", index=False)
    rms_summary = _summarize(
        rms_matched,
        ["dataset", "attack", "target_rms_percent_case_sigma"],
        (
            "epsilon_n",
            "psnr_db_robust_range",
            "ssim_axial_prostate",
            "adversarial_dice",
            "dice_drop",
            "adversarial_iou",
            "iou_drop",
            "absolute_volume_error_increase_percent",
            "hd95_increase_mm",
            "asd_increase_mm",
        ),
    )
    rms_summary.to_csv(output_dir / "rms_matched_summary.csv", index=False)
    paired_differences = paired_rms_differences(rms_matched, seed=args.seed)
    paired_differences.to_csv(
        output_dir / "rms_matched_paired_differences.csv", index=False
    )

    threshold_rows = failure_thresholds(paired, thresholds)
    threshold_rows.to_csv(
        output_dir / "segmentation_failure_thresholds_per_case.csv", index=False
    )
    failure_summary = summarize_failure_thresholds(threshold_rows)
    failure_summary.to_csv(
        output_dir / "segmentation_failure_thresholds_summary.csv", index=False
    )

    auc_per_case, auc_summary = damage_auc(rms_matched)
    auc_per_case.to_csv(output_dir / "damage_auc_per_case.csv", index=False)
    auc_summary.to_csv(output_dir / "damage_auc_summary.csv", index=False)

    measured_summary = measured_summary_with_clean(paired)
    measured_summary.to_csv(
        output_dir / "quality_damage_measured_summary.csv", index=False
    )
    pareto = mark_pareto_points(measured_summary)
    pareto.to_csv(output_dir / "pareto_operating_points.csv", index=False)

    dense_available, dense_matched = interpolate_rms_matched(paired, dense_targets)
    dense_available.to_csv(
        output_dir / "dense_rms_matched_available_per_case.csv", index=False
    )
    dense_common, dense_summary, dense_differences = (
        dense_paired_front_with_uncertainty(
            dense_matched,
            seed=args.seed,
            n_bootstrap=args.bootstrap_replicates,
        )
    )
    dense_common.to_csv(
        output_dir / "dense_rms_matched_paired_per_case.csv", index=False
    )
    dense_summary.to_csv(output_dir / "dense_rms_paired_front_summary.csv", index=False)
    dense_differences.to_csv(
        output_dir / "dense_rms_paired_front_differences.csv", index=False
    )

    class_dense_available, class_dense_matched = interpolate_rms_matched(
        class_paired,
        dense_targets,
        strata=("class",),
    )
    class_dense_available.to_csv(
        output_dir / "dense_rms_matched_available_per_class.csv", index=False
    )
    class_dense_common, class_dense_summary, class_dense_differences = (
        dense_paired_front_with_uncertainty(
            class_dense_matched,
            seed=args.seed + 500_000,
            n_bootstrap=args.bootstrap_replicates,
            strata=("class",),
        )
    )
    class_dense_common.to_csv(
        output_dir / "dense_rms_matched_paired_per_class.csv", index=False
    )
    class_dense_summary.to_csv(
        output_dir / "dense_rms_paired_per_class_summary.csv", index=False
    )
    class_dense_differences.to_csv(
        output_dir / "dense_rms_paired_per_class_differences.csv", index=False
    )

    figures = [
        plot_rms_matched(rms_summary, output_dir),
        plot_quality_damage(measured_summary, output_dir),
        plot_failure_thresholds(failure_summary, output_dir),
        plot_dense_paired_front(dense_summary, output_dir),
        plot_overlap_volume_damage(dense_summary, output_dir),
        plot_boundary_damage(dense_summary, output_dir),
        plot_per_class_damage(class_dense_summary, output_dir),
    ]
    report = write_report(
        output_dir,
        audit=audit,
        rms_summary=rms_summary,
        failure_summary=failure_summary,
        auc_summary=auc_summary,
        dense_summary=dense_summary,
        class_dense_summary=class_dense_summary,
        bootstrap_replicates=args.bootstrap_replicates,
        figures=figures,
    )
    fixed_cohorts = {
        str(dataset): int(group["n_cases"].iloc[0])
        for dataset, group in dense_summary.groupby("dataset", sort=True)
    }
    per_class_fixed_cohorts = {
        f"{dataset}/{class_name}": int(group["n_cases"].iloc[0])
        for (dataset, class_name), group in class_dense_summary.groupby(
            ["dataset", "class"], sort=True
        )
    }
    analysis_manifest.update(
        {
            "status": "complete",
            "join_audit": audit,
            "paired_rows": int(len(paired)),
            "class_rows": int(len(class_paired)),
            "dense_fixed_cohort_rows": int(len(dense_common)),
            "dense_fixed_cohorts": fixed_cohorts,
            "dense_per_class_fixed_cohorts": per_class_fixed_cohorts,
        }
    )
    manifest_path = output_dir / "analysis_manifest.json"
    manifest_tmp = manifest_path.with_suffix(".json.tmp")
    manifest_tmp.write_text(
        json.dumps(analysis_manifest, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    manifest_tmp.replace(manifest_path)
    print(f"Paired rows: {len(paired):,}", flush=True)
    print(f"RMS-matched paired rows: {len(rms_matched):,}", flush=True)
    print(f"Dense fixed-cohort paired rows: {len(dense_common):,}", flush=True)
    print(
        f"Maximum RMS reproduction difference: {audit['max_rms_relative_difference']:.4%}"
    )
    for path in [*figures, report]:
        print(f"Saved {path}", flush=True)


if __name__ == "__main__":
    main()
