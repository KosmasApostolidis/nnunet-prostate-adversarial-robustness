"""Analyze completed blinded radiologist visibility and adequacy ratings.

The script joins locked reader responses to the private truth keys, validates
allowed responses, and reports forced-choice detection, perceived visibility,
diagnostic/contouring adequacy, confidence, and repeat agreement. It never
creates or imputes reader ratings.
"""

from __future__ import annotations

import argparse
import math
from pathlib import Path

import matplotlib
import numpy as np
import pandas as pd

matplotlib.use("Agg")
import matplotlib.pyplot as plt


REPO_ROOT = Path(__file__).resolve().parents[1]
CALIBRATION_ROOT = (
    REPO_ROOT / "results" / "adv_rob_eval_results" / "image_quality_analysis"
)
DEFAULT_STUDY_DIR = CALIBRATION_ROOT / "radiologist_attack_reader_study"
DEFAULT_OUTPUT_DIR = DEFAULT_STUDY_DIR / "analysis"

ATTACKS = ["FGSM-BCE", "PGD-BCE", "APGD-BCE"]
ATTACK_LABELS = {
    "FGSM-BCE": "FGSM",
    "PGD-BCE": "PGD",
    "APGD-BCE": "APGD",
    "clean": "Clean",
}
ATTACK_COLORS = {
    "FGSM-BCE": "#0072B2",
    "PGD-BCE": "#E69F00",
    "APGD-BCE": "#D55E00",
    "clean": "#555555",
}
ATTACK_MARKERS = {
    "FGSM-BCE": "o",
    "PGD-BCE": "s",
    "APGD-BCE": "^",
    "clean": "D",
}
DATASET_LABELS = {"wg": "Whole gland (WG)", "zones": "Prostate zones"}


def _read_many(paths: list[Path]) -> pd.DataFrame:
    frames: list[pd.DataFrame] = []
    for path in paths:
        if not path.is_file():
            raise FileNotFoundError(path)
        frame = pd.read_csv(path, dtype={"reader_id": str})
        frame["source_score_file"] = str(path.resolve())
        frames.append(frame)
    if not frames:
        raise ValueError("at least one score file is required")
    return pd.concat(frames, ignore_index=True)


def _require_columns(frame: pd.DataFrame, required: set[str], label: str) -> None:
    missing = sorted(required - set(frame.columns))
    if missing:
        raise ValueError(f"{label} is missing columns: {missing}")


def _normalize_yes_no(values: pd.Series, label: str) -> pd.Series:
    normalized = values.astype(str).str.strip().str.lower()
    mapping = {
        "yes": True,
        "y": True,
        "1": True,
        "true": True,
        "no": False,
        "n": False,
        "0": False,
        "false": False,
    }
    invalid = sorted(set(normalized) - set(mapping))
    if invalid:
        raise ValueError(f"{label} contains invalid yes/no values: {invalid}")
    return normalized.map(mapping).astype(bool)


def _numeric_rating(values: pd.Series, label: str, lower: int, upper: int) -> pd.Series:
    numeric = pd.to_numeric(values, errors="coerce")
    invalid = numeric.isna() | (numeric < lower) | (numeric > upper)
    if invalid.any():
        examples = values[invalid].astype(str).head(5).tolist()
        raise ValueError(
            f"{label} must be an integer in [{lower}, {upper}]; examples: {examples}"
        )
    if not np.allclose(numeric, np.round(numeric)):
        raise ValueError(f"{label} must contain integer ratings")
    return numeric.astype(int)


def _reader_ids(frame: pd.DataFrame, label: str) -> pd.Series:
    if "reader_id" not in frame:
        raise ValueError(f"{label} is missing reader_id")
    readers = frame["reader_id"].fillna("").astype(str).str.strip()
    if (readers == "").any():
        raise ValueError(
            f"{label} contains blank reader_id values; complete a separate "
            "template for every reader"
        )
    return readers


def load_visibility_scores(score_paths: list[Path], truth_path: Path) -> pd.DataFrame:
    scores = _read_many(score_paths)
    truth = pd.read_csv(truth_path)
    _require_columns(
        scores,
        {
            "reader_id",
            "pair_id",
            "selected_altered_image",
            "difference_visible",
            "confidence_1_to_5",
        },
        "visibility scores",
    )
    _require_columns(
        truth,
        {
            "pair_id",
            "altered_side",
            "dataset",
            "case_id",
            "attack",
            "target_rms_percent_case_sigma",
            "is_reliability_repeat",
            "source_pair_id",
        },
        "visibility truth",
    )
    scores["reader_id"] = _reader_ids(scores, "visibility scores")
    scores["pair_id"] = scores["pair_id"].astype(str)
    selected = scores["selected_altered_image"].astype(str).str.strip().str.upper()
    invalid = sorted(set(selected) - {"A", "B"})
    if invalid:
        raise ValueError(
            "selected_altered_image must contain only A or B; "
            f"invalid values: {invalid}"
        )
    scores["selected_altered_image"] = selected
    scores["difference_visible_bool"] = _normalize_yes_no(
        scores["difference_visible"], "difference_visible"
    )
    scores["confidence_1_to_5"] = _numeric_rating(
        scores["confidence_1_to_5"], "visibility confidence", 1, 5
    )
    if scores.duplicated(["reader_id", "pair_id"]).any():
        raise ValueError("visibility scores contain duplicate reader/pair ratings")
    if truth.duplicated("pair_id").any():
        raise ValueError("visibility truth contains duplicate pair_id values")
    unknown = sorted(set(scores["pair_id"]) - set(truth["pair_id"].astype(str)))
    if unknown:
        raise ValueError(f"visibility scores contain unknown pair IDs: {unknown[:5]}")
    joined = scores.merge(
        truth,
        on="pair_id",
        how="left",
        validate="many_to_one",
        suffixes=("", "_truth"),
    )
    joined["correct"] = (
        joined["selected_altered_image"]
        == joined["altered_side"].astype(str).str.upper()
    )
    return joined


def load_adequacy_scores(score_paths: list[Path], truth_path: Path) -> pd.DataFrame:
    scores = _read_many(score_paths)
    truth = pd.read_csv(truth_path)
    _require_columns(
        scores,
        {
            "reader_id",
            "adequacy_item_id",
            "capsule_visibility_1_to_5",
            "zonal_anatomy_visibility_1_to_5",
            "adequate_for_segmentation_correction_yes_no",
            "overall_quality_1_to_3",
            "confidence_1_to_5",
        },
        "adequacy scores",
    )
    _require_columns(
        truth,
        {
            "adequacy_item_id",
            "dataset",
            "case_id",
            "condition",
            "attack",
            "target_rms_percent_case_sigma",
        },
        "adequacy truth",
    )
    scores["reader_id"] = _reader_ids(scores, "adequacy scores")
    scores["adequacy_item_id"] = scores["adequacy_item_id"].astype(str)
    scores["capsule_visibility_1_to_5"] = _numeric_rating(
        scores["capsule_visibility_1_to_5"],
        "capsule visibility",
        1,
        5,
    )
    zonal_raw = scores["zonal_anatomy_visibility_1_to_5"]
    zonal_text = zonal_raw.astype(str).str.strip().str.lower()
    zonal_missing = zonal_raw.isna() | zonal_text.isin({"", "na", "n/a", "nan"})
    zonal_numeric = pd.to_numeric(zonal_raw.where(~zonal_missing), errors="coerce")
    invalid_zonal = (~zonal_missing) & (
        zonal_numeric.isna() | (zonal_numeric < 1) | (zonal_numeric > 5)
    )
    if invalid_zonal.any():
        raise ValueError("zonal anatomy visibility must be 1–5 or NA")
    scores["zonal_anatomy_visibility_1_to_5"] = zonal_numeric
    scores["adequate_bool"] = _normalize_yes_no(
        scores["adequate_for_segmentation_correction_yes_no"],
        "adequate_for_segmentation_correction_yes_no",
    )
    scores["overall_quality_1_to_3"] = _numeric_rating(
        scores["overall_quality_1_to_3"], "overall quality", 1, 3
    )
    scores["confidence_1_to_5"] = _numeric_rating(
        scores["confidence_1_to_5"], "adequacy confidence", 1, 5
    )
    if scores.duplicated(["reader_id", "adequacy_item_id"]).any():
        raise ValueError("adequacy scores contain duplicate reader/item ratings")
    if truth.duplicated("adequacy_item_id").any():
        raise ValueError("adequacy truth contains duplicate item IDs")
    unknown = sorted(
        set(scores["adequacy_item_id"]) - set(truth["adequacy_item_id"].astype(str))
    )
    if unknown:
        raise ValueError(f"adequacy scores contain unknown item IDs: {unknown[:5]}")
    return scores.merge(
        truth,
        on="adequacy_item_id",
        how="left",
        validate="many_to_one",
        suffixes=("", "_truth"),
    )


def _wilson_interval(successes: int, total: int) -> tuple[float, float]:
    if total <= 0:
        return float("nan"), float("nan")
    z = 1.959963984540054
    proportion = successes / total
    denominator = 1.0 + z * z / total
    center = (proportion + z * z / (2.0 * total)) / denominator
    half = (
        z
        * math.sqrt(
            proportion * (1.0 - proportion) / total + z * z / (4.0 * total * total)
        )
        / denominator
    )
    return max(0.0, center - half), min(1.0, center + half)


def summarize_visibility(frame: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    for (dataset, attack, target), group in frame.groupby(
        ["dataset", "attack", "target_rms_percent_case_sigma"]
    ):
        n = len(group)
        correct = int(group["correct"].sum())
        visible = int(group["difference_visible_bool"].sum())
        accuracy_low, accuracy_high = _wilson_interval(correct, n)
        visible_low, visible_high = _wilson_interval(visible, n)
        rows.append(
            {
                "dataset": dataset,
                "attack": attack,
                "target_rms_percent_case_sigma": target,
                "n_ratings": n,
                "n_cases": int(group["case_id"].nunique()),
                "n_readers": int(group["reader_id"].nunique()),
                "detection_accuracy": correct / n,
                "detection_accuracy_ci95_low": accuracy_low,
                "detection_accuracy_ci95_high": accuracy_high,
                "difference_visible_fraction": visible / n,
                "difference_visible_ci95_low": visible_low,
                "difference_visible_ci95_high": visible_high,
                "mean_confidence": float(group["confidence_1_to_5"].mean()),
                "median_confidence": float(group["confidence_1_to_5"].median()),
            }
        )
    return pd.DataFrame(rows)


def visibility_reader_summary(frame: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    for reader_id, group in frame.groupby("reader_id"):
        low, high = _wilson_interval(int(group["correct"].sum()), len(group))
        rows.append(
            {
                "reader_id": reader_id,
                "n_ratings": len(group),
                "detection_accuracy": float(group["correct"].mean()),
                "detection_accuracy_ci95_low": low,
                "detection_accuracy_ci95_high": high,
                "difference_visible_fraction": float(
                    group["difference_visible_bool"].mean()
                ),
                "mean_confidence": float(group["confidence_1_to_5"].mean()),
            }
        )
    return pd.DataFrame(rows)


def visibility_repeat_agreement(frame: pd.DataFrame) -> pd.DataFrame:
    original = frame[~frame["is_reliability_repeat"].astype(bool)].set_index(
        ["reader_id", "pair_id"]
    )
    rows: list[dict[str, object]] = []
    repeats = frame[frame["is_reliability_repeat"].astype(bool)]
    for row in repeats.itertuples(index=False):
        key = (row.reader_id, str(row.source_pair_id))
        if key not in original.index:
            continue
        source = original.loc[key]
        if isinstance(source, pd.DataFrame):
            source = source.iloc[0]
        rows.append(
            {
                "reader_id": row.reader_id,
                "repeat_pair_id": row.pair_id,
                "source_pair_id": row.source_pair_id,
                "same_selected_side": (
                    row.selected_altered_image == source["selected_altered_image"]
                ),
                "same_visibility_response": (
                    bool(row.difference_visible_bool)
                    == bool(source["difference_visible_bool"])
                ),
                "confidence_absolute_difference": abs(
                    int(row.confidence_1_to_5) - int(source["confidence_1_to_5"])
                ),
            }
        )
    return pd.DataFrame(rows)


def summarize_adequacy(frame: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    group_columns = [
        "dataset",
        "condition",
        "attack",
        "target_rms_percent_case_sigma",
    ]
    for key, group in frame.groupby(group_columns):
        n = len(group)
        adequate = int(group["adequate_bool"].sum())
        low, high = _wilson_interval(adequate, n)
        zonal = group["zonal_anatomy_visibility_1_to_5"].dropna()
        rows.append(
            {
                **dict(zip(group_columns, key)),
                "n_ratings": n,
                "n_cases": int(group["case_id"].nunique()),
                "n_readers": int(group["reader_id"].nunique()),
                "adequate_fraction": adequate / n,
                "adequate_fraction_ci95_low": low,
                "adequate_fraction_ci95_high": high,
                "capsule_visibility_mean": float(
                    group["capsule_visibility_1_to_5"].mean()
                ),
                "capsule_visibility_median": float(
                    group["capsule_visibility_1_to_5"].median()
                ),
                "zonal_visibility_mean": (
                    float(zonal.mean()) if len(zonal) else float("nan")
                ),
                "zonal_visibility_median": (
                    float(zonal.median()) if len(zonal) else float("nan")
                ),
                "overall_quality_mean": float(group["overall_quality_1_to_3"].mean()),
                "overall_quality_median": float(
                    group["overall_quality_1_to_3"].median()
                ),
                "mean_confidence": float(group["confidence_1_to_5"].mean()),
            }
        )
    return pd.DataFrame(rows)


def adequacy_reader_summary(frame: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    for reader_id, group in frame.groupby("reader_id"):
        rows.append(
            {
                "reader_id": reader_id,
                "n_ratings": len(group),
                "adequate_fraction": float(group["adequate_bool"].mean()),
                "capsule_visibility_mean": float(
                    group["capsule_visibility_1_to_5"].mean()
                ),
                "zonal_visibility_mean": float(
                    group["zonal_anatomy_visibility_1_to_5"].mean()
                ),
                "overall_quality_mean": float(group["overall_quality_1_to_3"].mean()),
                "mean_confidence": float(group["confidence_1_to_5"].mean()),
            }
        )
    return pd.DataFrame(rows)


def plot_visibility(summary: pd.DataFrame, output_dir: Path) -> Path:
    fig, axes = plt.subplots(1, 2, figsize=(13, 5.2), sharex=True, sharey=True)
    for ax, dataset in zip(axes, ("wg", "zones")):
        for attack in ATTACKS:
            group = summary[
                (summary["dataset"] == dataset) & (summary["attack"] == attack)
            ].sort_values("target_rms_percent_case_sigma")
            x = group["target_rms_percent_case_sigma"].to_numpy(dtype=float)
            y = group["detection_accuracy"].to_numpy(dtype=float)
            lower = y - group["detection_accuracy_ci95_low"].to_numpy(dtype=float)
            upper = group["detection_accuracy_ci95_high"].to_numpy(dtype=float) - y
            ax.errorbar(
                x,
                y,
                yerr=np.vstack([lower, upper]),
                color=ATTACK_COLORS[attack],
                marker=ATTACK_MARKERS[attack],
                linewidth=2.0,
                capsize=3,
                label=ATTACK_LABELS[attack],
            )
        ax.axhline(
            0.5,
            color="#555555",
            linestyle="--",
            linewidth=1.2,
            label="Chance (50%)" if dataset == "wg" else None,
        )
        ax.set_title(DATASET_LABELS[dataset])
        ax.set_xlabel("RMS-matched perturbation (% of clean case SD)")
        ax.grid(alpha=0.25)
    axes[0].set_ylabel("Forced-choice attack detection accuracy")
    axes[0].set_ylim(0.35, 1.02)
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(
        handles,
        labels,
        loc="upper center",
        ncol=4,
        frameon=False,
        bbox_to_anchor=(0.5, 0.92),
    )
    fig.suptitle("Radiologist visibility of RMS-matched attacks", y=0.99)
    fig.text(
        0.5,
        0.02,
        "Points are pooled reader ratings; error bars are Wilson 95% confidence intervals.",
        ha="center",
        color="#444444",
    )
    fig.tight_layout(rect=(0, 0.06, 1, 0.87))
    path = output_dir / "01_radiologist_attack_visibility.png"
    fig.savefig(path, dpi=240, bbox_inches="tight")
    plt.close(fig)
    return path


def plot_adequacy(summary: pd.DataFrame, output_dir: Path) -> Path:
    fig, axes = plt.subplots(2, 2, figsize=(13, 8.5), sharex="col")
    for column, dataset in enumerate(("wg", "zones")):
        for attack in ["clean", *ATTACKS]:
            group = summary[
                (summary["dataset"] == dataset) & (summary["attack"] == attack)
            ].sort_values("target_rms_percent_case_sigma")
            if group.empty:
                continue
            x = group["target_rms_percent_case_sigma"].to_numpy(dtype=float)
            axes[0, column].plot(
                x,
                group["adequate_fraction"],
                color=ATTACK_COLORS[attack],
                marker=ATTACK_MARKERS[attack],
                linewidth=2.0,
                label=ATTACK_LABELS[attack],
            )
            axes[1, column].plot(
                x,
                group["overall_quality_mean"],
                color=ATTACK_COLORS[attack],
                marker=ATTACK_MARKERS[attack],
                linewidth=2.0,
                label=ATTACK_LABELS[attack],
            )
        axes[0, column].set_title(DATASET_LABELS[dataset])
        axes[0, column].set_ylabel("Adequate for segmentation correction")
        axes[0, column].set_ylim(-0.02, 1.02)
        axes[1, column].set_ylabel("Mean overall quality (1–3)")
        axes[1, column].set_ylim(0.9, 3.1)
        axes[1, column].set_xlabel("RMS-matched perturbation (% of clean case SD)")
        for row in range(2):
            axes[row, column].grid(alpha=0.25)
    handles, labels = axes[0, 0].get_legend_handles_labels()
    fig.legend(
        handles,
        labels,
        loc="upper center",
        ncol=4,
        frameon=False,
        bbox_to_anchor=(0.5, 0.93),
    )
    fig.suptitle("Blinded diagnostic and contouring adequacy ratings", y=0.99)
    fig.tight_layout(rect=(0, 0.03, 1, 0.88))
    path = output_dir / "02_radiologist_diagnostic_adequacy.png"
    fig.savefig(path, dpi=240, bbox_inches="tight")
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
        values = row._asdict()
        rendered: list[str] = []
        for column, _label in columns:
            value = values[column]
            rendered.append(
                format(value, formatters[column])
                if column in formatters and pd.notna(value)
                else str(value)
            )
        lines.append("| " + " | ".join(rendered) + " |")
    return "\n".join(lines)


def write_report(
    output_dir: Path,
    visibility_summary: pd.DataFrame,
    adequacy_summary: pd.DataFrame,
    repeat_summary: pd.DataFrame,
    figures: list[Path],
) -> Path:
    visibility = visibility_summary.copy()
    visibility["cohort"] = visibility["dataset"].map(DATASET_LABELS)
    visibility["attack_label"] = visibility["attack"].map(ATTACK_LABELS)
    adequacy = adequacy_summary.copy()
    adequacy["cohort"] = adequacy["dataset"].map(DATASET_LABELS)
    adequacy["attack_label"] = adequacy["attack"].map(ATTACK_LABELS)
    repeat_text = (
        "No completed reliability repeats were available."
        if repeat_summary.empty
        else (
            f"Across {len(repeat_summary)} completed repeats, altered-side "
            f"selection agreement was "
            f"{repeat_summary['same_selected_side'].mean():.1%} and visible/not-"
            f"visible agreement was "
            f"{repeat_summary['same_visibility_response'].mean():.1%}."
        )
    )
    text = [
        "# Blinded radiologist reader-study results",
        "",
        "## Perturbation visibility",
        "",
        _markdown_table(
            visibility,
            [
                ("cohort", "Cohort"),
                ("attack_label", "Attack"),
                ("target_rms_percent_case_sigma", "RMS target (% SD)"),
                ("n_ratings", "Ratings"),
                ("n_readers", "Readers"),
                ("detection_accuracy", "Detection accuracy"),
                ("difference_visible_fraction", "Reported visible"),
                ("mean_confidence", "Mean confidence"),
            ],
            {
                "target_rms_percent_case_sigma": ".1f",
                "detection_accuracy": ".1%",
                "difference_visible_fraction": ".1%",
                "mean_confidence": ".2f",
            },
        ),
        "",
        f"![Visibility]({figures[0].name})",
        "",
        "## Diagnostic and contouring adequacy",
        "",
        _markdown_table(
            adequacy,
            [
                ("cohort", "Cohort"),
                ("attack_label", "Condition"),
                ("target_rms_percent_case_sigma", "RMS target (% SD)"),
                ("n_ratings", "Ratings"),
                ("adequate_fraction", "Adequate fraction"),
                ("capsule_visibility_mean", "Mean capsule score"),
                ("overall_quality_mean", "Mean overall quality"),
            ],
            {
                "target_rms_percent_case_sigma": ".1f",
                "adequate_fraction": ".1%",
                "capsule_visibility_mean": ".2f",
                "overall_quality_mean": ".2f",
            },
        ),
        "",
        f"![Adequacy]({figures[1].name})",
        "",
        "## Repeat agreement",
        "",
        repeat_text,
        "",
        "## Interpretation limits",
        "",
        "- Detection above chance establishes reader visibility under this "
        "specific paired display task; it does not by itself establish loss of "
        "diagnostic adequacy.",
        "- Adequacy ratings concern the exported model-space T2 volumes and the "
        "specified contour-correction task, not a complete multiparametric "
        "PI-QUAL examination.",
        "- Pooled confidence intervals do not replace a pre-specified multi-reader "
        "multi-case inferential model when making confirmatory clinical claims.",
        "- Patient harm requires downstream clinical decision and outcome evidence.",
        "",
    ]
    path = output_dir / "README.md"
    path.write_text("\n".join(text), encoding="utf-8")
    return path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Analyze completed blinded radiologist attack-study ratings"
    )
    parser.add_argument("--visibility-scores", nargs="+", type=Path, required=True)
    parser.add_argument("--adequacy-scores", nargs="+", type=Path, required=True)
    parser.add_argument(
        "--visibility-truth",
        type=Path,
        default=DEFAULT_STUDY_DIR / "truth" / "visibility_truth_key.csv",
    )
    parser.add_argument(
        "--adequacy-truth",
        type=Path,
        default=DEFAULT_STUDY_DIR / "truth" / "adequacy_truth_key.csv",
    )
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    output_dir = args.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    visibility = load_visibility_scores(
        [path.resolve() for path in args.visibility_scores],
        args.visibility_truth.resolve(),
    )
    adequacy = load_adequacy_scores(
        [path.resolve() for path in args.adequacy_scores],
        args.adequacy_truth.resolve(),
    )
    visibility_summary = summarize_visibility(visibility)
    visibility_reader = visibility_reader_summary(visibility)
    repeat_summary = visibility_repeat_agreement(visibility)
    adequacy_summary = summarize_adequacy(adequacy)
    adequacy_reader = adequacy_reader_summary(adequacy)
    visibility_summary.to_csv(output_dir / "visibility_summary.csv", index=False)
    visibility_reader.to_csv(output_dir / "visibility_per_reader.csv", index=False)
    repeat_summary.to_csv(output_dir / "visibility_repeat_agreement.csv", index=False)
    adequacy_summary.to_csv(output_dir / "adequacy_summary.csv", index=False)
    adequacy_reader.to_csv(output_dir / "adequacy_per_reader.csv", index=False)
    figures = [
        plot_visibility(visibility_summary, output_dir),
        plot_adequacy(adequacy_summary, output_dir),
    ]
    report = write_report(
        output_dir,
        visibility_summary,
        adequacy_summary,
        repeat_summary,
        figures,
    )
    print(
        f"Analyzed {len(visibility):,} visibility and {len(adequacy):,} "
        "adequacy ratings.",
        flush=True,
    )
    for path in [*figures, report]:
        print(f"Saved {path}", flush=True)


if __name__ == "__main__":
    main()
