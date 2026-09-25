"""Size-insensitive confirmation of the anatomical base effect using ASD.

Reviewer 2 noted that the observed correlation between foreground area and Dice
drop may partly arise from the intrinsic size sensitivity of Dice rather than
greater adversarial susceptibility, and asked for the slice-level correlation
analysis to be clarified. Reviewer 1 asked whether the base/apex association
survives adjustment for area, clean Dice, perimeter-to-area ratio and patient
clustering. The manuscript answers both. Neither reviewer asked for a
size-insensitive metric.

This script goes beyond those requests: it repeats the regional contrasts with
the average symmetric surface distance (ASD), a boundary-distance metric in
millimetres carrying no size normalisation, under the same patient-clustered
GEE specification. Because Dice adjusted for area is still a size-sensitive
outcome, this is the check that can actually falsify the size-artefact
explanation of the base effect.

Its results appear in neither the manuscript nor the response letter, by
decision rather than oversight. Kept runnable for a later review round.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import statsmodels.api as sm
import statsmodels.formula.api as smf

REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPOSITORY_ROOT / "src"))

from mri_prostate_seg.experiments.slice_vulnerability.adjusted import (  # noqa: E402
    holm_adjust,
)

TASKS = ("wg", "zones")
CLASS_NAMES = ("WG", "TZ+CZ", "PZ")
ENDPOINT_EPSILON = 0.1
PRIMARY_PREDICTORS = [
    "z_log1p_area_mm2",
    "z_asd_clean_mm",
    "z_perimeter_area_ratio",
]
SHAPE_COLUMNS = [
    "task",
    "fold",
    "case_id",
    "slice_idx",
    "class",
    "epsilon",
    "patient_id",
    "row_spacing_mm",
    "column_spacing_mm",
    "log1p_area_mm2",
    "perimeter_area_ratio_mm_inv",
]


def load_asd_slice_table(asd_dir: Path) -> pd.DataFrame:
    """Pool the per-fold ASD-enabled slice tables for both tasks."""
    frames: list[pd.DataFrame] = []
    for task in TASKS:
        paths = sorted(asd_dir.glob(f"{task}_fold*/{task}_fold*_per_slice_fgsm.csv"))
        if not paths:
            raise FileNotFoundError(f"no per-slice ASD tables for task {task!r}")
        for path in paths:
            frame = pd.read_csv(path)
            frame["task"] = task
            frames.append(frame)
    pooled = pd.concat(frames, ignore_index=True)
    pooled["patient_id"] = (
        pooled["task"].astype(str)
        + "|"
        + pooled["fold"].astype(str)
        + "|"
        + pooled["case_id"].astype(str)
    )
    return pooled


def merge_shape_covariates(
    asd_frame: pd.DataFrame,
    enriched_path: Path,
) -> pd.DataFrame:
    """Attach the ground-truth shape covariates used by the Dice models."""
    shape = pd.read_csv(enriched_path, usecols=SHAPE_COLUMNS)
    merged = asd_frame.merge(
        shape,
        on=["task", "fold", "case_id", "slice_idx", "class", "epsilon"],
        how="inner",
        suffixes=("", "_shape"),
    )
    if "patient_id_shape" in merged.columns:
        mismatch = int((merged["patient_id"] != merged["patient_id_shape"]).sum())
        if mismatch:
            raise RuntimeError(f"patient_id mismatch on {mismatch} merged rows")
        merged = merged.drop(columns=["patient_id_shape"])
    return merged


def prepare_asd_model_frame(
    merged: pd.DataFrame,
    *,
    epsilon: float = ENDPOINT_EPSILON,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Convert ASD to millimetres and build the endpoint modelling frame.

    Returns the analysable rows and a per-class table recording how many
    endpoint slices were dropped because ASD is undefined (empty ground truth
    or empty prediction).
    """
    endpoint = merged[np.isclose(merged["epsilon"], epsilon)].copy()

    spacing = endpoint[["row_spacing_mm", "column_spacing_mm"]].to_numpy(dtype=float)
    if not np.allclose(spacing[:, 0], spacing[:, 1]):
        raise RuntimeError("in-plane spacing is anisotropic; per-axis ASD required")
    millimetres_per_pixel = spacing[:, 0]
    for column in ("asd_clean", "asd_adv", "asd_change"):
        endpoint[f"{column}_mm"] = (
            endpoint[column].to_numpy(dtype=float) * millimetres_per_pixel
        )

    usable = np.isfinite(endpoint["asd_change_mm"]) & np.isfinite(
        endpoint["asd_clean_mm"]
    )
    exclusions = (
        endpoint.assign(dropped=~usable)
        .groupby("class", observed=True)
        .agg(
            n_endpoint_slices=("dropped", "size"),
            n_dropped_undefined_asd=("dropped", "sum"),
        )
        .reset_index()
    )
    exclusions["dropped_fraction"] = (
        exclusions["n_dropped_undefined_asd"] / exclusions["n_endpoint_slices"]
    )
    return endpoint[usable].copy(), exclusions


def _standardize_within_class(frame: pd.DataFrame) -> pd.DataFrame:
    standardized = frame.copy()
    sources = {
        "z_log1p_area_mm2": "log1p_area_mm2",
        "z_asd_clean_mm": "asd_clean_mm",
        "z_perimeter_area_ratio": "perimeter_area_ratio_mm_inv",
    }
    for target, source in sources.items():
        values = standardized[source].to_numpy(dtype=float)
        standard_deviation = float(values.std(ddof=0))
        if not np.isfinite(standard_deviation) or standard_deviation <= 0.0:
            raise RuntimeError(f"cannot standardize {source}")
        standardized[target] = (values - float(values.mean())) / standard_deviation
    standardized["anatomical_region"] = pd.Categorical(
        standardized["anatomical_region"],
        categories=["Mid", "Base", "Apex"],
    )
    return standardized


def _term_label(term: str) -> str:
    if term.endswith("[T.Base]"):
        return "Base_vs_Mid"
    if term.endswith("[T.Apex]"):
        return "Apex_vs_Mid"
    return term


def fit_asd_gee(frame: pd.DataFrame, *, class_name: str) -> list[dict[str, object]]:
    """Fit the patient-clustered Gaussian GEE for ASD change in millimetres."""
    formula = (
        'asd_change_mm ~ C(anatomical_region, Treatment(reference="Mid")) + '
        + " + ".join(PRIMARY_PREDICTORS)
    )
    model = smf.gee(
        formula,
        groups="patient_id",
        data=frame,
        family=sm.families.Gaussian(),
        cov_struct=sm.cov_struct.Exchangeable(),
    )
    fit = model.fit(maxiter=200)
    confidence = fit.conf_int()
    rows: list[dict[str, object]] = []
    for term in fit.params.index:
        rows.append(
            {
                "outcome": "asd_change_mm",
                "model_spec": "primary_area_clean_asd_perimeter_area",
                "class": class_name,
                "attack": "FGSM",
                "epsilon": ENDPOINT_EPSILON,
                "term": _term_label(str(term)),
                "coefficient_mm": float(fit.params[term]),
                "standard_error": float(fit.bse[term]),
                "ci95_low": float(confidence.loc[term, 0]),
                "ci95_high": float(confidence.loc[term, 1]),
                "p_value": float(fit.pvalues[term]),
                "holm_adjusted_p_value": float("nan"),
                "n_patients": int(frame["patient_id"].nunique()),
                "n_slices": int(len(frame)),
                "working_correlation_estimate": float(
                    np.asarray(fit.cov_struct.dep_params).reshape(-1)[0]
                ),
                "converged": bool(fit.converged),
            }
        )
    return rows


def fit_all_classes(model_frame: pd.DataFrame) -> pd.DataFrame:
    """Fit every foreground class and Holm-adjust the regional contrasts."""
    rows: list[dict[str, object]] = []
    for class_name in CLASS_NAMES:
        class_frame = model_frame[model_frame["class"] == class_name]
        if class_frame.empty:
            raise RuntimeError(f"no endpoint rows for class {class_name!r}")
        rows.extend(
            fit_asd_gee(_standardize_within_class(class_frame), class_name=class_name)
        )

    result = pd.DataFrame(rows)
    contrasts = result["term"].isin(["Base_vs_Mid", "Apex_vs_Mid"])
    result.loc[contrasts, "holm_adjusted_p_value"] = holm_adjust(
        result.loc[contrasts, "p_value"].to_numpy(dtype=float)
    )
    return result


def regional_means(model_frame: pd.DataFrame) -> pd.DataFrame:
    """Unadjusted patient-then-region mean ASD change, for context."""
    per_patient = (
        model_frame.groupby(
            ["class", "anatomical_region", "patient_id"], observed=True
        )["asd_change_mm"]
        .mean()
        .reset_index()
    )
    return (
        per_patient.groupby(["class", "anatomical_region"], observed=True)[
            "asd_change_mm"
        ]
        .agg(mean_asd_change_mm="mean", n_patients="size")
        .reset_index()
    )


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
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=REPOSITORY_ROOT / "results" / "slice_anatomy_asd",
    )
    arguments = parser.parse_args()
    arguments.output_dir.mkdir(parents=True, exist_ok=True)

    print("Loading ASD slice tables")
    asd_frame = load_asd_slice_table(arguments.asd_dir)
    print(f"  pooled rows: {len(asd_frame):,}")

    print("Merging ground-truth shape covariates")
    merged = merge_shape_covariates(asd_frame, arguments.enriched_csv)
    print(f"  merged rows: {len(merged):,}")

    model_frame, exclusions = prepare_asd_model_frame(merged)
    print(
        f"\nEndpoint slices analysable at eps={ENDPOINT_EPSILON}: {len(model_frame):,}"
    )
    print(exclusions.to_string(index=False))

    print("\nFitting patient-clustered GEE models on ASD change (mm)")
    models = fit_all_classes(model_frame)
    means = regional_means(model_frame)

    models_path = arguments.output_dir / "asd_adjusted_anatomical_models.csv"
    means_path = arguments.output_dir / "asd_regional_means.csv"
    exclusions_path = arguments.output_dir / "asd_endpoint_exclusions.csv"
    models.to_csv(models_path, index=False)
    means.to_csv(means_path, index=False)
    exclusions.to_csv(exclusions_path, index=False)

    print("\nAdjusted regional contrasts (mm of ASD change):")
    contrasts = models[models["term"].isin(["Base_vs_Mid", "Apex_vs_Mid"])]
    print(
        contrasts[
            [
                "class",
                "term",
                "coefficient_mm",
                "ci95_low",
                "ci95_high",
                "holm_adjusted_p_value",
                "n_slices",
                "n_patients",
            ]
        ].to_string(index=False)
    )
    print("\nUnadjusted regional means:")
    print(means.to_string(index=False))
    print(f"\nWrote {models_path}")
    print(f"Wrote {means_path}")
    print(f"Wrote {exclusions_path}")


if __name__ == "__main__":
    main()
