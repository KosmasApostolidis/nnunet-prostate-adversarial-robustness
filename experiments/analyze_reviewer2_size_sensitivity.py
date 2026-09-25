"""Reviewer 2 analysis of Dice size sensitivity and adjusted anatomy effects.

This script enriches the corrected five-fold OOF slice table with ground-truth
shape covariates, performs patient-aware paired regional contrasts, fits a
patient-clustered GEE at FGSM epsilon 0.1, and regenerates Figure 2 with an
adjusted-effect panel.
"""

from __future__ import annotations

import argparse
import pickle
import shutil
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import scipy.stats as scipy_stats
import statsmodels.api as sm
import statsmodels.formula.api as smf
from statsmodels.stats.outliers_influence import variance_inflation_factor

REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPOSITORY_ROOT / "src"))

from mri_prostate_seg.attacks.config_loader import load_model_config  # noqa: E402
from mri_prostate_seg.experiments.slice_vulnerability.adjusted import (  # noqa: E402
    holm_adjust,
    patient_bootstrap_mean_ci,
    shape_covariates,
)
from mri_prostate_seg.experiments.slice_vulnerability.figures import (  # noqa: E402
    add_combined_wg_tz_pz_axes,
    plot_fg_size_vs_vulnerability_on_ax,
)


CLASS_IDS = {"WG": 1, "TZ+CZ": 1, "PZ": 2}
CLASS_COLORS = {"WG": "#1976D2", "TZ+CZ": "#D32F2F", "PZ": "#388E3C"}
PRIMARY_PREDICTORS = [
    "z_log1p_area_mm2",
    "z_dice_clean",
    "z_perimeter_area_ratio",
]
SENSITIVITY_PREDICTORS = [
    "z_log1p_area_mm2",
    "z_dice_clean",
    "z_compactness",
]


def _load_segmentation_and_spacing(
    data_dir: str,
    case_id: str,
) -> tuple[np.ndarray, float, float]:
    """Load only the preprocessed label and its in-plane physical spacing."""
    base = Path(data_dir)
    b2nd_path = base / f"{case_id}_seg.b2nd"
    npy_path = base / f"{case_id}_seg.npy"
    npz_path = base / f"{case_id}.npz"
    properties_path = base / f"{case_id}.pkl"

    if b2nd_path.is_file():
        import blosc2

        segmentation = np.asarray(blosc2.open(urlpath=str(b2nd_path), mode="r")[:])
    elif npy_path.is_file():
        segmentation = np.load(npy_path)
    elif npz_path.is_file():
        with np.load(npz_path) as archive:
            segmentation = np.asarray(archive["seg"])
    else:
        raise FileNotFoundError(f"no preprocessed segmentation for {case_id}")

    with properties_path.open("rb") as file:
        properties = pickle.load(file)
    spacing = np.asarray(properties["spacing"], dtype=float)
    if spacing.size != 3 or np.any(~np.isfinite(spacing)) or np.any(spacing <= 0.0):
        raise ValueError(f"invalid preprocessed spacing for {case_id}: {spacing}")

    ground_truth = np.asarray(segmentation, dtype=np.int64).squeeze()
    if ground_truth.ndim != 3:
        raise ValueError(
            f"expected a 3-D preprocessed label for {case_id}, got {ground_truth.shape}"
        )
    return ground_truth, float(spacing[-2]), float(spacing[-1])


def _read_corrected_slice_tables(input_dir: Path) -> pd.DataFrame:
    frames: list[pd.DataFrame] = []
    for task in ("wg", "zones"):
        path = input_dir / f"{task}_all_folds_per_slice_fgsm.csv"
        frame = pd.read_csv(path)
        frame.insert(0, "task", task)
        frames.append(frame)
    combined = pd.concat(frames, ignore_index=True)
    combined["patient_id"] = (
        combined["task"].astype(str)
        + "|"
        + combined["fold"].astype(str)
        + "|"
        + combined["case_id"].astype(str)
    )
    return combined


def _build_shape_table(slice_frame: pd.DataFrame) -> pd.DataFrame:
    key_columns = ["task", "fold", "case_id", "slice_idx", "class"]
    invariants = ["gt_area", "dice_clean", "anatomical_region"]
    for column in invariants:
        counts = slice_frame.groupby(key_columns, sort=False)[column].nunique(dropna=False)
        if int(counts.max()) != 1:
            raise RuntimeError(f"{column} changes across epsilon for a slice")

    unique_slices = slice_frame[
        key_columns + ["gt_area", "dice_clean", "anatomical_region", "patient_id"]
    ].drop_duplicates(key_columns)
    covariate_rows: list[dict[str, object]] = []

    for task, task_rows in unique_slices.groupby("task", sort=False):
        config = load_model_config(str(task))
        cases = list(task_rows.groupby(["fold", "case_id"], sort=False))
        for index, ((fold, case_id), case_rows) in enumerate(cases, start=1):
            ground_truth, row_spacing, column_spacing = (
                _load_segmentation_and_spacing(config.data_dir, str(case_id))
            )
            for record in case_rows.to_dict("records"):
                class_name = str(record["class"])
                class_id = CLASS_IDS[class_name]
                slice_index = int(record["slice_idx"])
                mask = ground_truth[slice_index] == class_id
                observed_pixels = int(mask.sum())
                expected_pixels = int(record["gt_area"])
                if observed_pixels != expected_pixels:
                    raise RuntimeError(
                        f"GT area mismatch for {task}/{case_id}/slice {slice_index}/"
                        f"{class_name}: {observed_pixels} != {expected_pixels}"
                    )
                covariates = shape_covariates(
                    mask,
                    row_spacing_mm=row_spacing,
                    column_spacing_mm=column_spacing,
                )
                covariate_rows.append(
                    {
                        "task": task,
                        "fold": int(fold),
                        "case_id": str(case_id),
                        "slice_idx": slice_index,
                        "class": class_name,
                        "row_spacing_mm": row_spacing,
                        "column_spacing_mm": column_spacing,
                        **covariates,
                    }
                )
            if index % 100 == 0 or index == len(cases):
                print(f"  {task}: shape covariates {index}/{len(cases)} cases")

    shape_frame = pd.DataFrame(covariate_rows)
    if shape_frame.duplicated(key_columns).any():
        raise RuntimeError("duplicate slice keys in shape covariates")
    return shape_frame


def _enrich_slice_table(slice_frame: pd.DataFrame, shape_frame: pd.DataFrame) -> pd.DataFrame:
    keys = ["task", "fold", "case_id", "slice_idx", "class"]
    enriched = slice_frame.merge(
        shape_frame,
        on=keys,
        how="left",
        validate="many_to_one",
    )
    shape_columns = [
        "gt_area_mm2",
        "gt_perimeter_mm",
        "perimeter_area_ratio_mm_inv",
        "compactness",
    ]
    if enriched[shape_columns].isna().any().any():
        missing = enriched.loc[enriched[shape_columns].isna().any(axis=1), keys]
        raise RuntimeError(f"missing shape covariates for {len(missing)} rows")
    enriched["log1p_area_mm2"] = np.log1p(enriched["gt_area_mm2"])
    return enriched


def _shape_summaries(enriched: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    keys = ["task", "fold", "case_id", "slice_idx", "class"]
    unique_slices = enriched.drop_duplicates(keys)
    variables = [
        "gt_area_mm2",
        "gt_perimeter_mm",
        "perimeter_area_ratio_mm_inv",
        "compactness",
        "dice_clean",
    ]
    summary_rows: list[dict[str, object]] = []
    correlation_rows: list[dict[str, object]] = []

    for class_name, class_frame in unique_slices.groupby("class", sort=False):
        row: dict[str, object] = {
            "class": class_name,
            "n_patients": int(class_frame["patient_id"].nunique()),
            "n_slices": int(len(class_frame)),
        }
        for variable in variables:
            values = class_frame[variable].to_numpy(dtype=float)
            row[f"{variable}_median"] = float(np.median(values))
            row[f"{variable}_q1"] = float(np.percentile(values, 25))
            row[f"{variable}_q3"] = float(np.percentile(values, 75))
        summary_rows.append(row)

        correlation_variables = [
            "log1p_area_mm2",
            "dice_clean",
            "perimeter_area_ratio_mm_inv",
            "compactness",
        ]
        for left_index, left in enumerate(correlation_variables):
            for right in correlation_variables[left_index + 1 :]:
                rho, p_value = scipy_stats.spearmanr(
                    class_frame[left].to_numpy(dtype=float),
                    class_frame[right].to_numpy(dtype=float),
                )
                correlation_rows.append(
                    {
                        "class": class_name,
                        "predictor_1": left,
                        "predictor_2": right,
                        "spearman_rho": float(rho),
                        "naive_slice_p_value": float(p_value),
                        "n_slices": int(len(class_frame)),
                    }
                )
    return pd.DataFrame(summary_rows), pd.DataFrame(correlation_rows)


def _vif_table(enriched: pd.DataFrame) -> pd.DataFrame:
    endpoint = enriched[np.isclose(enriched["epsilon"], 0.1)].copy()
    rows: list[dict[str, object]] = []
    raw_predictors = [
        "log1p_area_mm2",
        "dice_clean",
        "perimeter_area_ratio_mm_inv",
    ]
    for class_name, frame in endpoint.groupby("class", sort=False):
        design = sm.add_constant(frame[raw_predictors], has_constant="add")
        for index, predictor in enumerate(design.columns):
            rows.append(
                {
                    "class": class_name,
                    "predictor": predictor,
                    "vif": float(
                        variance_inflation_factor(
                            design.to_numpy(dtype=float),
                            index,
                        )
                    ),
                }
            )
    return pd.DataFrame(rows)


def _paired_region_statistics(
    enriched: pd.DataFrame,
    *,
    n_resamples: int,
    seed: int,
) -> pd.DataFrame:
    endpoint = enriched[np.isclose(enriched["epsilon"], 0.1)].copy()
    regional = (
        endpoint.groupby(
            ["class", "patient_id", "anatomical_region"],
            observed=True,
            sort=False,
        )["dice_drop"]
        .mean()
        .unstack("anatomical_region")
    )
    rows: list[dict[str, object]] = []
    comparison_index = 0
    for class_name in ("WG", "TZ+CZ", "PZ"):
        class_table = regional.loc[class_name]
        for region in ("Base", "Apex"):
            paired = class_table[[region, "Mid"]].dropna()
            differences = (paired[region] - paired["Mid"]).to_numpy(dtype=float)
            low, high = patient_bootstrap_mean_ci(
                differences,
                n_resamples=n_resamples,
                seed=seed + comparison_index,
            )
            comparison_index += 1
            if differences.size and np.any(np.abs(differences) > 0.0):
                p_value = float(
                    scipy_stats.wilcoxon(
                        differences,
                        alternative="two-sided",
                        zero_method="wilcox",
                    ).pvalue
                )
            else:
                p_value = float("nan")
            rows.append(
                {
                    "class": class_name,
                    "attack": "FGSM",
                    "epsilon": 0.1,
                    "contrast": f"{region}-Mid",
                    "n_paired_patients": int(differences.size),
                    "mean_paired_difference": float(np.mean(differences)),
                    "median_paired_difference": float(np.median(differences)),
                    "patient_bootstrap_ci95_low": low,
                    "patient_bootstrap_ci95_high": high,
                    "paired_wilcoxon_p_value": p_value,
                }
            )
    result = pd.DataFrame(rows)
    result["holm_adjusted_p_value"] = holm_adjust(
        result["paired_wilcoxon_p_value"].to_numpy(dtype=float)
    )
    return result


def _standardize_within_class(frame: pd.DataFrame) -> pd.DataFrame:
    standardized = frame.copy()
    sources = {
        "z_log1p_area_mm2": "log1p_area_mm2",
        "z_dice_clean": "dice_clean",
        "z_perimeter_area_ratio": "perimeter_area_ratio_mm_inv",
        "z_compactness": "compactness",
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


def _fit_one_gee(
    frame: pd.DataFrame,
    *,
    class_name: str,
    model_spec: str,
    predictors: list[str],
) -> list[dict[str, object]]:
    formula = (
        'dice_drop ~ C(anatomical_region, Treatment(reference="Mid")) + '
        + " + ".join(predictors)
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
                "model_spec": model_spec,
                "class": class_name,
                "attack": "FGSM",
                "epsilon": 0.1,
                "term": _term_label(str(term)),
                "coefficient": float(fit.params[term]),
                "standard_error": float(fit.bse[term]),
                "ci95_low": float(confidence.loc[term, 0]),
                "ci95_high": float(confidence.loc[term, 1]),
                "p_value": float(fit.pvalues[term]),
                "holm_adjusted_p_value": float("nan"),
                "n_patients": int(frame["patient_id"].nunique()),
                "n_slices": int(len(frame)),
                "working_correlation": "exchangeable",
                "working_correlation_estimate": float(
                    np.asarray(fit.cov_struct.dep_params).reshape(-1)[0]
                ),
                "converged": bool(fit.converged),
            }
        )
    return rows


def _fit_adjusted_models(enriched: pd.DataFrame) -> pd.DataFrame:
    endpoint = enriched[np.isclose(enriched["epsilon"], 0.1)].copy()
    rows: list[dict[str, object]] = []
    for class_name in ("WG", "TZ+CZ", "PZ"):
        class_frame = endpoint[endpoint["class"] == class_name].copy()
        class_frame = _standardize_within_class(class_frame)
        rows.extend(
            _fit_one_gee(
                class_frame,
                class_name=class_name,
                model_spec="primary_area_clean_dice_perimeter_area",
                predictors=PRIMARY_PREDICTORS,
            )
        )
        rows.extend(
            _fit_one_gee(
                class_frame,
                class_name=class_name,
                model_spec="sensitivity_area_clean_dice_compactness",
                predictors=SENSITIVITY_PREDICTORS,
            )
        )

    result = pd.DataFrame(rows)
    for model_spec in result["model_spec"].unique():
        mask = (result["model_spec"] == model_spec) & result["term"].isin(
            ["Base_vs_Mid", "Apex_vs_Mid"]
        )
        result.loc[mask, "holm_adjusted_p_value"] = holm_adjust(
            result.loc[mask, "p_value"].to_numpy(dtype=float)
        )
    return result


def _plot_adjusted_effects(ax: plt.Axes, model_rows: pd.DataFrame) -> None:
    primary = model_rows[
        (model_rows["model_spec"] == "primary_area_clean_dice_perimeter_area")
        & model_rows["term"].isin(["Base_vs_Mid", "Apex_vs_Mid"])
    ].copy()
    order = [
        ("WG", "Base_vs_Mid"),
        ("WG", "Apex_vs_Mid"),
        ("TZ+CZ", "Base_vs_Mid"),
        ("TZ+CZ", "Apex_vs_Mid"),
        ("PZ", "Base_vs_Mid"),
        ("PZ", "Apex_vs_Mid"),
    ]
    labels: list[str] = []
    for y_position, (class_name, term) in enumerate(reversed(order)):
        row = primary[
            (primary["class"] == class_name) & (primary["term"] == term)
        ].iloc[0]
        coefficient = float(row["coefficient"])
        low = float(row["ci95_low"])
        high = float(row["ci95_high"])
        ax.errorbar(
            coefficient,
            y_position,
            xerr=[[coefficient - low], [high - coefficient]],
            fmt="o",
            color=CLASS_COLORS[class_name],
            capsize=4,
            markersize=6,
        )
        labels.append(f"{class_name}: {term.replace('_vs_', '–')}")
    ax.axvline(0.0, color="black", linewidth=1.0, linestyle="--")
    ax.set_yticks(range(len(labels)))
    ax.set_yticklabels(labels, fontsize=8.5)
    # Reserve a footer band below the lowest effect so the explanatory note is
    # visible without covering a point, interval, tick label, or axis title.
    ax.set_ylim(-1.35, len(labels) - 0.45)
    ax.set_xlabel("Adjusted Dice-drop difference vs Mid", fontsize=10, labelpad=7)
    ax.set_title(
        "Patient-clustered GEE at ε=0.10\n"
        "Adjusted for area, clean Dice, and P/A",
        fontsize=11,
        fontweight="bold",
        pad=9,
    )
    ax.grid(True, axis="x", alpha=0.25)
    ax.tick_params(axis="x", labelsize=9, pad=3)
    ax.tick_params(axis="y", pad=4)
    ax.text(
        0.98,
        0.035,
        "Positive = greater drop than mid-gland;\n95% robust CI",
        transform=ax.transAxes,
        fontsize=9,
        ha="right",
        va="bottom",
        bbox={
            "boxstyle": "round,pad=0.25",
            "facecolor": "white",
            "edgecolor": "0.75",
            "alpha": 0.95,
            "linewidth": 0.7,
        },
        zorder=10,
    )


def _regenerate_figure_2(
    enriched: pd.DataFrame,
    model_rows: pd.DataFrame,
    output_dir: Path,
    paper_figures_dir: Path | None,
) -> None:
    endpoint = enriched[np.isclose(enriched["epsilon"], 0.1)]
    wg_records = endpoint[endpoint["task"] == "wg"].to_dict("records")
    zones_records = endpoint[endpoint["task"] == "zones"].to_dict("records")

    fig = plt.figure(figsize=(14, 10), layout="constrained")
    ax_wg, ax_tz, ax_pz, ax_adjusted = add_combined_wg_tz_pz_axes(
        fig,
        hspace=0.14,
        wspace=0.16,
        bottom_inner_wspace=0.3,
    )
    plot_fg_size_vs_vulnerability_on_ax(
        ax_wg,
        wg_records,
        "WG",
        0.1,
        "Whole Gland",
        CLASS_COLORS["WG"],
    )
    plot_fg_size_vs_vulnerability_on_ax(
        ax_tz,
        zones_records,
        "TZ+CZ",
        0.1,
        "Prostate Zones",
        CLASS_COLORS["TZ+CZ"],
    )
    plot_fg_size_vs_vulnerability_on_ax(
        ax_pz,
        zones_records,
        "PZ",
        0.1,
        "Prostate Zones",
        CLASS_COLORS["PZ"],
    )
    _plot_adjusted_effects(ax_adjusted, model_rows)
    fig.suptitle(
        "Foreground Area, Dice Drop, and Geometry-Adjusted Anatomy under FGSM",
        fontsize=14,
        fontweight="bold",
    )
    png_path = output_dir / "combined_wg_zones_fg_size_vs_vulnerability.png"
    pdf_path = output_dir / "combined_wg_zones_fg_size_vs_vulnerability.pdf"
    fig.savefig(png_path, dpi=600, bbox_inches="tight", pad_inches=0.08)
    fig.savefig(pdf_path, bbox_inches="tight", pad_inches=0.08)
    plt.close(fig)
    print(f"  Saved {png_path}")
    print(f"  Saved {pdf_path}")

    if paper_figures_dir is not None:
        paper_figures_dir.mkdir(parents=True, exist_ok=True)
        shutil.copy2(png_path, paper_figures_dir / png_path.name)
        shutil.copy2(pdf_path, paper_figures_dir / pdf_path.name)
        print(f"  Installed Figure 2 in {paper_figures_dir}")


def _write_report(
    output_dir: Path,
    models: pd.DataFrame,
    paired: pd.DataFrame,
    vif: pd.DataFrame,
) -> None:
    primary = models[
        models["model_spec"] == "primary_area_clean_dice_perimeter_area"
    ]
    path = output_dir / "reviewer_2_size_sensitivity_report.md"
    with path.open("w") as file:
        file.write("# Reviewer 2: Dice size-sensitivity analysis\n\n")
        file.write(
            "Analysis was restricted to the corrected, physically oriented five-fold "
            "OOF FGSM slice dataset. Ground-truth geometry was calculated in physical "
            "units with a grid-edge perimeter estimator. The primary endpoint model "
            "was a Gaussian GEE with exchangeable within-patient working correlation "
            "at epsilon=0.10. It adjusted for standardized log(1+area in mm^2), clean "
            "slice Dice, and perimeter/area.\n\n"
        )
        file.write("## Primary adjusted region effects\n\n")
        file.write("| Class | Contrast | Coefficient | 95% CI | Holm p |\n")
        file.write("|---|---:|---:|---:|---:|\n")
        for class_name in ("WG", "TZ+CZ", "PZ"):
            for term in ("Base_vs_Mid", "Apex_vs_Mid"):
                row = primary[
                    (primary["class"] == class_name) & (primary["term"] == term)
                ].iloc[0]
                file.write(
                    f"| {class_name} | {term.replace('_vs_', '–')} | "
                    f"{row['coefficient']:.4f} | "
                    f"[{row['ci95_low']:.4f}, {row['ci95_high']:.4f}] | "
                    f"{row['holm_adjusted_p_value']:.3g} |\n"
                )
        file.write("\n## Adjusted foreground-area terms\n\n")
        file.write("| Class | Coefficient per 1 SD log-area | 95% CI | p |\n")
        file.write("|---|---:|---:|---:|\n")
        for class_name in ("WG", "TZ+CZ", "PZ"):
            row = primary[
                (primary["class"] == class_name)
                & (primary["term"] == "z_log1p_area_mm2")
            ].iloc[0]
            file.write(
                f"| {class_name} | {row['coefficient']:.4f} | "
                f"[{row['ci95_low']:.4f}, {row['ci95_high']:.4f}] | "
                f"{row['p_value']:.3g} |\n"
            )
        file.write("\n## Paired patient-level regional contrasts\n\n")
        file.write("| Class | Contrast | n | Mean | 95% bootstrap CI | Holm p |\n")
        file.write("|---|---:|---:|---:|---:|---:|\n")
        for row in paired.to_dict("records"):
            file.write(
                f"| {row['class']} | {row['contrast']} | "
                f"{row['n_paired_patients']} | "
                f"{row['mean_paired_difference']:.4f} | "
                f"[{row['patient_bootstrap_ci95_low']:.4f}, "
                f"{row['patient_bootstrap_ci95_high']:.4f}] | "
                f"{row['holm_adjusted_p_value']:.3g} |\n"
            )
        file.write("\n## Collinearity diagnostic\n\n")
        maximum_vif = vif[vif["predictor"] != "const"]["vif"].max()
        file.write(f"Maximum primary-predictor VIF across classes: {maximum_vif:.3f}.\n\n")
        file.write(
            "## Interpretation boundary\n\n"
            "The negative foreground-area association is retained as a descriptive "
            "property of Dice drop. Because Dice is intrinsically more sensitive to "
            "the same absolute boundary error for a small object, neither the raw "
            "Spearman correlation nor the adjusted log-area coefficient establishes "
            "greater attack susceptibility. The adjusted region coefficients answer "
            "the narrower question of whether base/apex location remains associated "
            "with Dice drop conditional on measured size, clean Dice, shape complexity, "
            "and within-patient clustering.\n"
        )
    print(f"  Saved {path}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--input-dir",
        type=Path,
        default=REPOSITORY_ROOT / "rebuttal_paper2" / "slice_anatomy_corrected",
    )
    parser.add_argument("--output-dir", type=Path, default=None)
    parser.add_argument(
        "--paper-figures-dir",
        type=Path,
        default=REPOSITORY_ROOT / "paper" / "figures",
    )
    parser.add_argument("--bootstrap-resamples", type=int, default=10_000)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--reuse-augmented",
        action="store_true",
        help="Reuse all_folds_per_slice_with_shape.csv instead of reloading masks",
    )
    args = parser.parse_args()

    output_dir = args.output_dir or args.input_dir
    output_dir.mkdir(parents=True, exist_ok=True)
    augmented_path = output_dir / "all_folds_per_slice_with_shape.csv"

    if args.reuse_augmented:
        enriched = pd.read_csv(augmented_path)
    else:
        print("Loading corrected OOF slice tables")
        slice_frame = _read_corrected_slice_tables(args.input_dir)
        print("Calculating ground-truth shape covariates")
        shape_frame = _build_shape_table(slice_frame)
        enriched = _enrich_slice_table(slice_frame, shape_frame)
        enriched.to_csv(augmented_path, index=False, float_format="%.8f")
        print(f"  Saved {augmented_path}")

    shape_summary, correlations = _shape_summaries(enriched)
    shape_summary.to_csv(output_dir / "shape_covariate_summary.csv", index=False)
    correlations.to_csv(output_dir / "shape_predictor_correlations.csv", index=False)
    vif = _vif_table(enriched)
    vif.to_csv(output_dir / "shape_predictor_vif.csv", index=False)

    print("Calculating paired patient-level regional contrasts")
    paired = _paired_region_statistics(
        enriched,
        n_resamples=args.bootstrap_resamples,
        seed=args.seed,
    )
    paired.to_csv(output_dir / "patient_region_statistics.csv", index=False)

    print("Fitting patient-clustered GEE models")
    models = _fit_adjusted_models(enriched)
    models.to_csv(output_dir / "adjusted_anatomical_models.csv", index=False)

    _regenerate_figure_2(
        enriched,
        models,
        output_dir,
        args.paper_figures_dir,
    )
    _write_report(output_dir, models, paired, vif)
    print("Done.")


if __name__ == "__main__":
    main()
