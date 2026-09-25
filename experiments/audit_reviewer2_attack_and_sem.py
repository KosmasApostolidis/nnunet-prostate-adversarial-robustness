"""Reconstruct Reviewer 2 attack details and the original Table I SEM unit."""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import numpy as np
import pandas as pd


REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
EPSILONS = [0.0, 0.02, 0.04, 0.06, 0.08, 0.1]
ATTACKS = ["fgsm", "pgd", "a_pgd"]
ATTACK_LABELS = {"fgsm": "FGSM", "pgd": "PGD", "a_pgd": "Adaptive PGD-style"}
TASK_CLASSES = {"wg": ["WG"], "zones": ["TZ+CZ", "PZ"]}
METRICS = ["dice", "hd95", "asd"]


def _fold_metric_matrix(
    results_dir: Path,
    *,
    task: str,
    attack: str,
    class_name: str,
    metric: str,
) -> tuple[np.ndarray, np.ndarray]:
    means: list[np.ndarray] = []
    case_counts: list[np.ndarray] = []
    for fold in range(5):
        path = results_dir / f"{task}_{attack}_fold{fold}_{metric}.csv"
        frame = pd.read_csv(path)
        frame = frame[frame["class"] == class_name].copy()
        frame["epsilon"] = frame["epsilon"].astype(float)
        frame = frame.set_index("epsilon").reindex(EPSILONS)
        if frame["mean"].isna().any():
            raise RuntimeError(f"missing epsilon/class cell in {path}")
        means.append(frame["mean"].to_numpy(dtype=float))
        case_counts.append(frame["n"].to_numpy(dtype=int))
    return np.vstack(means), np.vstack(case_counts)


def _verify_frozen_cross_fold_summaries(results_dir: Path) -> int:
    checks = 0
    for task, classes in TASK_CLASSES.items():
        summary_folder = "WG" if task == "wg" else "Zones"
        for attack in ATTACKS:
            summary = pd.read_csv(
                results_dir / summary_folder / f"{attack}_summary.csv"
            )
            for class_name in classes:
                for metric in METRICS:
                    fold_means, _ = _fold_metric_matrix(
                        results_dir,
                        task=task,
                        attack=attack,
                        class_name=class_name,
                        metric=metric,
                    )
                    for epsilon_index, epsilon in enumerate(EPSILONS):
                        row = summary[
                            np.isclose(summary["epsilon"], epsilon)
                            & (summary["class"] == class_name)
                        ].iloc[0]
                        observed_mean = float(fold_means[:, epsilon_index].mean())
                        observed_sd = float(
                            fold_means[:, epsilon_index].std(ddof=1)
                        )
                        if not math.isclose(
                            observed_mean,
                            float(row[f"mean_{metric}"]),
                            abs_tol=1.5e-6,
                        ):
                            raise RuntimeError(
                                f"cross-fold mean mismatch: {task}/{attack}/"
                                f"{class_name}/{metric}/{epsilon}"
                            )
                        if not math.isclose(
                            observed_sd,
                            float(row[f"std_{metric}"]),
                            abs_tol=1.5e-6,
                        ):
                            raise RuntimeError(
                                f"cross-fold SD mismatch: {task}/{attack}/"
                                f"{class_name}/{metric}/{epsilon}"
                            )
                        checks += 1
    return checks


def _summary_row(
    *,
    attack: str,
    class_name: str,
    metric: str,
    summary_type: str,
    fold_values: np.ndarray,
    n_budgets_per_fold: int,
    fold_case_counts: np.ndarray,
) -> dict[str, object]:
    values = np.asarray(fold_values, dtype=float)
    fold_sd = float(values.std(ddof=1))
    return {
        "attack": attack,
        "class": class_name,
        "metric": metric,
        "summary_type": summary_type,
        "n_fold_observations": int(values.size),
        "n_budgets_averaged_within_fold": n_budgets_per_fold,
        "fold_case_counts": json.dumps([int(value) for value in fold_case_counts]),
        "fold_level_values": json.dumps(
            [round(float(value), 9) for value in values]
        ),
        "mean": float(values.mean()),
        "fold_level_sd": fold_sd,
        "sem": fold_sd / math.sqrt(values.size),
    }


def reconstruct_table_sem(results_dir: Path) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    for task, classes in TASK_CLASSES.items():
        for class_name in classes:
            for metric in METRICS:
                clean_means, clean_counts = _fold_metric_matrix(
                    results_dir,
                    task=task,
                    attack="fgsm",
                    class_name=class_name,
                    metric=metric,
                )
                rows.append(
                    _summary_row(
                        attack="Clean",
                        class_name=class_name,
                        metric=metric,
                        summary_type="clean_epsilon_0",
                        fold_values=clean_means[:, 0],
                        n_budgets_per_fold=1,
                        fold_case_counts=clean_counts[:, 0],
                    )
                )
            for attack in ATTACKS:
                for metric in METRICS:
                    fold_means, fold_counts = _fold_metric_matrix(
                        results_dir,
                        task=task,
                        attack=attack,
                        class_name=class_name,
                        metric=metric,
                    )
                    rows.append(
                        _summary_row(
                            attack=ATTACK_LABELS[attack],
                            class_name=class_name,
                            metric=metric,
                            summary_type="mean_nonzero_budgets",
                            fold_values=fold_means[:, 1:].mean(axis=1),
                            n_budgets_per_fold=5,
                            fold_case_counts=fold_counts[:, 1],
                        )
                    )
                    rows.append(
                        _summary_row(
                            attack=ATTACK_LABELS[attack],
                            class_name=class_name,
                            metric=metric,
                            summary_type="endpoint_epsilon_0.1",
                            fold_values=fold_means[:, -1],
                            n_budgets_per_fold=1,
                            fold_case_counts=fold_counts[:, -1],
                        )
                    )
    return pd.DataFrame(rows)


def _attack_audit_rows() -> list[dict[str, object]]:
    common = {
        "original_loss": (
            "valid-voxel mean softmax cross-entropy + foreground soft-Dice "
            "loss averaged across foreground classes"
        ),
        "binary_WG_handling": "two-channel mutually exclusive softmax (background/WG)",
        "multiclass_zonal_handling": (
            "three-channel mutually exclusive softmax (background/TZ+CZ/PZ)"
        ),
        "ignore_handling": "label -1 excluded from CE and Dice terms",
        "constraint": (
            "L-infinity projection about the clean preprocessed tensor, followed "
            "by per-volume clean-intensity min/max clipping"
        ),
        "seed": 42,
        "attack_space": "post-resampling, post-z-score nnU-Net input space",
        "inference": "zero-pad to stride divisibility; direct full-volume forward",
        "output_conversion": "unpad logits; voxelwise argmax",
        "postprocessing": "none",
    }
    return [
        {
            "attack": "FGSM",
            **common,
            "steps": 1,
            "step_size": "epsilon",
            "random_initialization": False,
            "runs_or_restarts": 1,
            "momentum": 0.0,
            "adaptation": "none",
            "best_iterate_tracking": False,
            "epsilon_sweep": "evaluated at each budget from the clean-input gradient",
        },
        {
            "attack": "PGD",
            **common,
            "steps": 20,
            "step_size": "max_epsilon / 20",
            "random_initialization": True,
            "runs_or_restarts": 1,
            "momentum": 0.0,
            "adaptation": "none",
            "best_iterate_tracking": False,
            "epsilon_sweep": (
                "shared max-budget trajectory with intermediate L-infinity "
                "projections for smaller budgets"
            ),
        },
        {
            "attack": "Adaptive PGD-style",
            **common,
            "steps": 20,
            "step_size": "initially 2 * epsilon / 20",
            "random_initialization": True,
            "runs_or_restarts": 1,
            "momentum": 0.9,
            "adaptation": (
                "fixed checkpoints at 25%, 50%, 75%; rate 0.1; halve/restart "
                "from best-so-far on insufficient improvement"
            ),
            "best_iterate_tracking": True,
            "epsilon_sweep": (
                "shared max-budget adaptive result projected to smaller budgets"
            ),
        },
    ]


def _write_sem_audit(
    output_dir: Path,
    reconstruction: pd.DataFrame,
    verified_cells: int,
) -> None:
    path = output_dir / "table1_original_sem_audit.md"
    with path.open("w") as file:
        file.write("# Original Table I SEM audit\n\n")
        file.write(
            f"Reconstruction verified {verified_cells} per-epsilon class/metric "
            "cross-fold mean and standard-deviation cells against the frozen "
            "campaign summaries.\n\n"
        )
        file.write("## Unit and denominator\n\n")
        file.write(
            "- The independent values entering every displayed SEM were the five "
            "fold-level validation-set means (n=5), not individual patients and "
            "not individual slices.\n"
            "- For a Table I **mean** cell, the five non-zero epsilon means were "
            "first averaged within each fold. The reported center is the mean of "
            "those five fold summaries and SEM = sample SD(fold summaries)/sqrt(5).\n"
            "- For a **worst** cell, each fold contributed its epsilon=0.10 mean; "
            "SEM = sample SD(fold endpoint means)/sqrt(5).\n"
            "- For **Clean**, each fold contributed its epsilon=0 mean and the same "
            "n=5 formula was used.\n"
            "- Patients were averaged within their validation fold before the SEM "
            "was calculated. Fold sizes were therefore equally weighted rather "
            "than pooled at the patient level. Attack strengths were not treated "
            "as independent SEM observations.\n\n"
        )
        file.write("## Cohort sizes represented inside fold means\n\n")
        for class_name in ("WG", "TZ+CZ", "PZ"):
            row = reconstruction[
                (reconstruction["attack"] == "Clean")
                & (reconstruction["class"] == class_name)
                & (reconstruction["metric"] == "dice")
            ].iloc[0]
            file.write(
                f"- {class_name}: fold case counts {row['fold_case_counts']} "
                f"(total {sum(json.loads(row['fold_case_counts']))}).\n"
            )
        file.write(
            "\nThese SEMs quantify dispersion among the five trained-fold "
            "validation means. They do not quantify variability from retraining "
            "under new random data partitions.\n"
        )
    print(f"  Saved {path}")


def _write_attack_audit(output_dir: Path, attack_frame: pd.DataFrame) -> None:
    path = output_dir / "attack_and_inference_audit.md"
    with path.open("w") as file:
        file.write("# Reviewer 2 attack and nnU-Net inference audit\n\n")
        file.write("## Exact attack objective\n\n")
        file.write(
            "The frozen campaign maximized the same scalar compound objective for "
            "FGSM, PGD, and the custom adaptive PGD-style attack: valid-voxel mean "
            "softmax cross-entropy plus foreground soft-Dice loss, with the Dice "
            "term averaged across foreground classes. WG is a mutually exclusive "
            "two-channel background/WG model, not a one-logit sigmoid model. The "
            "zonal network is a mutually exclusive three-channel model. Label -1 "
            "voxels outside the valid crop are excluded. Therefore the manuscript's "
            "former label “APGD-CE” was not an accurate description of the run.\n\n"
        )
        file.write("## Threat-model boundary and inference path\n\n")
        file.write(
            "The optimization variable is the stored nnU-Net preprocessed tensor, "
            "after resampling and z-score normalization. The attack does not "
            "differentiate through DICOM/NIfTI loading, raw-image resampling, or "
            "normalization. Each tensor and label are zero/ignore padded only to "
            "the network's stride divisibility, followed by a differentiable direct "
            "full-volume network forward. The prediction logits are unpadded and "
            "converted with voxelwise argmax. The attack/evaluation path does not "
            "use nnU-Net sliding-window Gaussian accumulation, test-time mirroring, "
            "resampling to the raw image grid, or connected-component postprocessing. "
            "Clean and adversarial predictions use this same path. The honest scope "
            "is therefore a post-preprocessing, model-input-space robustness audit, "
            "not an end-to-end attack through the deployed raw-image pipeline.\n\n"
        )
        file.write("## Constraint handling\n\n")
        file.write(
            "Every update is projected into the L-infinity ball around the clean "
            "preprocessed tensor and then clipped to that volume's clean minimum "
            "and maximum intensity. The epsilon values are consequently expressed "
            "in normalized model-input units.\n\n"
        )
        file.write("## Iterative attacks\n\n")
        file.write(
            "PGD used 20 sign-gradient steps, one configured random initialization, "
            "one run, no momentum, and a shared maximum-budget trajectory. The "
            "adaptive attack was a custom PGD-style implementation rather than "
            "canonical classification AutoAttack APGD-CE: it used 20 steps, one "
            "random initialization, momentum 0.9, initial step 2*epsilon/20, fixed "
            "25/50/75% checkpoints, rate 0.1, and best-so-far tracking. Random "
            "number generators were seeded at 42. These one-run settings can "
            "underestimate the strongest attainable attack and should not be "
            "called a certified worst case.\n\n"
        )
        file.write("## Machine-readable configuration\n\n")
        file.write(
            "See `attack_configuration_audit.csv` and "
            "`attack_configuration_audit.json` in this directory.\n"
        )
    print(f"  Saved {path}")

    records = attack_frame.to_dict("records")
    with (output_dir / "attack_configuration_audit.json").open("w") as file:
        json.dump(records, file, indent=2)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--results-dir",
        type=Path,
        default=(
            REPOSITORY_ROOT
            / "results"
            / "wg_zones_adversarial_robustness_results"
        ),
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=REPOSITORY_ROOT / "rebuttal_paper2" / "audit",
    )
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    verified_cells = _verify_frozen_cross_fold_summaries(args.results_dir)
    reconstruction = reconstruct_table_sem(args.results_dir)
    reconstruction.to_csv(
        args.output_dir / "table1_sem_reconstruction.csv",
        index=False,
        float_format="%.9f",
    )
    _write_sem_audit(args.output_dir, reconstruction, verified_cells)

    attack_frame = pd.DataFrame(_attack_audit_rows())
    attack_frame.to_csv(args.output_dir / "attack_configuration_audit.csv", index=False)
    _write_attack_audit(args.output_dir, attack_frame)
    print("Done.")


if __name__ == "__main__":
    main()
