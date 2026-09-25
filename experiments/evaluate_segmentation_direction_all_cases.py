"""Direction of adversarial segmentation damage for the full FGSM/PGD/APGD cohort.

Regenerates the same deterministic attacks as
``evaluate_adversarial_segmentation_all_cases.py`` (same seeds, same batch
planning), predicts clean and adversarial masks with the fold-all model, and
records where the output moved: ground-truth-aware voxel transitions, damage
rates per signed-distance band, area-weighted surface displacement, and zone
transitions.  Hard argmax masks are saved so later stages need no GPU.

Dice is recomputed here and never joined from the boundary table: APGD does
not reproduce bit-exactly between runs, and the success flag must describe
the perturbation whose direction is being measured.
"""

from __future__ import annotations

import argparse
import csv
import gc
import json
import math
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))
SRC_ROOT = REPO_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from experiments.adv_rob_eval_nnunet_nnunetrecenc import (  # noqa: E402
    IGNORE_LABEL,
    _pad_to,
    _plan_batches,
    build_model,
    load_sample,
)
from experiments.evaluate_adversarial_quality_all_cases import (  # noqa: E402
    ATTACK_LABELS,
    DEFAULT_EPS_N,
    _available_case_ids,
    _dataset_paths,
    apgd_bce_independent_batch,
    fgsm_bce_independent_batch,
    pgd_bce_independent_batch,
)
from experiments.evaluate_adversarial_segmentation_all_cases import (  # noqa: E402
    _configure_deterministic_execution,
    _predict_batch,
)
from mri_prostate_seg.experiments.epsilon_calibration import stable_seed  # noqa: E402
from mri_prostate_seg.experiments.segmentation_direction import (  # noqa: E402
    TargetGeometry,
    analyze_binary_target,
    attack_success,
    multiclass_transition_tables,
    target_geometry,
    transition_rows,
)

DEFAULT_RESULTS = (
    REPO_ROOT / "results" / "adv_rob_eval_results" / "image_quality_analysis"
)
DEFAULT_OUTPUT_DIR = DEFAULT_RESULTS / "all_cases_segmentation_direction"
SUMMARY_CSV = "case_direction_summary.csv"
PROFILE_CSV = "distance_direction_profile.csv"
TRANSITION_CSV = "class_transition_summary.csv"
MASK_DIR = "masks"
IDENTITY = ["dataset", "case_id", "epsilon_n", "attack"]
DEFAULT_TOLERANCE_MM = 0.5
ZONE_CLASS_IDS = [0, 1, 2]
ZONE_CLASS_NAMES = ["background", "TZ+CZ", "PZ"]

SUMMARY_FIELDNAMES = [
    *IDENTITY,
    "epsilon_norm",
    "attack_seed",
    "class_or_union",
    "target_valid",
    "spacing_z_mm",
    "spacing_y_mm",
    "spacing_x_mm",
    "clean_dice",
    "adv_dice",
    "delta_dice",
    "attack_success",
    "clean_volume_mm3",
    "adv_volume_mm3",
    "volume_change_mm3",
    "volume_change_pct",
    "gained_fg_mm3",
    "lost_fg_mm3",
    "induced_fp_mm3",
    "induced_fn_mm3",
    "corrected_fp_mm3",
    "corrected_fn_mm3",
    "harm_volume_mm3",
    "corrected_volume_mm3",
    "change_bias",
    "damage_bias",
    "centroid_shift_norm_mm",
    "changed_voxels_outside_valid",
    "median_surface_motion_mm",
    "q05_surface_motion_mm",
    "q10_surface_motion_mm",
    "q90_surface_motion_mm",
    "q95_surface_motion_mm",
    "median_abs_surface_motion_mm",
    "rms_surface_motion_mm",
    "outward_surface_fraction",
    "inward_surface_fraction",
    "stable_surface_fraction",
    "worsened_outward_surface_fraction",
    "worsened_inward_surface_fraction",
    "corrected_surface_fraction",
    "clean_surface_area_mm2",
    "gt_surface_area_mm2",
    "surface_metric_valid",
    "surface_failure_reason",
    "displacement_tolerance_mm",
]
PROFILE_FIELDNAMES = [
    *IDENTITY,
    "class_or_union",
    "band",
    "left_mm",
    "right_mm",
    "n_voxels",
    "induced_fp_rate",
    "induced_fn_rate",
    "corrected_fp_rate",
    "corrected_fn_rate",
    "net_fg_change_per_100_voxels",
    "gained_fg_voxels",
    "lost_fg_voxels",
    "at_risk_fp_voxels",
    "at_risk_fn_voxels",
]
TRANSITION_FIELDNAMES = [
    *IDENTITY,
    "transition_type",
    "source_class",
    "target_class",
    "voxel_count",
    "physical_volume_mm3",
    "row_normalized_rate",
]


def targets_for(
    dataset_key: str,
    ground_truth: np.ndarray,
    clean_pred: np.ndarray,
    adv_pred: np.ndarray,
) -> list[tuple[str, np.ndarray, np.ndarray, np.ndarray]]:
    g, c, a = (np.asarray(x) for x in (ground_truth, clean_pred, adv_pred))
    if dataset_key == "wg":
        return [("WG", g == 1, c == 1, a == 1)]
    return [
        ("union", g > 0, c > 0, a > 0),
        ("TZ+CZ", g == 1, c == 1, a == 1),
        ("PZ", g == 2, c == 2, a == 2),
    ]


def expected_targets(dataset_key: str) -> set[str]:
    return {"WG"} if dataset_key == "wg" else {"union", "TZ+CZ", "PZ"}


def head_target(dataset_key: str) -> str:
    return "WG" if dataset_key == "wg" else "union"


def completed_keys(summary_csv: Path) -> set[tuple[str, str, int, str]]:
    if not summary_csv.is_file() or summary_csv.stat().st_size == 0:
        return set()
    frame = pd.read_csv(summary_csv, usecols=[*IDENTITY, "class_or_union"])
    done: set[tuple[str, str, int, str]] = set()
    for key, group in frame.groupby(IDENTITY, sort=False):
        if set(group["class_or_union"].astype(str)) == expected_targets(str(key[0])):
            done.add((str(key[0]), str(key[1]), int(key[2]), str(key[3])))
    return done


def mask_path(
    output_dir: Path,
    dataset_key: str,
    case_id: str,
    attack_label: str | None = None,
    epsilon_n: int | None = None,
) -> Path:
    base = output_dir / MASK_DIR / dataset_key / case_id
    if attack_label is None:
        return base / "clean.npz"
    return base / f"{attack_label}_eps{int(epsilon_n)}.npz"


def _save_mask(path: Path, mask: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(".npz.partial")
    with temporary.open("wb") as handle:
        np.savez_compressed(handle, mask=np.asarray(mask, dtype=np.uint8))
    temporary.replace(path)


def _invalid_summary(tolerance_mm: float) -> dict[str, object]:
    row: dict[str, object] = {k: float("nan") for k in SUMMARY_FIELDNAMES}
    row.update(
        target_valid=0,
        surface_metric_valid=0.0,
        surface_failure_reason="empty_gt",
        attack_success=0,
        displacement_tolerance_mm=tolerance_mm,
    )
    return row


def direction_rows(
    dataset_key: str,
    case_id: str,
    epsilon_n: int,
    attack_label: str,
    attack_seed: int,
    spacing: tuple[float, float, float],
    ground_truth: np.ndarray,
    clean_pred: np.ndarray,
    adv_pred: np.ndarray,
    *,
    valid: np.ndarray,
    tolerance_mm: float,
    geometries: dict[str, TargetGeometry | None],
) -> tuple[list[dict[str, object]], list[dict[str, object]], list[dict[str, object]]]:
    identity = {
        "dataset": dataset_key,
        "case_id": case_id,
        "epsilon_n": int(epsilon_n),
        "attack": attack_label,
    }
    summaries: list[dict[str, object]] = []
    profiles: list[dict[str, object]] = []
    for name, g, c, a in targets_for(dataset_key, ground_truth, clean_pred, adv_pred):
        geometry = geometries.get(name)
        if geometry is None:
            row = _invalid_summary(tolerance_mm)
        else:
            summary, profile = analyze_binary_target(
                g,
                c,
                a,
                valid=valid,
                spacing=spacing,
                tolerance_mm=tolerance_mm,
                geometry=geometry,
            )
            row = {
                **summary,
                "target_valid": 1,
                "displacement_tolerance_mm": tolerance_mm,
            }
            profiles.extend({**identity, "class_or_union": name, **p} for p in profile)
        row.update(
            identity,
            epsilon_norm=epsilon_n / 255.0,
            attack_seed=attack_seed,
            class_or_union=name,
            spacing_z_mm=spacing[0],
            spacing_y_mm=spacing[1],
            spacing_x_mm=spacing[2],
        )
        summaries.append(row)
    head = next(r for r in summaries if r["class_or_union"] == head_target(dataset_key))
    success = int(attack_success(float(head["delta_dice"])))
    for row in summaries:
        row["attack_success"] = success
    transitions: list[dict[str, object]] = []
    if dataset_key == "zones":
        tables = multiclass_transition_tables(
            ground_truth, clean_pred, adv_pred, valid=valid, class_ids=ZONE_CLASS_IDS
        )
        transitions = [
            {**identity, **r}
            for r in transition_rows(
                tables,
                class_ids=ZONE_CLASS_IDS,
                class_names=ZONE_CLASS_NAMES,
                ground_truth=ground_truth,
                clean_pred=clean_pred,
                valid=valid,
                spacing=spacing,
            )
        ]
    ordered = [
        {k: r.get(k, float("nan")) for k in SUMMARY_FIELDNAMES} for r in summaries
    ]
    return ordered, profiles, transitions


def _append(path: Path, rows: list[dict[str, object]], fieldnames: list[str]) -> None:
    if not rows:
        return
    header = not path.is_file() or path.stat().st_size == 0
    with path.open("a", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames, extrasaction="ignore")
        if header:
            writer.writeheader()
        writer.writerows(rows)
        handle.flush()


def evaluate_dataset(
    dataset_key: str,
    *,
    attacks: list[str],
    epsilon_n_values: list[int],
    attack_steps: int,
    batch_size: int,
    max_cases: int | None,
    base_seed: int,
    output_dir: Path,
    completed: set[tuple[str, str, int, str]],
    device: torch.device,
    tolerance_mm: float,
    save_masks: bool,
    start_time: float,
    total_keys: int,
    attack_valid_only: bool = False,
) -> None:
    paths = _dataset_paths(dataset_key)
    case_ids = _available_case_ids(paths["data"])
    if max_cases is not None:
        case_ids = case_ids[:max_cases]
    requested_for_dataset = {
        (dataset_key, case_id, epsilon_n, ATTACK_LABELS[attack])
        for case_id in case_ids
        for attack in attacks
        for epsilon_n in epsilon_n_values
    }
    if requested_for_dataset.issubset(completed):
        print(f"[{dataset_key}] all requested rows already complete", flush=True)
        return

    print(f"[{dataset_key}] loading fold-all UNet", flush=True)
    model, div_factors, num_classes = build_model(str(paths["checkpoint"]), device)
    model.eval()
    model.requires_grad_(False)
    batches = _plan_batches(str(paths["data"]), case_ids, div_factors, batch_size)
    print(
        f"[{dataset_key}] {len(case_ids)} cases in {len(batches)} stable batches "
        f"(requested batch size {batch_size})",
        flush=True,
    )

    for batch_index, batch_case_ids in enumerate(batches, start=1):
        batch_keys = {
            (dataset_key, case_id, epsilon_n, ATTACK_LABELS[attack])
            for case_id in batch_case_ids
            for attack in attacks
            for epsilon_n in epsilon_n_values
        }
        if batch_keys.issubset(completed):
            continue

        loaded: list[tuple[str, np.ndarray, np.ndarray, tuple[float, ...]]] = []
        for case_id in batch_case_ids:
            image_np, seg_np, spacing = load_sample(str(paths["data"]), case_id)
            loaded.append((case_id, image_np, np.asarray(seg_np), spacing))

        dimensions = [record[1].shape[1:] for record in loaded]
        maximum = [max(shape[index] for shape in dimensions) for index in range(3)]
        target_shape = [
            int(math.ceil(value / factor) * factor)
            for value, factor in zip(maximum, div_factors)
        ]
        images: list[torch.Tensor] = []
        labels: list[torch.Tensor] = []
        original_shapes: list[tuple[int, ...]] = []
        segmentations: list[np.ndarray] = []
        spacings: list[tuple[float, float, float]] = []
        for _case_id, image_np, seg_np, spacing in loaded:
            image_tensor = torch.from_numpy(image_np[np.newaxis]).to(device)
            label_tensor = torch.from_numpy(seg_np[np.newaxis].astype(np.float32)).to(
                device
            )
            images.append(_pad_to(image_tensor, target_shape))
            labels.append(
                _pad_to(
                    label_tensor,
                    target_shape,
                    mode="constant",
                    value=float(IGNORE_LABEL),
                )
            )
            original_shapes.append(tuple(int(value) for value in image_np.shape[1:]))
            segmentation = np.asarray(seg_np).squeeze()
            if segmentation.ndim == 4:
                segmentation = segmentation[0]
            if segmentation.ndim != 3:
                raise ValueError(f"expected 3-D segmentation, got {segmentation.shape}")
            segmentations.append(segmentation)
            spacings.append(tuple(float(value) for value in spacing))
        image_batch = torch.cat(images, dim=0)
        label_batch = torch.cat(labels, dim=0)
        # Same rule as the attribution drivers' ``attackable_mask``: the deployed
        # pipeline regenerates ignore-labelled voxels after any image-level
        # attack, so a perturbation there cannot be realised.
        perturbation_mask = (
            (label_batch != IGNORE_LABEL) if attack_valid_only else None
        )

        clean_prediction_batch = _predict_batch(model, image_batch)
        clean_predictions: list[np.ndarray] = []
        for index, original_shape in enumerate(original_shapes):
            slices = tuple(slice(0, value) for value in original_shape)
            prediction = clean_prediction_batch[(index, *slices)].detach().cpu().numpy()
            clean_predictions.append(prediction)

        valids = [seg >= 0 for seg in segmentations]
        geometries: list[dict[str, TargetGeometry | None]] = []
        for index, (prediction, seg, spacing) in enumerate(
            zip(clean_predictions, segmentations, spacings)
        ):
            geometries.append(
                {
                    name: (target_geometry(g, c, spacing) if g.any() else None)
                    for name, g, c, _ in targets_for(
                        dataset_key, seg, prediction, prediction
                    )
                }
            )
            if save_masks:
                _save_mask(
                    mask_path(output_dir, dataset_key, loaded[index][0]), prediction
                )

        for attack_key in attacks:
            attack_label = ATTACK_LABELS[attack_key]
            for epsilon_n in epsilon_n_values:
                missing_indices = [
                    index
                    for index, (case_id, _image, _seg, _spacing) in enumerate(loaded)
                    if (dataset_key, case_id, epsilon_n, attack_label) not in completed
                ]
                if not missing_indices:
                    continue
                seeds = [
                    stable_seed(
                        dataset_key,
                        case_id,
                        f"{attack_key}_bce",
                        epsilon_n,
                        base_seed=base_seed,
                    )
                    for case_id, _image, _seg, _spacing in loaded
                ]
                with torch.enable_grad():
                    if attack_key == "fgsm":
                        adversarial_batch = fgsm_bce_independent_batch(
                            model,
                            image_batch,
                            label_batch,
                            epsilon_n / 255.0,
                            num_classes,
                            perturbation_mask=perturbation_mask,
                        )
                    elif attack_key == "pgd":
                        adversarial_batch = pgd_bce_independent_batch(
                            model,
                            image_batch,
                            label_batch,
                            epsilon_n / 255.0,
                            num_classes,
                            n_steps=attack_steps,
                            seeds=seeds,
                            perturbation_mask=perturbation_mask,
                        )
                    elif attack_key == "apgd":
                        adversarial_batch = apgd_bce_independent_batch(
                            model,
                            image_batch,
                            label_batch,
                            epsilon_n / 255.0,
                            num_classes,
                            n_steps=attack_steps,
                            seeds=seeds,
                            perturbation_mask=perturbation_mask,
                        )
                    else:  # pragma: no cover - guarded by argparse
                        raise ValueError(f"unknown attack: {attack_key}")
                adversarial_prediction_batch = _predict_batch(model, adversarial_batch)

                adversarial_predictions: list[np.ndarray] = []
                for index in missing_indices:
                    original_shape = original_shapes[index]
                    slices = tuple(slice(0, value) for value in original_shape)
                    adversarial_predictions.append(
                        adversarial_prediction_batch[(index, *slices)]
                        .detach()
                        .cpu()
                        .numpy()
                    )

                summary_rows: list[dict[str, object]] = []
                profile_rows: list[dict[str, object]] = []
                transition_rows_: list[dict[str, object]] = []
                for position, index in enumerate(missing_indices):
                    case_id = loaded[index][0]
                    adv = adversarial_predictions[position]
                    if save_masks:
                        _save_mask(
                            mask_path(
                                output_dir,
                                dataset_key,
                                case_id,
                                attack_label,
                                epsilon_n,
                            ),
                            adv,
                        )
                    s, p, t = direction_rows(
                        dataset_key,
                        case_id,
                        epsilon_n,
                        attack_label,
                        -1 if attack_key == "fgsm" else seeds[index],
                        spacings[index],
                        segmentations[index],
                        clean_predictions[index],
                        adv,
                        valid=valids[index],
                        tolerance_mm=tolerance_mm,
                        geometries=geometries[index],
                    )
                    summary_rows.extend(s)
                    profile_rows.extend(p)
                    transition_rows_.extend(t)
                _append(output_dir / PROFILE_CSV, profile_rows, PROFILE_FIELDNAMES)
                _append(
                    output_dir / TRANSITION_CSV, transition_rows_, TRANSITION_FIELDNAMES
                )
                _append(
                    output_dir / SUMMARY_CSV, summary_rows, SUMMARY_FIELDNAMES
                )  # last: the resume marker
                for index in missing_indices:
                    completed.add(
                        (
                            dataset_key,
                            loaded[index][0],
                            epsilon_n,
                            attack_label,
                        )
                    )
                del adversarial_batch, adversarial_prediction_batch

        elapsed = time.monotonic() - start_time
        completed_count = len(completed)
        rate = completed_count / max(elapsed, 1e-9)
        remaining = max(total_keys - completed_count, 0)
        eta_minutes = remaining / max(rate, 1e-9) / 60.0
        print(
            f"[{dataset_key}] batch {batch_index}/{len(batches)}; "
            f"keys {completed_count}/{total_keys}; ETA {eta_minutes:.1f} min",
            flush=True,
        )
        del (
            image_batch,
            label_batch,
            clean_prediction_batch,
            images,
            labels,
            clean_predictions,
            valids,
            geometries,
        )
        if device.type == "cuda":
            torch.cuda.empty_cache()

    del model
    gc.collect()
    if device.type == "cuda":
        torch.cuda.empty_cache()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Evaluate the direction of adversarial segmentation damage for "
            "every FGSM/PGD/APGD case"
        )
    )
    parser.add_argument(
        "--datasets", nargs="+", choices=["wg", "zones"], default=["wg", "zones"]
    )
    parser.add_argument("--eps-n", nargs="+", type=int, default=DEFAULT_EPS_N)
    parser.add_argument(
        "--attacks",
        nargs="+",
        choices=list(ATTACK_LABELS),
        default=list(ATTACK_LABELS),
    )
    parser.add_argument("--attack-steps", type=int, default=20)
    parser.add_argument("--batch-size", type=int, default=2)
    parser.add_argument("--max-cases", type=int, default=None)
    parser.add_argument("--seed", type=int, default=20260721)
    parser.add_argument("--device", type=str, default=None)
    parser.add_argument(
        "--deterministic",
        action=argparse.BooleanOptionalAction,
        default=True,
        help=(
            "enforce deterministic PyTorch/CUDA kernels so regenerated attacks "
            "remain paired across runs (default: enabled)"
        ),
    )
    parser.add_argument(
        "--displacement-tolerance-mm", type=float, default=DEFAULT_TOLERANCE_MM
    )
    parser.add_argument(
        "--save-masks",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="save hard argmax clean/adversarial masks as .npz (default: enabled)",
    )
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument(
        "--attack-valid-only",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="confine the perturbation to non-ignore voxels (excludes the zero-filled "
        "margin outside the dilated gland on zones, and the model-grid padding). "
        "The published run went without it; use a fresh --output-dir.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.attack_steps < 1:
        raise ValueError("--attack-steps must be positive")
    if args.batch_size < 1:
        raise ValueError("--batch-size must be positive")
    epsilon_n_values = sorted(set(int(value) for value in args.eps_n))
    if not epsilon_n_values or any(value <= 0 for value in epsilon_n_values):
        raise ValueError("--eps-n values must be positive")
    if args.displacement_tolerance_mm <= 0:
        raise ValueError("--displacement-tolerance-mm must be positive")
    attacks = list(dict.fromkeys(str(value) for value in args.attacks))
    _configure_deterministic_execution(enabled=args.deterministic, seed=args.seed)

    output_dir = args.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    device = (
        torch.device(args.device)
        if args.device
        else torch.device("cuda" if torch.cuda.is_available() else "cpu")
    )
    run_manifest = {
        "datasets": list(args.datasets),
        "epsilon_n_values": epsilon_n_values,
        "attacks": attacks,
        "attack_steps": int(args.attack_steps),
        "batch_size": int(args.batch_size),
        "deterministic": bool(args.deterministic),
        "max_cases": args.max_cases,
        "seed": int(args.seed),
        "device": str(device),
        "displacement_tolerance_mm": float(args.displacement_tolerance_mm),
        "save_masks": bool(args.save_masks),
        "attack_valid_only": bool(args.attack_valid_only),
        "tables": [SUMMARY_CSV, PROFILE_CSV, TRANSITION_CSV],
        "checkpoints": {
            dataset_key: str(_dataset_paths(dataset_key)["checkpoint"].resolve())
            for dataset_key in args.datasets
        },
    }
    (output_dir / "run_manifest.json").write_text(
        json.dumps(run_manifest, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    requested_cases = {
        dataset_key: _available_case_ids(_dataset_paths(dataset_key)["data"])
        for dataset_key in args.datasets
    }
    if args.max_cases is not None:
        requested_cases = {
            key: values[: args.max_cases] for key, values in requested_cases.items()
        }
    requested_keys = {
        (dataset_key, case_id, epsilon_n, ATTACK_LABELS[attack])
        for dataset_key, case_ids in requested_cases.items()
        for case_id in case_ids
        for attack in attacks
        for epsilon_n in epsilon_n_values
    }
    completed = completed_keys(output_dir / SUMMARY_CSV)
    completed.intersection_update(requested_keys)
    print(f"Device: {device}", flush=True)
    print(f"Output: {output_dir}", flush=True)
    print(
        f"Requested: {sum(len(values) for values in requested_cases.values())} "
        f"cases, {len(requested_keys)} case-attack-epsilon keys; "
        f"already complete: {len(completed)}",
        flush=True,
    )

    start_time = time.monotonic()
    for dataset_key in args.datasets:
        evaluate_dataset(
            dataset_key,
            attacks=attacks,
            epsilon_n_values=epsilon_n_values,
            attack_steps=args.attack_steps,
            batch_size=args.batch_size,
            max_cases=args.max_cases,
            base_seed=args.seed,
            output_dir=output_dir,
            completed=completed,
            device=device,
            tolerance_mm=args.displacement_tolerance_mm,
            save_masks=args.save_masks,
            start_time=start_time,
            total_keys=len(requested_keys),
            attack_valid_only=bool(args.attack_valid_only),
        )

    if requested_keys.issubset(completed):
        print(f"Complete: {len(completed)}/{len(requested_keys)} keys", flush=True)
    else:
        print(
            f"Incomplete: {len(requested_keys - completed)} keys remain; "
            "rerun the same command to resume",
            flush=True,
        )


if __name__ == "__main__":
    main()
