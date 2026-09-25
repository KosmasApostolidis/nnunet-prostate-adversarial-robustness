"""Measure paired segmentation damage for the full FGSM/PGD/APGD cohort.

The companion image-quality evaluator deliberately omitted inference after
attack generation. This resumable pass regenerates the same deterministic
attacks and records clean-versus-adversarial Dice, IoU, HD95, and ASD for every
case, attack, epsilon, and foreground class. Image-quality metrics are computed
from that same in-memory adversarial volume, avoiding invalid joins when an
adaptive attack follows a slightly different numerical trajectory on rerun.
"""

from __future__ import annotations

import argparse
import csv
import gc
import json
import math
import os
import sys
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np
import pandas as pd
import torch


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from experiments.adv_rob_eval_nnunet_nnunetrecenc import (  # noqa: E402
    DATASETS,
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
    _use_mask_for_norm,
    apgd_bce_independent_batch,
    fgsm_bce_independent_batch,
    pgd_bce_independent_batch,
)
from mri_prostate_seg.experiments.epsilon_calibration import (  # noqa: E402
    perturbation_quality,
    stable_seed,
)
from mri_prostate_seg.metrics.segmentation import (  # noqa: E402
    class_metrics_triple,
    dice_score,
    iou_score,
)

DEFAULT_OUTPUT_DIR = (
    REPO_ROOT
    / "results"
    / "adv_rob_eval_results"
    / "image_quality_analysis"
    / "all_cases_adversarial_segmentation_with_boundaries"
)
FIELDNAMES = [
    "dataset",
    "case_id",
    "epsilon_n",
    "epsilon_norm",
    "attack",
    "attack_steps",
    "attack_precision",
    "attack_seed",
    "class",
    "class_label",
    "spacing_z_mm",
    "spacing_y_mm",
    "spacing_x_mm",
    "linf_norm_recomputed",
    "rms_norm_recomputed",
    "linf_norm",
    "rms_norm",
    "rms_percent_case_sigma",
    "mae_norm",
    "robust_signal_range_norm",
    "psnr_db_robust_range",
    "ssim_axial_prostate",
    "epsilon_saturation_fraction",
    "max_ssim_slices",
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
]


def _configure_deterministic_execution(*, enabled: bool, seed: int) -> None:
    """Make repeated attack-generation passes numerically reproducible."""

    if not enabled:
        return
    os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    torch.use_deterministic_algorithms(True)


def _expected_classes(dataset_key: str) -> set[str]:
    return {*DATASETS[dataset_key]["class_names"], "macro"}


def _completed_keys(output_csv: Path) -> set[tuple[str, str, int, str]]:
    """Return only keys for which every expected foreground row is present."""

    if not output_csv.is_file() or output_csv.stat().st_size == 0:
        return set()
    frame = pd.read_csv(
        output_csv,
        usecols=["dataset", "case_id", "epsilon_n", "attack", "class"],
    )
    completed: set[tuple[str, str, int, str]] = set()
    for key, group in frame.groupby(
        ["dataset", "case_id", "epsilon_n", "attack"], sort=False
    ):
        dataset_key = str(key[0])
        if set(group["class"].astype(str)) == _expected_classes(dataset_key):
            completed.add((dataset_key, str(key[1]), int(key[2]), str(key[3])))
    return completed


def _seed_completed_rows(
    output_csv: Path,
    seed_csv: Path,
    requested_keys: set[tuple[str, str, int, str]],
) -> int:
    """Atomically reuse complete, numerically identical compatible keys."""

    if output_csv.resolve() == seed_csv.resolve():
        return 0
    if not seed_csv.is_file():
        raise FileNotFoundError(f"resume source does not exist: {seed_csv}")

    complete_seed_keys = _completed_keys(seed_csv).intersection(requested_keys)
    if not complete_seed_keys:
        return 0
    before = _completed_keys(output_csv)
    seed = pd.read_csv(seed_csv)
    missing_columns = set(FIELDNAMES).difference(seed.columns)
    if missing_columns:
        raise ValueError(f"resume source is missing columns: {sorted(missing_columns)}")
    row_keys = list(
        zip(
            seed["dataset"].astype(str),
            seed["case_id"].astype(str),
            seed["epsilon_n"].astype(int),
            seed["attack"].astype(str),
            strict=True,
        )
    )
    seed = seed.loc[[key in complete_seed_keys for key in row_keys], FIELDNAMES]
    identity_columns = ["dataset", "case_id", "epsilon_n", "attack", "class"]
    if seed.duplicated(identity_columns).any():
        raise ValueError("resume source contains duplicate identity rows")

    if output_csv.is_file() and output_csv.stat().st_size:
        current = pd.read_csv(output_csv)
        overlap = current.merge(
            seed,
            on=identity_columns,
            how="inner",
            suffixes=("_current", "_seed"),
        )
        for column in set(FIELDNAMES).difference(identity_columns):
            current_values = overlap[f"{column}_current"]
            seed_values = overlap[f"{column}_seed"]
            if pd.api.types.is_numeric_dtype(current_values):
                equal = np.isclose(
                    current_values.to_numpy(dtype=float),
                    seed_values.to_numpy(dtype=float),
                    rtol=1e-12,
                    atol=1e-12,
                    equal_nan=True,
                )
            else:
                equal = (
                    current_values.astype("string").fillna("<NA>")
                    == seed_values.astype("string").fillna("<NA>")
                ).to_numpy()
            if not bool(np.all(equal)):
                raise ValueError(
                    f"resume source conflicts with existing rows in {column!r}"
                )
        if complete_seed_keys.issubset(before):
            return 0
        combined = pd.concat([current[FIELDNAMES], seed], ignore_index=True)
        combined = combined.drop_duplicates(identity_columns, keep="first")
    else:
        combined = seed

    temporary = output_csv.with_suffix(".csv.resume.tmp")
    combined.to_csv(temporary, index=False)
    temporary.replace(output_csv)
    after = _completed_keys(output_csv)
    return len(after.difference(before))


def _append_rows(output_csv: Path, rows: list[dict[str, object]]) -> None:
    if not rows:
        return
    needs_header = not output_csv.is_file() or output_csv.stat().st_size == 0
    with output_csv.open("a", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=FIELDNAMES)
        if needs_header:
            writer.writeheader()
        writer.writerows(rows)
        handle.flush()


def _finite_difference(adversarial: float, clean: float) -> float:
    if not np.isfinite(adversarial) or not np.isfinite(clean):
        return float("nan")
    return float(adversarial - clean)


def _volume_error_percent(predicted_voxels: int, target_voxels: int) -> float:
    if target_voxels <= 0:
        return float("nan")
    return 100.0 * (predicted_voxels - target_voxels) / target_voxels


def _class_metrics(
    prediction: np.ndarray,
    target: np.ndarray,
    spacing: tuple[float, float, float],
    class_names: list[str],
    *,
    boundary_metrics: bool,
) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    for class_label, class_name in enumerate(class_names, start=1):
        prediction_mask = prediction == class_label
        target_mask = target == class_label
        dice = dice_score(prediction_mask, target_mask)
        iou = iou_score(prediction_mask, target_mask)
        if boundary_metrics:
            _boundary_dice, hd95, asd = class_metrics_triple(
                prediction_mask.astype(np.uint8),
                target_mask.astype(np.uint8),
                voxel_spacing=spacing,
            )
        else:
            hd95 = asd = float("nan")
        rows.append(
            {
                "class": class_name,
                "class_label": class_label,
                "dice": dice,
                "iou": iou,
                "hd95_mm": hd95,
                "asd_mm": asd,
                "target_foreground_voxels": int(target_mask.sum()),
                "prediction_foreground_voxels": int(prediction_mask.sum()),
            }
        )

    macro: dict[str, object] = {
        "class": "macro",
        "class_label": 0,
        "target_foreground_voxels": int((target > 0).sum()),
        "prediction_foreground_voxels": int((prediction > 0).sum()),
    }
    for metric in ("dice", "iou", "hd95_mm", "asd_mm"):
        values = np.asarray([row[metric] for row in rows], dtype=float)
        macro[metric] = (
            float(np.nanmean(values)) if np.isfinite(values).any() else float("nan")
        )
    rows.append(macro)
    return rows


def _class_metrics_batch(
    predictions: list[np.ndarray],
    targets: list[np.ndarray],
    spacings: list[tuple[float, float, float]],
    class_names: list[str],
    *,
    boundary_metrics: bool,
    workers: int,
) -> list[list[dict[str, object]]]:
    """Compute per-case metrics concurrently without copying process memory."""

    if not (len(predictions) == len(targets) == len(spacings)):
        raise ValueError("prediction, target, and spacing batches must align")

    def compute(index: int) -> list[dict[str, object]]:
        return _class_metrics(
            predictions[index],
            targets[index],
            spacings[index],
            class_names,
            boundary_metrics=boundary_metrics,
        )

    if not boundary_metrics or workers == 1 or len(predictions) < 2:
        return [compute(index) for index in range(len(predictions))]
    with ThreadPoolExecutor(max_workers=min(workers, len(predictions))) as executor:
        return list(executor.map(compute, range(len(predictions))))


def _segmentation_rows(
    *,
    dataset_key: str,
    case_id: str,
    epsilon_n: int,
    attack: str,
    attack_steps: int,
    attack_precision: str,
    attack_seed: int,
    spacing: tuple[float, float, float],
    linf_norm: float,
    rms_norm: float,
    quality_metrics: dict[str, float],
    max_ssim_slices: int,
    clean_metrics: list[dict[str, object]],
    adversarial_metrics: list[dict[str, object]],
) -> list[dict[str, object]]:
    clean_by_class = {str(row["class"]): row for row in clean_metrics}
    rows: list[dict[str, object]] = []
    for adversarial in adversarial_metrics:
        class_name = str(adversarial["class"])
        clean = clean_by_class[class_name]
        target_voxels = int(adversarial["target_foreground_voxels"])
        clean_voxels = int(clean["prediction_foreground_voxels"])
        adversarial_voxels = int(adversarial["prediction_foreground_voxels"])
        clean_dice = float(clean["dice"])
        adversarial_dice = float(adversarial["dice"])
        clean_iou = float(clean["iou"])
        adversarial_iou = float(adversarial["iou"])
        clean_hd95 = float(clean["hd95_mm"])
        adversarial_hd95 = float(adversarial["hd95_mm"])
        clean_asd = float(clean["asd_mm"])
        adversarial_asd = float(adversarial["asd_mm"])
        rows.append(
            {
                "dataset": dataset_key,
                "case_id": case_id,
                "epsilon_n": epsilon_n,
                "epsilon_norm": epsilon_n / 255.0,
                "attack": attack,
                "attack_steps": attack_steps,
                "attack_precision": attack_precision,
                "attack_seed": attack_seed,
                "class": class_name,
                "class_label": int(adversarial["class_label"]),
                "spacing_z_mm": spacing[0],
                "spacing_y_mm": spacing[1],
                "spacing_x_mm": spacing[2],
                "linf_norm_recomputed": linf_norm,
                "rms_norm_recomputed": rms_norm,
                "linf_norm": quality_metrics["linf_norm"],
                "rms_norm": quality_metrics["rms_norm"],
                "rms_percent_case_sigma": quality_metrics["rms_percent_case_sigma"],
                "mae_norm": quality_metrics["mae_norm"],
                "robust_signal_range_norm": quality_metrics["robust_signal_range_norm"],
                "psnr_db_robust_range": quality_metrics["psnr_db_robust_range"],
                "ssim_axial_prostate": quality_metrics["ssim_axial_prostate"],
                "epsilon_saturation_fraction": quality_metrics[
                    "epsilon_saturation_fraction"
                ],
                "max_ssim_slices": max_ssim_slices,
                "clean_dice": clean_dice,
                "adversarial_dice": adversarial_dice,
                "dice_drop": clean_dice - adversarial_dice,
                "clean_iou": clean_iou,
                "adversarial_iou": adversarial_iou,
                "iou_drop": clean_iou - adversarial_iou,
                "clean_hd95_mm": clean_hd95,
                "adversarial_hd95_mm": adversarial_hd95,
                "hd95_increase_mm": _finite_difference(adversarial_hd95, clean_hd95),
                "clean_asd_mm": clean_asd,
                "adversarial_asd_mm": adversarial_asd,
                "asd_increase_mm": _finite_difference(adversarial_asd, clean_asd),
                "target_foreground_voxels": target_voxels,
                "clean_prediction_foreground_voxels": clean_voxels,
                "adversarial_prediction_foreground_voxels": adversarial_voxels,
                "clean_volume_error_percent": _volume_error_percent(
                    clean_voxels, target_voxels
                ),
                "adversarial_volume_error_percent": _volume_error_percent(
                    adversarial_voxels, target_voxels
                ),
            }
        )
    return rows


def _predict_batch(model: torch.nn.Module, images: torch.Tensor) -> torch.Tensor:
    with torch.no_grad():
        return model(images).argmax(dim=1)


def evaluate_dataset(
    dataset_key: str,
    *,
    attacks: list[str],
    epsilon_n_values: list[int],
    attack_steps: int,
    batch_size: int,
    use_bf16: bool,
    max_cases: int | None,
    base_seed: int,
    output_csv: Path,
    completed: set[tuple[str, str, int, str]],
    device: torch.device,
    boundary_metrics: bool,
    boundary_workers: int,
    max_ssim_slices: int,
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
    class_names = list(DATASETS[dataset_key]["class_names"])
    use_mask = _use_mask_for_norm(paths["plans"])
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

        with torch.autocast(
            device_type=device.type,
            dtype=torch.bfloat16,
            enabled=use_bf16,
        ):
            clean_prediction_batch = _predict_batch(model, image_batch)
        clean_predictions: list[np.ndarray] = []
        for index, original_shape in enumerate(original_shapes):
            slices = tuple(slice(0, value) for value in original_shape)
            prediction = clean_prediction_batch[(index, *slices)].detach().cpu().numpy()
            clean_predictions.append(prediction)
        clean_metrics = _class_metrics_batch(
            clean_predictions,
            segmentations,
            spacings,
            class_names,
            boundary_metrics=boundary_metrics,
            workers=boundary_workers,
        )

        for attack_key in attacks:
            attack_label = ATTACK_LABELS[attack_key]
            steps = 1 if attack_key == "fgsm" else attack_steps
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
                with (
                    torch.enable_grad(),
                    torch.autocast(
                        device_type=device.type,
                        dtype=torch.bfloat16,
                        enabled=use_bf16,
                    ),
                ):
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
                with torch.autocast(
                    device_type=device.type,
                    dtype=torch.bfloat16,
                    enabled=use_bf16,
                ):
                    adversarial_prediction_batch = _predict_batch(
                        model, adversarial_batch
                    )

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
                adversarial_metrics_for_missing = _class_metrics_batch(
                    adversarial_predictions,
                    [segmentations[index] for index in missing_indices],
                    [spacings[index] for index in missing_indices],
                    class_names,
                    boundary_metrics=boundary_metrics,
                    workers=boundary_workers,
                )
                adversarial_metrics_by_index = dict(
                    zip(missing_indices, adversarial_metrics_for_missing)
                )

                rows: list[dict[str, object]] = []
                for index in missing_indices:
                    case_id = loaded[index][0]
                    original_shape = original_shapes[index]
                    slices = tuple(slice(0, value) for value in original_shape)
                    clean_array = (
                        image_batch[(index, 0, *slices)]
                        .detach()
                        .cpu()
                        .numpy()
                        .astype(np.float32)
                    )
                    adversarial_array = (
                        adversarial_batch[(index, 0, *slices)]
                        .detach()
                        .cpu()
                        .numpy()
                        .astype(np.float32)
                    )
                    valid_mask = (
                        segmentations[index] >= 0
                        if use_mask
                        else np.ones_like(segmentations[index], dtype=bool)
                    )
                    quality_metrics = perturbation_quality(
                        clean_array,
                        adversarial_array,
                        valid_mask=valid_mask,
                        slice_selector=segmentations[index] > 0,
                        epsilon=epsilon_n / 255.0,
                        max_ssim_slices=max_ssim_slices,
                    )
                    rows.extend(
                        _segmentation_rows(
                            dataset_key=dataset_key,
                            case_id=case_id,
                            epsilon_n=epsilon_n,
                            attack=attack_label,
                            attack_steps=steps,
                            attack_precision=("bf16-autocast" if use_bf16 else "fp32"),
                            attack_seed=(-1 if attack_key == "fgsm" else seeds[index]),
                            spacing=spacings[index],
                            linf_norm=quality_metrics["linf_norm"],
                            rms_norm=quality_metrics["rms_norm"],
                            quality_metrics=quality_metrics,
                            max_ssim_slices=max_ssim_slices,
                            clean_metrics=clean_metrics[index],
                            adversarial_metrics=adversarial_metrics_by_index[index],
                        )
                    )
                _append_rows(output_csv, rows)
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
        )
        if device.type == "cuda":
            torch.cuda.empty_cache()

    del model
    gc.collect()
    if device.type == "cuda":
        torch.cuda.empty_cache()


def _describe(values: pd.Series) -> dict[str, float]:
    array = values.to_numpy(dtype=float)
    array = array[np.isfinite(array)]
    if not len(array):
        return {
            "mean": float("nan"),
            "median": float("nan"),
            "q05": float("nan"),
            "q95": float("nan"),
        }
    return {
        "mean": float(np.mean(array)),
        "median": float(np.median(array)),
        "q05": float(np.quantile(array, 0.05)),
        "q95": float(np.quantile(array, 0.95)),
    }


def _write_summary(output_csv: Path, summary_csv: Path) -> None:
    frame = pd.read_csv(output_csv)
    rows: list[dict[str, object]] = []
    metrics = (
        "clean_dice",
        "adversarial_dice",
        "dice_drop",
        "clean_iou",
        "adversarial_iou",
        "iou_drop",
        "hd95_increase_mm",
        "asd_increase_mm",
        "adversarial_volume_error_percent",
        "rms_percent_case_sigma",
        "psnr_db_robust_range",
        "ssim_axial_prostate",
    )
    for (dataset, attack, epsilon_n, class_name), group in frame.groupby(
        ["dataset", "attack", "epsilon_n", "class"]
    ):
        row: dict[str, object] = {
            "dataset": dataset,
            "attack": attack,
            "epsilon_n": int(epsilon_n),
            "epsilon_norm": int(epsilon_n) / 255.0,
            "class": class_name,
            "n_cases": int(group["case_id"].nunique()),
        }
        for metric in metrics:
            for statistic, value in _describe(group[metric]).items():
                row[f"{metric}_{statistic}"] = value
        rows.append(row)
    pd.DataFrame(rows).sort_values(["dataset", "attack", "epsilon_n", "class"]).to_csv(
        summary_csv, index=False
    )


def _write_same_pass_quality(output_csv: Path, quality_csv: Path) -> None:
    frame = pd.read_csv(output_csv)
    macro = frame[frame["class"] == "macro"].copy()
    columns = [
        "dataset",
        "case_id",
        "epsilon_n",
        "epsilon_norm",
        "attack",
        "attack_steps",
        "attack_precision",
        "attack_seed",
        "linf_norm",
        "rms_norm",
        "rms_percent_case_sigma",
        "mae_norm",
        "robust_signal_range_norm",
        "psnr_db_robust_range",
        "ssim_axial_prostate",
        "epsilon_saturation_fraction",
        "max_ssim_slices",
    ]
    macro[columns].sort_values(["dataset", "case_id", "attack", "epsilon_n"]).to_csv(
        quality_csv, index=False
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Evaluate paired clean/adversarial segmentation metrics for every "
            "FGSM/PGD/APGD case"
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
    parser.add_argument("--bf16", action="store_true")
    parser.add_argument("--max-cases", type=int, default=None)
    parser.add_argument("--seed", type=int, default=20260721)
    parser.add_argument("--max-ssim-slices", type=int, default=5)
    parser.add_argument(
        "--boundary-workers",
        type=int,
        default=2,
        help="threads used for per-case HD95/ASD computation (default: 2)",
    )
    parser.add_argument("--device", type=str, default=None)
    parser.add_argument(
        "--deterministic",
        action=argparse.BooleanOptionalAction,
        default=True,
        help=(
            "enforce deterministic PyTorch/CUDA kernels so regenerated attacks "
            "and boundary metrics remain paired across runs (default: enabled)"
        ),
    )
    parser.add_argument(
        "--skip-boundary-metrics",
        action="store_true",
        help="compute Dice and IoU only; HD95 and ASD are left missing",
    )
    parser.add_argument(
        "--resume-from",
        type=Path,
        default=None,
        help=(
            "reuse fully complete keys from a compatible prior evaluation; "
            "overlapping rows must match exactly"
        ),
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
    if args.boundary_workers < 1:
        raise ValueError("--boundary-workers must be positive")
    epsilon_n_values = sorted(set(int(value) for value in args.eps_n))
    if not epsilon_n_values or any(value <= 0 for value in epsilon_n_values):
        raise ValueError("--eps-n values must be positive")
    attacks = list(dict.fromkeys(str(value) for value in args.attacks))
    _configure_deterministic_execution(enabled=args.deterministic, seed=args.seed)

    output_dir = args.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    output_csv = output_dir / "adversarial_segmentation_all_cases.csv"
    summary_csv = output_dir / "adversarial_segmentation_all_cases_summary.csv"
    quality_csv = output_dir / "adversarial_quality_same_pass.csv"
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
        "boundary_metrics": not args.skip_boundary_metrics,
        "boundary_workers": int(args.boundary_workers),
        "deterministic": bool(args.deterministic),
        "bf16_autocast": bool(args.bf16),
        "max_cases": args.max_cases,
        "max_ssim_slices": int(args.max_ssim_slices),
        "seed": int(args.seed),
        "device": str(device),
        "resume_from": str(args.resume_from.resolve()) if args.resume_from else None,
        "attack_valid_only": bool(args.attack_valid_only),
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
    seeded_keys = 0
    if args.resume_from is not None:
        seeded_keys = _seed_completed_rows(
            output_csv,
            args.resume_from.resolve(),
            requested_keys,
        )
    completed = _completed_keys(output_csv)
    completed.intersection_update(requested_keys)
    print(f"Device: {device}", flush=True)
    print(f"Output: {output_csv}", flush=True)
    print(
        f"Requested: {sum(len(values) for values in requested_cases.values())} "
        f"cases, {len(requested_keys)} case-attack-epsilon keys; "
        f"already complete: {len(completed)}",
        flush=True,
    )
    print(
        "Boundary metrics: "
        f"{'disabled' if args.skip_boundary_metrics else 'HD95 and ASD enabled'}",
        flush=True,
    )
    if seeded_keys:
        print(
            f"Reused {seeded_keys} complete keys from {args.resume_from.resolve()}",
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
            use_bf16=args.bf16,
            max_cases=args.max_cases,
            base_seed=args.seed,
            output_csv=output_csv,
            completed=completed,
            device=device,
            boundary_metrics=not args.skip_boundary_metrics,
            boundary_workers=args.boundary_workers,
            max_ssim_slices=args.max_ssim_slices,
            start_time=start_time,
            total_keys=len(requested_keys),
            attack_valid_only=bool(args.attack_valid_only),
        )

    if requested_keys.issubset(completed):
        _write_summary(output_csv, summary_csv)
        _write_same_pass_quality(output_csv, quality_csv)
        print(f"Complete: {len(completed)}/{len(requested_keys)} keys", flush=True)
        print(f"Saved {summary_csv}", flush=True)
        print(f"Saved {quality_csv}", flush=True)
    else:
        print(
            f"Incomplete: {len(requested_keys - completed)} keys remain; "
            "rerun the same command to resume",
            flush=True,
        )


if __name__ == "__main__":
    main()
