"""Calibrate adversarial epsilon for the preprocessed prostate-MRI datasets.

The nnU-Net inputs evaluated by this repository are per-case z-score normalized.
MRI signal values are arbitrary units rather than standardized physical units, so
this experiment reports epsilon primarily as a fraction of the within-case signal
standard deviation.  It also:

* verifies the raw and post-preprocessing mean/std for both datasets;
* reconstructs the exact per-case z-score transform from the raw NIfTI files;
* measures PSNR, axial SSIM, L-inf, RMS and an image-derived high-frequency noise
  proxy for report-compatible APGD-BCE perturbations;
* compares each adversarial perturbation with RMS- and L-inf-matched Gaussian and
  Rician-magnitude-proxy noise; and
* reports the corresponding segmentation Dice on deterministic case samples.

Raw MRI intensities remain scanner/reconstruction-dependent arbitrary units
(AU), not a standardized physical unit. The exact inverse conversion here means
the original per-case AU scale, on the resampled model grid.

Example (quick smoke run)::

    python experiments/epsilon_physical_calibration.py \
        --stats-cases 8 --attack-cases 1 --noise-trials 1 \
        --eps-n 0 16 --apgd-steps 2

Representative run::

    python experiments/epsilon_physical_calibration.py \
        --stats-cases 128 --attack-cases 12 --noise-trials 3
"""

from __future__ import annotations

import argparse
import gc
import json
import pickle
import sys
from pathlib import Path
from typing import Iterable

import matplotlib
import numpy as np
import pandas as pd
import torch
from nnunetv2.imageio.simpleitk_reader_writer import SimpleITKIO
from nnunetv2.preprocessing.cropping.cropping import crop_to_nonzero

matplotlib.use("Agg")
import matplotlib.pyplot as plt


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from experiments.adv_rob_eval_nnunet_nnunetrecenc import (  # noqa: E402
    DATASETS,
    IGNORE_LABEL,
    NNUNET_PATHS,
    TRAINERS,
    _pad_to,
    apgd_bce,
    build_model,
    load_sample,
    pad_to_divisible,
    unpad,
)
from mri_prostate_seg.experiments.epsilon_calibration import (  # noqa: E402
    estimate_high_frequency_noise_sigma,
    matched_gaussian_noise,
    matched_rician_proxy_noise,
    normalized_epsilon_mapping,
    perturbation_quality,
    stable_seed,
)
from mri_prostate_seg.metrics.segmentation import dice_score  # noqa: E402


DEFAULT_OUT_DIR = REPO_ROOT / "results" / "adv_rob_eval_results" / "image_quality_analysis"
DEFAULT_EPS_N = [0, 2, 4, 8, 16, 32]
PERTURBATION_LABELS = {
    "apgd_bce": "APGD-BCE",
    "gaussian_rms_matched": "Gaussian (RMS matched)",
    "rician_proxy_rms_matched": "Rician proxy (RMS matched)",
    "clean": "Clean",
}
PERTURBATION_COLORS = {
    "apgd_bce": "#D55E00",
    "gaussian_rms_matched": "#0072B2",
    "rician_proxy_rms_matched": "#009E73",
    "clean": "#333333",
}


def _dataset_paths(dataset_key: str) -> dict[str, Path]:
    cfg = DATASETS[dataset_key]
    dataset_name = cfg["dataset_name"]
    root = Path(NNUNET_PATHS)
    return {
        "data": root / "nnUNet_preprocessed" / dataset_name / cfg["data_subdir"],
        "plans": root / "nnUNet_preprocessed" / dataset_name / "nnUNetPlans.json",
        "raw": root / "nnUNet_raw" / dataset_name,
        "checkpoint": root
        / "nnUNet_results"
        / dataset_name
        / TRAINERS["unet"]
        / "fold_all"
        / "checkpoint_final.pth",
    }


def _available_case_ids(data_dir: Path) -> list[str]:
    return sorted(
        path.name[: -len(".b2nd")]
        for path in data_dir.glob("*.b2nd")
        if not path.name.endswith("_seg.b2nd")
    )


def _sample_ids(
    case_ids: list[str],
    count: int,
    *,
    dataset_key: str,
    seed: int,
) -> list[str]:
    if count <= 0 or count >= len(case_ids):
        return list(case_ids)
    rng = np.random.default_rng(stable_seed("stats", dataset_key, base_seed=seed))
    chosen = rng.choice(np.asarray(case_ids, dtype=object), size=count, replace=False)
    return sorted(str(x) for x in chosen)


def _normalization_config(plans_path: Path) -> dict[str, object]:
    plans = json.loads(plans_path.read_text())
    configuration = plans["configurations"]["3d_fullres"]
    intensity = plans["foreground_intensity_properties_per_channel"]["0"]
    return {
        "scheme": configuration["normalization_schemes"][0],
        "use_mask_for_norm": bool(configuration["use_mask_for_norm"][0]),
        "transpose_forward": [int(value) for value in plans["transpose_forward"]],
        "cohort_raw_fg_mean_au": float(intensity["mean"]),
        "cohort_raw_fg_std_au": float(intensity["std"]),
        "cohort_raw_fg_p005_au": float(intensity["percentile_00_5"]),
        "cohort_raw_fg_p995_au": float(intensity["percentile_99_5"]),
        "cohort_raw_fg_min_au": float(intensity["min"]),
        "cohort_raw_fg_max_au": float(intensity["max"]),
    }


def _seg_volume(seg_np: np.ndarray) -> np.ndarray:
    seg = np.asarray(seg_np).squeeze()
    if seg.ndim == 4:
        seg = seg[0]
    if seg.ndim != 3:
        raise ValueError(f"expected a 3-D segmentation, got shape {seg.shape}")
    return seg


def _valid_mask(seg: np.ndarray, use_mask_for_norm: bool) -> np.ndarray:
    return (seg >= 0) if use_mask_for_norm else np.ones_like(seg, dtype=bool)


def _resolve_case_file(raw_dir: Path, folders: tuple[str, ...], filename: str) -> Path:
    for folder in folders:
        candidate = raw_dir / folder / filename
        if candidate.is_file():
            return candidate
    searched = ", ".join(str(raw_dir / folder / filename) for folder in folders)
    raise FileNotFoundError(f"could not find raw case file; searched: {searched}")


def _raw_normalization_record(
    case_id: str,
    *,
    data_dir: Path,
    raw_dir: Path,
    transpose_forward: list[int],
    use_mask_for_norm: bool,
    max_slices: int,
) -> dict[str, object]:
    """Reproduce nnU-Net's pre-resampling per-case z-score statistics."""

    image_path = _resolve_case_file(
        raw_dir, ("imagesTr", "ImagesTr"), f"{case_id}_0000.nii.gz"
    )
    label_path = _resolve_case_file(
        raw_dir, ("labelsTr", "LabelsTr"), f"{case_id}.nii.gz"
    )
    properties_path = data_dir / f"{case_id}.pkl"
    properties = pickle.loads(properties_path.read_bytes())

    reader = SimpleITKIO()
    raw, _ = reader.read_images((str(image_path),))
    raw_seg, _ = reader.read_seg(str(label_path))
    raw = raw.astype(np.float32)
    spatial_axes = [axis + 1 for axis in transpose_forward]
    raw = raw.transpose([0, *spatial_axes])
    raw_seg = raw_seg.transpose([0, *spatial_axes])
    cropped, cropped_seg, recomputed_bbox = crop_to_nonzero(raw, raw_seg)

    stored_bbox = [
        [int(value) for value in pair] for pair in properties["bbox_used_for_cropping"]
    ]
    recomputed_bbox = [[int(value) for value in pair] for pair in recomputed_bbox]
    bbox_matches = recomputed_bbox == stored_bbox
    stored_shape = tuple(
        int(value) for value in properties["shape_after_cropping_and_before_resampling"]
    )
    shape_matches = tuple(cropped.shape[1:]) == stored_shape
    if not bbox_matches or not shape_matches:
        raise RuntimeError(
            f"raw/preprocessed crop mismatch for {case_id}: "
            f"bbox={recomputed_bbox} vs {stored_bbox}, "
            f"shape={tuple(cropped.shape[1:])} vs {stored_shape}"
        )

    image = np.asarray(cropped[0], dtype=np.float32)
    segmentation = np.asarray(cropped_seg[0])
    norm_mask = (
        segmentation >= 0 if use_mask_for_norm else np.ones_like(image, dtype=bool)
    )
    foreground = segmentation > 0
    values = image[norm_mask]
    raw_mean = float(values.mean())
    raw_std = float(values.std())
    if not np.isfinite(raw_std) or raw_std <= 0:
        raise RuntimeError(f"invalid raw normalization std for {case_id}: {raw_std}")
    p005, p995 = np.percentile(values.astype(np.float64), [0.5, 99.5])
    raw_hf_sigma = estimate_high_frequency_noise_sigma(
        image,
        valid_mask=norm_mask,
        slice_selector=foreground,
        max_slices=max_slices,
    )
    return {
        "raw_image_voxels": int(image.size),
        "raw_normalization_voxels": int(norm_mask.sum()),
        "raw_nonzero_crop_fraction": float((segmentation >= 0).mean()),
        "raw_norm_mean_au": raw_mean,
        "raw_norm_std_au": raw_std,
        "raw_norm_p005_au": float(p005),
        "raw_norm_p995_au": float(p995),
        "raw_norm_min_au": float(values.min()),
        "raw_norm_max_au": float(values.max()),
        "raw_negative_fraction": float(np.mean(values < 0)),
        "raw_zero_fraction": float(np.mean(values == 0)),
        "raw_zero_normalized": -raw_mean / raw_std,
        "raw_grid_hf_noise_proxy_sigma_au": float(raw_hf_sigma),
        "raw_grid_hf_noise_proxy_sigma_norm": float(raw_hf_sigma / raw_std),
        "raw_crop_bbox_matches_preprocessing": bbox_matches,
        "raw_crop_shape_matches_preprocessing": shape_matches,
    }


def _normalization_record(
    dataset_key: str,
    case_id: str,
    data_dir: Path,
    *,
    raw_dir: Path,
    transpose_forward: list[int],
    use_mask_for_norm: bool,
    max_slices: int,
) -> dict[str, object]:
    image_np, seg_np, _spacing = load_sample(str(data_dir), case_id)
    image = np.asarray(image_np[0], dtype=np.float32)
    seg = _seg_volume(seg_np)
    valid = _valid_mask(seg, use_mask_for_norm)
    values = image[valid].astype(np.float64)
    foreground = seg > 0
    model_grid_hf_sigma = estimate_high_frequency_noise_sigma(
        image,
        valid_mask=valid,
        slice_selector=foreground,
        max_slices=max_slices,
    )
    p005, p995 = np.percentile(values, [0.5, 99.5])
    raw_record = _raw_normalization_record(
        case_id,
        data_dir=data_dir,
        raw_dir=raw_dir,
        transpose_forward=transpose_forward,
        use_mask_for_norm=use_mask_for_norm,
        max_slices=max_slices,
    )
    return {
        "dataset": dataset_key,
        "case_id": case_id,
        "depth": int(image.shape[0]),
        "height": int(image.shape[1]),
        "width": int(image.shape[2]),
        "valid_voxels": int(valid.sum()),
        "normalized_mean": float(values.mean()),
        "normalized_std": float(values.std()),
        "normalized_p005": float(p005),
        "normalized_p995": float(p995),
        "normalized_min": float(values.min()),
        "normalized_max": float(values.max()),
        "model_grid_hf_noise_proxy_sigma_norm": float(model_grid_hf_sigma),
        # The legacy name now denotes the raw-grid image-derived noise estimate
        # expressed in model-normalized units, for direct perturbation ratios.
        "high_frequency_noise_sigma_norm": raw_record[
            "raw_grid_hf_noise_proxy_sigma_norm"
        ],
        **raw_record,
    }


def _summarize_normalization(
    dataset_key: str,
    records: pd.DataFrame,
    *,
    total_cases: int,
    config: dict[str, object],
    raw_dir: Path,
) -> dict[str, object]:
    raw_images_present = any((raw_dir / "imagesTr").glob("*.nii.gz")) or any(
        (raw_dir / "ImagesTr").glob("*.nii.gz")
    )
    return {
        "dataset": dataset_key,
        "normalization_scheme": config["scheme"],
        "use_mask_for_norm": config["use_mask_for_norm"],
        "total_cases": int(total_cases),
        "cases_checked": int(len(records)),
        "median_normalized_mean": float(records["normalized_mean"].median()),
        "max_abs_normalized_mean": float(records["normalized_mean"].abs().max()),
        "median_normalized_std": float(records["normalized_std"].median()),
        "q05_normalized_std": float(records["normalized_std"].quantile(0.05)),
        "q95_normalized_std": float(records["normalized_std"].quantile(0.95)),
        "median_hf_noise_proxy_sigma_norm": float(
            records["high_frequency_noise_sigma_norm"].median()
        ),
        "q05_hf_noise_proxy_sigma_norm": float(
            records["high_frequency_noise_sigma_norm"].quantile(0.05)
        ),
        "q95_hf_noise_proxy_sigma_norm": float(
            records["high_frequency_noise_sigma_norm"].quantile(0.95)
        ),
        "median_model_grid_hf_noise_proxy_sigma_norm": float(
            records["model_grid_hf_noise_proxy_sigma_norm"].median()
        ),
        "median_raw_norm_mean_au": float(records["raw_norm_mean_au"].median()),
        "q05_raw_norm_mean_au": float(records["raw_norm_mean_au"].quantile(0.05)),
        "q95_raw_norm_mean_au": float(records["raw_norm_mean_au"].quantile(0.95)),
        "median_raw_norm_std_au": float(records["raw_norm_std_au"].median()),
        "q05_raw_norm_std_au": float(records["raw_norm_std_au"].quantile(0.05)),
        "q95_raw_norm_std_au": float(records["raw_norm_std_au"].quantile(0.95)),
        "median_raw_grid_hf_noise_proxy_sigma_au": float(
            records["raw_grid_hf_noise_proxy_sigma_au"].median()
        ),
        "median_raw_negative_fraction": float(
            records["raw_negative_fraction"].median()
        ),
        "crop_validation_failures": int(
            (
                ~(
                    records["raw_crop_bbox_matches_preprocessing"].astype(bool)
                    & records["raw_crop_shape_matches_preprocessing"].astype(bool)
                )
            ).sum()
        ),
        "cohort_raw_fg_mean_au": config["cohort_raw_fg_mean_au"],
        "cohort_raw_fg_std_au": config["cohort_raw_fg_std_au"],
        "cohort_raw_fg_p005_au": config["cohort_raw_fg_p005_au"],
        "cohort_raw_fg_p995_au": config["cohort_raw_fg_p995_au"],
        "raw_training_images_present": bool(raw_images_present),
        "exact_per_case_raw_inverse_available": bool(raw_images_present),
        "acquisition_noise_measurement_available": False,
        "raw_image_noise_proxy_available": True,
    }


def _epsilon_mapping_record(
    dataset_key: str,
    epsilon_n: int,
    records: pd.DataFrame,
    config: dict[str, object],
) -> dict[str, object]:
    row = {
        "dataset": dataset_key,
        **normalized_epsilon_mapping(
            epsilon_n,
            cohort_foreground_std_au=float(config["cohort_raw_fg_std_au"]),
        ),
    }
    epsilon = float(row["epsilon_norm"])
    exact_raw = epsilon * records["raw_norm_std_au"].to_numpy(dtype=np.float64)
    raw_noise_norm = records["raw_grid_hf_noise_proxy_sigma_norm"].to_numpy(
        dtype=np.float64
    )
    noise_ratios = np.divide(
        epsilon,
        raw_noise_norm,
        out=np.full_like(raw_noise_norm, np.nan),
        where=raw_noise_norm > 0,
    )
    row.update(
        {
            "exact_case_raw_linf_median_au": float(np.median(exact_raw)),
            "exact_case_raw_linf_q05_au": float(np.quantile(exact_raw, 0.05)),
            "exact_case_raw_linf_q95_au": float(np.quantile(exact_raw, 0.95)),
            "epsilon_over_raw_noise_proxy_median": float(np.nanmedian(noise_ratios)),
            "epsilon_over_raw_noise_proxy_q05": float(
                np.nanquantile(noise_ratios, 0.05)
            ),
            "epsilon_over_raw_noise_proxy_q95": float(
                np.nanquantile(noise_ratios, 0.95)
            ),
            "raw_mapping_status": "exact_per_case_zscore_inverse_summarized_across_cases",
            "raw_units": "original MRI arbitrary units (AU)",
        }
    )
    return row


def _stratified_attack_ids(stats: pd.DataFrame, count: int) -> list[str]:
    ordered = stats.sort_values(
        ["high_frequency_noise_sigma_norm", "case_id"], na_position="last"
    )["case_id"].tolist()
    if count <= 0 or count >= len(ordered):
        return [str(x) for x in ordered]
    indices = np.linspace(0, len(ordered) - 1, count).round().astype(int)
    return [str(ordered[i]) for i in dict.fromkeys(indices)]


def _pad_attack_inputs(
    image_np: np.ndarray,
    seg_np: np.ndarray,
    div_factors: list[int],
    device: torch.device,
) -> tuple[torch.Tensor, torch.Tensor, tuple[int, ...]]:
    x = torch.from_numpy(image_np[np.newaxis]).to(device)
    y = torch.from_numpy(seg_np[np.newaxis].astype(np.float32)).to(device)
    x_padded, orig_shape = pad_to_divisible(x, div_factors)
    y_padded = _pad_to(y, list(x_padded.shape[2:]), value=float(IGNORE_LABEL))
    return x_padded, y_padded, orig_shape


def _restore_artificial_padding(
    altered: torch.Tensor,
    clean: torch.Tensor,
    orig_shape: tuple[int, ...],
) -> torch.Tensor:
    valid = torch.zeros_like(clean, dtype=torch.bool)
    slices = tuple([slice(None), slice(None)] + [slice(0, int(s)) for s in orig_shape])
    valid[slices] = True
    return torch.where(valid, altered, clean)


def _predict_tensor(
    model: torch.nn.Module,
    image_padded: torch.Tensor,
    orig_shape: tuple[int, ...],
) -> np.ndarray:
    with torch.no_grad():
        logits = unpad(model(image_padded), orig_shape)
    return logits.argmax(dim=1).squeeze(0).cpu().numpy().astype(np.int64)


def _predict_numpy(
    model: torch.nn.Module,
    image: np.ndarray,
    div_factors: list[int],
    device: torch.device,
) -> np.ndarray:
    x = torch.from_numpy(image[np.newaxis, np.newaxis].astype(np.float32)).to(device)
    x_padded, orig_shape = pad_to_divisible(x, div_factors)
    pred = _predict_tensor(model, x_padded, orig_shape)
    del x, x_padded
    return pred


def _dice_records(
    dataset_key: str,
    case_id: str,
    epsilon_n: int,
    perturbation: str,
    trial: int,
    prediction: np.ndarray,
    target: np.ndarray,
    class_names: list[str],
) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    per_class: list[float] = []
    for label, class_name in enumerate(class_names, start=1):
        value = dice_score(prediction == label, target == label)
        per_class.append(value)
        rows.append(
            {
                "dataset": dataset_key,
                "case_id": case_id,
                "epsilon_n": epsilon_n,
                "epsilon_norm": epsilon_n / 255.0,
                "perturbation": perturbation,
                "trial": trial,
                "class": class_name,
                "dice": value,
            }
        )
    rows.append(
        {
            "dataset": dataset_key,
            "case_id": case_id,
            "epsilon_n": epsilon_n,
            "epsilon_norm": epsilon_n / 255.0,
            "perturbation": perturbation,
            "trial": trial,
            "class": "macro",
            "dice": float(np.mean(per_class)),
        }
    )
    return rows


def _quality_record(
    dataset_key: str,
    case_id: str,
    epsilon_n: int,
    perturbation: str,
    trial: int,
    clean: np.ndarray,
    altered: np.ndarray,
    *,
    valid_mask: np.ndarray,
    foreground: np.ndarray,
    raw_noise_sigma_norm: float,
    raw_noise_sigma_au: float,
    case_raw_mean: float,
    case_raw_std: float,
    cohort_raw_std: float,
    max_ssim_slices: int,
) -> dict[str, object]:
    epsilon = epsilon_n / 255.0
    metrics = perturbation_quality(
        clean,
        altered,
        valid_mask=valid_mask,
        slice_selector=foreground,
        epsilon=epsilon,
        high_frequency_noise_sigma=raw_noise_sigma_norm,
        max_ssim_slices=max_ssim_slices,
    )
    return {
        "dataset": dataset_key,
        "case_id": case_id,
        "epsilon_n": epsilon_n,
        "epsilon_norm": epsilon,
        "perturbation": perturbation,
        "trial": trial,
        **metrics,
        "case_raw_norm_mean_au": case_raw_mean,
        "case_raw_norm_std_au": case_raw_std,
        "raw_grid_hf_noise_proxy_sigma_au": raw_noise_sigma_au,
        "linf_raw_au_exact": metrics["linf_norm"] * case_raw_std,
        "rms_raw_au_exact": metrics["rms_norm"] * case_raw_std,
        "mae_raw_au_exact": metrics["mae_norm"] * case_raw_std,
        "linf_over_raw_noise_proxy": (
            metrics["linf_norm"] / raw_noise_sigma_norm
            if raw_noise_sigma_norm > 0
            else float("nan")
        ),
        "rms_over_raw_noise_proxy": (
            metrics["rms_norm"] / raw_noise_sigma_norm
            if raw_noise_sigma_norm > 0
            else float("nan")
        ),
        # Affine z-score inversion leaves both metrics unchanged when their
        # data range is transformed consistently.
        "psnr_db_raw_au_exact": metrics["psnr_db_robust_range"],
        "ssim_raw_au_exact": metrics["ssim_axial_prostate"],
        "cohort_raw_linf_proxy_au": metrics["linf_norm"] * cohort_raw_std,
        "cohort_raw_rms_proxy_au": metrics["rms_norm"] * cohort_raw_std,
    }


def _torch_seed(seed: int) -> None:
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def _evaluate_perturbations(
    dataset_key: str,
    attack_ids: list[str],
    *,
    data_dir: Path,
    checkpoint: Path,
    config: dict[str, object],
    normalization_stats: pd.DataFrame,
    eps_n: list[int],
    noise_trials: int,
    apgd_steps: int,
    seed: int,
    max_ssim_slices: int,
    device: torch.device,
    attack_valid_only: bool = False,
) -> tuple[list[dict[str, object]], list[dict[str, object]]]:
    print(f"\n[{dataset_key}] loading clean UNet checkpoint")
    model, div_factors, num_classes = build_model(str(checkpoint), device)
    model.eval()
    class_names = list(DATASETS[dataset_key]["class_names"])
    use_mask = bool(config["use_mask_for_norm"])
    cohort_raw_std = float(config["cohort_raw_fg_std_au"])
    stats_lookup = normalization_stats.set_index("case_id", verify_integrity=True)
    quality_rows: list[dict[str, object]] = []
    dice_rows: list[dict[str, object]] = []

    for case_index, case_id in enumerate(attack_ids, start=1):
        image_np, seg_np, _spacing = load_sample(str(data_dir), case_id)
        clean = np.asarray(image_np[0], dtype=np.float32)
        target = _seg_volume(seg_np)
        valid = _valid_mask(target, use_mask)
        foreground = target > 0
        case_stats = stats_lookup.loc[case_id]
        case_raw_mean = float(case_stats["raw_norm_mean_au"])
        case_raw_std = float(case_stats["raw_norm_std_au"])
        raw_noise_sigma_au = float(case_stats["raw_grid_hf_noise_proxy_sigma_au"])
        raw_noise_sigma_norm = raw_noise_sigma_au / case_raw_std
        raw_zero_normalized = -case_raw_mean / case_raw_std
        x_padded, y_padded, orig_shape = _pad_attack_inputs(
            image_np, seg_np, div_factors, device
        )
        clean_pred = _predict_tensor(model, x_padded, orig_shape)
        quality_rows.append(
            _quality_record(
                dataset_key,
                case_id,
                0,
                "clean",
                0,
                clean,
                clean,
                valid_mask=valid,
                foreground=foreground,
                raw_noise_sigma_norm=raw_noise_sigma_norm,
                raw_noise_sigma_au=raw_noise_sigma_au,
                case_raw_mean=case_raw_mean,
                case_raw_std=case_raw_std,
                cohort_raw_std=cohort_raw_std,
                max_ssim_slices=max_ssim_slices,
            )
        )
        dice_rows.extend(
            _dice_records(
                dataset_key,
                case_id,
                0,
                "clean",
                0,
                clean_pred,
                target,
                class_names,
            )
        )

        for epsilon_n in [value for value in eps_n if value > 0]:
            epsilon = epsilon_n / 255.0
            attack_seed = stable_seed(
                dataset_key, case_id, "apgd_bce", epsilon_n, base_seed=seed
            )
            _torch_seed(attack_seed)
            with torch.enable_grad():
                x_adv = apgd_bce(
                    model,
                    x_padded,
                    y_padded,
                    epsilon,
                    num_classes,
                    n_steps=apgd_steps,
                    # deployed pipeline regenerates ignore-labelled voxels
                    # (zero-filled margin on zones, padding) after any attack
                    perturbation_mask=(
                        (y_padded != IGNORE_LABEL) if attack_valid_only else None
                    ),
                )
            x_adv = _restore_artificial_padding(x_adv, x_padded, orig_shape)
            adv = unpad(x_adv, orig_shape).squeeze(0).squeeze(0).cpu().numpy()
            adv_pred = _predict_tensor(model, x_adv, orig_shape)
            adv_quality = _quality_record(
                dataset_key,
                case_id,
                epsilon_n,
                "apgd_bce",
                0,
                clean,
                adv,
                valid_mask=valid,
                foreground=foreground,
                raw_noise_sigma_norm=raw_noise_sigma_norm,
                raw_noise_sigma_au=raw_noise_sigma_au,
                case_raw_mean=case_raw_mean,
                case_raw_std=case_raw_std,
                cohort_raw_std=cohort_raw_std,
                max_ssim_slices=max_ssim_slices,
            )
            quality_rows.append(adv_quality)
            dice_rows.extend(
                _dice_records(
                    dataset_key,
                    case_id,
                    epsilon_n,
                    "apgd_bce",
                    0,
                    adv_pred,
                    target,
                    class_names,
                )
            )
            target_rms = float(adv_quality["rms_norm"])

            for trial in range(noise_trials):
                trial_seed = stable_seed(
                    dataset_key, case_id, epsilon_n, trial, base_seed=seed
                )
                gaussian = matched_gaussian_noise(
                    clean,
                    valid_mask=valid,
                    epsilon=epsilon,
                    target_rms=target_rms,
                    rng=np.random.default_rng(trial_seed),
                )
                rician = matched_rician_proxy_noise(
                    clean,
                    valid_mask=valid,
                    epsilon=epsilon,
                    target_rms=target_rms,
                    rng=np.random.default_rng(trial_seed + 1),
                    raw_zero_normalized=raw_zero_normalized,
                )
                for perturbation, altered in (
                    ("gaussian_rms_matched", gaussian),
                    ("rician_proxy_rms_matched", rician),
                ):
                    quality_rows.append(
                        _quality_record(
                            dataset_key,
                            case_id,
                            epsilon_n,
                            perturbation,
                            trial,
                            clean,
                            altered,
                            valid_mask=valid,
                            foreground=foreground,
                            raw_noise_sigma_norm=raw_noise_sigma_norm,
                            raw_noise_sigma_au=raw_noise_sigma_au,
                            case_raw_mean=case_raw_mean,
                            case_raw_std=case_raw_std,
                            cohort_raw_std=cohort_raw_std,
                            max_ssim_slices=max_ssim_slices,
                        )
                    )
                    pred = _predict_numpy(model, altered, div_factors, device)
                    dice_rows.extend(
                        _dice_records(
                            dataset_key,
                            case_id,
                            epsilon_n,
                            perturbation,
                            trial,
                            pred,
                            target,
                            class_names,
                        )
                    )

            del x_adv
            if device.type == "cuda":
                torch.cuda.empty_cache()
            print(
                f"  [{dataset_key} {case_index}/{len(attack_ids)}] {case_id} "
                f"epsilon={epsilon_n}/255 rms={target_rms:.5f}"
            )

        del x_padded, y_padded
        if device.type == "cuda":
            torch.cuda.empty_cache()

    del model
    gc.collect()
    if device.type == "cuda":
        torch.cuda.empty_cache()
    return quality_rows, dice_rows


def _case_averaged_summary(
    frame: pd.DataFrame,
    *,
    group_columns: list[str],
    metrics: Iterable[str],
) -> pd.DataFrame:
    metrics = list(metrics)
    case_group = [*group_columns, "case_id"]
    by_case = frame.groupby(case_group, as_index=False)[metrics].mean()
    output: list[dict[str, object]] = []
    for keys, group in by_case.groupby(group_columns, sort=True):
        if not isinstance(keys, tuple):
            keys = (keys,)
        row: dict[str, object] = dict(zip(group_columns, keys))
        row["n_cases"] = int(group["case_id"].nunique())
        for metric in metrics:
            values = group[metric].to_numpy(dtype=np.float64)
            values = values[np.isfinite(values)]
            if not len(values):
                row[f"{metric}_mean"] = float("nan")
                row[f"{metric}_median"] = float("nan")
                row[f"{metric}_q05"] = float("nan")
                row[f"{metric}_q95"] = float("nan")
                continue
            row[f"{metric}_mean"] = float(values.mean())
            row[f"{metric}_median"] = float(np.median(values))
            row[f"{metric}_q05"] = float(np.quantile(values, 0.05))
            row[f"{metric}_q95"] = float(np.quantile(values, 0.95))
        output.append(row)
    return pd.DataFrame(output)


def _plot_quality(summary: pd.DataFrame, out_dir: Path) -> None:
    metrics = [
        ("psnr_db_robust_range", "PSNR (dB; robust signal range)"),
        ("ssim_axial_prostate", "Axial SSIM (prostate slices)"),
        (
            "rms_over_raw_noise_proxy",
            "Perturbation RMS / raw-image noise proxy (median)",
        ),
    ]
    datasets = [key for key in ("wg", "zones") if key in set(summary["dataset"])]
    fig, axes = plt.subplots(
        len(datasets), len(metrics), figsize=(16, 5 * len(datasets))
    )
    axes = np.atleast_2d(axes)
    for row_index, dataset_key in enumerate(datasets):
        for col_index, (metric, ylabel) in enumerate(metrics):
            ax = axes[row_index, col_index]
            for perturbation in (
                "apgd_bce",
                "gaussian_rms_matched",
                "rician_proxy_rms_matched",
            ):
                group = summary[
                    (summary["dataset"] == dataset_key)
                    & (summary["perturbation"] == perturbation)
                ].sort_values("epsilon_n")
                if group.empty:
                    continue
                x = group["epsilon_n"].to_numpy(dtype=float)
                center_stat = (
                    "median" if metric == "rms_over_raw_noise_proxy" else "mean"
                )
                center = group[f"{metric}_{center_stat}"].to_numpy(dtype=float)
                q05 = group[f"{metric}_q05"].to_numpy(dtype=float)
                q95 = group[f"{metric}_q95"].to_numpy(dtype=float)
                ax.plot(
                    x,
                    center,
                    marker="o",
                    linewidth=2,
                    color=PERTURBATION_COLORS[perturbation],
                    label=PERTURBATION_LABELS[perturbation],
                )
                ax.fill_between(
                    x,
                    q05,
                    q95,
                    color=PERTURBATION_COLORS[perturbation],
                    alpha=0.12,
                )
            ax.set_title(f"{dataset_key.upper()} — {ylabel}")
            ax.set_xlabel("Normalized-space epsilon, ε_norm = n/255 (legacy n)")
            ax.set_ylabel(ylabel)
            ax.grid(alpha=0.25)
            ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(out_dir / "image_quality_vs_epsilon.png", dpi=220, bbox_inches="tight")
    plt.close(fig)


def _plot_dice(summary: pd.DataFrame, out_dir: Path) -> None:
    macro = summary[summary["class"] == "macro"]
    datasets = [key for key in ("wg", "zones") if key in set(macro["dataset"])]
    fig, axes = plt.subplots(1, len(datasets), figsize=(7.5 * len(datasets), 5.5))
    axes = np.atleast_1d(axes)
    for ax, dataset_key in zip(axes, datasets):
        clean = macro[
            (macro["dataset"] == dataset_key) & (macro["perturbation"] == "clean")
        ]
        clean_mean = (
            float(clean["dice_mean"].iloc[0]) if not clean.empty else float("nan")
        )
        for perturbation in (
            "apgd_bce",
            "gaussian_rms_matched",
            "rician_proxy_rms_matched",
        ):
            group = macro[
                (macro["dataset"] == dataset_key)
                & (macro["perturbation"] == perturbation)
            ].sort_values("epsilon_n")
            if group.empty:
                continue
            x = np.r_[0.0, group["epsilon_n"].to_numpy(dtype=float)]
            mean = np.r_[clean_mean, group["dice_mean"].to_numpy(dtype=float)]
            q05 = np.r_[clean_mean, group["dice_q05"].to_numpy(dtype=float)]
            q95 = np.r_[clean_mean, group["dice_q95"].to_numpy(dtype=float)]
            ax.plot(
                x,
                mean,
                marker="o",
                linewidth=2,
                color=PERTURBATION_COLORS[perturbation],
                label=PERTURBATION_LABELS[perturbation],
            )
            ax.fill_between(
                x,
                q05,
                q95,
                color=PERTURBATION_COLORS[perturbation],
                alpha=0.12,
            )
        ax.set_title(f"{dataset_key.upper()} — macro Dice")
        ax.set_xlabel("Normalized-space epsilon, ε_norm = n/255 (legacy n)")
        ax.set_ylabel("Dice")
        ax.set_ylim(0.0, 1.02)
        ax.grid(alpha=0.25)
        ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(
        out_dir / "dice_adversarial_vs_matched_noise.png", dpi=220, bbox_inches="tight"
    )
    plt.close(fig)


def _plot_mapping(mapping: pd.DataFrame, out_dir: Path) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(13, 5))
    for dataset_key, group in mapping.groupby("dataset"):
        group = group.sort_values("epsilon_n")
        axes[0].plot(
            group["epsilon_n"],
            group["epsilon_percent_within_case_sigma"],
            marker="o",
            linewidth=2,
            label=dataset_key.upper(),
        )
        (raw_line,) = axes[1].plot(
            group["epsilon_n"],
            group["exact_case_raw_linf_median_au"],
            marker="o",
            linewidth=2,
            label=dataset_key.upper(),
        )
        axes[1].fill_between(
            group["epsilon_n"].to_numpy(dtype=float),
            group["exact_case_raw_linf_q05_au"].to_numpy(dtype=float),
            group["exact_case_raw_linf_q95_au"].to_numpy(dtype=float),
            color=raw_line.get_color(),
            alpha=0.12,
        )
    axes[0].set_title("Exact normalized-space interpretation")
    axes[0].set_ylabel("L-inf budget (% of within-case sigma)")
    axes[1].set_title("Exact per-case raw-AU inversion (median; 5–95%)")
    axes[1].set_ylabel("L-inf budget (original MRI arbitrary units)")
    for ax in axes:
        ax.set_xlabel("Normalized-space epsilon, ε_norm = n/255 (legacy n)")
        ax.grid(alpha=0.25)
        ax.legend()
    fig.tight_layout()
    fig.savefig(out_dir / "epsilon_scale_mapping.png", dpi=220, bbox_inches="tight")
    plt.close(fig)


def _markdown_table(headers: list[str], rows: list[list[object]]) -> list[str]:
    out = ["| " + " | ".join(headers) + " |"]
    out.append("| " + " | ".join(["---"] * len(headers)) + " |")
    for row in rows:
        out.append("| " + " | ".join(str(value) for value in row) + " |")
    return out


def _write_report(
    out_dir: Path,
    normalization_summary: pd.DataFrame,
    mapping: pd.DataFrame,
    quality_summary: pd.DataFrame,
    dice_summary: pd.DataFrame,
    *,
    stats_cases: int,
    attack_cases: int,
    noise_trials: int,
    apgd_steps: int,
    seed: int,
) -> None:
    lines = [
        "# Physical interpretation of adversarial epsilon",
        "",
        "## Bottom line",
        "",
        "Both datasets use per-case `ZScoreNormalization`. Therefore epsilon is an absolute "
        "distance in normalized model-input space: `epsilon_norm = n/255` equals "
        "`100*n/255` percent of the within-case pre-resampling standard deviation. MRI "
        "intensities are arbitrary units, not a standardized physical scale.",
        "",
        "The raw training NIfTI files are available. This audit reproduces nnU-Net's exact "
        "per-case crop and mask, then recomputes the mean and standard deviation used before "
        "resampling. Consequently `delta_raw_AU = delta_norm * sigma_case_raw_AU` is an exact "
        "intensity-scale inversion (on the model grid). The AU scale remains case- and scanner-"
        "dependent and is not comparable like a calibrated physical unit.",
        "",
        "No noise-only scan, repeated acquisition, DICOM noise map, or complex k-space data is "
        "available, so true acquisition-noise sigma cannot be measured. The report instead "
        "uses an explicitly labeled raw-image high-frequency noise proxy; edges and texture can "
        "inflate it.",
        "",
        "## Normalization verification",
        "",
    ]
    norm_rows: list[list[object]] = []
    for _, row in normalization_summary.iterrows():
        norm_rows.append(
            [
                str(row["dataset"]).upper(),
                int(row["cases_checked"]),
                str(row["normalization_scheme"]),
                bool(row["use_mask_for_norm"]),
                f"{row['median_normalized_mean']:.3e}",
                f"{row['median_normalized_std']:.5f}",
                f"{row['median_raw_norm_mean_au']:.2f}",
                (
                    f"{row['median_raw_norm_std_au']:.2f} "
                    f"[{row['q05_raw_norm_std_au']:.2f}, {row['q95_raw_norm_std_au']:.2f}]"
                ),
                f"{row['median_raw_grid_hf_noise_proxy_sigma_au']:.2f}",
            ]
        )
    lines.extend(
        _markdown_table(
            [
                "Dataset",
                "Cases checked",
                "Scheme",
                "Mask normalization",
                "Median mean",
                "Median std",
                "Median raw mean (AU)",
                "Raw std median [Q05, Q95] (AU)",
                "Raw HF-noise proxy sigma (AU)",
            ],
            norm_rows,
        )
    )
    lines.extend(["", "## Epsilon mapping", ""])
    map_rows: list[list[object]] = []
    for _, row in mapping.iterrows():
        map_rows.append(
            [
                str(row["dataset"]).upper(),
                int(row["epsilon_n"]),
                f"{row['epsilon_norm']:.5f}",
                f"{row['epsilon_percent_within_case_sigma']:.3f}%",
                (
                    f"{row['exact_case_raw_linf_median_au']:.2f} "
                    f"[{row['exact_case_raw_linf_q05_au']:.2f}, "
                    f"{row['exact_case_raw_linf_q95_au']:.2f}]"
                ),
                f"{row['epsilon_over_raw_noise_proxy_median']:.2f}",
            ]
        )
    lines.extend(
        _markdown_table(
            [
                "Dataset",
                "Normalized-space grid (legacy n/255)",
                "epsilon_norm",
                "% of case sigma",
                "Exact raw-AU L-inf median [Q05, Q95]",
                "L-inf / raw-noise proxy (median)",
            ],
            map_rows,
        )
    )

    lines.extend(["", "## APGD versus magnitude-matched noise", ""])
    compare = quality_summary[
        (quality_summary["epsilon_n"].isin([16, 32]))
        & (quality_summary["perturbation"] != "clean")
    ]
    quality_rows: list[list[object]] = []
    for _, row in compare.iterrows():
        quality_rows.append(
            [
                str(row["dataset"]).upper(),
                int(row["epsilon_n"]),
                PERTURBATION_LABELS[str(row["perturbation"])],
                f"{row['rms_norm_median']:.5f}",
                (
                    f"{row['rms_raw_au_exact_median']:.2f} "
                    f"[{row['rms_raw_au_exact_q05']:.2f}, "
                    f"{row['rms_raw_au_exact_q95']:.2f}]"
                ),
                f"{row['psnr_db_robust_range_mean']:.2f}",
                f"{row['ssim_axial_prostate_mean']:.5f}",
                f"{row['rms_over_raw_noise_proxy_median']:.2f}",
            ]
        )
    lines.extend(
        _markdown_table(
            [
                "Dataset",
                "Normalized epsilon (legacy n/255)",
                "Perturbation",
                "RMS median (normalized)",
                "RMS exact raw-AU median [Q05, Q95]",
                "PSNR dB",
                "SSIM",
                "RMS / raw-noise proxy (median)",
            ],
            quality_rows,
        )
    )

    lines.extend(["", "## Segmentation response", ""])
    dice_compare = dice_summary[
        (dice_summary["epsilon_n"].isin([0, 16, 32]))
        & (dice_summary["class"] == "macro")
    ]
    dice_rows: list[list[object]] = []
    for _, row in dice_compare.iterrows():
        dice_rows.append(
            [
                str(row["dataset"]).upper(),
                int(row["epsilon_n"]),
                PERTURBATION_LABELS[str(row["perturbation"])],
                f"{row['dice_mean']:.4f}",
                f"{row['dice_q05']:.4f}",
                int(row["n_cases"]),
            ]
        )
    lines.extend(
        _markdown_table(
            [
                "Dataset",
                "Normalized epsilon (legacy n/255)",
                "Perturbation",
                "Mean Dice",
                "Q05 Dice",
                "Cases",
            ],
            dice_rows,
        )
    )

    lines.extend(
        [
            "",
            "## Required labeling in future reports",
            "",
            "Use labels such as `epsilon_norm=0.06275 (legacy 16/255; 6.275% of "
            "within-case sigma)`. Do not call 16/255 sixteen MRI gray levels. Keep the "
            "legacy grid for continuity, but every axis and table must identify it as "
            "normalized-space epsilon.",
            "",
            "## Limitations",
            "",
            "- Exact raw-AU values use each case's reproduced pre-resampling z-score standard "
            "deviation. They are exact intensity-scale inversions, but MRI AU are not "
            "standardized across cases, scanners, or reconstructions.",
            "- `cohort_raw_*_proxy_au` is retained in the CSV only for comparison with the old "
            "dataset-fingerprint interpretation; use `*_raw_au_exact` for conclusions.",
            "- The Immerkaer high-frequency estimator is an image-derived acquisition-noise "
            "proxy, not a direct scanner noise measurement. Edges and fine anatomy can inflate "
            "it. A noise-only or repeated acquisition is needed for a measured sigma.",
            "- Rician noise uses the exact normalized location of raw intensity zero. It remains "
            "a proxy where stored images contain negative/processed intensities or otherwise "
            "violate the magnitude-MRI Rician model.",
            "- PSNR uses each clean volume's 0.5th-to-99.5th percentile range. PSNR and SSIM "
            "are invariant to an affine raw/normalized conversion when the data range is "
            "scaled consistently.",
            "- APGD quality measurements use deterministic, batch-size-one runs of the report's "
            "custom APGD-BCE implementation; this is a scale-calibration experiment, not an "
            "independent robustness certification.",
            "- Dice and image-quality comparisons use 12 cases per dataset, stratified over the "
            "raw-image noise proxy. They are an experiment panel, not a population estimate; "
            "repeat on a held-out external cohort for inferential claims.",
            "",
            "## Run configuration",
            "",
            (
                "- Normalization cases per dataset: all available"
                if stats_cases <= 0
                else f"- Normalization cases requested per dataset: {stats_cases}"
            ),
            f"- Attack cases requested per dataset: {attack_cases}",
            f"- Matched-noise trials: {noise_trials}",
            f"- APGD steps: {apgd_steps}",
            f"- Seed: {seed}",
            "",
        ]
    )
    (out_dir / "epsilon_physical_interpretation.md").write_text(
        "\n".join(lines), encoding="utf-8"
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Calibrate normalized-space adversarial epsilon against MRI image scales"
    )
    parser.add_argument(
        "--datasets", nargs="+", choices=["wg", "zones"], default=["wg", "zones"]
    )
    parser.add_argument("--eps-n", nargs="+", type=int, default=DEFAULT_EPS_N)
    parser.add_argument(
        "--stats-cases",
        type=int,
        default=0,
        help="cases per dataset for normalization audit; 0 checks every case (default)",
    )
    parser.add_argument("--attack-cases", type=int, default=12)
    parser.add_argument("--noise-trials", type=int, default=3)
    parser.add_argument("--apgd-steps", type=int, default=20)
    parser.add_argument("--max-ssim-slices", type=int, default=5)
    parser.add_argument("--seed", type=int, default=20260721)
    parser.add_argument("--device", type=str, default=None)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUT_DIR)
    parser.add_argument(
        "--attack-valid-only",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="confine the APGD perturbation to non-ignore voxels (zones: the zero-filled "
        "margin outside the dilated gland, and the padding). The published run went "
        "without it.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if any(value < 0 for value in args.eps_n):
        raise ValueError("epsilon numerators must be non-negative")
    if args.noise_trials < 1:
        raise ValueError("--noise-trials must be at least 1")
    if args.apgd_steps < 1:
        raise ValueError("--apgd-steps must be at least 1")

    out_dir = args.output_dir.resolve()
    out_dir.mkdir(parents=True, exist_ok=True)
    device = (
        torch.device(args.device)
        if args.device
        else torch.device("cuda" if torch.cuda.is_available() else "cpu")
    )
    print(f"Device: {device}")
    print(f"Output: {out_dir}")

    all_stats: list[dict[str, object]] = []
    norm_summaries: list[dict[str, object]] = []
    mapping_rows: list[dict[str, object]] = []
    all_quality: list[dict[str, object]] = []
    all_dice: list[dict[str, object]] = []

    for dataset_key in args.datasets:
        paths = _dataset_paths(dataset_key)
        config = _normalization_config(paths["plans"])
        case_ids = _available_case_ids(paths["data"])
        stats_ids = _sample_ids(
            case_ids,
            args.stats_cases,
            dataset_key=dataset_key,
            seed=args.seed,
        )
        print(
            f"\n[{dataset_key}] checking normalization on {len(stats_ids)}/{len(case_ids)} cases"
        )
        dataset_stats: list[dict[str, object]] = []
        for index, case_id in enumerate(stats_ids, start=1):
            dataset_stats.append(
                _normalization_record(
                    dataset_key,
                    case_id,
                    paths["data"],
                    raw_dir=paths["raw"],
                    transpose_forward=list(config["transpose_forward"]),
                    use_mask_for_norm=bool(config["use_mask_for_norm"]),
                    max_slices=args.max_ssim_slices,
                )
            )
            if index % 25 == 0 or index == len(stats_ids):
                print(f"  normalization {index}/{len(stats_ids)}")
        stats_frame = pd.DataFrame(dataset_stats)
        all_stats.extend(dataset_stats)
        norm_summaries.append(
            _summarize_normalization(
                dataset_key,
                stats_frame,
                total_cases=len(case_ids),
                config=config,
                raw_dir=paths["raw"],
            )
        )
        for epsilon_n in sorted(set(args.eps_n)):
            mapping_rows.append(
                _epsilon_mapping_record(dataset_key, epsilon_n, stats_frame, config)
            )

        attack_ids = _stratified_attack_ids(stats_frame, args.attack_cases)
        print(
            f"[{dataset_key}] perturbation cases stratified by HF-noise proxy: "
            f"{len(attack_ids)}"
        )
        quality, dice = _evaluate_perturbations(
            dataset_key,
            attack_ids,
            data_dir=paths["data"],
            checkpoint=paths["checkpoint"],
            config=config,
            normalization_stats=stats_frame,
            eps_n=sorted(set(args.eps_n)),
            noise_trials=args.noise_trials,
            apgd_steps=args.apgd_steps,
            seed=args.seed,
            max_ssim_slices=args.max_ssim_slices,
            device=device,
            attack_valid_only=bool(args.attack_valid_only),
        )
        all_quality.extend(quality)
        all_dice.extend(dice)

    stats_df = pd.DataFrame(all_stats)
    norm_summary_df = pd.DataFrame(norm_summaries)
    mapping_df = pd.DataFrame(mapping_rows)
    quality_df = pd.DataFrame(all_quality)
    dice_df = pd.DataFrame(all_dice)

    quality_metrics = [
        "linf_norm",
        "rms_norm",
        "mae_norm",
        "linf_percent_case_sigma",
        "rms_percent_case_sigma",
        "psnr_db_robust_range",
        "ssim_axial_prostate",
        "linf_over_hf_noise",
        "rms_over_hf_noise",
        "epsilon_saturation_fraction",
        "linf_raw_au_exact",
        "rms_raw_au_exact",
        "mae_raw_au_exact",
        "linf_over_raw_noise_proxy",
        "rms_over_raw_noise_proxy",
        "psnr_db_raw_au_exact",
        "ssim_raw_au_exact",
        "cohort_raw_linf_proxy_au",
        "cohort_raw_rms_proxy_au",
    ]
    quality_summary_df = _case_averaged_summary(
        quality_df,
        group_columns=["dataset", "epsilon_n", "epsilon_norm", "perturbation"],
        metrics=quality_metrics,
    )
    dice_summary_df = _case_averaged_summary(
        dice_df,
        group_columns=["dataset", "epsilon_n", "epsilon_norm", "perturbation", "class"],
        metrics=["dice"],
    )

    outputs = {
        "normalization_per_case.csv": stats_df,
        "normalization_summary.csv": norm_summary_df,
        "epsilon_mapping.csv": mapping_df,
        "perturbation_quality_per_case.csv": quality_df,
        "perturbation_quality_summary.csv": quality_summary_df,
        "segmentation_dice_per_case.csv": dice_df,
        "segmentation_dice_summary.csv": dice_summary_df,
    }
    for filename, frame in outputs.items():
        frame.to_csv(out_dir / filename, index=False)
        print(f"Saved {out_dir / filename}")

    _plot_mapping(mapping_df, out_dir)
    _plot_quality(quality_summary_df, out_dir)
    _plot_dice(dice_summary_df, out_dir)
    _write_report(
        out_dir,
        norm_summary_df,
        mapping_df,
        quality_summary_df,
        dice_summary_df,
        stats_cases=args.stats_cases,
        attack_cases=args.attack_cases,
        noise_trials=args.noise_trials,
        apgd_steps=args.apgd_steps,
        seed=args.seed,
    )
    print(f"Saved {out_dir / 'epsilon_physical_interpretation.md'}")
    print("\nCalibration complete.")


if __name__ == "__main__":
    main()
