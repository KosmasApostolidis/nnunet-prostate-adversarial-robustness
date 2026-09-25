"""Measure FGSM/PGD/APGD PSNR and SSIM on every preprocessed case.

The original full-cohort adversarial evaluation retained segmentation metrics
but not adversarial images, PSNR, or SSIM. Those image-quality metrics cannot be
reconstructed from Dice CSV files, so this script regenerates perturbations for
the clean fold-all UNet and writes each case/attack/epsilon row immediately.

Only image-quality metrics are computed; matched-noise controls and segmentation
inference are intentionally omitted to keep the full-cohort pass tractable.

All attacks maximize the same masked one-hot BCE loss and use the same L-infinity
budget and per-case intensity bounds. FGSM uses one gradient-sign step. PGD and
APGD use deterministic case-specific random starts so interrupted runs resume
reproducibly and batching does not couple cases.
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
import torch.nn.functional as F


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from experiments.adv_rob_eval_nnunet_nnunetrecenc import (  # noqa: E402
    DATASETS,
    IGNORE_LABEL,
    NNUNET_PATHS,
    TRAINERS,
    _pad_to,
    _plan_batches,
    build_model,
    load_sample,
)
from mri_prostate_seg.experiments.epsilon_calibration import (  # noqa: E402
    perturbation_quality,
    stable_seed,
)


DEFAULT_OUTPUT_DIR = (
    REPO_ROOT
    / "results"
    / "adv_rob_eval_results"
    / "image_quality_analysis"
    / "all_cases_adversarial_quality"
)
DEFAULT_EPS_N = [2, 4, 8, 16, 32]
ATTACK_LABELS = {
    "fgsm": "FGSM-BCE",
    "pgd": "PGD-BCE",
    "apgd": "APGD-BCE",
}
FIELDNAMES = [
    "dataset",
    "case_id",
    "epsilon_n",
    "epsilon_norm",
    "epsilon_percent_case_sigma",
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
    "case_raw_norm_mean_au",
    "case_raw_norm_std_au",
    "linf_raw_au_exact",
    "rms_raw_au_exact",
    "mae_raw_au_exact",
    "max_ssim_slices",
]


def _dataset_paths(
    dataset_key: str, trainer_dir: str = TRAINERS["unet"]
) -> dict[str, Path]:
    """``trainer_dir`` is the nnU-Net results folder name (``<Trainer>__<Plans>__<config>``)
    whose ``fold_all/checkpoint_final.pth`` is attacked; default = the clean model."""
    config = DATASETS[dataset_key]
    dataset_name = config["dataset_name"]
    root = Path(NNUNET_PATHS)
    return {
        "data": root / "nnUNet_preprocessed" / dataset_name / config["data_subdir"],
        "plans": root / "nnUNet_preprocessed" / dataset_name / "nnUNetPlans.json",
        "checkpoint": root
        / "nnUNet_results"
        / dataset_name
        / trainer_dir
        / "fold_all"
        / "checkpoint_final.pth",
    }


def _available_case_ids(data_dir: Path) -> list[str]:
    return sorted(
        path.name[: -len(".b2nd")]
        for path in data_dir.glob("*.b2nd")
        if not path.name.endswith("_seg.b2nd")
    )


def _use_mask_for_norm(plans_path: Path) -> bool:
    plans = json.loads(plans_path.read_text())
    return bool(plans["configurations"]["3d_fullres"]["use_mask_for_norm"][0])


def _seg_volume(seg_np: np.ndarray) -> np.ndarray:
    segmentation = np.asarray(seg_np).squeeze()
    if segmentation.ndim == 4:
        segmentation = segmentation[0]
    if segmentation.ndim != 3:
        raise ValueError(f"expected 3-D segmentation, got {segmentation.shape}")
    return segmentation


def _samplewise_bounds(x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    dimensions = tuple(range(1, x.ndim))
    return (
        x.amin(dim=dimensions, keepdim=True),
        x.amax(dim=dimensions, keepdim=True),
    )


def _per_sample_bce(
    logits: torch.Tensor, target: torch.Tensor, num_classes: int
) -> torch.Tensor:
    """Return the evaluator's masked BCE loss independently for each sample."""

    t = target.squeeze(1).long()
    valid = (t != IGNORE_LABEL).unsqueeze(1).float()
    t_safe = t.clamp_min(0)
    one_hot = torch.zeros_like(logits)
    one_hot.scatter_(1, t_safe.unsqueeze(1), 1.0)
    one_hot = one_hot * valid
    elementwise = F.binary_cross_entropy_with_logits(logits, one_hot, reduction="none")
    reduction_dims = tuple(range(1, elementwise.ndim))
    denominator = valid.expand_as(elementwise).sum(dim=reduction_dims).clamp(min=1.0)
    return (elementwise * valid).sum(dim=reduction_dims) / denominator


def _confine(adv: torch.Tensor, clean: torch.Tensor, mask: torch.Tensor | None) -> torch.Tensor:
    """Zero the perturbation outside ``mask`` (bool, broadcastable to ``adv``).

    ``None`` leaves the attack unconstrained (the campaign default).  The zones
    inputs are zero-filled outside the dilated gland (seg == -1) and zero-padded
    to the model grid; the deployed pipeline regenerates both after any
    image-level attack, so a perturbation there cannot be realised.
    """
    if mask is None:
        return adv
    return torch.where(mask, adv, clean)


def _deterministic_initial_noise(
    x: torch.Tensor,
    epsilon: float,
    seeds: list[int],
) -> torch.Tensor:
    pieces: list[torch.Tensor] = []
    for index, seed in enumerate(seeds):
        generator = torch.Generator(device=x.device)
        generator.manual_seed(int(seed))
        noise = torch.empty_like(x[index : index + 1]).uniform_(
            -epsilon, epsilon, generator=generator
        )
        pieces.append(noise)
    return torch.cat(pieces, dim=0)


def apgd_bce_independent_batch(
    model: torch.nn.Module,
    x: torch.Tensor,
    y: torch.Tensor,
    epsilon: float,
    num_classes: int,
    *,
    n_steps: int,
    seeds: list[int],
    perturbation_mask: torch.Tensor | None = None,
) -> torch.Tensor:
    """APGD-BCE with independent per-sample state and deterministic starts.

    Model forwards are batched, while best-loss tracking and step-size halving
    remain sample-specific. Thus batching is a throughput optimization rather
    than a cross-case attack coupling.
    """

    if len(seeds) != len(x):
        raise ValueError("one deterministic seed is required per sample")
    if epsilon <= 0:
        return x.detach().clone()

    lower, upper = _samplewise_bounds(x)
    clean = x.detach()
    initial_noise = _deterministic_initial_noise(clean, epsilon, seeds)
    adversarial = _confine(
        (clean + initial_noise).clamp(lower, upper), clean, perturbation_mask
    )
    best = adversarial.clone()

    shape = [len(x), *([1] * (x.ndim - 1))]
    alpha = torch.full(shape, 2.0 * epsilon, device=x.device, dtype=x.dtype)
    best_loss = torch.full((len(x),), -float("inf"), device=x.device)
    sum_loss = torch.zeros((len(x),), device=x.device)
    count = 0

    checkpoints: list[int] = []
    width = max(1, math.ceil(0.22 * n_steps))
    checkpoints.append(width)
    while checkpoints[-1] < n_steps:
        remaining = n_steps - checkpoints[-1]
        width = max(1, math.ceil(0.22 * remaining))
        checkpoints.append(min(checkpoints[-1] + width, n_steps))
    checkpoint_index = 0

    for step in range(1, n_steps + 1):
        requested = adversarial.clone().requires_grad_(True)
        per_sample_loss = _per_sample_bce(model(requested), y, num_classes)
        gradient = torch.autograd.grad(per_sample_loss.sum(), requested)[0].detach()

        with torch.no_grad():
            loss_values = per_sample_loss.detach()
            sum_loss += loss_values
            count += 1
            gradient_step = (
                (adversarial + alpha * gradient.sign())
                .clamp(clean - epsilon, clean + epsilon)
                .clamp(lower, upper)
            )
            updated = _confine(
                (0.75 * gradient_step + 0.25 * adversarial)
                .clamp(clean - epsilon, clean + epsilon)
                .clamp(lower, upper),
                clean,
                perturbation_mask,
            )
            improved = loss_values > best_loss
            best = torch.where(improved.reshape(shape), updated, best)
            best_loss = torch.maximum(best_loss, loss_values)
            adversarial = updated

        if (
            checkpoint_index < len(checkpoints)
            and step == checkpoints[checkpoint_index]
        ):
            average = sum_loss / max(count, 1)
            halve = average <= best_loss * 0.95
            alpha = torch.where(halve.reshape(shape), alpha * 0.5, alpha)
            sum_loss.zero_()
            count = 0
            checkpoint_index += 1

    return best.detach()


def fgsm_bce_independent_batch(
    model: torch.nn.Module,
    x: torch.Tensor,
    y: torch.Tensor,
    epsilon: float,
    num_classes: int,
    *,
    perturbation_mask: torch.Tensor | None = None,
) -> torch.Tensor:
    """One-step FGSM-BCE with an independent loss contribution per sample."""

    if epsilon <= 0:
        return x.detach().clone()

    clean = x.detach()
    lower, upper = _samplewise_bounds(clean)
    requested = clean.clone().requires_grad_(True)
    per_sample_loss = _per_sample_bce(model(requested), y, num_classes)
    gradient = torch.autograd.grad(per_sample_loss.sum(), requested)[0].detach()
    return _confine(
        (clean + epsilon * gradient.sign())
        .clamp(clean - epsilon, clean + epsilon)
        .clamp(lower, upper),
        clean,
        perturbation_mask,
    ).detach()


def pgd_bce_independent_batch(
    model: torch.nn.Module,
    x: torch.Tensor,
    y: torch.Tensor,
    epsilon: float,
    num_classes: int,
    *,
    n_steps: int,
    seeds: list[int],
    perturbation_mask: torch.Tensor | None = None,
) -> torch.Tensor:
    """Fixed-step PGD-BCE with deterministic, independent random starts."""

    if len(seeds) != len(x):
        raise ValueError("one deterministic seed is required per sample")
    if epsilon <= 0:
        return x.detach().clone()
    if n_steps < 1:
        raise ValueError("n_steps must be positive")

    clean = x.detach()
    lower, upper = _samplewise_bounds(clean)
    initial_noise = _deterministic_initial_noise(clean, epsilon, seeds)
    adversarial = _confine(
        (clean + initial_noise).clamp(lower, upper), clean, perturbation_mask
    )
    step_size = epsilon / n_steps

    for _step in range(n_steps):
        requested = adversarial.detach().requires_grad_(True)
        per_sample_loss = _per_sample_bce(model(requested), y, num_classes)
        gradient = torch.autograd.grad(per_sample_loss.sum(), requested)[0].detach()
        with torch.no_grad():
            adversarial = _confine(
                (requested + step_size * gradient.sign())
                .clamp(clean - epsilon, clean + epsilon)
                .clamp(lower, upper),
                clean,
                perturbation_mask,
            )
    return adversarial.detach()


def _load_normalization_lookup(path: Path) -> pd.DataFrame:
    if not path.is_file():
        raise FileNotFoundError(
            "the all-case normalization audit is required for exact raw-AU conversion: "
            f"{path}"
        )
    frame = pd.read_csv(path)
    return frame.set_index(["dataset", "case_id"], verify_integrity=True)


def _completed_keys(output_csv: Path) -> set[tuple[str, str, int, str]]:
    if not output_csv.is_file() or output_csv.stat().st_size == 0:
        return set()
    frame = pd.read_csv(
        output_csv, usecols=["dataset", "case_id", "epsilon_n", "attack"]
    )
    return {
        (str(row.dataset), str(row.case_id), int(row.epsilon_n), str(row.attack))
        for row in frame.itertuples(index=False)
    }


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


def _quality_row(
    dataset_key: str,
    case_id: str,
    epsilon_n: int,
    attack: str,
    attack_steps: int,
    attack_precision: str,
    attack_seed: int,
    clean: np.ndarray,
    altered: np.ndarray,
    segmentation: np.ndarray,
    *,
    use_mask_for_norm: bool,
    raw_mean: float,
    raw_std: float,
    max_ssim_slices: int,
) -> dict[str, object]:
    valid = (
        segmentation >= 0
        if use_mask_for_norm
        else np.ones_like(segmentation, dtype=bool)
    )
    metrics = perturbation_quality(
        clean,
        altered,
        valid_mask=valid,
        slice_selector=segmentation > 0,
        epsilon=epsilon_n / 255.0,
        max_ssim_slices=max_ssim_slices,
    )
    return {
        "dataset": dataset_key,
        "case_id": case_id,
        "epsilon_n": epsilon_n,
        "epsilon_norm": epsilon_n / 255.0,
        "epsilon_percent_case_sigma": 100.0 * epsilon_n / 255.0,
        "attack": attack,
        "attack_steps": attack_steps,
        "attack_precision": attack_precision,
        "attack_seed": attack_seed,
        "linf_norm": metrics["linf_norm"],
        "rms_norm": metrics["rms_norm"],
        "rms_percent_case_sigma": metrics["rms_percent_case_sigma"],
        "mae_norm": metrics["mae_norm"],
        "robust_signal_range_norm": metrics["robust_signal_range_norm"],
        "psnr_db_robust_range": metrics["psnr_db_robust_range"],
        "ssim_axial_prostate": metrics["ssim_axial_prostate"],
        "epsilon_saturation_fraction": metrics["epsilon_saturation_fraction"],
        "case_raw_norm_mean_au": raw_mean,
        "case_raw_norm_std_au": raw_std,
        "linf_raw_au_exact": metrics["linf_norm"] * raw_std,
        "rms_raw_au_exact": metrics["rms_norm"] * raw_std,
        "mae_raw_au_exact": metrics["mae_norm"] * raw_std,
        "max_ssim_slices": max_ssim_slices,
    }


def evaluate_dataset(
    dataset_key: str,
    *,
    attacks: list[str],
    epsilon_n_values: list[int],
    attack_steps: int,
    batch_size: int,
    use_bf16: bool,
    max_cases: int | None,
    max_ssim_slices: int,
    base_seed: int,
    output_csv: Path,
    normalization: pd.DataFrame,
    completed: set[tuple[str, str, int, str]],
    device: torch.device,
    start_time: float,
    total_rows: int,
    attack_valid_only: bool = False,
) -> None:
    paths = _dataset_paths(dataset_key)
    case_ids = _available_case_ids(paths["data"])
    if max_cases is not None:
        case_ids = case_ids[:max_cases]
    case_ids = [
        case_id
        for case_id in case_ids
        if any(
            (dataset_key, case_id, epsilon_n, ATTACK_LABELS[attack]) not in completed
            for attack in attacks
            for epsilon_n in epsilon_n_values
        )
    ]
    if not case_ids:
        print(f"[{dataset_key}] all requested rows already complete", flush=True)
        return

    use_mask = _use_mask_for_norm(paths["plans"])
    print(f"[{dataset_key}] loading fold-all UNet", flush=True)
    model, div_factors, num_classes = build_model(str(paths["checkpoint"]), device)
    model.eval()
    model.requires_grad_(False)
    batches = _plan_batches(str(paths["data"]), case_ids, div_factors, batch_size)
    print(
        f"[{dataset_key}] {len(case_ids)} remaining cases in {len(batches)} batches "
        f"(requested batch size {batch_size})",
        flush=True,
    )

    for batch_index, batch_case_ids in enumerate(batches, start=1):
        loaded: list[tuple[str, np.ndarray, np.ndarray]] = []
        for case_id in batch_case_ids:
            image_np, seg_np, _spacing = load_sample(str(paths["data"]), case_id)
            loaded.append((case_id, image_np, np.asarray(seg_np)))

        dimensions = [record[1].shape[1:] for record in loaded]
        maximum = [max(shape[index] for shape in dimensions) for index in range(3)]
        target = [
            int(math.ceil(value / factor) * factor)
            for value, factor in zip(maximum, div_factors)
        ]
        images: list[torch.Tensor] = []
        labels: list[torch.Tensor] = []
        original_shapes: list[tuple[int, ...]] = []
        clean_arrays: list[np.ndarray] = []
        segmentations: list[np.ndarray] = []
        for _case_id, image_np, seg_np in loaded:
            image_tensor = torch.from_numpy(image_np[np.newaxis]).to(device)
            label_tensor = torch.from_numpy(seg_np[np.newaxis].astype(np.float32)).to(
                device
            )
            images.append(_pad_to(image_tensor, target))
            labels.append(
                _pad_to(
                    label_tensor,
                    target,
                    mode="constant",
                    value=float(IGNORE_LABEL),
                )
            )
            original_shapes.append(tuple(int(value) for value in image_np.shape[1:]))
            clean_arrays.append(np.asarray(image_np[0], dtype=np.float32))
            segmentations.append(_seg_volume(seg_np))
        image_batch = torch.cat(images, dim=0)
        label_batch = torch.cat(labels, dim=0)
        # Same rule as the attribution drivers' ``attackable_mask``: the deployed
        # pipeline regenerates ignore-labelled voxels after any image-level
        # attack, so a perturbation there cannot be realised.
        perturbation_mask = (
            (label_batch != IGNORE_LABEL) if attack_valid_only else None
        )

        for attack_key in attacks:
            attack_label = ATTACK_LABELS[attack_key]
            steps = 1 if attack_key == "fgsm" else attack_steps
            for epsilon_n in epsilon_n_values:
                missing_indices = [
                    index
                    for index, (case_id, _image, _seg) in enumerate(loaded)
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
                    for case_id, _image, _seg in loaded
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
                rows: list[dict[str, object]] = []
                for index in missing_indices:
                    case_id = loaded[index][0]
                    original = original_shapes[index]
                    slices = tuple(slice(0, value) for value in original)
                    altered = (
                        adversarial_batch[(index, 0, *slices)]
                        .detach()
                        .cpu()
                        .numpy()
                        .astype(np.float32)
                    )
                    normalization_row = normalization.loc[(dataset_key, case_id)]
                    rows.append(
                        _quality_row(
                            dataset_key,
                            case_id,
                            epsilon_n,
                            attack_label,
                            steps,
                            "bf16-autocast" if use_bf16 else "fp32",
                            -1 if attack_key == "fgsm" else seeds[index],
                            clean_arrays[index],
                            altered,
                            segmentations[index],
                            use_mask_for_norm=use_mask,
                            raw_mean=float(normalization_row["raw_norm_mean_au"]),
                            raw_std=float(normalization_row["raw_norm_std_au"]),
                            max_ssim_slices=max_ssim_slices,
                        )
                    )
                _append_rows(output_csv, rows)
                for row in rows:
                    completed.add(
                        (
                            str(row["dataset"]),
                            str(row["case_id"]),
                            int(row["epsilon_n"]),
                            str(row["attack"]),
                        )
                    )
                del adversarial_batch

        completed_count = len(completed)
        elapsed = time.monotonic() - start_time
        rate = completed_count / max(elapsed, 1e-9)
        remaining = max(total_rows - completed_count, 0)
        eta_minutes = remaining / max(rate, 1e-9) / 60.0
        print(
            f"[{dataset_key}] batch {batch_index}/{len(batches)}; "
            f"rows {completed_count}/{total_rows}; ETA {eta_minutes:.1f} min",
            flush=True,
        )
        del image_batch, label_batch, images, labels
        if device.type == "cuda":
            torch.cuda.empty_cache()

    del model
    gc.collect()
    if device.type == "cuda":
        torch.cuda.empty_cache()


def _write_summary(output_csv: Path, summary_csv: Path) -> None:
    frame = pd.read_csv(output_csv)
    rows: list[dict[str, object]] = []
    for (dataset, attack, epsilon_n), group in frame.groupby(
        ["dataset", "attack", "epsilon_n"]
    ):
        row: dict[str, object] = {
            "dataset": dataset,
            "attack": attack,
            "epsilon_n": int(epsilon_n),
            "epsilon_norm": float(group["epsilon_norm"].iloc[0]),
            "n_cases": int(group["case_id"].nunique()),
        }
        for metric in (
            "rms_norm",
            "rms_percent_case_sigma",
            "psnr_db_robust_range",
            "ssim_axial_prostate",
        ):
            values = group[metric].to_numpy(dtype=float)
            row[f"{metric}_mean"] = float(np.mean(values))
            row[f"{metric}_median"] = float(np.median(values))
            row[f"{metric}_q05"] = float(np.quantile(values, 0.05))
            row[f"{metric}_q95"] = float(np.quantile(values, 0.95))
        rows.append(row)
    pd.DataFrame(rows).sort_values(["dataset", "attack", "epsilon_n"]).to_csv(
        summary_csv, index=False
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Evaluate FGSM/PGD/APGD PSNR/SSIM for every preprocessed MRI case"
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
        help="attacks to evaluate (default: fgsm pgd apgd)",
    )
    parser.add_argument("--attack-steps", type=int, default=20)
    parser.add_argument("--batch-size", type=int, default=2)
    parser.add_argument(
        "--bf16",
        action="store_true",
        help="use bfloat16 autocast during attack generation; PSNR/SSIM remain fp32",
    )
    parser.add_argument("--max-cases", type=int, default=None)
    parser.add_argument("--max-ssim-slices", type=int, default=5)
    parser.add_argument("--seed", type=int, default=20260721)
    parser.add_argument("--device", type=str, default=None)
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
    attacks = list(dict.fromkeys(str(value) for value in args.attacks))

    output_dir = args.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    output_csv = output_dir / "adversarial_quality_all_cases.csv"
    summary_csv = output_dir / "adversarial_quality_all_cases_summary.csv"
    normalization_path = (
        REPO_ROOT
        / "results"
        / "adv_rob_eval_results"
        / "image_quality_analysis"
        / "normalization_per_case.csv"
    )
    normalization = _load_normalization_lookup(normalization_path)
    (output_dir / "run_manifest.json").write_text(
        json.dumps(
            {
                "datasets": list(args.datasets),
                "attacks": attacks,
                "epsilon_n_values": epsilon_n_values,
                "attack_steps": int(args.attack_steps),
                "batch_size": int(args.batch_size),
                "bf16_autocast": bool(args.bf16),
                "max_cases": args.max_cases,
                "max_ssim_slices": int(args.max_ssim_slices),
                "seed": int(args.seed),
                "attack_valid_only": bool(args.attack_valid_only),
            },
            indent=2,
            sort_keys=True,
        )
        + "\n",
        encoding="utf-8",
    )
    completed = _completed_keys(output_csv)
    device = (
        torch.device(args.device)
        if args.device
        else torch.device("cuda" if torch.cuda.is_available() else "cpu")
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
    completed.intersection_update(requested_keys)
    total_rows = len(requested_keys)
    print(f"Device: {device}", flush=True)
    print(f"Output: {output_csv}", flush=True)
    print(
        f"Requested: {sum(len(values) for values in requested_cases.values())} cases, "
        f"{total_rows} case-attack-epsilon rows; already complete: {len(completed)}",
        flush=True,
    )
    print(f"Attacks: {[ATTACK_LABELS[value] for value in attacks]}", flush=True)

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
            max_ssim_slices=args.max_ssim_slices,
            base_seed=args.seed,
            output_csv=output_csv,
            normalization=normalization,
            completed=completed,
            device=device,
            start_time=start_time,
            total_rows=total_rows,
            attack_valid_only=bool(args.attack_valid_only),
        )

    if requested_keys.issubset(completed):
        _write_summary(output_csv, summary_csv)
        print(f"Complete: {len(completed)}/{total_rows} rows", flush=True)
        print(f"Saved {summary_csv}", flush=True)
    else:
        missing = len(requested_keys - completed)
        print(
            f"Incomplete: {missing} rows remain; rerun the same command to resume",
            flush=True,
        )


if __name__ == "__main__":
    main()
