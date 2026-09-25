"""Prepare a blinded multi-reader study of adversarial prostate MRI quality.

The script creates:

* balanced, stratified case/attack/RMS selections;
* public visibility and diagnostic-adequacy manifests with hashed IDs only;
* separate scoring templates;
* a private truth key; and
* optionally, RMS-calibrated NIfTI stimuli and five-slice PNG previews.

No reader outcomes are generated. The private truth directory must not be
available to readers during scoring.
"""

from __future__ import annotations

import argparse
import hashlib
import math
import shutil
import sys
from pathlib import Path

import matplotlib
import numpy as np
import pandas as pd
import torch

matplotlib.use("Agg")
import matplotlib.pyplot as plt


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from experiments.adv_rob_eval_nnunet_nnunetrecenc import (  # noqa: E402
    IGNORE_LABEL,
    _pad_to,
    build_model,
    load_sample,
)
from experiments.evaluate_adversarial_quality_all_cases import (  # noqa: E402
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
from mri_prostate_seg.metrics.segmentation import dice_score  # noqa: E402


CALIBRATION_ROOT = (
    REPO_ROOT / "results" / "adv_rob_eval_results" / "image_quality_analysis"
)
DEFAULT_MATCHED_CSV = (
    CALIBRATION_ROOT
    / "attack_quality_damage_experiments"
    / "rms_matched_paired_per_case.csv"
)
DEFAULT_OUTPUT_DIR = CALIBRATION_ROOT / "radiologist_attack_reader_study"

ATTACKS = ["FGSM-BCE", "PGD-BCE", "APGD-BCE"]
ATTACK_KEYS = {
    "FGSM-BCE": "fgsm",
    "PGD-BCE": "pgd",
    "APGD-BCE": "apgd",
}
DATASET_LABELS = {"wg": "Whole gland (WG)", "zones": "Prostate zones"}


def _hash_id(prefix: str, value: str, seed: int) -> str:
    digest = hashlib.blake2s(
        f"{seed}|{prefix}|{value}".encode(),
        digest_size=10,
    ).hexdigest()
    return f"{prefix}_{digest.upper()}"


def _select_evenly(
    candidates: pd.DataFrame,
    count: int,
    *,
    score_column: str,
) -> pd.DataFrame:
    ordered = candidates.sort_values([score_column, "case_id"]).reset_index(drop=True)
    if len(ordered) < count:
        raise ValueError(
            f"requested {count} cases but only {len(ordered)} eligible cases remain"
        )
    indices = np.linspace(0, len(ordered) - 1, count).round().astype(int)
    indices = list(dict.fromkeys(int(value) for value in indices))
    if len(indices) < count:
        unused = [index for index in range(len(ordered)) if index not in indices]
        indices.extend(unused[: count - len(indices)])
    return ordered.iloc[indices[:count]].copy()


def select_conditions(
    matched: pd.DataFrame,
    *,
    rms_targets: list[float],
    cases_per_dataset_target: int,
) -> pd.DataFrame:
    required = {
        "dataset",
        "case_id",
        "attack",
        "target_rms_percent_case_sigma",
        "epsilon_n",
        "rms_norm",
        "ssim_axial_prostate",
        "dice_drop",
        "paired_complete",
    }
    missing = sorted(required - set(matched.columns))
    if missing:
        raise ValueError(f"RMS-matched CSV is missing columns: {missing}")
    matched = matched[
        matched["attack"].isin(ATTACKS)
        & matched["dataset"].isin(DATASET_LABELS)
        & matched["paired_complete"].astype(bool)
    ].copy()
    selected: list[pd.DataFrame] = []
    used_cases: dict[str, set[str]] = {dataset: set() for dataset in DATASET_LABELS}
    for dataset in ("wg", "zones"):
        for target_index, target in enumerate(rms_targets):
            target_rows = matched[
                (matched["dataset"] == dataset)
                & np.isclose(
                    matched["target_rms_percent_case_sigma"].to_numpy(dtype=float),
                    target,
                )
            ].copy()
            attack_counts = target_rows.groupby("case_id")["attack"].nunique()
            eligible_ids = attack_counts[attack_counts == len(ATTACKS)].index
            target_rows = target_rows[target_rows["case_id"].isin(eligible_ids)]
            case_scores = (
                target_rows.groupby("case_id", as_index=False)["dice_drop"]
                .median()
                .rename(columns={"dice_drop": "selection_damage_score"})
            )
            case_scores = case_scores[~case_scores["case_id"].isin(used_cases[dataset])]
            chosen = _select_evenly(
                case_scores,
                cases_per_dataset_target,
                score_column="selection_damage_score",
            )
            chosen["selection_rank"] = np.arange(len(chosen))
            chosen["target_index"] = target_index
            chosen_ids = set(chosen["case_id"].astype(str))
            used_cases[dataset].update(chosen_ids)
            chosen_rows = target_rows[
                target_rows["case_id"].astype(str).isin(chosen_ids)
            ].merge(chosen, on="case_id", validate="many_to_one")
            selected.append(chosen_rows)
    result = pd.concat(selected, ignore_index=True)
    attack_order = {attack: index for index, attack in enumerate(ATTACKS)}
    result["attack_order"] = result["attack"].map(attack_order)
    return result.sort_values(
        [
            "dataset",
            "target_rms_percent_case_sigma",
            "selection_rank",
            "attack_order",
        ]
    ).reset_index(drop=True)


def build_study_manifests(
    conditions: pd.DataFrame,
    *,
    reader_blocks: int,
    seed: int,
    repeat_fraction: float,
) -> dict[str, pd.DataFrame]:
    if reader_blocks < 1:
        raise ValueError("reader_blocks must be positive")
    rng = np.random.default_rng(seed)
    condition_rows: list[dict[str, object]] = []
    visibility_truth: list[dict[str, object]] = []
    adequacy_truth: list[dict[str, object]] = []
    for row in conditions.itertuples(index=False):
        condition_key = (
            f"{row.dataset}|{row.case_id}|{row.attack}|"
            f"{row.target_rms_percent_case_sigma:g}"
        )
        condition_id = _hash_id("COND", condition_key, seed)
        block = (
            int(row.selection_rank) + int(row.attack_order) + int(row.target_index)
        ) % reader_blocks + 1
        pair_id = _hash_id("PAIR", condition_key, seed)
        stimulus_a_id = _hash_id("STIM", f"{pair_id}|A", seed)
        stimulus_b_id = _hash_id("STIM", f"{pair_id}|B", seed)
        altered_side = str(rng.choice(["A", "B"]))
        condition_record = {
            **row._asdict(),
            "condition_id": condition_id,
            "reader_block": block,
            "estimated_epsilon_n": float(row.epsilon_n),
            "attack_key": ATTACK_KEYS[str(row.attack)],
            "attack_seed": stable_seed(
                str(row.dataset),
                str(row.case_id),
                f"reader_{ATTACK_KEYS[str(row.attack)]}",
                int(round(float(row.target_rms_percent_case_sigma) * 1000)),
                base_seed=seed,
            ),
        }
        condition_rows.append(condition_record)
        visibility_truth.append(
            {
                "pair_id": pair_id,
                "reader_block": block,
                "stimulus_a_id": stimulus_a_id,
                "stimulus_b_id": stimulus_b_id,
                "altered_side": altered_side,
                "condition_id": condition_id,
                "dataset": row.dataset,
                "case_id": row.case_id,
                "attack": row.attack,
                "target_rms_percent_case_sigma": row.target_rms_percent_case_sigma,
                "estimated_epsilon_n": row.epsilon_n,
                "is_reliability_repeat": False,
                "source_pair_id": "",
            }
        )
        adequacy_truth.append(
            {
                "adequacy_item_id": _hash_id("ADEQ", condition_key, seed),
                "reader_block": block,
                "stimulus_id": _hash_id("STIM", f"{condition_id}|ADEQ", seed),
                "condition_id": condition_id,
                "dataset": row.dataset,
                "case_id": row.case_id,
                "condition": "attacked",
                "attack": row.attack,
                "target_rms_percent_case_sigma": row.target_rms_percent_case_sigma,
                "estimated_epsilon_n": row.epsilon_n,
                "is_reliability_repeat": False,
                "source_adequacy_item_id": "",
            }
        )

    condition_frame = pd.DataFrame(condition_rows)
    visibility_frame = pd.DataFrame(visibility_truth)
    adequacy_frame = pd.DataFrame(adequacy_truth)

    unique_cases = (
        condition_frame[
            [
                "dataset",
                "case_id",
                "selection_rank",
                "target_index",
            ]
        ]
        .drop_duplicates()
        .reset_index(drop=True)
    )
    clean_rows: list[dict[str, object]] = []
    for row in unique_cases.itertuples(index=False):
        key = f"{row.dataset}|{row.case_id}|clean"
        clean_rows.append(
            {
                "adequacy_item_id": _hash_id("ADEQ", key, seed),
                "reader_block": (int(row.selection_rank) + int(row.target_index))
                % reader_blocks
                + 1,
                "stimulus_id": _hash_id("STIM", f"{key}|ADEQ", seed),
                "condition_id": "",
                "dataset": row.dataset,
                "case_id": row.case_id,
                "condition": "clean",
                "attack": "clean",
                "target_rms_percent_case_sigma": 0.0,
                "estimated_epsilon_n": 0.0,
                "is_reliability_repeat": False,
                "source_adequacy_item_id": "",
            }
        )
    adequacy_frame = pd.concat(
        [adequacy_frame, pd.DataFrame(clean_rows)], ignore_index=True
    )

    if repeat_fraction > 0:
        for block in range(1, reader_blocks + 1):
            candidates = visibility_frame[visibility_frame["reader_block"] == block]
            repeat_count = int(round(len(candidates) * repeat_fraction))
            if repeat_count:
                sampled = candidates.sample(n=repeat_count, random_state=seed + block)
                repeats: list[dict[str, object]] = []
                for row in sampled.itertuples(index=False):
                    record = row._asdict()
                    source_pair = str(record["pair_id"])
                    repeat_pair = _hash_id("PAIR", f"{source_pair}|repeat", seed)
                    record.update(
                        {
                            "pair_id": repeat_pair,
                            "stimulus_a_id": _hash_id("STIM", f"{repeat_pair}|A", seed),
                            "stimulus_b_id": _hash_id("STIM", f"{repeat_pair}|B", seed),
                            "is_reliability_repeat": True,
                            "source_pair_id": source_pair,
                        }
                    )
                    repeats.append(record)
                visibility_frame = pd.concat(
                    [visibility_frame, pd.DataFrame(repeats)],
                    ignore_index=True,
                )

    visibility_frame["display_order"] = visibility_frame.groupby("reader_block")[
        "pair_id"
    ].transform(
        lambda values: pd.Series(
            rng.permutation(np.arange(1, len(values) + 1)),
            index=values.index,
        )
    )
    adequacy_frame["display_order"] = adequacy_frame.groupby("reader_block")[
        "adequacy_item_id"
    ].transform(
        lambda values: pd.Series(
            rng.permutation(np.arange(1, len(values) + 1)),
            index=values.index,
        )
    )
    visibility_frame = visibility_frame.sort_values(
        ["reader_block", "display_order"]
    ).reset_index(drop=True)
    adequacy_frame = adequacy_frame.sort_values(
        ["reader_block", "display_order"]
    ).reset_index(drop=True)

    visibility_public = visibility_frame[
        [
            "reader_block",
            "display_order",
            "pair_id",
            "stimulus_a_id",
            "stimulus_b_id",
        ]
    ].copy()
    visibility_public["image_a_nifti"] = visibility_public["stimulus_a_id"].map(
        lambda value: f"stimuli/{value}.nii.gz"
    )
    visibility_public["image_b_nifti"] = visibility_public["stimulus_b_id"].map(
        lambda value: f"stimuli/{value}.nii.gz"
    )
    visibility_public["image_a_preview"] = visibility_public["stimulus_a_id"].map(
        lambda value: f"previews/{value}.png"
    )
    visibility_public["image_b_preview"] = visibility_public["stimulus_b_id"].map(
        lambda value: f"previews/{value}.png"
    )

    adequacy_public = adequacy_frame[
        [
            "reader_block",
            "display_order",
            "adequacy_item_id",
            "stimulus_id",
        ]
    ].copy()
    adequacy_public["image_nifti"] = adequacy_public["stimulus_id"].map(
        lambda value: f"stimuli/{value}.nii.gz"
    )
    adequacy_public["image_preview"] = adequacy_public["stimulus_id"].map(
        lambda value: f"previews/{value}.png"
    )

    visibility_scores = visibility_public[
        ["reader_block", "display_order", "pair_id"]
    ].copy()
    visibility_scores.insert(0, "reader_id", "")
    visibility_scores["selected_altered_image"] = ""
    visibility_scores["difference_visible"] = ""
    visibility_scores["confidence_1_to_5"] = ""
    visibility_scores["comments"] = ""

    adequacy_scores = adequacy_public[
        ["reader_block", "display_order", "adequacy_item_id"]
    ].copy()
    adequacy_scores.insert(0, "reader_id", "")
    adequacy_scores["capsule_visibility_1_to_5"] = ""
    adequacy_scores["zonal_anatomy_visibility_1_to_5"] = ""
    adequacy_scores["adequate_for_segmentation_correction_yes_no"] = ""
    adequacy_scores["overall_quality_1_to_3"] = ""
    adequacy_scores["confidence_1_to_5"] = ""
    adequacy_scores["comments"] = ""

    return {
        "conditions": condition_frame,
        "visibility_truth": visibility_frame,
        "visibility_public": visibility_public,
        "visibility_scores": visibility_scores,
        "adequacy_truth": adequacy_frame,
        "adequacy_public": adequacy_public,
        "adequacy_scores": adequacy_scores,
    }


def _pad_case(
    image_np: np.ndarray,
    seg_np: np.ndarray,
    div_factors: list[int],
    device: torch.device,
) -> tuple[torch.Tensor, torch.Tensor, tuple[int, ...]]:
    image = torch.from_numpy(image_np[np.newaxis]).to(device)
    target_shape = [
        int(math.ceil(size / factor) * factor)
        for size, factor in zip(image_np.shape[1:], div_factors)
    ]
    label = torch.from_numpy(seg_np[np.newaxis].astype(np.float32)).to(device)
    return (
        _pad_to(image, target_shape),
        _pad_to(
            label,
            target_shape,
            mode="constant",
            value=float(IGNORE_LABEL),
        ),
        tuple(int(value) for value in image_np.shape[1:]),
    )


def _generate_attack(
    model: torch.nn.Module,
    image: torch.Tensor,
    target: torch.Tensor,
    *,
    attack_key: str,
    epsilon: float,
    num_classes: int,
    attack_steps: int,
    attack_seed: int,
) -> torch.Tensor:
    with torch.enable_grad():
        if attack_key == "fgsm":
            return fgsm_bce_independent_batch(
                model, image, target, epsilon, num_classes
            )
        if attack_key == "pgd":
            return pgd_bce_independent_batch(
                model,
                image,
                target,
                epsilon,
                num_classes,
                n_steps=attack_steps,
                seeds=[attack_seed],
            )
        if attack_key == "apgd":
            return apgd_bce_independent_batch(
                model,
                image,
                target,
                epsilon,
                num_classes,
                n_steps=attack_steps,
                seeds=[attack_seed],
            )
    raise ValueError(f"unknown attack key: {attack_key}")


def _rescale_delta_to_target_rms(
    clean: np.ndarray,
    altered: np.ndarray,
    *,
    valid_mask: np.ndarray,
    target_rms: float,
    lower: float,
    upper: float,
    iterations: int = 40,
) -> tuple[np.ndarray, float]:
    delta = altered.astype(np.float64) - clean.astype(np.float64)
    if target_rms <= 0:
        return clean.astype(np.float32).copy(), 0.0
    initial = float(np.sqrt(np.mean(delta[valid_mask] ** 2)))
    if initial <= 0:
        raise RuntimeError("attack produced a zero perturbation")

    def candidate(scale: float) -> tuple[np.ndarray, float]:
        image = np.clip(clean.astype(np.float64) + scale * delta, lower, upper)
        rms = float(
            np.sqrt(
                np.mean((image[valid_mask] - clean.astype(np.float64)[valid_mask]) ** 2)
            )
        )
        return image, rms

    low = 0.0
    high = max(1.0, target_rms / initial)
    image, rms = candidate(high)
    while rms < target_rms and high < 1024:
        high *= 2.0
        image, rms = candidate(high)
    if rms < target_rms:
        raise RuntimeError(
            f"target RMS {target_rms:.6g} is unattainable after intensity clipping"
        )
    best = image
    best_rms = rms
    for _iteration in range(iterations):
        middle = 0.5 * (low + high)
        image, rms = candidate(middle)
        if abs(rms - target_rms) < abs(best_rms - target_rms):
            best, best_rms = image, rms
        if rms < target_rms:
            low = middle
        else:
            high = middle
    return best.astype(np.float32), best_rms


def _save_nifti(
    path: Path,
    image_zyx: np.ndarray,
    spacing_zyx: tuple[float, float, float],
) -> None:
    import nibabel as nib

    image_xyz = np.transpose(image_zyx.astype(np.float32), (2, 1, 0))
    affine = np.diag([spacing_zyx[2], spacing_zyx[1], spacing_zyx[0], 1.0])
    nib.save(nib.Nifti1Image(image_xyz, affine), path)


def _preview_slices(segmentation: np.ndarray, count: int = 5) -> list[int]:
    foreground_by_slice = (
        (segmentation > 0).reshape(segmentation.shape[0], -1).sum(axis=1)
    )
    available = np.flatnonzero(foreground_by_slice > 0)
    if not len(available):
        available = np.arange(segmentation.shape[0])
    indices = np.linspace(0, len(available) - 1, min(count, len(available)))
    return [int(available[int(round(index))]) for index in indices]


def _save_preview(
    path: Path,
    image: np.ndarray,
    slice_indices: list[int],
    *,
    vmin: float,
    vmax: float,
) -> None:
    fig, axes = plt.subplots(
        1, len(slice_indices), figsize=(2.4 * len(slice_indices), 2.6)
    )
    axes_array = np.atleast_1d(axes)
    for ax, slice_index in zip(axes_array, slice_indices):
        ax.imshow(image[slice_index], cmap="gray", vmin=vmin, vmax=vmax)
        ax.set_title(f"slice {slice_index}", fontsize=8)
        ax.axis("off")
    fig.tight_layout(pad=0.25)
    fig.savefig(path, dpi=180, bbox_inches="tight", facecolor="black")
    plt.close(fig)


def _macro_dice(prediction: np.ndarray, target: np.ndarray, num_classes: int) -> float:
    return float(
        np.mean(
            [
                dice_score(prediction == class_label, target == class_label)
                for class_label in range(1, num_classes)
            ]
        )
    )


def export_stimuli(
    manifests: dict[str, pd.DataFrame],
    output_dir: Path,
    *,
    attack_steps: int,
    device: torch.device,
    max_ssim_slices: int,
) -> pd.DataFrame:
    stimuli_dir = output_dir / "public" / "stimuli"
    previews_dir = output_dir / "public" / "previews"
    stimuli_dir.mkdir(parents=True, exist_ok=True)
    previews_dir.mkdir(parents=True, exist_ok=True)
    conditions = manifests["conditions"].copy()
    generated: dict[str, dict[str, object]] = {}
    audit_rows: list[dict[str, object]] = []

    for dataset in ("wg", "zones"):
        dataset_conditions = conditions[conditions["dataset"] == dataset]
        if dataset_conditions.empty:
            continue
        paths = _dataset_paths(dataset)
        use_mask = _use_mask_for_norm(paths["plans"])
        model, div_factors, num_classes = build_model(str(paths["checkpoint"]), device)
        model.eval()
        model.requires_grad_(False)
        for case_id, case_conditions in dataset_conditions.groupby("case_id"):
            image_np, seg_np, spacing = load_sample(str(paths["data"]), case_id)
            clean = np.asarray(image_np[0], dtype=np.float32)
            segmentation = np.asarray(seg_np).squeeze()
            if segmentation.ndim == 4:
                segmentation = segmentation[0]
            valid = (
                segmentation >= 0
                if use_mask
                else np.ones_like(segmentation, dtype=bool)
            )
            lower = float(clean.min())
            upper = float(clean.max())
            image_padded, target_padded, original_shape = _pad_case(
                image_np, seg_np, div_factors, device
            )
            slices = tuple(slice(0, value) for value in original_shape)
            with torch.no_grad():
                clean_prediction = (
                    model(image_padded)
                    .argmax(dim=1)[(0, *slices)]
                    .detach()
                    .cpu()
                    .numpy()
                )
            clean_macro_dice = _macro_dice(clean_prediction, segmentation, num_classes)
            slice_indices = _preview_slices(segmentation)
            display_values = clean[valid].astype(np.float64)
            vmin, vmax = np.quantile(display_values, [0.01, 0.99])
            generated[f"clean|{dataset}|{case_id}"] = {
                "image": clean,
                "spacing": tuple(float(value) for value in spacing),
                "slice_indices": slice_indices,
                "vmin": float(vmin),
                "vmax": float(vmax),
            }
            for row in case_conditions.itertuples(index=False):
                epsilon = float(row.estimated_epsilon_n) / 255.0
                adversarial_padded = _generate_attack(
                    model,
                    image_padded,
                    target_padded,
                    attack_key=str(row.attack_key),
                    epsilon=epsilon,
                    num_classes=num_classes,
                    attack_steps=attack_steps,
                    attack_seed=int(row.attack_seed),
                )
                altered = (
                    adversarial_padded[(0, 0, *slices)]
                    .detach()
                    .cpu()
                    .numpy()
                    .astype(np.float32)
                )
                calibrated, actual_rms = _rescale_delta_to_target_rms(
                    clean,
                    altered,
                    valid_mask=valid,
                    target_rms=float(row.rms_norm),
                    lower=lower,
                    upper=upper,
                )
                metrics = perturbation_quality(
                    clean,
                    calibrated,
                    valid_mask=valid,
                    slice_selector=segmentation > 0,
                    epsilon=None,
                    max_ssim_slices=max_ssim_slices,
                )
                calibrated_padded = image_padded.clone()
                calibrated_tensor = torch.from_numpy(calibrated).to(
                    device=device,
                    dtype=image_padded.dtype,
                )
                calibrated_padded[(0, 0, *slices)] = calibrated_tensor
                with torch.no_grad():
                    calibrated_prediction = (
                        model(calibrated_padded)
                        .argmax(dim=1)[(0, *slices)]
                        .detach()
                        .cpu()
                        .numpy()
                    )
                calibrated_macro_dice = _macro_dice(
                    calibrated_prediction, segmentation, num_classes
                )
                generated[str(row.condition_id)] = {
                    "image": calibrated,
                    "spacing": tuple(float(value) for value in spacing),
                    "slice_indices": slice_indices,
                    "vmin": float(vmin),
                    "vmax": float(vmax),
                }
                audit_rows.append(
                    {
                        "condition_id": row.condition_id,
                        "dataset": dataset,
                        "case_id": case_id,
                        "attack": row.attack,
                        "target_rms_percent_case_sigma": (
                            row.target_rms_percent_case_sigma
                        ),
                        "estimated_epsilon_n": row.estimated_epsilon_n,
                        "pre_rescale_attack_seed": row.attack_seed,
                        "actual_rms_norm": actual_rms,
                        "actual_rms_percent_case_sigma": metrics[
                            "rms_percent_case_sigma"
                        ],
                        "rms_match_relative_error": (actual_rms - float(row.rms_norm))
                        / max(float(row.rms_norm), 1e-12),
                        "actual_linf_norm": metrics["linf_norm"],
                        "actual_psnr_db": metrics["psnr_db_robust_range"],
                        "actual_ssim": metrics["ssim_axial_prostate"],
                        "clean_macro_dice": clean_macro_dice,
                        "stimulus_macro_dice": calibrated_macro_dice,
                        "stimulus_macro_dice_drop": (
                            clean_macro_dice - calibrated_macro_dice
                        ),
                        "calibration_method": (
                            "attack_at_interpolated_epsilon_then_residual_rescaling"
                        ),
                    }
                )
                del adversarial_padded, calibrated_padded
            del image_padded, target_padded
        del model
        if device.type == "cuda":
            torch.cuda.empty_cache()

    written: set[str] = set()

    def write_stimulus(stimulus_id: str, source_key: str) -> None:
        if stimulus_id in written:
            return
        source = generated[source_key]
        _save_nifti(
            stimuli_dir / f"{stimulus_id}.nii.gz",
            np.asarray(source["image"]),
            source["spacing"],
        )
        _save_preview(
            previews_dir / f"{stimulus_id}.png",
            np.asarray(source["image"]),
            list(source["slice_indices"]),
            vmin=float(source["vmin"]),
            vmax=float(source["vmax"]),
        )
        written.add(stimulus_id)

    condition_lookup = conditions.set_index("condition_id")
    for row in manifests["visibility_truth"].itertuples(index=False):
        condition = condition_lookup.loc[row.condition_id]
        clean_key = f"clean|{condition.dataset}|{condition.case_id}"
        altered_key = str(row.condition_id)
        source_a = altered_key if row.altered_side == "A" else clean_key
        source_b = altered_key if row.altered_side == "B" else clean_key
        write_stimulus(str(row.stimulus_a_id), source_a)
        write_stimulus(str(row.stimulus_b_id), source_b)
    for row in manifests["adequacy_truth"].itertuples(index=False):
        if row.condition == "clean":
            source_key = f"clean|{row.dataset}|{row.case_id}"
        else:
            source_key = str(row.condition_id)
        write_stimulus(str(row.stimulus_id), source_key)
    return pd.DataFrame(audit_rows)


def _write_protocol(
    output_dir: Path,
    *,
    manifests: dict[str, pd.DataFrame],
    rms_targets: list[float],
    exported: bool,
) -> Path:
    visibility = manifests["visibility_public"]
    adequacy = manifests["adequacy_public"]
    text = f"""# Blinded radiologist adversarial-image study

## Status

Study design and scoring files are ready. Stimulus export is {"complete" if exported else "not run; use --export-stimuli"}. No radiologist responses have been generated or inferred.

## Study population

- Datasets: Whole gland and prostate zones
- Attacks: FGSM, PGD, and APGD
- RMS-matched targets: {", ".join(f"{value:g}% SD" for value in rms_targets)}
- Visibility pairs: {len(visibility):,}
- Diagnostic-adequacy items: {len(adequacy):,}
- Reader blocks: {visibility["reader_block"].nunique()}

Cases are stratified across the observed median Dice-damage distribution. The same selected case supports all three attacks at a given RMS target, while Latin-style reader blocks reduce repeated-anatomy exposure within a reader.

The default six cases per dataset and RMS target constitute a pilot design.
Use pilot variability and the intended number of readers in a formal
multi-reader multi-case power calculation before a confirmatory clinical study.

## Blinding

Give readers only the `public/` directory and the appropriate scoring template. Keep `truth/` inaccessible until every score file has been locked. Public filenames are deterministic hashes and contain no case, attack, epsilon, RMS, or altered-side label.

## Task 1: perturbation visibility

For each pair, inspect image A and image B under the same display conditions.

1. Select which image is altered: A or B. A response is required.
2. Record whether the difference is consciously visible: yes or no.
3. Record confidence from 1 (guessing) to 5 (certain).

The primary endpoint is forced-choice accuracy. Secondary endpoints are the visible fraction, confidence, attack-specific accuracy, RMS-specific accuracy, and intra-reader repeat agreement.

## Task 2: diagnostic/contouring adequacy

Review each single image without access to its clean counterpart.

1. Capsule visibility: 1 (non-diagnostic) to 5 (excellent).
2. Zonal-anatomy visibility: 1 to 5; use NA for WG cases if not assessable.
3. Adequate for manual segmentation correction: yes or no.
4. Overall quality: 1 (inadequate), 2 (adequate with limitations), or 3 (fully adequate).
5. Confidence: 1 to 5.

This is a model-space T2-weighted image study, not a complete PI-QUAL examination. Do not assign a full PI-QUAL score without all required clinical sequences and acquisition information.

## Reading conditions

- Use a diagnostic display in a controlled environment.
- Permit identical window/level adjustment for both images in a visibility pair.
- Randomize by the supplied display order.
- Run visibility and adequacy in separate sessions with a washout interval.
- Assign at least two independent readers to each block when estimating
  inter-reader variability. If readers must review multiple blocks, separate
  them by a washout interval and preserve the supplied block-specific order.
- Do not reveal attack prevalence or the truth key.
- Copy one scoring template per reader; use a pseudonymous reader ID.

## Files

- `public/visibility_manifest.csv`: blinded A/B order.
- `public/adequacy_manifest.csv`: blinded single-image order.
- `public/visibility_scores_template.csv`: visibility responses.
- `public/adequacy_scores_template.csv`: adequacy responses.
- `truth/visibility_truth_key.csv`: altered side and attack metadata.
- `truth/adequacy_truth_key.csv`: clean/attack metadata.
- `truth/selected_conditions.csv`: case selection and estimated epsilon.
- `truth/stimulus_export_audit.csv`: exact exported RMS/PSNR/SSIM and paired
  segmentation Dice, when exported.

After ratings are complete, run `analyze_radiologist_attack_study.py` with the combined reader score files.
"""
    path = output_dir / "STUDY_PROTOCOL.md"
    path.write_text(text, encoding="utf-8")
    return path


def write_package(
    manifests: dict[str, pd.DataFrame],
    output_dir: Path,
    *,
    export: bool,
    attack_steps: int,
    device: torch.device,
    max_ssim_slices: int,
    rms_targets: list[float],
) -> list[Path]:
    public_dir = output_dir / "public"
    truth_dir = output_dir / "truth"
    public_dir.mkdir(parents=True, exist_ok=True)
    truth_dir.mkdir(parents=True, exist_ok=True)
    paths = [
        public_dir / "visibility_manifest.csv",
        public_dir / "adequacy_manifest.csv",
        public_dir / "visibility_scores_template.csv",
        public_dir / "adequacy_scores_template.csv",
        truth_dir / "selected_conditions.csv",
        truth_dir / "visibility_truth_key.csv",
        truth_dir / "adequacy_truth_key.csv",
    ]
    manifests["visibility_public"].to_csv(paths[0], index=False)
    manifests["adequacy_public"].to_csv(paths[1], index=False)
    manifests["visibility_scores"].to_csv(paths[2], index=False)
    manifests["adequacy_scores"].to_csv(paths[3], index=False)
    manifests["conditions"].to_csv(paths[4], index=False)
    manifests["visibility_truth"].to_csv(paths[5], index=False)
    manifests["adequacy_truth"].to_csv(paths[6], index=False)
    if export:
        audit = export_stimuli(
            manifests,
            output_dir,
            attack_steps=attack_steps,
            device=device,
            max_ssim_slices=max_ssim_slices,
        )
        audit_path = truth_dir / "stimulus_export_audit.csv"
        audit.to_csv(audit_path, index=False)
        paths.append(audit_path)
    protocol = _write_protocol(
        output_dir,
        manifests=manifests,
        rms_targets=rms_targets,
        exported=export,
    )
    paths.append(protocol)
    return paths


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Prepare a blinded RMS-matched radiologist reader study"
    )
    parser.add_argument("--matched-csv", type=Path, default=DEFAULT_MATCHED_CSV)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--rms-targets", nargs="+", type=float, default=[1.0, 4.0, 7.0])
    parser.add_argument("--cases-per-dataset-target", type=int, default=6)
    parser.add_argument("--reader-blocks", type=int, default=3)
    parser.add_argument("--repeat-fraction", type=float, default=0.10)
    parser.add_argument("--seed", type=int, default=20260722)
    parser.add_argument("--export-stimuli", action="store_true")
    parser.add_argument("--attack-steps", type=int, default=20)
    parser.add_argument("--max-ssim-slices", type=int, default=5)
    parser.add_argument("--device", type=str, default=None)
    parser.add_argument(
        "--replace-existing-stimuli",
        action="store_true",
        help="remove only the existing public stimuli/previews before export",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.cases_per_dataset_target < 1:
        raise ValueError("--cases-per-dataset-target must be positive")
    if args.reader_blocks < 1:
        raise ValueError("--reader-blocks must be positive")
    if not 0 <= args.repeat_fraction < 1:
        raise ValueError("--repeat-fraction must be in [0, 1)")
    rms_targets = sorted(set(float(value) for value in args.rms_targets))
    if not rms_targets or any(value <= 0 for value in rms_targets):
        raise ValueError("reader-study RMS targets must be positive")
    matched_path = args.matched_csv.resolve()
    if not matched_path.is_file():
        raise FileNotFoundError(
            f"missing RMS-matched analysis: {matched_path}; run "
            "analyze_attack_quality_damage.py first"
        )
    output_dir = args.output_dir.resolve()
    if args.replace_existing_stimuli:
        for path in (
            output_dir / "public" / "stimuli",
            output_dir / "public" / "previews",
        ):
            if path.is_dir():
                shutil.rmtree(path)

    matched = pd.read_csv(matched_path)
    conditions = select_conditions(
        matched,
        rms_targets=rms_targets,
        cases_per_dataset_target=args.cases_per_dataset_target,
    )
    manifests = build_study_manifests(
        conditions,
        reader_blocks=args.reader_blocks,
        seed=args.seed,
        repeat_fraction=args.repeat_fraction,
    )
    device = (
        torch.device(args.device)
        if args.device
        else torch.device("cuda" if torch.cuda.is_available() else "cpu")
    )
    paths = write_package(
        manifests,
        output_dir,
        export=args.export_stimuli,
        attack_steps=args.attack_steps,
        device=device,
        max_ssim_slices=args.max_ssim_slices,
        rms_targets=rms_targets,
    )
    print(
        f"Selected {conditions[['dataset', 'case_id']].drop_duplicates().shape[0]} "
        f"unique cases and {len(conditions)} attack conditions",
        flush=True,
    )
    print(
        f"Visibility pairs: {len(manifests['visibility_public'])}; "
        f"adequacy items: {len(manifests['adequacy_public'])}",
        flush=True,
    )
    for path in paths:
        print(f"Saved {path}", flush=True)


if __name__ == "__main__":
    main()
