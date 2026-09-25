"""Spectral-shell energy-utility attribution (SSEUA): GPU driver.

Reads the same cohort as ``evaluate_edge_attribution_all_cases.py``, regenerates
the same adversarial perturbations, and partitions each one into fixed
radial-frequency shells instead of anatomical edge patches.  Every helper that
is not spectral is imported from the edge driver; that file is not modified.

Re-attacking a defended checkpoint: ``--trainer-dir <Trainer>__<Plans>__3d_fullres``
selects whose ``fold_all/checkpoint_final.pth`` is attacked (default: the clean
model).  Use a fresh ``--output-dir``; the run manifest records the trainer and
refuses to resume into a directory produced with a different one.  The published
RMS drift references belong to the clean model, so they are skipped for any other
trainer unless ``--quality-csv`` is given explicitly.

Spec: docs/superpowers/specs/2026-09-14-spectral-shell-attribution-design.md
"""

from __future__ import annotations

import argparse
import gc
import json
import math
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import pandas as pd
import torch

REPO_ROOT = Path(__file__).resolve().parents[1]
for _root in (REPO_ROOT, REPO_ROOT / "src"):
    if str(_root) not in sys.path:
        sys.path.insert(0, str(_root))

from experiments.adv_rob_eval_nnunet_nnunetrecenc import (  # noqa: E402
    IGNORE_LABEL,
    _pad_to,
    build_model,
    load_sample,
)
from experiments.evaluate_adversarial_quality_all_cases import (  # noqa: E402
    ATTACK_LABELS,
    TRAINERS,
    _available_case_ids,
    _dataset_paths,
    _per_sample_bce,
)
from experiments.evaluate_adversarial_segmentation_all_cases import (  # noqa: E402
    _configure_deterministic_execution,
)
from experiments.evaluate_edge_attribution_all_cases import (  # noqa: E402
    DEFAULT_QUALITY_CSV,
    DEFAULT_RESULTS,
    IDENTITY,
    INTERACTION_EPS,
    METRICS,
    QC_CSV,
    QC_FIELDNAMES,
    RMS_DRIFT_FLAG,
    _append,
    _metrics,
    _rms_check,
    _rms_reference,
    _save_npz,
    _unpad,
    case_ids_for_subset,
    condition_damages,
    primary_fn,
    run_attack,
)
from mri_prostate_seg.experiments.edge_attribution import (  # noqa: E402
    concentration_indices,
    damage,
    k_alpha,
    subcohort_case_ids,
)
from mri_prostate_seg.experiments.edge_attribution.spectral import (  # noqa: E402
    BANDS,
    BAND_OF_SHELL,
    N_SHELLS,
    ShellPartition,
    SpectralDelta,
    parseval_error,
    reference_shell_edges,
    shell_attribution,
    shell_energy,
    shell_masks,
    split_delta,
)
from mri_prostate_seg.experiments.edge_attribution.spectral_analysis import (  # noqa: E402
    RANKINGS,
    band_indices,
    count_matched_draws,
    cumulative_masks,
    energy_matched_draws,
    k_to_cpm,
    permutation_p,
    ranking_orders,
)
from mri_prostate_seg.experiments.edge_attribution.torch_ops import (  # noqa: E402
    SpectralOperator,
    evaluate_interventions,
    greedy_restoration,
    integrated_attribution,
    pairwise_interactions,
)
from mri_prostate_seg.experiments.segmentation_direction import (  # noqa: E402
    attack_success,
)

DEFAULT_OUTPUT_DIR = DEFAULT_RESULTS / "all_cases_spectral_attribution"
SUMMARY_CSV = "case_summary.csv"
SHELL_CSV = "shell_metrics.csv"
CURVES_CSV = "subset_curves.csv"
CONTROL_CSV = "control_summary.csv"
SPECTRA_DIR = "spectra"
STAGES = ("energy", "causal", "interaction")
CASE_SUBSETS = ("all", "causal", "interaction")
DEFAULT_EPS_N = (16, 32)
DEFAULT_ATTACKS = ("pgd", "apgd")
REMOVE_ALL_GATE = 0.02
PARSEVAL_GATE = 1e-6
IG_COMPLETENESS_GATE = 0.05
_NAN = float("nan")

# ``ATLAS_TYPE`` is what ``aggregate.CONDITION`` groups on; the shell count is
# part of the literal so a Phase B run at 12 or 48 shells never merges with 24.
N_SHELLS_ACTIVE = N_SHELLS
ATLAS_TYPE = f"radial_shells_{N_SHELLS}"

SHELL_FIELDNAMES = [
    *IDENTITY,
    "stage",
    "patch_id",
    "band",
    "anatomical_category",
    "f_low_cpm",
    "f_high_cpm",
    "n_coefficients_weighted",
    "shell_empty",
    "screened",
    "energy",
    "energy_share",
    "fold_enrichment",
    "ig_attribution",
    *[f"necessity_{m}" for m in METRICS],
    *[f"sufficiency_{m}" for m in METRICS],
    "fractional_recovery_dice",
    "linf_excess",
]
SUMMARY_FIELDNAMES = [
    *IDENTITY,
    "stage",
    "attack_seed",
    "attack_success",
    "non_damaging",
    *[f"clean_{m}" for m in METRICS],
    *[f"adv_{m}" for m in METRICS],
    *[f"full_{m}_damage" for m in METRICS],
    "linf_norm",
    "rms_norm",
    "rms_reproduction_rel_error",
    "epsilon_saturation_fraction",
    "n_shells",
    "n_shells_empty",
    "parseval_error",
    "total_energy_valid",
    "dc_energy_share",
    "out_energy_share",
    "lf_energy_share",
    "mf_energy_share",
    "hf_energy_share",
    "n_eff",
    "n_eff_norm",
    "entropy_concentration",
    "energy_gini",
    "number_of_patches",
    "n_patches_screened",
    "ig_steps_used",
    "ig_completeness_error",
    "ig_positive_total",
    "remove_all_damage_dice",
    "utility_gini",
    "utility_n_eff",
    *[f"k50_{r}" for r in RANKINGS],
    *[f"k80_{r}" for r in RANKINGS],
    "f_k50_frequency_low_first_cpm",
    "f_k80_frequency_low_first_cpm",
    "f_k50_frequency_high_first_cpm",
    "k50_remove",
    "k80_remove",
    "k50_keep",
    "k80_keep",
    "damage_removed_top_10pct",
    "damage_kept_top_10pct",
    *[f"{b.lower()}_necessity_dice" for b in BANDS],
    *[f"{b.lower()}_sufficiency_dice" for b in BANDS],
    "jaccard_lf_top8_utility",
    "spearman_frequency_utility",
    "spearman_energy_utility",
    "spearman_ig_utility",
    "greedy_steps",
    "greedy_k95",
    # a-priori band controls.  The ESEUA-required columns carry their spectral
    # meaning for the LF band so ``aggregate.cohort_controls`` works unchanged.
    "observed_band_primary",
    "observed_top10_recovery",
    "observed_matched_recovery",
    "observed_pool_size",
    "control_pool_size",
    "matched_pool_size_median",
    "control_drawable_fraction",
    "n_trials_skipped_undersized",
    "n_controls_volume",
    "n_controls_energy",
    "control_volume_recovery_median",
    "control_energy_recovery_median",
    "control_permutation_p",
    "control_permutation_p_pooled",
    *[f"{b.lower()}_observed_recovery" for b in BANDS],
    *[f"{b.lower()}_control_permutation_p" for b in BANDS],
    *[f"{b.lower()}_control_drawable_fraction" for b in BANDS],
]
CURVES_FIELDNAMES = [
    *IDENTITY,
    "stage",
    "ranking_type",
    "k",
    "evaluated",
    "patch_id",
    "f_high_cpm",
    "patch_fraction",
    "volume_fraction",
    "total_energy_fraction",
    "edge_energy_fraction",
    "damage_removed_fraction",
    "damage_kept_fraction",
    "hd95_removed",
    "asd_removed",
    "hd95_kept",
    "asd_kept",
    "linf_excess",
]
CONTROL_FIELDNAMES = [
    *IDENTITY,
    "stage",
    "control_type",
    "observed_band",
    "patch_id",
    "trial",
    "target_voxels",
    "control_voxels",
    "target_energy",
    "control_energy",
    "recovery_fraction",
    "patch_recovery_fraction",
    "pair_a",
    "pair_b",
    "joint_sufficiency",
    "interaction",
    "greedy_step",
    "greedy_recovered_fraction",
    "n_patches_in_joint",
    "control_shells",
]


# ---------------------------------------------------------------------------
# Per-case and per-condition state
# ---------------------------------------------------------------------------
@dataclass
class SpectralCase:
    dataset_key: str
    case_id: str
    spacing: tuple[float, float, float]
    seg: np.ndarray
    valid: np.ndarray
    x_pad: torch.Tensor
    y_pad: torch.Tensor
    padded_shape: tuple[int, ...]
    shape: tuple[int, ...]
    clean_pred: np.ndarray
    clean_loss: float
    clean_metrics: dict[str, float]
    partition: ShellPartition
    in_causal: bool
    in_interaction: bool


@dataclass
class SpectralCondition:
    attack_key: str
    attack_label: str
    epsilon_n: int
    seed: int
    adv_pad: torch.Tensor
    delta: np.ndarray
    split: SpectralDelta
    adv_pred: np.ndarray
    adv_metrics: dict[str, float]
    damages: dict[str, float]
    undefined: list[str]
    linf: float
    rms: float
    saturation: float
    rms_error: float
    stage: str = "energy"
    flags: list[str] = field(default_factory=list)


@dataclass
class Settings:
    stages: tuple[str, ...]
    ig_steps: int
    ig_batch: int
    iv_batch: int
    boundary_workers: int
    count_trials: int
    energy_trials: int
    energy_tol: float
    energy_max_tries: int
    save_spectra: bool
    base_seed: int
    attack_valid_only: bool = False


def _identity(case: SpectralCase, cond: SpectralCondition) -> dict[str, object]:
    return {
        "dataset": case.dataset_key,
        "case_id": case.case_id,
        "epsilon_n": cond.epsilon_n,
        "attack": cond.attack_label,
        "atlas_type": ATLAS_TYPE,
        "stage": cond.stage,
    }


def completed_keys(summary_csv: Path) -> set[tuple[str, str, int, str]]:
    if not summary_csv.is_file() or summary_csv.stat().st_size == 0:
        return set()
    frame = pd.read_csv(summary_csv, usecols=IDENTITY[:4])
    return {
        (str(r.dataset), str(r.case_id), int(r.epsilon_n), str(r.attack))
        for r in frame.itertuples()
    }


def spectral_case(
    model: torch.nn.Module,
    device: torch.device,
    div_factors: list[int],
    num_classes: int,
    dataset_key: str,
    case_id: str,
    image_np: np.ndarray,
    seg_np: np.ndarray,
    spacing: tuple[float, float, float],
    *,
    in_causal: bool,
    in_interaction: bool,
    n_shells: int = N_SHELLS,
) -> SpectralCase:
    """Everything per case that does not depend on the attack."""

    image = np.asarray(image_np, dtype=np.float32)
    if image.ndim == 3:
        image = image[np.newaxis]
    seg = np.asarray(seg_np).squeeze()
    if seg.ndim == 4:
        seg = seg[0]
    shape = tuple(int(v) for v in image.shape[1:])
    padded = tuple(int(math.ceil(v / f) * f) for v, f in zip(shape, div_factors))
    x_pad = _pad_to(torch.from_numpy(image[np.newaxis]).to(device), list(padded))
    y_pad = _pad_to(
        torch.from_numpy(seg[np.newaxis, np.newaxis].astype(np.float32)).to(device),
        list(padded),
        mode="constant",
        value=float(IGNORE_LABEL),
    )
    with torch.no_grad():
        logits = model(x_pad)
        clean_loss = float(_per_sample_bce(logits, y_pad, num_classes)[0].item())
        pred = logits.argmax(dim=1)
    clean_pred = _unpad(pred[0], shape).astype(np.int16)
    spacing = (float(spacing[0]), float(spacing[1]), float(spacing[2]))
    # The partition lives on the PADDED grid: the attack perturbs every input
    # voxel, including the divisibility margin and the ignore-label region, and
    # the model's receptive field sees all of it.  Leaving any part of delta
    # unpartitioned puts damage into the ``remove all`` endpoint (measured on the
    # smoke run: half the damage came from the outside-mask perturbation).
    partition = shell_masks(padded, spacing, reference_shell_edges(n_shells))
    return SpectralCase(
        dataset_key,
        case_id,
        spacing,
        seg,
        seg >= 0,
        x_pad,
        y_pad,
        padded,
        shape,
        clean_pred,
        clean_loss,
        _metrics(clean_pred, seg, spacing, dataset_key, clean_loss),
        partition,
        in_causal,
        in_interaction,
    )


def load_spectral_case(
    model: torch.nn.Module,
    device: torch.device,
    div_factors: list[int],
    num_classes: int,
    dataset_key: str,
    case_id: str,
    paths: dict[str, Path],
    **kwargs: object,
) -> SpectralCase:
    image_np, seg_np, spacing_raw = load_sample(str(paths["data"]), case_id)
    return spectral_case(
        model,
        device,
        div_factors,
        num_classes,
        dataset_key,
        case_id,
        image_np,
        seg_np,
        tuple(float(v) for v in spacing_raw),
        **kwargs,  # type: ignore[arg-type]
    )


def build_spectral_condition(
    model: torch.nn.Module,
    case: SpectralCase,
    attack_key: str,
    attack_label: str,
    epsilon_n: int,
    adv_pad: torch.Tensor,
    seed: int,
    num_classes: int,
    quality: pd.DataFrame | None,
) -> SpectralCondition:
    epsilon = epsilon_n / 255.0
    delta = _unpad((adv_pad - case.x_pad)[0, 0], case.shape).astype(np.float64)
    linf = float(np.abs(delta[case.valid]).max()) if case.valid.any() else 0.0
    if linf > epsilon + 1e-6:
        raise RuntimeError(f"L-inf {linf:.6f} exceeds epsilon {epsilon:.6f}")
    rms = float(np.sqrt(np.mean(delta[case.valid] ** 2)))
    saturation = float(np.mean(np.abs(delta[case.valid]) >= epsilon * (1 - 1e-6)))
    with torch.no_grad():
        logits = model(adv_pad)
        adv_loss = float(_per_sample_bce(logits, case.y_pad, num_classes)[0].item())
        adv_pred = _unpad(logits.argmax(dim=1)[0], case.shape).astype(np.int16)
    adv_metrics = _metrics(adv_pred, case.seg, case.spacing, case.dataset_key, adv_loss)
    damages, undefined = condition_damages(case.clean_metrics, adv_metrics)
    key = f"{case.dataset_key}/{case.case_id}/{attack_label}/eps{epsilon_n}"
    rms_error = _rms_check(
        rms,
        _rms_reference(
            quality, case.dataset_key, case.case_id, epsilon_n, attack_label
        ),
        key,
    )
    delta_pad = (adv_pad - case.x_pad)[0, 0].detach().cpu().numpy().astype(np.float64)
    return SpectralCondition(
        attack_key,
        attack_label,
        epsilon_n,
        seed,
        adv_pad,
        delta,
        split_delta(delta_pad, np.ones(delta_pad.shape, dtype=bool)),
        adv_pred,
        adv_metrics,
        damages,
        undefined,
        linf,
        rms,
        saturation,
        rms_error,
    )


# ---------------------------------------------------------------------------
# Stages
# ---------------------------------------------------------------------------
def _intervention_metrics(
    case: SpectralCase, result, primary
) -> list[dict[str, float]]:
    """Per-intervention metric dicts; surface metrics where a prediction was kept."""

    out: list[dict[str, float]] = []
    for i, dice_row in enumerate(result.dice):
        m = {"dice": primary(dice_row), "loss": float(result.loss[i])}
        if case.dataset_key == "zones":
            m["dice_tz"], m["dice_pz"] = float(dice_row[0]), float(dice_row[1])
        if i in result.preds:
            # Predictions are kept on the padded grid; score them on the crop.
            crop = tuple(slice(0, v) for v in case.shape)
            surf = _metrics(
                result.preds[i][crop], case.seg, case.spacing, case.dataset_key, 0.0
            )
            m["hd95"], m["asd"] = surf["hd95"], surf["asd"]
        out.append(m)
    return out


def _linf_excess(
    op: SpectralOperator, masks: np.ndarray, epsilon: float
) -> list[float]:
    """max|P_S delta| / epsilon per mask -- how far the projection leaves the ball."""

    out = []
    for m in masks:
        mask = torch.from_numpy(np.ascontiguousarray(m))[None, None].to(op.F.device)
        out.append(float(op._project(mask).abs().max().item()) / epsilon)
    return out


def _run(model, case, cond, masks, mode, num_classes, batch, op, *, keep_preds: bool):
    keep = set(range(masks.shape[0])) if keep_preds else set()
    return evaluate_interventions(
        model,
        _per_sample_bce,
        case.x_pad,
        cond.adv_pad,
        case.y_pad,
        np.ascontiguousarray(masks),
        mode=mode,
        num_classes=num_classes,
        batch=batch,
        keep_pred_indices=keep,
        crop=case.shape,
        operator=op,
    )


def energy_stage(
    case: SpectralCase, cond: SpectralCondition
) -> tuple[list[dict[str, object]], dict[str, object], np.ndarray]:
    part = case.partition
    energies = shell_energy(cond.split.F, part)
    total_valid = cond.split.energy_total
    e_sum = float(energies.sum())
    share = energies / e_sum if e_sum > 0 else np.full(energies.shape, _NAN)
    counts = part.counts
    with np.errstate(divide="ignore", invalid="ignore"):
        fold = np.where(counts > 0, share / (counts / counts.sum()), _NAN)
    rows: list[dict[str, object]] = []
    for k in range(part.n_shells):
        band = BAND_OF_SHELL(k + 1, part.n_shells)
        rows.append(
            {
                **_identity(case, cond),
                "patch_id": k + 1,
                "band": band,
                "anatomical_category": band,
                "f_low_cpm": float(part.edges[k]),
                "f_high_cpm": float(part.edges[k + 1]),
                "n_coefficients_weighted": float(counts[k]),
                "shell_empty": int(part.empty[k]),
                "screened": int(not part.empty[k]),
                "energy": float(energies[k]),
                "energy_share": float(share[k]),
                "fold_enrichment": float(fold[k]),
            }
        )
    conc = concentration_indices(share[~part.empty])
    delta_pad = (cond.adv_pad - case.x_pad)[0, 0].detach().cpu().numpy()
    parseval = parseval_error(cond.split.F, part, delta_pad)
    if parseval > PARSEVAL_GATE:
        raise AssertionError(f"Parseval violated: {parseval:.3e}")
    # Descriptive only: how much of the perturbation energy sits outside the
    # valid (non-ignore) region of the crop.  It IS partitioned like the rest.
    e_out = float((cond.delta[~case.valid] ** 2).sum())
    summary: dict[str, object] = {
        "n_shells": part.n_shells,
        "n_shells_empty": int(part.empty.sum()),
        "parseval_error": parseval,
        "total_energy_valid": total_valid,
        "dc_energy_share": cond.split.energy_dc / total_valid
        if total_valid > 0
        else _NAN,
        "out_energy_share": e_out / total_valid if total_valid > 0 else _NAN,
        **{
            f"{b.lower()}_energy_share": float(
                np.nansum(share[band_indices(b, part.n_shells)])
            )
            for b in BANDS
        },
        "n_eff": conc["n_eff"],
        "n_eff_norm": conc["n_eff_norm"],
        "entropy_concentration": conc["entropy_concentration"],
        "energy_gini": conc["gini"],
        "number_of_patches": int((~part.empty).sum()),
        "n_patches_screened": int((~part.empty).sum()),
    }
    return rows, summary, energies


def _spearman(a: np.ndarray, b: np.ndarray) -> float:
    from scipy.stats import spearmanr

    keep = np.isfinite(a) & np.isfinite(b)
    if keep.sum() < 3:
        return _NAN
    rho = spearmanr(a[keep], b[keep]).correlation
    return float(rho) if rho is not None and math.isfinite(rho) else _NAN


def causal_stage(
    model: torch.nn.Module,
    case: SpectralCase,
    cond: SpectralCondition,
    num_classes: int,
    settings: Settings,
    shell_rows: list[dict[str, object]],
    energies: np.ndarray,
) -> tuple[
    list[dict[str, object]], dict[str, object], np.ndarray, np.ndarray, SpectralOperator
]:
    part = case.partition
    primary = primary_fn(case.dataset_key)
    clean_primary = case.clean_metrics["dice"]
    d_full = cond.damages.get("dice", _NAN)
    damaging = math.isfinite(d_full) and d_full > 0
    epsilon = cond.epsilon_n / 255.0
    op = SpectralOperator.from_split(
        case.x_pad,
        cond.adv_pad,
        cond.split,
        crop=case.padded_shape,
        device=case.x_pad.device,
    )

    ig = integrated_attribution(
        model,
        _per_sample_bce,
        case.x_pad,
        cond.adv_pad,
        case.y_pad,
        num_classes=num_classes,
        steps=settings.ig_steps,
        batch=settings.ig_batch,
    )
    if ig.completeness_error > IG_COMPLETENESS_GATE:
        ig = integrated_attribution(
            model,
            _per_sample_bce,
            case.x_pad,
            cond.adv_pad,
            case.y_pad,
            num_classes=num_classes,
            steps=settings.ig_steps * 2,
            batch=settings.ig_batch,
        )
    assert ig.mean_gradient is not None
    # The partition is on the padded grid, so the IG gradient stays padded too.
    attributions = shell_attribution(
        cond.split.F, np.asarray(ig.mean_gradient, dtype=np.float64), part
    )

    live = np.nonzero(~part.empty)[0]
    masks = part.masks[live]
    rem = _run(
        model,
        case,
        cond,
        masks,
        "remove",
        num_classes,
        settings.iv_batch,
        op,
        keep_preds=True,
    )
    keep = _run(
        model,
        case,
        cond,
        masks,
        "keep",
        num_classes,
        settings.iv_batch,
        op,
        keep_preds=True,
    )
    rem_m = _intervention_metrics(case, rem, primary)
    keep_m = _intervention_metrics(case, keep, primary)
    excess = _linf_excess(op, masks, epsilon)
    necessity = np.full(part.n_shells, _NAN)
    sufficiency = np.full(part.n_shells, _NAN)
    for pos, k in enumerate(live):
        row = shell_rows[k]
        row["ig_attribution"] = float(attributions[k])
        for metric in METRICS:
            if metric in rem_m[pos]:
                v = damage(metric, case.clean_metrics[metric], rem_m[pos][metric])
                row[f"necessity_{metric}"] = (
                    cond.damages[metric] - v if math.isfinite(v) else _NAN
                )
            if metric in keep_m[pos]:
                row[f"sufficiency_{metric}"] = damage(
                    metric, case.clean_metrics[metric], keep_m[pos][metric]
                )
        necessity[k] = float(row.get("necessity_dice", _NAN))
        sufficiency[k] = float(row.get("sufficiency_dice", _NAN))
        row["fractional_recovery_dice"] = necessity[k] / d_full if damaging else _NAN
        row["linf_excess"] = excess[pos]

    orders = ranking_orders(necessity, energies, attributions, part.empty)
    summary: dict[str, object] = {
        "ig_steps_used": ig.steps,
        "ig_completeness_error": ig.completeness_error,
        "ig_positive_total": float(np.clip(attributions, 0, None).sum()),
    }
    util_conc = concentration_indices(np.clip(np.nan_to_num(necessity[live]), 0, None))
    summary["utility_gini"], summary["utility_n_eff"] = (
        util_conc["gini"],
        util_conc["n_eff"],
    )
    e_total = float(energies.sum())
    c_total = float(part.counts.sum())
    curve_rows: list[dict[str, object]] = []
    for name, order in orders.items():
        cum = cumulative_masks(part.masks, order)
        r = _run(
            model,
            case,
            cond,
            cum,
            "remove",
            num_classes,
            settings.iv_batch,
            op,
            keep_preds=True,
        )
        s = _run(
            model,
            case,
            cond,
            cum,
            "keep",
            num_classes,
            settings.iv_batch,
            op,
            keep_preds=True,
        )
        r_m = _intervention_metrics(case, r, primary)
        s_m = _intervention_metrics(case, s, primary)
        ex = _linf_excess(op, cum, epsilon)
        removed = np.array([clean_primary - m["dice"] for m in r_m])
        kept = np.array([clean_primary - m["dice"] for m in s_m])
        rem_frac = (
            (d_full - removed) / d_full if damaging else np.full(removed.shape, _NAN)
        )
        keep_frac = kept / d_full if damaging else np.full(kept.shape, _NAN)
        cum_e = (
            np.cumsum(energies[order]) / e_total
            if e_total > 0
            else np.full(len(order), _NAN)
        )
        cum_c = np.cumsum(part.counts[order]) / c_total
        for i, shell in enumerate(order):
            k = i + 1
            curve_rows.append(
                {
                    **_identity(case, cond),
                    "ranking_type": name,
                    "k": k,
                    "evaluated": 1,
                    "patch_id": int(shell) + 1,
                    "f_high_cpm": k_to_cpm(order, k, part.edges),
                    "patch_fraction": k / len(order),
                    "volume_fraction": float(cum_c[i]),
                    "total_energy_fraction": float(cum_e[i]),
                    "edge_energy_fraction": float(cum_e[i]),
                    "damage_removed_fraction": float(rem_frac[i]),
                    "damage_kept_fraction": float(keep_frac[i]),
                    "hd95_removed": r_m[i].get("hd95", _NAN),
                    "asd_removed": r_m[i].get("asd", _NAN),
                    "hd95_kept": s_m[i].get("hd95", _NAN),
                    "asd_kept": s_m[i].get("asd", _NAN),
                    "linf_excess": ex[i],
                }
            )
        summary[f"k50_{name}"] = k_alpha(rem_frac, 0.5)
        summary[f"k80_{name}"] = k_alpha(rem_frac, 0.8)
        if name == "frequency_low_first":
            summary["f_k50_frequency_low_first_cpm"] = k_to_cpm(
                order, summary[f"k50_{name}"], part.edges
            )
            summary["f_k80_frequency_low_first_cpm"] = k_to_cpm(
                order, summary[f"k80_{name}"], part.edges
            )
            summary["remove_all_damage_dice"] = float(removed[-1])
            if abs(float(removed[-1])) > REMOVE_ALL_GATE:
                cond.flags.append(f"remove_all_nonzero:{float(removed[-1]):.4f}")
        elif name == "frequency_high_first":
            summary["f_k50_frequency_high_first_cpm"] = k_to_cpm(
                order, summary[f"k50_{name}"], part.edges
            )
        elif name == "utility":
            summary["k50_remove"] = summary[f"k50_{name}"]
            summary["k80_remove"] = summary[f"k80_{name}"]
            summary["k50_keep"] = k_alpha(keep_frac, 0.5)
            summary["k80_keep"] = k_alpha(keep_frac, 0.8)
            top = max(1, int(math.ceil(0.10 * len(order))))
            summary["damage_removed_top_10pct"] = float(rem_frac[top - 1])
            summary["damage_kept_top_10pct"] = float(keep_frac[top - 1])

    # A-priori band necessity / sufficiency (one remove + one keep per band).
    band_masks = []
    for b in BANDS:
        idx = [i for i in band_indices(b, part.n_shells) if i in set(live.tolist())]
        band_masks.append(
            part.masks[idx].any(axis=0) if idx else np.zeros_like(part.masks[0])
        )
    band_stack = np.stack(band_masks)
    rb = _run(
        model,
        case,
        cond,
        band_stack,
        "remove",
        num_classes,
        settings.iv_batch,
        op,
        keep_preds=False,
    )
    kb = _run(
        model,
        case,
        cond,
        band_stack,
        "keep",
        num_classes,
        settings.iv_batch,
        op,
        keep_preds=False,
    )
    for i, b in enumerate(BANDS):
        summary[f"{b.lower()}_necessity_dice"] = d_full - (
            clean_primary - primary(rb.dice[i])
        )
        summary[f"{b.lower()}_sufficiency_dice"] = clean_primary - primary(kb.dice[i])
    top8 = {int(v) for v in orders["utility"][: max(1, part.n_shells // 3)]}
    lf = set(band_indices("LF", part.n_shells))
    summary["jaccard_lf_top8_utility"] = len(top8 & lf) / len(top8 | lf)
    freq = -np.arange(part.n_shells, dtype=np.float64)
    summary["spearman_frequency_utility"] = _spearman(freq, necessity)
    summary["spearman_energy_utility"] = _spearman(energies, necessity)
    summary["spearman_ig_utility"] = _spearman(attributions, necessity)
    return curve_rows, summary, necessity, sufficiency, op


def interaction_stage(
    model: torch.nn.Module,
    case: SpectralCase,
    cond: SpectralCondition,
    num_classes: int,
    settings: Settings,
    sufficiency: np.ndarray,
    energies: np.ndarray,
    op: SpectralOperator,
) -> tuple[list[dict[str, object]], dict[str, object]]:
    part = case.partition
    primary = primary_fn(case.dataset_key)
    clean_primary = case.clean_metrics["dice"]
    d_full = cond.damages.get("dice", _NAN)
    damaging = math.isfinite(d_full) and d_full > 0
    live = [int(k) for k in np.nonzero(~part.empty)[0]]
    masks_by_id = {k + 1: part.masks[k] for k in live}
    base = {**_identity(case, cond), "patch_id": -1, "trial": -1}
    rows: list[dict[str, object]] = []

    greedy = greedy_restoration(
        model,
        _per_sample_bce,
        case.x_pad,
        cond.adv_pad,
        case.y_pad,
        masks_by_id,
        num_classes=num_classes,
        primary=primary,
        clean_primary=clean_primary,
        d_full=d_full,
        batch=settings.iv_batch,
        crop=case.shape,
        operator=op,
    )
    for g in greedy:
        rows.append(
            {
                **base,
                "control_type": "greedy",
                "patch_id": g["patch_id"],
                "greedy_step": g["step"],
                "greedy_recovered_fraction": g["recovered_fraction"],
                "recovery_fraction": g["recovered_fraction"],
            }
        )
    summary: dict[str, object] = {
        "greedy_steps": len(greedy),
        "greedy_k95": next(
            (g["step"] for g in greedy if g["recovered_fraction"] >= 0.95), _NAN
        ),
    }
    pairs = pairwise_interactions(
        model,
        _per_sample_bce,
        case.x_pad,
        cond.adv_pad,
        case.y_pad,
        masks_by_id,
        num_classes=num_classes,
        primary=primary,
        clean_primary=clean_primary,
        sufficiency_by_id={k + 1: float(sufficiency[k]) for k in live},
        batch=settings.iv_batch,
        crop=case.shape,
        operator=op,
    )
    for p in pairs:
        rows.append(
            {
                **base,
                "control_type": "pairwise",
                "pair_a": p["patch_a"],
                "pair_b": p["patch_b"],
                "joint_sufficiency": p["joint_sufficiency"],
                "interaction": p["interaction"],
            }
        )

    rng = np.random.default_rng(settings.base_seed + abs(cond.seed) + cond.epsilon_n)
    pooled_obs: list[float] = []
    pooled_ctl: list[float] = []
    for b in BANDS:
        idx = [i for i in band_indices(b, part.n_shells) if i in live]
        if not idx:
            continue
        band_mask = part.masks[idx].any(axis=0)[None]
        obs = _run(
            model, case, cond, band_mask, "remove", num_classes, 1, op, keep_preds=False
        )
        observed = (
            float((d_full - (clean_primary - primary(obs.dice[0]))) / d_full)
            if damaging
            else _NAN
        )
        target_e = float(energies[idx].sum())
        count = count_matched_draws(idx, part.n_shells, settings.count_trials, rng)
        energy_d, _tries = energy_matched_draws(
            idx,
            energies,
            settings.energy_trials,
            rng,
            tol=settings.energy_tol,
            max_tries=settings.energy_max_tries,
        )
        band_obs: list[float] = []
        band_ctl: list[float] = []
        energy_recs: list[float] = []
        for ctype, draws in (("count_matched", count), ("energy_matched", energy_d)):
            if not draws:
                continue
            cm = np.stack([part.masks[d].any(axis=0) for d in draws])
            res = _run(
                model,
                case,
                cond,
                cm,
                "remove",
                num_classes,
                settings.iv_batch,
                op,
                keep_preds=False,
            )
            for t, (d, dice_row) in enumerate(zip(draws, res.dice)):
                rec = (
                    float((d_full - (clean_primary - primary(dice_row))) / d_full)
                    if damaging
                    else _NAN
                )
                rows.append(
                    {
                        **base,
                        "control_type": f"{ctype}_joint",
                        "observed_band": b,
                        "trial": t,
                        "target_voxels": len(idx),
                        "control_voxels": len(d),
                        "target_energy": target_e,
                        "control_energy": float(energies[d].sum()),
                        "recovery_fraction": rec,
                        "patch_recovery_fraction": observed,
                        "n_patches_in_joint": len(idx),
                        "control_shells": " ".join(str(v + 1) for v in d),
                    }
                )
                if ctype == "count_matched":
                    band_obs.append(observed)
                    band_ctl.append(rec)
                else:
                    energy_recs.append(rec)
                pooled_obs.append(observed)
                pooled_ctl.append(rec)
        drawable = len(energy_d) / settings.energy_trials
        summary[f"{b.lower()}_observed_recovery"] = observed
        summary[f"{b.lower()}_control_permutation_p"] = permutation_p(
            band_obs, band_ctl
        )
        summary[f"{b.lower()}_control_drawable_fraction"] = drawable
        if b == "LF":
            summary.update(
                {
                    "observed_band_primary": "LF",
                    "observed_top10_recovery": observed,
                    "observed_matched_recovery": observed,
                    "observed_pool_size": float(len(idx)),
                    "control_pool_size": float(len(idx)),
                    "matched_pool_size_median": float(len(idx)),
                    "control_drawable_fraction": drawable,
                    "n_trials_skipped_undersized": settings.energy_trials
                    - len(energy_d),
                    "n_controls_volume": len(count),
                    "n_controls_energy": len(energy_d),
                    "control_volume_recovery_median": float(np.median(band_ctl))
                    if band_ctl
                    else _NAN,
                    "control_energy_recovery_median": float(np.median(energy_recs))
                    if energy_recs
                    else _NAN,
                    "control_permutation_p": permutation_p(band_obs, band_ctl),
                }
            )
    summary["control_permutation_p_pooled"] = permutation_p(pooled_obs, pooled_ctl)
    return rows, summary


def process_condition(
    model: torch.nn.Module,
    case: SpectralCase,
    cond: SpectralCondition,
    num_classes: int,
    settings: Settings,
    output_dir: Path,
) -> None:
    shell_rows, summary, energies = energy_stage(case, cond)
    cond.stage = "energy"
    summary.update(
        {
            **_identity(case, cond),
            "attack_seed": cond.seed,
            "attack_success": int(attack_success(-cond.damages.get("dice", _NAN))),
            "non_damaging": int(not (cond.damages.get("dice", _NAN) > 0)),
            **{f"clean_{m}": case.clean_metrics.get(m, _NAN) for m in METRICS},
            **{f"adv_{m}": cond.adv_metrics.get(m, _NAN) for m in METRICS},
            **{f"full_{m}_damage": cond.damages.get(m, _NAN) for m in METRICS},
            "linf_norm": cond.linf,
            "rms_norm": cond.rms,
            "rms_reproduction_rel_error": cond.rms_error,
            "epsilon_saturation_fraction": cond.saturation,
        }
    )
    curve_rows: list[dict[str, object]] = []
    control_rows: list[dict[str, object]] = []
    if "causal" in settings.stages and case.in_causal:
        cond.stage = "causal"
        curve_rows, s2, _necessity, sufficiency, op = causal_stage(
            model, case, cond, num_classes, settings, shell_rows, energies
        )
        summary.update(s2)
        if (
            "interaction" in settings.stages
            and case.in_interaction
            and cond.epsilon_n in INTERACTION_EPS
        ):
            cond.stage = "interaction"
            control_rows, s3 = interaction_stage(
                model, case, cond, num_classes, settings, sufficiency, energies, op
            )
            summary.update(s3)
    summary["stage"] = cond.stage
    for rows in (shell_rows, curve_rows, control_rows):
        for r in rows:
            r["stage"] = cond.stage
    if cond.rms_error > RMS_DRIFT_FLAG:
        cond.flags.append("rms_drift_above_flag")
    if settings.save_spectra:
        _save_npz(
            output_dir
            / SPECTRA_DIR
            / case.dataset_key
            / case.case_id
            / f"{cond.attack_label}_eps{cond.epsilon_n}.npz",
            shell_energy=energies,
            shell_edges_cpm=case.partition.edges,
            shell_counts=case.partition.counts,
            shell_attribution=np.array(
                [float(r.get("ig_attribution", _NAN)) for r in shell_rows]
            ),
        )
    _append(output_dir / SHELL_CSV, shell_rows, SHELL_FIELDNAMES)
    _append(output_dir / CURVES_CSV, curve_rows, CURVES_FIELDNAMES)
    _append(output_dir / CONTROL_CSV, control_rows, CONTROL_FIELDNAMES)
    _append(
        output_dir / QC_CSV,
        [
            {
                **_identity(case, cond),
                "n_flags": len(cond.flags),
                "flags": ";".join(cond.flags),
            }
        ],
        QC_FIELDNAMES,
    )
    _append(output_dir / SUMMARY_CSV, [summary], SUMMARY_FIELDNAMES)


# ---------------------------------------------------------------------------
# Dataset loop, CLI
# ---------------------------------------------------------------------------
def evaluate_dataset(
    dataset_key: str,
    *,
    attacks: list[str],
    epsilon_n_values: list[int],
    attack_steps: int,
    max_cases: int | None,
    case_subset: str,
    causal_cases: int,
    interaction_cases: int,
    force_interaction: bool,
    output_dir: Path,
    completed: set[tuple[str, str, int, str]],
    device: torch.device,
    settings: Settings,
    quality: pd.DataFrame | None,
    start_time: float,
    total_keys: int,
    trainer_dir: str = TRAINERS["unet"],
) -> None:
    paths = _dataset_paths(dataset_key, trainer_dir)
    all_case_ids = _available_case_ids(paths["data"])
    causal_set = set(subcohort_case_ids(all_case_ids, causal_cases))
    interaction_set = set(subcohort_case_ids(sorted(causal_set), interaction_cases))
    case_ids = case_ids_for_subset(
        all_case_ids, case_subset, causal_cases, interaction_cases
    )
    case_ids = case_ids[:max_cases] if max_cases is not None else case_ids
    requested = {
        (dataset_key, c, e, ATTACK_LABELS[a])
        for c in case_ids
        for a in attacks
        for e in epsilon_n_values
    }
    if requested <= completed:
        print(f"[{dataset_key}] all requested rows already complete", flush=True)
        return
    print(f"[{dataset_key}] loading fold-all UNet", flush=True)
    model, div_factors, num_classes = build_model(str(paths["checkpoint"]), device)
    model.eval()
    model.requires_grad_(False)
    for index, case_id in enumerate(case_ids, start=1):
        keys = {
            (dataset_key, case_id, e, ATTACK_LABELS[a])
            for a in attacks
            for e in epsilon_n_values
        }
        if keys <= completed:
            continue
        case = load_spectral_case(
            model,
            device,
            div_factors,
            num_classes,
            dataset_key,
            case_id,
            paths,
            in_causal=case_id in causal_set or force_interaction,
            in_interaction=case_id in interaction_set or force_interaction,
            n_shells=N_SHELLS_ACTIVE,
        )
        for attack_key in attacks:
            label = ATTACK_LABELS[attack_key]
            for epsilon_n in epsilon_n_values:
                if (dataset_key, case_id, epsilon_n, label) in completed:
                    continue
                t0 = time.monotonic()
                adv_pad, seed = run_attack(
                    model,
                    case,  # type: ignore[arg-type]  # duck-typed: x_pad, y_pad, ids
                    attack_key,
                    epsilon_n,
                    num_classes,
                    attack_steps=attack_steps,
                    base_seed=settings.base_seed,
                    valid_only=settings.attack_valid_only,
                )
                cond = build_spectral_condition(
                    model,
                    case,
                    attack_key,
                    label,
                    epsilon_n,
                    adv_pad,
                    seed,
                    num_classes,
                    quality,
                )
                process_condition(model, case, cond, num_classes, settings, output_dir)
                completed.add((dataset_key, case_id, epsilon_n, label))
                print(
                    f"[{dataset_key}] {case_id} {label} eps{epsilon_n}: stage={cond.stage} "
                    f"{time.monotonic() - t0:.1f}s",
                    flush=True,
                )
                del adv_pad, cond
        elapsed = time.monotonic() - start_time
        rate = len(completed) / max(elapsed, 1e-9)
        eta = max(total_keys - len(completed), 0) / max(rate, 1e-9) / 60.0
        print(
            f"[{dataset_key}] case {index}/{len(case_ids)} ({case_id}); "
            f"keys {len(completed)}/{total_keys}; ETA {eta:.1f} min",
            flush=True,
        )
        del case
        if device.type == "cuda":
            torch.cuda.empty_cache()
    del model
    gc.collect()
    if device.type == "cuda":
        torch.cuda.empty_cache()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument(
        "--datasets", nargs="+", choices=["wg", "zones"], default=["wg", "zones"]
    )
    parser.add_argument("--eps-n", nargs="+", type=int, default=list(DEFAULT_EPS_N))
    parser.add_argument(
        "--attacks",
        nargs="+",
        choices=list(ATTACK_LABELS),
        default=list(DEFAULT_ATTACKS),
    )
    parser.add_argument(
        "--stages", nargs="+", choices=list(STAGES), default=list(STAGES)
    )
    parser.add_argument("--attack-steps", type=int, default=20)
    parser.add_argument("--max-cases", type=int, default=None)
    parser.add_argument("--case-subset", choices=list(CASE_SUBSETS), default="causal")
    parser.add_argument("--causal-cases", type=int, default=100)
    parser.add_argument("--interaction-cases", type=int, default=30)
    parser.add_argument(
        "--force-interaction",
        action="store_true",
        help="treat every selected case as in the interaction subcohort (pilots)",
    )
    parser.add_argument("--n-shells", type=int, default=N_SHELLS)
    parser.add_argument("--seed", type=int, default=20260721)
    parser.add_argument("--device", type=str, default=None)
    parser.add_argument(
        "--deterministic", action=argparse.BooleanOptionalAction, default=True
    )
    parser.add_argument("--ig-steps", type=int, default=32)
    parser.add_argument("--ig-batch", type=int, default=8)
    parser.add_argument("--iv-batch", type=int, default=8)
    parser.add_argument("--boundary-workers", type=int, default=4)
    parser.add_argument("--count-trials", type=int, default=20)
    parser.add_argument("--energy-trials", type=int, default=10)
    parser.add_argument("--energy-tol", type=float, default=0.10)
    parser.add_argument("--energy-max-tries", type=int, default=2000)
    parser.add_argument(
        "--save-spectra", action=argparse.BooleanOptionalAction, default=True
    )
    parser.add_argument("--quality-csv", type=Path, default=DEFAULT_QUALITY_CSV)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument(
        "--attack-valid-only",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="confine the perturbation to non-ignore voxels (excludes the zero-filled "
        "margin outside the dilated gland on zones, and the model-grid padding). "
        "The published campaign ran without it; use a fresh --output-dir.",
    )
    parser.add_argument(
        "--trainer-dir",
        type=str,
        default=TRAINERS["unet"],
        help="nnU-Net results folder (<Trainer>__<Plans>__<config>) whose fold_all "
        "checkpoint is attacked, e.g. a defended fine-tune. Use a fresh --output-dir.",
    )
    return parser.parse_args()


def main() -> None:
    global ATLAS_TYPE, N_SHELLS_ACTIVE
    args = parse_args()
    for name in (
        "attack_steps",
        "ig_steps",
        "ig_batch",
        "iv_batch",
        "causal_cases",
        "interaction_cases",
        "count_trials",
        "energy_trials",
        "n_shells",
    ):
        if getattr(args, name) < 1:
            raise ValueError(f"--{name.replace('_', '-')} must be positive")
    if args.n_shells % 3:
        raise ValueError("--n-shells must be a multiple of 3 so the bands are terciles")
    N_SHELLS_ACTIVE = int(args.n_shells)
    ATLAS_TYPE = f"radial_shells_{N_SHELLS_ACTIVE}"
    epsilon_n_values = sorted({int(v) for v in args.eps_n})
    if not epsilon_n_values or any(v <= 0 for v in epsilon_n_values):
        raise ValueError("--eps-n values must be positive")
    attacks = list(dict.fromkeys(str(v) for v in args.attacks))
    stages = tuple(s for s in STAGES if s in set(args.stages))
    if "interaction" in stages and "causal" not in stages:
        raise ValueError("--stages interaction requires causal")
    _configure_deterministic_execution(enabled=args.deterministic, seed=args.seed)
    output_dir = args.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    device = (
        torch.device(args.device)
        if args.device
        else torch.device("cuda" if torch.cuda.is_available() else "cpu")
    )
    settings = Settings(
        stages,
        args.ig_steps,
        args.ig_batch,
        args.iv_batch,
        args.boundary_workers,
        args.count_trials,
        args.energy_trials,
        args.energy_tol,
        args.energy_max_tries,
        bool(args.save_spectra),
        int(args.seed),
        bool(args.attack_valid_only),
    )
    quality = None
    if (
        args.trainer_dir != TRAINERS["unet"] or args.attack_valid_only
    ) and args.quality_csv == DEFAULT_QUALITY_CSV:
        print(
            f"trainer {args.trainer_dir!r}, attack_valid_only={bool(args.attack_valid_only)}: "
            "published RMS references are for the clean model attacked without "
            "confinement; drift check skipped (pass --quality-csv to enable)",
            flush=True,
        )
    elif args.quality_csv.is_file():
        quality = pd.read_csv(
            args.quality_csv,
            usecols=["dataset", "case_id", "epsilon_n", "attack", "rms_norm"],
        ).set_index(["dataset", "case_id", "epsilon_n", "attack"])
    edges = reference_shell_edges(N_SHELLS_ACTIVE)
    manifest_path = output_dir / "run_manifest.json"
    manifest = {
        "datasets": list(args.datasets),
        "epsilon_n_values": epsilon_n_values,
        "attacks": attacks,
        "stages": list(stages),
        "attack_steps": int(args.attack_steps),
        "max_cases": args.max_cases,
        "case_subset": args.case_subset,
        "causal_cases": int(args.causal_cases),
        "interaction_cases": int(args.interaction_cases),
        "force_interaction": bool(args.force_interaction),
        "n_shells": N_SHELLS_ACTIVE,
        "atlas_type": ATLAS_TYPE,
        "shell_edges_cpm": [None if math.isinf(e) else float(e) for e in edges],
        "seed": int(args.seed),
        "device": str(device),
        "ig_steps": int(args.ig_steps),
        "ig_batch": int(args.ig_batch),
        "iv_batch": int(args.iv_batch),
        "count_trials": int(args.count_trials),
        "energy_trials": int(args.energy_trials),
        "energy_tol": float(args.energy_tol),
        "energy_max_tries": int(args.energy_max_tries),
        "interaction_eps": list(INTERACTION_EPS),
        "tables": [SUMMARY_CSV, SHELL_CSV, CURVES_CSV, CONTROL_CSV, QC_CSV],
        "trainer_dir": str(args.trainer_dir),
        "attack_valid_only": bool(args.attack_valid_only),
        "checkpoints": {
            d: str(_dataset_paths(d, args.trainer_dir)["checkpoint"].resolve())
            for d in args.datasets
        },
    }
    if manifest_path.is_file():
        previous = json.loads(manifest_path.read_text())
        # manifests written before --trainer-dir existed were all clean-model runs
        previous.setdefault("trainer_dir", TRAINERS["unet"])
        previous.setdefault("attack_valid_only", False)  # older runs never confined
        for key in (
            "trainer_dir",
            "attack_valid_only",
            "seed",
            "attack_steps",
            "ig_steps",
            "n_shells",
            "shell_edges_cpm",
            "causal_cases",
            "interaction_cases",
            "stages",
            "iv_batch",
            "ig_batch",
            "count_trials",
            "energy_trials",
            "energy_tol",
            "energy_max_tries",
        ):
            if previous.get(key) != json.loads(json.dumps(manifest[key])):
                raise ValueError(
                    f"resume parameter mismatch for {key!r}: {previous.get(key)} != {manifest[key]}"
                )
    manifest_path.write_text(json.dumps(manifest, indent=2, sort_keys=True))
    completed = completed_keys(output_dir / SUMMARY_CSV)
    total_keys = 0
    for d in args.datasets:
        ids = case_ids_for_subset(
            _available_case_ids(_dataset_paths(d)["data"]),
            args.case_subset,
            int(args.causal_cases),
            int(args.interaction_cases),
        )
        ids = ids[: args.max_cases] if args.max_cases is not None else ids
        total_keys += len(ids) * len(attacks) * len(epsilon_n_values)
    start = time.monotonic()
    for dataset_key in args.datasets:
        evaluate_dataset(
            dataset_key,
            attacks=attacks,
            epsilon_n_values=epsilon_n_values,
            attack_steps=int(args.attack_steps),
            max_cases=args.max_cases,
            case_subset=args.case_subset,
            causal_cases=int(args.causal_cases),
            interaction_cases=int(args.interaction_cases),
            force_interaction=bool(args.force_interaction),
            output_dir=output_dir,
            completed=completed,
            device=device,
            settings=settings,
            quality=quality,
            start_time=start,
            total_keys=total_keys,
            trainer_dir=str(args.trainer_dir),
        )
    print(
        f"done: {len(completed)} keys in {(time.monotonic() - start) / 60:.1f} min",
        flush=True,
    )


if __name__ == "__main__":
    main()
