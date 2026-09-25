"""Edge-specific energy–utility attribution for the FGSM/PGD/APGD cohort.

Regenerates the same deterministic attacks as
``evaluate_adversarial_quality_all_cases.py`` one case at a time, builds the
clean-derived edge atlases once per case (cached under ``atlases/``), and runs
the requested stages: ``energy`` (per-patch energy accounting), ``causal``
(Integrated Gradients, patch removal and retention), ``interaction`` (greedy
restoration, pairwise interactions, matched controls).  Rows are appended per
case; ``case_summary.csv`` is the resume marker.

Dice is recomputed here and never joined from the boundary table: APGD does
not reproduce bit-exactly between runs, and every quantity must describe the
perturbation measured in this run.

Re-attacking a defended checkpoint: ``--trainer-dir <Trainer>__<Plans>__3d_fullres``
selects whose ``fold_all/checkpoint_final.pth`` is attacked (default: the clean
model).  Use a fresh ``--output-dir``; the run manifest records the trainer and
refuses to resume into a directory produced with a different one.  The published
RMS drift references belong to the clean model attacked without confinement, so
they are skipped for any other trainer or for ``--attack-valid-only`` unless
``--quality-csv`` is given explicitly.

``--attack-valid-only`` confines the perturbation to voxels whose label is not
``IGNORE_LABEL`` (the zero-filled margin outside the dilated gland on zones, and
the model-grid padding); default off, the published campaign ran without it.
"""

from __future__ import annotations

import argparse
import csv
import gc
import json
import math
import sys
import time
from collections.abc import Callable, Sequence
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field
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
    build_model,
    load_sample,
)
from experiments.evaluate_adversarial_quality_all_cases import (  # noqa: E402
    ATTACK_LABELS,
    DEFAULT_EPS_N,
    TRAINERS,
    _available_case_ids,
    _dataset_paths,
    _load_normalization_lookup,
    _per_sample_bce,
    apgd_bce_independent_batch,
    fgsm_bce_independent_batch,
    pgd_bce_independent_batch,
)
from experiments.evaluate_adversarial_segmentation_all_cases import (  # noqa: E402
    _configure_deterministic_execution,
)
from mri_prostate_seg.experiments.edge_attribution import (  # noqa: E402
    AtlasConfig,
    EdgeAtlas,
    atlas_flags,
    build_case_atlases,
    concentration_indices,
    conservation_error,
    cumulative_curves,
    damage,
    distance_band,
    energy_map,
    energy_matched_region,
    fractional,
    k_alpha,
    local_damage_direction,
    patch_energy_table,
    patch_static_features,
    primary_dice,
    rank_agreement,
    rank_order,
    screen_patches,
    subcohort_case_ids,
    surface_metrics,
    top_fraction_ids,
    volume_matched_region,
)
from mri_prostate_seg.experiments.edge_attribution.torch_ops import (  # noqa: E402
    evaluate_interventions,
    greedy_restoration,
    integrated_attribution,
    pairwise_interactions,
)
from mri_prostate_seg.experiments.epsilon_calibration import (  # noqa: E402
    matched_gaussian_noise,
    matched_rician_proxy_noise,
    stable_seed,
)
from mri_prostate_seg.experiments.perturbation_structure import (  # noqa: E402
    signed_distance_mm,
)
from mri_prostate_seg.experiments.segmentation_direction import (  # noqa: E402
    attack_success,
)

DEFAULT_RESULTS = (
    REPO_ROOT / "results" / "adv_rob_eval_results" / "image_quality_analysis"
)
DEFAULT_OUTPUT_DIR = DEFAULT_RESULTS / "all_cases_edge_attribution"
DEFAULT_QUALITY_CSV = (
    DEFAULT_RESULTS
    / "all_cases_adversarial_quality"
    / "adversarial_quality_all_cases.csv"
)
DEFAULT_NORMALIZATION_CSV = DEFAULT_RESULTS / "normalization_per_case.csv"
SUMMARY_CSV = "case_summary.csv"
PATCH_CSV = "patch_metrics.csv"
CURVES_CSV = "subset_curves.csv"
CONTROL_CSV = "control_summary.csv"
QC_CSV = "qc.csv"
ATLAS_DIR = "atlases"
MAP_DIR = "maps"
IDENTITY = ["dataset", "case_id", "epsilon_n", "attack", "atlas_type"]
STAGES = ("energy", "causal", "interaction")
# `flags` is a free-text, `;`-joined column (see `qc_rows.append` below) --
# any exception text folded into it must not itself contain `;` and must stay
# bounded so one pathological message can't blow up the QC row.
QC_DETAIL_MAX_CHARS = 200
INTERACTION_EPS = (16, 32)
EXPECTED_ATLASES: dict[str, tuple[str, ...]] = {
    "wg": ("native_edges", "outer_boundary_gt"),
    "zones": ("native_edges", "outer_boundary_gt", "zone_interface_gt"),
}
TOP_FRACTIONS = (0.01, 0.05, 0.10, 0.20)
LADDER = (1, 2, 3, 5, 8, 13, 21, 34, 55, 89)
SURFACE_TOP = 10
CONTROL_LABELS = {
    "gaussian": "gaussian_rms_matched",
    "rician": "rician_proxy_rms_matched",
}
METRICS = ("dice", "dice_tz", "dice_pz", "hd95", "asd", "loss")
_NAN = float("nan")

STATIC_KEYS = [
    "voxel_count",
    "volume_mm3",
    "centroid_z_mm",
    "centroid_y_mm",
    "centroid_x_mm",
    "signed_distance_min_mm",
    "signed_distance_median_mm",
    "signed_distance_max_mm",
    "anatomical_category",
    "edge_strength_mean",
    "edge_strength_median",
    "edge_strength_p90",
    "clean_entropy_mean",
    "clean_margin_mean",
    "overlap_native_edges",
    "overlap_outer_boundary_gt",
    "overlap_zone_interface_gt",
]
ENERGY_KEYS = [
    "energy",
    "energy_density",
    "energy_share_roi",
    "energy_share_fov",
    "edge_energy_share",
    "spatial_share",
    "fold_enrichment",
    "percentage_point_enrichment",
]
IG_KEYS = [
    "ig_signed",
    "ig_positive",
    "ig_negative",
    "ig_positive_share",
    "ig_per_energy",
]
CAUSAL_KEYS = [
    "screened",
    "screen_reason",
    "utility_rank",
    *[f"necessity_{m}" for m in METRICS],
    "fractional_recovery_dice",
    "necessity_per_energy",
    *[f"sufficiency_{m}" for m in METRICS],
    "fractional_reproduction_dice",
    "metric_undefined",
]
LOCAL_KEYS = [
    "local_induced_fp_mm3",
    "local_induced_fn_mm3",
    "local_signed_displacement_median_mm",
    "local_outward_displacement_p90_mm",
    "local_inward_displacement_p10_mm",
    "local_surface_vertices",
    "pz_to_tz_mm3",
    "tz_to_pz_mm3",
]
PATCH_FIELDNAMES = [
    *IDENTITY,
    "stage",
    "patch_id",
    *STATIC_KEYS,
    *ENERGY_KEYS,
    *IG_KEYS,
    *CAUSAL_KEYS,
    *LOCAL_KEYS,
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
    "roi_voxels",
    "fov_voxels",
    "total_energy_roi",
    "total_energy_fov",
    "edge_energy_fraction_roi",
    "number_of_patches",
    "conservation_error",
    "n_eff",
    "n_eff_norm",
    "entropy_concentration",
    "energy_gini",
    *[f"energy_top_{int(f * 100)}pct" for f in TOP_FRACTIONS],
    "n_patches_screened",
    "ig_steps_used",
    "ig_completeness_error",
    "ig_positive_total",
    "utility_gini",
    "utility_n_eff",
    *[f"damage_removed_top_{int(f * 100)}pct" for f in TOP_FRACTIONS],
    *[f"damage_kept_top_{int(f * 100)}pct" for f in TOP_FRACTIONS],
    "k50_remove",
    "k80_remove",
    "k50_keep",
    "k80_keep",
    "useful_subset_volume_fraction",
    "spearman_strength_energy",
    "spearman_energy_utility",
    "spearman_strength_utility",
    "jaccard_strength_energy",
    "jaccard_strength_utility",
    "jaccard_energy_utility",
    "greedy_steps",
    "greedy_k95",
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
]
CURVES_FIELDNAMES = [
    *IDENTITY,
    "stage",
    "ranking_type",
    "k",
    "evaluated",
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
]
CONTROL_FIELDNAMES = [
    *IDENTITY,
    "stage",
    "control_type",
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
]
QC_FIELDNAMES = [*IDENTITY, "stage", "n_flags", "flags"]


# ---------------------------------------------------------------------------
# Paths, resume, I/O
# ---------------------------------------------------------------------------
def expected_atlases(dataset_key: str) -> set[str]:
    return set(EXPECTED_ATLASES[dataset_key])


def completed_keys(summary_csv: Path) -> set[tuple[str, str, int, str]]:
    if not summary_csv.is_file() or summary_csv.stat().st_size == 0:
        return set()
    frame = pd.read_csv(summary_csv, usecols=IDENTITY)
    done: set[tuple[str, str, int, str]] = set()
    key_columns = IDENTITY[:4]
    for key, group in frame.groupby(key_columns, sort=False):
        if expected_atlases(str(key[0])) <= set(group["atlas_type"].astype(str)):
            done.add((str(key[0]), str(key[1]), int(key[2]), str(key[3])))
    return done


CASE_SUBSETS = ("all", "causal", "interaction")


def case_ids_for_subset(
    all_case_ids: list[str], subset: str, causal_cases: int, interaction_cases: int
) -> list[str]:
    """Restrict the case list to a nested subcohort, preserving the input order.

    Phase D re-runs only the ``interaction`` subcohort at its two budgets; without
    this filter a fresh output directory would redo ``energy`` for every case.
    """
    if subset not in CASE_SUBSETS:
        raise ValueError(f"case subset must be one of {CASE_SUBSETS}, got {subset!r}")
    if subset == "all":
        return list(all_case_ids)
    keep = set(subcohort_case_ids(all_case_ids, causal_cases))
    if subset == "interaction":
        keep = set(subcohort_case_ids(sorted(keep), interaction_cases))
    return [c for c in all_case_ids if c in keep]


def atlas_path(output_dir: Path, dataset_key: str, case_id: str) -> Path:
    return output_dir / ATLAS_DIR / dataset_key / f"{case_id}.npz"


def map_path(
    output_dir: Path, dataset_key: str, case_id: str, attack_label: str, epsilon_n: int
) -> Path:
    return (
        output_dir
        / MAP_DIR
        / dataset_key
        / case_id
        / f"{attack_label}_eps{int(epsilon_n)}.npz"
    )


def _save_npz(path: Path, **arrays: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(".npz.partial")
    with temporary.open("wb") as handle:
        np.savez_compressed(handle, **arrays)
    temporary.replace(path)


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


def load_or_build_atlases(
    path: Path,
    image: np.ndarray,
    seg: np.ndarray,
    spacing: tuple[float, float, float],
    *,
    config: AtlasConfig,
    clean_pred: np.ndarray | None = None,
) -> tuple[dict[str, EdgeAtlas], np.ndarray, np.ndarray]:
    """Read the cached atlases or build and cache them; a bad cache is rebuilt."""

    if path.is_file():
        try:
            with np.load(path) as data:
                names = [str(n) for n in data["names"]]
                if clean_pred is None or "cores_outer_boundary_pred" in data:
                    atlases = {
                        n: EdgeAtlas(n, data[f"cores_{n}"], data[f"tubes_{n}"])
                        for n in names
                    }
                    return (
                        atlases,
                        data["roi"].astype(bool),
                        data["strength"].astype(np.float64),
                    )
        except Exception as error:  # noqa: BLE001 - any unreadable cache is rebuilt
            print(f"[atlas] rebuilding {path.name}: {error}", flush=True)
    atlases, roi, strength = build_case_atlases(
        image, seg, spacing, config=config, clean_pred=clean_pred
    )
    arrays: dict[str, np.ndarray] = {
        "names": np.array(list(atlases), dtype=str),
        "roi": roi.astype(np.uint8),
        "strength": strength.astype(np.float32),
    }
    for name, atlas in atlases.items():
        arrays[f"cores_{name}"] = atlas.cores.astype(np.int32)
        arrays[f"tubes_{name}"] = atlas.tubes.astype(np.int32)
    _save_npz(path, **arrays)
    return atlases, roi, strength


# ---------------------------------------------------------------------------
# Small helpers
# ---------------------------------------------------------------------------
def pad_masks(masks: np.ndarray, padded_shape: tuple[int, ...]) -> np.ndarray:
    out = np.zeros((masks.shape[0], *padded_shape), dtype=bool)
    d, h, w = masks.shape[1:]
    out[:, :d, :h, :w] = masks
    return out


def primary_fn(dataset_key: str) -> Callable[[np.ndarray], float]:
    if dataset_key == "wg":
        return lambda row: float(row[0])
    if dataset_key == "zones":
        return lambda row: float(np.mean(row[:2]))
    raise ValueError(f"unknown dataset {dataset_key!r}")


def condition_damages(
    clean: dict[str, float], adv: dict[str, float]
) -> tuple[dict[str, float], list[str]]:
    damages: dict[str, float] = {}
    undefined: list[str] = []
    for metric in METRICS:
        if metric not in clean or metric not in adv:
            continue
        value = damage(metric, clean[metric], adv[metric])
        damages[metric] = value
        if math.isnan(value):
            undefined.append(metric)
    return damages, undefined


def curve_k_values(k_max: int) -> list[int]:
    ks = {k for k in LADDER if k <= k_max}
    ks |= {max(1, int(math.ceil(f * k_max))) for f in TOP_FRACTIONS}
    ks.add(k_max)
    return sorted(ks)


def _finite_or(value: object, default: float) -> float:
    try:
        number = float(value)  # type: ignore[arg-type]
    except (TypeError, ValueError):
        return default
    return number if math.isfinite(number) else default


def _unpad(tensor: torch.Tensor, shape: tuple[int, ...]) -> np.ndarray:
    slices = tuple(slice(0, v) for v in shape)
    return tensor[(..., *slices)].detach().cpu().numpy()


def _metrics(
    pred: np.ndarray,
    seg: np.ndarray,
    spacing: tuple[float, float, float],
    dataset_key: str,
    loss: float,
) -> dict[str, float]:
    out = primary_dice(pred, seg, dataset_key=dataset_key)
    out.update(surface_metrics(pred, seg, dataset_key=dataset_key, spacing=spacing))
    out["loss"] = float(loss)
    return out


def _rms_reference(
    quality: pd.DataFrame | None,
    dataset_key: str,
    case_id: str,
    epsilon_n: int,
    label: str,
) -> float:
    if quality is None:
        return _NAN
    try:
        return float(quality.loc[(dataset_key, case_id, epsilon_n, label), "rms_norm"])
    except KeyError:
        return _NAN


RMS_DRIFT_WARN = 1e-4
RMS_DRIFT_FLAG = 1e-2
RMS_DRIFT_FATAL = 0.25


def _rms_check(recomputed: float, published: float, key: str) -> float:
    """Compare this run's realized RMS with the published one.

    APGD takes sign() steps, so a ~1e-7 difference in GPU reduction order near a
    zero crossing flips a voxel by a full 2*alpha and compounds over 20 steps.
    That divergence is case-dependent and not rare: over the full 10-case pilot,
    2 of 5 whole-gland cases exceeded FLAG (3.7e-2 and 2.8e-2) while all 5 zones
    cases stayed under 5.3e-3. A systematic regeneration bug would move every
    case uniformly, so a whole-gland-skewed minority is trajectory divergence,
    not a different attack — and every quantity this driver reports
    comes from the delta regenerated in this run, which is internally consistent
    regardless. Above FLAG the condition is recorded in qc.csv and continues;
    only gross drift, which would mean a genuinely different attack, is fatal.
    Raising on a single case would also stall its shard forever, since the
    resume marker is never written and the restart returns to the same case.
    """

    if not math.isfinite(published) or published <= 0:
        return _NAN
    error = abs(recomputed - published) / published
    if error > RMS_DRIFT_FATAL:
        raise RuntimeError(f"RMS reproduction drift {error:.3e} for {key}")
    if error > RMS_DRIFT_WARN:
        print(f"[warn] RMS reproduction drift {error:.3e} for {key}", flush=True)
    return error


# ---------------------------------------------------------------------------
# Per-case and per-condition state
# ---------------------------------------------------------------------------
@dataclass
class CaseContext:
    dataset_key: str
    case_id: str
    spacing: tuple[float, float, float]
    seg: np.ndarray
    valid: np.ndarray
    gt_fg: np.ndarray
    roi: np.ndarray
    strength: np.ndarray
    atlases: dict[str, EdgeAtlas]
    signed_distance: np.ndarray
    x_pad: torch.Tensor
    y_pad: torch.Tensor
    padded_shape: tuple[int, ...]
    shape: tuple[int, ...]
    clean_probs: np.ndarray
    clean_pred: np.ndarray
    clean_loss: float
    clean_metrics: dict[str, float]
    static: dict[str, dict[int, dict[str, object]]]
    in_causal: bool
    in_interaction: bool


@dataclass
class Condition:
    attack_key: str
    attack_label: str
    epsilon_n: int
    seed: int
    adv_pad: torch.Tensor
    delta: np.ndarray
    energy: np.ndarray
    adv_pred: np.ndarray
    adv_metrics: dict[str, float]
    damages: dict[str, float]
    undefined: list[str]
    linf: float
    rms: float
    saturation: float
    rms_error: float
    ig_attribution: np.ndarray | None = None
    ig_steps: int = 0
    ig_completeness: float = _NAN
    stage: str = "energy"
    flags: list[str] = field(default_factory=list)


@dataclass
class Settings:
    stages: tuple[str, ...]
    ig_steps: int
    ig_batch: int
    iv_batch: int
    boundary_workers: int
    greedy_limit: int
    pairwise_limit: int
    controls_per_patch: int
    control_pool_size: int
    local_radius_mm: float
    save_maps: bool
    base_seed: int
    attack_valid_only: bool = False


def _identity(case: CaseContext, cond: Condition, atlas_name: str) -> dict[str, object]:
    return {
        "dataset": case.dataset_key,
        "case_id": case.case_id,
        "epsilon_n": cond.epsilon_n,
        "attack": cond.attack_label,
        "atlas_type": atlas_name,
        "stage": cond.stage,
    }


def build_case(
    model: torch.nn.Module,
    device: torch.device,
    div_factors: list[int],
    num_classes: int,
    dataset_key: str,
    case_id: str,
    paths: dict[str, Path],
    output_dir: Path,
    config: AtlasConfig,
    *,
    in_causal: bool,
    in_interaction: bool,
) -> CaseContext:
    image_np, seg_np, spacing_raw = load_sample(str(paths["data"]), case_id)
    spacing = tuple(float(v) for v in spacing_raw)
    seg = np.asarray(seg_np).squeeze()
    if seg.ndim == 4:
        seg = seg[0]
    shape = tuple(int(v) for v in image_np.shape[1:])
    padded = tuple(int(math.ceil(v / f) * f) for v, f in zip(shape, div_factors))
    x_pad = _pad_to(torch.from_numpy(image_np[np.newaxis]).to(device), list(padded))
    y_pad = _pad_to(
        torch.from_numpy(seg_np[np.newaxis].astype(np.float32)).to(device),
        list(padded),
        mode="constant",
        value=float(IGNORE_LABEL),
    )
    with torch.no_grad():
        logits = model(x_pad)
        clean_loss = float(_per_sample_bce(logits, y_pad, num_classes)[0].item())
        probs = torch.softmax(logits, dim=1)
        pred = logits.argmax(dim=1)
    clean_probs = _unpad(probs[0], shape)
    clean_pred = _unpad(pred[0], shape).astype(np.int16)
    valid = seg >= 0
    gt_fg = seg > 0
    atlases, roi, strength = load_or_build_atlases(
        atlas_path(output_dir, dataset_key, case_id),
        image_np[0],
        seg,
        spacing,
        config=config,
    )
    sd = signed_distance_mm(gt_fg, spacing)
    cores = {name: atlas.cores > 0 for name, atlas in atlases.items()}
    static: dict[str, dict[int, dict[str, object]]] = {}
    for name, atlas in atlases.items():
        rows = patch_static_features(
            atlas,
            spacing=spacing,
            signed_distance=sd,
            strength=strength,
            probs=clean_probs,
            other_cores=cores,
        )
        static[name] = {int(r["patch_id"]): dict(r) for r in rows}
    return CaseContext(
        dataset_key,
        case_id,
        spacing,
        seg,
        valid,
        gt_fg,
        roi,
        strength,
        atlases,
        sd,
        x_pad,
        y_pad,
        padded,
        shape,
        clean_probs,
        clean_pred,
        clean_loss,
        _metrics(clean_pred, seg, spacing, dataset_key, clean_loss),
        static,
        in_causal,
        in_interaction,
    )


def attackable_mask(y_pad: torch.Tensor) -> torch.Tensor:
    """Voxels the deployed pipeline exposes to an image-level attack: everything
    except the ignore-labelled region (zero-filled margin on zones, and padding)."""
    return y_pad != IGNORE_LABEL


def run_attack(
    model: torch.nn.Module,
    case: CaseContext,
    attack_key: str,
    epsilon_n: int,
    num_classes: int,
    *,
    attack_steps: int,
    base_seed: int,
    valid_only: bool = False,
) -> tuple[torch.Tensor, int]:
    mask = attackable_mask(case.y_pad) if valid_only else None
    seed = stable_seed(
        case.dataset_key,
        case.case_id,
        f"{attack_key}_bce",
        epsilon_n,
        base_seed=base_seed,
    )
    epsilon = epsilon_n / 255.0
    with torch.enable_grad():
        if attack_key == "fgsm":
            adv = fgsm_bce_independent_batch(
                model, case.x_pad, case.y_pad, epsilon, num_classes, perturbation_mask=mask
            )
        elif attack_key == "pgd":
            adv = pgd_bce_independent_batch(
                model,
                case.x_pad,
                case.y_pad,
                epsilon,
                num_classes,
                n_steps=attack_steps,
                seeds=[seed],
                perturbation_mask=mask,
            )
        elif attack_key == "apgd":
            adv = apgd_bce_independent_batch(
                model,
                case.x_pad,
                case.y_pad,
                epsilon,
                num_classes,
                n_steps=attack_steps,
                seeds=[seed],
                perturbation_mask=mask,
            )
        else:  # pragma: no cover - guarded by argparse
            raise ValueError(f"unknown attack: {attack_key}")
    return adv.detach(), (-1 if attack_key == "fgsm" else seed)


def build_condition(
    model: torch.nn.Module,
    case: CaseContext,
    attack_key: str,
    attack_label: str,
    epsilon_n: int,
    adv_pad: torch.Tensor,
    seed: int,
    num_classes: int,
    quality: pd.DataFrame | None,
) -> Condition:
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
    return Condition(
        attack_key,
        attack_label,
        epsilon_n,
        seed,
        adv_pad,
        delta,
        energy_map(delta),
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
def energy_stage(
    case: CaseContext, cond: Condition, name: str
) -> tuple[dict[int, dict[str, object]], dict[str, object]]:
    atlas = case.atlases[name]
    fov = case.valid
    table = patch_energy_table(cond.energy, atlas.tubes, roi=case.roi, fov=fov)
    rows: dict[int, dict[str, object]] = {}
    for r in table:
        pid = int(r["patch_id"])
        row: dict[str, object] = {**_identity(case, cond, name), "patch_id": pid}
        row.update(case.static[name].get(pid, {}))
        row.update(r)
        rows[pid] = row
    energies = np.array([r["energy"] for r in table], dtype=np.float64)
    edge_total = float(energies.sum())
    total_roi = float(cond.energy[case.roi].sum())
    conc = (
        concentration_indices(energies)
        if energies.size
        else {k: _NAN for k in ("n_eff", "n_eff_norm", "entropy_concentration", "gini")}
    )
    summary: dict[str, object] = {
        **_identity(case, cond, name),
        "attack_seed": cond.seed,
        "attack_success": int(attack_success(-cond.damages.get("dice", _NAN))),
        "non_damaging": int(not (cond.damages.get("dice", _NAN) > 0)),
        "linf_norm": cond.linf,
        "rms_norm": cond.rms,
        "rms_reproduction_rel_error": cond.rms_error,
        "epsilon_saturation_fraction": cond.saturation,
        "roi_voxels": int(case.roi.sum()),
        "fov_voxels": int(fov.sum()),
        "total_energy_roi": total_roi,
        "total_energy_fov": float(cond.energy[fov].sum()),
        "edge_energy_fraction_roi": edge_total / total_roi if total_roi > 0 else _NAN,
        "number_of_patches": int(energies.size),
        "conservation_error": conservation_error(cond.energy, atlas.tubes, case.roi),
        "n_eff": conc["n_eff"],
        "n_eff_norm": conc["n_eff_norm"],
        "entropy_concentration": conc["entropy_concentration"],
        "energy_gini": conc["gini"],
    }
    for metric in METRICS:
        summary[f"clean_{metric}"] = case.clean_metrics.get(metric, _NAN)
        summary[f"adv_{metric}"] = cond.adv_metrics.get(metric, _NAN)
        summary[f"full_{metric}_damage"] = cond.damages.get(metric, _NAN)
    if energies.size:
        order = rank_order(energies)
        cum = (
            np.cumsum(energies[order]) / edge_total
            if edge_total > 0
            else np.full(energies.size, _NAN)
        )
        for f in TOP_FRACTIONS:
            k = max(1, int(math.ceil(f * energies.size)))
            summary[f"energy_top_{int(f * 100)}pct"] = float(cum[k - 1])
    return rows, summary


def _surface_for(
    preds: dict[int, np.ndarray], case: CaseContext, workers: int
) -> dict[int, dict[str, float]]:
    def compute(item: tuple[int, np.ndarray]) -> tuple[int, dict[str, float]]:
        index, pred = item
        return index, surface_metrics(
            pred[: case.shape[0], : case.shape[1], : case.shape[2]],
            case.seg,
            dataset_key=case.dataset_key,
            spacing=case.spacing,
        )

    if not preds:
        return {}
    if workers <= 1 or len(preds) < 2:
        return dict(compute(item) for item in preds.items())
    with ThreadPoolExecutor(max_workers=min(workers, len(preds))) as pool:
        return dict(pool.map(compute, list(preds.items())))


def causal_stage(
    model: torch.nn.Module,
    case: CaseContext,
    cond: Condition,
    name: str,
    rows: dict[int, dict[str, object]],
    summary: dict[str, object],
    num_classes: int,
    settings: Settings,
    rng: np.random.Generator,
) -> tuple[list[dict[str, object]], dict[str, np.ndarray], list[int]]:
    """IG per patch, screening, removal/retention, curves. Returns (curve rows, maps, utility-ordered ids)."""

    atlas = case.atlases[name]
    ids = np.array(sorted(rows), dtype=np.int64)
    if ids.size == 0 or cond.ig_attribution is None:
        return [], {}, []
    attribution = cond.ig_attribution
    positive_total = float(np.maximum(attribution, 0.0)[case.roi].sum())
    for pid in ids:
        mask = atlas.tubes == pid
        a = attribution[mask]
        r = rows[int(pid)]
        r["ig_signed"] = float(a.sum())
        r["ig_positive"] = float(np.maximum(a, 0.0).sum())
        r["ig_negative"] = float(np.minimum(a, 0.0).sum())
        r["ig_positive_share"] = (
            r["ig_positive"] / positive_total if positive_total > 0 else _NAN
        )
        r["ig_per_energy"] = r["ig_positive"] / (float(r["energy"]) + 1e-12)
    summary["ig_positive_total"] = positive_total
    summary["ig_steps_used"] = cond.ig_steps
    summary["ig_completeness_error"] = cond.ig_completeness

    selected, reason = screen_patches(
        ids,
        ig_positive=np.array([rows[int(i)]["ig_positive"] for i in ids]),
        energy_fold=np.array([rows[int(i)]["fold_enrichment"] for i in ids]),
        uncertainty=np.array(
            [rows[int(i)].get("clean_entropy_mean", _NAN) for i in ids]
        ),
        rng=rng,
        use_energy=cond.attack_key != "fgsm",
    )
    for pid in ids:
        rows[int(pid)]["screened"] = int(int(pid) in reason)
        rows[int(pid)]["screen_reason"] = reason.get(int(pid), "")
    summary["n_patches_screened"] = int(selected.size)
    sel = [int(i) for i in selected]
    masks = pad_masks(np.stack([atlas.tubes == i for i in sel]), case.padded_shape)
    primary = primary_fn(case.dataset_key)
    clean_primary = float(case.clean_metrics["dice"])
    d_full = float(cond.damages.get("dice", _NAN))

    removal = evaluate_interventions(
        model,
        _per_sample_bce,
        case.x_pad,
        cond.adv_pad,
        case.y_pad,
        masks,
        mode="remove",
        num_classes=num_classes,
        batch=settings.iv_batch,
        crop=case.shape,
    )
    necessity = np.array(
        [d_full - (clean_primary - primary(row)) for row in removal.dice]
    )
    utility_order = rank_order(necessity)
    top_surface = {int(i) for i in utility_order[:SURFACE_TOP]}
    if top_surface:
        removal_top = evaluate_interventions(
            model,
            _per_sample_bce,
            case.x_pad,
            cond.adv_pad,
            case.y_pad,
            masks[sorted(top_surface)],
            mode="remove",
            num_classes=num_classes,
            batch=settings.iv_batch,
            keep_pred_indices=set(range(len(top_surface))),
            crop=case.shape,
        )
        removal_surface = {
            sorted(top_surface)[k]: v
            for k, v in _surface_for(
                removal_top.preds, case, settings.boundary_workers
            ).items()
        }
    else:
        removal_surface = {}
    retention = evaluate_interventions(
        model,
        _per_sample_bce,
        case.x_pad,
        cond.adv_pad,
        case.y_pad,
        masks,
        mode="keep",
        num_classes=num_classes,
        batch=settings.iv_batch,
        keep_pred_indices=top_surface,
        crop=case.shape,
    )
    retention_surface = _surface_for(retention.preds, case, settings.boundary_workers)

    for position, pid in enumerate(sel):
        r = rows[pid]
        rem_row, keep_row = removal.dice[position], retention.dice[position]
        rem = {"dice": primary(rem_row), "loss": float(removal.loss[position])}
        keep = {"dice": primary(keep_row), "loss": float(retention.loss[position])}
        if case.dataset_key == "zones":
            rem.update(dice_tz=float(rem_row[0]), dice_pz=float(rem_row[1]))
            keep.update(dice_tz=float(keep_row[0]), dice_pz=float(keep_row[1]))
        rem.update(removal_surface.get(position, {}))
        keep.update(retention_surface.get(position, {}))
        undefined: list[str] = []
        for metric in METRICS:
            if metric in rem:
                value = damage(metric, case.clean_metrics[metric], rem[metric])
                r[f"necessity_{metric}"] = (
                    cond.damages[metric] - value if math.isfinite(value) else _NAN
                )
                if not math.isfinite(value):
                    undefined.append(metric)
            if metric in keep:
                r[f"sufficiency_{metric}"] = damage(
                    metric, case.clean_metrics[metric], keep[metric]
                )
        r["fractional_recovery_dice"] = fractional(float(r["necessity_dice"]), d_full)
        r["fractional_reproduction_dice"] = fractional(
            float(r["sufficiency_dice"]), d_full
        )
        r["necessity_per_energy"] = float(r["necessity_dice"]) / (
            float(r["energy"]) + 1e-12
        )
        r["utility_rank"] = int(np.nonzero(utility_order == position)[0][0]) + 1
        r["metric_undefined"] = ";".join(undefined)

    ordered = [sel[int(i)] for i in utility_order]
    curves = _curves_for_rankings(
        model,
        case,
        cond,
        name,
        rows,
        sel,
        necessity,
        num_classes,
        settings,
        summary,
        d_full,
        clean_primary,
        primary,
    )
    _local_direction(case, cond, name, rows, sel, settings)
    weights = np.maximum(necessity, 0.0)
    util_conc = concentration_indices(weights)
    summary["utility_gini"] = util_conc["gini"]
    summary["utility_n_eff"] = util_conc["n_eff"]
    strength = np.array([rows[i]["edge_strength_mean"] for i in sel])
    fold = np.array([rows[i]["fold_enrichment"] for i in sel])
    summary.update(rank_agreement(strength, fold, necessity))
    top10 = top_fraction_ids(utility_order, np.array(sel), 0.10)
    summary["useful_subset_volume_fraction"] = float(
        np.isin(atlas.tubes, list(top10)).sum() / max(int(case.roi.sum()), 1)
    )
    maps = {
        f"necessity_{name}": _patch_map(atlas.tubes, dict(zip(sel, necessity))),
        f"sufficiency_{name}": _patch_map(
            atlas.tubes, {pid: float(rows[pid]["sufficiency_dice"]) for pid in sel}
        ),
    }
    return curves, maps, ordered


def _patch_map(tubes: np.ndarray, values: dict[int, float]) -> np.ndarray:
    out = np.zeros(tubes.shape, dtype=np.float32)
    for pid, value in values.items():
        out[tubes == pid] = value
    return out


def _curves_for_rankings(
    model: torch.nn.Module,
    case: CaseContext,
    cond: Condition,
    name: str,
    rows: dict[int, dict[str, object]],
    sel: list[int],
    necessity: np.ndarray,
    num_classes: int,
    settings: Settings,
    summary: dict[str, object],
    d_full: float,
    clean_primary: float,
    primary: Callable[[np.ndarray], float],
) -> list[dict[str, object]]:
    atlas = case.atlases[name]
    e_total = float(cond.energy[case.roi].sum())
    e_edges = float(sum(float(rows[i]["energy"]) for i in rows))
    v_roi = int(case.roi.sum())
    energy = np.array([rows[i]["energy"] for i in sel], dtype=np.float64)
    voxels = np.array([rows[i]["voxel_count"] for i in sel], dtype=np.float64)
    orders = {
        "utility": rank_order(necessity),
        "energy": rank_order(np.array([rows[i]["fold_enrichment"] for i in sel])),
        "strength": rank_order(np.array([rows[i]["edge_strength_mean"] for i in sel])),
        "ig": rank_order(np.array([rows[i]["ig_positive"] for i in sel])),
    }
    k_max = len(sel)
    out: list[dict[str, object]] = []
    for ranking, order in orders.items():
        if ranking == "energy" and cond.attack_key == "fgsm":
            continue
        ks = (
            curve_k_values(k_max)
            if ranking == "utility"
            else sorted({max(1, int(math.ceil(f * k_max))) for f in TOP_FRACTIONS})
        )
        ids = [sel[int(i)] for i in order]
        removed = np.full(k_max, _NAN)
        kept = np.full(k_max, _NAN)
        surface: dict[int, dict[str, float]] = {}
        joint = pad_masks(
            np.stack([np.isin(atlas.tubes, ids[:k]) for k in ks]), case.padded_shape
        )
        keep_idx = {
            i
            for i, k in enumerate(ks)
            if k in {max(1, int(math.ceil(f * k_max))) for f in TOP_FRACTIONS}
        }
        rem = evaluate_interventions(
            model,
            _per_sample_bce,
            case.x_pad,
            cond.adv_pad,
            case.y_pad,
            joint,
            mode="remove",
            num_classes=num_classes,
            batch=settings.iv_batch,
            keep_pred_indices=keep_idx,
            crop=case.shape,
        )
        keep = evaluate_interventions(
            model,
            _per_sample_bce,
            case.x_pad,
            cond.adv_pad,
            case.y_pad,
            joint,
            mode="keep",
            num_classes=num_classes,
            batch=settings.iv_batch,
            keep_pred_indices=keep_idx,
            crop=case.shape,
        )
        rem_surface = _surface_for(rem.preds, case, settings.boundary_workers)
        keep_surface = _surface_for(keep.preds, case, settings.boundary_workers)
        for i, k in enumerate(ks):
            removed[k - 1] = clean_primary - primary(rem.dice[i])
            kept[k - 1] = clean_primary - primary(keep.dice[i])
            surface[k] = {
                **{f"{m}_removed": v for m, v in rem_surface.get(i, {}).items()},
                **{f"{m}_kept": v for m, v in keep_surface.get(i, {}).items()},
            }
        curve = cumulative_curves(
            order,
            energy=energy,
            voxels=voxels,
            e_total=e_total,
            e_edges=e_edges,
            v_roi=v_roi,
            damage_removed=removed,
            damage_kept=kept,
            d_full=d_full,
        )
        for row in curve:
            k = int(row["k"])
            out.append(
                {
                    **_identity(case, cond, name),
                    "ranking_type": ranking,
                    "evaluated": int(k in ks),
                    **row,
                    **surface.get(k, {}),
                }
            )
        if ranking == "utility":
            removed_fraction = np.array(
                [r["damage_removed_fraction"] for r in curve], dtype=np.float64
            )
            kept_fraction = np.array(
                [r["damage_kept_fraction"] for r in curve], dtype=np.float64
            )
            for f in TOP_FRACTIONS:
                k = max(1, int(math.ceil(f * k_max)))
                summary[f"damage_removed_top_{int(f * 100)}pct"] = float(
                    removed_fraction[k - 1]
                )
                summary[f"damage_kept_top_{int(f * 100)}pct"] = float(
                    kept_fraction[k - 1]
                )
            evaluated_removed = np.where(
                np.isin(np.arange(1, k_max + 1), ks), removed_fraction, _NAN
            )
            evaluated_kept = np.where(
                np.isin(np.arange(1, k_max + 1), ks), kept_fraction, _NAN
            )
            summary["k50_remove"] = k_alpha(
                np.nan_to_num(evaluated_removed, nan=-1.0), 0.5
            )
            summary["k80_remove"] = k_alpha(
                np.nan_to_num(evaluated_removed, nan=-1.0), 0.8
            )
            summary["k50_keep"] = k_alpha(np.nan_to_num(evaluated_kept, nan=-1.0), 0.5)
            summary["k80_keep"] = k_alpha(np.nan_to_num(evaluated_kept, nan=-1.0), 0.8)
    return out


def _local_direction(
    case: CaseContext,
    cond: Condition,
    name: str,
    rows: dict[int, dict[str, object]],
    sel: list[int],
    settings: Settings,
) -> None:
    zones = case.dataset_key == "zones"
    local = local_damage_direction(
        case.atlases[name],
        spacing=case.spacing,
        gt_fg=case.gt_fg,
        clean_fg=case.clean_pred > 0,
        adv_fg=cond.adv_pred > 0,
        valid=case.valid,
        radius_mm=settings.local_radius_mm,
        seg_gt=case.seg if zones else None,
        seg_clean=case.clean_pred if zones else None,
        seg_adv=cond.adv_pred if zones else None,
        patch_ids=sel,
    )
    for r in local:
        rows[int(r["patch_id"])].update(r)


_PatchTargets = tuple[int, np.ndarray, float, tuple[int, int]]


def _patch_targets(
    atlas: EdgeAtlas,
    case: CaseContext,
    rows: dict[int, dict[str, object]],
    patch_id: int,
    cache: dict[int, _PatchTargets],
) -> _PatchTargets:
    """Volume, distance band, energy and z-range a matched draw must match."""

    cached = cache.get(patch_id)
    if cached is not None:
        return cached
    tube = atlas.tubes == patch_id
    z = np.nonzero(tube)[0]
    targets: _PatchTargets = (
        int(tube.sum()),
        distance_band(
            case.signed_distance,
            _finite_or(rows[patch_id].get("signed_distance_median_mm"), 0.0),
        ),
        _finite_or(rows[patch_id].get("energy"), 0.0),
        (int(z.min()), int(z.max())),
    )
    cache[patch_id] = targets
    return targets


def select_pool_top_q(
    recoveries: Sequence[float], regions: Sequence[np.ndarray], q: int
) -> np.ndarray:
    """Union of the ``q`` pool regions with the highest MEASURED recovery.

    The control arm's half of the selection symmetry: the observed arm removes
    the ``q`` patches with the highest measured necessity out of its screened
    set, so the control must select on the same functional out of a pool of the
    same size.  Taking the first ``q`` drawn instead is exactly the
    selection-on-outcome defect this function exists to prevent.
    """

    order = sorted(range(len(regions)), key=lambda i: -recoveries[i])
    return np.logical_or.reduce([regions[i] for i in order[:q]])


def select_observed_matched_subset(
    pool_ids: Sequence[int], necessity: dict[int, float], q: int
) -> list[int]:
    """The ``q`` highest-necessity patches OF THE POOL'S OWN ID SET.

    The observed arm's half of the matched joint control.  ``pool_ids`` is
    ``P_succ``: exactly the patches whose matched twin this trial's pool is made
    of, one twin each.  Ranking those -- rather than a uniform subsample of the
    whole screened set -- is what makes the two arms' candidate sets share a
    volume, band, energy and slice-range distribution patch for patch.

    Matched draws do NOT succeed uniformly: ``volume_matched_region`` has to
    grow a connected region of ``target_voxels`` inside a thin signed-distance
    band against per-trial depletion, so large patches fail far more often
    (measured on the pilot's own atlases: P(success) 0.95 on the smallest volume
    quartile against 0.59 on the largest, and a drawable set averaging 78% of
    the full set's patch volume).  An observed arm sampled uniformly from
    ``ordered`` would therefore select its top q out of the full size
    distribution while the control selects out of a size-truncated one; union
    volume drives recovery, so ``control_permutation_p`` would be biased small
    again -- the same direction as the volume-matching and selection-on-outcome
    biases already fixed here, and invisible in the outputs.

    ``P_succ`` is selected on drawability, i.e. on geometry, never on any
    measured recovery -- so this is matching, not selection on outcome.
    """

    return sorted(pool_ids, key=lambda pid: -necessity[pid])[:q]


def interaction_stage(
    model: torch.nn.Module,
    case: CaseContext,
    cond: Condition,
    name: str,
    rows: dict[int, dict[str, object]],
    summary: dict[str, object],
    ordered: list[int],
    num_classes: int,
    settings: Settings,
    rng: np.random.Generator,
) -> list[dict[str, object]]:
    atlas = case.atlases[name]
    primary = primary_fn(case.dataset_key)
    clean_primary = float(case.clean_metrics["dice"])
    d_full = float(cond.damages.get("dice", _NAN))
    out: list[dict[str, object]] = []
    if not ordered or not (d_full > 0):
        return out
    base = {**_identity(case, cond, name)}
    greedy_ids = ordered[: settings.greedy_limit]
    masks_by_id = {
        pid: pad_masks((atlas.tubes == pid)[None], case.padded_shape)[0]
        for pid in greedy_ids
    }
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
    )
    summary["greedy_steps"] = len(greedy)
    summary["greedy_k95"] = next(
        (g["step"] for g in greedy if float(g["recovered_fraction"]) >= 0.95), _NAN
    )
    out.extend(
        {
            **base,
            "control_type": "greedy",
            "patch_id": g["patch_id"],
            "greedy_step": g["step"],
            "greedy_recovered_fraction": g["recovered_fraction"],
            "recovery_fraction": g["incremental_recovery"],
        }
        for g in greedy
    )

    by_sufficiency = sorted(
        ordered,
        key=lambda pid: -_finite_or(rows[pid].get("sufficiency_dice"), -math.inf),
    )[: settings.pairwise_limit]
    pair_masks = {
        pid: masks_by_id.get(
            pid, pad_masks((atlas.tubes == pid)[None], case.padded_shape)[0]
        )
        for pid in by_sufficiency
    }
    pairs = pairwise_interactions(
        model,
        _per_sample_bce,
        case.x_pad,
        cond.adv_pad,
        case.y_pad,
        pair_masks,
        num_classes=num_classes,
        primary=primary,
        clean_primary=clean_primary,
        sufficiency_by_id={
            pid: float(rows[pid]["sufficiency_dice"]) for pid in by_sufficiency
        },
        batch=settings.iv_batch,
        crop=case.shape,
    )
    out.extend(
        {
            **base,
            "control_type": "pairwise",
            "pair_a": p["patch_a"],
            "pair_b": p["patch_b"],
            "joint_sufficiency": p["joint_sufficiency"],
            "interaction": p["interaction"],
        }
        for p in pairs
    )

    top10 = [pid for pid in ordered[: max(1, int(math.ceil(0.10 * len(ordered))))]][:10]
    joint_top = pad_masks(np.isin(atlas.tubes, top10)[None], case.padded_shape)
    observed = evaluate_interventions(
        model,
        _per_sample_bce,
        case.x_pad,
        cond.adv_pad,
        case.y_pad,
        joint_top,
        mode="remove",
        num_classes=num_classes,
        batch=1,
        crop=case.shape,
    )
    observed_recovery = float(
        (d_full - (clean_primary - primary(observed.dice[0]))) / d_full
    )
    summary["observed_top10_recovery"] = observed_recovery
    # Joint controls select q regions, so the useful-vs-random comparison must
    # be against the removal of q real patches -- not the full top-10 -- or a
    # volume mismatch alone would bias control_permutation_p toward "real edges
    # beat random controls".  WHICH q is decided per trial, out of the same
    # patches that trial's control pool was matched to; see the joint-control
    # block below.
    q = max(1, int(math.ceil(0.6 * len(top10))))
    # The matched control is only defined for atlases that do not coincide with
    # their own signed-distance band.  outer_boundary_gt's patches ARE that
    # band, so `roi & (tubes == 0)` inside it is nearly empty (4 of 60 drawable
    # on the pilot) and "a random region at the boundary that is not the
    # boundary" has no referent.  That atlas fails the M > q gate honestly and
    # reports NaN plus no_matched_controls / control_pool_undersized; widening
    # its band to make a number appear would be measuring something else.
    base_candidate = case.roi & (atlas.tubes == 0)
    # Per-(control_type, trial) working candidate: voxels drawn for one patch
    # within a trial are removed before the next patch's draw in that same
    # trial, so two regions in one trial can't overlap or coincide -- otherwise
    # their union is smaller than the disjoint observed tubes it's compared
    # against, biasing recovery low and the p-value small.
    trial_candidates: dict[str, list[np.ndarray]] = {
        t: [base_candidate.copy() for _ in range(settings.controls_per_patch)]
        for t in ("volume_matched", "energy_matched")
    }
    per_patch: dict[str, list[float]] = {"volume_matched": [], "energy_matched": []}
    target_cache: dict[int, _PatchTargets] = {}
    for pid in top10:
        tube = atlas.tubes == pid
        z = np.nonzero(tube)[0]
        band = distance_band(
            case.signed_distance,
            _finite_or(rows[pid].get("signed_distance_median_mm"), 0.0),
        )
        target_energy = _finite_or(rows[pid].get("energy"), 0.0)
        for control_type in ("volume_matched", "energy_matched"):
            drawn: list[tuple[int, np.ndarray]] = []
            for trial in range(settings.controls_per_patch):
                trial_candidate = trial_candidates[control_type][trial]
                if control_type == "volume_matched":
                    region = volume_matched_region(
                        target_voxels=int(tube.sum()),
                        band=band,
                        candidate=trial_candidate,
                        rng=rng,
                        slices=(int(z.min()), int(z.max())),
                    )
                else:
                    region = energy_matched_region(
                        energy=cond.energy,
                        target_energy=target_energy,
                        target_voxels=int(tube.sum()),
                        band=band,
                        candidate=trial_candidate,
                        rng=rng,
                        slices=(int(z.min()), int(z.max())),
                    )
                if region is not None:
                    drawn.append((trial, region))
                    trial_candidates[control_type][trial] = trial_candidate & ~region
            if not drawn:
                continue
            result = evaluate_interventions(
                model,
                _per_sample_bce,
                case.x_pad,
                cond.adv_pad,
                case.y_pad,
                pad_masks(np.stack([r for _, r in drawn]), case.padded_shape),
                mode="remove",
                num_classes=num_classes,
                batch=settings.iv_batch,
                crop=case.shape,
            )
            for (trial, region), dice_row in zip(drawn, result.dice):
                recovery = float(
                    (d_full - (clean_primary - primary(dice_row))) / d_full
                )
                per_patch[control_type].append(recovery)
                out.append(
                    {
                        **base,
                        "control_type": control_type,
                        "patch_id": pid,
                        "trial": trial,
                        "target_voxels": int(tube.sum()),
                        "control_voxels": int(region.sum()),
                        "target_energy": target_energy,
                        "control_energy": float(cond.energy[region].sum()),
                        "recovery_fraction": recovery,
                        "patch_recovery_fraction": rows[pid][
                            "fractional_recovery_dice"
                        ],
                    }
                )
    # Joint control: the SAME selection, out of the SAME M patches, on both
    # arms.  Two distinct defects have to stay fixed here at once.
    #
    # (1) Selection on outcome.  The observed arm removes the q patches with
    # the highest MEASURED necessity; q matched draws that were selected on
    # nothing are not a comparison, because under a null where every patch has
    # identical true necessity the observed statistic is still a maximum of
    # noisy measurements while each control is a single draw.  So the control
    # arm performs the same selection: measure every pool region's
    # single-region recovery with the identical functional the observed ranking
    # uses, and remove the union of the top q.
    #
    # (2) Matching of the two candidate SETS.  Each pool member is the matched
    # twin of one specific patch, and matched draws fail -- large targets, thin
    # bands, per-trial depletion -- so the realised pool is the twin set of
    # P_succ, the patches whose draw succeeded, and P_succ is systematically
    # short of the large ones.  The observed arm therefore ranks P_succ itself
    # (see select_observed_matched_subset), not the whole screened set and not
    # a uniform subsample of it: only then do the two arms' candidate sets have
    # the same volume/band/energy/slice distribution, patch for patch.
    #
    # Both arms are now "top q by their own single measurement out of the same
    # M elements", so the max-of-noise advantage and the volume advantage both
    # cancel by construction.  M is whatever the control side could realise:
    # --control-pool-size still caps the REQUESTED pool, but the realised M is
    # what both arms use, and the endpoint is conditional on drawability --
    # reported as control_drawable_fraction rather than left in a comment.
    requested_pool = (
        len(ordered)
        if settings.control_pool_size <= 0
        else min(int(settings.control_pool_size), len(ordered))
    )
    summary["observed_pool_size"] = len(ordered)
    necessity_by_id = {
        pid: _finite_or(rows[pid].get("necessity_dice"), -math.inf) for pid in ordered
    }
    realised_pools: list[int] = []
    matched_pools: list[int] = []
    skipped_undersized = 0
    observed_recoveries: list[float] = []
    # (control_t, observed_t) per qualifying trial: both arms share M within a
    # trial, so the permutation p-value pairs them trial-wise.
    paired: list[tuple[float, float]] = []
    for control_type in ("volume_matched", "energy_matched"):
        unions: list[tuple[int, np.ndarray, int, float, int]] = []
        for trial in range(settings.controls_per_patch):
            # Shuffled walk over `ordered`: in rank order the per-trial
            # candidate depletion starves the tail-rank matches first, which
            # biases the pool's size/energy/band mixture toward the
            # high-utility end of the observed selection's own pool.
            walk = [int(pid) for pid in rng.permutation(np.asarray(ordered))]
            # One entry per drawn patch, paired with the patch it was matched
            # to: the observed arm ranks exactly these ids, so the pool and its
            # id set can never desynchronise into a size-mismatched comparison.
            pool: list[tuple[int, np.ndarray]] = []
            for pid in walk[:requested_pool]:
                pool_voxels, pool_band, pool_energy, pool_slices = _patch_targets(
                    atlas, case, rows, pid, target_cache
                )
                trial_candidate = trial_candidates[control_type][trial]
                if control_type == "volume_matched":
                    region = volume_matched_region(
                        target_voxels=pool_voxels,
                        band=pool_band,
                        candidate=trial_candidate,
                        rng=rng,
                        slices=pool_slices,
                    )
                else:
                    region = energy_matched_region(
                        energy=cond.energy,
                        target_energy=pool_energy,
                        target_voxels=pool_voxels,
                        band=pool_band,
                        candidate=trial_candidate,
                        rng=rng,
                        slices=pool_slices,
                    )
                if region is None:
                    continue
                pool.append((pid, region))
                trial_candidates[control_type][trial] = trial_candidate & ~region
            realised_pools.append(len(pool))
            # One gate, not two.  There is no "requested" size to miss any
            # more: the observed arm is matched to the realised M, so a pool
            # short of the request is a narrower comparison, not a broken one
            # (that narrowing is reported as control_drawable_fraction).  What
            # still has to be excluded is M <= q, where the top q of the pool
            # IS the pool and the control arm performs no selection at all --
            # the old biased test wearing a flag, reachable by passing
            # --control-pool-size at or below q.
            if len(pool) <= q:
                skipped_undersized += 1
                continue
            matched_pools.append(len(pool))
            pool_ids = [pid for pid, _ in pool]
            pool_regions = [region for _, region in pool]
            measured = evaluate_interventions(
                model,
                _per_sample_bce,
                case.x_pad,
                cond.adv_pad,
                case.y_pad,
                pad_masks(np.stack(pool_regions), case.padded_shape),
                mode="remove",
                num_classes=num_classes,
                batch=settings.iv_batch,
                crop=case.shape,
            )
            # The literal functional the observed ranking uses (causal_stage's
            # necessity divided by d_full): a composite or sufficiency-based
            # score here would leave the two arms selecting on different
            # quantities and destroy the symmetry this block exists to create.
            recoveries = [
                float((d_full - (clean_primary - primary(dice_row))) / d_full)
                for dice_row in measured.dice
            ]
            # The observed arm for THIS trial: the q highest-necessity patches
            # of the pool's own id set, ranked on the necessity causal_stage
            # already measured -- one extra forward per trial, not a second
            # pool sweep.  necessity and the control's necessity/d_full are
            # rank-equivalent because d_full > 0 is a precondition of this
            # whole block (guarded at the top of interaction_stage).
            observed_ids = select_observed_matched_subset(pool_ids, necessity_by_id, q)
            observed_tubes = np.isin(atlas.tubes, observed_ids)
            observed_trial = evaluate_interventions(
                model,
                _per_sample_bce,
                case.x_pad,
                cond.adv_pad,
                case.y_pad,
                pad_masks(observed_tubes[None], case.padded_shape),
                mode="remove",
                num_classes=num_classes,
                batch=settings.iv_batch,
                crop=case.shape,
            )
            observed_t = float(
                (d_full - (clean_primary - primary(observed_trial.dice[0]))) / d_full
            )
            observed_voxels = int(observed_tubes.sum())
            observed_recoveries.append(observed_t)
            out.append(
                {
                    **base,
                    "control_type": "observed_matched_subset",
                    "patch_id": -1,
                    "trial": trial,
                    "control_voxels": observed_voxels,
                    "recovery_fraction": observed_t,
                    "n_patches_in_joint": len(observed_ids),
                }
            )
            unions.append(
                (
                    trial,
                    select_pool_top_q(recoveries, pool_regions, q),
                    min(q, len(pool)),
                    observed_t,
                    observed_voxels,
                )
            )
        if not unions:
            continue
        result = evaluate_interventions(
            model,
            _per_sample_bce,
            case.x_pad,
            cond.adv_pad,
            case.y_pad,
            pad_masks(np.stack([u for _, u, _, _, _ in unions]), case.padded_shape),
            mode="remove",
            num_classes=num_classes,
            batch=settings.iv_batch,
            crop=case.shape,
        )
        for (
            t,
            union,
            n_selected,
            observed_t,
            observed_voxels,
        ), dice_row in zip(unions, result.dice):
            recovery = float((d_full - (clean_primary - primary(dice_row))) / d_full)
            paired.append((recovery, observed_t))
            out.append(
                {
                    **base,
                    "control_type": f"{control_type}_joint",
                    "patch_id": -1,
                    "trial": t,
                    "target_voxels": observed_voxels,
                    "control_voxels": int(union.sum()),
                    "recovery_fraction": recovery,
                    # The comparison column must describe the same mask as
                    # target_voxels: this row's OWN paired observed statistic
                    # -- the trial-wise pairing control_permutation_p counts,
                    # not the top-10 recovery and not the median over trials
                    # that observed_matched_recovery now reports.
                    "patch_recovery_fraction": observed_t,
                    # The true size of the selected list (len(order[:q]) at
                    # selection time), not a hardcoded q -- the gate above makes
                    # them equal for every row emitted here, but this stays
                    # correct even if that invariant ever changes.
                    "n_patches_in_joint": n_selected,
                }
            )
    realised_median = float(np.median(realised_pools)) if realised_pools else _NAN
    summary["control_pool_size"] = realised_median
    summary["matched_pool_size_median"] = (
        float(np.median(matched_pools)) if matched_pools else _NAN
    )
    # What fraction of the screened set was drawable at all.  The endpoint is
    # conditional on drawability -- both arms are top q of M, not of
    # len(ordered) -- so the narrowing has to be readable off the output.
    # Deliberately the median over ALL trials, not just qualifying ones: an
    # atlas whose every trial is skipped (outer_boundary_gt, whose patches ARE
    # their own distance band) is exactly where this number matters most, and a
    # qualifying-trials-only definition would be NaN there.
    summary["control_drawable_fraction"] = (
        realised_median / len(ordered) if realised_pools else _NAN
    )
    # Median over qualifying trials of the per-trial subsampled observed
    # statistic, NOT a single fixed top-q removal: which q patches the observed
    # arm removes is decided per trial, from that trial's P_succ.
    summary["observed_matched_recovery"] = (
        float(np.median(observed_recoveries)) if observed_recoveries else _NAN
    )
    summary["n_trials_skipped_undersized"] = skipped_undersized
    # Any skipped trial thins the null distribution, and a minority of skips
    # can leave the median realised pool size healthy -- a median-based
    # trigger would hide that. Fire on any skip, not the median.
    summary["control_pool_undersized"] = bool(skipped_undersized > 0)
    summary["n_controls_volume"] = len(per_patch["volume_matched"])
    summary["n_controls_energy"] = len(per_patch["energy_matched"])
    summary["control_volume_recovery_median"] = (
        float(np.median(per_patch["volume_matched"]))
        if per_patch["volume_matched"]
        else _NAN
    )
    summary["control_energy_recovery_median"] = (
        float(np.median(per_patch["energy_matched"]))
        if per_patch["energy_matched"]
        else _NAN
    )
    # Paired trial-wise -- both arms share M within a trial -- and pooled
    # across the two control types.  10 trials x 2 control types keeps the
    # floor at 0.048.
    summary["control_permutation_p"] = (
        float(
            (sum(1 for control_t, obs_t in paired if control_t >= obs_t) + 1)
            / (len(paired) + 1)
        )
        if paired
        else _NAN
    )
    return out


def noise_control_rows(
    model: torch.nn.Module,
    case: CaseContext,
    cond: Condition,
    ordered_by_atlas: dict[str, list[int]],
    num_classes: int,
    settings: Settings,
    normalization: pd.DataFrame | None,
) -> tuple[list[dict[str, object]], list[dict[str, object]]]:
    """Gaussian / Rician δ matched to this condition's RMS: energy rows + top-10% removal."""

    patch_rows: list[dict[str, object]] = []
    summaries: list[dict[str, object]] = []
    clean = _unpad(case.x_pad[0, 0], case.shape).astype(np.float32)
    epsilon = cond.epsilon_n / 255.0
    for control_key, label in CONTROL_LABELS.items():
        rng = np.random.default_rng(
            stable_seed(
                case.dataset_key,
                case.case_id,
                control_key,
                cond.epsilon_n,
                0,
                base_seed=settings.base_seed,
            )
        )
        if control_key == "gaussian":
            altered = matched_gaussian_noise(
                clean,
                valid_mask=case.valid,
                epsilon=epsilon,
                target_rms=cond.rms,
                rng=rng,
            )
        else:
            if normalization is None:
                continue
            try:
                row = normalization.loc[(case.dataset_key, case.case_id)]
            except KeyError:
                cond.flags.append("no_normalization_row")
                continue
            raw_mean, raw_std = (
                float(row["raw_norm_mean_au"]),
                float(row["raw_norm_std_au"]),
            )
            altered = matched_rician_proxy_noise(
                clean,
                valid_mask=case.valid,
                epsilon=epsilon,
                target_rms=cond.rms,
                rng=rng,
                raw_zero_normalized=-raw_mean / raw_std,
            )
        delta = (altered - clean).astype(np.float64)
        noise_pad = case.x_pad.clone()
        noise_pad[0, 0, : case.shape[0], : case.shape[1], : case.shape[2]] = (
            torch.from_numpy(altered).to(case.x_pad.device)
        )
        with torch.no_grad():
            logits = model(noise_pad)
            loss = float(_per_sample_bce(logits, case.y_pad, num_classes)[0].item())
            pred = _unpad(logits.argmax(dim=1)[0], case.shape).astype(np.int16)
        metrics = _metrics(pred, case.seg, case.spacing, case.dataset_key, loss)
        damages, undefined = condition_damages(case.clean_metrics, metrics)
        noise = Condition(
            control_key,
            label,
            cond.epsilon_n,
            -1,
            noise_pad,
            delta,
            energy_map(delta),
            pred,
            metrics,
            damages,
            undefined,
            float(np.abs(delta[case.valid]).max()),
            float(np.sqrt(np.mean(delta[case.valid] ** 2))),
            float(np.mean(np.abs(delta[case.valid]) >= epsilon * (1 - 1e-6))),
            _NAN,
            stage="interaction",
        )
        for name in case.atlases:
            rows, summary = energy_stage(case, noise, name)
            top = ordered_by_atlas.get(name, [])[
                : max(1, int(math.ceil(0.10 * len(ordered_by_atlas.get(name, [1])))))
            ][:10]
            # attack_success, not `> 0`: matched noise drops Dice by ~2e-4, and a
            # recovery fraction over that denominator is noise, not a measurement.
            if top and attack_success(-damages.get("dice", _NAN)):
                joint = pad_masks(
                    np.isin(case.atlases[name].tubes, top)[None], case.padded_shape
                )
                res = evaluate_interventions(
                    model,
                    _per_sample_bce,
                    case.x_pad,
                    noise_pad,
                    case.y_pad,
                    joint,
                    mode="remove",
                    num_classes=num_classes,
                    batch=1,
                    crop=case.shape,
                )
                summary["observed_top10_recovery"] = float(
                    (
                        damages["dice"]
                        - (
                            float(case.clean_metrics["dice"])
                            - primary_fn(case.dataset_key)(res.dice[0])
                        )
                    )
                    / damages["dice"]
                )
            patch_rows.extend(rows.values())
            summaries.append(summary)
    return patch_rows, summaries


# ---------------------------------------------------------------------------
# Orchestration
# ---------------------------------------------------------------------------
def process_condition(
    model: torch.nn.Module,
    case: CaseContext,
    cond: Condition,
    num_classes: int,
    settings: Settings,
    output_dir: Path,
    normalization: pd.DataFrame | None,
) -> None:
    causal = "causal" in settings.stages and case.in_causal
    interaction = (
        "interaction" in settings.stages
        and case.in_interaction
        and cond.epsilon_n in INTERACTION_EPS
    )
    cond.stage = "interaction" if interaction else ("causal" if causal else "energy")
    if causal:
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
        if ig.completeness_error > 0.05:
            ig = integrated_attribution(
                model,
                _per_sample_bce,
                case.x_pad,
                cond.adv_pad,
                case.y_pad,
                num_classes=num_classes,
                steps=2 * settings.ig_steps,
                batch=settings.ig_batch,
            )
            if ig.completeness_error > 0.05:
                cond.flags.append("ig_completeness_exceeds_5pct")
        cond.ig_attribution = ig.attribution[
            : case.shape[0], : case.shape[1], : case.shape[2]
        ]
        cond.ig_steps = ig.steps
        cond.ig_completeness = ig.completeness_error
        # QC: a null removal must reproduce adv Dice, a full removal clean Dice.
        # batch=1, not 2: cuDNN picks kernels per batch shape, so a batch-2
        # forward of the same input differs from the batch-1 reference forward
        # by ~1e-5 Dice — enough to trip the 1e-9 gate on a comparison that is
        # supposed to be exact. Matching the reference batch size is what makes
        # this check mean anything.
        probe = np.stack(
            [np.zeros(case.padded_shape, bool), np.ones(case.padded_shape, bool)]
        )
        check = evaluate_interventions(
            model,
            _per_sample_bce,
            case.x_pad,
            cond.adv_pad,
            case.y_pad,
            probe,
            mode="remove",
            num_classes=num_classes,
            batch=1,
            crop=case.shape,
        )
        primary = primary_fn(case.dataset_key)
        if (
            abs(primary(check.dice[0]) - float(cond.adv_metrics["dice"])) > 1e-9
            or abs(primary(check.dice[1]) - float(case.clean_metrics["dice"])) > 1e-9
        ):
            raise RuntimeError(
                f"intervention Dice does not reproduce clean/adversarial Dice for {case.case_id}"
            )
    rng = np.random.default_rng(
        stable_seed(
            case.dataset_key,
            case.case_id,
            cond.attack_key,
            cond.epsilon_n,
            "eseua",
            base_seed=settings.base_seed,
        )
    )
    patch_rows: list[dict[str, object]] = []
    summaries: list[dict[str, object]] = []
    curve_rows: list[dict[str, object]] = []
    control_rows: list[dict[str, object]] = []
    qc_rows: list[dict[str, object]] = []
    maps: dict[str, np.ndarray] = {"energy": cond.energy.astype(np.float32)}
    if cond.ig_attribution is not None:
        maps["integrated_attribution"] = cond.ig_attribution.astype(np.float32)
    ordered_by_atlas: dict[str, list[int]] = {}
    for name, atlas in case.atlases.items():
        flags = (
            list(cond.flags)
            + atlas_flags(atlas, case.roi)
            + [f"undefined_{m}" for m in cond.undefined]
        )
        if math.isfinite(cond.rms_error) and cond.rms_error > RMS_DRIFT_FLAG:
            flags.append("rms_drift_above_flag")
        rows, summary = energy_stage(case, cond, name)
        if float(summary["conservation_error"]) > 1e-6:
            raise RuntimeError(
                f"energy conservation violated for {case.case_id}/{name}"
            )
        if causal and rows:
            curves, patch_maps, ordered = causal_stage(
                model, case, cond, name, rows, summary, num_classes, settings, rng
            )
            curve_rows.extend(curves)
            maps.update(patch_maps)
            ordered_by_atlas[name] = ordered
            if interaction:
                control_rows.extend(
                    interaction_stage(
                        model,
                        case,
                        cond,
                        name,
                        rows,
                        summary,
                        ordered,
                        num_classes,
                        settings,
                        rng,
                    )
                )
                if "n_controls_volume" not in summary:
                    # interaction_stage returned before drawing anything (no
                    # causal patches ordered, or d_full <= 0) -- distinct from
                    # having run and found the atlas has no candidate sampling
                    # space, which is what no_matched_controls below means.
                    flags.append("interaction_stage_skipped_no_causal_patches")
                elif (
                    summary.get("n_controls_volume", 0) == 0
                    and summary.get("n_controls_energy", 0) == 0
                ):
                    flags.append("no_matched_controls")
                if summary.get("control_pool_undersized"):
                    # Independent of the branch above: per-patch controls can
                    # draw fine while the much larger joint pool cannot, and a
                    # capped --control-pool-size leaves a smaller version of the
                    # selection bias that must stay visible.
                    flags.append("control_pool_undersized")
        if not (cond.damages.get("dice", _NAN) > 0):
            flags.append("non_damaging")
        qc_rows.append(
            {
                **_identity(case, cond, name),
                "n_flags": len(flags),
                "flags": ";".join(flags),
            }
        )
        patch_rows.extend(rows.values())
        summaries.append(summary)
    # Noise controls are written once per case/epsilon, on the APGD condition only.
    # Their identity key carries the control label rather than the source attack, so
    # emitting them per attack would collide: three attack shards would each write the
    # same key with a different perturbation. Matching to APGD's realized RMS is also
    # the convention evaluate_perturbation_structure_all_cases.py already uses.
    if interaction and cond.attack_key == "apgd":
        noise_patches, noise_summaries = noise_control_rows(
            model, case, cond, ordered_by_atlas, num_classes, settings, normalization
        )
        patch_rows.extend(noise_patches)
        summaries.extend(noise_summaries)
    if settings.save_maps and (causal or interaction):
        _save_npz(
            map_path(
                output_dir,
                case.dataset_key,
                case.case_id,
                cond.attack_label,
                cond.epsilon_n,
            ),
            **maps,
        )
    _append(output_dir / PATCH_CSV, patch_rows, PATCH_FIELDNAMES)
    _append(output_dir / CURVES_CSV, curve_rows, CURVES_FIELDNAMES)
    _append(output_dir / CONTROL_CSV, control_rows, CONTROL_FIELDNAMES)
    _append(output_dir / QC_CSV, qc_rows, QC_FIELDNAMES)
    _append(
        output_dir / SUMMARY_CSV, summaries, SUMMARY_FIELDNAMES
    )  # last: the resume marker


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
    output_dir: Path,
    completed: set[tuple[str, str, int, str]],
    device: torch.device,
    config: AtlasConfig,
    settings: Settings,
    quality: pd.DataFrame | None,
    normalization: pd.DataFrame | None,
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
        try:
            case = build_case(
                model,
                device,
                div_factors,
                num_classes,
                dataset_key,
                case_id,
                paths,
                output_dir,
                config,
                in_causal=case_id in causal_set,
                in_interaction=case_id in interaction_set,
            )
        except ValueError as exc:
            # An empty GT makes roi_mask raise (gradients.py:29); an
            # all-foreground GT does NOT raise (its boundary core is simply
            # empty via border_value=1 erosion, and the atlas is built with
            # zero patches). Any other ValueError inside build_case also
            # lands here -- hence the generic name and the folded-in message,
            # rather than a name that asserts one specific cause. Without this
            # guard the exception propagates, the case is never completed,
            # and a restart lands right back on it and fails again -- the
            # same permanent-stall mode already fixed twice in this driver.
            # Flag it, with the actual cause, and move on instead.
            detail = str(exc).replace(";", ",")[:QC_DETAIL_MAX_CHARS]
            flag = f"case_build_failed:{detail}"
            _append(
                output_dir / QC_CSV,
                [
                    {
                        "dataset": dataset_key,
                        "case_id": case_id,
                        "epsilon_n": -1,
                        "attack": "none",
                        "atlas_type": "none",
                        # No Condition/stage exists yet at this point in the
                        # pipeline; "energy" is the earliest real stage (and
                        # Condition.stage's own default), not a CLI-selectable
                        # STAGES member of its own.
                        "stage": "energy",
                        "n_flags": 1,
                        "flags": flag,
                    }
                ],
                QC_FIELDNAMES,
            )
            print(
                f"[{dataset_key}] case {index}/{len(case_ids)} ({case_id}): "
                f"{exc}; flagged {flag}",
                flush=True,
            )
            continue
        for attack_key in attacks:
            label = ATTACK_LABELS[attack_key]
            for epsilon_n in epsilon_n_values:
                if (dataset_key, case_id, epsilon_n, label) in completed:
                    continue
                adv_pad, seed = run_attack(
                    model,
                    case,
                    attack_key,
                    epsilon_n,
                    num_classes,
                    attack_steps=attack_steps,
                    base_seed=settings.base_seed,
                    valid_only=settings.attack_valid_only,
                )
                cond = build_condition(
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
                process_condition(
                    model, case, cond, num_classes, settings, output_dir, normalization
                )
                completed.add((dataset_key, case_id, epsilon_n, label))
                del adv_pad, cond
        elapsed = time.monotonic() - start_time
        rate = len(completed) / max(elapsed, 1e-9)
        eta = max(total_keys - len(completed), 0) / max(rate, 1e-9) / 60.0
        print(
            f"[{dataset_key}] case {index}/{len(case_ids)} ({case_id}); keys {len(completed)}/{total_keys}; ETA {eta:.1f} min",
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
    parser = argparse.ArgumentParser(
        description="Edge-specific energy–utility attribution for every FGSM/PGD/APGD case"
    )
    parser.add_argument(
        "--datasets", nargs="+", choices=["wg", "zones"], default=["wg", "zones"]
    )
    parser.add_argument("--eps-n", nargs="+", type=int, default=DEFAULT_EPS_N)
    parser.add_argument(
        "--attacks", nargs="+", choices=list(ATTACK_LABELS), default=list(ATTACK_LABELS)
    )
    parser.add_argument(
        "--stages", nargs="+", choices=list(STAGES), default=list(STAGES)
    )
    parser.add_argument("--attack-steps", type=int, default=20)
    parser.add_argument("--max-cases", type=int, default=None)
    parser.add_argument("--case-subset", choices=list(CASE_SUBSETS), default="all")
    parser.add_argument(
        "--causal-cases",
        type=int,
        default=100,
        help="evenly spaced subcohort per dataset for the causal stage",
    )
    parser.add_argument(
        "--interaction-cases",
        type=int,
        default=30,
        help="evenly spaced subcohort of the causal set for the interaction stage",
    )
    parser.add_argument("--seed", type=int, default=20260721)
    parser.add_argument("--device", type=str, default=None)
    parser.add_argument(
        "--deterministic", action=argparse.BooleanOptionalAction, default=True
    )
    parser.add_argument("--edge-quantile", type=float, default=0.90)
    parser.add_argument("--patch-extent-mm", type=float, default=6.0)
    parser.add_argument("--tube-radius-mm", type=float, default=2.0)
    parser.add_argument("--min-core-voxels", type=int, default=10)
    parser.add_argument("--roi-margin-mm", type=float, default=20.0)
    parser.add_argument("--ig-steps", type=int, default=32)
    parser.add_argument("--ig-batch", type=int, default=8)
    parser.add_argument("--iv-batch", type=int, default=8)
    parser.add_argument("--boundary-workers", type=int, default=4)
    parser.add_argument("--greedy-limit", type=int, default=30)
    parser.add_argument("--pairwise-limit", type=int, default=20)
    parser.add_argument("--controls-per-patch", type=int, default=10)
    parser.add_argument("--control-pool-size", type=int, default=0)
    parser.add_argument("--local-radius-mm", type=float, default=4.0)
    parser.add_argument(
        "--save-maps", action=argparse.BooleanOptionalAction, default=True
    )
    parser.add_argument(
        "--quality-csv",
        type=Path,
        default=DEFAULT_QUALITY_CSV,
        help="published quality table for the RMS reproduction check; skipped if absent",
    )
    parser.add_argument(
        "--normalization-csv",
        type=Path,
        default=DEFAULT_NORMALIZATION_CSV,
        help="per-case normalization audit for the Rician control; Rician skipped if absent",
    )
    parser.add_argument(
        "--trainer-dir",
        type=str,
        default=TRAINERS["unet"],
        help="nnU-Net results folder (<Trainer>__<Plans>__<config>) whose fold_all "
        "checkpoint is attacked, e.g. a defended fine-tune. Use a fresh --output-dir.",
    )
    parser.add_argument(
        "--attack-valid-only",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="confine the perturbation to non-ignore voxels (excludes the zero-filled "
        "margin outside the dilated gland on zones, and the model-grid padding). "
        "The published campaign ran without it; use a fresh --output-dir.",
    )
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    for name in (
        "attack_steps",
        "ig_steps",
        "ig_batch",
        "iv_batch",
        "boundary_workers",
        "causal_cases",
        "interaction_cases",
    ):
        if getattr(args, name) < 1:
            raise ValueError(f"--{name.replace('_', '-')} must be positive")
    epsilon_n_values = sorted({int(v) for v in args.eps_n})
    if not epsilon_n_values or any(v <= 0 for v in epsilon_n_values):
        raise ValueError("--eps-n values must be positive")
    attacks = list(dict.fromkeys(str(v) for v in args.attacks))
    stages = tuple(s for s in STAGES if s in set(args.stages))
    if "interaction" in stages and "causal" not in stages:
        raise ValueError(
            "--stages interaction requires causal (the interaction stage ranks the causal patches)"
        )
    _configure_deterministic_execution(enabled=args.deterministic, seed=args.seed)
    output_dir = args.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    device = (
        torch.device(args.device)
        if args.device
        else torch.device("cuda" if torch.cuda.is_available() else "cpu")
    )
    config = AtlasConfig(
        edge_quantile=args.edge_quantile,
        min_core_voxels=args.min_core_voxels,
        patch_extent_mm=args.patch_extent_mm,
        tube_radius_mm=args.tube_radius_mm,
        roi_margin_mm=args.roi_margin_mm,
    )
    settings = Settings(
        stages,
        args.ig_steps,
        args.ig_batch,
        args.iv_batch,
        args.boundary_workers,
        args.greedy_limit,
        args.pairwise_limit,
        args.controls_per_patch,
        args.control_pool_size,
        args.local_radius_mm,
        bool(args.save_maps),
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
    normalization = (
        _load_normalization_lookup(args.normalization_csv)
        if args.normalization_csv.is_file()
        else None
    )
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
        "seed": int(args.seed),
        "device": str(device),
        "atlas_config": config.__dict__,
        "ig_steps": int(args.ig_steps),
        "ig_batch": int(args.ig_batch),
        "iv_batch": int(args.iv_batch),
        "boundary_workers": int(args.boundary_workers),
        "greedy_limit": int(args.greedy_limit),
        "pairwise_limit": int(args.pairwise_limit),
        "controls_per_patch": int(args.controls_per_patch),
        "control_pool_size": int(args.control_pool_size),
        "local_radius_mm": float(args.local_radius_mm),
        "interaction_eps": list(INTERACTION_EPS),
        "tables": [SUMMARY_CSV, PATCH_CSV, CURVES_CSV, CONTROL_CSV, QC_CSV],
        "trainer_dir": str(args.trainer_dir),
        "attack_valid_only": bool(args.attack_valid_only),
        "checkpoints": {
            d: str(_dataset_paths(d, args.trainer_dir)["checkpoint"].resolve())
            for d in args.datasets
        },
    }
    if manifest_path.is_file():
        previous = json.loads(manifest_path.read_text())
        # manifests written before the two flags existed were clean-model, unconfined runs
        previous.setdefault("trainer_dir", TRAINERS["unet"])
        previous.setdefault("attack_valid_only", False)
        for key in (
            "seed",
            "trainer_dir",
            "attack_valid_only",
            "attack_steps",
            "ig_steps",
            "atlas_config",
            "causal_cases",
            "interaction_cases",
            "stages",
            "iv_batch",
            "ig_batch",
            "greedy_limit",
            "pairwise_limit",
            "controls_per_patch",
            "control_pool_size",
            "local_radius_mm",
        ):
            if previous.get(key) != json.loads(
                json.dumps(manifest[key])
            ):  # tuples round-trip as lists
                raise ValueError(
                    f"resume parameter mismatch for {key!r}: {previous.get(key)} != {manifest[key]}"
                )
    manifest_path.write_text(json.dumps(manifest, indent=2, sort_keys=True))
    completed = completed_keys(output_dir / SUMMARY_CSV)
    total_keys = sum(
        len(
            case_ids_for_subset(
                _available_case_ids(_dataset_paths(d)["data"]),
                args.case_subset,
                int(args.causal_cases),
                int(args.interaction_cases),
            )[: args.max_cases]
        )
        * len(attacks)
        * len(epsilon_n_values)
        for d in args.datasets
    )
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
            output_dir=output_dir,
            completed=completed,
            trainer_dir=str(args.trainer_dir),
            device=device,
            config=config,
            settings=settings,
            quality=quality,
            normalization=normalization,
            start_time=start,
            total_keys=total_keys,
        )
    print(f"Complete: {len(completed)}/{total_keys} keys", flush=True)


if __name__ == "__main__":
    main()
