"""Characterize the spatial organization of adversarial and random perturbations.

The full-cohort quality pass recorded PSNR and SSIM but not the perturbations
themselves, so descriptors that need the actual field require regenerating it.
Attacks are regenerated on GPU with the settings recorded in
``adversarial_quality_all_cases.csv`` (fp32, 20 steps, batch size 2, seed
20260721); random controls are regenerated from deterministic seeds.

Because APGD is regenerated anyway, each recomputed RMS is compared against the
published value, so this pass doubles as a reproduction check of the existing
image-quality results.
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


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))
SRC_ROOT = REPO_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

DEFAULT_RESULTS = (
    REPO_ROOT / "results" / "adv_rob_eval_results" / "image_quality_analysis"
)
DEFAULT_QUALITY_CSV = (
    DEFAULT_RESULTS
    / "all_cases_adversarial_quality"
    / "adversarial_quality_all_cases.csv"
)
DEFAULT_OUTPUT_DIR = DEFAULT_RESULTS / "all_cases_perturbation_structure"
DEFAULT_NORMALIZATION_CSV = DEFAULT_RESULTS / "normalization_per_case.csv"
DEFAULT_EPS_N = [2, 4, 8, 16, 32]

CONDITIONS = {
    "fgsm": "FGSM-BCE",
    "pgd": "PGD-BCE",
    "apgd": "APGD-BCE",
    "gaussian": "gaussian_rms_matched",
    "rician": "rician_proxy_rms_matched",
}
ATTACK_KEYS = ("fgsm", "pgd", "apgd")
CONTROL_KEYS = ("gaussian", "rician")
# The published random-noise control run
# (all_cases_random_noise_quality/random_noise_quality_all_cases.csv) used
# match_iterations=12 for the RMS-matching bisection in
# epsilon_calibration._match_parameter_by_rms. That function returns whichever
# candidate the bisection lands on at its last iteration, so a different depth
# returns a materially different field -- the library default of 32 would
# silently regenerate a control that does not match the published one, with no
# published_rms to catch the drift (controls never run check_reproduction).
PUBLISHED_MATCH_ITERATIONS = 12
# Controls carry no published RMS, so ``check_reproduction`` never runs on them.
# This mirrors the gate in evaluate_random_noise_quality_all_cases.py:288-293.
CONTROL_RMS_TOLERANCE = 2e-3
# Attack RMS drift is measured and reported, not fatal.  PGD and APGD take
# sign(gradient) each step, so a GPU reduction-order difference near a zero
# crossing flips that voxel by a full 2*alpha and compounds over 20 iterations:
# the smoke run measured 3.1e-3 median and 5.3e-3 max APGD drift from four rows
# alone, and the production pass regenerates ~14k attack rows, so a tail beyond
# 1e-2 is expected rather than a defect.  Aborting there was also unrecoverable:
# a failed row never enters ``completed``, so every resume regenerated the same
# case and died in the same place, with no way to skip it.  Only drift large
# enough to mean the run itself is misconfigured -- wrong seed, wrong step
# count, wrong precision, which lands orders of magnitude away -- still raises.
REPRO_DRIFT_TOLERANCE = 3e-3
# One threshold cannot serve all three attacks, because they do not reproduce on
# the same scale.  FGSM is a single deterministic step with no random start, so
# it is the sharp fingerprint of the configuration -- wrong checkpoint, epsilon,
# loss, or precision moves it at once -- and it reproduces to ~1e-5 in practice.
# PGD and APGD take sign(gradient) over 20 steps from a random start, so a GPU
# reduction-order difference near a zero crossing flips a voxel by a full
# 2*alpha and compounds; measured over the first 53 cases of the production run
# (batch size 1, against a published batch-size-2 run) APGD drift is median
# 1.5e-3, q99 4.1e-2, max 4.9e-2, and a tail fit projects ~6e-1 over the full
# 9,995 APGD rows.  A shared 1e-1 ceiling would therefore abort a healthy run --
# and unrecoverably, since a failed row never enters ``completed`` and every
# resume retries the same case.  So the iterative attacks are gated only against
# a wholly different perturbation, and drift is carried in
# ``rms_reproduction_rel_error`` plus the per-condition tally instead.
REPRO_FAIL_TOLERANCE = {"FGSM-BCE": 1e-3, "PGD-BCE": 1.0, "APGD-BCE": 1.0}
DEFAULT_REPRO_FAIL_TOLERANCE = 1.0
# Settings added after the first tables were written: a table without the column
# was produced under this value.
LEGACY_SETTING_DEFAULTS = {"attack_valid_only": 0}


def completed_keys(
    path: Path, *, settings: dict[str, int]
) -> set[tuple[str, str, int, str]]:
    """Resume support: which (dataset, case, epsilon, condition) rows exist.

    Refuses to resume a file written under different descriptor settings, since
    mixing them would silently produce a CSV whose rows are not comparable.
    """

    path = Path(path)
    if not path.is_file() or path.stat().st_size == 0:
        return set()
    frame = pd.read_csv(path)
    # A header-only file carries no rows to disagree about, so it must resume as
    # if empty.  Running the settings check on it reported the settings as
    # ``[]`` and refused to resume, which is both wrong and unrecoverable
    # without deleting the file by hand.
    if frame.empty:
        return set()
    for column, expected in settings.items():
        if column not in frame.columns:
            if column in LEGACY_SETTING_DEFAULTS:
                observed = {LEGACY_SETTING_DEFAULTS[column]}
            else:
                raise ValueError(
                    f"existing output lacks reproducibility field {column!r}"
                )
        else:
            observed = set(frame[column].astype(int).unique())
        if observed != {int(expected)}:
            raise ValueError(
                f"existing output uses {column}={sorted(observed)}, requested {expected}; "
                "choose a different output directory"
            )
    return {
        (str(row.dataset), str(row.case_id), int(row.epsilon_n), str(row.condition))
        for row in frame.itertuples(index=False)
    }


def spectra_case_subset(case_ids: list[str], *, count: int) -> list[str]:
    """Deterministic evenly spaced case selection over sorted ids.

    Same ``np.linspace`` idiom that ``axial_ssim_mean`` uses for slices, so the
    subset is reproducible without carrying an RNG.
    """

    ordered = sorted(case_ids)
    if count <= 0:
        return []
    if len(ordered) <= count:
        return ordered
    take = np.linspace(0, len(ordered) - 1, count).round().astype(int)
    return [ordered[int(i)] for i in dict.fromkeys(take)]


def reproduction_error(observed: float, expected: float) -> float:
    """Relative difference between a recomputed and a published RMS."""

    return abs(float(observed) - float(expected)) / max(abs(float(expected)), 1e-12)


def check_reproduction(
    observed: float,
    expected: float,
    *,
    label: str,
    fail: float = DEFAULT_REPRO_FAIL_TOLERANCE,
) -> float:
    """Compare a recomputed RMS with the published one.

    The tolerance is deliberately loose.  ``_plan_batches`` groups cases by
    identical padded shape so batching contributes no drift, but PGD and APGD
    take ``sign(gradient)`` each step: a tiny difference in GPU reduction order
    near a zero crossing flips that voxel's step by a full 2*alpha and compounds
    over 20 iterations.  FGSM, being one step, should reproduce far more tightly.

    Ordinary drift is returned, not warned about: it is recorded per row in
    ``rms_reproduction_rel_error`` and tallied per dataset by
    ``evaluate_dataset``.  Warning per row instead produced one ``UserWarning``
    per attack row -- each with a distinct message, so Python's dedup never
    collapsed them -- which buried every other warning the run emits.
    """

    error = reproduction_error(observed, expected)
    # A boundary value (e.g. observed exactly `fail` relative error away from
    # expected) is not guaranteed to land on either side of the threshold once
    # ``reproduction_error`` has divided two floats: 0.0505 vs 0.05 computes to
    # 0.010000000000000009, not 0.01. ``math.isclose`` absorbs that noise so a
    # genuinely boundary case is never misclassified purely by rounding.
    if error > fail and not math.isclose(error, fail, rel_tol=1e-9):
        raise RuntimeError(
            f"RMS reproduction failed for {label}: relative error {error:.3e} "
            f"(observed {observed:.8f}, published {expected:.8f}); this is far "
            "beyond GPU nondeterminism and means the run is misconfigured"
        )
    return error


def format_drift_summary(dataset_key: str, drift: dict[str, list[float]]) -> list[str]:
    """One reported line per condition, replacing the per-row warnings."""

    lines: list[str] = []
    for condition_label, errors in sorted(drift.items()):
        if not errors:
            continue
        values = np.asarray(errors, dtype=float)
        over = int((values > REPRO_DRIFT_TOLERANCE).sum())
        lines.append(
            f"[{dataset_key}] RMS reproduction {condition_label}: "
            f"n={values.size} median={np.median(values):.3e} "
            f"max={values.max():.3e}; {over} rows above "
            f"{REPRO_DRIFT_TOLERANCE:.0e}"
        )
    return lines


def check_control_rms_match(observed: float, target: float, *, label: str) -> float:
    """Verify a regenerated control actually hits the RMS it is matched to.

    Controls pass ``published_rms=None``, so ``check_reproduction`` never runs on
    them and nothing else in this script would notice a degraded match.  The
    entire paired comparison rests on the control carrying the same magnitude as
    the attack it is paired against: if the match silently broke, every paired
    difference would compare different magnitudes and the output would contain no
    trace of it.  The tolerance and message mirror the published random-noise
    gate so the two runs agree on what "matched" means.
    """

    error = reproduction_error(observed, target)
    if error > CONTROL_RMS_TOLERANCE:
        raise RuntimeError(
            f"RMS match failed for {label}: relative error {error:.6f} "
            f"(observed {observed:.8f}, target {target:.8f})"
        )
    return error


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
from experiments.evaluate_adversarial_quality_all_cases import (  # noqa: E402
    apgd_bce_independent_batch,
    fgsm_bce_independent_batch,
    pgd_bce_independent_batch,
)
from mri_prostate_seg.experiments.epsilon_calibration import (  # noqa: E402
    matched_gaussian_noise,
    matched_rician_proxy_noise,
    stable_seed,
)
from mri_prostate_seg.experiments.perturbation_structure import (  # noqa: E402
    BAND_NAMES,
    INTERIOR_BAND_NAMES,
    CaseGeometry,
    case_geometry,
    perturbation_structure,
)

import torch  # noqa: E402


FIELDNAMES = [
    "dataset",
    "case_id",
    "epsilon_n",
    "epsilon_norm",
    "condition",
    "seed",
    "rms_norm",
    "linf_norm",
    "rms_published_norm",
    "rms_reproduction_rel_error",
    "spectral_centroid_norm",
    "hf_mean_power_fraction",
    "spectral_slope",
    "spectral_slices_used",
    "energy_fraction_foreground",
    "volume_fraction_foreground",
    "foreground_energy_enrichment",
    *[
        f"{p}_band_{n}"
        for n in (*BAND_NAMES, *INTERIOR_BAND_NAMES)
        for p in ("energy_fraction", "volume_fraction", "enrichment")
    ],
    "spearman_absdelta_gradmag",
    "edge_energy_enrichment",
    "alignment_voxels_used",
    "gmsd_axial_prostate",
    # Every parameter that changes the numbers is recorded, so the CSV describes
    # the run that produced it and ``completed_keys`` can refuse to resume across
    # a mismatch.  ``attack_steps`` matters most: resuming with a different value
    # would mix 20-step and n-step rows *and* invalidate the reproduction audit.
    # ``max_gmsd_slices`` otherwise leaves no trace at all, which would make
    # ``gmsd_axial_prostate`` unreconstructable from the artifact.
    # ``spectra_cases`` changes no scalar, but it selects which cases reach
    # ``radial_spectra_subsample.csv``; without it recorded, resuming under a
    # different value silently produced a spectra table over an inconsistent
    # case set, because already-complete cases are never revisited.
    "base_seed",
    "attack_steps",
    "spectral_bins",
    "spectral_margin",
    "spectral_min_size",
    "alignment_voxels",
    "max_gmsd_slices",
    "spectra_cases",
    # 1 when the attack was confined to non-ignore voxels (zones: the zero-filled
    # margin outside the dilated gland, plus the model-grid padding), 0 for the
    # original unconfined runs.  Tables written before the flag existed lack the
    # column; ``completed_keys`` reads that as 0.
    "attack_valid_only",
]
SPECTRA_FIELDNAMES = [
    "dataset",
    "case_id",
    "epsilon_n",
    "condition",
    "freq_bin",
    "f_r",
    "power",
]


def _dataset_paths(dataset_key: str) -> dict[str, Path]:
    config = DATASETS[dataset_key]
    dataset_name = config["dataset_name"]
    root = Path(NNUNET_PATHS)
    return {
        "data": root / "nnUNet_preprocessed" / dataset_name / config["data_subdir"],
        "plans": root / "nnUNet_preprocessed" / dataset_name / "nnUNetPlans.json",
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


def _append_rows(
    path: Path, rows: list[dict[str, object]], fieldnames: list[str]
) -> None:
    if not rows:
        return
    needs_header = not path.is_file() or path.stat().st_size == 0
    with path.open("a", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        if needs_header:
            writer.writeheader()
        writer.writerows(rows)
        handle.flush()


def _load_quality_lookup(path: Path) -> pd.DataFrame:
    """Load the full-cohort quality CSV, indexed for O(1) published-RMS lookup."""

    if not path.is_file():
        raise FileNotFoundError(f"missing full-cohort adversarial quality CSV: {path}")
    frame = pd.read_csv(path)
    return frame.set_index(
        ["dataset", "case_id", "epsilon_n", "attack"], verify_integrity=True
    )


def _load_normalization_lookup(path: Path) -> pd.DataFrame:
    """Load the per-case normalization audit needed for the Rician zero proxy."""

    if not path.is_file():
        raise FileNotFoundError(
            "the all-case normalization audit is required for the Rician "
            f"zero-signal proxy: {path}"
        )
    frame = pd.read_csv(path)
    return frame.set_index(["dataset", "case_id"], verify_integrity=True)


def _published_rms(
    quality: pd.DataFrame,
    dataset_key: str,
    case_id: str,
    epsilon_n: int,
    condition_key: str,
) -> float | None:
    """Published RMS for an attack row, or ``None`` for a control condition."""

    if condition_key not in ATTACK_KEYS:
        return None
    label = CONDITIONS[condition_key]
    try:
        return float(quality.loc[(dataset_key, case_id, epsilon_n, label), "rms_norm"])
    except KeyError as exc:
        raise ValueError(
            f"no published quality row for {dataset_key}/{case_id}/{epsilon_n}/{label}"
        ) from exc


def build_row(
    dataset_key: str,
    case_id: str,
    epsilon_n: int,
    condition_key: str,
    clean: np.ndarray,
    altered: np.ndarray,
    segmentation: np.ndarray,
    *,
    valid: np.ndarray,
    spacing: tuple[float, float, float],
    published_rms: float | None,
    base_seed: int,
    settings: dict[str, int],
    options: dict[str, int],
    geometry: CaseGeometry | None = None,
) -> tuple[dict[str, object], np.ndarray, np.ndarray]:
    """Build one scalar row and radial spectrum, enforcing run invariants."""

    epsilon_norm = epsilon_n / 255.0
    delta = np.asarray(altered, dtype=np.float64) - np.asarray(clean, dtype=np.float64)
    ok = np.asarray(valid, dtype=bool)
    if ok.shape != delta.shape:
        raise ValueError(
            f"valid mask shape {ok.shape} does not match perturbation shape {delta.shape}"
        )
    if not ok.any():
        raise ValueError("valid mask contains no voxels")
    linf = float(np.max(np.abs(delta[ok])))
    if linf > epsilon_norm + 2e-6:
        raise RuntimeError(
            f"L-inf violation for {dataset_key}/{case_id}/{epsilon_n}/"
            f"{CONDITIONS[condition_key]}: {linf} > {epsilon_norm}"
        )

    scalars, centers, power = perturbation_structure(
        clean,
        altered,
        segmentation,
        valid_mask=ok,
        spacing=spacing,
        seed=stable_seed(
            dataset_key,
            case_id,
            "structure_subsample",
            epsilon_n,
            base_seed=base_seed,
        ),
        margin=options["spectral_margin"],
        bins=options["spectral_bins"],
        min_size=options["spectral_min_size"],
        max_voxels=options["alignment_voxels"],
        max_gmsd_slices=options["max_gmsd_slices"],
        geometry=geometry,
    )

    label = f"{dataset_key}/{case_id}/{epsilon_n}/{CONDITIONS[condition_key]}"
    repro_error = (
        float("nan")
        if published_rms is None
        else check_reproduction(
            scalars["rms_norm"],
            published_rms,
            label=label,
            fail=REPRO_FAIL_TOLERANCE.get(
                CONDITIONS[condition_key], DEFAULT_REPRO_FAIL_TOLERANCE
            ),
        )
    )
    row: dict[str, object] = {
        "dataset": dataset_key,
        "case_id": case_id,
        "epsilon_n": epsilon_n,
        "epsilon_norm": epsilon_norm,
        "condition": CONDITIONS[condition_key],
        "seed": base_seed,
        "linf_norm": linf,
        "rms_published_norm": (
            float("nan") if published_rms is None else float(published_rms)
        ),
        "rms_reproduction_rel_error": repro_error,
        **scalars,
        **settings,
    }
    return row, centers, power


def _spectrum_rows(
    row: dict[str, object], centers: np.ndarray, power: np.ndarray
) -> list[dict[str, object]]:
    """Expand one radial spectrum into the long-form subsample table."""

    if centers.shape != power.shape:
        raise ValueError(
            f"spectrum centers and power must have equal shape, got "
            f"{centers.shape} and {power.shape}"
        )
    return [
        {
            "dataset": row["dataset"],
            "case_id": row["case_id"],
            "epsilon_n": row["epsilon_n"],
            "condition": row["condition"],
            "freq_bin": int(index),
            "f_r": float(frequency),
            "power": float(value),
        }
        for index, (frequency, value) in enumerate(zip(centers, power))
    ]


def _valid_mask(segmentation: np.ndarray, *, use_mask_for_norm: bool) -> np.ndarray:
    return (
        segmentation >= 0
        if use_mask_for_norm
        else np.ones_like(segmentation, dtype=bool)
    )


def evaluate_dataset(
    dataset_key: str,
    *,
    conditions: list[str],
    epsilon_n_values: list[int],
    attack_steps: int,
    batch_size: int,
    max_cases: int | None,
    base_seed: int,
    settings: dict[str, int],
    options: dict[str, int],
    output_csv: Path,
    spectra_csv: Path,
    quality: pd.DataFrame,
    normalization: pd.DataFrame,
    completed: set[tuple[str, str, int, str]],
    spectra_cases: set[str],
    device: torch.device,
    start_time: float,
    total_rows: int,
    already_complete: int = 0,
) -> None:
    """Regenerate requested perturbations and append descriptors for one cohort.

    ``already_complete`` is the row count this process inherited from a previous
    run; the throughput estimate divides by the rows *this* process wrote, since
    inherited rows cost it no time.
    """

    paths = _dataset_paths(dataset_key)
    requested_case_ids = _available_case_ids(paths["data"])
    if max_cases is not None:
        requested_case_ids = requested_case_ids[:max_cases]
    case_ids = [
        case_id
        for case_id in requested_case_ids
        if any(
            (dataset_key, case_id, epsilon_n, CONDITIONS[condition]) not in completed
            for condition in conditions
            for epsilon_n in epsilon_n_values
        )
    ]
    if not case_ids:
        print(f"[{dataset_key}] all requested rows already complete", flush=True)
        return

    use_mask = _use_mask_for_norm(paths["plans"])
    attack_valid_only = bool(int(settings.get("attack_valid_only", 0)))
    needs_attacks = any(
        condition in ATTACK_KEYS
        and any(
            (dataset_key, case_id, epsilon_n, CONDITIONS[condition]) not in completed
            for case_id in case_ids
            for epsilon_n in epsilon_n_values
        )
        for condition in conditions
    )
    model: torch.nn.Module | None = None
    num_classes = 0
    if needs_attacks:
        print(f"[{dataset_key}] loading fold-all UNet", flush=True)
        model, div_factors, num_classes = build_model(str(paths["checkpoint"]), device)
        model.eval()
        model.requires_grad_(False)
    else:
        div_factors = [1, 1, 1]

    batches = _plan_batches(str(paths["data"]), case_ids, div_factors, batch_size)
    print(
        f"[{dataset_key}] {len(case_ids)} remaining cases in {len(batches)} batches "
        f"(requested batch size {batch_size})",
        flush=True,
    )
    drift: dict[str, list[float]] = {}

    for batch_index, batch_case_ids in enumerate(batches, start=1):
        loaded: list[
            tuple[str, np.ndarray, np.ndarray, tuple[float, float, float]]
        ] = []
        for case_id in batch_case_ids:
            image_np, seg_np, spacing = load_sample(str(paths["data"]), case_id)
            spacing_3d = tuple(float(value) for value in spacing)
            if len(spacing_3d) != 3:
                raise ValueError(
                    f"expected 3-D spacing for {dataset_key}/{case_id}, got {spacing}"
                )
            loaded.append((case_id, image_np, np.asarray(seg_np), spacing_3d))

        clean_arrays = [
            np.asarray(image_np[0], dtype=np.float32)
            for _case_id, image_np, _seg_np, _spacing in loaded
        ]
        segmentations = [
            _seg_volume(seg_np) for _case_id, _image_np, seg_np, _spacing in loaded
        ]
        valid_masks = [
            _valid_mask(segmentation, use_mask_for_norm=use_mask)
            for segmentation in segmentations
        ]
        spacings = [record[3] for record in loaded]
        # Built once per case and reused by all 25 (condition x epsilon) rows:
        # the signed distance transform alone costs 0.24 s of the 0.37 s a
        # whole-gland row takes, and nothing it contains can depend on which
        # perturbation is being measured.
        geometries = [
            case_geometry(
                clean_arrays[index],
                segmentations[index],
                valid_mask=valid_masks[index],
                spacing=spacings[index],
                margin=options["spectral_margin"],
                min_size=options["spectral_min_size"],
            )
            for index in range(len(loaded))
        ]

        batch_needs_attacks = model is not None and any(
            (dataset_key, case_id, epsilon_n, CONDITIONS[condition]) not in completed
            for case_id, _image_np, _seg_np, _spacing in loaded
            for condition in conditions
            if condition in ATTACK_KEYS
            for epsilon_n in epsilon_n_values
        )
        image_batch: torch.Tensor | None = None
        label_batch: torch.Tensor | None = None
        original_shapes: list[tuple[int, ...]] = []
        if batch_needs_attacks:
            dimensions = [record[1].shape[1:] for record in loaded]
            maximum = [max(shape[index] for shape in dimensions) for index in range(3)]
            target = [
                int(math.ceil(value / factor) * factor)
                for value, factor in zip(maximum, div_factors)
            ]
            images: list[torch.Tensor] = []
            labels: list[torch.Tensor] = []
            for _case_id, image_np, seg_np, _spacing in loaded:
                image_tensor = torch.from_numpy(image_np[np.newaxis]).to(device)
                label_tensor = torch.from_numpy(
                    seg_np[np.newaxis].astype(np.float32)
                ).to(device)
                images.append(_pad_to(image_tensor, target))
                labels.append(
                    _pad_to(
                        label_tensor,
                        target,
                        mode="constant",
                        value=float(IGNORE_LABEL),
                    )
                )
                original_shapes.append(
                    tuple(int(value) for value in image_np.shape[1:])
                )
            image_batch = torch.cat(images, dim=0)
            label_batch = torch.cat(labels, dim=0)
            # Same rule as the attribution drivers' ``attackable_mask``: the
            # deployed pipeline regenerates ignore-labelled voxels after any
            # image-level attack, so a perturbation there cannot be realised.
            perturbation_mask = (
                (label_batch != IGNORE_LABEL) if attack_valid_only else None
            )

        def write_result(
            index: int,
            epsilon_n: int,
            condition_key: str,
            altered: np.ndarray,
            published_rms: float | None,
            target_rms: float | None = None,
        ) -> None:
            case_id = loaded[index][0]
            row, centers, power = build_row(
                dataset_key,
                case_id,
                epsilon_n,
                condition_key,
                clean_arrays[index],
                altered,
                segmentations[index],
                valid=valid_masks[index],
                spacing=spacings[index],
                published_rms=published_rms,
                base_seed=base_seed,
                settings=settings,
                options=options,
                geometry=geometries[index],
            )
            if published_rms is not None:
                error = float(row["rms_reproduction_rel_error"])
                if math.isfinite(error):
                    drift.setdefault(CONDITIONS[condition_key], []).append(error)
            if target_rms is not None:
                check_control_rms_match(
                    float(row["rms_norm"]),
                    target_rms,
                    label=(
                        f"{dataset_key}/{case_id}/{epsilon_n}/"
                        f"{CONDITIONS[condition_key]}"
                    ),
                )
            _append_rows(output_csv, [row], FIELDNAMES)
            if case_id in spectra_cases:
                _append_rows(
                    spectra_csv,
                    _spectrum_rows(row, centers, power),
                    SPECTRA_FIELDNAMES,
                )
            completed.add((dataset_key, case_id, epsilon_n, CONDITIONS[condition_key]))

        if image_batch is not None and label_batch is not None:
            if model is None:  # pragma: no cover - protected by construction
                raise RuntimeError("attack batch was built without a model")
            for attack_key in (key for key in conditions if key in ATTACK_KEYS):
                attack_label = CONDITIONS[attack_key]
                for epsilon_n in epsilon_n_values:
                    missing_indices = [
                        index
                        for index, (case_id, _image, _seg, _spacing) in enumerate(
                            loaded
                        )
                        if (
                            dataset_key,
                            case_id,
                            epsilon_n,
                            attack_label,
                        )
                        not in completed
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
                        else:
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
                    for index in missing_indices:
                        original = original_shapes[index]
                        altered = (
                            adversarial_batch[
                                index,
                                0,
                                : original[0],
                                : original[1],
                                : original[2],
                            ]
                            .detach()
                            .cpu()
                            .numpy()
                            .astype(np.float32)
                        )
                        # The published RMS was measured on unconfined attacks;
                        # a confined run is not a reproduction of it, so the
                        # drift reference is skipped (as in the attribution
                        # drivers) rather than reported as disagreement.
                        published = (
                            None
                            if attack_valid_only
                            else _published_rms(
                                quality,
                                dataset_key,
                                loaded[index][0],
                                epsilon_n,
                                attack_key,
                            )
                        )
                        write_result(
                            index,
                            epsilon_n,
                            attack_key,
                            altered,
                            published,
                        )
                    del adversarial_batch

        for control_key in (key for key in conditions if key in CONTROL_KEYS):
            control_label = CONDITIONS[control_key]
            for epsilon_n in epsilon_n_values:
                for index, (case_id, _image, _seg, _spacing) in enumerate(loaded):
                    key = (dataset_key, case_id, epsilon_n, control_label)
                    if key in completed:
                        continue
                    target_rms = _published_rms(
                        quality, dataset_key, case_id, epsilon_n, "apgd"
                    )
                    if target_rms is None:  # pragma: no cover - APGD is an attack
                        raise RuntimeError("APGD target RMS unexpectedly missing")
                    rng = np.random.default_rng(
                        stable_seed(
                            dataset_key,
                            case_id,
                            control_key,
                            epsilon_n,
                            0,
                            base_seed=base_seed,
                        )
                    )
                    if control_key == "gaussian":
                        altered = matched_gaussian_noise(
                            clean_arrays[index],
                            valid_mask=valid_masks[index],
                            epsilon=epsilon_n / 255.0,
                            target_rms=target_rms,
                            rng=rng,
                            match_iterations=PUBLISHED_MATCH_ITERATIONS,
                        )
                    else:
                        normalization_row = normalization.loc[(dataset_key, case_id)]
                        raw_mean = float(normalization_row["raw_norm_mean_au"])
                        raw_std = float(normalization_row["raw_norm_std_au"])
                        if not np.isfinite(raw_std) or raw_std <= 0:
                            raise ValueError(
                                f"invalid raw normalization std for "
                                f"{dataset_key}/{case_id}: {raw_std}"
                            )
                        altered = matched_rician_proxy_noise(
                            clean_arrays[index],
                            valid_mask=valid_masks[index],
                            epsilon=epsilon_n / 255.0,
                            target_rms=target_rms,
                            rng=rng,
                            raw_zero_normalized=-raw_mean / raw_std,
                            match_iterations=PUBLISHED_MATCH_ITERATIONS,
                        )
                    write_result(
                        index,
                        epsilon_n,
                        control_key,
                        altered,
                        published_rms=None,
                        target_rms=target_rms,
                    )

        elapsed = time.monotonic() - start_time
        # Rows inherited from an earlier run cost this process no time, so
        # dividing the *total* by this run's elapsed time reported a wildly
        # optimistic ETA for every batch of a resumed run.
        written = len(completed) - already_complete
        remaining = max(total_rows - len(completed), 0)
        rate = written / max(elapsed, 1e-9)
        eta = (
            f"{remaining / rate / 60.0:.1f} min" if written and rate > 0 else "unknown"
        )
        print(
            f"[{dataset_key}] batch {batch_index}/{len(batches)}; "
            f"rows {len(completed)}/{total_rows} ({written} this run); ETA {eta}",
            flush=True,
        )
        del image_batch, label_batch
        if device.type == "cuda":
            torch.cuda.empty_cache()

    for line in format_drift_summary(dataset_key, drift):
        print(line, flush=True)

    if model is not None:
        del model
    gc.collect()
    if device.type == "cuda":
        torch.cuda.empty_cache()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Characterize adversarial and RMS-matched random perturbation structure"
        )
    )
    parser.add_argument(
        "--datasets", nargs="+", choices=["wg", "zones"], default=["wg", "zones"]
    )
    parser.add_argument("--eps-n", nargs="+", type=int, default=DEFAULT_EPS_N)
    parser.add_argument(
        "--conditions",
        nargs="+",
        choices=list(CONDITIONS),
        default=list(CONDITIONS),
        help="conditions to regenerate (default: all attacks and controls)",
    )
    parser.add_argument("--attack-steps", type=int, default=20)
    parser.add_argument("--batch-size", type=int, default=2)
    parser.add_argument("--max-cases", type=int, default=None)
    parser.add_argument("--spectral-bins", type=int, default=32)
    parser.add_argument("--spectral-margin", type=int, default=16)
    parser.add_argument("--spectral-min-size", type=int, default=32)
    parser.add_argument("--alignment-voxels", type=int, default=200_000)
    parser.add_argument("--spectra-cases", type=int, default=50)
    parser.add_argument(
        "--max-gmsd-slices",
        type=int,
        default=5,
        help=(
            "axial slices per case used for GMSD, evenly spaced over the "
            "foreground slices (must be at least 1)"
        ),
    )
    parser.add_argument("--seed", type=int, default=20260721)
    parser.add_argument("--device", type=str, default=None)
    parser.add_argument("--quality-csv", type=Path, default=DEFAULT_QUALITY_CSV)
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
    if args.max_cases is not None and args.max_cases < 1:
        raise ValueError("--max-cases must be positive")
    if args.spectral_bins < 2:
        raise ValueError("--spectral-bins must be at least 2")
    if args.spectral_margin < 0:
        raise ValueError("--spectral-margin must be non-negative")
    if args.spectral_min_size < 2:
        raise ValueError("--spectral-min-size must be at least 2")
    if args.alignment_voxels < 1:
        raise ValueError("--alignment-voxels must be positive")
    if args.spectra_cases < 0:
        raise ValueError("--spectra-cases must be non-negative")
    # Not "non-negative": ``axial_gmsd_mean`` reads any value below 1 as "use
    # every foreground slice", which silently turns a 5-slice budget into a
    # 24-slice one instead of disabling the family.
    if args.max_gmsd_slices < 1:
        raise ValueError("--max-gmsd-slices must be at least 1")

    epsilon_n_values = sorted(set(int(value) for value in args.eps_n))
    if not epsilon_n_values or any(value <= 0 for value in epsilon_n_values):
        raise ValueError("--eps-n values must be positive")
    datasets = list(dict.fromkeys(str(value) for value in args.datasets))
    conditions = list(dict.fromkeys(str(value) for value in args.conditions))

    output_dir = args.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    output_csv = output_dir / "perturbation_structure_all_cases.csv"
    spectra_csv = output_dir / "radial_spectra_subsample.csv"
    settings = {
        "base_seed": int(args.seed),
        "attack_steps": int(args.attack_steps),
        "spectral_bins": int(args.spectral_bins),
        "spectral_margin": int(args.spectral_margin),
        "spectral_min_size": int(args.spectral_min_size),
        "alignment_voxels": int(args.alignment_voxels),
        "max_gmsd_slices": int(args.max_gmsd_slices),
        "spectra_cases": int(args.spectra_cases),
        "attack_valid_only": int(bool(args.attack_valid_only)),
    }
    options = {
        "spectral_bins": int(args.spectral_bins),
        "spectral_margin": int(args.spectral_margin),
        "spectral_min_size": int(args.spectral_min_size),
        "alignment_voxels": int(args.alignment_voxels),
        "max_gmsd_slices": int(args.max_gmsd_slices),
    }
    quality = _load_quality_lookup(args.quality_csv.resolve())
    normalization = _load_normalization_lookup(DEFAULT_NORMALIZATION_CSV)
    completed = completed_keys(output_csv, settings=settings)
    device = (
        torch.device(args.device)
        if args.device
        else torch.device("cuda" if torch.cuda.is_available() else "cpu")
    )

    requested_cases = {
        dataset_key: _available_case_ids(_dataset_paths(dataset_key)["data"])
        for dataset_key in datasets
    }
    if args.max_cases is not None:
        requested_cases = {
            key: values[: args.max_cases] for key, values in requested_cases.items()
        }
    spectra_cases_by_dataset = {
        key: set(spectra_case_subset(values, count=args.spectra_cases))
        for key, values in requested_cases.items()
    }
    requested_keys = {
        (dataset_key, case_id, epsilon_n, CONDITIONS[condition])
        for dataset_key, case_ids in requested_cases.items()
        for case_id in case_ids
        for condition in conditions
        for epsilon_n in epsilon_n_values
    }
    completed.intersection_update(requested_keys)
    total_rows = len(requested_keys)
    print(f"Device: {device}", flush=True)
    print(f"Output: {output_csv}", flush=True)
    print(f"Spectra: {spectra_csv}", flush=True)
    print(
        f"Requested: {sum(len(values) for values in requested_cases.values())} cases, "
        f"{total_rows} case-condition-epsilon rows; already complete: "
        f"{len(completed)}",
        flush=True,
    )
    print(f"Conditions: {[CONDITIONS[value] for value in conditions]}", flush=True)

    already_complete = len(completed)
    start_time = time.monotonic()
    for dataset_key in datasets:
        evaluate_dataset(
            dataset_key,
            conditions=conditions,
            epsilon_n_values=epsilon_n_values,
            attack_steps=args.attack_steps,
            batch_size=args.batch_size,
            max_cases=args.max_cases,
            base_seed=args.seed,
            settings=settings,
            options=options,
            output_csv=output_csv,
            spectra_csv=spectra_csv,
            quality=quality,
            normalization=normalization,
            completed=completed,
            spectra_cases=spectra_cases_by_dataset[dataset_key],
            device=device,
            start_time=start_time,
            total_rows=total_rows,
            already_complete=already_complete,
        )

    if requested_keys.issubset(completed):
        print(f"Complete: {len(completed)}/{total_rows} rows", flush=True)
    else:
        missing = len(requested_keys - completed)
        print(
            f"Incomplete: {missing} rows remain; rerun the same command to resume",
            flush=True,
        )


if __name__ == "__main__":
    main()
