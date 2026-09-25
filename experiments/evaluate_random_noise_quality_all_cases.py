"""Measure RMS-matched random-noise PSNR/SSIM on every preprocessed case.

The full-cohort APGD quality pass records the realized perturbation RMS for
every case and epsilon. This script uses those values as case-specific targets
for random controls. Matching RMS and enforcing the same L-inf/domain bounds
separates perturbation magnitude from spatial/adversarial structure:

* PSNR should agree with APGD up to numerical matching error because PSNR is a
  deterministic function of RMS and the clean-image range.
* SSIM may differ because it is sensitive to the spatial organization of the
  perturbation.

Rows are appended after each case, and deterministic semantic seeds make a
resumed run byte-for-byte reproducible for each requested control.
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import sys
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))
SRC_ROOT = REPO_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from experiments.adv_rob_eval_nnunet_nnunetrecenc import (  # noqa: E402
    DATASETS,
    NNUNET_PATHS,
    load_sample,
)
from mri_prostate_seg.experiments.epsilon_calibration import (  # noqa: E402
    matched_gaussian_noise,
    matched_rician_proxy_noise,
    perturbation_quality,
    stable_seed,
)


DEFAULT_RESULTS = REPO_ROOT / "results" / "adv_rob_eval_results" / "image_quality_analysis"
DEFAULT_APGD_CSV = (
    DEFAULT_RESULTS
    / "all_cases_adversarial_quality"
    / "adversarial_quality_all_cases.csv"
)
DEFAULT_OUTPUT_DIR = DEFAULT_RESULTS / "all_cases_random_noise_quality"
DEFAULT_EPS_N = [2, 4, 8, 16, 32]
NOISE_NAMES = {
    "gaussian": "gaussian_rms_matched",
    "rician": "rician_proxy_rms_matched",
}
FIELDNAMES = [
    "dataset",
    "case_id",
    "epsilon_n",
    "epsilon_norm",
    "epsilon_percent_case_sigma",
    "perturbation",
    "trial",
    "base_seed",
    "noise_seed",
    "target_apgd_rms_norm",
    "target_apgd_psnr_db_robust_range",
    "target_apgd_ssim_axial_prostate",
    "linf_norm",
    "rms_norm",
    "rms_match_relative_error",
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
    "match_iterations",
]


@dataclass(frozen=True)
class NoiseRequest:
    epsilon_n: int
    epsilon_norm: float
    noise_type: str
    trial: int
    target_rms: float
    target_psnr: float
    target_ssim: float


@dataclass(frozen=True)
class CaseJob:
    dataset: str
    case_id: str
    data_dir: str
    use_mask_for_norm: bool
    raw_mean: float
    raw_std: float
    requests: tuple[NoiseRequest, ...]
    base_seed: int
    max_ssim_slices: int
    match_iterations: int


def _dataset_paths(dataset_key: str) -> dict[str, Path]:
    config = DATASETS[dataset_key]
    dataset_name = config["dataset_name"]
    root = Path(NNUNET_PATHS)
    return {
        "data": root / "nnUNet_preprocessed" / dataset_name / config["data_subdir"],
        "plans": root / "nnUNet_preprocessed" / dataset_name / "nnUNetPlans.json",
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


def _load_apgd_targets(path: Path, epsilon_values: set[int]) -> pd.DataFrame:
    if not path.is_file():
        raise FileNotFoundError(f"missing full-cohort APGD quality CSV: {path}")
    frame = pd.read_csv(path)
    if "attack" in frame.columns:
        frame = frame[frame["attack"].str.casefold() == "apgd-bce"].copy()
    frame = frame[frame["epsilon_n"].isin(epsilon_values)].copy()
    required = {
        "dataset",
        "case_id",
        "epsilon_n",
        "epsilon_norm",
        "rms_norm",
        "psnr_db_robust_range",
        "ssim_axial_prostate",
        "case_raw_norm_mean_au",
        "case_raw_norm_std_au",
    }
    missing = sorted(required - set(frame.columns))
    if missing:
        raise ValueError(f"{path} is missing required columns: {missing}")
    duplicates = frame.duplicated(["dataset", "case_id", "epsilon_n"], keep=False)
    if duplicates.any():
        raise ValueError(f"{path} contains duplicate case/epsilon APGD targets")
    if frame.empty:
        raise ValueError(f"{path} contains no requested APGD targets")
    return frame


def _completed_keys(
    output_csv: Path,
    *,
    base_seed: int,
    max_ssim_slices: int,
    match_iterations: int,
) -> set[tuple[str, str, int, str, int]]:
    if not output_csv.is_file() or output_csv.stat().st_size == 0:
        return set()
    frame = pd.read_csv(output_csv)
    settings = {
        "base_seed": base_seed,
        "max_ssim_slices": max_ssim_slices,
        "match_iterations": match_iterations,
    }
    for column, expected in settings.items():
        if column not in frame.columns:
            raise ValueError(f"existing output lacks reproducibility field {column!r}")
        observed = set(frame[column].astype(int).unique())
        if observed != {int(expected)}:
            raise ValueError(
                f"existing output uses {column}={sorted(observed)}, requested {expected}; "
                "choose a different output directory"
            )
    return {
        (
            str(row.dataset),
            str(row.case_id),
            int(row.epsilon_n),
            str(row.perturbation),
            int(row.trial),
        )
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


def _evaluate_case(job: CaseJob) -> list[dict[str, object]]:
    image_np, seg_np, _spacing = load_sample(job.data_dir, job.case_id)
    clean = np.asarray(image_np[0], dtype=np.float32)
    segmentation = _seg_volume(seg_np)
    valid = (
        segmentation >= 0
        if job.use_mask_for_norm
        else np.ones_like(segmentation, dtype=bool)
    )
    foreground = segmentation > 0
    raw_zero_normalized = -job.raw_mean / job.raw_std
    rows: list[dict[str, object]] = []

    for request in job.requests:
        noise_seed = stable_seed(
            job.dataset,
            job.case_id,
            request.noise_type,
            request.epsilon_n,
            request.trial,
            base_seed=job.base_seed,
        )
        rng = np.random.default_rng(noise_seed)
        common = {
            "valid_mask": valid,
            "epsilon": request.epsilon_norm,
            "target_rms": request.target_rms,
            "rng": rng,
            "match_iterations": job.match_iterations,
        }
        if request.noise_type == "gaussian":
            altered = matched_gaussian_noise(clean, **common)
        elif request.noise_type == "rician":
            altered = matched_rician_proxy_noise(
                clean,
                **common,
                raw_zero_normalized=raw_zero_normalized,
            )
        else:  # pragma: no cover - guarded by argparse and job construction
            raise ValueError(f"unknown noise type: {request.noise_type}")

        metrics = perturbation_quality(
            clean,
            altered,
            valid_mask=valid,
            slice_selector=foreground,
            epsilon=request.epsilon_norm,
            max_ssim_slices=job.max_ssim_slices,
        )
        relative_error = abs(metrics["rms_norm"] - request.target_rms) / max(
            request.target_rms, 1e-12
        )
        if metrics["linf_norm"] > request.epsilon_norm + 2e-6:
            raise RuntimeError(
                f"L-inf violation for {job.dataset}/{job.case_id}/"
                f"{request.epsilon_n}/{request.noise_type}: {metrics['linf_norm']}"
            )
        if relative_error > 2e-3:
            raise RuntimeError(
                f"RMS match failed for {job.dataset}/{job.case_id}/"
                f"{request.epsilon_n}/{request.noise_type}: relative error "
                f"{relative_error:.6f}"
            )

        rows.append(
            {
                "dataset": job.dataset,
                "case_id": job.case_id,
                "epsilon_n": request.epsilon_n,
                "epsilon_norm": request.epsilon_norm,
                "epsilon_percent_case_sigma": 100.0 * request.epsilon_norm,
                "perturbation": NOISE_NAMES[request.noise_type],
                "trial": request.trial,
                "base_seed": job.base_seed,
                "noise_seed": noise_seed,
                "target_apgd_rms_norm": request.target_rms,
                "target_apgd_psnr_db_robust_range": request.target_psnr,
                "target_apgd_ssim_axial_prostate": request.target_ssim,
                "linf_norm": metrics["linf_norm"],
                "rms_norm": metrics["rms_norm"],
                "rms_match_relative_error": relative_error,
                "rms_percent_case_sigma": metrics["rms_percent_case_sigma"],
                "mae_norm": metrics["mae_norm"],
                "robust_signal_range_norm": metrics["robust_signal_range_norm"],
                "psnr_db_robust_range": metrics["psnr_db_robust_range"],
                "ssim_axial_prostate": metrics["ssim_axial_prostate"],
                "epsilon_saturation_fraction": metrics[
                    "epsilon_saturation_fraction"
                ],
                "case_raw_norm_mean_au": job.raw_mean,
                "case_raw_norm_std_au": job.raw_std,
                "linf_raw_au_exact": metrics["linf_norm"] * job.raw_std,
                "rms_raw_au_exact": metrics["rms_norm"] * job.raw_std,
                "mae_raw_au_exact": metrics["mae_norm"] * job.raw_std,
                "max_ssim_slices": job.max_ssim_slices,
                "match_iterations": job.match_iterations,
            }
        )
    return rows


def _write_summary(output_csv: Path, summary_csv: Path) -> None:
    frame = pd.read_csv(output_csv)
    metrics = [
        "rms_norm",
        "rms_match_relative_error",
        "rms_percent_case_sigma",
        "psnr_db_robust_range",
        "ssim_axial_prostate",
    ]
    by_case = (
        frame.groupby(
            ["dataset", "case_id", "epsilon_n", "epsilon_norm", "perturbation"],
            as_index=False,
        )[metrics]
        .mean()
        .sort_values(["dataset", "case_id", "epsilon_n", "perturbation"])
    )
    rows: list[dict[str, object]] = []
    for keys, group in by_case.groupby(
        ["dataset", "epsilon_n", "epsilon_norm", "perturbation"], sort=True
    ):
        dataset, epsilon_n, epsilon_norm, perturbation = keys
        row: dict[str, object] = {
            "dataset": dataset,
            "epsilon_n": int(epsilon_n),
            "epsilon_norm": float(epsilon_norm),
            "perturbation": perturbation,
            "n_cases": int(group["case_id"].nunique()),
            "trials_per_case": int(
                frame[
                    (frame["dataset"] == dataset)
                    & (frame["epsilon_n"] == epsilon_n)
                    & (frame["perturbation"] == perturbation)
                ]["trial"].nunique()
            ),
        }
        for metric in metrics:
            values = group[metric].to_numpy(dtype=float)
            row[f"{metric}_mean"] = float(np.mean(values))
            row[f"{metric}_median"] = float(np.median(values))
            row[f"{metric}_q05"] = float(np.quantile(values, 0.05))
            row[f"{metric}_q95"] = float(np.quantile(values, 0.95))
        rows.append(row)
    pd.DataFrame(rows).sort_values(
        ["dataset", "epsilon_n", "perturbation"]
    ).to_csv(summary_csv, index=False)


def _build_jobs(
    targets: pd.DataFrame,
    *,
    datasets: list[str],
    noise_types: list[str],
    trials: int,
    max_cases: int | None,
    base_seed: int,
    max_ssim_slices: int,
    match_iterations: int,
    completed: set[tuple[str, str, int, str, int]],
) -> tuple[list[CaseJob], set[tuple[str, str, int, str, int]]]:
    jobs: list[CaseJob] = []
    requested_keys: set[tuple[str, str, int, str, int]] = set()
    for dataset in datasets:
        paths = _dataset_paths(dataset)
        available = _available_case_ids(paths["data"])
        if max_cases is not None:
            available = available[:max_cases]
        dataset_targets = targets[
            (targets["dataset"] == dataset) & targets["case_id"].isin(available)
        ]
        target_cases = set(dataset_targets["case_id"].astype(str))
        missing_targets = sorted(set(available) - target_cases)
        if missing_targets:
            raise ValueError(
                f"APGD source lacks requested targets for {len(missing_targets)} "
                f"{dataset} cases; first missing case: {missing_targets[0]}"
            )
        use_mask = _use_mask_for_norm(paths["plans"])
        for case_id, case_rows in dataset_targets.groupby("case_id", sort=True):
            case_rows = case_rows.sort_values("epsilon_n")
            first = case_rows.iloc[0]
            requests: list[NoiseRequest] = []
            for record in case_rows.itertuples(index=False):
                for noise_type in noise_types:
                    perturbation = NOISE_NAMES[noise_type]
                    for trial in range(trials):
                        key = (
                            dataset,
                            str(case_id),
                            int(record.epsilon_n),
                            perturbation,
                            trial,
                        )
                        requested_keys.add(key)
                        if key in completed:
                            continue
                        requests.append(
                            NoiseRequest(
                                epsilon_n=int(record.epsilon_n),
                                epsilon_norm=float(record.epsilon_norm),
                                noise_type=noise_type,
                                trial=trial,
                                target_rms=float(record.rms_norm),
                                target_psnr=float(record.psnr_db_robust_range),
                                target_ssim=float(record.ssim_axial_prostate),
                            )
                        )
            if requests:
                jobs.append(
                    CaseJob(
                        dataset=dataset,
                        case_id=str(case_id),
                        data_dir=str(paths["data"]),
                        use_mask_for_norm=use_mask,
                        raw_mean=float(first["case_raw_norm_mean_au"]),
                        raw_std=float(first["case_raw_norm_std_au"]),
                        requests=tuple(requests),
                        base_seed=base_seed,
                        max_ssim_slices=max_ssim_slices,
                        match_iterations=match_iterations,
                    )
                )
    return jobs, requested_keys


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Evaluate RMS-matched random-noise PSNR/SSIM controls"
    )
    parser.add_argument(
        "--datasets", nargs="+", choices=["wg", "zones"], default=["wg", "zones"]
    )
    parser.add_argument("--eps-n", nargs="+", type=int, default=DEFAULT_EPS_N)
    parser.add_argument(
        "--noise-types",
        nargs="+",
        choices=sorted(NOISE_NAMES),
        default=["gaussian"],
        help="random controls to generate; Gaussian is the primary baseline",
    )
    parser.add_argument("--trials", type=int, default=1)
    parser.add_argument("--workers", type=int, default=1)
    parser.add_argument(
        "--progress-every",
        type=int,
        default=100,
        help="print progress after approximately this many newly written rows",
    )
    parser.add_argument("--match-iterations", type=int, default=12)
    parser.add_argument("--max-cases", type=int, default=None)
    parser.add_argument("--max-ssim-slices", type=int, default=5)
    parser.add_argument("--seed", type=int, default=20260721)
    parser.add_argument("--apgd-csv", type=Path, default=DEFAULT_APGD_CSV)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.trials < 1:
        raise ValueError("--trials must be at least 1")
    if args.workers < 1:
        raise ValueError("--workers must be at least 1")
    if args.progress_every < 1:
        raise ValueError("--progress-every must be at least 1")
    if args.match_iterations < 1:
        raise ValueError("--match-iterations must be at least 1")
    if args.max_cases is not None and args.max_cases < 1:
        raise ValueError("--max-cases must be positive")
    epsilon_values = sorted(set(int(value) for value in args.eps_n))
    if not epsilon_values or any(value <= 0 for value in epsilon_values):
        raise ValueError("--eps-n values must be positive")
    noise_types = list(dict.fromkeys(str(value) for value in args.noise_types))
    datasets = list(dict.fromkeys(str(value) for value in args.datasets))

    output_dir = args.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    output_csv = output_dir / "random_noise_quality_all_cases.csv"
    summary_csv = output_dir / "random_noise_quality_all_cases_summary.csv"
    completed = _completed_keys(
        output_csv,
        base_seed=args.seed,
        max_ssim_slices=args.max_ssim_slices,
        match_iterations=args.match_iterations,
    )
    targets = _load_apgd_targets(args.apgd_csv.resolve(), set(epsilon_values))
    jobs, requested_keys = _build_jobs(
        targets,
        datasets=datasets,
        noise_types=noise_types,
        trials=args.trials,
        max_cases=args.max_cases,
        base_seed=args.seed,
        max_ssim_slices=args.max_ssim_slices,
        match_iterations=args.match_iterations,
        completed=completed,
    )
    completed.intersection_update(requested_keys)

    print(f"APGD targets: {args.apgd_csv.resolve()}", flush=True)
    print(f"Output: {output_csv}", flush=True)
    print(
        f"Requested {len(requested_keys)} control rows across {len(jobs)} remaining "
        f"case jobs; already complete: {len(completed)}; workers: {args.workers}",
        flush=True,
    )
    start = time.monotonic()
    initial_completed = len(completed)

    def record(rows: list[dict[str, object]]) -> None:
        _append_rows(output_csv, rows)
        for row in rows:
            completed.add(
                (
                    str(row["dataset"]),
                    str(row["case_id"]),
                    int(row["epsilon_n"]),
                    str(row["perturbation"]),
                    int(row["trial"]),
                )
            )
        elapsed = time.monotonic() - start
        done = len(completed)
        newly_done = done - initial_completed
        rate = newly_done / max(elapsed, 1e-9)
        remaining = len(requested_keys - completed)
        eta_minutes = remaining / max(rate, 1e-9) / 60.0
        crossed_interval = (
            newly_done // args.progress_every
            != max(newly_done - len(rows), 0) // args.progress_every
        )
        if not crossed_interval and remaining:
            return
        print(
            f"rows {done}/{len(requested_keys)}; ETA {eta_minutes:.1f} min",
            flush=True,
        )

    if args.workers == 1:
        for job in jobs:
            record(_evaluate_case(job))
    elif jobs:
        max_workers = min(args.workers, os.cpu_count() or args.workers)
        with ProcessPoolExecutor(max_workers=max_workers) as executor:
            futures = [executor.submit(_evaluate_case, job) for job in jobs]
            for future in as_completed(futures):
                record(future.result())

    if requested_keys.issubset(completed):
        _write_summary(output_csv, summary_csv)
        print(f"Complete: {len(requested_keys)}/{len(requested_keys)} rows", flush=True)
        print(f"Saved {summary_csv}", flush=True)
    else:  # pragma: no cover - only reachable after external interruption
        remaining = len(requested_keys - completed)
        print(f"Incomplete: {remaining} rows remain; rerun to resume", flush=True)


if __name__ == "__main__":
    main()
