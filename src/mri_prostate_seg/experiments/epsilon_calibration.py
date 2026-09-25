"""Pure numerical helpers for MRI epsilon-scale calibration.

MRI magnitude intensities are arbitrary units, and this repository's nnU-Net
inputs are normalized per case.  The helpers in this module therefore keep the
primary measurements in normalized units and in fractions of the within-case
standard deviation.  When the raw per-case normalization standard deviation is
available, multiplying a normalized perturbation by that value gives its exact
intensity scale in the original image's arbitrary units.
"""

from __future__ import annotations

import hashlib
import math
from typing import Callable

import numpy as np
from scipy.signal import convolve2d
from skimage.metrics import structural_similarity


_HF_KERNEL = np.asarray(
    [[1.0, -2.0, 1.0], [-2.0, 4.0, -2.0], [1.0, -2.0, 1.0]],
    dtype=np.float64,
)


def stable_seed(*parts: object, base_seed: int = 0) -> int:
    """Return a process-independent 31-bit seed for a semantic run key."""

    payload = "|".join([str(base_seed), *(str(p) for p in parts)]).encode("utf-8")
    digest = hashlib.sha256(payload).digest()
    return int.from_bytes(digest[:8], byteorder="big", signed=False) % (2**31 - 1)


def normalized_epsilon_mapping(
    epsilon_n: int,
    *,
    cohort_foreground_std_au: float,
) -> dict[str, float | int | str]:
    """Map a legacy ``n/255`` epsilon to defensible normalized-space units.

    The raw-AU value is only a cohort-level scale proxy.  Exact inversion would
    require the pre-normalization standard deviation for the individual case.
    """

    epsilon_norm = float(epsilon_n) / 255.0
    return {
        "epsilon_n": int(epsilon_n),
        "epsilon_norm": epsilon_norm,
        "epsilon_percent_within_case_sigma": 100.0 * epsilon_norm,
        "cohort_raw_linf_proxy_au": epsilon_norm * float(cohort_foreground_std_au),
        "recommended_label": (
            f"epsilon_norm={epsilon_norm:.5f} "
            f"(legacy {epsilon_n}/255; {100.0 * epsilon_norm:.3f}% of case sigma)"
        ),
    }


def _as_mask(mask: np.ndarray | None, shape: tuple[int, ...]) -> np.ndarray:
    if mask is None:
        return np.ones(shape, dtype=bool)
    out = np.asarray(mask, dtype=bool)
    if out.shape != shape:
        raise ValueError(f"mask shape {out.shape} does not match image shape {shape}")
    if not out.any():
        raise ValueError("mask contains no valid voxels")
    return out


def robust_data_range(
    image: np.ndarray,
    mask: np.ndarray | None = None,
    *,
    lower_percentile: float = 0.5,
    upper_percentile: float = 99.5,
) -> float:
    """Robust per-volume signal range used to normalize PSNR and SSIM."""

    arr = np.asarray(image, dtype=np.float64)
    valid = _as_mask(mask, arr.shape)
    lo, hi = np.percentile(arr[valid], [lower_percentile, upper_percentile])
    value = float(hi - lo)
    if not np.isfinite(value) or value <= 0:
        value = float(np.ptp(arr[valid]))
    return value if np.isfinite(value) and value > 0 else 1.0


def axial_ssim_mean(
    clean: np.ndarray,
    altered: np.ndarray,
    *,
    data_range: float,
    slice_selector: np.ndarray | None = None,
    max_slices: int = 5,
) -> float:
    """Mean 2-D SSIM on deterministic axial slices.

    Mixing through-plane voxels in a 3-D SSIM window is inappropriate for the
    strongly anisotropic 3 mm x 0.5 mm x 0.5 mm volumes, so SSIM is evaluated
    within axial planes.  If supplied, ``slice_selector`` chooses slices that
    contain the prostate foreground.
    """

    x = np.asarray(clean, dtype=np.float64)
    y = np.asarray(altered, dtype=np.float64)
    if x.shape != y.shape or x.ndim != 3:
        raise ValueError("axial_ssim_mean expects equal-shape 3-D volumes")

    if slice_selector is None:
        candidates = np.arange(x.shape[0], dtype=int)
    else:
        selector = np.asarray(slice_selector, dtype=bool)
        if selector.shape != x.shape:
            raise ValueError("slice_selector must have the same shape as the image")
        candidates = np.flatnonzero(selector.reshape(selector.shape[0], -1).any(axis=1))
        if not len(candidates):
            candidates = np.arange(x.shape[0], dtype=int)

    if max_slices > 0 and len(candidates) > max_slices:
        take = np.linspace(0, len(candidates) - 1, max_slices).round().astype(int)
        candidates = candidates[take]

    values = [
        float(
            structural_similarity(
                x[int(z)],
                y[int(z)],
                data_range=float(data_range),
            )
        )
        for z in candidates
        if min(x[int(z)].shape) >= 7
    ]
    return float(np.mean(values)) if values else float("nan")


def perturbation_quality(
    clean: np.ndarray,
    altered: np.ndarray,
    *,
    valid_mask: np.ndarray | None = None,
    slice_selector: np.ndarray | None = None,
    epsilon: float | None = None,
    high_frequency_noise_sigma: float | None = None,
    max_ssim_slices: int = 5,
) -> dict[str, float]:
    """Compute image-quality and perturbation-scale measurements."""

    x = np.asarray(clean, dtype=np.float64)
    y = np.asarray(altered, dtype=np.float64)
    if x.shape != y.shape:
        raise ValueError(f"shape mismatch: clean={x.shape}, altered={y.shape}")
    valid = _as_mask(valid_mask, x.shape)
    delta = y - x
    values = delta[valid]
    clean_values = x[valid]

    rms = float(np.sqrt(np.mean(values**2)))
    mae = float(np.mean(np.abs(values)))
    linf = float(np.max(np.abs(values)))
    case_std = float(clean_values.std())
    signal_range = robust_data_range(x, valid)
    psnr = float("inf") if rms == 0 else float(20.0 * np.log10(signal_range / rms))
    ssim = axial_ssim_mean(
        x,
        y,
        data_range=signal_range,
        slice_selector=slice_selector,
        max_slices=max_ssim_slices,
    )

    hf = float(high_frequency_noise_sigma or float("nan"))
    saturation = float("nan")
    if epsilon is not None and epsilon > 0:
        saturation = float(np.mean(np.abs(values) >= 0.99 * float(epsilon)))

    return {
        "linf_norm": linf,
        "rms_norm": rms,
        "mae_norm": mae,
        "clean_std_norm": case_std,
        "linf_percent_case_sigma": 100.0 * linf / max(case_std, 1e-12),
        "rms_percent_case_sigma": 100.0 * rms / max(case_std, 1e-12),
        "robust_signal_range_norm": signal_range,
        "psnr_db_robust_range": psnr,
        "ssim_axial_prostate": ssim,
        "high_frequency_noise_sigma_norm": hf,
        "linf_over_hf_noise": linf / hf if np.isfinite(hf) and hf > 0 else float("nan"),
        "rms_over_hf_noise": rms / hf if np.isfinite(hf) and hf > 0 else float("nan"),
        "epsilon_saturation_fraction": saturation,
    }


def estimate_high_frequency_noise_sigma(
    image: np.ndarray,
    *,
    valid_mask: np.ndarray | None = None,
    slice_selector: np.ndarray | None = None,
    max_slices: int = 5,
) -> float:
    """Estimate an image-derived high-frequency noise proxy.

    This is the Immerkaer Laplacian estimator, applied to axial slices.  It is
    useful as a repeatable image-texture/noise scale but is *not* a measurement
    of scanner acquisition noise: edges and fine anatomy can inflate it.
    """

    x = np.asarray(image, dtype=np.float64)
    if x.ndim != 3:
        raise ValueError("estimate_high_frequency_noise_sigma expects a 3-D volume")
    valid = _as_mask(valid_mask, x.shape)

    selector = (
        valid if slice_selector is None else np.asarray(slice_selector, dtype=bool)
    )
    if selector.shape != x.shape:
        raise ValueError("slice_selector must have the same shape as image")
    candidates = np.flatnonzero(selector.reshape(x.shape[0], -1).any(axis=1))
    if not len(candidates):
        candidates = np.arange(x.shape[0], dtype=int)
    if max_slices > 0 and len(candidates) > max_slices:
        take = np.linspace(0, len(candidates) - 1, max_slices).round().astype(int)
        candidates = candidates[take]

    estimates: list[float] = []
    for z in candidates:
        mask2d = valid[int(z)]
        coords = np.argwhere(mask2d)
        if not len(coords):
            continue
        lo = np.maximum(coords.min(axis=0), 0)
        hi = np.minimum(coords.max(axis=0) + 1, np.asarray(mask2d.shape))
        plane = x[int(z), lo[0] : hi[0], lo[1] : hi[1]]
        if min(plane.shape) < 3:
            continue
        response = convolve2d(plane, _HF_KERNEL, mode="valid")
        estimate = math.sqrt(math.pi / 2.0) * float(np.abs(response).sum())
        estimate /= 6.0 * float(response.size)
        if np.isfinite(estimate):
            estimates.append(estimate)
    return float(np.median(estimates)) if estimates else float("nan")


def _project_candidate(
    clean: np.ndarray,
    candidate: np.ndarray,
    *,
    valid_mask: np.ndarray,
    epsilon: float,
) -> np.ndarray:
    valid = _as_mask(valid_mask, clean.shape)
    clean64 = np.asarray(clean, dtype=np.float64)
    cand = np.asarray(candidate, dtype=np.float64)
    delta = np.clip(cand - clean64, -float(epsilon), float(epsilon))
    out = clean64 + delta
    lo = float(clean64[valid].min())
    hi = float(clean64[valid].max())
    out[valid] = np.clip(out[valid], lo, hi)
    out[~valid] = clean64[~valid]
    return out.astype(np.float32)


def _match_parameter_by_rms(
    clean: np.ndarray,
    *,
    valid_mask: np.ndarray,
    epsilon: float,
    target_rms: float,
    candidate_fn: Callable[[float], np.ndarray],
    iterations: int = 32,
) -> np.ndarray:
    if iterations < 1:
        raise ValueError("iterations must be at least 1")
    if target_rms <= 0 or epsilon <= 0:
        return np.asarray(clean, dtype=np.float32).copy()

    valid = _as_mask(valid_mask, clean.shape)

    def evaluated(scale: float) -> tuple[np.ndarray, float]:
        value = _project_candidate(
            clean,
            candidate_fn(scale),
            valid_mask=valid,
            epsilon=epsilon,
        )
        rms = float(np.sqrt(np.mean((value[valid] - clean[valid]) ** 2)))
        return value, rms

    low = 0.0
    high = max(float(target_rms), float(epsilon) / 4.0, 1e-8)
    best, achieved = evaluated(high)
    for _ in range(20):
        if achieved >= target_rms * 0.999:
            break
        high *= 2.0
        best, achieved = evaluated(high)

    for _ in range(iterations):
        mid = 0.5 * (low + high)
        candidate, achieved = evaluated(mid)
        best = candidate
        if achieved < target_rms:
            low = mid
        else:
            high = mid
    return best


def matched_gaussian_noise(
    clean: np.ndarray,
    *,
    valid_mask: np.ndarray,
    epsilon: float,
    target_rms: float,
    rng: np.random.Generator,
    match_iterations: int = 32,
) -> np.ndarray:
    """Gaussian noise matched to target RMS under the same L-inf/domain box."""

    base = rng.standard_normal(np.asarray(clean).shape)
    return _match_parameter_by_rms(
        np.asarray(clean),
        valid_mask=valid_mask,
        epsilon=epsilon,
        target_rms=target_rms,
        candidate_fn=lambda sigma: np.asarray(clean, dtype=np.float64) + sigma * base,
        iterations=match_iterations,
    )


def matched_rician_proxy_noise(
    clean: np.ndarray,
    *,
    valid_mask: np.ndarray,
    epsilon: float,
    target_rms: float,
    rng: np.random.Generator,
    raw_zero_normalized: float | None = None,
    match_iterations: int = 32,
) -> np.ndarray:
    """Rician-magnitude proxy matched to target RMS and L-inf.

    ``raw_zero_normalized`` should be ``-raw_mean / raw_std`` when the raw
    per-case z-score parameters are known. If omitted, the valid-region minimum
    is used as a zero-signal proxy. The returned perturbation is still best
    described as a Rician proxy when the stored MRI has negative values or has
    undergone processing that breaks the magnitude-image noise model.
    """

    x = np.asarray(clean, dtype=np.float64)
    valid = _as_mask(valid_mask, x.shape)
    zero = (
        float(raw_zero_normalized)
        if raw_zero_normalized is not None
        else float(x[valid].min())
    )
    signal = np.maximum(x - zero, 0.0)
    n1 = rng.standard_normal(x.shape)
    n2 = rng.standard_normal(x.shape)

    def candidate(sigma: float) -> np.ndarray:
        magnitude = np.sqrt((signal + sigma * n1) ** 2 + (sigma * n2) ** 2)
        # Express the magnitude-noise increment around the stored clean image.
        # This keeps candidate(0) == clean even for processed negative values.
        return x + magnitude - signal

    return _match_parameter_by_rms(
        x,
        valid_mask=valid,
        epsilon=epsilon,
        target_rms=target_rms,
        candidate_fn=candidate,
        iterations=match_iterations,
    )


__all__ = [
    "axial_ssim_mean",
    "estimate_high_frequency_noise_sigma",
    "matched_gaussian_noise",
    "matched_rician_proxy_noise",
    "normalized_epsilon_mapping",
    "perturbation_quality",
    "robust_data_range",
    "stable_seed",
]
