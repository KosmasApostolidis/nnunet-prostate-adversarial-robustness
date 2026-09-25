"""ROI construction and multiscale clean-image edge strength (plan §6, §10)."""

from __future__ import annotations

from collections.abc import Sequence

import numpy as np
from scipy.ndimage import gaussian_filter

from mri_prostate_seg.experiments.perturbation_structure import gradient_magnitude_mm


def roi_mask(
    foreground: np.ndarray,
    valid: np.ndarray,
    spacing: tuple[float, float, float],
    *,
    margin_mm_inplane: float = 20.0,
    margin_slices: int = 1,
) -> np.ndarray:
    """Bounding box of ``foreground`` grown by a physical in-plane margin and
    ``margin_slices`` through-plane slices, intersected with ``valid``."""

    fg = np.asarray(foreground, dtype=bool)
    ok = np.asarray(valid, dtype=bool)
    if fg.shape != ok.shape or fg.ndim != 3:
        raise ValueError("foreground and valid must be equal-shaped 3-D masks")
    if not fg.any():
        raise ValueError("foreground is empty")
    z, y, x = np.nonzero(fg)
    my = int(np.ceil(margin_mm_inplane / spacing[1]))
    mx = int(np.ceil(margin_mm_inplane / spacing[2]))
    roi = np.zeros_like(fg)
    roi[
        max(z.min() - margin_slices, 0) : z.max() + margin_slices + 1,
        max(y.min() - my, 0) : y.max() + my + 1,
        max(x.min() - mx, 0) : x.max() + mx + 1,
    ] = True
    return roi & ok


def robust_normalize(
    values: np.ndarray, domain: np.ndarray, *, eta: float = 1e-8
) -> np.ndarray:
    """``(v - Q50) / max(Q95 - Q50, eta)`` with quantiles taken over ``domain``."""

    inside = np.asarray(values, dtype=np.float64)[np.asarray(domain, dtype=bool)]
    if inside.size == 0:
        raise ValueError("domain contains no voxels")
    q50, q95 = np.quantile(inside, [0.50, 0.95])
    # max(), not (q95 - q50 + eta): the additive form biases every normalized
    # value, so a non-degenerate spread would not map q95 to exactly 1.0.
    return (np.asarray(values, dtype=np.float64) - q50) / max(float(q95 - q50), eta)


def multiscale_edge_strength(
    image: np.ndarray,
    spacing: tuple[float, float, float],
    *,
    sigmas_mm: Sequence[float],
    domain: np.ndarray,
) -> np.ndarray:
    """Max over scales of the robustly normalised physical gradient magnitude.

    Smoothing is in-plane only: with 3 mm slices every requested sigma is a
    fraction of a slice, so sigma_z is fixed at 0 (spec: anisotropy).
    """

    vol = np.asarray(image, dtype=np.float32)
    if vol.ndim != 3:
        raise ValueError(f"expected a 3-D image, got {vol.ndim}-D")
    if not sigmas_mm:
        raise ValueError("sigmas_mm must not be empty")
    maps = []
    for sigma_mm in sigmas_mm:
        sigma_vox = (0.0, float(sigma_mm) / spacing[1], float(sigma_mm) / spacing[2])
        smooth = gaussian_filter(vol, sigma=sigma_vox)
        maps.append(robust_normalize(gradient_magnitude_mm(smooth, spacing), domain))
    return np.max(np.stack(maps, axis=0), axis=0)


__all__ = ["multiscale_edge_strength", "robust_normalize", "roi_mask"]
