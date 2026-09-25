"""Gradient Magnitude Similarity Deviation (GMSD).

Port of the reference implementation in piq 0.8.0 (``piq/gmsd.py``), which
follows Xue et al., "Gradient Magnitude Similarity Deviation: A Highly Efficient
Perceptual Image Quality Index" (2013), arXiv:1308.3052.

piq itself cannot be imported in this environment: ``piq/__init__.py`` imports
``.perceptual``, which imports ``torchvision.models``, and torchvision raises
``operator torchvision::nms does not exist`` here.  Importing any piq submodule
executes the package ``__init__``, so the dependency is unusable and the part
that matters is reproduced instead.

GMSD is 0 for identical images and grows as gradient structure diverges.  It
complements SSIM: SSIM weighs luminance, contrast, and structure, while GMSD is
purely gradient-based.
"""

from __future__ import annotations

import numpy as np
from scipy.signal import convolve2d


GMSD_T = 170.0 / (255.0**2)

# Normalized 3x3 Prewitt kernel in x, matching piq.functional.prewitt_filter.
_PREWITT_X = (
    np.asarray([[-1.0, 0.0, 1.0], [-1.0, 0.0, 1.0], [-1.0, 0.0, 1.0]], dtype=np.float64)
    / 3.0
)


def _gradient_magnitude(image: np.ndarray) -> np.ndarray:
    """Prewitt gradient magnitude with zero padding.

    ``convolve2d`` flips the kernel relative to torch's cross-correlating
    ``conv2d``; the sign difference cancels because both components are squared.
    """
    gx = convolve2d(image, _PREWITT_X, mode="same", boundary="fill", fillvalue=0.0)
    gy = convolve2d(image, _PREWITT_X.T, mode="same", boundary="fill", fillvalue=0.0)
    return np.sqrt(gx**2 + gy**2)


def _average_pool_2x2(image: np.ndarray) -> np.ndarray:
    """2x2 mean pooling with the floor semantics of ``F.avg_pool2d``.

    A trailing odd row or column is dropped rather than raising.  This matters:
    piq pads both dimensions by ``max(H % 2, W % 2)``, which leaves one
    dimension odd whenever H and W have different parity, and 296 of the 603
    zones cases are mixed-parity in-plane (e.g. 96x117).  ``avg_pool2d`` floors
    in that case; a bare reshape would raise.
    """

    height = image.shape[0] - image.shape[0] % 2
    width = image.shape[1] - image.shape[1] % 2
    cropped = image[:height, :width]
    return cropped.reshape(height // 2, 2, width // 2, 2).mean(axis=(1, 3))


def gmsd_2d(x: np.ndarray, y: np.ndarray, *, t: float = GMSD_T) -> float:
    """GMSD between two 2-D arrays already scaled to [0, 1]."""

    a = np.asarray(x, dtype=np.float64)
    b = np.asarray(y, dtype=np.float64)
    if a.shape != b.shape or a.ndim != 2:
        raise ValueError(
            f"gmsd_2d expects equal-shape 2-D arrays, got {a.shape}, {b.shape}"
        )

    # piq pads the right and bottom edges by the same amount so both dimensions
    # become even before the 2x downsample.
    pad = max(a.shape[0] % 2, a.shape[1] % 2)
    if pad:
        a = np.pad(a, ((0, pad), (0, pad)), mode="constant")
        b = np.pad(b, ((0, pad), (0, pad)), mode="constant")

    a = _average_pool_2x2(a)
    b = _average_pool_2x2(b)

    ga = _gradient_magnitude(a)
    gb = _gradient_magnitude(b)
    gms = (2.0 * ga * gb + t) / (ga**2 + gb**2 + t)
    return float(np.sqrt(np.mean((gms - gms.mean()) ** 2)))


def axial_gmsd_mean(
    clean: np.ndarray,
    altered: np.ndarray,
    *,
    lower: float,
    upper: float,
    slice_selector: np.ndarray | None = None,
    max_slices: int = 5,
) -> float:
    """Mean GMSD over deterministic axial slices.

    Both volumes are mapped to [0, 1] with the *same* affine transform derived
    from the clean volume's robust percentiles, so GMSD's constant ``t`` keeps
    its intended scale on z-scored MRI.  Slice selection mirrors
    ``epsilon_calibration.axial_ssim_mean`` exactly.
    """

    x = np.asarray(clean, dtype=np.float64)
    y = np.asarray(altered, dtype=np.float64)
    if x.shape != y.shape or x.ndim != 3:
        raise ValueError("axial_gmsd_mean expects equal-shape 3-D volumes")
    span = float(upper) - float(lower)
    if not np.isfinite(span) or span <= 0:
        raise ValueError(f"invalid intensity bounds: lower={lower}, upper={upper}")

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

    def to_unit(v: np.ndarray) -> np.ndarray:
        return np.clip((v - float(lower)) / span, 0.0, 1.0)

    values = [gmsd_2d(to_unit(x[int(z)]), to_unit(y[int(z)])) for z in candidates]
    return float(np.mean(values))


__all__ = ["GMSD_T", "axial_gmsd_mean", "gmsd_2d"]
