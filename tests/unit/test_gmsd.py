"""Tests for the vendored GMSD port."""

from __future__ import annotations

import numpy as np
import pytest

from mri_prostate_seg.experiments.gmsd import GMSD_T, axial_gmsd_mean, gmsd_2d


def _torch_reference_gmsd(x: np.ndarray, y: np.ndarray, t: float = GMSD_T) -> float:
    """piq 0.8.0 ``gmsd()`` transcribed to run without importing piq.

    piq cannot be imported here (torchvision is broken), so the reference is
    reproduced in Torch to check the NumPy port against a different framework.
    """
    import torch
    import torch.nn.functional as F

    xt = torch.tensor(x, dtype=torch.float64)[None, None]
    yt = torch.tensor(y, dtype=torch.float64)[None, None]
    down = max(xt.shape[2] % 2, xt.shape[3] % 2)
    xt = F.pad(xt, [0, down, 0, down])
    yt = F.pad(yt, [0, down, 0, down])
    xt = F.avg_pool2d(xt, kernel_size=2, stride=2)
    yt = F.avg_pool2d(yt, kernel_size=2, stride=2)

    prewitt = (
        torch.tensor(
            [[[-1.0, 0.0, 1.0], [-1.0, 0.0, 1.0], [-1.0, 0.0, 1.0]]],
            dtype=torch.float64,
        )
        / 3
    )
    kernels = torch.stack([prewitt, prewitt.transpose(-1, -2)])

    def gradient_map(v: "torch.Tensor") -> "torch.Tensor":
        g = F.conv2d(v, kernels, padding=1)
        return torch.sqrt(torch.sum(g**2, dim=-3, keepdim=True))

    gx, gy = gradient_map(xt), gradient_map(yt)
    gms = (2.0 * gx * gy + t) / (gx**2 + gy**2 + t)
    return float(torch.sqrt(((gms - gms.mean()) ** 2).mean()))


def _pair(height: int, width: int, seed: int) -> tuple[np.ndarray, np.ndarray]:
    rng = np.random.default_rng(seed)
    x = rng.random((height, width))
    y = np.clip(x + 0.05 * rng.standard_normal((height, width)), 0.0, 1.0)
    return x, y


def test_identical_images_score_zero() -> None:
    x, _ = _pair(64, 64, seed=0)
    assert gmsd_2d(x, x) == pytest.approx(0.0, abs=1e-12)


def test_perturbed_images_score_above_zero() -> None:
    x, y = _pair(64, 64, seed=1)
    assert gmsd_2d(x, y) > 0.0


@pytest.mark.parametrize(
    "shape",
    [
        (64, 64),  # both even
        (65, 63),  # both odd
        (95, 119),  # both odd
        (24, 32),  # both even
        (64, 63),  # MIXED parity — even/odd
        (96, 117),  # MIXED parity — a real zones in-plane shape
        (75, 96),  # MIXED parity — odd/even, the other direction
    ],
)
def test_matches_torch_reference_including_odd_shapes(shape: tuple[int, int]) -> None:
    x, y = _pair(shape[0], shape[1], seed=2)
    assert gmsd_2d(x, y) == pytest.approx(
        _torch_reference_gmsd(x, y), rel=1e-9, abs=1e-12
    )


def test_axial_mean_selects_foreground_slices_and_scales_to_unit_range() -> None:
    rng = np.random.default_rng(3)
    clean = rng.standard_normal((10, 40, 40)) * 2.0
    altered = clean + 0.01 * rng.standard_normal((10, 40, 40))
    selector = np.zeros((10, 40, 40), dtype=bool)
    selector[4:7, 10:30, 10:30] = True

    value = axial_gmsd_mean(
        clean, altered, lower=-4.0, upper=4.0, slice_selector=selector, max_slices=5
    )
    assert 0.0 < value < 1.0

    identical = axial_gmsd_mean(
        clean, clean, lower=-4.0, upper=4.0, slice_selector=selector, max_slices=5
    )
    assert identical == pytest.approx(0.0, abs=1e-12)
