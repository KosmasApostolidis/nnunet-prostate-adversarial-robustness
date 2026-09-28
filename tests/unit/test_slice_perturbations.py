from __future__ import annotations

import numpy as np
import pytest
import torch

from mri_prostate_seg.experiments.slice_vulnerability.metrics import volume_metrics_mm
from mri_prostate_seg.experiments.slice_vulnerability.perturbations import (
    ATTACKS,
    sign_noise,
)


def test_sign_noise_moves_every_unclamped_voxel_by_exactly_eps():
    x = torch.zeros(1, 1, 3, 4, 4)
    x[..., 1:3, 1:3] = 0.5
    bounds = (torch.full((1, 1, 1, 1, 1), -1.0), torch.full((1, 1, 1, 1, 1), 1.0))
    out = sign_noise(x, 0.1, bounds, generator=torch.Generator().manual_seed(0))
    assert torch.allclose((out - x).abs(), torch.full_like(x, 0.1))


def test_sign_noise_respects_intensity_bounds():
    x = torch.linspace(0.0, 1.0, 48).reshape(1, 1, 3, 4, 4)
    bounds = (x.amin().reshape(1, 1, 1, 1, 1), x.amax().reshape(1, 1, 1, 1, 1))
    out = sign_noise(x, 0.3, bounds, generator=torch.Generator().manual_seed(1))
    assert out.min() >= 0.0 and out.max() <= 1.0
    assert (out - x).abs().max() <= 0.3 + 1e-7


def test_sign_noise_is_reproducible_for_a_seeded_generator():
    x = torch.randn(1, 1, 3, 4, 4)
    bounds = (x.amin().reshape(1, 1, 1, 1, 1), x.amax().reshape(1, 1, 1, 1, 1))
    a = sign_noise(x, 0.05, bounds, generator=torch.Generator().manual_seed(42))
    b = sign_noise(x, 0.05, bounds, generator=torch.Generator().manual_seed(42))
    assert torch.equal(a, b)


def test_sign_noise_with_zero_eps_returns_the_input():
    x = torch.randn(1, 1, 2, 3, 3)
    bounds = (x.amin().reshape(1, 1, 1, 1, 1), x.amax().reshape(1, 1, 1, 1, 1))
    assert torch.equal(sign_noise(x, 0.0, bounds), x)


def test_sign_noise_leaves_masked_out_voxels_untouched():
    x = torch.zeros(1, 1, 2, 3, 3)
    mask = torch.zeros_like(x, dtype=torch.bool)
    mask[..., 0, :, :] = True
    bounds = (torch.full((1, 1, 1, 1, 1), -1.0), torch.full((1, 1, 1, 1, 1), 1.0))
    out = sign_noise(
        x, 0.1, bounds, mask=mask, generator=torch.Generator().manual_seed(0)
    )
    assert torch.equal(out[..., 1, :, :], x[..., 1, :, :])


def test_attack_registry_names_the_five_arms():
    assert set(ATTACKS) == {"fgsm", "pgd", "as_pgd", "auto_pgd", "noise"}


def test_auto_pgd_arm_stays_inside_the_ball_and_bounds():
    torch.manual_seed(0)
    net = torch.nn.Conv3d(1, 2, 1)
    x = torch.randn(1, 1, 2, 4, 4)
    y = (x > 0).long()
    bounds = (x.amin().reshape(1, 1, 1, 1, 1), x.amax().reshape(1, 1, 1, 1, 1))
    out = ATTACKS["auto_pgd"](net, x, y, 0.1, 2, bounds, n_steps=5, generator=None)
    assert (out - x).abs().max() <= 0.1 + 1e-6
    assert out.min() >= bounds[0].item() - 1e-6 and out.max() <= bounds[1].item() + 1e-6


def _cube(
    shape: tuple[int, int, int], start: tuple[int, int, int], size: int
) -> np.ndarray:
    vol = np.zeros(shape, dtype=np.int64)
    d, h, w = start
    vol[d : d + size, h : h + size, w : w + size] = 1
    return vol


def test_volume_metrics_scale_distances_by_physical_spacing():
    gt = _cube((10, 10, 10), (3, 3, 3), 4)
    pred = _cube((10, 10, 10), (4, 3, 3), 4)  # shifted one slice along depth
    dice, hd95_mm, asd_mm = volume_metrics_mm(
        pred, gt, class_id=1, spacing=(3.0, 0.5, 0.5)
    )
    assert dice == pytest.approx(0.75)
    assert hd95_mm == pytest.approx(3.0)
    assert 0.0 < asd_mm <= 3.0


def test_volume_metrics_asd_is_symmetric():
    gt = _cube((12, 12, 12), (2, 2, 2), 8)
    pred = _cube((12, 12, 12), (4, 4, 4), 2)  # small cube deep inside the GT
    spacing = (3.0, 0.5, 0.5)
    _, _, asd_ab = volume_metrics_mm(pred, gt, class_id=1, spacing=spacing)
    _, _, asd_ba = volume_metrics_mm(gt, pred, class_id=1, spacing=spacing)
    assert asd_ab == pytest.approx(asd_ba)


def test_volume_metrics_exclude_ignore_voxels():
    gt = _cube((6, 6, 6), (1, 1, 1), 3)
    pred = gt.copy()
    gt[0] = -1
    pred[0] = 1  # predictions inside the ignore region must not count
    dice, _, _ = volume_metrics_mm(pred, gt, class_id=1, spacing=(1.0, 1.0, 1.0))
    assert dice == pytest.approx(1.0)


def test_volume_metrics_return_nan_distances_for_an_empty_prediction():
    gt = _cube((6, 6, 6), (1, 1, 1), 3)
    pred = np.zeros_like(gt)
    dice, hd95_mm, asd_mm = volume_metrics_mm(
        pred, gt, class_id=1, spacing=(3.0, 0.5, 0.5)
    )
    assert dice == pytest.approx(0.0, abs=1e-6)
    assert np.isnan(hd95_mm) and np.isnan(asd_mm)
