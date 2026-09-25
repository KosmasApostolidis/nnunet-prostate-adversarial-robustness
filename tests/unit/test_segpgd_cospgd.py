"""Contract tests for the SegPGD and CosPGD loss-variant PGD arms.

They pin the parts a reader would check against the papers: SegPGD's
(t - 1) / 2T schedule and correct/wrong voxel weighting (Eq. 4), CosPGD's
detached cosine scaling of the per-voxel cross-entropy, ignore-label handling,
and that both trajectories obey the L-infinity ball and the update convention
they share with the ``pgd`` arm.
"""

from __future__ import annotations

import pytest
import torch
import torch.nn.functional as F

from mri_prostate_seg.attacks.cospgd import cospgd_attack, cospgd_loss
from mri_prostate_seg.attacks.pgd import pgd_trajectory
from mri_prostate_seg.attacks.segpgd import (
    _ScheduledSegPGDLoss,
    segpgd_attack,
    segpgd_lambda,
    segpgd_loss,
    segpgd_trajectory,
)


class _ToySegmentationModel(torch.nn.Module):
    def __init__(self) -> None:
        super().__init__()
        torch.manual_seed(0)
        self.conv = torch.nn.Conv3d(1, 2, kernel_size=3, padding=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.conv(x)


def _toy_case() -> tuple[torch.Tensor, torch.Tensor]:
    x = torch.linspace(-1.0, 1.0, 4 * 8 * 8).reshape(1, 1, 4, 8, 8)
    y = torch.zeros((1, 1, 4, 8, 8), dtype=torch.float32)
    y[:, :, 1:3, 2:6, 2:6] = 1.0
    return x, y


def _mean_ce(logits: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
    return F.cross_entropy(logits, y.squeeze(1).long())


# --- SegPGD -----------------------------------------------------------------


def test_segpgd_lambda_is_paper_schedule() -> None:
    T = 20
    assert segpgd_lambda(1, T) == 0.0
    assert segpgd_lambda(T, T) == pytest.approx((T - 1) / (2 * T))
    assert segpgd_lambda(T, T) < 0.5
    with pytest.raises(ValueError):
        segpgd_lambda(0, T)
    with pytest.raises(ValueError):
        segpgd_lambda(T + 1, T)


def test_segpgd_loss_weights_correct_and_wrong_voxels() -> None:
    torch.manual_seed(1)
    logits = torch.randn(1, 2, 2, 3, 3)
    y = torch.randint(0, 2, (1, 1, 2, 3, 3)).float()
    labels = y.squeeze(1).long()
    per_voxel = F.cross_entropy(logits, labels, reduction="none")
    correct = logits.argmax(dim=1) == labels
    T = 10
    for t in (1, 5, T):
        lam = segpgd_lambda(t, T)
        expected = (
            (1 - lam) * per_voxel[correct].sum() + lam * per_voxel[~correct].sum()
        ) / per_voxel.numel()
        got = segpgd_loss(logits, y, 2, t, T)
        assert torch.allclose(got, expected)


def test_segpgd_loss_first_step_uses_only_correct_voxels() -> None:
    torch.manual_seed(2)
    logits = torch.randn(1, 2, 2, 3, 3)
    y = torch.randint(0, 2, (1, 1, 2, 3, 3)).float()
    labels = y.squeeze(1).long()
    wrong = logits.argmax(dim=1) != labels
    assert wrong.any(), "fixture needs at least one wrong voxel"
    per_voxel = F.cross_entropy(logits, labels, reduction="none")
    got = segpgd_loss(logits, y, 2, 1, 10)
    expected = per_voxel[~wrong].sum() / per_voxel.numel()
    assert torch.allclose(got, expected)


def test_segpgd_loss_excludes_ignore_voxels() -> None:
    torch.manual_seed(3)
    logits = torch.randn(1, 2, 2, 3, 3)
    y = torch.randint(0, 2, (1, 1, 2, 3, 3)).float()
    y_ignored = y.clone()
    y_ignored[0, 0, 0, 0, :] = -1.0
    got = segpgd_loss(logits, y_ignored, 2, 5, 10)
    assert torch.isfinite(got)
    # Recompute by hand over the valid voxels only.
    labels = y_ignored.squeeze(1).long()
    valid = labels != -1
    safe = torch.where(valid, labels, torch.zeros_like(labels))
    per_voxel = F.cross_entropy(logits, safe, reduction="none")
    correct = (logits.argmax(dim=1) == labels) & valid
    lam = segpgd_lambda(5, 10)
    expected = (
        (1 - lam) * per_voxel[correct].sum() + lam * per_voxel[valid & ~correct].sum()
    ) / valid.sum()
    assert torch.allclose(got, expected)


def test_segpgd_trajectory_advances_schedule_once_per_iteration() -> None:
    model = _ToySegmentationModel()
    x, y = _toy_case()
    scheduled = _ScheduledSegPGDLoss(5)
    pgd_trajectory(model, x, y, 0.1, 2, scheduled, n_steps=5, random_start=True)
    assert scheduled.steps_seen == [1, 2, 3, 4, 5]


def test_segpgd_restart_resets_schedule() -> None:
    model = _ToySegmentationModel()
    x, y = _toy_case()
    log_a: list[float] = []
    log_b: list[float] = []
    torch.manual_seed(0)
    segpgd_trajectory(model, x, y, 0.1, 2, 4, random_start=True, loss_log=log_a)
    torch.manual_seed(0)
    segpgd_trajectory(model, x, y, 0.1, 2, 4, random_start=True, loss_log=log_b)
    assert log_a == log_b  # a fresh counter per call: identical schedules


def test_segpgd_attack_stays_in_ball_and_bounds() -> None:
    model = _ToySegmentationModel()
    x, y = _toy_case()
    eps = 0.1
    x_adv = segpgd_attack(model, x, y, eps, 2, n_steps=5)
    assert x_adv.shape == x.shape
    assert float((x_adv - x).abs().max()) <= eps + 1e-6
    assert float(x_adv.min()) >= float(x.min()) - 1e-6
    assert float(x_adv.max()) <= float(x.max()) + 1e-6


def test_segpgd_eps_zero_is_noop() -> None:
    model = _ToySegmentationModel()
    x, y = _toy_case()
    assert torch.equal(segpgd_attack(model, x, y, 0.0, 2, n_steps=5), x)


# --- CosPGD -----------------------------------------------------------------


def test_cospgd_loss_matches_official_scaling() -> None:
    torch.manual_seed(4)
    logits = torch.randn(1, 2, 2, 3, 3)
    y = torch.randint(0, 2, (1, 1, 2, 3, 3)).float()
    labels = y.squeeze(1).long()
    per_voxel = F.cross_entropy(logits, labels, reduction="none")
    onehot = F.one_hot(labels, 2).movedim(-1, 1).float()
    cossim = F.cosine_similarity(F.softmax(logits, dim=1), onehot, dim=1)
    expected = (cossim * per_voxel).mean()
    assert torch.allclose(cospgd_loss(logits, y, 2), expected)


def test_cospgd_scale_is_detached() -> None:
    torch.manual_seed(5)
    logits = torch.randn(1, 2, 2, 3, 3, requires_grad=True)
    y = torch.randint(0, 2, (1, 1, 2, 3, 3)).float()
    labels = y.squeeze(1).long()
    cospgd_loss(logits, y, 2).backward()
    grad_cos = logits.grad.clone()

    logits2 = logits.detach().clone().requires_grad_(True)
    per_voxel = F.cross_entropy(logits2, labels, reduction="none")
    onehot = F.one_hot(labels, 2).movedim(-1, 1).float()
    cossim = F.cosine_similarity(F.softmax(logits2, dim=1), onehot, dim=1).detach()
    (cossim * per_voxel).mean().backward()
    assert torch.allclose(grad_cos, logits2.grad)


def test_cospgd_loss_is_at_most_cross_entropy() -> None:
    torch.manual_seed(6)
    logits = torch.randn(1, 2, 2, 3, 3)
    y = torch.randint(0, 2, (1, 1, 2, 3, 3)).float()
    assert float(cospgd_loss(logits, y, 2)) <= float(_mean_ce(logits, y)) + 1e-6
    assert float(cospgd_loss(logits, y, 2)) >= 0.0


def test_cospgd_loss_excludes_ignore_voxels() -> None:
    torch.manual_seed(7)
    logits = torch.randn(1, 2, 2, 3, 3)
    y = torch.randint(0, 2, (1, 1, 2, 3, 3)).float()
    y[0, 0, 0, 0, :] = -1.0
    got = cospgd_loss(logits, y, 2)
    assert torch.isfinite(got)
    labels = y.squeeze(1).long()
    valid = labels != -1
    safe = torch.where(valid, labels, torch.zeros_like(labels))
    per_voxel = F.cross_entropy(logits, safe, reduction="none")
    onehot = F.one_hot(safe, 2).movedim(-1, 1).float()
    cossim = F.cosine_similarity(F.softmax(logits, dim=1), onehot, dim=1)
    expected = (cossim * per_voxel)[valid].sum() / valid.sum()
    assert torch.allclose(got, expected)


def test_cospgd_attack_stays_in_ball_and_bounds() -> None:
    model = _ToySegmentationModel()
    x, y = _toy_case()
    eps = 0.1
    x_adv = cospgd_attack(model, x, y, eps, 2, n_steps=5)
    assert x_adv.shape == x.shape
    assert float((x_adv - x).abs().max()) <= eps + 1e-6
    assert float(x_adv.min()) >= float(x.min()) - 1e-6
    assert float(x_adv.max()) <= float(x.max()) + 1e-6


def test_cospgd_eps_zero_is_noop() -> None:
    model = _ToySegmentationModel()
    x, y = _toy_case()
    assert torch.equal(cospgd_attack(model, x, y, 0.0, 2, n_steps=5), x)


def test_loss_variants_raise_on_zero_steps() -> None:
    model = _ToySegmentationModel()
    x, y = _toy_case()
    with pytest.raises(ValueError):
        segpgd_attack(model, x, y, 0.1, 2, n_steps=0)
    with pytest.raises(ValueError):
        cospgd_attack(model, x, y, 0.1, 2, n_steps=0)
