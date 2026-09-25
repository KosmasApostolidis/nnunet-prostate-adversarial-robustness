"""Contract tests for the DAG and SEA arms.

DAG: the active-set objective (Eq. 1 over still-correct voxels), the L-infinity
normalised step, the early stop once every voxel is fooled, and the ball.
SEA: the three losses against hand computations from the official source,
the worst-Dice selection, and the ball.
"""

from __future__ import annotations

import pytest
import torch
import torch.nn.functional as F

from mri_prostate_seg.attacks.dag import (
    adversarial_labels,
    dag_attack,
    dag_loss,
    dag_trajectory,
)
from mri_prostate_seg.attacks.sea import (
    SEA_LOSSES,
    foreground_dice,
    js_loss,
    masked_ce_balanced_loss,
    masked_ce_loss,
    sea_trajectory,
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


def _random_logits(seed: int, num_classes: int = 2):
    torch.manual_seed(seed)
    logits = torch.randn(1, num_classes, 2, 3, 3)
    y = torch.randint(0, num_classes, (1, 1, 2, 3, 3)).float()
    return logits, y


# --- DAG --------------------------------------------------------------------


def test_adversarial_labels_binary_is_the_swap() -> None:
    labels = torch.tensor([[[0, 1, -1]]])
    adv = adversarial_labels(labels, 2)
    assert adv.tolist() == [[[1, 0, -1]]]


def test_adversarial_labels_multiclass_is_a_derangement() -> None:
    torch.manual_seed(0)
    labels = torch.arange(5).reshape(1, 1, 5)
    adv = adversarial_labels(labels, 5)
    assert bool((adv != labels).all())
    assert sorted(adv.flatten().tolist()) == [0, 1, 2, 3, 4]


def test_dag_loss_is_logit_gap_over_active_set() -> None:
    logits, y = _random_logits(1)
    labels = y.squeeze(1).long()
    adv = adversarial_labels(labels, 2)
    active = logits.argmax(dim=1) == labels
    f_true = logits.gather(1, labels.unsqueeze(1)).squeeze(1)
    f_adv = logits.gather(1, adv.unsqueeze(1)).squeeze(1)
    expected = (f_adv - f_true)[active].sum()
    got, n_active = dag_loss(logits, labels, adv)
    assert torch.allclose(got, expected)
    assert n_active == int(active.sum())


def test_dag_loss_ignores_ignore_voxels() -> None:
    logits, y = _random_logits(2)
    y[0, 0, 0, 0, :] = -1.0
    labels = y.squeeze(1).long()
    adv = adversarial_labels(labels, 2)
    got, n_active = dag_loss(logits, labels, adv)
    assert torch.isfinite(got)
    valid = labels != -1
    assert n_active == int(((logits.argmax(dim=1) == labels) & valid).sum())


def test_dag_step_has_linf_norm_gamma() -> None:
    model = _ToySegmentationModel()
    x, y = _toy_case()
    eps, k = 0.2, 4
    x_adv, _ = dag_trajectory(model, x, y, eps, 2, n_steps=1, step_size=eps / k)
    # One unprojected step: the largest voxel move is exactly gamma.
    assert float((x_adv - x).abs().max()) == pytest.approx(eps / k, rel=1e-4)


def test_dag_stops_when_every_voxel_is_fooled() -> None:
    class _AlwaysWrong(torch.nn.Module):
        def forward(self, x: torch.Tensor) -> torch.Tensor:
            # Predict background everywhere except where the label is background.
            logits = torch.zeros(x.shape[0], 2, *x.shape[2:])
            logits[:, 1] = 1.0  # class 1 everywhere
            return logits + 0.0 * x.sum()

    x = torch.zeros(1, 1, 2, 2, 2)
    y = torch.zeros(1, 1, 2, 2, 2)  # all background -> all wrong from the start
    log: list[float] = []
    x_adv, snaps = dag_trajectory(
        _AlwaysWrong(), x, y, 0.1, 2, n_steps=5, snapshot_eps=[0.05, 0.1], loss_log=log
    )
    assert len(log) == 1  # evaluated once, then stopped
    assert torch.equal(x_adv, x)
    assert set(snaps) == {0.05, 0.1}


def test_dag_attack_stays_in_ball_and_bounds() -> None:
    model = _ToySegmentationModel()
    x, y = _toy_case()
    eps = 0.1
    x_adv = dag_attack(model, x, y, eps, 2, n_steps=5)
    assert x_adv.shape == x.shape
    assert float((x_adv - x).abs().max()) <= eps + 1e-6
    assert float(x_adv.min()) >= float(x.min()) - 1e-6
    assert float(x_adv.max()) <= float(x.max()) + 1e-6


def test_dag_eps_zero_and_zero_steps() -> None:
    model = _ToySegmentationModel()
    x, y = _toy_case()
    assert torch.equal(dag_attack(model, x, y, 0.0, 2, n_steps=5), x)
    with pytest.raises(ValueError):
        dag_attack(model, x, y, 0.1, 2, n_steps=0)


# --- SEA losses --------------------------------------------------------------


def test_masked_ce_matches_official_definition() -> None:
    logits, y = _random_logits(3)
    labels = y.squeeze(1).long()
    per_voxel = F.cross_entropy(logits, labels, reduction="none")
    mask = (logits.argmax(dim=1) == labels).float()
    expected = (mask * per_voxel).mean()
    assert torch.allclose(masked_ce_loss(logits, y, 2), expected)


def test_masked_ce_balanced_weights_classes_inversely_to_size() -> None:
    logits, y = _random_logits(4)
    labels = y.squeeze(1).long()
    counts = torch.bincount(labels.flatten(), minlength=2).float()
    w = 1.0 / counts
    w = w / w.sum()
    per_voxel = F.cross_entropy(logits, labels, reduction="none", weight=w)
    mask = (logits.argmax(dim=1) == labels).float()
    expected = (mask * per_voxel).mean()
    assert torch.allclose(masked_ce_balanced_loss(logits, y, 2), expected)


def test_js_loss_matches_official_definition() -> None:
    logits, y = _random_logits(5)
    labels = y.squeeze(1).long()
    p = F.softmax(logits, dim=1)
    q = F.one_hot(labels, 2).movedim(-1, 1).float()
    m = (p + q) / 2
    ref = (
        F.kl_div(m.log(), p, reduction="none") + F.kl_div(m.log(), q, reduction="none")
    ) / 2
    expected = ref.sum(dim=1).mean()
    assert torch.allclose(js_loss(logits, y, 2), expected, atol=1e-6)


def test_js_loss_is_zero_for_a_perfect_prediction() -> None:
    labels = torch.tensor([[[[0, 1]]]])
    y = labels.unsqueeze(1).float()
    logits = torch.full((1, 2, 1, 1, 2), -50.0)
    logits[0, 0, 0, 0, 0] = 50.0
    logits[0, 1, 0, 0, 1] = 50.0
    assert float(js_loss(logits, y, 2)) == pytest.approx(0.0, abs=1e-6)


def test_sea_losses_exclude_ignore_voxels() -> None:
    logits, y = _random_logits(6)
    y[0, 0, 0, 0, :] = -1.0
    for fn in (masked_ce_loss, masked_ce_balanced_loss, js_loss):
        assert torch.isfinite(fn(logits, y, 2))


def test_foreground_dice_hard_argmax() -> None:
    logits = torch.zeros(1, 2, 1, 1, 4)
    logits[0, 1, 0, 0, :2] = 1.0  # predict fg on voxels 0,1
    y = torch.tensor([[[[[0.0, 1.0, 1.0, 0.0]]]]])  # fg on voxels 1,2
    assert foreground_dice(logits, y, 2) == pytest.approx(0.5)


def test_sea_trajectory_picks_lowest_dice_per_budget() -> None:
    model = _ToySegmentationModel()
    x, y = _toy_case()
    chosen: dict[float, str] = {}
    torch.manual_seed(0)
    x_adv, snaps = sea_trajectory(
        model, x, y, 0.1, 2, n_steps=3, snapshot_eps=[0.0, 0.05, 0.1], chosen_log=chosen
    )
    assert set(snaps) == {0.0, 0.05, 0.1}
    assert set(chosen) == {0.05, 0.1}
    assert all(name in SEA_LOSSES for name in chosen.values())
    assert torch.equal(snaps[0.1], x_adv)
    for e in (0.05, 0.1):
        assert float((snaps[e] - x).abs().max()) <= e + 1e-6


def test_sea_rejects_unknown_loss_and_zero_steps() -> None:
    model = _ToySegmentationModel()
    x, y = _toy_case()
    with pytest.raises(ValueError):
        sea_trajectory(model, x, y, 0.1, 2, n_steps=3, losses=("mce", "nope"))
    with pytest.raises(ValueError):
        sea_trajectory(model, x, y, 0.1, 2, n_steps=0)
