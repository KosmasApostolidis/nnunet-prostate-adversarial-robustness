"""Contract tests for Auto-PGD+.

These pin the five properties that separate this arm from the Auto-PGD port it
repairs: a geometric backbone that reaches the fine regime inside any budget, a
multiplier that can only hold the step at or below that backbone, backtracking
that reacts to a losing step immediately, restart hygiene, and an oscillation
window that never falls to a single sample.
"""

from __future__ import annotations

import math

import torch

from mri_prostate_seg.attacks.auto_pgd_plus import (
    MIN_OSCILLATION_WINDOW,
    MULTIPLIER_BOUNDS,
    _backbone_step,
    _checkpoint_steps,
    auto_pgd_plus_attack,
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


def _dice_ce(logits: torch.Tensor, target: torch.Tensor, num_classes: int):
    t = target.squeeze(1).long()
    ce = torch.nn.functional.cross_entropy(logits, t)
    probs = torch.softmax(logits, dim=1)
    p_fg = probs[:, 1:2]
    t_fg = (t == 1).float().unsqueeze(1)
    inter = (p_fg * t_fg).sum()
    denom = p_fg.sum() + t_fg.sum()
    return ce + 1.0 - (2.0 * inter + 1e-5) / (denom + 1e-5)


def test_backbone_runs_from_two_eps_to_the_floor() -> None:
    eps, k = 0.1, 20
    assert _backbone_step(1, k, eps) == 2.0 * eps
    assert abs(_backbone_step(k, k, eps) - eps / k) < 1e-12
    values = [_backbone_step(i, k, eps) for i in range(1, k + 1)]
    assert all(b < a for a, b in zip(values, values[1:]))


def test_backbone_is_geometric_not_linear() -> None:
    """Equal ratio between consecutive steps is what buys equal time per octave."""
    eps, k = 0.1, 20
    values = [_backbone_step(i, k, eps) for i in range(1, k + 1)]
    ratios = [b / a for a, b in zip(values, values[1:])]
    assert max(ratios) - min(ratios) < 1e-12


def test_backbone_reaches_the_fine_regime_at_a_short_budget() -> None:
    """The starvation fix: most of a k=20 run must use a step below eps.

    Measured Auto-PGD at k=20 spends only 43% of its iterations below eps and
    realises 3-5 of the 6 halvings it needs.
    """
    eps, k = 0.1, 20
    values = [_backbone_step(i, k, eps) for i in range(1, k + 1)]
    below = [v for v in values if v < eps]
    assert len(below) / k > 0.75
    octaves = {int(math.floor(math.log2(v / (eps / k)))) for v in values}
    assert len(octaves) >= 6


def test_backbone_is_budget_invariant() -> None:
    """k=20 and k=100 must both span the range, unlike Auto-PGD's constants."""
    for k in (20, 100):
        assert _backbone_step(1, k, 0.1) == 0.2
        assert abs(_backbone_step(k, k, 0.1) - 0.1 / k) < 1e-12


def test_oscillation_window_never_falls_to_one_sample() -> None:
    """At k=20 the reference's window reaches 1 from iteration 10 onward."""
    for k in (20, 100):
        steps = _checkpoint_steps(k)
        intervals = [b - a for a, b in zip([0] + steps, steps)]
        assert min(intervals) >= MIN_OSCILLATION_WINDOW


def test_multiplier_is_bounded_at_or_below_the_backbone() -> None:
    """apgd_updated's ratchets to 4.0 and pins the step at the 2*eps cap."""
    assert MULTIPLIER_BOUNDS[1] == 1.0
    assert 0.0 < MULTIPLIER_BOUNDS[0] < 1.0


def test_step_sizes_stay_within_the_floor_and_cap() -> None:
    model = _ToySegmentationModel()
    x, y = _toy_case()
    eps, k = 0.1, 20
    log: list[float] = []
    auto_pgd_plus_attack(
        model, x, y, eps, 2, _dice_ce, n_steps=k, random_start=False, step_size_log=log
    )
    assert len(log) == k
    assert all(eps / k - 1e-12 <= s <= 2.0 * eps + 1e-12 for s in log)


def test_perturbation_respects_the_ball_and_the_intensity_bounds() -> None:
    model = _ToySegmentationModel()
    x, y = _toy_case()
    eps = 0.05
    adv = auto_pgd_plus_attack(
        model, x, y, eps, 2, _dice_ce, n_steps=10, random_start=True
    )
    assert torch.max(torch.abs(adv - x)).item() <= eps + 1e-6
    assert adv.min().item() >= x.min().item() - 1e-6
    assert adv.max().item() <= x.max().item() + 1e-6


def test_returns_the_best_iterate_not_the_last() -> None:
    model = _ToySegmentationModel()
    x, y = _toy_case()
    losses: list[float] = []
    adv = auto_pgd_plus_attack(
        model, x, y, 0.1, 2, _dice_ce, n_steps=15, random_start=False, loss_log=losses
    )
    with torch.no_grad():
        returned = float(_dice_ce(model(adv), y, 2))
    assert returned >= max(losses) - 1e-5


def test_zero_epsilon_is_a_no_op() -> None:
    model = _ToySegmentationModel()
    x, y = _toy_case()
    adv = auto_pgd_plus_attack(model, x, y, 0.0, 2, _dice_ce)
    assert torch.equal(adv, x)


def test_rejects_a_non_positive_budget() -> None:
    model = _ToySegmentationModel()
    x, y = _toy_case()
    try:
        auto_pgd_plus_attack(model, x, y, 0.1, 2, _dice_ce, n_steps=0)
    except ValueError:
        return
    raise AssertionError("n_steps=0 must raise")
