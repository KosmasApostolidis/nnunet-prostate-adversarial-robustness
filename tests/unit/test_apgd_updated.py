"""Contract tests for the updated adaptive-step PGD (the improved-step arm).

These pin the three properties that make this arm different from the two attacks
it sits between: a step that spans the full useful range rather than a narrow
band, a checkpoint schedule that does not change shape with the budget, and a
bidirectional multiplier bounded on both sides.
"""

from __future__ import annotations

import torch

from mri_prostate_seg.attacks.apgd_updated import (
    _backbone_step,
    _checkpoint_steps,
    apgd_updated_attack,
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


def test_backbone_spans_from_two_eps_down_to_the_floor() -> None:
    """The whole point of the backbone: sweep the range both rivals miss."""
    eps, k = 0.1, 20
    assert _backbone_step(0.0, eps, k) == 2.0 * eps  # Auto-PGD's step
    assert abs(_backbone_step(1.0, eps, k) - eps / k) < 1e-12  # the campaign floor
    mid = _backbone_step(0.5, eps, k)
    assert eps / k < mid < 2.0 * eps
    # Monotonically decreasing across the run.
    values = [_backbone_step(i / 20, eps, k) for i in range(21)]
    assert all(b <= a + 1e-12 for a, b in zip(values, values[1:]))


def test_checkpoint_schedule_is_budget_invariant() -> None:
    """Same fractions at any budget, unlike Auto-PGD's k=100-tuned constants."""
    for n_steps in (20, 100):
        steps = _checkpoint_steps(n_steps)
        fractions = [s / n_steps for s in steps]
        assert all(0.0 < f < 1.0 for f in fractions)
        assert fractions == sorted(fractions)
        # Ten equal blocks means nine interior checkpoints at both budgets.
        assert len(steps) == 9
    assert _checkpoint_steps(20) == [2, 4, 6, 8, 10, 12, 14, 16, 18]
    assert _checkpoint_steps(100) == [10, 20, 30, 40, 50, 60, 70, 80, 90]


def test_step_stays_between_the_floor_and_two_eps() -> None:
    model = _ToySegmentationModel()
    x, y = _toy_case()
    eps, k = 0.1, 20
    sizes: list[float] = []
    apgd_updated_attack(
        model,
        x,
        y,
        eps=eps,
        num_classes=2,
        loss_fn=_dice_ce,
        n_steps=k,
        random_start=False,
        step_size_log=sizes,
    )
    assert len(sizes) == k
    assert max(sizes) <= 2.0 * eps + 1e-12
    assert min(sizes) >= eps / k - 1e-12
    # It must actually use a large step early on -- that is what the campaign
    # attack cannot do, and the reason it is 40x too fine at k=20.
    assert max(sizes) > 10.0 * (2.0 * eps / k)


def test_returned_iterate_respects_linf_and_intensity_bounds() -> None:
    model = _ToySegmentationModel()
    x, y = _toy_case()
    eps = 0.1
    x_adv = apgd_updated_attack(
        model,
        x,
        y,
        eps=eps,
        num_classes=2,
        loss_fn=_dice_ce,
        n_steps=20,
    )
    assert torch.max(torch.abs(x_adv - x)).item() <= eps + 1e-6
    assert x_adv.min().item() >= x.min().item() - 1e-6
    assert x_adv.max().item() <= x.max().item() + 1e-6


def test_returns_the_highest_loss_iterate_visited() -> None:
    model = _ToySegmentationModel()
    x, y = _toy_case()
    losses: list[float] = []
    x_adv = apgd_updated_attack(
        model,
        x,
        y,
        eps=0.1,
        num_classes=2,
        loss_fn=_dice_ce,
        n_steps=20,
        loss_log=losses,
    )
    with torch.no_grad():
        returned = float(_dice_ce(model(x_adv), y, 2))
    assert returned >= max(losses) - 1e-5


def test_zero_epsilon_is_a_no_op() -> None:
    model = _ToySegmentationModel()
    x, y = _toy_case()
    x_adv = apgd_updated_attack(
        model,
        x,
        y,
        eps=0.0,
        num_classes=2,
        loss_fn=_dice_ce,
        n_steps=20,
    )
    assert torch.equal(x_adv, x)


def test_trajectory_snapshots_respect_each_epsilon() -> None:
    from mri_prostate_seg.attacks.apgd_updated import apgd_updated_trajectory

    model = _ToySegmentationModel()
    x, y = _toy_case()
    snapshot_eps = [0.0, 0.02, 0.06, 0.1]
    x_adv, snaps = apgd_updated_trajectory(
        model,
        x,
        y,
        max_eps=0.1,
        num_classes=2,
        loss_fn=_dice_ce,
        n_steps=20,
        snapshot_eps=snapshot_eps,
    )
    assert set(snaps) == set(snapshot_eps)
    for eps, snap in snaps.items():
        assert torch.max(torch.abs(snap - x)).item() <= eps + 1e-6
    assert torch.equal(snaps[0.0], x)
    assert torch.equal(snaps[0.1], x_adv)
