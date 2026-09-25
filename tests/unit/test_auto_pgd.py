"""Contract tests for the Auto-PGD port.

The reference is vendored at docs/superpowers/specs/vendor/autopgd_base.py.
These tests pin the four properties that make the port a faithful Auto-PGD
rather than a differently-tuned PGD: the published checkpoint schedule, the
step size that only ever halves from 2*eps, the Linf and intensity bounds,
and the "return the highest-loss iterate" contract.
"""

from __future__ import annotations

import torch

from mri_prostate_seg.attacks.auto_pgd import _checkpoint_steps, auto_pgd_attack


class _ToySegmentationModel(torch.nn.Module):
    """Two-logit head whose foreground logit is a fixed linear map of the input."""

    def __init__(self) -> None:
        super().__init__()
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


def test_checkpoint_schedule_matches_the_published_constants() -> None:
    # n_iter_2 = 22, size_decr = 3, n_iter_min = 6 at k=100.
    assert _checkpoint_steps(100) == [22, 41, 57, 70, 80, 87, 93, 99]
    # At k=20 the schedule degenerates: n_iter_2 = 4, size_decr = 1,
    # n_iter_min = 1, so it fires every iteration from 10 onward.
    assert _checkpoint_steps(20) == [
        4,
        7,
        9,
        10,
        11,
        12,
        13,
        14,
        15,
        16,
        17,
        18,
        19,
        20,
    ]


def test_step_size_starts_at_two_eps_and_never_grows() -> None:
    model = _ToySegmentationModel()
    x, y = _toy_case()
    sizes: list[float] = []
    auto_pgd_attack(
        model,
        x,
        y,
        eps=0.1,
        num_classes=2,
        loss_fn=_dice_ce,
        n_steps=20,
        random_start=False,
        step_size_log=sizes,
    )
    assert sizes[0] == 0.2
    assert all(later <= earlier for earlier, later in zip(sizes, sizes[1:]))
    assert all(s in {0.2 / 2**j for j in range(len(sizes))} for s in sizes)


def test_returned_iterate_respects_linf_and_intensity_bounds() -> None:
    model = _ToySegmentationModel()
    x, y = _toy_case()
    eps = 0.1
    x_adv = auto_pgd_attack(
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
    x_adv = auto_pgd_attack(
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
    x_adv = auto_pgd_attack(
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
    from mri_prostate_seg.attacks.auto_pgd import auto_pgd_trajectory

    model = _ToySegmentationModel()
    x, y = _toy_case()
    snapshot_eps = [0.0, 0.02, 0.06, 0.1]
    x_adv, snaps = auto_pgd_trajectory(
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
    # The max-eps snapshot is the attack's own iterate, not a re-projection.
    assert torch.equal(snaps[0.1], x_adv)


def test_trajectory_at_zero_epsilon_returns_the_clean_volume() -> None:
    from mri_prostate_seg.attacks.auto_pgd import auto_pgd_trajectory

    model = _ToySegmentationModel()
    x, y = _toy_case()
    x_adv, snaps = auto_pgd_trajectory(
        model,
        x,
        y,
        max_eps=0.0,
        num_classes=2,
        loss_fn=_dice_ce,
        n_steps=20,
        snapshot_eps=[0.0],
    )
    assert torch.equal(x_adv, x)
    assert torch.equal(snaps[0.0], x)
