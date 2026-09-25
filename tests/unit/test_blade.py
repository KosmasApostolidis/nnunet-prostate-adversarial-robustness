"""Contract tests for BLADE: the epsilon ladder and its two new objectives.

The ladder: one snapshot per rung, each natively optimised inside its own ball,
each rung warm-started from the previous one. The objectives: the sign-flipped
boundary loss of Kervadec et al. and the frontier-margin loss, both checked
against hand computations.
"""

from __future__ import annotations

import numpy as np
import pytest
import torch

from mri_prostate_seg.attacks.auto_pgd import auto_pgd_attack
from mri_prostate_seg.attacks.blade_legacy import (
    BLADE_LOSSES,
    blade_trajectory,
    boundary_weighted_loss,
    frontier_margin_loss,
    ladder_trajectory,
    signed_distance_map,
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
    import torch.nn.functional as F

    t = target.squeeze(1).long()
    ce = F.cross_entropy(logits, t.clamp_min(0))
    p = F.softmax(logits, dim=1)[:, 1:2]
    tf = (t == 1).float().unsqueeze(1)
    inter = (p * tf).sum()
    return 1.0 - (2 * inter + 1e-5) / (p.sum() + tf.sum() + 1e-5) + ce


# --- auto_pgd x_init: additive, default-preserving ---------------------------


def test_auto_pgd_without_x_init_is_unchanged() -> None:
    model = _ToySegmentationModel()
    x, y = _toy_case()
    torch.manual_seed(7)
    a = auto_pgd_attack(model, x, y, 0.1, 2, _dice_ce, n_steps=3)
    torch.manual_seed(7)
    b = auto_pgd_attack(model, x, y, 0.1, 2, _dice_ce, n_steps=3, x_init=None)
    assert torch.equal(a, b)


def test_auto_pgd_x_init_replaces_the_random_start() -> None:
    model = _ToySegmentationModel()
    x, y = _toy_case()
    warm = torch.clamp(x + 0.05, x.min(), x.max())
    torch.manual_seed(7)
    from_warm = auto_pgd_attack(
        model, x, y, 0.1, 2, _dice_ce, n_steps=3, x_init=warm
    )
    torch.manual_seed(7)
    from_random = auto_pgd_attack(model, x, y, 0.1, 2, _dice_ce, n_steps=3)
    assert not torch.equal(from_warm, from_random)
    assert float((from_warm - x).abs().max()) <= 0.1 + 1e-6


def test_auto_pgd_x_init_is_projected_into_the_ball() -> None:
    model = _ToySegmentationModel()
    x, y = _toy_case()
    # A warm point outside the rung's ball must be pulled back onto it.
    warm = x + 0.5
    out = auto_pgd_attack(model, x, y, 0.05, 2, _dice_ce, n_steps=2, x_init=warm)
    assert float((out - x).abs().max()) <= 0.05 + 1e-6


# --- signed distance map -----------------------------------------------------


def test_signed_distance_map_is_negative_inside_positive_outside() -> None:
    y = torch.zeros(1, 1, 1, 5, 5)
    y[0, 0, 0, 2, 2] = 1.0
    phi = signed_distance_map(y)
    assert phi.shape == (1, 1, 1, 5, 5)
    assert float(phi[0, 0, 0, 2, 2]) < 0.0  # inside the object
    assert float(phi[0, 0, 0, 0, 0]) > 0.0  # far outside
    # Farther out means larger phi.
    assert float(phi[0, 0, 0, 0, 0]) > float(phi[0, 0, 0, 1, 2])


def test_signed_distance_map_all_background_is_finite() -> None:
    y = torch.zeros(1, 1, 1, 4, 4)
    phi = signed_distance_map(y)
    assert bool(torch.isfinite(phi).all())


def test_signed_distance_map_excludes_ignore_voxels() -> None:
    y = torch.zeros(1, 1, 1, 4, 4)
    y[0, 0, 0, 1, 1] = 1.0
    y[0, 0, 0, 3, :] = -1.0
    phi = signed_distance_map(y)
    assert float(phi[0, 0, 0, 3, 0]) == 0.0  # ignored voxels carry no weight


# --- boundary-weighted loss --------------------------------------------------


def test_boundary_loss_matches_hand_computation() -> None:
    import torch.nn.functional as F

    torch.manual_seed(3)
    logits = torch.randn(1, 2, 1, 4, 4)
    y = torch.zeros(1, 1, 1, 4, 4)
    y[0, 0, 0, 1:3, 1:3] = 1.0
    phi = signed_distance_map(y)
    p_fg = F.softmax(logits, dim=1)[:, 1:2]
    expected = (phi * p_fg).sum() / phi.abs().sum().clamp_min(1e-12)
    assert torch.allclose(boundary_weighted_loss(logits, y, 2), expected)


def test_boundary_loss_rewards_distant_false_positives() -> None:
    y = torch.zeros(1, 1, 1, 5, 5)
    y[0, 0, 0, 2, 2] = 1.0
    near = torch.full((1, 2, 1, 5, 5), -10.0)
    near[0, 1, 0, 2, 2] = 10.0  # perfect prediction
    far = torch.full((1, 2, 1, 5, 5), -10.0)
    far[0, 1, 0, 0, 0] = 10.0  # foreground put in the far corner instead
    assert float(boundary_weighted_loss(far, y, 2)) > float(
        boundary_weighted_loss(near, y, 2)
    )


# --- frontier margin loss ----------------------------------------------------


def test_frontier_margin_targets_the_smallest_positive_margins() -> None:
    # Four correct voxels with margins 4, 3, 2, 1; k_frac picks the smallest two.
    logits = torch.zeros(1, 2, 1, 1, 4)
    logits[0, 1, 0, 0, :] = torch.tensor([4.0, 3.0, 2.0, 1.0])
    y = torch.ones(1, 1, 1, 1, 4)
    got = frontier_margin_loss(logits, y, 2, k_frac=0.5)
    assert float(got) == pytest.approx(-(2.0 + 1.0) / 2.0)


def test_frontier_margin_ignores_already_wrong_voxels() -> None:
    logits = torch.zeros(1, 2, 1, 1, 3)
    logits[0, 1, 0, 0, :] = torch.tensor([5.0, 2.0, -3.0])  # last one already wrong
    y = torch.ones(1, 1, 1, 1, 3)
    got = frontier_margin_loss(logits, y, 2, k_frac=1.0)
    assert float(got) == pytest.approx(-(5.0 + 2.0) / 2.0)


def test_frontier_margin_ignores_ignore_label() -> None:
    logits = torch.zeros(1, 2, 1, 1, 3)
    logits[0, 1, 0, 0, :] = torch.tensor([5.0, 2.0, 1.0])
    y = torch.ones(1, 1, 1, 1, 3)
    y[0, 0, 0, 0, 2] = -1.0
    got = frontier_margin_loss(logits, y, 2, k_frac=1.0)
    assert float(got) == pytest.approx(-(5.0 + 2.0) / 2.0)


def test_frontier_margin_is_zero_when_everything_is_already_wrong() -> None:
    logits = torch.zeros(1, 2, 1, 1, 2)
    logits[0, 1, 0, 0, :] = torch.tensor([-1.0, -2.0])
    y = torch.ones(1, 1, 1, 1, 2)
    assert float(frontier_margin_loss(logits, y, 2)) == pytest.approx(0.0)


def test_frontier_margin_keeps_a_gradient() -> None:
    base = torch.zeros(1, 2, 1, 1, 4)
    base[0, 1, 0, 0, :] = torch.tensor([4.0, 3.0, 2.0, 1.0])
    logits = base.clone().requires_grad_(True)
    y = torch.ones(1, 1, 1, 1, 4)
    frontier_margin_loss(logits, y, 2, k_frac=0.5).backward()
    assert logits.grad is not None
    assert float(logits.grad.abs().sum()) > 0.0


# --- the ladder --------------------------------------------------------------


def test_ladder_emits_one_snapshot_per_rung_inside_its_own_ball() -> None:
    model = _ToySegmentationModel()
    x, y = _toy_case()
    rungs = [0.02, 0.05, 0.1]
    torch.manual_seed(0)
    x_adv, snaps = ladder_trajectory(
        model, x, y, 0.1, 2, _dice_ce, n_steps=2, snapshot_eps=rungs
    )
    assert set(snaps) == set(rungs)
    for e in rungs:
        assert float((snaps[e] - x).abs().max()) <= e + 1e-6
    assert torch.equal(snaps[0.1], x_adv)


def test_ladder_warm_starts_each_rung_from_the_previous_one() -> None:
    model = _ToySegmentationModel()
    x, y = _toy_case()
    seen: list[torch.Tensor] = []
    torch.manual_seed(0)
    _, snaps = ladder_trajectory(
        model, x, y, 0.1, 2, _dice_ce, n_steps=2,
        snapshot_eps=[0.02, 0.1], init_log=seen,
    )
    # Rung 1 starts from a random point; rung 2 starts from rung 1's output.
    assert len(seen) == 2
    assert seen[0] is None
    assert torch.equal(seen[1], snaps[0.02])


def test_ladder_always_includes_max_eps_as_a_rung() -> None:
    model = _ToySegmentationModel()
    x, y = _toy_case()
    torch.manual_seed(0)
    _, snaps = ladder_trajectory(
        model, x, y, 0.06, 2, _dice_ce, n_steps=2, snapshot_eps=[0.02]
    )
    assert 0.06 in snaps


def test_ladder_zero_eps_and_zero_steps() -> None:
    model = _ToySegmentationModel()
    x, y = _toy_case()
    x_adv, snaps = ladder_trajectory(
        model, x, y, 0.0, 2, _dice_ce, n_steps=2, snapshot_eps=[0.0]
    )
    assert torch.equal(x_adv, x)
    assert set(snaps) == {0.0}
    assert torch.equal(snaps[0.0], x)
    with pytest.raises(ValueError):
        ladder_trajectory(model, x, y, 0.1, 2, _dice_ce, n_steps=0)


def test_ladder_snapshot_zero_is_the_clean_volume() -> None:
    model = _ToySegmentationModel()
    x, y = _toy_case()
    torch.manual_seed(0)
    _, snaps = ladder_trajectory(
        model, x, y, 0.1, 2, _dice_ce, n_steps=2, snapshot_eps=[0.0, 0.1]
    )
    assert torch.equal(snaps[0.0], x)


# --- the ensemble ------------------------------------------------------------


def test_blade_picks_the_worst_dice_candidate_per_rung() -> None:
    model = _ToySegmentationModel()
    x, y = _toy_case()
    chosen: dict[float, str] = {}
    torch.manual_seed(0)
    x_adv, snaps = blade_trajectory(
        model, x, y, 0.1, 2, _dice_ce, n_steps=2,
        snapshot_eps=[0.0, 0.05, 0.1], chosen_log=chosen,
    )
    assert set(snaps) == {0.0, 0.05, 0.1}
    assert set(chosen) == {0.05, 0.1}
    assert all(name in BLADE_LOSSES for name in chosen.values())
    assert torch.equal(snaps[0.1], x_adv)
    for e in (0.05, 0.1):
        assert float((snaps[e] - x).abs().max()) <= e + 1e-6


def test_blade_rejects_unknown_loss_and_zero_steps() -> None:
    model = _ToySegmentationModel()
    x, y = _toy_case()
    with pytest.raises(ValueError):
        blade_trajectory(
            model, x, y, 0.1, 2, _dice_ce, n_steps=2, losses=("dice_ce", "nope")
        )
    with pytest.raises(ValueError):
        blade_trajectory(model, x, y, 0.1, 2, _dice_ce, n_steps=0)


def test_blade_respects_the_intensity_bounds() -> None:
    model = _ToySegmentationModel()
    x, y = _toy_case()
    torch.manual_seed(0)
    x_adv, _ = blade_trajectory(
        model, x, y, 0.1, 2, _dice_ce, n_steps=2, snapshot_eps=[0.1]
    )
    assert float(x_adv.min()) >= float(x.min()) - 1e-6
    assert float(x_adv.max()) <= float(x.max()) + 1e-6


# --- per-objective marginals -------------------------------------------------


@pytest.mark.parametrize("objective", ["dice_ce", "boundary", "frontier"])
def test_single_objective_ladder_needs_no_adjudication(objective: str) -> None:
    """Each marginal arm runs one ladder and reports it unchanged."""
    model = _ToySegmentationModel()
    x, y = _toy_case()
    chosen: dict[float, str] = {}
    torch.manual_seed(0)
    x_adv, snaps = blade_trajectory(
        model, x, y, 0.1, 2, _dice_ce, n_steps=2,
        snapshot_eps=[0.04, 0.1], losses=(objective,), chosen_log=chosen,
    )
    assert chosen == {0.04: objective, 0.1: objective}
    assert set(snaps) == {0.04, 0.1}
    for e in (0.04, 0.1):
        assert float((snaps[e] - x).abs().max()) <= e + 1e-6
    assert torch.equal(snaps[0.1], x_adv)


def test_marginals_and_ensemble_share_the_same_candidate_pool() -> None:
    """BLADE-3's winner at a rung is one of the three marginals' results."""
    model = _ToySegmentationModel()
    x, y = _toy_case()
    marginals = {}
    for objective in ("dice_ce", "boundary", "frontier"):
        torch.manual_seed(0)
        _, snaps = blade_trajectory(
            model, x, y, 0.1, 2, _dice_ce, n_steps=2,
            snapshot_eps=[0.1], losses=(objective,),
        )
        marginals[objective] = snaps[0.1]
    chosen: dict[float, str] = {}
    torch.manual_seed(0)
    _, snaps = blade_trajectory(
        model, x, y, 0.1, 2, _dice_ce, n_steps=2,
        snapshot_eps=[0.1], chosen_log=chosen,
    )
    assert torch.equal(snaps[0.1], marginals[chosen[0.1]])
