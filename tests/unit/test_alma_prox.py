"""Contract tests for the ALMA prox arm.

The prox against a brute-force minimisation of the same objective, the P2
penalty against its analytic derivative, the two-class constraint fallback, the
ignore label, and the campaign contract (budget ball, intensity bounds, the
clean image at eps = 0).
"""

from __future__ import annotations

import pytest
import torch

from mri_prostate_seg.attacks.alma_prox import (
    _PenaltyP2,
    alma_prox_trajectory,
    constraint_values,
    difference_of_logits,
    foreground_band,
    prox_linf_indicator,
)


class _ToySegmentationModel(torch.nn.Module):
    def __init__(self, num_classes: int = 2) -> None:
        super().__init__()
        torch.manual_seed(0)
        self.conv = torch.nn.Conv3d(1, num_classes, kernel_size=3, padding=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.conv(x)


def _toy_case() -> tuple[torch.Tensor, torch.Tensor]:
    x = torch.linspace(-1.0, 1.0, 4 * 8 * 8).reshape(1, 1, 4, 8, 8)
    y = torch.zeros((1, 1, 4, 8, 8), dtype=torch.float32)
    y[:, :, 1:3, 2:6, 2:6] = 1.0
    return x, y


def _infhot(logits: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
    return torch.zeros_like(logits).scatter_(1, labels.unsqueeze(1), float("inf"))


# --- the proximity operator -------------------------------------------------


def test_prox_matches_brute_force_minimisation() -> None:
    torch.manual_seed(0)
    delta = torch.randn(1, 6)
    lower = torch.full((1, 6), -0.8)
    upper = torch.full((1, 6), 0.9)
    metric = torch.rand(1, 6) + 0.5
    lam = torch.tensor([0.3])

    prox = prox_linf_indicator(delta, lam=lam, lower=lower, upper=upper, metric=metric)

    projected = delta.clamp(min=lower, max=upper)
    levels = torch.linspace(0.0, float(projected.abs().max()), 20_001)
    candidates = projected.clamp(min=-levels.unsqueeze(1), max=levels.unsqueeze(1))
    objective = ((candidates - delta).square() * metric).sum(dim=1) + 2 * lam * levels
    best = candidates[int(objective.argmin())]

    assert torch.allclose(prox.flatten(), best, atol=1e-3)


def test_prox_respects_the_box() -> None:
    delta = torch.tensor([[-5.0, 5.0, 0.1]])
    lower = torch.tensor([[-0.2, -0.2, -0.2]])
    upper = torch.tensor([[0.3, 0.3, 0.3]])
    prox = prox_linf_indicator(delta, lam=torch.tensor([0.0]), lower=lower, upper=upper)
    assert bool((prox >= lower - 1e-6).all()) and bool((prox <= upper + 1e-6).all())


# --- the Penalty-Lagrangian -------------------------------------------------


def test_penalty_p2_derivative_matches_autograd() -> None:
    y = torch.tensor(
        [-2.0, -0.5, 0.0, 0.5, 2.0], dtype=torch.double, requires_grad=True
    )
    rho = torch.full_like(y.detach(), 1.5)
    mu = torch.full_like(y.detach(), 0.25)
    analytic = torch.autograd.grad(_PenaltyP2.apply(y, rho, mu).sum(), y)[0]
    numeric = torch.autograd.gradcheck(
        lambda t: _PenaltyP2.apply(t, rho, mu), (y,), eps=1e-6, atol=1e-4
    )
    assert numeric
    assert bool((analytic > 0).all())  # P2 is increasing, so mu stays positive


# --- the constraint ---------------------------------------------------------


def test_two_classes_use_the_unnormalised_margin() -> None:
    torch.manual_seed(0)
    logits = torch.randn(1, 2, 2, 3, 3)
    labels = torch.randint(0, 2, (1, 2, 3, 3))
    infhot = _infhot(logits, labels)
    margin = difference_of_logits(logits, labels, infhot)
    values = constraint_values(logits, labels, infhot, num_classes=2, tolerance=1e-4)
    assert torch.allclose(values, margin + 1e-4)


def test_three_classes_use_the_difference_of_logits_ratio() -> None:
    torch.manual_seed(0)
    logits = torch.randn(1, 3, 2, 3, 3)
    labels = torch.randint(0, 3, (1, 2, 3, 3))
    infhot = _infhot(logits, labels)
    top3 = logits.topk(k=3, dim=1).values
    expected = (difference_of_logits(logits, labels, infhot) + 1e-4) / (
        top3[:, 0] - top3[:, 2] + 1e-8
    )
    values = constraint_values(logits, labels, infhot, num_classes=3, tolerance=1e-4)
    assert torch.allclose(values, expected)


def test_constraint_sign_flags_correct_voxels() -> None:
    # f_y largest -> positive constraint (violated); smallest -> negative.
    logits = torch.tensor([[[[[2.0]]], [[[0.0]]], [[[-1.0]]]]])
    correct = torch.zeros((1, 1, 1, 1), dtype=torch.long)
    wrong = torch.full((1, 1, 1, 1), 2, dtype=torch.long)
    for labels, positive in ((correct, True), (wrong, False)):
        value = constraint_values(
            logits, labels, _infhot(logits, labels), num_classes=3, tolerance=1e-4
        )
        assert bool((value > 0).all()) is positive


# --- the trajectory contract ------------------------------------------------


def test_zero_epsilon_returns_the_clean_image() -> None:
    x, y = _toy_case()
    x_adv, snaps = alma_prox_trajectory(
        _ToySegmentationModel(), x, y, 0.0, 2, n_steps=4, snapshot_eps=[0.0]
    )
    assert torch.equal(x_adv, x)
    assert set(snaps) == {0.0} and torch.equal(snaps[0.0], x)


def test_snapshots_stay_inside_their_ball_and_the_intensity_bounds() -> None:
    x, y = _toy_case()
    budgets = [0.0, 0.02, 0.06]
    _, snaps = alma_prox_trajectory(
        _ToySegmentationModel(), x, y, 0.06, 2, n_steps=12, snapshot_eps=budgets
    )
    assert set(snaps) == set(budgets)
    for eps, adv in snaps.items():
        assert float((adv - x).abs().max()) <= eps + 1e-6
        assert float(adv.min()) >= float(x.min()) - 1e-6
        assert float(adv.max()) <= float(x.max()) + 1e-6


def test_two_class_attack_actually_moves() -> None:
    # The DLR+ normaliser degenerates at two classes; the margin fallback must
    # still produce a perturbation.
    x, y = _toy_case()
    _, snaps = alma_prox_trajectory(
        _ToySegmentationModel(), x, y, 0.1, 2, n_steps=12, snapshot_eps=[0.1]
    )
    assert float((snaps[0.1] - x).abs().max()) > 0.0


def test_ignore_label_is_not_treated_as_background() -> None:
    x, y = _toy_case()
    model = _ToySegmentationModel()
    ignored, background = y.clone(), y.clone()
    ignored[:, :, 3:, :, :] = -1.0
    background[:, :, 3:, :, :] = 0.0
    _, a = alma_prox_trajectory(
        model, x, ignored, 0.05, 2, n_steps=8, snapshot_eps=[0.05]
    )
    _, b = alma_prox_trajectory(
        model, x, background, 0.05, 2, n_steps=8, snapshot_eps=[0.05]
    )
    assert not torch.allclose(a[0.05], b[0.05])


def test_all_ignored_labels_leave_the_image_untouched() -> None:
    # Every constraint is masked out, so the objective -- and its gradient --
    # are identically zero and the prox keeps delta at 0.
    x, y = _toy_case()
    _, snaps = alma_prox_trajectory(
        _ToySegmentationModel(),
        x,
        torch.full_like(y, -1.0),
        0.05,
        2,
        n_steps=6,
        snapshot_eps=[0.05],
    )
    assert torch.allclose(snaps[0.05], x)


def test_min_norm_log_records_one_value_per_sample() -> None:
    x, y = _toy_case()
    log: list[float] = []
    alma_prox_trajectory(
        _ToySegmentationModel(),
        x,
        y,
        0.1,
        2,
        n_steps=8,
        snapshot_eps=[0.1],
        min_norm_log=log,
    )
    assert len(log) == 1
    assert log[0] >= 0.0


def test_loss_log_has_one_entry_per_iteration() -> None:
    x, y = _toy_case()
    log: list[float] = []
    alma_prox_trajectory(
        _ToySegmentationModel(),
        x,
        y,
        0.05,
        2,
        n_steps=7,
        snapshot_eps=[0.05],
        loss_log=log,
    )
    assert len(log) == 7


def test_rejects_a_non_positive_budget_of_steps() -> None:
    x, y = _toy_case()
    with pytest.raises(ValueError, match="n_steps"):
        alma_prox_trajectory(_ToySegmentationModel(), x, y, 0.05, 2, n_steps=0)


# --- the constraint band ----------------------------------------------------


def test_band_is_the_foreground_when_the_radius_is_zero() -> None:
    _, y = _toy_case()
    labels = y.squeeze(1).long()
    band = foreground_band(labels, band_mm=0.0, spacing=(1.0, 1.0, 1.0))
    assert torch.equal(band, labels == 1)


def test_band_dilates_by_the_physical_radius_on_each_axis() -> None:
    labels = torch.zeros((1, 5, 9, 9), dtype=torch.long)
    labels[0, 2, 4, 4] = 1
    # 5 mm on a (3.0, 0.5, 0.5) mm grid -> radii (2, 10, 10), clipped by shape.
    band = foreground_band(labels, band_mm=5.0, spacing=(3.0, 0.5, 0.5))
    assert bool(band.all())
    # 1 mm -> radii (0, 2, 2): the whole in-plane 5x5 block on that slice only.
    band = foreground_band(labels, band_mm=1.0, spacing=(3.0, 0.5, 0.5))
    assert int(band.sum()) == 25
    assert bool(band[0, 2, 2:7, 2:7].all())
    assert not bool(band[0, 1].any()) and not bool(band[0, 3].any())


def test_band_excludes_ignored_foreground() -> None:
    labels = torch.full((1, 1, 3, 3), -1, dtype=torch.long)
    assert not bool(foreground_band(labels, 1.0, (1.0, 1.0, 1.0)).any())


def test_band_restricts_which_voxels_drive_the_attack() -> None:
    # Same image and same labels, but a band that covers everything versus one
    # that covers only the gland: the perturbations must differ.
    x, y = _toy_case()
    model = _ToySegmentationModel()
    common = dict(n_steps=8, snapshot_eps=[0.05], spacing=(1.0, 1.0, 1.0))
    _, tight = alma_prox_trajectory(
        model, x, y, 0.05, 2, constraint_band_mm=0.0, **common
    )
    _, wide = alma_prox_trajectory(
        model, x, y, 0.05, 2, constraint_band_mm=50.0, **common
    )
    _, unrestricted = alma_prox_trajectory(model, x, y, 0.05, 2, **common)
    assert not torch.allclose(tight[0.05], wide[0.05])
    # A band wide enough to cover the volume is the paper's setting again.
    assert torch.allclose(wide[0.05], unrestricted[0.05])


def test_empty_constrained_region_leaves_the_image_untouched() -> None:
    x, y = _toy_case()
    _, snaps = alma_prox_trajectory(
        _ToySegmentationModel(),
        x,
        torch.zeros_like(y),  # background only: no foreground, so no band
        0.05,
        2,
        n_steps=6,
        snapshot_eps=[0.05],
        constraint_band_mm=1.0,
        spacing=(1.0, 1.0, 1.0),
    )
    assert torch.allclose(snaps[0.05], x)


def test_each_budget_keeps_the_worst_iterate_that_fits_inside_it(monkeypatch) -> None:
    """The per-epsilon rule this port adds on top of the paper.

    ALMA prox has no epsilon, so each reported budget is served by the
    lowest-Dice iterate whose perturbation fits inside that ball. Record every
    iterate the optimiser produces and check the snapshots are exactly that.
    """
    from mri_prostate_seg.attacks import alma_prox as module
    from mri_prostate_seg.attacks.blade_updated import foreground_dice_per_sample

    x, y = _toy_case()
    model = _ToySegmentationModel()
    budgets = [0.02, 0.05]
    seen: list[torch.Tensor] = []
    original = module.prox_linf_indicator

    def spy(*args, **kwargs):
        out = original(*args, **kwargs)
        seen.append(out.detach().clone())
        return out

    monkeypatch.setattr(module, "prox_linf_indicator", spy)
    _, snaps = module.alma_prox_trajectory(
        model, x, y, max(budgets), 2, n_steps=15, snapshot_eps=budgets
    )
    labels = y.squeeze(1).long()

    def dice(image: torch.Tensor) -> float:
        with torch.no_grad():
            return float(foreground_dice_per_sample(model(image), labels, 2)[0])

    # The clean image is an iterate too: it is what a budget keeps if nothing
    # better fits inside it.
    iterates = [x] + [x + d for d in seen]
    for eps in budgets:
        fitting = [i for i in iterates if float((i - x).abs().max()) <= eps + 1e-12]
        assert fitting, f"no iterate fits inside {eps}"
        assert dice(snaps[eps]) == pytest.approx(min(dice(i) for i in fitting), abs=1e-9)
