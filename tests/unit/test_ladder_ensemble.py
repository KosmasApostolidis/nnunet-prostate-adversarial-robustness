"""ladder_ensemble reproduces BLADE and SEA exactly when given their objectives.

That equivalence is what makes the new factorial arms interpretable: an arm
built here differs from BLADE (or from SEA) only in the objectives or the
ladder, never in the machinery.
"""

from __future__ import annotations

import pytest
import torch

from mri_prostate_seg.attacks import blade_updated as mc
from mri_prostate_seg.attacks.blade_legacy import (
    blade_trajectory as legacy_blade,
)
from mri_prostate_seg.attacks.blade_legacy import (
    boundary_weighted_loss,
    frontier_margin_loss,
    signed_distance_map,
)
from mri_prostate_seg.attacks.ladder_ensemble import ladder_ensemble_trajectory
from mri_prostate_seg.attacks.sea import LOSS_FNS, sea_trajectory

EPS = [0.0, 0.02, 0.05, 0.1]


class _Toy(torch.nn.Module):
    def __init__(self, classes: int = 2) -> None:
        super().__init__()
        torch.manual_seed(0)
        self.conv = torch.nn.Conv3d(1, classes, kernel_size=3, padding=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.conv(x)


def _case(classes: int = 2) -> tuple[torch.Tensor, torch.Tensor]:
    x = torch.linspace(-1.0, 1.0, 4 * 8 * 8).reshape(1, 1, 4, 8, 8)
    y = torch.zeros((1, 1, 4, 8, 8))
    y[:, :, 1:3, 2:6, 2:6] = 1.0
    if classes == 3:
        y[:, :, 1:3, 2:6, 4:6] = 2.0
    return x, y


def _dice_ce(logits: torch.Tensor, y: torch.Tensor, nc: int) -> torch.Tensor:
    return torch.nn.functional.cross_entropy(logits, y.squeeze(1).long())


def _same(a: dict, b: dict) -> None:
    assert set(a) == set(b)
    for e in a:
        assert torch.equal(a[e], b[e]), f"snapshot at eps={e} differs"


def test_legacy_with_blade_objectives_reproduces_blade() -> None:
    model, (x, y) = _Toy(), _case()
    phi = signed_distance_map(y)
    objectives = {
        "dice_ce": _dice_ce,
        "boundary": lambda lg, t, nc: boundary_weighted_loss(lg, t, nc, phi=phi),
        "frontier": lambda lg, t, nc: frontier_margin_loss(lg, t, nc),
    }
    torch.manual_seed(1)
    _, ref = legacy_blade(model, x, y, 0.1, 2, _dice_ce, 3, snapshot_eps=EPS)
    torch.manual_seed(1)
    _, got = ladder_ensemble_trajectory(
        model, x, y, 0.1, 2, objectives, 3, snapshot_eps=EPS
    )
    _same(got, ref)


def test_legacy_single_rung_with_sea_objectives_is_sea() -> None:
    model, (x, y) = _Toy(), _case()
    for eps in (0.02, 0.1):
        torch.manual_seed(2)
        _, ref = sea_trajectory(model, x, y, eps, 2, 3, snapshot_eps=[eps])
        torch.manual_seed(2)
        _, got = ladder_ensemble_trajectory(
            model, x, y, eps, 2, dict(LOSS_FNS), 3, snapshot_eps=[eps]
        )
        _same(got, ref)


def test_mc_with_blade_objectives_reproduces_blade_mc() -> None:
    model, (x, y) = _Toy(classes=3), _case(classes=3)
    y = y.long()  # BLADE-MC takes integer labels; the driver passes y.long()
    objectives = {
        "dice_ce": lambda lg, t, nc: mc.multiclass_dice_ce_loss(
            lg, t, nc, ignore_label=-1
        ),
        "frontier": lambda lg, t, nc: mc.frontier_margin_loss(
            lg, t, nc, ignore_label=-1
        ),
    }
    torch.manual_seed(3)
    _, ref = mc.blade_trajectory(
        model, x, y, 0.1, 3, None, 3, snapshot_eps=EPS, losses=("dice_ce", "frontier")
    )
    torch.manual_seed(3)
    _, got = ladder_ensemble_trajectory(
        model, x, y, 0.1, 3, objectives, 3, snapshot_eps=EPS, impl="mc"
    )
    _same(got, ref)


@pytest.mark.parametrize("impl", ["legacy", "mc"])
def test_sea_objectives_on_the_ladder_stay_in_the_ball(impl: str) -> None:
    model, (x, y) = _Toy(classes=3), _case(classes=3)
    if impl == "mc":
        y = y.long()
    chosen: dict[float, str] = {}
    _, snaps = ladder_ensemble_trajectory(
        model,
        x,
        y,
        0.1,
        3,
        dict(LOSS_FNS),
        3,
        snapshot_eps=EPS,
        impl=impl,
        chosen_log=chosen,
    )
    assert torch.equal(snaps[0.0], x)
    for e in EPS[1:]:
        assert float((snaps[e] - x).abs().max()) <= e + 1e-6
    assert {chosen[e] for e in EPS[1:]} <= set(LOSS_FNS) | {"clean"}


def test_rejects_unknown_impl_and_empty_objectives() -> None:
    model, (x, y) = _Toy(), _case()
    with pytest.raises(ValueError, match="impl"):
        ladder_ensemble_trajectory(model, x, y, 0.1, 2, {"a": _dice_ce}, 3, impl="x")
    with pytest.raises(ValueError, match="non-empty"):
        ladder_ensemble_trajectory(model, x, y, 0.1, 2, {}, 3)
