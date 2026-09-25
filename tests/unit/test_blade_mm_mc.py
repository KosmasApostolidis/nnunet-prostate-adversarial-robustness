"""BLADE-MM-MC: BLADE-MC ladders, BLADE-MM adjudication."""

from __future__ import annotations

import torch

from mri_prostate_seg.attacks.blade_mm_mc import BLADE_LOSSES, blade_mm_mc_trajectory


class _Tiny(torch.nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.conv = torch.nn.Conv3d(1, 3, kernel_size=3, padding=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.conv(x)


def _case():
    torch.manual_seed(0)
    x = torch.randn(1, 1, 6, 8, 8)
    y = torch.zeros(1, 1, 6, 8, 8, dtype=torch.long)
    y[..., 1:5, 1:4, 1:7] = 1
    y[..., 1:5, 4:7, 1:7] = 2
    return x, y


def test_returns_snapshots_per_rung_and_logs_every_branch() -> None:
    x, y = _case()
    chosen: dict = {}
    cands: dict = {}
    adv, snaps = blade_mm_mc_trajectory(
        _Tiny(), x, y, 0.2, 3, n_steps=3, random_start=False,
        snapshot_eps=[0.1, 0.2], chosen_log=chosen, candidate_log=cands,
    )
    assert set(snaps) == {0.1, 0.2} and torch.equal(adv, snaps[0.2])
    for rung in (0.1, 0.2):
        assert chosen[rung] in BLADE_LOSSES
        assert set(cands[rung]) == set(BLADE_LOSSES)
        assert set(cands[rung][chosen[rung]]) == {"dice", "hd95", "asd"}
        assert (snaps[rung] - x).abs().max() <= 0.2 + 1e-6
        assert torch.equal(snaps[rung], snaps[rung])  # finite, no nan
        assert torch.isfinite(snaps[rung]).all()


def test_eps_zero_is_clean_and_ignore_label_is_accepted() -> None:
    x, y = _case()
    y[..., 0, :, :] = -1
    adv, snaps = blade_mm_mc_trajectory(_Tiny(), x, y, 0.0, 3, n_steps=2, snapshot_eps=[0.0])
    assert torch.equal(adv, x) and torch.equal(snaps[0.0], x)
    adv, _ = blade_mm_mc_trajectory(
        _Tiny(), x, y, 0.1, 3, n_steps=2, random_start=False, losses=("dice_ce", "frontier"),
    )
    assert torch.isfinite(adv).all()
