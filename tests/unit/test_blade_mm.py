"""Contract tests for BLADE-MM, the multi-metric adjudication variant.

BLADE-MM changes exactly one thing about BLADE: which candidate wins a rung.
BLADE keeps the lowest foreground Dice, which silently discards the candidate
the boundary objective exists to produce. BLADE-MM ranks the candidates on
every reported metric and keeps the one that is worst overall.
"""

from __future__ import annotations

import math

import pytest
import torch

from mri_prostate_seg.attacks.blade_mm import (
    METRIC_DIRECTION,
    blade_mm_trajectory,
    foreground_metrics,
    per_metric_oracle,
    select_by_rank,
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


# --- metric direction --------------------------------------------------------


def test_metric_direction_declares_what_damage_means() -> None:
    # Lower Dice is a stronger attack; higher surface distance is a stronger attack.
    assert METRIC_DIRECTION["dice"] == -1
    assert METRIC_DIRECTION["hd95"] == +1
    assert METRIC_DIRECTION["asd"] == +1


# --- foreground_metrics ------------------------------------------------------


def test_foreground_metrics_dice_matches_hand_computation() -> None:
    logits = torch.zeros(1, 2, 1, 1, 4)
    logits[0, 1, 0, 0, :2] = 1.0          # predict fg on voxels 0,1
    y = torch.tensor([[[[[0.0, 1.0, 1.0, 0.0]]]]])  # fg on voxels 1,2
    m = foreground_metrics(logits, y, 2)
    assert m["dice"] == pytest.approx(0.5, abs=1e-4)


def test_foreground_metrics_reports_surface_distances() -> None:
    logits = torch.full((1, 2, 4, 8, 8), 0.0)
    logits[0, 1, 1:3, 2:6, 2:6] = 5.0     # exactly the ground truth
    _, y = _toy_case()
    m = foreground_metrics(logits, y, 2)
    assert m["dice"] == pytest.approx(1.0, abs=1e-4)
    assert m["hd95"] == pytest.approx(0.0, abs=1e-6)
    assert m["asd"] == pytest.approx(0.0, abs=1e-6)


def test_foreground_metrics_empty_prediction_gives_nan_surfaces() -> None:
    logits = torch.zeros(1, 2, 4, 8, 8)
    logits[0, 0] = 5.0                    # predict background everywhere
    _, y = _toy_case()
    m = foreground_metrics(logits, y, 2)
    assert m["dice"] == pytest.approx(0.0, abs=1e-4)
    assert math.isnan(m["hd95"]) and math.isnan(m["asd"])


def test_foreground_metrics_excludes_ignore_voxels() -> None:
    logits = torch.zeros(1, 2, 4, 8, 8)
    logits[0, 1, 1:3, 2:6, 2:6] = 5.0
    _, y = _toy_case()
    y[0, 0, 3] = -1.0
    m = foreground_metrics(logits, y, 2)
    assert m["dice"] == pytest.approx(1.0, abs=1e-4)


# --- rank selection ----------------------------------------------------------


def test_rank_selection_prefers_the_boundary_candidate_dice_would_discard() -> None:
    cands = {
        "dice_ce":  {"dice": 0.50, "hd95": 10.0, "asd": 3.0},
        "frontier": {"dice": 0.40, "hd95": 5.0,  "asd": 2.0},
        "boundary": {"dice": 0.45, "hd95": 60.0, "asd": 20.0},
    }
    # Dice alone would pick 'frontier' (0.40, the lowest).
    assert min(cands, key=lambda n: cands[n]["dice"]) == "frontier"
    # Ranked across all three metrics, 'boundary' is worst overall.
    assert select_by_rank(cands) == "boundary"


def test_rank_selection_agrees_with_dice_when_one_candidate_dominates() -> None:
    cands = {
        "a": {"dice": 0.10, "hd95": 90.0, "asd": 40.0},
        "b": {"dice": 0.60, "hd95": 8.0, "asd": 2.0},
    }
    assert select_by_rank(cands) == "a"


def test_rank_selection_breaks_ties_on_dice() -> None:
    cands = {
        "a": {"dice": 0.30, "hd95": 20.0, "asd": 5.0},
        "b": {"dice": 0.20, "hd95": 20.0, "asd": 5.0},
    }
    assert select_by_rank(cands) == "b"


def test_rank_selection_ignores_metrics_it_cannot_rank() -> None:
    # Surface metrics undefined for both candidates: only Dice can decide.
    nan = float("nan")
    cands = {
        "a": {"dice": 0.30, "hd95": nan, "asd": nan},
        "b": {"dice": 0.05, "hd95": nan, "asd": nan},
    }
    assert select_by_rank(cands) == "b"


def test_rank_selection_single_candidate() -> None:
    assert select_by_rank({"only": {"dice": 0.4, "hd95": 3.0, "asd": 1.0}}) == "only"


# --- per-metric oracle -------------------------------------------------------


def test_per_metric_oracle_takes_the_worst_value_of_each_metric() -> None:
    cands = {
        "dice_ce":  {"dice": 0.50, "hd95": 10.0, "asd": 3.0},
        "frontier": {"dice": 0.40, "hd95": 5.0,  "asd": 2.0},
        "boundary": {"dice": 0.45, "hd95": 60.0, "asd": 20.0},
    }
    oracle = per_metric_oracle(cands)
    assert oracle["dice"] == (pytest.approx(0.40), "frontier")
    assert oracle["hd95"] == (pytest.approx(60.0), "boundary")
    assert oracle["asd"] == (pytest.approx(20.0), "boundary")


def test_per_metric_oracle_skips_undefined_metrics() -> None:
    nan = float("nan")
    cands = {"a": {"dice": 0.3, "hd95": nan, "asd": nan}}
    oracle = per_metric_oracle(cands)
    assert oracle["dice"][1] == "a"
    assert oracle["hd95"] is None and oracle["asd"] is None


# --- the trajectory ----------------------------------------------------------


def test_blade_mm_records_every_candidate_at_every_rung() -> None:
    model = _ToySegmentationModel()
    x, y = _toy_case()
    cand: dict[float, dict[str, dict[str, float]]] = {}
    chosen: dict[float, str] = {}
    torch.manual_seed(0)
    x_adv, snaps = blade_mm_trajectory(
        model, x, y, 0.1, 2, _dice_ce, n_steps=2,
        snapshot_eps=[0.0, 0.05, 0.1], candidate_log=cand, chosen_log=chosen,
    )
    assert set(snaps) == {0.0, 0.05, 0.1}
    assert set(cand) == {0.05, 0.1}
    for rung in (0.05, 0.1):
        assert set(cand[rung]) == {"dice_ce", "boundary", "frontier"}
        for metrics in cand[rung].values():
            assert set(metrics) == {"dice", "hd95", "asd"}
    assert set(chosen) == {0.05, 0.1}
    assert torch.equal(snaps[0.1], x_adv)


def test_blade_mm_snapshots_respect_the_ball_and_bounds() -> None:
    model = _ToySegmentationModel()
    x, y = _toy_case()
    torch.manual_seed(0)
    _, snaps = blade_mm_trajectory(
        model, x, y, 0.1, 2, _dice_ce, n_steps=2, snapshot_eps=[0.04, 0.1]
    )
    for e in (0.04, 0.1):
        assert float((snaps[e] - x).abs().max()) <= e + 1e-6
        assert float(snaps[e].min()) >= float(x.min()) - 1e-6
        assert float(snaps[e].max()) <= float(x.max()) + 1e-6


def test_blade_mm_zero_eps_and_zero_steps() -> None:
    model = _ToySegmentationModel()
    x, y = _toy_case()
    x_adv, snaps = blade_mm_trajectory(
        model, x, y, 0.0, 2, _dice_ce, n_steps=2, snapshot_eps=[0.0]
    )
    assert torch.equal(x_adv, x)
    assert set(snaps) == {0.0}
    with pytest.raises(ValueError):
        blade_mm_trajectory(model, x, y, 0.1, 2, _dice_ce, n_steps=0)


def test_blade_mm_rejects_unknown_loss() -> None:
    model = _ToySegmentationModel()
    x, y = _toy_case()
    with pytest.raises(ValueError):
        blade_mm_trajectory(
            model, x, y, 0.1, 2, _dice_ce, n_steps=2, losses=("dice_ce", "nope")
        )


def test_blade_mm_single_objective_needs_no_adjudication() -> None:
    model = _ToySegmentationModel()
    x, y = _toy_case()
    chosen: dict[float, str] = {}
    torch.manual_seed(0)
    _, snaps = blade_mm_trajectory(
        model, x, y, 0.1, 2, _dice_ce, n_steps=2, snapshot_eps=[0.1],
        losses=("dice_ce",), chosen_log=chosen,
    )
    assert chosen == {0.1: "dice_ce"}
    assert float((snaps[0.1] - x).abs().max()) <= 0.1 + 1e-6
