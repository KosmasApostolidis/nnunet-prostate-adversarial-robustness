"""Toy-model tests for the torch operations (plan §37.5–§37.8)."""

from __future__ import annotations

import numpy as np
import pytest
import torch

from experiments.evaluate_adversarial_quality_all_cases import _per_sample_bce
from mri_prostate_seg.experiments.edge_attribution.torch_ops import (
    argmax_class_dice,
    evaluate_interventions,
    greedy_restoration,
    integrated_attribution,
    pairwise_interactions,
)

SHAPE = (4, 16, 16)
NUM_CLASSES = 2


class _Linear(torch.nn.Module):
    """logit_1 = 4x, logit_0 = 0: class 1 wherever x > 0."""

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return torch.cat([torch.zeros_like(x), 4.0 * x], dim=1)


class _Gate(torch.nn.Module):
    """Predicts class 1 everywhere unless g(mean_A, mean_B) > 0.5; g = max or min."""

    def __init__(self, a: np.ndarray, b: np.ndarray, mode: str) -> None:
        super().__init__()
        self.register_buffer("a", torch.from_numpy(a.astype(np.float32)))
        self.register_buffer("b", torch.from_numpy(b.astype(np.float32)))
        self.mode = mode

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        sa = (x[:, 0] * self.a).sum(dim=(1, 2, 3)) / self.a.sum()
        sb = (x[:, 0] * self.b).sum(dim=(1, 2, 3)) / self.b.sum()
        g = torch.maximum(sa, sb) if self.mode == "max" else torch.minimum(sa, sb)
        logit1 = (20.0 * (0.5 - g)).view(-1, 1, 1, 1, 1).expand_as(x)
        return torch.cat([torch.zeros_like(x), logit1], dim=1)


def _region(z0: int, z1: int, y0: int, y1: int, x0: int, x1: int) -> np.ndarray:
    m = np.zeros(SHAPE, dtype=bool)
    m[z0:z1, y0:y1, x0:x1] = True
    return m


def _tensors(x_np: np.ndarray, gt_np: np.ndarray) -> tuple[torch.Tensor, torch.Tensor]:
    x = torch.from_numpy(x_np.astype(np.float32))[None, None]
    y = torch.from_numpy(gt_np.astype(np.float32))[None, None]
    return x, y


def _primary(row: np.ndarray) -> float:
    return float(row[0])


def test_argmax_class_dice_matches_numpy_and_handles_empty() -> None:
    pred = torch.tensor([[[[1, 1], [0, 0]]], [[[0, 0], [0, 0]]]])
    target = torch.tensor([[[[1, 0], [0, -1]]], [[[0, 0], [0, 0]]]])
    d = argmax_class_dice(pred, target, NUM_CLASSES)
    assert d.shape == (2, 1)
    assert d[0, 0].item() == pytest.approx(2 * 1 / (2 + 1))
    assert d[1, 0].item() == 1.0


def test_linear_model_attribution_and_interventions_localise_to_the_perturbed_patch() -> (
    None
):
    gt = _region(1, 3, 4, 12, 4, 12)
    x_np = gt.astype(np.float32)  # +1 inside, 0 outside
    patch_a = _region(1, 3, 4, 8, 4, 12)  # half of the gland
    patch_b = _region(1, 3, 8, 12, 4, 12)  # other half, unperturbed
    x_adv_np = x_np.copy()
    x_adv_np[patch_a] -= 2.0
    x, y = _tensors(x_np, gt)
    x_adv, _ = _tensors(x_adv_np, gt)
    model = _Linear().eval()
    model.requires_grad_(False)

    ig = integrated_attribution(
        model, _per_sample_bce, x, x_adv, y, num_classes=NUM_CLASSES, steps=32, batch=4
    )
    positive = np.maximum(ig.attribution, 0.0)
    assert positive[patch_a].sum() / positive.sum() > 0.95
    assert ig.completeness_error < 0.01
    assert ig.loss_adv > ig.loss_clean

    masks = np.stack([patch_a, patch_b])
    removal = evaluate_interventions(
        model,
        _per_sample_bce,
        x,
        x_adv,
        y,
        masks,
        mode="remove",
        num_classes=NUM_CLASSES,
        batch=2,
        keep_pred_indices={0},
    )
    clean_dice = 1.0
    adv_dice = 2 * patch_b.sum() / (patch_b.sum() + gt.sum())
    assert removal.dice[0, 0] == pytest.approx(
        clean_dice
    )  # restoring A recovers everything
    assert removal.dice[1, 0] == pytest.approx(adv_dice)  # restoring B changes nothing
    assert set(removal.preds) == {0} and removal.preds[0].shape == SHAPE
    retention = evaluate_interventions(
        model,
        _per_sample_bce,
        x,
        x_adv,
        y,
        masks,
        mode="keep",
        num_classes=NUM_CLASSES,
        batch=2,
    )
    assert retention.dice[0, 0] == pytest.approx(adv_dice)
    assert retention.dice[1, 0] == pytest.approx(clean_dice)


def test_no_perturbation_gives_zero_attribution_and_no_damage() -> None:
    gt = _region(1, 3, 4, 12, 4, 12)
    x, y = _tensors(gt.astype(np.float32), gt)
    model = _Linear().eval()
    ig = integrated_attribution(
        model, _per_sample_bce, x, x.clone(), y, num_classes=NUM_CLASSES, steps=4
    )
    assert np.abs(ig.attribution).max() == 0.0
    res = evaluate_interventions(
        model,
        _per_sample_bce,
        x,
        x.clone(),
        y,
        np.stack([_region(1, 3, 4, 8, 4, 12)]),
        mode="remove",
        num_classes=NUM_CLASSES,
    )
    assert res.dice[0, 0] == 1.0


def test_redundant_patches_have_zero_individual_necessity_but_greedy_recovers() -> None:
    a = _region(1, 3, 2, 6, 2, 14)
    b = _region(1, 3, 10, 14, 2, 14)
    gt = _region(0, 4, 0, 16, 0, 16)  # all ones so "predict 1 everywhere" is perfect
    x_np = np.zeros(SHAPE, dtype=np.float32)
    x_adv_np = x_np.copy()
    x_adv_np[a] = 1.0
    x_adv_np[b] = 1.0
    x, y = _tensors(x_np, gt)
    x_adv, _ = _tensors(x_adv_np, gt)
    model = _Gate(a, b, "max").eval()
    masks = {1: a, 2: b}
    res = evaluate_interventions(
        model,
        _per_sample_bce,
        x,
        x_adv,
        y,
        np.stack([a, b]),
        mode="remove",
        num_classes=NUM_CLASSES,
    )
    assert res.dice[0, 0] == pytest.approx(0.0) and res.dice[1, 0] == pytest.approx(
        0.0
    )  # still flipped
    rows = greedy_restoration(
        model,
        _per_sample_bce,
        x,
        x_adv,
        y,
        masks,
        num_classes=NUM_CLASSES,
        primary=_primary,
        clean_primary=1.0,
        d_full=1.0,
    )
    assert [r["step"] for r in rows] == [1, 2]
    assert rows[0]["recovered_fraction"] == pytest.approx(0.0)
    assert rows[1]["recovered_fraction"] == pytest.approx(1.0)


def test_synergistic_patches_show_positive_pairwise_interaction() -> None:
    a = _region(1, 3, 2, 6, 2, 14)
    b = _region(1, 3, 10, 14, 2, 14)
    gt = _region(0, 4, 0, 16, 0, 16)
    x_np = np.zeros(SHAPE, dtype=np.float32)
    x_adv_np = x_np.copy()
    x_adv_np[a] = 1.0
    x_adv_np[b] = 1.0
    x, y = _tensors(x_np, gt)
    x_adv, _ = _tensors(x_adv_np, gt)
    model = _Gate(a, b, "min").eval()
    keep = evaluate_interventions(
        model,
        _per_sample_bce,
        x,
        x_adv,
        y,
        np.stack([a, b]),
        mode="keep",
        num_classes=NUM_CLASSES,
    )
    singles = {1: 1.0 - float(keep.dice[0, 0]), 2: 1.0 - float(keep.dice[1, 0])}
    assert singles == {1: pytest.approx(0.0), 2: pytest.approx(0.0)}
    rows = pairwise_interactions(
        model,
        _per_sample_bce,
        x,
        x_adv,
        y,
        {1: a, 2: b},
        num_classes=NUM_CLASSES,
        primary=_primary,
        clean_primary=1.0,
        sufficiency_by_id=singles,
    )
    assert len(rows) == 1
    assert rows[0]["joint_sufficiency"] == pytest.approx(1.0)
    assert rows[0]["interaction"] == pytest.approx(1.0)


def test_interventions_reject_out_of_range_inputs() -> None:
    gt = _region(1, 3, 4, 12, 4, 12)
    x, y = _tensors(gt.astype(np.float32), gt)
    with pytest.raises(ValueError):
        evaluate_interventions(
            _Linear().eval(),
            _per_sample_bce,
            x,
            x,
            y,
            np.zeros((1, *SHAPE), bool),
            mode="sideways",
            num_classes=NUM_CLASSES,
        )
