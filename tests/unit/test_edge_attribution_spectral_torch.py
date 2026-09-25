"""Spectral intervention operator on the toy models (SSEUA spec, Interventions)."""

from __future__ import annotations

import numpy as np
import pytest
import torch

from experiments.evaluate_adversarial_quality_all_cases import _per_sample_bce
from mri_prostate_seg.experiments.edge_attribution.spectral import (
    N_SHELLS,
    reference_shell_edges,
    shell_masks,
    split_delta,
)
from mri_prostate_seg.experiments.edge_attribution.torch_ops import (
    SpatialOperator,
    SpectralOperator,
    evaluate_interventions,
    greedy_restoration,
    integrated_attribution,
    pairwise_interactions,
)
from tests.unit.test_edge_attribution_torch_ops import (
    NUM_CLASSES,
    SHAPE,
    _Linear,
    _primary,
    _tensors,
)

SPACING = (3.0, 0.5, 0.5)


def _case():
    rng = np.random.default_rng(3)
    x_np = -np.ones(SHAPE, dtype=np.float32)
    x_np[1:3, 4:12, 4:12] = 1.0
    gt = (x_np > 0).astype(np.float32)
    delta = rng.normal(scale=0.3, size=SHAPE)
    x, y = _tensors(x_np, gt)
    x_adv = x + torch.from_numpy(delta.astype(np.float32))[None, None]
    valid = np.ones(SHAPE, dtype=bool)
    split = split_delta(delta, valid)
    part = shell_masks(SHAPE, SPACING, reference_shell_edges(N_SHELLS))
    op = SpectralOperator.from_split(x, x_adv, split, crop=SHAPE, device=x.device)
    return x, x_adv, y, split, part, op


def test_spectral_operator_endpoints_and_linearity() -> None:
    x, x_adv, y, split, part, op = _case()
    none = torch.zeros((1, 1, *part.masks.shape[1:]), dtype=torch.bool)
    allm = torch.from_numpy(part.masks.any(axis=0))[None, None]
    assert torch.allclose(op.apply(x, x_adv, none, "remove"), x_adv, atol=1e-6)
    assert torch.allclose(op.apply(x, x_adv, allm, "keep"), x_adv, atol=1e-5)
    base = op.apply(x, x_adv, allm, "remove")
    assert torch.allclose(op.apply(x, x_adv, none, "keep"), base, atol=1e-6)
    # remove-all leaves x + delta_out + c0 (delta_out = 0 here)
    assert torch.allclose(base, x + split.c0, atol=1e-5)
    m = torch.from_numpy(part.masks[3])[None, None]
    lin = op.apply(x, x_adv, m, "remove") + op.apply(x, x_adv, m, "keep") - x_adv
    assert torch.allclose(lin, base, atol=1e-5)
    op.check(x, x_adv, op.apply(x, x_adv, m, "remove"), m, "remove")  # no raise


def test_evaluate_interventions_with_spectral_operator_returns_finite_rows() -> None:
    x, x_adv, y, split, part, op = _case()
    out = evaluate_interventions(
        _Linear(), _per_sample_bce, x, x_adv, y, part.masks, mode="remove",
        num_classes=NUM_CLASSES, batch=8, operator=op,
    )
    assert out.dice.shape == (N_SHELLS, NUM_CLASSES - 1)
    assert np.isfinite(out.dice).all() and np.isfinite(out.loss).all()


def test_spatial_default_is_unchanged() -> None:
    x, x_adv, y, split, part, op = _case()
    masks = np.zeros((2, *SHAPE), dtype=bool)
    masks[0, :, :8] = True
    masks[1, :, 8:] = True
    a = evaluate_interventions(
        _Linear(), _per_sample_bce, x, x_adv, y, masks, mode="remove",
        num_classes=NUM_CLASSES,
    )
    b = evaluate_interventions(
        _Linear(), _per_sample_bce, x, x_adv, y, masks, mode="remove",
        num_classes=NUM_CLASSES, operator=SpatialOperator(),
    )
    assert np.array_equal(a.dice, b.dice) and np.array_equal(a.loss, b.loss)


def test_spectral_masks_rejected_by_spatial_operator() -> None:
    x, x_adv, y, split, part, op = _case()
    with pytest.raises(ValueError):
        evaluate_interventions(
            _Linear(), _per_sample_bce, x, x_adv, y, part.masks, mode="remove",
            num_classes=NUM_CLASSES,
        )


def test_greedy_and_pairwise_accept_the_operator() -> None:
    x, x_adv, y, split, part, op = _case()
    model = _Linear()
    masks_by_id = {k + 1: part.masks[k] for k in range(4)}
    rows = greedy_restoration(
        model, _per_sample_bce, x, x_adv, y, masks_by_id, num_classes=NUM_CLASSES,
        primary=_primary, clean_primary=1.0, d_full=0.5, stop_fraction=2.0,
        operator=op,
    )
    assert [r["step"] for r in rows] == [1, 2, 3, 4]
    pairs = pairwise_interactions(
        model, _per_sample_bce, x, x_adv, y, masks_by_id, num_classes=NUM_CLASSES,
        primary=_primary, clean_primary=1.0,
        sufficiency_by_id={k: 0.0 for k in masks_by_id}, operator=op,
    )
    assert len(pairs) == 6


def test_ig_mean_gradient_reproduces_attribution() -> None:
    x, x_adv, y, split, part, op = _case()
    ig = integrated_attribution(
        _Linear(), _per_sample_bce, x, x_adv, y, num_classes=NUM_CLASSES, steps=4
    )
    assert ig.mean_gradient is not None and ig.mean_gradient.shape == SHAPE
    delta = (x_adv - x)[0, 0].numpy()
    assert np.allclose(ig.mean_gradient * delta, ig.attribution, atol=1e-6)
