"""Unit tests for the full-cohort FGSM/PGD/APGD quality helpers."""

from __future__ import annotations

import pandas as pd
import pytest
import torch

from experiments.evaluate_adversarial_quality_all_cases import (
    _completed_keys,
    fgsm_bce_independent_batch,
    pgd_bce_independent_batch,
)
from experiments.plot_adversarial_attack_psnr_ssim import (
    METRICS,
    _paired_against_apgd,
)


class _ToySegmentationModel(torch.nn.Module):
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return torch.cat((-x, x), dim=1)


def _toy_batch() -> tuple[torch.Tensor, torch.Tensor]:
    first = torch.linspace(-1.0, 1.0, 27).reshape(1, 1, 3, 3, 3)
    second = torch.linspace(-0.8, 1.2, 27).reshape(1, 1, 3, 3, 3)
    x = torch.cat((first, second), dim=0)
    y = torch.ones((2, 1, 3, 3, 3), dtype=torch.float32)
    return x, y


@pytest.mark.parametrize("attack", ["fgsm", "pgd"])
def test_attacks_are_bounded_and_deterministic(attack: str) -> None:
    model = _ToySegmentationModel()
    x, y = _toy_batch()
    epsilon = 0.1
    if attack == "fgsm":
        first = fgsm_bce_independent_batch(model, x, y, epsilon, 2)
        second = fgsm_bce_independent_batch(model, x, y, epsilon, 2)
    else:
        first = pgd_bce_independent_batch(
            model, x, y, epsilon, 2, n_steps=4, seeds=[11, 22]
        )
        second = pgd_bce_independent_batch(
            model, x, y, epsilon, 2, n_steps=4, seeds=[11, 22]
        )

    torch.testing.assert_close(first, second)
    assert float((first - x).abs().max()) <= epsilon + 1e-6
    assert float(first.min()) >= float(x.amin(dim=(1, 2, 3, 4)).min()) - 1e-6
    assert float(first.max()) <= float(x.amax(dim=(1, 2, 3, 4)).max()) + 1e-6


def test_pgd_batching_does_not_change_a_sample() -> None:
    model = _ToySegmentationModel()
    x, y = _toy_batch()
    batched = pgd_bce_independent_batch(model, x, y, 0.1, 2, n_steps=4, seeds=[11, 22])
    alone = pgd_bce_independent_batch(
        model, x[:1], y[:1], 0.1, 2, n_steps=4, seeds=[11]
    )
    torch.testing.assert_close(batched[:1], alone)


def test_attack_does_not_accumulate_unused_parameter_gradients() -> None:
    model = torch.nn.Conv3d(1, 2, kernel_size=1)
    x, y = _toy_batch()

    pgd_bce_independent_batch(
        model,
        x,
        y,
        0.1,
        2,
        n_steps=2,
        seeds=[11, 22],
    )

    assert all(parameter.grad is None for parameter in model.parameters())


def test_completed_keys_include_attack(tmp_path) -> None:
    path = tmp_path / "quality.csv"
    pd.DataFrame(
        {
            "dataset": ["wg", "wg"],
            "case_id": ["case", "case"],
            "epsilon_n": [8, 8],
            "attack": ["FGSM-BCE", "APGD-BCE"],
        }
    ).to_csv(path, index=False)
    assert _completed_keys(path) == {
        ("wg", "case", 8, "FGSM-BCE"),
        ("wg", "case", 8, "APGD-BCE"),
    }


def test_paired_comparison_subtracts_apgd_within_case() -> None:
    rows = []
    metric_values = {
        "APGD-BCE": [0.05, 5.0, 38.0, 0.90],
        "FGSM-BCE": [0.04, 4.0, 40.0, 0.95],
        "PGD-BCE": [0.045, 4.5, 39.0, 0.92],
    }
    for attack, values in metric_values.items():
        row = {
            "dataset": "wg",
            "case_id": "case",
            "epsilon_n": 16,
            "epsilon_norm": 16 / 255,
            "attack": attack,
        }
        row.update(dict(zip(METRICS, values)))
        rows.append(row)

    paired = _paired_against_apgd(pd.DataFrame(rows)).set_index("attack")
    assert paired.loc["FGSM-BCE", "psnr_db_robust_range_minus_apgd"] == pytest.approx(
        2.0
    )
    assert paired.loc["PGD-BCE", "ssim_axial_prostate_minus_apgd"] == pytest.approx(
        0.02
    )


# --- attackable-voxel mask ---------------------------------------------------
# Zones inputs are zero-filled outside the dilated gland (seg == -1) and padded
# with zeros; the deployed pipeline regenerates both after any image-level
# attack, so a perturbation there is not realisable.  `perturbation_mask`
# confines delta to the attackable voxels.


def _attacks_with_mask():
    from experiments.evaluate_adversarial_quality_all_cases import (
        apgd_bce_independent_batch,
    )

    def fgsm(model, x, y, eps, mask):
        return fgsm_bce_independent_batch(model, x, y, eps, 2, perturbation_mask=mask)

    def pgd(model, x, y, eps, mask):
        return pgd_bce_independent_batch(
            model, x, y, eps, 2, n_steps=3, seeds=[1, 2], perturbation_mask=mask
        )

    def apgd(model, x, y, eps, mask):
        return apgd_bce_independent_batch(
            model, x, y, eps, 2, n_steps=4, seeds=[1, 2], perturbation_mask=mask
        )

    return {"fgsm": fgsm, "pgd": pgd, "apgd": apgd}


@pytest.mark.parametrize("attack", ["fgsm", "pgd", "apgd"])
def test_mask_none_equals_all_true_mask_bitwise(attack: str) -> None:
    x, y = _toy_batch()
    run = _attacks_with_mask()[attack]
    unmasked = run(_ToySegmentationModel(), x, y, 0.1, None)
    all_true = run(_ToySegmentationModel(), x, y, 0.1, torch.ones_like(x, dtype=torch.bool))
    assert torch.equal(unmasked, all_true)


@pytest.mark.parametrize("attack", ["fgsm", "pgd", "apgd"])
def test_mask_confines_delta_to_attackable_voxels(attack: str) -> None:
    x, y = _toy_batch()
    mask = torch.ones_like(x, dtype=torch.bool)
    mask[:, :, 0] = False  # first slice not attackable
    mask[1, :, :, :, 2] = False  # and one column of the second sample
    adv = _attacks_with_mask()[attack](_ToySegmentationModel(), x, y, 0.1, mask)
    delta = adv - x
    assert torch.equal(delta[~mask], torch.zeros_like(delta[~mask]))
    assert (delta.abs() <= 0.1 + 1e-6).all()
    assert (delta[mask] != 0).any()


def test_mask_from_ignore_label_matches_y_pad_convention() -> None:
    from experiments.evaluate_adversarial_quality_all_cases import IGNORE_LABEL
    from experiments.evaluate_edge_attribution_all_cases import attackable_mask

    y = torch.zeros((1, 1, 2, 2, 2))
    y[0, 0, 1] = IGNORE_LABEL
    mask = attackable_mask(y)
    assert mask.shape == y.shape and mask.dtype == torch.bool
    assert mask[0, 0, 0].all() and not mask[0, 0, 1].any()


def test_recenc_apgd_mask_none_is_bitwise_unchanged_and_mask_confines() -> None:
    """The calibration script's APGD (``adv_rob_eval_nnunet_nnunetrecenc.apgd_bce``)
    takes the same ``perturbation_mask`` contract as the batch attacks."""

    from experiments.adv_rob_eval_nnunet_nnunetrecenc import IGNORE_LABEL, apgd_bce

    torch.manual_seed(0)
    model = torch.nn.Sequential(torch.nn.Conv3d(1, 2, 3, padding=1))
    for parameter in model.parameters():
        parameter.requires_grad_(False)
    x = torch.randn(1, 1, 4, 8, 8)
    y = torch.zeros(1, 1, 4, 8, 8)
    y[..., :4, :] = IGNORE_LABEL
    y[..., 4:, :4] = 1
    mask = y != IGNORE_LABEL
    eps = 8 / 255.0

    torch.manual_seed(1)
    unconfined = apgd_bce(model, x, y, eps, 2, n_steps=4)
    torch.manual_seed(1)
    none_mask = apgd_bce(model, x, y, eps, 2, n_steps=4, perturbation_mask=None)
    torch.manual_seed(1)
    confined = apgd_bce(model, x, y, eps, 2, n_steps=4, perturbation_mask=mask)

    assert torch.equal(unconfined, none_mask)
    assert torch.equal(confined[~mask], x[~mask])
    assert (confined[mask] - x[mask]).abs().max() <= eps + 1e-6
    assert (confined[mask] != x[mask]).any()


@pytest.mark.parametrize(
    "module_name",
    [
        "experiments.evaluate_adversarial_quality_all_cases",
        "experiments.evaluate_adversarial_segmentation_all_cases",
        "experiments.evaluate_segmentation_direction_all_cases",
        "experiments.generate_worst_case_perturbation_figures",
        "experiments.epsilon_physical_calibration",
    ],
)
def test_drivers_expose_attack_valid_only_off_by_default(module_name, monkeypatch) -> None:
    import importlib

    if module_name.endswith("epsilon_physical_calibration"):
        pytest.importorskip("nnunetv2")  # imported at module top; not in the test env
    module = importlib.import_module(module_name)
    monkeypatch.setattr("sys.argv", ["prog"])
    assert module.parse_args().attack_valid_only is False
    monkeypatch.setattr("sys.argv", ["prog", "--attack-valid-only"])
    assert module.parse_args().attack_valid_only is True
