"""``--attack-valid-only``: the attack must never see, and never deliver, a
perturbation outside the attackable mask (zones margin + padding)."""

from __future__ import annotations

import torch

from experiments.fgsm_adversarial_evaluation import (
    IGNORE_LABEL,
    ConfinedInput,
    attackable_mask,
    confine,
    fgsm_attack,
    pad_to_divisible,
)


class _Recording(torch.nn.Module):
    """Linear probe that keeps every input it was called with."""

    def __init__(self, num_classes: int = 3) -> None:
        super().__init__()
        self.conv = torch.nn.Conv3d(1, num_classes, kernel_size=1)
        self.seen: list[torch.Tensor] = []

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        self.seen.append(x.detach().clone())
        return self.conv(x)


def _zones_like_case():
    torch.manual_seed(0)
    x = torch.randn(1, 1, 5, 6, 7)
    y = torch.zeros(1, 1, 5, 6, 7)
    y[..., 1:4, 1:5, 2:6] = 1.0  # a gland
    y[..., 0, :, :] = IGNORE_LABEL  # zero-filled margin, as in the zones crops
    x[..., 0, :, :] = 0.0
    return x, y


def test_attackable_mask_excludes_ignore_label_and_padding() -> None:
    x, y = _zones_like_case()
    div = [4, 4, 4]
    mask = attackable_mask(y, div)
    x_padded, _ = pad_to_divisible(x, div)
    assert mask.shape == x_padded.shape and mask.dtype == torch.bool
    assert not mask[..., 0, :, :].any()  # ignore-labelled slab
    assert not mask[..., 5:, :, :].any()  # depth padding
    assert not mask[..., :, 6:, :].any() and not mask[..., :, :, 7:].any()
    assert mask[..., 1:5, :6, :7].all()  # everything with a valid label


def test_wrapper_hides_outside_mask_and_zeroes_its_gradient() -> None:
    x, y = _zones_like_case()
    div = [4, 4, 4]
    x_padded, _ = pad_to_divisible(x, div)
    mask = attackable_mask(y, div)
    base = _Recording()
    wrapped = ConfinedInput(base, x_padded, mask)
    assert list(wrapped.parameters()) == list(base.parameters())

    probe = (x_padded + 0.5).requires_grad_(True)
    wrapped(probe).sum().backward()
    seen = base.seen[-1]
    assert torch.equal(seen[mask], probe.detach()[mask])
    assert torch.equal(seen[~mask], x_padded[~mask])
    assert probe.grad is not None
    assert torch.count_nonzero(probe.grad[~mask]) == 0
    assert torch.count_nonzero(probe.grad[mask]) > 0


def test_fgsm_through_wrapper_is_confined_after_confine() -> None:
    x, y = _zones_like_case()
    div = [4, 4, 4]
    x_padded, _ = pad_to_divisible(x, div)
    y_padded, _ = pad_to_divisible(y, div)
    mask = attackable_mask(y, div)
    base = _Recording()
    x_adv = fgsm_attack(ConfinedInput(base, x_padded, mask), x_padded, y_padded, 0.1, 3)
    x_adv = confine(x_adv, x_padded, mask)
    delta = x_adv - x_padded
    assert torch.count_nonzero(delta[~mask]) == 0
    assert torch.count_nonzero(delta[mask]) > 0
    assert confine(x_adv, x_padded, None) is x_adv


def test_auto_pgd_iterate_callback_observes_every_forward_without_changing_result() -> None:
    from mri_prostate_seg.attacks.auto_pgd import auto_pgd_attack

    x, y = _zones_like_case()
    y = y.clamp_min(0)  # binary task, no ignore label needed here
    seen: list[tuple[int, tuple[int, ...]]] = []

    def loss(logits, target, num_classes):
        return logits.mean()

    torch.manual_seed(1)
    ref = auto_pgd_attack(_Recording(2), x, y, 0.1, 2, loss, n_steps=4, random_start=False)
    torch.manual_seed(1)
    out = auto_pgd_attack(
        _Recording(2), x, y, 0.1, 2, loss, n_steps=4, random_start=False,
        iterate_callback=lambda img, lg, step, val: seen.append((step, tuple(lg.shape))),
    )
    assert torch.equal(ref, out)
    assert [s for s, _ in seen] == [0, 1, 2, 3, 4]
    assert all(shape == (1, 2, 5, 6, 7) for _, shape in seen)
