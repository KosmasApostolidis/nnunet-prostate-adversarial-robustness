"""FGSM (Fast Gradient Sign Method) L∞ adversarial attack.

Single-step attack — fastest, weakest of the gradient-based family.
"""

from __future__ import annotations

import torch


def _samplewise_bounds(x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    spatial_dims = tuple(range(1, x.ndim))
    x_min = x.amin(dim=spatial_dims, keepdim=True)
    x_max = x.amax(dim=spatial_dims, keepdim=True)
    return x_min, x_max


def fgsm_attack(
    model: torch.nn.Module,
    x: torch.Tensor,
    y: torch.Tensor,
    eps: float,
    num_classes: int,
    loss_fn,
    x_bounds: tuple[torch.Tensor, torch.Tensor] | None = None,
    normalized_grad: bool = False,
) -> torch.Tensor:
    """Single-step FGSM attack.

    Args:
        model: Segmentation network (eval mode expected).
        x: Clean input ``[B, C, D, H, W]``.
        y: Ground-truth ``[B, 1, D, H, W]`` (int, -1 = ignore).
        eps: L∞ perturbation budget.
        num_classes: Number of classes.
        loss_fn: ``(logits, y, num_classes) -> scalar loss``.
        x_bounds: Optional ``(x_min, x_max)`` per-sample intensity bounds.
        normalized_grad: If True, use L2-normalized gradient direction.

    Returns:
        Adversarial example ``[B, C, D, H, W]``.
    """
    if eps == 0.0:
        return x.clone()

    x_orig = x.detach()
    if x_bounds is not None:
        x_min, x_max = x_bounds
    else:
        x_min, x_max = _samplewise_bounds(x_orig)

    saved_grad_states = [(p, p.requires_grad) for p in model.parameters()]
    for p, _ in saved_grad_states:
        p.requires_grad_(False)

    try:
        x_adv = x_orig.clone().requires_grad_(True)
        model.zero_grad(set_to_none=True)
        logits = model(x_adv)
        loss = loss_fn(logits, y, num_classes)
        loss.backward()
        grad = x_adv.grad
        if grad is None:
            return x_orig.clone()
        g = grad.detach()
        if normalized_grad:
            l2 = (
                g.pow(2)
                .sum(dim=tuple(range(1, g.ndim)), keepdim=True)
                .sqrt()
                .clamp_min(1e-12)
            )
            x_adv = x_orig + eps * (g / l2)
        else:
            x_adv = x_orig + eps * g.sign()
        x_adv = torch.max(torch.min(x_adv, x_orig + eps), x_orig - eps)
        x_adv = torch.clamp(x_adv, x_min, x_max)
    finally:
        for p, state in saved_grad_states:
            p.requires_grad_(state)

    return x_adv.detach()
