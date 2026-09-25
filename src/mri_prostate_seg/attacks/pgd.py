"""PGD (Projected Gradient Descent) L∞ adversarial attack.

Pure PyTorch — takes a model and tensors, returns adversarial examples.
No dependency on experiment scripts or file I/O.
"""

from __future__ import annotations

import warnings

import torch


def _project_linf(
    x_adv: torch.Tensor,
    x_orig: torch.Tensor,
    eps: float,
    x_min: torch.Tensor,
    x_max: torch.Tensor,
) -> torch.Tensor:
    """Project onto L∞ ball of radius `eps` around `x_orig`, then clamp to [x_min, x_max]."""
    x_adv = torch.max(torch.min(x_adv, x_orig + eps), x_orig - eps)
    return torch.clamp(x_adv, x_min, x_max)


def _samplewise_bounds(x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """Per-sample min/max intensity for intensity-bound clamping."""
    spatial_dims = tuple(range(1, x.ndim))
    x_min = x.amin(dim=spatial_dims, keepdim=True)
    x_max = x.amax(dim=spatial_dims, keepdim=True)
    return x_min, x_max


def pgd_trajectory(
    model: torch.nn.Module,
    x: torch.Tensor,
    y: torch.Tensor,
    max_eps: float,
    num_classes: int,
    loss_fn,
    n_steps: int,
    step_size: float | None = None,
    random_start: bool = False,
    momentum: float = 0.0,
    normalized_grad: bool = False,
    x_bounds: tuple[torch.Tensor, torch.Tensor] | None = None,
    snapshot_eps: list[float] | None = None,
    loss_log: list[float] | None = None,
) -> tuple[torch.Tensor, dict[float, torch.Tensor]]:
    """Run PGD at ``max_eps``, optionally snapshot intermediate L∞ projections.

    Args:
        model: Segmentation network (eval mode expected).
        x: Clean input tensor ``[B, C, D, H, W]``.
        y: Ground-truth labels ``[B, 1, D, H, W]`` (int, -1 = ignore).
        max_eps: L∞ perturbation budget.
        num_classes: Number of foreground + background classes.
        loss_fn: Callable ``(logits, y, num_classes) -> scalar loss``.
        n_steps: Number of PGD iterations.
        step_size: Per-step size. Default: ``max_eps / n_steps``.
        random_start: Uniform random init within L∞ ball.
        momentum: MI-FGSM momentum factor (0 = off).
        normalized_grad: L2-normalize gradient before sign step.
        x_bounds: Optional ``(x_min, x_max)`` per-sample intensity bounds.
        snapshot_eps: Epsilon values to snapshot along the trajectory.
        loss_log: If provided, append per-step loss values.

    Returns:
        ``(x_adv_final, snapshots)`` where snapshots maps ``eps -> x_adv``.
    """
    if max_eps == 0.0:
        clean = x.clone()
        snaps = {0.0: clean.clone()} if snapshot_eps else {}
        return clean, snaps

    if n_steps < 1:
        raise ValueError("n_steps must be >= 1 for PGD")

    if step_size is None:
        step_size = float(max_eps) / float(n_steps)
    if step_size > float(max_eps):
        warnings.warn(
            f"step_size ({step_size:.6g}) > max_eps ({max_eps:.6g})",
            stacklevel=2,
        )

    x_orig = x.detach()
    if x_bounds is not None:
        x_min, x_max = x_bounds
    else:
        x_min, x_max = _samplewise_bounds(x_orig)

    if random_start:
        delta = (torch.rand_like(x_orig) * 2.0 - 1.0) * float(max_eps)
        x_adv = _project_linf(x_orig + delta, x_orig, max_eps, x_min, x_max)
    else:
        x_adv = x_orig.clone()

    snapshot_eps_list = sorted(set(snapshot_eps or []))
    snapshots: dict[float, torch.Tensor] = {}
    if 0.0 in snapshot_eps_list:
        snapshots[0.0] = x_orig.clone()
    pending = [e for e in snapshot_eps_list if e > 0.0]

    saved_grad_states = [(p, p.requires_grad) for p in model.parameters()]
    for p, _ in saved_grad_states:
        p.requires_grad_(False)

    grad_accum = torch.zeros_like(x_orig)
    try:
        for k in range(1, n_steps + 1):
            x_adv = x_adv.detach().requires_grad_(True)
            model.zero_grad(set_to_none=True)
            logits = model(x_adv)
            loss = loss_fn(logits, y, num_classes)
            if loss_log is not None:
                loss_log.append(float(loss.detach().cpu()))
            loss.backward()
            grad = x_adv.grad
            if grad is None:
                clean = x_orig.clone()
                for e in pending:
                    snapshots[e] = clean.clone()
                snapshots[float(max_eps)] = clean
                return clean, snapshots
            g = grad.detach()
            if momentum > 0.0:
                l1 = (
                    g.abs()
                    .mean(dim=tuple(range(1, g.ndim)), keepdim=True)
                    .clamp_min(1e-12)
                )
                grad_accum = momentum * grad_accum + g / l1
                x_adv = x_adv.detach() + step_size * grad_accum.sign()
            elif normalized_grad:
                l2 = (
                    g.pow(2)
                    .sum(dim=tuple(range(1, g.ndim)), keepdim=True)
                    .sqrt()
                    .clamp_min(1e-12)
                )
                x_adv = x_adv.detach() + step_size * (g / l2)
            else:
                x_adv = x_adv.detach() + step_size * g.sign()
            x_adv = _project_linf(x_adv, x_orig, max_eps, x_min, x_max)

            still_pending = []
            for e in pending:
                if k * step_size + 1e-12 >= e or k == n_steps:
                    snapshots[e] = _project_linf(
                        x_adv.detach(), x_orig, e, x_min, x_max
                    ).clone()
                else:
                    still_pending.append(e)
            pending = still_pending
    finally:
        for p, state in saved_grad_states:
            p.requires_grad_(state)

    snapshots[float(max_eps)] = x_adv.detach()
    return x_adv.detach(), snapshots


def pgd_attack(
    model: torch.nn.Module,
    x: torch.Tensor,
    y: torch.Tensor,
    eps: float,
    num_classes: int,
    loss_fn,
    n_steps: int,
    step_size: float | None = None,
    random_start: bool = False,
    x_bounds: tuple[torch.Tensor, torch.Tensor] | None = None,
    momentum: float = 0.0,
    normalized_grad: bool = False,
) -> torch.Tensor:
    """Single-epsilon PGD — convenience wrapper around ``pgd_trajectory``."""
    if eps == 0.0:
        return x.clone()
    x_adv, _ = pgd_trajectory(
        model,
        x,
        y,
        eps,
        num_classes,
        loss_fn,
        n_steps=n_steps,
        step_size=step_size,
        random_start=random_start,
        x_bounds=x_bounds,
        momentum=momentum,
        normalized_grad=normalized_grad,
    )
    return x_adv


def pgd_with_restarts(
    model: torch.nn.Module,
    x: torch.Tensor,
    y: torch.Tensor,
    max_eps: float,
    num_classes: int,
    loss_fn,
    n_steps: int,
    step_size: float | None,
    momentum: float,
    normalized_grad: bool,
    x_bounds: tuple[torch.Tensor, torch.Tensor] | None,
    snapshot_eps: list[float] | None,
    num_restarts: int,
    loss_log: list[float] | None = None,
    random_start: bool = False,
) -> dict[float, torch.Tensor]:
    """Multiple restarts; keeps worst-loss perturbation per snapshot epsilon.

    Restart 0 honours ``random_start``; additional restarts are randomized, so
    ``num_restarts>1`` searches a superset of the single-restart attack."""
    best_snaps: dict[float, torch.Tensor] = {}
    best_loss: dict[float, float] = {}
    for r in range(max(1, num_restarts)):
        rs = (r > 0) or random_start
        _, snaps = pgd_trajectory(
            model,
            x,
            y,
            max_eps,
            num_classes,
            loss_fn,
            n_steps=n_steps,
            step_size=step_size,
            random_start=rs,
            momentum=momentum,
            normalized_grad=normalized_grad,
            x_bounds=x_bounds,
            snapshot_eps=snapshot_eps,
            loss_log=loss_log if r == 0 else None,
        )
        with torch.no_grad():
            for e, xa in snaps.items():
                lv = float(loss_fn(model(xa), y, num_classes).detach().cpu())
                if e not in best_loss or lv > best_loss[e]:
                    best_loss[e] = lv
                    best_snaps[e] = xa
    return best_snaps
