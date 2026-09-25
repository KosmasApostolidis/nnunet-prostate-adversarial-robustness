"""A-PGD (Adaptive Projected Gradient Descent) L∞ adversarial attack.

Extends PGD with adaptive step sizing: at checkpoint iterations the step size
is adjusted based on the loss trend. If loss stagnates the optimizer restarts
from the best-so-far point with halved step size, avoiding wasted iterations
on flat loss regions.

Reference: Croce & Hein, "Reliable evaluation of adversarial robustness
with an ensemble of diverse parameter-free attacks", ICML 2020.
"""

from __future__ import annotations

import warnings

import torch

from mri_prostate_seg.attacks.pgd import _project_linf, _samplewise_bounds


def a_pgd_attack(
    model: torch.nn.Module,
    x: torch.Tensor,
    y: torch.Tensor,
    eps: float,
    num_classes: int,
    loss_fn,
    n_steps: int = 20,
    step_size: float | None = None,
    random_start: bool = True,
    momentum: float = 0.9,
    x_bounds: tuple[torch.Tensor, torch.Tensor] | None = None,
    adaptation_rate: float = 0.1,
    checkpoints: list[float] | None = None,
    loss_log: list[float] | None = None,
) -> torch.Tensor:
    """Adaptive PGD attack.

    At each checkpoint fraction of total steps, evaluates whether the loss
    has improved. If it has, increases step size. If it has not, restarts
    from the best-so-far perturbation with halved step size.

    Args:
        model: Segmentation network (eval mode expected).
        x: Clean input ``[B, C, D, H, W]``.
        y: Ground-truth ``[B, 1, D, H, W]`` (int, -1 = ignore).
        eps: L∞ perturbation budget.
        num_classes: Number of classes.
        loss_fn: ``(logits, y, num_classes) -> scalar loss``.
        n_steps: Total PGD iterations.
        step_size: Initial step size (default: ``2 * eps / n_steps``).
        random_start: Random init within L∞ ball.
        momentum: Momentum factor for gradient accumulation.
        x_bounds: Optional ``(x_min, x_max)`` per-sample intensity bounds.
        adaptation_rate: Fraction by which step size is adjusted at checkpoints.
        checkpoints: Fractions of ``n_steps`` at which to adapt (e.g. ``[0.25, 0.5, 0.75]``).
        loss_log: If provided, append per-step loss values.

    Returns:
        Adversarial example tensor ``[B, C, D, H, W]``.
    """
    if eps == 0.0:
        return x.clone()

    if n_steps < 1:
        raise ValueError("n_steps must be >= 1 for A-PGD")

    if step_size is None:
        step_size = 2.0 * float(eps) / float(n_steps)

    if checkpoints is None:
        checkpoints = [0.25, 0.5, 0.75]

    checkpoint_steps = sorted({max(1, int(f * n_steps)) for f in checkpoints if 0.0 < f < 1.0})
    if checkpoint_steps and checkpoint_steps[-1] != n_steps:
        checkpoint_steps.append(n_steps)

    x_orig = x.detach()
    if x_bounds is not None:
        x_min, x_max = x_bounds
    else:
        x_min, x_max = _samplewise_bounds(x_orig)

    # Random start within L∞ ball
    if random_start:
        delta = (torch.rand_like(x_orig) * 2.0 - 1.0) * float(eps)
        x_adv = _project_linf(x_orig + delta, x_orig, eps, x_min, x_max)
    else:
        x_adv = x_orig.clone()

    saved_grad_states = [(p, p.requires_grad) for p in model.parameters()]
    for p, _ in saved_grad_states:
        p.requires_grad_(False)

    grad_accum = torch.zeros_like(x_orig)
    current_step_size = step_size
    best_x_adv = x_adv.clone()
    best_loss_val: float = float("-inf")

    try:
        for k in range(1, n_steps + 1):
            x_adv = x_adv.detach().requires_grad_(True)
            model.zero_grad(set_to_none=True)
            logits = model(x_adv)
            loss = loss_fn(logits, y, num_classes)
            loss_val = float(loss.detach().cpu())
            if loss_log is not None:
                loss_log.append(loss_val)
            loss.backward()
            grad = x_adv.grad
            if grad is None:
                return best_x_adv.clone()

            g = grad.detach()
            if momentum > 0.0:
                l1 = g.abs().mean(dim=tuple(range(1, g.ndim)), keepdim=True).clamp_min(1e-12)
                grad_accum = momentum * grad_accum + g / l1
                step_dir = grad_accum
            else:
                step_dir = g

            x_adv_pre = x_adv.detach()  # snapshot pre-step: loss was measured here
            x_adv = x_adv_pre + current_step_size * step_dir.sign()
            x_adv = _project_linf(x_adv, x_orig, eps, x_min, x_max)

            # Track best at the point where loss was measured (pre-update)
            if loss_val > best_loss_val:
                best_loss_val = loss_val
                best_x_adv = x_adv_pre.clone()

            # Adaptive step-size adjustment at checkpoints
            if k in checkpoint_steps and k < n_steps:
                with torch.no_grad():
                    current_loss = float(loss_fn(model(x_adv), y, num_classes).detach().cpu())

                if current_loss > best_loss_val * (1.0 + adaptation_rate):
                    # Loss improved — increase step size
                    current_step_size *= (1.0 + adaptation_rate)
                else:
                    # Loss stagnated or decreased — halve step and restart
                    current_step_size = max(current_step_size * 0.5, float(eps) / float(n_steps))
                    x_adv = best_x_adv.clone()
                    grad_accum.zero_()

    finally:
        for p, state in saved_grad_states:
            p.requires_grad_(state)

    # Final comparison: keep whichever has higher loss
    with torch.no_grad():
        final_loss = float(loss_fn(model(x_adv), y, num_classes).detach().cpu())
        best_final_loss = float(loss_fn(model(best_x_adv), y, num_classes).detach().cpu())
    return x_adv.detach() if final_loss >= best_final_loss else best_x_adv.detach()


def a_pgd_trajectory(
    model: torch.nn.Module,
    x: torch.Tensor,
    y: torch.Tensor,
    max_eps: float,
    num_classes: int,
    loss_fn,
    n_steps: int = 20,
    step_size: float | None = None,
    random_start: bool = True,
    momentum: float = 0.9,
    x_bounds: tuple[torch.Tensor, torch.Tensor] | None = None,
    adaptation_rate: float = 0.1,
    checkpoints: list[float] | None = None,
    snapshot_eps: list[float] | None = None,
    loss_log: list[float] | None = None,
) -> tuple[torch.Tensor, dict[float, torch.Tensor]]:
    """A-PGD with epsilon snapshots along the trajectory.

    Runs A-PGD at ``max_eps`` and projects snapshots to each ``snapshot_eps``.
    """
    if max_eps == 0.0:
        clean = x.clone()
        snaps = {0.0: clean.clone()} if snapshot_eps else {}
        return clean, snaps

    x_adv = a_pgd_attack(
        model, x, y, max_eps, num_classes, loss_fn,
        n_steps=n_steps, step_size=step_size, random_start=random_start,
        momentum=momentum, x_bounds=x_bounds,
        adaptation_rate=adaptation_rate, checkpoints=checkpoints,
        loss_log=loss_log,
    )

    x_orig = x.detach()
    if x_bounds is not None:
        x_min, x_max = x_bounds
    else:
        x_min, x_max = _samplewise_bounds(x_orig)

    snapshots: dict[float, torch.Tensor] = {}
    if snapshot_eps:
        for e in sorted(snapshot_eps):
            if e == 0.0:
                snapshots[0.0] = x_orig.clone()
            else:
                snapshots[e] = _project_linf(x_adv.clone(), x_orig, e, x_min, x_max)

    snapshots[float(max_eps)] = x_adv.clone()
    return x_adv, snapshots
