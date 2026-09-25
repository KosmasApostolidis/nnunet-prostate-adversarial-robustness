"""Adaptive-step PGD with a scheduled step size (the "improved step" arm).

A fifth attack arm for the whole-gland comparison. It keeps what the campaign's
adaptive-step PGD (``a_pgd_campaign.py``) does well and replaces the part that
does not work at a 20-iteration budget.

What the two existing attacks each get wrong
--------------------------------------------
``a_pgd_campaign`` starts at ``2*eps/k`` and can only ramp by 1.1x at three
checkpoints, so at eps=0.1, k=20 its step spans 0.010 to about 0.013 while the
useful scale is 0.2. It is 40x too fine and cannot climb out.

The original Auto-PGD (``auto_pgd.py``) fixes the step at ``2*eps`` and can only
halve. It saturates the ball's vertices from iteration 1, which is devastating at
large eps and useless at small eps, where it overshoots until enough halvings
have accumulated. Its step is monotonically non-increasing, so once it
over-shrinks it stays fine, with no floor to stop the collapse.

What this module does instead
-----------------------------
1. **Deterministic backbone.** The step follows a cosine decay from ``2*eps``
   down to ``eps/k`` across the run, so every budget sweeps the whole useful
   range of step sizes whether or not the adaptive rule ever fires. Both existing
   attacks are purely reactive, which is why neither adapts meaningfully inside
   20 iterations.
2. **Bidirectional adaptation with real range.** A multiplier on top of the
   backbone grows or shrinks by ``ADAPT_FACTOR`` at each checkpoint and is
   clamped to ``MULTIPLIER_BOUNDS``. Auto-PGD can only shrink; the campaign
   module's growth is too small to matter. The final step is clamped to
   ``[eps/k, 2*eps]``, giving a floor Auto-PGD lacks and a cap the campaign
   module lacks.
3. **Budget invariance.** Checkpoints are placed at fixed *fractions* of the
   run, so the schedule transfers between k=20 and k=100 unchanged. Auto-PGD's
   constants are tuned for k=100 and degenerate at k=20 (its checkpoints fire at
   4, 7, 9 and then every single iteration); the campaign module's step is tied
   to k through ``2*eps/k``.

Kept from ``a_pgd_campaign``: the gradient is normalized by its mean absolute
value before being accumulated with momentum, so the step size stays
interpretable in units of eps instead of being confounded with gradient
magnitude. That normalization is the campaign attack's genuine advantage over
Auto-PGD, which signs the raw gradient.

Fixed relative to ``a_pgd_campaign``: the best iterate is tracked against a
running maximum. The campaign module initializes its best loss to ``+inf`` so the
comparison never fires, which turns its "adaptation" into a monotone ramp.
Note that fixing that alone makes the attack *weaker* -- ``a_pgd.py`` does
exactly that and measures 0.611 against the campaign module's 0.518 on the
fold-0 probe, because with a working comparison the halving branch fires at
nearly every checkpoint and the step collapses. The fix is only useful paired
with the backbone and the cap above, which is why they ship together here.

This module is new work and is not part of any published result. Do not use it to
reproduce Table I -- that is ``a_pgd_campaign``'s job.
"""

from __future__ import annotations

import math

import torch

from mri_prostate_seg.attacks.pgd import _project_linf, _samplewise_bounds

# Checkpoint at every tenth of the run. Fractions rather than iteration counts is
# what makes the schedule independent of the budget.
CHECKPOINT_FRACTIONS = (0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9)
ADAPT_FACTOR = 1.2
MULTIPLIER_BOUNDS = (0.25, 4.0)
# Relative improvement in the running best loss required to count as progress.
IMPROVEMENT_TOL = 1e-3


def _checkpoint_steps(n_steps: int) -> list[int]:
    """1-based iteration indices at which the multiplier is updated."""
    steps = sorted(
        {
            max(1, min(n_steps - 1, int(round(f * n_steps))))
            for f in CHECKPOINT_FRACTIONS
            if 0.0 < f < 1.0
        }
    )
    return [s for s in steps if s < n_steps]


def _backbone_step(progress: float, eps: float, n_steps: int) -> float:
    """Cosine decay from 2*eps at the start to eps/n_steps at the end.

    ``progress`` runs 0 to 1 across the attack. The high end matches Auto-PGD's
    fixed step, so the run begins with the same reach; the low end matches the
    campaign module's floor, so it ends able to make fine corrections. Sweeping
    between them is what removes the need for the adaptive rule to find the right
    scale on its own inside a short budget.
    """
    high = 2.0 * eps
    low = eps / float(n_steps)
    return low + 0.5 * (high - low) * (1.0 + math.cos(math.pi * progress))


def apgd_updated_attack(
    model: torch.nn.Module,
    x: torch.Tensor,
    y: torch.Tensor,
    eps: float,
    num_classes: int,
    loss_fn,
    n_steps: int = 20,
    random_start: bool = True,
    momentum: float = 0.9,
    x_bounds: tuple[torch.Tensor, torch.Tensor] | None = None,
    adapt_factor: float = ADAPT_FACTOR,
    loss_log: list[float] | None = None,
    step_size_log: list[float] | None = None,
) -> torch.Tensor:
    """Run the updated adaptive-step PGD and return the highest-loss iterate.

    Args:
        model: Segmentation network (eval mode expected).
        x: Clean input ``[B, C, D, H, W]``.
        y: Ground truth ``[B, 1, D, H, W]``.
        eps: L-infinity perturbation budget.
        num_classes: Number of classes including background.
        loss_fn: ``(logits, y, num_classes) -> scalar loss`` to maximize.
        n_steps: Iteration budget. The schedule is budget-invariant.
        random_start: Uniform random init inside the ball, as the campaign
            attack does. Auto-PGD instead starts on the sphere.
        momentum: Decay for the normalized-gradient accumulator.
        x_bounds: Optional ``(x_min, x_max)`` per-sample intensity bounds.
        adapt_factor: Multiplier grow/shrink factor at each checkpoint.
        loss_log: If given, append the loss at every evaluated iterate.
        step_size_log: If given, append the step size used at every iteration.

    Returns:
        Adversarial example ``[B, C, D, H, W]``.
    """
    if eps == 0.0:
        return x.clone()
    if n_steps < 1:
        raise ValueError("n_steps must be >= 1 for scheduled A-PGD")

    x_orig = x.detach()
    if x_bounds is not None:
        x_min, x_max = x_bounds
    else:
        x_min, x_max = _samplewise_bounds(x_orig)

    if random_start:
        delta = (torch.rand_like(x_orig) * 2.0 - 1.0) * float(eps)
        x_adv = _project_linf(x_orig + delta, x_orig, eps, x_min, x_max)
    else:
        x_adv = x_orig.clone()

    saved_grad_states = [(p, p.requires_grad) for p in model.parameters()]
    for p, _ in saved_grad_states:
        p.requires_grad_(False)

    checkpoints = set(_checkpoint_steps(n_steps))
    floor = float(eps) / float(n_steps)
    cap = 2.0 * float(eps)
    multiplier = 1.0

    grad_accum = torch.zeros_like(x_orig)
    x_best = x_adv.clone()
    best_loss = -float("inf")
    best_loss_at_last_check = -float("inf")

    try:
        for i in range(1, n_steps + 1):
            progress = (i - 1) / float(n_steps)
            step_size = min(
                max(_backbone_step(progress, float(eps), n_steps) * multiplier, floor),
                cap,
            )
            if step_size_log is not None:
                step_size_log.append(step_size)

            x_adv = x_adv.detach().requires_grad_(True)
            model.zero_grad(set_to_none=True)
            loss = loss_fn(model(x_adv), y, num_classes)
            loss_val = float(loss.detach().cpu())
            if loss_log is not None:
                loss_log.append(loss_val)
            loss.backward()
            grad = x_adv.grad
            if grad is None:
                break
            grad = grad.detach()

            if loss_val > best_loss:
                best_loss = loss_val
                x_best = x_adv.detach().clone()

            with torch.no_grad():
                if momentum > 0.0:
                    scale = (
                        grad.abs()
                        .mean(dim=tuple(range(1, grad.ndim)), keepdim=True)
                        .clamp_min(1e-12)
                    )
                    grad_accum = momentum * grad_accum + grad / scale
                    direction = grad_accum
                else:
                    direction = grad
                x_adv = _project_linf(
                    x_adv.detach() + step_size * direction.sign(),
                    x_orig,
                    eps,
                    x_min,
                    x_max,
                )

                if i in checkpoints:
                    improved = (
                        best_loss > best_loss_at_last_check * (1.0 + IMPROVEMENT_TOL)
                        if math.isfinite(best_loss_at_last_check)
                        else True
                    )
                    if improved:
                        multiplier = min(
                            multiplier * adapt_factor, MULTIPLIER_BOUNDS[1]
                        )
                    else:
                        multiplier = max(
                            multiplier / adapt_factor, MULTIPLIER_BOUNDS[0]
                        )
                        # Only on a failed checkpoint do we give up the current
                        # trajectory; the backbone keeps shrinking regardless.
                        x_adv = x_best.clone()
                        grad_accum.zero_()
                    best_loss_at_last_check = best_loss
    finally:
        for p, state in saved_grad_states:
            p.requires_grad_(state)

    with torch.no_grad():
        final_loss = float(loss_fn(model(x_adv), y, num_classes).detach().cpu())
    return x_adv.detach() if final_loss >= best_loss else x_best.detach()


def apgd_updated_trajectory(
    model: torch.nn.Module,
    x: torch.Tensor,
    y: torch.Tensor,
    max_eps: float,
    num_classes: int,
    loss_fn,
    n_steps: int = 20,
    random_start: bool = True,
    momentum: float = 0.9,
    x_bounds: tuple[torch.Tensor, torch.Tensor] | None = None,
    adapt_factor: float = ADAPT_FACTOR,
    snapshot_eps: list[float] | None = None,
    loss_log: list[float] | None = None,
) -> tuple[torch.Tensor, dict[float, torch.Tensor]]:
    """Updated A-PGD at ``max_eps`` with L-infinity projections to smaller budgets.

    Mirrors ``a_pgd_campaign.a_pgd_trajectory`` so this arm uses the same
    shared-trajectory convention as every other arm in the comparison.
    """
    if max_eps == 0.0:
        clean = x.clone()
        snaps = {0.0: clean.clone()} if snapshot_eps else {}
        return clean, snaps

    x_adv = apgd_updated_attack(
        model,
        x,
        y,
        max_eps,
        num_classes,
        loss_fn,
        n_steps=n_steps,
        random_start=random_start,
        momentum=momentum,
        x_bounds=x_bounds,
        adapt_factor=adapt_factor,
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
