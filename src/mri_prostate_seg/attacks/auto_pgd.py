"""Original Auto-PGD (Croce and Hein, ICML 2020) ported to 3D segmentation.

Port of ``APGDAttack.attack_single_run``, Linf branch, from
``autoattack/autopgd_base.py`` in https://github.com/fra31/auto-attack. The
reference file this port was written against is vendored at
``docs/superpowers/specs/vendor/autopgd_base.py``.

This module exists to compare the published Auto-PGD against the repo's
adaptive-step PGD (``a_pgd_campaign.py``). The two are different attacks that
unfortunately share an abbreviation; nothing here should be used as a drop-in
for that module.

Preserved from the reference
----------------------------
* Step size initialized to ``2 * eps``, halved on a failed checkpoint, never
  increased, with no floor.
* Checkpoint schedule from the published constants: ``n_iter_2 = 0.22k``,
  ``size_decr = 0.03k``, ``n_iter_min = 0.06k``, with the interval shrinking
  after each checkpoint.
* Oscillation test: halve when at most ``rho`` of the last ``k`` iterations
  increased the loss, or when the step was not reduced at the previous
  checkpoint and the best loss has not improved since.
* On a reduction, restart from the best iterate and restore its gradient.
* Momentum applied to the *iterates*, not the gradient:
  ``x + (x_1 - x) * 0.75 + (x - x_old) * 0.25``.
* Random start on the eps-*sphere* (``x + eps * t / max|t|``), not uniform in
  the ball.
* Returns the iterate that attained the highest loss over the run.
* One gradient evaluation before the loop plus one per iteration, so a run of
  ``n_steps`` costs ``n_steps + 1`` gradients, as in the reference.

Deviations forced by the task
-----------------------------
1. Objective is the caller's ``loss_fn`` (Dice + cross-entropy in this repo).
   The reference DLR loss indexes the third-largest logit and is undefined for
   a two-logit head; plain cross-entropy is available but would make the arms
   differ in objective as well as in schedule.
2. The reference ``clamp(0, 1)`` image domain is replaced by the per-sample
   intensity clamp ``[x_min, x_max]`` the other attacks in this package use,
   because inputs are z-scored MRI volumes.
3. Batch size is one volume and the loss is a scalar, so the reference's
   per-sample bookkeeping degenerates to scalar tracking.
4. An optional ``x_init`` selects the starting iterate. It defaults to
   ``None``, in which case initialization is exactly the reference's, so no
   existing arm is affected. ``blade.py``'s epsilon ladder uses it to warm-start
   each radius from the one below.
5. ``n_steps`` defaults to 20 to match the campaign rather than the reference's
   100. The schedule degenerates at that budget: checkpoints fall at 4, 7, 9
   and then every iteration to 20, so the oscillation window in the tail is a
   single step. This is a property of running Auto-PGD on a short budget, not a
   porting choice.

One faithfully reproduced quirk: at the first checkpoint the reference reads
``loss_steps[j - k]`` with ``j - k == -1``, which wraps to the last, still
zero-initialized slot. ``_check_oscillation`` keeps that behaviour by indexing
a zero-initialized list the same way.
"""

from __future__ import annotations

import torch

from mri_prostate_seg.attacks.pgd import _project_linf, _samplewise_bounds


def _checkpoint_steps(n_steps: int) -> list[int]:
    """1-based iteration indices at which Auto-PGD adapts the step size.

    Mirrors the reference's ``counter3``/``k`` bookkeeping: the interval starts
    at ``n_iter_2`` and shrinks by ``size_decr`` after every checkpoint, with a
    lower bound of ``n_iter_min``.
    """
    interval = max(int(0.22 * n_steps), 1)
    size_decr = max(int(0.03 * n_steps), 1)
    interval_min = max(int(0.06 * n_steps), 1)

    steps: list[int] = []
    counter = 0
    for i in range(1, n_steps + 1):
        counter += 1
        if counter == interval:
            steps.append(i)
            interval = max(interval - size_decr, interval_min)
            counter = 0
    return steps


def _check_oscillation(loss_steps: list[float], j: int, k: int, rho: float) -> bool:
    """True when at most ``rho`` of the last ``k`` iterations increased the loss."""
    increased = sum(1.0 for c in range(k) if loss_steps[j - c] > loss_steps[j - c - 1])
    return increased <= k * rho


def auto_pgd_attack(
    model: torch.nn.Module,
    x: torch.Tensor,
    y: torch.Tensor,
    eps: float,
    num_classes: int,
    loss_fn,
    n_steps: int = 20,
    random_start: bool = True,
    x_bounds: tuple[torch.Tensor, torch.Tensor] | None = None,
    rho: float = 0.75,
    loss_log: list[float] | None = None,
    step_size_log: list[float] | None = None,
    x_init: torch.Tensor | None = None,
    iterate_callback=None,
) -> torch.Tensor:
    """Run the original Auto-PGD and return the highest-loss iterate.

    Args:
        model: Segmentation network (eval mode expected).
        x: Clean input ``[B, C, D, H, W]``.
        y: Ground truth ``[B, 1, D, H, W]``.
        eps: Linf perturbation budget.
        num_classes: Number of classes including background.
        loss_fn: ``(logits, y, num_classes) -> scalar loss`` to maximize.
        n_steps: Iteration budget. The published constants assume 100; see the
            module docstring for what changes at 20.
        random_start: Initialize on the eps-sphere, as the reference does.
        x_bounds: Optional ``(x_min, x_max)`` per-sample intensity bounds.
        rho: Oscillation threshold, the reference's ``thr_decr``.
        loss_log: If given, append the loss at every evaluated iterate.
        step_size_log: If given, append the step size in force at every step.
        x_init: Starting iterate, projected onto the eps-ball around ``x``
            before the run. Not part of the reference; when ``None`` (the
            default) the initialization is exactly the reference's, so every
            existing arm is unaffected. Used by the epsilon ladder in
            ``blade.py`` to warm-start a rung from the rung below it.
        iterate_callback: Optional ``(x_adv, logits, step, loss)`` observer
            called under ``no_grad`` after every forward pass (step 0 is the
            starting iterate). It sees each evaluated iterate at no extra
            gradient cost and cannot alter the trajectory; ``blade_updated``
            uses it to select candidates by Dice rather than by surrogate loss.

    Returns:
        Adversarial example ``[B, C, D, H, W]``.
    """
    if eps == 0.0:
        return x.clone()
    if n_steps < 1:
        raise ValueError("n_steps must be >= 1 for Auto-PGD")

    x_orig = x.detach()
    if x_bounds is not None:
        x_min, x_max = x_bounds
    else:
        x_min, x_max = _samplewise_bounds(x_orig)

    if x_init is not None:
        x_adv = _project_linf(x_init.detach().clone(), x_orig, eps, x_min, x_max)
    elif random_start:
        t = 2.0 * torch.rand_like(x_orig) - 1.0
        scale = t.abs().amax(dim=tuple(range(1, t.ndim)), keepdim=True).clamp_min(1e-12)
        x_adv = _project_linf(
            x_orig + float(eps) * t / scale, x_orig, eps, x_min, x_max
        )
    else:
        x_adv = x_orig.clone()

    saved_grad_states = [(p, p.requires_grad) for p in model.parameters()]
    for p, _ in saved_grad_states:
        p.requires_grad_(False)

    checkpoints = set(_checkpoint_steps(n_steps))
    intervals = _checkpoint_steps(n_steps)
    interval_by_step = {}
    previous = 0
    for step in intervals:
        interval_by_step[step] = step - previous
        previous = step

    step_size = 2.0 * float(eps)
    loss_steps = [0.0] * (n_steps + 1)

    try:
        # One gradient evaluation before the loop, as in the reference.
        x_adv = x_adv.detach().requires_grad_(True)
        model.zero_grad(set_to_none=True)
        logits = model(x_adv)
        loss = loss_fn(logits, y, num_classes)
        loss_val = float(loss.detach().cpu())
        if iterate_callback is not None:
            with torch.no_grad():
                iterate_callback(x_adv.detach(), logits.detach(), 0, loss_val)
        loss.backward()
        grad = x_adv.grad.detach().clone()
        del logits

        x_adv = x_adv.detach()
        x_adv_old = x_adv.clone()
        x_best = x_adv.clone()
        grad_best = grad.clone()
        best_loss = loss_val
        loss_best_last_check = best_loss
        reduced_last_check = True

        for i in range(1, n_steps + 1):
            if step_size_log is not None:
                step_size_log.append(step_size)

            with torch.no_grad():
                x_cur = x_adv
                grad2 = x_cur - x_adv_old
                x_adv_old = x_cur.clone()
                a = 0.75 if i > 1 else 1.0
                x_1 = _project_linf(
                    x_cur + step_size * grad.sign(), x_orig, eps, x_min, x_max
                )
                x_adv = _project_linf(
                    x_cur + (x_1 - x_cur) * a + grad2 * (1.0 - a),
                    x_orig,
                    eps,
                    x_min,
                    x_max,
                )

            x_adv = x_adv.detach().requires_grad_(True)
            model.zero_grad(set_to_none=True)
            logits = model(x_adv)
            loss = loss_fn(logits, y, num_classes)
            loss_val = float(loss.detach().cpu())
            if loss_log is not None:
                loss_log.append(loss_val)
            if iterate_callback is not None:
                with torch.no_grad():
                    iterate_callback(x_adv.detach(), logits.detach(), i, loss_val)
            loss.backward()
            grad = x_adv.grad.detach().clone()
            del logits
            x_adv = x_adv.detach()

            loss_steps[i] = loss_val
            if loss_val > best_loss:
                best_loss = loss_val
                x_best = x_adv.clone()
                grad_best = grad.clone()

            if i in checkpoints:
                k = interval_by_step[i]
                oscillating = _check_oscillation(loss_steps, i, k, rho)
                no_improvement = (
                    not reduced_last_check and loss_best_last_check >= best_loss
                )
                reduce_now = oscillating or no_improvement
                reduced_last_check = reduce_now
                loss_best_last_check = best_loss

                if reduce_now:
                    step_size /= 2.0
                    x_adv = x_best.clone()
                    grad = grad_best.clone()
    finally:
        for p, state in saved_grad_states:
            p.requires_grad_(state)

    return x_best.detach()


def auto_pgd_trajectory(
    model: torch.nn.Module,
    x: torch.Tensor,
    y: torch.Tensor,
    max_eps: float,
    num_classes: int,
    loss_fn,
    n_steps: int = 20,
    random_start: bool = True,
    x_bounds: tuple[torch.Tensor, torch.Tensor] | None = None,
    rho: float = 0.75,
    snapshot_eps: list[float] | None = None,
    loss_log: list[float] | None = None,
) -> tuple[torch.Tensor, dict[float, torch.Tensor]]:
    """Auto-PGD at ``max_eps`` with Linf projections down to smaller budgets.

    Mirrors ``a_pgd_campaign.a_pgd_trajectory`` so the new arm uses the same
    snapshot convention as the frozen campaign. Note that native Auto-PGD is a
    per-eps attack, since its step size is ``2 * eps``; running it this way
    keeps it comparable with the campaign's projected rows, and at ``max_eps``
    the two conventions coincide.
    """
    if max_eps == 0.0:
        clean = x.clone()
        snaps = {0.0: clean.clone()} if snapshot_eps else {}
        return clean, snaps

    x_adv = auto_pgd_attack(
        model,
        x,
        y,
        max_eps,
        num_classes,
        loss_fn,
        n_steps=n_steps,
        random_start=random_start,
        x_bounds=x_bounds,
        rho=rho,
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
