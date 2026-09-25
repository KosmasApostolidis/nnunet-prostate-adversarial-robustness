"""Auto-PGD+ : an attempt to repair Auto-PGD for short budgets. IT DID NOT WORK.

NEGATIVE RESULT -- do not use this attack. It is kept so the five changes below
are not proposed again; every one of them was measured and every one of them
made Auto-PGD weaker. Use ``auto_pgd.py``.

Evidence
--------
Ablation at native eps=0.04, fold 0, 12 cases, k=20, paired per-case seeds, with
real dynamic range (baseline Dice spans 0.125 to 0.822, no case saturated).
Lower Dice is a stronger attack; the paired delta is against Auto-PGD.

    Auto-PGD (baseline)          0.5969      --        --
    + backtracking               0.5950   -0.0019    7 better / 5 worse
    + oscillation window floor   0.6012   +0.0043    2 / 9
    + geometric backbone         0.6294   +0.0325    3 / 9
    all five (this module)       0.6294   +0.0325    3 / 9
    + gradient momentum          0.6360   +0.0391    1 / 11
    geometric + grad momentum    0.6870   +0.0901    0 / 12

Only backtracking is even neutral, and it is indistinguishable from zero
(t = -0.46). See ``results/autopgd_plus_ablation/``.

Why the diagnosis was wrong
---------------------------
A k=20 probe at eps=0.1 showed Auto-PGD still improving at the final iteration,
reaching only 3-5 of the 6 halvings needed to get from ``2*eps`` to ``eps/32``,
and spending 19% of its iterations moving downhill. Those observations are
correct. The inferences drawn from them were not:

* **Downhill iterations are the search, not waste.** Auto-PGD at k=100 is the
  most non-monotone configuration measured (31% downhill, worst single-step loss
  drop 2.34) and is also the only one that drives Dice to exactly 0.0000. The
  checkpoint gating that delays a reduction is deliberate hysteresis. Change 3
  suppressed it and bought nothing.
* **Step-range coverage is not the lever.** The geometric backbone raised the
  fraction of iterations below eps from 42% to 80% and lost 9 of 12 cases.
* **The eps=0.1 comparison that motivated this module was saturated.** On 8
  fold-0 cases at eps=0.1, six return Dice ~0.0000 for every arm; the entire
  between-arm difference came from one case. Rankings measured there -- including
  ``apgd_updated`` appearing strongest -- do not survive at an eps where the
  model has not already collapsed.

The five changes, all refuted
-----------------------------
1. Geometric step backbone ``2*eps -> eps/k``, replacing event-driven halving.
2. Multiplier bounded to ``MULTIPLIER_BOUNDS`` = (0.25, 1.0), shrink-biased.
3. Immediate backtracking on any iteration that loses loss.
4. ``x_adv_old`` reset on restart, so momentum does not carry from an abandoned
   trajectory.
5. ``loss_steps[0]`` seeded with the pre-loop loss instead of a zero sentinel,
   and the oscillation window floored at ``MIN_OSCILLATION_WINDOW``.

Changes 4 and 5 are faithful-port deviations that are defensible on their own
terms (the reference's stale ``x_adv_old`` and zero sentinel are genuine quirks),
but neither helps, and 5 measurably hurts.

Not part of any published result. Do not use it to reproduce Table I -- that is
``a_pgd_campaign``'s job.
"""

from __future__ import annotations

import torch

from mri_prostate_seg.attacks.pgd import _project_linf, _samplewise_bounds

# The multiplier may only hold the step at or below the geometric backbone.
MULTIPLIER_BOUNDS = (0.25, 1.0)
# Applied to the multiplier on a backtrack, and its inverse on a clean interval.
BACKTRACK_FACTOR = 0.5
RECOVER_FACTOR = 1.25
# Fewer than this many iterations in an oscillation window makes the test react
# to a single sample; the reference lets it fall to 1 at short budgets.
MIN_OSCILLATION_WINDOW = 2


def _backbone_step(i: int, n_steps: int, eps: float) -> float:
    """Geometric decay from ``2*eps`` at iteration 1 to ``eps/n_steps`` at ``n_steps``.

    Geometric rather than linear or cosine because the quantity that matters is
    the *scale* of the step, and a halving schedule -- which is what Auto-PGD is
    reaching for -- is geometric. Equal iterations per octave is what guarantees
    the fine regime is reached inside the budget.
    """
    high = 2.0 * float(eps)
    low = float(eps) / float(n_steps)
    if n_steps == 1:
        return high
    progress = (i - 1) / float(n_steps - 1)
    return high * (low / high) ** progress


def _checkpoint_steps(n_steps: int) -> list[int]:
    """Auto-PGD's checkpoint schedule with the oscillation window floored.

    The published constants (``0.22k`` interval, shrinking by ``0.03k``, floored
    at ``0.06k``) are kept so the checkpoint *placement* stays recognisable, but
    the interval floor is raised to ``MIN_OSCILLATION_WINDOW``.
    """
    interval = max(int(0.22 * n_steps), MIN_OSCILLATION_WINDOW)
    size_decr = max(int(0.03 * n_steps), 1)
    interval_min = max(int(0.06 * n_steps), MIN_OSCILLATION_WINDOW)

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


def auto_pgd_plus_attack(
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
) -> torch.Tensor:
    """Run Auto-PGD+ and return the highest-loss iterate.

    Signature mirrors ``auto_pgd.auto_pgd_attack`` so the two are drop-in
    comparable in the driver.

    Args:
        model: Segmentation network (eval mode expected).
        x: Clean input ``[B, C, D, H, W]``.
        y: Ground truth ``[B, 1, D, H, W]``.
        eps: Linf perturbation budget.
        num_classes: Number of classes including background.
        loss_fn: ``(logits, y, num_classes) -> scalar loss`` to maximize.
        n_steps: Iteration budget. The backbone is budget-invariant.
        random_start: Initialize on the eps-sphere, as Auto-PGD does.
        x_bounds: Optional ``(x_min, x_max)`` per-sample intensity bounds.
        rho: Oscillation threshold, the reference's ``thr_decr``.
        loss_log: If given, append the loss at every evaluated iterate.
        step_size_log: If given, append the step size used at every iteration.

    Returns:
        Adversarial example ``[B, C, D, H, W]``.
    """
    if eps == 0.0:
        return x.clone()
    if n_steps < 1:
        raise ValueError("n_steps must be >= 1 for Auto-PGD+")

    x_orig = x.detach()
    if x_bounds is not None:
        x_min, x_max = x_bounds
    else:
        x_min, x_max = _samplewise_bounds(x_orig)

    if random_start:
        t = 2.0 * torch.rand_like(x_orig) - 1.0
        scale = t.abs().amax(dim=tuple(range(1, t.ndim)), keepdim=True).clamp_min(1e-12)
        x_adv = _project_linf(x_orig + float(eps) * t / scale, x_orig, eps, x_min, x_max)
    else:
        x_adv = x_orig.clone()

    saved_grad_states = [(p, p.requires_grad) for p in model.parameters()]
    for p, _ in saved_grad_states:
        p.requires_grad_(False)

    checkpoints = _checkpoint_steps(n_steps)
    interval_by_step: dict[int, int] = {}
    previous = 0
    for step in checkpoints:
        interval_by_step[step] = step - previous
        previous = step
    checkpoint_set = set(checkpoints)

    floor = float(eps) / float(n_steps)
    cap = 2.0 * float(eps)
    multiplier = 1.0
    loss_steps = [0.0] * (n_steps + 1)

    try:
        # One gradient evaluation before the loop, as in Auto-PGD.
        x_adv = x_adv.detach().requires_grad_(True)
        model.zero_grad(set_to_none=True)
        loss = loss_fn(model(x_adv), y, num_classes)
        loss_val = float(loss.detach().cpu())
        loss.backward()
        grad = x_adv.grad.detach().clone()

        x_adv = x_adv.detach()
        x_adv_old = x_adv.clone()
        x_best = x_adv.clone()
        grad_best = grad.clone()
        best_loss = loss_val
        previous_loss = loss_val
        # Change 5: the pre-loop loss, not a zero sentinel.
        loss_steps[0] = loss_val
        loss_best_last_check = best_loss
        reduced_last_check = True
        backtracked_since_check = False

        for i in range(1, n_steps + 1):
            step_size = min(
                max(_backbone_step(i, n_steps, float(eps)) * multiplier, floor), cap
            )
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
            loss = loss_fn(model(x_adv), y, num_classes)
            loss_val = float(loss.detach().cpu())
            if loss_log is not None:
                loss_log.append(loss_val)
            loss.backward()
            grad = x_adv.grad.detach().clone()
            x_adv = x_adv.detach()

            loss_steps[i] = loss_val
            if loss_val > best_loss:
                best_loss = loss_val
                x_best = x_adv.clone()
                grad_best = grad.clone()

            # Change 3: react to a losing step now, not at the next checkpoint.
            if loss_val < previous_loss:
                multiplier = max(
                    multiplier * BACKTRACK_FACTOR, MULTIPLIER_BOUNDS[0]
                )
                x_adv = x_best.clone()
                grad = grad_best.clone()
                # Change 4: do not carry momentum from the abandoned trajectory.
                x_adv_old = x_adv.clone()
                backtracked_since_check = True
            previous_loss = loss_val

            if i in checkpoint_set:
                k = interval_by_step[i]
                oscillating = _check_oscillation(loss_steps, i, k, rho)
                no_improvement = (
                    not reduced_last_check and loss_best_last_check >= best_loss
                )
                reduce_now = oscillating or no_improvement
                reduced_last_check = reduce_now
                loss_best_last_check = best_loss

                if reduce_now:
                    multiplier = max(
                        multiplier * BACKTRACK_FACTOR, MULTIPLIER_BOUNDS[0]
                    )
                    x_adv = x_best.clone()
                    grad = grad_best.clone()
                    x_adv_old = x_adv.clone()
                elif not backtracked_since_check:
                    # Change 2: recover only toward the backbone, never above it.
                    multiplier = min(
                        multiplier * RECOVER_FACTOR, MULTIPLIER_BOUNDS[1]
                    )
                backtracked_since_check = False
    finally:
        for p, state in saved_grad_states:
            p.requires_grad_(state)

    return x_best.detach()


def auto_pgd_plus_trajectory(
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
    """Auto-PGD+ at ``max_eps`` with Linf projections down to smaller budgets.

    Mirrors ``auto_pgd.auto_pgd_trajectory`` so the two share a convention. The
    same caveat applies and more strongly: the step schedule is tied to ``eps``,
    so a snapshot projected from ``max_eps`` is not the attack this module would
    run natively at that eps. Drive the campaign without ``--shared-trajectory``
    to measure the native attack.
    """
    if max_eps == 0.0:
        clean = x.clone()
        snaps = {0.0: clean.clone()} if snapshot_eps else {}
        return clean, snaps

    x_adv = auto_pgd_plus_attack(
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
