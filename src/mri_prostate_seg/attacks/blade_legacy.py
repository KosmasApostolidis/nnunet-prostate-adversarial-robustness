"""BLADE: a boundary-weighted epsilon-ladder ensemble attack for 3D segmentation.

BLADE is assembled from what the ten-arm whole-gland comparison in this
repository actually measured, and from what the Auto-PGD+ ablation refuted. It
changes nothing inside the optimiser; it is two wrappers and two objectives.

Three components
----------------
1. **The epsilon ladder** (``ladder_trajectory``). Every other trajectory in
   this package runs once at ``max_eps`` and projects the result down to the
   smaller reporting budgets. That understates every arm whose step size is
   tied to epsilon: Auto-PGD fixes its step at ``2*eps``, and a fold-0 probe
   measured native Auto-PGD at eps=0.04 scoring 0.63 Dice against 0.91 for the
   projection of the same run. The ladder instead optimises at the smallest
   rung, emits that snapshot, expands the ball to the next rung and *continues
   from the warm iterate*, and so on to ``max_eps``. Every snapshot is
   natively optimised inside its own ball, with the step correctly scaled for
   that rung. The warm point from rung ``r-1`` lies inside the ball of rung
   ``r``, so it is always feasible.

   Cost: ``len(rungs) * n_steps`` gradients, the same as running the attack
   cold once per epsilon -- the ladder pays no premium over a native per-eps
   campaign and gets a warm start for free.

   Relation to SEA's "red-eps" schedule: that one *descends* the radius
   (2eps -> 1.5eps -> eps) inside a single reported epsilon and reports one
   number. This one *ascends* across the reporting grid and reports every rung.

2. **The boundary-weighted objective** (``boundary_weighted_loss``). The
   campaign reports Dice, HD95 and ASD, and no arm in it optimises anything
   but region overlap. This is the boundary loss of Kervadec et al. -- the
   integral of the signed distance transform of the ground truth against the
   predicted foreground probability -- used with its sign flipped: what a
   robust segmenter is trained to minimise, the attack maximises. It rewards
   foreground probability placed far *outside* the gland, which is exactly the
   configuration that inflates the 95th-percentile Hausdorff distance, and
   suppresses it deep inside, which inflates the average surface distance.

3. **The frontier-margin objective** (``frontier_margin_loss``). DAG's active
   set (voxels the current iterate still classifies correctly) restricted to
   the ``k_frac`` fraction with the *smallest* logit margin -- the ones nearest
   to flipping. Greedy in flipped voxels per unit of budget, and complementary
   to the mass-driven soft-Dice term in ``dice_ce_loss``.

The ensemble (``blade_trajectory``) runs one ladder per objective and keeps,
per rung, the candidate with the lowest foreground Dice -- SEA's selection
rule, which is the one mechanism in the ten-arm comparison that demonstrably
paid (SEA is the strongest arm at eps=0.06).

What is deliberately NOT done
-----------------------------
Nothing touches Auto-PGD's inner loop. ``auto_pgd_plus.py`` measured five
plausible changes to it -- geometric step backbone, bounded multiplier,
immediate backtracking, momentum reset, oscillation-window floor -- and every
one made the attack weaker. Its non-monotone search is the engine, not a
defect. BLADE only chooses where each run starts and which objective it
ascends.

Budget accounting, so the comparison is matched
-----------------------------------------------
* ``BLADE-1`` (one objective): ``R * n_steps`` gradients per case, against a
  native per-epsilon single-objective arm's ``R * (n_steps + 1)``.
* ``BLADE-3`` (three objectives): three times that, against native per-epsilon
  SEA, which is also three Auto-PGD runs per epsilon.

Report whichever was run against the matched comparator.
"""

from __future__ import annotations

import numpy as np
import torch
import torch.nn.functional as F
from scipy.ndimage import distance_transform_edt

from mri_prostate_seg.attacks.auto_pgd import auto_pgd_attack
from mri_prostate_seg.attacks.pgd import _samplewise_bounds
from mri_prostate_seg.attacks.sea import foreground_dice

IGNORE_LABEL = -1
BLADE_LOSSES = ("dice_ce", "boundary", "frontier")
# Fraction of the still-correct voxels the frontier objective attacks.
DEFAULT_K_FRAC = 0.1


def signed_distance_map(target: torch.Tensor) -> torch.Tensor:
    """Signed Euclidean distance to the ground-truth foreground surface.

    Negative inside the gland, positive outside, magnitude in voxels. Ignore
    voxels are excluded from the foreground and are given zero weight, so they
    cannot contribute to the boundary objective.

    Args:
        target: ``[B, 1, D, H, W]`` labels.

    Returns:
        ``[B, 1, D, H, W]`` float map on ``target``'s device.
    """
    labels = target.squeeze(1).long()
    valid = labels != IGNORE_LABEL
    foreground = (labels >= 1) & valid

    fg_np = foreground.detach().cpu().numpy()
    valid_np = valid.detach().cpu().numpy()
    phi_np = np.zeros(fg_np.shape, dtype=np.float32)
    for b in range(fg_np.shape[0]):
        fg = fg_np[b]
        if not fg.any() or fg.all():
            # No surface exists; every voxel is equidistant from nothing.
            continue
        outside = distance_transform_edt(~fg)
        inside = distance_transform_edt(fg)
        phi_np[b] = (outside - inside).astype(np.float32)
    phi_np *= valid_np.astype(np.float32)

    phi = torch.from_numpy(phi_np).to(device=target.device, dtype=torch.float32)
    return phi.unsqueeze(1)


def boundary_weighted_loss(
    logits: torch.Tensor,
    target: torch.Tensor,
    num_classes: int,
    phi: torch.Tensor | None = None,
) -> torch.Tensor:
    """Kervadec's boundary loss with the sign flipped, as an attack objective.

        L = sum_i phi_i * p_fg(i) / sum_i |phi_i|

    ``phi`` is the signed distance map: positive outside the gland, negative
    inside. Ascending ``L`` moves foreground probability outward -- far
    false positives inflate HD95, and suppressed interior inflates ASD.

    Args:
        logits: ``[B, C, D, H, W]`` model output on the current iterate.
        target: ``[B, 1, D, H, W]`` labels.
        num_classes: Number of classes including background.
        phi: Precomputed signed distance map. Supply it once per case; the
            transform depends only on the ground truth, so recomputing it at
            every iteration is pure waste.
    """
    del num_classes
    if phi is None:
        phi = signed_distance_map(target)
    probabilities = F.softmax(logits, dim=1)
    p_fg = probabilities[:, 1:].sum(dim=1, keepdim=True)
    denominator = phi.abs().sum().clamp_min(1e-12)
    return (phi * p_fg).sum() / denominator


def frontier_margin_loss(
    logits: torch.Tensor,
    target: torch.Tensor,
    num_classes: int,
    k_frac: float = DEFAULT_K_FRAC,
) -> torch.Tensor:
    """Negative mean logit margin over the still-correct voxels closest to flipping.

    The margin of a voxel is ``logit(true) - max_{c != true} logit(c)``; it is
    positive exactly where the iterate is still correct. Ascending the negative
    of the ``k_frac`` smallest margins spends the budget on the voxels one
    small push from the decision surface, which maximises flipped voxels per
    unit of perturbation. Voxels already wrong, and ignore voxels, are excluded:
    a flipped voxel has nothing left to give.

    Returns a differentiable zero when every valid voxel is already wrong.
    """
    labels = target.squeeze(1).long()
    valid = labels != IGNORE_LABEL
    safe = torch.where(valid, labels, torch.zeros_like(labels))

    true_logit = logits.gather(1, safe.unsqueeze(1)).squeeze(1)
    masked = logits.scatter(
        1, safe.unsqueeze(1), torch.full_like(true_logit.unsqueeze(1), -float("inf"))
    )
    other_logit = masked.amax(dim=1)
    margin = true_logit - other_logit

    correct = (logits.detach().argmax(dim=1) == labels) & valid
    n_correct = int(correct.sum())
    if n_correct == 0:
        return logits.sum() * 0.0

    active = margin[correct]
    k = max(1, min(n_correct, int(k_frac * n_correct)))
    smallest, _ = torch.topk(active, k, largest=False)
    return -smallest.mean()


def ladder_trajectory(
    model: torch.nn.Module,
    x: torch.Tensor,
    y: torch.Tensor,
    max_eps: float,
    num_classes: int,
    loss_fn,
    n_steps: int,
    random_start: bool = True,
    x_bounds: tuple[torch.Tensor, torch.Tensor] | None = None,
    snapshot_eps: list[float] | None = None,
    loss_log: list[float] | None = None,
    rho: float = 0.75,
    init_log: list[torch.Tensor | None] | None = None,
) -> tuple[torch.Tensor, dict[float, torch.Tensor]]:
    """Ascending-epsilon Auto-PGD: one native run per rung, warm-started.

    Args:
        loss_fn: ``(logits, y, num_classes) -> scalar`` to maximize.
        n_steps: Iterations *per rung*, so the run costs ``R * n_steps``
            gradients for ``R`` rungs.
        snapshot_eps: The reporting grid. Positive entries become rungs;
            ``max_eps`` is always the last one. ``0.0`` returns the clean
            volume.
        init_log: If given, receives the warm point used to start each rung
            (``None`` for the first).

    Returns:
        ``(x_adv, snapshots)`` with the same contract as ``pgd_trajectory``.
    """
    if max_eps == 0.0:
        clean = x.clone()
        snaps = {0.0: clean.clone()} if snapshot_eps else {}
        return clean, snaps
    if n_steps < 1:
        raise ValueError("n_steps must be >= 1 for the epsilon ladder")

    x_orig = x.detach()
    if x_bounds is None:
        x_bounds = _samplewise_bounds(x_orig)

    requested = sorted({float(e) for e in (snapshot_eps or [])})
    snapshots: dict[float, torch.Tensor] = {}
    if 0.0 in requested:
        snapshots[0.0] = x_orig.clone()
    rungs = sorted({e for e in requested if 0.0 < e <= float(max_eps)} | {float(max_eps)})

    warm: torch.Tensor | None = None
    for rung in rungs:
        if init_log is not None:
            init_log.append(warm)
        x_adv = auto_pgd_attack(
            model,
            x,
            y,
            rung,
            num_classes,
            loss_fn,
            n_steps=n_steps,
            random_start=random_start and warm is None,
            x_bounds=x_bounds,
            rho=rho,
            loss_log=loss_log,
            x_init=warm,
        )
        snapshots[rung] = x_adv.clone()
        warm = x_adv

    return snapshots[float(max_eps)], snapshots


def blade_trajectory(
    model: torch.nn.Module,
    x: torch.Tensor,
    y: torch.Tensor,
    max_eps: float,
    num_classes: int,
    base_loss_fn,
    n_steps: int,
    random_start: bool = True,
    x_bounds: tuple[torch.Tensor, torch.Tensor] | None = None,
    snapshot_eps: list[float] | None = None,
    loss_log: list[float] | None = None,
    losses: tuple[str, ...] = BLADE_LOSSES,
    k_frac: float = DEFAULT_K_FRAC,
    chosen_log: dict[float, str] | None = None,
) -> tuple[torch.Tensor, dict[float, torch.Tensor]]:
    """One epsilon ladder per objective; per rung, keep the worst foreground Dice.

    Args:
        base_loss_fn: The ``dice_ce`` member, supplied by the caller so this
            package does not fork the driver's definition of it.
        losses: Subset of ``BLADE_LOSSES`` to ensemble. A single-element tuple
            gives the budget-matched BLADE-1.
        chosen_log: If given, maps each rung to the winning objective's name.
    """
    if max_eps == 0.0:
        clean = x.clone()
        snaps = {0.0: clean.clone()} if snapshot_eps else {}
        return clean, snaps
    if n_steps < 1:
        raise ValueError("n_steps must be >= 1 for BLADE")
    unknown = [name for name in losses if name not in BLADE_LOSSES]
    if unknown:
        raise ValueError(f"unknown BLADE loss(es): {unknown}")

    x_orig = x.detach()
    if x_bounds is None:
        x_bounds = _samplewise_bounds(x_orig)

    phi = signed_distance_map(y) if "boundary" in losses else None
    loss_fns = {
        "dice_ce": base_loss_fn,
        "boundary": lambda lg, t, nc: boundary_weighted_loss(lg, t, nc, phi=phi),
        "frontier": lambda lg, t, nc: frontier_margin_loss(lg, t, nc, k_frac=k_frac),
    }

    candidates: dict[str, dict[float, torch.Tensor]] = {}
    for i, name in enumerate(losses):
        _, snaps = ladder_trajectory(
            model,
            x,
            y,
            max_eps,
            num_classes,
            loss_fns[name],
            n_steps=n_steps,
            random_start=random_start,
            x_bounds=x_bounds,
            snapshot_eps=snapshot_eps,
            loss_log=loss_log if i == 0 else None,
            init_log=None,
        )
        candidates[name] = snaps

    rungs = sorted(set().union(*(set(s) for s in candidates.values())))
    snapshots: dict[float, torch.Tensor] = {}
    with torch.no_grad():
        for rung in rungs:
            if rung == 0.0:
                snapshots[0.0] = x_orig.clone()
                continue
            best_name, best_dice, best_x = None, float("inf"), None
            for name in losses:
                xe = candidates[name][rung]
                d = foreground_dice(model(xe), y, num_classes)
                if d < best_dice:
                    best_name, best_dice, best_x = name, d, xe
            snapshots[rung] = best_x.clone()
            if chosen_log is not None:
                chosen_log[rung] = best_name

    return snapshots[float(max_eps)], snapshots


def blade_attack(
    model: torch.nn.Module,
    x: torch.Tensor,
    y: torch.Tensor,
    eps: float,
    num_classes: int,
    base_loss_fn,
    n_steps: int,
    x_bounds: tuple[torch.Tensor, torch.Tensor] | None = None,
    losses: tuple[str, ...] = BLADE_LOSSES,
) -> torch.Tensor:
    """Single-epsilon BLADE: the ladder degenerates to one rung at ``eps``."""
    if eps == 0.0:
        return x.clone()
    x_adv, _ = blade_trajectory(
        model, x, y, eps, num_classes, base_loss_fn, n_steps,
        x_bounds=x_bounds, losses=losses,
    )
    return x_adv


__all__ = [
    "BLADE_LOSSES",
    "blade_attack",
    "blade_trajectory",
    "boundary_weighted_loss",
    "frontier_margin_loss",
    "ladder_trajectory",
    "signed_distance_map",
]
