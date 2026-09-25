"""ALMA prox: the minimum-norm segmentation attack of Rony, Pesquet, Ben Ayed.

Reference: J. Rony, J.-C. Pesquet, I. Ben Ayed, "Proximal Splitting Adversarial
Attack for Semantic Segmentation", CVPR 2023, arXiv:2206.07179. The formulation
below follows the paper's Eqs. 1, 3, 5-8 and 17 and its Algorithm 1, verified
against the PDF, and the authors' implementation
``adv_lib/attacks/segmentation/alma_prox.py`` (github.com/jeromerony/
adversarial-library), which is what github.com/jeromerony/alma_prox_segmentation
calls.

Every other arm in this repository maximises damage inside a fixed L-infinity
ball. ALMA prox does the opposite -- it minimises the perturbation norm subject
to the image being misclassified:

    min_d ||d||_inf   s.t.  argmax_k f(x + d)_{k,i} != y_i, i = 1..d,
                            x + d in X                                       (1)

The d non-differentiable misclassification constraints are replaced by smooth
ones on the difference-of-logits ratio,

    min_d ||d||_inf   s.t.  DLR+(f(x + d)_i, y_i) + eps <= 0, i = 1..d,
                            d in Delta                                       (3)

and (3) is solved by an augmented Lagrangian whose nonsmooth part is handled by
a proximal step instead of a penalty:

    h1(d) = ||d||_inf + iota_Delta(d)                                        (5)
    h2(d) = (m~^(t))^T P(w^(t) c^(t), rho^(t), mu^(t))                       (7)
    d^(t+1) = prox^H_{lam h1}( d^(t) - lam H^-1 grad h2(d^(t)) )            (17)

with

* ``P`` the Penalty-Lagrangian P2 of the ALMA paper (arXiv:2011.11857). Its
  derivative in the constraint gives the multiplier update, smoothed by an
  exponential moving average with weight ``alpha``.
* ``rho`` multiplied by ``gamma = 2`` every ``check_steps`` iterations (``M`` in
  the article) for the voxels whose constraint has not improved by the factor
  ``tau`` and which have not been adversarial since the last check.
* a constraint scale ``w``, adapted multiplicatively by ``scale_gamma`` so that
  the adversarial pixel fraction tracks ``nu = adv_threshold``                (8)
* constraint masking: the largest -- least satisfiable -- constraints are
  discarded, and the discarded fraction *grows* linearly from none at the first
  iteration to ``1 - nu`` at the last, which is Eq. 6 read as a percentile
  falling from 1 to ``nu``. The flag is called ``mask_decay`` after the
  reference, whose own docstring describes it backwards ("linearly decrease the
  number of discarded constraints"); the code there grows it, as here.        (6)
* ``H`` an RMSProp-style diagonal metric, and ``prox^H`` solved by the ternary
  search of Algorithm 1, since the prox of the L-infinity norm plus the box
  indicator has no closed form.

What is deliberately NOT taken from the paper
---------------------------------------------
* The paper attacks [0, 1] images, so its feasible set is ``Delta = [-x, 1-x]``.
  Our inputs are z-scored MRI volumes; ``Delta`` is built from the per-sample
  intensity bounds (``x_bounds``), the same clamp every other arm here uses.
* ``DLR+`` normalises the logit margin by ``top1 - top3``, which needs at least
  three classes. The whole-gland model has two. Taking ``top1 - top2`` instead
  would make the normaliser ``|margin|``, so ``DLR+ ~ 1 + eps/margin`` on correct
  voxels, whose derivative ``-eps/margin^2`` is ~1e-4: the attack would barely
  move. For fewer than three classes the constraint is therefore the plain
  difference of logits (``margin + eps <= 0``); the adaptive scale ``w`` absorbs
  the missing normalisation. The zones model (three classes) uses ``DLR+``
  unchanged.
* The campaign reports Dice/HD95/ASD on a fixed epsilon grid, and a minimum-norm
  attack has no epsilon. Each requested budget keeps the iterate with the lowest
  foreground Dice among those satisfying ``||d||_inf <= eps`` -- the usual way a
  fixed-budget row is read off a minimum-norm run. Two consequences worth
  stating when the numbers are quoted: above the minimal norm the attack does
  not spend the extra budget, so the row flattens; below it the row is a
  pre-convergence iterate, not a converged attack at that budget. The minimal
  norm itself -- ALMA prox's native output -- is available through
  ``min_norm_log``.
* The paper constrains every labelled pixel, which on a 2-D street scene is a
  reasonable set. On a 3-D prostate volume the gland is a few percent of the
  voxels, so requiring the paper's ``nu = 99%`` of all labelled voxels to be
  misclassified asks for nearly the whole background to flip as well: measured
  on fold 0, the unrestricted attack reaches 5.7% at ``||delta||_inf = 0.36``,
  and the minimal norm is never attained. ``constraint_band_mm`` therefore
  restricts the constraints to the foreground dilated by that many
  millimetres, so ``nu`` refers to the gland and the tissue around it -- the
  region whose misclassification is what "the segmentation was destroyed"
  means here. Even so, the band-restricted attack peaks at 94-99% within the
  budget, and at ``nu = 0.95`` one whole-gland case in three still had no
  minimal norm, so the arm is configured with ``nu = 0.90`` rather than the
  paper's 0.99 (see ``ALMA_ADV_THRESHOLD`` in the driver); note that ``nu``
  also sets the
  ``1 - nu`` fraction that constraint masking discards (Eq. 6), so loosening it
  loosens both, as in the reference. The snapshot selection is unaffected: it
  still scores foreground Dice over every labelled voxel. ``None`` restores the
  paper's constraint set.
* ``num_steps`` is the paper's 500. Fewer than about a hundred steps is not
  meaningful here: ``check_steps``, the mask decay and the learning-rate decay
  all assume hundreds of iterations, so the campaign's ``n_steps = 20`` would
  measure a crippled attack rather than ALMA prox.
* The reference decorates the prox with ``@torch.compile``; the pinned stack is
  torch 2.0.1, where that is unreliable, so it is dropped.
* Two small departures from the reference, both unreachable at the batch size
  this driver uses (1): a sample whose constrained region is empty scores
  ``adv_percent = 0`` rather than ``NaN``; and when constraint masking selects
  ``k = 0`` for one sample while another has ``k > 0``, nothing is discarded
  for that sample, where the reference's ``k - 1`` index would wrap to ``-1``
  and discard its ``max(k)`` largest constraints.
* ALMA prox starts at ``d = 0`` and is deterministic. ``random_start`` is
  accepted for signature compatibility and ignored, and restarts would all
  return the same perturbation, so only one run is ever performed.
"""

from __future__ import annotations

import math
from typing import Any

import torch
import torch.nn.functional as F

from mri_prostate_seg.attacks.blade_updated import foreground_dice_per_sample
from mri_prostate_seg.attacks.pgd import _project_linf, _samplewise_bounds

IGNORE_LABEL = -1
DEFAULT_NUM_STEPS = 500


def prox_linf_indicator(
    delta: torch.Tensor,
    lam: torch.Tensor,
    lower: torch.Tensor,
    upper: torch.Tensor,
    metric: torch.Tensor | None = None,
    tol: float = 1e-6,
    section: float = 1.0 / 3.0,
) -> torch.Tensor:
    """Proximity operator of ``lam*||.||_inf + iota_Lambda`` in the metric H.

    Algorithm 1 of the paper: the minimiser of
    ``1/2*||p - delta||_H^2 + lam*||p||_inf`` over the box ``Lambda`` is the
    box-projected ``delta`` clamped to some level ``beta >= 0``, and ``beta`` is
    found by a ternary search on the (unimodal) objective. ``section = 1/3``
    rather than the golden ratio, as in the reference: slightly slower, more
    stable numerically.

    Args:
        lam: ``[B]`` per-sample weight of the L-infinity term.
        lower, upper: bounds of ``Lambda``, broadcastable to ``delta``.
        metric: ``[B, ...]`` diagonal metric ``H``; identity if None.
        tol: absolute error on the level ``beta``.
    """
    shape = delta.shape
    delta = delta.flatten(1)
    lam = 2.0 * lam
    metric = metric.flatten(1) if metric is not None else None
    projected = delta.clamp(min=lower.flatten(1), max=upper.flatten(1))

    right = projected.norm(p=float("inf"), dim=1)
    left = torch.zeros_like(right)
    span = float(right.max().clamp_min(tol))
    steps = int(math.ceil(math.log(tol / span) / math.log(1.0 - section)))

    def objective(level: torch.Tensor) -> torch.Tensor:
        clamped = projected.clamp(min=-level.unsqueeze(1), max=level.unsqueeze(1))
        squared = (clamped - delta).square()
        if metric is not None:
            squared = squared * metric
        return squared.sum(dim=1) + level * lam

    for _ in range(max(steps, 0)):
        left_third = torch.lerp(left, right, weight=section)
        right_third = torch.lerp(left, right, weight=1.0 - section)
        keep_right = objective(left_third) >= objective(right_third)
        left = torch.where(keep_right, left_third, left)
        right = torch.where(keep_right, right, right_third)

    level = torch.lerp(left, right, weight=0.5)
    return projected.clamp(min=-level.unsqueeze(1), max=level.unsqueeze(1)).view(shape)


class _PenaltyP2(torch.autograd.Function):
    """Penalty-Lagrangian P2 of the ALMA paper, and its derivative in ``y``.

    Eq. 21 of arXiv:2011.11857: ``P(y, rho, mu) = mu*y + mu*rho*y^2 +
    rho^2*y^3/6`` for ``y >= 0`` and ``mu*y / (1 - rho*y)`` otherwise. The
    backward pass is the derivative with respect to the constraint only;
    ``rho`` and ``mu`` are parameters, not variables, which is also how the
    multiplier update reads ``P'`` off this function.

    Two notes on the reference implementation (``adv_lib``):

    * its forward clamps ``rho`` to ``>= 1`` in the negative branch, which the
      paper does not. With ``rho_init = 0.01`` that is a large difference, and
      it is the code that produced the published segmentation numbers, so the
      clamp is kept.
    * its backward returns ``mu*y + 2*mu*rho*y + rho^2*y^2/2`` for ``y >= 0``.
      That is not the derivative of its own forward: the first term should be
      ``mu``, not ``mu*y``. As written it is discontinuous at ``y = 0`` (0 from
      above, ``mu`` from below) and breaks the defining property that the
      multiplier update ``mu <- P'(0)`` returns ``mu``. The correct derivative
      is used here; ``tests/unit/test_alma_prox.py`` checks it with
      ``torch.autograd.gradcheck``.
    """

    @staticmethod
    def forward(
        ctx: Any, y: torch.Tensor, rho: torch.Tensor, mu: torch.Tensor
    ) -> torch.Tensor:
        positive = y >= 0
        ctx.save_for_backward(y, rho, mu, positive)
        upper = mu * y + mu * rho * y**2 + rho**2 * y**3 / 6.0
        lower = mu * y / (1.0 - rho.clamp(min=1) * y.clamp(max=0))
        return torch.where(positive, upper, lower)

    @staticmethod
    def backward(
        ctx: Any, grad_output: torch.Tensor
    ) -> tuple[torch.Tensor, None, None]:
        y, rho, mu, positive = ctx.saved_tensors
        upper = mu + 2.0 * mu * rho * y + rho**2 * y**2 / 2.0
        lower = mu / (1.0 - rho.clamp(min=1) * y.clamp(max=0)).square()
        return grad_output * torch.where(positive, upper, lower), None, None


def difference_of_logits(
    logits: torch.Tensor, labels: torch.Tensor, labels_infhot: torch.Tensor
) -> torch.Tensor:
    """``f_y - max_{k != y} f_k``: positive exactly where the voxel is correct."""
    true_logits = logits.gather(1, labels.unsqueeze(1)).squeeze(1)
    other_logits = (logits - labels_infhot).amax(dim=1)
    return true_logits - other_logits


def constraint_values(
    logits: torch.Tensor,
    labels: torch.Tensor,
    labels_infhot: torch.Tensor,
    num_classes: int,
    tolerance: float,
) -> torch.Tensor:
    """``DLR+ + tolerance`` (Eq. 3), or the raw margin when ``DLR+`` degenerates.

    With fewer than three classes the ``top1 - top3`` normaliser of ``DLR+`` does
    not exist and its two-class substitute kills the gradient (see the module
    docstring), so the unnormalised difference of logits is used instead.
    """
    margin = difference_of_logits(logits, labels, labels_infhot) + tolerance
    if num_classes < 3:
        return margin
    top3 = logits.topk(k=3, dim=1).values
    return margin / (top3[:, 0] - top3[:, 2] + 1e-8)


def foreground_band(
    labels: torch.Tensor,
    band_mm: float,
    spacing: tuple[float, float, float] | None = None,
    ignore_label: int = IGNORE_LABEL,
) -> torch.Tensor:
    """Foreground dilated by ``band_mm``, as a ``[B, D, H, W]`` boolean mask.

    The dilation is isotropic in millimetres: the per-axis voxel radius is
    ``round(band_mm / spacing_axis)``, so on the (3.0, 0.5, 0.5) mm grid used by
    both models a 5 mm band reaches 2 slices through-plane and 10 voxels
    in-plane. Without ``spacing`` the radius is ``round(band_mm)`` voxels on
    every axis. Implemented as a max-pool, which is a dilation by a rectangular
    structuring element.
    """
    radii = [
        max(0, int(round(band_mm / s)))
        for s in (spacing if spacing is not None else (1.0, 1.0, 1.0))
    ]
    foreground = ((labels >= 1) & (labels != ignore_label)).to(torch.float32)
    dilated = F.max_pool3d(
        foreground.unsqueeze(1),
        kernel_size=[2 * r + 1 for r in radii],
        stride=1,
        padding=radii,
    )
    return dilated.squeeze(1) > 0


def alma_prox_trajectory(
    model: torch.nn.Module,
    x: torch.Tensor,
    y: torch.Tensor,
    max_eps: float,
    num_classes: int,
    n_steps: int = DEFAULT_NUM_STEPS,
    step_size: float | None = None,
    random_start: bool = False,
    x_bounds: tuple[torch.Tensor, torch.Tensor] | None = None,
    snapshot_eps: list[float] | None = None,
    loss_log: list[float] | None = None,
    *,
    adv_threshold: float = 0.99,
    lr_init: float = 0.001,
    lr_reduction: float = 0.1,
    mu_init: float = 1.0,
    rho_init: float = 0.01,
    check_steps: int = 10,
    tau: float = 0.95,
    gamma: float = 2.0,
    alpha: float = 0.8,
    scale_min: float = 0.1,
    scale_max: float = 1.0,
    scale_init: float = 1.0,
    scale_gamma: float = 0.02,
    logit_tolerance: float = 1e-4,
    constraint_masking: bool = True,
    mask_decay: bool = True,
    ignore_label: int = IGNORE_LABEL,
    constraint_band_mm: float | None = None,
    spacing: tuple[float, float, float] | None = None,
    min_norm_log: list[float] | None = None,
) -> tuple[torch.Tensor, dict[float, torch.Tensor]]:
    """ALMA prox with the campaign's snapshot convention.

    Args:
        max_eps: Largest reported budget. It does not constrain the search --
            ALMA prox minimises the norm -- it only selects the returned iterate.
        n_steps: ALMA prox's ``num_steps``; the paper uses 500.
        step_size, random_start: accepted for signature compatibility, ignored.
        snapshot_eps: budgets to report. Each one receives the lowest-Dice
            iterate whose perturbation fits inside it.
        loss_log: if given, the augmented-Lagrangian objective per iteration.
        constraint_band_mm: if given, only the foreground dilated by this many
            millimetres carries a constraint (see the module docstring).
        spacing: ``(D, H, W)`` voxel spacing in mm for that dilation.
        min_norm_log: if given, receives one float per sample -- the smallest
            ``||d||_inf`` at which at least ``adv_threshold`` of the
            constrained voxels were misclassified, or ``inf`` if that never
            happened.

    Returns:
        ``(x_adv, snapshots)`` with the same semantics as ``pgd_trajectory``.
    """
    del step_size, random_start  # ALMA prox starts at delta = 0, deterministic.
    if max_eps == 0.0:
        clean = x.clone()
        return clean, ({0.0: clean.clone()} if snapshot_eps else {})
    if n_steps < 1:
        raise ValueError("n_steps must be >= 1 for ALMA prox")

    x_orig = x.detach()
    batch = x_orig.shape[0]
    x_min, x_max = x_bounds if x_bounds is not None else _samplewise_bounds(x_orig)
    lower, upper = x_min - x_orig, x_max - x_orig

    labels = y.squeeze(1).long()
    masks = labels != ignore_label
    if constraint_band_mm is not None:
        masks = masks & foreground_band(labels, constraint_band_mm, spacing, ignore_label)
    # A sample whose constrained region is empty scores adv_percent 0 rather
    # than NaN, so it simply never counts as adversarial.
    masks_sum = masks.flatten(1).sum(dim=1)
    masks_denominator = masks_sum.clamp_min(1)
    safe_labels = torch.where(masks, labels, torch.zeros_like(labels))
    masks_inf = torch.zeros_like(masks, dtype=x_orig.dtype).masked_fill_(
        ~masks, float("inf")
    )

    def batch_view(t: torch.Tensor) -> torch.Tensor:
        return t.view(batch, *[1] * (x_orig.ndim - 1))

    def constraint_view(t: torch.Tensor) -> torch.Tensor:
        return t.view(batch, *[1] * (labels.ndim - 1))

    delta = torch.zeros_like(x_orig, requires_grad=True)
    lr = torch.full((batch,), lr_init, device=x_orig.device, dtype=x_orig.dtype)
    second_moment = torch.zeros_like(x_orig)
    mu = torch.full_like(labels, mu_init, dtype=torch.double)
    rho = torch.full_like(labels, rho_init, dtype=torch.double)
    scale = torch.full_like(lr, scale_init)

    pixel_adv_found = torch.zeros_like(masks)
    adv_found = torch.zeros_like(lr, dtype=torch.bool)
    step_found = torch.full_like(lr, float(n_steps // 2))
    best_norm = torch.full_like(lr, float("inf"))
    k = ((1.0 - adv_threshold) * masks_sum).long()
    constraint_mask = masks
    prev_constraints = torch.zeros_like(masks, dtype=x_orig.dtype)

    budgets = sorted({e for e in (snapshot_eps or []) if e > 0.0})
    candidates = {e: x_orig.clone() for e in budgets}
    best_dice = {
        e: torch.full_like(lr, float("inf"), dtype=torch.double) for e in budgets
    }

    saved_grad_states = [(p, p.requires_grad) for p in model.parameters()]
    for p, _ in saved_grad_states:
        p.requires_grad_(False)

    try:
        for i in range(n_steps):
            adv_inputs = x_orig + delta
            logits = model(adv_inputs)
            norms = delta.detach().flatten(1).norm(p=float("inf"), dim=1)

            if i == 0:
                labels_infhot = torch.zeros_like(logits.detach()).scatter_(
                    1, safe_labels.unsqueeze(1), float("inf")
                )

            prediction = logits.detach().argmax(dim=1)
            pixel_is_adv = prediction != labels
            pixel_adv_found.logical_or_(pixel_is_adv)
            adv_percent = (
                pixel_is_adv & masks
            ).flatten(1).sum(dim=1) / masks_denominator
            is_adv = adv_percent >= adv_threshold
            best_norm = torch.where(is_adv & (norms < best_norm), norms, best_norm)
            # The learning rate only starts decaying once a sample has been
            # adversarial once, and only if that happens in the first half.
            if i < n_steps // 2:
                step_found = torch.where(
                    ~adv_found & is_adv,
                    torch.full_like(step_found, float(i)),
                    step_found,
                )
            adv_found = adv_found | is_adv

            if budgets:
                dice = foreground_dice_per_sample(
                    logits.detach(), labels, num_classes, ignore_label=ignore_label
                )
                for e in budgets:
                    better = (norms <= e + 1e-12) & (dice < best_dice[e])
                    if not bool(better.any()):
                        continue
                    best_dice[e] = torch.where(better, dice, best_dice[e])
                    candidates[e] = torch.where(
                        batch_view(better), adv_inputs.detach(), candidates[e]
                    )

            # Constraint scale: loosen while the attack is short of nu, tighten
            # once it is past it (Eq. 8).
            scale = (
                scale / torch.where(is_adv, 1.0 + scale_gamma, 1.0 - scale_gamma)
            ).clamp(min=scale_min, max=scale_max)
            constraints = constraint_view(scale) * constraint_values(
                logits, safe_labels, labels_infhot, num_classes, logit_tolerance
            )

            if constraint_masking:
                if mask_decay:
                    k = (
                        ((1.0 - adv_threshold) * masks_sum) * (i / max(n_steps - 1, 1))
                    ).long()
                if bool(k.any()):
                    top = (
                        constraints.detach()
                        .sub(masks_inf)
                        .flatten(1)
                        .topk(k=int(k.max()))
                        .values
                    )
                    xi = top.gather(1, k.clamp_min(1).unsqueeze(1) - 1).squeeze(1)
                    constraint_mask = masks & (constraints <= constraint_view(xi))

            if i == 0:
                prev_constraints = constraints.detach()
            elif (i + 1) % check_steps == 0:
                improved = (
                    constraints.detach() * constraint_mask <= tau * prev_constraints
                )
                rho = torch.where(~(pixel_adv_found | improved), gamma * rho, rho)
                prev_constraints = constraints.detach()
                pixel_adv_found.fill_(False)

            if i:
                c = constraints.to(dtype=mu.dtype).detach().requires_grad_(True)
                penalty_grad = torch.autograd.grad(
                    _PenaltyP2.apply(c, rho, mu)[constraint_mask].sum(),
                    c,
                    only_inputs=True,
                )[0]
                mu = mu.lerp(penalty_grad, weight=1.0 - alpha).clamp_(1e-12, 1.0)

            objective = (
                _PenaltyP2.apply(
                    constraints, rho.to(constraints.dtype), mu.to(constraints.dtype)
                )
                .mul(constraint_mask)
                .flatten(1)
                .sum(dim=1)
            )
            if loss_log is not None:
                loss_log.append(float(objective.detach().sum().cpu()))
            delta_grad = torch.autograd.grad(objective.sum(), delta, only_inputs=True)[
                0
            ]

            if lr_reduction != 1:
                tangent = (
                    lr_reduction
                    / (1.0 - lr_reduction)
                    * (n_steps - step_found).clamp_min(1)
                )
                lam = lr * (tangent / ((i - step_found).clamp_min(0) + tangent))
            else:
                lam = lr

            second_moment.mul_(alpha).addcmul_(
                delta_grad, delta_grad, value=1.0 - alpha
            )
            metric = second_moment.div(1.0 - alpha ** (i + 1)).sqrt().clamp_min(1e-8)

            with torch.no_grad():
                delta.data.addcmul_(delta_grad, batch_view(lam) / metric, value=-1.0)
                delta.data = prox_linf_indicator(
                    delta.data, lam=lam, lower=lower, upper=upper, metric=metric
                )
    finally:
        for p, state in saved_grad_states:
            p.requires_grad_(state)

    if min_norm_log is not None:
        min_norm_log.extend(float(v) for v in best_norm)

    snapshots: dict[float, torch.Tensor] = {}
    if snapshot_eps and 0.0 in snapshot_eps:
        snapshots[0.0] = x_orig.clone()
    for e in budgets:
        snapshots[e] = _project_linf(candidates[e], x_orig, e, x_min, x_max)
    x_adv = snapshots.get(float(max_eps))
    if x_adv is None:
        x_adv = _project_linf(
            candidates[budgets[-1]] if budgets else x_orig + delta.detach(),
            x_orig,
            float(max_eps),
            x_min,
            x_max,
        )
        snapshots[float(max_eps)] = x_adv
    return x_adv.clone(), snapshots


__all__ = [
    "alma_prox_trajectory",
    "constraint_values",
    "difference_of_logits",
    "prox_linf_indicator",
]
