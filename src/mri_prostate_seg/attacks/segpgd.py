"""SegPGD: PGD with the correct/wrong voxel loss split of Gu et al. (ECCV 2022).

Reference: J. Gu, H. Zhao, V. Tresp, P. Torr, "SegPGD: An Effective and
Efficient Adversarial Attack for Evaluating and Boosting Segmentation
Robustness", ECCV 2022, arXiv:2207.12391. The formulation below follows the
paper's Eq. 4 and Algorithm 1, verified against the PDF:

    L = (1 - lambda) / N * sum_{j in P^T} CE_j  +  lambda / N * sum_{k in P^F} CE_k
    lambda(t) = (t - 1) / (2 T),   t = 1..T

where P^T are the voxels the current adversarial prediction classifies
correctly, P^F the ones it gets wrong, N the total voxel count and CE the
per-voxel cross-entropy. The split is taken from the argmax of the current
logits; the schedule starts at lambda = 0 (only correct voxels drive the first
step) and ends just below 0.5. The official CosPGD repository's ``segpgd_scale``
uses ``iteration / (2 * iterations)`` with a 0-based iteration, which is the
same schedule.

What is deliberately NOT taken from the paper
---------------------------------------------
The paper attacks [0, 1] images with alpha = 0.01 at eps = 0.03. Our inputs are
z-scored MRI volumes, so that alpha has no meaning here. SegPGD is a loss
variant of PGD, and the campaign runs it with the campaign PGD's update rule and
step (``alpha = eps / n_steps``, sign gradient, uniform random start, L-infinity
projection plus intensity clamping) so the only difference between the ``pgd``
and ``segpgd`` arms is the objective. That isolates the loss effect, which is
the claim the paper makes.

The volumetric port at github.com/asif-hanif/segpgd replaces the per-voxel
cross-entropy with a Dice loss over each voxel set. That is a deviation from
the paper and is not reproduced here.

Implementation note: ``pgd_trajectory`` evaluates its loss callable exactly once
per iteration, in order, so the iteration index t is supplied by a counter that
advances on every call. ``segpgd_trajectory`` builds a fresh counter per call,
so each restart begins its schedule at lambda = 0 again. Never pass the counter
loss to code that evaluates it for scoring; that would advance the schedule.
"""

from __future__ import annotations

import torch
import torch.nn.functional as F

from mri_prostate_seg.attacks.pgd import pgd_trajectory

IGNORE_LABEL = -1


def segpgd_lambda(step: int, n_steps: int) -> float:
    """Paper's linear schedule, Eq. 5 first instance: (t - 1) / (2 T), t = 1..T."""
    if step < 1 or step > n_steps:
        raise ValueError(f"step must be in [1, {n_steps}], got {step}")
    return (step - 1) / (2.0 * n_steps)


def segpgd_loss(
    logits: torch.Tensor,
    target: torch.Tensor,
    num_classes: int,
    step: int,
    n_steps: int,
) -> torch.Tensor:
    """Eq. 4 of the paper for one iteration.

    Args:
        logits: ``[B, C, D, H, W]`` model output on the current adversarial input.
        target: ``[B, 1, D, H, W]`` labels; ``IGNORE_LABEL`` voxels are excluded
            from both sets and from the normalising count.
        num_classes: Unused beyond signature parity with the other losses.
        step: 1-based attack iteration t.
        n_steps: Total iterations T.
    """
    del num_classes
    labels = target.squeeze(1).long()
    valid = labels != IGNORE_LABEL
    safe_labels = torch.where(valid, labels, torch.zeros_like(labels))
    per_voxel = F.cross_entropy(logits, safe_labels, reduction="none")
    per_voxel = per_voxel * valid.to(per_voxel.dtype)

    predicted = logits.detach().argmax(dim=1)
    correct = (predicted == labels) & valid
    lam = segpgd_lambda(step, n_steps)
    weight = torch.where(
        correct,
        torch.full_like(per_voxel, 1.0 - lam),
        torch.full_like(per_voxel, lam),
    )
    n_valid = valid.sum().clamp_min(1).to(per_voxel.dtype)
    return (weight * per_voxel).sum() / n_valid


class _ScheduledSegPGDLoss:
    """Callable ``(logits, y, num_classes) -> loss`` that advances t on each call."""

    def __init__(self, n_steps: int) -> None:
        self.n_steps = n_steps
        self.step = 0
        self.steps_seen: list[int] = []

    def __call__(
        self, logits: torch.Tensor, target: torch.Tensor, num_classes: int
    ) -> torch.Tensor:
        self.step += 1
        self.steps_seen.append(self.step)
        return segpgd_loss(logits, target, num_classes, self.step, self.n_steps)


def segpgd_trajectory(
    model: torch.nn.Module,
    x: torch.Tensor,
    y: torch.Tensor,
    max_eps: float,
    num_classes: int,
    n_steps: int,
    step_size: float | None = None,
    random_start: bool = True,
    x_bounds: tuple[torch.Tensor, torch.Tensor] | None = None,
    snapshot_eps: list[float] | None = None,
    loss_log: list[float] | None = None,
) -> tuple[torch.Tensor, dict[float, torch.Tensor]]:
    """SegPGD at ``max_eps`` with the campaign PGD's update and snapshot rules.

    Delegates to ``pgd_trajectory`` (no momentum, sign gradient, default step
    ``max_eps / n_steps``) with the scheduled loss above. Returns the same
    ``(x_adv, snapshots)`` pair so it slots into the shared-trajectory protocol.
    """
    if max_eps == 0.0:
        clean = x.clone()
        snaps = {0.0: clean.clone()} if snapshot_eps else {}
        return clean, snaps
    if n_steps < 1:
        raise ValueError("n_steps must be >= 1 for SegPGD")

    scheduled = _ScheduledSegPGDLoss(n_steps)
    return pgd_trajectory(
        model,
        x,
        y,
        max_eps,
        num_classes,
        scheduled,
        n_steps=n_steps,
        step_size=step_size,
        random_start=random_start,
        momentum=0.0,
        normalized_grad=False,
        x_bounds=x_bounds,
        snapshot_eps=snapshot_eps,
        loss_log=loss_log,
    )


def segpgd_attack(
    model: torch.nn.Module,
    x: torch.Tensor,
    y: torch.Tensor,
    eps: float,
    num_classes: int,
    n_steps: int,
    step_size: float | None = None,
    random_start: bool = True,
    x_bounds: tuple[torch.Tensor, torch.Tensor] | None = None,
) -> torch.Tensor:
    """Single-epsilon SegPGD; returns the final iterate as the paper does."""
    if eps == 0.0:
        return x.clone()
    x_adv, _ = segpgd_trajectory(
        model,
        x,
        y,
        eps,
        num_classes,
        n_steps,
        step_size=step_size,
        random_start=random_start,
        x_bounds=x_bounds,
    )
    return x_adv
