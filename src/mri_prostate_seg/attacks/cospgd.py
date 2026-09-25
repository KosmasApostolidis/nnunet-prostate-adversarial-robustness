"""CosPGD: PGD with the cosine-similarity loss scaling of Agnihotri et al.

Reference: S. Agnihotri, S. Jung, M. Keuper, "CosPGD: an efficient white-box
adversarial attack for pixel-wise prediction tasks", ICML 2024 (PMLR 235),
arXiv:2302.02213. The formulation follows the paper's untargeted objective and
the official implementation ``cospgd/attack_implementations.py::cospgd_scale``
(github.com/shashankskagnihotri/cospgd), verified from the source:

    s_i = cos( softmax(f(x_adv))_i , onehot(y)_i )      over the class axis
    L   = mean_i  stopgrad(s_i) * CE_i

The per-voxel cross-entropy is scaled by how well the current prediction still
agrees with the label, so voxels the model already gets wrong contribute little
and the update concentrates on the ones still to be flipped. The official code
detaches the similarity (``cossim.detach() * loss``); so does this module. The
targeted variant (``1 - s_i``) is not needed here and is not implemented.

What is deliberately NOT taken from the paper
---------------------------------------------
The paper uses alpha = 0.01 on [0, 1] images at eps ~ 8/255. Our inputs are
z-scored MRI volumes, so that alpha has no meaning here. CosPGD is a loss
variant of PGD, and the campaign runs it with the campaign PGD's update rule
and step (``alpha = eps / n_steps``, sign gradient, uniform random start,
L-infinity projection plus intensity clamping) so the only difference between
the ``pgd`` and ``cospgd`` arms is the objective.
"""

from __future__ import annotations

import torch
import torch.nn.functional as F

from mri_prostate_seg.attacks.pgd import pgd_trajectory

IGNORE_LABEL = -1


def cospgd_loss(
    logits: torch.Tensor, target: torch.Tensor, num_classes: int
) -> torch.Tensor:
    """Untargeted CosPGD objective: cosine-scaled per-voxel cross-entropy.

    Args:
        logits: ``[B, C, D, H, W]`` model output on the current adversarial input.
        target: ``[B, 1, D, H, W]`` labels; ``IGNORE_LABEL`` voxels are excluded
            from the mean and never reach ``one_hot``.
        num_classes: Number of classes, used to build the one-hot target.
    """
    labels = target.squeeze(1).long()
    valid = labels != IGNORE_LABEL
    safe_labels = torch.where(valid, labels, torch.zeros_like(labels))

    per_voxel = F.cross_entropy(logits, safe_labels, reduction="none")
    probabilities = F.softmax(logits, dim=1)
    onehot = F.one_hot(safe_labels, num_classes=num_classes)
    onehot = onehot.movedim(-1, 1).to(probabilities.dtype)
    similarity = F.cosine_similarity(probabilities, onehot, dim=1).detach()

    scaled = similarity * per_voxel * valid.to(per_voxel.dtype)
    n_valid = valid.sum().clamp_min(1).to(per_voxel.dtype)
    return scaled.sum() / n_valid


def cospgd_trajectory(
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
    """CosPGD at ``max_eps`` with the campaign PGD's update and snapshot rules."""
    if max_eps == 0.0:
        clean = x.clone()
        snaps = {0.0: clean.clone()} if snapshot_eps else {}
        return clean, snaps
    if n_steps < 1:
        raise ValueError("n_steps must be >= 1 for CosPGD")

    return pgd_trajectory(
        model,
        x,
        y,
        max_eps,
        num_classes,
        cospgd_loss,
        n_steps=n_steps,
        step_size=step_size,
        random_start=random_start,
        momentum=0.0,
        normalized_grad=False,
        x_bounds=x_bounds,
        snapshot_eps=snapshot_eps,
        loss_log=loss_log,
    )


def cospgd_attack(
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
    """Single-epsilon CosPGD; returns the final iterate as the paper does."""
    if eps == 0.0:
        return x.clone()
    x_adv, _ = cospgd_trajectory(
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
