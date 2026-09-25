"""DAG: Dense Adversary Generation of Xie et al. (ICCV 2017).

Reference: C. Xie, J. Wang, Z. Zhang, Y. Zhou, L. Xie, A. Yuille, "Adversarial
Examples for Semantic Segmentation and Object Detection", ICCV 2017,
arXiv:1703.08603. The formulation follows the paper's Algorithm 1 and Eqs. 1-3,
verified against the PDF:

    T_m   = { t_n : argmax_c f_c(X_m, t_n) = l_n }              active target set
    r_m   = sum_{t_n in T_m} [ grad f_{l'_n}(X_m, t_n) - grad f_{l_n}(X_m, t_n) ]
    r'_m  = gamma / ||r_m||_inf * r_m
    X_m+1 = X_m + r'_m
    stop when T_m is empty or m reaches M_0

``f`` is the pre-softmax score (logit), ``l_n`` the true label of voxel ``t_n``
and ``l'_n`` an adversarial label drawn once per image from a random
permutation ``pi`` of the classes with ``pi(c) != c``. Only voxels the current
iterate still classifies correctly contribute, so the active set shrinks and the
attack spends its budget on what is left to flip. The perturbation is
normalised by its L-infinity norm, not signed, which is what distinguishes DAG
from the PGD family. There is no random start.

For the whole-gland model there are two classes, so the adversarial label is
deterministic: the other class.

What is deliberately NOT taken from the paper
---------------------------------------------
* The paper's ``gamma = 0.5`` is on the [0, 255] pixel scale and its
  perturbation is unbounded (it stops when every target is fooled or at
  ``M_0 = 200`` iterations). Our inputs are z-scored MRI volumes evaluated on a
  fixed L-infinity grid, so the step is ``gamma = eps / n_steps`` -- the same
  per-step reach as the ``pgd`` arm -- and every iterate is projected onto the
  eps-ball and clamped to the intensity range. Under that projection the
  maximum-displacement voxel reaches the ball boundary after ``n_steps`` steps.
* ``M_0 = 200`` becomes the campaign budget ``n_steps = 20``.
* The early stop is kept: if the active set empties, the attack returns the
  current iterate, and any pending snapshot budgets receive its projection.
"""

from __future__ import annotations

import torch

from mri_prostate_seg.attacks.pgd import _project_linf, _samplewise_bounds

IGNORE_LABEL = -1


def adversarial_labels(labels: torch.Tensor, num_classes: int) -> torch.Tensor:
    """Per-voxel adversarial label ``pi(l)`` from a derangement of the classes.

    The paper draws a random permutation with ``pi(c) != c`` once per image.
    With two classes the only derangement is the swap, so the result is
    deterministic. Ignore voxels keep their label; they are never active.
    """
    if num_classes < 2:
        raise ValueError("DAG needs at least two classes")
    classes = torch.arange(num_classes, device=labels.device)
    if num_classes == 2:
        perm = torch.tensor([1, 0], device=labels.device)
    else:
        perm = classes.clone()
        while bool((perm == classes).any()):
            perm = classes[torch.randperm(num_classes, device=labels.device)]
    valid = labels != IGNORE_LABEL
    safe = torch.where(valid, labels, torch.zeros_like(labels))
    return torch.where(valid, perm[safe], labels)


def dag_loss(
    logits: torch.Tensor,
    labels: torch.Tensor,
    adv_labels: torch.Tensor,
) -> tuple[torch.Tensor, int]:
    """Eq. 1 restricted to the active set, as a quantity to ascend.

    Returns ``(sum_{active} [f_adv - f_true], n_active)``. The paper minimises
    ``f_true - f_adv``; ascending its negative is the same update.
    """
    valid = labels != IGNORE_LABEL
    active = (logits.detach().argmax(dim=1) == labels) & valid
    safe_true = torch.where(valid, labels, torch.zeros_like(labels)).unsqueeze(1)
    safe_adv = torch.where(valid, adv_labels, torch.zeros_like(labels)).unsqueeze(1)
    f_true = logits.gather(1, safe_true).squeeze(1)
    f_adv = logits.gather(1, safe_adv).squeeze(1)
    objective = ((f_adv - f_true) * active.to(logits.dtype)).sum()
    return objective, int(active.sum())


def dag_trajectory(
    model: torch.nn.Module,
    x: torch.Tensor,
    y: torch.Tensor,
    max_eps: float,
    num_classes: int,
    n_steps: int,
    step_size: float | None = None,
    random_start: bool = False,
    x_bounds: tuple[torch.Tensor, torch.Tensor] | None = None,
    snapshot_eps: list[float] | None = None,
    loss_log: list[float] | None = None,
) -> tuple[torch.Tensor, dict[float, torch.Tensor]]:
    """DAG at ``max_eps`` with the campaign's snapshot convention.

    Args:
        step_size: ``gamma``; defaults to ``max_eps / n_steps``.
        random_start: Off by default -- the paper starts at the clean image.
            Accepted so the restart wrapper can randomise later restarts.
        loss_log: If given, append the active-set objective at every iterate.

    Returns:
        ``(x_adv, snapshots)`` with the same semantics as ``pgd_trajectory``.
    """
    if max_eps == 0.0:
        clean = x.clone()
        snaps = {0.0: clean.clone()} if snapshot_eps else {}
        return clean, snaps
    if n_steps < 1:
        raise ValueError("n_steps must be >= 1 for DAG")

    gamma = float(max_eps) / float(n_steps) if step_size is None else float(step_size)

    x_orig = x.detach()
    if x_bounds is not None:
        x_min, x_max = x_bounds
    else:
        x_min, x_max = _samplewise_bounds(x_orig)

    labels = y.squeeze(1).long()
    adv_labels = adversarial_labels(labels, num_classes)

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

    def _flush(iterate: torch.Tensor) -> None:
        for e in pending:
            snapshots[e] = _project_linf(iterate, x_orig, e, x_min, x_max).clone()
        pending.clear()

    saved_grad_states = [(p, p.requires_grad) for p in model.parameters()]
    for p, _ in saved_grad_states:
        p.requires_grad_(False)

    try:
        for k in range(1, n_steps + 1):
            x_adv = x_adv.detach().requires_grad_(True)
            model.zero_grad(set_to_none=True)
            objective, n_active = dag_loss(model(x_adv), labels, adv_labels)
            if loss_log is not None:
                loss_log.append(float(objective.detach().cpu()))
            if n_active == 0:
                # Every target is already fooled: the paper's termination.
                x_adv = x_adv.detach()
                _flush(x_adv)
                break
            objective.backward()
            grad = x_adv.grad
            if grad is None:
                x_adv = x_adv.detach()
                _flush(x_adv)
                break
            r = grad.detach()
            r_inf = r.abs().amax(dim=tuple(range(1, r.ndim)), keepdim=True)
            r = gamma * r / r_inf.clamp_min(1e-12)
            x_adv = _project_linf(x_adv.detach() + r, x_orig, max_eps, x_min, x_max)

            still_pending = []
            for e in pending:
                if k * gamma + 1e-12 >= e or k == n_steps:
                    snapshots[e] = _project_linf(
                        x_adv.detach(), x_orig, e, x_min, x_max
                    ).clone()
                else:
                    still_pending.append(e)
            pending[:] = still_pending
    finally:
        for p, state in saved_grad_states:
            p.requires_grad_(state)

    x_adv = x_adv.detach()
    _flush(x_adv)
    snapshots[float(max_eps)] = x_adv.clone()
    return x_adv, snapshots


def dag_attack(
    model: torch.nn.Module,
    x: torch.Tensor,
    y: torch.Tensor,
    eps: float,
    num_classes: int,
    n_steps: int,
    step_size: float | None = None,
    x_bounds: tuple[torch.Tensor, torch.Tensor] | None = None,
) -> torch.Tensor:
    """Single-epsilon DAG; returns the final iterate as the paper does."""
    if eps == 0.0:
        return x.clone()
    x_adv, _ = dag_trajectory(
        model, x, y, eps, num_classes, n_steps, step_size=step_size, x_bounds=x_bounds
    )
    return x_adv
