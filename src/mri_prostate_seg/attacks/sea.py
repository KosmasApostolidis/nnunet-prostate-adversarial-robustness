"""SEA: the Segmentation Ensemble Attack of Croce, Singh and Hein (2023).

Reference: F. Croce, N. D. Singh, M. Hein, "Towards Reliable Evaluation and
Fast Training of Robust Semantic Segmentation Models", arXiv:2306.12941;
official code github.com/nmndeep/robust-segmentation (``semseg/attacker.py``).
The three losses and the selection rule below were verified against that
source:

    L_MCE(u, y)     = 1[argmax_j u_j = y] * CE(u, y)            (``masked_cross_entropy``)
    L_MCE-Bal(u, y) = w_y * 1[argmax_j u_j = y] * CE(u, y)      (``masked_cross_entropy_balanced``)
    L_JS(u, y)      = D_JS( softmax(u) || e_y )                 (``js_div_fn`` / ``js_loss``)
                      D_JS(p||q) = ( KL(p||m) + KL(q||m) ) / 2,  m = (p + q) / 2

each averaged over the valid voxels of the volume. The mask is detached, as in
the reference. The ensemble runs APGD once per loss and keeps, per image, the
run with the worst value of the target metric (``apgd_restarts``: the candidate
replaces the incumbent when its accuracy is lower).

How the ensemble is instantiated here
-------------------------------------
* Optimiser: this repository's faithful Auto-PGD port (``auto_pgd.py``), i.e.
  APGD with the published step rule, at the campaign budget ``n_steps = 20``
  per loss. The ensemble therefore costs three times the gradient evaluations
  of a single-loss arm; that is the point of SEA and is reported as such.
* Selection: per snapshot budget, the loss whose projected iterate gives the
  lowest foreground Dice from the argmax of the logits -- the metric the
  campaign reports -- mirroring SEA's "worst image-wise metric" rule.
* Class weights for L_MCE-Bal: the paper takes ``1 / N_y`` with ``N_y`` the
  pixel count of class ``y`` over the dataset. Here ``N_y`` is the voxel count
  of class ``y`` in the case being attacked, normalised so the weights sum to
  one; the intent (equalise the classes' contribution) is the same, and no
  dataset-level statistic has to be threaded through the driver.

What is deliberately NOT taken from the paper
---------------------------------------------
* The progressive radius reduction ("red-eps": 2eps -> 1.5eps -> eps in a
  3:3:4 split of 300 iterations) is an optimiser schedule tuned for a
  300-iteration budget. At 20 iterations the slots would hold 6, 6 and 8
  steps, so it is not used; each loss runs plain Auto-PGD at ``eps``.
* The paper's fourth candidate in some later evaluations (a masked spherical
  loss) is not part of the arXiv SEA definition or the official
  ``criterion_dict``, and is not implemented.
"""

from __future__ import annotations

import torch
import torch.nn.functional as F

from mri_prostate_seg.attacks.auto_pgd import auto_pgd_attack
from mri_prostate_seg.attacks.pgd import _project_linf, _samplewise_bounds

IGNORE_LABEL = -1
SEA_LOSSES = ("mce", "mce_bal", "js")


def _valid_and_correct(
    logits: torch.Tensor, labels: torch.Tensor
) -> tuple[torch.Tensor, torch.Tensor]:
    valid = labels != IGNORE_LABEL
    correct = (logits.detach().argmax(dim=1) == labels) & valid
    return valid, correct


def masked_ce_loss(
    logits: torch.Tensor, target: torch.Tensor, num_classes: int
) -> torch.Tensor:
    """L_MCE: cross-entropy of the voxels still classified correctly, mean over valid voxels."""
    del num_classes
    labels = target.squeeze(1).long()
    valid, correct = _valid_and_correct(logits, labels)
    safe = torch.where(valid, labels, torch.zeros_like(labels))
    per_voxel = F.cross_entropy(logits, safe, reduction="none")
    n_valid = valid.sum().clamp_min(1).to(per_voxel.dtype)
    return (per_voxel * correct.to(per_voxel.dtype)).sum() / n_valid


def masked_ce_balanced_loss(
    logits: torch.Tensor, target: torch.Tensor, num_classes: int
) -> torch.Tensor:
    """L_MCE-Bal: L_MCE with per-class weights inversely proportional to class size."""
    labels = target.squeeze(1).long()
    valid, correct = _valid_and_correct(logits, labels)
    safe = torch.where(valid, labels, torch.zeros_like(labels))
    counts = torch.bincount(safe[valid], minlength=num_classes).to(logits.dtype)
    weights = torch.where(
        counts > 0, 1.0 / counts.clamp_min(1.0), torch.zeros_like(counts)
    )
    weights = weights / weights.sum().clamp_min(1e-12)
    per_voxel = F.cross_entropy(logits, safe, reduction="none", weight=weights)
    n_valid = valid.sum().clamp_min(1).to(per_voxel.dtype)
    return (per_voxel * correct.to(per_voxel.dtype)).sum() / n_valid


def js_loss(
    logits: torch.Tensor, target: torch.Tensor, num_classes: int
) -> torch.Tensor:
    """L_JS: Jensen-Shannon divergence between softmax and one-hot label, mean over valid voxels."""
    labels = target.squeeze(1).long()
    valid = labels != IGNORE_LABEL
    safe = torch.where(valid, labels, torch.zeros_like(labels))
    p = F.softmax(logits, dim=1)
    q = F.one_hot(safe, num_classes=num_classes).movedim(-1, 1).to(p.dtype)
    m = 0.5 * (p + q)
    log_m = m.clamp_min(1e-12).log()
    # F.kl_div(log_m, p) = sum p * (log p - log m); q's term is q * (log q - log m)
    # with 0 log 0 = 0, so on the one-hot support it is -log m.
    kl_p = (p * (p.clamp_min(1e-12).log() - log_m)).sum(dim=1)
    kl_q = (q * (q.clamp_min(1e-12).log() - log_m)).sum(dim=1)
    per_voxel = 0.5 * (kl_p + kl_q)
    n_valid = valid.sum().clamp_min(1).to(per_voxel.dtype)
    return (per_voxel * valid.to(per_voxel.dtype)).sum() / n_valid


LOSS_FNS = {
    "mce": masked_ce_loss,
    "mce_bal": masked_ce_balanced_loss,
    "js": js_loss,
}


def foreground_dice(
    logits: torch.Tensor, target: torch.Tensor, num_classes: int
) -> float:
    """Mean over foreground classes of the hard Dice from the argmax prediction."""
    labels = target.squeeze(1).long()
    valid = labels != IGNORE_LABEL
    pred = logits.argmax(dim=1)
    dices = []
    for c in range(1, num_classes):
        p = (pred == c) & valid
        t = (labels == c) & valid
        denom = int(p.sum()) + int(t.sum())
        if denom == 0:
            dices.append(1.0)
        else:
            dices.append(2.0 * int((p & t).sum()) / denom)
    return float(sum(dices) / max(len(dices), 1))


def sea_trajectory(
    model: torch.nn.Module,
    x: torch.Tensor,
    y: torch.Tensor,
    max_eps: float,
    num_classes: int,
    n_steps: int,
    random_start: bool = True,
    x_bounds: tuple[torch.Tensor, torch.Tensor] | None = None,
    snapshot_eps: list[float] | None = None,
    loss_log: list[float] | None = None,
    losses: tuple[str, ...] = SEA_LOSSES,
    chosen_log: dict[float, str] | None = None,
) -> tuple[torch.Tensor, dict[float, torch.Tensor]]:
    """SEA at ``max_eps``: Auto-PGD per loss, worst Dice per snapshot budget.

    Args:
        loss_log: If given, receives the first loss's per-iterate objective
            (the wrapper contract allows one curve per case).
        losses: Subset of ``SEA_LOSSES`` to ensemble.
        chosen_log: If given, maps each snapshot budget to the winning loss name.

    Returns:
        ``(x_adv, snapshots)``: ``x_adv`` is the winner at ``max_eps``.
    """
    if max_eps == 0.0:
        clean = x.clone()
        snaps = {0.0: clean.clone()} if snapshot_eps else {}
        return clean, snaps
    if n_steps < 1:
        raise ValueError("n_steps must be >= 1 for SEA")
    unknown = [name for name in losses if name not in LOSS_FNS]
    if unknown:
        raise ValueError(f"unknown SEA loss(es): {unknown}")

    x_orig = x.detach()
    if x_bounds is not None:
        x_min, x_max = x_bounds
    else:
        x_min, x_max = _samplewise_bounds(x_orig)

    budgets = sorted({float(e) for e in (snapshot_eps or [])} | {float(max_eps)})
    candidates: dict[str, torch.Tensor] = {}
    for i, name in enumerate(losses):
        candidates[name] = auto_pgd_attack(
            model,
            x,
            y,
            max_eps,
            num_classes,
            LOSS_FNS[name],
            n_steps=n_steps,
            random_start=random_start,
            x_bounds=(x_min, x_max),
            loss_log=loss_log if i == 0 else None,
        )

    snapshots: dict[float, torch.Tensor] = {}
    with torch.no_grad():
        for e in budgets:
            if e == 0.0:
                snapshots[0.0] = x_orig.clone()
                continue
            best_name, best_dice, best_x = None, float("inf"), None
            for name in losses:
                xe = (
                    candidates[name]
                    if e == float(max_eps)
                    else _project_linf(
                        candidates[name].clone(), x_orig, e, x_min, x_max
                    )
                )
                d = foreground_dice(model(xe), y, num_classes)
                if d < best_dice:
                    best_name, best_dice, best_x = name, d, xe
            snapshots[e] = best_x.clone()
            if chosen_log is not None:
                chosen_log[e] = best_name

    return snapshots[float(max_eps)], snapshots
