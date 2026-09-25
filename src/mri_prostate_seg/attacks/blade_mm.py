"""BLADE-MM: BLADE with multi-metric adjudication instead of a Dice-only oracle.

BLADE-MM changes exactly one thing about ``blade.py``, and it fixes a real
inconsistency in that design.

The inconsistency
-----------------
BLADE's ensemble admits a boundary objective whose entire purpose is to inflate
the surface metrics -- HD95 and ASD -- by making the network hallucinate
foreground far from the gland. It then adjudicates the rung with
``foreground_dice`` alone (``blade.py``, the selection loop in
``blade_trajectory``; SEA has the same rule). So the candidate that wrecked the
surface metrics is discarded whenever another candidate scores marginally lower
on overlap, and the HD95 and ASD that get reported are whatever the Dice winner
happened to produce. They are a floor on the boundary objective's effect, never
a measurement of it.

The fix
-------
Rank the candidates on every metric the campaign reports, and keep the one that
is worst overall:

* ``dice``  -- lower is a stronger attack        (``METRIC_DIRECTION`` = -1)
* ``hd95``  -- higher is a stronger attack       (+1)
* ``asd``   -- higher is a stronger attack       (+1)

For each metric the candidates are ranked by damage (rank 1 = most damaging).
The winner is the candidate with the lowest mean rank, ties broken by lower
Dice. Ranks rather than normalised magnitudes because Dice lives on [0, 1]
while HD95 reaches ~100 voxels: any scalarisation of the raw values is really a
choice of exchange rate between overlap and millimetres, and with three
candidates a min-max normalisation is dominated by whichever pair happens to
straddle the range. Rank aggregation needs no exchange rate and no constants.

Undefined metrics are excluded rather than imputed. HD95 and ASD do not exist
for an empty prediction, and the campaign already drops non-finite values when
averaging. A metric with fewer than two finite candidate values cannot separate
anything, so it contributes no ranks at all -- this stops a candidate winning a
metric merely by being the only one for which it was computable.

Reporting
---------
``select_by_rank`` answers "what does one perturbation achieve across all three
metrics at once" -- a realisable attack, one image per case. ``per_metric_oracle``
answers the different question "how vulnerable is the model on each metric" by
taking each metric's worst candidate independently. The two are not
interchangeable: the oracle's three numbers may come from three different
perturbations, and must be labelled as a per-metric bound rather than as the
result of a single attack. ``candidate_log`` carries every candidate's full
metric vector so both can be computed from one run.

Everything else -- the epsilon ladder, the three objectives, the untouched
Auto-PGD optimiser -- is ``blade.py``'s and is imported from it, so the two arms
differ only in adjudication and can be compared directly.
"""

from __future__ import annotations

import math

import numpy as np
import torch
from medpy.metric.binary import asd as _medpy_asd
from medpy.metric.binary import hd95 as _medpy_hd95

from mri_prostate_seg.attacks.blade_legacy import (
    BLADE_LOSSES,
    DEFAULT_K_FRAC,
    boundary_weighted_loss,
    frontier_margin_loss,
    ladder_trajectory,
    signed_distance_map,
)
from mri_prostate_seg.attacks.pgd import _samplewise_bounds

IGNORE_LABEL = -1

# +1: a larger value is a stronger attack. -1: a smaller value is.
METRIC_DIRECTION = {"dice": -1, "hd95": +1, "asd": +1}
METRICS = tuple(METRIC_DIRECTION)


def foreground_metrics(
    logits: torch.Tensor, target: torch.Tensor, num_classes: int
) -> dict[str, float]:
    """Dice, HD95 and ASD of the argmax prediction, averaged over foreground classes.

    Uses the same ``medpy`` implementations as the campaign's reported metrics,
    so a candidate is adjudicated on exactly the quantity that will be printed
    for it. Surface metrics are ``nan`` when either mask is empty, which is the
    same case the campaign drops when averaging.
    """
    labels = target.squeeze(1).long()
    valid = (labels != IGNORE_LABEL).cpu().numpy()
    pred = logits.detach().argmax(dim=1).cpu().numpy()
    gt = labels.cpu().numpy()

    dices: list[float] = []
    hd95s: list[float] = []
    asds: list[float] = []
    for c in range(1, num_classes):
        p = ((pred == c) & valid).astype(np.uint8)
        t = ((gt == c) & valid).astype(np.uint8)
        denom = int(p.sum()) + int(t.sum())
        dices.append(1.0 if denom == 0 else 2.0 * float((p & t).sum()) / denom)
        if p.sum() == 0 or t.sum() == 0:
            hd95s.append(math.nan)
            asds.append(math.nan)
            continue
        try:
            hd95s.append(float(_medpy_hd95(p, t)))
        except (RuntimeError, ValueError, TypeError):
            hd95s.append(math.nan)
        try:
            asds.append(float(_medpy_asd(p, t)))
        except (RuntimeError, ValueError, TypeError):
            asds.append(math.nan)

    def _mean(values: list[float]) -> float:
        finite = [v for v in values if math.isfinite(v)]
        return float(sum(finite) / len(finite)) if finite else math.nan

    return {"dice": _mean(dices), "hd95": _mean(hd95s), "asd": _mean(asds)}


def select_by_rank(candidates: dict[str, dict[str, float]]) -> str:
    """Name of the candidate that is worst overall across the reported metrics.

    Rank 1 on a metric is the most damaging value of it, and candidates with
    equal values share the average of the positions they span. The winner has
    the lowest mean rank over the metrics that could be ranked; ties go to the
    lower Dice. A metric with fewer than two finite values ranks nobody.
    """
    if not candidates:
        raise ValueError("select_by_rank needs at least one candidate")
    names = list(candidates)
    if len(names) == 1:
        return names[0]

    ranks: dict[str, list[float]] = {name: [] for name in names}
    for metric, direction in METRIC_DIRECTION.items():
        scored = [
            (name, candidates[name][metric])
            for name in names
            if math.isfinite(candidates[name].get(metric, math.nan))
        ]
        if len(scored) < 2:
            continue
        scored.sort(key=lambda kv: -direction * kv[1])
        # Average ranking: candidates with equal values share the mean of the
        # positions they span. Without this, a stable sort would hand tied
        # candidates different ranks and let dict order decide the rung.
        start = 0
        while start < len(scored):
            stop = start + 1
            while stop < len(scored) and scored[stop][1] == scored[start][1]:
                stop += 1
            shared = (start + stop + 1) / 2.0
            for name, _ in scored[start:stop]:
                ranks[name].append(shared)
            start = stop

    def key(name: str) -> tuple[float, float]:
        own = ranks[name]
        mean_rank = sum(own) / len(own) if own else float("inf")
        return (mean_rank, candidates[name]["dice"])

    return min(names, key=key)


def per_metric_oracle(
    candidates: dict[str, dict[str, float]],
) -> dict[str, tuple[float, str] | None]:
    """Worst value of each metric over the candidates, with the candidate that gave it.

    ``None`` for a metric no candidate could define. These three values may come
    from three different perturbations, so they bound the model's vulnerability
    per metric rather than describing one attack.
    """
    out: dict[str, tuple[float, str] | None] = {}
    for metric, direction in METRIC_DIRECTION.items():
        scored = [
            (name, candidates[name][metric])
            for name in candidates
            if math.isfinite(candidates[name].get(metric, math.nan))
        ]
        if not scored:
            out[metric] = None
            continue
        name, value = max(scored, key=lambda kv: direction * kv[1])
        out[metric] = (value, name)
    return out


def blade_mm_trajectory(
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
    candidate_log: dict[float, dict[str, dict[str, float]]] | None = None,
) -> tuple[torch.Tensor, dict[float, torch.Tensor]]:
    """BLADE's ladder ensemble, adjudicated on every reported metric.

    Args:
        base_loss_fn: The ``dice_ce`` member, supplied by the caller so this
            package does not fork the driver's definition of it.
        chosen_log: If given, maps each rung to the winning objective's name.
        candidate_log: If given, maps each rung to ``{objective: metrics}`` for
            every candidate -- the record ``per_metric_oracle`` and any other
            adjudication rule can be recomputed from without re-attacking.

    Returns:
        ``(x_adv, snapshots)`` with the same contract as ``blade_trajectory``.
    """
    if max_eps == 0.0:
        clean = x.clone()
        snaps = {0.0: clean.clone()} if snapshot_eps else {}
        return clean, snaps
    if n_steps < 1:
        raise ValueError("n_steps must be >= 1 for BLADE-MM")
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
        )
        candidates[name] = snaps

    rungs = sorted(set().union(*(set(s) for s in candidates.values())))
    snapshots: dict[float, torch.Tensor] = {}
    with torch.no_grad():
        for rung in rungs:
            if rung == 0.0:
                snapshots[0.0] = x_orig.clone()
                continue
            scored = {
                name: foreground_metrics(model(candidates[name][rung]), y, num_classes)
                for name in losses
            }
            winner = select_by_rank(scored)
            snapshots[rung] = candidates[winner][rung].clone()
            if chosen_log is not None:
                chosen_log[rung] = winner
            if candidate_log is not None:
                candidate_log[rung] = scored

    return snapshots[float(max_eps)], snapshots


def blade_mm_attack(
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
    """Single-epsilon BLADE-MM; the ladder degenerates to one rung at ``eps``."""
    if eps == 0.0:
        return x.clone()
    x_adv, _ = blade_mm_trajectory(
        model, x, y, eps, num_classes, base_loss_fn, n_steps,
        x_bounds=x_bounds, losses=losses,
    )
    return x_adv


__all__ = [
    "METRICS",
    "METRIC_DIRECTION",
    "blade_mm_attack",
    "blade_mm_trajectory",
    "foreground_metrics",
    "per_metric_oracle",
    "select_by_rank",
]
