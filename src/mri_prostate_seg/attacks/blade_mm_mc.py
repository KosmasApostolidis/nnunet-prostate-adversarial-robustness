"""BLADE-MM-MC: BLADE-MC's ladders, adjudicated the BLADE-MM way.

BLADE-MM (``blade_mm.py``) fixes BLADE's Dice-only oracle by ranking the three
per-objective candidates on every reported metric, but it builds those
candidates with the binary ``blade_legacy`` ladders, so on a multiclass task
(the prostate zones) its objectives never see the TZ/PZ interface. BLADE-MC
(``blade_updated.py``) has the multiclass objectives and per-class candidate
selection, but adjudicates a rung on foreground macro-Dice alone -- the same
blind spot BLADE-MM was written to remove.

This arm composes the two: one BLADE-MC ladder per objective yields that
branch's Dice-worst candidate per rung (candidates from earlier rungs and the
clean image stay eligible, as in BLADE-MC); the branch winners are then scored
with ``blade_mm.foreground_metrics`` -- already a per-class macro average of
Dice, HD95 and ASD -- and ``blade_mm.select_by_rank`` keeps the one that is
worst overall. Nothing is re-implemented: the ladders are BLADE-MC's, the
adjudication is BLADE-MM's, so the arm differs from ``blade`` (MC) only in
which candidate wins a rung, and from ``blade_mm`` only in the objectives and
in-branch selection being multiclass.

Cost: three BLADE-MC ladders plus three forward passes per rung.
"""

from __future__ import annotations

import torch

from mri_prostate_seg.attacks import blade_updated as mc
from mri_prostate_seg.attacks.blade_mm import foreground_metrics, select_by_rank

BLADE_LOSSES = mc.BLADE_LOSSES


def blade_mm_mc_trajectory(
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
    losses: tuple[str, ...] = BLADE_LOSSES,
    k_frac: float = mc.DEFAULT_K_FRAC,
    chosen_log: dict[float, str] | None = None,
    candidate_log: dict[float, dict[str, dict[str, float]]] | None = None,
    *,
    phi: torch.Tensor | None = None,
    ignore_label: int = mc.IGNORE_LABEL,
    rho: float = 0.75,
    spacing: mc.Spacing = None,
) -> tuple[torch.Tensor, dict[float, torch.Tensor]]:
    """BLADE-MC's per-objective ladders, adjudicated on Dice, HD95 and ASD.

    Args:
        y: Integer labels ``[B, 1, D, H, W]`` (``ignore_label`` allowed).
        phi: Optional trusted ``[B, C, D, H, W]`` signed-distance maps for the
            boundary branch (see ``blade_updated.blade_trajectory``).
        chosen_log / candidate_log: as in ``blade_mm_trajectory``.

    Returns:
        ``(x_adv, snapshots)`` with the same contract as ``blade_trajectory``.
    """
    if max_eps == 0.0:
        clean = x.clone()
        snaps = {0.0: clean.clone()} if snapshot_eps else {}
        return clean, snaps
    unknown = [name for name in losses if name not in mc.SUPPORTED_LOSSES]
    if unknown:
        raise ValueError(f"unknown BLADE loss(es): {unknown}")

    x_orig = x.detach()
    candidates: dict[str, dict[float, torch.Tensor]] = {}
    for i, name in enumerate(losses):
        _, snaps = mc.blade_trajectory(
            model,
            x,
            y,
            max_eps,
            num_classes,
            None,
            n_steps,
            random_start=random_start,
            x_bounds=x_bounds,
            snapshot_eps=snapshot_eps,
            loss_log=loss_log if i == 0 else None,
            losses=(name,),
            k_frac=k_frac,
            ignore_label=ignore_label,
            rho=rho,
            spacing=spacing,
            phi=phi if name == "boundary" else None,
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


__all__ = ["BLADE_LOSSES", "blade_mm_mc_trajectory"]
