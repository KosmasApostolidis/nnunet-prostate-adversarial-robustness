"""The BLADE epsilon ladder over an arbitrary set of objectives.

BLADE couples two things: the ascending-epsilon ladder (each rung's Auto-PGD
warm-started from the previous rung's result) and its own three objectives
(Dice+CE, boundary, frontier). ``blade_legacy.blade_trajectory`` and
``blade_updated.blade_trajectory`` accept only those objectives by name, so the
ladder cannot be tested with anyone else's losses. This module runs the same
ladder, with the same per-rung worst-Dice adjudication, over any
``{name: loss_fn}`` mapping -- for instance SEA's masked CE, balanced masked CE
and Jensen-Shannon losses -- so that the ladder's effect can be separated from
the choice of objectives.

Two implementations mirror the two BLADE modules exactly:

* ``impl="legacy"`` -- the binary whole-gland BLADE: per objective a
  ``blade_legacy.ladder_trajectory``, whose rung result is Auto-PGD's final
  iterate; per rung the objective with the lowest foreground Dice wins.
* ``impl="mc"`` -- BLADE-MC, used on the zones: per objective
  ``blade_updated._one_ladder``, which keeps the lowest-Dice iterate seen on a
  rung (the clean image included) and warm-starts the next rung from Auto-PGD's
  surrogate-best point; per rung the lowest Dice across objectives wins.

With BLADE's own objectives each implementation reproduces its BLADE module
bit for bit (tested), so any difference between an arm built here and BLADE is
the objectives, not the machinery.
"""

from __future__ import annotations

from collections.abc import Callable, Mapping

import torch

from mri_prostate_seg.attacks import blade_updated as mc
from mri_prostate_seg.attacks.blade_legacy import ladder_trajectory as legacy_ladder
from mri_prostate_seg.attacks.pgd import _samplewise_bounds
from mri_prostate_seg.attacks.sea import foreground_dice

LossFn = Callable[[torch.Tensor, torch.Tensor, int], torch.Tensor]
IMPLS = ("legacy", "mc")


def ladder_ensemble_trajectory(
    model: torch.nn.Module,
    x: torch.Tensor,
    y: torch.Tensor,
    max_eps: float,
    num_classes: int,
    objectives: Mapping[str, LossFn],
    n_steps: int,
    random_start: bool = True,
    x_bounds: tuple[torch.Tensor, torch.Tensor] | None = None,
    snapshot_eps: list[float] | None = None,
    loss_log: list[float] | None = None,
    impl: str = "legacy",
    chosen_log: dict[float, str] | None = None,
) -> tuple[torch.Tensor, dict[float, torch.Tensor]]:
    """One epsilon ladder per objective; per rung, keep the worst foreground Dice.

    Args:
        objectives: ``{name: loss_fn}``, each ``(logits, y, num_classes) ->
            scalar`` to maximise. Order matters only for ``loss_log`` (first
            objective) and for ties (earlier objective wins).
        impl: ``"legacy"`` (binary whole-gland BLADE) or ``"mc"`` (BLADE-MC).
        chosen_log: If given, maps each rung to the winning objective's name.

    Returns:
        ``(x_adv, snapshots)`` with the contract of ``blade_trajectory``.
    """
    if impl not in IMPLS:
        raise ValueError(f"impl must be one of {IMPLS}, got {impl!r}")
    if not objectives:
        raise ValueError("objectives must be non-empty")
    if impl == "mc":
        return _mc(
            model,
            x,
            y,
            max_eps,
            num_classes,
            objectives,
            n_steps,
            random_start,
            x_bounds,
            snapshot_eps,
            loss_log,
            chosen_log,
        )
    return _legacy(
        model,
        x,
        y,
        max_eps,
        num_classes,
        objectives,
        n_steps,
        random_start,
        x_bounds,
        snapshot_eps,
        loss_log,
        chosen_log,
    )


def _legacy(
    model,
    x,
    y,
    max_eps,
    num_classes,
    objectives,
    n_steps,
    random_start,
    x_bounds,
    snapshot_eps,
    loss_log,
    chosen_log,
):
    """``blade_legacy.blade_trajectory`` with the objective table passed in."""
    if max_eps == 0.0:
        clean = x.clone()
        return clean, ({0.0: clean.clone()} if snapshot_eps else {})
    if n_steps < 1:
        raise ValueError("n_steps must be >= 1")
    x_orig = x.detach()
    if x_bounds is None:
        x_bounds = _samplewise_bounds(x_orig)

    candidates: dict[str, dict[float, torch.Tensor]] = {}
    for i, (name, fn) in enumerate(objectives.items()):
        _, snaps = legacy_ladder(
            model,
            x,
            y,
            max_eps,
            num_classes,
            fn,
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
            for name in objectives:
                xe = candidates[name][rung]
                d = foreground_dice(model(xe), y, num_classes)
                if d < best_dice:
                    best_name, best_dice, best_x = name, d, xe
            snapshots[rung] = best_x.clone()
            if chosen_log is not None:
                chosen_log[rung] = best_name
    return snapshots[float(max_eps)], snapshots


def _mc(
    model,
    x,
    y,
    max_eps,
    num_classes,
    objectives,
    n_steps,
    random_start,
    x_bounds,
    snapshot_eps,
    loss_log,
    chosen_log,
):
    """``blade_updated.blade_trajectory`` (B = 1) with the objective table passed in."""
    ignore = mc.IGNORE_LABEL
    labels, rungs = mc._validate_inputs(
        model, x, y, num_classes, n_steps, max_eps, snapshot_eps, ignore
    )
    if x.shape[0] != 1:
        raise ValueError("impl='mc' supports one volume at a time")
    bounds = mc._validated_bounds(x, x_bounds)
    x = x.detach()
    if max_eps == 0:
        return x.clone(), {0.0: x.clone()}
    with mc._evaluation_mode(model):
        with torch.no_grad():
            logits = model(x)
            clean_dice = float(
                mc.foreground_dice_per_sample(
                    logits, labels, num_classes, ignore_label=ignore
                )[0]
            )
        del logits
        clean = mc._Candidate(x.clone(), clean_dice, "clean", 0.0, -1)
        winners = {rung: clean for rung in rungs}

        def emit(rung, candidate):
            if candidate.dice < winners[rung].dice:
                winners[rung] = candidate

        for j, (name, fn) in enumerate(objectives.items()):
            mc._one_ladder(
                model,
                x,
                labels,
                num_classes,
                fn,
                n_steps,
                rungs,
                bounds,
                random_start,
                0.75,
                ignore,
                clean,
                name,
                emit,
                loss_log=loss_log if j == 0 else None,
            )
    snapshots = {rung: c.image.detach() for rung, c in winners.items()}
    if chosen_log is not None:
        chosen_log.update({rung: c.source for rung, c in winners.items()})
    return snapshots[float(max_eps)], snapshots


__all__ = ["IMPLS", "LossFn", "ladder_ensemble_trajectory"]
