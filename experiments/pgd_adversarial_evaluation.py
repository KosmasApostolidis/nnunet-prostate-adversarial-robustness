"""
PGD (multi-step L∞) adversarial evaluation for nnU-Net checkpoints under
nnUnet_paths/nnUNet_results.

Same loss / input constraints as fgsm_adversarial_evaluation.py:
  - loss = dice_ce_loss (Dice + CE)
  - after each step: project to L∞ ball of radius epsilon around the clean image,
    then clamp to per-sample intensity bounds [x_min, x_max].

Improvements over the previous version:
  1. **Shared-trajectory mode** (``--shared-trajectory``): runs PGD once at
     ``max(eps_list)`` and snapshots+projects to every smaller epsilon along the
     way. Roughly an N× speedup over per-eps PGD when sweeping N epsilons.
  2. **Stronger attacks**: momentum (MI-PGD, ``--momentum``), normalized gradient
     (``--normalized-grad``), random restarts (``--num-restarts``) keeping the
     worst-loss perturbation per sample.
  3. **Reproducibility / correctness**: per-sample CSV (one row per
     case×eps×class), deterministic mode (``--deterministic``), optional save of
     adversarial volumes (``--save-adv-dir``), per-step loss curve logging
     (``--log-loss-curve``).
  4. **Reporting**: 95% CI confidence bands on plots, attack success rate (ASR:
     fraction of samples whose Dice drops by ≥ ``--asr-threshold`` vs. clean
     baseline), combined cross-model figure when running ``--model both``.
  5. **Runner**: tqdm progress, mixed-precision forward (``--amp``, attack
     gradient stays in fp32), resume-by-skipping cases already present in the
     per-sample CSV (``--resume``).

Backwards-compatible: ``pgd_attack`` is preserved as a thin wrapper.
"""

from __future__ import annotations

import argparse
import csv
import functools
import gc
import json
import os
import warnings
from collections import defaultdict
from contextlib import nullcontext

import numpy as np
import torch
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

try:
    from tqdm import tqdm
except ImportError:  # tqdm is optional — fall back to a no-op iterator wrapper

    def tqdm(it, **_):
        return it


import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from fgsm_adversarial_evaluation import (
    NNUNET_PATHS,
    ConfinedInput,
    attackable_mask,
    compute_metrics,
    confine,
    dice_ce_loss,
    get_checkpoint_for_fold,
    load_model,
    load_sample,
    load_validation_ids,
    pad_to_divisible,
    samplewise_bounds,
    unpad,
    _make_eval_config,
)

# Output-file stem; set from --attack so PGD and adaptive-step PGD runs never
# overwrite each other when they share an output directory.
ATTACK_STEM = "pgd"

# Attacks whose trajectory IS the epsilon grid. For these, --shared-trajectory
# does not mean "optimise once at max_eps and project down"; it means "hand the
# whole requested grid to the ladder as its rungs", and every reported epsilon
# is optimised natively inside its own ball. Recording the flag alone would
# therefore misdescribe the protocol in the run config, so _protocol_label()
# resolves the two meanings apart.
LADDER_ATTACKS = frozenset(
    {"blade", "blade1", "blade_mm", "blade_boundary", "blade_frontier"}
)

# Attacks that have no epsilon of their own: they minimise the perturbation
# norm, and the reported grid is read off the single trajectory they produce.
# They are always handed the whole grid in one call, whatever --shared-trajectory
# says, because running them once per epsilon would repeat an identical
# deterministic search.
MIN_NORM_ATTACKS = frozenset({"alma_prox"})

PROTOCOL_NOTES = {
    "ladder": (
        "Ascending-epsilon ladder: every reported epsilon is optimised natively "
        "inside its own L-inf ball, warm-started from the rung below. No "
        "snapshot is projected down from max_eps. --shared-trajectory was "
        "passed to deliver the epsilon grid as the ladder's rungs, not to "
        "select the projected protocol; the run is identical under either "
        "campaign protocol."
    ),
    "shared_trajectory": (
        "Shared-trajectory protocol: optimised once at max_eps, with every "
        "smaller epsilon obtained by L-inf projection of that single "
        "trajectory. Sub-maximal epsilons are projections, not native runs."
    ),
    "minimum_norm": (
        "Minimum-norm protocol: the attack does not take an epsilon. It "
        "minimises ||delta||_inf until at least adv_threshold of the labelled "
        "voxels are misclassified, is run once per case, and each reported "
        "epsilon receives the lowest-Dice iterate whose perturbation fits "
        "inside that ball. Above the minimal norm the attack does not spend "
        "the extra budget, so the row flattens; below it the row is a "
        "pre-convergence iterate, not a converged attack at that epsilon."
    ),
    "native_per_eps": (
        "Native per-epsilon protocol: the attack was run independently at this "
        "epsilon, with no projection from a larger budget."
    ),
}


def _protocol_label(attack: str, shared_trajectory: bool) -> str:
    """Name the epsilon protocol this run actually used."""
    if attack in LADDER_ATTACKS:
        return "ladder"
    if attack in MIN_NORM_ATTACKS:
        return "minimum_norm"
    return "shared_trajectory" if shared_trajectory else "native_per_eps"

# ALMA prox restricts its constraints to the foreground dilated by this many
# millimetres, mirroring configs/attacks/alma_prox.json. Constraining every
# labelled voxel -- the paper's setting -- is unreachable on a volume where the
# gland is a few percent of the voxels; see the attack module's docstring.
ALMA_CONSTRAINT_BAND_MM = 5.0

# Fraction of the constrained voxels that must be misclassified before the run
# counts as successful -- nu in the paper, which uses 0.99 on 2-D scenes. On
# these volumes the attack peaks at 0.93-0.95 of the band within the budget, so
# 0.99 and 0.95 both leave cases whose minimal norm is never defined; see the
# attack module. It is also the (1 - nu) fraction of largest constraints that
# masking discards (Eq. 6), so loosening it loosens both, as in the reference.
ALMA_ADV_THRESHOLD = 0.90

# Adaptive-step PGD hyperparameters, mirroring configs/attacks/a_pgd.json.
APGD_ADAPTATION_RATE = 0.1
APGD_CHECKPOINTS = [0.25, 0.5, 0.75]


# ---------------------------------------------------------------------------
# Core PGD
# ---------------------------------------------------------------------------


def _project_linf(
    x_adv: torch.Tensor,
    x_orig: torch.Tensor,
    eps: float,
    x_min: torch.Tensor,
    x_max: torch.Tensor,
) -> torch.Tensor:
    x_adv = torch.max(torch.min(x_adv, x_orig + eps), x_orig - eps)
    return torch.clamp(x_adv, x_min, x_max)


def pgd_trajectory(
    model: torch.nn.Module,
    x: torch.Tensor,
    y: torch.Tensor,
    max_eps: float,
    num_classes: int,
    n_steps: int,
    step_size: float | None = None,
    random_start: bool = False,
    momentum: float = 0.0,
    normalized_grad: bool = False,
    x_bounds: tuple[torch.Tensor, torch.Tensor] | None = None,
    snapshot_eps: list[float] | None = None,
    loss_log: list[float] | None = None,
):
    """Run PGD at ``max_eps`` and optionally snapshot intermediate L∞ projections.

    Returns ``(x_adv_final, snapshots)`` where ``snapshots`` is a dict
    ``{eps: x_adv_at_that_eps}`` containing one entry per requested ``snapshot_eps``
    (always including ``max_eps``). For ``eps == 0`` the snapshot is the clean
    image. The snapshot for any ``eps < max_eps`` is taken at iteration
    ``k* = max(1, ceil(eps / step_size))`` and then projected onto the
    L∞(eps) ball — guaranteeing constraint satisfaction.
    """
    if max_eps == 0.0:
        clean = x.clone()
        snaps = {0.0: clean.clone()} if snapshot_eps else {}
        return clean, snaps

    if n_steps < 1:
        raise ValueError("n_steps must be >= 1 for PGD")

    if step_size is None:
        step_size = float(max_eps) / float(n_steps)
    if step_size > float(max_eps):
        warnings.warn(
            f"step_size ({step_size:.6g}) > max_eps ({max_eps:.6g}); the L∞ "
            "projection still enforces correctness, but this likely indicates "
            "a misconfiguration.",
            stacklevel=2,
        )

    x_orig = x.detach()
    if x_bounds is not None:
        x_min, x_max = x_bounds
    else:
        x_min, x_max = samplewise_bounds(x_orig)

    if random_start:
        delta = (torch.rand_like(x_orig) * 2.0 - 1.0) * float(max_eps)
        x_adv = _project_linf(x_orig + delta, x_orig, max_eps, x_min, x_max)
    else:
        x_adv = x_orig.clone()

    snapshot_eps = sorted(set(snapshot_eps or []))
    snapshots: dict[float, torch.Tensor] = {}
    if 0.0 in snapshot_eps:
        snapshots[0.0] = x_orig.clone()
    pending = [e for e in snapshot_eps if e > 0.0]

    # Disable parameter grads — only x_adv needs them.
    saved_grad_states = [(p, p.requires_grad) for p in model.parameters()]
    for p, _ in saved_grad_states:
        p.requires_grad_(False)

    grad_accum = torch.zeros_like(x_orig)  # for momentum
    try:
        for k in range(1, n_steps + 1):
            x_adv = x_adv.detach().requires_grad_(True)
            model.zero_grad(set_to_none=True)
            logits = model(x_adv)
            loss = dice_ce_loss(logits, y, num_classes)
            if loss_log is not None:
                loss_log.append(float(loss.detach().cpu()))
            loss.backward()
            grad = x_adv.grad
            if grad is None:
                warnings.warn(
                    "x_adv.grad is None — loss does not depend on input "
                    "(e.g. all voxels masked). Returning clean image.",
                    stacklevel=2,
                )
                clean = x_orig.clone()
                for e in pending:
                    snapshots[e] = clean.clone()
                snapshots[float(max_eps)] = clean
                return clean, snapshots
            g = grad.detach()
            if momentum > 0.0:
                # MI-FGSM normalization: divide by L1 norm per sample
                l1 = (
                    g.abs()
                    .mean(dim=tuple(range(1, g.ndim)), keepdim=True)
                    .clamp_min(1e-12)
                )
                grad_accum = momentum * grad_accum + g / l1
                step_dir = grad_accum
            elif normalized_grad:
                l2 = (
                    g.pow(2)
                    .mean(dim=tuple(range(1, g.ndim)), keepdim=True)
                    .sqrt()
                    .clamp_min(1e-12)
                )
                step_dir = g / l2
            else:
                step_dir = g
            x_adv = x_adv.detach() + step_size * step_dir.sign()
            x_adv = _project_linf(x_adv, x_orig, max_eps, x_min, x_max)

            # Snapshot any pending eps whose budget k*step_size is now reached.
            still_pending = []
            for e in pending:
                if k * step_size + 1e-12 >= e or k == n_steps:
                    snapshots[e] = _project_linf(
                        x_adv.detach(), x_orig, e, x_min, x_max
                    ).clone()
                else:
                    still_pending.append(e)
            pending = still_pending
    finally:
        for p, state in saved_grad_states:
            p.requires_grad_(state)

    snapshots[float(max_eps)] = x_adv.detach()
    return x_adv.detach(), snapshots


def pgd_attack(
    model: torch.nn.Module,
    x: torch.Tensor,
    y: torch.Tensor,
    eps: float,
    num_classes: int,
    n_steps: int,
    step_size: float | None = None,
    random_start: bool = False,
    x_bounds=None,
    momentum: float = 0.0,
    normalized_grad: bool = False,
) -> torch.Tensor:
    """Backwards-compatible single-eps PGD wrapper around ``pgd_trajectory``."""
    if eps == 0.0:
        return x.clone()
    x_adv, _ = pgd_trajectory(
        model,
        x,
        y,
        eps,
        num_classes,
        n_steps=n_steps,
        step_size=step_size,
        random_start=random_start,
        x_bounds=x_bounds,
        momentum=momentum,
        normalized_grad=normalized_grad,
    )
    return x_adv


def pgd_with_restarts(
    model,
    x,
    y,
    max_eps,
    num_classes,
    n_steps,
    step_size,
    momentum,
    normalized_grad,
    x_bounds,
    snapshot_eps,
    num_restarts: int,
    loss_log: list[float] | None,
    random_start: bool = False,
):
    """Run multiple restarts; per snapshot eps keep the perturbation that
    yields the *highest* loss for the victim model (worst-case for defender).

    Restart 0 honours ``random_start``; every additional restart is randomized.
    With ``random_start=False`` the first trajectory therefore starts from the
    clean image, so ``num_restarts>1`` searches a superset of the single-restart
    attack and can only be as strong or stronger."""
    best_snaps: dict[float, torch.Tensor] = {}
    best_loss: dict[float, float] = {}
    for r in range(max(1, num_restarts)):
        rs = (r > 0) or random_start  # extra restarts are always randomized
        _, snaps = pgd_trajectory(
            model,
            x,
            y,
            max_eps,
            num_classes,
            n_steps=n_steps,
            step_size=step_size,
            random_start=rs,
            momentum=momentum,
            normalized_grad=normalized_grad,
            x_bounds=x_bounds,
            snapshot_eps=snapshot_eps,
            loss_log=loss_log if r == 0 else None,
        )
        with torch.no_grad():
            for e, xa in snaps.items():
                lv = float(dice_ce_loss(model(xa), y, num_classes).detach().cpu())
                if e not in best_loss or lv > best_loss[e]:
                    best_loss[e] = lv
                    best_snaps[e] = xa
    return best_snaps


# ---------------------------------------------------------------------------
# Reporting
# ---------------------------------------------------------------------------


def _agg(vals):
    vals = np.array(vals, dtype=np.float64)
    vals = vals[np.isfinite(vals)]
    n = len(vals)
    mean = float(vals.mean()) if n > 0 else np.nan
    std = float(vals.std(ddof=1)) if n > 1 else np.nan
    sem = std / np.sqrt(n) if n > 1 else np.nan
    ci95 = 1.96 * sem if n > 1 else np.nan
    return mean, std, sem, ci95, n


def _save_aggregate_csv(
    results, fg_classes, class_names, eps_list, model_key, output_dir, fold
):
    for metric_name in ["dice", "hd95", "asd"]:
        path = os.path.join(
            output_dir, f"{model_key}_{ATTACK_STEM}_fold{fold}_{metric_name}.csv"
        )
        with open(path, "w", newline="") as f:
            w = csv.writer(f)
            w.writerow(["epsilon", "class", "mean", "std", "sem", "ci95", "n"])
            for eps in eps_list:
                for c, cname in zip(fg_classes, class_names):
                    mean, std, sem, ci95, n = _agg(results[eps][c][metric_name])
                    w.writerow(
                        [
                            eps,
                            cname,
                            f"{mean:.6f}",
                            f"{std:.6f}",
                            f"{sem:.6f}",
                            f"{ci95:.6f}",
                            n,
                        ]
                    )
        print(f"  Saved {path}")


def _save_persample_csv(per_sample_rows, model_key, output_dir, fold):
    path = os.path.join(
        output_dir, f"{model_key}_{ATTACK_STEM}_fold{fold}_per_sample.csv"
    )
    new = not os.path.isfile(path)
    with open(path, "a", newline="") as f:
        w = csv.writer(f)
        if new:
            w.writerow(["case_id", "epsilon", "class", "dice", "hd95", "asd"])
        for row in per_sample_rows:
            w.writerow(row)
    print(f"  Updated {path} (+{len(per_sample_rows)} rows)")
    return path


def _read_completed_cases(persample_path: str) -> set[str]:
    if not os.path.isfile(persample_path):
        return set()
    done: set[str] = set()
    with open(persample_path) as f:
        r = csv.reader(f)
        next(r, None)
        for row in r:
            if row:
                done.add(row[0])
    return done


def _save_plots(
    results,
    fg_classes,
    class_names,
    eps_list,
    model_key,
    model_display,
    output_dir,
    fold,
):
    for metric_name, ylabel in [
        ("dice", "Dice"),
        ("hd95", "HD95 (vox)"),
        ("asd", "ASD (vox)"),
    ]:
        fig, ax = plt.subplots(figsize=(8, 5))
        for c, cname in zip(fg_classes, class_names):
            means, lo, hi = [], [], []
            for eps in eps_list:
                mean, _, _, ci95, _ = _agg(results[eps][c][metric_name])
                means.append(mean)
                lo.append(mean - (ci95 if np.isfinite(ci95) else 0.0))
                hi.append(mean + (ci95 if np.isfinite(ci95) else 0.0))
            (line,) = ax.plot(eps_list, means, marker="o", label=cname)
            ax.fill_between(eps_list, lo, hi, alpha=0.2, color=line.get_color())
        ax.set_xlabel("PGD Epsilon (L∞)")
        ax.set_ylabel(ylabel)
        ax.set_title(f"{model_display} — {ylabel} vs PGD ε (fold {fold}, 95% CI)")
        ax.legend()
        ax.grid(True, alpha=0.3)
        plt.tight_layout()
        path = os.path.join(
            output_dir, f"{model_key}_{ATTACK_STEM}_fold{fold}_{metric_name}.png"
        )
        plt.savefig(path, dpi=300, bbox_inches="tight")
        plt.close(fig)
        print(f"  Saved {path}")


def _attack_success_rate(per_sample, fg_classes, eps_list, threshold: float):
    """ASR per (class, eps) = fraction of samples whose Dice fell by ≥ threshold
    vs the clean (eps=0) Dice for the same case."""
    by_case = defaultdict(dict)  # (case, class) -> {eps: dice}
    for case_id, eps, cname, dice, _h, _a in per_sample:
        by_case[(case_id, cname)][float(eps)] = float(dice)
    asr: dict[tuple[str, float], float] = {}
    classes_seen = sorted({c for (_, c) in by_case})
    for cname in classes_seen:
        for eps in eps_list:
            if eps == 0.0:
                asr[(cname, eps)] = 0.0
                continue
            hits = total = 0
            for (cs, cn), d in by_case.items():
                if cn != cname or 0.0 not in d or eps not in d:
                    continue
                total += 1
                if (d[0.0] - d[eps]) >= threshold:
                    hits += 1
            asr[(cname, eps)] = (hits / total) if total else float("nan")
    return asr, classes_seen


def _save_asr(asr, classes, eps_list, model_key, output_dir, fold, threshold):
    path = os.path.join(output_dir, f"{model_key}_{ATTACK_STEM}_fold{fold}_asr.csv")
    with open(path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["epsilon", "class", "asr", "dice_drop_threshold"])
        for cname in classes:
            for eps in eps_list:
                w.writerow(
                    [
                        eps,
                        cname,
                        f"{asr.get((cname, eps), float('nan')):.6f}",
                        threshold,
                    ]
                )
    print(f"  Saved {path}")


def _print_summary(results, fg_classes, class_names, eps_list):
    print("\n  Summary (PGD):")
    print(f"  {'Epsilon':<10}", end="")
    for cname in class_names:
        print(
            f"  {cname + ' Dice':<14} {cname + ' HD95':<14} {cname + ' ASD':<14}",
            end="",
        )
    print()
    for eps in eps_list:
        print(f"  {eps:<10.3f}", end="")
        for c in fg_classes:
            for m in ["dice", "hd95", "asd"]:
                mean, *_ = _agg(results[eps][c][m])
                print(f"  {mean:<14.4f}", end="")
        print()
    print()


# ---------------------------------------------------------------------------
# Per-model evaluation loop
# ---------------------------------------------------------------------------


def a_pgd_with_restarts(
    model,
    x,
    y,
    max_eps,
    num_classes,
    n_steps,
    step_size,
    momentum,
    normalized_grad,
    x_bounds,
    snapshot_eps,
    num_restarts: int,
    loss_log: list[float] | None,
    random_start: bool = False,
):
    """Adaptive-step PGD with the same restart semantics as ``pgd_with_restarts``.

    Delegates to the manuscript's attack (``mri_prostate_seg.attacks.a_pgd``),
    the implementation behind the APGD row of Table I: Dice+CE objective, step
    size starting at 2*eps/k, checkpoints at 25/50/75% of the budget, momentum
    on the accumulated gradient. ``normalized_grad`` is accepted for signature
    compatibility and ignored, because the adaptive attack always normalizes by
    the mean absolute gradient.
    """
    # Table I's implementation, not the current a_pgd.py — see that module's
    # docstring for why the two differ in strength.
    from mri_prostate_seg.attacks.a_pgd_campaign import a_pgd_trajectory

    best_snaps: dict[float, torch.Tensor] = {}
    best_loss: dict[float, float] = {}
    for r in range(max(1, num_restarts)):
        rs = (r > 0) or random_start
        _, snaps = a_pgd_trajectory(
            model,
            x,
            y,
            max_eps,
            num_classes,
            dice_ce_loss,
            n_steps=n_steps,
            step_size=step_size,
            random_start=rs,
            momentum=momentum,
            x_bounds=x_bounds,
            adaptation_rate=APGD_ADAPTATION_RATE,
            checkpoints=APGD_CHECKPOINTS,
            snapshot_eps=snapshot_eps,
            loss_log=loss_log if r == 0 else None,
        )
        with torch.no_grad():
            for e, xa in snaps.items():
                lv = float(dice_ce_loss(model(xa), y, num_classes).detach().cpu())
                if e not in best_loss or lv > best_loss[e]:
                    best_loss[e] = lv
                    best_snaps[e] = xa
    return best_snaps


def auto_pgd_with_restarts(
    model,
    x,
    y,
    max_eps,
    num_classes,
    n_steps,
    step_size,
    momentum,
    normalized_grad,
    x_bounds,
    snapshot_eps,
    num_restarts: int,
    loss_log: list[float] | None,
    random_start: bool = False,
):
    """Original Auto-PGD with the same restart semantics as ``pgd_with_restarts``.

    Delegates to ``mri_prostate_seg.attacks.auto_pgd``, the port of Croce and
    Hein's published attack. ``step_size``, ``momentum`` and ``normalized_grad``
    are accepted for signature compatibility and ignored: Auto-PGD fixes the
    step at 2*eps, applies its momentum to the iterates with a hard-coded 0.75,
    and takes a raw gradient sign.
    """
    from mri_prostate_seg.attacks.auto_pgd import auto_pgd_trajectory

    best_snaps: dict[float, torch.Tensor] = {}
    best_loss: dict[float, float] = {}
    for r in range(max(1, num_restarts)):
        rs = (r > 0) or random_start
        _, snaps = auto_pgd_trajectory(
            model,
            x,
            y,
            max_eps,
            num_classes,
            dice_ce_loss,
            n_steps=n_steps,
            random_start=rs,
            x_bounds=x_bounds,
            snapshot_eps=snapshot_eps,
            loss_log=loss_log if r == 0 else None,
        )
        with torch.no_grad():
            for e, xa in snaps.items():
                lv = float(dice_ce_loss(model(xa), y, num_classes).detach().cpu())
                if e not in best_loss or lv > best_loss[e]:
                    best_loss[e] = lv
                    best_snaps[e] = xa
    return best_snaps


def auto_pgd_plus_with_restarts(
    model,
    x,
    y,
    max_eps,
    num_classes,
    n_steps,
    step_size,
    momentum,
    normalized_grad,
    x_bounds,
    snapshot_eps,
    num_restarts: int,
    loss_log: list[float] | None,
    random_start: bool = False,
):
    """Auto-PGD+ with the same restart semantics as ``pgd_with_restarts``.

    Delegates to ``mri_prostate_seg.attacks.auto_pgd_plus``, the repaired
    Auto-PGD. ``step_size``, ``momentum`` and ``normalized_grad`` are accepted
    for signature compatibility and ignored, as for ``auto_pgd``: the step comes
    from a geometric backbone, momentum is on the iterates at a hard-coded 0.75,
    and the gradient sign is taken raw.
    """
    from mri_prostate_seg.attacks.auto_pgd_plus import auto_pgd_plus_trajectory

    best_snaps: dict[float, torch.Tensor] = {}
    best_loss: dict[float, float] = {}
    for r in range(max(1, num_restarts)):
        rs = (r > 0) or random_start
        _, snaps = auto_pgd_plus_trajectory(
            model,
            x,
            y,
            max_eps,
            num_classes,
            dice_ce_loss,
            n_steps=n_steps,
            random_start=rs,
            x_bounds=x_bounds,
            snapshot_eps=snapshot_eps,
            loss_log=loss_log if r == 0 else None,
        )
        with torch.no_grad():
            for e, xa in snaps.items():
                lv = float(dice_ce_loss(model(xa), y, num_classes).detach().cpu())
                if e not in best_loss or lv > best_loss[e]:
                    best_loss[e] = lv
                    best_snaps[e] = xa
    return best_snaps


def apgd_updated_with_restarts(
    model,
    x,
    y,
    max_eps,
    num_classes,
    n_steps,
    step_size,
    momentum,
    normalized_grad,
    x_bounds,
    snapshot_eps,
    num_restarts: int,
    loss_log: list[float] | None,
    random_start: bool = False,
):
    """Scheduled adaptive-step PGD, the improved-step arm.

    Delegates to ``mri_prostate_seg.attacks.apgd_updated``, which replaces the
    campaign attack's narrow 2*eps/k..1.33x band with a cosine backbone spanning
    2*eps down to eps/k, a bidirectional multiplier, and budget-invariant
    checkpoints. ``step_size`` and ``normalized_grad`` are accepted for signature
    compatibility and ignored: the step is set by the schedule, and the gradient
    is always normalized by its mean absolute value before accumulation.
    """
    from mri_prostate_seg.attacks.apgd_updated import apgd_updated_trajectory

    best_snaps: dict[float, torch.Tensor] = {}
    best_loss: dict[float, float] = {}
    for r in range(max(1, num_restarts)):
        rs = (r > 0) or random_start
        _, snaps = apgd_updated_trajectory(
            model,
            x,
            y,
            max_eps,
            num_classes,
            dice_ce_loss,
            n_steps=n_steps,
            random_start=rs,
            momentum=momentum,
            x_bounds=x_bounds,
            snapshot_eps=snapshot_eps,
            loss_log=loss_log if r == 0 else None,
        )
        with torch.no_grad():
            for e, xa in snaps.items():
                lv = float(dice_ce_loss(model(xa), y, num_classes).detach().cpu())
                if e not in best_loss or lv > best_loss[e]:
                    best_loss[e] = lv
                    best_snaps[e] = xa
    return best_snaps


def _loss_variant_with_restarts(
    trajectory_fn,
    model,
    x,
    y,
    max_eps,
    num_classes,
    n_steps,
    step_size,
    x_bounds,
    snapshot_eps,
    num_restarts: int,
    loss_log: list[float] | None,
    random_start: bool,
):
    """Restart loop shared by the two loss-variant PGD arms.

    Same semantics as ``pgd_with_restarts``: restart 0 honours ``random_start``,
    later restarts are randomised, and the worst snapshot per epsilon is kept.
    Snapshots are scored with ``dice_ce_loss`` like every other wrapper, never
    with the attack's own objective -- SegPGD's is iteration-scheduled and must
    not be evaluated outside its trajectory.
    """
    best_snaps: dict[float, torch.Tensor] = {}
    best_loss: dict[float, float] = {}
    for r in range(max(1, num_restarts)):
        rs = (r > 0) or random_start
        _, snaps = trajectory_fn(
            model,
            x,
            y,
            max_eps,
            num_classes,
            n_steps=n_steps,
            step_size=step_size,
            random_start=rs,
            x_bounds=x_bounds,
            snapshot_eps=snapshot_eps,
            loss_log=loss_log if r == 0 else None,
        )
        with torch.no_grad():
            for e, xa in snaps.items():
                lv = float(dice_ce_loss(model(xa), y, num_classes).detach().cpu())
                if e not in best_loss or lv > best_loss[e]:
                    best_loss[e] = lv
                    best_snaps[e] = xa
    return best_snaps


def segpgd_with_restarts(
    model,
    x,
    y,
    max_eps,
    num_classes,
    n_steps,
    step_size,
    momentum,
    normalized_grad,
    x_bounds,
    snapshot_eps,
    num_restarts: int,
    loss_log: list[float] | None,
    random_start: bool = False,
):
    """SegPGD (Gu et al., ECCV 2022) with ``pgd_with_restarts`` semantics.

    Delegates to ``mri_prostate_seg.attacks.segpgd``. ``momentum`` and
    ``normalized_grad`` are accepted for signature compatibility and ignored:
    the arm uses the plain PGD sign update so only the objective differs from
    ``pgd``. ``step_size`` defaults to ``max_eps / n_steps`` as for ``pgd``.
    """
    from mri_prostate_seg.attacks.segpgd import segpgd_trajectory

    return _loss_variant_with_restarts(
        segpgd_trajectory, model, x, y, max_eps, num_classes, n_steps,
        step_size, x_bounds, snapshot_eps, num_restarts, loss_log, random_start,
    )


def cospgd_with_restarts(
    model,
    x,
    y,
    max_eps,
    num_classes,
    n_steps,
    step_size,
    momentum,
    normalized_grad,
    x_bounds,
    snapshot_eps,
    num_restarts: int,
    loss_log: list[float] | None,
    random_start: bool = False,
):
    """CosPGD (Agnihotri et al., ICML 2024) with ``pgd_with_restarts`` semantics.

    Delegates to ``mri_prostate_seg.attacks.cospgd``. ``momentum`` and
    ``normalized_grad`` are accepted for signature compatibility and ignored,
    as for ``segpgd``.
    """
    from mri_prostate_seg.attacks.cospgd import cospgd_trajectory

    return _loss_variant_with_restarts(
        cospgd_trajectory, model, x, y, max_eps, num_classes, n_steps,
        step_size, x_bounds, snapshot_eps, num_restarts, loss_log, random_start,
    )


def dag_with_restarts(
    model,
    x,
    y,
    max_eps,
    num_classes,
    n_steps,
    step_size,
    momentum,
    normalized_grad,
    x_bounds,
    snapshot_eps,
    num_restarts: int,
    loss_log: list[float] | None,
    random_start: bool = False,
):
    """DAG (Xie et al., ICCV 2017) with ``pgd_with_restarts`` semantics.

    Delegates to ``mri_prostate_seg.attacks.dag``. ``momentum`` and
    ``normalized_grad`` are accepted for signature compatibility and ignored:
    DAG normalises its gradient by the L-infinity norm by construction. The
    paper starts at the clean image, so ``random_start`` is off unless asked;
    later restarts are randomised as for the other arms.
    """
    from mri_prostate_seg.attacks.dag import dag_trajectory

    return _loss_variant_with_restarts(
        dag_trajectory, model, x, y, max_eps, num_classes, n_steps,
        step_size, x_bounds, snapshot_eps, num_restarts, loss_log, random_start,
    )


def sea_with_restarts(
    model,
    x,
    y,
    max_eps,
    num_classes,
    n_steps,
    step_size,
    momentum,
    normalized_grad,
    x_bounds,
    snapshot_eps,
    num_restarts: int,
    loss_log: list[float] | None,
    random_start: bool = False,
):
    """SEA (Croce, Singh, Hein 2023) with ``pgd_with_restarts`` semantics.

    Delegates to ``mri_prostate_seg.attacks.sea``: Auto-PGD once per loss in
    {masked CE, balanced masked CE, Jensen-Shannon}, worst foreground Dice per
    snapshot budget. ``step_size``, ``momentum`` and ``normalized_grad`` are
    ignored, as for ``auto_pgd``. Across restarts the snapshots are scored by
    ``dice_ce_loss`` like every other wrapper.
    """
    from mri_prostate_seg.attacks.sea import sea_trajectory

    def _trajectory(model, x, y, max_eps, num_classes, n_steps, step_size,
                    random_start, x_bounds, snapshot_eps, loss_log):
        del step_size
        return sea_trajectory(
            model, x, y, max_eps, num_classes, n_steps,
            random_start=random_start, x_bounds=x_bounds,
            snapshot_eps=snapshot_eps, loss_log=loss_log,
        )

    return _loss_variant_with_restarts(
        _trajectory, model, x, y, max_eps, num_classes, n_steps,
        step_size, x_bounds, snapshot_eps, num_restarts, loss_log, random_start,
    )


def alma_prox_with_restarts(
    model,
    x,
    y,
    max_eps,
    num_classes,
    n_steps,
    step_size,
    momentum,
    normalized_grad,
    x_bounds,
    snapshot_eps,
    num_restarts: int,
    loss_log: list[float] | None,
    random_start: bool = False,
    spacing: tuple[float, float, float] | None = None,
):
    """ALMA prox (Rony, Pesquet, Ben Ayed, CVPR 2023) with the wrapper contract.

    Delegates to ``mri_prostate_seg.attacks.alma_prox``. This is the only arm
    here that does not maximise damage inside a fixed ball: it minimises
    ``||delta||_inf`` subject to the image being misclassified, so ``max_eps``
    selects the returned iterate rather than constraining the search, and every
    requested epsilon is served from the one trajectory.

    ``step_size``, ``momentum``, ``normalized_grad`` and ``random_start`` are
    accepted for signature compatibility and ignored -- the attack starts at
    ``delta = 0`` and takes a proximal-gradient step in its own RMS metric.
    ``num_restarts`` is ignored too: the attack is deterministic, so every
    restart would return the same perturbation.

    Its constraints are restricted to the foreground dilated by
    ``ALMA_CONSTRAINT_BAND_MM``, using the plan's voxel spacing, so the
    adversarial-pixel threshold refers to the gland and the tissue around it
    rather than to a background-dominated whole volume.
    """
    from mri_prostate_seg.attacks.alma_prox import alma_prox_trajectory

    del step_size, momentum, normalized_grad, random_start, num_restarts
    _, snaps = alma_prox_trajectory(
        model,
        x,
        y,
        max_eps,
        num_classes,
        n_steps,
        x_bounds=x_bounds,
        snapshot_eps=snapshot_eps,
        loss_log=loss_log,
        adv_threshold=ALMA_ADV_THRESHOLD,
        constraint_band_mm=ALMA_CONSTRAINT_BAND_MM,
        spacing=spacing,
    )
    return snaps


# The BLADE arms that get the multiclass implementation on the zones task:
# BLADE-MC for the ladder arms, BLADE-MM-MC (BLADE-MC ladders, BLADE-MM
# adjudication) for blade_mm.
BLADE_MC_ARMS = ("blade", "blade1", "blade_boundary", "blade_frontier", "blade_mm")


def _plan_spacing(cfg: dict) -> tuple[float, float, float]:
    """(D, H, W) voxel spacing in mm of the preprocessed grid the attack runs on.

    Read from the dataset's ``nnUNetPlans.json`` for ``cfg["configuration"]``;
    every plan of a dataset (vanilla and ResEnc) resamples to the same target
    spacing, which is what the preprocessed arrays are stored at.
    """
    plans_path = os.path.join(
        os.path.dirname(os.path.abspath(cfg["data_dir"])), "nnUNetPlans.json"
    )
    with open(plans_path) as f:
        spacing = json.load(f)["configurations"][cfg["configuration"]]["spacing"]
    return tuple(float(v) for v in spacing)


def _blade_mc_distance_maps(
    y: torch.Tensor, num_classes: int, spacing: tuple[float, ...] | None = None
) -> torch.Tensor:
    """Per-class signed-distance maps for BLADE-MC on the zones crops.

    ``blade_updated.signed_distance_maps`` refuses labels whose ignore region
    is not rectangular padding, and the zones crops are zero-filled outside the
    dilated whole gland (label -1, non-rectangular). Each class surface lies
    inside that gland with a ring of background around it, so the map is well
    defined there; this builds it the way the legacy binary BLADE did -- the
    ignore region counts as "not the class" for the EDT and is then zeroed so
    it carries no weight -- one channel per foreground class. ``spacing``
    (D, H, W) puts the distances in mm; ``None`` keeps voxel units.
    """
    from scipy.ndimage import distance_transform_edt

    labels = y[:, 0].detach().cpu().numpy()
    maps = np.zeros((labels.shape[0], num_classes, *labels.shape[1:]), dtype=np.float32)
    for b, lab in enumerate(labels):
        valid = lab != -1
        for c in range(1, num_classes):
            member = lab == c
            if not member.any() or member.all():
                continue
            phi = distance_transform_edt(~member, sampling=spacing) - distance_transform_edt(
                member, sampling=spacing
            )
            maps[b, c] = phi.astype(np.float32) * valid
    return torch.from_numpy(maps).to(device=y.device)


def _blade_with_restarts(
    losses,
    model,
    x,
    y,
    max_eps,
    num_classes,
    n_steps,
    step_size,
    x_bounds,
    snapshot_eps,
    num_restarts: int,
    loss_log: list[float] | None,
    random_start: bool,
    blade_impl: str = "legacy",
    spacing: tuple[float, ...] | None = None,
):
    """Restart loop shared by the two BLADE arms.

    Under ``--shared-trajectory`` the requested epsilons become the ladder's
    rungs, so every snapshot is natively optimised inside its own ball rather
    than projected down from ``max_eps``. Without it the driver calls this once
    per epsilon and the ladder degenerates to a single rung, which is the plain
    native attack -- correct, but with no warm start to inherit.

    ``blade_impl`` selects the implementation: ``"legacy"`` is the binary
    whole-gland BLADE the published trees used; ``"mc"`` is BLADE-MC
    (``blade_updated``), the multiclass version the zones re-evaluation uses
    (multiclass Dice+CE, class-specific signed-distance maps, class-balanced
    frontier, Dice-selected candidates).
    """
    if blade_impl == "mc":
        from mri_prostate_seg.attacks import blade_updated as mc

        def _trajectory(model, x, y, max_eps, num_classes, n_steps, step_size,
                        random_start, x_bounds, snapshot_eps, loss_log):
            del step_size
            y_int = y.long()
            return mc.blade_trajectory(
                model, x, y_int, max_eps, num_classes, None, n_steps,
                random_start=random_start, x_bounds=x_bounds,
                snapshot_eps=snapshot_eps, loss_log=loss_log, losses=losses,
                spacing=spacing,
                phi=(
                    _blade_mc_distance_maps(y_int, num_classes, spacing)
                    if "boundary" in losses else None
                ),
            )
    else:
        from mri_prostate_seg.attacks.blade_legacy import blade_trajectory

        def _trajectory(model, x, y, max_eps, num_classes, n_steps, step_size,
                        random_start, x_bounds, snapshot_eps, loss_log):
            del step_size
            return blade_trajectory(
                model, x, y, max_eps, num_classes, dice_ce_loss, n_steps,
                random_start=random_start, x_bounds=x_bounds,
                snapshot_eps=snapshot_eps, loss_log=loss_log, losses=losses,
            )

    return _loss_variant_with_restarts(
        _trajectory, model, x, y, max_eps, num_classes, n_steps,
        step_size, x_bounds, snapshot_eps, num_restarts, loss_log, random_start,
    )


def blade_with_restarts(
    model, x, y, max_eps, num_classes, n_steps, step_size, momentum,
    normalized_grad, x_bounds, snapshot_eps, num_restarts: int,
    loss_log: list[float] | None, random_start: bool = False,
    blade_impl: str = "legacy",
    spacing: tuple[float, ...] | None = None,
):
    """BLADE-3: the epsilon ladder run once per objective, worst Dice per rung.

    Costs three ladders, so it is budget-matched against ``sea``, not against
    the single-objective arms. ``step_size``, ``momentum`` and
    ``normalized_grad`` are ignored, as for ``auto_pgd``.
    """
    from mri_prostate_seg.attacks.blade_legacy import BLADE_LOSSES

    return _blade_with_restarts(
        BLADE_LOSSES, model, x, y, max_eps, num_classes, n_steps, step_size,
        x_bounds, snapshot_eps, num_restarts, loss_log, random_start,
        blade_impl=blade_impl, spacing=spacing,
    )


def blade_boundary_with_restarts(
    model, x, y, max_eps, num_classes, n_steps, step_size, momentum,
    normalized_grad, x_bounds, snapshot_eps, num_restarts: int,
    loss_log: list[float] | None, random_start: bool = False,
    blade_impl: str = "legacy",
    spacing: tuple[float, ...] | None = None,
):
    """BLADE-Boundary: the ladder on the sign-flipped boundary loss alone.

    One of the three per-objective marginals of ``blade``. Budget-matched
    against the single-objective arms, and the arm that shows what the boundary
    objective achieves on HD95 and ASD when nothing else is competing with it.
    """
    return _blade_with_restarts(
        ("boundary",), model, x, y, max_eps, num_classes, n_steps, step_size,
        x_bounds, snapshot_eps, num_restarts, loss_log, random_start,
        blade_impl=blade_impl, spacing=spacing,
    )


def blade_frontier_with_restarts(
    model, x, y, max_eps, num_classes, n_steps, step_size, momentum,
    normalized_grad, x_bounds, snapshot_eps, num_restarts: int,
    loss_log: list[float] | None, random_start: bool = False,
    blade_impl: str = "legacy",
    spacing: tuple[float, ...] | None = None,
):
    """BLADE-Frontier: the ladder on the frontier-margin objective alone.

    The third per-objective marginal. Together with blade1 (dice_ce) and
    blade_boundary it decomposes ``blade`` into its parts at one third the cost
    each, so the ensemble's gain over the best single objective is measurable.
    """
    return _blade_with_restarts(
        ("frontier",), model, x, y, max_eps, num_classes, n_steps, step_size,
        x_bounds, snapshot_eps, num_restarts, loss_log, random_start,
        blade_impl=blade_impl, spacing=spacing,
    )


def blade_mm_with_restarts(
    model, x, y, max_eps, num_classes, n_steps, step_size, momentum,
    normalized_grad, x_bounds, snapshot_eps, num_restarts: int,
    loss_log: list[float] | None, random_start: bool = False,
    blade_impl: str = "legacy",
    spacing: tuple[float, ...] | None = None,
):
    """BLADE-MM: the ladder ensemble adjudicated on Dice, HD95 and ASD together.

    Same cost and same three objectives as ``blade``; the only difference is
    which candidate wins a rung. ``blade`` keeps the lowest foreground Dice,
    which discards the candidate the boundary objective exists to produce;
    BLADE-MM ranks the candidates on every reported metric.

    The restart loop adjudicates the same way rather than deferring to
    ``dice_ce_loss`` as the other wrappers do -- reintroducing a Dice-only
    oracle one level up would defeat the point of the arm. ``step_size``,
    ``momentum`` and ``normalized_grad`` are ignored, as for ``auto_pgd``.
    """
    from mri_prostate_seg.attacks.blade_mm import (
        blade_mm_trajectory, foreground_metrics, select_by_rank,
    )

    del step_size, momentum, normalized_grad

    if blade_impl == "mc":
        # BLADE-MM-MC: multiclass ladders (blade_updated), same adjudication.
        from mri_prostate_seg.attacks.blade_mm_mc import blade_mm_mc_trajectory

        y_int = y.long()
        phi = _blade_mc_distance_maps(y_int, num_classes, spacing)

        def _trajectory(rs, log):
            return blade_mm_mc_trajectory(
                model, x, y_int, max_eps, num_classes, n_steps,
                random_start=rs, x_bounds=x_bounds, snapshot_eps=snapshot_eps,
                loss_log=log, phi=phi, spacing=spacing,
            )
    else:
        def _trajectory(rs, log):
            return blade_mm_trajectory(
                model, x, y, max_eps, num_classes, dice_ce_loss, n_steps,
                random_start=rs, x_bounds=x_bounds, snapshot_eps=snapshot_eps,
                loss_log=log,
            )

    best_snaps: dict[float, torch.Tensor] = {}
    best_metrics: dict[float, dict[str, float]] = {}
    for r in range(max(1, num_restarts)):
        rs = (r > 0) or random_start
        _, snaps = _trajectory(rs, loss_log if r == 0 else None)
        with torch.no_grad():
            for e, xa in snaps.items():
                m = foreground_metrics(model(xa), y, num_classes)
                if e not in best_snaps:
                    best_snaps[e], best_metrics[e] = xa, m
                    continue
                pair = {"incumbent": best_metrics[e], "challenger": m}
                if select_by_rank(pair) == "challenger":
                    best_snaps[e], best_metrics[e] = xa, m
    return best_snaps


def blade1_with_restarts(
    model, x, y, max_eps, num_classes, n_steps, step_size, momentum,
    normalized_grad, x_bounds, snapshot_eps, num_restarts: int,
    loss_log: list[float] | None, random_start: bool = False,
    blade_impl: str = "legacy",
    spacing: tuple[float, ...] | None = None,
):
    """BLADE-1: the epsilon ladder on ``dice_ce_loss`` alone.

    One ladder, so it is budget-matched against the single-objective arms
    (``pgd``, ``a_pgd``, ``auto_pgd``, ``segpgd``, ``cospgd``).
    """
    return _blade_with_restarts(
        ("dice_ce",), model, x, y, max_eps, num_classes, n_steps, step_size,
        x_bounds, snapshot_eps, num_restarts, loss_log, random_start,
        blade_impl=blade_impl, spacing=spacing,
    )



def sea_ladder_with_restarts(
    model, x, y, max_eps, num_classes, n_steps, step_size, momentum,
    normalized_grad, x_bounds, snapshot_eps, num_restarts: int,
    loss_log: list[float] | None, random_start: bool = False,
    blade_impl: str = "legacy",
    spacing: tuple[float, ...] | None = None,
):
    """SEA-ladder: SEA's three losses run through BLADE's epsilon ladder.

    Factorial control. Same machinery and adjudication as ``blade`` (one ladder
    per objective, worst Dice per rung; ``blade_impl`` picks the legacy binary
    or the BLADE-MC ladder), with SEA's masked CE, balanced masked CE and
    Jensen-Shannon losses in place of BLADE's. Against ``blade`` it isolates the
    objectives on the ladder; against ``sea`` it isolates the ladder on SEA's
    objectives. 315 gradients per case at 20 steps, like both.
    """
    from mri_prostate_seg.attacks.ladder_ensemble import ladder_ensemble_trajectory
    from mri_prostate_seg.attacks.sea import LOSS_FNS

    del momentum, normalized_grad, spacing

    def _trajectory(model, x, y, max_eps, num_classes, n_steps, step_size,
                    random_start, x_bounds, snapshot_eps, loss_log):
        del step_size
        return ladder_ensemble_trajectory(
            model, x, y.long() if blade_impl == "mc" else y, max_eps, num_classes,
            dict(LOSS_FNS), n_steps, random_start=random_start, x_bounds=x_bounds,
            snapshot_eps=snapshot_eps, loss_log=loss_log, impl=blade_impl,
        )

    return _loss_variant_with_restarts(
        _trajectory, model, x, y, max_eps, num_classes, n_steps,
        step_size, x_bounds, snapshot_eps, num_restarts, loss_log, random_start,
    )


def blade_noladder_with_restarts(
    model, x, y, max_eps, num_classes, n_steps, step_size, momentum,
    normalized_grad, x_bounds, snapshot_eps, num_restarts: int,
    loss_log: list[float] | None, random_start: bool = False,
    blade_impl: str = "legacy",
    spacing: tuple[float, ...] | None = None,
):
    """BLADE-3 without the ladder: its three objectives, one rung per epsilon.

    Factorial control. Identical to ``blade_with_restarts``; what removes the
    ladder is the caller, which must invoke it once per epsilon with
    ``snapshot_eps=[eps]`` (as the driver does for every non-ladder arm), so each
    budget is a single rung with nothing to warm-start from. Against ``blade``
    it isolates the warm start; against ``sea`` it isolates BLADE's objectives
    from SEA's. 315 gradients per case at 20 steps.
    """
    return blade_with_restarts(
        model, x, y, max_eps, num_classes, n_steps, step_size, momentum,
        normalized_grad, x_bounds, snapshot_eps, num_restarts, loss_log,
        random_start, blade_impl=blade_impl, spacing=spacing,
    )


def evaluate_pgd(
    model_key: str,
    device: torch.device,
    max_samples: int | None,
    output_dir: str,
    cfg: dict,
    n_steps: int,
    step_size: float | None,
    random_start: bool,
    fold: int = 0,
    seed: int | None = None,
    momentum: float = 0.0,
    normalized_grad: bool = False,
    num_restarts: int = 1,
    shared_trajectory: bool = False,
    save_adv_dir: str | None = None,
    log_loss_curve: bool = False,
    amp: bool = False,
    resume: bool = False,
    asr_threshold: float = 0.1,
    attack: str = "pgd",
    attack_valid_only: bool = False,
):
    cfg = {**cfg}
    cfg["epsilons"] = sorted({float(e) for e in cfg["epsilons"]})

    print(f"\n{'=' * 64}")
    print(
        f"{attack.upper()}: {cfg['name']} ({model_key})  fold={fold}  steps={n_steps}  "
        f"restarts={num_restarts}  shared_traj={shared_trajectory}  "
        f"momentum={momentum}  norm_grad={normalized_grad}  amp={amp}  "
        f"valid_only={attack_valid_only}"
    )
    print(f"{'=' * 64}")

    os.makedirs(output_dir, exist_ok=True)
    if save_adv_dir:
        os.makedirs(save_adv_dir, exist_ok=True)

    checkpoint_path = cfg.get("checkpoint") or get_checkpoint_for_fold(cfg, fold)
    network, div_factors = load_model(
        checkpoint_path, cfg["configuration"], cfg["num_classes"], device
    )
    network.eval()

    val_ids = load_validation_ids(cfg["splits_json"], fold)
    if max_samples is not None:
        val_ids = val_ids[:max_samples]
    if not val_ids:
        raise ValueError(f"No val samples for fold {fold} in {cfg['splits_json']}.")

    print(f"Checkpoint: {checkpoint_path}")
    print(f"Validation samples: {len(val_ids)}  div_factors: {div_factors}")

    if seed is not None:
        torch.manual_seed(seed)
        np.random.seed(seed)

    num_classes = cfg["num_classes"]
    class_names = cfg["class_names"]
    eps_list = cfg["epsilons"]
    fg_classes = list(range(1, num_classes))
    max_eps = max(eps_list) if eps_list else 0.0

    persample_path = os.path.join(
        output_dir, f"{model_key}_{ATTACK_STEM}_fold{fold}_per_sample.csv"
    )
    completed = _read_completed_cases(persample_path) if resume else set()
    if completed:
        print(f"Resume: skipping {len(completed)} cases already in {persample_path}")

    results = {
        eps: {c: {"dice": [], "hd95": [], "asd": []} for c in fg_classes}
        for eps in eps_list
    }
    persample_rows: list[tuple] = []
    loss_curve_rows: list[tuple] = []

    amp_ctx = (
        torch.autocast(device_type="cuda", dtype=torch.float16)
        if (amp and device.type == "cuda")
        else nullcontext()
    )

    attack_fn = {
        "a_pgd": a_pgd_with_restarts,
        "auto_pgd": auto_pgd_with_restarts,
        "auto_pgd_plus": auto_pgd_plus_with_restarts,
        "apgd_updated": apgd_updated_with_restarts,
        "segpgd": segpgd_with_restarts,
        "cospgd": cospgd_with_restarts,
        "dag": dag_with_restarts,
        "sea": sea_with_restarts,
        "alma_prox": alma_prox_with_restarts,
        "blade": blade_with_restarts,
        "blade1": blade1_with_restarts,
        "blade_mm": blade_mm_with_restarts,
        "blade_boundary": blade_boundary_with_restarts,
        "blade_frontier": blade_frontier_with_restarts,
    }.get(attack, pgd_with_restarts)
    if attack in MIN_NORM_ATTACKS:
        # The constraint band is defined in millimetres, so the arm needs the
        # spacing of the grid it runs on.
        attack_fn = functools.partial(attack_fn, spacing=_plan_spacing(cfg))
    if model_key == "zones" and attack in BLADE_MC_ARMS:
        # The zones task is multiclass; the binary legacy BLADE would collapse
        # TZ+CZ and PZ into one foreground. BLADE-MC keeps them apart.
        attack_fn = functools.partial(
            attack_fn, blade_impl="mc", spacing=_plan_spacing(cfg)
        )

    pbar = tqdm(val_ids, desc=f"{model_key} fold{fold}", unit="case")
    for i, case_id in enumerate(pbar):
        if case_id in completed:
            continue

        img_np, seg_np = load_sample(cfg["data_dir"], case_id)
        orig_seg_np = np.asarray(seg_np, dtype=np.int64)

        x = torch.from_numpy(img_np[np.newaxis]).to(device)
        y = torch.from_numpy(orig_seg_np[np.newaxis].astype(np.float32)).to(device)
        x_min, x_max = samplewise_bounds(x)
        x_padded, orig_shape = pad_to_divisible(x, div_factors)
        y_padded, _ = pad_to_divisible(y, div_factors)
        # Under --attack-valid-only the attack only ever sees the clean image
        # outside the attackable mask, so its gradient there is zero for every
        # arm; the snapshots are confined again below before evaluation.
        mask = attackable_mask(y, div_factors) if attack_valid_only else None
        attacked = ConfinedInput(network, x_padded, mask) if mask is not None else network

        case_loss_curve: list[float] = [] if log_loss_curve else None  # type: ignore

        one_call = shared_trajectory or attack in MIN_NORM_ATTACKS
        if one_call and max_eps > 0.0:
            with torch.enable_grad():
                snaps = attack_fn(
                    attacked,
                    x_padded,
                    y_padded,
                    max_eps,
                    num_classes,
                    n_steps=n_steps,
                    step_size=step_size,
                    momentum=momentum,
                    normalized_grad=normalized_grad,
                    x_bounds=(x_min, x_max),
                    snapshot_eps=eps_list,
                    num_restarts=num_restarts,
                    loss_log=case_loss_curve,
                    random_start=random_start,
                )
            # ensure clean image is present for eps==0.0
            snaps.setdefault(0.0, x_padded.clone())
        else:
            snaps = {}
            for eps in eps_list:
                if eps == 0.0:
                    snaps[eps] = x_padded.clone()
                    continue
                with torch.enable_grad():
                    s = attack_fn(
                        attacked,
                        x_padded,
                        y_padded,
                        eps,
                        num_classes,
                        n_steps=n_steps,
                        step_size=step_size,
                        momentum=momentum,
                        normalized_grad=normalized_grad,
                        x_bounds=(x_min, x_max),
                        snapshot_eps=[eps],
                        num_restarts=num_restarts,
                        loss_log=case_loss_curve,
                        random_start=random_start,
                    )
                snaps[eps] = s[eps]

        for eps in eps_list:
            x_input = confine(snaps[eps], x_padded, mask)
            with torch.no_grad(), amp_ctx:
                logits_padded = network(x_input)
            logits = unpad(logits_padded.float(), orig_shape)
            metrics = compute_metrics(logits, orig_seg_np[np.newaxis], num_classes)
            for c, cname in zip(fg_classes, class_names):
                d, h, a = metrics[c]
                results[eps][c]["dice"].append(d)
                results[eps][c]["hd95"].append(h)
                results[eps][c]["asd"].append(a)
                persample_rows.append(
                    (case_id, eps, cname, f"{d:.6f}", f"{h:.6f}", f"{a:.6f}")
                )
            if save_adv_dir and eps > 0.0:
                np.savez_compressed(
                    os.path.join(
                        save_adv_dir,
                        f"{model_key}_fold{fold}_{case_id}_eps{eps:.4f}.npz",
                    ),
                    x_adv=unpad(x_input, orig_shape).detach().cpu().numpy(),
                )
            del logits_padded, logits

        if log_loss_curve and case_loss_curve:
            for step_idx, lv in enumerate(case_loss_curve):
                loss_curve_rows.append((case_id, step_idx, lv))

        # Periodically flush per-sample rows so resume works after interruption.
        if len(persample_rows) >= 500 or (i + 1) == len(val_ids):
            _save_persample_csv(persample_rows, model_key, output_dir, fold)
            persample_rows.clear()

        del x, y, x_padded, y_padded, x_min, x_max, snaps, mask, attacked

    if persample_rows:
        _save_persample_csv(persample_rows, model_key, output_dir, fold)

    if log_loss_curve and loss_curve_rows:
        path = os.path.join(
            output_dir, f"{model_key}_{ATTACK_STEM}_fold{fold}_loss_curve.csv"
        )
        with open(path, "w", newline="") as f:
            w = csv.writer(f)
            w.writerow(["case_id", "step", "loss"])
            w.writerows(loss_curve_rows)
        print(f"  Saved {path}")

    _save_aggregate_csv(
        results, fg_classes, class_names, eps_list, model_key, output_dir, fold
    )
    _save_plots(
        results,
        fg_classes,
        class_names,
        eps_list,
        model_key,
        cfg["name"],
        output_dir,
        fold,
    )

    # ASR computed from the freshly written per-sample CSV (covers resumed runs).
    all_rows: list[tuple] = []
    if os.path.isfile(persample_path):
        with open(persample_path) as f:
            r = csv.reader(f)
            next(r, None)
            for row in r:
                if not row:
                    continue
                all_rows.append(
                    (
                        row[0],
                        float(row[1]),
                        row[2],
                        float(row[3]),
                        float(row[4]),
                        float(row[5]),
                    )
                )
    asr, asr_classes = _attack_success_rate(
        all_rows, fg_classes, eps_list, asr_threshold
    )
    _save_asr(asr, asr_classes, eps_list, model_key, output_dir, fold, asr_threshold)

    _print_summary(results, fg_classes, class_names, eps_list)

    # Hand back the aggregate for cross-model plots.
    summary = {
        "model_key": model_key,
        "model_display": cfg["name"],
        "fold": fold,
        "eps_list": eps_list,
        "fg_classes": fg_classes,
        "class_names": class_names,
        "results": {
            eps: {
                c: {m: list(results[eps][c][m]) for m in ("dice", "hd95", "asd")}
                for c in fg_classes
            }
            for eps in eps_list
        },
    }

    del results
    del network
    gc.collect()
    if device.type == "cuda":
        torch.cuda.empty_cache()
    return summary


# ---------------------------------------------------------------------------
# Cross-model combined plot
# ---------------------------------------------------------------------------


def _combined_plots(summaries: list[dict], output_dir: str, fold: int):
    if len(summaries) < 2:
        return
    for metric_name, ylabel in [
        ("dice", "Dice"),
        ("hd95", "HD95 (vox)"),
        ("asd", "ASD (vox)"),
    ]:
        fig, ax = plt.subplots(figsize=(9, 5.5))
        for s in summaries:
            for c, cname in zip(s["fg_classes"], s["class_names"]):
                means, lo, hi = [], [], []
                for eps in s["eps_list"]:
                    mean, _, _, ci95, _ = _agg(s["results"][eps][c][metric_name])
                    means.append(mean)
                    lo.append(mean - (ci95 if np.isfinite(ci95) else 0.0))
                    hi.append(mean + (ci95 if np.isfinite(ci95) else 0.0))
                label = f"{s['model_display']} — {cname}"
                (line,) = ax.plot(s["eps_list"], means, marker="o", label=label)
                ax.fill_between(
                    s["eps_list"], lo, hi, alpha=0.15, color=line.get_color()
                )
        ax.set_xlabel("PGD Epsilon (L∞)")
        ax.set_ylabel(ylabel)
        ax.set_title(f"PGD robustness (combined, fold {fold}, 95% CI)")
        ax.legend(fontsize=8)
        ax.grid(True, alpha=0.3)
        plt.tight_layout()
        path = os.path.join(
            output_dir, f"combined_{ATTACK_STEM}_fold{fold}_{metric_name}.png"
        )
        plt.savefig(path, dpi=300, bbox_inches="tight")
        plt.close(fig)
        print(f"Saved combined plot: {path}")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def _apply_checkpoint_override(cfg: dict, checkpoint: str | None) -> dict:
    if checkpoint is None:
        return cfg
    if checkpoint == "":
        raise ValueError("--checkpoint was provided as an empty string")
    out = {**cfg, "checkpoint": os.path.abspath(checkpoint)}
    if not os.path.isfile(out["checkpoint"]):
        raise FileNotFoundError(f"Checkpoint not found: {out['checkpoint']}")
    return out


def main():
    parser = argparse.ArgumentParser(
        description="PGD (L∞) adversarial evaluation for nnU-Net models"
    )
    parser.add_argument(
        "--attack",
        choices=[
            "pgd", "a_pgd", "auto_pgd", "auto_pgd_plus", "apgd_updated",
            "segpgd", "cospgd", "dag", "sea", "alma_prox", "blade", "blade1",
            "blade_mm", "blade_boundary", "blade_frontier",
        ],
        default="pgd",
        help="pgd = fixed-step PGD; a_pgd = the manuscript's adaptive-step PGD "
        "(Table I's APGD); auto_pgd = the original Auto-PGD of Croce and Hein; "
        "apgd_updated = the improved-step arm, a cosine step backbone from "
        "2*eps down to eps/k with a bidirectional multiplier and "
        "budget-invariant checkpoints; auto_pgd_plus = a REFUTED attempt at "
        "repairing Auto-PGD for short budgets, kept only as a negative result "
        "(see its module docstring) -- it is weaker than auto_pgd, do not use "
        "it; segpgd = SegPGD (Gu et al., ECCV 2022), PGD whose per-voxel CE is "
        "split into correct/wrong voxels with a (1-lambda)/lambda schedule; "
        "cospgd = CosPGD (Agnihotri et al., ICML 2024), PGD whose per-voxel CE "
        "is scaled by the cosine similarity to the label. Both use the pgd "
        "update and step so only the objective differs; dag = Dense Adversary "
        "Generation (Xie et al., ICCV 2017), L-inf-normalised ascent on the "
        "logit gap of still-correct voxels, no random start; sea = Segmentation "
        "Ensemble Attack (Croce, Singh, Hein 2023), Auto-PGD per loss in "
        "{masked CE, balanced masked CE, Jensen-Shannon} keeping the worst "
        "Dice, three times the cost of a single-loss arm; alma_prox = ALMA "
        "prox (Rony, Pesquet, Ben Ayed, CVPR 2023), the minimum-norm arm: it "
        "minimises the L-inf norm subject to the image being misclassified "
        "rather than maximising damage inside a fixed ball, so it ignores the "
        "epsilon grid while optimising and each reported epsilon keeps the "
        "worst iterate that fits inside it (--pgd-steps 500 in the paper); "
        "blade = BLADE-3, the "
        "ascending-epsilon ladder run once per objective in {dice_ce, "
        "sign-flipped boundary loss, frontier margin} keeping the worst Dice "
        "per rung, budget-matched against sea; blade1 = BLADE-1, the same "
        "ladder on dice_ce alone, budget-matched against the single-objective "
        "arms; blade_mm = BLADE-MM, the same three-objective ladder as blade "
        "but adjudicated by ranking the candidates on Dice, HD95 and ASD "
        "together instead of on Dice alone; blade_boundary and blade_frontier "
        "= the ladder on one objective each, the per-objective marginals of "
        "blade. Under --shared-trajectory the ladder optimises natively at "
        "every requested epsilon instead of projecting down from the largest. "
        "Selecting anything other than pgd also "
        "switches the defaults for momentum and random start to that attack's "
        "configuration.",
    )
    parser.add_argument("--model", choices=["wg", "zones", "both"], default="both")
    parser.add_argument("--checkpoint", type=str, default=None)
    parser.add_argument("--fold", type=int, default=0)
    parser.add_argument("--max-samples", type=int, default=None)
    parser.add_argument("--output-dir", type=str, default="pgd_results")
    parser.add_argument("--epsilons-wg", type=float, nargs="+", default=None)
    parser.add_argument("--epsilons-zones", type=float, nargs="+", default=None)

    parser.add_argument("--pgd-steps", type=int, default=10)
    parser.add_argument(
        "--step-size",
        type=float,
        default=None,
        help="Step size α (default: max_eps / pgd-steps)",
    )
    parser.add_argument("--random-start", action="store_true")
    parser.add_argument("--seed", type=int, default=None)

    # Stronger attacks
    parser.add_argument(
        "--momentum",
        type=float,
        default=0.0,
        help="MI-PGD momentum decay μ (e.g. 1.0). 0 disables.",
    )
    parser.add_argument(
        "--normalized-grad",
        action="store_true",
        help="Use L2-normalized gradient direction (ignored if --momentum>0)",
    )
    parser.add_argument("--num-restarts", type=int, default=1)

    # Efficiency
    parser.add_argument(
        "--shared-trajectory",
        action="store_true",
        help="Run PGD once at max(eps) and snapshot smaller eps along the way",
    )
    parser.add_argument(
        "--amp", action="store_true", help="Mixed-precision forward at evaluation time"
    )

    # Reporting / IO
    parser.add_argument(
        "--save-adv-dir",
        type=str,
        default=None,
        help="If set, save adversarial volumes (.npz) to this dir",
    )
    parser.add_argument(
        "--log-loss-curve",
        action="store_true",
        help="Log per-step attack loss to a CSV",
    )
    parser.add_argument(
        "--asr-threshold",
        type=float,
        default=0.1,
        help="Dice drop threshold for attack success rate (default 0.1)",
    )
    parser.add_argument(
        "--resume", action="store_true", help="Skip cases already in the per-sample CSV"
    )
    parser.add_argument(
        "--deterministic",
        action="store_true",
        help="Enable deterministic algorithms (slower, reproducible)",
    )
    parser.add_argument(
        "--benchmark",
        action="store_true",
        help="Enable cuDNN autotuning. The attack repeats one shape for every "
        "step of a case, so the tuning cost is amortized. Ignored under "
        "--deterministic.",
    )
    parser.add_argument(
        "--attack-valid-only",
        action="store_true",
        help="Confine the perturbation to voxels with a valid label (label != "
        "-1) inside the unpadded volume. The zones inputs are zero-filled "
        "outside the dilated whole gland and zero-padded to the model grid, "
        "and the deployed pipeline regenerates both after any image-level "
        "attack, so a perturbation there cannot be realised. Default off "
        "reproduces the published (unconfined) numbers.",
    )

    args = parser.parse_args()

    global ATTACK_STEM
    ATTACK_STEM = args.attack
    if args.attack == "a_pgd":
        # configs/attacks/a_pgd.json: momentum 0.9 and a random start, unless the
        # caller asked for something else explicitly.
        if "--momentum" not in sys.argv:
            args.momentum = 0.9
        if "--random-start" not in sys.argv:
            args.random_start = True
    elif args.attack == "auto_pgd":
        # configs/attacks/auto_pgd.json: Auto-PGD always starts on the eps-sphere
        # and has no momentum or step-size knob of its own.
        if "--random-start" not in sys.argv:
            args.random_start = True
    elif args.attack == "auto_pgd_plus":
        # configs/attacks/auto_pgd_plus.json: like auto_pgd, it starts on the
        # eps-sphere and has no momentum or step-size knob of its own.
        if "--random-start" not in sys.argv:
            args.random_start = True
    elif args.attack == "apgd_updated":
        # configs/attacks/apgd_updated.json: same momentum and random start as
        # the campaign APGD, since only the step schedule differs between them.
        if "--momentum" not in sys.argv:
            args.momentum = 0.9
        if "--random-start" not in sys.argv:
            args.random_start = True
    elif args.attack in ("segpgd", "cospgd"):
        # configs/attacks/{segpgd,cospgd}.json: the pgd arm's random start and
        # no momentum; they are loss variants of pgd and share its update.
        if "--random-start" not in sys.argv:
            args.random_start = True
    elif args.attack in (
        "sea", "blade", "blade1", "blade_mm", "blade_boundary", "blade_frontier"
    ):
        # configs/attacks/{sea,blade,blade1,blade_mm}.json: Auto-PGD under the
        # hood, so the eps-sphere start of auto_pgd -- and for a ladder only its first
        # rung is randomised, the rest inherit the rung below. dag keeps the
        # parser default (clean start), as configs/attacks/dag.json and the
        # paper specify.
        if "--random-start" not in sys.argv:
            args.random_start = True

    if args.deterministic:
        os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
        torch.use_deterministic_algorithms(True, warn_only=True)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
    elif args.benchmark:
        torch.backends.cudnn.benchmark = True

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")
    print(f"nnUnet_paths root: {NNUNET_PATHS}")

    # Persist run config alongside results for traceability.
    os.makedirs(args.output_dir, exist_ok=True)
    run_config = dict(vars(args))
    # ``shared_trajectory`` records the flag as passed; ``protocol`` records what
    # that flag meant for this attack. They diverge for ladder attacks.
    run_config["protocol"] = _protocol_label(args.attack, args.shared_trajectory)
    run_config["blade_impl"] = (
        "mc" if args.model == "zones" and args.attack in BLADE_MC_ARMS else "legacy"
    ) if args.attack.startswith("blade") else None
    run_config["blade_mc_spacing_mm"] = (
        _plan_spacing(_make_eval_config("zones", args))
        if run_config["blade_impl"] == "mc" else None
    )
    run_config["protocol_note"] = PROTOCOL_NOTES[run_config["protocol"]]
    with open(
        os.path.join(args.output_dir, f"{ATTACK_STEM}_run_config_fold{args.fold}.json"),
        "w",
    ) as f:
        json.dump(run_config, f, indent=2, default=str)

    models_to_eval = ["wg", "zones"] if args.model == "both" else [args.model]
    summaries = []
    for model_key in models_to_eval:
        cfg = _make_eval_config(model_key, args)
        cfg = _apply_checkpoint_override(cfg, args.checkpoint)
        s = evaluate_pgd(
            model_key=model_key,
            device=device,
            max_samples=args.max_samples,
            output_dir=args.output_dir,
            cfg=cfg,
            n_steps=args.pgd_steps,
            step_size=args.step_size,
            random_start=args.random_start,
            fold=args.fold,
            seed=args.seed,
            momentum=args.momentum,
            normalized_grad=args.normalized_grad,
            num_restarts=args.num_restarts,
            shared_trajectory=args.shared_trajectory,
            save_adv_dir=args.save_adv_dir,
            log_loss_curve=args.log_loss_curve,
            amp=args.amp,
            resume=args.resume,
            attack=args.attack,
            asr_threshold=args.asr_threshold,
            attack_valid_only=args.attack_valid_only,
        )
        summaries.append(s)

    _combined_plots(summaries, args.output_dir, args.fold)
    print("\nDone.")


if __name__ == "__main__":
    main()
