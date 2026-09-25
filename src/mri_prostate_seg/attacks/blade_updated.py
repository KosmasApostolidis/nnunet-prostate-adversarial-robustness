"""BLADE-MC: multiclass, L-infinity, white-box attack for 3D segmentation.

Default independent objectives: multiclass Dice+CE, normalized class-specific
signed-distance loss, and class-balanced hard frontier margins. Each objective
has its own ascending-epsilon APGD warm-start chain. Every evaluated iterate is
eligible for foreground-macro-Dice selection, independently of APGD's surrogate
best-loss state. Clean and earlier feasible candidates remain eligible.

Background is label 0; mutually exclusive foreground classes are 1..C-1.
No anatomical zone IDs are hard-coded. Batched inputs are attacked sequentially
per volume, with independent optimizer state, distance maps, and selection.
See README.md for geometry assumptions, exact loss definitions, and limitations.
"""
from __future__ import annotations

from collections.abc import Callable, Iterator, Sequence
from contextlib import contextmanager
from dataclasses import dataclass
import math
from numbers import Integral
import warnings

import numpy as np
import torch
import torch.nn.functional as F
from scipy.ndimage import distance_transform_edt

from mri_prostate_seg.attacks.auto_pgd import auto_pgd_attack
from mri_prostate_seg.attacks.pgd import _samplewise_bounds

IGNORE_LABEL = -1
BLADE_LOSSES = ("dice_ce", "boundary", "frontier")
SUPPORTED_LOSSES = BLADE_LOSSES + ("frontier_soft",)
DEFAULT_K_FRAC = 0.1
LossFn = Callable[[torch.Tensor, torch.Tensor, int], torch.Tensor]
Spacing = Sequence[float] | np.ndarray | torch.Tensor | None


def _labels(target: torch.Tensor, num_classes: int, ignore_label: int) -> torch.Tensor:
    if not isinstance(num_classes, Integral) or isinstance(num_classes, bool) or num_classes < 2:
        raise ValueError("num_classes must be an integer >= 2 (including background)")
    if 0 <= ignore_label < num_classes:
        raise ValueError("ignore_label must lie outside the class ID range")
    if target.ndim == 5 and target.shape[1] == 1:
        labels = target[:, 0]
    elif target.ndim == 4:
        labels = target
    else:
        raise ValueError("target must have shape [B,1,D,H,W] or [B,D,H,W]")
    if any(s == 0 for s in labels.shape):
        raise ValueError("Empty batch/spatial dimensions are not supported")
    if labels.is_floating_point() or labels.is_complex() or labels.dtype == torch.bool:
        raise TypeError("target must contain integer class IDs, not probabilities")
    valid = labels != ignore_label
    if bool(((labels[valid] < 0) | (labels[valid] >= num_classes)).any()):
        raise ValueError("A non-ignored target label is outside [0, num_classes)")
    return labels.long()


def _check_logits(logits: torch.Tensor, labels: torch.Tensor, num_classes: int) -> None:
    expected = (labels.shape[0], num_classes, *labels.shape[1:])
    if not isinstance(logits, torch.Tensor) or tuple(logits.shape) != expected:
        raise ValueError(f"Model must return a logits tensor of shape {expected}; wrap other outputs")
    if not logits.is_floating_point() or logits.device != labels.device:
        raise ValueError("Logits must be floating point and on the target's device")
    if not bool(torch.isfinite(logits).all()):
        raise ValueError("Nonfinite model logits encountered")


def _working_logits(logits: torch.Tensor) -> torch.Tensor:
    # Avoid half-precision sum overflow on large 3D volumes; preserve float64 tests.
    return logits.float() if logits.dtype in (torch.float16, torch.bfloat16) else logits


def _spacing_array(spacing: Spacing, batch_size: int) -> np.ndarray:
    if spacing is None:
        return np.ones((batch_size, 3), dtype=np.float64)
    if isinstance(spacing, torch.Tensor):
        spacing = spacing.detach().cpu().numpy()
    result = np.asarray(spacing, dtype=np.float64)
    if result.shape == (3,):
        result = np.broadcast_to(result, (batch_size, 3)).copy()
    if result.shape != (batch_size, 3) or not np.isfinite(result).all() or (result <= 0).any():
        raise ValueError("spacing must contain positive finite (D,H,W) values, shape [3] or [B,3]")
    return result


def _weights(values: Sequence[float] | torch.Tensor | None, classes: int,
             device: torch.device, dtype: torch.dtype) -> torch.Tensor:
    result = torch.ones(classes, device=device, dtype=dtype) if values is None else torch.as_tensor(
        values, device=device, dtype=dtype
    ).detach()
    if result.shape != (classes,) or not bool(torch.isfinite(result).all()) or bool((result < 0).any()):
        raise ValueError("Class weights must be a length-C vector of finite nonnegative values")
    return result


def signed_distance_maps(
    target: torch.Tensor,
    num_classes: int,
    *,
    spacing: Spacing = None,
    ignore_label: int = IGNORE_LABEL,
) -> torch.Tensor:
    """Return [B,C,D,H,W] float32 maps; channel 0 is zero and not optimized.

    Each foreground channel is EDT(not class) - EDT(class), in supplied spacing
    units. Classes absent from, or filling, the valid domain receive a zero map.
    This is the original code's voxel-center signed-distance convention, not an
    exact subvoxel surface-distance implementation and not HD95 or ASD.

    Ignored labels are supported only as padding outside a rectangular fully
    labeled domain. Distances are computed inside that domain, never through
    ignored padding. Internal holes/nonrectangular unknown regions raise instead
    of being silently treated as background. Cropped patches should use maps
    computed on the full labeled volume and then cropped, passed via ``phi``.
    """
    labels = _labels(target, num_classes, ignore_label)
    spacings = _spacing_array(spacing, labels.shape[0])
    lab_np = labels.detach().cpu().numpy()
    maps = np.zeros((labels.shape[0], num_classes, *labels.shape[1:]), dtype=np.float32)
    for b, lab in enumerate(lab_np):
        valid = lab != ignore_label
        if not valid.any():
            continue
        coordinates = np.nonzero(valid)
        box = tuple(slice(int(a.min()), int(a.max()) + 1) for a in coordinates)
        if not valid[box].all():
            raise ValueError(
                "Boundary geometry cannot infer surfaces across ignored holes/nonrectangular "
                "unknown regions. Supply trusted full-volume distance maps via phi, or omit "
                "the boundary branch. Rectangular ignored padding is supported."
            )
        crop = lab[box]
        for c in range(1, num_classes):
            mask = crop == c
            if not mask.any() or mask.all():
                continue
            outside = distance_transform_edt(~mask, sampling=spacings[b])
            inside = distance_transform_edt(mask, sampling=spacings[b])
            maps[(b, c, *box)] = (outside - inside).astype(np.float32)
    return torch.from_numpy(maps).to(device=target.device)


def signed_distance_map(target: torch.Tensor) -> torch.Tensor:
    """Backward-compatible binary helper; use signed_distance_maps for C > 2."""
    warnings.warn("Use signed_distance_maps(target, num_classes, spacing=...) instead",
                  DeprecationWarning, stacklevel=2)
    # Do not silently union multiclass labels, the original source of zone blindness.
    return signed_distance_maps(target, 2)[:, 1:2]


def boundary_weighted_loss(
    logits: torch.Tensor,
    target: torch.Tensor,
    num_classes: int,
    phi: torch.Tensor | None = None,
    *,
    spacing: Spacing = None,
    ignore_label: int = IGNORE_LABEL,
    class_weights: Sequence[float] | torch.Tensor | None = None,
) -> torch.Tensor:
    """Maximize mean_c sum(phi_c*p_c)/sum(abs(phi_c)), per volume.

    Class weights have length C; the background entry is ignored. Foreground
    classes with zero maps or absent ground-truth support are omitted and weights
    are renormalized over the eligible classes of each volume. All-ineligible
    cases contribute a differentiable zero. Maps are fixed label geometry and
    are detached. Externally supplied maps must match class channel IDs exactly.
    """
    labels = _labels(target, num_classes, ignore_label)
    _check_logits(logits, labels, num_classes)
    z = _working_logits(logits)
    if phi is None:
        phi = signed_distance_maps(target, num_classes, spacing=spacing, ignore_label=ignore_label)
    # Preserve direct binary calls with the legacy [B,1,D,H,W] map.
    if num_classes == 2 and phi.shape == (labels.shape[0], 1, *labels.shape[1:]):
        phi = torch.cat((torch.zeros_like(phi), phi), dim=1)
    if tuple(phi.shape) != tuple(logits.shape) or phi.device != logits.device:
        raise ValueError("phi must match logits [B,C,D,H,W], with map c in channel c")
    if not bool(torch.isfinite(phi).all()):
        raise ValueError("phi contains nonfinite distances")
    valid = labels != ignore_label
    distances = torch.where(valid[:, None], phi.detach().to(z.dtype), 0.0)
    probabilities = F.softmax(z, dim=1)
    weights = _weights(class_weights, num_classes, z.device, z.dtype)
    if not bool((weights[1:] > 0).any()):
        raise ValueError("At least one foreground boundary weight must be positive")
    result = []
    for b in range(labels.shape[0]):
        numerators, denominators = [], []
        for c in range(1, num_classes):
            scale = distances[b, c].abs().sum()
            if not bool(((labels[b] == c) & valid[b]).any()) or not bool(scale > 0) or not bool(weights[c] > 0):
                continue
            numerators.append(weights[c] * (distances[b, c] * probabilities[b, c]).sum() / scale)
            denominators.append(weights[c])
        result.append(torch.stack(numerators).sum() / torch.stack(denominators).sum()
                      if numerators else z[b].sum() * 0.0)
    return torch.stack(result).mean()


def multiclass_dice_ce_loss(
    logits: torch.Tensor,
    target: torch.Tensor,
    num_classes: int,
    *,
    ignore_label: int = IGNORE_LABEL,
    dice_weight: float = 1.0,
    ce_weight: float = 1.0,
    smooth: float = 1e-5,
) -> torch.Tensor:
    """Foreground macro soft-Dice loss + valid-voxel multiclass CE.

    Soft Dice is (2*intersection+smooth)/(prediction_mass+target_mass+smooth).
    Every foreground channel is included, also an absent class with false-positive
    probability mass. CE includes background, is voxel-averaged per volume, and
    is not class-balanced. The total is averaged per volume. All-ignored samples
    have a differentiable zero. No one-hot B*C*D*H*W label allocation is required.
    """
    for name, value in (("dice_weight", dice_weight), ("ce_weight", ce_weight), ("smooth", smooth)):
        if not math.isfinite(float(value)) or value < 0:
            raise ValueError(f"{name} must be finite and nonnegative")
    if smooth <= 0 or dice_weight + ce_weight <= 0:
        raise ValueError("smooth and the total objective weight must be positive")
    labels = _labels(target, num_classes, ignore_label)
    _check_logits(logits, labels, num_classes)
    z = _working_logits(logits)
    valid = labels != ignore_label
    safe = torch.where(valid, labels, 0)
    probabilities = F.softmax(z, dim=1)
    ce = F.cross_entropy(z, safe, reduction="none")
    result = []
    for b in range(labels.shape[0]):
        if not bool(valid[b].any()):
            result.append(z[b].sum() * 0.0)
            continue
        terms = []
        for c in range(1, num_classes):
            prediction = probabilities[b, c][valid[b]]
            truth = (labels[b][valid[b]] == c).to(z.dtype)
            dice = (2.0 * (prediction * truth).sum() + smooth) / (prediction.sum() + truth.sum() + smooth)
            terms.append(1.0 - dice)
        result.append(dice_weight * torch.stack(terms).mean() + ce_weight * ce[b][valid[b]].mean())
    return torch.stack(result).mean()


def _margins(logits: torch.Tensor, labels: torch.Tensor, ignore_label: int) -> torch.Tensor:
    safe = torch.where(labels != ignore_label, labels, 0)
    truth = logits.gather(1, safe[:, None]).squeeze(1)
    rivals = logits.scatter(1, safe[:, None], float("-inf")).amax(dim=1)
    return truth - rivals


def frontier_margin_loss(
    logits: torch.Tensor,
    target: torch.Tensor,
    num_classes: int,
    k_frac: float = DEFAULT_K_FRAC,
    *,
    ignore_label: int = IGNORE_LABEL,
    class_weights: Sequence[float] | torch.Tensor | None = None,
) -> torch.Tensor:
    """Negative mean of smallest still-correct margins, selected CLASSWISE.

    All ground-truth-present classes (including background) enter a FIXED class
    denominator. A class with no still-correct voxels contributes zero without
    disappearing from that denominator. k=max(1,floor(k_frac*n_correct_class)).
    This loss still has active-set discontinuities: metric-best retention is
    mandatory, and does not turn the surrogate into a continuous objective.
    """
    if not math.isfinite(float(k_frac)) or not 0.0 < k_frac <= 1.0:
        raise ValueError("k_frac must be in (0,1]")
    labels = _labels(target, num_classes, ignore_label)
    _check_logits(logits, labels, num_classes)
    z = _working_logits(logits)
    valid = labels != ignore_label
    margin = _margins(z, labels, ignore_label)
    correct = (z.detach().argmax(dim=1) == labels) & valid
    weights = _weights(class_weights, num_classes, z.device, z.dtype)
    return _classwise_frontier(z, labels, valid, margin, weights, correct, k_frac, None)


def soft_frontier_margin_loss(
    logits: torch.Tensor,
    target: torch.Tensor,
    num_classes: int,
    *,
    temperature: float = 1.0,
    ignore_label: int = IGNORE_LABEL,
    class_weights: Sequence[float] | torch.Tensor | None = None,
) -> torch.Tensor:
    """Optional ablation: class-mean sigmoid(-margin/temperature), no hard mask.

    Unlike ``frontier``, previously flipped voxels remain in this continuous
    (piecewise differentiable) surrogate. This is a different attack objective,
    not an equivalent implementation or an empirically established improvement.
    """
    if not math.isfinite(float(temperature)) or temperature <= 0:
        raise ValueError("temperature must be finite and positive")
    labels = _labels(target, num_classes, ignore_label)
    _check_logits(logits, labels, num_classes)
    z = _working_logits(logits)
    valid = labels != ignore_label
    weights = _weights(class_weights, num_classes, z.device, z.dtype)
    return _classwise_frontier(z, labels, valid, _margins(z, labels, ignore_label),
                              weights, None, None, temperature)


def _classwise_frontier(z, labels, valid, margin, weights, correct, k_frac, temperature):
    result = []
    for b in range(labels.shape[0]):
        terms, active_weights = [], []
        for c in range(z.shape[1]):
            support = (labels[b] == c) & valid[b]
            if not bool(support.any()) or not bool(weights[c] > 0):
                continue
            if temperature is not None:
                term = torch.sigmoid(-margin[b][support] / temperature).mean()
            else:
                values = margin[b][support & correct[b]]
                k = max(1, int(k_frac * values.numel()))
                term = -values.topk(k, largest=False).values.mean() if values.numel() else z[b].sum() * 0.0
            terms.append(weights[c] * term)
            active_weights.append(weights[c])
        if not terms and bool(valid[b].any()):
            raise ValueError("At least one ground-truth-present frontier class must have positive weight")
        result.append(torch.stack(terms).sum() / torch.stack(active_weights).sum()
                      if terms else z[b].sum() * 0.0)
    return torch.stack(result).mean()


@torch.no_grad()
def foreground_dice_per_sample(
    logits: torch.Tensor, target: torch.Tensor, num_classes: int, *,
    ignore_label: int = IGNORE_LABEL,
) -> torch.Tensor:
    """[B] hard foreground macro Dice. Empty/empty=1; empty GT plus FP=0.

    No valid labels -> NaN, never a fabricated perfect score. BLADE entrypoints
    reject all-ignored samples. Integer counts and float64 division avoid half
    overflow and keep candidate comparisons reliable on large volumes.
    """
    labels = _labels(target, num_classes, ignore_label)
    _check_logits(logits, labels, num_classes)
    valid = labels != ignore_label
    prediction = logits.argmax(dim=1)
    dims = tuple(range(1, labels.ndim))
    scores = []
    for c in range(1, num_classes):
        p, t = (prediction == c) & valid, (labels == c) & valid
        denominator = p.sum(dim=dims) + t.sum(dim=dims)
        intersection = (p & t).sum(dim=dims)
        score = 2.0 * intersection.double() / denominator.clamp_min(1).double()
        scores.append(torch.where(denominator > 0, score, 1.0))
    result = torch.stack(scores, dim=1).mean(dim=1)
    return torch.where(valid.flatten(1).any(dim=1), result, torch.nan)


@contextmanager
def _evaluation_mode(model: torch.nn.Module) -> Iterator[None]:
    # Restore mixed per-submodule states, not just the root training flag.
    states = [(module, module.training) for module in model.modules()]
    try:
        model.eval()
        yield
    finally:
        for module, state in states:
            module.training = state


@dataclass
class _Candidate:
    image: torch.Tensor
    dice: float
    source: str
    origin_eps: float
    step: int


def _validate_inputs(model, x, y, num_classes, n_steps, max_eps, snapshot_eps, ignore_label):
    if torch.is_inference_mode_enabled():
        raise RuntimeError("BLADE requires autograd; run outside torch.inference_mode(). no_grad is supported.")
    if not isinstance(model, torch.nn.Module):
        raise TypeError("model must be a torch.nn.Module returning [B,C,D,H,W] logits")
    if x.ndim != 5 or any(s == 0 for s in x.shape) or not x.is_floating_point():
        raise ValueError("x must be a nonempty floating [B,input_channels,D,H,W] tensor")
    if x.dtype not in (torch.float32, torch.float64):
        raise TypeError("Attack x must be float32 or float64; do not quantize the perturbation in half precision")
    if not bool(torch.isfinite(x).all()):
        raise ValueError("x contains nonfinite intensities")
    labels = _labels(y, num_classes, ignore_label)
    if labels.device != x.device or labels.shape != (x.shape[0], *x.shape[2:]):
        raise ValueError("Image and target batch/spatial shapes and devices must agree")
    if not bool((labels != ignore_label).flatten(1).any(dim=1).all()):
        raise ValueError("Every attacked volume must contain at least one valid ground-truth voxel")
    if not isinstance(n_steps, Integral) or isinstance(n_steps, bool) or n_steps < 1:
        raise ValueError("n_steps must be an integer >= 1")
    if not math.isfinite(float(max_eps)) or max_eps < 0:
        raise ValueError("max_eps must be finite and nonnegative")
    budgets = set(float(e) for e in (snapshot_eps or [])) | {float(max_eps)}
    if any(not math.isfinite(e) or e < 0 or e > max_eps for e in budgets):
        raise ValueError("Every snapshot epsilon must be finite and within [0,max_eps]")
    return labels[:, None], sorted(budgets)


def _validated_bounds(x, x_bounds):
    if x_bounds is None:
        return _samplewise_bounds(x)
    if len(x_bounds) != 2:
        raise ValueError("x_bounds must be (lower,upper)")
    result = []
    for value in x_bounds:
        value = torch.as_tensor(value, device=x.device, dtype=x.dtype).detach()
        if value.ndim == 1 and value.shape[0] == x.shape[0]:
            value = value.view(-1, 1, 1, 1, 1)
        try:
            result.append(torch.broadcast_to(value, x.shape))
        except RuntimeError as exc:
            raise ValueError("x_bounds must broadcast to x (or be length-B vectors)") from exc
    low, high = result
    if not bool(torch.isfinite(low).all() & torch.isfinite(high).all()):
        raise ValueError("Intensity bounds must be finite")
    if bool(((low > high) | (x < low) | (x > high)).any()):
        raise ValueError("Intensity bounds must be ordered and contain the original clean image")
    return low, high


def _one_ladder(model, x, y, classes, loss_fn, n_steps, rungs, bounds, random_start,
                rho, ignore_label, clean, source, emit, loss_log=None,
                init_log=None, run_log=None, sample_index=0):
    """Stream each branch champion without storing every branch/rung volume."""
    champion = clean
    warm = None
    for rung in rungs:
        if rung == 0.0:
            emit(rung, clean)
            continue
        if init_log is not None:
            init_log.append(None if warm is None else warm.detach().clone())
        evaluations = 0

        def observe(image, logits, step, surrogate_value):
            nonlocal champion, evaluations
            evaluations += 1
            dice = float(foreground_dice_per_sample(logits, y, classes, ignore_label=ignore_label)[0])
            if not math.isfinite(dice):
                raise RuntimeError("Nonfinite candidate selection metric")
            if dice < champion.dice:  # strict comparison retains clean/earlier candidate on ties
                champion = _Candidate(image.detach().clone(), dice, source, rung, step)

        warm = auto_pgd_attack(
            model, x, y, rung, classes, loss_fn, n_steps=n_steps,
            random_start=random_start and warm is None, x_bounds=bounds, rho=rho,
            loss_log=loss_log, x_init=warm, iterate_callback=observe,
        )
        # warm is APGD's SURROGATE-best, not the metric-best. No state substitution.
        emit(rung, champion)
        if run_log is not None:
            run_log.append({
                "sample_index": sample_index, "objective": source, "epsilon": rung,
                "gradient_evaluations": evaluations, "model_forwards": evaluations,
                "n_steps": n_steps, "branch_best_dice": champion.dice,
                "candidate_origin_epsilon": champion.origin_eps,
                "candidate_origin_step": champion.step, "candidate_source": champion.source,
            })


def ladder_trajectory(
    model: torch.nn.Module, x: torch.Tensor, y: torch.Tensor, max_eps: float,
    num_classes: int, loss_fn: LossFn, n_steps: int,
    random_start: bool = True,
    x_bounds: tuple[torch.Tensor, torch.Tensor] | None = None,
    snapshot_eps: list[float] | None = None,
    loss_log: list[float] | None = None,
    rho: float = 0.75,
    init_log: list[torch.Tensor | None] | None = None,
    *, ignore_label: int = IGNORE_LABEL,
) -> tuple[torch.Tensor, dict[float, torch.Tensor]]:
    """Single-volume ladder returning METRIC-best snapshots, with clean retained.

    Previous positional arguments are retained. Multi-volume callers should use
    blade_trajectory, which loops independently over volumes. APGD optimizer-best
    images still warm-start the next rung; snapshots can differ from those images.
    """
    y, rungs = _validate_inputs(model, x, y, num_classes, n_steps, max_eps, snapshot_eps, ignore_label)
    if x.shape[0] != 1:
        raise ValueError("ladder_trajectory accepts B=1; use blade_trajectory for independent batched attacks")
    if not math.isfinite(float(rho)) or not 0 < rho <= 1:
        raise ValueError("rho must be in (0,1]")
    bounds = _validated_bounds(x, x_bounds)
    x = x.detach()
    snapshots = {}
    if max_eps == 0:
        return x.clone(), {0.0: x.clone()}
    with _evaluation_mode(model):
        with torch.no_grad():
            logits = model(x)
            clean = _Candidate(x.clone(), float(foreground_dice_per_sample(
                logits, y, num_classes, ignore_label=ignore_label)[0]), "clean", 0.0, -1)
        _one_ladder(model, x, y, num_classes, loss_fn, n_steps, rungs, bounds,
                    random_start, rho, ignore_label, clean, "custom",
                    lambda rung, candidate: snapshots.__setitem__(rung, candidate.image.clone()),
                    loss_log=loss_log, init_log=init_log)
    return snapshots[float(max_eps)], snapshots


def blade_trajectory(
    model: torch.nn.Module,
    x: torch.Tensor,
    y: torch.Tensor,
    max_eps: float,
    num_classes: int,
    base_loss_fn: LossFn | None = None,
    n_steps: int = 20,
    random_start: bool = True,
    x_bounds: tuple[torch.Tensor, torch.Tensor] | None = None,
    snapshot_eps: list[float] | None = None,
    loss_log: list[float] | None = None,
    losses: tuple[str, ...] = BLADE_LOSSES,
    k_frac: float = DEFAULT_K_FRAC,
    chosen_log: dict | None = None,
    *,
    spacing: Spacing = None,
    ignore_label: int = IGNORE_LABEL,
    boundary_class_weights: Sequence[float] | torch.Tensor | None = None,
    frontier_class_weights: Sequence[float] | torch.Tensor | None = None,
    frontier_temperature: float = 1.0,
    rho: float = 0.75,
    phi: torch.Tensor | None = None,
    metric_log: dict | None = None,
    run_log: list[dict] | None = None,
) -> tuple[torch.Tensor, dict[float, torch.Tensor]]:
    """BLADE-MC; returns (adversarial_at_max_eps, {eps: batch_of_candidates}).

    Parameters added to the old interface are keyword-only. ``base_loss_fn=None``
    uses this module's explicit multiclass Dice+CE. A custom callback remains the
    caller's responsibility: it must be a multiclass, ignore-aware scalar loss.
    ``spacing`` is (D,H,W) or [B,3] on the current tensor grid; None means unit voxels.
    ``phi`` optionally supplies trusted [B,C,D,H,W] full-volume-derived maps.
    Class weights have length C (background weight ignored for boundary).
    ``frontier_soft`` is available only when explicitly included in ``losses``.

    The model is temporarily evaluated, with all submodule training flags restored.
    Calls inside torch.no_grad work; torch.inference_mode is rejected. Use a
    deterministic differentiable inference wrapper returning full-resolution logits.

    Logs: chosen_log[eps] and metric_log[eps] are scalar for B=1, lists for B>1.
    The winner can be 'clean' or originate at an earlier epsilon. run_log appends
    one clean-score record plus M*R APGD records per volume (no extra gradient).
    loss_log records only the first requested branch, concatenated by case/rung.
    """
    y, rungs = _validate_inputs(model, x, y, num_classes, n_steps, max_eps, snapshot_eps, ignore_label)
    if not losses or len(set(losses)) != len(losses) or any(name not in SUPPORTED_LOSSES for name in losses):
        raise ValueError(f"losses must be a nonempty duplicate-free subset of {SUPPORTED_LOSSES}")
    if base_loss_fn is not None and not callable(base_loss_fn):
        raise TypeError("base_loss_fn must be callable or None")
    if not math.isfinite(float(k_frac)) or not 0 < k_frac <= 1:
        raise ValueError("k_frac must be in (0,1]")
    if not math.isfinite(float(frontier_temperature)) or frontier_temperature <= 0:
        raise ValueError("frontier_temperature must be finite and positive")
    if not math.isfinite(float(rho)) or not 0 < rho <= 1:
        raise ValueError("rho must be in (0,1]")
    spacings = _spacing_array(spacing, x.shape[0])
    bounds = _validated_bounds(x, x_bounds)
    x = x.detach()
    if phi is not None:
        if phi.shape != (x.shape[0], num_classes, *x.shape[2:]) or phi.device != x.device:
            raise ValueError("phi must be [B,C,D,H,W], same device as x")
        if not bool(torch.isfinite(phi).all()):
            raise ValueError("phi must be finite")
    if boundary_class_weights is not None:
        bw = _weights(boundary_class_weights, num_classes, x.device, x.dtype)
        if not bool((bw[1:] > 0).any()):
            raise ValueError("At least one foreground boundary class weight must be positive")
    if frontier_class_weights is not None:
        fw = _weights(frontier_class_weights, num_classes, x.device, x.dtype)
        for b in range(x.shape[0]):
            present = y[b][y[b] != ignore_label].unique()
            if not bool((fw[present] > 0).any()):
                raise ValueError("Each volume needs a present class with positive frontier weight")
    if max_eps == 0:
        if chosen_log is not None:
            chosen_log[0.0] = "clean" if x.shape[0] == 1 else ["clean"] * x.shape[0]
        # No forward pass: metric_log remains untouched for this no-op path.
        return x.clone(), {0.0: x.clone()}

    per_rung_images = {rung: [] for rung in rungs}
    per_rung_names = {rung: [] for rung in rungs}
    per_rung_scores = {rung: [] for rung in rungs}
    with _evaluation_mode(model):
        for b in range(x.shape[0]):
            xb, yb = x[b:b+1], y[b:b+1]
            bb = tuple(bound[b:b+1] for bound in bounds)
            maps = None
            if "boundary" in losses:
                maps = (phi[b:b+1].detach() if phi is not None else signed_distance_maps(
                    yb, num_classes, spacing=spacings[b], ignore_label=ignore_label))
            with torch.no_grad():
                logits = model(xb)
                clean_dice = float(foreground_dice_per_sample(logits, yb, num_classes, ignore_label=ignore_label)[0])
            del logits
            clean = _Candidate(xb.clone(), clean_dice, "clean", 0.0, -1)
            if run_log is not None:
                run_log.append({"sample_index": b, "objective": "clean", "epsilon": 0.0,
                                "gradient_evaluations": 0, "model_forwards": 1,
                                "branch_best_dice": clean_dice})
            winners = {rung: clean for rung in rungs}

            def emit(rung, candidate):
                if candidate.dice < winners[rung].dice:
                    winners[rung] = candidate

            objectives = {
                "dice_ce": base_loss_fn if base_loss_fn is not None else (lambda lg, t, nc: multiclass_dice_ce_loss(
                    lg, t, nc, ignore_label=ignore_label)),
                "boundary": lambda lg, t, nc: boundary_weighted_loss(
                    lg, t, nc, phi=maps, ignore_label=ignore_label, class_weights=boundary_class_weights),
                "frontier": lambda lg, t, nc: frontier_margin_loss(
                    lg, t, nc, k_frac=k_frac, ignore_label=ignore_label, class_weights=frontier_class_weights),
                "frontier_soft": lambda lg, t, nc: soft_frontier_margin_loss(
                    lg, t, nc, temperature=frontier_temperature, ignore_label=ignore_label,
                    class_weights=frontier_class_weights),
            }
            for j, name in enumerate(losses):
                _one_ladder(model, xb, yb, num_classes, objectives[name], n_steps, rungs,
                            bb, random_start, rho, ignore_label, clean, name, emit,
                            loss_log=loss_log if j == 0 else None, run_log=run_log, sample_index=b)
            for rung, candidate in winners.items():
                per_rung_images[rung].append(candidate.image)
                per_rung_names[rung].append(candidate.source)
                per_rung_scores[rung].append(candidate.dice)

    snapshots = {rung: torch.cat(images, dim=0).detach() for rung, images in per_rung_images.items()}
    for rung in rungs:
        if chosen_log is not None:
            chosen_log[rung] = per_rung_names[rung][0] if x.shape[0] == 1 else per_rung_names[rung]
        if metric_log is not None:
            metric_log[rung] = per_rung_scores[rung][0] if x.shape[0] == 1 else per_rung_scores[rung]
    return snapshots[float(max_eps)], snapshots


def blade_attack(
    model: torch.nn.Module, x: torch.Tensor, y: torch.Tensor, eps: float,
    num_classes: int, base_loss_fn: LossFn | None = None, n_steps: int = 20,
    x_bounds: tuple[torch.Tensor, torch.Tensor] | None = None,
    losses: tuple[str, ...] = BLADE_LOSSES,
    *,
    random_start: bool = True,
    spacing: Spacing = None,
    ignore_label: int = IGNORE_LABEL,
    k_frac: float = DEFAULT_K_FRAC,
    boundary_class_weights: Sequence[float] | torch.Tensor | None = None,
    frontier_class_weights: Sequence[float] | torch.Tensor | None = None,
    frontier_temperature: float = 1.0,
    rho: float = 0.75,
    phi: torch.Tensor | None = None,
    chosen_log: dict | None = None,
    metric_log: dict | None = None,
    run_log: list[dict] | None = None,
) -> torch.Tensor:
    """Single-epsilon convenience call; there are no implicit intermediate rungs."""
    result, _ = blade_trajectory(
        model, x, y, eps, num_classes, base_loss_fn, n_steps,
        random_start=random_start, x_bounds=x_bounds, losses=losses, k_frac=k_frac,
        spacing=spacing, ignore_label=ignore_label, boundary_class_weights=boundary_class_weights,
        frontier_class_weights=frontier_class_weights, frontier_temperature=frontier_temperature,
        rho=rho, phi=phi, chosen_log=chosen_log, metric_log=metric_log, run_log=run_log,
    )
    return result


__all__ = [
    "BLADE_LOSSES", "SUPPORTED_LOSSES", "blade_attack", "blade_trajectory",
    "ladder_trajectory", "signed_distance_map", "signed_distance_maps",
    "boundary_weighted_loss", "multiclass_dice_ce_loss", "frontier_margin_loss",
    "soft_frontier_margin_loss", "foreground_dice_per_sample",
]
