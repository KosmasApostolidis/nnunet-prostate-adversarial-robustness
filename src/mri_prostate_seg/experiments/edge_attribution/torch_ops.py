"""Model-touching operations: Integrated Gradients, interventions, greedy, pairwise.

Everything here works on the padded model grid.  ``loss_fn`` must be
per-sample (``[B]``) so that ``autograd.grad(loss.sum(), inputs)`` separates
across the batch; the cohort's ``_per_sample_bce`` satisfies this.
"""

from __future__ import annotations

from collections.abc import Callable, Iterator
from dataclasses import dataclass, field
from typing import Protocol

import numpy as np
import torch

LossFn = Callable[[torch.Tensor, torch.Tensor, int], torch.Tensor]


@dataclass(frozen=True)
class IGResult:
    attribution: np.ndarray
    loss_clean: float
    loss_adv: float
    completeness_error: float
    steps: int
    # Path-averaged gradient on the padded grid, [D, H, W].  ``attribution`` is
    # delta * mean_gradient; the spectral partition needs the factor on its own.
    mean_gradient: np.ndarray | None = None


@dataclass(frozen=True)
class InterventionResult:
    dice: np.ndarray
    loss: np.ndarray
    preds: dict[int, np.ndarray] = field(default_factory=dict)


def _check_5d(x: torch.Tensor, name: str) -> None:
    if x.ndim != 5 or x.shape[0] != 1:
        raise ValueError(f"{name} must be [1, C, D, H, W], got {tuple(x.shape)}")


def _loss_value(
    model: torch.nn.Module,
    loss_fn: LossFn,
    x: torch.Tensor,
    y: torch.Tensor,
    num_classes: int,
) -> float:
    with torch.no_grad():
        return float(loss_fn(model(x), y, num_classes).sum().item())


def integrated_attribution(
    model: torch.nn.Module,
    loss_fn: LossFn,
    x: torch.Tensor,
    x_adv: torch.Tensor,
    y: torch.Tensor,
    *,
    num_classes: int,
    steps: int = 32,
    batch: int = 8,
) -> IGResult:
    """Midpoint-rule Integrated Gradients of the attack loss along x -> x_adv."""

    _check_5d(x, "x")
    _check_5d(x_adv, "x_adv")
    if steps < 2:
        raise ValueError("steps must be at least 2")
    x0 = x.detach().float()
    delta = x_adv.detach().float() - x0
    accumulated = torch.zeros_like(x0)
    alphas = (torch.arange(steps, dtype=torch.float32, device=x0.device) + 0.5) / steps
    with torch.enable_grad():
        for start in range(0, steps, batch):
            chunk = alphas[start : start + batch]
            points = (x0 + chunk.view(-1, 1, 1, 1, 1) * delta).requires_grad_(True)
            per_sample = loss_fn(
                model(points), y.expand(points.shape[0], *y.shape[1:]), num_classes
            )
            (grad,) = torch.autograd.grad(per_sample.sum(), points)
            accumulated += grad.detach().sum(dim=0, keepdim=True)
    attribution = (delta * accumulated / steps).sum(dim=1)[0]
    mean_gradient = (accumulated / steps).sum(dim=1)[0]
    loss_clean = _loss_value(model, loss_fn, x0, y, num_classes)
    loss_adv = _loss_value(model, loss_fn, x_adv.detach().float(), y, num_classes)
    true_change = loss_adv - loss_clean
    completeness = abs(float(attribution.sum().item()) - true_change) / (
        abs(true_change) + 1e-8
    )
    return IGResult(
        attribution=attribution.detach().cpu().numpy().astype(np.float32),
        loss_clean=loss_clean,
        loss_adv=loss_adv,
        completeness_error=float(completeness),
        steps=int(steps),
        mean_gradient=mean_gradient.detach().cpu().numpy().astype(np.float32),
    )


def argmax_class_dice(
    pred: torch.Tensor, target: torch.Tensor, num_classes: int
) -> torch.Tensor:
    b = pred.shape[0]
    out = torch.empty((b, num_classes - 1), dtype=torch.float64, device=pred.device)
    p = pred.reshape(b, -1)
    t = target.reshape(b, -1)
    for c in range(1, num_classes):
        pc = p == c
        tc = t == c
        inter = (pc & tc).sum(dim=1).double()
        union = pc.sum(dim=1).double() + tc.sum(dim=1).double()
        out[:, c - 1] = torch.where(
            union > 0, 2.0 * inter / union.clamp_min(1.0), torch.ones_like(inter)
        )
    return out


class InterventionOperator(Protocol):
    """How a boolean mask turns (x, x_adv) into an intervened model input."""

    def mask_shape(self, x: torch.Tensor) -> tuple[int, ...]: ...

    def apply(
        self, x0: torch.Tensor, x1: torch.Tensor, mask: torch.Tensor, mode: str
    ) -> torch.Tensor: ...

    def check(
        self,
        x0: torch.Tensor,
        x1: torch.Tensor,
        out: torch.Tensor,
        mask: torch.Tensor,
        mode: str,
    ) -> None: ...


class SpatialOperator:
    """The voxel-mask intervention: torch.where inside the clean/adversarial envelope."""

    def mask_shape(self, x: torch.Tensor) -> tuple[int, ...]:
        return tuple(x.shape[2:])

    def apply(
        self, x0: torch.Tensor, x1: torch.Tensor, mask: torch.Tensor, mode: str
    ) -> torch.Tensor:
        # torch.where, not x + M*delta: the arithmetic form is not bit-identical to
        # x_adv / x in fp32, and the outside-mask assertion below is exact.
        if mode == "remove":
            return torch.where(mask, x0, x1)
        if mode == "keep":
            return torch.where(mask, x1, x0)
        raise ValueError(f"mode must be 'remove' or 'keep', got {mode!r}")

    def check(
        self,
        x0: torch.Tensor,
        x1: torch.Tensor,
        out: torch.Tensor,
        mask: torch.Tensor,
        mode: str,
    ) -> None:
        lower = torch.minimum(x0, x1)
        upper = torch.maximum(x0, x1)
        if bool((out < lower - 1e-6).any()) or bool((out > upper + 1e-6).any()):
            raise AssertionError("intervened input left the clean/adversarial envelope")
        reference = x1 if mode == "remove" else x0
        if bool((out[~mask.expand_as(out)] != reference[~mask.expand_as(out)]).any()):
            raise AssertionError("intervention changed voxels outside its mask")


class SpectralOperator:
    """Remove / keep a set of rfftn coefficients of the valid-masked perturbation.

    ``remove S`` = x_adv - P_S(delta); ``keep S`` = base + P_S(delta), with
    ``base = x_adv - P_all(delta)`` (x + delta_out + c0 inside the crop and
    x_adv in the padded margin).  A band-limited projection of an L-inf delta
    rings past epsilon, so there is no envelope to check; the linearity
    identity remove(S) + keep(S) - x_adv = base is asserted on the first batch
    of each call instead.
    """

    def __init__(
        self,
        F: torch.Tensor,
        base: torch.Tensor,
        crop: tuple[int, int, int],
        padded_shape: tuple[int, ...],
    ) -> None:
        self.F = F  # [D, H, W//2+1] complex, DC zeroed
        self.base = base  # [1, 1, Dp, Hp, Wp]
        self.crop = tuple(int(v) for v in crop)
        self.padded_shape = tuple(int(v) for v in padded_shape)
        self._checked = False

    @classmethod
    def from_split(
        cls,
        x_pad: torch.Tensor,
        adv_pad: torch.Tensor,
        split: object,
        *,
        crop: tuple[int, int, int],
        device: torch.device,
    ) -> SpectralOperator:
        F = torch.from_numpy(np.asarray(split.F).astype(np.complex64)).to(device)  # type: ignore[attr-defined]
        adv = adv_pad.detach().float()
        op = cls(F, adv, crop, tuple(x_pad.shape[2:]))
        all_mask = torch.ones((1, 1, *F.shape), dtype=torch.bool, device=device)
        all_mask[..., 0, 0, 0] = False
        op.base = adv - op._project(all_mask)
        return op

    def mask_shape(self, x: torch.Tensor) -> tuple[int, ...]:
        return tuple(self.F.shape)

    def _project(self, mask: torch.Tensor) -> torch.Tensor:
        coeff = torch.where(mask, self.F, torch.zeros_like(self.F))
        spatial = torch.fft.irfftn(coeff, s=self.crop, dim=(-3, -2, -1))
        pad: list[int] = []
        for full, part in zip(reversed(self.padded_shape), reversed(self.crop)):
            pad += [0, full - part]
        return torch.nn.functional.pad(spatial, pad)

    def apply(
        self, x0: torch.Tensor, x1: torch.Tensor, mask: torch.Tensor, mode: str
    ) -> torch.Tensor:
        p = self._project(mask)
        if mode == "remove":
            return x1 - p
        if mode == "keep":
            return self.base + p
        raise ValueError(f"mode must be 'remove' or 'keep', got {mode!r}")

    def check(
        self,
        x0: torch.Tensor,
        x1: torch.Tensor,
        out: torch.Tensor,
        mask: torch.Tensor,
        mode: str,
    ) -> None:
        if self._checked:
            return
        other = "keep" if mode == "remove" else "remove"
        both = out + self.apply(x0, x1, mask, other) - x1
        scale = float(x1.abs().max().item()) + 1e-6
        if not torch.allclose(
            both, self.base.expand_as(both), rtol=1e-5, atol=1e-5 * scale
        ):
            raise AssertionError(
                "spectral intervention is not linear: remove + keep != base"
            )
        self._checked = True


def _mask_batches(
    masks: np.ndarray, batch: int, device: torch.device
) -> Iterator[tuple[list[int], torch.Tensor]]:
    for start in range(0, masks.shape[0], batch):
        indices = list(range(start, min(start + batch, masks.shape[0])))
        yield indices, torch.from_numpy(masks[indices]).to(device)[:, None]


def evaluate_interventions(
    model: torch.nn.Module,
    loss_fn: LossFn,
    x: torch.Tensor,
    x_adv: torch.Tensor,
    y: torch.Tensor,
    masks: np.ndarray,
    *,
    mode: str,
    num_classes: int,
    batch: int = 8,
    keep_pred_indices: set[int] | None = None,
    crop: tuple[int, int, int] | None = None,
    operator: InterventionOperator | None = None,
) -> InterventionResult:
    """Forward every intervened input; Dice on the ``crop`` (unpadded) grid.

    ``operator`` decides what a mask means: the default ``SpatialOperator``
    takes voxel masks on the padded grid; ``SpectralOperator`` takes rfftn
    coefficient masks.
    """

    op = operator or SpatialOperator()
    if mode not in ("remove", "keep"):
        raise ValueError(f"mode must be 'remove' or 'keep', got {mode!r}")
    _check_5d(x, "x")
    if masks.ndim != 4 or tuple(masks.shape[1:]) != tuple(op.mask_shape(x)):
        raise ValueError(f"masks must be [K, *{tuple(op.mask_shape(x))}]")
    keep = keep_pred_indices or set()
    x0 = x.detach().float()
    x1 = x_adv.detach().float()
    window = tuple(slice(0, v) for v in (crop or tuple(x.shape[2:])))
    target = y[:, 0].long()[(slice(None), *window)]
    k = masks.shape[0]
    dice = np.full((k, num_classes - 1), np.nan)
    loss = np.full(k, np.nan)
    preds: dict[int, np.ndarray] = {}
    with torch.no_grad():
        for indices, mask in _mask_batches(masks.astype(bool), batch, x0.device):
            inputs = op.apply(x0, x1, mask, mode)
            op.check(x0.expand_as(inputs), x1.expand_as(inputs), inputs, mask, mode)
            logits = model(inputs)
            per_sample = loss_fn(
                logits, y.expand(inputs.shape[0], *y.shape[1:]), num_classes
            )
            argmax = logits.argmax(dim=1)
            d = argmax_class_dice(
                argmax[(slice(None), *window)],
                target.expand(inputs.shape[0], *target.shape[1:]),
                num_classes,
            )
            for row, i in enumerate(indices):
                dice[i] = d[row].cpu().numpy()
                loss[i] = float(per_sample[row].item())
                if i in keep:
                    preds[i] = argmax[row].to(torch.uint8).cpu().numpy()
    return InterventionResult(dice=dice, loss=loss, preds=preds)


def greedy_restoration(
    model: torch.nn.Module,
    loss_fn: LossFn,
    x: torch.Tensor,
    x_adv: torch.Tensor,
    y: torch.Tensor,
    masks_by_id: dict[int, np.ndarray],
    *,
    num_classes: int,
    primary: Callable[[np.ndarray], float],
    clean_primary: float,
    d_full: float,
    stop_fraction: float = 0.95,
    batch: int = 8,
    crop: tuple[int, int, int] | None = None,
    operator: InterventionOperator | None = None,
) -> list[dict[str, float | int]]:
    """Interaction-aware ranking: restore the patch giving the largest extra recovery."""

    remaining = sorted(masks_by_id)
    selected = np.zeros_like(next(iter(masks_by_id.values())), dtype=bool)
    previous_damage = d_full
    rows: list[dict[str, float | int]] = []
    step = 0
    while remaining:
        candidates = np.stack([selected | masks_by_id[i] for i in remaining])
        result = evaluate_interventions(
            model,
            loss_fn,
            x,
            x_adv,
            y,
            candidates,
            mode="remove",
            num_classes=num_classes,
            batch=batch,
            crop=crop,
            operator=operator,
        )
        damages = np.array([clean_primary - primary(row) for row in result.dice])
        best = int(np.argmin(damages))
        chosen = remaining.pop(best)
        selected |= masks_by_id[chosen]
        step += 1
        damage_after = float(damages[best])
        rows.append(
            {
                "step": step,
                "patch_id": int(chosen),
                "damage_after": damage_after,
                "recovered_fraction": float((d_full - damage_after) / d_full)
                if d_full > 0
                else float("nan"),
                "incremental_recovery": float(previous_damage - damage_after),
            }
        )
        previous_damage = damage_after
        if d_full > 0 and (d_full - damage_after) / d_full >= stop_fraction:
            break
    return rows


def pairwise_interactions(
    model: torch.nn.Module,
    loss_fn: LossFn,
    x: torch.Tensor,
    x_adv: torch.Tensor,
    y: torch.Tensor,
    masks_by_id: dict[int, np.ndarray],
    *,
    num_classes: int,
    primary: Callable[[np.ndarray], float],
    clean_primary: float,
    sufficiency_by_id: dict[int, float],
    batch: int = 8,
    crop: tuple[int, int, int] | None = None,
    operator: InterventionOperator | None = None,
) -> list[dict[str, float | int]]:
    ids = sorted(masks_by_id)
    pairs = [(a, b) for i, a in enumerate(ids) for b in ids[i + 1 :]]
    if not pairs:
        return []
    joint = np.stack([masks_by_id[a] | masks_by_id[b] for a, b in pairs])
    result = evaluate_interventions(
        model,
        loss_fn,
        x,
        x_adv,
        y,
        joint,
        mode="keep",
        num_classes=num_classes,
        batch=batch,
        crop=crop,
        operator=operator,
    )
    rows: list[dict[str, float | int]] = []
    for (a, b), row in zip(pairs, result.dice):
        joint_damage = clean_primary - primary(row)
        rows.append(
            {
                "patch_a": int(a),
                "patch_b": int(b),
                "joint_sufficiency": float(joint_damage),
                "interaction": float(
                    joint_damage - sufficiency_by_id[a] - sufficiency_by_id[b]
                ),
            }
        )
    return rows


__all__ = [
    "IGResult",
    "InterventionOperator",
    "InterventionResult",
    "SpatialOperator",
    "SpectralOperator",
    "argmax_class_dice",
    "evaluate_interventions",
    "greedy_restoration",
    "integrated_attribution",
    "pairwise_interactions",
]
