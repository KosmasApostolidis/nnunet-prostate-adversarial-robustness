"""Ground-truth-aware voxel transitions (spec §6).

Every changed voxel falls into exactly one of four groups: newly induced false
positive, newly induced false negative, corrected false positive, corrected
false negative.  Comparing the adversarial mask to the clean one alone would
count an accidental correction as damage (spec §47).

``valid`` is ``ground_truth >= 0``: the preprocessing writes ``-1`` where it
zeroed the image, and a voxel there can neither be at risk nor be damaged.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class BinaryTransitions:
    gained_fg: np.ndarray
    lost_fg: np.ndarray
    induced_fp: np.ndarray
    induced_fn: np.ndarray
    corrected_fp: np.ndarray
    corrected_fn: np.ndarray


def _require_same_shape(*arrays: np.ndarray) -> None:
    shapes = {a.shape for a in arrays}
    if len(shapes) != 1:
        raise ValueError(f"all masks must share one shape, got {sorted(shapes)}")


def binary_transition_masks(
    ground_truth: np.ndarray,
    clean_pred: np.ndarray,
    adv_pred: np.ndarray,
    *,
    valid: np.ndarray,
) -> BinaryTransitions:
    """Partition attack-induced label changes inside ``valid``."""

    g = np.asarray(ground_truth, dtype=bool)
    c = np.asarray(clean_pred, dtype=bool)
    a = np.asarray(adv_pred, dtype=bool)
    ok = np.asarray(valid, dtype=bool)
    _require_same_shape(g, c, a, ok)

    gained = ok & ~c & a
    lost = ok & c & ~a
    induced_fp = gained & ~g
    corrected_fn = gained & g
    induced_fn = lost & g
    corrected_fp = lost & ~g

    count = (
        induced_fp.astype(np.uint8)
        + induced_fn.astype(np.uint8)
        + corrected_fp.astype(np.uint8)
        + corrected_fn.astype(np.uint8)
    )
    if not np.array_equal(count > 0, gained | lost):
        raise AssertionError("transition partition is incomplete")
    if np.any(count > 1):
        raise AssertionError("transition categories overlap")
    return BinaryTransitions(
        gained, lost, induced_fp, induced_fn, corrected_fp, corrected_fn
    )


def physical_measure(mask: np.ndarray, spacing: tuple[float, ...]) -> float:
    """Volume (mm^3) of a binary mask."""

    m = np.asarray(mask, dtype=bool)
    if m.ndim != len(spacing):
        raise ValueError(f"spacing has {len(spacing)} entries for a {m.ndim}-D mask")
    return float(np.count_nonzero(m)) * float(np.prod(np.asarray(spacing, dtype=float)))


def _centroid_mm(mask: np.ndarray, spacing: tuple[float, ...]) -> np.ndarray:
    indices = np.argwhere(mask)
    if indices.size == 0:
        return np.full(mask.ndim, np.nan)
    return indices.mean(axis=0) * np.asarray(spacing, dtype=float)


def centroid_shift_norm_mm(
    clean_pred: np.ndarray, adv_pred: np.ndarray, spacing: tuple[float, ...]
) -> float:
    """Euclidean centroid displacement in mm; NaN when either mask is empty.

    The norm is orientation-invariant, so it needs no RAS conversion.  The
    signed components do (spec §16) and belong to Stage 2.
    """

    shift = _centroid_mm(np.asarray(adv_pred, bool), spacing) - _centroid_mm(
        np.asarray(clean_pred, bool), spacing
    )
    return float(np.linalg.norm(shift)) if np.all(np.isfinite(shift)) else float("nan")


def _bias(positive: float, negative: float, eps: float) -> float:
    total = positive + negative
    if total <= 0.0:
        return float("nan")
    return (positive - negative) / (total + eps)


def summarize_binary_transitions(
    transitions: BinaryTransitions,
    ground_truth: np.ndarray,
    clean_pred: np.ndarray,
    adv_pred: np.ndarray,
    *,
    valid: np.ndarray,
    spacing: tuple[float, ...],
    eps: float = 1e-12,
) -> dict[str, float]:
    """Directional volume metrics (spec §6.3).  Biases are NaN when undefined."""

    c = np.asarray(clean_pred, dtype=bool)
    a = np.asarray(adv_pred, dtype=bool)
    ok = np.asarray(valid, dtype=bool)
    v_gain = physical_measure(transitions.gained_fg, spacing)
    v_loss = physical_measure(transitions.lost_fg, spacing)
    v_ifp = physical_measure(transitions.induced_fp, spacing)
    v_ifn = physical_measure(transitions.induced_fn, spacing)
    v_cfp = physical_measure(transitions.corrected_fp, spacing)
    v_cfn = physical_measure(transitions.corrected_fn, spacing)
    v_clean = physical_measure(c & ok, spacing)
    v_adv = physical_measure(a & ok, spacing)
    return {
        "clean_volume_mm3": v_clean,
        "adv_volume_mm3": v_adv,
        "volume_change_mm3": v_adv - v_clean,
        "volume_change_pct": (
            float("nan") if v_clean == 0.0 else 100.0 * (v_adv - v_clean) / v_clean
        ),
        "gained_fg_mm3": v_gain,
        "lost_fg_mm3": v_loss,
        "induced_fp_mm3": v_ifp,
        "induced_fn_mm3": v_ifn,
        "corrected_fp_mm3": v_cfp,
        "corrected_fn_mm3": v_cfn,
        "harm_volume_mm3": v_ifp + v_ifn,
        "corrected_volume_mm3": v_cfp + v_cfn,
        "change_bias": _bias(v_gain, v_loss, eps),
        "damage_bias": _bias(v_ifp, v_ifn, eps),
        "centroid_shift_norm_mm": centroid_shift_norm_mm(c & ok, a & ok, spacing),
        "changed_voxels_outside_valid": float(np.count_nonzero((c != a) & ~ok)),
    }


__all__ = [
    "BinaryTransitions",
    "binary_transition_masks",
    "centroid_shift_norm_mm",
    "physical_measure",
    "summarize_binary_transitions",
]
