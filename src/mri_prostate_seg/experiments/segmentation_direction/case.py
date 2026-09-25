"""One binary target (WG, zone union, or one zone) under one condition."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from mri_prostate_seg.experiments.perturbation_structure import signed_distance_mm
from mri_prostate_seg.metrics.segmentation import dice_score

from .profile import directional_distance_profile
from .surface import iso_signed_distance_mm, surface_motion_summary
from .transitions import binary_transition_masks, summarize_binary_transitions

SUCCESS_DELTA_DICE = -0.01


@dataclass(frozen=True)
class TargetGeometry:
    """Distance fields that every condition of a case shares.

    ``phi_gt`` is the plain EDT field the bands use (same reference as the
    structure table); ``iso_gt``/``iso_clean`` are the half-voxel-corrected
    fields the surface metrics use.
    """

    phi_gt: np.ndarray
    iso_gt: np.ndarray
    iso_clean: np.ndarray | None


def target_geometry(
    gt_mask: np.ndarray, clean_mask: np.ndarray, spacing: tuple[float, float, float]
) -> TargetGeometry:
    g = np.asarray(gt_mask, dtype=bool)
    c = np.asarray(clean_mask, dtype=bool)
    if not g.any():
        raise ValueError("ground truth is empty")
    if g.all():
        raise ValueError("ground truth fills the volume")
    iso_clean = (
        iso_signed_distance_mm(c, spacing) if (c.any() and not c.all()) else None
    )
    return TargetGeometry(
        signed_distance_mm(g, spacing), iso_signed_distance_mm(g, spacing), iso_clean
    )


def attack_success(delta_dice: float, threshold: float = SUCCESS_DELTA_DICE) -> bool:
    return bool(np.isfinite(delta_dice) and delta_dice <= threshold)


def analyze_binary_target(
    gt_mask: np.ndarray,
    clean_mask: np.ndarray,
    adv_mask: np.ndarray,
    *,
    valid: np.ndarray,
    spacing: tuple[float, float, float],
    tolerance_mm: float,
    geometry: TargetGeometry | None = None,
) -> tuple[dict[str, float | str], list[dict[str, float | str]]]:
    """Summary row and eight profile rows for one target under one condition."""

    g = np.asarray(gt_mask, dtype=bool)
    c = np.asarray(clean_mask, dtype=bool)
    a = np.asarray(adv_mask, dtype=bool)
    ok = np.asarray(valid, dtype=bool)
    if geometry is None:
        geometry = target_geometry(g, c, spacing)
    if geometry.phi_gt.shape != g.shape:
        raise ValueError("geometry was built for another shape")

    transitions = binary_transition_masks(g, c, a, valid=ok)
    summary: dict[str, float | str] = {}
    summary.update(
        summarize_binary_transitions(transitions, g, c, a, valid=ok, spacing=spacing)
    )
    summary.update(
        surface_motion_summary(
            g,
            c,
            a,
            spacing=spacing,
            tolerance_mm=tolerance_mm,
            iso_gt=geometry.iso_gt,
            iso_clean=geometry.iso_clean,
        )
    )
    clean_dice = float(dice_score(c, g))
    adv_dice = float(dice_score(a, g))
    summary["clean_dice"] = clean_dice
    summary["adv_dice"] = adv_dice
    summary["delta_dice"] = adv_dice - clean_dice
    profile = directional_distance_profile(
        transitions, g, c, valid=ok, spacing=spacing, distance=geometry.phi_gt
    )
    return summary, profile


__all__ = [
    "SUCCESS_DELTA_DICE",
    "TargetGeometry",
    "analyze_binary_target",
    "attack_success",
    "target_geometry",
]
