"""Directional damage as a function of signed distance to the GT boundary (spec §7).

Uses the eight bands of the perturbation-energy profile so the energy and the
damage can be read on one axis (spec §25).  Rates divide by the voxels that were
clean-and-correct before the attack in that band (spec §7.2), never by raw
band size.
"""

from __future__ import annotations

import numpy as np

from mri_prostate_seg.experiments.perturbation_structure import (
    BOUNDARY_BAND_EDGES_MM,
    SIGNED_BAND_NAMES,
    signed_band_masks,
    signed_distance_mm,
)

from .transitions import BinaryTransitions

_LOW, _MID, _HIGH = BOUNDARY_BAND_EDGES_MM[1:]
SIGNED_BAND_EDGES_MM: dict[str, tuple[float, float]] = {
    "inside_beyond_10mm": (-np.inf, -_HIGH),
    "inside_5_10mm": (-_HIGH, -_MID),
    "inside_2_5mm": (-_MID, -_LOW),
    "inside_0_2mm": (-_LOW, 0.0),
    "0_2mm": (0.0, _LOW),
    "2_5mm": (_LOW, _MID),
    "5_10mm": (_MID, _HIGH),
    "beyond_10mm": (_HIGH, np.inf),
}
assert tuple(SIGNED_BAND_EDGES_MM) == SIGNED_BAND_NAMES


def _rate(numerator: int, denominator: int) -> float:
    return numerator / denominator if denominator else float("nan")


def directional_distance_profile(
    transitions: BinaryTransitions,
    ground_truth: np.ndarray,
    clean_pred: np.ndarray,
    *,
    valid: np.ndarray,
    spacing: tuple[float, float, float],
    distance: np.ndarray | None = None,
) -> list[dict[str, float | str]]:
    """One row per signed band with at-risk FP/FN rates and net change."""

    g = np.asarray(ground_truth, dtype=bool)
    c = np.asarray(clean_pred, dtype=bool)
    ok = np.asarray(valid, dtype=bool)
    if not g.any():
        raise ValueError("signed distance is undefined for an empty ground truth")
    phi = signed_distance_mm(g, spacing) if distance is None else np.asarray(distance)
    if phi.shape != g.shape:
        raise ValueError(f"distance shape {phi.shape} does not match {g.shape}")
    bands = signed_band_masks(phi)

    at_risk_fp = ok & ~g & ~c
    at_risk_fn = ok & g & c
    clean_fp = ok & ~g & c
    clean_fn = ok & g & ~c

    rows: list[dict[str, float | str]] = []
    for name in SIGNED_BAND_NAMES:
        band = bands[name] & ok
        left, right = SIGNED_BAND_EDGES_MM[name]
        n_band = int(np.count_nonzero(band))
        n_gain = int(np.count_nonzero(transitions.gained_fg & band))
        n_loss = int(np.count_nonzero(transitions.lost_fg & band))
        n_risk_fp = int(np.count_nonzero(at_risk_fp & band))
        n_risk_fn = int(np.count_nonzero(at_risk_fn & band))
        rows.append(
            {
                "band": name,
                "left_mm": float(left),
                "right_mm": float(right),
                "n_voxels": float(n_band),
                "induced_fp_rate": _rate(
                    int(np.count_nonzero(transitions.induced_fp & band)), n_risk_fp
                ),
                "induced_fn_rate": _rate(
                    int(np.count_nonzero(transitions.induced_fn & band)), n_risk_fn
                ),
                "corrected_fp_rate": _rate(
                    int(np.count_nonzero(transitions.corrected_fp & band)),
                    int(np.count_nonzero(clean_fp & band)),
                ),
                "corrected_fn_rate": _rate(
                    int(np.count_nonzero(transitions.corrected_fn & band)),
                    int(np.count_nonzero(clean_fn & band)),
                ),
                "net_fg_change_per_100_voxels": _rate(100 * (n_gain - n_loss), n_band),
                "gained_fg_voxels": float(n_gain),
                "lost_fg_voxels": float(n_loss),
                "at_risk_fp_voxels": float(n_risk_fp),
                "at_risk_fn_voxels": float(n_risk_fn),
            }
        )
    return rows


__all__ = ["SIGNED_BAND_EDGES_MM", "directional_distance_profile"]
