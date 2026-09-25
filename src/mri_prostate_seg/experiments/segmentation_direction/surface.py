"""Signed surface displacement from distance fields (spec §10-§11).

The clean prediction surface answers "which way did the attack move the
model's own boundary"; the ground-truth surface answers "did that movement
worsen or correct the error".  Both are area weighted: marching-cubes vertices
are not uniformly spaced, so each carries one third of its adjacent triangle
areas.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy.ndimage import distance_transform_edt, map_coordinates
from skimage.measure import marching_cubes

_NUMERIC_KEYS = (
    "median_surface_motion_mm",
    "q05_surface_motion_mm",
    "q10_surface_motion_mm",
    "q90_surface_motion_mm",
    "q95_surface_motion_mm",
    "median_abs_surface_motion_mm",
    "rms_surface_motion_mm",
    "outward_surface_fraction",
    "inward_surface_fraction",
    "stable_surface_fraction",
    "worsened_outward_surface_fraction",
    "worsened_inward_surface_fraction",
    "corrected_surface_fraction",
    "clean_surface_area_mm2",
    "gt_surface_area_mm2",
)


@dataclass(frozen=True)
class SurfaceMesh:
    vertices_mm: np.ndarray  # [N, 3], array-axis physical coordinates
    faces: np.ndarray  # [F, 3]
    vertex_area_mm2: np.ndarray  # [N]


def _vertex_area_weights(vertices: np.ndarray, faces: np.ndarray) -> np.ndarray:
    triangles = vertices[faces]
    cross = np.cross(
        triangles[:, 1] - triangles[:, 0], triangles[:, 2] - triangles[:, 0]
    )
    face_area = 0.5 * np.linalg.norm(cross, axis=1)
    weights = np.zeros(len(vertices), dtype=np.float64)
    for corner in range(3):
        np.add.at(weights, faces[:, corner], face_area / 3.0)
    return weights


def _mask_state(mask: np.ndarray) -> str:
    if not mask.any():
        return "empty"
    if mask.all():
        return "full"
    return "ok"


def iso_signed_distance_mm(
    mask: np.ndarray, spacing: tuple[float, float, float]
) -> np.ndarray:
    """Signed distance (negative inside) to the 0.5 iso-surface, in mm.

    A plain EDT measures to the nearest voxel *centre* on the other side, so
    it overstates every distance by half a voxel along the nearest-boundary
    direction: 0.25 mm in-plane, 1.5 mm through-plane here.  Marching cubes
    puts the surface midway between those centres, so this subtracts that
    half step (its length taken along the EDT's own nearest-neighbour vector).
    On phantoms this turns a 2.37 mm reading of a 2 mm expansion into 2.00 mm.
    The band profile keeps the plain EDT, matching the structure table.
    """

    m = np.asarray(mask, dtype=bool)
    s = np.asarray(spacing, dtype=float)[:, None, None, None]
    out = np.zeros(m.shape, dtype=float)
    position = np.indices(m.shape, dtype=float)
    for region, sign in ((~m, 1.0), (m, -1.0)):
        distance, nearest = distance_transform_edt(
            region, sampling=spacing, return_indices=True
        )
        vector = (nearest - position) * s
        norm = np.linalg.norm(vector, axis=0)
        unit = np.divide(vector, norm, out=np.zeros_like(vector), where=norm > 0)
        half_step = 0.5 * np.sqrt(np.sum((unit * s) ** 2, axis=0))
        out[region] = (sign * np.maximum(distance - half_step, 0.0))[region]
    return out


def extract_surface_mesh(
    mask: np.ndarray, spacing: tuple[float, float, float]
) -> SurfaceMesh:
    """Marching cubes at 0.5 on a zero-padded copy, so edge-touching masks close."""

    m = np.asarray(mask, dtype=bool)
    if m.ndim != 3:
        raise ValueError(f"expected a 3-D mask, got {m.ndim}-D")
    state = _mask_state(m)
    if state != "ok":
        raise ValueError(f"surface is undefined for a {state} mask")
    padded = np.pad(m, 1).astype(np.float32)
    vertices, faces, _normals, _values = marching_cubes(
        padded, level=0.5, spacing=spacing
    )
    vertices = vertices - np.asarray(spacing, dtype=float)[None, :]
    return SurfaceMesh(vertices, faces, _vertex_area_weights(vertices, faces))


def sample_field_at_vertices(
    field: np.ndarray, vertices_mm: np.ndarray, spacing: tuple[float, float, float]
) -> np.ndarray:
    """Trilinear sample of a voxel field at physical vertex positions."""

    coordinates = (vertices_mm / np.asarray(spacing, dtype=float)[None, :]).T
    return map_coordinates(
        np.asarray(field, dtype=float), coordinates, order=1, mode="nearest"
    )


def weighted_quantile(
    values: np.ndarray, weights: np.ndarray, quantiles: np.ndarray
) -> np.ndarray:
    values = np.asarray(values, dtype=float)
    weights = np.asarray(weights, dtype=float)
    quantiles = np.asarray(quantiles, dtype=float)
    keep = np.isfinite(values) & np.isfinite(weights) & (weights > 0)
    values, weights = values[keep], weights[keep]
    if values.size == 0:
        return np.full(quantiles.shape, np.nan)
    order = np.argsort(values, kind="stable")
    values, weights = values[order], weights[order]
    cumulative = (np.cumsum(weights) - 0.5 * weights) / weights.sum()
    return np.interp(quantiles, cumulative, values)


def _invalid(reason: str) -> dict[str, float | str]:
    out: dict[str, float | str] = {key: float("nan") for key in _NUMERIC_KEYS}
    out["surface_metric_valid"] = 0.0
    out["surface_failure_reason"] = reason
    return out


def _fraction(condition: np.ndarray, weights: np.ndarray) -> float:
    return float(weights[condition].sum() / weights.sum())


def surface_motion_summary(
    ground_truth: np.ndarray,
    clean_pred: np.ndarray,
    adv_pred: np.ndarray,
    *,
    spacing: tuple[float, float, float],
    tolerance_mm: float,
    iso_gt: np.ndarray | None = None,
    iso_clean: np.ndarray | None = None,
) -> dict[str, float | str]:
    """Area-weighted clean->adversarial motion and GT-relative worsening.

    ``iso_gt`` and ``iso_clean`` accept precomputed ``iso_signed_distance_mm``
    fields; both are identical across every attack and epsilon of a case.
    """

    g = np.asarray(ground_truth, dtype=bool)
    c = np.asarray(clean_pred, dtype=bool)
    a = np.asarray(adv_pred, dtype=bool)
    for name, mask in (("gt", g), ("clean", c), ("adv", a)):
        state = _mask_state(mask)
        if state != "ok":
            return _invalid(f"{state}_{name}")

    phi_g = iso_signed_distance_mm(g, spacing) if iso_gt is None else np.asarray(iso_gt)
    phi_c = (
        iso_signed_distance_mm(c, spacing)
        if iso_clean is None
        else np.asarray(iso_clean)
    )
    phi_a = iso_signed_distance_mm(a, spacing)

    clean_mesh = extract_surface_mesh(c, spacing)
    motion = sample_field_at_vertices(phi_c, clean_mesh.vertices_mm, spacing) - (
        sample_field_at_vertices(phi_a, clean_mesh.vertices_mm, spacing)
    )
    w_clean = clean_mesh.vertex_area_mm2

    gt_mesh = extract_surface_mesh(g, spacing)
    gt_at = sample_field_at_vertices(phi_g, gt_mesh.vertices_mm, spacing)
    e_clean = gt_at - sample_field_at_vertices(phi_c, gt_mesh.vertices_mm, spacing)
    e_adv = gt_at - sample_field_at_vertices(phi_a, gt_mesh.vertices_mm, spacing)
    worsening = np.abs(e_adv) - np.abs(e_clean)
    w_gt = gt_mesh.vertex_area_mm2

    tau = float(tolerance_mm)
    q = weighted_quantile(motion, w_clean, np.asarray([0.05, 0.10, 0.5, 0.90, 0.95]))
    return {
        "median_surface_motion_mm": float(q[2]),
        "q05_surface_motion_mm": float(q[0]),
        "q10_surface_motion_mm": float(q[1]),
        "q90_surface_motion_mm": float(q[3]),
        "q95_surface_motion_mm": float(q[4]),
        "median_abs_surface_motion_mm": float(
            weighted_quantile(np.abs(motion), w_clean, np.asarray([0.5]))[0]
        ),
        "rms_surface_motion_mm": float(
            np.sqrt(np.sum(w_clean * motion**2) / w_clean.sum())
        ),
        "outward_surface_fraction": _fraction(motion > tau, w_clean),
        "inward_surface_fraction": _fraction(motion < -tau, w_clean),
        "stable_surface_fraction": _fraction(np.abs(motion) <= tau, w_clean),
        "worsened_outward_surface_fraction": _fraction(
            (worsening > tau) & (e_adv > tau), w_gt
        ),
        "worsened_inward_surface_fraction": _fraction(
            (worsening > tau) & (e_adv < -tau), w_gt
        ),
        "corrected_surface_fraction": _fraction(worsening < -tau, w_gt),
        "clean_surface_area_mm2": float(w_clean.sum()),
        "gt_surface_area_mm2": float(w_gt.sum()),
        "surface_metric_valid": 1.0,
        "surface_failure_reason": "",
    }


__all__ = [
    "SurfaceMesh",
    "extract_surface_mesh",
    "iso_signed_distance_mm",
    "sample_field_at_vertices",
    "surface_motion_summary",
    "weighted_quantile",
]
