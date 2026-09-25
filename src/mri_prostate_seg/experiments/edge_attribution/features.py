"""Per-patch static features and local damage direction (plan §15, §27)."""

from __future__ import annotations

from collections.abc import Sequence

import numpy as np
from scipy.ndimage import distance_transform_edt

from mri_prostate_seg.experiments.segmentation_direction.surface import (
    extract_surface_mesh,
    iso_signed_distance_mm,
    sample_field_at_vertices,
)
from mri_prostate_seg.experiments.segmentation_direction.transitions import (
    binary_transition_masks,
)

from .tubes import EdgeAtlas

_NAN = float("nan")
BOUNDARY_MM = 2.0
PERI_MM = 10.0


def _category(median_sd: float, zone_overlap: float) -> str:
    if zone_overlap >= 0.5:
        return "zone_interface"
    if abs(median_sd) <= BOUNDARY_MM:
        return "outer_boundary"
    if median_sd < -BOUNDARY_MM:
        return "interior"
    if median_sd <= PERI_MM:
        return "peri_prostatic"
    return "background"


def patch_static_features(
    atlas: EdgeAtlas,
    *,
    spacing: tuple[float, float, float],
    signed_distance: np.ndarray,
    strength: np.ndarray,
    probs: np.ndarray | None,
    other_cores: dict[str, np.ndarray],
) -> list[dict[str, float | int | str]]:
    sp = np.asarray(spacing, dtype=np.float64)
    voxel_mm3 = float(np.prod(sp))
    tubes = atlas.tubes
    entropy = margin = None
    if probs is not None:
        p = np.clip(np.asarray(probs, dtype=np.float64), 1e-12, 1.0)
        entropy = -(p * np.log(p)).sum(axis=0)
        top2 = np.sort(p, axis=0)[-2:]
        margin = top2[1] - top2[0]
    rows: list[dict[str, float | int | str]] = []
    for patch_id in np.unique(tubes[tubes > 0]):
        mask = tubes == patch_id
        coords = np.argwhere(mask)
        centroid = (coords * sp).mean(axis=0)
        sd = np.asarray(signed_distance, dtype=np.float64)[mask]
        st = np.asarray(strength, dtype=np.float64)[mask]
        row: dict[str, float | int | str] = {
            "patch_id": int(patch_id),
            "voxel_count": int(mask.sum()),
            "volume_mm3": float(mask.sum() * voxel_mm3),
            "centroid_z_mm": float(centroid[0]),
            "centroid_y_mm": float(centroid[1]),
            "centroid_x_mm": float(centroid[2]),
            "signed_distance_min_mm": float(sd.min()),
            "signed_distance_median_mm": float(np.median(sd)),
            "signed_distance_max_mm": float(sd.max()),
            "edge_strength_mean": float(st.mean()),
            "edge_strength_median": float(np.median(st)),
            "edge_strength_p90": float(np.quantile(st, 0.9)),
            "clean_entropy_mean": float(entropy[mask].mean())
            if entropy is not None
            else _NAN,
            "clean_margin_mean": float(margin[mask].mean())
            if margin is not None
            else _NAN,
        }
        for name, core in other_cores.items():
            row[f"overlap_{name}"] = float(
                (mask & np.asarray(core, dtype=bool)).sum() / mask.sum()
            )
        row["anatomical_category"] = _category(
            float(row["signed_distance_median_mm"]),
            float(row.get("overlap_zone_interface_gt", 0.0)),
        )
        rows.append(row)
    return rows


def local_damage_direction(
    atlas: EdgeAtlas,
    *,
    spacing: tuple[float, float, float],
    gt_fg: np.ndarray,
    clean_fg: np.ndarray,
    adv_fg: np.ndarray,
    valid: np.ndarray,
    radius_mm: float = 4.0,
    seg_gt: np.ndarray | None = None,
    seg_clean: np.ndarray | None = None,
    seg_adv: np.ndarray | None = None,
    patch_ids: Sequence[int] | None = None,
) -> list[dict[str, float | int]]:
    sp = np.asarray(spacing, dtype=np.float64)
    voxel_mm3 = float(np.prod(sp))
    g = np.asarray(gt_fg, dtype=bool)
    c = np.asarray(clean_fg, dtype=bool)
    a = np.asarray(adv_fg, dtype=bool)
    ok = np.asarray(valid, dtype=bool)
    transitions = binary_transition_masks(g, c, a, valid=ok)
    zones = seg_gt is not None and seg_clean is not None and seg_adv is not None
    if zones:
        sc = np.asarray(seg_clean)
        sa = np.asarray(seg_adv)
        pz_to_tz = ok & (sc == 2) & (sa == 1)
        tz_to_pz = ok & (sc == 1) & (sa == 2)
    mesh = None
    disp_at_vertices = None
    if c.any() and not c.all():
        mesh = extract_surface_mesh(c, spacing)
        iso_adv = (
            iso_signed_distance_mm(a, spacing) if (a.any() and not a.all()) else None
        )
        if iso_adv is not None:
            disp_at_vertices = -sample_field_at_vertices(
                iso_adv, mesh.vertices_mm, spacing
            )
    tubes = atlas.tubes
    if not (tubes > 0).any():
        return []
    vertex_vox = None
    if mesh is not None:
        vertex_vox = np.clip(
            np.round(mesh.vertices_mm / sp).astype(int), 0, np.asarray(tubes.shape) - 1
        )
    distance, nearest = distance_transform_edt(
        tubes == 0, sampling=spacing, return_indices=True
    )
    nearest_label = tubes[tuple(nearest)]
    near = (distance <= radius_mm) & ok
    wanted = (
        np.unique(tubes[tubes > 0])
        if patch_ids is None
        else np.asarray(sorted(patch_ids))
    )
    rows: list[dict[str, float | int]] = []
    for patch_id in wanted:
        neighbourhood = near & (nearest_label == patch_id)
        row: dict[str, float | int] = {
            "patch_id": int(patch_id),
            "local_induced_fp_mm3": float(
                (transitions.induced_fp & neighbourhood).sum() * voxel_mm3
            ),
            "local_induced_fn_mm3": float(
                (transitions.induced_fn & neighbourhood).sum() * voxel_mm3
            ),
            "local_signed_displacement_median_mm": _NAN,
            "local_outward_displacement_p90_mm": _NAN,
            "local_inward_displacement_p10_mm": _NAN,
            "local_surface_vertices": 0,
        }
        if mesh is not None and disp_at_vertices is not None and vertex_vox is not None:
            inside = neighbourhood[tuple(vertex_vox.T)]
            if inside.any():
                d = disp_at_vertices[inside]
                row["local_signed_displacement_median_mm"] = float(np.median(d))
                row["local_outward_displacement_p90_mm"] = float(np.quantile(d, 0.9))
                row["local_inward_displacement_p10_mm"] = float(np.quantile(d, 0.1))
                row["local_surface_vertices"] = int(inside.sum())
        if zones:
            row["pz_to_tz_mm3"] = float((pz_to_tz & neighbourhood).sum() * voxel_mm3)
            row["tz_to_pz_mm3"] = float((tz_to_pz & neighbourhood).sum() * voxel_mm3)
        rows.append(row)
    return rows


__all__ = ["local_damage_direction", "patch_static_features"]
