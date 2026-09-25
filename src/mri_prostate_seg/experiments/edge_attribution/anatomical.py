"""Anatomical edge cores and the per-case atlas set (plan §9, §14)."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy.ndimage import binary_dilation, binary_erosion, generate_binary_structure

from .gradients import multiscale_edge_strength, roi_mask
from .patching import edge_cores
from .tubes import EdgeAtlas, build_atlas

STRUCTURE_6 = generate_binary_structure(3, 1)


@dataclass(frozen=True)
class AtlasConfig:
    sigmas_mm: tuple[float, ...] = (0.5, 1.0, 2.0)
    edge_quantile: float = 0.90
    min_core_voxels: int = 10
    patch_extent_mm: float = 6.0
    tube_radius_mm: float = 2.0
    roi_margin_mm: float = 20.0
    roi_margin_slices: int = 1


def foreground_mask(seg: np.ndarray) -> np.ndarray:
    return np.asarray(seg) > 0


def outer_boundary_core(foreground: np.ndarray) -> np.ndarray:
    fg = np.asarray(foreground, dtype=bool)
    return fg & ~binary_erosion(fg, structure=STRUCTURE_6, border_value=1)


def zone_interface_core(
    seg: np.ndarray, *, class_a: int = 1, class_b: int = 2
) -> np.ndarray:
    s = np.asarray(seg)
    a = s == class_a
    b = s == class_b
    a_touch = a & binary_dilation(b, structure=STRUCTURE_6)
    b_touch = b & binary_dilation(a, structure=STRUCTURE_6)
    return a_touch | b_touch


def build_case_atlases(
    image: np.ndarray,
    seg: np.ndarray,
    spacing: tuple[float, float, float],
    *,
    config: AtlasConfig,
    clean_pred: np.ndarray | None = None,
) -> tuple[dict[str, EdgeAtlas], np.ndarray, np.ndarray]:
    """Build every clean-derived atlas for one case.

    Returns ``(atlases, roi, strength)``; every atlas is restricted to ``roi``.
    """

    s = np.asarray(seg)
    valid = s >= 0
    fg = foreground_mask(s)
    roi = roi_mask(
        fg,
        valid,
        spacing,
        margin_mm_inplane=config.roi_margin_mm,
        margin_slices=config.roi_margin_slices,
    )
    strength = multiscale_edge_strength(
        image, spacing, sigmas_mm=config.sigmas_mm, domain=roi
    )

    def make(name: str, core: np.ndarray) -> EdgeAtlas:
        return build_atlas(
            name,
            core,
            spacing,
            domain=roi,
            extent_mm=config.patch_extent_mm,
            radius_mm=config.tube_radius_mm,
            min_voxels=config.min_core_voxels,
        )

    native = edge_cores(
        strength, roi, quantile=config.edge_quantile, min_voxels=config.min_core_voxels
    )
    atlases: dict[str, EdgeAtlas] = {
        "native_edges": make("native_edges", native > 0),
        "outer_boundary_gt": make("outer_boundary_gt", outer_boundary_core(fg)),
    }
    has_zones = bool((s == 2).any())
    if has_zones:
        atlases["zone_interface_gt"] = make("zone_interface_gt", zone_interface_core(s))
    if clean_pred is not None:
        p = np.asarray(clean_pred)
        atlases["outer_boundary_pred"] = make(
            "outer_boundary_pred", outer_boundary_core(p > 0)
        )
        if has_zones:
            atlases["zone_interface_pred"] = make(
                "zone_interface_pred", zone_interface_core(p)
            )
    return atlases, roi, strength


__all__ = [
    "AtlasConfig",
    "build_case_atlases",
    "foreground_mask",
    "outer_boundary_core",
    "zone_interface_core",
]
