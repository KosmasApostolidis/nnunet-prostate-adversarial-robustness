"""Disjoint nearest-core tubes and the atlas container (plan §13)."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy.ndimage import distance_transform_edt
from scipy.ndimage import label as cc_label

from .patching import STRUCTURE_26, drop_small_components, split_into_patches


def grow_tubes(
    patch_labels: np.ndarray,
    spacing: tuple[float, float, float],
    *,
    radius_mm: float,
    domain: np.ndarray,
) -> np.ndarray:
    """Expand labelled cores into disjoint tubes of physical radius ``radius_mm``."""

    cores = np.asarray(patch_labels)
    ok = np.asarray(domain, dtype=bool)
    if cores.ndim != 3 or cores.shape != ok.shape:
        raise ValueError("patch_labels and domain must be equal-shaped 3-D arrays")
    if radius_mm <= 0:
        raise ValueError("radius_mm must be positive")
    if not (cores > 0).any():
        return np.zeros(cores.shape, dtype=np.int32)
    distance, nearest = distance_transform_edt(
        cores == 0, sampling=spacing, return_indices=True
    )
    nearest_label = cores[tuple(nearest)]
    tubes = np.zeros(cores.shape, dtype=np.int32)
    within = (distance <= float(radius_mm)) & ok
    tubes[within] = nearest_label[within]
    tubes[(cores > 0) & ok] = cores[(cores > 0) & ok]
    return tubes


@dataclass(frozen=True)
class EdgeAtlas:
    name: str
    cores: np.ndarray  # int32 patch labels on core voxels
    tubes: np.ndarray  # int32 patch labels on tube voxels (disjoint)

    @property
    def n_patches(self) -> int:
        return int(self.tubes.max()) if self.tubes.size else 0


def build_atlas(
    name: str,
    core_mask: np.ndarray,
    spacing: tuple[float, float, float],
    *,
    domain: np.ndarray,
    extent_mm: float,
    radius_mm: float,
    min_voxels: int,
) -> EdgeAtlas:
    """Label, split, and tube a boolean core mask into a disjoint atlas."""

    ok = np.asarray(domain, dtype=bool)
    core = np.asarray(core_mask, dtype=bool) & ok
    labels, _ = cc_label(core, structure=STRUCTURE_26)
    labels = drop_small_components(labels, min_voxels)
    patches = split_into_patches(
        labels.astype(np.int32), spacing, extent_mm=extent_mm, min_voxels=min_voxels
    )
    tubes = grow_tubes(patches, spacing, radius_mm=radius_mm, domain=ok)
    return EdgeAtlas(name=name, cores=patches, tubes=tubes)


def atlas_flags(
    atlas: EdgeAtlas, domain: np.ndarray, *, max_patch_fraction: float = 0.05
) -> list[str]:
    ok = np.asarray(domain, dtype=bool)
    flags: list[str] = []
    if atlas.n_patches == 0:
        return ["no_patches"]
    counts = np.bincount(atlas.tubes[atlas.tubes > 0].ravel())
    if counts.size > 1 and counts[1:].max() > max_patch_fraction * ok.sum():
        flags.append("patch_exceeds_roi_fraction")
    if ((atlas.tubes > 0) & ~ok).any():
        flags.append("tube_outside_domain")
    if ((atlas.cores > 0) & (atlas.tubes != atlas.cores)).any():
        flags.append("core_not_in_tube")
    return flags


__all__ = ["EdgeAtlas", "atlas_flags", "build_atlas", "grow_tubes"]
