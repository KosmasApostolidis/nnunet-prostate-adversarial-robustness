"""Edge cores and deterministic physical patching (plan §11, §12)."""

from __future__ import annotations

import numpy as np
from scipy.ndimage import label as cc_label
from scipy.spatial import cKDTree

STRUCTURE_26 = np.ones((3, 3, 3), dtype=bool)


def edge_cores(
    strength: np.ndarray,
    domain: np.ndarray,
    *,
    quantile: float = 0.90,
    min_voxels: int = 10,
) -> np.ndarray:
    """26-connected components of the top-quantile strength voxels in ``domain``."""

    s = np.asarray(strength, dtype=np.float64)
    ok = np.asarray(domain, dtype=bool)
    if s.shape != ok.shape or s.ndim != 3:
        raise ValueError("strength and domain must be equal-shaped 3-D arrays")
    if not ok.any():
        raise ValueError("domain contains no voxels")
    threshold = float(np.quantile(s[ok], quantile))
    # `>`, not `>=`: on a tied plateau at the threshold (a mostly-flat strength
    # map) `>=` selects the whole plateau — in the degenerate case the entire
    # ROI — and builds an all-background "edge" atlas. `>` fails safe to no
    # cores, which atlas_flags reports as "no_patches".
    candidate = ok & (s > threshold)
    labels, count = cc_label(candidate, structure=STRUCTURE_26)
    if count == 0:
        return np.zeros(s.shape, dtype=np.int32)
    return drop_small_components(labels, min_voxels)


def drop_small_components(labels: np.ndarray, min_voxels: int) -> np.ndarray:
    """Remove labelled components below ``min_voxels`` and relabel 1..C."""

    sizes = np.bincount(np.asarray(labels).ravel())
    keep = sizes >= int(min_voxels)
    keep[0] = False
    remap = np.zeros(sizes.size, dtype=np.int32)
    remap[keep] = np.arange(1, int(keep.sum()) + 1, dtype=np.int32)
    return remap[np.asarray(labels)]


def farthest_point_seeds(points_mm: np.ndarray, k: int) -> np.ndarray:
    """Deterministic farthest-point sampling; first seed nearest the centroid."""

    pts = np.asarray(points_mm, dtype=np.float64)
    if pts.ndim != 2 or pts.shape[1] != 3:
        raise ValueError("points_mm must be [N, 3]")
    n = pts.shape[0]
    k = max(1, min(int(k), n))
    first = int(np.argmin(((pts - pts.mean(axis=0)) ** 2).sum(axis=1)))
    seeds = [first]
    nearest = ((pts - pts[first]) ** 2).sum(axis=1)
    for _ in range(k - 1):
        candidate = int(np.argmax(nearest))
        seeds.append(candidate)
        nearest = np.minimum(nearest, ((pts - pts[candidate]) ** 2).sum(axis=1))
    return np.asarray(seeds, dtype=np.int64)


def _merge_small(labels: np.ndarray, pts: np.ndarray, min_voxels: int) -> np.ndarray:
    """Reassign voxels of sub-minimum patches to the nearest other patch."""

    labels = labels.copy()
    while True:
        ids, counts = np.unique(labels, return_counts=True)
        small = ids[counts < min_voxels]
        if small.size == 0 or ids.size == 1:
            return labels
        victim = int(small[np.argmin(counts[counts < min_voxels])])
        others = labels != victim
        tree = cKDTree(pts[others])
        _, nearest = tree.query(pts[~others])
        labels[~others] = labels[others][nearest]


def split_into_patches(
    core_labels: np.ndarray,
    spacing: tuple[float, float, float],
    *,
    extent_mm: float = 6.0,
    min_voxels: int = 10,
) -> np.ndarray:
    """Split every core component into ~``extent_mm`` patches; IDs by centroid."""

    cores = np.asarray(core_labels)
    if cores.ndim != 3:
        raise ValueError("core_labels must be 3-D")
    sp = np.asarray(spacing, dtype=np.float64)
    out = np.zeros(cores.shape, dtype=np.int32)
    pieces: list[tuple[np.ndarray, np.ndarray]] = []  # (centroid_mm, coords)
    for component in np.unique(cores[cores > 0]):
        coords = np.argwhere(cores == component)
        pts = coords * sp
        # Seed count from patch VOLUME, not linear extent. Extent/target is only
        # correct for a 1-D curve; a surface sheet needs ~(extent/target)^2 seeds
        # and a blob ~(extent/target)^3. Seeding by volume gives ~target-sized
        # patches whatever the component's dimensionality.
        target_volume_mm3 = float(extent_mm) ** 3
        component_volume_mm3 = float(len(pts)) * float(np.prod(sp))
        k = max(1, int(round(component_volume_mm3 / target_volume_mm3)))
        seeds = farthest_point_seeds(pts, k)
        _, assignment = cKDTree(pts[seeds]).query(pts)
        # Re-split disconnected fragments of each seed region.
        local = np.zeros(cores.shape, dtype=np.int32)
        next_id = 1
        for seed_index in range(len(seeds)):
            member = coords[assignment == seed_index]
            mask = np.zeros(cores.shape, dtype=bool)
            mask[tuple(member.T)] = True
            frag, n_frag = cc_label(mask, structure=STRUCTURE_26)
            for f in range(1, n_frag + 1):
                local[frag == f] = next_id
                next_id += 1
        labels = local[tuple(coords.T)]
        labels = _merge_small(labels, pts, int(min_voxels))
        for patch in np.unique(labels):
            member = coords[labels == patch]
            pieces.append(((member * sp).mean(axis=0), member))
    order = sorted(range(len(pieces)), key=lambda i: tuple(pieces[i][0]))
    for new_id, index in enumerate(order, start=1):
        out[tuple(pieces[index][1].T)] = new_id
    return out


__all__ = [
    "drop_small_components",
    "edge_cores",
    "farthest_point_seeds",
    "split_into_patches",
]
