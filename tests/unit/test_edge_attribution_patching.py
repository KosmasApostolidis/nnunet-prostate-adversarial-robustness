"""Edge-core and patching phantoms (plan §11, §12)."""

from __future__ import annotations

import numpy as np

from mri_prostate_seg.experiments.edge_attribution.patching import (
    edge_cores,
    farthest_point_seeds,
    split_into_patches,
)

SPACING = (3.0, 0.5, 0.5)
SHAPE = (8, 64, 64)


def test_edge_cores_keeps_only_top_quantile_components_above_min_size() -> None:
    strength = np.zeros(SHAPE)
    strength[4, 20:40, 30] = 10.0  # a 20-voxel line
    strength[2, 5, 5] = 10.0  # an isolated voxel
    domain = np.ones(SHAPE, dtype=bool)
    labels = edge_cores(strength, domain, quantile=0.99, min_voxels=10)
    assert labels.max() == 1
    assert labels[4, 20:40, 30].min() == 1
    assert labels[2, 5, 5] == 0


def test_edge_cores_is_restricted_to_domain() -> None:
    strength = np.full(SHAPE, 1.0)
    strength[4, 20:40, 30] = 10.0
    domain = np.zeros(SHAPE, dtype=bool)
    domain[:, :, :25] = True
    labels = edge_cores(strength, domain, quantile=0.5, min_voxels=1)
    assert labels[4, 20:40, 30].max() == 0


def test_farthest_point_seeds_are_deterministic_and_spread() -> None:
    points = np.array([[0, 0, 0], [0, 0, 10], [0, 0, 20], [0, 0, 5]], dtype=float)
    seeds = farthest_point_seeds(points, 2)
    assert seeds.tolist() == farthest_point_seeds(points, 2).tolist()
    picked = points[seeds][:, 2]
    assert picked[0] == 10.0  # nearest the centroid (8.75)
    assert abs(picked[0] - picked[1]) >= 10.0  # farthest from it


def test_split_long_line_into_expected_number_of_patches() -> None:
    cores = np.zeros(SHAPE, dtype=np.int32)
    cores[4, 32, 8:56] = 1  # 48 voxels * 0.5 mm = 24 mm long
    patches = split_into_patches(cores, SPACING, extent_mm=6.0, min_voxels=3)
    # Seeding is by patch volume now (Fix 1), not linear extent: this line is
    # 1 voxel wide/thick, so its physical volume (48 * 0.75 mm^3 = 36 mm^3) is
    # far below the 6 mm cube target (216 mm^3) despite its 24 mm length, and
    # it stays a single patch. Was 4 under the old extent-based rule.
    assert patches.max() == 1
    assert (patches > 0).sum() == 48
    ids = np.unique(patches[patches > 0])
    assert ids.tolist() == [1]


def test_split_ids_are_sorted_by_centroid_and_invariant_to_label_order() -> None:
    cores = np.zeros(SHAPE, dtype=np.int32)
    cores[4, 10, 8:20] = 1
    cores[4, 50, 8:20] = 2
    a = split_into_patches(cores, SPACING, extent_mm=100.0, min_voxels=1)
    swapped = np.where(cores == 1, 2, np.where(cores == 2, 1, 0)).astype(np.int32)
    b = split_into_patches(swapped, SPACING, extent_mm=100.0, min_voxels=1)
    assert np.array_equal(a, b)
    assert a[4, 10, 8] == 1 and a[4, 50, 8] == 2


def test_split_merges_tiny_fragments_into_nearest_patch() -> None:
    # Volume seeding needs real volume: the old 32-voxel line was 24 mm^3
    # against a 216 mm^3 target, so k=1 and _merge_small was never reached --
    # `counts.min() >= 6` held trivially on a single 32-voxel patch. A 2-slice
    # 20x20 slab is 800 * 0.75 = 600 mm^3, so k=3, and the 12-voxel bridge
    # hanging off it is the sub-minimum fragment the split leaves behind.
    cores = np.zeros(SHAPE, dtype=np.int32)
    cores[3:5, 20:40, 20:40] = 1  # 800-voxel slab
    cores[3, 40:52, 20] = 1  # 12-voxel bridge
    seeds = 3  # round(812 * 0.75 mm^3 / 6^3 mm^3)
    # Pre-merge, the split really does produce a piece below min_voxels: with
    # nothing to merge, three pieces survive and the smallest is 14 voxels.
    unmerged = split_into_patches(cores, SPACING, extent_mm=6.0, min_voxels=1)
    assert unmerged.max() == seeds
    assert np.bincount(unmerged[unmerged > 0])[1:].min() < 20
    patches = split_into_patches(cores, SPACING, extent_mm=6.0, min_voxels=20)
    counts = np.bincount(patches[patches > 0])[1:]
    assert patches.max() < seeds  # only a merge can lose a patch
    assert counts.min() >= 20
    assert (patches > 0).sum() == 812  # absorbed, not dropped


def test_split_extent_is_physical_not_voxel() -> None:
    # The volume rule seeds on prod(spacing), which is orientation-independent,
    # so two differently-oriented lines can never discriminate here -- the old
    # phantom asserted 1 == 1 twice. Hold the VOXEL count fixed and change the
    # spacing instead: 600 voxels are 450 mm^3 at 3.0 x 0.5 x 0.5 mm but only
    # 75 mm^3 at 0.5 mm isotropic, a 6x difference against a 216 mm^3 target.
    cores = np.zeros((6, 20, 40), dtype=np.int32)
    cores[1:4, 5:15, 10:30] = 1
    assert (cores > 0).sum() == 600
    coarse = split_into_patches(cores, (3.0, 0.5, 0.5), extent_mm=6.0, min_voxels=1)
    fine = split_into_patches(cores, (0.5, 0.5, 0.5), extent_mm=6.0, min_voxels=1)
    assert coarse.max() == 2 and fine.max() == 1


def _patch_extents_mm(
    patches: np.ndarray, spacing: tuple[float, float, float]
) -> list[float]:
    sp = np.asarray(spacing, dtype=np.float64)
    extents = []
    for patch_id in np.unique(patches[patches > 0]):
        coords = np.argwhere(patches == patch_id) * sp
        extents.append(float((coords.max(axis=0) - coords.min(axis=0)).max()))
    return extents


def test_split_sheet_component_bounds_patch_extent() -> None:
    """A 2-D sheet needs ~(extent/target)^2 seeds, the case this fix targets:
    real outer_boundary_gt surfaces got 34x-oversized patches under the old
    extent-based rule. 60x60 voxels, 2 z-slices (6 mm) thick so the sheet's
    physical thickness matches extent_mm: the old rule gives k=ceil(29.5/6)=5
    and leaves patches up to ~29 mm across (median ~14.5 mm); the volume rule
    gives k~25 and keeps the median patch within 1.5x the 6 mm target. A few
    FPS-seeded corner cells running to ~2x are a pre-existing property of
    farthest-point-seeding + Voronoi assignment (out of this fix's scope), so
    the max bound is set looser than the median bound.
    """
    shape = (10, 80, 80)
    cores = np.zeros(shape, dtype=np.int32)
    cores[3:5, 10:70, 10:70] = 1  # 60x60 voxel sheet, 6 mm thick in z
    patches = split_into_patches(cores, SPACING, extent_mm=6.0, min_voxels=10)
    extents = _patch_extents_mm(patches, SPACING)
    assert max(extents) <= 2.0 * 6.0
    assert float(np.median(extents)) <= 1.5 * 6.0


def test_split_blob_component_bounds_patch_extent() -> None:
    """A 3-D blob needs ~(extent/target)^3 seeds. 24x24x8 voxels (24 mm thick
    in z, 12 mm in y/x): the old extent-based rule gives k=ceil(24/6)=4 and
    leaves patches up to ~15 mm across (median ~11.5 mm); the volume rule
    gives k~16 and keeps the median patch within 1.5x the 6 mm target. (A
    12x12x4 blob, as suggested in the fix note, gives k=2 under both the old
    and new rule -- it would not have caught this bug, so this uses a larger
    block where old and new diverge.)
    """
    shape = (20, 40, 40)
    cores = np.zeros(shape, dtype=np.int32)
    cores[2:10, 8:32, 8:32] = 1  # 8x24x24 voxel solid block
    patches = split_into_patches(cores, SPACING, extent_mm=6.0, min_voxels=10)
    extents = _patch_extents_mm(patches, SPACING)
    assert max(extents) <= 2.0 * 6.0
    assert float(np.median(extents)) <= 1.5 * 6.0
