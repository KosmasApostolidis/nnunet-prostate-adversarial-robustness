"""Tube growth and atlas phantoms (plan §13)."""

from __future__ import annotations

import numpy as np

from mri_prostate_seg.experiments.edge_attribution.tubes import (
    EdgeAtlas,
    atlas_flags,
    build_atlas,
    grow_tubes,
)

SPACING = (3.0, 0.5, 0.5)
SHAPE = (8, 64, 64)


def _two_cores() -> np.ndarray:
    cores = np.zeros(SHAPE, dtype=np.int32)
    cores[4, 32, 10:20] = 1
    cores[4, 32, 40:50] = 2
    return cores


def test_tube_radius_is_physical_and_in_plane_on_anisotropic_grid() -> None:
    tubes = grow_tubes(
        _two_cores(), SPACING, radius_mm=2.0, domain=np.ones(SHAPE, bool)
    )
    # 2 mm = 4 voxels in-plane, 0 slices through-plane.
    assert tubes[4, 32 + 4, 15] == 1 and tubes[4, 32 + 5, 15] == 0
    assert tubes[3].max() == 0 and tubes[5].max() == 0


def test_tube_reaches_adjacent_slice_on_isotropic_grid() -> None:
    tubes = grow_tubes(
        _two_cores(), (1.0, 1.0, 1.0), radius_mm=2.0, domain=np.ones(SHAPE, bool)
    )
    assert tubes[3, 32, 15] == 1 and tubes[6, 32, 15] == 1 and tubes[7, 32, 15] == 0


def test_tubes_are_disjoint_and_nearest_core_wins() -> None:
    cores = np.zeros(SHAPE, dtype=np.int32)
    cores[4, 32, 20] = 1
    cores[4, 32, 26] = 2  # 3 mm apart; 2 mm tubes overlap between them
    tubes = grow_tubes(cores, SPACING, radius_mm=2.0, domain=np.ones(SHAPE, bool))
    assert tubes[4, 32, 22] == 1 and tubes[4, 32, 24] == 2
    assert set(np.unique(tubes).tolist()) == {0, 1, 2}


def test_tubes_respect_domain_and_keep_core_ids() -> None:
    domain = np.ones(SHAPE, dtype=bool)
    domain[:, :, 45:] = False
    tubes = grow_tubes(_two_cores(), SPACING, radius_mm=2.0, domain=domain)
    assert tubes[:, :, 45:].max() == 0
    assert tubes[4, 32, 12] == 1


def test_build_atlas_and_flags_clean_case() -> None:
    core = np.zeros(SHAPE, dtype=bool)
    core[4, 32, 8:56] = True
    domain = np.ones(SHAPE, dtype=bool)
    atlas = build_atlas(
        "native_edges",
        core,
        SPACING,
        domain=domain,
        extent_mm=6.0,
        radius_mm=2.0,
        min_voxels=3,
    )
    assert isinstance(atlas, EdgeAtlas)
    # split_into_patches seeds by patch volume, not linear extent (Fix 1): this
    # is a 1-voxel-wide/thick, 48-voxel line, so its physical volume (36 mm^3)
    # is far below the 6 mm cube target (216 mm^3) despite its 24 mm length,
    # and it stays a single patch. Was 4 under the old extent-based rule.
    assert atlas.n_patches == 1
    assert atlas_flags(atlas, domain) == []


def test_flags_report_oversized_patch_and_no_patches() -> None:
    domain = np.zeros(SHAPE, dtype=bool)
    domain[4, 30:35, 8:56] = True
    core = np.zeros(SHAPE, dtype=bool)
    core[4, 32, 8:56] = True
    atlas = build_atlas(
        "x", core, SPACING, domain=domain, extent_mm=100.0, radius_mm=2.0, min_voxels=1
    )
    assert "patch_exceeds_roi_fraction" in atlas_flags(
        atlas, domain, max_patch_fraction=0.05
    )
    empty = EdgeAtlas("y", np.zeros(SHAPE, np.int32), np.zeros(SHAPE, np.int32))
    assert atlas_flags(empty, domain) == ["no_patches"]
