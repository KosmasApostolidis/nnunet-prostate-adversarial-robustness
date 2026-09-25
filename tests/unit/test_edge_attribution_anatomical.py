"""Anatomical atlas phantoms (plan §14)."""

from __future__ import annotations

import numpy as np

from mri_prostate_seg.experiments.edge_attribution.anatomical import (
    AtlasConfig,
    build_case_atlases,
    foreground_mask,
    outer_boundary_core,
    zone_interface_core,
)

SPACING = (3.0, 0.5, 0.5)
SHAPE = (10, 64, 64)


def _zones_seg() -> np.ndarray:
    seg = np.zeros(SHAPE, dtype=np.int16)
    seg[3:7, 20:44, 16:48] = 1  # TZ block
    seg[3:7, 20:44, 32:48] = 2  # PZ on the right half
    seg[:, :, :4] = -1  # ignore band
    return seg


def test_outer_boundary_core_is_one_voxel_inner_shell() -> None:
    fg = foreground_mask(_zones_seg())
    core = outer_boundary_core(fg)
    assert core[3, 20, 16] and not core[5, 30, 30]
    assert core.sum() < fg.sum()
    assert (core & ~fg).sum() == 0


def test_zone_interface_core_sits_on_both_sides_of_the_tz_pz_plane() -> None:
    core = zone_interface_core(_zones_seg())
    assert core[4, 30, 31] and core[4, 30, 32]
    assert not core[4, 30, 20] and not core[4, 30, 45]


def test_build_case_atlases_keys_and_domain() -> None:
    seg = _zones_seg()
    rng = np.random.default_rng(1)
    image = rng.normal(size=SHAPE).astype(np.float32)
    image[seg > 0] += 3.0
    atlases, roi, strength = build_case_atlases(
        image, seg, SPACING, config=AtlasConfig(min_core_voxels=3)
    )
    assert set(atlases) == {"native_edges", "outer_boundary_gt", "zone_interface_gt"}
    assert strength.shape == SHAPE and roi.dtype == bool
    assert not roi[:, :, :4].any()  # ignore band excluded
    for atlas in atlases.values():
        assert atlas.n_patches > 0
        assert not ((atlas.tubes > 0) & ~roi).any()


def test_wg_case_has_no_zone_interface_and_pred_atlases_are_optional() -> None:
    seg = np.where(_zones_seg() == 2, 1, _zones_seg()).astype(np.int16)
    image = np.zeros(SHAPE, dtype=np.float32)
    image[seg > 0] = 1.0
    atlases, _, _ = build_case_atlases(
        image, seg, SPACING, config=AtlasConfig(min_core_voxels=3)
    )
    assert set(atlases) == {"native_edges", "outer_boundary_gt"}
    pred = np.roll(seg, 2, axis=2)
    with_pred, _, _ = build_case_atlases(
        image, seg, SPACING, config=AtlasConfig(min_core_voxels=3), clean_pred=pred
    )
    assert "outer_boundary_pred" in with_pred
