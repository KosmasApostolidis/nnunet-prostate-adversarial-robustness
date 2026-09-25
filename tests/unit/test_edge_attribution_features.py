"""Static feature and local-direction phantoms (plan §15, §27)."""

from __future__ import annotations

import math

import numpy as np
import pytest

from mri_prostate_seg.experiments.edge_attribution.features import (
    local_damage_direction,
    patch_static_features,
)
from mri_prostate_seg.experiments.edge_attribution.tubes import EdgeAtlas
from mri_prostate_seg.experiments.perturbation_structure import signed_distance_mm

SPACING = (3.0, 0.5, 0.5)
SHAPE = (10, 64, 64)


def _ball(radius_mm: float, center=(15.0, 16.0, 16.0)) -> np.ndarray:
    grids = np.meshgrid(
        *[np.arange(n) * s for n, s in zip(SHAPE, SPACING)], indexing="ij"
    )
    return sum((g - c) ** 2 for g, c in zip(grids, center)) <= radius_mm**2


def _atlas_on_surface(fg: np.ndarray) -> EdgeAtlas:
    # Ball centre x = 16 mm, radius 10 mm: boundary at x = 26 mm (voxel 52) and 6 mm (voxel 12).
    labels = np.zeros(SHAPE, dtype=np.int32)
    labels[5, 32, 50:54] = 1  # right boundary at z = 15 mm
    labels[5, 32, 10:14] = 2  # left boundary
    return EdgeAtlas("t", labels, labels)


def test_static_features_geometry_and_category() -> None:
    fg = _ball(10.0)
    atlas = _atlas_on_surface(fg)
    sd = signed_distance_mm(fg, SPACING)
    strength = np.full(SHAPE, 0.5)
    strength[atlas.tubes == 1] = 2.0
    rows = patch_static_features(
        atlas,
        spacing=SPACING,
        signed_distance=sd,
        strength=strength,
        probs=None,
        other_cores={"outer_boundary_gt": fg & ~np.roll(fg, 1, axis=2)},
    )
    assert [r["patch_id"] for r in rows] == [1, 2]
    r = rows[0]
    assert r["volume_mm3"] == pytest.approx(4 * 0.75)
    assert r["centroid_z_mm"] == pytest.approx(15.0)
    assert r["edge_strength_mean"] == pytest.approx(2.0)
    assert r["anatomical_category"] in ("outer_boundary", "interior")
    assert 0.0 <= r["overlap_outer_boundary_gt"] <= 1.0
    assert math.isnan(r["clean_entropy_mean"])


def test_static_features_entropy_and_margin_from_probs() -> None:
    fg = _ball(10.0)
    atlas = _atlas_on_surface(fg)
    probs = np.zeros((2, *SHAPE))
    probs[0] = 0.5
    probs[1] = 0.5
    rows = patch_static_features(
        atlas,
        spacing=SPACING,
        signed_distance=signed_distance_mm(fg, SPACING),
        strength=np.ones(SHAPE),
        probs=probs,
        other_cores={},
    )
    assert rows[0]["clean_entropy_mean"] == pytest.approx(math.log(2))
    assert rows[0]["clean_margin_mean"] == pytest.approx(0.0)


def test_local_direction_detects_outward_growth_near_one_patch() -> None:
    gt = _ball(10.0)
    clean = gt.copy()
    adv = clean.copy()
    adv[4:7, 24:41, 46:60] = (
        True  # bulge overlapping the ball and protruding right, across patch 1's whole neighbourhood
    )
    atlas = _atlas_on_surface(gt)
    rows = local_damage_direction(
        atlas,
        spacing=SPACING,
        gt_fg=gt,
        clean_fg=clean,
        adv_fg=adv,
        valid=np.ones(SHAPE, bool),
        radius_mm=4.0,
    )
    right, left = rows[0], rows[1]
    assert right["local_induced_fp_mm3"] > 0 and left["local_induced_fp_mm3"] == 0
    assert right["local_signed_displacement_median_mm"] > 0
    assert abs(left["local_signed_displacement_median_mm"]) < 0.3
    assert right["local_induced_fn_mm3"] == 0


def test_local_direction_zone_transitions() -> None:
    gt_seg = np.where(_ball(10.0), 1, 0).astype(np.int16)
    gt_seg[:, :, 32:] = np.where(gt_seg[:, :, 32:] == 1, 2, 0)
    clean_seg = gt_seg.copy()
    adv_seg = clean_seg.copy()
    adv_seg[5, 28:37, 46:52] = (
        1  # PZ voxels just inside the right boundary relabelled TZ, near patch 1
    )
    atlas = _atlas_on_surface(gt_seg > 0)
    rows = local_damage_direction(
        atlas,
        spacing=SPACING,
        gt_fg=gt_seg > 0,
        clean_fg=clean_seg > 0,
        adv_fg=adv_seg > 0,
        valid=np.ones(SHAPE, bool),
        seg_gt=gt_seg,
        seg_clean=clean_seg,
        seg_adv=adv_seg,
    )
    assert rows[0]["pz_to_tz_mm3"] > 0 and rows[0]["tz_to_pz_mm3"] == 0
    assert rows[1]["pz_to_tz_mm3"] == 0


def test_local_direction_reports_nan_not_zero_when_displacement_is_unmeasurable() -> (
    None
):
    gt = _ball(10.0)
    atlas = _atlas_on_surface(gt)
    empty = np.zeros(SHAPE, dtype=bool)
    # No clean surface at all: every displacement column is unmeasurable.
    rows = local_damage_direction(
        atlas,
        spacing=SPACING,
        gt_fg=gt,
        clean_fg=empty,
        adv_fg=gt,
        valid=np.ones(SHAPE, bool),
        radius_mm=4.0,
    )
    for row in rows:
        assert math.isnan(row["local_signed_displacement_median_mm"])
        assert math.isnan(row["local_outward_displacement_p90_mm"])
        assert math.isnan(row["local_inward_displacement_p10_mm"])
        assert row["local_surface_vertices"] == 0
    # A clean surface exists but this patch's neighbourhood is far from it:
    # still NaN, and distinguishable from a genuine zero displacement.
    far = np.zeros(SHAPE, dtype=np.int32)
    far[1, 4, 4:8] = 1
    lone = EdgeAtlas("far", far, far)
    away = local_damage_direction(
        lone,
        spacing=SPACING,
        gt_fg=gt,
        clean_fg=gt,
        adv_fg=gt,
        valid=np.ones(SHAPE, bool),
        radius_mm=1.0,
    )
    assert math.isnan(away[0]["local_signed_displacement_median_mm"])
    assert away[0]["local_surface_vertices"] == 0
