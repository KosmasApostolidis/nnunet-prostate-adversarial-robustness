"""Surface-displacement phantoms (spec §10, §11, §43)."""

from __future__ import annotations

import numpy as np
import pytest

from mri_prostate_seg.experiments.segmentation_direction.surface import (
    extract_surface_mesh,
    iso_signed_distance_mm,
    sample_field_at_vertices,
    surface_motion_summary,
    weighted_quantile,
)

SPACING = (3.0, 0.5, 0.5)
SHAPE = (16, 80, 80)
CENTER = (24.0, 20.0, 20.0)


def _ball(radius_mm: float, center: tuple[float, float, float] = CENTER) -> np.ndarray:
    grids = np.meshgrid(
        *[np.arange(n) * s for n, s in zip(SHAPE, SPACING)], indexing="ij"
    )
    r2 = sum((g - c) ** 2 for g, c in zip(grids, center))
    return r2 <= radius_mm**2


def test_mesh_area_of_a_sphere_is_close_to_4_pi_r2() -> None:
    mesh = extract_surface_mesh(_ball(12.0), SPACING)
    area = float(mesh.vertex_area_mm2.sum())
    # Measured 19% over on 3 mm slices (staircase caps); a spacing bug is x6.
    assert area == pytest.approx(4 * np.pi * 12.0**2, rel=0.25)
    assert mesh.vertices_mm.shape[1] == 3
    assert np.all(mesh.vertices_mm >= 0)


def test_mesh_closes_when_the_mask_touches_the_array_edge() -> None:
    mask = np.zeros(SHAPE, dtype=bool)
    mask[0:4, 10:30, 10:30] = True  # touches z = 0
    mesh = extract_surface_mesh(mask, SPACING)
    expected = 2 * (10 * 10) + 4 * (12 * 10)  # box 12 x 10 x 10 mm
    assert float(mesh.vertex_area_mm2.sum()) == pytest.approx(expected, rel=0.1)


def test_weighted_quantile_matches_unweighted_for_equal_weights() -> None:
    values = np.arange(1.0, 101.0)
    q = weighted_quantile(values, np.ones_like(values), np.asarray([0.5]))
    assert q[0] == pytest.approx(50.5, abs=0.5)
    assert np.isnan(weighted_quantile(np.array([]), np.array([]), np.asarray([0.5]))[0])


def test_flat_slab_half_step_correction_in_plane_and_through_plane() -> None:
    # A flat slab isolates the half-voxel correction from marching-cubes
    # curvature noise: 0.25 mm in-plane (0.5 mm spacing), 1.5 mm through-plane
    # (3.0 mm spacing), on both sides of the boundary.
    mask = np.zeros(SHAPE, dtype=bool)
    mask[4:9, 20:60, 20:60] = True
    phi = iso_signed_distance_mm(mask, SPACING)
    assert phi[6, 20, 40] == pytest.approx(-0.25, abs=1e-6)
    assert phi[6, 19, 40] == pytest.approx(0.25, abs=1e-6)
    assert phi[4, 40, 40] == pytest.approx(-1.5, abs=1e-6)
    assert phi[3, 40, 40] == pytest.approx(1.5, abs=1e-6)


def test_uniform_expansion_moves_outward_by_two_mm() -> None:
    gt = _ball(10.0)
    adv = _ball(12.0)
    s = surface_motion_summary(gt, gt, adv, spacing=SPACING, tolerance_mm=0.5)
    assert s["surface_metric_valid"] == 1.0
    assert s["median_surface_motion_mm"] == pytest.approx(2.0, abs=0.1)
    assert s["outward_surface_fraction"] > 0.9
    assert s["inward_surface_fraction"] < 0.05
    assert s["worsened_outward_surface_fraction"] > 0.9
    assert s["corrected_surface_fraction"] < 0.05


def test_uniform_contraction_moves_inward_by_two_mm() -> None:
    gt = _ball(10.0)
    adv = _ball(8.0)
    s = surface_motion_summary(gt, gt, adv, spacing=SPACING, tolerance_mm=0.5)
    assert s["median_surface_motion_mm"] == pytest.approx(-2.0, abs=0.5)
    assert s["inward_surface_fraction"] > 0.9
    assert s["worsened_inward_surface_fraction"] > 0.9


def test_translation_has_balanced_outward_and_inward_sides() -> None:
    gt = _ball(10.0)
    adv = np.roll(gt, 4, axis=2)  # +2 mm along x
    s = surface_motion_summary(gt, gt, adv, spacing=SPACING, tolerance_mm=0.5)
    assert abs(s["median_surface_motion_mm"]) < 0.5
    # Measured 0.31 vs 0.40: the staircase caps are not symmetric under a shift.
    assert s["outward_surface_fraction"] == pytest.approx(
        s["inward_surface_fraction"], abs=0.15
    )
    assert s["outward_surface_fraction"] > 0.25
    assert s["rms_surface_motion_mm"] > 0.8


def test_a_correcting_attack_is_counted_as_corrected() -> None:
    gt = _ball(10.0)
    clean = _ball(12.0)  # clean over-segments by 2 mm
    adv = gt.copy()  # attack lands on the truth
    s = surface_motion_summary(gt, clean, adv, spacing=SPACING, tolerance_mm=0.5)
    assert s["corrected_surface_fraction"] > 0.9
    assert s["worsened_outward_surface_fraction"] < 0.05
    assert s["median_surface_motion_mm"] == pytest.approx(-2.0, abs=0.5)


def test_identical_masks_are_stable() -> None:
    gt = _ball(10.0)
    s = surface_motion_summary(gt, gt, gt, spacing=SPACING, tolerance_mm=0.5)
    assert s["median_surface_motion_mm"] == pytest.approx(0.0, abs=1e-6)
    assert s["stable_surface_fraction"] == pytest.approx(1.0)


def test_empty_adversarial_mask_is_flagged_not_crashed() -> None:
    gt = _ball(10.0)
    s = surface_motion_summary(
        gt, gt, np.zeros(SHAPE, bool), spacing=SPACING, tolerance_mm=0.5
    )
    assert s["surface_metric_valid"] == 0.0
    assert s["surface_failure_reason"] == "empty_adv"
    assert np.isnan(s["median_surface_motion_mm"])
    with pytest.raises(ValueError):
        extract_surface_mesh(np.zeros(SHAPE, bool), SPACING)
    with pytest.raises(ValueError):
        extract_surface_mesh(np.ones(SHAPE, bool), SPACING)


def test_isotropic_and_anisotropic_spacing_agree_in_mm() -> None:
    iso_shape = (40, 40, 40)
    grids = np.meshgrid(*[np.arange(n) * 1.0 for n in iso_shape], indexing="ij")
    r2 = sum((g - 20.0) ** 2 for g in grids)
    gt_iso, adv_iso = r2 <= 10.0**2, r2 <= 12.0**2
    iso = surface_motion_summary(
        gt_iso, gt_iso, adv_iso, spacing=(1.0, 1.0, 1.0), tolerance_mm=0.5
    )
    aniso = surface_motion_summary(
        _ball(10.0), _ball(10.0), _ball(12.0), spacing=SPACING, tolerance_mm=0.5
    )
    assert iso["median_surface_motion_mm"] == pytest.approx(
        aniso["median_surface_motion_mm"], abs=0.5
    )


def test_sample_field_reads_back_a_linear_ramp() -> None:
    field = np.arange(np.prod(SHAPE), dtype=float).reshape(SHAPE)
    verts = np.asarray([[3.0, 0.5, 1.0]])  # index (1, 1, 2)
    assert sample_field_at_vertices(field, verts, SPACING)[0] == pytest.approx(
        field[1, 1, 2]
    )
