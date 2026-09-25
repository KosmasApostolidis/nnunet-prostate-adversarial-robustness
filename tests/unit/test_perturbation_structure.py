"""Tests for perturbation structure descriptors."""

from __future__ import annotations

import numpy as np
import pytest
from scipy.ndimage import distance_transform_edt, gaussian_filter

from mri_prostate_seg.experiments.perturbation_structure import (
    BAND_NAMES,
    INTERIOR_BAND_NAMES,
    SIGNED_BAND_NAMES,
    alignment_descriptors,
    case_geometry,
    foreground_bbox_2d,
    largest_valid_rectangle,
    localization_descriptors,
    perturbation_structure,
    radial_power_spectrum,
    robust_percentile_bounds,
    spectral_crops,
    spectral_descriptors,
)


SPACING = (3.0, 0.5, 0.5)


def _foreground(shape: tuple[int, int, int] = (8, 64, 64)) -> np.ndarray:
    fg = np.zeros(shape, dtype=bool)
    fg[2:6, 20:44, 20:44] = True
    return fg


def test_robust_percentile_bounds_are_ordered_and_span_the_range() -> None:
    rng = np.random.default_rng(0)
    image = rng.standard_normal((8, 32, 32))
    lower, upper = robust_percentile_bounds(image)
    assert lower < upper
    assert upper - lower == pytest.approx(
        float(np.percentile(image, 99.5) - np.percentile(image, 0.5))
    )


def test_robust_percentile_bounds_honors_a_mask() -> None:
    image = np.zeros((4, 8, 8))
    image[0] = 100.0  # excluded by the mask below
    image[1:] = np.arange(192).reshape(3, 8, 8) / 192.0  # non-constant masked region
    mask = np.zeros((4, 8, 8), dtype=bool)
    mask[1:] = True
    lower, upper = robust_percentile_bounds(image, mask)
    assert upper < 1.0  # the 100.0 plane must not reach the percentiles


def test_robust_percentile_bounds_rejects_a_mismatched_mask() -> None:
    with pytest.raises(ValueError, match="does not match"):
        robust_percentile_bounds(np.zeros((4, 8, 8)), np.ones((4, 8, 9), dtype=bool))


def test_robust_percentile_bounds_rejects_an_empty_mask() -> None:
    with pytest.raises(ValueError, match="no valid voxels"):
        robust_percentile_bounds(np.zeros((4, 8, 8)), np.zeros((4, 8, 8), dtype=bool))


def test_robust_percentile_bounds_keeps_a_non_degenerate_range_for_uniform_data() -> (
    None
):
    image = np.ones((4, 8, 8)) * 42.0
    lower, upper = robust_percentile_bounds(image)
    assert lower == 42.0
    assert upper == 43.0
    assert upper > lower


def test_foreground_bbox_applies_margin_and_clips_to_slice() -> None:
    mask = np.zeros((64, 64), dtype=bool)
    mask[30:34, 30:34] = True
    rows, cols = foreground_bbox_2d(mask, margin=8)
    assert (rows.start, rows.stop) == (22, 42)
    assert (cols.start, cols.stop) == (22, 42)

    edge = np.zeros((64, 64), dtype=bool)
    edge[0:3, 60:64] = True
    rows, cols = foreground_bbox_2d(edge, margin=8)
    assert (rows.start, rows.stop) == (0, 11)
    assert (cols.start, cols.stop) == (52, 64)

    assert foreground_bbox_2d(np.zeros((64, 64), dtype=bool), margin=8) is None


def test_largest_valid_rectangle_excludes_every_invalid_voxel() -> None:
    ok = np.ones((20, 20), dtype=bool)
    ok[5:8, 5:8] = False  # interior hole a bounding box would swallow
    rows, cols = largest_valid_rectangle(ok)
    assert ok[rows, cols].all()
    assert (rows.stop - rows.start) * (cols.stop - cols.start) >= 20 * 12

    assert largest_valid_rectangle(np.zeros((10, 10), dtype=bool)) is None

    full = np.ones((10, 12), dtype=bool)
    rows, cols = largest_valid_rectangle(full)
    assert (rows.start, rows.stop, cols.start, cols.stop) == (0, 10, 0, 12)


def test_radial_power_spectrum_returns_normalized_increasing_bins() -> None:
    rng = np.random.default_rng(1)
    f_r, power = radial_power_spectrum(rng.standard_normal((64, 64)), bins=32)
    assert f_r.shape == power.shape == (32,)
    assert np.all(np.diff(f_r) > 0)
    assert f_r[0] >= 0.0 and f_r[-1] <= 1.0
    assert np.all(power >= 0.0)


def test_white_noise_is_flatter_and_higher_frequency_than_a_blurred_field() -> None:
    rng = np.random.default_rng(2)
    fg = _foreground()
    valid = np.ones_like(fg, dtype=bool)

    white = rng.standard_normal(fg.shape)
    blurred = gaussian_filter(rng.standard_normal(fg.shape), sigma=(0.0, 3.0, 3.0))

    white_scalars, _, _ = spectral_descriptors(
        white, foreground=fg, valid=valid, min_size=16
    )
    blurred_scalars, _, _ = spectral_descriptors(
        blurred, foreground=fg, valid=valid, min_size=16
    )

    assert white_scalars["spectral_slope"] == pytest.approx(0.0, abs=0.5)
    assert blurred_scalars["spectral_slope"] < white_scalars["spectral_slope"]
    assert (
        blurred_scalars["hf_mean_power_fraction"]
        < white_scalars["hf_mean_power_fraction"]
    )
    assert (
        blurred_scalars["spectral_centroid_norm"]
        < white_scalars["spectral_centroid_norm"]
    )


def test_spectral_descriptors_ignore_a_constant_offset() -> None:
    """A global brightness shift must not move the spectral scalars.

    Without mean removal the offset's entire energy lands in the DC coefficient
    and therefore in radial bin 0, which holds a handful of coefficients and
    whose power is a per-bin mean -- so it dominates both scalars, which sum over
    every bin.  This is not hypothetical for FGSM, whose median ε-saturation is
    0.9988 (WG): its δ is essentially a ±ε sign field, making the untreated
    scalars a direct function of sign imbalance inside each crop rather than of
    spatial organization.
    """
    rng = np.random.default_rng(40)
    fg = _foreground()
    valid = np.ones_like(fg, dtype=bool)
    delta = gaussian_filter(rng.standard_normal(fg.shape), sigma=(0.0, 2.0, 2.0))

    plain, _, plain_power = spectral_descriptors(
        delta, foreground=fg, valid=valid, min_size=16
    )
    shifted, _, shifted_power = spectral_descriptors(
        delta + 0.37, foreground=fg, valid=valid, min_size=16
    )

    for key in ("spectral_centroid_norm", "hf_mean_power_fraction", "spectral_slope"):
        assert shifted[key] == pytest.approx(plain[key], rel=1e-9, abs=1e-12), key
    # The per-bin curve is looser only because subtracting a large offset from
    # the patch changes the FFT's rounding: the bins that disagree hold ~1e-15
    # power, 14 orders below the peak.  The scalars above are the load-bearing
    # assertion and hold to 1e-9.
    assert shifted_power == pytest.approx(plain_power, rel=1e-6, abs=0.0)


def test_a_white_sign_field_reads_as_white_regardless_of_sign_imbalance() -> None:
    """Pin the falsifiable ordering: unstructured ≈ 0.5, structured far below.

    A ±1 sign field with no spatial organization must land near the white-noise
    value of 0.5 on both scalars even when its signs are imbalanced, while the
    sign of a *smooth* field -- which carries real structure -- must sit far
    below.  Absolute values are deliberately loose; the ordering is the claim.
    """
    rng = np.random.default_rng(41)
    fg = _foreground()
    valid = np.ones_like(fg, dtype=bool)

    # 70/30 sign imbalance, but no spatial correlation whatsoever.
    imbalanced = np.where(rng.random(fg.shape) < 0.7, 1.0, -1.0)
    structured = np.sign(
        gaussian_filter(rng.standard_normal(fg.shape), sigma=(0.0, 3.0, 3.0))
    )

    white, _, _ = spectral_descriptors(
        imbalanced, foreground=fg, valid=valid, min_size=16
    )
    smooth, _, _ = spectral_descriptors(
        structured, foreground=fg, valid=valid, min_size=16
    )

    assert white["spectral_centroid_norm"] == pytest.approx(0.5, abs=0.06)
    assert white["hf_mean_power_fraction"] == pytest.approx(0.5, abs=0.06)
    assert smooth["spectral_centroid_norm"] < 0.25
    assert smooth["hf_mean_power_fraction"] < 0.25


def test_spectral_descriptors_exclude_invalid_voxels_inside_the_crop() -> None:
    """An ignore region inside the crop must not reach the FFT.

    On the zones cohort 21.5% of the dilated foreground box is ignore-label, and
    the attack leaves its uniform random start there, so leaking those voxels in
    would measure injected white noise instead of the perturbation.
    """
    rng = np.random.default_rng(5)
    fg = np.zeros((3, 60, 60), dtype=bool)
    fg[1, 10:50, 10:50] = True
    valid = np.ones_like(fg)
    valid[1, 25:35, 25:35] = False  # ignore-label hole inside the foreground box

    delta = np.zeros(fg.shape)
    delta[1] = gaussian_filter(rng.standard_normal((60, 60)), sigma=3.0) * 1e-3
    delta[1, 25:35, 25:35] = rng.standard_normal((10, 10)) * 1e3  # white, huge

    scalars, _, _ = spectral_descriptors(
        delta, foreground=fg, valid=valid, margin=4, min_size=16
    )
    contaminated, _, _ = spectral_descriptors(
        delta, foreground=fg, valid=np.ones_like(fg), margin=4, min_size=16
    )
    assert scalars["hf_mean_power_fraction"] < contaminated["hf_mean_power_fraction"]
    assert scalars["spectral_slope"] < contaminated["spectral_slope"]


def test_spectral_descriptors_count_only_foreground_slices() -> None:
    rng = np.random.default_rng(3)
    fg = _foreground()
    valid = np.ones_like(fg, dtype=bool)
    scalars, f_r, power = spectral_descriptors(
        rng.standard_normal(fg.shape), foreground=fg, valid=valid, min_size=16, bins=32
    )
    assert scalars["spectral_slices_used"] == 4
    assert f_r.shape == power.shape == (32,)


def test_spectral_descriptors_reject_a_volume_with_no_usable_slice() -> None:
    rng = np.random.default_rng(4)
    fg = np.zeros((4, 20, 20), dtype=bool)
    fg[1, 9:11, 9:11] = True
    valid = np.ones_like(fg, dtype=bool)
    with pytest.raises(ValueError, match="no usable"):
        spectral_descriptors(
            rng.standard_normal(fg.shape),
            foreground=fg,
            valid=valid,
            margin=0,
            min_size=32,
        )


def test_uniform_perturbation_has_unit_enrichment_everywhere() -> None:
    rng = np.random.default_rng(10)
    fg = _foreground()
    valid = np.ones_like(fg, dtype=bool)
    delta = rng.standard_normal(fg.shape)

    out = localization_descriptors(delta, foreground=fg, valid=valid, spacing=SPACING)

    assert out["foreground_energy_enrichment"] == pytest.approx(1.0, abs=0.15)
    for name in BAND_NAMES:
        if out[f"volume_fraction_band_{name}"] > 0.02:
            assert out[f"enrichment_band_{name}"] == pytest.approx(1.0, abs=0.25)


def test_energy_confined_to_foreground_gives_inverse_volume_enrichment() -> None:
    rng = np.random.default_rng(11)
    fg = _foreground()
    valid = np.ones_like(fg, dtype=bool)
    delta = np.zeros(fg.shape)
    delta[fg] = rng.standard_normal(int(fg.sum()))

    out = localization_descriptors(delta, foreground=fg, valid=valid, spacing=SPACING)

    assert out["energy_fraction_foreground"] == pytest.approx(1.0)
    assert out["foreground_energy_enrichment"] == pytest.approx(
        1.0 / out["volume_fraction_foreground"]
    )


def test_energy_outside_foreground_gives_zero_enrichment() -> None:
    rng = np.random.default_rng(12)
    fg = _foreground()
    valid = np.ones_like(fg, dtype=bool)
    delta = np.zeros(fg.shape)
    delta[~fg] = rng.standard_normal(int((~fg).sum()))

    out = localization_descriptors(delta, foreground=fg, valid=valid, spacing=SPACING)

    assert out["energy_fraction_foreground"] == pytest.approx(0.0)
    assert out["foreground_energy_enrichment"] == pytest.approx(0.0)


def test_band_fractions_form_a_partition() -> None:
    rng = np.random.default_rng(13)
    fg = _foreground()
    valid = np.ones_like(fg, dtype=bool)
    out = localization_descriptors(
        rng.standard_normal(fg.shape), foreground=fg, valid=valid, spacing=SPACING
    )
    energy = sum(out[f"energy_fraction_band_{name}"] for name in BAND_NAMES)
    volume = sum(out[f"volume_fraction_band_{name}"] for name in BAND_NAMES)
    assert energy == pytest.approx(1.0)
    assert volume == pytest.approx(1.0)


def _deep_foreground() -> np.ndarray:
    """A mask thick enough to hold a >10 mm-deep interior band.

    ``_foreground`` spans four 3 mm slices, so nothing in it is more than 6 mm
    from the boundary and the deep bands would be empty by construction.
    """

    fg = np.zeros((14, 64, 64), dtype=bool)
    fg[2:12, 8:56, 8:56] = True
    return fg


def test_interior_bands_partition_the_pooled_inside_band() -> None:
    rng = np.random.default_rng(21)
    fg = _deep_foreground()
    valid = np.ones_like(fg, dtype=bool)
    out = localization_descriptors(
        rng.standard_normal(fg.shape), foreground=fg, valid=valid, spacing=SPACING
    )
    for prefix in ("energy_fraction", "volume_fraction"):
        pooled = out[f"{prefix}_band_inside"]
        split = sum(out[f"{prefix}_band_{name}"] for name in INTERIOR_BAND_NAMES)
        assert split == pytest.approx(pooled)


def test_signed_bands_form_a_partition() -> None:
    rng = np.random.default_rng(22)
    fg = _deep_foreground()
    valid = np.ones_like(fg, dtype=bool)
    out = localization_descriptors(
        rng.standard_normal(fg.shape), foreground=fg, valid=valid, spacing=SPACING
    )
    for prefix in ("energy_fraction", "volume_fraction"):
        total = sum(out[f"{prefix}_band_{name}"] for name in SIGNED_BAND_NAMES)
        assert total == pytest.approx(1.0)


def test_uniform_perturbation_has_unit_enrichment_in_every_signed_band() -> None:
    rng = np.random.default_rng(23)
    fg = _deep_foreground()
    valid = np.ones_like(fg, dtype=bool)
    out = localization_descriptors(
        rng.standard_normal(fg.shape), foreground=fg, valid=valid, spacing=SPACING
    )
    for name in SIGNED_BAND_NAMES:
        if out[f"volume_fraction_band_{name}"] > 0.02:
            assert out[f"enrichment_band_{name}"] == pytest.approx(1.0, abs=0.25)


def test_interior_bands_separate_boundary_hugging_from_deep_targeting() -> None:
    """The discrimination the pooled ``inside`` band cannot make.

    Both fields put all of their energy inside the mask, so the pooled band
    reports the same enrichment for each; only the signed profile says whether
    that energy hugs the inner face of the boundary or sits deep in the gland.
    """

    fg = _deep_foreground()
    valid = np.ones_like(fg, dtype=bool)
    depth = distance_transform_edt(fg, sampling=SPACING)

    hugging = np.zeros(fg.shape)
    hugging[fg & (depth < 2.0)] = 1.0
    deep = np.zeros(fg.shape)
    deep[depth >= 10.0] = 1.0

    hugging_out = localization_descriptors(
        hugging, foreground=fg, valid=valid, spacing=SPACING
    )
    deep_out = localization_descriptors(
        deep, foreground=fg, valid=valid, spacing=SPACING
    )

    assert hugging_out["enrichment_band_inside"] == pytest.approx(
        deep_out["enrichment_band_inside"]
    )
    assert hugging_out["energy_fraction_band_inside_0_2mm"] == pytest.approx(1.0)
    assert hugging_out["energy_fraction_band_inside_beyond_10mm"] == pytest.approx(0.0)
    assert deep_out["energy_fraction_band_inside_beyond_10mm"] == pytest.approx(1.0)
    assert deep_out["energy_fraction_band_inside_0_2mm"] == pytest.approx(0.0)


@pytest.mark.parametrize(
    "column, expected_band",
    [
        (35, "0_2mm"),  # 1.5 mm — just inside
        (36, "2_5mm"),  # exactly 2.0 mm — the cutoff belongs to the upper band
        (41, "2_5mm"),  # 4.5 mm — just inside
        (42, "5_10mm"),  # exactly 5.0 mm
        (51, "5_10mm"),  # 9.5 mm — just inside
        (52, "beyond_10mm"),  # exactly 10.0 mm
    ],
)
def test_band_edges_are_half_open_lower_inclusive(
    column: int, expected_band: str
) -> None:
    """Pin which side of each cutoff a boundary voxel falls on.

    Without this, a regression that flipped every band's inclusive side
    (``<=``/``>`` for ``<``/``>=``) would keep the partition gap-free and
    overlap-free, so the partition and millimetre tests would both still pass.
    Bands are half-open ``[low, high)``: an exactly-2.0 mm voxel is ``2_5mm``.
    """
    fg = np.zeros((5, 64, 64), dtype=bool)
    fg[2, 32, 32] = True
    valid = np.ones_like(fg)
    delta = np.zeros(fg.shape)
    delta[2, 32, column] = 1.0

    out = localization_descriptors(delta, foreground=fg, valid=valid, spacing=SPACING)
    assert out[f"energy_fraction_band_{expected_band}"] == pytest.approx(1.0)
    for name in BAND_NAMES:
        if name != expected_band:
            assert out[f"energy_fraction_band_{name}"] == pytest.approx(0.0)


def test_bands_are_measured_in_millimetres_not_voxels() -> None:
    """One step in z is 3 mm, one step in-plane is 0.5 mm."""
    fg = np.zeros((5, 32, 32), dtype=bool)
    fg[2, 16, 16] = True
    valid = np.ones_like(fg, dtype=bool)

    delta = np.zeros(fg.shape)
    delta[1, 16, 16] = 1.0  # 3.0 mm away -> band 2-5 mm
    out_z = localization_descriptors(delta, foreground=fg, valid=valid, spacing=SPACING)
    assert out_z["energy_fraction_band_2_5mm"] == pytest.approx(1.0)

    delta = np.zeros(fg.shape)
    delta[2, 16, 18] = 1.0  # 1.0 mm away -> band 0-2 mm
    out_xy = localization_descriptors(
        delta, foreground=fg, valid=valid, spacing=SPACING
    )
    assert out_xy["energy_fraction_band_0_2mm"] == pytest.approx(1.0)


def test_perturbation_proportional_to_gradient_is_perfectly_rank_aligned() -> None:
    rng = np.random.default_rng(20)
    clean = gaussian_filter(rng.standard_normal((8, 48, 48)), sigma=(0.0, 2.0, 2.0))
    valid = np.ones(clean.shape, dtype=bool)

    gy, gx = np.gradient(clean, SPACING[1], SPACING[2], axis=(1, 2))
    gz = np.gradient(clean, SPACING[0], axis=0)
    gradient = np.sqrt(gz**2 + gy**2 + gx**2)

    out = alignment_descriptors(
        gradient, clean, valid=valid, spacing=SPACING, seed=1, max_voxels=50_000
    )
    assert out["spearman_absdelta_gradmag"] == pytest.approx(1.0, abs=1e-6)
    assert out["edge_energy_enrichment"] > 1.0


def test_independent_perturbation_is_rank_uncorrelated_with_structure() -> None:
    rng = np.random.default_rng(21)
    clean = gaussian_filter(rng.standard_normal((8, 48, 48)), sigma=(0.0, 2.0, 2.0))
    valid = np.ones(clean.shape, dtype=bool)
    delta = rng.standard_normal(clean.shape)

    out = alignment_descriptors(
        delta, clean, valid=valid, spacing=SPACING, seed=2, max_voxels=50_000
    )
    assert out["spearman_absdelta_gradmag"] == pytest.approx(0.0, abs=0.05)
    assert out["edge_energy_enrichment"] == pytest.approx(1.0, abs=0.2)


def test_energy_on_top_decile_edges_enriches_by_ten() -> None:
    rng = np.random.default_rng(22)
    clean = gaussian_filter(rng.standard_normal((8, 48, 48)), sigma=(0.0, 2.0, 2.0))
    valid = np.ones(clean.shape, dtype=bool)

    gy, gx = np.gradient(clean, SPACING[1], SPACING[2], axis=(1, 2))
    gz = np.gradient(clean, SPACING[0], axis=0)
    gradient = np.sqrt(gz**2 + gy**2 + gx**2)
    threshold = np.quantile(gradient, 0.9)

    delta = np.zeros(clean.shape)
    top = gradient >= threshold
    delta[top] = 1.0

    out = alignment_descriptors(
        delta, clean, valid=valid, spacing=SPACING, seed=3, max_voxels=50_000
    )
    assert out["edge_energy_enrichment"] == pytest.approx(10.0, rel=0.05)


def test_alignment_reports_the_voxel_count_it_actually_used() -> None:
    """Pin ``alignment_voxels_used`` on both sides of the subsample threshold.

    ``count = min(max_voxels, magnitude.size)`` is computed before the branch,
    so an off-by-one in ``count < magnitude.size`` cannot change ``count`` --
    that branch only swaps in an equal-length permutation. What each assertion
    actually guards: the capped case (``max_voxels=1_000``) catches always
    reporting the population size instead of the cap; the uncapped case
    (``max_voxels=population * 10``) catches hard-coding ``max_voxels`` instead
    of the true count.
    """
    rng = np.random.default_rng(24)
    clean = gaussian_filter(rng.standard_normal((6, 40, 40)), sigma=(0.0, 2.0, 2.0))
    valid = np.ones(clean.shape, dtype=bool)
    delta = rng.standard_normal(clean.shape)
    population = int(valid.sum())  # 9600

    capped = alignment_descriptors(
        delta, clean, valid=valid, spacing=SPACING, seed=1, max_voxels=1_000
    )
    assert capped["alignment_voxels_used"] == 1_000

    uncapped = alignment_descriptors(
        delta, clean, valid=valid, spacing=SPACING, seed=1, max_voxels=population * 10
    )
    assert uncapped["alignment_voxels_used"] == population


def test_alignment_rejects_mismatched_shapes() -> None:
    rng = np.random.default_rng(25)
    clean = rng.standard_normal((6, 40, 40))
    with pytest.raises(ValueError, match="equal-shape"):
        alignment_descriptors(
            rng.standard_normal((6, 40, 41)),
            clean,
            valid=np.ones(clean.shape, dtype=bool),
            spacing=SPACING,
            seed=1,
        )


def test_alignment_rejects_an_empty_valid_mask() -> None:
    rng = np.random.default_rng(26)
    clean = rng.standard_normal((6, 40, 40))
    with pytest.raises(ValueError, match="no voxels"):
        alignment_descriptors(
            rng.standard_normal(clean.shape),
            clean,
            valid=np.zeros(clean.shape, dtype=bool),
            spacing=SPACING,
            seed=1,
        )


def test_alignment_rejects_a_zero_energy_perturbation() -> None:
    rng = np.random.default_rng(27)
    clean = rng.standard_normal((6, 40, 40))
    with pytest.raises(ValueError, match="no energy"):
        alignment_descriptors(
            np.zeros(clean.shape),
            clean,
            valid=np.ones(clean.shape, dtype=bool),
            spacing=SPACING,
            seed=1,
        )


def test_alignment_is_deterministic_for_a_fixed_seed() -> None:
    rng = np.random.default_rng(23)
    clean = gaussian_filter(rng.standard_normal((8, 64, 64)), sigma=(0.0, 2.0, 2.0))
    valid = np.ones(clean.shape, dtype=bool)
    delta = rng.standard_normal(clean.shape)

    first = alignment_descriptors(
        delta, clean, valid=valid, spacing=SPACING, seed=7, max_voxels=1_000
    )
    second = alignment_descriptors(
        delta, clean, valid=valid, spacing=SPACING, seed=7, max_voxels=1_000
    )
    other = alignment_descriptors(
        delta, clean, valid=valid, spacing=SPACING, seed=8, max_voxels=1_000
    )
    assert first == second
    assert first["spearman_absdelta_gradmag"] != other["spearman_absdelta_gradmag"]


def _case(seed: int = 30) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    rng = np.random.default_rng(seed)
    clean = gaussian_filter(rng.standard_normal((8, 64, 64)), sigma=(0.0, 2.0, 2.0))
    segmentation = np.zeros((8, 64, 64), dtype=np.int16)
    segmentation[2:6, 20:44, 20:44] = 1
    valid = np.ones(clean.shape, dtype=bool)
    return clean, segmentation, valid


def test_aggregator_returns_every_family() -> None:
    clean, segmentation, valid = _case()
    rng = np.random.default_rng(31)
    altered = clean + 0.01 * rng.standard_normal(clean.shape)

    scalars, f_r, power = perturbation_structure(
        clean,
        altered,
        segmentation,
        valid_mask=valid,
        spacing=SPACING,
        seed=5,
        min_size=16,
    )

    for key in (
        "rms_norm",
        "spectral_centroid_norm",
        "hf_mean_power_fraction",
        "spectral_slope",
        "energy_fraction_foreground",
        "foreground_energy_enrichment",
        "enrichment_band_0_2mm",
        "spearman_absdelta_gradmag",
        "edge_energy_enrichment",
        "gmsd_axial_prostate",
    ):
        assert key in scalars, key
    assert f_r.shape == power.shape == (32,)
    assert scalars["rms_norm"] == pytest.approx(
        float(np.sqrt(np.mean((altered - clean)[valid] ** 2)))
    )


def test_precomputed_geometry_reproduces_the_recomputed_descriptors() -> None:
    """The whole point of ``case_geometry`` is that it changes no number.

    It only skips work that depends on the clean image and segmentation alone,
    so every scalar and the radial spectrum must match the recomputed path
    exactly -- not approximately, since the same arithmetic runs either way.
    """

    clean, segmentation, valid = _case()
    rng = np.random.default_rng(31)
    altered = clean + 0.01 * rng.standard_normal(clean.shape)
    geometry = case_geometry(
        clean,
        segmentation,
        valid_mask=valid,
        spacing=SPACING,
        min_size=16,
    )

    expected, expected_f_r, expected_power = perturbation_structure(
        clean,
        altered,
        segmentation,
        valid_mask=valid,
        spacing=SPACING,
        seed=5,
        min_size=16,
    )
    observed, observed_f_r, observed_power = perturbation_structure(
        clean,
        altered,
        segmentation,
        valid_mask=valid,
        spacing=SPACING,
        seed=5,
        min_size=16,
        geometry=geometry,
    )

    # assert_equal rather than ==: an empty band yields NaN enrichment, and
    # NaN != NaN would fail here for a descriptor that reproduced exactly.
    np.testing.assert_equal(observed, expected)
    np.testing.assert_array_equal(observed_f_r, expected_f_r)
    np.testing.assert_array_equal(observed_power, expected_power)
    assert geometry.crops == spectral_crops(
        segmentation > 0, valid, margin=16, min_size=16
    )


def test_geometry_built_for_another_case_is_rejected() -> None:
    """A geometry silently applied to the wrong case would corrupt a whole run."""

    clean, segmentation, valid = _case()
    geometry = case_geometry(
        clean[:, :32, :32],
        segmentation[:, :32, :32],
        valid_mask=valid[:, :32, :32],
        spacing=SPACING,
        min_size=16,
    )

    with pytest.raises(ValueError, match="geometry was built for shape"):
        perturbation_structure(
            clean,
            clean + 0.01,
            segmentation,
            valid_mask=valid,
            spacing=SPACING,
            seed=5,
            min_size=16,
            geometry=geometry,
        )


def test_geometry_built_with_other_crop_settings_is_rejected() -> None:
    """``crops`` bakes in ``margin`` and ``min_size``; a mismatch must not pass."""

    clean, segmentation, valid = _case()
    geometry = case_geometry(
        clean, segmentation, valid_mask=valid, spacing=SPACING, margin=8, min_size=16
    )

    with pytest.raises(ValueError, match="margin"):
        perturbation_structure(
            clean,
            clean + 0.01,
            segmentation,
            valid_mask=valid,
            spacing=SPACING,
            seed=5,
            margin=16,
            min_size=16,
            geometry=geometry,
        )


def test_aggregator_rejects_a_mismatched_valid_mask() -> None:
    """A wrong-shaped mask must raise a domain error, not a numpy IndexError.

    ``delta[ok]`` is indexed before any sub-family validates ``ok``, so without
    this guard the failure surfaces as ``IndexError: boolean index did not
    match``. This module's convention is a clear ``ValueError``.
    """
    clean, segmentation, _valid = _case()
    rng = np.random.default_rng(32)
    altered = clean + 0.01 * rng.standard_normal(clean.shape)
    wrong = np.ones((clean.shape[0], clean.shape[1], clean.shape[2] + 1), dtype=bool)

    with pytest.raises(ValueError, match="equal-shape"):
        perturbation_structure(
            clean,
            altered,
            segmentation,
            valid_mask=wrong,
            spacing=SPACING,
            seed=5,
            min_size=16,
        )


def test_aggregator_rejects_mismatched_volumes() -> None:
    clean, segmentation, valid = _case()
    with pytest.raises(ValueError, match="equal-shape"):
        perturbation_structure(
            clean,
            clean[:, :, :-1],
            segmentation,
            valid_mask=valid,
            spacing=SPACING,
            seed=5,
            min_size=16,
        )


def test_aggregator_rejects_an_unperturbed_pair() -> None:
    clean, segmentation, valid = _case()
    with pytest.raises(ValueError, match="no energy"):
        perturbation_structure(
            clean,
            clean,
            segmentation,
            valid_mask=valid,
            spacing=SPACING,
            seed=5,
            min_size=16,
        )


def test_public_signed_helpers_are_the_private_ones() -> None:
    from mri_prostate_seg.experiments import perturbation_structure as ps

    fg = _foreground()
    distance = ps.signed_distance_mm(fg, SPACING)
    assert distance is not None
    assert np.array_equal(distance, ps._signed_distance_mm(fg, SPACING))
    bands = ps.signed_band_masks(distance)
    assert set(bands) == set(ps._band_masks(distance))
    assert set(SIGNED_BAND_NAMES).issubset(bands)
    # The eight signed bands partition the volume.
    total = sum(bands[name].astype(int) for name in SIGNED_BAND_NAMES)
    assert np.array_equal(total, np.ones_like(total))
