"""Descriptors of how a perturbation is spatially organized.

These complement PSNR and SSIM, which measure *how much* an image changed but
say nothing about *how* the change is arranged.  Three families are provided:

* spectral -- in-plane radial power spectrum of the perturbation,
* localization -- where the perturbation energy sits relative to anatomy,
* alignment -- whether the perturbation follows image structure.

Everything is descriptive.  RMS-matched Gaussian noise is white and uniform over
the valid mask *by construction*, so its flat spectrum and unit enrichment are
analytic references rather than findings; the informative quantity is always the
adversarial value measured against them.

Spectral work is done in-plane rather than in 3-D because voxels are
(3.0, 0.5, 0.5) mm with only 22-24 slices in z: a radially averaged 3-D spectrum
would average frequencies six times apart.  This is the same reasoning
``epsilon_calibration.axial_ssim_mean`` applies to SSIM.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy.ndimage import distance_transform_edt
from scipy.stats import spearmanr

from mri_prostate_seg.experiments.gmsd import axial_gmsd_mean


def robust_percentile_bounds(
    image: np.ndarray,
    mask: np.ndarray | None = None,
    *,
    lower_percentile: float = 0.5,
    upper_percentile: float = 99.5,
) -> tuple[float, float]:
    """Robust intensity endpoints, the pair behind ``robust_data_range``."""

    arr = np.asarray(image, dtype=np.float64)
    if mask is None:
        values = arr
    else:
        selector = np.asarray(mask, dtype=bool)
        # Mirror epsilon_calibration._as_mask: a shape mismatch must raise a
        # clear domain error, not an opaque numpy IndexError.
        if selector.shape != arr.shape:
            raise ValueError(
                f"mask shape {selector.shape} does not match image shape {arr.shape}"
            )
        values = arr[selector]
    if values.size == 0:
        raise ValueError("mask contains no valid voxels")
    lower, upper = np.percentile(values, [lower_percentile, upper_percentile])
    if not np.isfinite(upper - lower) or upper <= lower:
        lower, upper = float(values.min()), float(values.max())
    # Keep the range non-degenerate.  Task 5 passes these bounds to
    # ``axial_gmsd_mean``, which raises on ``upper - lower <= 0``, so a
    # zero-width interval on genuinely constant data would abort the case.
    if upper <= lower:
        upper = lower + 1.0
    return float(lower), float(upper)


def foreground_bbox_2d(
    mask_slice: np.ndarray, *, margin: int
) -> tuple[slice, slice] | None:
    """Bounding box of a 2-D mask, dilated by ``margin`` and clipped."""

    mask = np.asarray(mask_slice, dtype=bool)
    if mask.ndim != 2:
        raise ValueError("foreground_bbox_2d expects a 2-D mask")
    rows = np.flatnonzero(mask.any(axis=1))
    cols = np.flatnonzero(mask.any(axis=0))
    if not len(rows) or not len(cols):
        return None
    row_start = max(int(rows[0]) - margin, 0)
    row_stop = min(int(rows[-1]) + 1 + margin, mask.shape[0])
    col_start = max(int(cols[0]) - margin, 0)
    col_stop = min(int(cols[-1]) + 1 + margin, mask.shape[1])
    return slice(row_start, row_stop), slice(col_start, col_stop)


def largest_valid_rectangle(valid_slice: np.ndarray) -> tuple[slice, slice] | None:
    """Largest all-valid axis-aligned rectangle, by the maximal-rectangle method.

    Runs in O(H*W): sweep rows maintaining per-column runs of consecutive valid
    voxels, and for each row solve largest-rectangle-in-histogram with a stack.
    The result contains no invalid voxel at all, so the FFT never sees one.

    The rectangle depends only on the segmentation, which is identical across
    attacks and controls for a given case, so every condition is measured on the
    same crop and paired comparisons stay valid.
    """

    ok = np.asarray(valid_slice, dtype=bool)
    if ok.ndim != 2:
        raise ValueError("largest_valid_rectangle expects a 2-D mask")
    if not ok.any():
        return None

    height, width = ok.shape
    runs = np.zeros(width, dtype=int)
    best_area = 0
    best = (0, 0, 0, 0)
    for row in range(height):
        runs = np.where(ok[row], runs + 1, 0)
        stack: list[tuple[int, int]] = []
        for col in range(width + 1):
            current = int(runs[col]) if col < width else 0
            start = col
            while stack and stack[-1][1] >= current:
                left, tall = stack.pop()
                area = tall * (col - left)
                if area > best_area:
                    best_area = area
                    best = (row - tall + 1, row + 1, left, col)
                start = left
            stack.append((start, current))

    if best_area <= 0:
        return None
    row_start, row_stop, col_start, col_stop = best
    return slice(row_start, row_stop), slice(col_start, col_stop)


def radial_power_spectrum(
    field: np.ndarray, *, bins: int = 32
) -> tuple[np.ndarray, np.ndarray]:
    """Radially binned 2-D power spectrum on a Nyquist-normalized axis.

    Frequencies are normalized per axis before the radial coordinate is formed,
    so spectra from differently sized crops (WG 256x256 vs zones ~95-141) share
    one comparable axis.  A separable Hann window suppresses edge leakage from
    the crop boundary.

    The crop mean is removed first.  A constant offset over the crop is a global
    brightness shift that carries no information about *spatial organization*,
    yet its whole energy lands in the DC coefficient and therefore in radial bin
    0 -- a bin holding only 5-97 coefficients out of 4k-65k, whose ``power`` is a
    per-bin *mean*.  Left in, it dominates ``spectral_centroid_norm`` and
    ``hf_mean_power_fraction``: a structureless white sign field with a 60/40 sign
    imbalance reads as centroid 0.26 instead of 0.50.  Genuine structure is
    untouched (a smooth field's centroid moves 0.0309 -> 0.0313), so removing the
    mean strictly dominates keeping it.
    """

    patch = np.asarray(field, dtype=np.float64)
    if patch.ndim != 2:
        raise ValueError("radial_power_spectrum expects a 2-D array")
    if bins < 2:
        raise ValueError("bins must be at least 2")

    height, width = patch.shape
    patch = patch - patch.mean()
    window = np.outer(np.hanning(height), np.hanning(width))
    spectrum = np.abs(np.fft.fft2(patch * window)) ** 2

    fy = np.fft.fftfreq(height)[:, np.newaxis] * 2.0  # /Nyquist -> [-1, 1]
    fx = np.fft.fftfreq(width)[np.newaxis, :] * 2.0
    radius = np.sqrt(fy**2 + fx**2) / np.sqrt(2.0)

    edges = np.linspace(0.0, 1.0, bins + 1)
    index = np.clip(np.digitize(radius.ravel(), edges) - 1, 0, bins - 1)
    totals = np.bincount(index, weights=spectrum.ravel(), minlength=bins)
    counts = np.bincount(index, minlength=bins).astype(np.float64)
    power = np.divide(totals, counts, out=np.zeros(bins), where=counts > 0)
    centers = 0.5 * (edges[:-1] + edges[1:])
    return centers, power


def spectral_crops(
    foreground: np.ndarray,
    valid: np.ndarray,
    *,
    margin: int = 16,
    min_size: int = 32,
) -> tuple[tuple[int, slice, slice], ...]:
    """Per-slice ``(z, rows, cols)`` crops the spectral family measures on.

    Split out of ``spectral_descriptors`` because it depends only on the
    segmentation, which is identical across every attack and control of a case:
    the evaluator computes it once per case instead of 25 times (see
    ``case_geometry``).
    """

    fg = np.asarray(foreground, dtype=bool)
    ok = np.asarray(valid, dtype=bool)
    _require_equal_shape_3d("spectral_crops", fg, ok)

    crops: list[tuple[int, slice, slice]] = []
    for z in np.flatnonzero(fg.reshape(fg.shape[0], -1).any(axis=1)):
        # ``fg & ok`` would be a no-op: foreground is ``seg > 0`` and valid is
        # ``seg >= 0``, so foreground is always a subset of valid.
        box = foreground_bbox_2d(fg[int(z)], margin=margin)
        if box is None:
            continue
        rows, cols = box
        # Trimming the bounding box is not enough — invalid voxels sit *inside*
        # it.  On the zones cohort 21.5% of the dilated box is ignore-label
        # (worst slice 52%), and the loss ignores those voxels, so PGD and APGD
        # leave their uniform random start there.  Feeding them to the FFT would
        # measure injected white noise and bias zones toward the very "whiter
        # spectrum" reading this study exists to test.  Shrinking the margin does
        # not help (4% invalid even at margin 0, because the zones foreground is
        # non-convex), so take the largest all-valid rectangle inside the box.
        inner = largest_valid_rectangle(ok[int(z), rows, cols])
        if inner is None:
            continue
        inner_rows, inner_cols = inner
        rows = slice(rows.start + inner_rows.start, rows.start + inner_rows.stop)
        cols = slice(cols.start + inner_cols.start, cols.start + inner_cols.stop)
        if (rows.stop - rows.start) < min_size or (cols.stop - cols.start) < min_size:
            continue
        crops.append((int(z), rows, cols))
    return tuple(crops)


def spectral_descriptors(
    delta: np.ndarray,
    *,
    foreground: np.ndarray,
    valid: np.ndarray,
    margin: int = 16,
    bins: int = 32,
    min_size: int = 32,
    crops: tuple[tuple[int, slice, slice], ...] | None = None,
) -> tuple[dict[str, float], np.ndarray, np.ndarray]:
    """Spectral family for a perturbation volume.

    Each foreground-containing slice is cropped to the foreground bounding box
    dilated by ``margin`` and clipped to the valid region.  Cropping is not
    cosmetic: where the loss ignores voxels (zones uses ``use_mask_for_norm``),
    the gradient there is zero, so PGD and APGD leave their uniform random start
    in place.  Including those voxels would measure injected white noise.

    ``hf_mean_power_fraction`` and ``spectral_centroid_norm`` are computed from
    the radially binned *per-bin mean* power, not from raw coefficient energy:
    with the count-weighted alternative, outer bins hold most coefficients and
    both scalars would be dominated by bin population rather than by spectral
    shape.  Hence the name -- the fraction is of the mean-power profile above
    half Nyquist (0.5 for a white field), not of the perturbation's total energy
    above half Nyquist.  The two differ, the second is what a reader assumes
    when a spectral scalar is called an energy fraction, and this one was called
    ``hf_energy_fraction`` until that was noticed.  It is read against the white
    reference, which is where its meaning comes from.
    """

    d = np.asarray(delta, dtype=np.float64)
    fg = np.asarray(foreground, dtype=bool)
    ok = np.asarray(valid, dtype=bool)
    _require_equal_shape_3d("spectral_descriptors", d, fg, ok)
    if crops is None:
        crops = spectral_crops(fg, ok, margin=margin, min_size=min_size)

    spectra: list[np.ndarray] = []
    centers = np.zeros(bins)
    for z, rows, cols in crops:
        centers, power = radial_power_spectrum(d[z, rows, cols], bins=bins)
        spectra.append(power)

    if not spectra:
        raise ValueError(
            "no usable foreground slice for spectral analysis "
            f"(min_size={min_size}, margin={margin})"
        )

    power = np.mean(np.vstack(spectra), axis=0)
    total = float(power.sum())
    if total <= 0:
        raise ValueError("perturbation carries no spectral energy")

    centroid = float((centers * power).sum() / total)
    high = float(power[centers > 0.5].sum() / total)

    band = (centers >= 0.05) & (centers <= 0.5) & (power > 0)
    if band.sum() >= 2:
        slope = float(np.polyfit(np.log10(centers[band]), np.log10(power[band]), 1)[0])
    else:
        slope = float("nan")

    scalars = {
        "spectral_centroid_norm": centroid,
        "hf_mean_power_fraction": high,
        "spectral_slope": slope,
        "spectral_slices_used": float(len(spectra)),
    }
    return scalars, centers, power


BOUNDARY_BAND_EDGES_MM = (0.0, 2.0, 5.0, 10.0)
BAND_NAMES = ("inside", "0_2mm", "2_5mm", "5_10mm", "beyond_10mm")
# ``inside`` pools the whole interior, so a positive value there cannot separate
# a perturbation spread through the gland from one hugging the inner face of the
# boundary.  These split that pool by depth, using the same edges mirrored
# inwards: ``inside_0_2mm`` is the -2..0 mm shell, ``inside_beyond_10mm`` the
# deep interior.  ``BAND_NAMES`` is left alone so published columns, the
# partition it forms, and every existing figure are unaffected.
INTERIOR_BAND_NAMES = (
    "inside_0_2mm",
    "inside_2_5mm",
    "inside_5_10mm",
    "inside_beyond_10mm",
)
# The signed profile, deep interior to far exterior.
SIGNED_BAND_NAMES = (
    *reversed(INTERIOR_BAND_NAMES),
    "0_2mm",
    "2_5mm",
    "5_10mm",
    "beyond_10mm",
)


def _signed_distance_mm(
    foreground: np.ndarray, spacing: tuple[float, float, float]
) -> np.ndarray:
    """Distance to the foreground boundary in millimetres, negative inside."""

    outside = distance_transform_edt(~foreground, sampling=spacing)
    inside = distance_transform_edt(foreground, sampling=spacing)
    return np.where(foreground, -inside, outside)


def _band_masks(distance: np.ndarray) -> dict[str, np.ndarray]:
    low, mid, high = BOUNDARY_BAND_EDGES_MM[1:]
    # Depth is positive inside, so the interior bands read with the same
    # comparisons as the exterior ones.  Outside voxels have depth <= 0 and so
    # fall out of every interior band without a second mask.
    depth = -distance
    return {
        "inside": distance < 0,
        "0_2mm": (distance >= 0) & (distance < low),
        "2_5mm": (distance >= low) & (distance < mid),
        "5_10mm": (distance >= mid) & (distance < high),
        "beyond_10mm": distance >= high,
        "inside_0_2mm": (distance < 0) & (depth < low),
        "inside_2_5mm": (depth >= low) & (depth < mid),
        "inside_5_10mm": (depth >= mid) & (depth < high),
        "inside_beyond_10mm": depth >= high,
    }


# Public names for the direction analysis (segmentation_direction package).
# The underscored bindings stay so nothing already importing them moves.
signed_distance_mm = _signed_distance_mm
signed_band_masks = _band_masks


def localization_descriptors(
    delta: np.ndarray,
    *,
    foreground: np.ndarray,
    valid: np.ndarray,
    spacing: tuple[float, float, float],
    bands: dict[str, np.ndarray] | None = None,
) -> dict[str, float]:
    """Where the perturbation spends its energy relative to the segmentation.

    Enrichment is the energy fraction divided by the volume fraction, so 1.0
    means "spread uniformly".  RMS-matched random noise sits at 1.0 by
    construction; that is the reference the adversarial value is read against.
    """

    d = np.asarray(delta, dtype=np.float64)
    fg = np.asarray(foreground, dtype=bool)
    ok = np.asarray(valid, dtype=bool)
    _require_equal_shape_3d("localization_descriptors", d, fg, ok)
    if not ok.any():
        raise ValueError("valid mask contains no voxels")

    energy = d**2
    total_energy = float(energy[ok].sum())
    if total_energy <= 0:
        raise ValueError("perturbation carries no energy")
    total_voxels = float(ok.sum())

    fg_valid = fg & ok
    energy_fraction = float(energy[fg_valid].sum()) / total_energy
    volume_fraction = float(fg_valid.sum()) / total_voxels

    out: dict[str, float] = {
        "energy_fraction_foreground": energy_fraction,
        "volume_fraction_foreground": volume_fraction,
        "foreground_energy_enrichment": (
            energy_fraction / volume_fraction if volume_fraction > 0 else float("nan")
        ),
    }

    if bands is None:
        spacing_mm = (float(spacing[0]), float(spacing[1]), float(spacing[2]))
        bands = _band_masks(_signed_distance_mm(fg, spacing_mm))
    for name, band in bands.items():
        selected = band & ok
        band_energy = float(energy[selected].sum()) / total_energy
        band_volume = float(selected.sum()) / total_voxels
        out[f"energy_fraction_band_{name}"] = band_energy
        out[f"volume_fraction_band_{name}"] = band_volume
        out[f"enrichment_band_{name}"] = (
            band_energy / band_volume if band_volume > 0 else float("nan")
        )
    return out


def gradient_magnitude_mm(
    image: np.ndarray, spacing: tuple[float, float, float]
) -> np.ndarray:
    """Physically scaled gradient magnitude (units per mm).

    Passing spacing to ``np.gradient`` matters here: an unweighted gradient
    would treat one 3 mm step in z as equal to one 0.5 mm step in-plane.
    """

    gz, gy, gx = np.gradient(np.asarray(image, dtype=np.float64), *spacing)
    return np.sqrt(gz**2 + gy**2 + gx**2)


def alignment_descriptors(
    delta: np.ndarray,
    clean: np.ndarray,
    *,
    valid: np.ndarray,
    spacing: tuple[float, float, float],
    seed: int,
    max_voxels: int = 200_000,
    gradient: np.ndarray | None = None,
) -> dict[str, float]:
    """Whether the perturbation follows the image's own structure."""

    d = np.asarray(delta, dtype=np.float64)
    x = np.asarray(clean, dtype=np.float64)
    ok = np.asarray(valid, dtype=bool)
    _require_equal_shape_3d("alignment_descriptors", d, x, ok)
    if not ok.any():
        raise ValueError("valid mask contains no voxels")

    if gradient is None:
        spacing_mm = (float(spacing[0]), float(spacing[1]), float(spacing[2]))
        gradient = gradient_magnitude_mm(x, spacing_mm)
    magnitude = np.abs(d)[ok]
    structure = gradient[ok]
    energy = magnitude**2
    total_energy = float(energy.sum())
    if total_energy <= 0:
        raise ValueError("perturbation carries no energy")

    threshold = float(np.quantile(structure, 0.9))
    top = structure >= threshold
    top_share = float(top.sum()) / float(top.size)
    edge_enrichment = (
        float(energy[top].sum()) / total_energy / top_share
        if top_share > 0
        else float("nan")
    )

    count = int(min(max_voxels, magnitude.size))
    if count < magnitude.size:
        rng = np.random.default_rng(int(seed))
        index = rng.choice(magnitude.size, size=count, replace=False)
        magnitude = magnitude[index]
        structure = structure[index]

    correlation = spearmanr(magnitude, structure).statistic
    return {
        "spearman_absdelta_gradmag": float(correlation),
        "edge_energy_enrichment": edge_enrichment,
        "alignment_voxels_used": float(count),
    }


def _require_equal_shape_3d(
    caller: str, reference: np.ndarray, *others: np.ndarray
) -> None:
    """Shared shape guard for the descriptor families.

    ``spectral_descriptors``, ``localization_descriptors`` and
    ``alignment_descriptors`` each carried a copy of this three-clause check;
    the aggregator would have been a fourth, so it is extracted here.
    """

    if reference.ndim != 3:
        raise ValueError(f"{caller} expects 3-D arrays, got {reference.ndim}-D")
    for other in others:
        if other.shape != reference.shape:
            raise ValueError(
                f"{caller} expects equal-shape 3-D arrays, got "
                f"{reference.shape} and {other.shape}"
            )


@dataclass(frozen=True)
class CaseGeometry:
    """The parts of a case's descriptors that no condition can change.

    Spectral crops, boundary-distance bands, the clean image's gradient
    magnitude, and its robust intensity bounds are all functions of the clean
    image and the segmentation alone, so they are identical for every attack and
    control of a case.  Recomputing them per row cost 0.30 s of the 0.37 s a
    whole-gland row takes -- the signed distance transform alone is 0.24 s --
    which is 25x more work than the run needs (5 conditions x 5 epsilons).
    """

    shape: tuple[int, ...]
    margin: int
    min_size: int
    crops: tuple[tuple[int, slice, slice], ...]
    bands: dict[str, np.ndarray]
    gradient: np.ndarray
    bounds: tuple[float, float]


def case_geometry(
    clean: np.ndarray,
    segmentation: np.ndarray,
    *,
    valid_mask: np.ndarray,
    spacing: tuple[float, float, float],
    margin: int = 16,
    min_size: int = 32,
) -> CaseGeometry:
    """Precompute everything ``perturbation_structure`` reuses across conditions."""

    x = np.asarray(clean, dtype=np.float64)
    ok = np.asarray(valid_mask, dtype=bool)
    _require_equal_shape_3d("case_geometry", x, ok)
    foreground = np.asarray(segmentation) > 0
    spacing_mm = (float(spacing[0]), float(spacing[1]), float(spacing[2]))
    return CaseGeometry(
        shape=x.shape,
        margin=int(margin),
        min_size=int(min_size),
        crops=spectral_crops(foreground, ok, margin=margin, min_size=min_size),
        bands=_band_masks(_signed_distance_mm(foreground, spacing_mm)),
        gradient=gradient_magnitude_mm(x, spacing_mm),
        bounds=robust_percentile_bounds(x, ok),
    )


def _require_matching_geometry(
    geometry: CaseGeometry, shape: tuple[int, ...], *, margin: int, min_size: int
) -> None:
    """Reject a precomputed geometry that does not belong to this call."""

    if geometry.shape != shape:
        raise ValueError(f"geometry was built for shape {geometry.shape}, got {shape}")
    if (geometry.margin, geometry.min_size) != (int(margin), int(min_size)):
        raise ValueError(
            f"geometry was built with margin={geometry.margin}, "
            f"min_size={geometry.min_size}; got margin={margin}, min_size={min_size}"
        )


def perturbation_structure(
    clean: np.ndarray,
    altered: np.ndarray,
    segmentation: np.ndarray,
    *,
    valid_mask: np.ndarray,
    spacing: tuple[float, float, float],
    seed: int,
    margin: int = 16,
    bins: int = 32,
    min_size: int = 32,
    max_voxels: int = 200_000,
    max_gmsd_slices: int = 5,
    geometry: CaseGeometry | None = None,
) -> tuple[dict[str, float], np.ndarray, np.ndarray]:
    """All four descriptor families for one clean/altered pair.

    ``geometry`` is an optional ``case_geometry`` result for this exact case; it
    only skips recomputation and never changes a number.  It is validated rather
    than trusted, because a geometry built from another case or under another
    crop setting would silently produce wrong descriptors for a whole run.
    """

    x = np.asarray(clean, dtype=np.float64)
    y = np.asarray(altered, dtype=np.float64)
    ok = np.asarray(valid_mask, dtype=bool)
    # ``ok`` is included deliberately: it is indexed as ``delta[ok]`` below, and an
    # unchecked mismatch would surface as a raw numpy IndexError.  This module's
    # convention (see ``robust_percentile_bounds``) is a clear domain error instead.
    _require_equal_shape_3d("perturbation_structure", x, y, ok)
    foreground = np.asarray(segmentation) > 0
    if geometry is not None:
        _require_matching_geometry(geometry, x.shape, margin=margin, min_size=min_size)

    delta = y - x
    if not np.any(delta[ok]):
        raise ValueError("perturbation carries no energy")

    scalars: dict[str, float] = {"rms_norm": float(np.sqrt(np.mean(delta[ok] ** 2)))}
    spectral, centers, power = spectral_descriptors(
        delta,
        foreground=foreground,
        valid=ok,
        margin=margin,
        bins=bins,
        min_size=min_size,
        crops=None if geometry is None else geometry.crops,
    )
    scalars.update(spectral)
    scalars.update(
        localization_descriptors(
            delta,
            foreground=foreground,
            valid=ok,
            spacing=spacing,
            bands=None if geometry is None else geometry.bands,
        )
    )
    scalars.update(
        alignment_descriptors(
            delta,
            x,
            valid=ok,
            spacing=spacing,
            seed=seed,
            max_voxels=max_voxels,
            gradient=None if geometry is None else geometry.gradient,
        )
    )

    lower, upper = (
        robust_percentile_bounds(x, ok) if geometry is None else geometry.bounds
    )
    scalars["gmsd_axial_prostate"] = axial_gmsd_mean(
        x,
        y,
        lower=lower,
        upper=upper,
        slice_selector=foreground,
        max_slices=max_gmsd_slices,
    )
    return scalars, centers, power


__all__ = [
    "BAND_NAMES",
    "BOUNDARY_BAND_EDGES_MM",
    "INTERIOR_BAND_NAMES",
    "SIGNED_BAND_NAMES",
    "CaseGeometry",
    "alignment_descriptors",
    "case_geometry",
    "foreground_bbox_2d",
    "gradient_magnitude_mm",
    "localization_descriptors",
    "perturbation_structure",
    "radial_power_spectrum",
    "robust_percentile_bounds",
    "signed_band_masks",
    "signed_distance_mm",
    "spectral_crops",
    "spectral_descriptors",
]
