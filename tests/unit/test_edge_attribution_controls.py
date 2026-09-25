"""Matched-control sampler phantoms (plan §29)."""

from __future__ import annotations

import numpy as np

from mri_prostate_seg.experiments.edge_attribution.controls import (
    distance_band,
    energy_matched_region,
    volume_matched_region,
)

SHAPE = (8, 40, 40)


def test_distance_band_contains_reference_and_respects_edges() -> None:
    sd = (
        np.linspace(-15, 15, SHAPE[2])[None, None, :]
        .repeat(SHAPE[0], 0)
        .repeat(SHAPE[1], 1)
    )
    band = distance_band(sd, 1.0)
    assert band[0, 0, np.argmin(np.abs(sd[0, 0] - 1.0))]
    assert not band[0, 0, np.argmin(np.abs(sd[0, 0] - 3.0))]


def test_volume_matched_region_has_requested_size_band_and_slices() -> None:
    band = np.zeros(SHAPE, dtype=bool)
    band[:, 10:30, 10:30] = True
    candidate = np.ones(SHAPE, dtype=bool)
    region = volume_matched_region(
        target_voxels=25,
        band=band,
        candidate=candidate,
        rng=np.random.default_rng(0),
        slices=(3, 4),
    )
    assert region is not None and region.sum() == 25
    assert (region & ~band).sum() == 0
    z = np.nonzero(region)[0]
    assert z.min() >= 3 and z.max() <= 4


def test_volume_matched_region_returns_none_when_impossible() -> None:
    band = np.zeros(SHAPE, dtype=bool)
    band[2, 5, 5] = True
    assert (
        volume_matched_region(
            target_voxels=10,
            band=band,
            candidate=np.ones(SHAPE, bool),
            rng=np.random.default_rng(0),
            slices=(2, 2),
            max_attempts=3,
        )
        is None
    )


def test_energy_matched_region_is_within_tolerance() -> None:
    rng = np.random.default_rng(1)
    energy = rng.random(SHAPE)
    band = np.ones(SHAPE, dtype=bool)
    target = 0.5 * 20
    region = energy_matched_region(
        energy=energy,
        target_energy=target,
        target_voxels=20,
        band=band,
        candidate=np.ones(SHAPE, bool),
        rng=rng,
        slices=(0, 7),
        tolerance=0.05,
    )
    assert region is not None
    assert abs(energy[region].sum() - target) <= 0.05 * target
