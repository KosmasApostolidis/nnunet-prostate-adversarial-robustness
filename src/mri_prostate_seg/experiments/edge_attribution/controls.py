"""Volume- and energy-matched random control regions (plan §29)."""

from __future__ import annotations

from collections import deque

import numpy as np

BAND_EDGES_MM: tuple[float, ...] = (-10.0, -5.0, -2.0, 0.0, 2.0, 5.0, 10.0)
_OFFSETS = [
    (dz, dy, dx)
    for dz in (-1, 0, 1)
    for dy in (-1, 0, 1)
    for dx in (-1, 0, 1)
    if (dz, dy, dx) != (0, 0, 0)
]


def distance_band(
    signed_distance: np.ndarray,
    reference_mm: float,
    *,
    edges_mm: tuple[float, ...] = BAND_EDGES_MM,
) -> np.ndarray:
    edges = np.asarray((-np.inf, *edges_mm, np.inf), dtype=np.float64)
    index = int(np.searchsorted(edges, reference_mm, side="right") - 1)
    sd = np.asarray(signed_distance, dtype=np.float64)
    return (sd >= edges[index]) & (sd < edges[index + 1])


def volume_matched_region(
    *,
    target_voxels: int,
    band: np.ndarray,
    candidate: np.ndarray,
    rng: np.random.Generator,
    slices: tuple[int, int],
    max_attempts: int = 50,
) -> np.ndarray | None:
    """26-connected BFS region of ``target_voxels`` inside ``band ∩ candidate``."""

    allowed = np.asarray(band, dtype=bool) & np.asarray(candidate, dtype=bool)
    z0, z1 = slices
    allowed[:z0] = False
    allowed[z1 + 1 :] = False
    seeds = np.argwhere(allowed)
    if seeds.size == 0:
        return None
    shape = allowed.shape
    for _ in range(max_attempts):
        seed = seeds[rng.integers(seeds.shape[0])]
        # Unpacked, not `tuple(int(v) for v in seed)`: the generator form infers
        # tuple[int, ...], which does not satisfy deque[tuple[int, int, int]].
        start = (int(seed[0]), int(seed[1]), int(seed[2]))
        region = np.zeros(shape, dtype=bool)
        region[start] = True
        queue: deque[tuple[int, int, int]] = deque([start])
        count = 1
        while queue and count < target_voxels:
            z, y, x = queue.popleft()
            for dz, dy, dx in _OFFSETS:
                n = (z + dz, y + dy, x + dx)
                if not (
                    0 <= n[0] < shape[0]
                    and 0 <= n[1] < shape[1]
                    and 0 <= n[2] < shape[2]
                ):
                    continue
                if allowed[n] and not region[n]:
                    region[n] = True
                    count += 1
                    queue.append(n)
                    if count >= target_voxels:
                        break
        if count >= target_voxels:
            return region
    return None


def energy_matched_region(
    *,
    energy: np.ndarray,
    target_energy: float,
    target_voxels: int,
    band: np.ndarray,
    candidate: np.ndarray,
    rng: np.random.Generator,
    slices: tuple[int, int],
    tolerance: float = 0.05,
    max_attempts: int = 200,
) -> np.ndarray | None:
    e = np.asarray(energy, dtype=np.float64)
    for _ in range(max_attempts):
        region = volume_matched_region(
            target_voxels=target_voxels,
            band=band,
            candidate=candidate,
            rng=rng,
            slices=slices,
            max_attempts=5,
        )
        if region is None:
            # A failed 5-seed batch is unlucky, not impossible: keep drawing until
            # this function's own budget is spent. (When the band truly holds no
            # seeds, volume_matched_region returns immediately, so this stays cheap.)
            continue
        if abs(float(e[region].sum()) - target_energy) <= tolerance * target_energy:
            return region
    return None


__all__ = [
    "BAND_EDGES_MM",
    "distance_band",
    "energy_matched_region",
    "volume_matched_region",
]
