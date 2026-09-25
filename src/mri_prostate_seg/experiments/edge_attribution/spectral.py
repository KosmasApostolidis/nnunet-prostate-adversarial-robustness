"""Radial-frequency shells of the perturbation (SSEUA spec, Partition / Energy).

Pure numpy.  A shell is a set of ``rfftn`` coefficients with radial frequency in
``(f_{k-1}, f_k]`` cycles/mm; shell edges are fixed on a reference grid so every
case and both datasets share one physical partition.  The one-sided ``rfftn``
layout stores each ``kx > 0`` bin once for its conjugate pair, so every count and
energy is weighted by the Hermitian multiplicity (a port of the band-limited
adversarial-training study's ``grids.py`` / ``masks.py``, which is TensorFlow and
lives in another repository).
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

REFERENCE_SHAPE: tuple[int, int, int] = (24, 256, 256)
REFERENCE_SPACING: tuple[float, float, float] = (3.0, 0.5, 0.5)
N_SHELLS = 24
BANDS = ("LF", "MF", "HF")


def BAND_OF_SHELL(k: int, n_shells: int = N_SHELLS) -> str:  # noqa: N802
    """Tercile label of 1-based shell ``k``; shells 1-8 / 9-16 / 17-24 for 24."""
    if not 1 <= k <= n_shells:
        raise ValueError(f"shell must be in 1..{n_shells}, got {k}")
    return BANDS[min((k - 1) * 3 // n_shells, 2)]


def hermitian_weights(width: int) -> np.ndarray:
    """How many full-spectrum coefficients each stored last-axis bin stands for."""
    stored = width // 2 + 1
    w = np.full(stored, 2.0)
    w[0] = 1.0
    if width % 2 == 0:
        w[-1] = 1.0
    return w


def _radial_cpm(
    shape: tuple[int, int, int],
    spacing: tuple[float, float, float],
    *,
    one_sided: bool,
) -> np.ndarray:
    d, h, w = shape
    sz, sy, sx = spacing
    fz = np.fft.fftfreq(d, d=sz)
    fy = np.fft.fftfreq(h, d=sy)
    fx = np.fft.rfftfreq(w, d=sx) if one_sided else np.fft.fftfreq(w, d=sx)
    return np.sqrt(
        fz[:, None, None] ** 2 + fy[None, :, None] ** 2 + fx[None, None, :] ** 2
    )


def equal_count_edges(
    shape: tuple[int, int, int],
    spacing: tuple[float, float, float],
    n_shells: int,
) -> np.ndarray:
    """Shell edges giving equal unweighted non-DC coefficient counts on ``shape``.

    Bandat spec section 8.1 at ``n_shells`` levels: quantiles of the FULL-grid
    radial frequency with DC excluded.  ``edges[0] = 0``, ``edges[-1] = inf``.
    """
    if n_shells < 1:
        raise ValueError("n_shells must be positive")
    radial = _radial_cpm(shape, spacing, one_sided=False).ravel()
    non_dc = radial[radial > 0]
    inner = np.quantile(non_dc, np.arange(1, n_shells) / n_shells)
    return np.concatenate([[0.0], inner, [np.inf]])


def reference_shell_edges(n_shells: int = N_SHELLS) -> np.ndarray:
    """The fixed partition every case uses: equal-count edges on the reference grid."""
    return equal_count_edges(REFERENCE_SHAPE, REFERENCE_SPACING, n_shells)


def assert_hermitian_symmetric(mask: np.ndarray) -> None:
    """An rfftn mask must equal its conjugate image on the kx = 0 and last planes.

    Only the first and last stored planes are self-conjugate (the last one when
    ``W`` is even); a radial mask is symmetric on every plane, so checking the
    last plane unconditionally is safe for the masks built here.
    """
    if mask.ndim != 3:
        raise ValueError("mask must be [D, H, W//2+1]")
    for kx in (0, mask.shape[-1] - 1):
        plane = mask[:, :, kx]
        mirrored = np.roll(np.roll(plane[::-1, ::-1], 1, axis=0), 1, axis=1)
        if not np.array_equal(plane, mirrored):
            raise AssertionError(f"mask is not Hermitian-symmetric on plane kx={kx}")


@dataclass(frozen=True)
class ShellPartition:
    masks: np.ndarray  # [K, D, H, W//2+1] bool
    weights: np.ndarray  # [W//2+1]
    counts: np.ndarray  # [K] weighted coefficient counts
    edges: np.ndarray  # [K+1] cycles/mm
    shape: tuple[int, int, int]
    spacing: tuple[float, float, float]
    empty: np.ndarray  # [K] bool

    @property
    def n_shells(self) -> int:
        return int(self.masks.shape[0])

    @property
    def weights_3d(self) -> np.ndarray:
        return np.broadcast_to(self.weights, self.masks.shape[1:])


def shell_masks(
    shape: tuple[int, int, int],
    spacing: tuple[float, float, float],
    edges: np.ndarray,
) -> ShellPartition:
    """Boolean rfftn masks for ``(edges[k-1], edges[k]]`` on this case's grid."""
    if len(shape) != 3 or len(spacing) != 3:
        raise ValueError("shape and spacing must be triples")
    shape = (int(shape[0]), int(shape[1]), int(shape[2]))
    spacing = (float(spacing[0]), float(spacing[1]), float(spacing[2]))
    edges = np.asarray(edges, dtype=np.float64)
    radial = _radial_cpm(shape, spacing, one_sided=True)
    non_dc = radial > 0
    masks = np.stack(
        [
            non_dc & (radial > edges[k - 1]) & (radial <= edges[k])
            for k in range(1, edges.size)
        ]
    )
    weights = hermitian_weights(shape[2])
    counts = (masks * weights).sum(axis=(1, 2, 3)).astype(np.float64)
    for mask in masks:
        assert_hermitian_symmetric(mask)
    return ShellPartition(masks, weights, counts, edges, shape, spacing, counts == 0)


@dataclass(frozen=True)
class SpectralDelta:
    delta_out: np.ndarray  # delta outside the valid mask, spatial
    c0: float  # mean of delta inside the valid mask
    F: np.ndarray  # rfftn(delta * valid) with DC zeroed, complex128
    energy_total: float  # sum (delta*valid)^2
    energy_dc: float
    energy_out: float


def split_delta(delta: np.ndarray, valid: np.ndarray) -> SpectralDelta:
    """The three invariant components: outside-valid, DC, and the shelled spectrum."""
    delta = np.asarray(delta, dtype=np.float64)
    valid = np.asarray(valid, dtype=bool)
    if delta.shape != valid.shape or delta.ndim != 3:
        raise ValueError("delta and valid must be equal-shaped 3-D arrays")
    inside = delta * valid
    F = np.fft.rfftn(inside)
    n = float(delta.size)
    c0 = float(F[0, 0, 0].real / n)
    F = F.copy()
    F[0, 0, 0] = 0.0
    return SpectralDelta(
        delta * ~valid,
        c0,
        F,
        float((inside**2).sum()),
        float(n * c0 * c0),
        float((delta[~valid] ** 2).sum()),
    )


def shell_energy(F: np.ndarray, partition: ShellPartition) -> np.ndarray:
    """Parseval energy per shell: sum w |F|^2 / N."""
    power = (np.abs(F) ** 2) * partition.weights_3d
    n = float(np.prod(partition.shape))
    return np.array([float(power[m].sum()) / n for m in partition.masks])


def parseval_error(
    F: np.ndarray, partition: ShellPartition, delta_valid: np.ndarray
) -> float:
    """|sum_k E_k + E_DC - ||delta_valid||^2| / ||delta_valid||^2 (0 if no energy)."""
    dv = np.asarray(delta_valid, dtype=np.float64)
    total = float((dv**2).sum())
    if total <= 0:
        return 0.0
    dc = float(dv.mean() ** 2) * float(np.prod(partition.shape))
    return abs(shell_energy(F, partition).sum() + dc - total) / total


def project(F: np.ndarray, mask: np.ndarray, shape: tuple[int, int, int]) -> np.ndarray:
    """Real-space component of the coefficients selected by ``mask``."""
    return np.fft.irfftn(np.where(mask, F, 0.0), s=tuple(shape)).real


def shell_attribution(
    F: np.ndarray, grad: np.ndarray, partition: ShellPartition
) -> np.ndarray:
    """<P_k delta, grad> for every shell via one FFT of ``grad``."""
    G = np.fft.rfftn(np.asarray(grad, dtype=np.float64))
    cross = (F * np.conj(G)).real * partition.weights_3d
    n = float(np.prod(partition.shape))
    return np.array([float(cross[m].sum()) / n for m in partition.masks])


__all__ = [
    "BANDS",
    "BAND_OF_SHELL",
    "N_SHELLS",
    "REFERENCE_SHAPE",
    "REFERENCE_SPACING",
    "ShellPartition",
    "SpectralDelta",
    "assert_hermitian_symmetric",
    "equal_count_edges",
    "hermitian_weights",
    "parseval_error",
    "project",
    "reference_shell_edges",
    "shell_attribution",
    "shell_energy",
    "shell_masks",
    "split_delta",
]
