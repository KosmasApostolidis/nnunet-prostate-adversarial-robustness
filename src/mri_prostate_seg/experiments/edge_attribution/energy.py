"""Per-patch perturbation energy and concentration indices (plan §16, §25)."""

from __future__ import annotations

import numpy as np

_NAN = float("nan")


def energy_map(delta: np.ndarray) -> np.ndarray:
    d = np.asarray(delta, dtype=np.float64)
    if d.ndim == 3:
        return d * d
    if d.ndim == 4:
        return (d * d).sum(axis=0)
    raise ValueError(f"unexpected perturbation shape {d.shape}")


def patch_energy_table(
    energy: np.ndarray,
    tubes: np.ndarray,
    *,
    roi: np.ndarray,
    fov: np.ndarray,
) -> list[dict[str, float | int]]:
    e = np.asarray(energy, dtype=np.float64)
    t = np.asarray(tubes)
    r = np.asarray(roi, dtype=bool)
    f = np.asarray(fov, dtype=bool)
    if not (e.shape == t.shape == r.shape == f.shape):
        raise ValueError("energy, tubes, roi, and fov must share one shape")
    total_roi = float(e[r].sum())
    total_fov = float(e[f].sum())
    in_tube = (t > 0) & r
    edge_total = float(e[in_tube].sum())
    roi_voxels = int(r.sum())
    ids = np.unique(t[in_tube])
    if ids.size == 0:
        return []
    sums = np.bincount(t[in_tube].ravel(), weights=e[in_tube].ravel())
    counts = np.bincount(t[in_tube].ravel())
    rows: list[dict[str, float | int]] = []
    for patch_id in ids:
        n = int(counts[patch_id])
        energy_j = float(sums[patch_id])
        p_roi = energy_j / total_roi if total_roi > 0 else 0.0
        p_fov = energy_j / total_fov if total_fov > 0 else 0.0
        p_edge = energy_j / edge_total if edge_total > 0 else 0.0
        q = n / roi_voxels if roi_voxels > 0 else 0.0
        rows.append(
            {
                "patch_id": int(patch_id),
                "voxel_count": n,
                "energy": energy_j,
                "energy_density": energy_j / n if n else _NAN,
                "energy_share_roi": p_roi,
                "energy_share_fov": p_fov,
                "edge_energy_share": p_edge,
                "spatial_share": q,
                "fold_enrichment": p_roi / q if q > 0 else _NAN,
                "percentage_point_enrichment": 100.0 * (p_roi - q),
            }
        )
    return rows


def conservation_error(
    energy: np.ndarray, tubes: np.ndarray, domain: np.ndarray
) -> float:
    e = np.asarray(energy, dtype=np.float64)
    t = np.asarray(tubes)
    ok = np.asarray(domain, dtype=bool)
    total = float(e[ok].sum())
    outside = float(e[ok & (t == 0)].sum())
    per_patch = np.bincount(t[ok & (t > 0)].ravel(), weights=e[ok & (t > 0)].ravel())
    return abs(total - outside - float(per_patch.sum())) / (total + 1e-12)


def gini(values: np.ndarray) -> float:
    v = np.sort(np.asarray(values, dtype=np.float64).ravel())
    n = v.size
    if n == 0 or v.sum() <= 0:
        return _NAN
    index = np.arange(1, n + 1)
    return float((2.0 * (index * v).sum()) / (n * v.sum()) - (n + 1.0) / n)


def concentration_indices(weights: np.ndarray) -> dict[str, float]:
    w = np.asarray(weights, dtype=np.float64).ravel()
    k = w.size
    total = w.sum()
    if k < 2 or total <= 0:
        return {
            key: _NAN
            for key in (
                "n_eff",
                "n_eff_norm",
                "entropy_norm",
                "entropy_concentration",
                "gini",
            )
        }
    w = w / total
    n_eff = 1.0 / float((w * w).sum())
    positive = w[w > 0]
    entropy_norm = float(-(positive * np.log(positive)).sum() / np.log(k))
    return {
        "n_eff": n_eff,
        "n_eff_norm": n_eff / k,
        "entropy_norm": entropy_norm,
        "entropy_concentration": 1.0 - entropy_norm,
        "gini": gini(w),
    }


__all__ = [
    "concentration_indices",
    "conservation_error",
    "energy_map",
    "gini",
    "patch_energy_table",
]
