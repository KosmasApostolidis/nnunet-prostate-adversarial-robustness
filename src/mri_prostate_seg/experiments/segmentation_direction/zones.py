"""Multiclass label transitions for the zone model (spec §19).

``new_harm`` is the matrix that matters: a voxel that was correct under the
clean prediction and changed class under attack.  Corrections and
wrong-to-wrong moves are kept apart from it (spec §19.4).
"""

from __future__ import annotations

import numpy as np

from .transitions import physical_measure

TRANSITION_TYPES = ("raw", "new_harm", "correction", "wrong_to_wrong")


def multiclass_transition_tables(
    ground_truth: np.ndarray,
    clean_pred: np.ndarray,
    adv_pred: np.ndarray,
    *,
    valid: np.ndarray,
    class_ids: list[int],
) -> dict[str, np.ndarray]:
    g = np.asarray(ground_truth)
    c = np.asarray(clean_pred)
    a = np.asarray(adv_pred)
    ok = np.asarray(valid, dtype=bool)
    if not (g.shape == c.shape == a.shape == ok.shape):
        raise ValueError("all label maps must share one shape")
    n = len(class_ids)
    raw = np.zeros((n, n), dtype=np.int64)
    harm = np.zeros((n, n), dtype=np.int64)
    fix = np.zeros((n, n), dtype=np.int64)
    wrong = np.zeros((n, n), dtype=np.int64)
    for i, source in enumerate(class_ids):
        from_source = ok & (c == source)
        for j, target in enumerate(class_ids):
            moved = from_source & (a == target)
            raw[i, j] = np.count_nonzero(moved)
            if source == target:
                continue
            harm[i, j] = np.count_nonzero(moved & (g == source))
            fix[i, j] = np.count_nonzero(moved & (g == target))
            wrong[i, j] = np.count_nonzero(moved & (g != source) & (g != target))
    off_diagonal = ~np.eye(n, dtype=bool)
    if not np.array_equal(
        raw[off_diagonal],
        harm[off_diagonal] + fix[off_diagonal] + wrong[off_diagonal],
    ):
        raise AssertionError("zone transition partition is incomplete")
    return {"raw": raw, "new_harm": harm, "correction": fix, "wrong_to_wrong": wrong}


def transition_rows(
    tables: dict[str, np.ndarray],
    *,
    class_ids: list[int],
    class_names: list[str],
    ground_truth: np.ndarray,
    clean_pred: np.ndarray,
    valid: np.ndarray,
    spacing: tuple[float, float, float],
) -> list[dict[str, object]]:
    """Long-format rows; ``row_normalized_rate`` follows spec §19.2 for harm."""

    g = np.asarray(ground_truth)
    c = np.asarray(clean_pred)
    ok = np.asarray(valid, dtype=bool)
    voxel_mm3 = physical_measure(np.ones((1,) * g.ndim, dtype=bool), spacing)
    clean_correct = [int(np.count_nonzero(ok & (c == k) & (g == k))) for k in class_ids]
    clean_total = [int(np.count_nonzero(ok & (c == k))) for k in class_ids]
    rows: list[dict[str, object]] = []
    for kind in TRANSITION_TYPES:
        matrix = tables[kind]
        for i, source in enumerate(class_names):
            denominator = clean_correct[i] if kind == "new_harm" else clean_total[i]
            for j, target in enumerate(class_names):
                count = int(matrix[i, j])
                rows.append(
                    {
                        "transition_type": kind,
                        "source_class": source,
                        "target_class": target,
                        "voxel_count": count,
                        "physical_volume_mm3": count * voxel_mm3,
                        "row_normalized_rate": count / denominator
                        if denominator
                        else float("nan"),
                    }
                )
    return rows


__all__ = ["TRANSITION_TYPES", "multiclass_transition_tables", "transition_rows"]
