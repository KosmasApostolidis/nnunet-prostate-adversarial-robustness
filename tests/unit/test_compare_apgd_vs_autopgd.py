"""Unit tests for the four-arm comparison, on synthetic CSVs."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from experiments.compare_apgd_vs_autopgd import (
    fold_summary,
    load_per_sample,
    paired_delta,
)

EPSILONS = [0.0, 0.1]


def _write_per_sample(root, attack, fold, dice_by_case, duplicate=False):
    rows = []
    for case_id, dice in dice_by_case.items():
        for eps in EPSILONS:
            rows.append((case_id, eps, "WG", 0.9 if eps == 0.0 else dice, 5.0, 1.0))
    frame = pd.DataFrame(
        rows, columns=["case_id", "epsilon", "class", "dice", "hd95", "asd"]
    )
    if duplicate:
        frame = pd.concat([frame, frame], ignore_index=True)
    frame.to_csv(root / f"wg_{attack}_fold{fold}_per_sample.csv", index=False)


def test_load_per_sample_drops_duplicate_runs(tmp_path) -> None:
    _write_per_sample(tmp_path, "a_pgd", 0, {"c1": 0.5, "c2": 0.4}, duplicate=True)
    frame = load_per_sample(tmp_path, "a_pgd", folds=(0,))
    assert len(frame) == 4
    assert frame.groupby(["case_id", "epsilon"]).size().max() == 1


def test_load_per_sample_rejects_disagreeing_duplicates(tmp_path) -> None:
    _write_per_sample(tmp_path, "a_pgd", 0, {"c1": 0.5}, duplicate=False)
    path = tmp_path / "wg_a_pgd_fold0_per_sample.csv"
    frame = pd.read_csv(path)
    clashing = frame.copy()
    clashing["dice"] = clashing["dice"] + 0.5
    pd.concat([frame, clashing]).to_csv(path, index=False)
    with pytest.raises(ValueError, match="disagree"):
        load_per_sample(tmp_path, "a_pgd", folds=(0,))


def test_fold_summary_uses_the_spread_across_fold_means(tmp_path) -> None:
    for fold, value in enumerate([0.40, 0.50, 0.60]):
        _write_per_sample(tmp_path, "a_pgd", fold, {"c1": value, "c2": value})
    frame = load_per_sample(tmp_path, "a_pgd", folds=(0, 1, 2))
    summary = fold_summary(frame, "dice")
    row = summary[np.isclose(summary["epsilon"], 0.1)].iloc[0]
    assert row["mean"] == pytest.approx(0.50)
    assert row["sem"] == pytest.approx(np.std([0.4, 0.5, 0.6], ddof=1) / np.sqrt(3))
    assert row["n_folds"] == 3


def test_paired_delta_is_computed_within_case(tmp_path) -> None:
    # c1 and c2 are common to both arms (the merge intersection); c4 is only
    # in auto_pgd and c3 is only in a_pgd, so the case-id join drops both and
    # n_cases must come out as the 2-case intersection, not the 4-case union.
    # The two files also list their cases in different orders (c4, c1, c2 vs.
    # c2, c3, c1), so the shared cases c1/c2 land in different relative order
    # too, and a positional (non-merging) implementation would pair up the
    # wrong rows.
    _write_per_sample(tmp_path, "auto_pgd", 0, {"c4": 0.20, "c1": 0.55, "c2": 0.60})
    _write_per_sample(tmp_path, "a_pgd", 0, {"c2": 0.30, "c3": 0.90, "c1": 0.50})
    left = load_per_sample(tmp_path, "auto_pgd", folds=(0,))
    right = load_per_sample(tmp_path, "a_pgd", folds=(0,))
    delta = paired_delta(left, right, "dice", n_boot=200, seed=42)
    row = delta[np.isclose(delta["epsilon"], 0.1)].iloc[0]

    # Correct paired delta, merged on case_id (intersection {c1, c2}):
    #   c1: 0.55 - 0.50 = 0.05
    #   c2: 0.60 - 0.30 = 0.30
    #   mean = (0.05 + 0.30) / 2 = 0.175
    #
    # Two plausible wrong implementations land elsewhere, and land on the
    # *same* wrong number as each other because they both reduce to a sum
    # over equal-length arrays:
    #   - Unpaired difference of group means, over all 3 cases per arm:
    #       mean([0.20, 0.55, 0.60]) - mean([0.30, 0.90, 0.50])
    #       = 0.45 - 0.566667 = -0.116667
    #   - Positional subtraction of the dice columns as written to disk
    #     (auto_pgd order c4, c1, c2 against a_pgd order c2, c3, c1):
    #       [0.20 - 0.30, 0.55 - 0.90, 0.60 - 0.50]
    #       = [-0.10, -0.35, 0.10], mean = -0.35 / 3 = -0.116667
    # Both wrong values (-0.116667) are far from the correct 0.175, so this
    # fixture discriminates the paired implementation from either mistake.
    # n_cases must be the 2-case {c1, c2} intersection, not the 4-case union
    # {c1, c2, c3, c4}.
    assert row["delta"] == pytest.approx(0.175)
    assert row["n_cases"] == 2
    assert row["ci95_low"] <= row["delta"] <= row["ci95_high"]
