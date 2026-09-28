"""Pure pieces of experiments/slice_statistics.py, on synthetic data."""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

REPO_ROOT = Path(__file__).resolve().parents[2]
for path in (REPO_ROOT, REPO_ROOT / "src"):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

from experiments.slice_statistics import (  # noqa: E402
    fixed_effects_ols,
    label_source,
    paired_excess,
)


def _synthetic_slices(base_effect: float, seed: int = 0) -> pd.DataFrame:
    """Patients with their own intercepts, an area trend and a planted base effect."""
    rng = np.random.default_rng(seed)
    rows = []
    for patient in range(80):
        intercept = rng.normal(0.0, 0.5)
        for slice_idx in range(12):
            region = "Base" if slice_idx < 3 else "Apex" if slice_idx >= 9 else "Mid"
            area = rng.uniform(100.0, 2000.0)
            clean = rng.uniform(0.6, 1.0)
            drop = (
                intercept
                + base_effect * (region == "Base")
                - 0.05 * np.log(area)
                + 0.2 * clean
                + rng.normal(0.0, 0.01)
            )
            rows.append(
                {
                    "case_id": f"P{patient:03d}",
                    "slice_idx": slice_idx,
                    "anatomical_region": region,
                    "area_mm2": area,
                    "dice_clean": clean,
                    "dice_drop": drop,
                }
            )
    return pd.DataFrame(rows)


def test_fixed_effects_recovers_planted_base_coefficient() -> None:
    frame = _synthetic_slices(base_effect=0.07)

    result = fixed_effects_ols(frame, "dice_drop", n_boot=200, seed=42)

    base = result.set_index("term").loc["base"]
    apex = result.set_index("term").loc["apex"]
    assert base["estimate"] == pytest.approx(0.07, abs=0.005)
    assert base["ci95_low"] < 0.07 < base["ci95_high"]
    assert apex["estimate"] == pytest.approx(0.0, abs=0.005)
    assert set(result["term"]) == {"base", "apex", "clean"}


def test_fixed_effects_absorbs_patient_intercepts() -> None:
    frame = _synthetic_slices(base_effect=0.0)
    shifted = frame.assign(
        dice_drop=frame["dice_drop"] + frame["case_id"].str[1:].astype(int) * 10.0
    )

    a = fixed_effects_ols(frame, "dice_drop", n_boot=10, seed=1)
    b = fixed_effects_ols(shifted, "dice_drop", n_boot=10, seed=1)

    np.testing.assert_allclose(a["estimate"], b["estimate"], atol=1e-8)


def test_label_source_classifies_picai_and_in_house_ids() -> None:
    ids = pd.Series(
        [
            "ProstateWG_10002",
            "ProstateWG_100280262407226510129745299155934395567",
            "ProstateWG_1234",
            "ProstateWG_123456",
            "ProstateWG_1000a",
        ]
    )

    sources = label_source(ids)

    assert sources.tolist() == [
        "PI-CAI",
        "in-house",
        "in-house",
        "in-house",
        "in-house",
    ]


def _arm(drops: list[float], slices: list[int]) -> pd.DataFrame:
    return pd.DataFrame(
        {
            "class": "WG",
            "case_id": "ProstateWG_10002",
            "slice_idx": slices,
            "dice_drop": drops,
        }
    )


def test_paired_excess_is_one_to_one() -> None:
    adversarial = _arm([0.5, 0.3, 0.2], [3, 1, 2])
    noise = _arm([0.1, 0.2, 0.05], [1, 2, 3])

    merged = paired_excess(adversarial, noise, "dice_drop")

    assert len(merged) == 3
    by_slice = merged.set_index("slice_idx")
    assert by_slice.loc[3, "excess"] == pytest.approx(0.45)
    assert by_slice.loc[1, "excess"] == pytest.approx(0.2)
    assert by_slice.loc[1, "noise_dice_drop"] == pytest.approx(0.1)


def test_paired_excess_raises_on_duplicate_slices() -> None:
    adversarial = _arm([0.5, 0.3], [1, 2])
    noise = _arm([0.1, 0.2, 0.3], [1, 1, 2])

    with pytest.raises(ValueError):
        paired_excess(adversarial, noise, "dice_drop")


def test_paired_excess_raises_on_unpaired_slices() -> None:
    adversarial = _arm([0.5, 0.3], [1, 2])
    noise = _arm([0.1], [1])

    with pytest.raises(ValueError):
        paired_excess(adversarial, noise, "dice_drop")
