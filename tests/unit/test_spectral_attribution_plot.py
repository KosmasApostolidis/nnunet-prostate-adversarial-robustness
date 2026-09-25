"""The pre-registered SSEUA rule, evaluated by code on synthetic summaries."""

from __future__ import annotations

import pandas as pd

from experiments.plot_spectral_attribution import (
    CONFIRMED,
    PARTIAL,
    REFUTED,
    verdict,
)


def _summary(lf_suff, k50, p, dataset=("wg", "wg", "zones", "zones"), eps=(16, 32, 16, 32)):
    rows = []
    for i, (d, e) in enumerate(zip(dataset, eps)):
        for case in range(3):
            rows.append(
                {
                    "dataset": d,
                    "case_id": f"{d}{case}",
                    "epsilon_n": e,
                    "attack": "PGD-BCE",
                    "atlas_type": "radial_shells_24",
                    "full_dice_damage": 0.5,
                    "lf_sufficiency_dice": lf_suff[i] * 0.5,
                    "k50_frequency_low_first": k50[i],
                    "control_permutation_p": p[i],
                }
            )
    return pd.DataFrame(rows)


def test_confirmed_when_all_three_hold_in_three_of_four_conditions() -> None:
    out = verdict(_summary([0.8, 0.7, 0.9, 0.3], [4, 6, 8, 20], [0.01, 0.02, 0.03, 0.5]))
    assert out["verdict"] == CONFIRMED
    assert out["n_conditions_all_three"] == 3


def test_refuted_when_sufficiency_fails_in_three_of_four() -> None:
    out = verdict(_summary([0.1, 0.2, 0.3, 0.9], [4, 6, 8, 8], [0.01] * 4))
    assert out["verdict"] == REFUTED


def test_refuted_when_k50_needs_hf() -> None:
    out = verdict(_summary([0.9] * 4, [20, 20, 20, 4], [0.01] * 4))
    assert out["verdict"] == REFUTED


def test_partial_otherwise() -> None:
    out = verdict(_summary([0.9, 0.9, 0.1, 0.1], [4, 4, 4, 4], [0.01, 0.01, 0.2, 0.2]))
    assert out["verdict"] == PARTIAL
