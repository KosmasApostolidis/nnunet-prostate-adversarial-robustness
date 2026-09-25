"""Paired defended-vs-baseline comparison on fabricated merged tables."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from experiments import compare_smoothfield_defense as cmp

CASES = [f"case_{i:03d}" for i in range(12)]


def _summary(shift: float, k50_nan_cases: int = 0, seed: int = 0) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    rows = []
    for attack in ("PGD-BCE", "APGD-BCE"):
        for eps in (16, 32):
            for i, case in enumerate(CASES):
                base_damage = 0.4 + 0.02 * i
                rows.append(
                    {
                        "dataset": "wg",
                        "case_id": case,
                        "epsilon_n": eps,
                        "attack": attack,
                        "atlas_type": "radial_shells_24",
                        "stage": "causal",
                        "clean_dice": 0.95 + 0.001 * rng.standard_normal(),
                        "full_dice_damage": base_damage + shift,
                        "lf_energy_share": 0.8 + shift,
                        "lf_sufficiency_dice": 0.98 + shift,
                        "lf_necessity_dice": 0.99 + shift,
                        "k50_frequency_low_first": np.nan if i < k50_nan_cases else 2.0,
                        "f_k50_frequency_low_first_cpm": 0.3,
                        "damage_removed_top_10pct": 0.9,
                        "lf_control_permutation_p": 0.048,
                    }
                )
    return pd.DataFrame(rows)


def test_pairing_is_exact_on_identity_and_delta_recovers_the_injected_shift() -> None:
    base = _summary(0.0)
    defended = _summary(-0.1)
    table, notes = cmp.compare(base, defended, "spectral")
    assert notes == []
    dmg = table[table.metric == "full_dice_damage"]
    assert len(dmg) == 4  # 2 attacks x 2 eps
    assert (dmg.n_paired == len(CASES)).all()
    assert np.allclose(dmg.delta_median, -0.1)
    assert (dmg.delta_ci_lo <= -0.1).all() and (dmg.delta_ci_hi >= -0.1).all()
    assert (dmg.wilcoxon_p < 0.01).all()
    assert np.allclose(dmg.defended_median, dmg.baseline_median - 0.1)


def test_nan_metric_values_reduce_n_defined_not_n_paired() -> None:
    base = _summary(0.0, k50_nan_cases=5)
    defended = _summary(0.0, k50_nan_cases=2)
    table, _ = cmp.compare(base, defended, "spectral")
    k50 = table[table.metric == "k50_frequency_low_first"]
    assert (k50.n_paired == len(CASES)).all()
    assert (k50.n_defined == len(CASES) - 5).all()  # NaN in either arm drops the pair
    assert np.isnan(k50.wilcoxon_p).all()  # all deltas zero -> no test


def test_unshared_cases_are_dropped_and_missing_metric_is_noted() -> None:
    base = _summary(0.0)
    defended = _summary(0.0)[lambda f: f.case_id != CASES[0]].drop(
        columns=["lf_energy_share"]
    )
    table, notes = cmp.compare(base, defended, "spectral")
    assert (table.n_paired == len(CASES) - 1).all()
    assert any("lf_energy_share" in n for n in notes)
    assert "lf_energy_share" not in set(table.metric)


def test_duplicate_identity_rows_are_refused() -> None:
    base = _summary(0.0)
    with pytest.raises(Exception):
        cmp.pair_arms(base, pd.concat([base, base.iloc[:1]]))


def test_edge_family_uses_atlas_type_in_the_key_and_report_renders(tmp_path) -> None:
    def edge(shift):
        f = _summary(shift)[
            [
                "dataset",
                "case_id",
                "epsilon_n",
                "attack",
                "clean_dice",
                "full_dice_damage",
            ]
        ]
        out = []
        for atlas in ("native_edges", "outer_boundary_gt"):
            g = f.copy()
            g["atlas_type"] = atlas
            g["edge_energy_fraction_roi"] = (
                0.2 + (0.05 if atlas == "native_edges" else 0.0) + shift
            )
            g["damage_removed_top_10pct"] = 0.05 + shift
            g["damage_kept_top_10pct"] = 0.01
            out.append(g)
        return pd.concat(out, ignore_index=True)

    table, notes = cmp.compare(edge(0.0), edge(0.02), "edge")
    assert set(table.atlas_type) == {"native_edges", "outer_boundary_gt"}
    assert any("control_permutation_p" in n for n in notes)
    report = cmp.render_report(table, "edge", notes, n_shared=4 * len(CASES) * 2)
    assert (
        "`edge_energy_fraction_roi`" in report
        and "| wg | PGD-BCE | 16 | native_edges |" in report
    )
    (tmp_path / "r.md").write_text(report)


def test_lf_band_metrics_are_compared_as_fractions_of_the_damage() -> None:
    # The driver stores lf_{sufficiency,necessity}_dice in absolute Dice; the report
    # must compare them as fractions of full_dice_damage so arms with different damage
    # are comparable.
    base = _summary(0.0)
    defended = _summary(0.0)
    base["lf_sufficiency_dice"] = 0.5 * base["full_dice_damage"]
    defended["lf_sufficiency_dice"] = 0.8 * defended["full_dice_damage"]
    base["lf_necessity_dice"] = 0.25 * base["full_dice_damage"]
    defended["lf_necessity_dice"] = 1.0 * defended["full_dice_damage"]
    table, _ = cmp.compare(base, defended, "spectral")
    suff = table[table.metric == "lf_sufficiency_fraction"]
    nec = table[table.metric == "lf_necessity_fraction"]
    assert len(suff) == 4 and len(nec) == 4
    assert np.allclose(suff.baseline_median, 0.5) and np.allclose(suff.defended_median, 0.8)
    assert np.allclose(nec.delta_median, 0.75)
    assert "lf_sufficiency_dice" not in set(table.metric)


def test_shell_profile_figure_legend_uses_the_arm_labels(tmp_path, monkeypatch) -> None:
    pytest.importorskip("matplotlib")
    ident = ["dataset", "case_id", "epsilon_n", "attack", "atlas_type"]
    rows = [
        {**dict(zip(ident, ("zones", c, 16, "PGD-BCE", "radial_shells_24"))),
         "patch_id": k, "f_low_cpm": 0.1 * k, "necessity_dice": 0.01 * k}
        for c in CASES[:3] for k in range(3)
    ]
    for arm in ("a", "b"):
        (tmp_path / arm).mkdir()
        pd.DataFrame(rows).to_parquet(tmp_path / arm / "shell_metrics.parquet")
    seen: list[str] = []
    import matplotlib.axes

    real = matplotlib.axes.Axes.plot

    def spy(self, *args, **kwargs):
        seen.append(kwargs.get("label"))
        return real(self, *args, **kwargs)

    monkeypatch.setattr(matplotlib.axes.Axes, "plot", spy)
    ok = cmp.shell_profile_figure(
        tmp_path / "a", tmp_path / "b", pd.DataFrame(rows)[ident].drop_duplicates(),
        tmp_path / "fig.png", labels=("unconfined", "confined"),
    )
    assert ok and seen == ["unconfined", "confined"]
