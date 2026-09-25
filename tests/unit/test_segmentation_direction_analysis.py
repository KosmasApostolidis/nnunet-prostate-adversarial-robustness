"""Merge invariants for the direction tables."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from experiments.evaluate_segmentation_direction_all_cases import (
    PROFILE_CSV,
    PROFILE_FIELDNAMES,
    SUMMARY_CSV,
    SUMMARY_FIELDNAMES,
    TRANSITION_CSV,
    TRANSITION_FIELDNAMES,
)
from experiments.merge_segmentation_direction_shards import (
    PROFILE_IDENTITY,
    TARGET_IDENTITY,
    TRANSITION_IDENTITY,
    _read,
    check_tables,
)
from experiments.plot_segmentation_direction import (
    PRIMARY_METRICS,
    join_energy,
    paired,
    profile_summary,
    summarize,
    transition_summary,
)
from mri_prostate_seg.experiments.perturbation_structure import SIGNED_BAND_NAMES


def _summary_row(**overrides: object) -> dict[str, object]:
    row: dict[str, object] = {k: 0.0 for k in SUMMARY_FIELDNAMES}
    row.update(
        dataset="wg",
        case_id="c",
        epsilon_n=2,
        attack="PGD-BCE",
        class_or_union="WG",
        target_valid=1,
        surface_metric_valid=1.0,
        surface_failure_reason="",
        attack_success=0,
        stable_surface_fraction=1.0,
        gained_fg_mm3=2.0,
        lost_fg_mm3=1.0,
        volume_change_mm3=1.0,
        induced_fp_mm3=2.0,
        corrected_fn_mm3=0.0,
        induced_fn_mm3=1.0,
        corrected_fp_mm3=0.0,
        change_bias=1 / 3,
        damage_bias=1 / 3,
    )
    row.update(overrides)
    return row


def _profile(summary: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for _, s in summary.iterrows():
        for band in SIGNED_BAND_NAMES:
            r = {k: 0.0 for k in PROFILE_FIELDNAMES}
            r.update(
                dataset=s.dataset,
                case_id=s.case_id,
                epsilon_n=s.epsilon_n,
                attack=s.attack,
                class_or_union=s.class_or_union,
                band=band,
                n_voxels=10.0,
                at_risk_fp_voxels=1.0,
                at_risk_fn_voxels=1.0,
            )
            rows.append(r)
    return pd.DataFrame(rows)


def _empty_transitions() -> pd.DataFrame:
    return pd.DataFrame(columns=TRANSITION_FIELDNAMES)


def test_consistent_tables_pass() -> None:
    # The eps=32 row is an emptied adversarial mask: surface metrics and the
    # centroid shift are NaN by whitelist, volumes stay finite.
    emptied = _summary_row(
        epsilon_n=32,
        surface_metric_valid=0.0,
        surface_failure_reason="empty_adv",
        adv_volume_mm3=0.0,
        centroid_shift_norm_mm=np.nan,
        median_surface_motion_mm=np.nan,
        stable_surface_fraction=np.nan,
        gained_fg_mm3=0.0,
        lost_fg_mm3=5.0,
        volume_change_mm3=-5.0,
        induced_fp_mm3=0.0,
        induced_fn_mm3=5.0,
        change_bias=-1.0,
        damage_bias=-1.0,
    )
    # The eps=8 row is a case with an empty clean mask: volume_change_pct is
    # NaN by the clean_volume_mm3 == 0 whitelist.
    empty_clean = _summary_row(
        epsilon_n=8, clean_volume_mm3=0.0, volume_change_pct=np.nan
    )
    summary = pd.DataFrame(
        [_summary_row(), _summary_row(epsilon_n=4), emptied, empty_clean]
    )
    check_tables(summary, _profile(summary), _empty_transitions(), expected_keys=4)


def test_partition_violation_is_reported() -> None:
    summary = pd.DataFrame([_summary_row(volume_change_mm3=5.0)])
    with pytest.raises(SystemExit, match="gained - lost"):
        check_tables(summary, _profile(summary), _empty_transitions(), expected_keys=1)


def test_unwhitelisted_nan_is_reported() -> None:
    summary = pd.DataFrame([_summary_row(median_surface_motion_mm=np.nan)])
    with pytest.raises(SystemExit, match="median_surface_motion_mm"):
        check_tables(summary, _profile(summary), _empty_transitions(), expected_keys=1)


def test_volume_change_pct_nan_with_positive_clean_volume_is_reported() -> None:
    summary = pd.DataFrame(
        [_summary_row(clean_volume_mm3=5.0, volume_change_pct=np.nan)]
    )
    with pytest.raises(SystemExit, match="volume_change_pct"):
        check_tables(summary, _profile(summary), _empty_transitions(), expected_keys=1)


def test_missing_profile_band_is_reported() -> None:
    summary = pd.DataFrame([_summary_row()])
    profile = _profile(summary).iloc[:-1]
    with pytest.raises(SystemExit, match="profile"):
        check_tables(summary, profile, _empty_transitions(), expected_keys=1)


def test_wrong_key_count_is_reported() -> None:
    summary = pd.DataFrame([_summary_row()])
    with pytest.raises(SystemExit, match="keys"):
        check_tables(summary, _profile(summary), _empty_transitions(), expected_keys=2)


def test_inconsistent_clean_dice_across_shards_is_reported() -> None:
    summary = pd.DataFrame([_summary_row(), _summary_row(epsilon_n=4, clean_dice=0.9)])
    with pytest.raises(SystemExit, match="clean_dice"):
        check_tables(summary, _profile(summary), _empty_transitions(), expected_keys=2)


def test_inconsistent_clean_volume_across_shards_is_reported() -> None:
    summary = pd.DataFrame(
        [_summary_row(), _summary_row(epsilon_n=4, clean_volume_mm3=9.0)]
    )
    with pytest.raises(SystemExit, match="clean_volume_mm3"):
        check_tables(summary, _profile(summary), _empty_transitions(), expected_keys=2)


def test_inconsistent_gt_surface_area_across_shards_is_reported() -> None:
    summary = pd.DataFrame(
        [_summary_row(), _summary_row(epsilon_n=4, gt_surface_area_mm2=9.0)]
    )
    with pytest.raises(SystemExit, match="gt_surface_area_mm2"):
        check_tables(summary, _profile(summary), _empty_transitions(), expected_keys=2)


def test_inconsistent_displacement_tolerance_is_reported() -> None:
    summary = pd.DataFrame(
        [_summary_row(), _summary_row(epsilon_n=4, displacement_tolerance_mm=1.0)]
    )
    with pytest.raises(SystemExit, match="displacement_tolerance_mm"):
        check_tables(summary, _profile(summary), _empty_transitions(), expected_keys=2)


def test_attack_success_inconsistent_with_delta_dice_is_reported() -> None:
    # delta_dice <= -0.01 should force attack_success == 1; this row claims 0.
    summary = pd.DataFrame([_summary_row(delta_dice=-0.5, attack_success=0)])
    with pytest.raises(SystemExit, match="attack_success"):
        check_tables(summary, _profile(summary), _empty_transitions(), expected_keys=1)


def test_read_normalises_empty_reason_column_from_csv(tmp_path: Path) -> None:
    # An all-empty surface_failure_reason column round-trips through
    # pd.read_csv as float64 NaN, not the empty string it was written as.
    # _read must undo that so check_tables sees a string column.
    summary = pd.DataFrame([_summary_row()])
    profile = _profile(summary)
    transitions = _empty_transitions()

    shard_dir = tmp_path / "shard0"
    shard_dir.mkdir()
    summary.to_csv(shard_dir / SUMMARY_CSV, index=False)
    profile.to_csv(shard_dir / PROFILE_CSV, index=False)
    transitions.to_csv(shard_dir / TRANSITION_CSV, index=False)

    read_summary = _read(tmp_path, SUMMARY_CSV, TARGET_IDENTITY)
    read_profile = _read(tmp_path, PROFILE_CSV, PROFILE_IDENTITY)
    read_transitions = _read(tmp_path, TRANSITION_CSV, TRANSITION_IDENTITY)

    assert read_summary["surface_failure_reason"].dtype == object
    assert (read_summary["surface_failure_reason"] == "").all()
    check_tables(read_summary, read_profile, read_transitions, expected_keys=1)


def _cohort() -> pd.DataFrame:
    rows = []
    for case in ("a", "b", "c", "d"):
        for attack, bias in (("FGSM-BCE", 0.0), ("PGD-BCE", 0.5), ("APGD-BCE", 0.9)):
            for eps in (16, 32):
                rows.append(
                    _summary_row(
                        case_id=case,
                        attack=attack,
                        epsilon_n=eps,
                        damage_bias=bias + (0.01 if case == "a" else 0.0),
                        median_surface_motion_mm=bias,
                        worsened_outward_surface_fraction=bias,
                        worsened_inward_surface_fraction=0.0,
                        centroid_shift_norm_mm=bias,
                        attack_success=int(bias > 0),
                        delta_dice=-bias,
                    )
                )
    frame = pd.DataFrame(rows)
    frame.loc[
        (frame.case_id == "d") & (frame.attack == "APGD-BCE"), "surface_metric_valid"
    ] = 0.0
    frame.loc[
        (frame.case_id == "d") & (frame.attack == "APGD-BCE"), "surface_failure_reason"
    ] = "empty_adv"
    frame.loc[
        (frame.case_id == "d") & (frame.attack == "APGD-BCE"),
        "median_surface_motion_mm",
    ] = np.nan
    return frame


def test_summarize_reports_both_strata_and_exclusions() -> None:
    out = summarize(_cohort(), metrics=PRIMARY_METRICS)
    apgd = out[
        (out.attack == "APGD-BCE")
        & (out.epsilon_n == 16)
        & (out.metric == "median_surface_motion_mm")
    ]
    assert set(apgd.stratum) == {"all", "successful"}
    row = apgd[apgd.stratum == "all"].iloc[0]
    assert row.n == 3 and row.n_excluded == 1 and "empty_adv" in row.exclusion_reasons
    assert row.ci_low <= row["median"] <= row.ci_high
    fgsm = out[
        (out.attack == "FGSM-BCE")
        & (out.stratum == "successful")
        & (out.metric == "damage_bias")
    ]
    assert (fgsm.n == 0).all()


def test_paired_apgd_minus_pgd_is_positive() -> None:
    out = paired(_cohort(), metrics=("damage_bias",))
    row = out[(out.comparison == "APGD-BCE - PGD-BCE") & (out.epsilon_n == 16)].iloc[0]
    assert row.median_diff == pytest.approx(0.4)
    assert row.n == 4
    within = out[(out.comparison == "eps32 - eps16") & (out.attack == "APGD-BCE")].iloc[
        0
    ]
    assert within.median_diff == pytest.approx(0.0)


def test_profile_summary_and_energy_join() -> None:
    summary = _cohort()
    profile = _profile(summary)
    profile["induced_fp_rate"] = 0.25
    ps = profile_summary(profile)
    assert set(ps.columns) >= {
        "band",
        "induced_fp_rate_median",
        "induced_fn_rate_median",
    }
    structure = pd.DataFrame(
        [
            {
                "dataset": "wg",
                "case_id": c,
                "epsilon_n": e,
                "condition": a,
                **{
                    f"enrichment_band_{b}": 1.0 + i
                    for i, b in enumerate(SIGNED_BAND_NAMES)
                },
            }
            for c in "abcd"
            for e in (16, 32)
            for a in ("FGSM-BCE", "PGD-BCE", "APGD-BCE")
        ]
    )
    joined = join_energy(ps, structure)
    assert "energy_enrichment_median" in joined
    assert joined.loc[
        joined.band == "inside_beyond_10mm", "energy_enrichment_median"
    ].iloc[0] == pytest.approx(1.0)
    assert joined["energy_enrichment_median"].notna().all()


def test_transition_summary_medians_rates() -> None:
    rows = [
        {
            "dataset": "zones",
            "case_id": c,
            "epsilon_n": 16,
            "attack": "APGD-BCE",
            "transition_type": "new_harm",
            "source_class": "TZ+CZ",
            "target_class": "PZ",
            "voxel_count": 1,
            "physical_volume_mm3": 0.75,
            "row_normalized_rate": r,
        }
        for c, r in (("a", 0.1), ("b", 0.3), ("c", 0.2))
    ]
    out = transition_summary(pd.DataFrame(rows))
    assert out.iloc[0]["row_normalized_rate_median"] == pytest.approx(0.2)
