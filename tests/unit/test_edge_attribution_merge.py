"""Merge invariants without GPU."""

from __future__ import annotations

import json

import numpy as np
import pandas as pd
import pytest

from experiments.evaluate_edge_attribution_all_cases import (
    CONTROL_CSV,
    CURVES_CSV,
    PATCH_CSV,
    QC_CSV,
    SUMMARY_CSV,
)
from experiments.merge_edge_attribution_shards import check_invariants, merge_shards


def _summary(case: str, atlases: list[str], attack: str = "APGD-BCE") -> pd.DataFrame:
    return pd.DataFrame(
        {
            "dataset": "wg",
            "case_id": case,
            "epsilon_n": 16,
            "attack": attack,
            "atlas_type": atlases,
            "stage": "energy",
            "conservation_error": 0.0,
            "number_of_patches": 2,
            "full_dice_damage": 0.1,
        }
    )


def _patches(case: str, atlases: list[str]) -> pd.DataFrame:
    rows = []
    for atlas in atlases:
        for pid, share in ((1, 0.4), (2, 0.6)):
            rows.append(
                {
                    "dataset": "wg",
                    "case_id": case,
                    "epsilon_n": 16,
                    "attack": "APGD-BCE",
                    "atlas_type": atlas,
                    "patch_id": pid,
                    "edge_energy_share": share,
                    "energy": share,
                }
            )
    return pd.DataFrame(rows)


def _curves(case: str) -> pd.DataFrame:
    return pd.DataFrame(
        {
            "dataset": "wg",
            "case_id": case,
            "epsilon_n": 16,
            "attack": "APGD-BCE",
            "atlas_type": "native_edges",
            "ranking_type": "utility",
            "k": [1, 2],
            "volume_fraction": [0.1, 0.2],
            "damage_removed_fraction": [0.3, np.nan],
        }
    )


def _write_shard(root, name: str, cases: list[str]) -> None:
    d = root / name
    d.mkdir(parents=True)
    pd.concat(
        [_summary(c, ["native_edges", "outer_boundary_gt"]) for c in cases]
    ).to_csv(d / SUMMARY_CSV, index=False)
    pd.concat(
        [_patches(c, ["native_edges", "outer_boundary_gt"]) for c in cases]
    ).to_csv(d / PATCH_CSV, index=False)
    pd.concat([_curves(c) for c in cases]).to_csv(d / CURVES_CSV, index=False)
    pd.DataFrame(
        columns=[
            "dataset",
            "case_id",
            "epsilon_n",
            "attack",
            "atlas_type",
            "control_type",
        ]
    ).to_csv(d / CONTROL_CSV, index=False)
    pd.DataFrame(
        {
            "dataset": "wg",
            "case_id": cases,
            "epsilon_n": 16,
            "attack": "APGD-BCE",
            "atlas_type": "native_edges",
            "n_flags": 0,
            "flags": "",
        }
    ).to_csv(d / QC_CSV, index=False)


def test_merge_concatenates_and_writes_parquet(tmp_path) -> None:
    _write_shard(tmp_path / "shards", "wg_apgd", ["a", "b"])
    _write_shard(tmp_path / "shards", "wg_pgd", ["c"])
    out = tmp_path / "merged"
    counts = merge_shards(
        [tmp_path / "shards" / "wg_apgd", tmp_path / "shards" / "wg_pgd"],
        out,
        expected_keys=3,
    )
    assert counts[SUMMARY_CSV] == 6 and counts["patch_metrics.parquet"] == 12
    assert (out / "patch_metrics.parquet").is_file() and (
        out / "subset_curves.parquet"
    ).is_file()
    assert (out / SUMMARY_CSV).is_file() and (out / "merge_manifest.json").is_file()
    merged = pd.read_parquet(out / "patch_metrics.parquet")
    assert merged["case_id"].nunique() == 3


def test_merge_refuses_duplicate_keys_and_wrong_count(tmp_path) -> None:
    _write_shard(tmp_path / "shards", "one", ["a"])
    _write_shard(tmp_path / "shards", "two", ["a"])
    with pytest.raises(ValueError, match="duplicate"):
        merge_shards(
            [tmp_path / "shards" / "one", tmp_path / "shards" / "two"],
            tmp_path / "m",
            expected_keys=None,
        )
    _write_shard(tmp_path / "s2", "one", ["a"])
    with pytest.raises(ValueError, match="expected"):
        merge_shards([tmp_path / "s2" / "one"], tmp_path / "m2", expected_keys=5)


def test_merge_drops_crash_resume_duplicate_rows_and_reports_count(tmp_path) -> None:
    # Pins the controller's ruling: a crash between the driver appending
    # patch_metrics and appending case_summary (its resume marker) leaves rows
    # behind that get rewritten on retry. The merge must drop exact duplicates
    # from the four non-marker tables and record how many, but summary itself
    # must never be deduplicated (a duplicate summary row is a real conflict).
    d = tmp_path / "shards" / "one"
    d.mkdir(parents=True)
    _summary("a", ["native_edges", "outer_boundary_gt"]).to_csv(
        d / SUMMARY_CSV, index=False
    )
    patches = _patches("a", ["native_edges", "outer_boundary_gt"])
    replayed = pd.concat([patches, patches.iloc[[0]]], ignore_index=True)
    replayed.to_csv(d / PATCH_CSV, index=False)
    _curves("a").to_csv(d / CURVES_CSV, index=False)
    pd.DataFrame(
        columns=[
            "dataset",
            "case_id",
            "epsilon_n",
            "attack",
            "atlas_type",
            "control_type",
        ]
    ).to_csv(d / CONTROL_CSV, index=False)
    pd.DataFrame(
        {
            "dataset": "wg",
            "case_id": ["a"],
            "epsilon_n": 16,
            "attack": "APGD-BCE",
            "atlas_type": "native_edges",
            "n_flags": 0,
            "flags": "",
        }
    ).to_csv(d / QC_CSV, index=False)

    out = tmp_path / "merged"
    merge_shards([d], out, expected_keys=None)

    merged = pd.read_parquet(out / "patch_metrics.parquet")
    assert len(merged) == len(patches)
    manifest = json.loads((out / "merge_manifest.json").read_text())
    assert manifest["duplicate_rows_dropped"][PATCH_CSV] == 1


def test_invariants_catch_missing_atlas_bad_shares_and_nonmonotone_volume() -> None:
    summary = _summary("a", ["native_edges"])  # outer_boundary_gt missing
    patches = _patches("a", ["native_edges"])
    patches.loc[0, "edge_energy_share"] = 0.9  # shares now sum to 1.5
    curves = _curves("a")
    curves.loc[1, "volume_fraction"] = 0.05  # decreasing
    problems = check_invariants(summary, patches, curves)
    assert any("atlas" in p for p in problems)
    assert any("edge_energy_share" in p for p in problems)
    assert any("volume_fraction" in p for p in problems)
    assert (
        check_invariants(
            _summary("a", ["native_edges", "outer_boundary_gt"]),
            _patches("a", ["native_edges", "outer_boundary_gt"]),
            _curves("a"),
        )
        == []
    )


def test_invariants_catch_voxel_count_mismatch_across_shards() -> None:
    # Same (dataset, case_id, atlas_type, patch_id) but two different epsilons
    # (i.e. two shards' attack conditions), each shard having rebuilt its own
    # atlas independently -- voxel_count differing here means the two shards
    # disagree about what "patch 1" even is, which every cross-attack
    # comparison downstream assumes cannot happen.
    rows = [
        {
            "dataset": "wg",
            "case_id": "a",
            "epsilon_n": epsilon_n,
            "attack": "APGD-BCE",
            "atlas_type": "native_edges",
            "patch_id": 1,
            "edge_energy_share": 1.0,
            "energy": 1.0,
            "voxel_count": voxel_count,
        }
        for epsilon_n, voxel_count in ((8, 100), (16, 150))
    ]
    problems = check_invariants(
        _summary("a", ["native_edges", "outer_boundary_gt"]),
        pd.DataFrame(rows),
        _curves("a"),
    )
    assert any("voxel_count" in p for p in problems)


def test_invariants_flag_zero_edge_energy_share_and_empty_real_attack_slice() -> None:
    zero_share = _patches("a", ["native_edges", "outer_boundary_gt"])
    zero_share["edge_energy_share"] = 0.0  # every patch reports zero share
    problems = check_invariants(
        _summary("a", ["native_edges", "outer_boundary_gt"]), zero_share, _curves("a")
    )
    assert any("edge_energy_share" in p for p in problems)

    noise_only = _summary(
        "a", ["native_edges", "outer_boundary_gt"], attack="gaussian_rms_matched"
    )
    problems = check_invariants(
        noise_only, _patches("a", ["native_edges", "outer_boundary_gt"]), _curves("a")
    )
    assert any("real-attack" in p for p in problems)


def test_expected_keys_default_applies_only_to_the_published_shard_tree(tmp_path) -> None:
    from experiments import merge_edge_attribution_shards as m

    # published campaign tree: the 29,985 gate stays on unless overridden
    assert m.resolve_expected_keys(None, m.DEFAULT_SHARDS) == m.EXPECTED_KEYS
    # any other tree (defended / confined re-runs): no count check unless asked for
    assert m.resolve_expected_keys(None, tmp_path / "shards_smoothfield") is None
    assert m.resolve_expected_keys(400, tmp_path / "shards_smoothfield") == 400
    assert m.resolve_expected_keys(0, m.DEFAULT_SHARDS) is None  # 0 = explicit skip
