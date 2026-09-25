"""Validate and atomically merge segmentation-direction shards.

Nothing here recomputes a metric: the shards are the measurement, the merge
concatenates, checks the invariants below, and refuses to write on failure.
Masks are not copied; ``merge_manifest.json`` records each shard's ``masks/``
root so readers resolve a mask through it.
"""

from __future__ import annotations

import argparse
import json
import sys
from collections.abc import Sequence
from pathlib import Path

import numpy as np
import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from experiments.evaluate_segmentation_direction_all_cases import (  # noqa: E402
    DEFAULT_OUTPUT_DIR,
    IDENTITY,
    MASK_DIR,
    PROFILE_CSV,
    SUMMARY_CSV,
    TRANSITION_CSV,
    expected_targets,
    head_target,
)
from mri_prostate_seg.experiments.perturbation_structure import (  # noqa: E402
    SIGNED_BAND_NAMES,
)
from mri_prostate_seg.experiments.segmentation_direction import (  # noqa: E402
    attack_success,
)

EXPECTED_KEYS = 1_396 * 15 + 603 * 15  # 29,985
SURFACE_COLUMNS = (
    "median_surface_motion_mm",
    "q05_surface_motion_mm",
    "q10_surface_motion_mm",
    "q90_surface_motion_mm",
    "q95_surface_motion_mm",
    "median_abs_surface_motion_mm",
    "rms_surface_motion_mm",
    "outward_surface_fraction",
    "inward_surface_fraction",
    "stable_surface_fraction",
    "worsened_outward_surface_fraction",
    "worsened_inward_surface_fraction",
    "corrected_surface_fraction",
    "clean_surface_area_mm2",
    "gt_surface_area_mm2",
)
FRACTION_COLUMNS = tuple(c for c in SURFACE_COLUMNS if c.endswith("_fraction"))
# NaN only where surface_metric_valid == 0: an empty mask undefines all of these.
EMPTY_MASK_COLUMNS = (*SURFACE_COLUMNS, "centroid_shift_norm_mm")
TARGET_IDENTITY = [*IDENTITY, "class_or_union"]
PROFILE_IDENTITY = [*TARGET_IDENTITY, "band"]
TRANSITION_IDENTITY = [*IDENTITY, "transition_type", "source_class", "target_class"]
CLEAN_PREDICTION_GROUP = ["dataset", "case_id", "class_or_union"]
MAX_OFFENDING_KEYS = 20
MAX_OFFENDING_ROWS = 5


def _close(a: pd.Series, b: pd.Series, atol: float = 1e-6) -> np.ndarray:
    return np.isclose(a.to_numpy(float), b.to_numpy(float), atol=atol, equal_nan=True)


def _format_list(items: Sequence[object], cap: int) -> str:
    shown = items[:cap]
    text = ", ".join(str(item) for item in shown)
    remaining = len(items) - len(shown)
    if remaining > 0:
        text += f" and {remaining} more"
    return text


def _identity_tuples(
    frame: pd.DataFrame, mask: np.ndarray, identity: list[str] | None = None
) -> list[tuple[object, ...]]:
    columns = identity if identity is not None else TARGET_IDENTITY
    rows = frame.loc[mask, columns]
    return [tuple(r) for r in rows.itertuples(index=False)][:MAX_OFFENDING_ROWS]


def _format_offenders(offenders: list[tuple[object, ...]]) -> str:
    # ``offenders`` is already capped to MAX_OFFENDING_ROWS by _identity_tuples.
    return ", ".join(str(item) for item in offenders)


def _check_summary(summary: pd.DataFrame, *, expected_keys: int) -> list[str]:
    failures: list[str] = []
    keys = summary[IDENTITY].drop_duplicates()
    if len(keys) != expected_keys:
        failures.append(f"{len(keys)} keys, expected {expected_keys}")
    # ``main`` deduplicates before calling this, so the check below only
    # fires for a caller that passes raw shard frames; it stays as a guard.
    if summary.duplicated(TARGET_IDENTITY).any():
        failures.append("duplicate (key, class_or_union) rows")

    missing_target_keys: list[tuple[object, ...]] = []
    inconsistent_success_keys: list[tuple[object, ...]] = []
    wrong_success_keys: list[tuple[object, ...]] = []
    for key, group in summary.groupby(IDENTITY, sort=False):
        dataset_key = str(key[0])
        if set(group["class_or_union"].astype(str)) != expected_targets(dataset_key):
            missing_target_keys.append(key)
            continue
        if group["attack_success"].nunique() != 1:
            inconsistent_success_keys.append(key)
            continue
        head_rows = group[group["class_or_union"] == head_target(dataset_key)]
        if head_rows.empty:
            continue
        head_delta = float(head_rows["delta_dice"].iloc[0])
        expected_success = int(attack_success(head_delta))
        actual_success = int(group["attack_success"].iloc[0])
        if actual_success != expected_success:
            wrong_success_keys.append(key)
    if missing_target_keys:
        failures.append(
            f"{len(missing_target_keys)} keys lack a target row: "
            f"{_format_list(missing_target_keys, MAX_OFFENDING_KEYS)}"
        )
    if inconsistent_success_keys:
        failures.append(
            f"attack_success varies within {len(inconsistent_success_keys)} keys: "
            f"{_format_list(inconsistent_success_keys, MAX_OFFENDING_KEYS)}"
        )
    if wrong_success_keys:
        failures.append(
            "attack_success != int(delta_dice <= -0.01) of the head target for "
            f"{len(wrong_success_keys)} keys: "
            f"{_format_list(wrong_success_keys, MAX_OFFENDING_KEYS)}"
        )

    for column, condition_frame in (
        ("clean_dice", summary),
        ("clean_volume_mm3", summary),
        ("gt_surface_area_mm2", summary[summary["surface_metric_valid"] == 1.0]),
    ):
        if condition_frame.empty:
            continue
        counts = condition_frame.groupby(CLEAN_PREDICTION_GROUP)[column].nunique(
            dropna=False
        )
        offenders = counts[counts > 1].index.tolist()
        if offenders:
            failures.append(
                f"{column} is not constant per (dataset, case_id, class_or_union) "
                f"for {len(offenders)} groups: "
                f"{_format_list(offenders, MAX_OFFENDING_KEYS)}"
            )
    if summary["displacement_tolerance_mm"].nunique(dropna=False) > 1:
        failures.append("displacement_tolerance_mm is not constant across the table")

    valid = summary[summary["target_valid"] == 1]
    for message, left, right in (
        (
            "volume_change != gained - lost",
            valid["volume_change_mm3"],
            valid["gained_fg_mm3"] - valid["lost_fg_mm3"],
        ),
        (
            "gained != induced_fp + corrected_fn",
            valid["gained_fg_mm3"],
            valid["induced_fp_mm3"] + valid["corrected_fn_mm3"],
        ),
        (
            "lost != induced_fn + corrected_fp",
            valid["lost_fg_mm3"],
            valid["induced_fn_mm3"] + valid["corrected_fp_mm3"],
        ),
    ):
        bad = ~_close(left, right)
        if bad.any():
            offenders = _identity_tuples(valid, bad)
            failures.append(f"{message} (offenders: {_format_offenders(offenders)})")

    surface_ok = (valid["surface_metric_valid"] == 1.0).to_numpy()
    for column in EMPTY_MASK_COLUMNS:
        nan = ~np.isfinite(valid[column].to_numpy(float))
        bad = nan & surface_ok
        if bad.any():
            offenders = _identity_tuples(valid, bad)
            failures.append(
                f"{column} is NaN on a valid-surface row "
                f"(offenders: {_format_offenders(offenders)})"
            )
    for column, description, condition in (
        (
            "change_bias",
            "its total is positive",
            (valid["gained_fg_mm3"] + valid["lost_fg_mm3"]) > 0,
        ),
        (
            "damage_bias",
            "its total is positive",
            (valid["induced_fp_mm3"] + valid["induced_fn_mm3"]) > 0,
        ),
        (
            "volume_change_pct",
            "clean_volume_mm3 is nonzero",
            valid["clean_volume_mm3"] != 0.0,
        ),
    ):
        nan = ~np.isfinite(valid[column].to_numpy(float))
        bad = nan & condition.to_numpy(bool)
        if bad.any():
            offenders = _identity_tuples(valid, bad)
            failures.append(
                f"{column} is NaN where {description} "
                f"(offenders: {_format_offenders(offenders)})"
            )
    skip = set(EMPTY_MASK_COLUMNS) | {"change_bias", "damage_bias", "volume_change_pct"}
    for column in valid.select_dtypes(include=[np.number]).columns:
        if column in skip:
            continue
        bad = ~np.isfinite(valid[column].to_numpy(float))
        if bad.any():
            offenders = _identity_tuples(valid, bad)
            failures.append(
                f"{column} has non-finite values "
                f"(offenders: {_format_offenders(offenders)})"
            )
    invalid = summary[summary["target_valid"] == 0]
    if not (invalid["surface_failure_reason"] == "empty_gt").all():
        failures.append("target_valid == 0 row without empty_gt reason")

    ok_rows = valid[surface_ok]
    for column in FRACTION_COLUMNS:
        v = ok_rows[column].to_numpy(float)
        bad = (v < 0) | (v > 1)
        if bad.any():
            offenders = _identity_tuples(ok_rows, bad)
            failures.append(
                f"{column} outside [0, 1] (offenders: {_format_offenders(offenders)})"
            )
    motion = (
        ok_rows["outward_surface_fraction"]
        + ok_rows["inward_surface_fraction"]
        + ok_rows["stable_surface_fraction"]
    ).to_numpy(float)
    if not np.allclose(motion, 1.0, atol=1e-9):
        failures.append("outward + inward + stable != 1")
    worsen = (
        ok_rows["worsened_outward_surface_fraction"]
        + ok_rows["worsened_inward_surface_fraction"]
        + ok_rows["corrected_surface_fraction"]
    ).to_numpy(float)
    if (worsen > 1 + 1e-9).any():
        failures.append("worsened + corrected > 1")
    return failures


def _check_profile(summary: pd.DataFrame, profile: pd.DataFrame) -> list[str]:
    failures: list[str] = []
    valid = summary[summary["target_valid"] == 1]
    expected = len(valid) * len(SIGNED_BAND_NAMES)
    if len(profile) != expected:
        failures.append(f"profile has {len(profile)} rows, expected {expected}")
    bands = profile.groupby(TARGET_IDENTITY, sort=False)["band"].apply(tuple)
    if not all(b == SIGNED_BAND_NAMES for b in bands):
        failures.append("profile bands are not exactly SIGNED_BAND_NAMES per target")
    for column, risk in (
        ("induced_fp_rate", "at_risk_fp_voxels"),
        ("induced_fn_rate", "at_risk_fn_voxels"),
    ):
        v = profile[column].to_numpy(float)
        r = profile[risk].to_numpy(float)
        bad = (np.isnan(v) & (r > 0)) | (v < 0) | (v > 1)
        if bad.any():
            offenders = _identity_tuples(profile, bad, identity=PROFILE_IDENTITY)
            failures.append(
                f"profile {column} invalid (offenders: {_format_offenders(offenders)})"
            )
    totals = (
        profile.groupby(TARGET_IDENTITY, sort=False)["n_voxels"].sum().reset_index()
    )
    per_case = totals.groupby(["dataset", "case_id", "class_or_union"])[
        "n_voxels"
    ].nunique()
    if (per_case > 1).any():
        failures.append("profile band voxel totals differ across conditions of a case")
    return failures


def check_tables(
    summary: pd.DataFrame,
    profile: pd.DataFrame,
    transitions: pd.DataFrame,
    *,
    expected_keys: int,
) -> None:
    failures = _check_summary(summary, expected_keys=expected_keys)
    failures += _check_profile(summary, profile)
    zones_keys = summary[summary["dataset"] == "zones"][IDENTITY].drop_duplicates()
    if len(transitions) != 36 * len(zones_keys):
        failures.append(
            f"transitions has {len(transitions)} rows, expected {36 * len(zones_keys)}"
        )
    if failures:
        raise SystemExit("merge refused:\n" + "\n".join(f"  - {f}" for f in failures))


def _read(shard_root: Path, name: str, identity: list[str]) -> pd.DataFrame:
    parts = sorted(shard_root.glob(f"*/{name}"))
    if not parts:
        raise SystemExit(f"no {name} under {shard_root}")
    frame = pd.concat([pd.read_csv(p) for p in parts], ignore_index=True)
    # An all-empty surface_failure_reason column reads back as float64 NaN
    # (pd.read_csv infers numeric when nothing in the column is non-empty);
    # normalise it to string so the invariant checks see a string column,
    # not a spuriously "non-finite numeric" one.
    if "surface_failure_reason" in frame.columns:
        frame["surface_failure_reason"] = (
            frame["surface_failure_reason"].fillna("").astype(str)
        )
    # A key redone after a crash re-appends its profile/transition rows; the
    # summary row is the resume marker and is written once, so keep the last.
    return frame.drop_duplicates(identity, keep="last")


def _write_atomically(frame: pd.DataFrame, destination: Path) -> None:
    temporary = destination.with_suffix(destination.suffix + ".partial")
    frame.to_csv(temporary, index=False)
    temporary.replace(destination)


def main() -> None:
    parser = argparse.ArgumentParser(description="Merge segmentation-direction shards")
    parser.add_argument(
        "--shard-root",
        type=Path,
        default=DEFAULT_OUTPUT_DIR.parent / "all_cases_segmentation_direction_shards",
    )
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--expected-keys", type=int, default=EXPECTED_KEYS)
    args = parser.parse_args()

    summary = _read(args.shard_root, SUMMARY_CSV, TARGET_IDENTITY)
    profile = _read(args.shard_root, PROFILE_CSV, PROFILE_IDENTITY)
    transitions = _read(args.shard_root, TRANSITION_CSV, TRANSITION_IDENTITY)
    check_tables(summary, profile, transitions, expected_keys=args.expected_keys)

    args.output_dir.mkdir(parents=True, exist_ok=True)
    _write_atomically(
        summary.sort_values(TARGET_IDENTITY), args.output_dir / SUMMARY_CSV
    )
    _write_atomically(profile, args.output_dir / PROFILE_CSV)
    _write_atomically(transitions, args.output_dir / TRANSITION_CSV)
    shards = sorted(p for p in args.shard_root.iterdir() if p.is_dir())
    manifest = {
        "shards": [str(p) for p in shards],
        "mask_roots": [str(p / MASK_DIR) for p in shards],
        "rows": {
            SUMMARY_CSV: len(summary),
            PROFILE_CSV: len(profile),
            TRANSITION_CSV: len(transitions),
        },
    }
    (args.output_dir / "merge_manifest.json").write_text(
        json.dumps(manifest, indent=2) + "\n"
    )
    print(f"merged {len(summary)} summary rows -> {args.output_dir / SUMMARY_CSV}")


if __name__ == "__main__":
    main()
