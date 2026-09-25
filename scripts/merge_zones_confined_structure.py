"""Assemble the confined perturbation-structure table and its spectra tables.

The zones attack rows are regenerated with ``--attack-valid-only`` by
``scripts/run_zones_confined_structure_shards.sh`` (audit Finding 1).  WG rows
and the RMS-matched controls did not change and are carried over from the
published table.  Nothing here recomputes a descriptor.

Outputs, in ``--output-dir``:
  perturbation_structure_all_cases.csv      49,975 rows
  radial_spectra_all_cases_epsilon16.csv    full-cohort e16 spectra (Figure 13)
  radial_spectra_subsample.csv              50-case-per-cohort spectra (Figure 19)
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from experiments.evaluate_perturbation_structure_all_cases import (  # noqa: E402
    ATTACK_KEYS,
    CONDITIONS,
    spectra_case_subset,
)
from experiments.merge_perturbation_structure_shards import (  # noqa: E402
    EXPECTED_ROWS,
    IDENTITY_COLUMNS,
    STRUCTURE_CSV,
    _check,
    _write_atomically,
)

ROOT = REPO_ROOT / "results/adv_rob_eval_results/image_quality_analysis"
SUBSAMPLE_CSV = "radial_spectra_subsample.csv"
E16_CSV = "radial_spectra_all_cases_epsilon16.csv"
ATTACK_LABELS = {CONDITIONS[key] for key in ATTACK_KEYS}
SUBSAMPLE_COUNT = 50
BINS = 32


def _is_zones_attack(frame: pd.DataFrame) -> pd.Series:
    return (frame["dataset"] == "zones") & frame["condition"].isin(ATTACK_LABELS)


def _read_shards(shard_root: Path, name: str) -> pd.DataFrame:
    parts = [shard_root / f"zones_{key}" / name for key in ATTACK_KEYS]
    parts = [p for p in parts if p.is_file()]
    if len(parts) != len(ATTACK_KEYS):
        raise SystemExit(f"expected {len(ATTACK_KEYS)} shards with {name}, found {parts}")
    return pd.concat([pd.read_csv(p) for p in parts], ignore_index=True)


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise SystemExit(f"merge refused: {message}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--published-dir", type=Path, default=ROOT / "perturbation_structure_signed")
    parser.add_argument("--output-dir", type=Path, default=ROOT / "perturbation_structure_confined")
    parser.add_argument(
        "--controls-shard",
        type=Path,
        default=None,
        help="zones gaussian/rician rows re-matched to the confined APGD RMS; "
        "without it the published control rows are carried over",
    )
    args = parser.parse_args()
    shard_root = args.output_dir / "shards"

    published = pd.read_csv(args.published_dir / STRUCTURE_CSV)
    published["attack_valid_only"] = 0  # the published run never confined
    shards = _read_shards(shard_root, STRUCTURE_CSV)
    _require(_is_zones_attack(shards).all(), "shards contain rows other than zones attacks")
    _require(int(shards["attack_valid_only"].min()) == 1, "a shard row is not confined")
    replaced = published[_is_zones_attack(published)]
    _require(len(shards) == len(replaced), f"shards hold {len(shards)} rows, table has {len(replaced)} zones attack rows")
    structure = pd.concat([published[~_is_zones_attack(published)], shards], ignore_index=True)
    controls_spectra = None
    if args.controls_shard is not None:
        controls = pd.read_csv(args.controls_shard / STRUCTURE_CSV)
        controls["attack_valid_only"] = controls.get("attack_valid_only", 0)
        is_zc = (structure["dataset"] == "zones") & ~structure["condition"].isin(ATTACK_LABELS)
        _require(set(controls["dataset"]) == {"zones"} and not controls["condition"].isin(ATTACK_LABELS).any(), "controls shard is not zones controls")
        _require(len(controls) == int(is_zc.sum()), f"controls shard holds {len(controls)} rows, table has {int(is_zc.sum())}")
        structure = pd.concat([structure[~is_zc], controls], ignore_index=True)
        controls_spectra = pd.read_csv(args.controls_shard / SUBSAMPLE_CSV)
    structure = structure.sort_values(IDENTITY_COLUMNS).reset_index(drop=True)
    # Confined attack rows have no published-RMS reference (NaN by design); the
    # shard checker only tolerates that on control rows, so check a copy with a
    # finite placeholder in those two columns for the confined rows.
    checked = structure.copy()
    confined = checked["attack_valid_only"].astype(int) == 1
    for column in ("rms_published_norm", "rms_reproduction_rel_error"):
        checked.loc[confined, column] = checked.loc[confined, column].fillna(0.0)
    _check(checked, expected_rows=EXPECTED_ROWS)

    spectra = _read_shards(shard_root, SUBSAMPLE_CSV)
    zones_cases = sorted(structure.loc[structure["dataset"] == "zones", "case_id"].unique())
    per_key = spectra.groupby(IDENTITY_COLUMNS).size()
    _require(len(per_key) == len(shards) and (per_key == BINS).all(), "shard spectra do not cover every zones attack row with 32 bins")

    e16 = pd.read_csv(args.published_dir / E16_CSV)
    e16_new = pd.concat([e16[~_is_zones_attack(e16)], spectra[spectra["epsilon_n"] == 16]], ignore_index=True)
    if controls_spectra is not None:
        is_zc = (e16_new["dataset"] == "zones") & ~e16_new["condition"].isin(ATTACK_LABELS)
        e16_new = pd.concat([e16_new[~is_zc], controls_spectra[controls_spectra["epsilon_n"] == 16]], ignore_index=True)
    _require(len(e16_new) == len(e16), f"e16 spectra rows {len(e16_new)} != {len(e16)}")

    subsample = pd.read_csv(args.published_dir / SUBSAMPLE_CSV)
    subset = set(spectra_case_subset(zones_cases, count=SUBSAMPLE_COUNT))
    _require(subset == set(subsample.loc[subsample["dataset"] == "zones", "case_id"]), "50-case subsample differs from the published one")
    sub_new = pd.concat([subsample[~_is_zones_attack(subsample)], spectra[spectra["case_id"].isin(subset)]], ignore_index=True)
    if controls_spectra is not None:
        is_zc = (sub_new["dataset"] == "zones") & ~sub_new["condition"].isin(ATTACK_LABELS)
        sub_new = pd.concat([sub_new[~is_zc], controls_spectra[controls_spectra["case_id"].isin(subset)]], ignore_index=True)
    _require(len(sub_new) == len(subsample), f"subsample spectra rows {len(sub_new)} != {len(subsample)}")

    args.output_dir.mkdir(parents=True, exist_ok=True)
    _write_atomically(structure, args.output_dir / STRUCTURE_CSV)
    _write_atomically(e16_new.sort_values(IDENTITY_COLUMNS + ["freq_bin"]), args.output_dir / E16_CSV)
    _write_atomically(sub_new.sort_values(IDENTITY_COLUMNS + ["freq_bin"]), args.output_dir / SUBSAMPLE_CSV)
    print(f"{len(structure)} structure rows, {len(e16_new)} e16 spectra rows, {len(sub_new)} subsample rows -> {args.output_dir}")


if __name__ == "__main__":
    main()
