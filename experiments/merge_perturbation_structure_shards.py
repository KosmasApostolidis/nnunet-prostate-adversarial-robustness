"""Validate and atomically merge signed-band perturbation-structure shards.

The attack rows are regenerated six ways by ``scripts/run_signed_band_shards.sh``
(one shard per dataset x condition) and the control rows by a single CPU run.
This merges the seven parts into one table, refusing to write unless every
invariant in the shard README holds.

Nothing here recomputes a descriptor: the shards are the measurement, and the
merge only concatenates and checks.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from experiments.evaluate_perturbation_structure_all_cases import (  # noqa: E402
    DEFAULT_OUTPUT_DIR,
)
from mri_prostate_seg.experiments.perturbation_structure import (  # noqa: E402
    INTERIOR_BAND_NAMES,
    SIGNED_BAND_NAMES,
)

IDENTITY_COLUMNS = ["dataset", "case_id", "epsilon_n", "condition"]
STRUCTURE_CSV = "perturbation_structure_all_cases.csv"
SPECTRA_CSV = "radial_spectra_subsample.csv"
EXPECTED_ROWS = 49_975

# ``inside_beyond_10mm`` is empty for a gland with no voxel deeper than 10 mm,
# and an empty band's enrichment is 0/0.  The whole case drops out together --
# every epsilon and every condition -- so the paired contrasts stay balanced.
EMPTY_DEEP_BAND_COLUMN = "enrichment_band_inside_beyond_10mm"
# Controls have no published RMS to reproduce against; the live table carries
# the same nulls.
CONTROL_ONLY_NULL_COLUMNS = ("rms_published_norm", "rms_reproduction_rel_error")


def _read_parts(shard_root: Path, controls_dir: Path) -> pd.DataFrame:
    parts = sorted(shard_root.glob(f"*/{STRUCTURE_CSV}"))
    if not parts:
        raise SystemExit(f"no shards under {shard_root}")
    control_csv = controls_dir / STRUCTURE_CSV
    if not control_csv.is_file():
        raise SystemExit(f"missing control rows at {control_csv}")
    frames = [pd.read_csv(path) for path in (*parts, control_csv)]
    return pd.concat(frames, ignore_index=True)


def _read_spectra(shard_root: Path, controls_dir: Path) -> pd.DataFrame:
    parts = sorted(shard_root.glob(f"*/{SPECTRA_CSV}"))
    frames = [pd.read_csv(path) for path in (*parts, controls_dir / SPECTRA_CSV)]
    merged = pd.concat(frames, ignore_index=True)
    # Every shard runs ``spectra_case_subset`` over the same sorted case list,
    # so the selection must be identical; a mismatch means someone sharded with
    # --max-cases and the subsample no longer describes one cohort.
    for dataset, group in merged.groupby("dataset"):
        selections = {
            frozenset(part.case_id.unique()) for _, part in group.groupby("condition")
        }
        if len(selections) != 1:
            raise SystemExit(
                f"{dataset}: shards disagree on the spectra case subsample"
            )
    return merged


def _check(frame: pd.DataFrame, *, expected_rows: int) -> None:
    failures: list[str] = []

    if len(frame) != expected_rows:
        failures.append(f"row count {len(frame)}, expected {expected_rows}")
    duplicates = int(frame.duplicated(IDENTITY_COLUMNS).sum())
    if duplicates:
        failures.append(f"{duplicates} duplicate identity keys")

    is_control = frame["condition"].str.contains("matched")
    for column in frame.select_dtypes(include=[np.number]).columns:
        nonfinite = ~np.isfinite(frame[column])
        if not nonfinite.any():
            continue
        if column in CONTROL_ONLY_NULL_COLUMNS and (nonfinite == is_control).all():
            continue
        if column == EMPTY_DEEP_BAND_COLUMN:
            empty = frame["volume_fraction_band_inside_beyond_10mm"] == 0
            if (nonfinite == empty).all():
                continue
            failures.append(f"{column} is non-finite where the band is not empty")
            continue
        failures.append(f"{column} has {int(nonfinite.sum())} non-finite values")

    signed = frame[[f"energy_fraction_band_{n}" for n in SIGNED_BAND_NAMES]].sum(axis=1)
    deviation = float(np.nanmax(np.abs(signed - 1.0)))
    if deviation > 1e-9:
        failures.append(f"signed bands sum to 1 +/- {deviation:.3e}")

    interior = frame[[f"energy_fraction_band_{n}" for n in INTERIOR_BAND_NAMES]].sum(
        axis=1
    )
    deviation = float(
        np.nanmax(np.abs(interior - frame["energy_fraction_band_inside"]))
    )
    if deviation > 1e-9:
        failures.append(f"interior bands miss 'inside' by {deviation:.3e}")

    controls = frame[is_control]
    for name in SIGNED_BAND_NAMES:
        values = np.asarray(controls[f"enrichment_band_{name}"], dtype=float)
        values = values[np.isfinite(values)]
        median = float(np.median(values))
        if abs(median - 1.0) > 1e-2:
            failures.append(f"control median enrichment in {name} is {median:.4f}")

    if failures:
        raise SystemExit(
            "merge refused:\n" + "\n".join(f"  - {line}" for line in failures)
        )


def _write_atomically(frame: pd.DataFrame, destination: Path) -> None:
    temporary = destination.with_suffix(destination.suffix + ".partial")
    frame.to_csv(temporary, index=False)
    temporary.replace(destination)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Merge signed-band perturbation-structure shards"
    )
    default_root = DEFAULT_OUTPUT_DIR.parent
    parser.add_argument(
        "--shard-root",
        type=Path,
        default=default_root / "all_cases_signed_bands_attacks_shards",
    )
    parser.add_argument(
        "--controls-dir",
        type=Path,
        default=default_root / "all_cases_signed_bands_controls_staging",
    )
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--expected-rows", type=int, default=EXPECTED_ROWS)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    structure = _read_parts(args.shard_root, args.controls_dir)
    _check(structure, expected_rows=args.expected_rows)
    spectra = _read_spectra(args.shard_root, args.controls_dir)

    args.output_dir.mkdir(parents=True, exist_ok=True)
    _write_atomically(
        structure.sort_values(IDENTITY_COLUMNS), args.output_dir / STRUCTURE_CSV
    )
    _write_atomically(spectra, args.output_dir / SPECTRA_CSV)
    print(f"merged {len(structure)} rows -> {args.output_dir / STRUCTURE_CSV}")
    print(f"merged {len(spectra)} spectra rows -> {args.output_dir / SPECTRA_CSV}")


if __name__ == "__main__":
    main()
