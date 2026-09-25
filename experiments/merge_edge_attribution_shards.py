"""Validate and atomically merge edge-attribution shards.

Nothing here recomputes a metric: the shards are the measurement, the merge
concatenates, checks the invariants below, and refuses to write on failure.
Atlases and maps are not copied; ``merge_manifest.json`` records each shard's
root so readers resolve them through it.  ``patch_metrics`` and
``subset_curves`` are written as parquet (they are the large tables); the
other three stay CSV.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from experiments.evaluate_adversarial_quality_all_cases import ATTACK_LABELS  # noqa: E402
from experiments.evaluate_edge_attribution_all_cases import (  # noqa: E402
    CONTROL_CSV,
    CURVES_CSV,
    DEFAULT_OUTPUT_DIR,
    IDENTITY,
    PATCH_CSV,
    QC_CSV,
    SUMMARY_CSV,
    expected_atlases,
)

DEFAULT_SHARDS = DEFAULT_OUTPUT_DIR.parent / "all_cases_edge_attribution_shards"
EXPECTED_KEYS = (
    1_396 * 15 + 603 * 15
)  # 29,985 case-conditions for the full energy stage
TABLES = (SUMMARY_CSV, PATCH_CSV, CURVES_CSV, CONTROL_CSV, QC_CSV)


def resolve_expected_keys(explicit: int | None, shards_dir: Path) -> int | None:
    """The 29,985 gate belongs to the published campaign tree; any other tree
    (defended or confined re-runs) is unchecked unless a count is given. 0 skips."""
    if explicit is not None:
        return explicit or None
    return EXPECTED_KEYS if Path(shards_dir) == DEFAULT_SHARDS else None
PARQUET = {PATCH_CSV: "patch_metrics.parquet", CURVES_CSV: "subset_curves.parquet"}
KEY = IDENTITY[:4]

# The spectral driver writes the same five tables with the per-unit table named
# shell_metrics.csv and a single atlas_type literal; everything else (identity,
# resume duplicates, Phase D superseding, curves) merges the same way.
SHELL_CSV = "shell_metrics.csv"
FAMILIES: dict[str, dict[str, object]] = {
    "edge": {"unit_table": PATCH_CSV, "atlas_type": None},
    "spectral": {"unit_table": SHELL_CSV, "atlas_type": "radial_shells_24"},
}


def _read(shard: Path, name: str) -> pd.DataFrame:
    path = shard / name
    if not path.is_file() or path.stat().st_size == 0:
        return pd.DataFrame()
    return pd.read_csv(path)


def check_invariants(
    summary: pd.DataFrame,
    patches: pd.DataFrame,
    curves: pd.DataFrame,
    *,
    family: str = "edge",
) -> list[str]:
    problems: list[str] = []
    if summary.empty:
        return ["summary is empty"]
    real_attack_summary = summary[summary["attack"].isin(ATTACK_LABELS.values())]
    expected_atlas = FAMILIES[family]["atlas_type"]
    if expected_atlas is not None:
        foreign = set(summary["atlas_type"].astype(str)) - {expected_atlas}
        if foreign:
            problems.append(f"atlas_type must be {expected_atlas!r}, found {sorted(foreign)}")
        return problems
    if real_attack_summary.empty:
        # An empty real-attack slice makes the per-key atlas loop below iterate
        # zero times and pass vacuously -- exactly the --expected-keys 0 path
        # used for pilot and partial merges, where that silence would hide a
        # real "no attack rows made it into this shard" bug.
        problems.append("no real-attack summary rows found")
    for key, group in real_attack_summary.groupby(KEY, sort=False):
        want = expected_atlases(str(key[0]))
        have = set(group["atlas_type"].astype(str))
        if not want <= have:
            problems.append(f"{key}: missing atlas rows {sorted(want - have)}")
    if "conservation_error" in summary and (summary["conservation_error"] > 1e-6).any():
        problems.append("conservation_error above 1e-6")
    if not patches.empty and "edge_energy_share" in patches:
        share = patches.groupby(IDENTITY, sort=False)["edge_energy_share"].sum()
        # No `& (share > 0)` guard: an identity that appears here has patches by
        # construction (groupby only yields existing groups), so a zero sum is
        # exactly as much a failure to sum to 1 as any other value -- and this
        # check is the only cross-shard guard against duplicated patch rows, so
        # letting an all-zero identity pass vacuously defeats it.
        bad = share[(share > 1 + 1e-6) | (share < 1 - 1e-6)]
        if not bad.empty:
            problems.append(
                f"edge_energy_share does not sum to 1 for {len(bad)} identities"
            )
    if not patches.empty and "voxel_count" in patches:
        # Every cross-attack comparison downstream assumes (dataset, case_id,
        # atlas_type, patch_id) denotes the same region in all three attack
        # shards, but each shard rebuilds its atlases independently and
        # nothing else checks that. voxel_count is the cheap proxy for "same
        # region": it must not vary across attacks/epsilons for a fixed patch.
        voxel_counts = patches.groupby(
            ["dataset", "case_id", "atlas_type", "patch_id"], sort=False
        )["voxel_count"].nunique()
        inconsistent = voxel_counts[voxel_counts > 1]
        if not inconsistent.empty:
            problems.append(
                f"voxel_count differs across shards for {len(inconsistent)} "
                "(dataset, case_id, atlas_type, patch_id) identities"
            )
    if not curves.empty and "volume_fraction" in curves:
        for key, group in curves.groupby([*IDENTITY, "ranking_type"], sort=False):
            v = group.sort_values("k")["volume_fraction"].to_numpy(dtype=float)
            if (np.diff(v) < -1e-12).any():
                problems.append(f"{key}: volume_fraction not monotone")
                break
    return problems


def _concat(shard_dirs: list[Path], name: str) -> pd.DataFrame:
    if not shard_dirs:
        return pd.DataFrame()
    return pd.concat([_read(s, name) for s in shard_dirs], ignore_index=True)


def merge_shards(
    shard_dirs: list[Path],
    output_dir: Path,
    *,
    expected_keys: int | None,
    superseding_dirs: list[Path] = (),
    excluded: list[str] = (),
    family: str = "edge",
) -> dict[str, int]:
    if family not in FAMILIES:
        raise ValueError(f"family must be one of {sorted(FAMILIES)}, got {family!r}")
    unit_table = str(FAMILIES[family]["unit_table"])
    tables = tuple(unit_table if name == PATCH_CSV else name for name in TABLES)
    parquet = {unit_table: unit_table.replace(".csv", ".parquet"), CURVES_CSV: PARQUET[CURVES_CSV]}
    frames = {name: _concat(shard_dirs, name) for name in tables}
    # Phase D re-runs its subcohort's case-conditions end to end (energy, causal,
    # interaction) into its own shard tree, so for every key it holds, its rows
    # in every table replace the Phase C rows: same measurement, one stage
    # higher, plus the control rows Phase C never wrote.
    superseding = {name: _concat(list(superseding_dirs), name) for name in tables}
    superseded_rows: dict[str, int] = {}
    superseded_keys = pd.DataFrame(columns=KEY)
    if not superseding[SUMMARY_CSV].empty:
        superseded_keys = superseding[SUMMARY_CSV][KEY].drop_duplicates()
        for name in tables:
            base = frames[name]
            if base.empty:
                keep = np.zeros(0, dtype=bool)
            else:
                keep = ~base.merge(
                    superseded_keys, on=KEY, how="left", indicator=True
                )["_merge"].eq("both").to_numpy()
            superseded_rows[name] = int((~keep).sum())
            frames[name] = pd.concat(
                [base[keep], superseding[name]], ignore_index=True
            )
    # Crash-resume replays: the driver appends these four before case_summary, its
    # resume marker, so an aborted condition leaves rows behind that are written
    # again when it is retried. The replay is deterministic and byte-identical, so
    # exact-duplicate rows are the artefact, not a measurement. Counts are reported
    # in the manifest — a silent drop would be worse than the duplicates.
    dropped: dict[str, int] = {}
    for name in (unit_table, CURVES_CSV, CONTROL_CSV, QC_CSV):
        before = len(frames[name])
        frames[name] = frames[name].drop_duplicates(ignore_index=True)
        dropped[name] = before - len(frames[name])
    summary = frames[SUMMARY_CSV]
    if summary.empty:
        raise ValueError("no summary rows found in the shards")
    duplicates = summary.duplicated(subset=IDENTITY).sum()
    if duplicates:
        raise ValueError(f"{duplicates} duplicate identity rows across shards")
    # Noise-control rows carry attack ∈ {gaussian_rms_matched, rician_proxy_rms_matched};
    # only real attack rows count toward the expected case-condition total.
    keys = summary[summary["attack"].isin(ATTACK_LABELS.values())][
        KEY
    ].drop_duplicates()
    if expected_keys is not None and len(keys) != expected_keys:
        raise ValueError(f"expected {expected_keys} case-conditions, found {len(keys)}")
    problems = check_invariants(
        summary, frames[unit_table], frames[CURVES_CSV], family=family
    )
    if problems:
        raise ValueError("invariants failed:\n  " + "\n  ".join(problems))
    output_dir.mkdir(parents=True, exist_ok=True)
    counts: dict[str, int] = {}
    for name, frame in frames.items():
        if name in parquet:
            target = output_dir / parquet[name]
            temporary = target.with_suffix(".parquet.partial")
            frame.to_parquet(temporary, index=False)
            counts[parquet[name]] = int(len(frame))
        else:
            target = output_dir / name
            temporary = target.with_suffix(".csv.partial")
            frame.to_csv(temporary, index=False)
            counts[name] = int(len(frame))
        temporary.replace(target)
    manifest = {
        "table_family": family,
        "shards": [str(s.resolve()) for s in shard_dirs],
        "excluded_shards": list(excluded),
        "row_counts": counts,
        "case_conditions": int(len(keys)),
        "atlas_roots": [str((s / "atlases").resolve()) for s in shard_dirs],
        "map_roots": [str((s / "maps").resolve()) for s in shard_dirs],
        "duplicate_rows_dropped": dropped,
        "superseding_shards": [str(s.resolve()) for s in superseding_dirs],
        "superseded_case_conditions": int(len(superseded_keys)),
        "superseded_rows_dropped": superseded_rows,
    }
    manifest_target = output_dir / "merge_manifest.json"
    manifest_temporary = manifest_target.with_suffix(".json.partial")
    manifest_temporary.write_text(json.dumps(manifest, indent=2))
    manifest_temporary.replace(manifest_target)
    return counts


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Merge edge-attribution shards behind invariants"
    )
    parser.add_argument("--shards-dir", type=Path, default=DEFAULT_SHARDS)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument(
        "--exclude",
        nargs="*",
        default=[],
        metavar="SHARD",
        help="shard directory names to leave out of the merge (e.g. a shard that "
        "was stopped part-way and must not enter the cohort)",
    )
    parser.add_argument(
        "--phase-d-dir",
        type=Path,
        default=None,
        help="Phase D shard tree whose rows supersede Phase C rows for the same "
        "case-conditions (default: <shards-dir>/phase_d when it exists)",
    )
    parser.add_argument(
        "--expected-keys",
        type=int,
        default=None,
        help="case-condition count the merge must find; default 29,985 for the "
        "published shard tree, unchecked for any other tree; pass 0 to skip",
    )
    parser.add_argument(
        "--table-family",
        choices=sorted(FAMILIES),
        default="edge",
        help="edge: patch_metrics + edge atlases; spectral: shell_metrics + radial shells",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    phase_d_dir = args.phase_d_dir or args.shards_dir / "phase_d"
    shard_dirs = sorted(
        p
        for p in args.shards_dir.iterdir()
        if p.is_dir()
        and not p.name.startswith("_")
        and p != phase_d_dir
        and p.name not in set(args.exclude)
    )
    if not shard_dirs:
        raise FileNotFoundError(f"no shard directories under {args.shards_dir}")
    superseding_dirs = (
        sorted(p for p in phase_d_dir.iterdir() if p.is_dir() and not p.name.startswith("_"))
        if phase_d_dir.is_dir()
        else []
    )
    counts = merge_shards(
        shard_dirs,
        args.output_dir,
        expected_keys=resolve_expected_keys(args.expected_keys, args.shards_dir),
        superseding_dirs=superseding_dirs,
        excluded=args.exclude,
        family=args.table_family,
    )
    for name, count in counts.items():
        print(f"{name}: {count} rows", flush=True)
    manifest = json.loads((args.output_dir / "merge_manifest.json").read_text())
    for name, dropped in manifest["duplicate_rows_dropped"].items():
        if dropped:
            print(f"{name}: dropped {dropped} crash-resume duplicate rows", flush=True)
    if manifest["superseded_case_conditions"]:
        print(
            f"phase D superseded {manifest['superseded_case_conditions']} "
            f"case-conditions from {len(manifest['superseding_shards'])} shard(s)",
            flush=True,
        )


if __name__ == "__main__":
    main()
