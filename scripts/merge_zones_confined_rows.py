"""Replace the zones rows of a published per-case table with confined shards.

``--kind quality``: ``adversarial_quality_all_cases.csv`` — the zones attack rows
(FGSM/PGD/APGD) are replaced by the ``--attack-valid-only`` shards; WG rows are
carried over.  ``--kind noise``: ``random_noise_quality_all_cases.csv`` — every
zones row is replaced by the controls re-matched to the confined APGD RMS.
The summary CSV is regenerated with the driver's own writer.  Nothing here
recomputes a metric.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

IDENTITY = {
    "quality": ["dataset", "case_id", "epsilon_n", "attack"],
    "noise": ["dataset", "case_id", "epsilon_n", "perturbation", "trial"],
}
TABLE = {
    "quality": "adversarial_quality_all_cases.csv",
    "noise": "random_noise_quality_all_cases.csv",
}


def _writer(kind: str):
    if kind == "quality":
        from experiments.evaluate_adversarial_quality_all_cases import _write_summary
    else:
        from experiments.evaluate_random_noise_quality_all_cases import _write_summary
    return _write_summary


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--kind", choices=list(TABLE), required=True)
    parser.add_argument("--published", type=Path, required=True, help="the superseded table")
    parser.add_argument("--shards", type=Path, nargs="+", required=True, help="shard CSVs (zones rows)")
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()

    identity = IDENTITY[args.kind]
    published = pd.read_csv(args.published)
    shards = pd.concat([pd.read_csv(p) for p in args.shards], ignore_index=True)
    if set(shards["dataset"]) != {"zones"}:
        raise SystemExit("shards must contain zones rows only")
    if list(shards.columns) != list(published.columns):
        raise SystemExit(f"column mismatch: {list(shards.columns)} vs {list(published.columns)}")
    is_zones = published["dataset"] == "zones"
    replaced = published[is_zones]
    if args.kind == "quality":
        replaced = replaced[replaced["attack"].isin(set(shards["attack"]))]
        keep = ~(is_zones & published["attack"].isin(set(shards["attack"])))
    else:
        keep = ~is_zones
    if len(shards) != len(replaced):
        raise SystemExit(f"shards hold {len(shards)} rows, table has {len(replaced)} rows to replace")
    if set(map(tuple, shards[identity].itertuples(index=False))) != set(map(tuple, replaced[identity].itertuples(index=False))):
        raise SystemExit("shard identity keys differ from the rows they replace")
    merged = pd.concat([published[keep], shards], ignore_index=True).sort_values(identity).reset_index(drop=True)
    if len(merged) != len(published) or merged.duplicated(identity).any():
        raise SystemExit("merged table has the wrong row count or duplicate keys")

    args.output_dir.mkdir(parents=True, exist_ok=True)
    out = args.output_dir / TABLE[args.kind]
    tmp = out.with_suffix(".csv.partial")
    merged.to_csv(tmp, index=False)
    tmp.replace(out)
    _writer(args.kind)(out, out.with_name(out.stem + "_summary.csv"))
    print(f"{len(merged)} rows ({len(shards)} replaced) -> {out}")


if __name__ == "__main__":
    main()
