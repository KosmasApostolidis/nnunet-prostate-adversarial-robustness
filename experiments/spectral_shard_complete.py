"""Exit 0 only when every named spectral shard has exactly its expected keys.

Progress is counted as unique (dataset, case_id, epsilon_n, attack) keys in
``case_summary.csv`` -- one row per key here, but counting keys keeps the gate
honest if that ever changes -- and a shard with duplicate identity rows fails
the gate outright (the ESEUA duplicate-launch incident wrote 22 keys twice).

Usage: python experiments/spectral_shard_complete.py --root <shards_dir> \
           --expected-keys 200 wg_pgd wg_apgd zones_pgd zones_apgd
"""

from __future__ import annotations

import argparse
import csv
import sys
from collections import Counter
from pathlib import Path


def shard_keys(summary_csv: Path) -> Counter[tuple[str, str, str, str]]:
    keys: Counter[tuple[str, str, str, str]] = Counter()
    if summary_csv.is_file():
        with summary_csv.open(newline="") as handle:
            for row in csv.DictReader(handle):
                keys[(row["dataset"], row["case_id"], row["epsilon_n"], row["attack"])] += 1
    return keys


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("shards", nargs="+")
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--expected-keys", type=int, required=True)
    args = parser.parse_args()
    done = True
    for shard in args.shards:
        keys = shard_keys(args.root / shard / "case_summary.csv")
        n = len(keys)
        dup = sum(1 for c in keys.values() if c > 1)
        pct = 100 * n / args.expected_keys if args.expected_keys else 0.0
        print(f"  {shard:11s} {n:5d}/{args.expected_keys} ({pct:5.1f}%) duplicates={dup}")
        done &= n == args.expected_keys and dup == 0
    sys.exit(0 if done else 1)


if __name__ == "__main__":
    main()
