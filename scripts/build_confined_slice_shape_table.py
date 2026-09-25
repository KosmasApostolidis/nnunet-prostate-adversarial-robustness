"""Rebuild ``all_folds_per_slice_with_shape.csv`` with confined zones attack rows.

The shape covariates (area in mm², perimeter, compactness, ...) depend only on
the ground-truth masks, so they are carried over from the published table
(``rebuttal_paper2/slice_anatomy_corrected``); only the attack-dependent
columns of the zones rows are replaced from a confined-attack per-slice table.
WG rows are unchanged (FGSM on WG has no ignore region or padding).
"""

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[1]
KEY = ["fold", "case_id", "slice_idx", "class", "epsilon"]
ATTACK_COLUMNS = [
    "dice_adv", "dice_drop", "hd95_adv", "hd95_change", "asd_adv", "asd_change",
]
INVARIANT_COLUMNS = ["gt_area", "dice_clean", "hd95_clean", "asd_clean", "anatomical_region"]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument(
        "--published",
        type=Path,
        default=REPO_ROOT / "rebuttal_paper2/slice_anatomy_corrected/all_folds_per_slice_with_shape.csv",
    )
    parser.add_argument("--zones-confined", type=Path, required=True, help="zones_all_folds_per_slice_fgsm.csv from a --attack-valid-only run")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    published = pd.read_csv(args.published)
    confined = pd.read_csv(args.zones_confined)
    zones = published[published["task"] == "zones"]
    merged = zones.drop(columns=ATTACK_COLUMNS).merge(
        confined[KEY + ATTACK_COLUMNS + INVARIANT_COLUMNS],
        on=KEY, how="inner", suffixes=("", "_confined"), validate="one_to_one",
    )
    if len(merged) != len(zones):
        raise SystemExit(f"key mismatch: {len(merged)} joined rows vs {len(zones)} zones rows")
    for column in INVARIANT_COLUMNS:
        left, right = merged[column], merged[f"{column}_confined"]
        if column == "anatomical_region":
            ok = (left == right).all()
        else:
            ok = ((left - right).abs().fillna(0) <= 1e-6).all()
        if not ok:
            raise SystemExit(f"attack-invariant column {column!r} differs between the two runs")
    merged = merged.drop(columns=[f"{c}_confined" for c in INVARIANT_COLUMNS])[published.columns]
    out = pd.concat([published[published["task"] == "wg"], merged], ignore_index=True)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    out.to_csv(args.output, index=False, float_format="%.8f")
    print(f"{len(out)} rows ({len(merged)} zones rows replaced) -> {args.output}")


if __name__ == "__main__":
    main()
