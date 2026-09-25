"""Integrity check for a campaign tree's per-sample CSVs.

Usage: python experiments/verify_campaign_integrity.py <campaign_dir> [...]
Exits 1 if any structural problem is found, so it can gate downstream steps.

Checks each file for: parseability, a correct header, six fields on every row,
a torn final line, out-of-range values, and duplicate (case,eps,class) keys
whose values disagree beyond the campaign's own DUPLICATE_TOL.
"""
import csv, glob, math, os, sys
from collections import defaultdict

TOL = 0.02
HEADER = ["case_id", "epsilon", "class", "dice", "hd95", "asd"]
EPS_GRID = {0.0, 0.02, 0.04, 0.06, 0.08, 0.1}

roots = sys.argv[1:]
files = []
for r in roots:
    files += sorted(glob.glob(os.path.join(r, "*", "*_per_sample.csv")))

problems = defaultdict(list)
stats = {"files": 0, "rows": 0, "nan_hd95": 0, "nan_asd": 0, "nan_dice": 0, "dupe_keys": 0}

for path in files:
    stats["files"] += 1
    rel = os.path.relpath(path)
    with open(path, newline="") as f:
        raw = f.read()
    if not raw.endswith("\n"):
        problems["torn_final_line"].append(rel)
    lines = raw.splitlines()
    if not lines:
        problems["empty_file"].append(rel); continue
    rdr = list(csv.reader(lines))
    if rdr[0] != HEADER:
        problems["bad_header"].append(f"{rel}: {rdr[0]}")
        continue
    seen = {}
    for i, row in enumerate(rdr[1:], start=2):
        if len(row) != 6:
            problems["bad_field_count"].append(f"{rel}:{i} -> {len(row)} fields")
            continue
        cid, eps, cls, d, h, a = row
        try:
            eps = float(eps); d = float(d); h = float(h); a = float(a)
        except ValueError:
            problems["unparseable_number"].append(f"{rel}:{i} -> {row}")
            continue
        stats["rows"] += 1
        if eps not in EPS_GRID:
            problems["eps_off_grid"].append(f"{rel}:{i} -> {eps}")
        if math.isnan(d): stats["nan_dice"] += 1
        elif not (0.0 <= d <= 1.0):
            problems["dice_out_of_range"].append(f"{rel}:{i} -> {d}")
        if math.isnan(h): stats["nan_hd95"] += 1
        elif h < 0: problems["negative_hd95"].append(f"{rel}:{i} -> {h}")
        if math.isnan(a): stats["nan_asd"] += 1
        elif a < 0: problems["negative_asd"].append(f"{rel}:{i} -> {a}")
        key = (cid, eps, cls)
        if key in seen:
            stats["dupe_keys"] += 1
            pd_, ph, pa = seen[key]
            worst = max(
                abs(d - pd_) if not (math.isnan(d) or math.isnan(pd_)) else 0.0,
                0.0,
            )
            if worst > TOL:
                problems["duplicate_disagrees"].append(f"{rel} {key} dice {pd_} vs {d}")
        else:
            seen[key] = (d, h, a)

print(f"files={stats['files']}  rows={stats['rows']:,}")
print(f"duplicate keys={stats['dupe_keys']}  "
      f"nan: dice={stats['nan_dice']} hd95={stats['nan_hd95']} asd={stats['nan_asd']}")
if not problems:
    print("\nNO STRUCTURAL PROBLEMS FOUND")
else:
    print()
    for k, v in problems.items():
        print(f"{k}: {len(v)}")
        for item in v[:5]:
            print(f"    {item}")
    sys.exit(1)
