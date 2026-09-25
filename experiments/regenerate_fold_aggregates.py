"""Rewrite per-fold aggregate CSVs whose rows carry n=0 (NaN means).

``pgd_adversarial_evaluation.py`` writes ``<model>_<arm>_fold<k>_{dice,hd95,asd}.csv``
from the cases it scored in that run; a ``--resume`` run that finds every case
already done scores nothing and leaves n=0 / NaN rows. This recomputes those
files from the per-sample CSV beside them with the same ``_agg`` statistics.

Usage: python experiments/regenerate_fold_aggregates.py <campaign_dir> [...]
"""
import csv, glob, os, sys

import numpy as np
import pandas as pd

METRICS = ("dice", "hd95", "asd")


def _agg(vals):
    vals = np.asarray(vals, dtype=np.float64)
    vals = vals[np.isfinite(vals)]
    n = len(vals)
    mean = float(vals.mean()) if n > 0 else np.nan
    std = float(vals.std(ddof=1)) if n > 1 else np.nan
    sem = std / np.sqrt(n) if n > 1 else np.nan
    ci95 = 1.96 * sem if n > 1 else np.nan
    return mean, std, sem, ci95, n


def _needs_regen(path):
    if not os.path.exists(path):
        return True
    t = pd.read_csv(path)
    return bool((t["n"] == 0).any())


def main(roots):
    rewritten = 0
    for root in roots:
        for ps in sorted(glob.glob(os.path.join(root, "*", "*_per_sample.csv"))):
            stem = ps[: -len("_per_sample.csv")]
            targets = [f"{stem}_{m}.csv" for m in METRICS]
            if not any(_needs_regen(t) for t in targets):
                continue
            df = pd.read_csv(ps)
            classes = list(dict.fromkeys(df["class"]))
            eps_list = sorted(df["epsilon"].unique())
            for m, path in zip(METRICS, targets):
                with open(path, "w", newline="") as f:
                    w = csv.writer(f)
                    w.writerow(["epsilon", "class", "mean", "std", "sem", "ci95", "n"])
                    for eps in eps_list:
                        for c in classes:
                            sel = df[(df["epsilon"] == eps) & (df["class"] == c)][m]
                            mean, std, sem, ci95, n = _agg(sel)
                            w.writerow([eps, c, f"{mean:.6f}", f"{std:.6f}",
                                        f"{sem:.6f}", f"{ci95:.6f}", n])
                rewritten += 1
            print(f"regenerated {os.path.relpath(stem)}_{{dice,hd95,asd}}.csv")
    print(f"files rewritten: {rewritten}")


if __name__ == "__main__":
    main(sys.argv[1:])
