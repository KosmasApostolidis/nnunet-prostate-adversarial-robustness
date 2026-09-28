"""Summarise the camera-ready re-derivation (FGSM, PGD, AS-PGD, noise control).

Reads the per-fold outputs of ``slice_vulnerability_analysis.py --attack <arm>``
and writes, per arm, the statistics the manuscript cites: Table I (3-D metrics,
distances in mm, mean +/- fold SEM), regional Delta-Dice and clean Dice per
anatomical region, the area/Delta-Dice Spearman correlation, and the
area-quartile Base-Mid / Apex-Mid contrasts, all at every epsilon. The
statistics reuse the published helpers so FGSM reproduces Figs. 1-2.
"""

from __future__ import annotations

import argparse
import glob
import sys
from pathlib import Path

import numpy as np
import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))
sys.path.insert(0, str(REPO_ROOT / "experiments"))

from mri_prostate_seg.experiments.slice_vulnerability.figures import (  # noqa: E402
    _bootstrap_mean_ci,
    _cluster_bootstrap_spearman,
    _patient_means,
)
from plot_fig2_area_stratified import (  # noqa: E402
    N_QUARTILES,
    SEED,
    _stratified_contrasts,
)

CLASSES = {"wg": ("WG",), "zones": ("TZ+CZ", "PZ")}
REGIONS = ("Base", "Mid", "Apex")
MM2_PER_PIXEL = 0.5 * 0.5


def load_arm(root: Path, arm: str, kind: str) -> pd.DataFrame:
    """Concatenate ``<model>_fold<k>_per_<kind>_<arm>.csv`` across folds."""
    frames = []
    for model in CLASSES:
        paths = sorted(
            glob.glob(
                str(
                    root
                    / arm
                    / f"{model}_fold*"
                    / f"{model}_fold*_per_{kind}_{arm}.csv"
                )
            )
        )
        for path in paths:
            frame = pd.read_csv(path)
            frame["task"] = model
            frames.append(frame)
    if not frames:
        raise FileNotFoundError(f"no per-{kind} CSVs for {arm} under {root}")
    return pd.concat(frames, ignore_index=True)


def table_one(cases: pd.DataFrame) -> pd.DataFrame:
    """Mean +/- fold SEM, per epsilon and averaged over epsilon > 0."""
    rows = []
    for (cls, eps), g in cases.groupby(["class", "epsilon"]):
        per_fold = g.groupby("fold")[["dice", "hd95_mm", "asd_mm"]].mean()
        rows.append(_summary_row(cls, f"{eps:g}", per_fold))
    attacked = cases[cases["epsilon"] > 0]
    for cls, g in attacked.groupby("class"):
        per_fold = g.groupby("fold")[["dice", "hd95_mm", "asd_mm"]].mean()
        rows.append(_summary_row(cls, "avg_eps>0", per_fold))
    return pd.DataFrame(rows)


def _summary_row(cls: str, eps: str, per_fold: pd.DataFrame) -> dict:
    row = {"class": cls, "epsilon": eps, "n_folds": len(per_fold)}
    for metric in per_fold.columns:
        values = per_fold[metric].to_numpy()
        row[metric] = values.mean()
        row[f"{metric}_sem"] = values.std(ddof=1) / np.sqrt(len(values))
    return row


def regional(slices: pd.DataFrame) -> pd.DataFrame:
    """Patient-weighted mean Delta-Dice and clean Dice per region, per epsilon."""
    rows = []
    for (cls, eps), g in slices.groupby(["class", "epsilon"]):
        for region in REGIONS:
            records = g[g["anatomical_region"] == region].to_dict("records")
            row = {"class": cls, "epsilon": eps, "region": region}
            for key, label in (
                ("dice_drop", "dice_drop"),
                ("dice_clean", "dice_clean"),
            ):
                mean, low, high = _bootstrap_mean_ci(_patient_means(records, key))
                row.update({label: mean, f"{label}_low": low, f"{label}_high": high})
            row["n_patients"] = len(_patient_means(records, "dice_drop"))
            rows.append(row)
    return pd.DataFrame(rows)


def spearman(slices: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for (cls, eps), g in slices.groupby(["class", "epsilon"]):
        if eps == 0:
            continue
        keys = list(zip(g["fold"].astype(int), g["case_id"].astype(str)))
        rho, _, low, high, n = _cluster_bootstrap_spearman(
            g["gt_area"].to_numpy(float) * MM2_PER_PIXEL,
            g["dice_drop"].to_numpy(float),
            keys,
        )
        rows.append(
            {
                "class": cls,
                "epsilon": eps,
                "rho": rho,
                "ci95_low": low,
                "ci95_high": high,
                "n_patients": n,
                "n_slices": len(g),
            }
        )
    return pd.DataFrame(rows)


def quartile_contrasts(slices: pd.DataFrame) -> pd.DataFrame:
    rng = np.random.default_rng(SEED)
    rows = []
    for eps in sorted(e for e in slices["epsilon"].unique() if e > 0):
        frame = slices[np.isclose(slices["epsilon"], eps) & (slices["gt_area"] > 0)]
        for cls in ("WG", "TZ+CZ", "PZ"):
            subset = frame[frame["class"] == cls].copy()
            if subset.empty:
                continue
            subset["patient_id"] = subset["case_id"]
            subset["gt_area_mm2"] = subset["gt_area"] * MM2_PER_PIXEL
            subset["area_quartile"] = pd.qcut(
                subset["gt_area_mm2"], N_QUARTILES, labels=False
            )
            stats = _stratified_contrasts(subset, rng)
            for q in range(N_QUARTILES):
                for r, term in enumerate(("Base_vs_Mid", "Apex_vs_Mid")):
                    low, high = stats["low"][r, q], stats["high"][r, q]
                    rows.append(
                        {
                            "epsilon": eps,
                            "class": cls,
                            "area_quartile": f"Q{q + 1}",
                            "term": term,
                            "difference": stats["point"][r, q],
                            "ci95_low": low,
                            "ci95_high": high,
                            "positive_ci_excludes_zero": bool(low > 0),
                            "n_patients": int(stats["n_patients"][r, q]),
                        }
                    )
    return pd.DataFrame(rows)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument(
        "--root", type=Path, default=REPO_ROOT / "results" / "camera_ready_rederived"
    )
    parser.add_argument("--arms", nargs="+", default=["fgsm", "noise", "pgd", "auto_pgd"])
    parser.add_argument(
        "--out", type=Path, default=None, help="default: <root>/analysis"
    )
    args = parser.parse_args()
    out = args.out or args.root / "analysis"
    out.mkdir(parents=True, exist_ok=True)
    for arm in args.arms:
        slices = load_arm(args.root, arm, "slice")
        for name, frame in (
            ("regional", regional(slices)),
            ("spearman", spearman(slices)),
            ("quartile_contrasts", quartile_contrasts(slices)),
        ):
            frame.to_csv(out / f"{arm}_{name}.csv", index=False)
        try:
            table_one(load_arm(args.root, arm, "case")).to_csv(
                out / f"{arm}_table1.csv", index=False
            )
        except FileNotFoundError as error:
            print(f"  {arm}: {error}")
        print(f"  {arm}: written to {out}")


if __name__ == "__main__":
    main()
