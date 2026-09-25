"""Clinical proxies of the replay study: PSA-density error (A1) and QC evasion (D).

Reads the direction-run summaries (``results/blade_attack/mechanism/direction``,
nnU-Net, 20 steps, 60 fold-0 cases per model) and writes to
``results/blade_attack/mechanism/clinical/``.

A1 -- PSA density
    PSAD = PSA / gland volume, and PSA does not change under attack, so the
    factor by which an attack misestimates PSA density is
    ``V_clean / V_adv`` (< 1: PSAD underestimated, i.e. the patient looks lower
    risk). The gland is ``WG`` for the whole-gland model and the zone union
    ``gland`` for the zones model. Per (model, arm, epsilon): quantiles of the
    factor and the share of cases misestimated by at least 20% and by at least
    2x in either direction. The 20% / 2x cut-offs are descriptive, not clinical
    thresholds; a decision-flip analysis needs each patient's PSA.

D -- QC evasion
    A QC designer sees only the output, not the ground truth. The rules applied
    here are the ones computable from volumes (the replay did not store masks,
    so connected-component and topology rules are out of scope):
      * gland volume inside the [1st, 99th] percentile of ground-truth gland
        volumes of the whole dataset (all cases, all folds);
      * zones model only: TZ+CZ fraction of the gland inside the [1st, 99th]
        percentile of the ground-truth fraction.
    A case is "damaged" when a target's Dice drops by at least 0.10 from the
    clean prediction (for the zones model: either zone). Reported per
    (model, arm, epsilon): QC pass rate of clean outputs (false-alarm
    baseline), QC pass rate of attacked outputs, and the share of damaged cases
    that still pass QC ("undetected damage").

Usage::

    PYTHONPATH=src python experiments/analyze_clinical_proxies.py
"""

from __future__ import annotations

import argparse
import glob
import os
import sys

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import nibabel as nib  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from run_twelve_arm_wg_campaign import ARM_LABELS  # noqa: E402

ROOT = os.path.join("results", "blade_attack", "mechanism")
RAW = os.path.join("nnUnet_paths", "nnUNet_raw")
LABEL_DIRS = {
    "wg": os.path.join(RAW, "Dataset016_WgSegmentationPNetAndPicai", "labelsTr"),
    "zones": os.path.join(
        RAW, "Dataset019_ProstateZonesSegmentationWgFilteredLessDilated", "labelsTr"
    ),
}
GLAND = {"wg": "WG", "zones": "gland"}
DAMAGE_DROP = 0.10
PERCENTILES = (1.0, 99.0)
# Colours: the dataviz reference palette's 8 categorical slots in fixed order
# (validated: CVD separation and normal-vision floor pass; the CSVs are the
# table view that the low-contrast slots require).
ARM_ORDER = (
    "fgsm",
    "pgd",
    "auto_pgd",
    "auto_pgd_r3",
    "sea",
    "blade1",
    "blade",
    "blade_mm",
    "blade_boundary",
)
ARM_COLOUR = dict(
    zip(
        ARM_ORDER[1:],
        (
            "#2a78d6",
            "#eb6834",
            "#1baf7a",
            "#eda100",
            "#e87ba4",
            "#008300",
            "#4a3aa7",
            "#e34948",
        ),
    )
)
# Nine arms, eight slots: FGSM (single-step PGD) shares PGD's hue and is told
# apart by a dashed line / hatched box (composite encoding, never a 9th hue).
ARM_COLOUR["fgsm"] = ARM_COLOUR["pgd"]
LINESTYLE = {"fgsm": "--"}
HATCH = {"fgsm": "////"}


def ground_truth_volumes(task: str, cache: str) -> pd.DataFrame:
    """Per-case ground-truth gland (and TZ+CZ) volume in mm^3 from the raw labels."""
    if os.path.isfile(cache):
        return pd.read_csv(cache)
    rows = []
    for path in sorted(glob.glob(os.path.join(LABEL_DIRS[task], "*.nii.gz"))):
        img = nib.load(path)
        lab = np.asarray(img.dataobj)
        vox = float(np.prod(img.header.get_zooms()[:3]))
        row = {
            "case_file": os.path.basename(path),
            "gland_mm3": float((lab >= 1).sum()) * vox,
        }
        if task == "zones":
            row["tzcz_mm3"] = float((lab == 1).sum()) * vox
        rows.append(row)
    t = pd.DataFrame(rows)
    t.to_csv(cache, index=False)
    return t


def qc_ranges(gt: dict[str, pd.DataFrame]) -> dict[str, dict[str, tuple[float, float]]]:
    lo, hi = PERCENTILES
    out = {}
    for task, t in gt.items():
        r = {"gland_mm3": tuple(np.percentile(t["gland_mm3"], [lo, hi]))}
        if task == "zones":
            frac = t["tzcz_mm3"] / t["gland_mm3"]
            r["tzcz_fraction"] = tuple(np.percentile(frac, [lo, hi]))
        out[task] = r
    return out


def per_case_table(d: pd.DataFrame) -> pd.DataFrame:
    """One row per (task, arm, case, epsilon) with gland volumes, zone volumes and damage."""
    rows = []
    for (task, arm, case, eps), g in d.groupby(["task", "arm", "case_id", "epsilon"]):
        g = g.set_index("target")
        gl = g.loc[GLAND[task]]
        row = {
            "task": task,
            "arm": arm,
            "case_id": case,
            "epsilon": eps,
            "clean_gland_mm3": gl["clean_volume_mm3"],
            "adv_gland_mm3": gl["adv_volume_mm3"],
        }
        if task == "zones":
            tz = g.loc["TZ+CZ"]
            row["clean_tzcz_fraction"] = tz["clean_volume_mm3"] / max(
                gl["clean_volume_mm3"], 1e-9
            )
            row["adv_tzcz_fraction"] = tz["adv_volume_mm3"] / max(
                gl["adv_volume_mm3"], 1e-9
            )
            drop = max(
                g.loc["TZ+CZ", "clean_dice"] - g.loc["TZ+CZ", "adv_dice"],
                g.loc["PZ", "clean_dice"] - g.loc["PZ", "adv_dice"],
            )
        else:
            drop = gl["clean_dice"] - gl["adv_dice"]
        row["dice_drop"] = drop
        rows.append(row)
    t = pd.DataFrame(rows)
    t["psad_factor"] = t["clean_gland_mm3"] / t["adv_gland_mm3"].clip(lower=1e-9)
    return t


def passes_qc(t: pd.DataFrame, which: str, ranges) -> pd.Series:
    ok = pd.Series(True, index=t.index)
    for task, r in ranges.items():
        m = t["task"] == task
        lo, hi = r["gland_mm3"]
        v = t.loc[m, f"{which}_gland_mm3"]
        ok.loc[m] &= (v >= lo) & (v <= hi)
        if "tzcz_fraction" in r:
            flo, fhi = r["tzcz_fraction"]
            f = t.loc[m, f"{which}_tzcz_fraction"]
            ok.loc[m] &= (f >= flo) & (f <= fhi)
    return ok


def summarize(t: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    psad, qc = [], []
    for (task, arm, eps), g in t.groupby(["task", "arm", "epsilon"]):
        f = g["psad_factor"]
        psad.append(
            {
                "task": task,
                "arm": arm,
                "epsilon": eps,
                "n_cases": len(g),
                "median_factor": f.median(),
                "q10": f.quantile(0.10),
                "q90": f.quantile(0.90),
                "share_under_20pct": (f <= 1 / 1.2).mean(),
                "share_over_20pct": (f >= 1.2).mean(),
                "share_under_2x": (f <= 0.5).mean(),
                "share_over_2x": (f >= 2.0).mean(),
            }
        )
        damaged = g["dice_drop"] >= DAMAGE_DROP
        qc.append(
            {
                "task": task,
                "arm": arm,
                "epsilon": eps,
                "n_cases": len(g),
                "clean_pass_rate": g["qc_clean"].mean(),
                "adv_pass_rate": g["qc_adv"].mean(),
                "n_damaged": int(damaged.sum()),
                "damaged_pass_rate": g.loc[damaged, "qc_adv"].mean()
                if damaged.any()
                else np.nan,
                "undetected_damage_share": (damaged & g["qc_adv"]).mean(),
            }
        )
    return pd.DataFrame(psad), pd.DataFrame(qc)


def _arms(t: pd.DataFrame) -> list[str]:
    return [a for a in ARM_ORDER if a in set(t["arm"])]


def plot_psad(t: pd.DataFrame, out: str) -> str:
    tasks = [("wg", "Whole-gland model"), ("zones", "Zones model (gland union)")]
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.4), sharey=True, layout="constrained")
    for ax, (task, title) in zip(axes, tasks):
        sub = t[(t["task"] == task) & np.isclose(t["epsilon"], 0.06)]
        arms = _arms(sub)
        data = [
            np.log2(sub.loc[sub["arm"] == a, "psad_factor"].clip(1e-3)) for a in arms
        ]
        bp = ax.boxplot(data, patch_artist=True, widths=0.6, showfliers=False)
        for patch, a in zip(bp["boxes"], arms):
            patch.set_facecolor(ARM_COLOUR[a])
            patch.set_edgecolor("#1a1a1a")
            if a in HATCH:
                patch.set_facecolor("white")
                patch.set_edgecolor(ARM_COLOUR[a])
                patch.set_hatch(HATCH[a])
        for med in bp["medians"]:
            med.set_color("#1a1a1a")
        ax.axhline(0, color="#6b6b6b", linewidth=0.8)
        ax.axhspan(
            np.log2(1 / 1.2), np.log2(1.2), color="#6b6b6b", alpha=0.12, zorder=0
        )
        ax.set_xticks(range(1, len(arms) + 1))
        ax.set_xticklabels(
            [ARM_LABELS[a] for a in arms], rotation=35, ha="right", fontsize=8
        )
        ax.set_yticks([-3, -2, -1, 0, 1, 2, 3])
        ax.set_yticklabels(["1/8", "1/4", "1/2", "1", "2", "4", "8"])
        ax.set_title(title, fontsize=11, fontweight="bold")
        ax.grid(axis="y", alpha=0.3)
        ax.spines[["top", "right"]].set_visible(False)
    axes[0].set_ylabel("PSA density, attacked / clean\n(= V_clean / V_adv, log scale)")
    fig.suptitle(
        "PSA-density misestimation per attack at eps = 0.06 (60 cases; shaded = within 20%)",
        fontsize=12,
        fontweight="bold",
    )
    path = os.path.join(out, "psad_factor_eps0.06.png")
    fig.savefig(path, dpi=300, bbox_inches="tight")
    plt.close(fig)
    return path


def plot_qc(qc: pd.DataFrame, out: str) -> str:
    tasks = [("wg", "Whole-gland model"), ("zones", "Zones model")]
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.2), sharey=True, layout="constrained")
    for ax, (task, title) in zip(axes, tasks):
        sub = qc[qc["task"] == task]
        for arm in _arms(sub):
            r = sub[sub["arm"] == arm].sort_values("epsilon")
            ax.plot(
                r["epsilon"],
                100 * r["undetected_damage_share"],
                marker="o",
                markersize=7,
                markerfacecolor="white" if arm in LINESTYLE else ARM_COLOUR[arm],
                markeredgewidth=1.6,
                color=ARM_COLOUR[arm],
                linestyle=LINESTYLE.get(arm, "-"),
                linewidth=1.8,
                label=ARM_LABELS[arm],
            )
        ax.set_title(title, fontsize=11, fontweight="bold")
        ax.set_xlabel(r"$\varepsilon$")
        ax.grid(alpha=0.3)
        ax.spines[["top", "right"]].set_visible(False)
    axes[0].set_ylabel(
        "share of all 60 cases that are damaged\n(Dice drop >= 0.10) and pass volume QC (%)"
    )
    axes[1].legend(fontsize=7, ncol=2, loc="upper left", handlelength=3)
    fig.suptitle(
        "Undetected damage: attacked outputs that a volume-based QC would accept",
        fontsize=12,
        fontweight="bold",
    )
    path = os.path.join(out, "qc_undetected_damage.png")
    fig.savefig(path, dpi=300, bbox_inches="tight")
    plt.close(fig)
    return path


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--output-dir", default=os.path.join(ROOT, "clinical"))
    args = parser.parse_args()
    os.makedirs(args.output_dir, exist_ok=True)

    gt = {
        task: ground_truth_volumes(
            task, os.path.join(args.output_dir, f"gt_volumes_{task}.csv")
        )
        for task in LABEL_DIRS
    }
    ranges = qc_ranges(gt)
    pd.DataFrame(
        [
            {
                "task": k,
                "rule": r,
                "low": v[0],
                "high": v[1],
                "n_reference_cases": len(gt[k]),
            }
            for k, rr in ranges.items()
            for r, v in rr.items()
        ]
    ).to_csv(os.path.join(args.output_dir, "qc_ranges.csv"), index=False)

    d = pd.concat(
        [
            pd.read_csv(f, low_memory=False)
            for f in glob.glob(os.path.join(ROOT, "direction", "summary_*.csv"))
        ]
    )
    t = per_case_table(d)
    t["qc_clean"] = passes_qc(t, "clean", ranges)
    t["qc_adv"] = passes_qc(t, "adv", ranges)
    t.to_csv(os.path.join(args.output_dir, "clinical_per_case.csv"), index=False)
    psad, qc = summarize(t)
    psad.to_csv(os.path.join(args.output_dir, "psad_factor_summary.csv"), index=False)
    qc.to_csv(os.path.join(args.output_dir, "qc_evasion_summary.csv"), index=False)
    for p in (plot_psad(t, args.output_dir), plot_qc(qc, args.output_dir)):
        print("wrote", p)
    print(pd.DataFrame(ranges).to_string())


if __name__ == "__main__":
    main()
