"""C1: radiotherapy-style geometric consequences of the attacked gland masks.

Reads the masks saved by ``run_blade_mechanism_study.py --save-masks``
(``results/blade_attack/mechanism/masks``: per case a reference file with the
ground truth, the clean prediction and the voxel spacing, and one file per
arm x epsilon with the attacked prediction) and writes
``results/blade_attack/mechanism/rt_geometry/``.

The planning target is built the way a planner would from a contour: the gland
(whole-gland label, or the zone union for the zones model) dilated by an
isotropic ``--margin-mm`` margin in physical space (anisotropic voxel spacing
handled by the distance transform). Per (model, arm, case, epsilon):

* ``gt_coverage``: share of the ground-truth gland inside the target built from
  the attacked contour (below 0.95 = part of the true gland would be left
  outside the treated volume, i.e. under-dosing risk). The same number for the
  clean contour is the no-attack baseline.
* ``ptv_inflation``: target volume from the attacked contour / from the clean
  contour.
* ``excess_normal_mm3``: target volume outside the ideal target (ground truth
  + margin), i.e. tissue that would be treated without need.
* ``hd95_mm``: symmetric 95th-percentile surface distance, attacked vs clean
  gland; ``centroid_shift_mm``: attacked vs clean gland centroid.

Only voxels with a valid label (not the -1 ignore region of the zones crops)
are used. Geometric proxies only: no dose, no organs at risk, no lesion
targets (the data has no lesion masks).

Usage::

    PYTHONPATH=src python experiments/analyze_rt_geometry.py
"""

from __future__ import annotations

import argparse
import glob
import os
import re
import sys
from multiprocessing import Pool

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from scipy.ndimage import binary_erosion, distance_transform_edt  # noqa: E402

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from analyze_clinical_proxies import ARM_COLOUR, ARM_ORDER, HATCH, LINESTYLE  # noqa: E402
from run_twelve_arm_wg_campaign import ARM_LABELS  # noqa: E402

ROOT = os.path.join("results", "blade_attack", "mechanism")
COVERAGE_OK = 0.95
TAIL_RE = re.compile(r"^(.+)_steps(\d+)_eps([0-9.]+)\.npz$")


def gland_of(labels: np.ndarray) -> np.ndarray:
    return labels >= 1


def ptv(mask: np.ndarray, spacing: np.ndarray, margin: float) -> np.ndarray:
    if not mask.any():
        return mask.copy()
    return distance_transform_edt(~mask, sampling=spacing) <= margin


def _surface(mask: np.ndarray) -> np.ndarray:
    return mask & ~binary_erosion(mask)


def hd95(a: np.ndarray, b: np.ndarray, spacing: np.ndarray) -> float:
    if not a.any() or not b.any():
        return float("nan")
    sa, sb = _surface(a), _surface(b)
    da = distance_transform_edt(~sb, sampling=spacing)[sa]
    db = distance_transform_edt(~sa, sampling=spacing)[sb]
    return float(np.percentile(np.concatenate([da, db]), 95))


def centroid_mm(mask: np.ndarray, spacing: np.ndarray) -> np.ndarray:
    idx = np.argwhere(mask)
    return idx.mean(axis=0) * spacing if len(idx) else np.full(3, np.nan)


def metrics(
    pred: np.ndarray, gt: np.ndarray, gt_ptv: np.ndarray, spacing, margin
) -> dict:
    vox = float(np.prod(spacing))
    target = ptv(pred, spacing, margin)
    n_gt = max(int(gt.sum()), 1)
    return {
        "gland_mm3": float(pred.sum()) * vox,
        "ptv_mm3": float(target.sum()) * vox,
        "gt_coverage": float((gt & target).sum()) / n_gt,
        "excess_normal_mm3": float((target & ~gt_ptv).sum()) * vox,
        "_target": target,
    }


def analyse_case(args: tuple) -> list[dict]:
    task, case_id, ref_path, adv_paths, margin = args
    ref = np.load(ref_path)
    labels, spacing = ref["gt"], ref["spacing"]
    valid = labels != -1
    gt = gland_of(labels) & valid
    clean = gland_of(ref["clean"]) & valid
    gt_ptv = ptv(gt, spacing, margin)
    base = metrics(clean, gt, gt_ptv, spacing, margin)
    c_centroid = centroid_mm(clean, spacing)
    rows = []
    for path, arm, steps, eps in adv_paths:
        adv = gland_of(np.load(path)["adv"]) & valid
        a = metrics(adv, gt, gt_ptv, spacing, margin)
        rows.append(
            {
                "task": task,
                "case_id": case_id,
                "arm": arm,
                "steps": int(steps),
                "epsilon": float(eps),
                "margin_mm": margin,
                "clean_gt_coverage": base["gt_coverage"],
                "adv_gt_coverage": a["gt_coverage"],
                "clean_ptv_mm3": base["ptv_mm3"],
                "adv_ptv_mm3": a["ptv_mm3"],
                "ptv_inflation": a["ptv_mm3"] / max(base["ptv_mm3"], 1e-9),
                "clean_excess_normal_mm3": base["excess_normal_mm3"],
                "adv_excess_normal_mm3": a["excess_normal_mm3"],
                "hd95_mm": hd95(adv, clean, spacing),
                "centroid_shift_mm": float(
                    np.linalg.norm(centroid_mm(adv, spacing) - c_centroid)
                ),
            }
        )
    return rows


def collect(mask_dir: str, margin: float, workers: int) -> pd.DataFrame:
    """One job per case; attacked masks found by stripping the case prefix."""
    jobs = []
    for ref in sorted(glob.glob(os.path.join(mask_dir, "*_reference.npz"))):
        stem = os.path.basename(ref)[: -len("_reference.npz")]
        task, case_id = stem.split("_", 1)
        advs = []
        for p in glob.glob(os.path.join(mask_dir, f"{stem}_*_steps*_eps*.npz")):
            m = TAIL_RE.match(os.path.basename(p)[len(stem) + 1 :])
            if m and m.group(1) in ARM_ORDER:
                advs.append((p, m.group(1), int(m.group(2)), float(m.group(3))))
        jobs.append((task, case_id, ref, sorted(advs), margin))
    with Pool(workers) as pool:
        parts = pool.map(analyse_case, jobs)
    return pd.DataFrame([r for part in parts for r in part])


def summarize(t: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for (task, arm, eps), g in t.groupby(["task", "arm", "epsilon"]):
        rows.append(
            {
                "task": task,
                "arm": arm,
                "epsilon": eps,
                "n_cases": g["case_id"].nunique(),
                "median_adv_gt_coverage": g["adv_gt_coverage"].median(),
                "median_clean_gt_coverage": g["clean_gt_coverage"].median(),
                "share_adv_coverage_below_95": (
                    g["adv_gt_coverage"] < COVERAGE_OK
                ).mean(),
                "share_clean_coverage_below_95": (
                    g["clean_gt_coverage"] < COVERAGE_OK
                ).mean(),
                "median_ptv_inflation": g["ptv_inflation"].median(),
                "share_ptv_inflation_over_1p5": (g["ptv_inflation"] > 1.5).mean(),
                "median_excess_normal_cm3": g["adv_excess_normal_mm3"].median() / 1000,
                "median_clean_excess_normal_cm3": g["clean_excess_normal_mm3"].median()
                / 1000,
                "median_hd95_mm": g["hd95_mm"].median(),
                "median_centroid_shift_mm": g["centroid_shift_mm"].median(),
            }
        )
    return pd.DataFrame(rows)


def _arms(t: pd.DataFrame) -> list[str]:
    return [a for a in ARM_ORDER if a in set(t["arm"])]


def plot_coverage(s: pd.DataFrame, out: str, margin: float) -> str:
    tasks = [("wg", "Whole-gland model"), ("zones", "Zones model (gland union)")]
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.2), sharey=True, layout="constrained")
    for ax, (task, title) in zip(axes, tasks):
        sub = s[s["task"] == task]
        for arm in _arms(sub):
            r = sub[sub["arm"] == arm].sort_values("epsilon")
            ax.plot(
                r["epsilon"],
                100 * r["share_adv_coverage_below_95"],
                marker="o",
                markersize=7,
                markeredgewidth=1.6,
                markerfacecolor="white" if arm in LINESTYLE else ARM_COLOUR[arm],
                color=ARM_COLOUR[arm],
                linestyle=LINESTYLE.get(arm, "-"),
                linewidth=1.8,
                label=ARM_LABELS[arm],
            )
        clean = 100 * sub["share_clean_coverage_below_95"].iloc[0]
        ax.axhline(
            clean, color="#6b6b6b", linewidth=1, linestyle=":", label="clean prediction"
        )
        ax.set_title(title, fontsize=11, fontweight="bold")
        ax.set_xlabel(r"$\varepsilon$")
        ax.grid(alpha=0.3)
        ax.spines[["top", "right"]].set_visible(False)
    axes[0].set_ylabel(
        f"cases whose target (contour + {margin:g} mm)\n"
        "misses > 5% of the true gland (%)"
    )
    axes[1].legend(fontsize=7, ncol=2, loc="upper left", handlelength=3)
    fig.suptitle(
        "Under-dosing risk: true gland left outside the planning target",
        fontsize=12,
        fontweight="bold",
    )
    path = os.path.join(out, "rt_coverage_failure.png")
    fig.savefig(path, dpi=300, bbox_inches="tight")
    plt.close(fig)
    return path


def plot_inflation(t: pd.DataFrame, out: str, margin: float) -> str:
    tasks = [("wg", "Whole-gland model"), ("zones", "Zones model (gland union)")]
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.4), sharey=True, layout="constrained")
    for ax, (task, title) in zip(axes, tasks):
        sub = t[(t["task"] == task) & np.isclose(t["epsilon"], 0.06)]
        arms = _arms(sub)
        data = [
            np.log2(sub.loc[sub["arm"] == a, "ptv_inflation"].clip(1e-3)) for a in arms
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
        ax.set_xticks(range(1, len(arms) + 1))
        ax.set_xticklabels(
            [ARM_LABELS[a] for a in arms], rotation=35, ha="right", fontsize=8
        )
        ax.set_yticks([-2, -1, 0, 1, 2, 3])
        ax.set_yticklabels(["1/4", "1/2", "1", "2", "4", "8"])
        ax.set_title(title, fontsize=11, fontweight="bold")
        ax.grid(axis="y", alpha=0.3)
        ax.spines[["top", "right"]].set_visible(False)
    axes[0].set_ylabel(
        f"planning-target volume, attacked / clean\n(contour + {margin:g} mm, log scale)"
    )
    fig.suptitle(
        "Planning-target inflation at eps = 0.06 (60 cases per model)",
        fontsize=12,
        fontweight="bold",
    )
    path = os.path.join(out, "rt_ptv_inflation_eps0.06.png")
    fig.savefig(path, dpi=300, bbox_inches="tight")
    plt.close(fig)
    return path


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--mask-dir", default=os.path.join(ROOT, "masks"))
    parser.add_argument("--output-dir", default=os.path.join(ROOT, "rt_geometry"))
    parser.add_argument("--margin-mm", type=float, default=5.0)
    parser.add_argument("--workers", type=int, default=16)
    args = parser.parse_args()
    os.makedirs(args.output_dir, exist_ok=True)
    t = collect(args.mask_dir, args.margin_mm, args.workers)
    t.to_csv(os.path.join(args.output_dir, "rt_geometry_per_case.csv"), index=False)
    s = summarize(t)
    s.to_csv(os.path.join(args.output_dir, "rt_geometry_summary.csv"), index=False)
    for p in (
        plot_coverage(s, args.output_dir, args.margin_mm),
        plot_inflation(t, args.output_dir, args.margin_mm),
    ):
        print("wrote", p)
    print(t.groupby(["task", "arm"]).case_id.nunique().unstack("task").to_string())


if __name__ == "__main__":
    main()
