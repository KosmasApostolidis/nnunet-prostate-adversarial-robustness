"""D+: topology QC rules on the saved masks, added to the volume QC of D.

Reads ``results/blade_attack/mechanism/masks`` (clean, attacked and
ground-truth argmax masks, 9 arms x 60 fold-0 cases x 2 models, 20 steps) and
the volume-QC ranges written by ``analyze_clinical_proxies.py``. Writes
``results/blade_attack/mechanism/topology_qc/``.

A QC designer sees only the output. Rules, each applied to the gland (and, for
the zones model, to each zone), using 26-connectivity and only valid voxels:

* ``volume``      -- D's rule: gland volume (and TZ+CZ share) inside the
                     dataset's [1st, 99th] ground-truth percentiles.
* ``components``  -- at most one connected component of at least
                     ``--min-component-cm3`` (a prostate, and each zone, is one
                     object; detached islands are implausible).
* ``z_contiguous``-- the gland occupies a contiguous run of slices (no empty
                     slice between its first and last slice).
* ``nested``      -- zones model only: every TZ+CZ component shares a boundary
                     with PZ (the transition zone is not detached from the
                     peripheral zone).

Each rule's false-alarm rate is measured on the clean predictions and on the
ground truth. A case is "damaged" when a target's Dice falls by >= 0.10 from
the clean prediction (zones: either zone). Per (model, arm, epsilon) the
output reports the share of damaged cases each rule catches and the share of
all cases that are damaged yet pass volume-only QC vs volume + topology QC.

Usage::

    PYTHONPATH=src python experiments/analyze_topology_qc.py
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
from scipy.ndimage import binary_dilation, generate_binary_structure, label  # noqa: E402

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from analyze_clinical_proxies import ARM_COLOUR, ARM_ORDER, LINESTYLE  # noqa: E402
from run_twelve_arm_wg_campaign import ARM_LABELS  # noqa: E402

ROOT = os.path.join("results", "blade_attack", "mechanism")
TAIL_RE = re.compile(r"^(.+)_steps(\d+)_eps([0-9.]+)\.npz$")
STRUCT = generate_binary_structure(3, 3)  # 26-connectivity
DAMAGE_DROP = 0.10


def dice(a: np.ndarray, b: np.ndarray) -> float:
    denom = int(a.sum()) + int(b.sum())
    return 2.0 * int((a & b).sum()) / denom if denom else 1.0


def n_big_components(mask: np.ndarray, min_vox: int) -> int:
    if not mask.any():
        return 0
    lab, n = label(mask, structure=STRUCT)
    sizes = np.bincount(lab.ravel())[1:]
    return int((sizes >= min_vox).sum())


def z_contiguous(mask: np.ndarray) -> bool:
    z = np.flatnonzero(mask.any(axis=(1, 2)))
    return bool(len(z)) and (z[-1] - z[0] + 1 == len(z))


def zones_touch(tz: np.ndarray, pz: np.ndarray) -> bool:
    """Every TZ+CZ component of any size shares a boundary with PZ."""
    if not tz.any() or not pz.any():
        return False
    pz_ring = binary_dilation(pz, structure=STRUCT)
    lab, n = label(tz, structure=STRUCT)
    touching = np.unique(lab[pz_ring & tz])
    return int((touching > 0).sum()) == n


def features(labels: np.ndarray, task: str, spacing: np.ndarray, min_vox: int) -> dict:
    """QC-relevant features of one label map (prediction or ground truth)."""
    gland = labels >= 1
    vox = float(np.prod(spacing))
    out = {
        "gland_mm3": float(gland.sum()) * vox,
        "gland_components": n_big_components(gland, min_vox),
        "gland_z_contiguous": z_contiguous(gland),
    }
    if task == "zones":
        tz, pz = labels == 1, labels == 2
        out["tzcz_fraction"] = float(tz.sum()) / max(float(gland.sum()), 1.0)
        out["tzcz_components"] = n_big_components(tz, min_vox)
        out["pz_components"] = n_big_components(pz, min_vox)
        out["zones_touch"] = zones_touch(tz, pz)
    return out


def analyse_case(job: tuple) -> list[dict]:
    task, case_id, ref_path, advs, min_cm3 = job
    ref = np.load(ref_path)
    gt, spacing = ref["gt"], ref["spacing"]
    valid = gt != -1
    min_vox = max(1, int(round(min_cm3 * 1000 / float(np.prod(spacing)))))

    def clip(x: np.ndarray) -> np.ndarray:
        return np.where(valid, x, 0)

    clean = clip(ref["clean"].astype(np.int16))
    targets = {"wg": [("WG", 1)], "zones": [("TZ+CZ", 1), ("PZ", 2)]}[task]
    clean_dice = {n: dice(clean == c, gt == c) for n, c in targets}
    rows = [
        {
            "task": task,
            "case_id": case_id,
            "arm": "__gt__",
            "epsilon": 0.0,
            **features(clip(gt), task, spacing, min_vox),
        },
        {
            "task": task,
            "case_id": case_id,
            "arm": "__clean__",
            "epsilon": 0.0,
            **features(clean, task, spacing, min_vox),
        },
    ]
    for path, arm, _steps, eps in advs:
        adv = clip(np.load(path)["adv"].astype(np.int16))
        drop = max(clean_dice[n] - dice(adv == c, gt == c) for n, c in targets)
        rows.append(
            {
                "task": task,
                "case_id": case_id,
                "arm": arm,
                "epsilon": eps,
                "dice_drop": drop,
                **features(adv, task, spacing, min_vox),
            }
        )
    return rows


def collect(mask_dir: str, min_cm3: float, workers: int) -> pd.DataFrame:
    jobs = []
    for ref in sorted(glob.glob(os.path.join(mask_dir, "*_reference.npz"))):
        stem = os.path.basename(ref)[: -len("_reference.npz")]
        task, case_id = stem.split("_", 1)
        advs = []
        for p in glob.glob(os.path.join(mask_dir, f"{stem}_*_steps*_eps*.npz")):
            m = TAIL_RE.match(os.path.basename(p)[len(stem) + 1 :])
            if m and m.group(1) in ARM_ORDER:
                advs.append((p, m.group(1), int(m.group(2)), float(m.group(3))))
        jobs.append((task, case_id, ref, sorted(advs), min_cm3))
    with Pool(workers) as pool:
        parts = pool.map(analyse_case, jobs)
    return pd.DataFrame([r for part in parts for r in part])


def apply_rules(t: pd.DataFrame, ranges: pd.DataFrame) -> pd.DataFrame:
    t = t.copy()
    rng = {(r.task, r.rule): (r.low, r.high) for r in ranges.itertuples()}
    vol_ok = pd.Series(True, index=t.index)
    for task in ("wg", "zones"):
        m = t["task"] == task
        lo, hi = rng[(task, "gland_mm3")]
        vol_ok[m] = t.loc[m, "gland_mm3"].between(lo, hi)
        if (task, "tzcz_fraction") in rng:
            flo, fhi = rng[(task, "tzcz_fraction")]
            vol_ok[m] &= t.loc[m, "tzcz_fraction"].between(flo, fhi)
    t["rule_volume"] = vol_ok
    comp = t["gland_components"] == 1
    zones = t["task"] == "zones"
    comp[zones] &= (t.loc[zones, "tzcz_components"] <= 1) & (
        t.loc[zones, "pz_components"] <= 1
    )
    t["rule_components"] = comp
    t["rule_z_contiguous"] = t["gland_z_contiguous"].astype(bool)
    nested = pd.Series(True, index=t.index)
    nested[zones] = t.loc[zones, "zones_touch"].astype(bool)
    t["rule_nested"] = nested
    for r in ("rule_volume", "rule_components", "rule_z_contiguous", "rule_nested"):
        t[r] = t[r].astype(bool)
    t["pass_volume"] = t["rule_volume"]
    t["pass_all"] = t[
        ["rule_volume", "rule_components", "rule_z_contiguous", "rule_nested"]
    ].all(axis=1)
    return t


RULES = (
    "rule_volume",
    "rule_components",
    "rule_z_contiguous",
    "rule_nested",
    "pass_all",
)


def false_alarms(t: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for (task, arm), g in t[t["arm"].isin(["__gt__", "__clean__"])].groupby(
        ["task", "arm"]
    ):
        rows.append(
            {
                "task": task,
                "reference": arm.strip("_"),
                "n_cases": len(g),
                **{f"fail_{r}": 1 - g[r].mean() for r in RULES},
            }
        )
    return pd.DataFrame(rows)


def summarize(t: pd.DataFrame) -> pd.DataFrame:
    rows = []
    att = t[~t["arm"].str.startswith("__")]
    for (task, arm, eps), g in att.groupby(["task", "arm", "epsilon"]):
        dmg = g["dice_drop"] >= DAMAGE_DROP
        row = {
            "task": task,
            "arm": arm,
            "epsilon": eps,
            "n_cases": len(g),
            "n_damaged": int(dmg.sum()),
            "undetected_volume_only": (dmg & g["pass_volume"]).mean(),
            "undetected_volume_plus_topology": (dmg & g["pass_all"]).mean(),
        }
        for r in RULES:
            row[f"catch_{r}"] = (~g.loc[dmg, r]).mean() if dmg.any() else np.nan
        rows.append(row)
    return pd.DataFrame(rows)


def plot(s: pd.DataFrame, out: str) -> str:
    tasks = [("wg", "Whole-gland model"), ("zones", "Zones model")]
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.2), sharey=True, layout="constrained")
    for ax, (task, title) in zip(axes, tasks):
        sub = s[s["task"] == task]
        for arm in [a for a in ARM_ORDER if a in set(sub["arm"])]:
            r = sub[sub["arm"] == arm].sort_values("epsilon")
            style = dict(
                color=ARM_COLOUR[arm],
                linestyle=LINESTYLE.get(arm, "-"),
                marker="o",
                markersize=6,
                markeredgewidth=1.4,
                markerfacecolor="white" if arm in LINESTYLE else ARM_COLOUR[arm],
                linewidth=1.8,
                label=ARM_LABELS[arm],
            )
            ax.plot(r["epsilon"], 100 * r["undetected_volume_plus_topology"], **style)
        ax.set_title(title, fontsize=11, fontweight="bold")
        ax.set_xlabel(r"$\varepsilon$")
        ax.grid(alpha=0.3)
        ax.spines[["top", "right"]].set_visible(False)
    axes[0].set_ylabel(
        "share of all 60 cases that are damaged\nand pass volume + topology QC (%)"
    )
    axes[1].legend(fontsize=7, ncol=2, loc="upper left", handlelength=3)
    fig.suptitle(
        "Undetected damage after adding topology rules to the volume QC",
        fontsize=12,
        fontweight="bold",
    )
    path = os.path.join(out, "topology_qc_undetected_damage.png")
    fig.savefig(path, dpi=300, bbox_inches="tight")
    plt.close(fig)
    return path


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--mask-dir", default=os.path.join(ROOT, "masks"))
    parser.add_argument(
        "--ranges", default=os.path.join(ROOT, "clinical", "qc_ranges.csv")
    )
    parser.add_argument("--output-dir", default=os.path.join(ROOT, "topology_qc"))
    parser.add_argument("--min-component-cm3", type=float, default=0.5)
    parser.add_argument("--workers", type=int, default=16)
    args = parser.parse_args()
    os.makedirs(args.output_dir, exist_ok=True)
    t = apply_rules(
        collect(args.mask_dir, args.min_component_cm3, args.workers),
        pd.read_csv(args.ranges),
    )
    t.to_csv(os.path.join(args.output_dir, "topology_qc_per_case.csv"), index=False)
    fa = false_alarms(t)
    fa.to_csv(
        os.path.join(args.output_dir, "topology_qc_false_alarms.csv"), index=False
    )
    s = summarize(t)
    s.to_csv(os.path.join(args.output_dir, "topology_qc_summary.csv"), index=False)
    print("wrote", plot(s, args.output_dir))
    print(fa.round(3).to_string(index=False))


if __name__ == "__main__":
    main()
