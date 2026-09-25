"""A2 + A3: PSA-density decisions under attack, and the clean-volume check.

Joins the whole-gland replay volumes (``results/blade_attack/mechanism/direction``,
9 arms, 60 fold-0 cases, nnU-Net, 20 steps) with PI-CAI clinical data
(``data_external/picai/wg_case_clinical.csv``; ``ProstateWG_<n>`` = PI-CAI
patient ``<n>``, see that folder's README). Writes to
``results/blade_attack/mechanism/clinical/``.

A3 -- clean-volume validation
    Clean-model gland volume vs PI-CAI ``prostate_volume`` (radiologist
    estimate) on the linked replay cases, with the ground-truth volume vs PI-CAI
    on all linked cases as the reference agreement.

A2 -- decision flips
    PSA density = PSA / gland volume (cm^3), computed from the clean and from
    the attacked segmentation (PI-CAI's own ``psad`` is missing for ~30% of
    cases, so it is recomputed consistently). A flip is a crossing of the
    biopsy threshold (``--thresholds``, default 0.10 / 0.15 / 0.20 ng/mL/cc):
    ``high_to_low`` (clean at or above, attacked below: an indication lost) and
    ``low_to_high`` (clean below, attacked at or above: an unnecessary
    indication). Reported per (arm, epsilon, threshold) over all linked cases
    with PSA and, separately, over csPCa-positive patients (``case_csPCa`` =
    YES), where losing the indication is the harmful direction.

Usage::

    PYTHONPATH=src python experiments/analyze_psad_decisions.py
"""

from __future__ import annotations

import argparse
import glob
import os
import sys

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from scipy import stats  # noqa: E402

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from analyze_clinical_proxies import ARM_COLOUR, ARM_ORDER, LINESTYLE  # noqa: E402
from run_twelve_arm_wg_campaign import ARM_LABELS  # noqa: E402

ROOT = os.path.join("results", "blade_attack", "mechanism")
CLINICAL = os.path.join("data_external", "picai", "wg_case_clinical.csv")
# BLADE-3 and BLADE-MM coincide often: filled square under a hollow diamond.
MARKER = {
    "blade": dict(marker="s", markersize=6),
    "blade_mm": dict(marker="D", markersize=8, markerfacecolor="none", linestyle="--"),
}


def load(clinical_path: str) -> tuple[pd.DataFrame, pd.DataFrame]:
    clin = pd.read_csv(clinical_path)
    d = pd.concat(
        [
            pd.read_csv(f, low_memory=False)
            for f in glob.glob(os.path.join(ROOT, "direction", "summary_*.csv"))
        ]
    )
    d = d[(d["task"] == "wg") & (d["target"] == "WG")]
    d = d[["arm", "case_id", "epsilon", "clean_volume_mm3", "adv_volume_mm3"]]
    linked = clin[clin["n_studies"] == 1]
    return d.merge(linked, on="case_id", how="inner"), clin


def clean_volume_check(t: pd.DataFrame, clin: pd.DataFrame) -> pd.DataFrame:
    rows = []
    c = t.drop_duplicates("case_id").dropna(subset=["prostate_volume"])
    g = clin[(clin["n_studies"] == 1)].dropna(subset=["prostate_volume"])
    for name, x, y, n in (
        (
            "clean model vs PI-CAI (replay subset)",
            c["clean_volume_mm3"] / 1000,
            c["prostate_volume"],
            len(c),
        ),
        (
            "ground truth vs PI-CAI (all linked)",
            g["gt_gland_mm3"] / 1000,
            g["prostate_volume"],
            len(g),
        ),
        (
            "clean model vs ground truth (replay subset)",
            c["clean_volume_mm3"] / 1000,
            c["gt_gland_mm3"] / 1000,
            len(c),
        ),
    ):
        ratio = np.asarray(x) / np.asarray(y)
        rows.append(
            {
                "comparison": name,
                "n_cases": n,
                "spearman": stats.spearmanr(x, y).correlation,
                "pearson": stats.pearsonr(x, y)[0],
                "median_ratio": float(np.median(ratio)),
                "ratio_q25": float(np.percentile(ratio, 25)),
                "ratio_q75": float(np.percentile(ratio, 75)),
            }
        )
    return pd.DataFrame(rows)


def flips(t: pd.DataFrame, thresholds: list[float]) -> pd.DataFrame:
    t = t.dropna(subset=["psa"]).copy()
    t["psad_clean"] = t["psa"] / (t["clean_volume_mm3"] / 1000)
    t["psad_adv"] = t["psa"] / (t["adv_volume_mm3"].clip(lower=1.0) / 1000)
    rows = []
    for (arm, eps), g in t.groupby(["arm", "epsilon"]):
        for thr in thresholds:
            for cohort, sub in (("all", g), ("csPCa", g[g["case_csPCa"] == "YES"])):
                hi_c = sub["psad_clean"] >= thr
                hi_a = sub["psad_adv"] >= thr
                rows.append(
                    {
                        "arm": arm,
                        "epsilon": eps,
                        "threshold": thr,
                        "cohort": cohort,
                        "n_cases": len(sub),
                        "n_above_clean": int(hi_c.sum()),
                        "high_to_low": int((hi_c & ~hi_a).sum()),
                        "low_to_high": int((~hi_c & hi_a).sum()),
                        "share_indications_lost": (hi_c & ~hi_a).sum()
                        / max(int(hi_c.sum()), 1),
                        "share_any_flip": ((hi_c != hi_a).mean()),
                    }
                )
    return pd.DataFrame(rows)


def plot(f: pd.DataFrame, out: str, thr: float) -> str:
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.2), sharey=True, layout="constrained")
    for ax, cohort, title in (
        (axes[0], "all", "All linked patients with PSA"),
        (axes[1], "csPCa", "Patients with clinically significant cancer"),
    ):
        sub = f[(f["cohort"] == cohort) & np.isclose(f["threshold"], thr)]
        for arm in [a for a in ARM_ORDER if a in set(sub["arm"])]:
            r = sub[sub["arm"] == arm].sort_values("epsilon")
            style = dict(
                marker="o",
                markersize=6,
                markeredgewidth=1.4,
                markerfacecolor="white" if arm in LINESTYLE else ARM_COLOUR[arm],
                color=ARM_COLOUR[arm],
                linestyle=LINESTYLE.get(arm, "-"),
                linewidth=1.8,
                label=ARM_LABELS[arm],
            )
            style.update(MARKER.get(arm, {}))
            ax.plot(r["epsilon"], 100 * r["share_indications_lost"], **style)
        n = int(sub["n_above_clean"].iloc[0]) if len(sub) else 0
        ax.set_title(
            f"{title}\n({n} above threshold on the clean segmentation)",
            fontsize=10,
            fontweight="bold",
        )
        ax.set_xlabel(r"$\varepsilon$")
        ax.grid(alpha=0.3)
        ax.spines[["top", "right"]].set_visible(False)
    axes[0].set_ylabel(
        f"biopsy indications lost (%)\nPSA density drops below {thr:g} ng/mL/cc"
    )
    axes[1].legend(fontsize=7, ncol=2, loc="upper left", handlelength=3)
    fig.suptitle(
        "PSA-density decision flips caused by attacking the whole-gland segmentation",
        fontsize=12,
        fontweight="bold",
    )
    path = os.path.join(out, f"psad_indications_lost_thr{thr:g}.png")
    fig.savefig(path, dpi=300, bbox_inches="tight")
    plt.close(fig)
    return path


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--clinical", default=CLINICAL)
    parser.add_argument("--output-dir", default=os.path.join(ROOT, "clinical"))
    parser.add_argument(
        "--thresholds", type=float, nargs="+", default=[0.10, 0.15, 0.20]
    )
    args = parser.parse_args()
    t, clin = load(args.clinical)
    t.to_csv(os.path.join(args.output_dir, "psad_linked_per_case.csv"), index=False)
    v = clean_volume_check(t, clin)
    v.to_csv(os.path.join(args.output_dir, "clean_volume_validation.csv"), index=False)
    f = flips(t, args.thresholds)
    f.to_csv(os.path.join(args.output_dir, "psad_decision_flips.csv"), index=False)
    print("wrote", plot(f, args.output_dir, 0.15))
    print(v.round(3).to_string(index=False))


if __name__ == "__main__":
    main()
