"""Budget-matched tier-300 statistics: BLADE-3 / BLADE-MM vs SEA / Auto-PGD x3.

Family (the one ``whole_gland/native_per_eps/PROTOCOL.md`` certifies as
budget-matched): ``blade`` and ``blade_mm`` against ``sea`` and
``auto_pgd_r3`` -- four arms that each spend 300 gradient evaluations over the
whole epsilon curve, so a win cannot be bought with extra compute. Run on the
whole gland (nnU-Net, ``whole_gland/native_per_eps``) and on TZ+CZ and PZ
(nnU-Net, confined zones campaign ``zones_confined_mc/campaign``), all three
metrics. Writes ``results/blade_attack/cross_dataset/blade_tier300_*.csv``.

  (0) Holm-Bonferroni over the four contrasts, on the raw budget-averaged
      Wilcoxon p-values of the existing pairwise tables (nothing re-tested):
      per dataset x metric (m = 4) and pooled (m = 36);
  (A) paired Wilcoxon separately at every epsilon, Holm over the 20-test
      (4 contrasts x 5 epsilon) family within each dataset x metric;
  (B) patient-level area under the robustness curve, trapezoid over the full
      six-point grid normalised to a mean value, paired Wilcoxon + Holm(m=4);
  (C) linear mixed model with case as a random intercept -- an additive model
      for the epsilon-averaged arm contrast, and an arm x epsilon model whose
      likelihood-ratio test asks whether that contrast is constant in epsilon;
  (D) case-clustered percentile bootstrap CIs on the median paired difference.

Sign convention matches ``analyze_main_comparison_stats.py``: every effect is
multiplied by METRIC_DIRECTION so **positive always means the BLADE arm did
more damage**, on Dice as well as on the distance metrics.

Recovered from the 2026-09-02 scratch scripts (``holm.py``,
``tier300_deep_stats.py``) that produced the first version of these files from
the unconfined zones campaign; only the zones inputs differ.

Usage::

    PYTHONPATH=src python experiments/analyze_blade_tier300.py
"""

from __future__ import annotations

import argparse
import glob
import os
import time
import warnings

import numpy as np
import pandas as pd
from scipy import stats

warnings.filterwarnings("ignore")

ROOT = os.path.join("results", "blade_attack")
OUT_DIR = os.path.join(ROOT, "cross_dataset")
WG_DIR = os.path.join(ROOT, "whole_gland", "native_per_eps")
ZONES_DIR = os.path.join(ROOT, "zones_confined_mc", "campaign")

# dataset -> (per-arm results directory, class filter)
DATASETS = {
    "Whole gland": (WG_DIR, "WG"),
    "TZ+CZ": (ZONES_DIR, "TZ+CZ"),
    "PZ": (ZONES_DIR, "PZ"),
}
# dataset -> budget-averaged pairwise table (source of the raw p-values for Holm)
PAIRWISE = {
    "Whole gland": os.path.join(WG_DIR, "main_comparison_wg_budget_avg_pairwise.csv"),
    "TZ+CZ": os.path.join(
        ZONES_DIR, "main_comparison_zones_TZCZ_budget_avg_pairwise.csv"
    ),
    "PZ": os.path.join(ZONES_DIR, "main_comparison_zones_PZ_budget_avg_pairwise.csv"),
}

# Positive means "attacked harder", exactly as in analyze_main_comparison_stats.py.
METRIC_DIRECTION = {"dice": -1, "hd95": +1, "asd": +1}
METRIC_LABEL = {"dice": "Dice", "hd95": "HD95 (mm)", "asd": "ASD (mm)"}
ARM_LABEL = {
    "blade": "BLADE-3",
    "blade_mm": "BLADE-MM",
    "sea": "SEA",
    "auto_pgd_r3": "Auto-PGD x3",
}

EPS_GRID = [0.0, 0.02, 0.04, 0.06, 0.08, 0.10]
EPS_ATTACK = EPS_GRID[1:]
# A p-value below ~1e-308 underflows to a literal 0.0, which must never be
# reported as "p = 0". Clip to the smallest positive double and flag it.
TINY = 5e-324


# --------------------------------------------------------------------------- io


def load_arm(directory: str, arm: str, klass: str) -> pd.DataFrame:
    """All five folds of one arm, filtered to one class, epsilon de-fuzzed."""
    # The directory name is the arm; the file stem is not (auto_pgd_r3 writes
    # files stemmed "auto_pgd"), so glob on the suffix and let the directory
    # do the disambiguation.
    files = sorted(glob.glob(os.path.join(directory, arm, "*_per_sample.csv")))
    if len(files) != 5:
        raise SystemExit(f"{arm}: expected 5 fold files, found {len(files)}")
    frame = pd.concat((pd.read_csv(f) for f in files), ignore_index=True)
    frame = frame[frame["class"] == klass].copy()
    frame["epsilon"] = frame["epsilon"].round(2)
    seen = set(frame["epsilon"].unique())
    if seen != set(EPS_GRID):
        raise SystemExit(f"{arm}/{klass}: epsilon grid is {sorted(seen)}")
    dup = frame.duplicated(["case_id", "epsilon"]).sum()
    if dup:
        raise SystemExit(f"{arm}/{klass}: {dup} duplicated (case_id, epsilon) rows")
    return frame


def wide(frame: pd.DataFrame, metric: str) -> pd.DataFrame:
    """case_id x epsilon table of one metric."""
    return frame.pivot(index="case_id", columns="epsilon", values=metric).sort_index()


# ------------------------------------------------------------------ statistics


def holm(p: list[float]) -> list[float]:
    """Holm-Bonferroni step-down adjusted p-values, monotonicity enforced."""
    order = sorted(range(len(p)), key=lambda i: p[i])
    m, adj, running = len(p), [0.0] * len(p), 0.0
    for rank, i in enumerate(order):
        running = max(running, min(1.0, (m - rank) * p[i]))
        adj[i] = running
    return adj


def paired(diff: np.ndarray) -> dict:
    """Two-sided Wilcoxon signed-rank plus the effect sizes that go with it."""
    n = len(diff)
    nonzero = diff[diff != 0]
    if nonzero.size == 0:
        return dict(
            n=n, median_diff=0.0, p_raw=1.0, rank_biserial=0.0, wins=0, losses=0
        )
    _, p = stats.wilcoxon(diff, alternative="two-sided")
    ranks = stats.rankdata(np.abs(nonzero))
    pos = ranks[nonzero > 0].sum()
    neg = ranks[nonzero < 0].sum()
    return dict(
        n=n,
        median_diff=float(np.median(diff)),
        p_raw=float(p),
        rank_biserial=float((pos - neg) / (pos + neg)),
        wins=int((diff > 0).sum()),
        losses=int((diff < 0).sum()),
    )


def boot_ci(
    diff: np.ndarray, rng: np.random.Generator, n_boot: int
) -> tuple[float, float]:
    """Percentile CI on the median paired difference, resampling whole cases."""
    n = len(diff)
    meds = np.empty(n_boot)
    for lo in range(0, n_boot, 500):
        hi = min(lo + 500, n_boot)
        idx = rng.integers(0, n, size=(hi - lo, n))
        meds[lo:hi] = np.median(diff[idx], axis=1)
    return float(np.percentile(meds, 2.5)), float(np.percentile(meds, 97.5))


def mixed_models(
    frame: pd.DataFrame, metric: str, arms: list[str], pairs: list[tuple[str, str]]
) -> tuple[list[dict], dict]:
    """Random-intercept LMM on the attacked epsilons; contrasts + interaction LRT."""
    import statsmodels.formula.api as smf

    sign = METRIC_DIRECTION[metric]
    long = frame[frame.epsilon > 0].copy()
    # arms[0] is the reference level so its dummy drops out of the design.
    long["arm"] = pd.Categorical(long["arm"], categories=arms, ordered=False)
    long["eps"] = pd.Categorical(long["epsilon"])
    long = long.rename(columns={metric: "y"})

    # Both fits are ML, not REML -- the likelihood-ratio test below compares
    # models that differ in their fixed effects, where REML likelihoods are not
    # comparable. lbfgs drives the group variance to the boundary and returns a
    # singular Hessian on the 1,396-case whole-gland design; bfgs, powell, cg and
    # nm all agree on the same optimum there, so bfgs leads the fallback chain.
    opt = ["bfgs", "powell", "cg"]
    add = smf.mixedlm("y ~ C(arm) + C(eps)", long, groups=long["case_id"]).fit(
        reml=False, method=opt
    )
    inter = smf.mixedlm("y ~ C(arm) * C(eps)", long, groups=long["case_id"]).fit(
        reml=False, method=opt
    )

    names = list(add.fe_params.index)  # t_test takes fixed effects only

    def dummy(arm: str) -> np.ndarray:
        v = np.zeros(len(names))
        if arm != arms[0]:
            v[names.index(f"C(arm)[T.{arm}]")] = 1.0
        return v

    rows = []
    for blade_arm, base_arm in pairs:
        contrast = (sign * (dummy(blade_arm) - dummy(base_arm))).reshape(1, -1)
        t = add.t_test(contrast)
        p = float(np.ravel(t.pvalue)[0])
        rows.append(
            dict(
                blade=ARM_LABEL.get(blade_arm, blade_arm),
                opponent=ARM_LABEL.get(base_arm, base_arm),
                lmm_estimate=float(np.ravel(t.effect)[0]),
                lmm_se=float(np.ravel(t.sd)[0]),
                lmm_z=float(np.ravel(t.statistic)[0]),
                lmm_p=max(p, TINY),
                lmm_p_underflow=bool(p == 0.0),
                lmm_ci_low=float(t.conf_int()[0][0]),
                lmm_ci_high=float(t.conf_int()[0][1]),
            )
        )

    df = len(inter.fe_params) - len(add.fe_params)
    lr = 2.0 * (inter.llf - add.llf)
    lr_p = float(stats.chi2.sf(lr, df))
    lrt = dict(
        lr_stat=float(lr),
        lr_df=int(df),
        lr_p=max(lr_p, TINY),
        lr_p_underflow=bool(lr_p == 0.0),
        icc=float(add.cov_re.iloc[0, 0] / (add.cov_re.iloc[0, 0] + add.scale)),
        group_var=float(add.cov_re.iloc[0, 0]),
        residual_var=float(add.scale),
        converged=bool(add.converged and inter.converged),
    )
    return rows, lrt


# ------------------------------------------------------------------- analyses


def holm_table(blade: list[str], baselines: list[str]) -> pd.DataFrame:
    """(0): Holm over the budget-averaged pairwise p-values, m = 4 and m = 36."""
    rows = []
    for dataset, path in PAIRWISE.items():
        src = pd.read_csv(path)
        for arm in blade:
            for opp in baselines:
                hit = src[
                    ((src.arm_a == arm) & (src.arm_b == opp))
                    | ((src.arm_a == opp) & (src.arm_b == arm))
                ]
                for _, r in hit.iterrows():
                    flip = r.arm_a == opp
                    rows.append(
                        dict(
                            dataset=dataset,
                            metric=METRIC_LABEL[r.metric],
                            metric_key=r.metric,
                            blade=ARM_LABEL[arm],
                            opponent=ARM_LABEL[opp],
                            median_blade=r.median_b if flip else r.median_a,
                            median_opp=r.median_a if flip else r.median_b,
                            hl_shift=-r.hl_shift if flip else r.hl_shift,
                            rank_biserial=-r.rank_biserial if flip else r.rank_biserial,
                            n=int(r.n_cases),
                            p_raw=r.p_value,
                            blade_more_damaging=(not r.a_more_damaging)
                            if flip
                            else bool(r.a_more_damaging),
                        )
                    )
    tab = pd.DataFrame(rows)
    expected = len(PAIRWISE) * len(METRIC_LABEL) * len(blade) * len(baselines)
    if len(tab) != expected:
        raise SystemExit(f"holm: expected {expected} contrasts, found {len(tab)}")

    tab["p_holm_within"] = 0.0
    for _, grp in tab.groupby(["dataset", "metric_key"], sort=False):
        tab.loc[grp.index, "p_holm_within"] = holm(list(grp.p_raw))
    tab["p_holm_pooled"] = holm(list(tab.p_raw))
    tab["sig_within"] = tab.p_holm_within < 0.05
    tab["sig_pooled"] = tab.p_holm_pooled < 0.05
    tab["verdict"] = [
        ("BLADE wins" if b else "BLADE loses") if s else "no separation"
        for b, s in zip(tab.blade_more_damaging, tab.sig_within)
    ]
    order = {"Whole gland": 0, "TZ+CZ": 1, "PZ": 2}
    mo = {"dice": 0, "hd95": 1, "asd": 2}
    tab = tab.sort_values(
        by=["dataset", "metric_key", "blade", "opponent"],
        key=lambda c: (
            c.map(order)
            if c.name == "dataset"
            else (c.map(mo) if c.name == "metric_key" else c)
        ),
    )
    cols = [
        "dataset",
        "metric",
        "blade",
        "opponent",
        "n",
        "median_blade",
        "median_opp",
        "hl_shift",
        "rank_biserial",
        "p_raw",
        "p_holm_within",
        "p_holm_pooled",
        "sig_within",
        "sig_pooled",
        "verdict",
    ]
    return tab[cols]


def deep_stats(
    blade: list[str], baselines: list[str], n_boot: int, seed: int
) -> dict[str, list[dict]]:
    """(A)-(D) from the per-sample CSVs."""
    arms = baselines[:1] + baselines[1:] + blade  # reference first
    pairs = [(b, o) for b in blade for o in baselines]
    rng = np.random.default_rng(seed)
    per_eps, auc_rows, lmm_rows, lrt_rows, boot_rows = [], [], [], [], []

    for dataset, (directory, klass) in DATASETS.items():
        t0 = time.time()
        data = {arm: load_arm(directory, arm, klass) for arm in arms}
        ref = set(data[arms[0]].case_id)
        for arm in arms[1:]:
            if set(data[arm].case_id) != ref:
                raise SystemExit(f"{dataset}: {arm} case set differs from {arms[0]}")
        print(f"[{dataset}] {len(ref)} cases x {len(arms)} arms", flush=True)

        for metric, sign in METRIC_DIRECTION.items():
            mats = {arm: wide(data[arm], metric) for arm in arms}
            # trapezoid over the full grid, normalised to a mean value on [0, 0.10]
            w = np.array([0.1, 0.2, 0.2, 0.2, 0.2, 0.1])
            auc = {
                arm: pd.Series(
                    mats[arm][EPS_GRID].to_numpy() @ w, index=mats[arm].index
                )
                for arm in arms
            }
            avg = {arm: mats[arm][EPS_ATTACK].mean(axis=1) for arm in arms}

            block = []
            for blade_arm, base_arm in pairs:
                for eps in EPS_ATTACK:
                    d = sign * (mats[blade_arm][eps] - mats[base_arm][eps]).to_numpy()
                    block.append(
                        dict(
                            dataset=dataset,
                            metric=METRIC_LABEL[metric],
                            metric_key=metric,
                            epsilon=eps,
                            blade=ARM_LABEL.get(blade_arm, blade_arm),
                            opponent=ARM_LABEL.get(base_arm, base_arm),
                            median_blade=float(mats[blade_arm][eps].median()),
                            median_opponent=float(mats[base_arm][eps].median()),
                            **paired(d),
                        )
                    )
            for row, ph in zip(block, holm([r["p_raw"] for r in block])):
                row["p_holm_m20"] = ph
                row["significant"] = ph < 0.05
            per_eps.extend(block)

            aucs, boots = [], []
            for blade_arm, base_arm in pairs:
                d_auc = sign * (auc[blade_arm] - auc[base_arm]).to_numpy()
                d_avg = sign * (avg[blade_arm] - avg[base_arm]).to_numpy()
                common = dict(
                    dataset=dataset,
                    metric=METRIC_LABEL[metric],
                    metric_key=metric,
                    blade=ARM_LABEL.get(blade_arm, blade_arm),
                    opponent=ARM_LABEL.get(base_arm, base_arm),
                )
                aucs.append(
                    dict(
                        **common,
                        auc_blade=float(auc[blade_arm].median()),
                        auc_opponent=float(auc[base_arm].median()),
                        **paired(d_auc),
                    )
                )
                lo_a, hi_a = boot_ci(d_auc, rng, n_boot)
                lo_b, hi_b = boot_ci(d_avg, rng, n_boot)
                boots.append(
                    dict(
                        **common,
                        auc_median_diff=float(np.median(d_auc)),
                        auc_ci_low=lo_a,
                        auc_ci_high=hi_a,
                        auc_ci_excludes_zero=bool(lo_a > 0 or hi_a < 0),
                        budgetavg_median_diff=float(np.median(d_avg)),
                        budgetavg_ci_low=lo_b,
                        budgetavg_ci_high=hi_b,
                        budgetavg_ci_excludes_zero=bool(lo_b > 0 or hi_b < 0),
                    )
                )
            for row, ph in zip(aucs, holm([r["p_raw"] for r in aucs])):
                row["p_holm_m4"] = ph
                row["significant"] = ph < 0.05
            auc_rows.extend(aucs)
            boot_rows.extend(boots)

            long = pd.concat(
                [data[arm].assign(arm=arm) for arm in arms], ignore_index=True
            )
            rows, lrt = mixed_models(long, metric, arms, pairs)
            for r in rows:
                r.update(
                    dataset=dataset, metric=METRIC_LABEL[metric], metric_key=metric
                )
            lmm_rows.extend(rows)
            lrt_rows.append(
                dict(
                    dataset=dataset,
                    metric=METRIC_LABEL[metric],
                    metric_key=metric,
                    **lrt,
                )
            )
            print(
                f"  {metric:<5} done  ({time.time() - t0:.0f}s cumulative)", flush=True
            )

    return {
        "per_epsilon_wilcoxon": per_eps,
        "auc_robustness_curve": auc_rows,
        "mixedlm_contrasts": lmm_rows,
        "mixedlm_interaction_lrt": lrt_rows,
        "clustered_bootstrap_ci": boot_rows,
    }


# ----------------------------------------------------------------------- driver


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--blade", nargs="+", default=["blade", "blade_mm"])
    ap.add_argument("--baselines", nargs="+", default=["sea", "auto_pgd_r3"])
    ap.add_argument("--n-boot", type=int, default=2000)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--output-dir", default=OUT_DIR)
    ap.add_argument("--stem", default="blade_tier300")
    args = ap.parse_args()

    stem = os.path.join(args.output_dir, args.stem)
    tab = holm_table(args.blade, args.baselines)
    tab.to_csv(f"{stem}_holm_corrected.csv", index=False)
    print(f"wrote {stem}_holm_corrected.csv ({len(tab)} rows)", flush=True)

    for name, rows in deep_stats(
        args.blade, args.baselines, args.n_boot, args.seed
    ).items():
        pd.DataFrame(rows).to_csv(f"{stem}_{name}.csv", index=False)
        print(f"wrote {stem}_{name}.csv ({len(rows)} rows)", flush=True)


if __name__ == "__main__":
    main()
