"""Round-3 review reanalysis for ``paper_resenc/manuscript.tex`` (items R4, S7).

Re-analyses existing per-sample outputs only, with the loaders, bootstrap and
seed of ``analyze_resenc_revision_evidence.py``.

* E29: per fold, per attack, at epsilon = 0.02 and budget-averaged, the mean
  encoder (ResEnc-M - nnU-Net), regime (ResEnc-XL - ResEnc-M) and total steps,
  and the paired regime-minus-encoder difference with a 95% percentile CI from
  2,000 case resamples within the fold (R4: fold consistency as a CI of the
  difference, not as CI non-overlap).
* E30: fold-level summary of E29, i.e. mean and SD over the five folds and a
  t interval with 4 degrees of freedom (S7: model-level, not case-level,
  variation), plus the number of folds whose difference CI lies above zero.
* E31: R9/R10 attack checks from ``resenc_attack_sanity_subset.py`` (WG fold-0
  subset): mean Dice, architecture/regime/total steps and their paired
  difference with case-bootstrap CIs, per loss, iteration count and epsilon.

Usage::

    PYTHONPATH=src python experiments/analyze_resenc_round3_evidence.py
"""

from __future__ import annotations

import argparse
import os
import sys

import numpy as np
import pandas as pd
from scipy import stats

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import analyze_resenc_revision_evidence as rev  # noqa: E402

from mri_prostate_seg.experiments import paired_contrasts as pc  # noqa: E402

ENDPOINTS = {"eps0.02": 0.02, "budget_avg": None}


def e29_per_fold(data: rev.Data) -> pd.DataFrame:
    rows = []
    for c in rev.TREES:
        for arm in rev.ARMS8:
            for label, eps in ENDPOINTS.items():
                mat = data.mat(c, arm, "dice", eps)
                folds = data.fold_labels(c, mat.index)
                for f in rev.FOLDS:
                    sub = mat[folds == f]
                    u, m, x = (
                        sub[a].to_numpy() for a in ("unet", "resenc_m", "resenc_xl")
                    )
                    idx = pc.stratified_bootstrap_indices(
                        np.zeros(len(sub)), rev.N_BOOT, rev.SEED
                    )
                    diff = (x - m) - (m - u)
                    ci = rev.boot_mean(diff, idx)
                    rows.append(
                        {
                            "class_name": c,
                            "arm": arm,
                            "endpoint": label,
                            "fold": f,
                            "n_cases": len(sub),
                            "encoder": float(np.mean(m - u)),
                            "regime": float(np.mean(x - m)),
                            "total": float(np.mean(x - u)),
                            "share_regime": pc.ratio_of_means(x - m, x - u),
                            "regime_minus_encoder": ci["mean"],
                            "diff_ci_lo": ci["ci_lo"],
                            "diff_ci_hi": ci["ci_hi"],
                        }
                    )
    return pd.DataFrame(rows)


def e33_per_case_steps(data: rev.Data, arm: str = "auto_pgd") -> pd.DataFrame:
    """Per-case budget-averaged architecture and regime steps (new Fig. 2)."""
    frames = []
    for c in rev.TREES:
        mat = data.mat(c, arm, "dice", None)
        frames.append(
            pd.DataFrame(
                {
                    "class_name": c,
                    "arm": arm,
                    "case_id": mat.index,
                    "fold": data.fold_labels(c, mat.index),
                    "architecture": mat["resenc_m"] - mat["unet"],
                    "regime": mat["resenc_xl"] - mat["resenc_m"],
                }
            )
        )
    return pd.concat(frames, ignore_index=True)


def e35_per_case_dice(data: rev.Data, arm: str = "auto_pgd") -> pd.DataFrame:
    """Per-case clean and budget-averaged attacked Dice per model (new Fig. 2)."""
    frames = []
    for c in rev.TREES:
        for condition, mat in (
            ("clean", data.clean(c)),
            (arm, data.mat(c, arm, "dice", None)),
        ):
            long = mat.reset_index().melt(
                id_vars=mat.index.name or "index", var_name="arch", value_name="dice"
            )
            long.columns = ["case_id", "arch", "dice"]
            frames.append(long.assign(class_name=c, condition=condition))
    return pd.concat(frames, ignore_index=True)


def e36_per_case_all_metrics(data: rev.Data) -> pd.DataFrame:
    """Per-case clean and budget-averaged attacked Dice/HD95/ASD per model and
    attack (new Fig. 1). HD95/ASD average only the budgets where they are defined."""
    frames = []
    for c in rev.TREES:
        for metric in ("dice", "hd95", "asd"):
            clean = pd.DataFrame(
                {
                    a: rev.build_matrix(
                        {"pgd": data.frames[c][a]["pgd"]}, metric, 0.0, ["pgd"], c
                    )["pgd"]
                    for a in rev.ARCHS
                }
            )
            for condition, mat in [("clean", clean)] + [
                (arm, data.mat(c, arm, metric, None)) for arm in rev.NON_DAG
            ]:
                long = mat.rename_axis("case_id").reset_index().melt(
                    id_vars="case_id", var_name="arch", value_name="value"
                )
                frames.append(long.assign(class_name=c, metric=metric, condition=condition))
    return pd.concat(frames, ignore_index=True).dropna(subset=["value"])


def e33_summary(steps: pd.DataFrame) -> pd.DataFrame:
    """Mean step with fold-stratified bootstrap CI, and the paired Wilcoxon test."""
    rows = []
    for c, g in steps.groupby("class_name", sort=False):
        idx = pc.stratified_bootstrap_indices(
            g["fold"].to_numpy(), rev.N_BOOT, rev.SEED
        )
        row = {"class_name": c, "n_cases": len(g)}
        for name in ("architecture", "regime"):
            ci = rev.boot_mean(g[name].to_numpy(), idx)
            row.update(
                {
                    name: ci["mean"],
                    f"{name}_ci_lo": ci["ci_lo"],
                    f"{name}_ci_hi": ci["ci_hi"],
                }
            )
        row["wilcoxon_p_regime_vs_architecture"] = rev.wilcoxon_p(
            (g["regime"] - g["architecture"]).to_numpy()
        )
        rows.append(row)
    return pd.DataFrame(rows)


def e30_fold_summary(per_fold: pd.DataFrame) -> pd.DataFrame:
    rows = []
    t_crit = stats.t.ppf(0.975, df=len(rev.FOLDS) - 1)
    for (c, arm, ep), g in per_fold.groupby(
        ["class_name", "arm", "endpoint"], sort=False
    ):
        row = {"class_name": c, "arm": arm, "endpoint": ep, "n_folds": len(g)}
        for col in (
            "encoder",
            "regime",
            "total",
            "share_regime",
            "regime_minus_encoder",
        ):
            mean, sd = float(g[col].mean()), float(g[col].std(ddof=1))
            row[f"{col}_mean"] = mean
            row[f"{col}_sd"] = sd
            if col == "regime_minus_encoder":
                half = t_crit * sd / np.sqrt(len(g))
                row["diff_t_lo"], row["diff_t_hi"] = mean - half, mean + half
        row["folds_diff_ci_above_0"] = int((g["diff_ci_lo"] > 0).sum())
        row["folds_diff_ci_below_0"] = int((g["diff_ci_hi"] < 0).sum())
        rows.append(row)
    return pd.DataFrame(rows)


def e31_attack_sanity(sanity_dir: str) -> pd.DataFrame:
    """R9/R10: steps under each (loss, iterations, epsilon) of the fold-0 subset.

    Reads ``resenc_attack_sanity_subset.py`` output; only cases finished by all
    three models enter, so every row is paired.
    """
    runs = pd.concat(
        [
            pd.read_csv(os.path.join(sanity_dir, f"wg_{a}_fold0.csv"))
            for a in ("unet", "resenc_m", "resenc_xl")
        ],
        ignore_index=True,
    )
    rows = []
    for (loss, iters, eps), g in runs.groupby(["loss", "iters", "eps"]):
        mat = g.pivot_table(index="case_id", columns="arch", values="dice").dropna()
        if mat.shape[1] < 3:
            continue
        rows.append({"loss": loss, "iters": iters, "eps": eps, **_step_row(mat)})
    return pd.DataFrame(rows)


def _step_row(mat: pd.DataFrame) -> dict:
    """Mean Dice per model and the paired steps with case-bootstrap CIs."""
    u, m, x = (mat[a].to_numpy() for a in ("unet", "resenc_m", "resenc_xl"))
    idx = pc.stratified_bootstrap_indices(np.zeros(len(mat)), rev.N_BOOT, rev.SEED)
    row = {
        "n_cases": len(mat),
        "dice_unet": u.mean(),
        "dice_resenc_m": m.mean(),
        "dice_resenc_xl": x.mean(),
    }
    for name, d in (
        ("architecture", m - u),
        ("regime", x - m),
        ("total", x - u),
        ("regime_minus_architecture", (x - m) - (m - u)),
    ):
        ci = rev.boot_mean(d, idx)
        row.update(
            {
                name: ci["mean"],
                f"{name}_ci_lo": ci["ci_lo"],
                f"{name}_ci_hi": ci["ci_hi"],
            }
        )
    row["share_regime"] = pc.ratio_of_means(x - m, x - u)
    row["share_regime_ci_lo"], row["share_regime_ci_hi"] = pc.percentile_ci(
        pc.resampled_means(x - m, idx) / pc.resampled_means(x - u, idx)
    )
    return row


def e32_transfer(sanity_dir: str) -> pd.DataFrame:
    """REV-23: each source model's perturbation scored on all three targets.

    Reads ``resenc_transfer_subset.py`` output. Per (source, epsilon) row, the
    ``dice_*`` columns are the target models' Dice on that source's inputs, and
    the steps compare the targets on identical inputs.
    """
    runs = pd.read_csv(os.path.join(sanity_dir, "wg_transfer_fold0.csv"))
    # Same cases as E31, so the two checks describe one subset.
    subset = pd.read_csv(os.path.join(sanity_dir, "wg_unet_fold0.csv"))["case_id"]
    runs = runs[runs["case_id"].isin(subset.unique())]
    rows = []
    for (source, eps), g in runs.groupby(["source", "eps"]):
        mat = g.pivot_table(index="case_id", columns="target", values="dice").dropna()
        rows.append({"source": source, "eps": eps, **_step_row(mat)})
    return pd.DataFrame(rows)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--output-dir", default=os.path.join(rev.STATS_DIR, "revision"))
    parser.add_argument(
        "--sanity-only", action="store_true", help="only E31 (R9/R10 subset runs)"
    )
    args = parser.parse_args()
    os.makedirs(args.output_dir, exist_ok=True)
    sanity = e31_attack_sanity(os.path.join(args.output_dir, "attack_sanity"))
    sanity.to_csv(
        os.path.join(args.output_dir, "E31_attack_sanity_steps.csv"), index=False
    )
    print(f"wrote E31_attack_sanity_steps.csv: {len(sanity)} rows")
    if os.path.exists(
        os.path.join(args.output_dir, "attack_sanity", "wg_transfer_fold0.csv")
    ):
        transfer = e32_transfer(os.path.join(args.output_dir, "attack_sanity"))
        transfer.to_csv(
            os.path.join(args.output_dir, "E32_transfer_steps.csv"), index=False
        )
        print(f"wrote E32_transfer_steps.csv: {len(transfer)} rows")
    if args.sanity_only:
        return
    data = rev.Data(rev.load_pooled_frames())
    per_fold = e29_per_fold(data)
    for df, name in (
        (per_fold, "E29_per_fold_eps002_difference.csv"),
        (e30_fold_summary(per_fold), "E30_fold_level_summary.csv"),
        (steps := e33_per_case_steps(data), "E33_per_case_steps_auto_pgd.csv"),
        (e33_summary(steps), "E33_step_summary_auto_pgd.csv"),
        (e35_per_case_dice(data), "E35_per_case_dice_clean_auto_pgd.csv"),
        (e36_per_case_all_metrics(data), "E36_per_case_all_metrics.csv"),
    ):
        df.to_csv(os.path.join(args.output_dir, name), index=False)
        print(f"wrote {name}: {len(df)} rows")


if __name__ == "__main__":
    main()
