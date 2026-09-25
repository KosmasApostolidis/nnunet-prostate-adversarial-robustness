"""Analyse the replay study: where each attack's damage lands, and how it scales.

Reads ``results/blade_attack/mechanism/{direction,scaling}/summary_*.csv`` (and
the ``profile_*.csv`` damage profiles) written by
``run_blade_mechanism_study.py`` and writes, into ``mechanism/analysis/``:

Direction (20 steps, nnU-Net, fold-0 subset)
  * ``direction_medians.csv`` -- per (task, target, arm, epsilon) medians of
    induced false-positive / false-negative volume, damage bias
    ((FP - FN) / (FP + FN)), volume change, centroid shift, TZ<->PZ swap volume,
    and the perturbation's energy share per signed band around the gland.
  * ``direction_tests.csv`` -- per-case budget-averaged paired Wilcoxon of
    BLADE-3 against Auto-PGD, Auto-PGD x3 and SEA on those quantities (BH within
    each target x quantity family).
  * figures: FP/FN volume per arm, damage-rate profile by signed distance,
    perturbation-energy profile, zone swap.

Scaling (nnU-Net WG; the 20-step point comes from the direction run)
  * ``scaling_summary.csv`` -- mean budget-averaged and per-epsilon Dice with a
    case-bootstrap 95% CI per (arm, steps), plus gradients per case.
  * ``scaling_tests.csv`` -- per step count, paired Wilcoxon of BLADE-3 vs SEA,
    BLADE-3 vs Auto-PGD x3 and BLADE-DiceCE vs Auto-PGD (budget-matched pairs).
  * figures: Dice against gradients per case, budget-averaged and per epsilon.

Usage::

    PYTHONPATH=src python experiments/analyze_blade_mechanism.py
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
from analyze_main_comparison_stats import (  # noqa: E402
    ALPHA,
    _fdr_bh,
    _hodges_lehmann,
    _rank_biserial,
)
from plot_attack_comparison import ARM_COLOUR  # noqa: E402
from run_twelve_arm_wg_campaign import ARM_LABELS  # noqa: E402

from mri_prostate_seg.experiments.perturbation_structure import SIGNED_BAND_NAMES  # noqa: E402

ROOT = os.path.join("results", "blade_attack", "mechanism")
DIRECTION_QUANTITIES = (
    "adv_dice",
    "induced_fp_mm3",
    "induced_fn_mm3",
    "damage_bias",
    "volume_change_pct",
    "centroid_shift_norm_mm",
)
ENERGY_COLS = [f"energy_share_{b}" for b in SIGNED_BAND_NAMES]
REFERENCE = "blade"
BASELINES = ("auto_pgd", "auto_pgd_r3", "sea")
SCALING_ARMS = ("auto_pgd", "auto_pgd_r3", "sea", "blade1", "blade")
SCALING_PAIRS = (("blade", "sea"), ("blade", "auto_pgd_r3"), ("blade1", "auto_pgd"))
BAND_LABELS = {
    "inside_beyond_10mm": ">10 in",
    "inside_5_10mm": "5-10 in",
    "inside_2_5mm": "2-5 in",
    "inside_0_2mm": "0-2 in",
    "0_2mm": "0-2 out",
    "2_5mm": "2-5 out",
    "5_10mm": "5-10 out",
    "beyond_10mm": ">10 out",
}


# BLADE-3 and BLADE-MM often coincide; a filled square under a larger hollow
# diamond on a dashed line keeps both visible where they overlap.
ARM_MARKER = {
    "blade": dict(marker="s", markersize=6),
    "blade_mm": dict(marker="D", markersize=8, markerfacecolor="none",
        linestyle="--", markeredgewidth=1.5),
}


def _marker(arm: str) -> dict:
    return ARM_MARKER.get(arm, dict(marker="o"))


def read(pattern: str) -> pd.DataFrame:
    files = sorted(glob.glob(pattern))
    if not files:
        return pd.DataFrame()
    return pd.concat([pd.read_csv(f) for f in files], ignore_index=True)


def _complete_cases(
    t: pd.DataFrame, arms: tuple[str, ...], keys: list[str]
) -> pd.DataFrame:
    """Keep only cases every arm has finished, so every comparison is paired."""
    have = t.groupby(keys + ["case_id"])["arm"].nunique().reset_index()
    full = have[have["arm"] == len(arms)].drop(columns="arm")
    return t.merge(full, on=keys + ["case_id"])


def _paired(a: np.ndarray, b: np.ndarray) -> dict[str, float]:
    diff = a - b
    diff = diff[np.isfinite(diff)]
    nz = diff[diff != 0]
    w_p = float("nan") if nz.size == 0 else float(stats.wilcoxon(nz)[1])
    hl, lo, hi = _hodges_lehmann(diff)
    return {
        "n_cases": int(diff.size),
        "median_a": float(np.nanmedian(a)),
        "median_b": float(np.nanmedian(b)),
        "hl_shift": hl,
        "hl_ci95_low": lo,
        "hl_ci95_high": hi,
        "p_value": w_p,
        "rank_biserial": _rank_biserial(diff),
    }


# ---------------------------------------------------------------- direction


def direction_analysis(out: str) -> list[str]:
    d = read(os.path.join(ROOT, "direction", "summary_*.csv"))
    if d.empty:
        return []
    arms = tuple(a for a in ARM_LABELS if a in set(d["arm"]))
    d = _complete_cases(d, arms, ["task", "target"])
    cols = [
        c
        for c in (*DIRECTION_QUANTITIES, *ENERGY_COLS, "zone_swap_mm3")
        if c in d.columns
    ]
    med = d.groupby(["task", "target", "arm", "epsilon"])[cols].median().reset_index()
    med["n_cases"] = (
        d.groupby(["task", "target", "arm", "epsilon"])["case_id"].nunique().values
    )
    med.to_csv(os.path.join(out, "direction_medians.csv"), index=False)

    per_case = (
        d.groupby(["task", "target", "arm", "case_id"])[cols].mean().reset_index()
    )
    rows = []
    for (task, target), sub in per_case.groupby(["task", "target"]):
        for q in cols:
            fam = []
            wide = sub.pivot(index="case_id", columns="arm", values=q)
            for base in BASELINES:
                if base not in wide or REFERENCE not in wide:
                    continue
                pair = wide[[REFERENCE, base]].dropna()
                fam.append(
                    {
                        "task": task,
                        "target": target,
                        "quantity": q,
                        "arm_a": REFERENCE,
                        "arm_b": base,
                        **_paired(pair[REFERENCE].to_numpy(), pair[base].to_numpy()),
                    }
                )
            if fam:
                f = pd.DataFrame(fam)
                f["p_adj_bh"] = _fdr_bh(f["p_value"].to_numpy())
                f["significant"] = f["p_adj_bh"] < ALPHA
                rows.append(f)
    pd.concat(rows, ignore_index=True).to_csv(
        os.path.join(out, "direction_tests.csv"), index=False
    )

    written = [_plot_fp_fn(med, arms, out), _plot_energy(med, arms, out)]
    prof = read(os.path.join(ROOT, "direction", "profile_*.csv"))
    if not prof.empty:
        prof = prof.merge(
            d[["task", "target", "arm", "case_id"]].drop_duplicates(),
            on=["task", "target", "arm", "case_id"],
        )
        written.append(_plot_profile(prof, arms, out))
    if "zone_swap_mm3" in med.columns:
        written.append(_plot_swap(med, arms, out))
    return written


def _targets(med: pd.DataFrame) -> list[tuple[str, str]]:
    order = [("wg", "WG"), ("zones", "TZ+CZ"), ("zones", "PZ")]
    have = set(zip(med["task"], med["target"]))
    return [t for t in order if t in have]


def _plot_fp_fn(med: pd.DataFrame, arms: tuple[str, ...], out: str) -> str:
    targets = _targets(med)
    fig, axes = plt.subplots(
        1, len(targets), figsize=(5.2 * len(targets), 4.2), layout="constrained"
    )
    axes = np.atleast_1d(axes)
    for ax, (task, target) in zip(axes, targets):
        sub = med[
            (med["task"] == task)
            & (med["target"] == target)
            & np.isclose(med["epsilon"], 0.06)
        ]
        sub = sub.set_index("arm").loc[list(arms)]
        x = np.arange(len(arms))
        fp = sub["induced_fp_mm3"].to_numpy() / 1000
        fn = sub["induced_fn_mm3"].to_numpy() / 1000
        ax.bar(x - 0.2, fp, width=0.4, color="#D55E00", label="induced false positive")
        ax.bar(x + 0.2, fn, width=0.4, color="#0072B2", label="induced false negative")
        ax.set_xticks(x)
        ax.set_xticklabels(
            [ARM_LABELS[a] for a in arms], rotation=40, ha="right", fontsize=8
        )
        ax.set_title(f"{target}", fontsize=11, fontweight="bold")
        ax.grid(axis="y", alpha=0.3)
    axes[0].set_ylabel("median volume per case (cm$^3$)")
    axes[0].legend(fontsize=8)
    fig.suptitle(
        r"Direction of damage at $\varepsilon$ = 0.06: over- vs under-segmentation",
        fontsize=12,
        fontweight="bold",
    )
    path = os.path.join(out, "direction_fp_fn_eps0.06.png")
    fig.savefig(path, dpi=300, bbox_inches="tight")
    plt.close(fig)
    return path


def _plot_energy(med: pd.DataFrame, arms: tuple[str, ...], out: str) -> str:
    tasks = [t for t in ("wg", "zones") if t in set(med["task"])]
    fig, axes = plt.subplots(
        1, len(tasks), figsize=(5.6 * len(tasks), 4.0), layout="constrained"
    )
    axes = np.atleast_1d(axes)
    for ax, task in zip(axes, tasks):
        target = "WG" if task == "wg" else "gland"
        sub = med[
            (med["task"] == task)
            & (med["target"] == target)
            & np.isclose(med["epsilon"], 0.06)
        ]
        for arm in arms:
            r = sub[sub["arm"] == arm][ENERGY_COLS].iloc[0].to_numpy()
            ax.plot(
                range(len(ENERGY_COLS)),
                r,
                **_marker(arm),
                color=ARM_COLOUR[arm],
                label=ARM_LABELS[arm],
            )
        ax.axvline(3.5, color="0.4", linewidth=0.8, linestyle=":")
        ax.set_xticks(range(len(ENERGY_COLS)))
        ax.set_xticklabels(
            [BAND_LABELS[b] for b in SIGNED_BAND_NAMES], rotation=40, fontsize=8
        )
        ax.set_xlabel("signed distance to the ground-truth gland boundary (mm)")
        ax.set_title(
            "Whole gland" if task == "wg" else "Zones (gland union)",
            fontsize=11,
            fontweight="bold",
        )
        ax.grid(alpha=0.3)
    axes[0].set_ylabel("share of perturbation energy")
    axes[-1].legend(fontsize=7, ncol=2)
    fig.suptitle(
        r"Where the perturbation sits, $\varepsilon$ = 0.06 (median over cases)",
        fontsize=12,
        fontweight="bold",
    )
    path = os.path.join(out, "direction_energy_profile_eps0.06.png")
    fig.savefig(path, dpi=300, bbox_inches="tight")
    plt.close(fig)
    return path


def _plot_profile(prof: pd.DataFrame, arms: tuple[str, ...], out: str) -> str:
    prof = prof[np.isclose(prof["epsilon"], 0.06)]
    targets = _targets(prof)
    rate_cols = [c for c in ("induced_fp_rate", "induced_fn_rate") if c in prof.columns]
    fig, axes = plt.subplots(
        len(rate_cols),
        len(targets),
        figsize=(5.0 * len(targets), 3.6 * len(rate_cols)),
        layout="constrained",
        squeeze=False,
    )
    for j, (task, target) in enumerate(targets):
        sub = prof[(prof["task"] == task) & (prof["target"] == target)]
        agg = sub.groupby(["arm", "band"])[rate_cols].median().reset_index()
        for i, col in enumerate(rate_cols):
            ax = axes[i, j]
            for arm in arms:
                r = (
                    agg[agg["arm"] == arm]
                    .set_index("band")
                    .reindex(SIGNED_BAND_NAMES)[col]
                )
                ax.plot(
                    range(len(SIGNED_BAND_NAMES)),
                    r.to_numpy(),
                    **_marker(arm),
                    color=ARM_COLOUR[arm],
                    label=ARM_LABELS[arm],
                )
            ax.axvline(3.5, color="0.4", linewidth=0.8, linestyle=":")
            ax.set_xticks(range(len(SIGNED_BAND_NAMES)))
            ax.set_xticklabels(
                [BAND_LABELS[b] for b in SIGNED_BAND_NAMES], rotation=40, fontsize=7
            )
            ax.set_title(
                f"{target}: {col.replace('_', ' ')}", fontsize=10, fontweight="bold"
            )
            ax.grid(alpha=0.3)
    axes[0, 0].legend(fontsize=6, ncol=2)
    fig.suptitle(
        r"Damage rate by signed distance to the true boundary, $\varepsilon$ = 0.06",
        fontsize=12,
        fontweight="bold",
    )
    path = os.path.join(out, "direction_damage_profile_eps0.06.png")
    fig.savefig(path, dpi=300, bbox_inches="tight")
    plt.close(fig)
    return path


def _plot_swap(med: pd.DataFrame, arms: tuple[str, ...], out: str) -> str:
    sub = med[(med["task"] == "zones") & (med["target"] == "gland")]
    fig, ax = plt.subplots(figsize=(5.4, 3.8), layout="constrained")
    for arm in arms:
        r = sub[sub["arm"] == arm].sort_values("epsilon")
        ax.plot(
            r["epsilon"],
            r["zone_swap_mm3"] / 1000,
            **_marker(arm),
            color=ARM_COLOUR[arm],
            label=ARM_LABELS[arm],
        )
    ax.set_xlabel(r"$\varepsilon$")
    ax.set_ylabel("median TZ<->PZ swapped volume (cm$^3$)")
    ax.grid(alpha=0.3)
    ax.legend(fontsize=7, ncol=2)
    ax.set_title(
        "Zone confusion induced by each attack", fontsize=11, fontweight="bold"
    )
    path = os.path.join(out, "direction_zone_swap.png")
    fig.savefig(path, dpi=300, bbox_inches="tight")
    plt.close(fig)
    return path


# ------------------------------------------------------------------ scaling


def _boot_ci(
    values: np.ndarray, n_boot: int = 2000, seed: int = 42
) -> tuple[float, float]:
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, len(values), size=(n_boot, len(values)))
    means = values[idx].mean(axis=1)
    return float(np.percentile(means, 2.5)), float(np.percentile(means, 97.5))


def scaling_analysis(out: str) -> list[str]:
    s = read(os.path.join(ROOT, "scaling", "summary_*.csv"))
    d = read(os.path.join(ROOT, "direction", "summary_*.csv"))
    if s.empty:
        return []
    keep = [
        "task",
        "arch",
        "arm",
        "steps",
        "case_id",
        "n_backward",
        "epsilon",
        "target",
        "adv_dice",
    ]
    parts = [s[keep]]
    if not d.empty:
        parts.append(d[(d["task"] == "wg") & d["arm"].isin(SCALING_ARMS)][keep])
    t = pd.concat(parts, ignore_index=True)
    t = t[t["arm"].isin(SCALING_ARMS)]
    t = _complete_cases(t, SCALING_ARMS, ["steps"])
    # a step count is only reported once every arm has the same cases
    per_case = (
        t.groupby(["arm", "steps", "case_id"])
        .agg(dice=("adv_dice", "mean"), grads=("n_backward", "first"))
        .reset_index()
    )

    rows = []
    for (arm, steps), g in per_case.groupby(["arm", "steps"]):
        lo, hi = _boot_ci(g["dice"].to_numpy())
        rows.append(
            {
                "arm": arm,
                "steps": steps,
                "epsilon": "budget_avg",
                "n_cases": len(g),
                "grads_per_case": g["grads"].mean(),
                "mean_dice": g["dice"].mean(),
                "ci95_low": lo,
                "ci95_high": hi,
            }
        )
    for (arm, steps, eps), g in t.groupby(["arm", "steps", "epsilon"]):
        lo, hi = _boot_ci(g["adv_dice"].to_numpy())
        rows.append(
            {
                "arm": arm,
                "steps": steps,
                "epsilon": eps,
                "n_cases": g["case_id"].nunique(),
                "grads_per_case": g["n_backward"].mean(),
                "mean_dice": g["adv_dice"].mean(),
                "ci95_low": lo,
                "ci95_high": hi,
            }
        )
    summ = pd.DataFrame(rows)
    summ.to_csv(os.path.join(out, "scaling_summary.csv"), index=False)

    tests = []
    for steps, g in per_case.groupby("steps"):
        wide = g.pivot(index="case_id", columns="arm", values="dice")
        fam = []
        for a, b in SCALING_PAIRS:
            # positive shift = first arm leaves lower Dice (more damage)
            fam.append(
                {
                    "steps": steps,
                    "arm_a": a,
                    "arm_b": b,
                    **_paired(wide[b].to_numpy(), wide[a].to_numpy()),
                }
            )
        f = pd.DataFrame(fam)
        f["p_adj_bh"] = _fdr_bh(f["p_value"].to_numpy())
        f["significant"] = f["p_adj_bh"] < ALPHA
        tests.append(f)
    pd.concat(tests, ignore_index=True).to_csv(
        os.path.join(out, "scaling_tests.csv"), index=False
    )
    return [_plot_scaling(summ, out)]


def _plot_scaling(summ: pd.DataFrame, out: str) -> str:
    panels = ["budget_avg", 0.02, 0.04, 0.06, 0.08, 0.1]
    fig, axes = plt.subplots(2, 3, figsize=(13.5, 7.6), layout="constrained")
    for ax, eps in zip(axes.ravel(), panels):
        sub = summ[summ["epsilon"].astype(str) == str(eps)]
        for arm in SCALING_ARMS:
            r = sub[sub["arm"] == arm].sort_values("grads_per_case")
            ax.errorbar(
                r["grads_per_case"],
                r["mean_dice"],
                yerr=[r["mean_dice"] - r["ci95_low"], r["ci95_high"] - r["mean_dice"]],
                marker="o",
                markersize=4,
                capsize=2,
                color=ARM_COLOUR[arm],
                label=ARM_LABELS[arm],
            )
        ax.set_xscale("log")
        ax.grid(alpha=0.3, which="both")
        ax.set_title(
            "budget-averaged" if eps == "budget_avg" else rf"$\varepsilon$ = {eps}",
            fontsize=10,
            fontweight="bold",
        )
        ax.set_xlabel("gradient evaluations per case (log)")
    for ax in axes[:, 0]:
        ax.set_ylabel("mean Dice (95% bootstrap CI)")
    axes[0, 0].legend(fontsize=8)
    fig.suptitle(
        "Iteration scaling on nnU-Net whole gland: does BLADE's lead survive more steps?",
        fontsize=12,
        fontweight="bold",
    )
    path = os.path.join(out, "scaling_dice_vs_gradients.png")
    fig.savefig(path, dpi=300, bbox_inches="tight")
    plt.close(fig)
    return path


# ------------------------------------------------------- factorial controls

# name -> (first arm, second arm, what the pair isolates). Every pair spends
# 315 gradients per case at 20 steps except the BLADE-DiceCE line, 105 each.
FACTORIAL_CONTRASTS = {
    "warm_start_blade_losses": (
        "blade",
        "blade_noladder",
        "ladder, BLADE's losses (BLADE-3 vs BLADE-3 no-ladder)",
    ),
    "warm_start_sea_losses": (
        "sea_ladder",
        "sea",
        "ladder, SEA's losses (SEA-ladder vs SEA)",
    ),
    "losses_no_ladder": ("blade_noladder", "sea", "BLADE vs SEA losses, no ladder"),
    "losses_on_ladder": ("blade", "sea_ladder", "BLADE vs SEA losses, on the ladder"),
    "ensemble_vs_restarts_ladder": (
        "blade",
        "blade1_r3",
        "3 objectives vs 3 restarts, on the ladder",
    ),
    "ladder_single_loss": (
        "blade1",
        "auto_pgd",
        "ladder, Dice+CE only (105 gradients)",
    ),
}
FACTORIAL_TARGETS = (("wg", "WG"), ("zones", "TZ+CZ"), ("zones", "PZ"))


def factorial_arms_analysis(out: str) -> list[str]:
    """Paired Dice contrasts of the three factorial controls on the replay subset.

    Existing arms come from the direction run and the controls from
    ``factorial_arms/``; both are 20 steps on the same fold-0 cases, and only
    cases every arm finished are used. Positive = first arm leaves lower Dice.
    """
    new = read(os.path.join(ROOT, "factorial_arms", "summary_*.csv"))
    old = read(os.path.join(ROOT, "direction", "summary_*.csv"))
    if new.empty or old.empty:
        return []
    arms = sorted({a for pair in FACTORIAL_CONTRASTS.values() for a in pair[:2]})
    keep = ["task", "arm", "case_id", "epsilon", "target", "adv_dice"]
    t = pd.concat([old[keep], new[keep]], ignore_index=True)
    t = t[t["arm"].isin(arms)]
    rows = []
    for task, target in FACTORIAL_TARGETS:
        sub = t[(t["task"] == task) & (t["target"] == target)]
        per_case = sub.groupby(["arm", "case_id"])["adv_dice"].mean().unstack("arm")
        if per_case.empty or not set(arms) <= set(per_case.columns):
            continue
        per_case = per_case.dropna(subset=arms)
        fam = []
        for name, (a, b, desc) in FACTORIAL_CONTRASTS.items():
            fam.append(
                {
                    "task": task,
                    "target": target,
                    "contrast": name,
                    "arm_a": a,
                    "arm_b": b,
                    "description": desc,
                    **_paired(per_case[b].to_numpy(), per_case[a].to_numpy()),
                }
            )
        inter = (
            (per_case["blade_noladder"] - per_case["blade"])
            - (per_case["sea"] - per_case["sea_ladder"])
        ).to_numpy()
        fam.append(
            {
                "task": task,
                "target": target,
                "contrast": "interaction",
                "arm_a": "",
                "arm_b": "",
                "description": "ladder gain with BLADE's losses minus with SEA's",
                **_paired(inter, np.zeros_like(inter)),
            }
        )
        f = pd.DataFrame(fam)
        f["p_adj_bh"] = _fdr_bh(f["p_value"].to_numpy())
        f["significant"] = f["p_adj_bh"] < ALPHA
        rows.append(f)
    if not rows:
        return []
    table = pd.concat(rows, ignore_index=True)
    table.to_csv(os.path.join(out, "factorial_arms_tests.csv"), index=False)
    means = t.groupby(["task", "target", "arm"])["adv_dice"].mean().reset_index()
    means.to_csv(os.path.join(out, "factorial_arms_mean_dice.csv"), index=False)
    return [_plot_factorial_arms(table, out)]


def _plot_factorial_arms(table: pd.DataFrame, out: str) -> str:
    order = [*FACTORIAL_CONTRASTS, "interaction"]
    targets = [
        t
        for t in FACTORIAL_TARGETS
        if ((table["task"] == t[0]) & (table["target"] == t[1])).any()
    ]
    fig, axes = plt.subplots(
        1,
        len(targets),
        figsize=(4.8 * len(targets), 4.2),
        sharey=True,
        layout="constrained",
    )
    axes = np.atleast_1d(axes)
    y = np.arange(len(order))
    for ax, (task, target) in zip(axes, targets):
        sub = table[(table["task"] == task) & (table["target"] == target)]
        sub = sub.set_index("contrast").loc[order]
        x = sub["hl_shift"].to_numpy()
        err = np.vstack([x - sub["hl_ci95_low"], sub["hl_ci95_high"] - x])
        sig = sub["significant"].to_numpy(dtype=bool)
        ax.errorbar(
            x, y, xerr=err, fmt="none", ecolor="#1a1a1a", elinewidth=1, capsize=2
        )
        ax.scatter(
            x[sig], y[sig], color="#2a78d6", s=36, zorder=3, label="significant (BH)"
        )
        ax.scatter(
            x[~sig],
            y[~sig],
            facecolors="white",
            edgecolors="#2a78d6",
            s=36,
            zorder=3,
            label="not significant",
        )
        ax.axvline(0, color="#6b6b6b", linewidth=0.8)
        ax.set_yticks(y)
        ax.set_yticklabels([sub.loc[c, "description"] for c in order], fontsize=8)
        ax.invert_yaxis()
        ax.set_title(target, fontsize=11, fontweight="bold")
        ax.set_xlabel("HL Dice shift (positive = first arm more damaging)")
        ax.grid(axis="x", alpha=0.3)
    axes[-1].legend(fontsize=8, loc="lower right")
    fig.suptitle(
        "Factorial controls on the replay subset (budget-averaged Dice, 95% CI)",
        fontsize=12,
        fontweight="bold",
    )
    path = os.path.join(out, "factorial_arms_contrasts.png")
    fig.savefig(path, dpi=300, bbox_inches="tight")
    plt.close(fig)
    return path


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--output-dir", default=os.path.join(ROOT, "analysis"))
    args = parser.parse_args()
    os.makedirs(args.output_dir, exist_ok=True)
    for path in [
        *direction_analysis(args.output_dir),
        *scaling_analysis(args.output_dir),
        *factorial_arms_analysis(args.output_dir),
    ]:
        print("wrote", path)


if __name__ == "__main__":
    main()
