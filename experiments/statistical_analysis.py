"""
Statistical Robustness Analysis for FGSM Adversarial Evaluation Results.

Reads per-sample CSV data (produced by the extended FGSM evaluation) and
generates:
  - Wilcoxon signed-rank tests (clean vs each epsilon)
  - Bootstrap 95 % confidence intervals
  - Per-sample scatter / Bland-Altman (one figure per class; ε ∈ {0, 0.5, 1})
  - Violin / box-plot distributions across epsilons
  - Correlation between WG and Zone robustness — one figure per zone class, all ε

Can also run in *summary-only* mode using aggregate CSVs (mean/std/sem)
when per-sample data is not yet available.

Wilcoxon paired tests report Cohen's d_z (paired) and Benjamini–Hochberg FDR
on the family of all tests in each table.
"""

import os
import argparse
import numpy as np
import pandas as pd
from scipy import stats
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

METRICS = ["dice", "hd95", "asd"]
METRIC_LABELS = {"dice": "Dice", "hd95": "HD95 (mm)", "asd": "ASD (mm)"}

# Defaults next to this script so analysis works when cwd is not the repo root.
_SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
_DEFAULT_INPUT_DIR = os.path.join(_SCRIPT_DIR, "fgsm_results")
_DEFAULT_OUTPUT_DIR = os.path.join(_SCRIPT_DIR, "statistical_results")


def _mask_eps(series: pd.Series, target: float) -> pd.Series:
    """Robust match for epsilon column (floats from CSV may not match exactly)."""
    return pd.Series(
        np.isclose(series.astype(float), float(target), rtol=1e-9, atol=0.0),
        index=series.index,
    )


def _epsilons_for_plots(df: pd.DataFrame, cli_eps: list[float] | None) -> list[float]:
    """If cli_eps is given and non-empty, use it; else all ε > 0 present in data."""
    if cli_eps is not None and len(cli_eps) > 0:
        return sorted(float(x) for x in cli_eps)
    out = sorted(
        float(e)
        for e in df["epsilon"].unique()
        if not np.isclose(float(e), 0.0, rtol=1e-9, atol=0.0)
    )
    return out


def _assert_unique_case_class_eps(df: pd.DataFrame, model_name: str) -> None:
    """One row per (case_id, epsilon, class) — expected schema."""
    dup = df.duplicated(["case_id", "epsilon", "class"], keep=False)
    if dup.any():
        bad = df.loc[dup, ["case_id", "epsilon", "class"]].drop_duplicates()
        raise ValueError(
            f"{model_name}: duplicate rows for (case_id, epsilon, class). Example:\n{bad.head()}"
        )


def _load_per_sample(input_dir: str, model_key: str):
    """Load per-sample CSV produced by the extended evaluation.
    Expected columns: case_id, epsilon, class, dice, hd95, asd
    Returns DataFrame or None if file does not exist.
    """
    path = os.path.join(input_dir, f"{model_key}_fgsm_per_sample.csv")
    if not os.path.isfile(path):
        return None
    return pd.read_csv(path)


def _load_aggregate(input_dir: str, model_key: str):
    """Load aggregate CSVs (one per metric) as produced by the original FGSM script."""
    frames = {}
    for metric in METRICS:
        path = os.path.join(input_dir, f"{model_key}_fgsm_{metric}.csv")
        if os.path.isfile(path):
            frames[metric] = pd.read_csv(path)
    return frames if frames else None


def _fdr_bh(pvals: np.ndarray) -> np.ndarray:
    """Benjamini–Hochberg adjusted p-values (two-sided tests)."""
    p = np.asarray(pvals, dtype=float)
    n = len(p)
    out = np.full(n, np.nan, dtype=float)
    finite = np.isfinite(p)
    if not finite.any():
        return out
    finite_idx = np.where(finite)[0]
    pv = p[finite_idx]
    m = len(pv)
    order = np.argsort(pv)
    sorted_p = pv[order]
    adj = sorted_p * m / np.arange(1, m + 1, dtype=float)
    adj = np.minimum.accumulate(adj[::-1])[::-1]
    adj = np.clip(adj, 0.0, 1.0)
    result = np.empty(m)
    result[order] = adj
    out[finite_idx] = result
    return out


def _bootstrap_ci(data: np.ndarray, n_boot: int = 10_000, ci: float = 0.95, seed: int = 42):
    rng = np.random.RandomState(seed)
    data = data[np.isfinite(data)]
    if len(data) < 2:
        return np.nan, np.nan
    idx = rng.randint(0, len(data), size=(n_boot, len(data)))
    boot_means = data[idx].mean(axis=1)
    alpha = (1.0 - ci) / 2.0
    return float(np.percentile(boot_means, 100 * alpha)), float(np.percentile(boot_means, 100 * (1 - alpha)))


def _cohens_d_paired(a: np.ndarray, b: np.ndarray) -> float:
    """Paired effect size (Cohen's d_z): mean(difference) / SD(differences)."""
    diff = np.asarray(a, dtype=np.float64) - np.asarray(b, dtype=np.float64)
    diff = diff[np.isfinite(diff)]
    if len(diff) < 2:
        return np.nan
    sd = diff.std(ddof=1)
    if sd == 0:
        return 0.0 if np.allclose(diff, 0.0) else np.nan
    return float(diff.mean() / sd)


def _wilcoxon_paired(a: np.ndarray, b: np.ndarray):
    """Paired Wilcoxon with explicit options; compatible with older SciPy."""
    kw = {"zero_method": "wilcox"}
    try:
        return stats.wilcoxon(a, b, **kw, method="auto")
    except TypeError:
        return stats.wilcoxon(a, b, **kw)


# ---------------------------------------------------------------------------
# Per-sample analyses
# ---------------------------------------------------------------------------

def wilcoxon_tests(df: pd.DataFrame, model_key: str, output_dir: str):
    """Paired Wilcoxon signed-rank test: clean (eps=0) vs each epsilon, per class per metric."""
    os.makedirs(output_dir, exist_ok=True)
    rows = []
    classes = sorted(df["class"].unique())
    epsilons = sorted(df["epsilon"].unique())
    eps0 = 0.0

    for cls in classes:
        clean = df[_mask_eps(df["epsilon"], eps0) & (df["class"] == cls)]
        for metric in METRICS:
            clean_vals = clean.set_index("case_id")[metric]
            for eps in epsilons:
                if np.isclose(float(eps), eps0, rtol=1e-9, atol=0.0):
                    continue
                adv = df[_mask_eps(df["epsilon"], eps) & (df["class"] == cls)]
                adv_vals = adv.set_index("case_id")[metric]
                common = clean_vals.index.intersection(adv_vals.index)
                a = clean_vals.loc[common].values.astype(np.float64)
                b = adv_vals.loc[common].values.astype(np.float64)
                mask = np.isfinite(a) & np.isfinite(b)
                a, b = a[mask], b[mask]
                if len(a) < 3:
                    continue
                if len(a) < 10:
                    print(
                        f"    Warning: Wilcoxon n={len(a)} < 10 for {model_key} "
                        f"class={cls!r} {metric} eps={eps}"
                    )
                try:
                    stat, pval = _wilcoxon_paired(a, b)
                except ValueError:
                    stat, pval = np.nan, np.nan
                d = _cohens_d_paired(a, b)
                rows.append({
                    "model": model_key, "class": cls, "metric": metric,
                    "epsilon": eps, "wilcoxon_stat": stat,
                    "p_value": pval, "cohens_dz": d, "n_pairs": len(a),
                })
    out = pd.DataFrame(rows)
    if len(out):
        out["p_value_fdr_bh"] = _fdr_bh(out["p_value"].values)
    path = os.path.join(output_dir, f"{model_key}_wilcoxon_tests.csv")
    out.to_csv(path, index=False, float_format="%.6g")
    print(f"  Saved {path}")
    return out


def bootstrap_confidence_intervals(df: pd.DataFrame, model_key: str, output_dir: str):
    os.makedirs(output_dir, exist_ok=True)
    rows = []
    for cls in sorted(df["class"].unique()):
        for eps in sorted(df["epsilon"].unique()):
            sub = df[_mask_eps(df["epsilon"], float(eps)) & (df["class"] == cls)]
            for metric in METRICS:
                vals = sub[metric].values.astype(np.float64)
                lo, hi = _bootstrap_ci(vals)
                rows.append({
                    "model": model_key, "class": cls, "epsilon": eps,
                    "metric": metric, "mean": float(np.nanmean(vals)),
                    "ci_lower": lo, "ci_upper": hi,
                })
    out = pd.DataFrame(rows)
    path = os.path.join(output_dir, f"{model_key}_bootstrap_ci.csv")
    out.to_csv(path, index=False, float_format="%.6f")
    print(f"  Saved {path}")
    return out


def _categorical_colors(n: int):
    """n distinct colors (tab20 if n>10, else tab10)."""
    cmap = plt.get_cmap("tab20" if n > 10 else "tab10")
    return [cmap(i % cmap.N) for i in range(n)]


# Scatter and Bland-Altman always use these ε (must exist in per-sample CSV for a series to appear).
SCATTER_BLAND_EPS: list[float] = [0.0, 0.5, 1.0]


def scatter_clean_vs_adversarial(
    df: pd.DataFrame, model_key: str, output_dir: str, _target_epsilons: list[float],
):
    """One figure per class: clean vs adversarial Dice for ε ∈ {0, 0.5, 1}.
    `_target_epsilons` is unused (main passes plot_eps for API symmetry)."""
    os.makedirs(output_dir, exist_ok=True)
    classes = sorted(df["class"].unique())
    eps0 = 0.0
    eps_list = SCATTER_BLAND_EPS
    colors = _categorical_colors(len(eps_list))

    for cls in classes:
        clean = df[_mask_eps(df["epsilon"], eps0) & (df["class"] == cls)].set_index("case_id")["dice"]
        fig, ax = plt.subplots(figsize=(7, 7))
        all_x, all_y = [], []
        n_plotted = 0
        for j, eps in enumerate(eps_list):
            adv = df[_mask_eps(df["epsilon"], eps) & (df["class"] == cls)].set_index("case_id")["dice"]
            common = clean.index.intersection(adv.index)
            if len(common) < 5:
                continue
            x, y = clean.loc[common].values, adv.loc[common].values
            all_x.append(x)
            all_y.append(y)
            ax.scatter(
                x, y, s=14, alpha=0.45, edgecolors="none",
                color=colors[j], label=f"ε={eps}",
            )
            n_plotted += 1

        if n_plotted == 0:
            plt.close(fig)
            continue

        xy_min = min(np.min(np.concatenate(all_x)), np.min(np.concatenate(all_y)))
        xy_max = max(np.max(np.concatenate(all_x)), np.max(np.concatenate(all_y)))
        lims = [xy_min - 0.02, xy_max + 0.02]
        ax.plot(lims, lims, "k--", alpha=0.35, linewidth=1.2, label="y = x")
        ax.set_xlim(lims)
        ax.set_ylim(lims)
        ax.set_xlabel("Clean Dice (ε=0)")
        ax.set_ylabel("Adversarial Dice")
        safe_cls = cls.replace("+", "").replace("/", "")
        ax.set_title(
            f"{model_key.upper()} — {cls}: Clean vs Adversarial Dice (ε ∈ {{0, 0.5, 1}})"
        )
        ax.legend(fontsize=8, loc="lower right", ncol=1)
        ax.grid(True, alpha=0.2)
        plt.tight_layout()
        path = os.path.join(output_dir, f"{model_key}_{safe_cls}_scatter.png")
        fig.savefig(path, dpi=200, bbox_inches="tight")
        plt.close(fig)
        print(f"  Saved {path}")


def bland_altman(
    df: pd.DataFrame, model_key: str, output_dir: str, _target_epsilons: list[float],
):
    """One figure per class: Bland-Altman for ε ∈ {0, 0.5, 1} (color-matched LOA)."""
    os.makedirs(output_dir, exist_ok=True)
    classes = sorted(df["class"].unique())
    eps0 = 0.0
    eps_list = SCATTER_BLAND_EPS
    colors = _categorical_colors(len(eps_list))

    for cls in classes:
        clean = df[_mask_eps(df["epsilon"], eps0) & (df["class"] == cls)].set_index("case_id")["dice"]
        fig, ax = plt.subplots(figsize=(8, 6))
        any_pts = False
        for j, eps in enumerate(eps_list):
            adv = df[_mask_eps(df["epsilon"], eps) & (df["class"] == cls)].set_index("case_id")["dice"]
            common = clean.index.intersection(adv.index)
            if len(common) < 5:
                continue
            x, y = clean.loc[common].values, adv.loc[common].values
            mean_vals = (x + y) / 2.0
            diff_vals = x - y
            md = diff_vals.mean()
            sd = diff_vals.std(ddof=1)
            n = len(diff_vals)
            c = colors[j]
            ax.scatter(
                mean_vals, diff_vals, s=12, alpha=0.45, edgecolors="none",
                color=c, label=f"ε={eps} (n={n})",
            )
            ax.axhline(md, color=c, linestyle="-", alpha=0.85, linewidth=1.1)
            ax.axhline(md + 1.96 * sd, color=c, linestyle="--", alpha=0.65, linewidth=0.9)
            ax.axhline(md - 1.96 * sd, color=c, linestyle="--", alpha=0.65, linewidth=0.9)
            any_pts = True

        if not any_pts:
            plt.close(fig)
            continue

        ax.set_xlabel("Mean Dice [(clean + adv) / 2]")
        ax.set_ylabel("Difference [clean − adv]")
        safe_cls = cls.replace("+", "").replace("/", "")
        ax.set_title(
            f"{model_key.upper()} — {cls}: Bland-Altman (ε ∈ {{0, 0.5, 1}})\n"
            "Solid = mean diff; dashed = ±1.96 SD (per ε)"
        )
        ax.legend(fontsize=7, loc="best", ncol=1)
        ax.grid(True, alpha=0.2)
        plt.tight_layout()
        path = os.path.join(output_dir, f"{model_key}_{safe_cls}_bland_altman.png")
        fig.savefig(path, dpi=200, bbox_inches="tight")
        plt.close(fig)
        print(f"  Saved {path}")


def violin_plots(df: pd.DataFrame, model_key: str, output_dir: str):
    """Violin plots of per-sample metrics at each epsilon."""
    os.makedirs(output_dir, exist_ok=True)
    classes = sorted(df["class"].unique())

    for metric in METRICS:
        for cls in classes:
            sub = df[df["class"] == cls]
            epsilons = sorted(sub["epsilon"].unique())
            eps_plot = []
            data_per_eps = []
            for eps in epsilons:
                vals = sub[_mask_eps(sub["epsilon"], float(eps))][metric].values.astype(np.float64)
                vals = vals[np.isfinite(vals)]
                if len(vals) == 0:
                    continue
                eps_plot.append(eps)
                data_per_eps.append(vals)
            if not data_per_eps:
                continue
            if any(len(v) < 2 for v in data_per_eps):
                print(
                    f"    Warning: violin {model_key} {cls} {metric}: some ε have <2 finite values"
                )

            fig, ax = plt.subplots(figsize=(max(8, len(eps_plot) * 0.9), 5))
            parts = ax.violinplot(data_per_eps, positions=range(len(eps_plot)),
                                  showmeans=True, showmedians=True, showextrema=False)
            for pc in parts["bodies"]:
                pc.set_alpha(0.6)
            ax.set_xticks(range(len(eps_plot)))
            ax.set_xticklabels([f"{e:.1f}" for e in eps_plot], fontsize=8)
            ax.set_xlabel("FGSM Epsilon")
            ax.set_ylabel(METRIC_LABELS[metric])
            safe_cls = cls.replace("+", "").replace("/", "")
            ax.set_title(f"{model_key.upper()} — {cls}: {METRIC_LABELS[metric]} Distribution")
            ax.grid(True, alpha=0.2, axis="y")
            plt.tight_layout()
            path = os.path.join(output_dir, f"{model_key}_{safe_cls}_violin_{metric}.png")
            fig.savefig(path, dpi=200, bbox_inches="tight")
            plt.close(fig)
            print(f"  Saved {path}")


def correlation_wg_zones(
    wg_df: pd.DataFrame,
    zones_df: pd.DataFrame,
    output_dir: str,
    target_epsilons: list[float],
):
    """One figure per zone class: WG vs zone Dice drop for all ε (colors + stats in legend)."""
    os.makedirs(output_dir, exist_ok=True)
    eps0 = 0.0
    eps_list = list(target_epsilons)
    colors = _categorical_colors(len(eps_list))

    wg_clean = wg_df[_mask_eps(wg_df["epsilon"], eps0)].set_index("case_id")["dice"]

    for cls in sorted(zones_df["class"].unique()):
        fig, ax = plt.subplots(figsize=(7.5, 7))
        n_plotted = 0
        for j, eps in enumerate(eps_list):
            wg_adv = wg_df[_mask_eps(wg_df["epsilon"], eps)].set_index("case_id")["dice"]
            wg_drop = (wg_clean - wg_adv).dropna()

            z_clean = zones_df[
                _mask_eps(zones_df["epsilon"], eps0) & (zones_df["class"] == cls)
            ].set_index("case_id")["dice"]
            z_adv = zones_df[
                _mask_eps(zones_df["epsilon"], eps) & (zones_df["class"] == cls)
            ].set_index("case_id")["dice"]
            z_drop = (z_clean - z_adv).dropna()

            common = wg_drop.index.intersection(z_drop.index)
            if len(common) < 3:
                continue
            if len(common) < 10:
                print(
                    f"    Warning: WG–zones correlation n={len(common)} < 10 "
                    f"(class={cls!r}, ε={eps})"
                )
            x, y = wg_drop.loc[common].values, z_drop.loc[common].values
            r_p, p_p = stats.pearsonr(x, y)
            r_s, p_s = stats.spearmanr(x, y)
            lab = (
                f"ε={eps}: n={len(common)}, "
                f"r={r_p:.2f}, ρ={r_s:.2f}"
            )
            ax.scatter(
                x, y, s=16, alpha=0.42, edgecolors="none",
                color=colors[j % len(colors)], label=lab,
            )
            n_plotted += 1

        if n_plotted == 0:
            plt.close(fig)
            continue

        ax.set_xlabel("WG Dice Drop")
        ax.set_ylabel(f"{cls} Dice Drop")
        safe_cls = cls.replace("+", "").replace("/", "")
        ax.set_title(f"WG vs {cls} Dice Drop (all ε)")
        ax.legend(fontsize=7, loc="best")
        ax.grid(True, alpha=0.2)
        plt.tight_layout()
        path = os.path.join(output_dir, f"corr_wg_{safe_cls}.png")
        fig.savefig(path, dpi=200, bbox_inches="tight")
        plt.close(fig)
        print(f"  Saved {path}")


# ---------------------------------------------------------------------------
# Aggregate-only fallback analyses
# ---------------------------------------------------------------------------

def aggregate_summary_table(input_dir: str, output_dir: str):
    """Generate a combined degradation summary table from aggregate CSVs."""
    os.makedirs(output_dir, exist_ok=True)
    rows = []
    for mk in ("wg", "zones"):
        agg = _load_aggregate(input_dir, mk)
        if agg is None:
            continue
        for metric, mdf in agg.items():
            for _, r in mdf.iterrows():
                rows.append({
                    "model": mk, "class": r["class"], "epsilon": r["epsilon"],
                    "metric": metric, "mean": r["mean"], "std": r["std"],
                })
    if not rows:
        print(f"  No aggregate CSVs found under {os.path.abspath(input_dir)} "
              f"(expected {os.path.join(input_dir, 'wg_fgsm_dice.csv')} etc.).")
        return
    df = pd.DataFrame(rows)

    clean = df[_mask_eps(df["epsilon"], 0.0)][["model", "class", "metric", "mean"]].rename(
        columns={"mean": "clean_mean"}
    )
    merged = df.merge(clean, on=["model", "class", "metric"], how="left")
    eps = np.finfo(float).eps
    denom = np.where(np.abs(merged["clean_mean"].astype(float)) > eps, merged["clean_mean"], np.nan)
    merged["pct_change"] = ((merged["mean"] - merged["clean_mean"]) / denom) * 100.0
    merged["pct_change_undef"] = np.abs(merged["clean_mean"].astype(float)) <= eps

    path = os.path.join(output_dir, "degradation_summary.csv")
    merged.to_csv(path, index=False, float_format="%.4f")
    print(f"  Saved {path}")

    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    for i, metric in enumerate(METRICS):
        ax = axes[i]
        sub = merged[merged["metric"] == metric]
        for _, grp in sub.groupby(["model", "class"]):
            label = f"{grp['model'].iloc[0].upper()} {grp['class'].iloc[0]}"
            ax.plot(grp["epsilon"], grp["pct_change"], marker="o", markersize=4, label=label)
        ax.set_xlabel("FGSM Epsilon")
        ax.set_ylabel("% Change from Clean")
        ax.set_title(f"{METRIC_LABELS[metric]} — % Degradation")
        ax.legend(fontsize=7)
        ax.grid(True, alpha=0.2)
        ax.axhline(0, color="black", linewidth=0.5)
    plt.tight_layout()
    path = os.path.join(output_dir, "degradation_pct_change.png")
    fig.savefig(path, dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved {path}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="Statistical analysis of FGSM evaluation results")
    parser.add_argument(
        "--input-dir",
        default=_DEFAULT_INPUT_DIR,
        help=f"Directory with FGSM CSV results (default: {_DEFAULT_INPUT_DIR})",
    )
    parser.add_argument(
        "--output-dir",
        default=_DEFAULT_OUTPUT_DIR,
        help=f"Output directory (default: {_DEFAULT_OUTPUT_DIR})",
    )
    parser.add_argument(
        "--plot-eps",
        type=float,
        nargs="*",
        default=None,
        help="ε list for WG–zones correlation (and filtering). "
        "Scatter & Bland-Altman use fixed ε ∈ {0, 0.5, 1}. Default: all ε>0 in per-sample data.",
    )
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)

    print("=" * 60)
    print("Statistical Robustness Analysis")
    print("=" * 60)
    print(f"  Input:  {os.path.abspath(args.input_dir)}")
    print(f"  Output: {os.path.abspath(args.output_dir)}")

    aggregate_summary_table(args.input_dir, args.output_dir)

    wg_df = _load_per_sample(args.input_dir, "wg")
    zones_df = _load_per_sample(args.input_dir, "zones")

    for mk, df in [("wg", wg_df), ("zones", zones_df)]:
        if df is None:
            p = os.path.join(args.input_dir, f"{mk}_fgsm_per_sample.csv")
            print(
                f"\n  [{mk}] Per-sample CSV not found:\n"
                f"        {os.path.abspath(p)}\n"
                f"        Run FGSM evaluation with per-sample CSV export, e.g.:\n"
                f"        python {os.path.join(_SCRIPT_DIR, 'fgsm_adversarial_evaluation.py')} "
                f"--model {mk} --output-dir {os.path.abspath(args.input_dir)}"
            )
            continue
        print(f"\n  [{mk}] Per-sample data loaded: {len(df)} rows")
        _assert_unique_case_class_eps(df, mk)
        plot_eps = _epsilons_for_plots(df, args.plot_eps)
        wilcoxon_tests(df, mk, args.output_dir)
        bootstrap_confidence_intervals(df, mk, args.output_dir)
        if plot_eps:
            scatter_clean_vs_adversarial(df, mk, args.output_dir, plot_eps)
            bland_altman(df, mk, args.output_dir, plot_eps)
        else:
            print(f"  [{mk}] No ε>0 in data for scatter/Bland-Altman (set --plot-eps).")
        violin_plots(df, mk, args.output_dir)

    if wg_df is not None and zones_df is not None:
        plot_eps = _epsilons_for_plots(
            pd.concat([wg_df, zones_df], ignore_index=True),
            args.plot_eps,
        )

        def _both_have_eps(e: float) -> bool:
            return not wg_df[_mask_eps(wg_df["epsilon"], e)].empty and not zones_df[
                _mask_eps(zones_df["epsilon"], e)
            ].empty

        plot_eps = [e for e in plot_eps if _both_have_eps(e)]
        if plot_eps:
            correlation_wg_zones(wg_df, zones_df, args.output_dir, plot_eps)
        else:
            print("\n  Cross-model correlation skipped (no overlapping ε>0 in both CSVs).")
    else:
        print("\n  Cross-model correlation skipped (need both per-sample CSVs).")

    print("\nDone.")


if __name__ == "__main__":
    main()
