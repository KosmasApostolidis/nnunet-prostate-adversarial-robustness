"""
Compare Standard vs Robust models under FGSM adversarial perturbations.

Reads pre-computed CSV results from fgsm_results/ and fgsm_results_robust/
and produces comparison plots (Dice, HD95, ASD) plus summary tables for
both the Whole-Gland (WG) and Prostate-Zones (TZ+CZ, PZ) models.
"""

import os
import numpy as np
import pandas as pd
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy import stats

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
STANDARD_DIR = os.path.join(SCRIPT_DIR, "fgsm_results")
ROBUST_DIR = os.path.join(SCRIPT_DIR, "fgsm_results_robust")
OUTPUT_DIR_WG = os.path.join(SCRIPT_DIR, "fgsm_comparison")
OUTPUT_DIR_ZONES = os.path.join(SCRIPT_DIR, "fgsm_comparison_zones")

WG_METRICS = {
    "dice": {
        "file_std": "wg_fgsm_dice.csv",
        "file_rob": "wg_robust_fgsm_dice.csv",
        "label": "Dice Score",
        "higher_better": True,
    },
    "hd95": {
        "file_std": "wg_fgsm_hd95.csv",
        "file_rob": "wg_robust_fgsm_hd95.csv",
        "label": "HD95 (mm)",
        "higher_better": False,
    },
    "asd": {
        "file_std": "wg_fgsm_asd.csv",
        "file_rob": "wg_robust_fgsm_asd.csv",
        "label": "ASD (mm)",
        "higher_better": False,
    },
}

ZONES_METRICS = {
    "dice": {
        "file_std": "zones_fgsm_dice.csv",
        "file_rob": "zones_robust_fgsm_dice.csv",
        "label": "Dice Score",
        "higher_better": True,
    },
    "hd95": {
        "file_std": "zones_fgsm_hd95.csv",
        "file_rob": "zones_robust_fgsm_hd95.csv",
        "label": "HD95 (mm)",
        "higher_better": False,
    },
    "asd": {
        "file_std": "zones_fgsm_asd.csv",
        "file_rob": "zones_robust_fgsm_asd.csv",
        "label": "ASD (mm)",
        "higher_better": False,
    },
}

ZONE_CLASSES = ["TZ+CZ", "PZ"]

COLORS = {"Standard": "#1f77b4", "Robust": "#d62728"}


def load_summary(path, class_filter=None):
    df = pd.read_csv(path)
    if class_filter is not None and "class" in df.columns:
        df = df[df["class"] == class_filter].reset_index(drop=True)
    return df


def load_per_sample(path, class_filter=None):
    df = pd.read_csv(path)
    if class_filter is not None and "class" in df.columns:
        df = df[df["class"] == class_filter].reset_index(drop=True)
    return df


def plot_metric_comparison(
    metric_key,
    cfg,
    std_df,
    rob_df,
    output_dir,
    title_prefix="Whole Gland",
    file_prefix="comparison",
):
    """Single metric: overlay standard vs robust with error bands."""
    fig, ax = plt.subplots(figsize=(9, 5.5))

    for label, df, color in [
        ("Standard", std_df, COLORS["Standard"]),
        ("Robust", rob_df, COLORS["Robust"]),
    ]:
        eps = df["epsilon"].values
        mean = df["mean"].values
        sem = df["sem"].values
        ax.plot(
            eps, mean, marker="o", color=color, linewidth=2.2, label=label, zorder=3
        )
        ax.fill_between(eps, mean - sem, mean + sem, color=color, alpha=0.15, zorder=2)

    ax.set_xlabel("FGSM Epsilon", fontsize=13)
    ax.set_ylabel(cfg["label"], fontsize=13)
    ax.set_title(
        f"{title_prefix} — {cfg['label']} vs FGSM Epsilon",
        fontsize=14,
        fontweight="bold",
    )
    ax.legend(fontsize=12, framealpha=0.9)
    ax.grid(True, alpha=0.3)
    ax.set_xticks(std_df["epsilon"].values)
    plt.tight_layout()
    path = os.path.join(output_dir, f"{file_prefix}_{metric_key}.png")
    plt.savefig(path, dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved {path}")


def plot_degradation(
    std_summaries,
    rob_summaries,
    metrics_dict,
    output_dir,
    title_prefix="",
    file_prefix="comparison",
):
    """Relative degradation from epsilon=0 baseline for each metric."""
    fig, axes = plt.subplots(1, 3, figsize=(18, 5.5))

    for ax, (metric_key, cfg) in zip(axes, metrics_dict.items()):
        std_df = std_summaries[metric_key]
        rob_df = rob_summaries[metric_key]

        std_base = std_df.loc[std_df["epsilon"] == 0.0, "mean"].values[0]
        rob_base = rob_df.loc[rob_df["epsilon"] == 0.0, "mean"].values[0]

        eps = std_df["epsilon"].values
        if cfg["higher_better"]:
            std_deg = (std_base - std_df["mean"].values) / std_base * 100
            rob_deg = (rob_base - rob_df["mean"].values) / rob_base * 100
            ylabel = "Relative Degradation (%)"
        else:
            std_deg = (std_df["mean"].values - std_base) / std_base * 100
            rob_deg = (rob_df["mean"].values - rob_base) / rob_base * 100
            ylabel = "Relative Increase (%)"

        ax.plot(
            eps,
            std_deg,
            marker="s",
            color=COLORS["Standard"],
            linewidth=2.2,
            label="Standard",
        )
        ax.plot(
            eps,
            rob_deg,
            marker="o",
            color=COLORS["Robust"],
            linewidth=2.2,
            label="Robust",
        )

        ax.set_xlabel("FGSM Epsilon", fontsize=12)
        ax.set_ylabel(ylabel, fontsize=12)
        ax.set_title(cfg["label"], fontsize=13, fontweight="bold")
        ax.legend(fontsize=11)
        ax.grid(True, alpha=0.3)
        ax.set_xticks(eps)

    suptitle = "Relative Performance Change from Clean Baseline (ε=0)"
    if title_prefix:
        suptitle = f"{title_prefix}: {suptitle}"
    fig.suptitle(suptitle, fontsize=14, fontweight="bold", y=1.02)
    plt.tight_layout()
    path = os.path.join(output_dir, f"{file_prefix}_degradation.png")
    plt.savefig(path, dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved {path}")


def plot_combined(
    std_summaries,
    rob_summaries,
    metrics_dict,
    output_dir,
    title="Whole Gland: Standard vs Robust Model under FGSM Attack",
    file_prefix="comparison",
):
    """All three metrics in a single figure (3 subplots)."""
    fig, axes = plt.subplots(1, 3, figsize=(18, 5.5))

    for ax, (metric_key, cfg) in zip(axes, metrics_dict.items()):
        std_df = std_summaries[metric_key]
        rob_df = rob_summaries[metric_key]

        for label, df, color in [
            ("Standard", std_df, COLORS["Standard"]),
            ("Robust", rob_df, COLORS["Robust"]),
        ]:
            eps = df["epsilon"].values
            mean = df["mean"].values
            sem = df["sem"].values
            ax.plot(eps, mean, marker="o", color=color, linewidth=2.2, label=label)
            ax.fill_between(eps, mean - sem, mean + sem, color=color, alpha=0.15)

        ax.set_xlabel("FGSM Epsilon", fontsize=12)
        ax.set_ylabel(cfg["label"], fontsize=12)
        ax.set_title(cfg["label"], fontsize=13, fontweight="bold")
        ax.legend(fontsize=11)
        ax.grid(True, alpha=0.3)
        ax.set_xticks(std_df["epsilon"].values)

    fig.suptitle(title, fontsize=14, fontweight="bold", y=1.02)
    plt.tight_layout()
    path = os.path.join(output_dir, f"{file_prefix}_combined.png")
    plt.savefig(path, dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved {path}")


def paired_statistical_tests(
    std_ps_path,
    rob_ps_path,
    output_dir,
    class_filter=None,
    file_prefix="statistical_comparison",
):
    """Per-sample paired Wilcoxon signed-rank test at each epsilon."""
    std_ps = load_per_sample(std_ps_path, class_filter=class_filter)
    rob_ps = load_per_sample(rob_ps_path, class_filter=class_filter)

    epsilons = sorted(std_ps["epsilon"].unique())
    rows = []

    for eps in epsilons:
        std_sub = (
            std_ps[std_ps["epsilon"] == eps]
            .sort_values("case_id")
            .reset_index(drop=True)
        )
        rob_sub = (
            rob_ps[rob_ps["epsilon"] == eps]
            .sort_values("case_id")
            .reset_index(drop=True)
        )

        assert (std_sub["case_id"].values == rob_sub["case_id"].values).all(), (
            f"Case IDs mismatch at epsilon={eps}"
        )

        for metric in ["dice", "hd95", "asd"]:
            s_vals = std_sub[metric].values.astype(float)
            r_vals = rob_sub[metric].values.astype(float)

            diff = r_vals - s_vals
            mean_diff = np.nanmean(diff)

            finite_mask = np.isfinite(diff) & (diff != 0)
            if finite_mask.sum() >= 10:
                stat, p_val = stats.wilcoxon(s_vals[finite_mask], r_vals[finite_mask])
            else:
                stat, p_val = np.nan, np.nan

            rows.append(
                {
                    "epsilon": eps,
                    "metric": metric,
                    "std_mean": np.nanmean(s_vals),
                    "rob_mean": np.nanmean(r_vals),
                    "mean_diff": mean_diff,
                    "wilcoxon_stat": stat,
                    "p_value": p_val,
                    "significant_005": "Yes"
                    if p_val < 0.05
                    else ("No" if np.isfinite(p_val) else "N/A"),
                }
            )

    results_df = pd.DataFrame(rows)
    path = os.path.join(output_dir, f"{file_prefix}.csv")
    results_df.to_csv(path, index=False, float_format="%.6f")
    print(f"  Saved {path}")
    return results_df


def print_summary_table(
    std_summaries,
    rob_summaries,
    metrics_dict,
    stat_df,
    banner="WHOLE GLAND: STANDARD vs ROBUST MODEL — FGSM ADVERSARIAL COMPARISON",
):
    """Print a formatted comparison table to stdout."""
    print("\n" + "=" * 100)
    print(banner)
    print("=" * 100)

    for metric_key, cfg in metrics_dict.items():
        std_df = std_summaries[metric_key]
        rob_df = rob_summaries[metric_key]

        print(f"\n--- {cfg['label']} ---")
        header = f"{'Epsilon':<10} {'Standard':>12} {'Robust':>12} {'Diff':>12} {'Better':>10} {'p-value':>12}"
        print(header)
        print("-" * len(header))

        for _, srow in std_df.iterrows():
            eps = srow["epsilon"]
            rrow = rob_df[rob_df["epsilon"] == eps].iloc[0]

            s_mean = srow["mean"]
            r_mean = rrow["mean"]
            diff = r_mean - s_mean

            if cfg["higher_better"]:
                better = "Robust" if diff > 0 else "Standard"
            else:
                better = "Robust" if diff < 0 else "Standard"

            stat_row = stat_df[
                (stat_df["epsilon"] == eps) & (stat_df["metric"] == metric_key)
            ]
            p_val = stat_row["p_value"].values[0] if len(stat_row) > 0 else np.nan
            p_str = f"{p_val:.4f}" if np.isfinite(p_val) else "N/A"
            sig = "*" if (np.isfinite(p_val) and p_val < 0.05) else ""

            print(
                f"{eps:<10.1f} {s_mean:>12.4f} {r_mean:>12.4f} {diff:>+12.4f} {better:>10} {p_str:>11}{sig}"
            )

    print("\n* = significant at p < 0.05 (Wilcoxon signed-rank test)")
    print()


def run_wg_comparison():
    """Run the Whole-Gland standard-vs-robust comparison."""
    os.makedirs(OUTPUT_DIR_WG, exist_ok=True)
    print("\n" + "#" * 70)
    print("# WHOLE GLAND COMPARISON")
    print("#" * 70)
    print("Loading WG summary CSVs...")

    std_summaries, rob_summaries = {}, {}
    for metric_key, cfg in WG_METRICS.items():
        std_summaries[metric_key] = load_summary(
            os.path.join(STANDARD_DIR, cfg["file_std"])
        )
        rob_summaries[metric_key] = load_summary(
            os.path.join(ROBUST_DIR, cfg["file_rob"])
        )

    print("Generating WG comparison plots...")
    for metric_key, cfg in WG_METRICS.items():
        plot_metric_comparison(
            metric_key,
            cfg,
            std_summaries[metric_key],
            rob_summaries[metric_key],
            OUTPUT_DIR_WG,
        )

    plot_combined(
        std_summaries,
        rob_summaries,
        WG_METRICS,
        OUTPUT_DIR_WG,
        title="Whole Gland: Standard vs Robust Model under FGSM Attack",
    )
    plot_degradation(
        std_summaries,
        rob_summaries,
        WG_METRICS,
        OUTPUT_DIR_WG,
        title_prefix="Whole Gland",
    )

    print("Running WG paired statistical tests...")
    stat_df = paired_statistical_tests(
        os.path.join(STANDARD_DIR, "wg_fgsm_per_sample.csv"),
        os.path.join(ROBUST_DIR, "wg_robust_fgsm_per_sample.csv"),
        OUTPUT_DIR_WG,
    )

    print_summary_table(
        std_summaries,
        rob_summaries,
        WG_METRICS,
        stat_df,
        banner="WHOLE GLAND: STANDARD vs ROBUST MODEL — FGSM ADVERSARIAL COMPARISON",
    )
    print(f"WG outputs saved to: {OUTPUT_DIR_WG}/")


def run_zones_comparison():
    """Run the Prostate-Zones standard-vs-robust comparison (per zone class)."""
    os.makedirs(OUTPUT_DIR_ZONES, exist_ok=True)
    print("\n" + "#" * 70)
    print("# PROSTATE ZONES COMPARISON")
    print("#" * 70)

    for zone_cls in ZONE_CLASSES:
        safe_name = zone_cls.replace("+", "").replace("/", "")
        zone_dir = os.path.join(OUTPUT_DIR_ZONES, safe_name)
        os.makedirs(zone_dir, exist_ok=True)

        print(f"\n--- Zone: {zone_cls} ---")
        print(f"Loading {zone_cls} summary CSVs...")

        std_summaries, rob_summaries = {}, {}
        for metric_key, cfg in ZONES_METRICS.items():
            std_summaries[metric_key] = load_summary(
                os.path.join(STANDARD_DIR, cfg["file_std"]), class_filter=zone_cls
            )
            rob_summaries[metric_key] = load_summary(
                os.path.join(ROBUST_DIR, cfg["file_rob"]), class_filter=zone_cls
            )

        print(f"Generating {zone_cls} comparison plots...")
        for metric_key, cfg in ZONES_METRICS.items():
            plot_metric_comparison(
                metric_key,
                cfg,
                std_summaries[metric_key],
                rob_summaries[metric_key],
                zone_dir,
                title_prefix=f"Prostate Zones — {zone_cls}",
                file_prefix="comparison",
            )

        plot_combined(
            std_summaries,
            rob_summaries,
            ZONES_METRICS,
            zone_dir,
            title=f"Prostate Zones ({zone_cls}): Standard vs Robust Model under FGSM Attack",
        )
        plot_degradation(
            std_summaries,
            rob_summaries,
            ZONES_METRICS,
            zone_dir,
            title_prefix=f"Prostate Zones — {zone_cls}",
        )

        print(f"Running {zone_cls} paired statistical tests...")
        stat_df = paired_statistical_tests(
            os.path.join(STANDARD_DIR, "zones_fgsm_per_sample.csv"),
            os.path.join(ROBUST_DIR, "zones_robust_fgsm_per_sample.csv"),
            zone_dir,
            class_filter=zone_cls,
        )

        print_summary_table(
            std_summaries,
            rob_summaries,
            ZONES_METRICS,
            stat_df,
            banner=f"PROSTATE ZONES ({zone_cls}): STANDARD vs ROBUST MODEL — FGSM ADVERSARIAL COMPARISON",
        )

    print(f"Zones outputs saved to: {OUTPUT_DIR_ZONES}/")


def main():
    run_wg_comparison()
    run_zones_comparison()
    print("\nDone — all comparisons complete.")


if __name__ == "__main__":
    main()
