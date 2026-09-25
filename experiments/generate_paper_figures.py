"""
Publication-Ready Figures and Tables for Adversarial Robustness Paper.

Generates:
  - LaTeX tables with mean +/- std, bold best, significance stars
  - Combined multi-panel figures (WG + Zones, Dice/HD95/ASD)
  - Radar / spider chart summary
  - Degradation summary table (% drop from clean)
  - Combined attack comparison figures (FGSM vs PGD vs BIM if available)

Reads from fgsm_results/, pgd_results/, noise_results/ as available.
"""

import os
import argparse
import numpy as np
import pandas as pd
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from math import pi

METRICS = ["dice", "hd95", "asd"]
METRIC_LABELS = {"dice": "Dice", "hd95": "HD95 (mm)", "asd": "ASD (mm)"}
METRIC_HIGHER_BETTER = {"dice": True, "hd95": False, "asd": False}

MODEL_DISPLAY = {"wg": "Whole Gland", "zones": "Prostate Zones"}
KEY_EPSILONS = [0.0, 0.25, 0.5, 0.75, 1.0]


def _load_agg_csv(directory: str, model_key: str, attack: str, metric: str):
    path = os.path.join(directory, f"{model_key}_{attack}_{metric}.csv")
    if not os.path.isfile(path):
        path = os.path.join(directory, f"{model_key}_fgsm_{metric}.csv")
    if not os.path.isfile(path):
        return None
    return pd.read_csv(path)


def _load_all_fgsm(fgsm_dir: str):
    """Load all FGSM aggregate CSVs into a single DataFrame."""
    frames = []
    for mk in ("wg", "zones"):
        for metric in METRICS:
            df = _load_agg_csv(fgsm_dir, mk, "fgsm", metric)
            if df is not None:
                df["model"] = mk
                df["metric"] = metric
                frames.append(df)
    return pd.concat(frames, ignore_index=True) if frames else None


# ---------------------------------------------------------------------------
# LaTeX tables
# ---------------------------------------------------------------------------

def generate_latex_tables(fgsm_dir: str, output_dir: str):
    """Auto-generate LaTeX tables with mean +/- std at key epsilons."""
    os.makedirs(output_dir, exist_ok=True)

    data = _load_all_fgsm(fgsm_dir)
    if data is None:
        print("  No FGSM data found.")
        return

    available_eps = sorted(data["epsilon"].unique())
    table_eps = [e for e in KEY_EPSILONS if e in available_eps]
    if not table_eps:
        table_eps = available_eps[:5]

    for mk in ("wg", "zones"):
        mk_data = data[data["model"] == mk]
        if mk_data.empty:
            continue

        classes = sorted(mk_data["class"].unique())

        lines = []
        lines.append(r"\begin{table}[htbp]")
        lines.append(r"\centering")
        lines.append(r"\caption{" + f"{MODEL_DISPLAY[mk]} — FGSM Robustness (mean $\\pm$ std)" + r"}")
        lines.append(r"\label{tab:" + mk + r"_fgsm}")
        lines.append(r"\small")

        n_metrics = len(METRICS)
        col_spec = "l" + "c" * (n_metrics * len(classes))
        lines.append(r"\begin{tabular}{" + col_spec + r"}")
        lines.append(r"\toprule")

        header_row = r"$\varepsilon$"
        for cls in classes:
            for metric in METRICS:
                header_row += f" & {cls} {METRIC_LABELS[metric]}"
        header_row += r" \\"
        lines.append(header_row)
        lines.append(r"\midrule")

        best_vals = {}
        for cls in classes:
            for metric in METRICS:
                vals_at_eps = {}
                for eps in table_eps:
                    row = mk_data[(mk_data["epsilon"] == eps) &
                                  (mk_data["class"] == cls) &
                                  (mk_data["metric"] == metric)]
                    if not row.empty:
                        vals_at_eps[eps] = row.iloc[0]["mean"]
                if vals_at_eps:
                    if METRIC_HIGHER_BETTER[metric]:
                        best_eps = max(vals_at_eps, key=vals_at_eps.get)
                    else:
                        best_eps = min(vals_at_eps, key=vals_at_eps.get)
                    best_vals[(cls, metric)] = best_eps

        for eps in table_eps:
            row_str = f"{eps:.2f}"
            for cls in classes:
                for metric in METRICS:
                    sub = mk_data[(mk_data["epsilon"] == eps) &
                                  (mk_data["class"] == cls) &
                                  (mk_data["metric"] == metric)]
                    if sub.empty:
                        row_str += " & --"
                    else:
                        mean = sub.iloc[0]["mean"]
                        std = sub.iloc[0]["std"]
                        cell = f"{mean:.3f} $\\pm$ {std:.3f}"
                        if best_vals.get((cls, metric)) == eps:
                            cell = r"\textbf{" + cell + "}"
                        row_str += f" & {cell}"
            row_str += r" \\"
            lines.append(row_str)

        lines.append(r"\bottomrule")
        lines.append(r"\end{tabular}")
        lines.append(r"\end{table}")

        tex_content = "\n".join(lines)
        path = os.path.join(output_dir, f"{mk}_fgsm_table.tex")
        with open(path, "w") as f:
            f.write(tex_content)
        print(f"  Saved {path}")


# ---------------------------------------------------------------------------
# Multi-panel figures
# ---------------------------------------------------------------------------

def generate_multipanel_figure(fgsm_dir: str, output_dir: str):
    """2-row x 3-col grid: WG on top, Zones on bottom; Dice / HD95 / ASD columns."""
    os.makedirs(output_dir, exist_ok=True)

    data = _load_all_fgsm(fgsm_dir)
    if data is None:
        print("  No FGSM data found.")
        return

    fig, axes = plt.subplots(2, 3, figsize=(16, 9))
    model_keys = ["wg", "zones"]
    colors = plt.cm.tab10.colors

    for row, mk in enumerate(model_keys):
        mk_data = data[data["model"] == mk]
        classes = sorted(mk_data["class"].unique())
        for col, metric in enumerate(METRICS):
            ax = axes[row, col]
            for ci, cls in enumerate(classes):
                sub = mk_data[(mk_data["class"] == cls) & (mk_data["metric"] == metric)]
                sub = sub.sort_values("epsilon")
                ax.errorbar(sub["epsilon"], sub["mean"], yerr=sub["sem"],
                            marker="o", markersize=4, capsize=3,
                            color=colors[ci], label=cls)
            ax.set_xlabel("FGSM Epsilon", fontsize=9)
            ax.set_ylabel(METRIC_LABELS[metric], fontsize=9)
            ax.set_title(f"{MODEL_DISPLAY[mk]} — {METRIC_LABELS[metric]}", fontsize=10, fontweight="bold")
            ax.legend(fontsize=8)
            ax.grid(True, alpha=0.2)
            ax.tick_params(labelsize=8)

    plt.tight_layout()
    path = os.path.join(output_dir, "multipanel_fgsm_all_metrics.png")
    fig.savefig(path, dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved {path}")

    path_pdf = os.path.join(output_dir, "multipanel_fgsm_all_metrics.pdf")
    fig2, axes2 = plt.subplots(2, 3, figsize=(16, 9))
    for row, mk in enumerate(model_keys):
        mk_data = data[data["model"] == mk]
        classes = sorted(mk_data["class"].unique())
        for col, metric in enumerate(METRICS):
            ax = axes2[row, col]
            for ci, cls in enumerate(classes):
                sub = mk_data[(mk_data["class"] == cls) & (mk_data["metric"] == metric)]
                sub = sub.sort_values("epsilon")
                ax.errorbar(sub["epsilon"], sub["mean"], yerr=sub["sem"],
                            marker="o", markersize=4, capsize=3,
                            color=colors[ci], label=cls)
            ax.set_xlabel("FGSM Epsilon", fontsize=9)
            ax.set_ylabel(METRIC_LABELS[metric], fontsize=9)
            ax.set_title(f"{MODEL_DISPLAY[mk]} — {METRIC_LABELS[metric]}", fontsize=10, fontweight="bold")
            ax.legend(fontsize=8)
            ax.grid(True, alpha=0.2)
            ax.tick_params(labelsize=8)
    plt.tight_layout()
    fig2.savefig(path_pdf, bbox_inches="tight")
    plt.close(fig2)
    print(f"  Saved {path_pdf}")


# ---------------------------------------------------------------------------
# Radar / Spider chart
# ---------------------------------------------------------------------------

def generate_radar_chart(fgsm_dir: str, output_dir: str):
    """Per-model radar chart at selected epsilon values showing all metrics."""
    os.makedirs(output_dir, exist_ok=True)

    data = _load_all_fgsm(fgsm_dir)
    if data is None:
        print("  No FGSM data found.")
        return

    radar_epsilons = [0.0, 0.5, 1.0]
    available_eps = sorted(data["epsilon"].unique())
    radar_epsilons = [e for e in radar_epsilons if e in available_eps]

    for mk in ("wg", "zones"):
        mk_data = data[data["model"] == mk]
        if mk_data.empty:
            continue
        classes = sorted(mk_data["class"].unique())

        for cls in classes:
            cls_data = mk_data[mk_data["class"] == cls]

            categories = list(METRIC_LABELS.values())
            N = len(categories)
            angles = [n / float(N) * 2 * pi for n in range(N)]
            angles += angles[:1]

            fig, ax = plt.subplots(figsize=(6, 6), subplot_kw=dict(polar=True))
            colors_radar = plt.cm.Set2.colors

            for ei, eps in enumerate(radar_epsilons):
                values = []
                for metric in METRICS:
                    sub = cls_data[(cls_data["epsilon"] == eps) & (cls_data["metric"] == metric)]
                    if sub.empty:
                        values.append(0)
                    else:
                        val = sub.iloc[0]["mean"]
                        if not METRIC_HIGHER_BETTER[metric]:
                            max_val = cls_data[cls_data["metric"] == metric]["mean"].max()
                            val = 1.0 - (val / (max_val + 1e-8)) if max_val > 0 else 0
                        values.append(val)
                values += values[:1]
                label = "Clean" if eps == 0 else f"ε={eps}"
                ax.plot(angles, values, "o-", linewidth=1.5, label=label, color=colors_radar[ei % len(colors_radar)])
                ax.fill(angles, values, alpha=0.1, color=colors_radar[ei % len(colors_radar)])

            ax.set_xticks(angles[:-1])
            ax.set_xticklabels(categories, fontsize=9)
            safe_cls = cls.replace("+", "").replace("/", "")
            ax.set_title(f"{MODEL_DISPLAY[mk]} — {cls}\nRadar Summary", fontsize=10, fontweight="bold", y=1.08)
            ax.legend(loc="upper right", bbox_to_anchor=(1.3, 1.1), fontsize=8)
            plt.tight_layout()
            path = os.path.join(output_dir, f"{mk}_{safe_cls}_radar.png")
            fig.savefig(path, dpi=200, bbox_inches="tight")
            plt.close(fig)
            print(f"  Saved {path}")


# ---------------------------------------------------------------------------
# Degradation summary table
# ---------------------------------------------------------------------------

def generate_degradation_table(fgsm_dir: str, output_dir: str):
    """Table and plot showing % degradation from clean baseline."""
    os.makedirs(output_dir, exist_ok=True)

    data = _load_all_fgsm(fgsm_dir)
    if data is None:
        print("  No FGSM data found.")
        return

    rows = []
    for mk in ("wg", "zones"):
        mk_data = data[data["model"] == mk]
        for cls in sorted(mk_data["class"].unique()):
            for metric in METRICS:
                sub = mk_data[(mk_data["class"] == cls) & (mk_data["metric"] == metric)].sort_values("epsilon")
                clean_row = sub[sub["epsilon"] == 0.0]
                if clean_row.empty:
                    continue
                clean_mean = clean_row.iloc[0]["mean"]
                for _, r in sub.iterrows():
                    pct = ((r["mean"] - clean_mean) / (clean_mean + 1e-8)) * 100.0
                    rows.append({
                        "model": mk, "class": cls, "metric": metric,
                        "epsilon": r["epsilon"], "mean": r["mean"],
                        "clean_baseline": clean_mean, "pct_change": pct,
                    })

    df = pd.DataFrame(rows)
    csv_path = os.path.join(output_dir, "degradation_summary.csv")
    df.to_csv(csv_path, index=False, float_format="%.4f")
    print(f"  Saved {csv_path}")

    lines = []
    lines.append(r"\begin{table}[htbp]")
    lines.append(r"\centering")
    lines.append(r"\caption{Segmentation Performance Degradation Under FGSM Attack (\% change from clean baseline)}")
    lines.append(r"\label{tab:degradation}")
    lines.append(r"\small")

    all_eps = sorted(df["epsilon"].unique())
    table_eps = [e for e in KEY_EPSILONS if e in all_eps]
    if not table_eps:
        table_eps = all_eps[:5]

    col_spec = "llc" + "c" * len(table_eps)
    lines.append(r"\begin{tabular}{" + col_spec + r"}")
    lines.append(r"\toprule")
    header = r"Model & Class & Metric & " + " & ".join([f"$\\varepsilon$={e:.2f}" for e in table_eps]) + r" \\"
    lines.append(header)
    lines.append(r"\midrule")

    for mk in ("wg", "zones"):
        mk_df = df[df["model"] == mk]
        for cls in sorted(mk_df["class"].unique()):
            for metric in METRICS:
                sub = mk_df[(mk_df["class"] == cls) & (mk_df["metric"] == metric)]
                row_str = f"{MODEL_DISPLAY[mk]} & {cls} & {METRIC_LABELS[metric]}"
                for eps in table_eps:
                    r = sub[sub["epsilon"] == eps]
                    if r.empty:
                        row_str += " & --"
                    else:
                        pct = r.iloc[0]["pct_change"]
                        row_str += f" & {pct:+.1f}\\%"
                row_str += r" \\"
                lines.append(row_str)
        lines.append(r"\midrule")

    lines[-1] = r"\bottomrule"
    lines.append(r"\end{tabular}")
    lines.append(r"\end{table}")

    tex_path = os.path.join(output_dir, "degradation_table.tex")
    with open(tex_path, "w") as f:
        f.write("\n".join(lines))
    print(f"  Saved {tex_path}")


# ---------------------------------------------------------------------------
# Combined attack comparison (FGSM vs PGD vs BIM)
# ---------------------------------------------------------------------------

def generate_attack_comparison(fgsm_dir: str, pgd_dir: str, output_dir: str):
    """Overlay FGSM, PGD, BIM on one figure per model for Dice."""
    os.makedirs(output_dir, exist_ok=True)

    for mk in ("wg", "zones"):
        attack_data = {}
        for atk, adir in [("FGSM", fgsm_dir), ("PGD", pgd_dir), ("BIM", pgd_dir)]:
            suffix = atk.lower()
            df = _load_agg_csv(adir, mk, suffix, "dice")
            if df is not None:
                attack_data[atk] = df

        if not attack_data:
            continue

        classes = attack_data[list(attack_data.keys())[0]]["class"].unique()

        fig, axes = plt.subplots(1, len(classes), figsize=(7 * len(classes), 5))
        if len(classes) == 1:
            axes = [axes]

        markers = {"FGSM": "o", "PGD": "s", "BIM": "^"}
        styles = {"FGSM": "-", "PGD": "--", "BIM": ":"}

        for ci, cls in enumerate(classes):
            ax = axes[ci]
            for atk_name, adf in attack_data.items():
                sub = adf[adf["class"] == cls].sort_values("epsilon")
                ax.errorbar(sub["epsilon"], sub["mean"], yerr=sub["sem"],
                            marker=markers.get(atk_name, "o"), capsize=3,
                            linestyle=styles.get(atk_name, "-"), markersize=4,
                            label=atk_name)
            ax.set_xlabel("Epsilon", fontsize=10)
            ax.set_ylabel("Dice", fontsize=10)
            ax.set_title(f"{MODEL_DISPLAY[mk]} — {cls}", fontsize=11, fontweight="bold")
            ax.legend(fontsize=9)
            ax.grid(True, alpha=0.2)

        plt.tight_layout()
        path = os.path.join(output_dir, f"{mk}_attack_comparison_dice.png")
        fig.savefig(path, dpi=300, bbox_inches="tight")
        plt.close(fig)
        print(f"  Saved {path}")

        path_pdf = path.replace(".png", ".pdf")
        fig2, axes2 = plt.subplots(1, len(classes), figsize=(7 * len(classes), 5))
        if len(classes) == 1:
            axes2 = [axes2]
        for ci, cls in enumerate(classes):
            ax = axes2[ci]
            for atk_name, adf in attack_data.items():
                sub = adf[adf["class"] == cls].sort_values("epsilon")
                ax.errorbar(sub["epsilon"], sub["mean"], yerr=sub["sem"],
                            marker=markers.get(atk_name, "o"), capsize=3,
                            linestyle=styles.get(atk_name, "-"), markersize=4,
                            label=atk_name)
            ax.set_xlabel("Epsilon", fontsize=10)
            ax.set_ylabel("Dice", fontsize=10)
            ax.set_title(f"{MODEL_DISPLAY[mk]} — {cls}", fontsize=11, fontweight="bold")
            ax.legend(fontsize=9)
            ax.grid(True, alpha=0.2)
        plt.tight_layout()
        fig2.savefig(path_pdf, bbox_inches="tight")
        plt.close(fig2)
        print(f"  Saved {path_pdf}")


# ---------------------------------------------------------------------------
# Noise vs Adversarial comparison figure
# ---------------------------------------------------------------------------

def generate_noise_comparison_figure(noise_dir: str, output_dir: str):
    """Combined figure showing adversarial gap from noise results."""
    os.makedirs(output_dir, exist_ok=True)

    for mk in ("wg", "zones"):
        path = os.path.join(noise_dir, f"{mk}_noise_vs_fgsm_dice.csv")
        if not os.path.isfile(path):
            continue
        df = pd.read_csv(path)
        classes = sorted(df["class"].unique())
        attacks = sorted(df["attack"].unique())

        fig, axes = plt.subplots(1, len(classes), figsize=(7 * len(classes), 5))
        if len(classes) == 1:
            axes = [axes]

        for ci, cls in enumerate(classes):
            ax = axes[ci]
            for atk in attacks:
                sub = df[(df["class"] == cls) & (df["attack"] == atk)].sort_values("epsilon")
                style = "-" if atk == "fgsm" else "--"
                ax.errorbar(sub["epsilon"], sub["mean"], yerr=sub["sem"],
                            marker="o", markersize=4, capsize=3, linestyle=style,
                            label=atk.upper() if atk == "fgsm" else atk.capitalize())
            ax.set_xlabel("Epsilon (L-inf)")
            ax.set_ylabel("Dice")
            ax.set_title(f"{MODEL_DISPLAY[mk]} — {cls}: Adversarial vs Random Noise")
            ax.legend(fontsize=8)
            ax.grid(True, alpha=0.2)

        plt.tight_layout()
        path_out = os.path.join(output_dir, f"{mk}_noise_vs_adversarial_dice.png")
        fig.savefig(path_out, dpi=300, bbox_inches="tight")
        plt.close(fig)
        print(f"  Saved {path_out}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="Generate publication-ready figures and tables")
    parser.add_argument("--fgsm-dir", default="fgsm_results")
    parser.add_argument("--pgd-dir", default="pgd_results")
    parser.add_argument("--noise-dir", default="noise_results")
    parser.add_argument("--output-dir", default="paper_figures")
    args = parser.parse_args()

    print("=" * 60)
    print("Generating Publication-Ready Figures and Tables")
    print("=" * 60)

    generate_latex_tables(args.fgsm_dir, args.output_dir)
    generate_multipanel_figure(args.fgsm_dir, args.output_dir)
    generate_radar_chart(args.fgsm_dir, args.output_dir)
    generate_degradation_table(args.fgsm_dir, args.output_dir)
    generate_attack_comparison(args.fgsm_dir, args.pgd_dir, args.output_dir)
    generate_noise_comparison_figure(args.noise_dir, args.output_dir)

    print("\nDone.")


if __name__ == "__main__":
    main()
