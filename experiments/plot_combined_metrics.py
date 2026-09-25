"""Combined clean-vs-PGD-AT line plots (Dice/HD95/ASD) across architectures, per dataset.

Reads the fold_all summary CSVs written by adv_rob_eval_nnunet_nnunetrecenc
(``{dataset}_{trainer}[_advft]_adv_rob.csv``) and, for each dataset (WG, Zones),
renders one PNG with a 3x3 grid: rows = attack (FGSM/PGD/APGD), columns = metric
(Dice/HD95/ASD). Each panel overlays all four architectures (UNet, ResEnc-M/L/XL),
dashed = before adversarial training (Clean), solid = after (PGD-AT).

Usage::

    python experiments/plot_combined_metrics.py
    python experiments/plot_combined_metrics.py --results-dir <dir> --out-dir <dir>
"""

from __future__ import annotations

import argparse
import csv
import os

import matplotlib
from matplotlib.lines import Line2D

matplotlib.use("Agg")
import matplotlib.pyplot as plt

matplotlib.rcParams.update(
    {
        "font.family": "serif",
        "font.serif": ["Times New Roman", "DejaVu Serif"],
        "font.size": 11,
        "axes.titlesize": 11,
        "axes.labelsize": 10,
        "figure.dpi": 150,
        "savefig.bbox": "tight",
    }
)

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DEFAULT_RESULTS_DIR = os.path.join(REPO_ROOT, "results", "adv_rob_eval_results")

DATASET_LABELS = {"wg": "WG", "zones": "Zones"}
ARCH_LABELS = {
    "unet": "UNet",
    "resenc_m": "ResEnc-M",
    "resenc_l": "ResEnc-L",
    "resenc_xl": "ResEnc-XL",
}
ATTACKS = ["FGSM", "PGD", "APGD"]
METRICS = ["dice", "hd95", "asd"]
METRIC_LABELS = {"dice": "Dice", "hd95": "HD95 (mm)", "asd": "ASD (mm)"}
CLEAN_LS, ADVFT_LS = "--", "-"
MAX_EPS = 16 / 255 + 1e-6


def load_rows(path: str) -> list[dict]:
    with open(path, newline="") as f:
        return list(csv.DictReader(f))


def series(
    rows: list[dict], attack: str, metric: str
) -> tuple[list[float], list[float]]:
    """Return (epsilons, means) for one attack/metric, sorted by epsilon."""
    pts = sorted(
        (
            (float(r["epsilon"]), float(r[f"{metric}_mean"]))
            for r in rows
            if r["attack"] == attack and float(r["epsilon"]) <= MAX_EPS
        ),
        key=lambda p: p[0],
    )
    return [p[0] for p in pts], [p[1] for p in pts]


def load_dataset(ds: str, results_dir: str) -> dict[str, list[dict]]:
    loaded = {}
    for arch in ARCH_LABELS:
        for suffix in ("", "_advft"):
            path = os.path.join(results_dir, "data", f"{ds}_{arch}{suffix}_adv_rob.csv")
            if not os.path.isfile(path):
                print(f"  [SKIP] missing {path}")
                continue
            loaded[f"{arch}{suffix}"] = load_rows(path)
    return loaded


def plot_dataset(ds: str, results_dir: str, out_path: str) -> None:
    cmap = plt.get_cmap("tab10")
    loaded = load_dataset(ds, results_dir)

    fig, axes = plt.subplots(
        len(ATTACKS), len(METRICS), figsize=(13, 11), squeeze=False
    )
    fig.suptitle(
        f"{DATASET_LABELS[ds]} — Clean vs. PGD-AT across architectures",
        fontsize=15,
        fontweight="bold",
        y=0.995,
    )
    fig.text(
        0.5,
        0.965,
        "dashed = Clean   solid = PGD-AT",
        ha="center",
        fontsize=10,
        style="italic",
    )

    for row_i, attack in enumerate(ATTACKS):
        for col_i, metric in enumerate(METRICS):
            ax = axes[row_i][col_i]
            epss_ref: list[float] = []
            for arch_i, arch in enumerate(ARCH_LABELS):
                color = cmap(arch_i % 10)
                for suffix, ls in (("", CLEAN_LS), ("_advft", ADVFT_LS)):
                    key = f"{arch}{suffix}"
                    if key not in loaded:
                        continue
                    epss, means = series(loaded[key], attack, metric)
                    if not epss:
                        continue
                    epss_ref = epss
                    ax.plot(
                        epss,
                        means,
                        ls=ls,
                        color=color,
                        lw=1.8,
                        marker="o",
                        markersize=3.5,
                        markerfacecolor="white",
                        markeredgewidth=1.0,
                    )
            ax.set_title(f"{attack} — {METRIC_LABELS[metric]}")
            ax.set_xlabel("ε (·/255, L∞)")
            ax.set_ylabel(METRIC_LABELS[metric])
            if epss_ref:
                ax.set_xticks(epss_ref)
                ax.set_xticklabels([str(round(e * 255)) for e in epss_ref])
            ax.grid(True, alpha=0.3)

    legend_handles = [
        Line2D([0], [0], color=cmap(i % 10), lw=2.5, label=ARCH_LABELS[arch])
        for i, arch in enumerate(ARCH_LABELS)
    ]
    fig.legend(
        handles=legend_handles,
        loc="lower center",
        ncol=4,
        fontsize=10,
        bbox_to_anchor=(0.5, -0.01),
        frameon=True,
    )
    fig.tight_layout(rect=(0, 0.04, 1, 0.96))
    fig.savefig(out_path)
    plt.close(fig)
    print(f"Combined metrics plot written: {out_path}")


def main() -> None:
    ap = argparse.ArgumentParser(
        description="Combined Dice/HD95/ASD clean-vs-PGD-AT line plots per dataset"
    )
    ap.add_argument("--results-dir", default=DEFAULT_RESULTS_DIR)
    ap.add_argument("--out-dir", default=None)
    args = ap.parse_args()
    out_dir = args.out_dir or os.path.join(args.results_dir, "figures")
    os.makedirs(out_dir, exist_ok=True)
    for ds in DATASET_LABELS:
        out_path = os.path.join(out_dir, f"{ds}_combined_metrics.png")
        plot_dataset(ds, args.results_dir, out_path)


if __name__ == "__main__":
    main()
