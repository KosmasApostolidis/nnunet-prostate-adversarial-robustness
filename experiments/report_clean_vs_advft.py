"""Patient-level clean-vs-PGD-AT robustness report → one multi-page PDF.

Reads the per-case CSVs written by adv_rob_eval_nnunet_nnunetrecenc.save_percase_csv
(``{dataset}_{arch}[_advft]_percase.csv``) and builds, in order:

  * one section per architecture (WG: UNet/M/L/XL, then Zones: UNet/M/L):
      - a patient-level slopegraph (Dice, clean→PGD-AT at ε=8/255, one thin line
        per patient + bold mean),
      - robustness curves (Dice / HD95 / ASD vs ε, clean vs PGD-AT, each with
        mean ± 95% CI across patients);
  * an AGGREGATE summary LAST: all architectures' mean Dice-vs-ε (clean vs
    PGD-AT) plus a Dice@ε=8/255 table (clean / advft / Δ).

Usage::

    python experiments/report_clean_vs_advft.py
    python experiments/report_clean_vs_advft.py --attack AutoAttack --results-dir <dir>
"""

from __future__ import annotations

import argparse
import csv
import os
from collections import defaultdict

import matplotlib
import numpy as np

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages

matplotlib.rcParams.update(
    {
        "font.family": "serif",
        "font.serif": ["Times New Roman", "DejaVu Serif"],
        "font.size": 11,
        "axes.titlesize": 12,
        "axes.labelsize": 11,
        "figure.dpi": 120,
        "savefig.bbox": "tight",
    }
)

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DEFAULT_RESULTS_DIR = os.path.join(REPO_ROOT, "results", "adv_rob_eval_results", "clean_vs_advft")

SLOPE_EPS = 8 / 255  # ε for the patient-level paired slopegraph
CURVE_METRICS = ["dice", "hd95", "asd"]
METRIC_LABELS = {"dice": "Dice", "hd95": "HD95 (mm)", "asd": "ASD (mm)"}
ALL_METRICS = ["dice", "hd95", "asd", "iou"]  # columns present in the per-case CSV
ATTACK_PREFERENCE = ["AutoAttack", "APGD-CE", "PGD", "FGSM"]

CLEAN_C, ADVFT_C = "#0072B2", "#D55E00"

# (dataset_key, arch_key, dataset_label, arch_label) — per-arch first, in this order
COMPARISONS = [
    ("wg", "unet", "WG", "UNet"),
    ("wg", "resenc_m", "WG", "ResEnc-M"),
    ("wg", "resenc_l", "WG", "ResEnc-L"),
    ("wg", "resenc_xl", "WG", "ResEnc-XL"),
    ("zones", "unet", "Zones", "UNet"),
    ("zones", "resenc_m", "Zones", "ResEnc-M"),
    ("zones", "resenc_l", "Zones", "ResEnc-L"),
]


# ---------------------------------------------------------------------------
# Loading
# ---------------------------------------------------------------------------
def load_percase(path):
    """Return data[attack][epsilon][metric] = {case_id: value}."""
    data = defaultdict(lambda: defaultdict(lambda: defaultdict(dict)))
    with open(path, newline="") as f:
        for row in csv.DictReader(f):
            atk = row["attack"]
            eps = float(row["epsilon"])
            cid = row["case_id"]
            for m in ALL_METRICS:
                if m in row:
                    v = row[m]
                    data[atk][eps][m][cid] = np.nan if v == "nan" else float(v)
    return data


def pick_attack(dc, da):
    present = set(dc) & set(da)
    for pref in ATTACK_PREFERENCE:
        if pref in present:
            return pref
    return sorted(present)[0] if present else None


def nearest_eps(epsilons, target):
    return min(epsilons, key=lambda e: abs(e - target))


def mean_ci(values):
    """Mean and 95% normal CI half-width (1.96·SEM) over finite values."""
    v = np.asarray(values, dtype=np.float64)
    v = v[np.isfinite(v)]
    if v.size == 0:
        return np.nan, np.nan
    mean = float(v.mean())
    sem = float(v.std(ddof=1) / np.sqrt(v.size)) if v.size > 1 else 0.0
    return mean, 1.96 * sem


# ---------------------------------------------------------------------------
# Per-architecture page
# ---------------------------------------------------------------------------
def _slopegraph(ax, dc, da, attack):
    """Dice slopegraph clean→PGD-AT at ε≈SLOPE_EPS, one line per shared patient."""
    epss = sorted(set(dc[attack]) & set(da[attack]))
    e = nearest_eps(epss, SLOPE_EPS)
    clean = dc[attack][e]["dice"]
    advft = da[attack][e]["dice"]
    common = sorted(set(clean) & set(advft))
    for cid in common:
        c, a = clean[cid], advft[cid]
        if np.isfinite(c) and np.isfinite(a):
            ax.plot([0, 1], [c, a], color="0.6", lw=0.4, alpha=0.4, zorder=1)
    mc = np.nanmean([clean[c] for c in common])
    ma = np.nanmean([advft[c] for c in common])
    ax.plot(
        [0, 1],
        [mc, ma],
        color="black",
        lw=2.6,
        marker="o",
        markersize=7,
        markerfacecolor="white",
        markeredgewidth=1.6,
        zorder=5,
        label="mean",
    )
    ax.set_xlim(-0.25, 1.25)
    ax.set_xticks([0, 1])
    ax.set_xticklabels([f"Clean\n{mc:.3f}", f"PGD-AT\n{ma:.3f}"])
    ax.set_ylabel("Dice")
    ax.set_title(f"Patient pairing @ ε={round(e * 255)}/255  (n={len(common)})")
    ax.grid(True, axis="y", alpha=0.3)


def _curve(ax, dc, da, attack, metric):
    epss = sorted(set(dc[attack]) & set(da[attack]))
    for data, color, label in ((dc, CLEAN_C, "Clean"), (da, ADVFT_C, "PGD-AT")):
        means, cis = [], []
        for e in epss:
            mean, ci = mean_ci(list(data[attack][e][metric].values()))
            means.append(mean)
            cis.append(ci)
        means, cis = np.array(means), np.array(cis)
        ax.fill_between(epss, means - cis, means + cis, color=color, alpha=0.15, lw=0)
        ax.plot(
            epss,
            means,
            marker="o",
            color=color,
            lw=2.0,
            markersize=5,
            markerfacecolor="white",
            markeredgewidth=1.2,
            label=label,
        )
    ax.set_xlabel("ε (·/255, L∞)")
    ax.set_ylabel(METRIC_LABELS[metric])
    ax.set_title(f"{METRIC_LABELS[metric]} vs ε  (mean ± 95% CI)")
    ax.set_xticks(epss)
    ax.set_xticklabels([str(round(e * 255)) for e in epss])
    ax.grid(True, alpha=0.3)
    if metric == "dice":
        ax.legend(loc="best", fontsize=10)


def arch_page(pdf, dc, da, attack, ds_label, arch_label):
    fig, axes = plt.subplots(2, 2, figsize=(11, 8.5))
    fig.suptitle(
        f"{ds_label} — {arch_label}   (attack: {attack})",
        fontsize=15,
        fontweight="bold",
        y=0.98,
    )
    _slopegraph(axes[0][0], dc, da, attack)
    _curve(axes[0][1], dc, da, attack, "dice")
    _curve(axes[1][0], dc, da, attack, "hd95")
    _curve(axes[1][1], dc, da, attack, "asd")
    fig.tight_layout(rect=(0, 0, 1, 0.96))
    pdf.savefig(fig)
    plt.close(fig)


# ---------------------------------------------------------------------------
# Aggregate summary page (LAST)
# ---------------------------------------------------------------------------
def aggregate_page(pdf, sections):
    """sections: list of (ds_label, arch_label, attack, dc, da)."""
    fig = plt.figure(figsize=(11, 8.5))
    fig.suptitle(
        "Aggregate summary — mean Dice vs ε (all architectures)",
        fontsize=15,
        fontweight="bold",
        y=0.98,
    )
    ax = fig.add_axes((0.08, 0.42, 0.88, 0.48))
    cmap = plt.get_cmap("tab10")
    table_rows = []
    ref_epss = sorted(
        set(sections[0][3][sections[0][2]]) & set(sections[0][4][sections[0][2]])
    )
    for i, (dsl, al, attack, dc, da) in enumerate(sections):
        epss = sorted(set(dc[attack]) & set(da[attack]))
        color = cmap(i % 10)
        cm = [mean_ci(list(dc[attack][e]["dice"].values()))[0] for e in epss]
        am = [mean_ci(list(da[attack][e]["dice"].values()))[0] for e in epss]
        ax.plot(epss, cm, ls="--", color=color, lw=1.4, alpha=0.8)
        ax.plot(
            epss,
            am,
            ls="-",
            color=color,
            lw=2.0,
            marker="o",
            markersize=4,
            label=f"{dsl}/{al}",
        )
        e8 = nearest_eps(epss, SLOPE_EPS)
        c8 = mean_ci(list(dc[attack][e8]["dice"].values()))[0]
        a8 = mean_ci(list(da[attack][e8]["dice"].values()))[0]
        table_rows.append([f"{dsl}/{al}", f"{c8:.3f}", f"{a8:.3f}", f"{a8 - c8:+.3f}"])
    ax.set_xlabel("ε (·/255, L∞)")
    ax.set_ylabel("mean Dice")
    ax.set_xticks(ref_epss)
    ax.set_xticklabels([str(round(e * 255)) for e in ref_epss])
    ax.grid(True, alpha=0.3)
    ax.legend(fontsize=8, ncol=2, loc="lower left")
    ax.text(
        0.99,
        0.98,
        "solid = PGD-AT   dashed = Clean",
        transform=ax.transAxes,
        ha="right",
        va="top",
        fontsize=9,
        style="italic",
    )

    tax = fig.add_axes((0.08, 0.05, 0.88, 0.28))
    tax.axis("off")
    tbl = tax.table(
        cellText=table_rows,
        colLabels=[
            "Architecture",
            f"Clean Dice@{round(SLOPE_EPS * 255)}/255",
            "PGD-AT",
            "Δ",
        ],
        cellLoc="center",
        loc="center",
    )
    tbl.auto_set_font_size(False)
    tbl.set_fontsize(9)
    tbl.scale(1, 1.4)
    pdf.savefig(fig)
    plt.close(fig)


# ---------------------------------------------------------------------------
# Driver
# ---------------------------------------------------------------------------
def build_report(results_dir, out_path, attack_override=None):
    sections = []
    with PdfPages(out_path) as pdf:
        for dk, ak, dsl, al in COMPARISONS:
            clean_csv = os.path.join(results_dir, f"{dk}_{ak}_percase.csv")
            advft_csv = os.path.join(results_dir, f"{dk}_{ak}_advft_percase.csv")
            if not (os.path.isfile(clean_csv) and os.path.isfile(advft_csv)):
                print(f"  [SKIP] {dk}/{ak}: per-case CSV(s) missing")
                continue
            dc, da = load_percase(clean_csv), load_percase(advft_csv)
            attack = attack_override or pick_attack(dc, da)
            if attack is None or attack not in dc or attack not in da:
                print(f"  [SKIP] {dk}/{ak}: attack {attack!r} not in both CSVs")
                continue
            arch_page(pdf, dc, da, attack, dsl, al)
            sections.append((dsl, al, attack, dc, da))
            print(f"  → page: {dsl}/{al} (attack={attack})")

        if not sections:
            print("  [WARN] no architectures rendered — no report produced")
            return 0
        aggregate_page(pdf, sections)
        print("  → page: aggregate summary")
    print(f"Report written: {out_path}  ({len(sections)} architectures + summary)")
    return len(sections)


def main():
    ap = argparse.ArgumentParser(description="Patient-level clean-vs-PGD-AT PDF report")
    ap.add_argument("--results-dir", default=DEFAULT_RESULTS_DIR)
    ap.add_argument("--out", default=None)
    ap.add_argument(
        "--attack", default=None, help="force a specific attack (default: auto)"
    )
    args = ap.parse_args()
    out = args.out or os.path.join(args.results_dir, "patient_report.pdf")
    build_report(args.results_dir, out, attack_override=args.attack)


if __name__ == "__main__":
    main()
