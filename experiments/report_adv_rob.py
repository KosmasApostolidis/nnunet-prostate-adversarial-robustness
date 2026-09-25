"""Patient-level adversarial-robustness report for the nnU-Net architecture sweep.

Reads the per-case CSVs written by ``adv_rob_eval_nnunet_nnunetrecenc.save_percase_csv``
(``{dataset}_{trainer}_percase.csv``) and builds one multi-page PDF. For each
attack, in order:

  * one page per architecture (dataset × trainer):
      - a patient-level paired Dice slopegraph (clean ε=0 → worst-case ε, one
        thin line per patient + bold mean),
      - robustness curves (Dice / HD95 / ASD vs ε, each mean ± 95% CI across
        patients);
  * an AGGREGATE summary LAST: every architecture's mean Dice-vs-ε overlaid plus
    a Dice@0 / Dice@worst / Δ table.

Single model per architecture (no clean-vs-PGD-AT pairing — that lives in
``report_clean_vs_advft.py``); each patient's paired line therefore connects its
own clean and worst-case-ε Dice.

Usage::

    python experiments/report_adv_rob.py
    python experiments/report_adv_rob.py --results-dir <dir> --attack APGD
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
DEFAULT_RESULTS_DIR = os.path.join(REPO_ROOT, "results", "adv_rob_eval_results")

SLOPE_EPS = 32 / 255  # worst-case ε the paired slopegraph pairs clean Dice against
ALL_METRICS = ["dice", "hd95", "asd", "iou"]  # columns present in the per-case CSV
CURVE_METRICS = ["dice", "hd95", "asd"]
METRIC_LABELS = {"dice": "Dice", "hd95": "HD95 (mm)", "asd": "ASD (mm)", "iou": "IoU"}
CURVE_C = "#0072B2"

# Dataset / trainer keys and display labels — mirror the harness registry so the
# report can find ``{ds}_{tr}_percase.csv`` and label each architecture page.
DATASET_LABELS = {"wg": "WG", "zones": "Zones"}
TRAINER_LABELS = {
    "unet": "nnU-Net UNet",
    "resenc_m": "nnU-Net ResEnc-M",
    "resenc_l": "nnU-Net ResEnc-L",
    "resenc_xl": "nnU-Net ResEnc-XL",
    "unet_advft": "nnU-Net UNet (PGD-AT)",
    "resenc_m_advft": "nnU-Net ResEnc-M (PGD-AT)",
    "resenc_l_advft": "nnU-Net ResEnc-L (PGD-AT)",
    "resenc_xl_advft": "nnU-Net ResEnc-XL (PGD-AT)",
}
# Preferred column/page order for attacks; any others found are appended sorted.
ATTACK_ORDER = ["FGSM", "PGD", "APGD"]


# ---------------------------------------------------------------------------
# Loading
# ---------------------------------------------------------------------------
def load_percase(path: str) -> dict:
    """Read a ``*_percase.csv`` into data[attack][epsilon][metric] = {case_id: v}.

    Rows from every fold are pooled by ``case_id`` (nnU-Net val folds are
    disjoint, so each patient contributes once), giving one patient population
    per (attack, ε) for the mean/CI curves and the paired slopegraph.
    """
    data: dict = defaultdict(lambda: defaultdict(lambda: defaultdict(dict)))
    with open(path, newline="") as f:
        for row in csv.DictReader(f):
            atk, eps, cid = row["attack"], float(row["epsilon"]), row["case_id"]
            for m in ALL_METRICS:
                if m in row:
                    v = row[m]
                    data[atk][eps][m][cid] = np.nan if v == "nan" else float(v)
    return data


def mean_ci(values) -> tuple[float, float]:
    """Mean and 95% normal CI half-width (1.96·SEM) over finite values."""
    v = np.asarray(list(values), dtype=np.float64)
    v = v[np.isfinite(v)]
    if v.size == 0:
        return float("nan"), float("nan")
    mean = float(v.mean())
    sem = float(v.std(ddof=1) / np.sqrt(v.size)) if v.size > 1 else 0.0
    return mean, 1.96 * sem


def nearest_eps(epsilons: list[float], target: float) -> float:
    return min(epsilons, key=lambda e: abs(e - target))


# ---------------------------------------------------------------------------
# Per-architecture page
# ---------------------------------------------------------------------------
def slopegraph(ax, data: dict, attack: str) -> None:
    """Per-patient Dice slopegraph clean (ε=0) → worst-case ε (one line/patient)."""
    epss = sorted(data[attack])
    e_lo, e_hi = min(epss), nearest_eps(epss, SLOPE_EPS)
    clean, adv = data[attack][e_lo]["dice"], data[attack][e_hi]["dice"]
    common = sorted(set(clean) & set(adv))
    for cid in common:
        c, a = clean[cid], adv[cid]
        if np.isfinite(c) and np.isfinite(a):
            ax.plot([0, 1], [c, a], color="0.6", lw=0.4, alpha=0.4, zorder=1)
    mc = float(np.nanmean([clean[c] for c in common])) if common else float("nan")
    ma = float(np.nanmean([adv[c] for c in common])) if common else float("nan")
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
    ax.set_xticklabels([f"Clean\n{mc:.3f}", f"ε={round(e_hi * 255)}/255\n{ma:.3f}"])
    ax.set_ylabel("Dice")
    ax.set_title(
        f"Patient pairing: clean → ε={round(e_hi * 255)}/255  (n={len(common)})"
    )
    ax.grid(True, axis="y", alpha=0.3)


def curve(ax, data: dict, attack: str, metric: str) -> None:
    """One model's ``metric`` vs ε: mean ± 95% CI across patients."""
    epss = sorted(data[attack])
    means, cis = [], []
    for e in epss:
        mean, ci = mean_ci(data[attack][e][metric].values())
        means.append(mean)
        cis.append(ci)
    means, cis = np.array(means), np.array(cis)
    ax.fill_between(epss, means - cis, means + cis, color=CURVE_C, alpha=0.15, lw=0)
    ax.plot(
        epss,
        means,
        marker="o",
        color=CURVE_C,
        lw=2.0,
        markersize=5,
        markerfacecolor="white",
        markeredgewidth=1.2,
    )
    ax.set_xlabel("ε (·/255, L∞)")
    ax.set_ylabel(METRIC_LABELS[metric])
    ax.set_title(f"{METRIC_LABELS[metric]} vs ε  (mean ± 95% CI)")
    ax.set_xticks(epss)
    ax.set_xticklabels([str(round(e * 255)) for e in epss])
    ax.grid(True, alpha=0.3)


def arch_page(pdf, data: dict, attack: str, ds_label: str, arch_label: str) -> None:
    """One page for a single architecture: paired slopegraph + 3 robustness curves."""
    fig, axes = plt.subplots(2, 2, figsize=(11, 8.5))
    fig.suptitle(
        f"{ds_label} — {arch_label}   (attack: {attack})",
        fontsize=15,
        fontweight="bold",
        y=0.98,
    )
    slopegraph(axes[0][0], data, attack)
    curve(axes[0][1], data, attack, "dice")
    curve(axes[1][0], data, attack, "hd95")
    curve(axes[1][1], data, attack, "asd")
    fig.tight_layout(rect=(0, 0, 1, 0.96))
    pdf.savefig(fig)
    plt.close(fig)


# ---------------------------------------------------------------------------
# Aggregate summary page (LAST, per attack)
# ---------------------------------------------------------------------------
def aggregate_page(pdf, sections: list, attack: str) -> None:
    """sections: list of (ds_label, arch_label, data). Mean Dice-vs-ε overlay + table."""
    fig = plt.figure(figsize=(11, 8.5))
    fig.suptitle(
        f"Aggregate summary — mean Dice vs ε  ({attack}, all architectures)",
        fontsize=15,
        fontweight="bold",
        y=0.98,
    )
    ax = fig.add_axes((0.08, 0.42, 0.88, 0.48))
    cmap = plt.get_cmap("tab10")
    ref_epss = sorted(sections[0][2][attack])
    table_rows = []
    for i, (dsl, al, data) in enumerate(sections):
        epss = sorted(data[attack])
        color = cmap(i % 10)
        means = [mean_ci(data[attack][e]["dice"].values())[0] for e in epss]
        ax.plot(
            epss,
            means,
            ls="-",
            color=color,
            lw=2.0,
            marker="o",
            markersize=4,
            label=f"{dsl}/{al}",
        )
        e_lo, e_hi = min(epss), nearest_eps(epss, SLOPE_EPS)
        d0 = mean_ci(data[attack][e_lo]["dice"].values())[0]
        dhi = mean_ci(data[attack][e_hi]["dice"].values())[0]
        table_rows.append(
            [f"{dsl}/{al}", f"{d0:.3f}", f"{dhi:.3f}", f"{dhi - d0:+.3f}"]
        )
    ax.set_xlabel("ε (·/255, L∞)")
    ax.set_ylabel("mean Dice")
    ax.set_xticks(ref_epss)
    ax.set_xticklabels([str(round(e * 255)) for e in ref_epss])
    ax.grid(True, alpha=0.3)
    ax.legend(fontsize=8, ncol=2, loc="lower left")

    tax = fig.add_axes((0.08, 0.05, 0.88, 0.28))
    tax.axis("off")
    tbl = tax.table(
        cellText=table_rows,
        colLabels=[
            "Architecture",
            "Dice@0",
            f"Dice@{round(SLOPE_EPS * 255)}/255",
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
def build_report(
    results_dir: str, out_path: str, attack_override: str | None = None
) -> int:
    """Build the multi-page PDF. Returns the number of pages written.

    Discovers every ``{ds}_{tr}_percase.csv`` present in ``results_dir`` (for the
    known dataset/trainer keys), then emits, per attack, one page per architecture
    followed by the aggregate summary.
    """
    loaded: dict = {}
    for ds in DATASET_LABELS:
        for tr in TRAINER_LABELS:
            path = os.path.join(results_dir, "data", f"{ds}_{tr}_percase.csv")
            if os.path.isfile(path):
                loaded[(ds, tr)] = load_percase(path)
    if not loaded:
        print(f"  [WARN] no *_percase.csv in {results_dir} — no report produced")
        return 0

    present: set = set()
    for data in loaded.values():
        present.update(data)
    if attack_override is not None:
        if attack_override not in present:
            print(f"  [WARN] attack {attack_override!r} not found in any CSV")
            return 0
        attacks = [attack_override]
    else:
        attacks = [a for a in ATTACK_ORDER if a in present]
        attacks += sorted(present - set(attacks))

    n_pages = 0
    with PdfPages(out_path) as pdf:
        for atk in attacks:
            sections = []
            for (ds, tr), data in loaded.items():
                if atk not in data:
                    continue
                dsl, al = DATASET_LABELS[ds], TRAINER_LABELS[tr]
                arch_page(pdf, data, atk, dsl, al)
                sections.append((dsl, al, data))
                n_pages += 1
            if sections:
                aggregate_page(pdf, sections, atk)
                n_pages += 1
            print(f"  → {atk}: {len(sections)} architecture pages + aggregate")
    print(f"Report written: {out_path}  ({n_pages} pages)")
    return n_pages


def main() -> None:
    ap = argparse.ArgumentParser(
        description="Patient-level adversarial-robustness PDF report (nnU-Net sweep)"
    )
    ap.add_argument("--results-dir", default=DEFAULT_RESULTS_DIR)
    ap.add_argument("--out", default=None)
    ap.add_argument(
        "--attack",
        default=None,
        help="restrict the report to one attack (default: all attacks present)",
    )
    args = ap.parse_args()
    out = args.out or os.path.join(args.results_dir, "patient_report.pdf")
    build_report(args.results_dir, out, attack_override=args.attack)


if __name__ == "__main__":
    main()
