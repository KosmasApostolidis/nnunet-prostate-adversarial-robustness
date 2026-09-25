"""Combined clean + worst-case robustness summary across all 16 combos.

Reads the fold_all summary CSVs written by adv_rob_eval_nnunet_nnunetrecenc
(``{dataset}_{trainer}[_advft]_adv_rob.csv``) and writes one wide CSV with, for
each (dataset, architecture, variant) combo: clean Dice/HD95/ASD and
worst-case (ε=16/255) Dice/HD95/ASD for FGSM, PGD and APGD (mean ± SEM).

16 combos = 2 datasets (WG, Zones) x 4 architectures (UNet, ResEnc-M/L/XL) x
2 variants (Clean, PGD-AT fine-tuned).

Usage::

    python experiments/generate_combined_summary_table.py
    python experiments/generate_combined_summary_table.py --results-dir <dir> --out <path>
"""

from __future__ import annotations

import argparse
import csv
import os

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DEFAULT_RESULTS_DIR = os.path.join(REPO_ROOT, "results", "adv_rob_eval_results")
DEFAULT_TEX_OUT = os.path.join(REPO_ROOT, "paper", "table_combined_summary.tex")

DATASET_LABELS = {"wg": "WG", "zones": "Zones"}
ARCH_LABELS = {
    "unet": "UNet",
    "resenc_m": "ResEnc-M",
    "resenc_l": "ResEnc-L",
    "resenc_xl": "ResEnc-XL",
}
VARIANTS = [("Clean", ""), ("PGD-AT", "_advft")]
ATTACKS = ["FGSM", "PGD", "APGD"]
METRICS = ["dice", "hd95", "asd"]
WORST_EPS = 16 / 255


def load_rows(path: str) -> list[dict]:
    with open(path, newline="") as f:
        return list(csv.DictReader(f))


def clean_row(rows: list[dict]) -> dict:
    return next(r for r in rows if float(r["epsilon"]) == 0.0)


def worst_row(rows: list[dict], attack: str) -> dict:
    candidates = [r for r in rows if r["attack"] == attack]
    return min(candidates, key=lambda r: abs(float(r["epsilon"]) - WORST_EPS))


def build_table(results_dir: str) -> list[dict]:
    table = []
    for ds in DATASET_LABELS:
        for arch in ARCH_LABELS:
            for variant, suffix in VARIANTS:
                path = os.path.join(
                    results_dir, "data", f"{ds}_{arch}{suffix}_adv_rob.csv"
                )
                if not os.path.isfile(path):
                    print(f"  [SKIP] missing {path}")
                    continue
                rows = load_rows(path)
                entry: dict[str, str | float] = {
                    "dataset": DATASET_LABELS[ds],
                    "architecture": ARCH_LABELS[arch],
                    "variant": variant,
                }
                c = clean_row(rows)
                for m in METRICS:
                    entry[f"clean_{m}"] = float(c[f"{m}_mean"])
                    entry[f"clean_{m}_sem"] = float(c[f"{m}_sem"])
                for atk in ATTACKS:
                    w = worst_row(rows, atk)
                    for m in METRICS:
                        entry[f"{atk.lower()}_worst_{m}"] = float(w[f"{m}_mean"])
                        entry[f"{atk.lower()}_worst_{m}_sem"] = float(w[f"{m}_sem"])
                table.append(entry)
    return table


def write_csv(table: list[dict], out_path: str) -> None:
    if not table:
        print("  [WARN] no combos found — no table written")
        return
    with open(out_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(table[0].keys()))
        writer.writeheader()
        writer.writerows(table)
    print(f"Combined summary table written: {out_path}  ({len(table)} rows)")


def print_markdown(table: list[dict]) -> None:
    header = [
        "Dataset",
        "Arch",
        "Variant",
        "Clean Dice",
        "FGSM@16 Dice",
        "PGD@16 Dice",
        "APGD@16 Dice",
    ]
    print("| " + " | ".join(header) + " |")
    print("|" + "|".join(["---"] * len(header)) + "|")
    for e in table:
        print(
            "| "
            + " | ".join(
                [
                    e["dataset"],
                    e["architecture"],
                    e["variant"],
                    f"{e['clean_dice']:.3f}",
                    f"{e['fgsm_worst_dice']:.3f}",
                    f"{e['pgd_worst_dice']:.3f}",
                    f"{e['apgd_worst_dice']:.3f}",
                ]
            )
            + " |"
        )


DICE_COLS = ["clean_dice", "fgsm_worst_dice", "pgd_worst_dice", "apgd_worst_dice"]


def _cell(entry: dict, col: str, is_best: bool) -> str:
    text = f"{entry[col]:.3f}\\pm{entry[col + '_sem']:.3f}"
    return f"$\\bm{{{text}}}$" if is_best else f"${text}$"


def write_latex(table: list[dict], out_path: str) -> None:
    if not table:
        print("  [WARN] no combos found — no LaTeX table written")
        return

    lines = [
        "\\begin{table*}[!t]",
        "\\centering",
        "\\caption{Clean ($\\varepsilon=0$) and worst-case ($\\varepsilon=16/255$, "
        "$L_\\infty$) adversarial robustness (Dice, mean\\,$\\pm$\\,SEM across "
        "patients, fold=\\textquotedblleft all\\textquotedblright\\ checkpoints) "
        "for four nnU-Net encoder variants on the whole-gland (WG) and "
        "zonal (Zones) segmentation tasks, comparing standard training (Clean) "
        "against PGD adversarial fine-tuning (PGD-AT). Bold marks the "
        "best-performing configuration in each column within a dataset.}",
        "\\label{tab:combined_summary}",
        "\\setlength{\\tabcolsep}{4pt}",
        "\\footnotesize",
        "\\begin{tabular}{@{}lllcccc@{}}",
        "\\toprule",
        "Dataset & Architecture & Training & Clean & FGSM@16 & PGD@16 & APGD@16 \\\\",
        "\\midrule",
    ]

    for ds_idx, ds in enumerate(DATASET_LABELS.values()):
        block = [e for e in table if e["dataset"] == ds]
        best = {col: max(e[col] for e in block) for col in DICE_COLS}
        for arch_idx, arch in enumerate(ARCH_LABELS.values()):
            arch_rows = [e for e in block if e["architecture"] == arch]
            for row_idx, entry in enumerate(arch_rows):
                cells = [
                    _cell(entry, col, entry[col] == best[col]) for col in DICE_COLS
                ]
                prefix = ""
                if row_idx == 0:
                    prefix += f"\\multirow{{2}}{{*}}{{{arch}}} & "
                else:
                    prefix += " & "
                if arch_idx == 0 and row_idx == 0:
                    prefix = f"\\multirow{{8}}{{*}}{{{ds}}} & " + prefix
                else:
                    prefix = " & " + prefix
                lines.append(
                    prefix + f"{entry['variant']} & " + " & ".join(cells) + " \\\\"
                )
            if arch_idx < len(ARCH_LABELS) - 1:
                lines.append("\\cmidrule(lr){2-7}")
        if ds_idx < len(DATASET_LABELS) - 1:
            lines.append("\\midrule")

    lines += ["\\bottomrule", "\\end{tabular}", "\\end{table*}"]

    with open(out_path, "w") as f:
        f.write("\n".join(lines) + "\n")
    print(f"LaTeX table written: {out_path}")


def main() -> None:
    ap = argparse.ArgumentParser(
        description="Combined 16-combo adversarial robustness summary table"
    )
    ap.add_argument("--results-dir", default=DEFAULT_RESULTS_DIR)
    ap.add_argument("--out", default=None)
    ap.add_argument("--tex-out", default=DEFAULT_TEX_OUT)
    args = ap.parse_args()
    out = args.out or os.path.join(args.results_dir, "combined_summary_table.csv")
    table = build_table(args.results_dir)
    write_csv(table, out)
    print_markdown(table)
    write_latex(table, args.tex_out)


if __name__ == "__main__":
    main()
