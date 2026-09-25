"""Paired with/without comparison of an attribution campaign against a defended model.

Joins the ``case_summary.csv`` of a baseline (undefended) merged directory with the
same table from a defended run on the exact identity key, so every comparison is a
paired, per-case difference on the same case, attack and budget.  Per condition it
reports the medians of both arms, the median paired delta with a case-resampled
bootstrap CI, and a Wilcoxon signed-rank p, for the metrics that the SSEUA/ESEUA
defense acceptance test names.

Usage:
    python experiments/compare_smoothfield_defense.py --table-family spectral \\
        --baseline-merged <SSEUA>/results/merged \\
        --defended-merged <SSEUA>/results/merged_smoothfield \\
        --output-dir <SSEUA>/results/analysis_smoothfield

The undefended tables are read only.  A metric column missing from either arm is
skipped with a note, never invented.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import wilcoxon

IDENTITY = ["dataset", "case_id", "epsilon_n", "attack", "atlas_type"]
CONDITION = ["dataset", "attack", "epsilon_n", "atlas_type"]

# metric -> what a *decrease* means for the defense (used only in the report text)
METRICS: dict[str, dict[str, str]] = {
    "spectral": {
        "clean_dice": "clean quality (must not fall)",
        "full_dice_damage": "primary: total Dice damage at fixed epsilon",
        "lf_energy_share": "attacker's LF energy share (< 0.526 makes the energy-matched control drawable)",
        "lf_sufficiency_fraction": "keep-only-LF reproduces this fraction of the damage",
        "lf_necessity_fraction": "remove-LF recovers this fraction of the damage",
        "k50_frequency_low_first": "shells (low-first) for half the damage; NaN when never reached",
        "f_k50_frequency_low_first_cpm": "frequency (cyc/mm) at which half the damage is reached",
        "damage_removed_top_10pct": "oracle top-10 % shell removal recovers this fraction",
        "lf_control_permutation_p": "LF vs count-matched random shells (interaction cases only)",
    },
    "edge": {
        "clean_dice": "clean quality (must not fall)",
        "full_dice_damage": "primary: total Dice damage at fixed epsilon",
        "edge_energy_fraction_roi": "edge share of perturbation energy (paired delta = fold-enrichment delta; the WG atlases are model-independent)",
        "damage_removed_top_10pct": "removing the top-10 % edge patches recovers this fraction",
        "damage_kept_top_10pct": "keeping only the top-10 % edge patches reproduces this fraction",
        "control_permutation_p": "edge patches vs matched non-edge controls (interaction cases only)",
    },
}
BOOTSTRAP_DRAWS = 2000


def load_summary(merged_dir: Path) -> pd.DataFrame:
    frame = pd.read_csv(merged_dir / "case_summary.csv")
    missing = [c for c in IDENTITY if c not in frame.columns]
    if missing:
        raise ValueError(
            f"{merged_dir}: case_summary.csv lacks identity columns {missing}"
        )
    return frame


def _with_band_fractions(summary: pd.DataFrame) -> pd.DataFrame:
    """The driver stores the LF band interventions in absolute Dice; report them as
    fractions of the damage so arms with different damage are comparable."""
    out = summary.copy()
    for band in ("sufficiency", "necessity"):
        col = f"lf_{band}_dice"
        if col in out and "full_dice_damage" in out:
            out[f"lf_{band}_fraction"] = out[col] / out["full_dice_damage"]
    return out


def pair_arms(baseline: pd.DataFrame, defended: pd.DataFrame) -> pd.DataFrame:
    """Inner join on the identity key; one row per paired case-condition."""
    merged = baseline.merge(
        defended,
        on=IDENTITY,
        how="inner",
        suffixes=("_base", "_def"),
        validate="one_to_one",
    )
    if merged.empty:
        raise ValueError("no case-conditions shared between the two arms")
    return merged


def bootstrap_median_ci(
    values: np.ndarray, seed: int, draws: int = BOOTSTRAP_DRAWS
) -> tuple[float, float]:
    if len(values) < 2:
        return float("nan"), float("nan")
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, len(values), size=(draws, len(values)))
    medians = np.median(values[idx], axis=1)
    return float(np.quantile(medians, 0.025)), float(np.quantile(medians, 0.975))


def compare_metric(pairs: pd.DataFrame, metric: str, seed: int) -> pd.DataFrame:
    base_col, def_col = f"{metric}_base", f"{metric}_def"
    rows = []
    for cond, group in pairs.groupby(CONDITION, sort=True):
        both = group[[base_col, def_col]].astype(float).dropna()
        delta = (both[def_col] - both[base_col]).to_numpy()
        lo, hi = bootstrap_median_ci(delta, seed)
        if len(delta) >= 2 and np.any(delta != 0):
            p = float(wilcoxon(delta, zero_method="wilcox").pvalue)
        else:
            p = float("nan")
        rows.append(
            {
                **dict(zip(CONDITION, cond)),
                "metric": metric,
                "n_paired": int(len(group)),
                "n_defined": int(len(both)),
                "baseline_median": float(both[base_col].median())
                if len(both)
                else float("nan"),
                "defended_median": float(both[def_col].median())
                if len(both)
                else float("nan"),
                "delta_median": float(np.median(delta)) if len(delta) else float("nan"),
                "delta_ci_lo": lo,
                "delta_ci_hi": hi,
                "wilcoxon_p": p,
            }
        )
    return pd.DataFrame(rows)


def compare(
    baseline: pd.DataFrame,
    defended: pd.DataFrame,
    table_family: str,
    seed: int = 20260916,
) -> tuple[pd.DataFrame, list[str]]:
    pairs = pair_arms(_with_band_fractions(baseline), _with_band_fractions(defended))
    tables, notes = [], []
    for metric in METRICS[table_family]:
        if f"{metric}_base" not in pairs or f"{metric}_def" not in pairs:
            notes.append(f"metric {metric!r} absent from one arm; skipped")
            continue
        tables.append(compare_metric(pairs, metric, seed))
    if not tables:
        raise ValueError("no comparable metric present in both arms")
    return pd.concat(tables, ignore_index=True), notes


def _fmt(x: float, digits: int = 3) -> str:
    return "—" if pd.isna(x) else f"{x:.{digits}f}"


def render_report(
    table: pd.DataFrame,
    table_family: str,
    notes: list[str],
    n_shared: int,
    *,
    title: str = "Smooth-field defense: paired comparison",
    labels: tuple[str, str] = ("baseline", "defended"),
) -> str:
    base, treat = labels
    lines = [
        f"# {title} ({table_family})",
        "",
        f"Paired case-conditions shared by both arms: {n_shared}. Medians over cases; delta = "
        f"{treat} − {base} on the same case; CI = 95 % case-resampled bootstrap of the median "
        "delta; p = Wilcoxon signed-rank. Rows with n_defined < n_paired have NaN in one arm "
        "(e.g. K50 never reached).",
        "",
    ]
    for metric, meaning in METRICS[table_family].items():
        sub = table[table.metric == metric]
        if sub.empty:
            continue
        lines += [f"## `{metric}` — {meaning}", ""]
        lines += [
            f"| dataset | attack | ε | atlas | n | {base} | {treat} | Δ median | 95 % CI | p |",
            "|---|---|---|---|---|---|---|---|---|---|",
        ]
        for _, r in sub.iterrows():
            lines.append(
                f"| {r.dataset} | {r.attack} | {int(r.epsilon_n)} | {r.atlas_type} | "
                f"{int(r.n_defined)}/{int(r.n_paired)} | {_fmt(r.baseline_median)} | "
                f"{_fmt(r.defended_median)} | {_fmt(r.delta_median)} | "
                f"[{_fmt(r.delta_ci_lo)}, {_fmt(r.delta_ci_hi)}] | {_fmt(r.wilcoxon_p, 4)} |"
            )
        lines.append("")
    if notes:
        lines += ["## Notes", ""] + [f"- {n}" for n in notes] + [""]
    return "\n".join(lines)


def shell_profile_figure(
    baseline_dir: Path,
    defended_dir: Path,
    pairs_key: pd.DataFrame,
    out: Path,
    labels: tuple[str, str] = ("baseline", "defended"),
) -> bool:
    """Median single-shell necessity per shell, both arms, one panel per condition."""
    paths = [d / "shell_metrics.parquet" for d in (baseline_dir, defended_dir)]
    if not all(p.is_file() for p in paths):
        return False
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    frames = []
    for arm, p in zip(("baseline", "defended"), paths):
        f = pd.read_parquet(
            p, columns=IDENTITY + ["patch_id", "f_low_cpm", "necessity_dice"]
        ).merge(pairs_key, on=IDENTITY)
        f["arm"] = arm
        frames.append(f)
    shells = pd.concat(frames)
    conds = sorted(shells.groupby(["attack", "epsilon_n"]).groups)
    fig, axes = plt.subplots(
        1, len(conds), figsize=(4.2 * len(conds), 3.4), squeeze=False
    )
    for ax, (attack, eps) in zip(axes[0], conds):
        sub = shells[(shells.attack == attack) & (shells.epsilon_n == eps)]
        for arm, label, style in zip(("baseline", "defended"), labels, ("-", "--")):
            prof = (
                sub[sub.arm == arm]
                .groupby("patch_id")
                .agg(f=("f_low_cpm", "first"), nec=("necessity_dice", "median"))
                .sort_index()
            )
            ax.plot(prof.f, prof.nec, style, marker="o", ms=3, label=label)
        ax.set_title(f"{attack} ε{int(eps)}")
        ax.set_xlabel("shell lower edge (cyc/mm)")
        ax.set_ylabel("median single-shell necessity (Dice)")
        ax.legend(frameon=False)
    fig.tight_layout()
    fig.savefig(out, dpi=150)
    plt.close(fig)
    return True


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--baseline-merged", type=Path, required=True)
    parser.add_argument("--defended-merged", type=Path, required=True)
    parser.add_argument("--table-family", choices=list(METRICS), required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--seed", type=int, default=20260916)
    parser.add_argument("--title", default="Smooth-field defense: paired comparison")
    parser.add_argument(
        "--labels",
        nargs=2,
        default=("baseline", "defended"),
        metavar=("BASELINE", "TREATMENT"),
        help="arm names in the report (CSV columns stay baseline_*/defended_*)",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    baseline = load_summary(args.baseline_merged)
    defended = load_summary(args.defended_merged)
    table, notes = compare(baseline, defended, args.table_family, args.seed)
    pairs_key = pair_arms(baseline, defended)[IDENTITY]
    args.output_dir.mkdir(parents=True, exist_ok=True)
    table.to_csv(args.output_dir / "defense_comparison.csv", index=False)
    figure_written = False
    if args.table_family == "spectral":
        figure_written = shell_profile_figure(
            args.baseline_merged,
            args.defended_merged,
            pairs_key,
            args.output_dir / "defense_shell_necessity.png",
            labels=tuple(args.labels),
        )
    report = render_report(
        table, args.table_family, notes, len(pairs_key),
        title=args.title, labels=tuple(args.labels),
    )
    (args.output_dir / "DEFENSE_COMPARISON.md").write_text(report)
    (args.output_dir / "defense_comparison_manifest.json").write_text(
        json.dumps(
            {
                "baseline_merged": str(args.baseline_merged.resolve()),
                "defended_merged": str(args.defended_merged.resolve()),
                "table_family": args.table_family,
                "paired_case_conditions": int(len(pairs_key)),
                "metrics": sorted(table.metric.unique()),
                "figure": figure_written,
                "notes": notes,
            },
            indent=2,
        )
    )
    print(
        f"{len(pairs_key)} paired case-conditions; wrote {args.output_dir}", flush=True
    )


if __name__ == "__main__":
    main()
