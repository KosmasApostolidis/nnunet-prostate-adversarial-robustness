"""Cohort tables, figures and report for the spectral-shell attribution tables.

Reads a merged (or pilot) ``all_cases_spectral_attribution`` directory, reuses
the edge-attribution cohort aggregators where the schema matches, adds the
spectral figures, and evaluates the pre-registered rule from the spec by code.

Spec: docs/superpowers/specs/2026-09-14-spectral-shell-attribution-design.md
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import matplotlib
import numpy as np
import pandas as pd

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

REPO_ROOT = Path(__file__).resolve().parents[1]
for _root in (REPO_ROOT, REPO_ROOT / "src"):
    if str(_root) not in sys.path:
        sys.path.insert(0, str(_root))

from experiments.evaluate_spectral_attribution_all_cases import (  # noqa: E402
    CONTROL_CSV,
    CURVES_CSV,
    DEFAULT_OUTPUT_DIR,
    SHELL_CSV,
    SUMMARY_CSV,
)
from mri_prostate_seg.experiments.edge_attribution.aggregate import (  # noqa: E402
    CONDITION,
    anatomical_category_summary,
    check_summary_schema,
    cohort_boundary_recovery,
    cohort_concentration,
    cohort_controls,
    cohort_curves,
    median_ci,
)
from mri_prostate_seg.experiments.edge_attribution.spectral import (  # noqa: E402
    N_SHELLS,
    reference_shell_edges,
)
from mri_prostate_seg.experiments.edge_attribution.spectral_analysis import (  # noqa: E402
    RANKINGS,
)

CONFIRMED, REFUTED, PARTIAL = "CONFIRMED", "REFUTED", "PARTIAL"
RULE_EPS = (16, 32)
RULE_ATTACK = "PGD-BCE"
RULE_MIN_CONDITIONS = 3
LF_SHELL_MAX = 8
HF_SHELL_MIN = 17
SUFFICIENCY_MIN = 0.50
P_MAX = 0.05
_NAN = float("nan")


def _read(path: Path) -> pd.DataFrame:
    if path.with_suffix(".parquet").is_file():
        return pd.read_parquet(path.with_suffix(".parquet"))
    return pd.read_csv(path) if path.is_file() else pd.DataFrame()


def load_tables(merged_dir: Path) -> dict[str, pd.DataFrame]:
    summary = _read(merged_dir / SUMMARY_CSV)
    check_summary_schema(summary, str(merged_dir / SUMMARY_CSV))
    return {
        "summary": summary,
        "shells": _read(merged_dir / SHELL_CSV),
        "curves": _read(merged_dir / CURVES_CSV),
        "controls": _read(merged_dir / CONTROL_CSV),
    }


# --------------------------------------------------------------------------- rule
def verdict(summary: pd.DataFrame) -> dict[str, object]:
    """Evaluate the spec's pre-registered rule on the PGD eps 16/32 conditions.

    Per condition (dataset x eps), case-clustered medians of: LF sufficiency as
    a fraction of full Dice damage, K50 under the frequency-low-first ranking,
    and the LF count-matched permutation p.  CONFIRMED if all three hold in at
    least RULE_MIN_CONDITIONS of the four; REFUTED if sufficiency fails in at
    least RULE_MIN_CONDITIONS or K50 lies in HF in at least that many; PARTIAL otherwise.
    """

    s = summary[
        (summary["attack"] == RULE_ATTACK) & summary["epsilon_n"].isin(RULE_EPS)
    ].copy()
    s["lf_sufficiency_fraction"] = s["lf_sufficiency_dice"] / s["full_dice_damage"]
    rows = []
    for (dataset, eps), g in s.groupby(["dataset", "epsilon_n"]):
        suff = median_ci(g["lf_sufficiency_fraction"].tolist())[1]
        k50 = median_ci(g["k50_frequency_low_first"].tolist())[1]
        p = median_ci(g["control_permutation_p"].tolist())[1]
        rows.append(
            {
                "dataset": dataset,
                "epsilon_n": int(eps),
                "n_cases": int(g["case_id"].nunique()),
                "lf_sufficiency_fraction": suff,
                "k50_frequency_low_first": k50,
                "control_permutation_p": p,
                "suff_ok": bool(np.isfinite(suff) and suff >= SUFFICIENCY_MIN),
                "k50_in_lf": bool(np.isfinite(k50) and k50 <= LF_SHELL_MAX),
                "k50_needs_hf": bool(np.isfinite(k50) and k50 >= HF_SHELL_MIN),
                "p_ok": bool(np.isfinite(p) and p <= P_MAX),
            }
        )
    table = pd.DataFrame(rows)
    if table.empty:
        return {
            "verdict": PARTIAL,
            "n_conditions": 0,
            "n_conditions_all_three": 0,
            "table": table,
        }
    all_three = int((table["suff_ok"] & table["k50_in_lf"] & table["p_ok"]).sum())
    suff_fail = int((~table["suff_ok"]).sum())
    hf_needed = int(table["k50_needs_hf"].sum())
    # Refutation first; with four conditions and a 3-of-4 bar the two cannot both hold.
    if suff_fail >= RULE_MIN_CONDITIONS or hf_needed >= RULE_MIN_CONDITIONS:
        result = REFUTED
    elif all_three >= RULE_MIN_CONDITIONS:
        result = CONFIRMED
    else:
        result = PARTIAL
    return {
        "verdict": result,
        "n_conditions": int(len(table)),
        "n_conditions_all_three": all_three,
        "n_conditions_sufficiency_failed": suff_fail,
        "n_conditions_k50_in_hf": hf_needed,
        "table": table,
    }


# --------------------------------------------------------------------------- cohort
def band_energy_summary(summary: pd.DataFrame) -> pd.DataFrame:
    out = []
    for keys, g in summary.groupby(list(CONDITION), observed=True):
        row = dict(zip(CONDITION, keys))
        row["n_cases"] = int(g["case_id"].nunique())
        for col in (
            "lf_energy_share",
            "mf_energy_share",
            "hf_energy_share",
            "dc_energy_share",
            "lf_sufficiency_dice",
            "lf_necessity_dice",
            "mf_sufficiency_dice",
            "hf_sufficiency_dice",
            "f_k50_frequency_low_first_cpm",
            "f_k80_frequency_low_first_cpm",
            "k50_frequency_low_first",
            "remove_all_damage_dice",
            "jaccard_lf_top8_utility",
            "lf_control_permutation_p",
            "lf_control_drawable_fraction",
            "mf_control_permutation_p",
            "hf_control_permutation_p",
        ):
            if col in g:
                low, med, high = median_ci(g[col].tolist())
                row[f"{col}_median"], row[f"{col}_ci_low"], row[f"{col}_ci_high"] = (
                    med,
                    low,
                    high,
                )
        out.append(row)
    return pd.DataFrame(out)


def shell_profile(shells: pd.DataFrame) -> pd.DataFrame:
    """Per-shell cohort medians of energy share, attribution, necessity, sufficiency."""

    out = []
    group_cols = [*CONDITION, "patch_id"]
    for keys, g in shells.groupby(group_cols, observed=True):
        row = dict(zip(group_cols, keys))
        row["band"] = str(g["band"].iloc[0])
        row["f_high_cpm"] = float(g["f_high_cpm"].iloc[0])
        row["n_cases"] = int(g["case_id"].nunique())
        for col in (
            "energy_share",
            "ig_attribution",
            "necessity_dice",
            "sufficiency_dice",
            "fractional_recovery_dice",
            "necessity_hd95",
            "linf_excess",
        ):
            if col in g:
                low, med, high = median_ci(g[col].tolist())
                row[f"{col}_median"], row[f"{col}_ci_low"], row[f"{col}_ci_high"] = (
                    med,
                    low,
                    high,
                )
        out.append(row)
    return pd.DataFrame(out).sort_values(group_cols).reset_index(drop=True)


def write_cohort_csvs(
    tables: dict[str, pd.DataFrame], out_dir: Path
) -> dict[str, pd.DataFrame]:
    out_dir.mkdir(parents=True, exist_ok=True)
    frames = {
        "band_summary": band_energy_summary(tables["summary"]),
        "shell_profile": shell_profile(tables["shells"]),
        "cohort_curves": cohort_curves(tables["curves"]),
        "cohort_concentration": cohort_concentration(tables["summary"]),
        "band_utility": anatomical_category_summary(tables["shells"]),
        "cohort_controls": cohort_controls(tables["summary"], tables["controls"]),
        "boundary_recovery": cohort_boundary_recovery(
            tables["summary"], tables["curves"]
        ),
    }
    for name, frame in frames.items():
        frame.to_csv(out_dir / f"{name}.csv", index=False)
    return frames


# --------------------------------------------------------------------------- figures
def _conditions(frame: pd.DataFrame) -> list[tuple]:
    return sorted(set(map(tuple, frame[["dataset", "attack", "epsilon_n"]].to_numpy())))


def fig_shell_layout(out: Path) -> None:
    edges = reference_shell_edges(N_SHELLS)
    fy = np.fft.fftshift(np.fft.fftfreq(256, d=0.5))
    fx = np.fft.fftshift(np.fft.fftfreq(256, d=0.5))
    r = np.sqrt(fy[:, None] ** 2 + fx[None, :] ** 2)
    shell = np.digitize(r, edges[1:-1]) + 1
    fig, ax = plt.subplots(figsize=(5, 4.5))
    im = ax.imshow(
        shell, extent=(fx[0], fx[-1], fy[0], fy[-1]), cmap="viridis", origin="lower"
    )
    for k, lab in ((8, "LF|MF"), (16, "MF|HF")):
        c = plt.Circle((0, 0), edges[k], fill=False, color="white", lw=1.2)
        ax.add_patch(c)
        ax.text(
            edges[k] * 0.72,
            edges[k] * 0.72,
            f"{lab} {edges[k]:.3f}",
            color="white",
            fontsize=8,
        )
    ax.set_xlabel("f_x (cycles/mm)")
    ax.set_ylabel("f_y (cycles/mm)")
    ax.set_title("24 radial shells on the in-plane reference grid (k_z = 0)")
    fig.colorbar(im, ax=ax, label="shell")
    fig.tight_layout()
    fig.savefig(out / "01_shell_layout.png", dpi=150)
    plt.close(fig)


def fig_shell_profile(profile: pd.DataFrame, out: Path) -> None:
    conds = _conditions(profile)
    fig, axes = plt.subplots(3, 1, figsize=(8, 9), sharex=True)
    for cond in conds:
        d, a, e = cond
        g = profile[
            (profile.dataset == d) & (profile.attack == a) & (profile.epsilon_n == e)
        ]
        label = f"{d} {a} ε{int(e)}"
        for ax, col in zip(
            axes, ("energy_share", "fractional_recovery_dice", "sufficiency_dice")
        ):
            ax.plot(g["patch_id"], g[f"{col}_median"], marker="o", ms=3, label=label)
            ax.fill_between(
                g["patch_id"], g[f"{col}_ci_low"], g[f"{col}_ci_high"], alpha=0.15
            )
    for ax, title in zip(
        axes,
        (
            "energy share per shell",
            "single-shell necessity / full Dice damage",
            "single-shell sufficiency (Dice damage kept)",
        ),
    ):
        ax.set_title(title, fontsize=10)
        for x in (8.5, 16.5):
            ax.axvline(x, color="grey", lw=0.8, ls="--")
        ax.grid(alpha=0.3)
    axes[-1].set_xlabel("shell (1 = lowest frequency); dashed = LF|MF|HF")
    axes[0].legend(fontsize=7, ncol=2)
    fig.tight_layout()
    fig.savefig(out / "02_shell_profile.png", dpi=150)
    plt.close(fig)


def fig_cumulative(curves: pd.DataFrame, band_summary: pd.DataFrame, out: Path) -> None:
    conds = _conditions(curves)
    n = len(conds)
    cols = min(3, n) if n else 1
    rows = int(np.ceil(n / cols)) if n else 1
    fig, axes = plt.subplots(
        rows, cols, figsize=(4.2 * cols, 3.4 * rows), squeeze=False
    )
    for ax, cond in zip(axes.ravel(), conds):
        d, a, e = cond
        for r in RANKINGS:
            g = curves[
                (curves.dataset == d)
                & (curves.attack == a)
                & (curves.epsilon_n == e)
                & (curves.ranking_type == r)
            ]
            if g.empty:
                continue
            ax.plot(
                g["k"], g["damage_removed_fraction_median"], label=r.replace("_", " ")
            )
        b = band_summary[
            (band_summary.dataset == d)
            & (band_summary.attack == a)
            & (band_summary.epsilon_n == e)
        ]
        if not b.empty and np.isfinite(b["k50_frequency_low_first_median"].iloc[0]):
            ax.axvline(
                b["k50_frequency_low_first_median"].iloc[0], color="k", ls=":", lw=1
            )
        ax.axhline(0.5, color="grey", lw=0.6)
        ax.set_title(f"{d} {a} ε{int(e)}", fontsize=9)
        ax.set_ylim(-0.05, 1.05)
        ax.grid(alpha=0.3)
    for ax in axes.ravel()[n:]:
        ax.axis("off")
    axes[0, 0].legend(fontsize=7)
    fig.supxlabel("shells removed (cumulative)")
    fig.supylabel("fraction of Dice damage recovered")
    fig.tight_layout()
    fig.savefig(out / "03_cumulative_curves.png", dpi=150)
    plt.close(fig)


def fig_bands(band_summary: pd.DataFrame, out: Path) -> None:
    if band_summary.empty:
        return
    labels = [
        f"{r.dataset}\n{r.attack}\nε{int(r.epsilon_n)}"
        for r in band_summary.itertuples()
    ]
    x = np.arange(len(labels))
    fig, axes = plt.subplots(1, 2, figsize=(11, 3.8))
    w = 0.26
    for i, b in enumerate(("lf", "mf", "hf")):
        axes[0].bar(
            x + (i - 1) * w,
            band_summary[f"{b}_energy_share_median"],
            w,
            label=b.upper(),
        )
        axes[1].bar(
            x + (i - 1) * w,
            band_summary[f"{b}_sufficiency_dice_median"],
            w,
            label=b.upper(),
        )
    axes[0].set_title("energy share per band")
    axes[1].set_title("keep-band-only: Dice damage retained")
    for ax in axes:
        ax.set_xticks(x)
        ax.set_xticklabels(labels, fontsize=7)
        ax.grid(axis="y", alpha=0.3)
    axes[0].legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(out / "04_bands.png", dpi=150)
    plt.close(fig)


def fig_greedy(controls: pd.DataFrame, out: Path) -> None:
    g = controls[controls["control_type"] == "greedy"]
    if g.empty:
        return
    conds = _conditions(g)
    fig, ax = plt.subplots(figsize=(6, 4))
    for cond in conds:
        d, a, e = cond
        h = g[(g.dataset == d) & (g.attack == a) & (g.epsilon_n == e)]
        med = h.groupby("greedy_step")["greedy_recovered_fraction"].median()
        ax.plot(med.index, med.values, marker="o", ms=3, label=f"{d} {a} ε{int(e)}")
    ax.set_xlabel("greedy step")
    ax.set_ylabel("Dice damage recovered (median over cases)")
    ax.grid(alpha=0.3)
    ax.legend(fontsize=7)
    fig.tight_layout()
    fig.savefig(out / "05_greedy.png", dpi=150)
    plt.close(fig)


def fig_controls(cohort_controls_frame: pd.DataFrame, out: Path) -> None:
    c = cohort_controls_frame[cohort_controls_frame["n_cases"] > 0]
    if c.empty:
        return
    labels = [f"{r.dataset}\n{r.attack}\nε{int(r.epsilon_n)}" for r in c.itertuples()]
    x = np.arange(len(labels))
    fig, ax = plt.subplots(figsize=(7, 3.8))
    ax.bar(x - 0.18, c["observed_median"], 0.36, label="LF band removed")
    ax.bar(x + 0.18, c["control_median"], 0.36, label="8 random non-LF shells removed")
    for xi, p in zip(x, c["permutation_p_median"]):
        ax.text(
            xi,
            max(c["observed_median"].max(), c["control_median"].max()) * 1.02,
            f"p={p:.2f}",
            ha="center",
            fontsize=7,
        )
    ax.set_xticks(x)
    ax.set_xticklabels(labels, fontsize=7)
    ax.set_ylabel("Dice damage recovered")
    ax.legend(fontsize=8)
    ax.grid(axis="y", alpha=0.3)
    fig.tight_layout()
    fig.savefig(out / "06_lf_vs_control.png", dpi=150)
    plt.close(fig)


# --------------------------------------------------------------------------- report
def _fmt(v: object, d: int = 3) -> str:
    try:
        f = float(v)  # type: ignore[arg-type]
    except (TypeError, ValueError):
        return str(v)
    return "—" if not np.isfinite(f) else f"{f:.{d}f}"


def write_report(tables, frames, rule, out: Path) -> None:
    bs = frames["band_summary"]
    lines = [
        "# Spectral-shell attribution (SSEUA) — cohort report",
        "",
        f"Cases: {tables['summary']['case_id'].nunique()}; conditions: {len(bs)}; "
        f"shells: {N_SHELLS}; case-clustered bootstrap medians (2,000 resamples).",
        "",
        "## Pre-registered rule",
        "",
        f"Confirmed if, in ≥ {RULE_MIN_CONDITIONS} of the 4 PGD ε16/32 conditions, keep-LF-only retains "
        f"≥ {SUFFICIENCY_MIN:.0%} of the Dice damage, K50 (frequency-low-first) ≤ shell {LF_SHELL_MAX}, "
        f"and LF beats the count-matched control at p ≤ {P_MAX}. Refuted if sufficiency fails in "
        f"≥ {RULE_MIN_CONDITIONS} or K50 ≥ shell {HF_SHELL_MIN} in ≥ {RULE_MIN_CONDITIONS}; "
        "refutation is tested first. n is the causal cohort; p rests on the 30-case interaction subcohort.",
        "",
        f"**Verdict: {rule['verdict']}** — all three held in {rule['n_conditions_all_three']} of "
        f"{rule['n_conditions']} conditions.",
        "",
        "| Dataset | ε | n | LF sufficiency / damage | K50 (shell) | p (count-matched) |",
        "| --- | --- | --- | --- | --- | --- |",
    ]
    for r in rule["table"].itertuples():
        lines.append(
            f"| {r.dataset} | {r.epsilon_n} | {r.n_cases} | {_fmt(r.lf_sufficiency_fraction)} "
            f"| {_fmt(r.k50_frequency_low_first, 1)} | {_fmt(r.control_permutation_p)} |"
        )
    lines += [
        "",
        "## Bands: energy, sufficiency, necessity, cut-offs",
        "",
        "| Dataset | Attack | ε | n | LF / MF / HF energy | DC | keep-LF-only | remove-LF | f_K50 cyc/mm | f_K80 | remove-all | LF p | LF energy-drawable |",
        "| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |",
    ]
    for r in bs.itertuples():
        g = lambda c, d=3: _fmt(getattr(r, c, _NAN), d)  # noqa: E731
        lines.append(
            f"| {r.dataset} | {r.attack} | {int(r.epsilon_n)} | {r.n_cases} "
            f"| {g('lf_energy_share_median')} / {g('mf_energy_share_median')} / {g('hf_energy_share_median')} "
            f"| {g('dc_energy_share_median')} | {g('lf_sufficiency_dice_median')} | {g('lf_necessity_dice_median')} "
            f"| {g('f_k50_frequency_low_first_cpm_median')} | {g('f_k80_frequency_low_first_cpm_median')} "
            f"| {g('remove_all_damage_dice_median')} | {g('lf_control_permutation_p_median')} "
            f"| {g('lf_control_drawable_fraction_median', 2)} |"
        )
    lines += [
        "",
        "Sufficiency and necessity are Dice-damage units (not fractions). `remove-all` is the",
        "damage left when every shell is removed (DC and outside-mask components stay); it",
        "must be ≈ 0. `f_K50` is the upper edge of the last shell removed when half the damage",
        "returns under the frequency-low-first ranking. Interventions are counterfactuals that",
        "leave the L∞ ball (`linf_excess` in the shell table).",
        "",
        "Figures: `01_shell_layout.png`, `02_shell_profile.png`, `03_cumulative_curves.png`,",
        "`04_bands.png`, `05_greedy.png`, `06_lf_vs_control.png`. Tables: `band_summary.csv`,",
        "`shell_profile.csv`, `cohort_curves.csv`, `cohort_concentration.csv`, `band_utility.csv`,",
        "`cohort_controls.csv`, `boundary_recovery.csv`.",
    ]
    (out / "SPECTRAL_ATTRIBUTION_REPORT.md").write_text(
        "\n".join(lines) + "\n", encoding="utf-8"
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--merged-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--output-dir", type=Path, default=None)
    args = parser.parse_args()
    out = args.output_dir or args.merged_dir / "analysis"
    out.mkdir(parents=True, exist_ok=True)
    tables = load_tables(args.merged_dir)
    frames = write_cohort_csvs(tables, out)
    rule = verdict(tables["summary"])
    rule["table"].to_csv(out / "rule_evaluation.csv", index=False)
    fig_shell_layout(out)
    fig_shell_profile(frames["shell_profile"], out)
    fig_cumulative(frames["cohort_curves"], frames["band_summary"], out)
    fig_bands(frames["band_summary"], out)
    fig_greedy(tables["controls"], out)
    fig_controls(frames["cohort_controls"], out)
    write_report(tables, frames, rule, out)
    print(
        f"verdict {rule['verdict']} ({rule['n_conditions_all_three']}/{rule['n_conditions']}); wrote {out}"
    )


if __name__ == "__main__":
    main()
