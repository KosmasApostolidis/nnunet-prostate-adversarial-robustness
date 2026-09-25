"""Two tables of budget- and fold-averaged results per attack, models as columns.

Table 1 (WG): columns nnU-Net, ResEnc-M, ResEnc-L, ResEnc-XL.
Table 2 (zones): the same four models x {TZ+CZ, PZ}.
Rows: attack x metric; each cell is the fold mean from the campaign summaries
averaged over the five attacked budgets (eps = 0.02 ... 0.1), matching
``analyze_main_comparison_stats.py``. The eps = 0 clean row is reported once.

Usage::

    PYTHONPATH=src python experiments/build_attack_budget_avg_tables.py \
        --output-dir results/blade_attack/attack_comparison
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
from build_attack_comparison_table import MODELS, load  # noqa: E402
from run_twelve_arm_wg_campaign import ARM_LABELS, ARMS, METRICS  # noqa: E402


def budget_avg(long: pd.DataFrame, classes: list[str]) -> pd.DataFrame:
    sub = long[(long["epsilon"] > 0) & long["class"].isin(classes)]
    # protocol order, but every BLADE variant grouped at the bottom
    present = [a for a in ARMS if a in set(sub["attack"])]
    arms = [a for a in present if not a.startswith("blade")] + [
        a for a in present if a.startswith("blade")
    ]
    avg = sub.groupby(["metric", "attack", "model", "class"], observed=True)["mean"].mean()
    wide = avg.unstack(["model", "class"])
    cols = [(m, c) for m in MODELS for c in classes]
    wide = wide.reindex(columns=pd.MultiIndex.from_tuples(cols))
    wide = wide.reindex(
        pd.MultiIndex.from_product([METRICS, arms], names=["metric", "attack"])
    )
    clean = (
        long[(long["epsilon"] == 0) & long["class"].isin(classes)]
        .groupby(["metric", "model", "class"], observed=True)["mean"]
        .mean()
        .unstack(["model", "class"])
        .reindex(columns=wide.columns)
    )
    clean.index = pd.MultiIndex.from_tuples(
        [(m, "clean") for m in clean.index], names=["metric", "attack"]
    )
    out = pd.concat([clean, wide]).sort_index(level="metric", sort_remaining=False)
    out = out.reindex(
        [(m, a) for m in METRICS for a in ["clean", *arms]]
    )
    out.index = pd.MultiIndex.from_tuples(
        [(m, ARM_LABELS.get(a, a)) for m, a in out.index], names=["metric", "attack"]
    )
    if len(classes) == 1:
        out.columns = [m for m, _ in out.columns]
    else:
        out.columns = [f"{m} {c}" for m, c in out.columns]
    return sort_by_damage(out)


def sort_by_damage(table: pd.DataFrame) -> pd.DataFrame:
    """Within each metric block: clean first, then arms least to most damaging
    (row mean across all columns; Dice descending, HD95/ASD ascending)."""
    blocks = []
    for metric in METRICS:
        block = table.xs(metric, level="metric", drop_level=False)
        clean, arms = block.iloc[[0]], block.iloc[1:]
        order = arms.mean(axis=1).sort_values(ascending=metric != "dice").index
        blocks.append(pd.concat([clean, arms.loc[order]]))
    return pd.concat(blocks)


HIGHLIGHT = '<span style="color:blue">**{:.3f}**</span>'


def to_markdown(table: pd.DataFrame) -> str:
    """Rows in protocol order; per column the highest Dice / lowest HD95 and ASD
    among the attacks (clean excluded) is bold blue."""
    parts = []
    for metric in METRICS:
        sub = table.xs(metric, level="metric")
        best = sub.iloc[1:].idxmax() if metric == "dice" else sub.iloc[1:].idxmin()
        fmt = sub.map(lambda v: f"{v:.3f}")
        for col, arm in best.items():
            fmt.loc[arm, col] = HIGHLIGHT.format(sub.loc[arm, col])
        parts.append(f"## {metric}\n\n" + fmt.to_markdown())
    return "\n\n".join(parts) + "\n"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument(
        "--output-dir", type=Path, default=Path("results/blade_attack/attack_comparison")
    )
    args = parser.parse_args()
    long = pd.concat([load(m) for m in MODELS], ignore_index=True)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    for name, classes in (("wg", ["WG"]), ("zones", ["TZ+CZ", "PZ"])):
        table = budget_avg(long, classes)
        stem = args.output_dir / f"attack_comparison_budget_avg_{name}"
        table.to_csv(stem.with_suffix(".csv"), float_format="%.4f")
        stem.with_suffix(".md").write_text(to_markdown(table))
        print(f"Saved {stem}.csv / .md ({table.shape[0]} rows x {table.shape[1]} cols)")


if __name__ == "__main__":
    main()
