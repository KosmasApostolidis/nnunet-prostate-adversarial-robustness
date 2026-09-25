"""One table of fold-averaged results per attack: classes as columns.

Rows: model (nnU-Net / ResEnc-M / ResEnc-L / ResEnc-XL) x attack x epsilon x
metric; columns: WG, TZ+CZ, PZ (mean across the five folds, from the campaign
summary CSVs). WG comes from the whole-gland trees, TZ+CZ / PZ from the
confined zones trees.

Usage::

    PYTHONPATH=src python experiments/build_attack_comparison_table.py \
        --output-dir results/blade_attack/attack_comparison
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
from run_twelve_arm_wg_campaign import ARM_LABELS, ARMS, METRICS  # noqa: E402

WG_ROOT = Path("results/blade_attack/whole_gland")
ZONES_ROOT = Path("results/blade_attack/zones_confined_mc")
MODELS = {
    "nnU-Net": ("native_per_eps/main_comparison_wg_summary.csv", "campaign"),
    "ResEnc-M": ("native_per_eps_resenc_m/thirteen_arm_wg_summary.csv", "campaign_resenc_m"),
    "ResEnc-L": ("native_per_eps_resenc_l/thirteen_arm_wg_summary.csv", "campaign_resenc_l"),
    "ResEnc-XL": ("native_per_eps_resenc_xl/thirteen_arm_wg_summary.csv", "campaign_resenc_xl"),
}
CLASSES = ["WG", "TZ+CZ", "PZ"]


def load(model: str) -> pd.DataFrame:
    wg_rel, zones_dir = MODELS[model]
    paths = [WG_ROOT / wg_rel, ZONES_ROOT / zones_dir / "thirteen_arm_zones_summary.csv"]
    frames = []
    for path in paths:
        if not path.is_file():
            sys.exit(f"summary not found: {path}")
        frames.append(pd.read_csv(path))
    table = pd.concat(frames, ignore_index=True)
    table.insert(0, "model", model)
    return table


def build() -> pd.DataFrame:
    long = pd.concat([load(m) for m in MODELS], ignore_index=True)
    wide = long.pivot_table(
        index=["metric", "model", "attack", "epsilon"], columns="class", values="mean"
    ).reset_index()[["metric", "model", "attack", "epsilon", *CLASSES]]
    wide["metric"] = pd.Categorical(wide["metric"], METRICS, ordered=True)
    wide["model"] = pd.Categorical(wide["model"], list(MODELS), ordered=True)
    wide["attack"] = pd.Categorical(wide["attack"], ARMS, ordered=True)
    wide = wide.sort_values(["metric", "model", "attack", "epsilon"]).reset_index(drop=True)
    wide["attack"] = wide["attack"].map(ARM_LABELS)
    return wide


def to_markdown(wide: pd.DataFrame) -> str:
    parts = []
    for metric in METRICS:
        sub = wide[wide["metric"] == metric].drop(columns="metric")
        parts.append(f"## {metric}\n\n" + sub.to_markdown(index=False, floatfmt=".3f"))
    return "\n\n".join(parts) + "\n"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument(
        "--output-dir", type=Path, default=Path("results/blade_attack/attack_comparison")
    )
    args = parser.parse_args()
    wide = build()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    csv = args.output_dir / "attack_comparison_table.csv"
    md = args.output_dir / "attack_comparison_table.md"
    wide.to_csv(csv, index=False, float_format="%.4f")
    md.write_text(to_markdown(wide))
    print(f"Saved {csv} ({len(wide)} rows) and {md}")


if __name__ == "__main__":
    main()
