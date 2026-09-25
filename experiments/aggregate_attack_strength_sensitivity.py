"""Aggregate the attack-strength sweep into a table and a figure.

Reads the per-sample CSVs written by run_attack_strength_sensitivity.py and
reports mean Dice per configuration, pooling the fold subsamples. Because every
configuration attacks the same cases, the comparison across iteration counts and
restart counts is paired, so the change relative to the reported 20-iteration,
single-restart setting is reported per case rather than as a difference of
independent means.
"""

from __future__ import annotations

import argparse
import re
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_ROOT = REPO_ROOT / "rebuttal_paper2" / "attack_strength_sensitivity"
BASELINE = (20, 1)
CLASS_COLORS = {"WG": "#1f77b4", "TZ+CZ": "#ff7f0e", "PZ": "#2ca02c"}


def load(root: Path) -> pd.DataFrame:
    frames = []
    for path in sorted(root.glob("steps*_restarts*/*/*_per_sample.csv")):
        match = re.search(r"steps(\d+)_restarts(\d+)", str(path))
        fold = re.search(r"fold(\d+)_per_sample", path.name)
        if match is None or fold is None:
            continue
        frame = pd.read_csv(path)
        frame["steps"] = int(match.group(1))
        frame["restarts"] = int(match.group(2))
        frame["fold"] = int(fold.group(1))
        frames.append(frame)
    if not frames:
        raise SystemExit(f"No per-sample CSVs under {root}")
    return pd.concat(frames, ignore_index=True)


def _paired_bootstrap_ci(
    deltas: pd.Series, n_boot: int = 10000, seed: int = 0
) -> tuple[float, float]:
    """Percentile CI for a paired mean, resampling cases with replacement.

    Cases are patients and each arm attacks the same ones, so the case is the
    independent unit and the pairing is preserved by resampling case-level
    differences rather than the two arms separately.
    """
    values = deltas.to_numpy(dtype=float)
    if values.size < 2:
        return float("nan"), float("nan")
    rng = np.random.default_rng(seed)
    draws = rng.integers(0, values.size, size=(n_boot, values.size))
    means = values[draws].mean(axis=1)
    return float(np.percentile(means, 2.5)), float(np.percentile(means, 97.5))


def report_checks(paired: pd.DataFrame, data: pd.DataFrame) -> None:
    """Print the design's sanity checks.

    Only the restart arm carries a per-case guarantee. Restart 0 starts from the
    clean image, so N restarts search a superset of the single-restart attack and
    can only raise the attack loss. Adding iterations is *not* a superset: the
    step size is alpha = eps / k, so a larger k is a different trajectory, and an
    individual case may end up marginally better off. Only the mean is expected
    to fall. Selection also maximizes the Dice+CE loss rather than minimizing
    Dice, so even the restart arm may move a single case's Dice slightly up.
    """
    print("\nSanity checks")

    restart_arms = sorted({r for r in data["restarts"].unique() if r > 1})
    for restarts in restart_arms:
        for steps in sorted(data[data["restarts"] == restarts]["steps"].unique()):
            multi = data[(data["steps"] == steps) & (data["restarts"] == restarts)]
            single = data[(data["steps"] == steps) & (data["restarts"] == 1)]
            if single.empty or multi.empty:
                continue
            merged = multi.merge(
                single[["case_id", "class", "epsilon", "dice"]],
                on=["case_id", "class", "epsilon"],
                suffixes=("_multi", "_single"),
            )
            gap = merged["dice_multi"] - merged["dice_single"]
            label = f"{steps}x{restarts} vs {steps}x1"
            if gap.max() <= 0.01:
                print(
                    f"  [ok] {label}: restarts never weaken the attack "
                    f"(worst case {gap.max():+.4f}, mean {gap.mean():+.4f})"
                )
            else:
                worst = merged.loc[gap.idxmax()]
                print(
                    f"  [WARN] {label}: restart set may have regressed; worst "
                    f"{gap.max():+.4f} ({worst['case_id']}, {worst['class']}, "
                    f"eps={worst['epsilon']})"
                )

    endpoint = paired[np.isclose(paired["epsilon"], 0.1)]
    for class_name, block in endpoint.groupby("class"):
        means = block.groupby(["steps", "restarts"])["delta"].mean().sort_index()
        single = means[[i for i in means.index if i[1] == 1]]
        monotone = all(
            single.iloc[i] >= single.iloc[i + 1] - 1e-9
            for i in range(len(single) - 1)
        )
        flag = "ok" if monotone else "WARN"
        print(f"  [{flag}] {class_name}: mean Dice drop monotone in iterations")

    improved = paired[paired["delta"] > 1e-6]
    print(
        f"  [info] {len(improved)}/{len(paired)} case-eps pairs sit above the "
        "20-iteration baseline; expected at the case level since alpha = eps/k "
        "makes each arm a different trajectory"
    )


def summarize(data: pd.DataFrame) -> pd.DataFrame:
    keys = ["class", "epsilon", "steps", "restarts"]
    summary = (
        data.groupby(keys)["dice"]
        .agg(mean_dice="mean", sd="std", n_cases="size")
        .reset_index()
    )
    baseline = data[
        (data["steps"] == BASELINE[0]) & (data["restarts"] == BASELINE[1])
    ][["case_id", "class", "epsilon", "dice"]].rename(columns={"dice": "dice_baseline"})
    paired = data.merge(baseline, on=["case_id", "class", "epsilon"], how="inner")
    paired["delta"] = paired["dice"] - paired["dice_baseline"]
    deltas = (
        paired.groupby(keys)["delta"]
        .agg(mean_delta="mean", sd_delta="std", n_paired="size")
        .reset_index()
    )
    bounds = (
        paired.groupby(keys)["delta"]
        .apply(lambda d: pd.Series(_paired_bootstrap_ci(d), index=["ci_lo", "ci_hi"]))
        .unstack()
        .reset_index()
    )
    merged = summary.merge(deltas, on=keys, how="left")
    return merged.merge(bounds, on=keys, how="left")


def paired_frame(data: pd.DataFrame) -> pd.DataFrame:
    """Case-level differences against the reported 20-iteration arm."""
    baseline = data[
        (data["steps"] == BASELINE[0]) & (data["restarts"] == BASELINE[1])
    ][["case_id", "class", "epsilon", "dice"]].rename(columns={"dice": "dice_baseline"})
    paired = data.merge(baseline, on=["case_id", "class", "epsilon"], how="inner")
    paired["delta"] = paired["dice"] - paired["dice_baseline"]
    return paired


def plot(summary: pd.DataFrame, out_path: Path) -> None:
    classes = [c for c in ("WG", "TZ+CZ", "PZ") if c in set(summary["class"])]
    fig, axes = plt.subplots(
        1, len(classes), figsize=(5.2 * len(classes), 4.4), squeeze=False
    )
    configs = sorted(
        {(int(s), int(r)) for s, r in zip(summary["steps"], summary["restarts"])}
    )
    for index, class_name in enumerate(classes):
        ax = axes[0, index]
        subset = summary[summary["class"] == class_name]
        for steps, restarts in configs:
            row = subset[
                (subset["steps"] == steps) & (subset["restarts"] == restarts)
            ].sort_values("epsilon")
            if row.empty:
                continue
            is_baseline = (steps, restarts) == BASELINE
            ax.plot(
                row["epsilon"],
                row["mean_dice"],
                marker="o" if is_baseline else "s",
                markersize=5,
                linewidth=2.2 if is_baseline else 1.4,
                linestyle="-" if restarts == 1 else "--",
                label=f"k={steps}, {restarts} restart" + ("s" if restarts > 1 else ""),
            )
        ax.set_title(class_name, fontsize=12, fontweight="bold")
        ax.set_xlabel(r"Attack strength $\varepsilon$")
        if index == 0:
            ax.set_ylabel("Mean Dice")
        ax.grid(alpha=0.3)
        ax.legend(fontsize=8)
    fig.suptitle(
        "PGD Dice under increasing iterations and restarts",
        fontsize=13,
        fontweight="bold",
    )
    fig.tight_layout()
    fig.savefig(out_path, dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved {out_path}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=DEFAULT_ROOT)
    args = parser.parse_args()

    data = load(args.root)
    summary = summarize(data)
    csv_path = args.root / "attack_strength_summary.csv"
    summary.to_csv(csv_path, index=False)
    print(f"  Saved {csv_path}")

    endpoint = summary[np.isclose(summary["epsilon"], 0.1)]
    print("\nAt epsilon = 0.1 (delta is paired against the 20-iteration arm):")
    shown = endpoint.copy()
    shown["delta_95ci"] = [
        f"[{lo:+.4f}, {hi:+.4f}]" for lo, hi in zip(shown["ci_lo"], shown["ci_hi"])
    ]
    print(
        shown[
            ["class", "steps", "restarts", "mean_dice", "mean_delta", "delta_95ci", "n_cases"]
        ].to_string(index=False)
    )
    report_checks(paired_frame(data), data)
    plot(summary, args.root / "attack_strength_sensitivity.png")


if __name__ == "__main__":
    main()
