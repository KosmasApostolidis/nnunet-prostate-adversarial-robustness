"""Compare the original Auto-PGD against FGSM, PGD and the manuscript's APGD.

The FGSM, PGD and APGD arms come from the frozen campaign in
``results/wg_zones_adversarial_robustness_results``; only the Auto-PGD arm is
new. Both cover the same 1,396 whole-gland cases, folds and epsilon values, so
the arms are joined rather than recomputed.

Two hazards this script handles explicitly:

* The per-sample CSVs contain two runs of each arm. Every ``(case_id, epsilon)``
  key appears twice, the two runs agreeing to about 4e-4 mean Dice. Rows are
  deduplicated on first occurrence and a disagreement beyond ``DUPLICATE_TOL``
  is an error rather than a silent average.
* That agreement also sets the resolution of the comparison: a difference in
  mean Dice below roughly 0.005 is within run-to-run noise and should not be
  read as an effect.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

REPO_ROOT = Path(__file__).resolve().parents[1]
CAMPAIGN_DIR = REPO_ROOT / "results" / "wg_zones_adversarial_robustness_results"
NEW_ARM_DIR = REPO_ROOT / "results" / "autopgd_comparison"
FOLDS = (0, 1, 2, 3, 4)
METRICS = ("dice", "hd95", "asd")
ARMS = ("fgsm", "pgd", "a_pgd", "auto_pgd")
ARM_LABELS = {
    "fgsm": "FGSM",
    "pgd": "PGD",
    "a_pgd": "APGD (ours)",
    "auto_pgd": "Auto-PGD [18]",
}
DUPLICATE_TOL = 0.02
NOISE_FLOOR = 0.005
N_BOOTSTRAP = 2000
SEED = 42


def load_per_sample(
    root: Path, attack: str, folds: tuple[int, ...] = FOLDS
) -> pd.DataFrame:
    """Deduplicated per-case rows for one arm, with a ``fold`` column.

    The campaign CSVs hold two runs per key. Keeping the first occurrence is
    equivalent to averaging them to within 4e-4 mean Dice, but a larger
    disagreement means the file mixes genuinely different configurations, which
    is an error worth failing on.
    """
    frames = []
    for fold in folds:
        path = root / f"wg_{attack}_fold{fold}_per_sample.csv"
        frame = pd.read_csv(path)
        spread = frame.groupby(["case_id", "epsilon"])["dice"].agg(
            lambda s: s.max() - s.min()
        )
        if float(spread.max()) > DUPLICATE_TOL:
            worst = spread.idxmax()
            raise ValueError(
                f"{path.name}: duplicate rows for {worst} disagree by "
                f"{float(spread.max()):.4f} Dice, above the {DUPLICATE_TOL} "
                "tolerance; the file mixes different runs."
            )
        frame = frame.drop_duplicates(subset=["case_id", "epsilon"], keep="first")
        frame["fold"] = fold
        frames.append(frame)
    return pd.concat(frames, ignore_index=True)


def fold_summary(per_sample: pd.DataFrame, metric: str) -> pd.DataFrame:
    """Mean and fold-level SEM per epsilon, in the Table I convention.

    The reported uncertainty is the spread across fold means divided by the
    square root of the number of folds, not the spread across cases.
    """
    fold_means = (
        per_sample.groupby(["epsilon", "fold"])[metric].mean().reset_index()
    )
    rows = []
    for epsilon, group in fold_means.groupby("epsilon"):
        values = group[metric].to_numpy(dtype=float)
        values = values[np.isfinite(values)]
        n = len(values)
        sem = float(np.std(values, ddof=1) / np.sqrt(n)) if n > 1 else np.nan
        rows.append(
            {
                "epsilon": float(epsilon),
                "mean": float(values.mean()) if n else np.nan,
                "sem": sem,
                "n_folds": n,
            }
        )
    return pd.DataFrame(rows).sort_values("epsilon").reset_index(drop=True)


def paired_delta(
    left: pd.DataFrame,
    right: pd.DataFrame,
    metric: str,
    n_boot: int = N_BOOTSTRAP,
    seed: int = SEED,
) -> pd.DataFrame:
    """Per-case ``left - right`` differences with a case bootstrap CI.

    The difference is taken inside each case before averaging, so both arms of
    the contrast come from the same cases.
    """
    merged = left.merge(
        right,
        on=["case_id", "epsilon", "class"],
        suffixes=("_left", "_right"),
    )
    merged["delta"] = merged[f"{metric}_left"] - merged[f"{metric}_right"]
    rng = np.random.default_rng(seed)
    rows = []
    for epsilon, group in merged.groupby("epsilon"):
        values = group["delta"].to_numpy(dtype=float)
        values = values[np.isfinite(values)]
        if len(values) == 0:
            continue
        draws = np.array(
            [
                values[rng.integers(0, len(values), len(values))].mean()
                for _ in range(n_boot)
            ]
        )
        rows.append(
            {
                "epsilon": float(epsilon),
                "delta": float(values.mean()),
                "ci95_low": float(np.percentile(draws, 2.5)),
                "ci95_high": float(np.percentile(draws, 97.5)),
                "n_cases": int(len(values)),
            }
        )
    return pd.DataFrame(rows).sort_values("epsilon").reset_index(drop=True)


def _arm_root(attack: str) -> Path:
    return NEW_ARM_DIR if attack == "auto_pgd" else CAMPAIGN_DIR


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, default=NEW_ARM_DIR)
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    arms = {a: load_per_sample(_arm_root(a), a) for a in ARMS}

    summaries = []
    for attack, frame in arms.items():
        for metric in METRICS:
            table = fold_summary(frame, metric)
            table.insert(0, "metric", metric)
            table.insert(0, "attack", attack)
            summaries.append(table)
    summary = pd.concat(summaries, ignore_index=True)
    summary.to_csv(args.output_dir / "four_arm_summary.csv", index=False)
    print(summary.to_string(index=False))

    delta = paired_delta(arms["auto_pgd"], arms["a_pgd"], "dice")
    delta["within_noise"] = delta["delta"].abs() < NOISE_FLOOR
    delta.to_csv(args.output_dir / "autopgd_minus_apgd_dice.csv", index=False)
    print("\nAuto-PGD minus APGD, paired per case, Dice:")
    print(delta.to_string(index=False))

    fig, ax = plt.subplots(figsize=(6.0, 4.2), layout="constrained")
    for attack in ARMS:
        table = summary[(summary["attack"] == attack) & (summary["metric"] == "dice")]
        ax.errorbar(
            table["epsilon"], table["mean"], yerr=table["sem"],
            marker="o", capsize=3, linewidth=1.6, label=ARM_LABELS[attack],
        )
    ax.set_xlabel(r"$\varepsilon$ ($L_\infty$, after z-score normalization)")
    ax.set_ylabel("Mean Dice")
    ax.set_title("Whole gland under four attacks at matched budget")
    ax.grid(alpha=0.3)
    ax.legend()
    for suffix in (".png", ".pdf"):
        fig.savefig(
            (args.output_dir / "four_arm_dice").with_suffix(suffix),
            dpi=600, bbox_inches="tight",
        )
    plt.close(fig)
    print(f"\nSaved to {args.output_dir}")


if __name__ == "__main__":
    main()
