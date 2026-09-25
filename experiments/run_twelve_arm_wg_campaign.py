"""Multi-arm whole-gland adversarial campaign (currently twelve attacks).

Runs FGSM, PGD, the manuscript's adaptive-step APGD, the original Auto-PGD of
Croce and Hein, the scheduled-step APGD, Auto-PGD+, SegPGD, CosPGD, DAG, SEA
and the two BLADE arms against the whole-gland nnU-Net over folds 0-4 and the epsilon grid, then averages
Dice, HD95 and ASD across folds. The number of attacks is derived from ``ARMS``
and is written into the summary CSV name, the figure name and the figure title,
so the outputs always state how many attacks the campaign compared.

Why this exists alongside the frozen campaign
---------------------------------------------
``results/wg_zones_adversarial_robustness_results`` already holds FGSM, PGD and
APGD from the published campaign. This driver recomputes every arm under one
harness instead of reusing those numbers, so every column sees the same code,
the same seed and the same run, and are internally comparable.

It also means the PGD column here will NOT match the published table. The
attacks package is untracked in git, and a fold-0 probe measured plain PGD at
0.629 where the recorded value was 0.568, so ``pgd.py`` has changed since the
published numbers were produced. The frozen module ``a_pgd_campaign.py``
reproduces to 0.001, so the divergence is specific to the PGD path. Use this
table for within-table comparisons; cite the published table for published
numbers.

Attack settings follow ``configs/attacks/*.json``: 20 iterations and one restart
for every iterative arm, a random start for all of them, momentum 0.9 for APGD.
PGD needs ``--random-start`` passed explicitly because, unlike the other two, the
driver does not switch that default on for it. Auto-PGD takes no step-size or
momentum argument at all -- it fixes its step at 2*eps by construction, and
apgd_updated sets its step from a cosine backbone rather than an argument.

The apgd_updated arm is new work, not part of any published result; it exists
to test whether the campaign attack's narrow step range is what limits it.

The auto_pgd_plus arm is a REFUTED attempt to speed up Auto-PGD: on a paired
12-case fold-0 ablation at native eps=0.04 it was weaker than Auto-PGD
(``results/autopgd_plus_ablation/README.md``). It is included here so the
negative result is documented at full scale under the same protocol as the
other arms, not because it is recommended.

SegPGD (Gu et al., ECCV 2022) and CosPGD (Agnihotri et al., ICML 2024) are
loss variants of PGD: they keep the pgd arm's update, step (eps/k), random
start and projection and change only the objective, so their columns isolate
the effect of the segmentation-specific loss.

DAG (Xie et al., ICCV 2017) ascends the logit gap of the voxels it has not yet
flipped with an L-infinity-normalised (not signed) step and no random start.
SEA (Croce, Singh and Hein, 2023) runs Auto-PGD once per loss in {masked CE,
class-balanced masked CE, Jensen-Shannon} and keeps the worst Dice per case,
so it spends three times the gradient evaluations of every other arm.

BLADE is this repository's own attack (``src/mri_prostate_seg/attacks/blade.py``).
It replaces the shared-trajectory protocol with an ascending-epsilon ladder:
Auto-PGD optimises at the smallest epsilon, then each larger ball continues from
the previous rung's iterate, so every reported epsilon is optimised natively
inside its own ball instead of being projected down from the largest. ``blade1``
runs the ladder on ``dice_ce_loss`` alone; ``blade`` runs it once per objective
in {dice_ce, sign-flipped Kervadec boundary loss, frontier margin} and keeps the
lowest foreground Dice per rung.

Budget, stated plainly because the two BLADE arms are not free. With ``R``
epsilons on the grid, ``blade1`` costs ``R * n_steps`` gradients per case and
``blade`` costs ``3R * n_steps``, whereas a shared-trajectory arm costs
``n_steps`` and shared-trajectory SEA costs ``3 * n_steps``. Under
``--native-per-eps`` -- where every other arm is also run once per epsilon --
the budgets match: ``blade1`` against the single-objective arms and ``blade``
against SEA. Read the default shared-trajectory table for what each attack
achieves, and the native table for a matched-cost comparison.

The two BLADE arms always run in ladder mode, with the whole epsilon grid as
their rungs, whether or not ``--native-per-eps`` is passed. Splitting them into
one call per epsilon would discard the warm start, which is the thing being
measured.

The default output directory is ``results/blade_attack/whole_gland/shared_trajectory``
for ``--model wg`` and ``results/blade_attack/zones/campaign`` for
``--model zones``. The summary CSV and the figure are named by the run that
produces them (``--figure-name``), not by the directory. A ``--native-per-eps``
run belongs in a different tree -- pass ``--output-dir`` explicitly, the way
``results/blade_attack/whole_gland/native_per_eps`` was produced.

Usage::

    python experiments/run_twelve_arm_wg_campaign.py                 # run then aggregate
    python experiments/run_twelve_arm_wg_campaign.py --aggregate-only
    python experiments/run_twelve_arm_wg_campaign.py --max-samples 2 # smoke test
    python experiments/run_twelve_arm_wg_campaign.py --arms dag sea --jobs 5
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

REPO_ROOT = Path(__file__).resolve().parents[1]
PGD_DRIVER = REPO_ROOT / "experiments" / "pgd_adversarial_evaluation.py"
FGSM_DRIVER = REPO_ROOT / "experiments" / "fgsm_adversarial_evaluation.py"

ARMS = (
    "fgsm", "pgd", "a_pgd", "auto_pgd", "apgd_updated", "auto_pgd_plus",
    "segpgd", "cospgd", "dag", "sea", "blade1", "blade", "blade_mm",
    "auto_pgd_r3", "blade_boundary", "blade_frontier",
)
# Arms whose trajectory IS the epsilon grid: they must see every epsilon in one
# call, so the driver never splits them per epsilon.
LADDER_ARMS = (
    "blade1", "blade", "blade_mm", "blade_boundary", "blade_frontier",
)
# An arm whose name is not the driver's --attack value, because it is that
# attack under different settings rather than a different attack.
ARM_ATTACK = {"auto_pgd_r3": "auto_pgd"}
# Restart count per arm. auto_pgd_r3 is the compute-matched control for the
# three-ladder BLADE-3 and three-loss SEA: one objective, three restarts.
ARM_RESTARTS = {"auto_pgd_r3": 3}
_COUNT_WORDS = {
    5: "five", 6: "six", 7: "seven", 8: "eight", 9: "nine", 10: "ten",
    11: "eleven", 12: "twelve", 13: "thirteen", 14: "fourteen",
    15: "fifteen", 16: "sixteen",
}
N_ARMS = len(ARMS)


def _count_word(n: int) -> str:
    """Spelled-out arm count for the output names and the figure title."""
    return _COUNT_WORDS.get(n, str(n))


ARM_COUNT = _count_word(N_ARMS)

# Default output tree per model, relative to ``results/blade_attack``. A
# --native-per-eps run writes a different protocol and belongs in
# whole_gland/native_per_eps, so it must pass --output-dir explicitly.
DEFAULT_OUTPUT_SUBDIR = {
    "wg": Path("whole_gland") / "shared_trajectory",
    "zones": Path("zones") / "campaign",
}
ARM_LABELS = {
    "fgsm": "FGSM",
    "pgd": "PGD",
    "a_pgd": "APGD",
    "auto_pgd": "Auto-PGD",
    "apgd_updated": "APGD-updated",
    "auto_pgd_plus": "Auto-PGD+",
    "segpgd": "SegPGD",
    "cospgd": "CosPGD",
    "dag": "DAG",
    "sea": "SEA",
    "blade1": "BLADE-DiceCE",
    "blade": "BLADE-3",
    "blade_mm": "BLADE-MM",
    "auto_pgd_r3": "Auto-PGD x3",
    "blade_boundary": "BLADE-Boundary",
    "blade_frontier": "BLADE-Frontier",
}
# Colourblind-safe (Okabe-Ito). The two schedule variants of the same family --
# the campaign APGD and its scheduled successor -- share a hue family so the
# comparison the figure exists for reads at a glance. Auto-PGD and its refuted
# successor Auto-PGD+ likewise share the dashed line style, and the two loss
# variants of PGD (SegPGD, CosPGD) share a dotted one; DAG and SEA, which
# are neither, take dash-dot. The two BLADE arms take a densely dash-dotted
# style used by nothing else, because what separates them from every other arm
# is the trajectory protocol rather than the objective or the step rule.
ARM_STYLE = {
    "fgsm": ("#999999", "o", "-"),
    "pgd": ("#0072B2", "s", "-"),
    "a_pgd": ("#D55E00", "^", "-"),
    "auto_pgd": ("#009E73", "D", "--"),
    "apgd_updated": ("#CC79A7", "v", "-"),
    "auto_pgd_plus": ("#000000", "P", "--"),
    "segpgd": ("#E69F00", "X", ":"),
    "cospgd": ("#56B4E9", "*", ":"),
    "dag": ("#D55E00", "h", "-."),
    "sea": ("#009E73", "8", "-."),
    "blade1": ("#CC79A7", "<", (0, (3, 1, 1, 1))),
    "blade": ("#0072B2", ">", (0, (3, 1, 1, 1))),
    "blade_mm": ("#000000", "d", (0, (3, 1, 1, 1))),
    "auto_pgd_r3": ("#009E73", "D", (0, (5, 2))),
    "blade_boundary": ("#D55E00", "v", (0, (3, 1, 1, 1))),
    "blade_frontier": ("#E69F00", "^", (0, (3, 1, 1, 1))),
}
METRIC_LABELS = {
    "dice": "Dice",
    "hd95": "HD95 (mm)",
    "asd": "ASD (mm)",
}
MODEL_CONFIG_DIR = REPO_ROOT / "configs" / "models"
NNUNET_RESULTS = REPO_ROOT / "nnUnet_paths" / "nnUNet_results"

# Architectures the campaign can attack. The key is the short name used on the
# command line and in the output tree; the value is the nnU-Net results trainer
# directory. "unet" is the default the published campaigns ran on.
TRAINERS = {
    "unet": "nnUNetTrainer__nnUNetPlans__3d_fullres",
    "resenc_m": "nnUNetTrainer__nnUNetResEncUNetMPlans__3d_fullres",
    "resenc_l": "nnUNetTrainer__nnUNetResEncUNetLPlans__3d_fullres",
    "resenc_xl": "nnUNetTrainer__nnUNetResEncUNetXLPlans__3d_fullres",
}


def model_dataset_folder(model: str) -> str:
    """Dataset directory for a model, from its shipped config."""
    with open(MODEL_CONFIG_DIR / f"{model}.json") as handle:
        return json.load(handle)["dataset_folder"]


def checkpoint_for(model: str, arch: str, fold: int) -> Path:
    """Checkpoint the driver must attack for one architecture and fold."""
    return (
        NNUNET_RESULTS
        / model_dataset_folder(model)
        / TRAINERS[arch]
        / f"fold_{fold}"
        / "checkpoint_final.pth"
    )


def model_classes(model: str) -> tuple[str, ...]:
    """Foreground class names for a model, from its shipped config.

    Read rather than hardcoded so the campaign cannot disagree with the config
    the driver itself loads. ``wg`` declares no ``class_names`` because it has
    one unnamed foreground class, which the per-sample CSVs label ``WG``.
    """
    with open(MODEL_CONFIG_DIR / f"{model}.json") as handle:
        config = json.load(handle)
    return tuple(config.get("class_names") or ("WG",))


FOLDS = (0, 1, 2, 3, 4)
EPSILONS = ("0", "0.02", "0.04", "0.06", "0.08", "0.1")
METRICS = ("dice", "hd95", "asd")
N_STEPS = 20
NUM_RESTARTS = 1
SEED = 42
# Two runs of the same arm on the same case agree to about 4e-4 mean Dice, so a
# duplicate pair differing by more than this is mixed configurations, not noise.
DUPLICATE_TOL = 0.02


def _arm_dir(output_root: Path, arm: str) -> Path:
    return output_root / arm


def arm_commands(
    arm: str,
    output_root: Path,
    max_samples: int | None,
    resume: bool,
    shared_trajectory: bool = True,
    model: str = "wg",
    arch: str = "unet",
    attack_valid_only: bool = False,
) -> list[list[str]]:
    """Driver invocations for one attack over every fold, writing into its own dir.

    With ``shared_trajectory`` (the default) each iterative arm runs once at the
    largest epsilon and the smaller budgets are L-infinity projections of that
    trajectory. With it off the driver attacks each epsilon natively, which is
    about five times the work but is the only way to compare arms whose step
    size is tied to epsilon -- Auto-PGD fixes its step at ``2*eps``, so a
    projected row understates it. FGSM is single-step and therefore native
    either way.
    """
    out = _arm_dir(output_root, arm)
    out.mkdir(parents=True, exist_ok=True)

    epsilon_flag = f"--epsilons-{model}"
    if arm == "fgsm":
        # The FGSM driver sweeps every fold in one process.
        commands = [
            [
                sys.executable,
                str(FGSM_DRIVER),
                "--model",
                model,
                "--all-folds",
                "--folds",
                *[str(f) for f in FOLDS],
                epsilon_flag,
                *EPSILONS,
                "--output-dir",
                str(out),
            ]
        ]
        if arch != "unet":
            # The FGSM driver resolves a checkpoint per fold, so it takes the
            # trainer directory rather than one checkpoint path.
            commands[0].extend(["--trainer", TRAINERS[arch]])
    else:
        commands = []
        for fold in FOLDS:
            command = [
                sys.executable,
                str(PGD_DRIVER),
                "--attack",
                ARM_ATTACK.get(arm, arm),
                "--model",
                model,
                "--fold",
                str(fold),
                "--pgd-steps",
                str(N_STEPS),
                "--num-restarts",
                str(ARM_RESTARTS.get(arm, NUM_RESTARTS)),
                "--seed",
                str(SEED),
                epsilon_flag,
                *EPSILONS,
                "--output-dir",
                str(out),
            ]
            if arch != "unet":
                # The PGD driver takes one checkpoint per invocation, and it is
                # invoked once per fold, so the fold's own checkpoint goes here.
                command.extend(
                    ["--checkpoint", str(checkpoint_for(model, arch, fold))]
                )
            if shared_trajectory or arm in LADDER_ARMS:
                # A ladder arm's snapshots ARE its rungs: it needs the whole
                # grid in one call to warm-start each epsilon from the one
                # below. Under --native-per-eps it therefore keeps this flag
                # while every other arm loses it -- and at that point the
                # budgets match, because the others are being run once per
                # epsilon too.
                command.append("--shared-trajectory")
            if arm == "pgd":
                # configs/attacks/pgd.json sets random_start; the driver only
                # auto-enables it for a_pgd and auto_pgd.
                command.append("--random-start")
            if resume:
                command.append("--resume")
            commands.append(command)

    if max_samples is not None:
        for command in commands:
            command.extend(["--max-samples", str(max_samples)])
    if attack_valid_only:
        # Both drivers take the same flag; the PGD driver records it in the
        # per-fold run_config JSON, which is how a confined tree is told apart.
        for command in commands:
            command.append("--attack-valid-only")

    return commands


def _run_one(command: list[str], arm: str) -> None:
    started = time.time()
    print(f"\n>>> {' '.join(command[1:])}", flush=True)
    result = subprocess.run(command, cwd=REPO_ROOT)
    if result.returncode != 0:
        raise SystemExit(
            f"{arm} failed with exit code {result.returncode}; fix the cause and "
            "re-run with --resume rather than aggregating a partial arm."
        )
    print(f"<<< {arm} step done in {time.time() - started:.0f}s", flush=True)


def run_all(
    arms: list[str], output_root: Path, max_samples: int | None,
    resume: bool, jobs: int, shared_trajectory: bool = True,
    model: str = "wg", arch: str = "unet", attack_valid_only: bool = False,
) -> None:
    """Run every (arm, fold) job, optionally several at once on the same GPU.

    Sequential runs leave the GPU badly underused -- a single 3D case at batch
    size 1 measured about 15% utilisation -- so the wall-clock win here comes
    from overlapping independent jobs, not from faster kernels. Every kernel-level
    option was measured and rejected: cuDNN TF32 is already on by default, matmul
    TF32 gives x0.99 while shifting mean Dice by 0.0146, cudnn.benchmark gives
    x1.00, channels_last_3d costs 32%, and torch.compile costs 2.4x because 20
    short attack steps never amortise the compile.

    Jobs are independent processes writing to per-arm directories, so this changes
    scheduling only, never numerics. Each holds its own copy of the network, so
    raise ``jobs`` only as far as GPU memory allows.
    """
    jobs_list = [
        (command, arm)
        for arm in arms
        for command in arm_commands(
            arm, output_root, max_samples, resume, shared_trajectory, model, arch,
            attack_valid_only,
        )
    ]
    if jobs <= 1:
        for command, arm in jobs_list:
            _run_one(command, arm)
        return

    print(f"Running {len(jobs_list)} jobs, {jobs} at a time.", flush=True)
    with ThreadPoolExecutor(max_workers=jobs) as pool:
        futures = {pool.submit(_run_one, c, a): (c, a) for c, a in jobs_list}
        for future in as_completed(futures):
            future.result()


def load_arm(output_root: Path, arm: str) -> pd.DataFrame:
    """Per-case rows for one arm across folds, deduplicated, with a fold column.

    Both drivers emit the same ``case_id,epsilon,class,dice,hd95,asd`` per-sample
    schema, so the arms are read through one path regardless of which driver
    produced them. A resumed run can append a case twice; identical duplicates
    are dropped and a disagreement beyond DUPLICATE_TOL is an error, because it
    means the file mixes runs made under different settings.
    """
    frames = []
    for fold in FOLDS:
        matches = sorted(
            _arm_dir(output_root, arm).glob(f"*_fold{fold}_per_sample.csv")
        )
        if not matches:
            raise FileNotFoundError(
                f"no per-sample CSV for {arm} fold {fold} under "
                f"{_arm_dir(output_root, arm)}; run that arm before aggregating."
            )
        frame = pd.read_csv(matches[0])
        # Keyed on class as well: a multi-class model writes one row per
        # foreground class per (case, epsilon), and those rows legitimately
        # disagree. Without the class in the key the zone rows read as
        # duplicates -- the spread check would fire on the TZ+CZ/PZ Dice
        # difference and the dedup would silently discard PZ.
        key = ["case_id", "epsilon", "class"]
        spread = frame.groupby(key)["dice"].agg(lambda s: s.max() - s.min())
        if len(spread) and float(spread.max()) > DUPLICATE_TOL:
            raise ValueError(
                f"{matches[0].name}: duplicate rows for {spread.idxmax()} disagree "
                f"by {float(spread.max()):.4f} Dice, above {DUPLICATE_TOL}; the "
                "file mixes different runs."
            )
        frame = frame.drop_duplicates(subset=key, keep="first")
        frame["fold"] = fold
        frames.append(frame)
    return pd.concat(frames, ignore_index=True)


def fold_summary(per_case: pd.DataFrame, metric: str) -> pd.DataFrame:
    """Average across folds per epsilon, with the fold-level SEM.

    Each fold contributes one mean and the five fold means are averaged, so the
    uncertainty is the spread across folds divided by sqrt(n_folds) -- the
    convention the published table uses -- not the much smaller spread across
    cases. HD95 and ASD are undefined for an empty prediction, so non-finite
    values are dropped per fold rather than poisoning the fold's mean.
    """
    rows = []
    for epsilon, group in per_case.groupby("epsilon"):
        fold_means = []
        for _, fold_group in group.groupby("fold"):
            values = fold_group[metric].to_numpy(dtype=float)
            values = values[np.isfinite(values)]
            if len(values):
                fold_means.append(values.mean())
        fold_means = np.asarray(fold_means, dtype=float)
        n = len(fold_means)
        rows.append(
            {
                "epsilon": float(epsilon),
                "mean": float(fold_means.mean()) if n else np.nan,
                "sd_folds": float(fold_means.std(ddof=1)) if n > 1 else np.nan,
                "sem": float(fold_means.std(ddof=1) / np.sqrt(n)) if n > 1 else np.nan,
                "n_folds": n,
            }
        )
    return pd.DataFrame(rows).sort_values("epsilon").reset_index(drop=True)


def aggregate(
    output_root: Path,
    arms: tuple[str, ...] | list[str] = ARMS,
    classes: tuple[str, ...] = ("WG",),
) -> pd.DataFrame:
    """Summarize the requested arms, one block per foreground class.

    Only the requested arms are read, so a subset run aggregates cleanly
    instead of failing on an arm that was never run. The ``class`` column is
    always present, carrying the single name ``WG`` for the binary model, so
    one schema serves both models.
    """
    tables = []
    for arm in arms:
        per_case = load_arm(output_root, arm)
        for class_name in classes:
            rows = per_case[per_case["class"] == class_name]
            if rows.empty:
                raise ValueError(
                    f"{arm}: no rows for class {class_name!r}; the per-sample "
                    f"CSVs carry {sorted(per_case['class'].unique())}."
                )
            for metric in METRICS:
                table = fold_summary(rows, metric)
                table.insert(0, "metric", metric)
                table.insert(0, "class", class_name)
                table.insert(0, "attack", arm)
                tables.append(table)
    return pd.concat(tables, ignore_index=True)


def print_report(
    summary: pd.DataFrame,
    arms: tuple[str, ...] | list[str] = ARMS,
    class_name: str | None = None,
) -> None:
    if class_name is not None:
        summary = summary[summary["class"] == class_name]
    for metric in METRICS:
        suffix = f" -- {class_name}" if class_name else ""
        print(
            f"\n=== {metric.upper()}{suffix} "
            f"(mean +/- SEM across {len(FOLDS)} folds) ==="
        )
        header = f"{'epsilon':>8}" + "".join(f"{ARM_LABELS[a]:>22}" for a in arms)
        print(header)
        for epsilon in sorted(summary["epsilon"].unique()):
            cells = [f"{epsilon:>8.2f}"]
            for arm in arms:
                row = summary[
                    (summary["attack"] == arm)
                    & (summary["metric"] == metric)
                    & np.isclose(summary["epsilon"], epsilon)
                ]
                if row.empty or not np.isfinite(row["mean"].iloc[0]):
                    cells.append(f"{'n/a':>22}")
                else:
                    cells.append(
                        f"{row['mean'].iloc[0]:>15.4f} +/-{row['sem'].iloc[0]:.4f}"
                    )
            print("".join(cells))


def plot_summary(
    summary: pd.DataFrame,
    target: Path,
    arms: tuple[str, ...] | list[str] = ARMS,
    title_subject: str = "Whole-gland",
    class_name: str | None = None,
) -> None:
    """One subplot per metric: metric against epsilon, one line per attack.

    Error bars are the fold-level SEM, the same quantity the tables report, so
    the figure and the tables cannot disagree. Dice, HD95 and ASD differ by two
    orders of magnitude, which is why they get separate axes rather than a shared
    one; HD95 and ASD are drawn on a log scale because an attack that erases the
    prediction sends them to the volume diagonal while a weak attack leaves them
    near zero.
    """
    if class_name is not None:
        summary = summary[summary["class"] == class_name]
    fig, axes = plt.subplots(1, 3, figsize=(15, 4.4), layout="constrained")
    for ax, metric in zip(axes, METRICS, strict=True):
        for arm in arms:
            rows = summary[
                (summary["attack"] == arm) & (summary["metric"] == metric)
            ].sort_values("epsilon")
            if rows.empty:
                continue
            colour, marker, linestyle = ARM_STYLE[arm]
            ax.errorbar(
                rows["epsilon"], rows["mean"], yerr=rows["sem"],
                color=colour, marker=marker, linestyle=linestyle,
                markersize=5, capsize=3, linewidth=1.6, label=ARM_LABELS[arm],
            )
        ax.set_xlabel(r"$\varepsilon$ ($L_\infty$, after z-score normalization)")
        ax.set_ylabel(METRIC_LABELS[metric])
        ax.set_title(METRIC_LABELS[metric], fontsize=12, fontweight="bold")
        ax.grid(alpha=0.3)
        if metric != "dice":
            ax.set_yscale("log")
    axes[0].set_ylim(-0.02, 1.0)
    axes[0].legend(fontsize=8, loc="lower left", ncol=2)
    plotted = _count_word(len(arms))
    fig.suptitle(
        f"{title_subject} nnU-Net under {plotted} adversarial attacks "
        f"(mean +/- SEM across {len(FOLDS)} folds)",
        fontsize=13, fontweight="bold",
    )
    for suffix in (".png", ".pdf"):
        fig.savefig(target.with_suffix(suffix), dpi=600, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved {target.with_suffix('.png')} / .pdf")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument(
        "--model",
        choices=("wg", "zones"),
        default="wg",
        help="Segmentation model to attack. 'zones' is the three-class model "
        "(background, TZ+CZ, PZ) and produces one summary block and one "
        "figure per foreground class.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=None,
        help="Root for per-arm result directories and the summary CSV. "
        "Defaults to results/blade_attack/whole_gland/shared_trajectory for "
        "--model wg and results/blade_attack/zones/campaign for --model zones.",
    )
    parser.add_argument(
        "--arms",
        nargs="+",
        choices=ARMS,
        default=list(ARMS),
        help=f"Subset of arms to run (default: all {ARM_COUNT}).",
    )
    parser.add_argument(
        "--figure-arms",
        nargs="+",
        choices=ARMS,
        default=None,
        help="Plot only these arms, into a figure named for how many were "
        "drawn (e.g. nine_arm_wg_metrics.png). The summary CSV still covers "
        "every aggregated arm, so restricting the figure never restricts the "
        "table behind it. Default: whatever --arms aggregated.",
    )
    parser.add_argument(
        "--figure-name",
        type=str,
        default=None,
        help="Base name for the figure, without extension. Defaults to the "
        "spelled-out count of the arms drawn, which collides when two "
        "different subsets happen to have the same size.",
    )
    parser.add_argument(
        "--aggregate-only",
        action="store_true",
        help="Skip the attacks and summarize existing per-arm CSVs.",
    )
    parser.add_argument(
        "--max-samples",
        type=int,
        default=None,
        help="Cases per fold; use a small value to smoke-test the wiring.",
    )
    parser.add_argument(
        "--jobs",
        type=int,
        default=1,
        help="Driver processes to run concurrently on the one GPU. The workload "
        "leaves the GPU about 85%% idle, so 3-4 cuts wall-clock roughly "
        "proportionally. Scheduling only; results are unchanged.",
    )
    parser.add_argument(
        "--resume",
        action="store_true",
        help="Skip cases already present in an arm's per-sample CSV.",
    )
    parser.add_argument(
        "--arch",
        choices=sorted(TRAINERS),
        default="unet",
        help="Trained architecture to attack (default: unet, the one the "
        "published campaigns used). Anything else routes the drivers at that "
        "trainer's checkpoints and, unless --output-dir says otherwise, into a "
        "sibling output tree suffixed with the architecture.",
    )
    parser.add_argument(
        "--native-per-eps",
        action="store_true",
        help="Attack every epsilon natively instead of projecting one "
        "trajectory run at the largest epsilon. About five times the work, "
        "and the only protocol under which arms whose step size is tied to "
        "epsilon (the Auto-PGD family, step 2*eps) are represented fairly.",
    )
    parser.add_argument(
        "--attack-valid-only",
        action="store_true",
        help="Pass --attack-valid-only to both drivers: confine every arm's "
        "perturbation to voxels with a valid label inside the unpadded volume. "
        "Required for a faithful zones campaign (its inputs are zero-filled "
        "outside the dilated gland and zero-padded, neither of which survives "
        "the deployed pipeline); a no-op in effect for wg. Default off "
        "reproduces the published trees.",
    )
    args = parser.parse_args()

    if args.output_dir is None:
        args.output_dir = (
            REPO_ROOT / "results" / "blade_attack" / DEFAULT_OUTPUT_SUBDIR[args.model]
        )
        if args.arch != "unet":
            # Per-arm output filenames carry no architecture, so a non-default
            # architecture must never share a tree with the published unet
            # campaign it would otherwise overwrite.
            args.output_dir = args.output_dir.parent / f"{args.output_dir.name}_{args.arch}"
    args.output_dir.mkdir(parents=True, exist_ok=True)
    classes = model_classes(args.model)

    if not args.aggregate_only:
        run_all(
            args.arms, args.output_dir, args.max_samples, args.resume, args.jobs,
            shared_trajectory=not args.native_per_eps, model=args.model,
            arch=args.arch, attack_valid_only=args.attack_valid_only,
        )

    summary = aggregate(args.output_dir, args.arms, classes)
    # Named for the arms actually aggregated, not for len(ARMS): a subset run
    # must not write a file whose name claims the full campaign.
    target = (
        args.output_dir
        / f"{_count_word(len(args.arms))}_arm_{args.model}_summary.csv"
    )
    summary.to_csv(target, index=False)

    figure_arms = args.figure_arms or args.arms
    missing = [a for a in figure_arms if a not in set(args.arms)]
    if missing:
        raise SystemExit(
            f"--figure-arms names {missing}, which --arms did not aggregate; "
            "add them to --arms or drop them from the figure."
        )
    stem = (
        args.figure_name
        or f"{_count_word(len(figure_arms))}_arm_{args.model}_metrics"
    )
    # One report block and one figure per foreground class. A multi-class model
    # has no single curve to draw: TZ+CZ and PZ fail differently, and averaging
    # them would hide exactly the asymmetry a zonal evaluation exists to find.
    for class_name in classes:
        print_report(summary, args.arms, class_name)
        suffix = "" if len(classes) == 1 else f"_{class_name.replace('+', '')}"
        plot_summary(
            summary,
            args.output_dir / f"{stem}{suffix}",
            figure_arms,
            title_subject="Whole-gland" if args.model == "wg" else class_name,
            class_name=class_name,
        )
    print(f"\nSaved {target}")


if __name__ == "__main__":
    main()
