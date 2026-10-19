"""Supplementary evidence tables for ``paper_resenc/results_evidence.md``.

Computes numbers the tracked ``analyze_attack_architecture_stats.py`` does
not itself produce: clean (epsilon=0) Dice, the raw budget-averaged
per-arm/architecture/class table, the WG encoder-vs-batch decomposition, the
per-epsilon Auto-PGD/SEA view (with a bootstrap CI), a batch-step
significance table across all three classes, model parameter counts, the
per-fold training schedule, the per-attack gradient budget, a per-fold
repeat of the encoder/batch/total decomposition (reviewer Major 2), and the
forced-foreground patch count implied by each checkpoint's oversampling
setting (reviewer Major 1). Reuses the same helper functions the tracked,
verified statistics pipeline uses (``load_per_sample``, ``mean_over_epsilons``,
``build_matrix``, ``_hodges_lehmann``, ``_fdr_bh`` from
``analyze_main_comparison_stats.py``), so the methodology matches.

Inputs: the per-sample CSVs under the four architectures' WG/zones trees
(``results/blade_attack/{whole_gland,zones_confined_mc}/...``), plus
``cross_arch_wilcoxon_WG.csv`` from a prior ``analyze_attack_architecture_stats.py``
run in ``--output-dir`` (for the encoder/total Hodges-Lehmann reference
points used by the WG decomposition), plus the checkpoints, ``debug.json``
and ``plans.json`` under ``nnUnet_paths/nnUNet_results`` (for the
parameter-count, training-schedule and foreground-oversampling tables).

Outputs, written to ``--output-dir``:

- ``evidence_clean_dice.csv``
- ``evidence_budget_avg.csv``
- ``evidence_wg_batch_decomposition.csv``
- ``evidence_per_epsilon_apgd_sea.csv``
- ``evidence_per_epsilon_ci.csv`` (Gap G9)
- ``evidence_batch_step_significance.csv`` (Gaps G4, G10)
- ``evidence_model_parameter_counts.csv`` (Gap G2; requires nnunetv2)
- ``evidence_training_schedule.csv`` (Gap G3)
- ``evidence_gradient_budgets.csv`` (Gap G5)
- ``evidence_per_fold_shifts.csv`` (reviewer Major 2)
- ``evidence_foreground_oversampling.csv`` (reviewer Major 1)

Usage::

    PYTHONPATH=src python experiments/analyze_resenc_paper_evidence.py \\
        --arms fgsm pgd segpgd cospgd dag auto_pgd auto_pgd_r3 sea \\
        --output-dir results/blade_attack/attack_comparison/stats_no_blade
"""

from __future__ import annotations

import argparse
import glob
import json
import os
import sys
from typing import cast

import numpy as np
import pandas as pd
import torch
from scipy import stats

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from analyze_main_comparison_stats import (  # noqa: E402
    _fdr_bh,
    _hodges_lehmann,
    build_matrix,
    load_per_sample,
    mean_over_epsilons,
)
from analyze_attack_architecture_stats import ARCHS, RESULTS, TREES  # noqa: E402
from run_twelve_arm_wg_campaign import (  # noqa: E402
    NNUNET_RESULTS,
    TRAINERS,
    checkpoint_for,
    model_dataset_folder,
)

ARMS8 = ["fgsm", "pgd", "segpgd", "cospgd", "dag", "auto_pgd", "auto_pgd_r3", "sea"]
EPSILONS = [0.02, 0.04, 0.06, 0.08, 0.1]
METRICS = ("dice", "hd95", "asd")
# Every arm's per-sample data is five independently trained fold models, each
# validation case belonging to exactly one fold (verified: the five per-fold
# case_id sets for a tree are pairwise disjoint and union to its full cohort).
FOLDS = (0, 1, 2, 3, 4)
# The two datasets underlying the paper's 3 evaluation classes: "zones" trains
# one multi-class network that is scored separately on TZ+CZ and PZ, so model
# parameter counts and training schedules are per dataset, not per class.
MODEL_KEYS = ("wg", "zones")
MODEL_DATASET_LABEL = {"wg": "WG", "zones": "TZ+CZ/PZ (zones)"}

Frames = dict[str, dict[str, dict[str, pd.DataFrame]]]


def load_frames(arms: list[str]) -> Frames:
    """Per-sample frames for every (class, architecture, arm) in ``TREES``."""
    frames: Frames = {}
    for class_name, class_trees in TREES.items():
        trees = class_trees[0]
        frames[class_name] = {
            arch: {arm: load_per_sample(os.path.join(RESULTS, t), arm) for arm in arms}
            for arch, t in trees.items()
        }
    return frames


def load_per_sample_fold(campaign_dir: str, arm: str, fold: int) -> pd.DataFrame:
    """One arm's per-sample CSV for a single fold (its own, disjoint cases).

    Same directory convention as ``load_per_sample`` (``wg_blade_fold0`` and
    ``zones_blade1_fold0`` both match), but reads exactly one fold instead of
    concatenating all five, so callers can pair cases within a fold without
    mixing in the other four folds' non-overlapping case IDs.
    """
    pattern = os.path.join(campaign_dir, arm, f"*_fold{fold}_per_sample.csv")
    files = sorted(glob.glob(pattern))
    if len(files) != 1:
        raise FileNotFoundError(
            f"expected exactly one per-sample CSV for arm {arm!r} fold {fold}: "
            f"{pattern} matched {files}"
        )
    frame = pd.read_csv(files[0])
    frame["arm"] = arm
    return frame


def load_frames_by_fold(
    arms: list[str], folds: tuple[int, ...] = FOLDS
) -> dict[int, Frames]:
    """Per-sample frames for every (fold, class, architecture, arm) in ``TREES``."""
    fold_frames: dict[int, Frames] = {}
    for fold in folds:
        frames: Frames = {}
        for class_name, class_trees in TREES.items():
            trees = class_trees[0]
            frames[class_name] = {
                arch: {
                    arm: load_per_sample_fold(os.path.join(RESULTS, t), arm, fold)
                    for arm in arms
                }
                for arch, t in trees.items()
            }
        fold_frames[fold] = frames
    return fold_frames


def clean_dice_table(frames: Frames) -> pd.DataFrame:
    """Clean (epsilon=0) Dice per class x architecture.

    The clean segmentation is shared across every arm in a tree, so any arm's
    epsilon=0 rows give the same numbers; "pgd" is used throughout.
    """
    rows: list[dict[str, object]] = []
    for class_name in TREES:
        for arch in ARCHS:
            df = frames[class_name][arch]["pgd"]
            clean = df[(df["epsilon"] == 0) & (df["class"] == class_name)]
            rows.append(
                {
                    "class_name": class_name,
                    "arch": arch,
                    "n_cases": len(clean),
                    "mean_dice": clean["dice"].mean(),
                    "median_dice": clean["dice"].median(),
                }
            )
    return pd.DataFrame(rows)


def budget_avg_table(frames: Frames, arms: list[str]) -> pd.DataFrame:
    """Budget-averaged mean/median per arm x architecture x class x metric."""
    rows: list[dict[str, object]] = []
    for class_name in TREES:
        for metric in METRICS:
            for arch in ARCHS:
                mat = mean_over_epsilons(
                    frames[class_name][arch], arms, metric, EPSILONS, class_name
                )
                for arm in arms:
                    rows.append(
                        {
                            "class_name": class_name,
                            "metric": metric,
                            "arch": arch,
                            "arm": arm,
                            "n_cases": len(mat),
                            "mean": mat[arm].mean(),
                            "median": mat[arm].median(),
                        }
                    )
    return pd.DataFrame(rows)


def share_of_total(step_hl: float, total_hl: float) -> float:
    """The fraction of a total Hodges-Lehmann shift one step accounts for.

    NaN when there is no total shift to divide into (``total_hl == 0``): the
    ratio is undefined, not zero.
    """
    if total_hl == 0:
        return float("nan")
    return step_hl / total_hl


def batch_encoder_decomposition_row(
    mat: pd.DataFrame, metric: str, arm: str, hl_encoder: float, hl_total: float
) -> dict[str, object]:
    """One row of the WG encoder-vs-batch decomposition.

    ``mat`` is a case x architecture Dice/HD95/ASD matrix restricted to one
    arm's budget-averaged metric (columns at least ``resenc_m``, ``resenc_l``,
    ``resenc_xl``, one row per case, paired by case). ``hl_encoder`` is the
    Hodges-Lehmann shift nnU-Net -> ResEnc-M and ``hl_total`` is nnU-Net ->
    ResEnc-XL, both computed elsewhere (Section 3's cross-architecture
    output); this function computes the two batch-size steps
    (ResEnc-M -> ResEnc-L, ResEnc-M -> ResEnc-XL) directly and paired by case,
    plus each step's share of the total shift.
    """
    diff_l_m = (mat["resenc_l"] - mat["resenc_m"]).to_numpy()
    diff_xl_m = (mat["resenc_xl"] - mat["resenc_m"]).to_numpy()
    hl_l_m, lo_l_m, hi_l_m = _hodges_lehmann(diff_l_m)
    hl_xl_m, lo_xl_m, hi_xl_m = _hodges_lehmann(diff_xl_m)
    return {
        "metric": metric,
        "arm": arm,
        "n_cases": len(mat),
        "hl_encoder_unet_to_m": hl_encoder,
        "hl_batch_l_minus_m": hl_l_m,
        "hl_batch_l_minus_m_ci_lo": lo_l_m,
        "hl_batch_l_minus_m_ci_hi": hi_l_m,
        "hl_batch_xl_minus_m": hl_xl_m,
        "hl_batch_xl_minus_m_ci_lo": lo_xl_m,
        "hl_batch_xl_minus_m_ci_hi": hi_xl_m,
        "hl_total_unet_to_xl": hl_total,
        "share_encoder_of_total": share_of_total(hl_encoder, hl_total),
        "share_batch_xl_of_total": share_of_total(hl_xl_m, hl_total),
    }


def wg_batch_decomposition_table(
    frames: Frames, arms: list[str], cross_arch_wilcoxon_wg: pd.DataFrame
) -> pd.DataFrame:
    """WG-only: encoder effect (nnU-Net -> ResEnc-M) vs. batch effect
    (ResEnc-M -> ResEnc-L / ResEnc-M -> ResEnc-XL), per arm and metric.

    ``cross_arch_wilcoxon_wg`` is ``cross_arch_wilcoxon_WG.csv`` from a prior
    ``analyze_attack_architecture_stats.py`` run over the same ``arms``, the
    source of the encoder (comparator=resenc_m) and total
    (comparator=resenc_xl) Hodges-Lehmann shifts against nnU-Net.
    """
    cross_budget = cross_arch_wilcoxon_wg[
        cross_arch_wilcoxon_wg["epsilon"] == "budget_avg"
    ]
    rows: list[dict[str, object]] = []
    for metric in METRICS:
        for arm in arms:
            mat_cols = {}
            for arch in ARCHS:
                one = {arm: frames["WG"][arch][arm]}
                m = mean_over_epsilons(one, [arm], metric, EPSILONS, "WG")
                mat_cols[arch] = m[arm]
            mat = pd.DataFrame(mat_cols).dropna(axis=0, how="any")

            enc_row = cross_budget[
                (cross_budget["arm"] == arm)
                & (cross_budget["metric"] == metric)
                & (cross_budget["comparator"] == "resenc_m")
            ].iloc[0]
            total_row = cross_budget[
                (cross_budget["arm"] == arm)
                & (cross_budget["metric"] == metric)
                & (cross_budget["comparator"] == "resenc_xl")
            ].iloc[0]
            rows.append(
                batch_encoder_decomposition_row(
                    mat,
                    metric,
                    arm,
                    hl_encoder=float(enc_row["hl_shift"]),
                    hl_total=float(total_row["hl_shift"]),
                )
            )
    return pd.DataFrame(rows)


def per_fold_shift_row(
    mat: pd.DataFrame, class_name: str, fold: int, arm: str
) -> dict[str, object]:
    """One fold's encoder/batch/total Dice decomposition for one class and arm.

    Unlike ``wg_batch_decomposition_table``, ``hl_encoder`` (nnU-Net ->
    ResEnc-M) and ``hl_total`` (nnU-Net -> ResEnc-XL) are computed here
    directly from ``mat`` rather than read from a prior
    ``analyze_attack_architecture_stats.py`` run: that script's
    ``cross_arch_wilcoxon_*.csv`` files are full-cohort, not per-fold. ``mat``
    is one fold's case x architecture budget-averaged Dice matrix, paired by
    case within that fold.
    """
    diff_encoder = (mat["resenc_m"] - mat["unet"]).to_numpy()
    diff_total = (mat["resenc_xl"] - mat["unet"]).to_numpy()
    hl_encoder, _, _ = _hodges_lehmann(diff_encoder)
    hl_total, _, _ = _hodges_lehmann(diff_total)
    row = batch_encoder_decomposition_row(
        mat, metric="dice", arm=arm, hl_encoder=hl_encoder, hl_total=hl_total
    )
    return {"class_name": class_name, "fold": fold, **row}


def per_fold_shifts_table(
    fold_frames: dict[int, Frames], arms: list[str]
) -> pd.DataFrame:
    """Per-fold WG/TZ+CZ/PZ encoder/batch/total Dice decomposition (reviewer Major 2).

    Each configuration is five independently trained fold models, and every
    validation case belongs to exactly one fold; this repeats
    ``per_fold_shift_row`` for that fold's own cases only, so the five rows
    per (class, arm) are five independent paired comparisons rather than one
    comparison inflated by pooling non-independent folds.
    """
    rows: list[dict[str, object]] = []
    for fold, frames in fold_frames.items():
        for class_name in TREES:
            for arm in arms:
                mat_cols = {}
                for arch in ARCHS:
                    one = {arm: frames[class_name][arch][arm]}
                    m = mean_over_epsilons(one, [arm], "dice", EPSILONS, class_name)
                    mat_cols[arch] = m[arm]
                mat = pd.DataFrame(mat_cols).dropna(axis=0, how="any")
                rows.append(per_fold_shift_row(mat, class_name, fold, arm))
    df = pd.DataFrame(rows)
    front = ["class_name", "fold", "arm"]
    return df[front + [c for c in df.columns if c not in front]]


def fold_agreement_summary(
    per_fold_df: pd.DataFrame, class_name: str, arm: str
) -> dict[str, object]:
    """Cross-fold min/max/mean/SD and sign/ordering agreement for one (class, arm).

    Summarises ``per_fold_shifts_table``'s five rows for a (class, arm) pair:
    whether each of the four shifts (encoder, the two batch steps, total) has
    the same sign in every fold, and whether the batch step
    (ResEnc-XL minus ResEnc-M) exceeds the encoder step
    (ResEnc-M minus nnU-Net) in every fold -- the two claims Section 1's
    per-fold robustness check needs.
    """
    sub = per_fold_df[
        (per_fold_df["class_name"] == class_name) & (per_fold_df["arm"] == arm)
    ]
    shift_columns = (
        "hl_encoder_unet_to_m",
        "hl_batch_l_minus_m",
        "hl_batch_xl_minus_m",
        "hl_total_unet_to_xl",
        "share_batch_xl_of_total",
    )
    row: dict[str, object] = {"class_name": class_name, "arm": arm, "n_folds": len(sub)}
    for col in shift_columns:
        values = sub[col].to_numpy(dtype=float)
        signs = np.sign(values)
        row[f"{col}_min"] = float(values.min())
        row[f"{col}_max"] = float(values.max())
        row[f"{col}_mean"] = float(values.mean())
        row[f"{col}_sd"] = (
            float(values.std(ddof=1)) if len(values) > 1 else float("nan")
        )
        row[f"{col}_sign_consistent"] = bool(np.all(signs == signs[0]))
    ordering_holds = (
        sub["hl_batch_xl_minus_m"].to_numpy() > sub["hl_encoder_unet_to_m"].to_numpy()
    )
    row["batch_gt_encoder_every_fold"] = bool(np.all(ordering_holds))
    row["batch_gt_encoder_n_folds_holding"] = int(ordering_holds.sum())
    return row


def per_epsilon_table(
    frames: Frames, arms: tuple[str, str] = ("auto_pgd", "sea")
) -> pd.DataFrame:
    """Per-epsilon Dice for the given arms, per architecture, per class."""
    rows: list[dict[str, object]] = []
    for class_name in TREES:
        for arm in arms:
            for arch in ARCHS:
                for eps in EPSILONS:
                    m = build_matrix(
                        {arm: frames[class_name][arch][arm]},
                        "dice",
                        eps,
                        [arm],
                        class_name,
                    )
                    rows.append(
                        {
                            "class_name": class_name,
                            "arm": arm,
                            "arch": arch,
                            "epsilon": eps,
                            "n_cases": len(m),
                            "mean_dice": m[arm].mean(),
                            "median_dice": m[arm].median(),
                        }
                    )
    return pd.DataFrame(rows)


def bootstrap_mean_ci(
    values: np.ndarray, n_resamples: int = 2000, seed: int = 42
) -> tuple[float, float]:
    """95% percentile case-bootstrap CI of the mean (Gap G9).

    Draws ``n_resamples`` samples of ``len(values)`` cases each, with
    replacement, and takes the 2.5th/97.5th percentile of the resulting
    distribution of resample means. Non-finite entries are dropped first, so
    a mix of valid and NaN cases (e.g. an undefined HD95) does not propagate
    NaN into every resample. Returns ``(nan, nan)`` when nothing is finite.
    """
    finite = values[np.isfinite(values)]
    n = len(finite)
    if n == 0:
        return float("nan"), float("nan")
    rng = np.random.default_rng(seed)
    resample_idx = rng.integers(0, n, size=(n_resamples, n))
    resample_means = finite[resample_idx].mean(axis=1)
    lo, hi = np.percentile(resample_means, [2.5, 97.5])
    return float(lo), float(hi)


def per_epsilon_ci_table(
    frames: Frames,
    arms: tuple[str, str] = ("auto_pgd", "sea"),
    n_resamples: int = 2000,
    seed: int = 42,
) -> pd.DataFrame:
    """Per-epsilon mean Dice with a 95% case-bootstrap CI, for Fig. 1 (Gap G9).

    Same (class, arm, arch, epsilon) grid as ``per_epsilon_table``, with a
    bootstrap CI added on the mean. Written to its own file rather than
    folded into ``per_epsilon_table``'s output, so the existing
    ``evidence_per_epsilon_apgd_sea.csv`` is left untouched.
    """
    rows: list[dict[str, object]] = []
    for class_name in TREES:
        for arm in arms:
            for arch in ARCHS:
                for eps in EPSILONS:
                    m = build_matrix(
                        {arm: frames[class_name][arch][arm]},
                        "dice",
                        eps,
                        [arm],
                        class_name,
                    )
                    values = m[arm].to_numpy()
                    ci_lo, ci_hi = bootstrap_mean_ci(values, n_resamples, seed)
                    rows.append(
                        {
                            "class_name": class_name,
                            "arm": arm,
                            "arch": arch,
                            "epsilon": eps,
                            "n_cases": len(m),
                            "mean_dice": float(values.mean()),
                            "ci_lo": ci_lo,
                            "ci_hi": ci_hi,
                        }
                    )
    return pd.DataFrame(rows)


def batch_step_significance_row(
    mat: pd.DataFrame, class_name: str, metric: str, arm: str
) -> dict[str, object]:
    """One row of paired batch-step HL shifts, CIs and BH-adjusted Wilcoxon p.

    ``mat`` is a case x architecture matrix (one arm's budget-averaged
    metric, at least ``resenc_m``, ``resenc_l``, ``resenc_xl`` columns). The
    two batch steps -- ResEnc-L minus ResEnc-M and ResEnc-XL minus ResEnc-M
    -- are the 2-comparison family Benjamini-Hochberg is applied within, the
    same per-case-paired, BH-within-family pattern
    ``analyze_attack_architecture_stats.cross_arch_tests`` uses for the
    vs-nnU-Net contrasts. Closes Gap G4 (TZ+CZ/PZ batch decomposition) and
    Gap G10 (WG batch-step significance, alongside the existing HL/CI-only
    ``evidence_wg_batch_decomposition.csv``).
    """
    diff_l_m = (mat["resenc_l"] - mat["resenc_m"]).to_numpy()
    diff_xl_m = (mat["resenc_xl"] - mat["resenc_m"]).to_numpy()
    hl_l_m, lo_l_m, hi_l_m = _hodges_lehmann(diff_l_m)
    hl_xl_m, lo_xl_m, hi_xl_m = _hodges_lehmann(diff_xl_m)

    p_values = []
    for diff in (diff_l_m, diff_xl_m):
        nz = diff[diff != 0]
        p_values.append(float("nan") if nz.size == 0 else float(stats.wilcoxon(nz)[1]))
    p_adj = _fdr_bh(np.array(p_values))

    return {
        "class_name": class_name,
        "metric": metric,
        "arm": arm,
        "n_cases": len(mat),
        "hl_batch_l_minus_m": hl_l_m,
        "hl_batch_l_minus_m_ci_lo": lo_l_m,
        "hl_batch_l_minus_m_ci_hi": hi_l_m,
        "p_value_batch_l_minus_m": p_values[0],
        "p_adj_batch_l_minus_m": float(p_adj[0]),
        "hl_batch_xl_minus_m": hl_xl_m,
        "hl_batch_xl_minus_m_ci_lo": lo_xl_m,
        "hl_batch_xl_minus_m_ci_hi": hi_xl_m,
        "p_value_batch_xl_minus_m": p_values[1],
        "p_adj_batch_xl_minus_m": float(p_adj[1]),
    }


def batch_step_significance_table(frames: Frames, arms: list[str]) -> pd.DataFrame:
    """Batch-step HL shifts + BH-adjusted Wilcoxon p, for every class, metric, arm."""
    rows: list[dict[str, object]] = []
    for class_name in TREES:
        for metric in METRICS:
            for arm in arms:
                mat_cols = {}
                for arch in ARCHS:
                    one = {arm: frames[class_name][arch][arm]}
                    m = mean_over_epsilons(one, [arm], metric, EPSILONS, class_name)
                    mat_cols[arch] = m[arm]
                mat = pd.DataFrame(mat_cols).dropna(axis=0, how="any")
                rows.append(batch_step_significance_row(mat, class_name, metric, arm))
    return pd.DataFrame(rows)


def model_parameter_counts_table() -> pd.DataFrame:
    """Trainable parameter count per dataset x architecture, from plans.json (Gap G2).

    Instantiates each network on CPU via nnunetv2's own
    ``get_network_from_plans`` (construction only, no forward/backward pass,
    no GPU), so the count matches what actually trained. Confirms
    ResEnc-M/L/XL are the same network on both datasets (see
    ``paper_resenc/results_evidence.md`` Section 0). Requires ``nnunetv2`` to
    be importable, hence the lazy import: nothing else in this module needs
    it.
    """
    from nnunetv2.utilities.get_network_from_plans import get_network_from_plans
    from nnunetv2.utilities.plans_handling.plans_handler import PlansManager

    rows: list[dict[str, object]] = []
    for model_key in MODEL_KEYS:
        dataset_dir = os.path.join(NNUNET_RESULTS, model_dataset_folder(model_key))
        with open(os.path.join(dataset_dir, TRAINERS["unet"], "dataset.json")) as f:
            dataset_json = json.load(f)
        for arch in ARCHS:
            plans_path = os.path.join(dataset_dir, TRAINERS[arch], "plans.json")
            with open(plans_path) as f:
                plans = json.load(f)
            plans_manager = PlansManager(plans)
            config = plans_manager.get_configuration("3d_fullres")
            label_manager = plans_manager.get_label_manager(dataset_json)
            network = get_network_from_plans(
                config.network_arch_class_name,
                config.network_arch_init_kwargs,
                config.network_arch_init_kwargs_req_import,
                len(dataset_json["channel_names"]),
                label_manager.num_segmentation_heads,
                allow_init=True,
                deep_supervision=True,
            )
            n_params = sum(p.numel() for p in network.parameters() if p.requires_grad)
            rows.append(
                {
                    "dataset": MODEL_DATASET_LABEL[model_key],
                    "arch": arch,
                    "network_class": config.network_arch_class_name.rsplit(".", 1)[-1],
                    "n_trainable_params": int(n_params),
                }
            )
    return pd.DataFrame(rows)


# nnUNetTrainer.__init__'s class default (nnunetv2 2.6.4,
# nnunetv2/training/nnUNetTrainer/nnUNetTrainer.py): "self.num_iterations_per_epoch
# = 250". Not itself stored in a checkpoint; applicable here because every
# checkpoint's ``trainer_name`` is the unmodified ``nnUNetTrainer`` (verified
# per fold in ``training_schedule_table``, not assumed), so no custom trainer
# subclass could have overridden it.
NUM_ITERATIONS_PER_EPOCH_DEFAULT = 250


def training_schedule_table(folds: tuple[int, ...] = (0, 1, 2, 3, 4)) -> pd.DataFrame:
    """Per-fold training schedule read from each checkpoint (Gap G3).

    No usable epoch-by-epoch ``training_log_*.txt`` or ``debug.json`` exists
    under these checkpoint directories (the on-disk logs are restart headers
    only, no per-epoch lines); the ground truth used here is each
    checkpoint's own embedded ``logging``/``optimizer_state`` state, written
    once at the end of training. ``num_epochs`` is the checkpoint's
    ``current_epoch`` (the actual number of epochs trained, not an assumed
    default); ``initial_lr`` is the first entry of the checkpoint's own
    logged learning-rate curve, not the optimizer's final (decayed) rate.
    """
    rows: list[dict[str, object]] = []
    for model_key in MODEL_KEYS:
        for arch in ARCHS:
            for fold in folds:
                path = checkpoint_for(model_key, arch, fold)
                ckpt = torch.load(str(path), map_location="cpu", weights_only=False)
                lrs = ckpt["logging"]["lrs"]
                param_group = ckpt["optimizer_state"]["param_groups"][0]
                rows.append(
                    {
                        "dataset": MODEL_DATASET_LABEL[model_key],
                        "arch": arch,
                        "fold": fold,
                        "trainer_name": ckpt["trainer_name"],
                        "num_epochs": int(ckpt["current_epoch"]),
                        "num_epochs_logged": len(lrs),
                        "initial_lr": float(lrs[0]),
                        "optimizer": "SGD",
                        "momentum": param_group.get("momentum"),
                        "nesterov": param_group.get("nesterov"),
                        "weight_decay": param_group.get("weight_decay"),
                        "num_iterations_per_epoch": NUM_ITERATIONS_PER_EPOCH_DEFAULT,
                    }
                )
    return pd.DataFrame(rows)


# nnUNetTrainer.__init__'s class default (reviewer Major 1). Verified identical
# in the nnunetv2 git history at both v2.2.1 (base_data_loader.py:144, the
# pre-refactor file) and v2.6.4 (nnUNetTrainer.py:147), the version range
# spanning this repo's documented pin (CLAUDE.md: nnunetv2==2.2.1) and its
# requirements.txt pin (2.6.4); used as the "unet" trainer's value below,
# whose checkpoints carry no debug.json to read it from directly.
OVERSAMPLE_FOREGROUND_PERCENT_DEFAULT = 0.33


def forced_foreground_count(
    batch_size: int, oversample_foreground_percent: float
) -> int:
    """Patches per batch nnU-Net's dataloader forces to contain foreground.

    The exact rule (installed nnunetv2 2.6.4,
    ``nnunetv2/training/dataloading/data_loader.py:73``,
    ``nnUNetDataLoader._oversample_last_XX_percent``, called once per batch
    slot ``j`` at ``data_loader.py:176``): slot ``j`` is forced when
    ``not j < round(batch_size * (1 - oversample_foreground_percent))``, i.e.
    the last ``batch_size - round(batch_size * (1 - p))`` slots of the batch.
    Identical in the nnunetv2 git history at v2.2.1
    (``base_data_loader.py:49``, pre-2.4 file layout) and v2.6.4, and
    ``probabilistic_oversampling`` (the alternative, non-deterministic rule)
    defaults to ``False`` in both, so this deterministic count applies.
    """
    not_forced = round(batch_size * (1 - oversample_foreground_percent))
    return batch_size - not_forced


def _debug_json_path(model_key: str, arch: str, fold: int) -> str:
    dataset_dir = os.path.join(NNUNET_RESULTS, model_dataset_folder(model_key))
    return os.path.join(dataset_dir, TRAINERS[arch], f"fold_{fold}", "debug.json")


def _read_plans_batch_size(model_key: str, arch: str) -> int:
    dataset_dir = os.path.join(NNUNET_RESULTS, model_dataset_folder(model_key))
    plans_path = os.path.join(dataset_dir, TRAINERS[arch], "plans.json")
    with open(plans_path) as f:
        plans = json.load(f)
    return int(plans["configurations"]["3d_fullres"]["batch_size"])


def _read_debug_oversample(
    model_key: str, arch: str, folds: tuple[int, ...]
) -> tuple[float, int] | None:
    """A checkpoint's own ``(oversample_foreground_percent, batch_size)``.

    Verified identical across every fold that has a ``debug.json`` (all folds
    do for resenc_m/l/xl; none do for "unet" in this repo, hence the
    ``None`` return the caller falls back on).
    """
    values: list[tuple[float, int]] = []
    for fold in folds:
        path = _debug_json_path(model_key, arch, fold)
        if not os.path.exists(path):
            continue
        with open(path) as f:
            debug = json.load(f)
        values.append(
            (float(debug["oversample_foreground_percent"]), int(debug["batch_size"]))
        )
    if not values:
        return None
    if any(v != values[0] for v in values):
        raise ValueError(
            f"debug.json disagrees across folds for {model_key}/{arch}: {values}"
        )
    return values[0]


def foreground_oversampling_table(folds: tuple[int, ...] = FOLDS) -> pd.DataFrame:
    """Forced-foreground patches per batch, per dataset x architecture (reviewer Major 1).

    ``oversample_foreground_percent`` and ``batch_size`` come from each
    checkpoint's own ``debug.json`` where one exists (resenc_m/l/xl); the
    "unet" trainer's checkpoints carry no ``debug.json`` in this repo, so its
    oversample percent falls back to the nnUNetTrainer class default
    (``OVERSAMPLE_FOREGROUND_PERCENT_DEFAULT``) while its batch size still
    comes from its own ``plans.json`` (not a default: it is the actually
    configured batch size, and it does differ between datasets: 2 for WG, 13
    for zones).
    """
    rows: list[dict[str, object]] = []
    for model_key in MODEL_KEYS:
        for arch in ARCHS:
            batch_size = _read_plans_batch_size(model_key, arch)
            debug = _read_debug_oversample(model_key, arch, folds)
            if debug is None:
                oversample = OVERSAMPLE_FOREGROUND_PERCENT_DEFAULT
                source = "nnUNetTrainer class default (no debug.json for this trainer)"
            else:
                oversample, debug_batch_size = debug
                if debug_batch_size != batch_size:
                    raise ValueError(
                        f"batch_size mismatch for {model_key}/{arch}: "
                        f"plans.json={batch_size}, debug.json={debug_batch_size}"
                    )
                source = "debug.json (identical across all folds)"
            forced = forced_foreground_count(batch_size, oversample)
            rows.append(
                {
                    "dataset": MODEL_DATASET_LABEL[model_key],
                    "arch": arch,
                    "batch_size": batch_size,
                    "oversample_foreground_percent": oversample,
                    "oversample_source": source,
                    "forced_foreground_patches_per_batch": forced,
                    "forced_foreground_share": forced / batch_size,
                }
            )
    return pd.DataFrame(rows)


# Gradient (backward-pass) evaluations per case per epsilon, read directly
# from the attack code (Gap G5). FGSM is one forward+backward
# (fgsm.py:61). PGD/SegPGD/CosPGD/DAG loop ``for k in range(1, n_steps + 1)``
# with one ``backward()`` per iteration (pgd.py:111,118; segpgd.py and
# cospgd.py delegate to the same ``pgd_trajectory``; dag.py:153,164), at
# ``n_steps=20`` (configs/attacks/{pgd,segpgd,cospgd,dag}.json). Auto-PGD
# costs ``n_steps + 1`` gradients -- one evaluation before the loop plus one
# per iteration (auto_pgd.py:30 docstring; backward calls at lines 187 and
# 229) -- at ``n_steps=20`` (configs/attacks/auto_pgd.json), so 21. Auto-PGD
# x3 reruns the full 21-gradient trajectory for 3 independent restarts
# (experiments/run_twelve_arm_wg_campaign.py:144, ``ARM_RESTARTS =
# {"auto_pgd_r3": 3}``; experiments/pgd_adversarial_evaluation.py:667,
# ``auto_pgd_with_restarts``'s ``for r in range(max(1, num_restarts))``), so
# 3 x 21 = 63. SEA runs Auto-PGD once per loss over 3 losses
# (sea.py:54, ``SEA_LOSSES = ("mce", "mce_bal", "js")``; sea.py:180-192 calls
# ``auto_pgd_attack`` once per loss), so 3 x 21 = 63 as well -- the same
# budget as Auto-PGD x3, by a different route.
GRADIENT_BUDGETS: tuple[dict[str, object], ...] = (
    {
        "attack": "FGSM",
        "n_steps": 1,
        "num_restarts": 1,
        "gradients_per_case_per_eps": 1,
        "source": "src/mri_prostate_seg/attacks/fgsm.py:61 (single loss.backward())",
    },
    {
        "attack": "PGD",
        "n_steps": 20,
        "num_restarts": 1,
        "gradients_per_case_per_eps": 20,
        "source": (
            "src/mri_prostate_seg/attacks/pgd.py:111,118 "
            "(for k in range(1, n_steps+1): loss.backward()); "
            "configs/attacks/pgd.json n_steps=20"
        ),
    },
    {
        "attack": "SegPGD",
        "n_steps": 20,
        "num_restarts": 1,
        "gradients_per_case_per_eps": 20,
        "source": (
            "src/mri_prostate_seg/attacks/segpgd.py:109-146 delegates to "
            "pgd_trajectory (pgd.py:111,118); configs/attacks/segpgd.json n_steps=20"
        ),
    },
    {
        "attack": "CosPGD",
        "n_steps": 20,
        "num_restarts": 1,
        "gradients_per_case_per_eps": 20,
        "source": (
            "src/mri_prostate_seg/attacks/cospgd.py:64-96 delegates to "
            "pgd_trajectory (pgd.py:111,118); configs/attacks/cospgd.json n_steps=20"
        ),
    },
    {
        "attack": "DAG",
        "n_steps": 20,
        "num_restarts": 1,
        "gradients_per_case_per_eps": 20,
        "source": (
            "src/mri_prostate_seg/attacks/dag.py:153,164 "
            "(for k in range(1, n_steps+1): objective.backward()); "
            "configs/attacks/dag.json n_steps=20"
        ),
    },
    {
        "attack": "Auto-PGD",
        "n_steps": 20,
        "num_restarts": 1,
        "gradients_per_case_per_eps": 21,
        "source": (
            "src/mri_prostate_seg/attacks/auto_pgd.py:30 "
            "('n_steps costs n_steps + 1 gradients'), backward calls at "
            "lines 187 (initial) and 229 (loop); configs/attacks/auto_pgd.json n_steps=20"
        ),
    },
    {
        "attack": "Auto-PGD x3",
        "n_steps": 20,
        "num_restarts": 3,
        "gradients_per_case_per_eps": 63,
        "source": (
            "experiments/run_twelve_arm_wg_campaign.py:144 "
            "(ARM_RESTARTS = {'auto_pgd_r3': 3}); "
            "experiments/pgd_adversarial_evaluation.py:667 "
            "(auto_pgd_with_restarts: for r in range(max(1, num_restarts))), "
            "each restart costing 21 gradients (auto_pgd.py:30)"
        ),
    },
    {
        "attack": "SEA",
        "n_steps": 20,
        "num_restarts": 1,
        "gradients_per_case_per_eps": 63,
        "source": (
            "src/mri_prostate_seg/attacks/sea.py:54 "
            "(SEA_LOSSES = ('mce', 'mce_bal', 'js'), 3 losses) and :180-192 "
            "(auto_pgd_attack called once per loss); configs/attacks/sea.json"
        ),
    },
)


def gradient_budget_table(n_epsilons: int = len(EPSILONS)) -> pd.DataFrame:
    """Gradients per case, per attack: per epsilon and summed over the eps grid (Gap G5)."""
    rows: list[dict[str, object]] = []
    for entry in GRADIENT_BUDGETS:
        row = dict(entry)
        per_eps = cast(int, entry["gradients_per_case_per_eps"])
        row["gradients_summed_over_grid"] = per_eps * n_epsilons
        rows.append(row)
    return pd.DataFrame(rows)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument(
        "--output-dir",
        default=os.path.join(RESULTS, "attack_comparison", "stats_no_blade"),
    )
    parser.add_argument("--arms", nargs="+", default=ARMS8)
    args = parser.parse_args()
    os.makedirs(args.output_dir, exist_ok=True)

    frames = load_frames(args.arms)

    clean_df = clean_dice_table(frames)
    clean_df.to_csv(
        os.path.join(args.output_dir, "evidence_clean_dice.csv"), index=False
    )

    budget_df = budget_avg_table(frames, args.arms)
    budget_df.to_csv(
        os.path.join(args.output_dir, "evidence_budget_avg.csv"), index=False
    )

    cross_arch_wilcoxon_wg = pd.read_csv(
        os.path.join(args.output_dir, "cross_arch_wilcoxon_WG.csv")
    )
    decomp_df = wg_batch_decomposition_table(frames, args.arms, cross_arch_wilcoxon_wg)
    decomp_df.to_csv(
        os.path.join(args.output_dir, "evidence_wg_batch_decomposition.csv"),
        index=False,
    )

    eps_df = per_epsilon_table(frames)
    eps_df.to_csv(
        os.path.join(args.output_dir, "evidence_per_epsilon_apgd_sea.csv"), index=False
    )

    eps_ci_df = per_epsilon_ci_table(frames)
    eps_ci_df.to_csv(
        os.path.join(args.output_dir, "evidence_per_epsilon_ci.csv"), index=False
    )

    batch_sig_df = batch_step_significance_table(frames, args.arms)
    batch_sig_df.to_csv(
        os.path.join(args.output_dir, "evidence_batch_step_significance.csv"),
        index=False,
    )

    params_df = model_parameter_counts_table()
    params_df.to_csv(
        os.path.join(args.output_dir, "evidence_model_parameter_counts.csv"),
        index=False,
    )

    schedule_df = training_schedule_table()
    schedule_df.to_csv(
        os.path.join(args.output_dir, "evidence_training_schedule.csv"), index=False
    )

    grad_df = gradient_budget_table()
    grad_df.to_csv(
        os.path.join(args.output_dir, "evidence_gradient_budgets.csv"), index=False
    )

    fold_frames = load_frames_by_fold(args.arms)
    per_fold_df = per_fold_shifts_table(fold_frames, args.arms)
    per_fold_df.to_csv(
        os.path.join(args.output_dir, "evidence_per_fold_shifts.csv"), index=False
    )
    for class_name in TREES:
        for arm in args.arms:
            summary = fold_agreement_summary(per_fold_df, class_name, arm)
            print(
                f"  per-fold agreement {class_name}/{arm}: "
                f"batch_gt_encoder_every_fold={summary['batch_gt_encoder_every_fold']} "
                f"({summary['batch_gt_encoder_n_folds_holding']}/{summary['n_folds']}), "
                f"share_batch_xl_of_total mean={summary['share_batch_xl_of_total_mean']:.3f} "
                f"[{summary['share_batch_xl_of_total_min']:.3f}, "
                f"{summary['share_batch_xl_of_total_max']:.3f}]"
            )

    oversampling_df = foreground_oversampling_table()
    oversampling_df.to_csv(
        os.path.join(args.output_dir, "evidence_foreground_oversampling.csv"),
        index=False,
    )

    print(
        f"clean_dice: {len(clean_df)} rows, budget_avg: {len(budget_df)} rows, "
        f"wg_batch_decomposition: {len(decomp_df)} rows, "
        f"per_epsilon_apgd_sea: {len(eps_df)} rows, "
        f"per_epsilon_ci: {len(eps_ci_df)} rows, "
        f"batch_step_significance: {len(batch_sig_df)} rows, "
        f"model_parameter_counts: {len(params_df)} rows, "
        f"training_schedule: {len(schedule_df)} rows, "
        f"gradient_budgets: {len(grad_df)} rows, "
        f"per_fold_shifts: {len(per_fold_df)} rows, "
        f"foreground_oversampling: {len(oversampling_df)} rows"
    )


if __name__ == "__main__":
    main()
