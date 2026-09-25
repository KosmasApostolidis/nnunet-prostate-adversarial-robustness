"""
Failure Case and Error Analysis for FGSM Adversarial Evaluation.

Analyses:
  - Worst-case patient ranking by Dice drop (clean -> adversarial)
  - Error type breakdown: over-segmentation vs under-segmentation (FP / FN volume)
  - Volume analysis: predicted vs GT volume ratio at each epsilon
  - Worst-case slice visualisation for top-K most degraded patients

Worst-case slice selection uses foreground for the class under study; small or
off-plane structures may not be fully visible on the chosen slice.
"""

import os
import gc
import math
import random
import argparse
import numpy as np
import torch
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd

import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from fgsm_adversarial_evaluation import (
    MODEL_CONFIGS,
    load_model,
    pad_to_divisible,
    unpad,
    load_validation_ids,
    load_sample,
    fgsm_attack,
    compute_metrics,
    _logits_to_seg,
    _pick_foreground_slice,
    _make_panel,
    _masked_class_masks_vol,
    IGNORE_LABEL,
)

# Single tolerance for ε matching (CSV floats, CLI, and cache keys).
_EPS_RTOL = 0.0
_EPS_ATOL = 1e-9


def _eps_close(a: float, b: float) -> bool:
    return math.isclose(float(a), float(b), rel_tol=_EPS_RTOL, abs_tol=_EPS_ATOL)


def _is_zero_eps(eps: float) -> bool:
    return abs(float(eps)) <= _EPS_ATOL


def _df_eps_mask(df, eps: float) -> np.ndarray:
    """Boolean mask for rows matching epsilon (robust to int/float)."""
    return np.isclose(
        df["epsilon"].astype(float), float(eps), rtol=_EPS_RTOL, atol=_EPS_ATOL
    )


def _eps_in_list(eps_list: list[float], eps: float) -> bool:
    return any(_eps_close(x, eps) for x in eps_list)


def _resolve_eps_in_list(eps_list: list[float], target: float) -> float | None:
    """Return the list element that matches target, or None."""
    for e in eps_list:
        if _eps_close(float(e), float(target)):
            return float(e)
    return None


def _compute_volumes_and_errors(
    pred_bin: np.ndarray, gt_bin: np.ndarray, voxel_vol_mm3: float = 1.0
):
    """Compute volume-based metrics for a single class.
    Returns (pred_vol, gt_vol, fp_vol, fn_vol) in mm^3.
    """
    pred_vol = float(pred_bin.sum()) * voxel_vol_mm3
    gt_vol = float(gt_bin.sum()) * voxel_vol_mm3
    fp = float(((pred_bin == 1) & (gt_bin == 0)).sum()) * voxel_vol_mm3
    fn = float(((pred_bin == 0) & (gt_bin == 1)).sum()) * voxel_vol_mm3
    return pred_vol, gt_vol, fp, fn


def evaluate_failure_cases(
    model_key: str,
    device: torch.device,
    target_epsilons: list[float] | None = None,
    max_samples: int | None = None,
    top_k: int = 10,
    output_dir: str = "results/failure_results",
    voxel_spacing_mm: tuple[float, float, float] = (0.5, 0.5, 3.0),
    vis_adv_epsilon: float | None = None,
    sample_seed: int | None = None,
):
    cfg = MODEL_CONFIGS[model_key]
    print(f"\n{'=' * 60}")
    print(f"Failure Analysis: {cfg['name']} ({model_key})")
    print(f"{'=' * 60}")

    os.makedirs(output_dir, exist_ok=True)

    network, div_factors = load_model(
        cfg["checkpoint"], cfg["configuration"], cfg["num_classes"], device
    )
    network.eval()

    val_ids = load_validation_ids(cfg["splits_json"])
    if max_samples is not None and max_samples < len(val_ids):
        if sample_seed is not None:
            rng = random.Random(sample_seed)
            val_ids = rng.sample(val_ids, max_samples)
        else:
            val_ids = val_ids[:max_samples]
    if not val_ids:
        print("No validation samples; exiting.")
        del network
        return

    num_classes = cfg["num_classes"]
    class_names = cfg["class_names"]
    fg_classes = list(range(1, num_classes))
    voxel_vol = float(np.prod(voxel_spacing_mm))

    if target_epsilons is None:
        target_epsilons = [0.0, 0.5, 1.0]
    target_epsilons = sorted({float(e) for e in target_epsilons})

    rows = []

    for i, case_id in enumerate(val_ids):
        img_np, seg_np = load_sample(cfg["data_dir"], case_id)
        orig_seg_np = np.asarray(seg_np, dtype=np.int64)

        x = torch.from_numpy(img_np[np.newaxis]).to(device)
        y = torch.from_numpy(orig_seg_np[np.newaxis].astype(np.float32)).to(device)
        x_padded, orig_shape = pad_to_divisible(x, div_factors)
        y_padded, _ = pad_to_divisible(y, div_factors)

        cached_clean = None

        for eps in target_epsilons:
            if cached_clean is not None and _is_zero_eps(eps):
                logits, pred = cached_clean
            else:
                if _is_zero_eps(eps):
                    x_input = x_padded
                else:
                    with torch.enable_grad():
                        x_input = fgsm_attack(
                            network, x_padded, y_padded, eps, num_classes
                        )
                with torch.no_grad():
                    logits_padded = network(x_input)
                logits = unpad(logits_padded, orig_shape)
                pred = _logits_to_seg(logits, num_classes)
                if not _is_zero_eps(eps):
                    del x_input, logits_padded
                else:
                    del logits_padded
                if _is_zero_eps(eps):
                    cached_clean = (logits, pred)

            gt = orig_seg_np[0]
            metrics = compute_metrics(logits, orig_seg_np[np.newaxis], num_classes)

            for c, cname in zip(fg_classes, class_names):
                gt_c, pred_c = _masked_class_masks_vol(gt, pred, c)
                pred_vol, gt_vol, fp_vol, fn_vol = _compute_volumes_and_errors(
                    pred_c, gt_c, voxel_vol
                )
                d, h, a = metrics[c]
                vol_ratio = pred_vol / gt_vol if gt_vol > 0 else np.nan
                rows.append(
                    {
                        "case_id": case_id,
                        "epsilon": float(eps),
                        "class": cname,
                        "dice": d,
                        "hd95": h,
                        "asd": a,
                        "pred_vol_mm3": pred_vol,
                        "gt_vol_mm3": gt_vol,
                        "vol_ratio": vol_ratio,
                        "fp_vol_mm3": fp_vol,
                        "fn_vol_mm3": fn_vol,
                    }
                )

        if (i + 1) % 10 == 0 or (i + 1) == len(val_ids):
            print(f"  [{i + 1}/{len(val_ids)}] processed")

        del x, y, x_padded, y_padded

    df = pd.DataFrame(rows)

    csv_path = os.path.join(output_dir, f"{model_key}_failure_analysis.csv")
    df.to_csv(csv_path, index=False, float_format="%.6f")
    print(f"  Saved {csv_path}")

    _worst_case_ranking(df, model_key, class_names, target_epsilons, top_k, output_dir)
    _error_type_plots(df, model_key, class_names, target_epsilons, output_dir)
    _volume_plots(df, model_key, class_names, target_epsilons, output_dir)
    _worst_case_visualization(
        network,
        div_factors,
        df,
        model_key,
        cfg,
        device,
        num_classes,
        fg_classes,
        class_names,
        target_epsilons,
        top_k,
        output_dir,
        vis_adv_epsilon=vis_adv_epsilon,
    )

    del network
    gc.collect()
    if device.type == "cuda":
        torch.cuda.empty_cache()


def _worst_case_ranking(df, model_key, class_names, target_epsilons, top_k, output_dir):
    """Rank patients by Dice drop from clean to adversarial."""
    eps0 = 0.0
    if not _eps_in_list(target_epsilons, eps0):
        print(
            "  Skipping worst-case ranking: ε=0 not in target_epsilons (no clean baseline)."
        )
        return

    for cname in class_names:
        clean = df[_df_eps_mask(df, eps0) & (df["class"] == cname)][
            ["case_id", "dice"]
        ].rename(columns={"dice": "dice_clean"})
        for eps in target_epsilons:
            if _is_zero_eps(float(eps)):
                continue
            adv = df[_df_eps_mask(df, eps) & (df["class"] == cname)][
                ["case_id", "dice"]
            ].rename(columns={"dice": "dice_adv"})
            merged = clean.merge(adv, on="case_id")
            if len(merged) == 0:
                print(
                    f"  Skipping worst-case ranking plot (no paired rows): {cname} ε={eps}"
                )
                continue
            merged["dice_drop"] = merged["dice_clean"] - merged["dice_adv"]
            merged = merged.sort_values("dice_drop", ascending=False)

            path = os.path.join(
                output_dir,
                f"{model_key}_{cname.replace('+', '').replace('/', '')}_worst_cases_eps{eps}.csv",
            )
            merged.to_csv(path, index=False, float_format="%.6f")
            print(f"  Saved {path}")

            fig, ax = plt.subplots(figsize=(10, 5))
            top = merged.head(top_k)
            ax.barh(range(len(top)), top["dice_drop"].values, color="tomato", alpha=0.7)
            ax.set_yticks(range(len(top)))
            ax.set_yticklabels(top["case_id"].values, fontsize=7)
            ax.set_xlabel("Dice Drop (clean − adversarial)")
            safe_cls = cname.replace("+", "").replace("/", "")
            ax.set_title(
                f"{model_key.upper()} — {cname}: Top-{top_k} Worst Affected (ε={eps})"
            )
            ax.invert_yaxis()
            ax.grid(True, alpha=0.2, axis="x")
            plt.tight_layout()
            path = os.path.join(
                output_dir, f"{model_key}_{safe_cls}_worst_cases_eps{eps}.png"
            )
            fig.savefig(path, dpi=200, bbox_inches="tight")
            plt.close(fig)
            print(f"  Saved {path}")


def _error_type_plots(df, model_key, class_names, target_epsilons, output_dir):
    """Bar chart of mean FP and FN volumes at each epsilon."""
    for cname in class_names:
        sub = df[df["class"] == cname]
        eps_labels = []
        mean_fp, mean_fn = [], []
        for eps in target_epsilons:
            ep_sub = sub[_df_eps_mask(sub, eps)]
            eps_labels.append(f"{eps:.1f}")
            mean_fp.append(
                float(ep_sub["fp_vol_mm3"].mean()) if len(ep_sub) else np.nan
            )
            mean_fn.append(
                float(ep_sub["fn_vol_mm3"].mean()) if len(ep_sub) else np.nan
            )

        x_pos = np.arange(len(eps_labels))
        width = 0.35
        fig, ax = plt.subplots(figsize=(8, 5))
        ax.bar(
            x_pos - width / 2,
            mean_fp,
            width,
            label="FP Volume (over-seg)",
            color="salmon",
            alpha=0.8,
        )
        ax.bar(
            x_pos + width / 2,
            mean_fn,
            width,
            label="FN Volume (under-seg)",
            color="cornflowerblue",
            alpha=0.8,
        )
        ax.set_xticks(x_pos)
        ax.set_xticklabels(eps_labels)
        ax.set_xlabel("FGSM Epsilon")
        ax.set_ylabel("Mean Volume (mm³)")
        safe_cls = cname.replace("+", "").replace("/", "")
        ax.set_title(f"{model_key.upper()} — {cname}: Over-seg vs Under-seg")
        ax.legend()
        ax.grid(True, alpha=0.2, axis="y")
        plt.tight_layout()
        path = os.path.join(output_dir, f"{model_key}_{safe_cls}_error_types.png")
        fig.savefig(path, dpi=200, bbox_inches="tight")
        plt.close(fig)
        print(f"  Saved {path}")


def _volume_plots(df, model_key, class_names, target_epsilons, output_dir):
    """Scatter plot of volume ratio (pred / GT) at each epsilon."""
    rng = np.random.RandomState(42)
    for cname in class_names:
        sub = df[df["class"] == cname]
        fig, ax = plt.subplots(figsize=(8, 5))
        for eps in target_epsilons:
            ep_sub = sub[_df_eps_mask(sub, eps)]
            ratios = ep_sub["vol_ratio"].dropna().values
            jitter = rng.uniform(-0.02, 0.02, len(ratios))
            ax.scatter(
                np.full(len(ratios), eps) + jitter,
                ratios,
                s=8,
                alpha=0.3,
                label=f"ε={eps:.1f}",
            )
        ax.axhline(1.0, color="black", linestyle="--", alpha=0.5, label="Perfect (1.0)")
        ax.set_xlabel("FGSM Epsilon")
        ax.set_ylabel("Volume Ratio (Pred / GT)")
        safe_cls = cname.replace("+", "").replace("/", "")
        ax.set_title(f"{model_key.upper()} — {cname}: Volume Ratio vs Epsilon")
        ax.legend(fontsize=7, ncol=2)
        ax.grid(True, alpha=0.2)
        plt.tight_layout()
        path = os.path.join(output_dir, f"{model_key}_{safe_cls}_volume_ratio.png")
        fig.savefig(path, dpi=200, bbox_inches="tight")
        plt.close(fig)
        print(f"  Saved {path}")

        means, stds = [], []
        for eps in target_epsilons:
            r = sub[_df_eps_mask(sub, eps)]["vol_ratio"].dropna()
            means.append(float(r.mean()) if len(r) else np.nan)
            if len(r) > 1:
                stds.append(float(r.std(ddof=1)))
            else:
                stds.append(0.0)

        fig, ax = plt.subplots(figsize=(8, 5))
        ax.errorbar(target_epsilons, means, yerr=stds, marker="s", capsize=4)
        ax.axhline(1.0, color="black", linestyle="--", alpha=0.5)
        ax.set_xlabel("FGSM Epsilon")
        ax.set_ylabel("Mean Volume Ratio (Pred / GT)")
        ax.set_title(f"{model_key.upper()} — {cname}: Mean Volume Ratio")
        ax.grid(True, alpha=0.2)
        plt.tight_layout()
        path = os.path.join(output_dir, f"{model_key}_{safe_cls}_mean_volume_ratio.png")
        fig.savefig(path, dpi=200, bbox_inches="tight")
        plt.close(fig)
        print(f"  Saved {path}")


def _worst_case_visualization(
    network,
    div_factors,
    df,
    model_key,
    cfg,
    device,
    num_classes,
    fg_classes,
    class_names,
    target_epsilons,
    top_k,
    output_dir,
    vis_adv_epsilon: float | None = None,
):
    """Visualise slices for the worst-affected patients (clean vs one adversarial ε)."""
    eps0 = 0.0
    if not _eps_in_list(target_epsilons, eps0):
        print("  Skipping worst-case visualization: ε=0 not in target_epsilons.")
        return
    pos_eps = [float(e) for e in target_epsilons if float(e) > _EPS_ATOL]
    if not pos_eps:
        print("  Skipping worst-case visualization: no positive ε in list.")
        return

    if vis_adv_epsilon is not None:
        adv_eps = _resolve_eps_in_list(list(target_epsilons), float(vis_adv_epsilon))
        if adv_eps is None or adv_eps <= _EPS_ATOL:
            print(
                f"  Skipping worst-case visualization: --vis-adv-epsilon {vis_adv_epsilon} "
                "not in target_epsilons or not positive."
            )
            return
    else:
        adv_eps = max(pos_eps)

    print(
        f"  Worst-case slice visualisation: comparing clean (ε=0) vs adversarial ε={adv_eps}."
    )

    for cname in class_names:
        clean_df = df[_df_eps_mask(df, eps0) & (df["class"] == cname)][
            ["case_id", "dice"]
        ].rename(columns={"dice": "dice_clean"})
        adv_df = df[_df_eps_mask(df, adv_eps) & (df["class"] == cname)][
            ["case_id", "dice"]
        ].rename(columns={"dice": "dice_adv"})
        merged = clean_df.merge(adv_df, on="case_id")
        merged["dice_drop"] = merged["dice_clean"] - merged["dice_adv"]
        n_worst = min(top_k, len(merged)) if len(merged) else 0
        if n_worst == 0:
            continue
        worst = merged.nlargest(n_worst, "dice_drop")

        for _, row in worst.iterrows():
            case_id = row["case_id"]
            img_np, seg_np = load_sample(cfg["data_dir"], case_id)
            orig_seg_np = np.asarray(seg_np, dtype=np.int64)
            gt = orig_seg_np[0]

            x = torch.from_numpy(img_np[np.newaxis]).to(device)
            y = torch.from_numpy(orig_seg_np[np.newaxis].astype(np.float32)).to(device)
            x_padded, orig_shape = pad_to_divisible(x, div_factors)
            y_padded, _ = pad_to_divisible(y, div_factors)

            c_idx = fg_classes[class_names.index(cname)]
            gt_c_fg = ((gt == c_idx) & (gt != IGNORE_LABEL)).astype(np.uint8)
            slice_idx = _pick_foreground_slice(gt_c_fg)
            mri_slice = img_np[0, slice_idx]
            vmin, vmax = np.percentile(mri_slice, [1, 99])

            panels = []
            titles = []
            cached_clean = None
            for eps in (eps0, adv_eps):
                if cached_clean is not None and _is_zero_eps(eps):
                    logits, pred = cached_clean
                else:
                    if _is_zero_eps(eps):
                        x_input = x_padded
                    else:
                        with torch.enable_grad():
                            x_input = fgsm_attack(
                                network, x_padded, y_padded, eps, num_classes
                            )
                    with torch.no_grad():
                        logits_padded = network(x_input)
                    logits = unpad(logits_padded, orig_shape)
                    pred = _logits_to_seg(logits, num_classes)
                    if not _is_zero_eps(eps):
                        del x_input, logits_padded
                    else:
                        del logits_padded
                    if _is_zero_eps(eps):
                        cached_clean = (logits, pred)

                gt_c_slice, pred_c_slice = _masked_class_masks_vol(
                    gt[slice_idx], pred[slice_idx], c_idx
                )
                panel = _make_panel(mri_slice, gt_c_slice, pred_c_slice, vmin, vmax)
                panels.append(panel)
                m = compute_metrics(logits, orig_seg_np[np.newaxis], num_classes)
                dice_val = m[c_idx][0]
                if _is_zero_eps(eps):
                    label = "Clean (ε=0)"
                else:
                    label = f"ε={eps}"
                titles.append(f"{label}\nDice: {dice_val:.3f}")

            fig, axes = plt.subplots(1, 2, figsize=(8, 4))
            for j, (panel, title) in enumerate(zip(panels, titles)):
                axes[j].imshow(panel, interpolation="nearest")
                axes[j].set_title(title, fontsize=9)
                axes[j].axis("off")
            safe_cls = cname.replace("+", "").replace("/", "")
            fig.suptitle(
                f"Worst Case: {case_id} — {cname} (slice {slice_idx}; adv ε={adv_eps})",
                fontsize=10,
                fontweight="bold",
            )
            plt.tight_layout()
            path = os.path.join(
                output_dir,
                f"{model_key}_{safe_cls}_worst_{case_id}_slice{slice_idx}.png",
            )
            fig.savefig(path, dpi=200, bbox_inches="tight")
            plt.close(fig)
            print(f"  Saved {path}")

            del x, y, x_padded, y_padded


def main():
    parser = argparse.ArgumentParser(description="Failure case and error analysis")
    parser.add_argument("--model", choices=["wg", "zones", "both"], default="both")
    parser.add_argument(
        "--max-samples",
        type=int,
        default=None,
        help="Limit validation cases: first N in split order unless --sample-seed is set",
    )
    parser.add_argument(
        "--top-k", type=int, default=10, help="Number of worst cases to report"
    )
    parser.add_argument("--output-dir", default="results/failure_results")
    parser.add_argument(
        "--epsilons",
        type=float,
        nargs="+",
        default=[0.0, 0.5, 1.0],
        help="Epsilon values to evaluate",
    )
    parser.add_argument(
        "--voxel-spacing",
        type=float,
        nargs=3,
        default=[0.5, 0.5, 3.0],
        metavar=("SX", "SY", "SZ"),
        help="Voxel spacing in mm (sx sy sz). Default is illustrative; set to your dataset.",
    )
    parser.add_argument(
        "--vis-adv-epsilon",
        type=float,
        default=None,
        help="Adversarial ε for worst-case slice figures (default: max positive ε in --epsilons)",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=None,
        help="Seed for torch / numpy / random (reproducibility)",
    )
    parser.add_argument(
        "--sample-seed",
        type=int,
        default=None,
        help="If set with --max-samples, subsample validation cases randomly (reproducible)",
    )
    args = parser.parse_args()

    if args.seed is not None:
        random.seed(args.seed)
        np.random.seed(args.seed)
        torch.manual_seed(args.seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(args.seed)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    eps_raw = [float(e) for e in args.epsilons]
    if not any(_is_zero_eps(e) for e in eps_raw):
        print(
            "Warning: ε=0 not in --epsilons; adding clean baseline for ranking / visualisation."
        )
        eps_raw = [0.0] + eps_raw
    eps_sorted = sorted({float(e) for e in eps_raw})

    voxel_spacing = (
        float(args.voxel_spacing[0]),
        float(args.voxel_spacing[1]),
        float(args.voxel_spacing[2]),
    )

    models = ["wg", "zones"] if args.model == "both" else [args.model]
    for mk in models:
        evaluate_failure_cases(
            mk,
            device,
            target_epsilons=eps_sorted,
            max_samples=args.max_samples,
            top_k=args.top_k,
            output_dir=args.output_dir,
            voxel_spacing_mm=voxel_spacing,
            vis_adv_epsilon=args.vis_adv_epsilon,
            sample_seed=args.sample_seed,
        )

    print("\nDone.")


if __name__ == "__main__":
    main()
