"""
Sensitivity and Perturbation Analysis for nnU-Net Segmentation Models.

Analyses:
  - Input gradient saliency maps (|dL/dx|) for clean inputs
  - FGSM perturbation spatial heatmaps (eps * sign(grad))
  - Class-conditional sensitivity (gradient w.r.t. per-class Dice; binary: Dice+BCE
    matching dice_ce_loss)
  - Overlay visualisations on MRI slices

Segmentation labels preserve IGNORE_LABEL (-1) for nnU-Net cropped regions.
Overlay panels normalise saliency by the max on that slice (display only; not
comparable across figures).
"""

import os
import gc
import random
import argparse
import numpy as np
import torch
import torch.nn.functional as F
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import Normalize

import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from fgsm_adversarial_evaluation import (
    MODEL_CONFIGS,
    load_model,
    pad_to_divisible,
    unpad,
    load_validation_ids,
    load_sample,
    dice_ce_loss,
    _pick_foreground_slice,
    IGNORE_LABEL,
)


def _eps_filename_tag(eps: float) -> str:
    """Compact epsilon string for filenames."""
    return f"{eps:.6g}".replace(".", "p").replace("-", "m")


def _compute_saliency(model, x_padded, y_padded, num_classes, orig_shape):
    """Compute |dL/dx| saliency map for the full loss."""
    model.zero_grad(set_to_none=True)
    x_in = x_padded.clone().detach().requires_grad_(True)
    logits = model(x_in)
    loss = dice_ce_loss(logits, y_padded, num_classes)
    loss.backward()
    saliency = x_in.grad.detach().abs()
    return unpad(saliency, orig_shape).cpu().numpy()[0, 0]


def _compute_perturbation_map(
    model, x_padded, y_padded, num_classes, orig_shape, eps: float
):
    """Compute the FGSM perturbation map: eps * sign(dL/dx)."""
    model.zero_grad(set_to_none=True)
    x_in = x_padded.clone().detach().requires_grad_(True)
    logits = model(x_in)
    loss = dice_ce_loss(logits, y_padded, num_classes)
    loss.backward()
    grad_sign = x_in.grad.detach().sign()
    perturbation = eps * grad_sign
    return unpad(perturbation, orig_shape).cpu().numpy()[0, 0]


def _compute_class_saliency(
    model, x_padded, y_padded, num_classes, orig_shape, target_class: int
):
    """Compute |dL/dx| where L is class-specific (Dice for that class; WG: same as dice_ce_loss)."""
    model.zero_grad(set_to_none=True)
    x_in = x_padded.clone().detach().requires_grad_(True)
    logits = model(x_in)

    if num_classes == 2 and target_class == 1:
        t = y_padded.squeeze(1)
        valid = (t != IGNORE_LABEL).float().unsqueeze(1)
        target_fg = (t == 1).float().unsqueeze(1)
        probs = torch.sigmoid(logits[:, 1:2])
        ce_raw = F.binary_cross_entropy_with_logits(
            logits[:, 1:2], target_fg, reduction="none"
        )
        ce = (ce_raw * valid).sum() / (valid.sum() + 1e-8)
        target_oh = target_fg * valid
        inter = (probs * target_oh).sum()
        denom = (probs * valid).sum() + target_oh.sum()
        dice_loss = 1.0 - (2.0 * inter + 1e-5) / (denom + 1e-5)
        loss = dice_loss + ce
    else:
        t = y_padded.squeeze(1)
        valid = (t != IGNORE_LABEL).float()
        probs_c = torch.softmax(logits, dim=1)[:, target_class]
        target_c = (t == target_class).float() * valid
        inter = (probs_c * target_c).sum()
        denom = (probs_c * valid).sum() + target_c.sum()
        loss = 1.0 - (2.0 * inter + 1e-5) / (denom + 1e-5)

    loss.backward()
    saliency = x_in.grad.detach().abs()
    return unpad(saliency, orig_shape).cpu().numpy()[0, 0]


def _save_saliency_figure(
    mri_slice, saliency_slice, gt_slice, title, output_path, cmap="hot", alpha=0.5
):
    """Overlay saliency on MRI with GT contour."""
    vmin_mri, vmax_mri = np.percentile(mri_slice, [1, 99])
    normed = np.clip((mri_slice - vmin_mri) / (vmax_mri - vmin_mri + 1e-8), 0, 1)

    fig, axes = plt.subplots(1, 3, figsize=(14, 4.5))

    axes[0].imshow(normed, cmap="gray", interpolation="nearest")
    if gt_slice is not None and gt_slice.max() > 0:
        axes[0].contour(gt_slice, levels=[0.5], colors=["cyan"], linewidths=1)
    axes[0].set_title("MRI + GT contour", fontsize=9)
    axes[0].axis("off")

    axes[1].imshow(saliency_slice, cmap=cmap, interpolation="nearest")
    axes[1].set_title("Saliency |∇L|", fontsize=9)
    axes[1].axis("off")

    axes[2].imshow(normed, cmap="gray", interpolation="nearest")
    # Per-slice max scaling for display (not comparable across patients/exports).
    sal_norm = saliency_slice / (saliency_slice.max() + 1e-8)
    axes[2].imshow(sal_norm, cmap=cmap, alpha=alpha, interpolation="nearest")
    if gt_slice is not None and gt_slice.max() > 0:
        axes[2].contour(gt_slice, levels=[0.5], colors=["cyan"], linewidths=1)
    axes[2].set_title("Overlay", fontsize=9)
    axes[2].axis("off")

    fig.suptitle(title, fontsize=10, fontweight="bold")
    plt.tight_layout()
    fig.savefig(output_path, dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved {output_path}")


def _save_perturbation_figure(
    mri_slice, perturbation_slice, gt_slice, title, output_path
):
    """Visualise the FGSM perturbation spatial distribution."""
    vmin_mri, vmax_mri = np.percentile(mri_slice, [1, 99])
    normed = np.clip((mri_slice - vmin_mri) / (vmax_mri - vmin_mri + 1e-8), 0, 1)
    abs_pert = np.abs(perturbation_slice)

    fig, axes = plt.subplots(1, 3, figsize=(14, 4.5))

    axes[0].imshow(normed, cmap="gray", interpolation="nearest")
    axes[0].set_title("Clean MRI", fontsize=9)
    axes[0].axis("off")

    im = axes[1].imshow(
        perturbation_slice,
        cmap="RdBu_r",
        interpolation="nearest",
        norm=Normalize(vmin=-abs_pert.max(), vmax=abs_pert.max()),
    )
    axes[1].set_title("Perturbation (signed)", fontsize=9)
    axes[1].axis("off")
    fig.colorbar(im, ax=axes[1], fraction=0.046, pad=0.04)

    axes[2].imshow(abs_pert, cmap="hot", interpolation="nearest")
    if gt_slice is not None and gt_slice.max() > 0:
        axes[2].contour(gt_slice, levels=[0.5], colors=["cyan"], linewidths=1)
    axes[2].set_title("|Perturbation| + GT contour", fontsize=9)
    axes[2].axis("off")

    fig.suptitle(title, fontsize=10, fontweight="bold")
    plt.tight_layout()
    fig.savefig(output_path, dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved {output_path}")


def analyze_sensitivity(
    model_key: str,
    device: torch.device,
    n_patients: int = 3,
    target_eps: float = 0.5,
    output_dir: str = "sensitivity_results",
    seed: int | None = None,
):
    cfg = MODEL_CONFIGS[model_key]
    print(f"\n{'=' * 60}")
    print(f"Sensitivity Analysis: {cfg['name']} ({model_key})")
    print(f"{'=' * 60}")

    if seed is not None:
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)

    network, div_factors = load_model(
        cfg["checkpoint"], cfg["configuration"], cfg["num_classes"], device
    )
    network.eval()

    val_ids = load_validation_ids(cfg["splits_json"])
    if not val_ids:
        print("No validation IDs; exiting.")
        del network
        return
    k = min(n_patients, len(val_ids))
    selected = random.sample(val_ids, k)

    num_classes = cfg["num_classes"]
    class_names = cfg["class_names"]
    fg_classes = list(range(1, num_classes))

    os.makedirs(output_dir, exist_ok=True)
    eps_tag = _eps_filename_tag(target_eps)

    for case_id in selected:
        img_np, seg_np = load_sample(cfg["data_dir"], case_id)
        orig_seg_np = np.asarray(seg_np, dtype=np.int64)
        gt = orig_seg_np[0]

        x = torch.from_numpy(img_np[np.newaxis]).to(device)
        y = torch.from_numpy(orig_seg_np[np.newaxis].astype(np.float32)).to(device)
        x_padded, orig_shape = pad_to_divisible(x, div_factors)
        y_padded, _ = pad_to_divisible(y, div_factors)

        slice_idx = _pick_foreground_slice(gt)
        mri_slice = img_np[0, slice_idx]
        gt_slice = gt[slice_idx]
        print(f"  Patient={case_id}, slice={slice_idx}")

        with torch.enable_grad():
            saliency = _compute_saliency(
                network, x_padded, y_padded, num_classes, orig_shape
            )
        _save_saliency_figure(
            mri_slice,
            saliency[slice_idx],
            (gt_slice > 0).astype(np.uint8),
            title=f"{cfg['name']} — Saliency — {case_id} (slice {slice_idx})",
            output_path=os.path.join(output_dir, f"{model_key}_{case_id}_saliency.png"),
        )

        with torch.enable_grad():
            pert_map = _compute_perturbation_map(
                network, x_padded, y_padded, num_classes, orig_shape, eps=target_eps
            )
        _save_perturbation_figure(
            mri_slice,
            pert_map[slice_idx],
            (gt_slice > 0).astype(np.uint8),
            title=f"{cfg['name']} — FGSM Perturbation (ε={target_eps}) — {case_id}",
            output_path=os.path.join(
                output_dir, f"{model_key}_{case_id}_perturbation_eps{eps_tag}.png"
            ),
        )

        class_sal_volumes: dict[str, np.ndarray] = {}
        for c, cname in zip(fg_classes, class_names):
            with torch.enable_grad():
                class_sal_volumes[cname] = _compute_class_saliency(
                    network, x_padded, y_padded, num_classes, orig_shape, target_class=c
                )
            safe_cls = cname.replace("+", "").replace("/", "")
            gt_c = (gt_slice == c).astype(np.uint8)
            _save_saliency_figure(
                mri_slice,
                class_sal_volumes[cname][slice_idx],
                gt_c,
                title=(
                    f"{cfg['name']} — {cname} saliency (∇Dice"
                    f"{'+CE' if num_classes == 2 else ''}) — {case_id} (slice {slice_idx})"
                ),
                output_path=os.path.join(
                    output_dir, f"{model_key}_{case_id}_{safe_cls}_class_saliency.png"
                ),
                cmap="inferno",
            )

        if len(class_sal_volumes) > 1:
            ncols = len(class_sal_volumes) + 1
            fig, axes = plt.subplots(1, ncols, figsize=(5 * ncols, 4.5))
            axes = np.atleast_1d(axes).ravel()
            vmin_mri, vmax_mri = np.percentile(mri_slice, [1, 99])
            normed = np.clip(
                (mri_slice - vmin_mri) / (vmax_mri - vmin_mri + 1e-8), 0, 1
            )
            axes[0].imshow(normed, cmap="gray")
            for c_idx in fg_classes:
                gt_c = (gt_slice == c_idx).astype(np.uint8)
                if gt_c.max() > 0:
                    axes[0].contour(gt_c, levels=[0.5], linewidths=1)
            axes[0].set_title("MRI + GT", fontsize=9)
            axes[0].axis("off")

            for j, (cname, sal_vol) in enumerate(class_sal_volumes.items()):
                axes[j + 1].imshow(
                    sal_vol[slice_idx], cmap="inferno", interpolation="nearest"
                )
                axes[j + 1].set_title(f"|∇Dice_{{{cname}}}|", fontsize=9)
                axes[j + 1].axis("off")

            fig.suptitle(
                f"{cfg['name']} — Class Dice Saliency — {case_id}",
                fontsize=10,
                fontweight="bold",
            )
            plt.tight_layout()
            path = os.path.join(
                output_dir, f"{model_key}_{case_id}_class_comparison.png"
            )
            fig.savefig(path, dpi=200, bbox_inches="tight")
            plt.close(fig)
            print(f"  Saved {path}")

        del x, y, x_padded, y_padded
        if device.type == "cuda":
            torch.cuda.empty_cache()

    del network
    gc.collect()
    if device.type == "cuda":
        torch.cuda.empty_cache()


def main():
    parser = argparse.ArgumentParser(
        description="Sensitivity and perturbation analysis"
    )
    parser.add_argument("--model", choices=["wg", "zones", "both"], default="both")
    parser.add_argument(
        "--n-patients",
        type=int,
        default=3,
        help="Number of random patients to visualise",
    )
    parser.add_argument(
        "--eps", type=float, default=0.5, help="Epsilon for perturbation heatmap"
    )
    parser.add_argument("--output-dir", default="sensitivity_results")
    parser.add_argument("--seed", type=int, default=None)
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    models = ["wg", "zones"] if args.model == "both" else [args.model]
    for mk in models:
        analyze_sensitivity(
            mk,
            device,
            n_patients=args.n_patients,
            target_eps=args.eps,
            output_dir=args.output_dir,
            seed=args.seed,
        )

    print("\nDone.")


if __name__ == "__main__":
    main()
