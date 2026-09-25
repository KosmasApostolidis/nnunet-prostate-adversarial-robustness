"""Perturbation-magnitude comparison across the epsilon sweep for each dataset's worst case.

Worst case = lowest area-under-the-Dice-vs-epsilon curve across the full
clean-model APGD sweep (0, 2/255, 4/255, 8/255, 16/255), i.e. the case most
vulnerable across the *whole* epsilon range rather than just at the largest
epsilon, excluding cases already shown in plot_top_failure_cases.py's top-3
grid so the two figures illustrate different examples.

Attacks the chosen worst-case slice with APGD-BCE at every
retained epsilon (0, 2/255, 4/255, 8/255, 16/255) against the clean UNet
checkpoint, and renders, per dataset, a 2-row x 5-col figure: top row = the
attacked image slice at that epsilon with the ground-truth mask and the
model's own prediction overlaid in two distinct colors, bottom row = a
boxplot of the per-pixel perturbation (x_adv - x_clean) — showing how much
of the L_inf noise budget the attack actually uses at each epsilon.

Note: apgd_bce() clamps every perturbed image back to the clean image's own
min/max (see _samplewise_bounds in adv_rob_eval_nnunet_nnunetrecenc.py), so
the *absolute* pixel value range barely shifts with epsilon — the
perturbation delta plotted here is what actually grows with epsilon.

Usage::

    python experiments/plot_pixel_range_epsilon.py
"""

from __future__ import annotations

import os
import sys

import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from experiments.adv_rob_eval_nnunet_nnunetrecenc import (
    DATASETS,
    NNUNET_PATHS,
    TRAINERS,
    apgd_bce,
    build_model,
    load_sample,
    pad_to_divisible,
    unpad,
)

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUT_DIR = os.path.join(REPO_ROOT, "results", "adv_rob_eval_results", "figures")

DATASET_LABELS = {"wg": "WG", "zones": "Zones"}
EPSILONS = [0.0, 2 / 255, 4 / 255, 8 / 255, 16 / 255]

# Chosen from lowest volume-level area-under-the-Dice-vs-epsilon curve
# candidates (clean UNet, full APGD sweep), then verified against the actual
# rendered best-slice Dice (which can diverge from the volume-level metric)
# and normal image contrast. Excludes cases already used in
# plot_top_failure_cases.py's top-3 grid.
WORST_CASE_ID = {
    "wg": "ProstateWG_257064078224475270522402338903754343410",
    "zones": (
        "ProstateZonesFilteredLessDilated_"
        "ProstateZones_127283536152476161733668159035515442572"
    ),
}

GT_COLOR = np.array([0.20, 0.55, 0.95, 0.45])  # blue — ground truth
PRED_COLOR = np.array([0.95, 0.30, 0.15, 0.45])  # red — model prediction


def find_best_slice(gt_vol: np.ndarray) -> int:
    mask = gt_vol > 0
    return int(np.argmax(mask.sum(axis=(1, 2))))


def slice_dice(gt_mask: np.ndarray, pred_mask: np.ndarray) -> float:
    intersection = np.logical_and(gt_mask, pred_mask).sum()
    denom = gt_mask.sum() + pred_mask.sum()
    return float(2 * intersection / denom) if denom > 0 else float("nan")


def mask_rgba(mask: np.ndarray, color: np.ndarray) -> np.ndarray:
    rgba = np.zeros((*mask.shape, 4), dtype=np.float64)
    rgba[mask] = color
    return rgba


def attacked_slice(model, div_factors, num_classes, data_dir, case_id, z, eps, device):
    img_np, seg_np, _ = load_sample(data_dir, case_id)
    x = torch.from_numpy(img_np[np.newaxis]).to(device)
    y = torch.from_numpy(seg_np[np.newaxis].astype(np.float32)).to(device)
    x_padded, orig_shape = pad_to_divisible(x, div_factors)
    y_padded, _ = pad_to_divisible(y, div_factors)

    with torch.enable_grad():
        x_adv = apgd_bce(model, x_padded, y_padded, eps, num_classes)
    with torch.no_grad():
        logits = model(x_adv)
    x_adv = unpad(x_adv, orig_shape)
    logits = unpad(logits.float(), orig_shape)

    slice_2d = x_adv[0, 0, z].detach().cpu().numpy()
    pred_vol = logits.argmax(dim=1).squeeze(0).cpu().numpy()
    pred_mask = pred_vol[z] > 0

    gt = np.asarray(seg_np).squeeze().astype(np.int64)
    if gt.ndim == 4:
        gt = gt[0]
    gt_mask = gt[z] > 0

    del x, y, x_padded, y_padded, x_adv, logits
    if device.type == "cuda":
        torch.cuda.empty_cache()
    return slice_2d, pred_mask, gt_mask


def plot_dataset(ds_key: str, device: torch.device) -> None:
    ds_cfg = DATASETS[ds_key]
    data_dir = os.path.join(
        NNUNET_PATHS,
        "nnUNet_preprocessed",
        ds_cfg["dataset_name"],
        ds_cfg["data_subdir"],
    )
    case_id = WORST_CASE_ID[ds_key]
    print(f"{ds_key}: worst case = {case_id}")

    ckpt = os.path.join(
        NNUNET_PATHS,
        "nnUNet_results",
        ds_cfg["dataset_name"],
        TRAINERS["unet"],
        "fold_all",
        "checkpoint_final.pth",
    )
    model, div_factors, num_classes = build_model(ckpt, device)
    model.eval()

    _, seg_np, _ = load_sample(data_dir, case_id)
    gt = np.asarray(seg_np).squeeze().astype(np.int64)
    if gt.ndim == 4:
        gt = gt[0]
    z = find_best_slice(gt)

    results = [
        attacked_slice(
            model, div_factors, num_classes, data_dir, case_id, z, eps, device
        )
        for eps in EPSILONS
    ]
    del model
    if device.type == "cuda":
        torch.cuda.empty_cache()

    slices = [r[0] for r in results]
    pred_masks = [r[1] for r in results]
    gt_mask = results[0][2]  # GT doesn't depend on eps

    clean_vmin, clean_vmax = np.percentile(slices[0][slices[0] > 0], [1, 99])
    deltas = [slc - slices[0] for slc in slices]

    fig, axes = plt.subplots(2, len(EPSILONS), figsize=(4 * len(EPSILONS), 8.5))
    for col, (eps, slc, pred_mask, delta) in enumerate(
        zip(EPSILONS, slices, pred_masks, deltas)
    ):
        ax_pred = axes[0][col]
        ax_pred.imshow(
            slc, cmap="gray", vmin=clean_vmin, vmax=clean_vmax, origin="lower", zorder=0
        )
        ax_pred.imshow(mask_rgba(gt_mask, GT_COLOR), origin="lower", zorder=1)
        ax_pred.imshow(mask_rgba(pred_mask, PRED_COLOR), origin="lower", zorder=2)
        dice = slice_dice(gt_mask, pred_mask)
        ax_pred.set_title(
            f"ε = {round(eps * 255)}/255  |  Dice = {dice:.3f}",
            fontsize=10.5,
            fontweight="bold",
        )
        ax_pred.axis("off")

        ax_box = axes[1][col]
        ax_box.boxplot(
            delta.ravel(),
            vert=True,
            widths=0.5,
            patch_artist=True,
            boxprops={"facecolor": "#8CA6DB", "edgecolor": "black"},
            medianprops={"color": "black"},
            flierprops={"markersize": 2, "alpha": 0.3},
        )
        ax_box.set_xticks([])
        ax_box.set_ylabel("adversarial noise δ (x_adv − x_clean)" if col == 0 else "")
        ax_box.text(
            0.5,
            0.98,
            f"[{delta.min():.3f}, {delta.max():.3f}]",
            transform=ax_box.transAxes,
            ha="center",
            va="top",
            fontsize=9,
        )
        ax_box.axhline(0, color="gray", lw=0.8, ls=":")
        ax_box.grid(True, axis="y", alpha=0.3)

    ymin = min(d.min() for d in deltas)
    ymax = max(d.max() for d in deltas)
    pad = 0.05 * (ymax - ymin)
    for col in range(len(EPSILONS)):
        axes[1][col].set_ylim(ymin - pad, ymax + pad)

    fig.suptitle(
        f"{DATASET_LABELS[ds_key]} — worst-case perturbation magnitude vs. APGD ε "
        f"(case {case_id})",
        fontsize=14,
        fontweight="bold",
    )
    legend_patches = [
        mpatches.Patch(color=GT_COLOR[:3], label="Ground truth"),
        mpatches.Patch(color=PRED_COLOR[:3], label="Model prediction"),
    ]
    fig.legend(
        handles=legend_patches,
        loc="lower center",
        ncol=2,
        fontsize=9.5,
        frameon=True,
        bbox_to_anchor=(0.5, -0.005),
    )
    fig.tight_layout(rect=(0, 0, 1, 0.95))

    os.makedirs(OUT_DIR, exist_ok=True)
    out_path = os.path.join(OUT_DIR, f"{ds_key}_pixel_range_epsilon.png")
    fig.savefig(
        out_path, dpi=300, bbox_inches="tight", facecolor="white", edgecolor="none"
    )
    plt.close(fig)
    print(f"Saved: {out_path}")


def main() -> None:
    plt.rcParams.update(
        {
            "font.family": "sans-serif",
            "font.sans-serif": ["DejaVu Sans"],
            "figure.dpi": 150,
        }
    )
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")
    for ds_key in DATASET_LABELS:
        plot_dataset(ds_key, device)


if __name__ == "__main__":
    main()
