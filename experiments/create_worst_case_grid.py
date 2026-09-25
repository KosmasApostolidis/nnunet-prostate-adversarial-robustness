"""Create 3x3 grid: rows=FGSM/PGD/APGD, cols=WG/TZ+CZ/PZ.
Filled-region overlay — TP green, FN blue (GT missed), FP orange (pred hallucination).
"""

import json
import os

import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import nibabel as nib
import numpy as np

PRED_DIR = os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
    "results", "wg_zones_adversarial_robustness_results", "worst_case_predictions",
)

ATTACKS = ["fgsm", "pgd", "a_pgd"]
CLASSES = ["WG", "TZCZ", "PZ"]
CLASS_LABELS = {"WG": "WG", "TZCZ": "TZ+CZ", "PZ": "PZ"}

# WG model: class 1 = WG.  Zones model: class 1 = TZ+CZ, class 2 = PZ.
CLASS_IDX = {"WG": 1, "TZCZ": 1, "PZ": 2}

ATTACK_LABELS = {"fgsm": "FGSM", "pgd": "PGD", "a_pgd": "APGD"}

# Overlay colours — alpha-composited onto greyscale MRI
FN_COLOR = np.array([0.20, 0.47, 0.84, 0.55])  # blue   — GT not captured
FP_COLOR = np.array([0.96, 0.42, 0.17, 0.55])  # orange — spurious prediction
TP_COLOR = np.array([0.18, 0.80, 0.44, 0.65])  # green  — correct overlap


def load_nifti(path):
    # cases_metadata.json stores absolute paths from the machine that produced
    # it, so resolve every entry against the local prediction directory.
    img = nib.load(os.path.join(PRED_DIR, os.path.basename(path)))
    return img.get_fdata(), img.affine


def dice_3d(gt_mask, pred_mask):
    """Case-level 3D Dice of the masks actually plotted."""
    denominator = gt_mask.sum() + pred_mask.sum()
    if denominator == 0:
        return float("nan")
    return 2.0 * np.logical_and(gt_mask, pred_mask).sum() / denominator


def find_best_slice(gt_vol, cls_idx):
    mask = (gt_vol == cls_idx)
    return int(np.argmax(mask.sum(axis=(1, 2))))


def build_overlay_rgba(gt_mask, pred_mask):
    """Return (H, W, 4) RGBA overlay with TP / FN / FP regions.

    Compositing order (bottom to top): FN → FP → TP, so TP sits on top
    and each region gets its assigned colour without edge artifacts.
    """
    h, w = gt_mask.shape
    overlay = np.zeros((h, w, 4), dtype=np.float64)

    fn = gt_mask & ~pred_mask
    fp = ~gt_mask & pred_mask
    tp = gt_mask & pred_mask

    # Layer 1 — false negatives (GT missed by model)
    overlay[fn] = FN_COLOR[np.newaxis, :]

    # Layer 2 — false positives (model hallucinated outside GT)
    overlay[fp] = FP_COLOR[np.newaxis, :]

    # Layer 3 — true positives (agreement)
    overlay[tp] = TP_COLOR[np.newaxis, :]

    return overlay


def render_slice(ax, img_slice, gt_mask, pred_mask, title):
    # MRI background
    vmin, vmax = np.percentile(img_slice[img_slice > 0], [5, 99.5])
    ax.imshow(img_slice, cmap="gray", vmin=vmin, vmax=vmax,
              origin="lower", zorder=0)

    overlay = build_overlay_rgba(gt_mask, pred_mask)
    ax.imshow(overlay, origin="lower", zorder=1)

    ax.set_title(title, fontsize=10, fontweight="bold", pad=4)
    ax.axis("off")


def main():
    with open(os.path.join(PRED_DIR, "cases_metadata.json")) as f:
        meta = json.load(f)

    plt.rcParams.update({
        "font.family": "sans-serif",
        "font.sans-serif": ["DejaVu Sans"],
        "figure.dpi": 150,
    })

    fig, axes = plt.subplots(3, 3, figsize=(14, 13))
    fig.subplots_adjust(wspace=0.06, hspace=0.15,
                        left=0.02, right=0.98, bottom=0.06, top=0.93)

    for row_idx, atk in enumerate(ATTACKS):
        for col_idx, cls_name in enumerate(CLASSES):
            ax = axes[row_idx, col_idx]
            cls_idx = CLASS_IDX[cls_name]

            entry = next(
                (e for e in meta
                 if e["attack"] == atk and
                 e["class_name"].replace("+", "") == cls_name),
                None,
            )
            if entry is None:
                ax.text(0.5, 0.5, "N/A", ha="center", va="center",
                        transform=ax.transAxes, fontsize=14)
                ax.axis("off")
                continue

            gt_vol, _ = load_nifti(entry["gt_path"])
            pred_vol, _ = load_nifti(entry["pred_path"])
            img_vol, _ = load_nifti(entry["img_path"])

            z = find_best_slice(gt_vol, cls_idx)
            # Computed from the stored masks rather than read from
            # entry["target_dice"], which is a hard-coded selection target and
            # does not describe the prediction saved for this case.
            dice_val = dice_3d(gt_vol == cls_idx, pred_vol == cls_idx)

            render_slice(
                ax,
                img_vol[z],
                gt_vol[z] == cls_idx,
                pred_vol[z] == cls_idx,
                f"{ATTACK_LABELS[atk]} | {CLASS_LABELS[cls_name]}\nDice = {dice_val:.3f}",
            )

    fig.suptitle(
        "Adversarial Robustness at $\\varepsilon = 0.1$ — Representative Cases",
        fontsize=13, fontweight="bold", y=0.97,
    )

    legend_patches = [
        mpatches.Patch(color=FN_COLOR[:3], label="False Negative (GT missed)"),
        mpatches.Patch(color=FP_COLOR[:3], label="False Positive (spurious)"),
        mpatches.Patch(color=TP_COLOR[:3], label="True Positive (match)"),
    ]
    fig.legend(handles=legend_patches, loc="lower center",
               ncol=3, fontsize=9.5, frameon=True,
               bbox_to_anchor=(0.5, -0.01))

    out_path = os.path.join(PRED_DIR, "worst_case_3x3_grid.png")
    fig.savefig(out_path, dpi=300, bbox_inches="tight",
                facecolor="white", edgecolor="none")
    print(f"Saved: {out_path}")
    plt.close(fig)


if __name__ == "__main__":
    main()
