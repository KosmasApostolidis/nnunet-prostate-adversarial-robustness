"""Qualitative clean-vs-PGD-AT comparison for the top-3 failure cases per dataset.

Reads the .npz volumes written by generate_top_failure_predictions.py and
renders, per dataset, a 3 (case) x 4 (Clean overlay | Clean confidence
heatmap | PGD-AT overlay | PGD-AT confidence heatmap) grid: the best
foreground slice with a TP/FN/FP overlay (matching create_worst_case_grid.py's
compositing) next to the model's own per-pixel softmax confidence in its
predicted class, plus the macro Dice each model achieved under APGD @
epsilon=16/255 on that same case.

Usage::

    python experiments/plot_top_failure_cases.py
"""

from __future__ import annotations

import os
import sys

import matplotlib.colors as mcolors
import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from experiments.generate_top_failure_predictions import top_cases

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
PRED_DIR = os.path.join(REPO_ROOT, "results", "adv_rob_eval_results", "top_failure_predictions")
OUT_DIR = os.path.join(REPO_ROOT, "results", "adv_rob_eval_results", "figures")

DATASET_LABELS = {"wg": "WG", "zones": "Zones"}
N_TOP = 3

FN_COLOR = np.array([0.20, 0.47, 0.84, 0.55])  # blue   — GT not captured
FP_COLOR = np.array([0.96, 0.42, 0.17, 0.55])  # orange — spurious prediction
TP_COLOR = np.array([0.18, 0.80, 0.44, 0.65])  # green  — correct overlap


def find_best_slice(gt_vol: np.ndarray) -> int:
    mask = gt_vol > 0
    return int(np.argmax(mask.sum(axis=(1, 2))))


def build_overlay_rgba(gt_mask: np.ndarray, pred_mask: np.ndarray) -> np.ndarray:
    overlay = np.zeros((*gt_mask.shape, 4), dtype=np.float64)
    overlay[gt_mask & ~pred_mask] = FN_COLOR
    overlay[~gt_mask & pred_mask] = FP_COLOR
    overlay[gt_mask & pred_mask] = TP_COLOR
    return overlay


def render_slice(
    ax, img_slice: np.ndarray, gt_mask: np.ndarray, pred_mask: np.ndarray, title: str
) -> None:
    vmin, vmax = np.percentile(img_slice[img_slice > 0], [5, 99.5])
    ax.imshow(img_slice, cmap="gray", vmin=vmin, vmax=vmax, origin="lower", zorder=0)
    ax.imshow(build_overlay_rgba(gt_mask, pred_mask), origin="lower", zorder=1)
    ax.set_title(title, fontsize=10, fontweight="bold", pad=4)
    ax.axis("off")


def render_confidence(
    ax, img_slice: np.ndarray, confidence_slice: np.ndarray, title: str
) -> None:
    """Per-pixel softmax confidence in the model's own predicted class.

    Bright = the model was sure of its prediction there (whether or not
    that prediction was correct); dark = the model was unsure.
    """
    vmin, vmax = np.percentile(img_slice[img_slice > 0], [5, 99.5])
    ax.imshow(img_slice, cmap="gray", vmin=vmin, vmax=vmax, origin="lower", zorder=0)

    cmap = plt.get_cmap("plasma")
    norm = mcolors.Normalize(vmin=0.0, vmax=1.0)
    rgba = cmap(norm(confidence_slice))
    rgba[..., 3] = 0.6
    ax.imshow(rgba, origin="lower", zorder=1)

    sm = plt.cm.ScalarMappable(cmap=cmap, norm=norm)
    cbar = plt.colorbar(sm, ax=ax, fraction=0.046, pad=0.04)
    cbar.set_label("confidence", fontsize=6.5)
    cbar.ax.tick_params(labelsize=6)

    ax.set_title(title, fontsize=10, fontweight="bold", pad=4)
    ax.axis("off")


def plot_dataset(ds_key: str) -> None:
    cases = top_cases(ds_key)
    fig, axes = plt.subplots(N_TOP, 4, figsize=(18, 4.2 * N_TOP))
    fig.subplots_adjust(
        wspace=0.15, hspace=0.25, left=0.045, right=0.99, bottom=0.08, top=0.9
    )

    for row, case_id in enumerate(cases):
        for pair, (variant, label) in enumerate(
            (("clean", "Clean"), ("advft", "PGD-AT"))
        ):
            data = np.load(os.path.join(PRED_DIR, f"{ds_key}_{variant}_{case_id}.npz"))
            img, gt, pred, confidence, dice = (
                data["img"],
                data["gt"],
                data["pred"],
                data["confidence"],
                float(data["dice"]),
            )
            z = find_best_slice(gt)
            gt_mask, pred_mask = gt[z] > 0, pred[z] > 0
            overlay_col, heatmap_col = 2 * pair, 2 * pair + 1
            render_slice(
                axes[row][overlay_col],
                img[z],
                gt_mask,
                pred_mask,
                f"{label}  |  Dice = {dice:.3f}",
            )
            render_confidence(
                axes[row][heatmap_col],
                img[z],
                confidence[z],
                f"{label}  |  confidence",
            )
        axes[row][0].text(
            -0.05,
            0.5,
            f"Case {row + 1}",
            transform=axes[row][0].transAxes,
            rotation=90,
            va="center",
            ha="right",
            fontsize=11,
            fontweight="bold",
        )

    fig.suptitle(
        f"{DATASET_LABELS[ds_key]} — Top-{N_TOP} clean-model APGD failures rescued by PGD-AT "
        r"($\varepsilon=16/255$)",
        fontsize=13,
        fontweight="bold",
        y=0.985,
    )
    legend_patches = [
        mpatches.Patch(color=FN_COLOR[:3], label="False Negative (GT missed)"),
        mpatches.Patch(color=FP_COLOR[:3], label="False Positive (spurious)"),
        mpatches.Patch(color=TP_COLOR[:3], label="True Positive (match)"),
    ]
    fig.legend(
        handles=legend_patches,
        loc="lower center",
        ncol=3,
        fontsize=9.5,
        frameon=True,
        bbox_to_anchor=(0.5, -0.005),
    )

    os.makedirs(OUT_DIR, exist_ok=True)
    out_path = os.path.join(OUT_DIR, f"{ds_key}_top_failures.png")
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
    for ds_key in DATASET_LABELS:
        plot_dataset(ds_key)


if __name__ == "__main__":
    main()
