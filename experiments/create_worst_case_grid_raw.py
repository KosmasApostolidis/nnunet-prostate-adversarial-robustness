"""Create the 3x3 worst-case grid on the original (un-preprocessed) T2W slices.

Same predictions and ground truth as ``create_worst_case_grid.py``; only the
greyscale background changes. nnU-Net's preprocessing for both cohorts is a
crop plus a z-score, with no resampling (the cohorts are already at the target
3.0 x 0.5 x 0.5 mm spacing), so the predictions can be pasted back into the
original field of view exactly, with no interpolation of the masks.
"""

from __future__ import annotations

import glob
import json
import os
import pickle
import re

import matplotlib

matplotlib.use("Agg")

import matplotlib.patches as mpatches  # noqa: E402
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
import nibabel as nib  # noqa: E402
import SimpleITK as sitk  # noqa: E402

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
PRED_DIR = os.path.join(
    REPO_ROOT,
    "results",
    "wg_zones_adversarial_robustness_results",
    "single_case_predictions",
)
RESULTS_DIR = os.path.dirname(PRED_DIR)
NNUNET_BASE = os.path.join(REPO_ROOT, "nnUnet_paths")
EPSILON = 0.1

DATASET_DIRS = {
    "wg": "Dataset016_WgSegmentationPNetAndPicai",
    "zones": "Dataset019_ProstateZonesSegmentationWgFilteredLessDilated",
}

ATTACKS = ["fgsm", "pgd", "a_pgd"]
CLASSES = ["WG", "TZCZ", "PZ"]
CLASS_LABELS = {"WG": "WG", "TZCZ": "TZ+CZ", "PZ": "PZ"}
CLASS_IDX = {"WG": 1, "TZCZ": 1, "PZ": 2}
ATTACK_LABELS = {"fgsm": "FGSM", "pgd": "PGD", "a_pgd": "APGD"}

FN_COLOR = np.array([0.20, 0.47, 0.84, 0.55])  # blue   — GT not captured
FP_COLOR = np.array([0.96, 0.42, 0.17, 0.55])  # orange — spurious prediction
TP_COLOR = np.array([0.18, 0.80, 0.44, 0.65])  # green  — correct overlap

# Margin in pixels kept around the prostate bounding box. Full 256x256 renders
# the zonal structures too small at column width.
MARGIN = 36


def load_pkl(model_key: str, case_id: str) -> dict:
    path = os.path.join(
        NNUNET_BASE,
        "nnUNet_preprocessed",
        DATASET_DIRS[model_key],
        "nnUNetPlans_3d_fullres",
        f"{case_id}.pkl",
    )
    with open(path, "rb") as handle:
        return pickle.load(handle)


def load_raw_volume(model_key: str, case_id: str) -> np.ndarray:
    """Original T2W volume for display.

    Dataset019's images are the whole-gland images multiplied by a dilated WG
    mask, so ~91% of each zonal volume is zero. For the 445 patients present in
    both cohorts the unmasked acquisition is available in Dataset016 under the
    same identifier suffix, with identical shape, spacing and origin, and with
    identical intensities wherever the masked volume is non-zero. Prefer it, so
    the zonal panels show the same unprocessed image as the WG panel.
    """
    if model_key == "zones":
        match = re.search(r"(\d{20,})$", case_id)
        if match:
            unmasked = os.path.join(
                NNUNET_BASE,
                "nnUNet_raw",
                DATASET_DIRS["wg"],
                "imagesTr",
                f"ProstateWG_{match.group(1)}_0000.nii.gz",
            )
            if os.path.exists(unmasked):
                return sitk.GetArrayFromImage(sitk.ReadImage(unmasked)).astype(
                    np.float32
                )

    path = os.path.join(
        NNUNET_BASE,
        "nnUNet_raw",
        DATASET_DIRS[model_key],
        "imagesTr",
        f"{case_id}_0000.nii.gz",
    )
    return sitk.GetArrayFromImage(sitk.ReadImage(path)).astype(np.float32)


def load_raw_label(model_key: str, case_id: str) -> np.ndarray:
    path = os.path.join(
        NNUNET_BASE,
        "nnUNet_raw",
        DATASET_DIRS[model_key],
        "labelsTr",
        f"{case_id}.nii.gz",
    )
    return sitk.GetArrayFromImage(sitk.ReadImage(path)).astype(np.int16)


def paste_back(volume: np.ndarray, pkl: dict, fill: int = 0) -> np.ndarray:
    """Place a cropped volume back into the original field of view."""
    (z0, z1), (y0, y1), (x0, x1) = pkl["bbox_used_for_cropping"]
    full = np.full(tuple(pkl["shape_before_cropping"]), fill, dtype=volume.dtype)
    full[z0:z1, y0:y1, x0:x1] = volume
    return full


def dice_3d(gt_mask: np.ndarray, pred_mask: np.ndarray) -> float:
    denominator = gt_mask.sum() + pred_mask.sum()
    if denominator == 0:
        return float("nan")
    return 2.0 * np.logical_and(gt_mask, pred_mask).sum() / denominator


def reported_dice(model_key: str, attack: str, case_id: str, class_label: str) -> float:
    """This patient's own epsilon=0.1 Dice from the evaluation run behind Table I.

    Printed next to the plotted Dice as a drift check: the panels are
    regenerated here, so the iterative attacks cannot be expected to land on
    the evaluation's exact adversarial example.
    """
    pattern = os.path.join(
        RESULTS_DIR, f"{model_key}_{attack}_fold*_per_sample.csv"
    )
    for path in sorted(glob.glob(pattern)):
        frame = pd.read_csv(path)
        row = frame[
            (frame["case_id"] == case_id)
            & (frame["class"] == class_label)
            & np.isclose(frame["epsilon"], EPSILON)
        ]
        if not row.empty:
            return float(row["dice"].iloc[0])
    raise LookupError(
        f"no epsilon={EPSILON} result for {case_id} / {attack} / {class_label}"
    )


def build_overlay_rgba(gt_mask: np.ndarray, pred_mask: np.ndarray) -> np.ndarray:
    overlay = np.zeros((*gt_mask.shape, 4), dtype=np.float64)
    overlay[gt_mask & ~pred_mask] = FN_COLOR
    overlay[~gt_mask & pred_mask] = FP_COLOR
    overlay[gt_mask & pred_mask] = TP_COLOR
    return overlay


def shared_slice(masks_by_class: dict[str, np.ndarray]) -> int:
    """Axial level that represents every class well.

    Each class's per-slice area is scaled by its own maximum, and the slice
    maximising the smallest of those scaled areas is returned. Taking the plain
    largest-area slice would follow the whole gland and can land where the thin
    peripheral zone is barely present.
    """
    scaled = []
    for mask in masks_by_class.values():
        area = mask.sum(axis=(1, 2)).astype(np.float64)
        peak = area.max()
        scaled.append(area / peak if peak > 0 else area)
    return int(np.argmax(np.min(np.stack(scaled), axis=0)))


def shared_crop_window(masks: list[np.ndarray], shape: tuple) -> tuple:
    """One row/column window covering every mask, clipped to the image."""
    union = np.zeros(shape, dtype=bool)
    for mask in masks:
        union |= mask
    ys, xs = np.where(union)
    if ys.size == 0:
        return slice(0, shape[0]), slice(0, shape[1])
    y0 = max(int(ys.min()) - MARGIN, 0)
    y1 = min(int(ys.max()) + MARGIN + 1, shape[0])
    x0 = max(int(xs.min()) - MARGIN, 0)
    x1 = min(int(xs.max()) + MARGIN + 1, shape[1])
    return slice(y0, y1), slice(x0, x1)


def render_slice(ax, img_slice, gt_mask, pred_mask, title) -> None:
    finite = img_slice[np.isfinite(img_slice)]
    vmin, vmax = np.percentile(finite, [1.0, 99.0])
    ax.imshow(img_slice, cmap="gray", vmin=vmin, vmax=vmax, origin="lower", zorder=0)
    ax.imshow(build_overlay_rgba(gt_mask, pred_mask), origin="lower", zorder=1)
    ax.set_title(title, fontsize=11, fontweight="bold", pad=3)
    ax.axis("off")


def main() -> None:
    with open(os.path.join(PRED_DIR, "cases_metadata.json")) as handle:
        meta = json.load(handle)

    plt.rcParams.update(
        {
            "font.family": "sans-serif",
            "font.sans-serif": ["DejaVu Sans"],
            "figure.dpi": 150,
        }
    )

    fig, axes = plt.subplots(3, 3, figsize=(14, 11.2))
    fig.subplots_adjust(
        wspace=0.04, hspace=0.08, left=0.01, right=0.99, bottom=0.04, top=0.95
    )

    panels: list[dict] = []
    for row_idx, attack in enumerate(ATTACKS):
        for col_idx, cls_name in enumerate(CLASSES):
            ax = axes[row_idx, col_idx]
            cls_idx = CLASS_IDX[cls_name]
            entry = next(
                (
                    e
                    for e in meta
                    if e["attack"] == attack
                    and e["class_name"].replace("+", "") == cls_name
                ),
                None,
            )
            if entry is None:
                ax.text(
                    0.5,
                    0.5,
                    "N/A",
                    ha="center",
                    va="center",
                    transform=ax.transAxes,
                    fontsize=14,
                )
                ax.axis("off")
                continue

            model_key, case_id = entry["model_key"], entry["case_id"]
            pkl = load_pkl(model_key, case_id)

            # Written with nibabel in (D, H, W) order, so read them back with
            # nibabel; SimpleITK would reverse the axes.
            gt_pre = (
                nib.load(os.path.join(PRED_DIR, os.path.basename(entry["gt_path"])))
                .get_fdata()
                .astype(np.int16)
            )
            pred_pre = (
                nib.load(os.path.join(PRED_DIR, os.path.basename(entry["pred_path"])))
                .get_fdata()
                .astype(np.int16)
            )

            raw = load_raw_volume(model_key, case_id)
            gt_full = paste_back(gt_pre, pkl)
            pred_full = paste_back(pred_pre, pkl)

            # Alignment check: the pasted ground truth must reproduce the raw
            # label map exactly. This validates the bounding box, the slice
            # offset, and the axis order in one comparison.
            raw_label = load_raw_label(model_key, case_id)
            matches = np.array_equal(gt_full == cls_idx, raw_label == cls_idx)
            print(
                f"{attack:6} {CLASS_LABELS[cls_name]:6} "
                f"gt-vs-raw-label exact: {matches}"
            )

            # Titles quote the Dice of the masks actually drawn, so no panel
            # can print a number its own overlay contradicts.
            dice_val = dice_3d(gt_pre == cls_idx, pred_pre == cls_idx)
            print(
                f"{attack:6} {CLASS_LABELS[cls_name]:6} "
                f"dice plotted={dice_val:.3f}  evaluated="
                f"{reported_dice(model_key, attack, case_id, CLASS_LABELS[cls_name]):.3f}"
            )

            panels.append(
                {
                    "ax": ax,
                    "class": cls_name,
                    "raw": raw,
                    "gt": gt_full == cls_idx,
                    "pred": pred_full == cls_idx,
                    "title": f"{ATTACK_LABELS[attack]} | {CLASS_LABELS[cls_name]}  (Dice = {dice_val:.3f})",
                }
            )

    # One axial level for the whole grid. Both cohorts share the patient's
    # original geometry, so the ground-truth volumes are already co-registered.
    z = shared_slice({p["class"]: p["gt"] for p in panels})
    print(f"shared axial slice: z = {z}")

    # One window too, so every panel shares a field of view and a scale.
    rows, cols = shared_crop_window(
        [p["gt"][z] for p in panels], panels[0]["raw"][z].shape
    )
    for panel in panels:
        render_slice(
            panel["ax"],
            panel["raw"][z][rows, cols],
            panel["gt"][z][rows, cols],
            panel["pred"][z][rows, cols],
            panel["title"],
        )

    fig.suptitle(
        "Adversarial Robustness at $\\varepsilon = 0.1$ — One Representative Patient",
        fontsize=14,
        fontweight="bold",
        y=0.995,
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
        fontsize=10.5,
        frameon=True,
        bbox_to_anchor=(0.5, 0.004),
    )

    out_path = os.path.join(PRED_DIR, "worst_case_3x3_grid_raw.png")
    fig.savefig(
        out_path, dpi=300, bbox_inches="tight", facecolor="white", edgecolor="none"
    )
    print(f"Saved: {out_path}")
    plt.close(fig)


if __name__ == "__main__":
    main()
