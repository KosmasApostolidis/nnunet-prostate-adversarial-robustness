"""Clean-vs-PGD-AT predictions for the top-3 UNet failure cases per dataset.

Selects, from the existing ``{ds}_unet[_advft]_percase.csv`` files, the 3 cases
where the clean UNet model fails most completely under APGD @ epsilon=16/255
(Dice=0) and PGD-AT fine-tuning rescues it the most (largest advft Dice).
Re-runs the same APGD-BCE attack
(``experiments.adv_rob_eval_nnunet_nnunetrecenc.apgd_bce``) against each
model's own fold_all checkpoint to reproduce the reported prediction, and
saves GT/pred/confidence/image volumes as ``.npz`` for
``plot_top_failure_cases.py``.

Usage::

    python experiments/generate_top_failure_predictions.py
"""

from __future__ import annotations

import csv
import os
import sys

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
from mri_prostate_seg.metrics.segmentation import class_metrics_triple

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RESULTS_DIR = os.path.join(REPO_ROOT, "results", "adv_rob_eval_results")
DATA_DIR = os.path.join(RESULTS_DIR, "data")
OUT_DIR = os.path.join(RESULTS_DIR, "top_failure_predictions")
EPS = 16 / 255
N_TOP = 3


def load_dice(path: str) -> dict[str, float]:
    d = {}
    with open(path, newline="") as f:
        for r in csv.DictReader(f):
            if r["attack"] == "APGD" and abs(float(r["epsilon"]) - EPS) < 1e-4:
                d[r["case_id"]] = float(r["dice"])
    return d


def top_cases(ds_key: str) -> list[str]:
    """Worst clean-model Dice first, tie-broken by the largest PGD-AT rescue."""
    clean = load_dice(os.path.join(DATA_DIR, f"{ds_key}_unet_percase.csv"))
    advft = load_dice(os.path.join(DATA_DIR, f"{ds_key}_unet_advft_percase.csv"))
    common = sorted(set(clean) & set(advft))
    ranked = sorted(common, key=lambda c: (clean[c], -advft[c]))
    return ranked[:N_TOP]


def predict(model, div_factors, num_classes, data_dir, case_id, device):
    img_np, seg_np, spacing = load_sample(data_dir, case_id)
    x = torch.from_numpy(img_np[np.newaxis]).to(device)
    y = torch.from_numpy(seg_np[np.newaxis].astype(np.float32)).to(device)
    x_padded, orig_shape = pad_to_divisible(x, div_factors)
    y_padded, _ = pad_to_divisible(y, div_factors)

    with torch.enable_grad():
        x_adv = apgd_bce(model, x_padded, y_padded, EPS, num_classes)
    with torch.no_grad():
        logits = model(x_adv)
    logits = unpad(logits.float(), orig_shape)
    probs = torch.softmax(logits, dim=1)
    pred = logits.argmax(dim=1).squeeze(0).cpu().numpy().astype(np.int64)
    confidence = probs.max(dim=1).values.squeeze(0).cpu().numpy().astype(np.float32)

    gt = np.asarray(seg_np).squeeze().astype(np.int64)
    if gt.ndim == 4:
        gt = gt[0]
    img = img_np[0].astype(np.float32)

    del x, y, x_padded, y_padded, x_adv, logits, probs
    if device.type == "cuda":
        torch.cuda.empty_cache()
    return img, gt, pred, confidence, spacing


def macro_dice(gt: np.ndarray, pred: np.ndarray, num_classes: int, spacing) -> float:
    dices = []
    for c in range(1, num_classes):
        dice, _, _ = class_metrics_triple(
            (pred == c).astype(np.uint8),
            (gt == c).astype(np.uint8),
            voxel_spacing=spacing,
        )
        if np.isfinite(dice):
            dices.append(dice)
    return float(np.mean(dices)) if dices else float("nan")


def main() -> None:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")
    os.makedirs(OUT_DIR, exist_ok=True)

    for ds_key in ("wg", "zones"):
        ds_cfg = DATASETS[ds_key]
        data_dir = os.path.join(
            NNUNET_PATHS,
            "nnUNet_preprocessed",
            ds_cfg["dataset_name"],
            ds_cfg["data_subdir"],
        )
        cases = top_cases(ds_key)
        print(f"\n{ds_key}: top-{N_TOP} cases = {cases}")

        for variant, trainer_key in (("clean", "unet"), ("advft", "unet_advft")):
            trainer_name = TRAINERS[trainer_key]
            ckpt = os.path.join(
                NNUNET_PATHS,
                "nnUNet_results",
                ds_cfg["dataset_name"],
                trainer_name,
                "fold_all",
                "checkpoint_final.pth",
            )
            print(f"  [{variant}] loading {ckpt}")
            model, div_factors, num_classes = build_model(ckpt, device)
            model.eval()

            for case_id in cases:
                img, gt, pred, confidence, spacing = predict(
                    model, div_factors, num_classes, data_dir, case_id, device
                )
                dice = macro_dice(gt, pred, num_classes, spacing)
                print(f"    {case_id}: macro Dice = {dice:.4f}")
                out_path = os.path.join(OUT_DIR, f"{ds_key}_{variant}_{case_id}.npz")
                np.savez_compressed(
                    out_path,
                    img=img,
                    gt=gt,
                    pred=pred,
                    confidence=confidence,
                    spacing=np.array(spacing),
                    dice=dice,
                )

            del model
            if device.type == "cuda":
                torch.cuda.empty_cache()

    print(f"\nDone. Predictions saved under {OUT_DIR}")


if __name__ == "__main__":
    main()
