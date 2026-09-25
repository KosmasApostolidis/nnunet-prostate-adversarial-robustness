"""Generate predicted segmentations for the 9 representative worst-case
(attack x model) combinations at epsilon=0.1.
"""

from __future__ import annotations

import json
import os
import pickle
import sys

import nibabel as nib
import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from fgsm_adversarial_evaluation import (
    MODEL_CONFIGS,
    dice_ce_loss,
    load_model,
    pad_to_divisible,
    samplewise_bounds,
    unpad,
)

# Table I's APGD is a_pgd_campaign, not a_pgd; see that module's docstring.
# The two differ in strength, so the figure must use the campaign one.
from mri_prostate_seg.attacks.a_pgd_campaign import a_pgd_attack
from mri_prostate_seg.attacks.pgd import _project_linf
from mri_prostate_seg.attacks.fgsm import fgsm_attack
from mri_prostate_seg.attacks.pgd import pgd_with_restarts
from mri_prostate_seg.utils.seeding import set_seed

import blosc2

CASES = [
    {
        "case_id": "ProstateWG_10552",
        "fold": 3,
        "attack": "fgsm",
        "model": "wg",
        "class_name": "WG",
        "target_dice": 0.814,
    },
    {
        "case_id": "ProstateWG_10913",
        "fold": 1,
        "attack": "pgd",
        "model": "wg",
        "class_name": "WG",
        "target_dice": 0.548,
    },
    {
        "case_id": "ProstateWG_75475649828150579970034301886067519673",
        "fold": 1,
        "attack": "a_pgd",
        "model": "wg",
        "class_name": "WG",
        "target_dice": 0.488,
    },
    {
        "case_id": "ProstateZonesFilteredLessDilated_ProstateZones_68774001862807440234041763844337454679",
        "fold": 4,
        "attack": "fgsm",
        "model": "zones",
        "class_name": "TZ+CZ",
        "target_dice": 0.849,
    },
    {
        "case_id": "ProstateZonesFilteredLessDilated_ProstateZones_88753932021518240523885925625518043086",
        "fold": 0,
        "attack": "pgd",
        "model": "zones",
        "class_name": "TZ+CZ",
        "target_dice": 0.638,
    },
    {
        "case_id": "ProstateZonesFilteredLessDilated_ProstateZones_208619811284332587660445080721806168019",
        "fold": 1,
        "attack": "a_pgd",
        "model": "zones",
        "class_name": "TZ+CZ",
        "target_dice": 0.142,
    },
    {
        "case_id": "ProstateZonesFilteredLessDilated_ProstateZones_119",
        "fold": 3,
        "attack": "fgsm",
        "model": "zones",
        "class_name": "PZ",
        "target_dice": 0.765,
    },
    {
        "case_id": "ProstateZonesFilteredLessDilated_ProstateZones_101",
        "fold": 2,
        "attack": "pgd",
        "model": "zones",
        "class_name": "PZ",
        "target_dice": 0.468,
    },
    {
        "case_id": "ProstateZonesFilteredLessDilated_ProstateZones_113",
        "fold": 0,
        "attack": "a_pgd",
        "model": "zones",
        "class_name": "PZ",
        "target_dice": 0.061,
    },
]

OUTPUT_DIR = os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
    "results",
    "wg_zones_adversarial_robustness_results",
    "worst_case_predictions",
)

NNUNET_BASE = os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
    "nnUnet_paths",
)

CONFIG_DIR = os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
    "configs",
    "attacks",
)

EPS = 0.1
SEED = 42

# Load the same hyperparameters the main evaluation uses, so the qualitative
# figure cannot drift from the attack configuration reported in the paper.
ATTACK_CFG = {}
for _name in ("pgd", "a_pgd"):
    with open(os.path.join(CONFIG_DIR, f"{_name}.json")) as _f:
        ATTACK_CFG[_name] = json.load(_f)


def resolve_data_dir(cfg):
    """Rebase the config's data_dir onto this checkout's nnUnet_paths."""
    tail = cfg["data_dir"].split("nnUnet_paths" + os.sep, 1)[-1]
    return os.path.join(NNUNET_BASE, tail)


def load_data(cfg, case_id):
    data_dir = resolve_data_dir(cfg)
    b2nd_path = os.path.join(data_dir, f"{case_id}.b2nd")
    seg_b2nd_path = os.path.join(data_dir, f"{case_id}_seg.b2nd")
    pkl_path = os.path.join(data_dir, f"{case_id}.pkl")
    img_data = blosc2.open(urlpath=b2nd_path, mode="r")
    seg_data = blosc2.open(urlpath=seg_b2nd_path, mode="r")
    img = np.asarray(img_data[:], dtype=np.float32)
    seg = np.asarray(seg_data[:], dtype=np.int64)
    with open(pkl_path, "rb") as f:
        meta = pickle.load(f)
    return img, seg, meta


def run_attack(
    network, x_padded, y_padded, eps, num_classes, attack_type, x_bounds, device
):
    cfg = ATTACK_CFG.get(attack_type, {})
    if attack_type == "fgsm":
        with torch.enable_grad():
            return fgsm_attack(
                network,
                x_padded,
                y_padded,
                eps,
                num_classes,
                dice_ce_loss,
                x_bounds=x_bounds,
                normalized_grad=False,
            )
    elif attack_type == "pgd":
        with torch.enable_grad():
            snaps = pgd_with_restarts(
                network,
                x_padded,
                y_padded,
                eps,
                num_classes,
                dice_ce_loss,
                n_steps=cfg["n_steps"],
                step_size=cfg["step_size"],
                momentum=cfg["momentum"],
                normalized_grad=cfg["normalized_grad"],
                x_bounds=x_bounds,
                snapshot_eps=[eps],
                num_restarts=cfg["num_restarts"],
            )
            return snaps[eps]
    elif attack_type == "a_pgd":
        # Mirrors run_adversarial_eval.py: one call at max_eps, and the
        # eps == max_eps snapshot is the returned iterate with no further
        # projection, which the attack has already applied.
        with torch.enable_grad():
            return a_pgd_attack(
                network,
                x_padded,
                y_padded,
                eps,
                num_classes,
                dice_ce_loss,
                n_steps=cfg["n_steps"],
                step_size=cfg["step_size"],
                random_start=cfg["random_start"],
                momentum=cfg["momentum"],
                x_bounds=x_bounds,
                adaptation_rate=cfg["adaptation_rate"],
                checkpoints=cfg["checkpoints"],
            )
    else:
        raise ValueError(f"Unknown attack: {attack_type}")


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    metadata_entries = []

    for i, case in enumerate(CASES):
        case_id = case["case_id"]
        model_key = case["model"]
        attack_type = case["attack"]
        fold = case["fold"]
        class_name = case["class_name"]
        cfg = MODEL_CONFIGS[model_key]

        print(
            f"\n[{i + 1}/9] {attack_type} x {class_name}  case={case_id}  fold={fold}"
        )

        checkpoint_path = os.path.join(
            NNUNET_BASE,
            "nnUNet_results",
            cfg["dataset_folder"],
            cfg["trainer"],
            f"fold_{fold}",
            "checkpoint_final.pth",
        )
        network, div_factors = load_model(
            checkpoint_path, cfg["configuration"], cfg["num_classes"], device
        )
        network.eval()

        img_np, seg_np, meta = load_data(cfg, case_id)

        # Seed per case so each panel is reproducible independently of the
        # order in which the nine cases are generated.
        set_seed(SEED)

        x = torch.from_numpy(img_np[np.newaxis]).to(device)
        y = torch.from_numpy(seg_np[np.newaxis].astype(np.float32)).to(device)
        x_min, x_max = samplewise_bounds(x)
        x_padded, orig_shape = pad_to_divisible(x, div_factors)
        y_padded, _ = pad_to_divisible(y, div_factors)

        x_adv = run_attack(
            network,
            x_padded,
            y_padded,
            EPS,
            cfg["num_classes"],
            attack_type,
            (x_min, x_max),
            device,
        )

        with torch.no_grad():
            logits_padded = network(x_adv)
        logits = unpad(logits_padded.float(), orig_shape)
        pred = torch.argmax(logits, dim=1).squeeze(0).cpu().numpy()
        gt = seg_np[0].astype(np.int64)

        # The figure must show the input the model actually saw, so unpad the
        # exact adversarial tensor that produced `pred` rather than the clean
        # volume. `linf` verifies the saved image carries a real perturbation
        # inside the epsilon budget.
        adv_np = unpad(x_adv.float(), orig_shape)[0, 0].cpu().numpy()
        linf = float((x_adv - x_padded).abs().max())
        print(f"  perturbation Linf = {linf:.4f}  (eps = {EPS})")

        spacing = meta["sitk_stuff"]["spacing"]
        origin = meta["sitk_stuff"]["origin"]
        direction = np.array(meta["sitk_stuff"]["direction"]).reshape(3, 3)
        affine = np.eye(4)
        affine[:3, :3] = direction * np.array(spacing)
        affine[:3, 3] = origin

        safe_name = f"{attack_type}_{class_name.replace('+', '')}"

        gt_path = os.path.join(OUTPUT_DIR, f"{safe_name}_gt.nii.gz")
        nib.save(nib.Nifti1Image(gt.astype(np.float32), affine), gt_path)
        print(f"  GT -> {gt_path}")

        pred_path = os.path.join(OUTPUT_DIR, f"{safe_name}_pred.nii.gz")
        nib.save(nib.Nifti1Image(pred.astype(np.float32), affine), pred_path)
        print(f"  PRED -> {pred_path}")

        img_path = os.path.join(OUTPUT_DIR, f"{safe_name}_img.nii.gz")
        nib.save(nib.Nifti1Image(adv_np.astype(np.float32), affine), img_path)
        print(f"  ADV IMG -> {img_path}")

        clean_path = os.path.join(OUTPUT_DIR, f"{safe_name}_img_clean.nii.gz")
        nib.save(nib.Nifti1Image(img_np[0].astype(np.float32), affine), clean_path)

        metadata_entries.append(
            {
                "attack": attack_type,
                "class_name": class_name,
                "model_key": model_key,
                "case_id": case_id,
                "fold": fold,
                "target_dice": case["target_dice"],
                "gt_path": gt_path,
                "pred_path": pred_path,
                "img_path": img_path,
                "clean_img_path": clean_path,
                "perturbation_linf": linf,
                "spacing": list(spacing),
                "shape": list(gt.shape),
            }
        )

        del network, x, y, x_padded, y_padded, x_adv, logits
        if device.type == "cuda":
            torch.cuda.empty_cache()

    meta_json = os.path.join(OUTPUT_DIR, "cases_metadata.json")
    with open(meta_json, "w") as f:
        json.dump(metadata_entries, f, indent=2)
    print(f"\nMetadata -> {meta_json}")
    print("Done.")


if __name__ == "__main__":
    main()
