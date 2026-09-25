"""
FGSM White-Box Adversarial Robustness Evaluation for nnU-Net Segmentation Models.

Evaluates whole-gland (WG) on Dataset016 and prostate zones (TZ+CZ / PZ) on
Dataset019 under FGSM at multiple epsilon levels. Metrics: Dice, HD95, ASD.

Single-fold mode (default): one checkpoint (fold_k) and that fold's val split.
Use ``--all-folds`` to run all cross-validation folds: each fold's
``checkpoint_final.pth`` on ``splits_final.json[k]['val']``, write per-fold
CSVs/plots, a macro summary CSV (mean across folds of each fold's validation
mean), and figures with one curve per fold plus the macro average (± SEM across
folds).
"""

import os
import json
import csv
import gc
import math
import random
import argparse
import numpy as np
import torch
import torch.nn.functional as F
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from medpy.metric.binary import hd95, asd

from importlib import import_module


NNUNET_PATHS = os.environ.get(
    "NNUNET_PATHS",
    os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "nnUnet_paths"),
)
_GT_COLOR = np.array([0.12, 0.46, 0.70, 0.50])     # blue  (RGBA)
_PRED_COLOR = np.array([1.00, 0.50, 0.05, 0.50])   # orange (RGBA)

def _checkpoint_path(dataset_folder: str, trainer: str, fold: int) -> str:
    return os.path.join(
        NNUNET_PATHS,
        "nnUNet_results",
        dataset_folder,
        trainer,
        f"fold_{fold}",
        "checkpoint_final.pth",
    )


MODEL_CONFIGS = {
    "wg": {
        "name": "Whole Gland",
        "epsilons": [0, 0.02, 0.04, 0.06, 0.08, 0.1],
        "dataset_folder": "Dataset016_WgSegmentationPNetAndPicai",
        "trainer": "nnUNetTrainer__nnUNetPlans__3d_fullres",
        "data_dir": os.path.join(
            NNUNET_PATHS,
            "nnUNet_preprocessed",
            "Dataset016_WgSegmentationPNetAndPicai",
            "nnUNetPlans_3d_fullres",
        ),
        "splits_json": os.path.join(
            NNUNET_PATHS,
            "nnUNet_preprocessed",
            "Dataset016_WgSegmentationPNetAndPicai",
            "splits_final.json",
        ),
        "num_classes": 2,
        "class_names": ["WG"],
        "configuration": "3d_fullres",
    },
    "zones": {
        "name": "Prostate Zones",
        "epsilons": [0, 0.02, 0.04, 0.06, 0.08, 0.1],
        "dataset_folder": "Dataset019_ProstateZonesSegmentationWgFilteredLessDilated",
        "trainer": "nnUNetTrainer__nnUNetPlans__3d_fullres",
        "data_dir": os.path.join(
            NNUNET_PATHS,
            "nnUNet_preprocessed",
            "Dataset019_ProstateZonesSegmentationWgFilteredLessDilated",
            "nnUNetPlans_3d_fullres",
        ),
        "splits_json": os.path.join(
            NNUNET_PATHS,
            "nnUNet_preprocessed",
            "Dataset019_ProstateZonesSegmentationWgFilteredLessDilated",
            "splits_final.json",
        ),
        "num_classes": 3,
        "class_names": ["TZ+CZ", "PZ"],
        "configuration": "3d_fullres",
    },
}


def get_checkpoint_for_fold(cfg: dict, fold: int) -> str:
    return _checkpoint_path(cfg["dataset_folder"], cfg["trainer"], fold)

# nnU-Net preprocessed labels: -1 = ignore (outside crop)
IGNORE_LABEL = -1


def _resolve_class(fqn: str):
    """Import a class from its fully-qualified name, e.g. 'torch.nn.LeakyReLU'."""
    if fqn is None:
        return None
    module_path, cls_name = fqn.rsplit(".", 1)
    return getattr(import_module(module_path), cls_name)

def _strip_prefix(state_dict: dict) -> dict:
    """Drop torch.compile / DataParallel key prefixes so load_state_dict is strict."""
    out = {}
    for k, v in state_dict.items():
        for prefix in ("_orig_mod.", "module."):
            if k.startswith(prefix):
                k = k[len(prefix):]
        out[k] = v
    return out


def load_model(checkpoint_path: str, configuration: str, num_classes: int, device: torch.device):
    """Build the network named by the checkpoint's architecture spec.
    The class comes from the plans (PlainConvUNet for nnUNetPlans,
    ResidualEncoderUNet for the ResEnc plans), so any trained variant loads.
    Returns (network, divisibility_factors) where divisibility_factors is a list of
    per-spatial-dimension factors that input sizes must be divisible by.
    """
    try:
        ckpt = torch.load(checkpoint_path, map_location="cpu", weights_only=True)
    except Exception:
        # nnU-Net checkpoints embed plans/dataset_json; full unpickle required for those.
        ckpt = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    plans = ckpt["init_args"]["plans"]
    arch = plans["configurations"][configuration]["architecture"]
    net_cls = _resolve_class(arch["network_class_name"])
    arch_kwargs = dict(arch["arch_kwargs"])

    for key in arch.get("_kw_requires_import", []):
        if arch_kwargs.get(key) is not None:
            arch_kwargs[key] = _resolve_class(arch_kwargs[key])

    dataset_json = ckpt["init_args"]["dataset_json"]
    input_channels = len(dataset_json["channel_names"])

    arch_kwargs["input_channels"] = input_channels
    arch_kwargs["num_classes"] = num_classes
    arch_kwargs["deep_supervision"] = False

    strides = arch_kwargs["strides"]
    ndim = len(strides[0])
    div_factors = [1] * ndim
    for s in strides:
        for d in range(ndim):
            div_factors[d] *= s[d]

    network = net_cls(**arch_kwargs)
    network.load_state_dict(_strip_prefix(ckpt["network_weights"]))
    network.to(device)
    network.eval()
    return network, div_factors

# ---------------------------------------------------------------------------
# Spatial padding to satisfy network stride constraints
# ---------------------------------------------------------------------------

def pad_to_divisible(x: torch.Tensor, div_factors: list):
    """Pad the spatial dimensions (last len(div_factors) dims) to the next multiple.
    Returns (padded_tensor, original_spatial_shape).
    """
    spatial_dims = x.shape[-len(div_factors):]
    pad_amounts = []
    for size, factor in zip(reversed(spatial_dims), reversed(div_factors)):
        remainder = int(size) % factor
        p = (factor - remainder) % factor
        pad_amounts.extend([0, p])
    if all(p == 0 for p in pad_amounts):
        return x, list(spatial_dims)
    return F.pad(x, pad_amounts, mode="constant", value=0), list(spatial_dims)

def unpad(x: torch.Tensor, original_spatial_shape: list):
    """Remove padding to restore original spatial dimensions."""
    slices = [slice(None)] * (x.ndim - len(original_spatial_shape))
    for s in original_spatial_shape:
        slices.append(slice(0, s))
    return x[tuple(slices)]


IGNORE_LABEL = -1


def attackable_mask(y: torch.Tensor, div_factors: list) -> torch.Tensor:
    """Bool mask of the voxels an image-level attack can actually reach.

    The zones inputs are zero-filled outside the dilated whole gland (label
    -1) and zero-padded to the model grid; the deployed pipeline regenerates
    both after any attack, so a perturbation there cannot be realised. Built
    from the unpadded labels so the padding (label 0 once padded) is excluded.
    """
    valid = (y != IGNORE_LABEL).float()
    return pad_to_divisible(valid, div_factors)[0].bool()


class ConfinedInput(torch.nn.Module):
    """Feed the wrapped network ``where(mask, x, clean)``.

    The gradient with respect to ``x`` outside ``mask`` is exactly zero, so an
    attack that touches the model only through ``forward`` never moves those
    voxels in any direction the model can see; ``confine`` strips whatever a
    random start left there before the perturbed image is evaluated or saved.
    """

    def __init__(self, network: torch.nn.Module, clean: torch.Tensor, mask: torch.Tensor):
        super().__init__()
        self.network = network
        self.register_buffer("clean", clean.detach())
        self.register_buffer("mask", mask)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.network(torch.where(self.mask, x, self.clean))


def confine(x_adv: torch.Tensor, clean: torch.Tensor, mask: torch.Tensor | None) -> torch.Tensor:
    """Zero the perturbation outside ``mask``; ``None`` leaves it untouched."""
    if mask is None:
        return x_adv
    return torch.where(mask, x_adv, clean)

# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------

def load_validation_ids(splits_json: str, fold: int = 0) -> list:
    with open(splits_json) as f:
        splits = json.load(f)
    if not splits:
        raise ValueError(f"splits JSON is empty: {splits_json}")
    if fold < 0 or fold >= len(splits):
        raise IndexError(
            f"fold {fold} out of range for {splits_json} (len(splits)={len(splits)})"
        )
    if "val" not in splits[fold]:
        raise KeyError(
            f"splits[{fold}] has no 'val' key in {splits_json}; keys={list(splits[fold].keys())}"
        )
    return splits[fold]["val"]

def load_sample(data_dir: str, case_id: str):
    """Return (image, seg) as numpy arrays.
    image: float32 [C, D, H, W], seg: int [1, D, H, W] with -1 outside crop region.
    Supports .npy, .npz, and .b2nd (blosc2) formats.
    """
    img_path = os.path.join(data_dir, f"{case_id}.npy")
    seg_path = os.path.join(data_dir, f"{case_id}_seg.npy")
    npz_path = os.path.join(data_dir, f"{case_id}.npz")
    b2nd_path = os.path.join(data_dir, f"{case_id}.b2nd")
    seg_b2nd_path = os.path.join(data_dir, f"{case_id}_seg.b2nd")

    if os.path.isfile(img_path):
        img = np.load(img_path).astype(np.float32)
        seg = np.load(seg_path)
    elif os.path.isfile(npz_path):
        data = np.load(npz_path)
        img = data["data"].astype(np.float32)
        seg = data["seg"]
    elif os.path.isfile(b2nd_path):
        import blosc2
        data = blosc2.open(urlpath=b2nd_path, mode="r")
        seg_data = blosc2.open(urlpath=seg_b2nd_path, mode="r")
        img = np.asarray(data[:], dtype=np.float32)
        seg = np.asarray(seg_data[:])
    else:
        raise FileNotFoundError(f"No data file found for {case_id} in {data_dir}")
    return img, seg

# ---------------------------------------------------------------------------
# FGSM attack
# ---------------------------------------------------------------------------

def samplewise_bounds(x: torch.Tensor):
    x_min = x.amin(dim=tuple(range(1, x.ndim)), keepdim=True)
    x_max = x.amax(dim=tuple(range(1, x.ndim)), keepdim=True)
    return x_min, x_max

def dice_ce_loss(logits: torch.Tensor, target: torch.Tensor, num_classes: int):
    if num_classes == 2:
        # Standard nnUNet two-class: softmax CE + soft Dice on foreground (not BCE on one logit).
        t = target.squeeze(1).long()
        ce = F.cross_entropy(logits, t, ignore_index=IGNORE_LABEL)
        probs = F.softmax(logits, dim=1)
        p_fg = probs[:, 1:2]
        valid = (t != IGNORE_LABEL).float().unsqueeze(1)
        t_fg = (t == 1).float().unsqueeze(1) * valid
        p_fg = p_fg * valid
        inter = (p_fg * t_fg).sum()
        denom = p_fg.sum() + t_fg.sum()
        dice_loss = 1.0 - (2.0 * inter + 1e-5) / (denom + 1e-5)
    else:
        target_long = target.squeeze(1).long()
        ce = F.cross_entropy(logits, target_long, ignore_index=-1)
        t = target.squeeze(1)
        valid = (t != IGNORE_LABEL).float()
        dice_loss = torch.tensor(0.0, device=logits.device)
        for c in range(1, num_classes):
            probs_c = torch.softmax(logits, dim=1)[:, c]
            target_c = (t == c).float() * valid
            inter = (probs_c * target_c).sum()
            denom = (probs_c * valid).sum() + target_c.sum()
            dice_loss = dice_loss + 1.0 - (2.0 * inter + 1e-5) / (denom + 1e-5)
        dice_loss = dice_loss / (num_classes - 1)
    return dice_loss + ce


def fgsm_attack(
    model: torch.nn.Module,
    x: torch.Tensor,
    y: torch.Tensor,
    eps: float,
    num_classes: int,
):
    """Single-step FGSM with L-inf projection and sample-wise domain clipping."""
    if math.isclose(eps, 0.0, abs_tol=0.0, rel_tol=0.0):
        return x.clone()
    x_adv = x.clone().detach().requires_grad_(True)
    logits = model(x_adv)
    loss = dice_ce_loss(logits, y, num_classes)
    model.zero_grad(set_to_none=True)
    loss.backward()
    grad_sign = x_adv.grad.detach().sign()
    x_out = x.detach() + eps * grad_sign
    x_min, x_max = samplewise_bounds(x)
    x_out = torch.clamp(x_out, x_min, x_max)
    return x_out.detach()

# ---------------------------------------------------------------------------
# Metrics
# ---------------------------------------------------------------------------

def _masked_class_masks_vol(gt_vol: np.ndarray, pred_vol: np.ndarray, class_id: int):
    """Binary GT/pred for class_id; ignore voxels (GT == -1) excluded from both masks."""
    valid = gt_vol != IGNORE_LABEL
    gt_bin = ((gt_vol == class_id) & valid).astype(np.uint8)
    pred_bin = ((pred_vol == class_id) & valid).astype(np.uint8)
    return gt_bin, pred_bin


def compute_metrics(logits: torch.Tensor, target_np: np.ndarray, num_classes: int):
    """Compute per-class Dice, HD95, ASD. Returns dict[class_idx] -> (dice, hd95, asd)."""
    results = {}
    if num_classes == 2:
        pred_vol = logits.argmax(dim=1)[0].detach().cpu().numpy().astype(np.uint8)
        gt_vol = target_np[0, 0]
        gt_bin, pred_bin = _masked_class_masks_vol(gt_vol, pred_vol, 1)
        results[1] = _class_metrics(pred_bin, gt_bin)
    else:
        pred_classes = torch.argmax(torch.softmax(logits, dim=1), dim=1)
        pred_np = pred_classes.detach().cpu().numpy()[0]
        gt_np = target_np[0, 0]
        for c in range(1, num_classes):
            gt_bin, pred_bin = _masked_class_masks_vol(gt_np, pred_np, c)
            results[c] = _class_metrics(pred_bin, gt_bin)
    return results


def _class_metrics(pred_bin: np.ndarray, gt_bin: np.ndarray):
    inter = (pred_bin * gt_bin).sum()
    denom = pred_bin.sum() + gt_bin.sum()
    dice_val = float((2.0 * inter + 1e-6) / (denom + 1e-6))
    try:
        hd95_val = float(hd95(pred_bin, gt_bin))
    except (RuntimeError, ValueError, TypeError):
        hd95_val = np.nan
    try:
        asd_val = float(asd(pred_bin, gt_bin))
    except (RuntimeError, ValueError, TypeError):
        asd_val = np.nan
    return dice_val, hd95_val, asd_val

def _make_eval_config(model_key: str, args: argparse.Namespace | None) -> dict:
    """Shallow copy of MODEL_CONFIGS entry; optional CLI epsilon overrides; float eps keys."""
    cfg = {**MODEL_CONFIGS[model_key]}
    if args is not None:
        trainer = getattr(args, "trainer", None)
        if trainer:
            # Attack a different trained variant (ResEnc-M/L/XL, PGD-AT) of the
            # same dataset. Overriding the trainer rather than a single
            # checkpoint path keeps --all-folds working: every fold resolves
            # under the same trainer directory.
            trainer_dir = os.path.join(
                NNUNET_PATHS, "nnUNet_results", cfg["dataset_folder"], trainer
            )
            if not os.path.isdir(trainer_dir):
                raise FileNotFoundError(f"Trainer directory not found: {trainer_dir}")
            cfg["trainer"] = trainer
        if model_key == "wg" and args.epsilons_wg is not None:
            cfg["epsilons"] = sorted(args.epsilons_wg)
        elif model_key == "zones" and args.epsilons_zones is not None:
            cfg["epsilons"] = sorted(args.epsilons_zones)
    cfg["epsilons"] = [float(e) for e in cfg["epsilons"]]
    return cfg


# ---------------------------------------------------------------------------
# Evaluation loop
# ---------------------------------------------------------------------------

def evaluate_model(
    model_key: str,
    device: torch.device,
    max_samples: int | None = None,
    output_dir: str = "fgsm_results",
    cfg: dict | None = None,
    fold: int = 0,
    save_outputs: bool = True,
    file_suffix: str = "",
    attack_valid_only: bool = False,
) -> dict:
    """Run FGSM evaluation on one fold's checkpoint and validation split.

    Returns the same nested ``results`` dict used for CSV/plots (epsilon -> class -> metric -> list).
    ``file_suffix``: e.g. ``\"\"`` for legacy ``{model_key}_fgsm_*.csv`` or ``\"_fold0\"`` for
    ``{model_key}_fgsm_fold0_*.csv``. When ``save_outputs`` is False, only the dict is returned.
    """
    base = cfg if cfg is not None else MODEL_CONFIGS[model_key]
    cfg = {**base}
    cfg["epsilons"] = [float(e) for e in cfg["epsilons"]]

    checkpoint_path = get_checkpoint_for_fold(cfg, fold)

    print(f"\n{'='*60}")
    print(f"Evaluating: {cfg['name']} ({model_key})  |  fold {fold}")
    print(f"Checkpoint: {checkpoint_path}")
    print(f"{'='*60}")

    os.makedirs(output_dir, exist_ok=True)

    network, div_factors = load_model(
        checkpoint_path, cfg["configuration"], cfg["num_classes"], device
    )
    val_ids = load_validation_ids(cfg["splits_json"], fold)
    if max_samples is not None:
        val_ids = val_ids[:max_samples]
    print(f"Validation samples: {len(val_ids)}")
    print(f"Divisibility factors: {div_factors}")

    num_classes = cfg["num_classes"]
    class_names = cfg["class_names"]
    eps_list = cfg["epsilons"]
    fg_classes = list(range(1, num_classes))

    results = {
        eps: {c: {"dice": [], "hd95": [], "asd": []} for c in fg_classes}
        for eps in eps_list
    }

    for i, case_id in enumerate(val_ids):
        img_np, seg_np = load_sample(cfg["data_dir"], case_id)
        orig_seg_np = np.asarray(seg_np, dtype=np.int64)

        x = torch.from_numpy(img_np[np.newaxis]).to(device)
        y = torch.from_numpy(orig_seg_np[np.newaxis].astype(np.float32)).to(device)

        x_padded, orig_shape = pad_to_divisible(x, div_factors)
        y_padded, _ = pad_to_divisible(y, div_factors)
        mask = attackable_mask(y, div_factors) if attack_valid_only else None
        attacked = ConfinedInput(network, x_padded, mask) if mask is not None else network

        for eps in eps_list:
            with torch.enable_grad():
                x_input = fgsm_attack(attacked, x_padded, y_padded, eps, num_classes)
            x_input = confine(x_input, x_padded, mask)
            with torch.no_grad():
                logits_padded = network(x_input)
            logits = unpad(logits_padded, orig_shape)
            metrics = compute_metrics(logits, orig_seg_np[np.newaxis], num_classes)
            for c in fg_classes:
                d, h, a = metrics[c]
                results[eps][c]["dice"].append(d)
                results[eps][c]["hd95"].append(h)
                results[eps][c]["asd"].append(a)
            del x_input, logits_padded, logits

        if (i + 1) % 10 == 0 or (i + 1) == len(val_ids):
            print(f"  [{i+1}/{len(val_ids)}] processed")

        del x, y, x_padded, y_padded

    if save_outputs:
        _save_csv(results, fg_classes, class_names, eps_list, model_key, output_dir, file_suffix)
        _save_per_sample_csv(
            results, fg_classes, class_names, eps_list, val_ids, model_key, output_dir, file_suffix
        )
        _save_plots(results, fg_classes, class_names, eps_list, model_key, cfg["name"], output_dir, file_suffix)
        _print_summary(results, fg_classes, class_names, eps_list)

    del network
    gc.collect()
    if device.type == "cuda":
        torch.cuda.empty_cache()

    return results

# ---------------------------------------------------------------------------
# CSV export
# ---------------------------------------------------------------------------

def _save_csv(results, fg_classes, class_names, eps_list, model_key, output_dir, file_suffix: str = ""):
    for metric_name in ["dice", "hd95", "asd"]:
        path = os.path.join(output_dir, f"{model_key}_fgsm{file_suffix}_{metric_name}.csv")
        with open(path, "w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(["epsilon", "class", "mean", "std", "sem", "n"])
            for eps in eps_list:
                for c, cname in zip(fg_classes, class_names):
                    vals = np.array(results[eps][c][metric_name], dtype=np.float64)
                    vals = vals[np.isfinite(vals)]
                    n = len(vals)
                    mean = float(vals.mean()) if n > 0 else np.nan
                    std = float(vals.std(ddof=1)) if n > 1 else np.nan
                    sem = std / np.sqrt(n) if n > 1 else np.nan
                    writer.writerow([eps, cname, f"{mean:.6f}", f"{std:.6f}", f"{sem:.6f}", n])
        print(f"  Saved {path}")

def _save_per_sample_csv(
    results, fg_classes, class_names, eps_list, val_ids, model_key, output_dir, file_suffix: str = ""
):
    """Export one row per (case, epsilon, class) with individual metric values."""
    path = os.path.join(output_dir, f"{model_key}_fgsm{file_suffix}_per_sample.csv")
    with open(path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["case_id", "epsilon", "class", "dice", "hd95", "asd"])
        for eps in eps_list:
            for c, cname in zip(fg_classes, class_names):
                for idx, case_id in enumerate(val_ids):
                    d = results[eps][c]["dice"][idx]
                    h = results[eps][c]["hd95"][idx]
                    a = results[eps][c]["asd"][idx]
                    writer.writerow([case_id, eps, cname, f"{d:.6f}", f"{h:.6f}", f"{a:.6f}"])
    print(f"  Saved {path}")

# ---------------------------------------------------------------------------
# Plotting
# ---------------------------------------------------------------------------

def _save_plots(
    results,
    fg_classes,
    class_names,
    eps_list,
    model_key,
    model_display: str,
    output_dir,
    file_suffix: str = "",
):
    # medpy distances are in voxel units unless spacing is passed
    for metric_name, ylabel in [
        ("dice", "Dice"),
        ("hd95", "HD95 (vox)"),
        ("asd", "ASD (vox)"),
    ]:
        fig, ax = plt.subplots(figsize=(8, 5))
        for c, cname in zip(fg_classes, class_names):
            means, sems = [], []
            for eps in eps_list:
                vals = np.array(results[eps][c][metric_name], dtype=np.float64)
                vals = vals[np.isfinite(vals)]
                n = len(vals)
                means.append(vals.mean() if n > 0 else np.nan)
                sems.append(vals.std(ddof=1) / np.sqrt(n) if n > 1 else 0.0)
            ax.errorbar(eps_list, means, yerr=sems, marker="o", capsize=4, label=cname)
        ax.set_xlabel("FGSM Epsilon")
        ax.set_ylabel(ylabel)
        ax.set_title(f"{model_display} — {ylabel} vs FGSM Epsilon")
        ax.legend()
        ax.grid(True, alpha=0.3)
        plt.tight_layout()
        path = os.path.join(output_dir, f"{model_key}_fgsm{file_suffix}_{metric_name}.png")
        plt.savefig(path, dpi=300, bbox_inches="tight")
        plt.close(fig)
        print(f"  Saved {path}")


def _mean_sem_over_samples(values: list) -> tuple[float, float]:
    vals = np.array(values, dtype=np.float64)
    vals = vals[np.isfinite(vals)]
    n = len(vals)
    if n == 0:
        return float("nan"), 0.0
    mean = float(vals.mean())
    sem = float(vals.std(ddof=1) / np.sqrt(n)) if n > 1 else 0.0
    return mean, sem


def _save_macro_csv(
    results_by_fold: dict[int, dict],
    fg_classes: list,
    class_names: list,
    eps_list: list,
    model_key: str,
    output_dir: str,
):
    """Macro average = mean across folds of each fold's validation-set mean (per ε, class, metric)."""
    path = os.path.join(output_dir, f"{model_key}_fgsm_all_folds_macro_summary.csv")
    folds_sorted = sorted(results_by_fold.keys())
    with open(path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(
            ["epsilon", "class", "metric", "macro_mean", "sem_across_folds", "n_folds"]
        )
        for eps in eps_list:
            for c, cname in zip(fg_classes, class_names):
                for metric_name in ["dice", "hd95", "asd"]:
                    fold_means = []
                    for fd in folds_sorted:
                        r = results_by_fold[fd]
                        m, _ = _mean_sem_over_samples(r[eps][c][metric_name])
                        if np.isfinite(m):
                            fold_means.append(m)
                    n_f = len(fold_means)
                    if n_f == 0:
                        macro = float("nan")
                        sem_f = float("nan")
                    else:
                        arr = np.array(fold_means, dtype=np.float64)
                        macro = float(arr.mean())
                        sem_f = float(arr.std(ddof=1) / np.sqrt(n_f)) if n_f > 1 else 0.0
                    writer.writerow([eps, cname, metric_name, f"{macro:.6f}", f"{sem_f:.6f}", n_f])
    print(f"  Saved {path}")


def _save_plots_per_fold_and_macro(
    results_by_fold: dict[int, dict],
    fg_classes: list,
    class_names: list,
    eps_list: list,
    model_key: str,
    model_display: str,
    output_dir: str,
):
    """One figure per metric: each class gets a subplot with one curve per fold + macro average ± SEM across folds."""
    folds_sorted = sorted(results_by_fold.keys())
    tab10 = plt.get_cmap("tab10")

    for metric_name, ylabel in [
        ("dice", "Dice"),
        ("hd95", "HD95 (mm)"),
        ("asd", "ASD (mm)"),
    ]:
        n_cls = len(fg_classes)
        fig, axes = plt.subplots(1, n_cls, figsize=(5.5 * n_cls, 5), squeeze=False)
        axes = axes[0]

        for ax_idx, (c, cname) in enumerate(zip(fg_classes, class_names)):
            ax = axes[ax_idx]
            fold_means_per_eps: list[list[float]] = []

            for fi, fd in enumerate(folds_sorted):
                r = results_by_fold[fd]
                means = []
                for eps in eps_list:
                    m, sem = _mean_sem_over_samples(r[eps][c][metric_name])
                    means.append(m)
                fold_means_per_eps.append(means)
                color = tab10((fi % 10) / 10.0)
                ax.plot(
                    eps_list,
                    means,
                    marker="o",
                    linestyle="-",
                    linewidth=1.6,
                    color=color,
                    alpha=0.85,
                    label=f"Fold {fd}",
                )

            if fold_means_per_eps:
                mat = np.array(fold_means_per_eps, dtype=np.float64)
                macro = np.nanmean(mat, axis=0)
                n_f = mat.shape[0]
                if n_f > 1:
                    sem_macro = np.nanstd(mat, axis=0, ddof=1) / np.sqrt(n_f)
                else:
                    sem_macro = np.zeros_like(macro)
                ax.errorbar(
                    eps_list,
                    macro,
                    yerr=sem_macro,
                    marker="s",
                    linestyle="--",
                    linewidth=2.4,
                    color="black",
                    capsize=4,
                    zorder=10,
                    label="Macro avg (across folds)",
                )

            ax.set_xlabel("FGSM ε")
            ax.set_ylabel(ylabel)
            ax.set_title(cname)
            ax.legend(fontsize=8, loc="best")
            ax.grid(True, alpha=0.3)

        fig.suptitle(
            f"{model_display} — {ylabel} vs FGSM attack (per-fold + macro)\n",
            fontsize=11,
            fontweight="bold",
            y=1.02,
        )
        plt.tight_layout()
        out_path = os.path.join(output_dir, f"{model_key}_fgsm_all_folds_{metric_name}.png")
        plt.savefig(out_path, dpi=300, bbox_inches="tight")
        plt.close(fig)
        print(f"  Saved {out_path}")


def evaluate_all_folds_for_model(
    model_key: str,
    device: torch.device,
    max_samples: int | None,
    output_dir: str,
    cfg: dict,
    folds: list[int] | None = None,
    attack_valid_only: bool = False,
) -> dict[int, dict]:
    """Run FGSM for each fold checkpoint on that fold's val split; save per-fold artifacts + macro plots/CSV."""
    splits_path = cfg["splits_json"]
    with open(splits_path) as f:
        splits = json.load(f)
    n_splits = len(splits)
    fold_list = folds if folds is not None else list(range(n_splits))

    results_by_fold: dict[int, dict] = {}
    for fold in fold_list:
        if fold < 0 or fold >= n_splits:
            print(f"  Skip invalid fold {fold} (splits length {n_splits})")
            continue
        ckpt = get_checkpoint_for_fold(cfg, fold)
        if not os.path.isfile(ckpt):
            print(f"  Skip fold {fold}: missing checkpoint {ckpt}")
            continue
        suffix = f"_fold{fold}"
        res = evaluate_model(
            model_key,
            device,
            max_samples=max_samples,
            output_dir=output_dir,
            cfg=cfg,
            fold=fold,
            save_outputs=True,
            file_suffix=suffix,
            attack_valid_only=attack_valid_only,
        )
        results_by_fold[fold] = res

    if not results_by_fold:
        print(f"  No fold results for {model_key}; skipping macro plots.")
        return results_by_fold

    fg_classes = list(range(1, cfg["num_classes"]))
    class_names = cfg["class_names"]
    eps_list = [float(e) for e in cfg["epsilons"]]
    _save_macro_csv(
        results_by_fold, fg_classes, class_names, eps_list, model_key, output_dir
    )
    _save_plots_per_fold_and_macro(
        results_by_fold,
        fg_classes,
        class_names,
        eps_list,
        model_key,
        cfg["name"],
        output_dir,
    )
    return results_by_fold


# ---------------------------------------------------------------------------
# Summary printing
# ---------------------------------------------------------------------------

def _print_summary(results, fg_classes, class_names, eps_list):
    print("\n  Summary:")
    print(f"  {'Epsilon':<10}", end="")
    for cname in class_names:
        print(f"  {cname+' Dice':<14} {cname+' HD95':<14} {cname+' ASD':<14}", end="")
    print()
    for eps in eps_list:
        print(f"  {eps:<10.3f}", end="")
        for c in fg_classes:
            for m in ["dice", "hd95", "asd"]:
                vals = np.array(results[eps][c][m], dtype=np.float64)
                vals = vals[np.isfinite(vals)]
                mean = vals.mean() if len(vals) > 0 else np.nan
                print(f"  {mean:<14.4f}", end="")
        print()
    print()

# ---------------------------------------------------------------------------
# Slice visualisation for a random patient
# ---------------------------------------------------------------------------

def _logits_to_seg(logits: torch.Tensor, num_classes: int) -> np.ndarray:
    """Convert raw logits [1, C, D, H, W] to an integer segmentation map [D, H, W]."""
    if num_classes == 2:
        return (torch.sigmoid(logits[0, 1]) > 0.5).cpu().numpy().astype(np.uint8)
    return torch.argmax(torch.softmax(logits, dim=1), dim=1)[0].cpu().numpy().astype(np.uint8)

def _pick_foreground_slice(seg_3d: np.ndarray) -> int:
    """Return the index of a random axial slice that contains foreground."""
    fg_slices = [s for s in range(seg_3d.shape[0]) if seg_3d[s].max() > 0]
    if not fg_slices:
        return seg_3d.shape[0] // 2
    return random.choice(fg_slices)

def _overlay_binary(mri_rgb: np.ndarray, mask: np.ndarray, rgba: np.ndarray):
    """Burn an RGBA colour into *mri_rgb* wherever *mask* is True (in-place)."""
    alpha = rgba[3]
    mri_rgb[mask] = mri_rgb[mask] * (1.0 - alpha) + rgba[:3] * alpha

def _make_panel(mri_slice: np.ndarray, gt_bin: np.ndarray, pred_bin: np.ndarray, vmin: float, vmax: float) -> np.ndarray:
    """Return an RGB image: MRI in grey, GT overlay in blue, prediction in orange."""
    normed = np.clip((mri_slice - vmin) / (vmax - vmin + 1e-8), 0.0, 1.0)
    rgb = np.stack([normed, normed, normed], axis=-1).astype(np.float64)
    _overlay_binary(rgb, gt_bin > 0, _GT_COLOR)
    _overlay_binary(rgb, pred_bin > 0, _PRED_COLOR)
    return np.clip(rgb, 0.0, 1.0)

def _save_vis_figure(panels: list, eps_list: list, dice_per_eps: list,
                     title: str, output_path: str, legend_labels: tuple):
    """One row of panels (one per epsilon), GT in blue + prediction in orange."""
    n = len(eps_list)
    fig, axes = plt.subplots(1, n, figsize=(3.2 * n, 3.8))
    if n == 1:
        axes = [axes]

    for j, (panel, eps, dice_val) in enumerate(zip(panels, eps_list, dice_per_eps)):
        ax = axes[j]
        ax.imshow(panel, interpolation="nearest")
        eps_label = f"ε={eps}" if not math.isclose(eps, 0.0, abs_tol=0.0, rel_tol=0.0) else "Clean (ε=0)"
        ax.set_title(f"{eps_label}\nDice: {dice_val:.3f}", fontsize=9)
        ax.axis("off")

    gt_patch = plt.Line2D([0], [0], marker="s", color="w",
                          markerfacecolor=_GT_COLOR[:3], markersize=10,
                          label=legend_labels[0])
    pred_patch = plt.Line2D([0], [0], marker="s", color="w",
                            markerfacecolor=_PRED_COLOR[:3], markersize=10,
                            label=legend_labels[1])
    fig.legend(handles=[gt_patch, pred_patch], loc="lower center",
               ncol=2, fontsize=9, frameon=True, bbox_to_anchor=(0.5, -0.02))

    fig.suptitle(title, fontsize=11, fontweight="bold")
    fig.savefig(output_path, dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved {output_path}")

def visualize_random_patient(
    device: torch.device,
    output_dir: str = "fgsm_results",
    seed: int | None = None,
    args: argparse.Namespace | None = None,
):
    """For a random validation patient, visualise GT (blue) and prediction (orange)
    overlaid on the same MRI slice for every epsilon.

    - Whole Gland  -> 1 figure  (WG class)
    - Prostate Zones -> 2 separate figures  (TZ+CZ and PZ)
    """
    if seed is not None:
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)

    os.makedirs(output_dir, exist_ok=True)

    for model_key in ("wg", "zones"):
        cfg = _make_eval_config(model_key, args)
        num_classes = cfg["num_classes"]

        vis_fold = getattr(args, "vis_fold", 0) if args is not None else 0
        network, div_factors = load_model(
            get_checkpoint_for_fold(cfg, vis_fold), cfg["configuration"], num_classes, device
        )
        val_ids = load_validation_ids(cfg["splits_json"], vis_fold)
        case_id = random.choice(val_ids)
        img_np, seg_np = load_sample(cfg["data_dir"], case_id)
        gt = np.asarray(seg_np[0], dtype=np.int64)

        slice_idx = _pick_foreground_slice(gt)
        print(f"  [{cfg['name']}] patient={case_id}  slice={slice_idx}")

        x = torch.from_numpy(img_np[np.newaxis]).to(device)
        y_torch = torch.from_numpy(np.asarray(seg_np, dtype=np.float32)[np.newaxis]).to(device)
        x_padded, orig_shape = pad_to_divisible(x, div_factors)
        y_padded, _ = pad_to_divisible(y_torch, div_factors)
        mask = (
            attackable_mask(y_torch, div_factors)
            if args is not None and getattr(args, "attack_valid_only", False)
            else None
        )
        attacked = ConfinedInput(network, x_padded, mask) if mask is not None else network

        mri_slice = img_np[0, slice_idx]
        vmin_img, vmax_img = np.percentile(mri_slice, [1, 99])
        gt_slice = gt[slice_idx]

        eps_list = cfg["epsilons"]
        fg_classes = list(range(1, num_classes))
        panels_per_class = {c: [] for c in fg_classes}
        dice_per_class = {c: [] for c in fg_classes}

        for eps in eps_list:
            with torch.enable_grad():
                x_input = fgsm_attack(attacked, x_padded, y_padded, eps, num_classes)
            x_input = confine(x_input, x_padded, mask)
            with torch.no_grad():
                logits_padded = network(x_input)
            logits = unpad(logits_padded, orig_shape)
            pred = _logits_to_seg(logits, num_classes)
            pred_slice = pred[slice_idx]

            metrics = compute_metrics(logits, gt[np.newaxis, np.newaxis], num_classes)

            for c in fg_classes:
                gt_c, pred_c = _masked_class_masks_vol(gt_slice, pred_slice, c)
                panel = _make_panel(mri_slice, gt_c, pred_c, vmin_img, vmax_img)
                panels_per_class[c].append(panel)
                dice_per_class[c].append(metrics[c][0])
            del x_input, logits_padded, logits

        if model_key == "wg":
            _save_vis_figure(
                panels_per_class[1], eps_list, dice_per_class[1],
                title=(f"Whole Gland — FGSM Visualisation\n"
                       f"Patient: {case_id}   Slice: {slice_idx}"),
                output_path=os.path.join(output_dir, "wg_fgsm_slice_vis.png"),
                legend_labels=("Ground Truth", "Prediction"),
            )
        else:
            for c, cname in zip(fg_classes, cfg["class_names"]):
                safe_name = cname.replace("+", "").replace("/", "")
                _save_vis_figure(
                    panels_per_class[c], eps_list, dice_per_class[c],
                    title=(f"Prostate Zones — {cname} — FGSM Visualisation\n"
                           f"Patient: {case_id}   Slice: {slice_idx}"),
                    output_path=os.path.join(
                        output_dir, f"zones_{safe_name}_fgsm_slice_vis.png"),
                    legend_labels=(f"Ground Truth ({cname})", f"Prediction ({cname})"),
                )

        del network
        gc.collect()
        torch.cuda.empty_cache() if device.type == "cuda" else None

# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="FGSM adversarial robustness evaluation for nnU-Net models")
    parser.add_argument(
        "--model",
        choices=["wg", "zones", "both"],
        default="both",
        help="Which model to evaluate (default: both)",
    )
    parser.add_argument(
        "--max-samples",
        type=int,
        default=None,
        help="Limit number of validation samples (for quick testing)",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="fgsm_results",
        help="Directory for CSV and plot outputs",
    )
    parser.add_argument(
        "--epsilons-wg",
        type=float,
        nargs="+",
        default=None,
        help="Custom epsilon values for the Whole Gland model",
    )
    parser.add_argument(
        "--epsilons-zones",
        type=float,
        nargs="+",
        default=None,
        help="Custom epsilon values for the Prostate Zones model",
    )
    parser.add_argument(
        "--visualize",
        action="store_true",
        help="Generate slice visualisation for a random patient (both models)",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=None,
        help="Random seed for reproducible patient/slice selection in --visualize",
    )
    parser.add_argument(
        "--all-folds",
        action="store_true",
        help=(
            "Run FGSM for every fold (Dataset016 / wg and Dataset019 / zones): "
            "fold_k checkpoint on splits[k]['val'], then save per-fold CSV/plots plus "
            "macro summary and combined figures (per-fold curves + macro average)."
        ),
    )
    parser.add_argument(
        "--fold",
        type=int,
        default=0,
        help="Single-fold mode only: which fold index (default 0). Ignored with --all-folds.",
    )
    parser.add_argument(
        "--folds",
        type=int,
        nargs="+",
        default=None,
        help="With --all-folds: only these fold indices (default: all splits in splits_final.json).",
    )
    parser.add_argument(
        "--vis-fold",
        type=int,
        default=0,
        help="With --visualize: checkpoint fold to load (default 0).",
    )
    parser.add_argument(
        "--trainer",
        type=str,
        default=None,
        help=(
            "nnU-Net results trainer directory to attack instead of the default "
            "nnUNetTrainer__nnUNetPlans__3d_fullres, e.g. "
            "nnUNetTrainer__nnUNetResEncUNetXLPlans__3d_fullres. Resolved under "
            "the same dataset, so every fold comes from this trainer."
        ),
    )
    parser.add_argument(
        "--attack-valid-only",
        action="store_true",
        help=(
            "Confine the perturbation to voxels with a valid label (label != -1) "
            "inside the unpadded volume. The zones inputs are zero-filled outside "
            "the dilated whole gland and zero-padded to the model grid, and the "
            "deployed pipeline regenerates both after any image-level attack, so "
            "a perturbation there cannot be realised. Default off reproduces the "
            "published (unconfined) numbers."
        ),
    )
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    if args.visualize:
        visualize_random_patient(
            device, output_dir=args.output_dir, seed=args.seed, args=args
        )
    else:
        models_to_eval = ["wg", "zones"] if args.model == "both" else [args.model]
        for model_key in models_to_eval:
            cfg = _make_eval_config(model_key, args)
            if args.all_folds:
                evaluate_all_folds_for_model(
                    model_key,
                    device,
                    max_samples=args.max_samples,
                    output_dir=args.output_dir,
                    cfg=cfg,
                    folds=args.folds,
                    attack_valid_only=args.attack_valid_only,
                )
            else:
                file_suffix = "" if args.fold == 0 else f"_fold{args.fold}"
                evaluate_model(
                    model_key,
                    device,
                    max_samples=args.max_samples,
                    output_dir=args.output_dir,
                    cfg=cfg,
                    fold=args.fold,
                    save_outputs=True,
                    file_suffix=file_suffix,
                    attack_valid_only=args.attack_valid_only,
                )

    print("\nDone.")

if __name__ == "__main__":
    main()
