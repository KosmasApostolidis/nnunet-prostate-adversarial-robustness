"""
Cross-Model Transferability and Cascaded Pipeline Attack Analysis.

Analyses:
  1. Transferability: Generate adversarial examples using the WG model and
     evaluate them on the Zones model (and vice versa).
  2. Cascaded attack: Attack the WG model with FGSM, then feed the
     adversarial input to the Zones model to measure downstream degradation.
     This is the most realistic threat model for the deployed pipeline.
"""

import os
import csv
import gc
import argparse
import numpy as np
import torch
import torch.nn.functional as F
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

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
)


# ---------------------------------------------------------------------------
# Transferability
# ---------------------------------------------------------------------------


def evaluate_transferability(
    device: torch.device,
    max_samples: int | None = None,
    output_dir: str = "transfer_results",
):
    """Generate adversarial examples from one model, evaluate on the other."""
    os.makedirs(output_dir, exist_ok=True)

    configs = {k: MODEL_CONFIGS[k] for k in ("wg", "zones")}
    networks = {}
    div_factors_map = {}
    for mk, cfg in configs.items():
        net, divf = load_model(
            cfg["checkpoint"], cfg["configuration"], cfg["num_classes"], device
        )
        networks[mk] = net
        div_factors_map[mk] = divf

    transfer_pairs = [("wg", "zones"), ("zones", "wg")]

    for source_key, target_key in transfer_pairs:
        src_cfg = configs[source_key]
        tgt_cfg = configs[target_key]
        src_net = networks[source_key]
        tgt_net = networks[target_key]
        src_div = div_factors_map[source_key]
        tgt_div = div_factors_map[target_key]

        src_val = set(load_validation_ids(src_cfg["splits_json"]))
        tgt_val = set(load_validation_ids(tgt_cfg["splits_json"]))
        common_ids = sorted(src_val & tgt_val)
        if max_samples is not None:
            common_ids = common_ids[:max_samples]

        if not common_ids:
            print(
                f"\n  No overlapping patients between {source_key} and {target_key}. Skipping."
            )
            continue

        print(f"\n{'=' * 60}")
        print(f"Transfer: {source_key} -> {target_key} ({len(common_ids)} patients)")
        print(f"{'=' * 60}")

        tgt_num_classes = tgt_cfg["num_classes"]
        tgt_class_names = tgt_cfg["class_names"]
        tgt_fg = list(range(1, tgt_num_classes))
        src_num_classes = src_cfg["num_classes"]
        eps_list = src_cfg["epsilons"]

        results_transfer = {
            eps: {c: {"dice": [], "hd95": [], "asd": []} for c in tgt_fg}
            for eps in eps_list
        }
        results_clean = {
            eps: {c: {"dice": [], "hd95": [], "asd": []} for c in tgt_fg}
            for eps in eps_list
        }

        for i, case_id in enumerate(common_ids):
            src_img, src_seg = load_sample(src_cfg["data_dir"], case_id)
            tgt_img, tgt_seg = load_sample(tgt_cfg["data_dir"], case_id)
            src_seg_clean = np.clip(src_seg, 0, None)
            tgt_seg_clean = np.clip(tgt_seg, 0, None)

            x_src = torch.from_numpy(src_img[np.newaxis]).to(device)
            y_src = torch.from_numpy(src_seg_clean[np.newaxis].astype(np.float32)).to(
                device
            )
            x_src_pad, src_orig = pad_to_divisible(x_src, src_div)
            y_src_pad, _ = pad_to_divisible(y_src, src_div)

            x_tgt = torch.from_numpy(tgt_img[np.newaxis]).to(device)
            x_tgt_pad, tgt_orig = pad_to_divisible(x_tgt, tgt_div)

            for eps in eps_list:
                with torch.enable_grad():
                    x_adv_src = fgsm_attack(
                        src_net, x_src_pad, y_src_pad, eps, src_num_classes
                    )

                x_adv_src_unpad = unpad(x_adv_src, src_orig)
                tgt_spatial = x_tgt.shape[-len(tgt_div) :]
                src_spatial = x_adv_src_unpad.shape[-len(tgt_div) :]

                if list(tgt_spatial) != list(src_spatial):
                    x_transfer = F.interpolate(
                        x_adv_src_unpad,
                        size=list(tgt_spatial),
                        mode="trilinear",
                        align_corners=False,
                    )
                else:
                    x_transfer = x_adv_src_unpad

                x_transfer_pad, _ = pad_to_divisible(x_transfer, tgt_div)
                with torch.no_grad():
                    logits_t = unpad(tgt_net(x_transfer_pad), tgt_orig)
                    logits_c = unpad(tgt_net(x_tgt_pad), tgt_orig)

                m_t = compute_metrics(
                    logits_t, tgt_seg_clean[np.newaxis], tgt_num_classes
                )
                m_c = compute_metrics(
                    logits_c, tgt_seg_clean[np.newaxis], tgt_num_classes
                )
                for c in tgt_fg:
                    for metric_idx, mname in enumerate(["dice", "hd95", "asd"]):
                        results_transfer[eps][c][mname].append(m_t[c][metric_idx])
                        results_clean[eps][c][mname].append(m_c[c][metric_idx])

            if (i + 1) % 10 == 0 or (i + 1) == len(common_ids):
                print(f"  [{i + 1}/{len(common_ids)}] processed")

            del x_src, y_src, x_src_pad, y_src_pad, x_tgt, x_tgt_pad
            torch.cuda.empty_cache() if device.type == "cuda" else None

        _save_transfer_results(
            results_transfer,
            results_clean,
            tgt_fg,
            tgt_class_names,
            eps_list,
            source_key,
            target_key,
            output_dir,
        )

    for net in networks.values():
        del net
    gc.collect()
    torch.cuda.empty_cache() if device.type == "cuda" else None


def _save_transfer_results(
    results_transfer,
    results_clean,
    fg_classes,
    class_names,
    eps_list,
    source_key,
    target_key,
    output_dir,
):
    for metric in ["dice", "hd95", "asd"]:
        path = os.path.join(
            output_dir, f"transfer_{source_key}_to_{target_key}_{metric}.csv"
        )
        with open(path, "w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(
                ["epsilon", "class", "transfer_mean", "transfer_std", "clean_mean", "n"]
            )
            for eps in eps_list:
                for c, cname in zip(fg_classes, class_names):
                    t_vals = np.array(
                        results_transfer[eps][c][metric], dtype=np.float64
                    )
                    t_vals = t_vals[np.isfinite(t_vals)]
                    c_vals = np.array(results_clean[eps][c][metric], dtype=np.float64)
                    c_vals = c_vals[np.isfinite(c_vals)]
                    writer.writerow(
                        [
                            eps,
                            cname,
                            f"{t_vals.mean():.6f}" if len(t_vals) > 0 else "nan",
                            f"{t_vals.std(ddof=1):.6f}" if len(t_vals) > 1 else "nan",
                            f"{c_vals.mean():.6f}" if len(c_vals) > 0 else "nan",
                            len(t_vals),
                        ]
                    )
        print(f"  Saved {path}")

    for metric, ylabel in [
        ("dice", "Dice"),
        ("hd95", "HD95 (mm)"),
        ("asd", "ASD (mm)"),
    ]:
        fig, ax = plt.subplots(figsize=(8, 5))
        for c, cname in zip(fg_classes, class_names):
            t_means = []
            c_means = []
            for eps in eps_list:
                t_vals = np.array(results_transfer[eps][c][metric], dtype=np.float64)
                t_vals = t_vals[np.isfinite(t_vals)]
                c_vals = np.array(results_clean[eps][c][metric], dtype=np.float64)
                c_vals = c_vals[np.isfinite(c_vals)]
                t_means.append(t_vals.mean() if len(t_vals) > 0 else np.nan)
                c_means.append(c_vals.mean() if len(c_vals) > 0 else np.nan)
            ax.plot(eps_list, t_means, "o-", markersize=4, label=f"{cname} (transfer)")
            ax.plot(
                eps_list,
                c_means,
                "s--",
                markersize=4,
                alpha=0.5,
                label=f"{cname} (clean)",
            )
        ax.set_xlabel("Source Epsilon")
        ax.set_ylabel(ylabel)
        ax.set_title(f"Transfer Attack: {source_key} → {target_key} — {ylabel}")
        ax.legend(fontsize=8)
        ax.grid(True, alpha=0.3)
        plt.tight_layout()
        path = os.path.join(
            output_dir, f"transfer_{source_key}_to_{target_key}_{metric}.png"
        )
        fig.savefig(path, dpi=200, bbox_inches="tight")
        plt.close(fig)
        print(f"  Saved {path}")


# ---------------------------------------------------------------------------
# Cascaded pipeline attack
# ---------------------------------------------------------------------------


def evaluate_cascaded_attack(
    device: torch.device,
    max_samples: int | None = None,
    output_dir: str = "transfer_results",
):
    """
    Attack WG model with FGSM, then evaluate the same adversarial input on
    the Zones model. Measures how upstream adversarial perturbations propagate
    through the two-stage pipeline.
    """
    os.makedirs(output_dir, exist_ok=True)

    wg_cfg = MODEL_CONFIGS["wg"]
    z_cfg = MODEL_CONFIGS["zones"]

    wg_net, wg_div = load_model(
        wg_cfg["checkpoint"], wg_cfg["configuration"], wg_cfg["num_classes"], device
    )
    z_net, z_div = load_model(
        z_cfg["checkpoint"], z_cfg["configuration"], z_cfg["num_classes"], device
    )

    wg_val = set(load_validation_ids(wg_cfg["splits_json"]))
    z_val = set(load_validation_ids(z_cfg["splits_json"]))
    common_ids = sorted(wg_val & z_val)
    if max_samples is not None:
        common_ids = common_ids[:max_samples]

    if not common_ids:
        print("  No overlapping patients. Skipping cascaded attack.")
        return

    print(f"\n{'=' * 60}")
    print(f"Cascaded Attack: WG → Zones ({len(common_ids)} patients)")
    print(f"{'=' * 60}")

    eps_list = wg_cfg["epsilons"]
    z_num_classes = z_cfg["num_classes"]
    z_class_names = z_cfg["class_names"]
    z_fg = list(range(1, z_num_classes))
    wg_num_classes = wg_cfg["num_classes"]

    wg_results = {eps: {1: {"dice": []}} for eps in eps_list}
    zone_results = {
        eps: {c: {"dice": [], "hd95": [], "asd": []} for c in z_fg} for eps in eps_list
    }

    for i, case_id in enumerate(common_ids):
        wg_img, wg_seg = load_sample(wg_cfg["data_dir"], case_id)
        z_img, z_seg = load_sample(z_cfg["data_dir"], case_id)
        wg_seg_clean = np.clip(wg_seg, 0, None)
        z_seg_clean = np.clip(z_seg, 0, None)

        x_wg = torch.from_numpy(wg_img[np.newaxis]).to(device)
        y_wg = torch.from_numpy(wg_seg_clean[np.newaxis].astype(np.float32)).to(device)
        x_wg_pad, wg_orig = pad_to_divisible(x_wg, wg_div)
        y_wg_pad, _ = pad_to_divisible(y_wg, wg_div)

        x_z = torch.from_numpy(z_img[np.newaxis]).to(device)
        x_z_pad, z_orig = pad_to_divisible(x_z, z_div)

        for eps in eps_list:
            with torch.enable_grad():
                x_adv = fgsm_attack(wg_net, x_wg_pad, y_wg_pad, eps, wg_num_classes)

            with torch.no_grad():
                wg_logits = unpad(wg_net(x_adv), wg_orig)
            wg_metrics = compute_metrics(
                wg_logits, wg_seg_clean[np.newaxis], wg_num_classes
            )
            wg_results[eps][1]["dice"].append(wg_metrics[1][0])

            x_adv_unpad = unpad(x_adv, wg_orig)
            z_spatial = x_z.shape[-len(z_div) :]
            wg_spatial = x_adv_unpad.shape[-len(z_div) :]
            if list(z_spatial) != list(wg_spatial):
                x_cascade = F.interpolate(
                    x_adv_unpad,
                    size=list(z_spatial),
                    mode="trilinear",
                    align_corners=False,
                )
            else:
                x_cascade = x_adv_unpad
            x_cascade_pad, _ = pad_to_divisible(x_cascade, z_div)

            with torch.no_grad():
                z_logits = unpad(z_net(x_cascade_pad), z_orig)
            z_metrics = compute_metrics(
                z_logits, z_seg_clean[np.newaxis], z_num_classes
            )
            for c in z_fg:
                d, h, a = z_metrics[c]
                zone_results[eps][c]["dice"].append(d)
                zone_results[eps][c]["hd95"].append(h)
                zone_results[eps][c]["asd"].append(a)

        if (i + 1) % 10 == 0 or (i + 1) == len(common_ids):
            print(f"  [{i + 1}/{len(common_ids)}] processed")

        del x_wg, y_wg, x_wg_pad, y_wg_pad, x_z, x_z_pad
        torch.cuda.empty_cache() if device.type == "cuda" else None

    path = os.path.join(output_dir, "cascaded_attack_results.csv")
    with open(path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(
            ["epsilon", "wg_dice_mean", "wg_dice_std"]
            + [
                f"{cn}_{m}"
                for cn in z_class_names
                for m in ["dice_mean", "dice_std", "hd95_mean", "asd_mean"]
            ]
        )
        for eps in eps_list:
            wg_d = np.array(wg_results[eps][1]["dice"], dtype=np.float64)
            row = [
                eps,
                f"{wg_d.mean():.6f}",
                f"{wg_d.std(ddof=1):.6f}" if len(wg_d) > 1 else "nan",
            ]
            for c, cn in zip(z_fg, z_class_names):
                for m in ["dice", "hd95", "asd"]:
                    vals = np.array(zone_results[eps][c][m], dtype=np.float64)
                    vals = vals[np.isfinite(vals)]
                    if m == "dice":
                        row.append(f"{vals.mean():.6f}" if len(vals) > 0 else "nan")
                        row.append(
                            f"{vals.std(ddof=1):.6f}" if len(vals) > 1 else "nan"
                        )
                    else:
                        row.append(f"{vals.mean():.6f}" if len(vals) > 0 else "nan")
            writer.writerow(row)
    print(f"  Saved {path}")

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    wg_means = [np.nanmean(wg_results[eps][1]["dice"]) for eps in eps_list]
    axes[0].plot(eps_list, wg_means, "o-", label="WG Dice (attacked)")
    axes[0].set_xlabel("FGSM Epsilon (applied to WG)")
    axes[0].set_ylabel("Dice")
    axes[0].set_title("WG Model Dice Under Attack")
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)

    for c, cn in zip(z_fg, z_class_names):
        z_means = [np.nanmean(zone_results[eps][c]["dice"]) for eps in eps_list]
        axes[1].plot(eps_list, z_means, "o-", markersize=4, label=cn)
    axes[1].set_xlabel("FGSM Epsilon (applied to WG)")
    axes[1].set_ylabel("Dice")
    axes[1].set_title("Zones Model Dice (Cascaded Attack)")
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)

    plt.tight_layout()
    path = os.path.join(output_dir, "cascaded_attack_plot.png")
    fig.savefig(path, dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved {path}")

    del wg_net, z_net
    gc.collect()
    torch.cuda.empty_cache() if device.type == "cuda" else None


def main():
    parser = argparse.ArgumentParser(
        description="Transferability and cascaded pipeline attack analysis"
    )
    parser.add_argument(
        "--analysis", choices=["transfer", "cascaded", "both"], default="both"
    )
    parser.add_argument("--max-samples", type=int, default=None)
    parser.add_argument("--output-dir", default="transfer_results")
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    if args.analysis in ("transfer", "both"):
        evaluate_transferability(
            device, max_samples=args.max_samples, output_dir=args.output_dir
        )
    if args.analysis in ("cascaded", "both"):
        evaluate_cascaded_attack(
            device, max_samples=args.max_samples, output_dir=args.output_dir
        )

    print("\nDone.")


if __name__ == "__main__":
    main()
