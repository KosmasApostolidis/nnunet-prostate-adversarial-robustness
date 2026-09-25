"""
Uncertainty Estimation for nnU-Net Segmentation Models.

Analyses:
  - Monte Carlo (MC) Dropout: enable dropout at inference, run N forward passes,
    compute voxel-wise prediction variance.
  - Entropy maps: voxel-wise softmax/sigmoid entropy on clean and adversarial inputs.
  - Calibration analysis: reliability diagrams and Expected Calibration Error (ECE)
    on foreground voxels only (avoids background-dominated ECE).

Population entropy CSVs report mean entropy over foreground (gt > 0) voxels.
"""

import os
import csv
import gc
import math
import random
import argparse
import numpy as np
import torch
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
    _pick_foreground_slice,
)


# ---------------------------------------------------------------------------
# MC Dropout utilities
# ---------------------------------------------------------------------------


def _enable_dropout(model: torch.nn.Module) -> int:
    """Set all dropout layers to train mode so they remain active at inference.
    Returns the number of dropout modules toggled.
    """
    n = 0
    for m in model.modules():
        if isinstance(m, (torch.nn.Dropout, torch.nn.Dropout2d, torch.nn.Dropout3d)):
            m.train()
            n += 1
    return n


def _epsilon_label(eps: float) -> str:
    return "Clean" if math.isclose(eps, 0.0, abs_tol=0.0, rel_tol=0.0) else f"ε={eps}"


def _axes_row(n_cols: int):
    """Return (fig, axes) with axes always 1D indexable; handles n_cols==1."""
    fig, axes = plt.subplots(1, n_cols, figsize=(4 * n_cols, 4))
    if n_cols == 1:
        axes = np.array([axes])
    else:
        axes = np.asarray(axes).ravel()
    return fig, axes


def _get_probs(logits: torch.Tensor, num_classes: int) -> torch.Tensor:
    """Convert logits to probabilities.
    Returns tensor of shape [1, C, D, H, W] where C = num_classes-1 for binary,
    or full C for multi-class.
    """
    if num_classes == 2:
        return torch.sigmoid(logits[:, 1:2])
    return torch.softmax(logits, dim=1)


def mc_dropout_inference(
    model: torch.nn.Module,
    x_padded: torch.Tensor,
    orig_shape: list,
    num_classes: int,
    n_forward: int = 20,
) -> tuple[np.ndarray, np.ndarray]:
    """Run N forward passes with dropout enabled.
    Returns (mean_probs, var_probs) as numpy arrays [C, D, H, W].
    """
    n_drop = _enable_dropout(model)
    if n_drop == 0:
        print("  Warning: no Dropout layers found; MC dropout variance will be ~0.")
    all_probs = []
    try:
        for _ in range(n_forward):
            with torch.no_grad():
                logits = unpad(model(x_padded), orig_shape)
                probs = _get_probs(logits, num_classes)
                all_probs.append(probs.cpu().numpy()[0])
    finally:
        model.eval()

    stacked = np.stack(all_probs, axis=0)
    mean_probs = stacked.mean(axis=0)
    var_probs = stacked.var(axis=0)
    if var_probs.ndim != 4:
        raise ValueError(
            f"expected var_probs ndim 4 [C,D,H,W], got shape {var_probs.shape}"
        )
    return mean_probs, var_probs


# ---------------------------------------------------------------------------
# Entropy maps
# ---------------------------------------------------------------------------


def _compute_entropy(logits: torch.Tensor, num_classes: int) -> np.ndarray:
    """Voxel-wise entropy of the softmax/sigmoid output. Returns [D, H, W]."""
    if num_classes == 2:
        p = torch.sigmoid(logits[0, 1]).detach().cpu().numpy()
        p = np.clip(p, 1e-7, 1.0 - 1e-7)
        return -(p * np.log(p) + (1 - p) * np.log(1 - p))
    else:
        probs = torch.softmax(logits, dim=1)[0].detach().cpu().numpy()
        probs = np.clip(probs, 1e-7, 1.0)
        return -(probs * np.log(probs)).sum(axis=0)


# ---------------------------------------------------------------------------
# Calibration
# ---------------------------------------------------------------------------


def _compute_calibration(
    probs_flat: np.ndarray, labels_flat: np.ndarray, n_bins: int = 15
):
    """Compute reliability diagram data and ECE.
    probs_flat: predicted probability of the positive class.
    labels_flat: binary ground truth.
    """
    bin_edges = np.linspace(0, 1, n_bins + 1)
    bin_accs = np.zeros(n_bins)
    bin_confs = np.zeros(n_bins)
    bin_counts = np.zeros(n_bins, dtype=np.int64)

    for b in range(n_bins):
        lo, hi = bin_edges[b], bin_edges[b + 1]
        if b == n_bins - 1:
            mask = (probs_flat >= lo) & (probs_flat <= hi)
        else:
            mask = (probs_flat >= lo) & (probs_flat < hi)
        if mask.sum() == 0:
            continue
        bin_accs[b] = labels_flat[mask].mean()
        bin_confs[b] = probs_flat[mask].mean()
        bin_counts[b] = mask.sum()

    total = bin_counts.sum()
    ece = float(np.sum(bin_counts * np.abs(bin_accs - bin_confs)) / max(total, 1))
    return bin_accs, bin_confs, bin_counts, ece


def _save_reliability_diagram(
    bin_accs, bin_confs, bin_counts, ece, n_bins, title, output_path
):
    fig, ax = plt.subplots(figsize=(6, 5))
    bar_width = 1.0 / n_bins
    positions = np.linspace(bar_width / 2, 1.0 - bar_width / 2, n_bins)

    mask = bin_counts > 0
    ax.bar(
        positions[mask],
        bin_accs[mask],
        width=bar_width * 0.9,
        color="steelblue",
        alpha=0.7,
        label="Accuracy",
    )
    ax.plot([0, 1], [0, 1], "k--", alpha=0.4, label="Perfect calibration")
    ax.scatter(
        positions[mask],
        bin_confs[mask],
        color="red",
        s=20,
        zorder=5,
        label="Avg confidence",
    )
    ax.set_xlabel("Mean Predicted Probability")
    ax.set_ylabel("Fraction of Positives")
    ax.set_title(f"{title}\nECE = {ece:.4f}")
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.2)
    plt.tight_layout()
    fig.savefig(output_path, dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved {output_path}")


# ---------------------------------------------------------------------------
# Main analysis
# ---------------------------------------------------------------------------


def analyze_uncertainty(
    model_key: str,
    device: torch.device,
    n_patients: int = 5,
    n_forward: int = 20,
    target_epsilons: list[float] | None = None,
    output_dir: str = "uncertainty_results",
    seed: int | None = None,
    n_bins: int = 15,
):
    cfg = MODEL_CONFIGS[model_key]
    print(f"\n{'=' * 60}")
    print(f"Uncertainty Analysis: {cfg['name']} ({model_key})")
    print(f"{'=' * 60}")

    if seed is not None:
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)

    if target_epsilons is None:
        target_epsilons = [0.0, 0.5, 1.0]

    network, div_factors = load_model(
        cfg["checkpoint"], cfg["configuration"], cfg["num_classes"], device
    )
    val_ids = load_validation_ids(cfg["splits_json"])
    if not val_ids:
        raise ValueError("No validation IDs in splits; check splits_json and data.")
    k = min(n_patients, len(val_ids))
    selected = random.sample(val_ids, k)
    num_classes = cfg["num_classes"]
    class_names = cfg["class_names"]
    fg_classes = list(range(1, num_classes))

    os.makedirs(output_dir, exist_ok=True)

    all_calib_data = {eps: {"probs": [], "labels": []} for eps in target_epsilons}

    for case_id in selected:
        img_np, seg_np = load_sample(cfg["data_dir"], case_id)
        orig_seg_np = np.clip(seg_np, 0, None)
        gt = orig_seg_np[0]

        x = torch.from_numpy(img_np[np.newaxis]).to(device)
        y = torch.from_numpy(orig_seg_np[np.newaxis].astype(np.float32)).to(device)
        x_padded, orig_shape = pad_to_divisible(x, div_factors)
        y_padded, _ = pad_to_divisible(y, div_factors)

        slice_idx = _pick_foreground_slice(gt)
        mri_slice = img_np[0, slice_idx]
        vmin_mri, vmax_mri = np.percentile(mri_slice, [1, 99])
        mri_normed = np.clip(
            (mri_slice - vmin_mri) / (vmax_mri - vmin_mri + 1e-8), 0, 1
        )

        print(f"  Patient={case_id}, slice={slice_idx}")

        # One FGSM per epsilon; reuse for MC dropout, entropy, and calibration.
        x_adv_by_eps: dict[float, torch.Tensor] = {}
        for eps in target_epsilons:
            with torch.enable_grad():
                x_adv_by_eps[eps] = fgsm_attack(
                    network, x_padded, y_padded, eps, num_classes
                )

        # --- MC Dropout ---
        mc_panels = []
        mc_titles = []
        for eps in target_epsilons:
            x_input = x_adv_by_eps[eps]
            mean_p, var_p = mc_dropout_inference(
                network, x_input, orig_shape, num_classes, n_forward
            )
            total_var = var_p.sum(axis=0)
            mc_panels.append(total_var[slice_idx])
            label = _epsilon_label(eps)
            mc_titles.append(f"{label}\nmax var: {total_var[slice_idx].max():.4f}")

        n_eps = len(target_epsilons)
        fig, axes = _axes_row(n_eps + 1)
        axes[0].imshow(mri_normed, cmap="gray")
        if gt[slice_idx].max() > 0:
            axes[0].contour(
                gt[slice_idx] > 0, levels=[0.5], colors=["cyan"], linewidths=1
            )
        axes[0].set_title("MRI + GT", fontsize=9)
        axes[0].axis("off")
        for j, (panel, title) in enumerate(zip(mc_panels, mc_titles)):
            im = axes[j + 1].imshow(panel, cmap="magma", interpolation="nearest")
            axes[j + 1].set_title(title, fontsize=9)
            axes[j + 1].axis("off")
            fig.colorbar(im, ax=axes[j + 1], fraction=0.046, pad=0.04)
        fig.suptitle(
            f"{cfg['name']} — MC Dropout Variance — {case_id} (slice {slice_idx})",
            fontsize=10,
            fontweight="bold",
        )
        plt.tight_layout()
        path = os.path.join(output_dir, f"{model_key}_{case_id}_mc_dropout.png")
        fig.savefig(path, dpi=200, bbox_inches="tight")
        plt.close(fig)
        print(f"  Saved {path}")

        # --- Entropy Maps ---
        entropy_panels = []
        entropy_titles = []
        for eps in target_epsilons:
            with torch.no_grad():
                logits = unpad(network(x_adv_by_eps[eps]), orig_shape)
            ent = _compute_entropy(logits, num_classes)
            entropy_panels.append(ent[slice_idx])
            label = _epsilon_label(eps)
            entropy_titles.append(f"{label}\nmean H: {ent[slice_idx].mean():.4f}")

        fig, axes = _axes_row(n_eps + 1)
        axes[0].imshow(mri_normed, cmap="gray")
        if gt[slice_idx].max() > 0:
            axes[0].contour(
                gt[slice_idx] > 0, levels=[0.5], colors=["cyan"], linewidths=1
            )
        axes[0].set_title("MRI + GT", fontsize=9)
        axes[0].axis("off")
        for j, (panel, title) in enumerate(zip(entropy_panels, entropy_titles)):
            im = axes[j + 1].imshow(panel, cmap="viridis", interpolation="nearest")
            axes[j + 1].set_title(title, fontsize=9)
            axes[j + 1].axis("off")
            fig.colorbar(im, ax=axes[j + 1], fraction=0.046, pad=0.04)
        fig.suptitle(
            f"{cfg['name']} — Entropy Map — {case_id} (slice {slice_idx})",
            fontsize=10,
            fontweight="bold",
        )
        plt.tight_layout()
        path = os.path.join(output_dir, f"{model_key}_{case_id}_entropy.png")
        fig.savefig(path, dpi=200, bbox_inches="tight")
        plt.close(fig)
        print(f"  Saved {path}")

        # --- Calibration (foreground voxels only; ECE is not dominated by background) ---
        fg_flat = gt.ravel() > 0
        if np.any(fg_flat):
            for eps in target_epsilons:
                with torch.no_grad():
                    logits = unpad(network(x_adv_by_eps[eps]), orig_shape)
                probs = _get_probs(logits, num_classes).cpu().numpy()[0]

                for c in fg_classes:
                    if num_classes == 2:
                        p_flat = probs[0].ravel()[fg_flat]
                    else:
                        p_flat = probs[c].ravel()[fg_flat]
                    gt_c = (gt == c).astype(np.float32).ravel()[fg_flat]
                    all_calib_data[eps]["probs"].append(p_flat)
                    all_calib_data[eps]["labels"].append(gt_c)

        del x, y, x_padded, y_padded
        if device.type == "cuda":
            torch.cuda.empty_cache()

    # --- Calibration diagrams ---
    calib_rows = []
    for eps in target_epsilons:
        if not all_calib_data[eps]["probs"]:
            continue
        probs_all = np.concatenate(all_calib_data[eps]["probs"])
        labels_all = np.concatenate(all_calib_data[eps]["labels"])
        bin_accs, bin_confs, bin_counts, ece = _compute_calibration(
            probs_all, labels_all, n_bins
        )
        label = _epsilon_label(eps)
        _save_reliability_diagram(
            bin_accs,
            bin_confs,
            bin_counts,
            ece,
            n_bins,
            title=f"{cfg['name']} — Reliability Diagram ({label}) — foreground voxels",
            output_path=os.path.join(
                output_dir, f"{model_key}_reliability_eps{eps}.png"
            ),
        )
        calib_rows.append({"model": model_key, "epsilon": eps, "ECE": ece})

    if calib_rows:
        import pandas as pd

        pd.DataFrame(calib_rows).to_csv(
            os.path.join(output_dir, f"{model_key}_ece_summary.csv"),
            index=False,
            float_format="%.6f",
        )
        print(f"  Saved {model_key}_ece_summary.csv")

        fig, ax = plt.subplots(figsize=(7, 4))
        eces = [r["ECE"] for r in calib_rows]
        ax.bar([f"{r['epsilon']}" for r in calib_rows], eces, color="teal", alpha=0.7)
        ax.set_xlabel("FGSM Epsilon")
        ax.set_ylabel("ECE")
        ax.set_title(
            f"{cfg['name']} — Expected Calibration Error vs Epsilon (foreground)"
        )
        ax.grid(True, alpha=0.2, axis="y")
        plt.tight_layout()
        path = os.path.join(output_dir, f"{model_key}_ece_vs_epsilon.png")
        fig.savefig(path, dpi=200, bbox_inches="tight")
        plt.close(fig)
        print(f"  Saved {path}")

    del network
    gc.collect()
    if device.type == "cuda":
        torch.cuda.empty_cache()


# ---------------------------------------------------------------------------
# Population-level entropy analysis (all patients, all epsilons)
# ---------------------------------------------------------------------------


def compute_population_entropy(
    model_key: str,
    device: torch.device,
    epsilons: list[float] | None = None,
    max_samples: int | None = None,
    output_dir: str = "uncertainty_results",
    seed: int | None = None,
):
    """Compute mean voxel entropy (foreground voxels only) per patient for every epsilon.
    Saves a CSV with one row per (case, epsilon) and returns a dict
    {epsilon: list_of_mean_entropies}.
    """
    cfg = MODEL_CONFIGS[model_key]
    print(f"\n{'=' * 60}")
    print(f"Population Entropy: {cfg['name']} ({model_key})")
    print(f"{'=' * 60}")

    if epsilons is None:
        epsilons = cfg["epsilons"]

    if seed is not None:
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)

    network, div_factors = load_model(
        cfg["checkpoint"], cfg["configuration"], cfg["num_classes"], device
    )
    val_ids = load_validation_ids(cfg["splits_json"])
    if not val_ids:
        raise ValueError("No validation IDs in splits; check splits_json and data.")
    if max_samples is not None and max_samples < len(val_ids):
        rng = random.Random(seed) if seed is not None else random
        val_ids = rng.sample(val_ids, max_samples)
    num_classes = cfg["num_classes"]

    os.makedirs(output_dir, exist_ok=True)

    entropy_per_eps = {eps: [] for eps in epsilons}

    csv_path = os.path.join(output_dir, f"{model_key}_entropy_per_sample.csv")
    with open(csv_path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["case_id", "epsilon", "mean_entropy_fg"])

        for i, case_id in enumerate(val_ids):
            img_np, seg_np = load_sample(cfg["data_dir"], case_id)
            orig_seg_np = np.clip(seg_np, 0, None)
            gt = orig_seg_np[0]
            fg = gt > 0

            x = torch.from_numpy(img_np[np.newaxis]).to(device)
            y = torch.from_numpy(orig_seg_np[np.newaxis].astype(np.float32)).to(device)
            x_padded, orig_shape = pad_to_divisible(x, div_factors)
            y_padded, _ = pad_to_divisible(y, div_factors)

            for eps in epsilons:
                with torch.enable_grad():
                    x_input = fgsm_attack(network, x_padded, y_padded, eps, num_classes)
                with torch.no_grad():
                    logits = unpad(network(x_input), orig_shape)
                ent = _compute_entropy(logits, num_classes)
                if np.any(fg):
                    mean_ent = float(ent[fg].mean())
                else:
                    mean_ent = float("nan")
                entropy_per_eps[eps].append(mean_ent)
                writer.writerow([case_id, eps, f"{mean_ent:.6f}"])

            if (i + 1) % 10 == 0 or (i + 1) == len(val_ids):
                print(f"  [{i + 1}/{len(val_ids)}] processed")

            del x, y, x_padded, y_padded
            if device.type == "cuda":
                torch.cuda.empty_cache()

    print(f"  Saved {csv_path}")

    agg_path = os.path.join(output_dir, f"{model_key}_entropy_vs_epsilon.csv")
    with open(agg_path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["model", "epsilon", "mean_entropy_fg", "std_entropy_fg", "n"])
        for eps in epsilons:
            vals = np.array(entropy_per_eps[eps])
            writer.writerow(
                [
                    model_key,
                    eps,
                    f"{vals.mean():.6f}",
                    f"{vals.std(ddof=1):.6f}" if len(vals) > 1 else "nan",
                    len(vals),
                ]
            )
    print(f"  Saved {agg_path}")

    del network
    gc.collect()
    if device.type == "cuda":
        torch.cuda.empty_cache()

    return entropy_per_eps


def plot_combined_entropy(output_dir: str = "uncertainty_results"):
    """Load entropy-vs-epsilon CSVs for both models and plot them together."""
    import pandas as pd

    frames = []
    for mk in ("wg", "zones"):
        path = os.path.join(output_dir, f"{mk}_entropy_vs_epsilon.csv")
        if os.path.isfile(path):
            frames.append(pd.read_csv(path))
    if not frames:
        print("  No entropy CSVs found to plot.")
        return

    df = pd.concat(frames, ignore_index=True)

    # Support both legacy and new column names
    mean_col = "mean_entropy_fg" if "mean_entropy_fg" in df.columns else "mean_entropy"
    std_col = "std_entropy_fg" if "std_entropy_fg" in df.columns else "std_entropy"

    model_labels = {"wg": "Whole Gland", "zones": "Prostate Zones"}
    colors = {"wg": "#1f77b4", "zones": "#ff7f0e"}

    fig, ax = plt.subplots(figsize=(9, 5.5))
    for mk in df["model"].unique():
        sub = df[df["model"] == mk].sort_values("epsilon")
        label = model_labels.get(mk, mk)
        ax.errorbar(
            sub["epsilon"],
            sub[mean_col],
            yerr=sub[std_col],
            marker="o",
            capsize=4,
            markersize=5,
            linewidth=1.8,
            color=colors.get(mk),
            label=label,
        )
        ax.fill_between(
            sub["epsilon"],
            sub[mean_col] - sub[std_col],
            sub[mean_col] + sub[std_col],
            alpha=0.15,
            color=colors.get(mk),
        )

    ax.set_xlabel("FGSM Epsilon", fontsize=11)
    ax.set_ylabel("Mean entropy (foreground voxels)", fontsize=11)
    ax.set_title(
        "Mean entropy (± std) vs FGSM ε — foreground", fontsize=12, fontweight="bold"
    )
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.25)
    plt.tight_layout()

    path_png = os.path.join(output_dir, "combined_entropy_vs_epsilon.png")
    fig.savefig(path_png, dpi=300, bbox_inches="tight")
    path_pdf = os.path.join(output_dir, "combined_entropy_vs_epsilon.pdf")
    fig.savefig(path_pdf, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved {path_png}")
    print(f"  Saved {path_pdf}")


def main():
    parser = argparse.ArgumentParser(description="Uncertainty estimation analysis")
    parser.add_argument("--model", choices=["wg", "zones", "both"], default="both")
    parser.add_argument(
        "--n-patients",
        type=int,
        default=5,
        help="Number of patients for per-patient visualisations",
    )
    parser.add_argument(
        "--n-forward", type=int, default=20, help="Number of MC dropout forward passes"
    )
    parser.add_argument("--epsilons", type=float, nargs="+", default=[0.0, 0.5, 1.0])
    parser.add_argument("--output-dir", default="uncertainty_results")
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument(
        "--max-samples",
        type=int,
        default=None,
        help="Limit validation samples for population entropy",
    )
    subparsers = parser.add_subparsers(dest="command")

    subparsers.add_parser(
        "per-patient",
        help="Per-patient visualisations (MC dropout, entropy maps, calibration)",
    )
    subparsers.add_parser(
        "entropy", help="Population-level entropy across all patients and epsilons"
    )
    subparsers.add_parser(
        "all", help="Run both per-patient and population entropy analyses"
    )

    args = parser.parse_args()

    if args.command is None:
        args.command = "all"

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    models = ["wg", "zones"] if args.model == "both" else [args.model]

    if args.command in ("per-patient", "all"):
        for mk in models:
            analyze_uncertainty(
                mk,
                device,
                n_patients=args.n_patients,
                n_forward=args.n_forward,
                target_epsilons=sorted(args.epsilons),
                output_dir=args.output_dir,
                seed=args.seed,
            )

    if args.command in ("entropy", "all"):
        for mk in models:
            compute_population_entropy(
                mk,
                device,
                epsilons=sorted(args.epsilons),
                max_samples=args.max_samples,
                output_dir=args.output_dir,
                seed=args.seed,
            )
        plot_combined_entropy(args.output_dir)

    print("\nDone.")


if __name__ == "__main__":
    main()
