"""Generate the manuscript's Figures 1-2: worst-case perturbation magnitude vs APGD epsilon.

Each figure shows, for one worst-affected case, how the model output degrades as the
APGD budget grows across epsilon in {0, 2, 4, 8, 16}/255:

* Top row  - a representative axial slice of the *perturbed* scan with the reference
  mask (blue, solid) and the model prediction (red, dashed) overlaid, annotated with
  the resulting per-case Dice (WG Dice for whole gland; macro Dice for zones).
* Bottom row - the distribution of the applied voxel-wise perturbation (delta = adv - clean,
  in normalized-input units), with the +/- epsilon_norm L-inf bounds marked.

The attack, model loading, preprocessing, seeding, and Dice are the *same* code path used by
``evaluate_adversarial_segmentation_all_cases.py`` to produce
``all_cases_adversarial_segmentation*/adversarial_segmentation_all_cases.csv``. With the default
epsilon grid, step count, and base seed, the annotated per-case Dice reproduce that CSV, so the
figure is a faithful visualization of the recorded worst case rather than a separate pipeline.

Example
-------
    python experiments/generate_worst_case_perturbation_figures.py --datasets wg zones
    python experiments/generate_worst_case_perturbation_figures.py --datasets wg --case-id ProstateWG_10211
"""

from __future__ import annotations

import argparse
import math
import os
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.lines import Line2D  # noqa: E402

REPO_ROOT = Path(__file__).resolve().parents[1]
# Support the repo's src/ layout whether or not the package is pip-installed.
for _root in (REPO_ROOT / "src", REPO_ROOT):
    if str(_root) not in sys.path:
        sys.path.insert(0, str(_root))

from experiments.adv_rob_eval_nnunet_nnunetrecenc import (  # noqa: E402
    DATASETS,
    IGNORE_LABEL,
    _pad_to,
    build_model,
    load_sample,
)
from experiments.evaluate_adversarial_quality_all_cases import (  # noqa: E402
    _dataset_paths,
    apgd_bce_independent_batch,
)
from mri_prostate_seg.experiments.epsilon_calibration import stable_seed  # noqa: E402
from mri_prostate_seg.metrics.segmentation import dice_score  # noqa: E402

DEFAULT_EPS_N = [0, 2, 4, 8, 16]
DEFAULT_STEPS = 20
DEFAULT_BASE_SEED = 20260721
DEFAULT_SELECT_EPS_N = 16
DEFAULT_MIN_CLEAN = 0.90
DEFAULT_OUTPUT_DIR = (
    REPO_ROOT
    / "results"
    / "adv_rob_eval_results"
    / "image_quality_analysis"
    / "worst_case_perturbation_figures"
)
DEFAULT_PERCASE_CSV = (
    REPO_ROOT
    / "results"
    / "adv_rob_eval_results"
    / "image_quality_analysis"
    / "all_cases_adversarial_segmentation_with_boundaries"
    / "adversarial_segmentation_all_cases.csv"
)
FIGURE_INDEX = {"wg": 1, "zones": 2}
GT_COLOR = "#1f6fff"
PRED_COLOR = "#ff2b2b"


def configure_deterministic(seed: int) -> None:
    """Match the calibration run's numerics so attacks are reproducible."""

    os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    torch.use_deterministic_algorithms(True)


def head_class(dataset_key: str) -> str:
    """The Dice reported in the manuscript's panel titles."""

    return "WG" if dataset_key == "wg" else "macro"


def select_worst_case(
    dataset_key: str, *, percase_csv: Path, select_eps_n: int, min_clean: float
) -> str:
    """Pick the near-perfect case that collapses hardest at the top budget."""

    if not percase_csv.is_file():
        raise SystemExit(
            f"per-case CSV not found: {percase_csv}\n"
            "Pass --case-id explicitly, or point --percase-csv at the segmentation output."
        )
    frame = pd.read_csv(
        percase_csv,
        usecols=[
            "dataset",
            "case_id",
            "attack",
            "class",
            "epsilon_n",
            "clean_dice",
            "adversarial_dice",
        ],
    )
    subset = frame[
        (frame["dataset"] == dataset_key)
        & (frame["attack"] == "APGD-BCE")
        & (frame["class"] == head_class(dataset_key))
        & (frame["epsilon_n"] == select_eps_n)
        & (frame["clean_dice"] >= min_clean)
    ]
    if subset.empty:
        raise SystemExit(
            f"no {dataset_key} case with clean Dice >= {min_clean} at eps={select_eps_n}; "
            "lower --min-clean or pass --case-id."
        )
    row = subset.sort_values("adversarial_dice", kind="stable").iloc[0]
    print(
        f"[{dataset_key}] selected {row['case_id']} "
        f"(clean {row['clean_dice']:.3f} -> APGD@{select_eps_n} {row['adversarial_dice']:.3f})",
        flush=True,
    )
    return str(row["case_id"])


def run_case_sweep(
    dataset_key: str,
    case_id: str,
    *,
    eps_n_values: list[int],
    steps: int,
    base_seed: int,
    device: torch.device,
    model: torch.nn.Module,
    div_factors: list[int],
    num_classes: int,
    data_dir: str,
    attack_valid_only: bool = False,
) -> dict[str, object]:
    """Run APGD at each budget for one case, mirroring the segmentation evaluator."""

    image_np, seg_np, spacing = load_sample(data_dir, case_id)
    original_shape = tuple(int(v) for v in image_np.shape[1:])
    target_shape = [
        int(math.ceil(size / factor) * factor)
        for size, factor in zip(original_shape, div_factors)
    ]
    image_tensor = _pad_to(
        torch.from_numpy(image_np[np.newaxis]).to(device), target_shape
    )
    label_tensor = _pad_to(
        torch.from_numpy(seg_np[np.newaxis].astype(np.float32)).to(device),
        target_shape,
        mode="constant",
        value=float(IGNORE_LABEL),
    )
    depth, height, width = original_shape
    perturbation_mask = (label_tensor != IGNORE_LABEL) if attack_valid_only else None
    clean_volume = np.asarray(image_np[0], dtype=np.float32)
    ground_truth = np.asarray(seg_np).squeeze().astype(np.int16)
    class_names = list(DATASETS[dataset_key]["class_names"])

    results: list[dict[str, object]] = []
    for eps_n in eps_n_values:
        if eps_n == 0:
            adversarial_tensor = image_tensor
        else:
            seeds = [
                stable_seed(
                    dataset_key, case_id, "apgd_bce", eps_n, base_seed=base_seed
                )
            ]
            with torch.enable_grad():
                adversarial_tensor = apgd_bce_independent_batch(
                    model,
                    image_tensor,
                    label_tensor,
                    eps_n / 255.0,
                    num_classes,
                    n_steps=steps,
                    seeds=seeds,
                    perturbation_mask=perturbation_mask,
                )
        with torch.no_grad():
            prediction_tensor = model(adversarial_tensor).argmax(dim=1)

        prediction = (
            prediction_tensor[0, :depth, :height, :width]
            .detach()
            .cpu()
            .numpy()
            .astype(np.int16)
        )
        adversarial_volume = (
            adversarial_tensor[0, 0, :depth, :height, :width]
            .detach()
            .cpu()
            .numpy()
            .astype(np.float32)
        )
        delta = adversarial_volume - clean_volume
        per_class = {
            name: float(dice_score(prediction == label, ground_truth == label))
            for label, name in enumerate(class_names, start=1)
        }
        macro = float(np.nanmean(list(per_class.values())))
        results.append(
            {
                "eps_n": eps_n,
                "eps_norm": eps_n / 255.0,
                "adversarial": adversarial_volume,
                "prediction": prediction,
                "delta": delta,
                "per_class": per_class,
                "macro": macro,
                "head_dice": per_class["WG"] if dataset_key == "wg" else macro,
                "rms_norm": float(np.sqrt(np.mean(np.square(delta)))),
                "linf_norm": float(np.max(np.abs(delta))),
            }
        )
        if eps_n != 0 and device.type == "cuda":
            del adversarial_tensor
            torch.cuda.empty_cache()

    return {
        "dataset_key": dataset_key,
        "case_id": case_id,
        "clean": clean_volume,
        "ground_truth": ground_truth,
        "class_names": class_names,
        "spacing": tuple(float(v) for v in spacing),
        "results": results,
    }


def _display_slice(ground_truth: np.ndarray) -> int:
    """Axial slice with the largest reference-foreground area (fixed across budgets)."""

    foreground_area = (ground_truth > 0).sum(axis=(1, 2))
    return int(np.argmax(foreground_area))


def render_figure(sweep: dict[str, object], *, output_path: Path, dpi: int) -> None:
    dataset_key = str(sweep["dataset_key"])
    results = list(sweep["results"])  # type: ignore[arg-type]
    ground_truth = np.asarray(sweep["ground_truth"])
    clean = np.asarray(sweep["clean"])
    class_names = list(sweep["class_names"])  # type: ignore[arg-type]
    slice_index = _display_slice(ground_truth)
    window_low, window_high = np.percentile(clean[slice_index], [1.0, 99.0])
    max_eps_norm = (
        max((float(r["eps_norm"]) for r in results), default=0.0) or 1.0 / 255.0
    )
    dice_label = "WG Dice" if dataset_key == "wg" else "macro Dice"

    columns = len(results)
    fig, axes = plt.subplots(2, columns, figsize=(3.05 * columns, 6.6), squeeze=False)
    for column, result in enumerate(results):
        top = axes[0][column]
        adversarial_slice = np.asarray(result["adversarial"])[slice_index]
        top.imshow(adversarial_slice, cmap="gray", vmin=window_low, vmax=window_high)
        gt_slice = ground_truth[slice_index]
        pred_slice = np.asarray(result["prediction"])[slice_index]
        for label in range(1, len(class_names) + 1):
            if np.any(gt_slice == label):
                top.contour(
                    gt_slice == label, levels=[0.5], colors=GT_COLOR, linewidths=1.4
                )
            if np.any(pred_slice == label):
                top.contour(
                    pred_slice == label,
                    levels=[0.5],
                    colors=PRED_COLOR,
                    linewidths=1.1,
                    linestyles="dashed",
                )
        top.set_title(
            rf"$\varepsilon = {int(result['eps_n'])}/255$" + "\n"
            f"{dice_label} = {float(result['head_dice']):.3f}",
            fontsize=10,
        )
        top.set_xticks([])
        top.set_yticks([])

        bottom = axes[1][column]
        delta = np.asarray(result["delta"]).ravel()
        bottom.hist(
            delta,
            bins=61,
            range=(-max_eps_norm, max_eps_norm),
            color="#4c4c4c",
            edgecolor="none",
        )
        eps_norm = float(result["eps_norm"])
        if eps_norm > 0:
            for bound in (-eps_norm, eps_norm):
                bottom.axvline(bound, color=PRED_COLOR, linestyle=":", linewidth=1.0)
        bottom.set_yscale("log")
        bottom.set_xlim(-max_eps_norm, max_eps_norm)
        bottom.set_title(
            "no perturbation"
            if eps_norm == 0
            else f"RMS = {float(result['rms_norm']):.4f}",
            fontsize=9,
        )
        bottom.tick_params(labelsize=7)
        if column == 0:
            top.set_ylabel("axial slice", fontsize=9)
            bottom.set_ylabel("voxel count (log)", fontsize=8)
        bottom.set_xlabel("perturbation\n(norm. units)", fontsize=8)

    handles = [
        Line2D([0], [0], color=GT_COLOR, lw=1.6, label="reference mask"),
        Line2D(
            [0],
            [0],
            color=PRED_COLOR,
            lw=1.4,
            linestyle="dashed",
            label="model prediction",
        ),
    ]
    task_name = "Whole Gland" if dataset_key == "wg" else "Prostate Zones"
    fig.subplots_adjust(top=0.87, bottom=0.20, hspace=0.42, wspace=0.12)
    fig.suptitle(
        f"{task_name} — worst-case perturbation magnitude versus APGD $\\varepsilon$",
        fontsize=12,
        y=0.985,
    )
    fig.legend(
        handles=handles,
        loc="lower center",
        ncol=2,
        frameon=False,
        fontsize=9,
        bbox_to_anchor=(0.5, 0.02),
    )
    fig.text(
        0.012,
        0.012,
        f"case {sweep['case_id']}",
        ha="left",
        va="bottom",
        fontsize=7.5,
        color="#666666",
    )
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=dpi, bbox_inches="tight")
    fig.savefig(output_path.with_suffix(".pdf"), bbox_inches="tight")
    plt.close(fig)
    print(f"[{dataset_key}] wrote {output_path} (+ .pdf)", flush=True)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--datasets", nargs="+", default=["wg", "zones"], choices=["wg", "zones"]
    )
    parser.add_argument(
        "--case-id",
        default=None,
        help="Override auto-selection (only valid when a single dataset is requested).",
    )
    parser.add_argument("--eps-n", type=int, nargs="+", default=DEFAULT_EPS_N)
    parser.add_argument("--steps", type=int, default=DEFAULT_STEPS)
    parser.add_argument("--base-seed", type=int, default=DEFAULT_BASE_SEED)
    parser.add_argument("--select-eps-n", type=int, default=DEFAULT_SELECT_EPS_N)
    parser.add_argument("--min-clean", type=float, default=DEFAULT_MIN_CLEAN)
    parser.add_argument("--percase-csv", type=Path, default=DEFAULT_PERCASE_CSV)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument(
        "--attack-valid-only",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="confine the perturbation to non-ignore voxels (excludes the zero-filled "
        "margin outside the dilated gland on zones, and the model-grid padding). "
        "The published run went without it; use a fresh --output-dir.",
    )
    parser.add_argument(
        "--device", default=None, help="cuda / cpu (default: cuda if available)"
    )
    parser.add_argument("--dpi", type=int, default=200)
    parser.add_argument(
        "--no-deterministic",
        action="store_true",
        help="Skip torch deterministic mode (use if an op lacks a deterministic kernel).",
    )
    args = parser.parse_args()
    if args.case_id is not None and len(args.datasets) != 1:
        parser.error("--case-id requires exactly one --datasets value")
    return args


def main() -> None:
    args = parse_args()
    if not args.no_deterministic:
        configure_deterministic(args.base_seed)
    device = torch.device(
        args.device if args.device else ("cuda" if torch.cuda.is_available() else "cpu")
    )
    print(f"device: {device}", flush=True)

    for dataset_key in args.datasets:
        paths = _dataset_paths(dataset_key)
        case_id = args.case_id or select_worst_case(
            dataset_key,
            percase_csv=args.percase_csv,
            select_eps_n=args.select_eps_n,
            min_clean=args.min_clean,
        )
        model, div_factors, num_classes = build_model(str(paths["checkpoint"]), device)
        model.eval()
        model.requires_grad_(False)
        sweep = run_case_sweep(
            dataset_key,
            case_id,
            eps_n_values=list(args.eps_n),
            steps=args.steps,
            base_seed=args.base_seed,
            device=device,
            model=model,
            div_factors=div_factors,
            num_classes=num_classes,
            data_dir=str(paths["data"]),
            attack_valid_only=bool(args.attack_valid_only),
        )
        trajectory = ", ".join(
            f"{int(r['eps_n'])}:{float(r['head_dice']):.3f}" for r in sweep["results"]
        )
        print(
            f"[{dataset_key}] {case_id} {head_class(dataset_key)} Dice by eps -> {trajectory}",
            flush=True,
        )
        output_path = (
            args.output_dir
            / f"fig{FIGURE_INDEX[dataset_key]}_{dataset_key}_worst_case_perturbation.png"
        )
        render_figure(sweep, output_path=output_path, dpi=args.dpi)
        del model
        if device.type == "cuda":
            torch.cuda.empty_cache()


if __name__ == "__main__":
    main()
