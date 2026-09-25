"""Qualitative direction overlays for representative cases (spec §33).

For one case at one APGD-BCE epsilon, renders six panels on a shared axial
slice and crop: (1) clean MRI with the ground-truth contour and the clean
prediction contour, (2) the applied perturbation delta on a symmetric colour
scale, (3) perturbation energy delta^2, (4) clean MRI with the adversarial
prediction contour, (5) a semantic overlay of induced FP/FN and corrections,
and (6) a text box whose numbers are read directly from
``case_direction_summary.csv`` rather than recomputed here.

APGD does not reproduce bit-exactly between runs (constraints.md), so the
contours in panels 1-5 are a fresh regeneration of the sweep and may drift
slightly from the numbers quoted in panel 6; panel 6 is the source of truth.

Example
-------
    python experiments/plot_segmentation_direction_cases.py \
        --case-id wg:ProstateWG_10000 --eps-n 16
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.figure import Figure  # noqa: E402

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))
SRC_ROOT = REPO_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from experiments.adv_rob_eval_nnunet_nnunetrecenc import build_model  # noqa: E402
from experiments.evaluate_adversarial_quality_all_cases import (  # noqa: E402
    _dataset_paths,
)
from experiments.evaluate_segmentation_direction_all_cases import (  # noqa: E402
    DEFAULT_OUTPUT_DIR,
    SUMMARY_CSV,
    head_target,
)
from experiments.generate_worst_case_perturbation_figures import (  # noqa: E402
    DEFAULT_BASE_SEED,
    DEFAULT_MIN_CLEAN,
    DEFAULT_SELECT_EPS_N,
    DEFAULT_STEPS,
    _display_slice,
    head_class,
    run_case_sweep,
    select_worst_case,
)
from mri_prostate_seg.experiments.segmentation_direction.transitions import (  # noqa: E402
    binary_transition_masks,
)

APGD_ATTACK_LABEL = "APGD-BCE"
DEFAULT_EPS_N_PANELS = [8, 16, 32]
DEFAULT_SELECT_MODES = ["median", "worst", "inward"]
TEXT_BOX_KEYS = [
    "delta_dice",
    "damage_bias",
    "median_surface_motion_mm",
    "worsened_outward_surface_fraction",
    "worsened_inward_surface_fraction",
]


def _crop(gt: np.ndarray, pad: int = 16) -> tuple[slice, slice]:
    """In-plane bounding box of the GT footprint (any slice), padded by ``pad``."""

    ys, xs = np.nonzero(gt.any(axis=0))
    return (
        slice(max(ys.min() - pad, 0), ys.max() + pad),
        slice(max(xs.min() - pad, 0), xs.max() + pad),
    )


def overlay_panels(
    case: dict[str, object],
    *,
    eps_n: int,
    target: str,
    summary_row: pd.Series,
) -> Figure:
    """Six-panel qualitative overlay for one case at one APGD-BCE epsilon."""

    results = case["results"]
    result = next(r for r in results if r["eps_n"] == eps_n)
    gt_all = case["ground_truth"]
    clean_pred_all = next(r for r in results if r["eps_n"] == 0)["prediction"]
    label = 1 if target in ("WG", "TZ+CZ") else 2 if target == "PZ" else None
    gt = (gt_all > 0) if label is None else (gt_all == label)  # type: ignore[operator]
    clean_pred = (clean_pred_all > 0) if label is None else (clean_pred_all == label)
    adv_pred = (
        (result["prediction"] > 0) if label is None else (result["prediction"] == label)
    )
    tr = binary_transition_masks(gt, clean_pred, adv_pred, valid=gt_all >= 0)
    z = _display_slice(gt)
    rows, cols = _crop(gt)

    fig, axes = plt.subplots(2, 3, figsize=(13, 8))
    image = case["clean"][z, rows, cols]  # type: ignore[index]
    for ax in axes.ravel()[:5]:
        ax.imshow(image, cmap="gray")
        ax.set_axis_off()

    axes[0, 0].contour(gt[z, rows, cols], levels=[0.5], colors="#1f6fff")
    axes[0, 0].contour(
        clean_pred[z, rows, cols], levels=[0.5], colors="#ff2b2b", linestyles="--"
    )
    axes[0, 0].set_title("clean, GT (blue) / clean pred (red)")

    delta = result["delta"][z, rows, cols]
    lim = float(np.abs(delta).max()) or 1.0
    axes[0, 1].imshow(delta, cmap="RdBu_r", vmin=-lim, vmax=lim)
    axes[0, 1].set_title("δ (symmetric)")

    axes[0, 2].imshow(delta**2, cmap="magma")
    axes[0, 2].set_title("δ² energy")

    axes[1, 0].contour(adv_pred[z, rows, cols], levels=[0.5], colors="#ff9f1c")
    axes[1, 0].set_title("adversarial pred (orange)")

    overlay = np.zeros((*image.shape, 4))
    overlay[tr.induced_fp[z, rows, cols]] = (1.0, 0.3, 0.1, 0.6)
    overlay[tr.induced_fn[z, rows, cols]] = (0.1, 0.4, 1.0, 0.6)
    overlay[(tr.corrected_fp | tr.corrected_fn)[z, rows, cols]] = (0.1, 0.8, 0.2, 0.6)
    axes[1, 1].imshow(overlay)
    axes[1, 1].set_title("induced FP (warm) / FN (cool) / corrected (green)")

    axes[1, 2].set_axis_off()
    axes[1, 2].text(
        0.0,
        0.9,
        "\n".join(f"{k}: {summary_row[k]:.3f}" for k in TEXT_BOX_KEYS),
        va="top",
        family="monospace",
    )

    fig.suptitle(
        f"{case['dataset_key']} {case['case_id']} {target} "
        f"APGD-BCE ε={eps_n}/255 slice {z}"
    )
    fig.tight_layout()
    return fig


def _head_rows(frame: pd.DataFrame, dataset_key: str, eps_n: int) -> pd.DataFrame:
    target = head_target(dataset_key)
    return frame[
        (frame["dataset"] == dataset_key)
        & (frame["attack"] == APGD_ATTACK_LABEL)
        & (frame["class_or_union"] == target)
        & (frame["epsilon_n"] == eps_n)
    ]


def _select_median_case(frame: pd.DataFrame, dataset_key: str, eps_n: int) -> str:
    rows = _head_rows(frame, dataset_key, eps_n).dropna(subset=["damage_bias"])
    if rows.empty:
        raise SystemExit(
            f"no {dataset_key} rows with a damage_bias at eps={eps_n} to pick a median case"
        )
    median = rows["damage_bias"].median()
    idx = (rows["damage_bias"] - median).abs().idxmin()
    return str(rows.loc[idx, "case_id"])


def _select_inward_case(frame: pd.DataFrame, dataset_key: str, eps_n: int) -> str:
    column = "worsened_inward_surface_fraction"
    rows = _head_rows(frame, dataset_key, eps_n).dropna(subset=[column])
    if rows.empty:
        raise SystemExit(
            f"no {dataset_key} rows with {column} at eps={eps_n} to pick the atypical case"
        )
    idx = rows[column].idxmax()
    return str(rows.loc[idx, "case_id"])


def _write_worst_selection_csv(
    frame: pd.DataFrame, dataset_key: str, output_dir: Path
) -> Path:
    """``select_worst_case`` reads ``class``/``adversarial_dice``; write that shape."""

    target = head_target(dataset_key)
    subset = frame[
        (frame["dataset"] == dataset_key) & (frame["class_or_union"] == target)
    ].copy()
    subset["class"] = head_class(dataset_key)
    subset["adversarial_dice"] = subset["adv_dice"]
    output_dir.mkdir(parents=True, exist_ok=True)
    temp_path = output_dir / f"_worst_case_selection_{dataset_key}.csv"
    subset[
        [
            "dataset",
            "case_id",
            "attack",
            "class",
            "epsilon_n",
            "clean_dice",
            "adversarial_dice",
        ]
    ].to_csv(temp_path, index=False)
    return temp_path


def _select_case_ids(
    frame: pd.DataFrame,
    dataset_key: str,
    *,
    select: list[str],
    select_eps_n: int,
    output_dir: Path,
) -> list[str]:
    case_ids: list[str] = []
    for mode in select:
        if mode == "worst":
            temp_csv = _write_worst_selection_csv(frame, dataset_key, output_dir)
            case_id = select_worst_case(
                dataset_key,
                percase_csv=temp_csv,
                select_eps_n=select_eps_n,
                min_clean=DEFAULT_MIN_CLEAN,
            )
        elif mode == "median":
            case_id = _select_median_case(frame, dataset_key, select_eps_n)
        else:
            case_id = _select_inward_case(frame, dataset_key, select_eps_n)
        if case_id not in case_ids:
            case_ids.append(case_id)
    return case_ids


def _parse_case_id_specs(specs: list[str]) -> dict[str, list[str]]:
    explicit: dict[str, list[str]] = {}
    for spec in specs:
        dataset_key, sep, case_id = spec.partition(":")
        if not sep or not case_id or dataset_key not in ("wg", "zones"):
            raise SystemExit(
                f"--case-id must be dataset:case_id (wg or zones), got {spec!r}"
            )
        explicit.setdefault(dataset_key, []).append(case_id)
    return explicit


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--datasets", nargs="+", default=["wg", "zones"], choices=["wg", "zones"]
    )
    parser.add_argument(
        "--case-id",
        action="append",
        default=None,
        help=(
            "Explicit dataset:case_id, repeatable; overrides --select for that "
            "dataset only (other --datasets entries still use --select)."
        ),
    )
    parser.add_argument(
        "--select",
        nargs="+",
        default=DEFAULT_SELECT_MODES,
        choices=DEFAULT_SELECT_MODES,
    )
    parser.add_argument("--select-eps-n", type=int, default=DEFAULT_SELECT_EPS_N)
    parser.add_argument("--eps-n", type=int, nargs="+", default=DEFAULT_EPS_N_PANELS)
    parser.add_argument(
        "--summary-csv", type=Path, default=DEFAULT_OUTPUT_DIR / SUMMARY_CSV
    )
    parser.add_argument(
        "--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR / "case_overlays"
    )
    parser.add_argument(
        "--device", default=None, help="cuda / cpu (default: cuda if available)"
    )
    parser.add_argument("--seed", type=int, default=DEFAULT_BASE_SEED)
    parser.add_argument("--steps", type=int, default=DEFAULT_STEPS)
    parser.add_argument("--dpi", type=int, default=200)
    parser.add_argument(
        "--attack-valid-only",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="confine the perturbation to non-ignore voxels (excludes the zero-filled "
        "margin outside the dilated gland on zones, and the model-grid padding). "
        "The published run went without it.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    device = torch.device(
        args.device if args.device else ("cuda" if torch.cuda.is_available() else "cpu")
    )
    print(f"device: {device}", flush=True)
    args.output_dir.mkdir(parents=True, exist_ok=True)

    explicit = _parse_case_id_specs(args.case_id) if args.case_id else {}
    summary_frame = pd.read_csv(args.summary_csv)
    dataset_keys = list(dict.fromkeys([*args.datasets, *explicit]))

    for dataset_key in dataset_keys:
        paths = _dataset_paths(dataset_key)
        if dataset_key in explicit:
            case_ids = list(dict.fromkeys(explicit[dataset_key]))
        else:
            case_ids = _select_case_ids(
                summary_frame,
                dataset_key,
                select=args.select,
                select_eps_n=args.select_eps_n,
                output_dir=args.output_dir,
            )
        target = head_target(dataset_key)

        print(f"[{dataset_key}] loading fold-all UNet", flush=True)
        model, div_factors, num_classes = build_model(str(paths["checkpoint"]), device)
        model.eval()
        model.requires_grad_(False)

        for case_id in case_ids:
            sweep = run_case_sweep(
                dataset_key,
                case_id,
                eps_n_values=[0, *args.eps_n],
                steps=args.steps,
                base_seed=args.seed,
                device=device,
                model=model,
                div_factors=div_factors,
                num_classes=num_classes,
                data_dir=str(paths["data"]),
                attack_valid_only=bool(args.attack_valid_only),
            )
            for eps_n in args.eps_n:
                rows = summary_frame[
                    (summary_frame["dataset"] == dataset_key)
                    & (summary_frame["case_id"] == case_id)
                    & (summary_frame["epsilon_n"] == eps_n)
                    & (summary_frame["attack"] == APGD_ATTACK_LABEL)
                    & (summary_frame["class_or_union"] == target)
                ]
                if rows.empty:
                    print(
                        f"[{dataset_key}] {case_id} eps={eps_n}: no summary row, skipping",
                        flush=True,
                    )
                    continue
                fig = overlay_panels(
                    sweep, eps_n=eps_n, target=target, summary_row=rows.iloc[0]
                )
                output_path = (
                    args.output_dir / f"{dataset_key}_{case_id}_eps{eps_n}.png"
                )
                fig.savefig(output_path, dpi=args.dpi, bbox_inches="tight")
                plt.close(fig)
                print(f"[{dataset_key}] wrote {output_path}", flush=True)

        del model
        if device.type == "cuda":
            torch.cuda.empty_cache()


if __name__ == "__main__":
    main()
