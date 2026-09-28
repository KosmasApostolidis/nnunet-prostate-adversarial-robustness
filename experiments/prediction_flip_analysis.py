"""Per-slice prediction flip (adversarial vs clean prediction) for FGSM and noise.

Answers the label-uncertainty objection to the slice-level analysis: if the
base/apex excess in Dice drop came from noisy reference masks at the gland
ends, it would vanish when the adversarial prediction is compared with the
model's own clean prediction instead of the reference. Reuses the runner's
loading, padding and attack code so the perturbations match the main runs
(same seed, same epsilon grid).

Writes ``<output-dir>/<attack>/<model>_fold<k>_prediction_flip.csv``.
"""

from __future__ import annotations

import argparse
import csv
import os
import sys

import numpy as np
import torch

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(REPO_ROOT, "src"))

from mri_prostate_seg.attacks.config_loader import load_model_config  # noqa: E402
from mri_prostate_seg.experiments.slice_vulnerability.anatomy import (  # noqa: E402
    derive_anatomy_mapping,
)
from mri_prostate_seg.experiments.slice_vulnerability.metrics import (  # noqa: E402
    masked_class_masks,
    slice_dice,
)
from mri_prostate_seg.experiments.slice_vulnerability.prediction_flip import (  # noqa: E402
    prediction_flip_by_slice,
)
from mri_prostate_seg.experiments.slice_vulnerability.runner import (  # noqa: E402
    _attack_predictions,
    _fgsm_predictions,
)
from mri_prostate_seg.experiments.slice_vulnerability.runtime import (  # noqa: E402
    load_network,
    load_preprocessed_case,
    load_validation_ids,
    pad_to_divisible,
)

HEADER = [
    "fold",
    "case_id",
    "slice_idx",
    "class",
    "epsilon",
    "gt_area",
    "anatomical_region",
    "prostate_relative_position",
    "dice_clean",
    "flip_dice",
]


def case_rows(
    network, config, plans, divisibility, fold, case_id, attack, generator, device
):
    image, segmentation, properties = load_preprocessed_case(config.data_dir, case_id)
    ground_truth = np.asarray(segmentation, dtype=np.int64)[0]
    anatomy = derive_anatomy_mapping(
        ground_truth,
        properties,
        transpose_forward=list(plans.get("transpose_forward", [0, 1, 2])),
    )
    image_padded, original_shape = pad_to_divisible(
        torch.from_numpy(image[np.newaxis]).to(device), divisibility
    )
    target_padded, _ = pad_to_divisible(
        torch.from_numpy(np.asarray(segmentation, dtype=np.int64)[np.newaxis]).to(
            device
        ),
        divisibility,
    )
    epsilons = [float(e) for e in config.epsilons if float(e) > 0]
    if attack == "fgsm":
        clean, adversarial = _fgsm_predictions(
            network,
            image_padded,
            target_padded,
            original_shape,
            epsilons,
            config.num_classes,
        )
    else:
        clean, adversarial = _attack_predictions(
            network,
            image_padded,
            target_padded,
            original_shape,
            epsilons,
            config.num_classes,
            attack=attack,
            n_steps=1,
            generator=generator,
        )
    rows = []
    for class_id, class_name in zip(range(1, config.num_classes), config.class_names):
        for epsilon in epsilons:
            for flip in prediction_flip_by_slice(
                clean, adversarial[epsilon], ground_truth, class_id
            ):
                s = flip["slice_idx"]
                gt_bin, clean_bin = masked_class_masks(
                    ground_truth[s], clean[s], class_id
                )
                fields = anatomy.slice_fields(s)
                rows.append(
                    {
                        "fold": fold,
                        "case_id": case_id,
                        "slice_idx": s,
                        "class": class_name,
                        "epsilon": epsilon,
                        "gt_area": flip["gt_area"],
                        "anatomical_region": fields["anatomical_region"],
                        "prostate_relative_position": fields[
                            "prostate_relative_position"
                        ],
                        "dice_clean": slice_dice(clean_bin, gt_bin),
                        "flip_dice": flip["flip_dice"],
                    }
                )
    return rows


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--model", choices=["wg", "zones"], required=True)
    parser.add_argument("--fold", type=int, required=True)
    parser.add_argument("--attack", choices=["fgsm", "noise"], required=True)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--max-samples", type=int, default=None)
    parser.add_argument("--output-dir", default="results/prediction_flip")
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    config = load_model_config(args.model)
    network, divisibility, plans = load_network(config, args.fold, device)
    case_ids = load_validation_ids(config.splits_json, args.fold)[: args.max_samples]
    generator = (
        torch.Generator().manual_seed(args.seed) if args.attack == "noise" else None
    )

    out_dir = os.path.join(args.output_dir, args.attack)
    os.makedirs(out_dir, exist_ok=True)
    out_path = os.path.join(
        out_dir, f"{args.model}_fold{args.fold}_prediction_flip.csv"
    )
    with open(out_path, "w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=HEADER)
        writer.writeheader()
        for index, case_id in enumerate(case_ids):
            writer.writerows(
                case_rows(
                    network,
                    config,
                    plans,
                    divisibility,
                    args.fold,
                    case_id,
                    args.attack,
                    generator,
                    device,
                )
            )
            if (index + 1) % 20 == 0 or index + 1 == len(case_ids):
                print(f"  [{index + 1}/{len(case_ids)}] processed", flush=True)
    print(f"wrote {out_path}")


if __name__ == "__main__":
    main()
