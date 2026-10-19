"""Where does the zonal input's non-zero region come from? (ResEnc paper, REV-4)

The zonal images are zero outside a dilated gland region. This compares, on a
random sample of training cases, that non-zero region with the reference zone
union (TZ+CZ or PZ) dilated by a ball of radius 3, 5 and 7 voxels (SimpleITK,
the same filter as ``utils.image_proc.mask_dilation``). Jaccard ~1 at one
radius means the region is the dilated reference mask.

    python experiments/check_zonal_confinement_source.py --n 12
"""

from __future__ import annotations

import argparse
import csv
import os
import random

import numpy as np
import SimpleITK as sitk

DATASET = (
    "nnUnet_paths/nnUNet_raw/Dataset019_ProstateZonesSegmentationWgFilteredLessDilated"
)
OUT = (
    "results/blade_attack/attack_comparison/stats_no_blade/revision/"
    "E34_zonal_confinement_source.csv"
)
RADII = (3, 5, 7)


def jaccard(a: np.ndarray, b: np.ndarray) -> float:
    return float((a & b).sum() / max((a | b).sum(), 1))


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--n", type=int, default=12)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--output", default=OUT)
    args = ap.parse_args()

    labels = sorted(os.listdir(os.path.join(DATASET, "labelsTr")))
    random.seed(args.seed)
    rows = []
    for name in random.sample(labels, args.n):
        lab = sitk.ReadImage(os.path.join(DATASET, "labelsTr", name))
        img = sitk.ReadImage(
            os.path.join(DATASET, "imagesTr", name.replace(".nii.gz", "_0000.nii.gz"))
        )
        nonzero = sitk.GetArrayFromImage(img) != 0
        zones = sitk.Cast(lab > 0, sitk.sitkUInt8)
        row = {
            "case": name,
            "zone_voxels_outside_nonzero": int(
                ((sitk.GetArrayFromImage(lab) > 0) & ~nonzero).sum()
            ),
        }
        for r in RADII:
            dilate = sitk.BinaryDilateImageFilter()
            dilate.SetKernelType(sitk.sitkBall)
            dilate.SetKernelRadius(r)
            dilated = sitk.GetArrayFromImage(dilate.Execute(zones)).astype(bool)
            row[f"jaccard_r{r}"] = round(jaccard(nonzero, dilated), 4)
        rows.append(row)
        print(row, flush=True)

    with open(args.output, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    print(f"wrote {args.output}")


if __name__ == "__main__":
    main()
