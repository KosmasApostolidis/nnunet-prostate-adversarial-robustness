"""Convert the attack budget from normalized units to raw T2W intensity.

nnU-Net normalizes each volume by its own mean and standard deviation
(ZScoreNormalization with use_mask_for_norm false), so an epsilon expressed in
the post-normalization space corresponds to eps * sigma in native scanner
units, where sigma is that volume's own intensity standard deviation. This
reports the median and IQR of sigma over a random sample of volumes, which is
what Reviewer 1 question 5 asks for.
"""

from __future__ import annotations

import argparse
import random
from pathlib import Path

import numpy as np
import SimpleITK as sitk

REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_IMAGES = (
    REPO_ROOT
    / "nnUnet_paths"
    / "nnUNet_raw"
    / "Dataset016_WgSegmentationPNetAndPicai"
    / "imagesTr"
)
EPSILONS = (0.02, 0.1)


def volume_sigma(path: Path) -> float:
    """Intensity SD over the nonzero bounding box, as nnU-Net crops before norm."""
    array = sitk.GetArrayFromImage(sitk.ReadImage(str(path))).astype(np.float32)
    nonzero = np.argwhere(array > 0)
    if nonzero.size:
        low = nonzero.min(axis=0)
        high = nonzero.max(axis=0) + 1
        array = array[low[0] : high[0], low[1] : high[1], low[2] : high[2]]
    return float(array.std())


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--images", type=Path, default=DEFAULT_IMAGES)
    parser.add_argument("--sample-size", type=int, default=200)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    files = sorted(args.images.glob("*_0000.nii.gz"))
    if not files:
        raise SystemExit(f"No images found under {args.images}")
    random.seed(args.seed)
    sample = random.sample(files, min(args.sample_size, len(files)))

    sigmas = np.array([volume_sigma(path) for path in sample])
    print(f"volumes sampled: {sigmas.size} of {len(files)}")
    print(
        f"per-volume sigma: median {np.median(sigmas):.1f} "
        f"(IQR {np.percentile(sigmas, 25):.1f}-{np.percentile(sigmas, 75):.1f}) "
        "arbitrary scanner units"
    )
    for epsilon in EPSILONS:
        shifts = epsilon * sigmas
        print(
            f"  epsilon={epsilon}: raw shift median {np.median(shifts):.1f} "
            f"(IQR {np.percentile(shifts, 25):.1f}-{np.percentile(shifts, 75):.1f})"
        )


if __name__ == "__main__":
    main()
