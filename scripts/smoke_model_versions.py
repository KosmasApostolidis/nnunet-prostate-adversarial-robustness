"""Run one WG or Zones model version on a single NIfTI and report label counts.

Exercises the same predictor bootstrap the pipeline uses (resolve_model +
_init_predictor) without the DICOM/MedProIO preprocessing, so it runs in any
env that has nnunetv2 2.6.4 and the checkpoints.  Exit 1 on an empty mask.
"""

from __future__ import annotations

import argparse
import os
import sys
import time

import numpy as np
import SimpleITK as sitk
import torch

from mri_prostate_seg.models import nnunet_call as nc
from nnunetv2.inference.predict_from_raw_data import nnUNetPredictor

DATASETS = {
    "WG": "Dataset016_WgSegmentationPNetAndPicai",
    "ZONES": "Dataset019_ProstateZonesSegmentationWgFilteredLessDilated",
}


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--kind", choices=sorted(DATASETS), required=True)
    ap.add_argument("--version", choices=nc.MODEL_VERSIONS, required=True)
    ap.add_argument("--image", required=True, help="one T2 NIfTI (…_0000.nii.gz)")
    ap.add_argument("--out-dir", required=True)
    args = ap.parse_args()

    trainer_folder, use_folds = nc.resolve_model(args.kind, args.version)
    model_dir = os.path.join(nc.nnUNet_results, DATASETS[args.kind], trainer_folder)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    predictor = nnUNetPredictor(
        tile_step_size=0.5,
        use_gaussian=True,
        use_mirroring=True,
        perform_everything_on_device=device.type == "cuda",
        device=device,
        verbose=False,
        verbose_preprocessing=False,
        allow_tqdm=False,
    )
    nc._init_predictor(predictor, model_dir, use_folds)

    os.makedirs(args.out_dir, exist_ok=True)
    out_prefix = os.path.join(args.out_dir, f"{args.kind}_{args.version}")
    t0 = time.time()
    predictor.predict_from_files(
        [[args.image]],
        [out_prefix],
        save_probabilities=False,
        overwrite=True,
        num_processes_preprocessing=1,
        num_processes_segmentation_export=1,
        folder_with_segs_from_prev_stage=None,
        num_parts=1,
        part_id=0,
    )
    mask = sitk.GetArrayFromImage(sitk.ReadImage(out_prefix + predictor.dataset_json["file_ending"]))
    labels, counts = np.unique(mask, return_counts=True)
    print(
        f"{args.kind} {args.version}: {trainer_folder} folds={use_folds} device={device.type} "
        f"{time.time() - t0:.1f}s labels={dict(zip(labels.tolist(), counts.tolist()))}"
    )
    if int((mask > 0).sum()) == 0:
        print("FAIL: empty mask", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
