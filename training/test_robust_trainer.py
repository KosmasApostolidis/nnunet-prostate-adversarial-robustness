#!/usr/bin/env python
"""
Minimal test: verify nnUNetTrainerRobustLoss train_step runs without errors.
Requires: nnUNet paths set, preprocessed Dataset016 present.
Run: python test_robust_trainer.py
"""
import os
import sys

# Set paths before importing nnunetv2
os.environ["nnUNet_raw"] = os.path.join(
    os.path.dirname(__file__), "nnUnet_paths", "nnUNet_raw"
)
os.environ["nnUNet_preprocessed"] = os.path.join(
    os.path.dirname(__file__), "nnUnet_paths", "nnUNet_preprocessed"
)
os.environ["nnUNet_results"] = os.path.join(
    os.path.dirname(__file__), "nnUnet_paths", "nnUNet_results"
)

from batchgenerators.utilities.file_and_folder_operations import join, load_json
from nnunetv2.paths import nnUNet_preprocessed, nnUNet_results
from nnunetv2.training.nnUNetTrainer.variants.loss.nnUNetTrainerRobustLoss import (
    nnUNetTrainerRobustLoss,
)
from nnunetv2.utilities.find_class_by_name import recursive_find_python_class
import nnunetv2
import torch


def test_trainer_discovery():
    tr = recursive_find_python_class(
        join(nnunetv2.__path__[0], "training", "nnUNetTrainer"),
        "nnUNetTrainerRobustLoss",
        "nnunetv2.training.nnUNetTrainer",
    )
    assert tr is not None, "nnUNetTrainerRobustLoss not found"
    print("OK: nnUNetTrainerRobustLoss is discoverable")


def test_robust_loss_utils():
    from nnunetv2.training.nnUNetTrainer.variants.loss.robust_loss_utils import (
        boundary_band_mask,
        kl_binary_per_voxel,
        samplewise_bounds,
    )

    x = torch.randn(2, 1, 8, 16, 16)
    x_min, x_max = samplewise_bounds(x)
    assert x_min.shape == (2, 1, 1, 1, 1)
    assert x_max.shape == (2, 1, 1, 1, 1)

    y = torch.randint(0, 2, (2, 1, 8, 16, 16)).float()
    band = boundary_band_mask(y, ksize=3)
    assert band.shape == y.shape

    p = torch.sigmoid(torch.randn(2, 1, 8, 16, 16))
    q = torch.sigmoid(torch.randn(2, 1, 8, 16, 16))
    kl = kl_binary_per_voxel(p, q)
    assert kl.shape == p.shape
    assert kl.isfinite().all()

    print("OK: robust_loss_utils functions work")


if __name__ == "__main__":
    test_trainer_discovery()
    test_robust_loss_utils()
    print("\nAll tests passed.")
