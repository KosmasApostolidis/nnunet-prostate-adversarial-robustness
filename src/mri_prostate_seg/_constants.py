"""Shared named constants used across experiments, evaluation, and visualisation.

Centralising these here keeps every script importing the same value: the
adversarial evaluation pipeline, the boundary-decomposition experiments,
and the figure-rendering helpers all agreed on these but used to define
them independently.
"""

from __future__ import annotations

import numpy as np

# nnU-Net preprocessed labels use -1 to mark "ignore" voxels (outside the crop
# or excluded by the dataset's ignore mask). Every evaluator filters on this
# constant before computing metrics; do not change without coordinating with
# the dataset preprocessor.
IGNORE_LABEL: int = -1

# RGBA overlay colours used by figure renderers for ground-truth vs predicted
# foreground. Alpha = 0.5 lets the underlying T2w slice show through.
GT_COLOR: np.ndarray = np.array([0.12, 0.46, 0.70, 0.50])  # blue
PRED_COLOR: np.ndarray = np.array([1.00, 0.50, 0.05, 0.50])  # orange

# Voxel-axis layout assumed throughout the codebase: (z, y, x) for 3D arrays.
# Centralised so axis reductions in metrics/visualisation are consistent.
VOXEL_AXES_3D: tuple[int, int, int] = (0, 1, 2)

__all__ = [
    "IGNORE_LABEL",
    "GT_COLOR",
    "PRED_COLOR",
    "VOXEL_AXES_3D",
]
