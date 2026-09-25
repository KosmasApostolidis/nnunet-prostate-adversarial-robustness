"""Whole-gland model preprocessing primitives — moved from Utils/wg_model.py."""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any

import numpy as np
import SimpleITK as sitk


class ImagePreprocessor(ABC):
    """Abstract base for medical image preprocessors.

    Provides a ``convert_to_float`` helper and a required ``normalize``
    interface that concrete subclasses must implement.
    """
    @abstractmethod
    def normalize(self, image: Any) -> Any:
        """Normalize a SimpleITK image."""

    def convert_to_float(self, image: Any) -> Any:
        """Convert image to float32 for processing."""
        return sitk.Cast(image, sitk.sitkFloat32)


class ZScoreNormalizer(ImagePreprocessor):
    """Z-score normalizer using SimpleITK StatisticsImageFilter.

    Casts the input image to float32, computes per-volume mean and
    standard deviation, and returns a zero-mean unit-variance image.
    """
    def normalize(self, image: Any) -> Any:
        original_image = image
        try:
            image = self.convert_to_float(image)
        except ValueError:
            image = original_image

        stats = sitk.StatisticsImageFilter()
        stats.Execute(image)

        mean = stats.GetMean()
        std = stats.GetSigma()

        normalized_image = sitk.ShiftScale(image, -mean, 1 / std if std > 0 else 1)
        return normalized_image


class ThresholdMaskFlattener:
    """Binarize soft tissue-probability masks at a configurable threshold.

    Attributes:
        threshold: Values above this are set to 1, below to 0.
    """
    def __init__(self, threshold: float = 0.5) -> None:
        self.threshold = threshold

    def flatten_mask(self, sitk_image: Any) -> Any:
        array = sitk.GetArrayFromImage(sitk_image)

        binary_mask = (array > self.threshold).astype(np.uint8)

        binary_sitk_image = sitk.GetImageFromArray(binary_mask)
        binary_sitk_image.CopyInformation(sitk_image)

        return binary_sitk_image
