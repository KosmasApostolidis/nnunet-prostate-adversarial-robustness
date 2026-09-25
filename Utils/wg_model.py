"""Back-compat shim — content moved to ``mri_prostate_seg.models.wg_model``."""

from __future__ import annotations

from mri_prostate_seg.models.wg_model import (  # noqa: F401
    ImagePreprocessor,
    ThresholdMaskFlattener,
    ZScoreNormalizer,
)
