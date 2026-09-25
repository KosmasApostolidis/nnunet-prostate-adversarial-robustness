"""Back-compat shim — content moved to ``mri_prostate_seg.pipeline.segmentor``."""

from __future__ import annotations

from mri_prostate_seg.pipeline.segmentor import (  # noqa: F401
    Segmentor,
    segmentor_pipeline_operation,
)
