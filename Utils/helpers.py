"""Back-compat shim — content moved to ``mri_prostate_seg.utils.image_proc``.

Re-exports the public API explicitly so static analysers (pyflakes, ruff, mypy)
can resolve symbols without `from … import *`. Add new names here when you
introduce them in ``image_proc`` and need them on the legacy ``Utils.helpers``
path.
"""

from __future__ import annotations

from mri_prostate_seg.utils.image_proc import (
    DeleteRedundantfiles,
    ImageProcessing,
    ImageProcessorClass,
    MaskPostProcessor,
    ZoneProcessor,
    create_binary_masks,
    filter_ser,
    initial_processing,
    mask_dilation,
    nnUNet_raw,
    outputs_saving,
    process_mask,
    process_masks,
    remove_small_components,
    resample,
)

__all__ = [
    "DeleteRedundantfiles",
    "ImageProcessing",
    "ImageProcessorClass",
    "MaskPostProcessor",
    "ZoneProcessor",
    "create_binary_masks",
    "filter_ser",
    "initial_processing",
    "mask_dilation",
    "nnUNet_raw",
    "outputs_saving",
    "process_mask",
    "process_masks",
    "remove_small_components",
    "resample",
]
