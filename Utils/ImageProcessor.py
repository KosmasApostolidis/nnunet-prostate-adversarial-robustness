"""Back-compat shim — content moved to ``mri_prostate_seg.utils.image_proc``."""

from __future__ import annotations

from mri_prostate_seg.utils.image_proc import (  # noqa: F401
    ImageProcessing,
    create_binary_masks,
    filter_ser,
    mask_dilation,
    process_mask,
    remove_small_components,
    resample,
)
