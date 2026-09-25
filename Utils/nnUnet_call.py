"""Back-compat shim — content moved to ``mri_prostate_seg.models.nnunet_call``."""

from __future__ import annotations

from mri_prostate_seg.models.nnunet_call import (  # noqa: F401
    BaseNNUnetModule,
    WGNNUnet,
    ZonesNNUnet,
)
