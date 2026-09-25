"""Input validation and loading helpers — formerly Utils/InputCheck.py."""

from __future__ import annotations

import os
from typing import Any, Iterable

import SimpleITK as sitk


def load_nii_gz_files(files_to_load: Iterable[str]) -> dict[str, Any]:
    """Load .nii.gz files into a dict keyed by basename (without extension).

    The path can be a directory containing .nii.gz files or a single file.
    Each entry maps the basename (split on the first dot) to a SimpleITK image.
    """

    nii_gz_objects: dict[str, Any] = {}

    for file_path in files_to_load:
        file_key = os.path.basename(file_path).split(".")[0]
        nii_gz_objects[file_key] = sitk.ReadImage(file_path)

    return nii_gz_objects
