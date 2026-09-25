"""initial_processing must create the nnUNet_raw ImagesTs folder it writes into.

Regression test: the folder was created relative to the CWD while the image was
written under ``nnUNet_raw``, which only worked when a placeholder folder already
existed there (it does in the repo, not in a slim Docker image).
"""

from __future__ import annotations

import os
import sys
import types

import numpy as np
import pytest
import SimpleITK as sitk


@pytest.fixture
def image_proc(monkeypatch, tmp_path):
    # image_proc imports MedProIO at module level; stub it (unused here).
    monkeypatch.setitem(
        sys.modules, "MedProIO", types.SimpleNamespace(Coregistrator=object, CropAndPad=object, Resampler=object)
    )
    monkeypatch.chdir(tmp_path)
    loaded_before = set(sys.modules)
    from mri_prostate_seg.utils import image_proc as ip

    raw = tmp_path / "nnUnet_paths" / "nnUNet_raw"
    raw.mkdir(parents=True)
    monkeypatch.setattr(ip, "nnUNet_raw", str(raw))
    monkeypatch.setattr(ip, "ImageProcessing", lambda img: img)  # skip MedProIO preprocessing
    try:
        yield ip, raw
    finally:
        for name in set(sys.modules) - loaded_before:
            sys.modules.pop(name, None)


def test_initial_processing_creates_imagests_under_nnunet_raw(image_proc, tmp_path):
    ip, raw = image_proc
    img = sitk.GetImageFromArray(np.zeros((4, 8, 8), dtype=np.float32))

    out = ip.initial_processing({"case1": img})

    assert set(out) == {"case1"}
    written = raw / "Dataset016_WgSegmentationPNetAndPicai" / "ImagesTs" / "ProstateWG_case1_0000.nii.gz"
    assert written.is_file(), f"{written} not written"
    assert not (tmp_path / "Dataset016_WgSegmentationPNetAndPicai").exists(), "folder created relative to CWD"
