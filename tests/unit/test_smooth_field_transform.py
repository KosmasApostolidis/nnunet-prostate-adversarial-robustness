"""Smooth intensity-field augmentation (SSEUA defense candidate 1).

Needs the nnU-Net fork on the path (PYTHONPATH=src:<fork>) and the prostagent
environment; skipped elsewhere.
"""
from __future__ import annotations

import json
from pathlib import Path

import pytest
import torch

pytest.importorskip("batchgeneratorsv2")
pytest.importorskip("nnunetv2")
sft = pytest.importorskip(
    "nnunetv2.training.nnUNetTrainer.variants.data_augmentation.smooth_field_transform"
)
from batchgeneratorsv2.transforms.utils.random import RandomTransform  # noqa: E402

SPACING = (3.0, 0.5, 0.5)
SHAPE = (24, 64, 64)
F_MAX = 0.34


def _radial_energy_split(field: torch.Tensor, f_max: float) -> tuple[float, float]:
    spec = torch.fft.rfftn(field).abs() ** 2
    freqs = [torch.fft.fftfreq(n, d=s) for n, s in zip(SHAPE[:-1], SPACING[:-1])]
    freqs.append(torch.fft.rfftfreq(SHAPE[-1], d=SPACING[-1]))
    radius = torch.sqrt(sum(g * g for g in torch.meshgrid(*freqs, indexing="ij")))
    inside = spec[radius <= f_max].sum().item()
    outside = spec[radius > f_max].sum().item()
    return inside, outside


def test_field_is_band_limited_dc_free_and_linf_normalised():
    g = torch.Generator().manual_seed(0)
    field = sft.band_limited_field(SHAPE, SPACING, F_MAX, generator=g)
    assert field.shape == SHAPE
    inside, outside = _radial_energy_split(field, F_MAX)
    assert inside > 0
    assert outside <= 1e-6 * inside
    assert abs(field.mean().item()) < 1e-5
    assert field.abs().amax().item() == pytest.approx(1.0)


def test_field_rejects_cutoff_below_lowest_frequency():
    with pytest.raises(ValueError):
        sft.band_limited_field(SHAPE, SPACING, 1e-6)


def test_transform_adds_scaled_field_to_every_channel_and_leaves_seg_alone():
    torch.manual_seed(1)
    t = sft.SmoothIntensityFieldTransform(SPACING, F_MAX, amplitude=(0.2, 0.2))
    image = torch.zeros((2, *SHAPE))
    seg = torch.randint(0, 2, (1, *SHAPE))
    out = t(image=image.clone(), segmentation=seg.clone())
    assert out["image"].shape == image.shape
    assert torch.equal(out["segmentation"], seg)
    delta = out["image"]
    assert torch.allclose(delta[0], delta[1])
    assert delta.abs().amax().item() == pytest.approx(0.2, rel=1e-5)
    inside, outside = _radial_energy_split(delta[0], F_MAX)
    assert outside <= 1e-6 * inside


def test_transform_validates_amplitude():
    with pytest.raises(ValueError):
        sft.SmoothIntensityFieldTransform(SPACING, F_MAX, amplitude=(0.3, 0.1))


@pytest.fixture(scope="module")
def d016_plans_and_dataset_json():
    root = Path(__file__).resolve().parents[2] / "nnUnet_paths" / "nnUNet_preprocessed" / "Dataset016_WgSegmentationPNetAndPicai"
    if not (root / "nnUNetPlans.json").exists():
        pytest.skip("Dataset016 preprocessed plans not present")
    plans = json.loads((root / "nnUNetPlans.json").read_text())
    dataset_json = json.loads((root / "dataset.json").read_text())
    return plans, dataset_json


def test_trainer_requires_checkpoint_and_inserts_field_after_spatial(d016_plans_and_dataset_json, tmp_path, monkeypatch):
    monkeypatch.setenv("nnUNet_results", str(tmp_path))
    trainer_mod = pytest.importorskip(
        "nnunetv2.training.nnUNetTrainer.variants.data_augmentation.nnUNetTrainerSmoothFieldFinetune"
    )
    plans, dataset_json = d016_plans_and_dataset_json
    # The fork's nnUNetTrainer pops this key; nnunetv2.run.run_training injects it.
    def fresh_plans():
        return {**plans, "continue_training": False}
    dataset_json = {k: v for k, v in dataset_json.items() if k != "smooth_field"}
    with pytest.raises(ValueError):
        trainer_mod.nnUNetTrainerSmoothFieldFinetune(fresh_plans(), "3d_fullres", "all", dataset_json, torch.device("cpu"))

    dataset_json["smooth_field"] = {
        "finetune_checkpoint": "/nonexistent.pth",
        "f_max_cyc_per_mm": 0.25,
        "amplitude": [0.1, 0.2],
        "p_apply": 0.7,
        "num_epochs": 3,
    }
    tr = trainer_mod.nnUNetTrainerSmoothFieldFinetune(fresh_plans(), "3d_fullres", "all", dataset_json, torch.device("cpu"))
    assert tr.num_epochs == 3 and tr.initial_lr == 1e-3
    rotation, dummy_2d, _, mirror = tr.configure_rotation_dummyDA_mirroring_and_inital_patch_size()
    compose = tr.get_training_transforms(
        tr.configuration_manager.patch_size, rotation, tr._get_deep_supervision_scales(), mirror, dummy_2d,
        use_mask_for_norm=tr.configuration_manager.use_mask_for_norm, is_cascaded=False,
        foreground_labels=tr.label_manager.foreground_labels, regions=None, ignore_label=None,
    )
    names = [type(t).__name__ for t in compose.transforms]
    idx = next(i for i, t in enumerate(compose.transforms)
               if isinstance(t, RandomTransform) and isinstance(t.transform, sft.SmoothIntensityFieldTransform))
    assert "SpatialTransform" in names[:idx]
    if dummy_2d:
        assert "Convert2DTo3DTransform" in names[:idx]
    assert not any(isinstance(t, RandomTransform) for t in compose.transforms[:idx])
    field = compose.transforms[idx]
    assert field.apply_probability == 0.7
    assert field.transform.f_max_cyc_per_mm == 0.25
    assert field.transform.amplitude == (0.1, 0.2)
    assert field.transform.spacing_mm == tuple(tr.configuration_manager.spacing)
