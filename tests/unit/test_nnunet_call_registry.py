"""Model-version registry and predictor bootstrap in mri_prostate_seg.models.nnunet_call."""

from __future__ import annotations

import os

import pytest
import torch

from mri_prostate_seg.models import nnunet_call as nc

WG_DS = "Dataset016_WgSegmentationPNetAndPicai"
ZONES_DS = "Dataset019_ProstateZonesSegmentationWgFilteredLessDilated"


def test_module_exposes_wrappers():
    assert hasattr(nc, "WGNNUnet")
    assert hasattr(nc, "ZonesNNUnet")


# --- registry / env resolution -------------------------------------------------


def test_default_version_is_v1(monkeypatch):
    monkeypatch.delenv("WG_MODEL", raising=False)
    assert nc.resolve_model("WG") == ("nnUNetTrainer__nnUNetPlans__3d_fullres", (0,))


def test_env_selects_v2(monkeypatch):
    monkeypatch.setenv("ZONES_MODEL", "v2")
    assert nc.resolve_model("ZONES") == (
        "nnUNetTrainerPGDATFinetune__nnUNetResEncUNetLPlans__3d_fullres",
        ("all",),
    )


def test_explicit_version_beats_env(monkeypatch):
    monkeypatch.setenv("WG_MODEL", "v2")
    assert nc.resolve_model("WG", "v1") == ("nnUNetTrainer__nnUNetPlans__3d_fullres", (0,))


def test_unknown_version_raises_with_env_name_and_value(monkeypatch):
    monkeypatch.setenv("WG_MODEL", "v3")
    with pytest.raises(ValueError, match=r"^WG_MODEL='v3' not in \('v1', 'v2'\)$"):
        nc.resolve_model("WG")


def test_unknown_kind_raises():
    with pytest.raises(KeyError):
        nc.resolve_model("PROSTATE")


@pytest.mark.parametrize("kind,dataset", [("WG", WG_DS), ("ZONES", ZONES_DS)])
@pytest.mark.parametrize("version", ["v1", "v2"])
def test_registry_entries_exist_on_disk(kind, dataset, version):
    folder, folds = nc.MODEL_REGISTRY[kind][version]
    root = os.path.join(nc.nnUNet_results, dataset, folder)
    if not os.path.isdir(root):
        pytest.skip(f"{root} not present on this host")
    for name in ("plans.json", "dataset.json"):
        assert os.path.isfile(os.path.join(root, name)), name
    for fold in folds:
        ckpt = os.path.join(root, f"fold_{fold}", "checkpoint_final.pth")
        assert os.path.isfile(ckpt), ckpt


# --- predictor bootstrap --------------------------------------------------------

from nnunetv2.inference.predict_from_raw_data import nnUNetPredictor  # noqa: E402


def _zones_model_dir(version: str) -> str:
    folder, folds = nc.MODEL_REGISTRY["ZONES"][version]
    model_dir = os.path.join(nc.nnUNet_results, ZONES_DS, folder)
    ckpt = os.path.join(model_dir, f"fold_{folds[0]}", "checkpoint_final.pth")
    if not os.path.isfile(ckpt):
        pytest.skip(f"{ckpt} not present on this host")
    return model_dir


def _cpu_predictor() -> nnUNetPredictor:
    return nnUNetPredictor(
        device=torch.device("cpu"),
        perform_everything_on_device=False,
        verbose=False,
        allow_tqdm=False,
    )


def test_upstream_init_rejects_the_custom_trainer_of_v2():
    """Documents why _init_predictor exists: stock nnunetv2 cannot resolve the trainer."""
    model_dir = _zones_model_dir("v2")
    with pytest.raises(RuntimeError, match="Unable to locate trainer class nnUNetTrainerPGDATFinetune"):
        _cpu_predictor().initialize_from_trained_model_folder(model_dir, ("all",))


@pytest.mark.parametrize(
    "version,network_cls,trainer_name",
    [("v1", "PlainConvUNet", "nnUNetTrainer"), ("v2", "ResidualEncoderUNet", "nnUNetTrainerPGDATFinetune")],
)
def test_init_predictor_builds_network_without_trainer_lookup(version, network_cls, trainer_name):
    model_dir = _zones_model_dir(version)
    _, folds = nc.MODEL_REGISTRY["ZONES"][version]
    predictor = _cpu_predictor()

    nc._init_predictor(predictor, model_dir, folds)

    assert type(predictor.network).__name__ == network_cls
    assert predictor.trainer_name == trainer_name
    assert predictor.allowed_mirroring_axes is not None
    assert len(predictor.list_of_parameters) == len(folds)
    assert predictor.label_manager.num_segmentation_heads == 3  # background, TZ+CZ, PZ
    assert predictor.configuration_manager.network_arch_class_name.endswith(network_cls)


# --- wrapper wiring -------------------------------------------------------------


@pytest.mark.parametrize(
    "cls,kind,dataset,env_var",
    [(nc.WGNNUnet, "WG", WG_DS, "WG_MODEL"), (nc.ZonesNNUnet, "ZONES", ZONES_DS, "ZONES_MODEL")],
)
def test_wrapper_resolves_version_from_env(monkeypatch, cls, kind, dataset, env_var):
    calls: list[tuple[str, tuple]] = []
    monkeypatch.setattr(nc, "_init_predictor", lambda predictor, model_dir, use_folds: calls.append((model_dir, use_folds)))
    monkeypatch.setenv(env_var, "v2")

    cls(input_path=dataset, output_path="Outcomes")

    folder, folds = nc.MODEL_REGISTRY[kind]["v2"]
    assert calls == [(os.path.join(nc.nnUNet_results, dataset, folder), folds)]


def test_wrapper_defaults_to_v1(monkeypatch):
    calls: list[tuple[str, tuple]] = []
    monkeypatch.setattr(nc, "_init_predictor", lambda predictor, model_dir, use_folds: calls.append((model_dir, use_folds)))
    monkeypatch.delenv("WG_MODEL", raising=False)

    nc.WGNNUnet(input_path=WG_DS, output_path="OutcomesWG")

    assert calls == [(os.path.join(nc.nnUNet_results, WG_DS, "nnUNetTrainer__nnUNetPlans__3d_fullres"), (0,))]


def test_wrapper_rejects_unknown_version(monkeypatch):
    monkeypatch.setattr(nc, "_init_predictor", lambda *a: pytest.fail("must not load a model"))
    monkeypatch.setenv("ZONES_MODEL", "latest")
    with pytest.raises(ValueError, match=r"ZONES_MODEL='latest'"):
        nc.ZonesNNUnet(input_path=ZONES_DS, output_path="OutcomesZones")


# --- fail-fast at pipeline entry -----------------------------------------------


def test_pipeline_entry_rejects_bad_zones_version_before_any_inference(monkeypatch):
    """A bad ZONES_MODEL must fail before the WG GPU pass, not after it."""
    import sys
    import types

    # image_proc imports MedProIO at module level; it is absent from some envs and
    # irrelevant here, so stub the three names the import pulls in.
    monkeypatch.setitem(
        sys.modules, "MedProIO", types.SimpleNamespace(Coregistrator=object, CropAndPad=object, Resampler=object)
    )
    loaded_before = set(sys.modules)
    try:
        from mri_prostate_seg.pipeline import segmentor as seg

        monkeypatch.setattr(seg.Segmentor, "wg_model", lambda self, pats: pytest.fail("WG inference ran"))
        monkeypatch.setenv("ZONES_MODEL", "v9")
        with pytest.raises(ValueError, match=r"ZONES_MODEL='v9'"):
            seg.segmentor_pipeline_operation(output_volume="Outputs", pats={})
    finally:
        # Drop modules imported under the stub so other tests see the real env.
        for name in set(sys.modules) - loaded_before:
            sys.modules.pop(name, None)
