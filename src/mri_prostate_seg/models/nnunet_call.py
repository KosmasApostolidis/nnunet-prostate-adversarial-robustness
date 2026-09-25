"""nnU-Net predictor wrappers — moved from Utils/nnUnet_call.py.

The legacy file set ``nnUNet_*`` env vars at import time as a side effect.
That side effect is preserved here, but rooted at the repo paths resolved
in ``mri_prostate_seg.paths`` so editable installs from any CWD continue
to find the bundled ``nnUnet_paths/`` tree.
"""

from __future__ import annotations

import logging
import os
from abc import ABC, abstractmethod
from typing import Any

log = logging.getLogger(__name__)

# --- env setup must precede the nnunetv2 import ---
from mri_prostate_seg.paths import NNUNET_PATHS as _NNUNET_PATHS

os.environ.setdefault("nnUNet_raw", os.path.join(_NNUNET_PATHS, "nnUNet_raw"))
os.environ.setdefault(
    "nnUNet_preprocessed", os.path.join(_NNUNET_PATHS, "nnUNet_preprocessed")
)
os.environ.setdefault("nnUNet_results", os.path.join(_NNUNET_PATHS, "nnUNet_results"))

# Legacy code expected the relative literals; preserve those too so any
# downstream consumer that reads the env var sees a value byte-identical
# to the pre-restructure default when run from the repo root.
if os.environ.get("nnUNet_raw", "").startswith(_NNUNET_PATHS):
    pass  # no override

import torch  # noqa: E402
from batchgenerators.utilities.file_and_folder_operations import join, load_json  # noqa: E402
from nnunetv2.inference.predict_from_raw_data import nnUNetPredictor  # noqa: E402
from nnunetv2.paths import nnUNet_raw, nnUNet_results  # noqa: E402
from nnunetv2.training.nnUNetTrainer.nnUNetTrainer import nnUNetTrainer  # noqa: E402
from nnunetv2.utilities.label_handling.label_handling import determine_num_input_channels  # noqa: E402
from nnunetv2.utilities.plans_handling.plans_handler import PlansManager  # noqa: E402


# ---------------------------------------------------------------------------
# Model-version registry.  The LSP6 desktop app selects a version per model
# through WG_MODEL / ZONES_MODEL (values below, default "v1").
# ---------------------------------------------------------------------------
MODEL_VERSIONS: tuple[str, ...] = ("v1", "v2")

# kind -> version -> (trainer folder under nnUNet_results/<dataset>/, use_folds)
MODEL_REGISTRY: dict[str, dict[str, tuple[str, tuple[int | str, ...]]]] = {
    "WG": {
        "v1": ("nnUNetTrainer__nnUNetPlans__3d_fullres", (0,)),
        "v2": ("nnUNetTrainerPGDATFinetune__nnUNetResEncUNetLPlans__3d_fullres", ("all",)),
    },
    "ZONES": {
        "v1": ("nnUNetTrainer__nnUNetPlans__3d_fullres", (0,)),
        "v2": ("nnUNetTrainerPGDATFinetune__nnUNetResEncUNetLPlans__3d_fullres", ("all",)),
    },
}
ENV_VARS: dict[str, str] = {"WG": "WG_MODEL", "ZONES": "ZONES_MODEL"}


def resolve_model(kind: str, version: str | None = None) -> tuple[str, tuple[int | str, ...]]:
    """Return ``(trainer_folder, use_folds)`` for ``kind`` (``"WG"`` or ``"ZONES"``).

    The version is ``version`` when given, else the env var named in
    ``ENV_VARS``, else ``"v1"``.  Unknown versions raise ``ValueError`` naming
    the env var so the message is actionable from the container log.
    """
    env_var = ENV_VARS[kind]
    chosen = version if version is not None else os.environ.get(env_var, "v1")
    if chosen not in MODEL_REGISTRY[kind]:
        raise ValueError(f"{env_var}={chosen!r} not in {MODEL_VERSIONS}")
    return MODEL_REGISTRY[kind][chosen]


def _init_predictor(
    predictor: nnUNetPredictor,
    model_dir: str,
    use_folds: tuple[int | str, ...],
    checkpoint_name: str = "checkpoint_final.pth",
) -> None:
    """``nnUNetPredictor.initialize_from_trained_model_folder`` minus the trainer lookup.

    Upstream resolves ``checkpoint["trainer_name"]`` inside the installed
    ``nnunetv2`` package and raises for custom trainers — the v2 checkpoints
    were saved by ``nnUNetTrainerPGDATFinetune``, which is not shipped.  None
    of our trainers override ``build_network_architecture``, so the plain
    ``nnUNetTrainer`` builder yields the identical network.  Everything else
    (fold naming, attribute set) mirrors nnunetv2 2.6.4.
    """
    dataset_json = load_json(join(model_dir, "dataset.json"))
    plans_manager = PlansManager(load_json(join(model_dir, "plans.json")))

    parameters = []
    for i, fold in enumerate(use_folds):
        checkpoint = torch.load(
            join(model_dir, f"fold_{fold}", checkpoint_name),
            map_location=torch.device("cpu"),
            weights_only=False,
        )
        if i == 0:
            trainer_name = checkpoint["trainer_name"]
            configuration_name = checkpoint["init_args"]["configuration"]
            inference_allowed_mirroring_axes = checkpoint.get("inference_allowed_mirroring_axes")
        parameters.append(checkpoint["network_weights"])

    configuration_manager = plans_manager.get_configuration(configuration_name)
    label_manager = plans_manager.get_label_manager(dataset_json)
    network = nnUNetTrainer.build_network_architecture(
        configuration_manager.network_arch_class_name,
        configuration_manager.network_arch_init_kwargs,
        configuration_manager.network_arch_init_kwargs_req_import,
        determine_num_input_channels(plans_manager, configuration_manager, dataset_json),
        label_manager.num_segmentation_heads,
        enable_deep_supervision=False,
    )
    network.load_state_dict(parameters[0])

    predictor.plans_manager = plans_manager
    predictor.configuration_manager = configuration_manager
    predictor.list_of_parameters = parameters
    predictor.network = network
    predictor.dataset_json = dataset_json
    predictor.trainer_name = trainer_name
    predictor.allowed_mirroring_axes = inference_allowed_mirroring_axes
    predictor.label_manager = label_manager


class BaseNNUnetModule(ABC):
    """Abstract base for nnU-Net v2 inference wrappers.

    Subclasses initialise a model folder and expose ``prediction`` plus
    ``return_paths`` for downstream pipeline stages.
    """
    def __init__(self, input_path: str, output_path: str) -> None:
        self.input_path = input_path
        self.output_path = output_path

    @abstractmethod
    def prediction(self) -> None:
        ...

    @abstractmethod
    def return_paths(self, *args: Any, **kwargs: Any) -> dict[str, Any]:
        ...


class WGNNUnet(BaseNNUnetModule):
    """Whole-gland nnU-Net predictor (Dataset016).

    Runs inference with a trained whole-gland model and returns
    per-patient binary segmentation and probability paths.
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    predictor = nnUNetPredictor(
        tile_step_size=0.5,
        use_gaussian=True,
        use_mirroring=True,
        perform_everything_on_device=device.type == "cuda",
        device=device,
        verbose=False,
        verbose_preprocessing=False,
        allow_tqdm=False,
    )

    def __init__(self, input_path: str, output_path: str) -> None:
        self.input_path = input_path
        self.output_path = output_path
        trainer_folder, use_folds = resolve_model("WG")
        model_dir = join(nnUNet_results, os.path.join(self.input_path, trainer_folder))
        log.info("WG model: %s folds=%s", model_dir, use_folds)
        _init_predictor(self.predictor, model_dir, use_folds)

    def prediction(self) -> None:
        self.predictor.predict_from_files(
            join(nnUNet_raw, os.path.join(self.input_path, "ImagesTs")),
            join(nnUNet_raw, self.output_path),
            save_probabilities=True,
            overwrite=True,
            num_processes_preprocessing=2,
            num_processes_segmentation_export=2,
            folder_with_segs_from_prev_stage=None,
            num_parts=1,
            part_id=0,
        )

    def return_paths(self, pats_for_wg: dict[str, Any]) -> dict[str, dict[str, str]]:
        pats_for_wg_inference: dict[str, dict[str, str]] = {}
        for key in pats_for_wg.keys():
            pats_for_wg_inference[key] = {
                "binary": os.path.join(
                    join(nnUNet_raw, self.output_path), f"ProstateWG_{key}.nii.gz"
                ),
                "probs": os.path.join(
                    join(nnUNet_raw, self.output_path), f"ProstateWG_{key}.npz"
                ),
            }
        return pats_for_wg_inference


class ZonesNNUnet(BaseNNUnetModule):
    """Prostate zone nnU-Net predictor (Dataset019).

    Runs inference with a trained zone-segmentation model and returns
    per-patient PZ and TZ binary masks and probability paths.
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    predictor = nnUNetPredictor(
        tile_step_size=0.5,
        use_gaussian=True,
        use_mirroring=True,
        perform_everything_on_device=device.type == "cuda",
        device=device,
        verbose=False,
        verbose_preprocessing=False,
        allow_tqdm=False,
    )

    def __init__(self, input_path: str, output_path: str) -> None:
        self.input_path = input_path
        self.output_path = output_path
        trainer_folder, use_folds = resolve_model("ZONES")
        model_dir = join(nnUNet_results, os.path.join(self.input_path, trainer_folder))
        log.info("ZONES model: %s folds=%s", model_dir, use_folds)
        _init_predictor(self.predictor, model_dir, use_folds)

    def prediction(self) -> None:
        self.predictor.predict_from_files(
            join(nnUNet_raw, os.path.join(self.input_path, "ImagesTs")),
            join(nnUNet_raw, "OutcomesZones"),
            save_probabilities=True,
            overwrite=True,
            num_processes_preprocessing=2,
            num_processes_segmentation_export=2,
            folder_with_segs_from_prev_stage=None,
            num_parts=1,
            part_id=0,
        )

    def return_paths(
        self, pats_for_wg_inference: dict[str, Any]
    ) -> dict[str, dict[str, str]]:
        pats_for_zones: dict[str, dict[str, str]] = {}
        for key in pats_for_wg_inference.keys():
            pats_for_zones[key] = {
                "binary": os.path.join(
                    join(nnUNet_raw, "OutcomesZones"),
                    f"ProstateZonesFilteredLessDilated_ProstateZones_{key}.nii.gz",
                ),
                "probs": os.path.join(
                    join(nnUNet_raw, "OutcomesZones"),
                    f"ProstateZonesFilteredLessDilated_ProstateZones_{key}.npz",
                ),
            }
        return pats_for_zones
