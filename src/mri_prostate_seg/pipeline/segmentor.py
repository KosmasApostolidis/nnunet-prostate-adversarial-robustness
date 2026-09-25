"""End-to-end segmentor pipeline — moved from Utils/segmentor_pipeline.py."""

from __future__ import annotations

import os
import warnings
from typing import Any

from mri_prostate_seg.models import nnunet_call
from mri_prostate_seg.utils import image_proc as helpers

warnings.filterwarnings("ignore")

nnUNet_raw: str = os.path.join("nnUnet_paths", "nnUNet_raw")


def _require(value: Any, name: str) -> Any:
    """Raise ValueError if a pipeline-stage prerequisite has not been set.

    Replaces `assert x is not None`, which `python -O` removes — turning real
    contract violations into silent NoneType errors at the next attribute access.
    """
    if value is None:
        raise ValueError(
            f"Segmentor stage prerequisite '{name}' is None; an earlier pipeline "
            "stage was skipped or failed. Call stages in order: wg_model → "
            "preparation_zones → zones_model → post_process_zones → saving → clean_workspace."
        )
    return value


def segmentor_pipeline_operation(output_volume: str, pats: dict[str, Any]) -> None:
    # Fail fast on a bad WG_MODEL / ZONES_MODEL before any inference runs;
    # otherwise an invalid zones version only surfaces after the WG pass.
    nnunet_call.resolve_model("WG")
    nnunet_call.resolve_model("ZONES")
    segmentor = Segmentor()
    segmentor.wg_model(pats)
    segmentor.preparation_zones(input_patients=pats)
    segmentor.zones_model()
    segmentor.post_process_zones(output_patient_folder=output_volume, pats=pats)
    segmentor.saving()
    segmentor.clean_workspace()


class Segmentor:
    """End-to-end prostate segmentation pipeline.

    Orchestrates whole-gland prediction, zone preparation, zone
    prediction, post-processing, output saving, and workspace cleanup.
    State is held in instance attributes shared across pipeline stages.
    """
    def __init__(self) -> None:
        self.pats_for_wg_inference: dict[str, dict[str, str]] | None = None
        self.pats_for_wg: dict[str, Any] | None = None
        self.wg_dict_original: dict[str, dict[str, str]] | None = None
        self.wg_dict_resampled: dict[str, dict[str, str]] | None = None
        self.pats_for_zones: dict[str, dict[str, str]] | None = None
        self.zones_original: dict[str, dict[str, str]] | None = None
        self.zones_resampled: dict[str, dict[str, str]] | None = None

    @staticmethod
    def preparation_wg(input_patients: dict[str, Any]) -> dict[str, Any]:
        return helpers.initial_processing(input_patients)

    def wg_model(self, input_patients: dict[str, Any]) -> None:
        self.pats_for_wg = self.preparation_wg(input_patients=input_patients)
        wg_nn = nnunet_call.WGNNUnet(
            input_path="Dataset016_WgSegmentationPNetAndPicai",
            output_path="OutcomesWG",
        )
        wg_nn.prediction()
        self.pats_for_wg_inference = wg_nn.return_paths(pats_for_wg=self.pats_for_wg)

    def preparation_zones(self, input_patients: dict[str, Any]) -> None:
        _require(self.pats_for_wg_inference, "pats_for_wg_inference")
        _require(self.pats_for_wg, "pats_for_wg")
        file_handling = helpers.ImageProcessorClass(
            base_output_path="Outputs", nnUNet_raw=nnUNet_raw
        )
        file_handling.process_images(
            self.pats_for_wg_inference, self.pats_for_wg, pats=input_patients
        )
        self.wg_dict_original, self.wg_dict_resampled = file_handling.get_paths()

    def zones_model(self) -> None:
        _require(self.pats_for_wg_inference, "pats_for_wg_inference")
        zones_nn = nnunet_call.ZonesNNUnet(
            input_path="Dataset019_ProstateZonesSegmentationWgFilteredLessDilated",
            output_path="OutcomesZones",
        )
        zones_nn.prediction()
        self.pats_for_zones = zones_nn.return_paths(
            pats_for_wg_inference=self.pats_for_wg_inference
        )

    def post_process_zones(
        self, output_patient_folder: str, pats: dict[str, Any]
    ) -> None:
        _require(self.pats_for_zones, "pats_for_zones")
        zone_handling = helpers.ZoneProcessor(output_patient_folder)
        zone_handling.process_zones(self.pats_for_zones, pats)
        self.zones_original, self.zones_resampled = zone_handling.get_paths()

    def saving(self) -> None:
        _require(self.wg_dict_original, "wg_dict_original")
        _require(self.zones_original, "zones_original")
        _require(self.wg_dict_resampled, "wg_dict_resampled")
        _require(self.zones_resampled, "zones_resampled")
        helpers.outputs_saving(
            self.wg_dict_original,
            self.zones_original,
            self.wg_dict_resampled,
            self.zones_resampled,
        )

    def clean_workspace(self) -> None:
        _require(self.pats_for_wg_inference, "pats_for_wg_inference")
        _require(self.pats_for_zones, "pats_for_zones")
        redundant = helpers.DeleteRedundantfiles()
        redundant.clean_workspace_wg(self.pats_for_wg_inference)
        redundant.clean_workspace_zones(self.pats_for_zones)
        redundant.clean_patients_directory(
            os.path.join(
                nnUNet_raw,
                os.path.join("Dataset016_WgSegmentationPNetAndPicai", "ImagesTs"),
            ),
            os.path.join(
                nnUNet_raw,
                os.path.join(
                    "Dataset019_ProstateZonesSegmentationWgFilteredLessDilated",
                    "ImagesTs",
                ),
            ),
        )
