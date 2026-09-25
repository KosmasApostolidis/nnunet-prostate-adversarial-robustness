"""Image processing helpers — merged from legacy Utils/helpers.py and Utils/ImageProcessor.py.

Functions and classes are reproduced verbatim so existing call sites
(``Utils.helpers.X``, ``Utils.ImageProcessor.X``) keep working through
the shim layer in ``Utils/`` while new code imports from
``mri_prostate_seg.utils.image_proc``.
"""

from __future__ import annotations

import json
import logging
import os
from typing import Any

import numpy as np
import SimpleITK as sitk
from MedProIO import Coregistrator, CropAndPad, Resampler

# nnUNet_raw is left as a relative path so existing call sites resolve it
# against the working directory exactly the way the legacy code did.
nnUNet_raw: str = os.path.join("nnUnet_paths", "nnUNet_raw")


# ---------------------------------------------------------------------------
# image-level helpers (formerly Utils/ImageProcessor.py)
# ---------------------------------------------------------------------------


def ImageProcessing(
    sitk_image: Any,
    spacing: tuple[float, float, float] = (0.5, 0.5, 3.0),
    target_size: tuple[int, int, int] = (256, 256, 24),
) -> Any:
    res = Resampler.ResamplerToSpacing()
    res.set_reference_image(sitk_image)
    res.set_spacing(spacing=spacing)
    res.execute_processing()
    tr = res.get_transformed_image()

    res = CropAndPad.VolumeCropperAndPadder()
    res.set_reference_image(tr)
    res.set_target_size(target_size=target_size)
    res.execute_processing()
    tr = res.get_transformed_image()
    return tr


def resample(fix: Any, mov: Any) -> Any:
    res = Coregistrator.SequenceResampler()
    res.set_reference_image(fix)
    res.set_moving_image(mov)
    res.execute_processing()
    tr = res.get_transformed_image()
    _ = res.get_issues()
    return tr


def mask_dilation(mask: Any) -> Any:
    radius = 5

    dilate_filter = sitk.BinaryDilateImageFilter()
    dilate_filter.SetKernelType(sitk.sitkBall)
    dilate_filter.SetKernelRadius(radius)

    dilated_mask = dilate_filter.Execute(mask)
    return dilated_mask


def filter_ser(image: Any, mask: Any) -> Any:
    if not (
        image.GetOrigin() == mask.GetOrigin()
        and image.GetSpacing() == mask.GetSpacing()
        and image.GetDirection() == mask.GetDirection()
    ):
        mask.SetOrigin(image.GetOrigin())
        mask.SetSpacing(image.GetSpacing())
        mask.SetDirection(image.GetDirection())
    return sitk.Mask(image, mask)


def remove_small_components(
    mask: Any,
    keep_largest_only: bool = True,
    size_threshold: int | None = None,
) -> Any:
    """Remove smaller non-connected components from a binary mask."""

    labeled_mask = sitk.ConnectedComponent(mask)

    label_stats = sitk.LabelShapeStatisticsImageFilter()
    label_stats.Execute(labeled_mask)

    labels = label_stats.GetLabels()
    if keep_largest_only:
        labels_to_keep = (
            [max(labels, key=label_stats.GetNumberOfPixels)] if labels else []
        )
    elif size_threshold is not None:
        labels_to_keep = [
            label
            for label in labels
            if label_stats.GetNumberOfPixels(label) >= size_threshold
        ]
    else:
        labels_to_keep = list(labels)

    output_image = sitk.Image(mask.GetSize(), sitk.sitkUInt8)
    output_image.CopyInformation(mask)

    for label in labels_to_keep:
        component = sitk.BinaryThreshold(
            labeled_mask,
            lowerThreshold=label,
            upperThreshold=label,
            insideValue=1,
            outsideValue=0,
        )
        output_image = sitk.Or(output_image, component)

    return output_image


def process_mask(mask: Any) -> Any:
    """Perform morphological closing on a SimpleITK mask."""
    closing_radius = (2, 2, 0)
    closed_mask = sitk.BinaryMorphologicalClosing(mask, closing_radius)
    return closed_mask


def create_binary_masks(image: Any) -> tuple[Any, Any]:
    """Create binary masks for a SimpleITK image with regions 0/1/2."""
    mask_tz = sitk.BinaryThreshold(
        image, lowerThreshold=1, upperThreshold=1, insideValue=1, outsideValue=0
    )
    mask_pz = sitk.BinaryThreshold(
        image, lowerThreshold=2, upperThreshold=2, insideValue=1, outsideValue=0
    )
    return mask_tz, mask_pz


# ---------------------------------------------------------------------------
# orchestration helpers (formerly Utils/helpers.py)
# ---------------------------------------------------------------------------


def outputs_saving(
    wg_dict_original: dict[str, dict[str, str]],
    zones_original: dict[str, dict[str, str]],
    wg_dict_resampled: dict[str, dict[str, str]],
    zones_resampled: dict[str, dict[str, str]],
) -> None:
    """Saves the original and resampled wg, tz and pz mask path dicts as JSON."""
    for k, v in wg_dict_original.items():
        v.update(zones_original[k])
    for k, v in wg_dict_resampled.items():
        v.update(zones_resampled[k])

    with open(
        os.path.join("Outputs", "ResampledToOriginalSegmentationPaths.json"), "w"
    ) as file:
        json.dump(wg_dict_resampled, file, indent=4)
    with open(os.path.join("Outputs", "nnOutputSegmentationPaths.json"), "w") as file:
        json.dump(wg_dict_original, file, indent=4)


def initial_processing(pats: dict[str, Any]) -> dict[str, Any]:
    """Performs image processing operations to prepare patients for nnU-Net WG model."""
    pats_for_wg: dict[str, Any] = {}
    for key, val in pats.items():
        processed = ImageProcessing(val)
        pats_for_wg.update({key: processed})

    os.makedirs(
        os.path.join(nnUNet_raw, "Dataset016_WgSegmentationPNetAndPicai", "ImagesTs"),
        exist_ok=True,
    )
    for k, v in pats_for_wg.items():
        sitk.WriteImage(
            v,
            os.path.join(
                nnUNet_raw,
                os.path.join(
                    os.path.join("Dataset016_WgSegmentationPNetAndPicai", "ImagesTs"),
                    f"ProstateWG_{k}_0000.nii.gz",
                ),
            ),
        )
    return pats_for_wg


class ImageProcessorClass:
    """Whole-gland inference results processor.

    Prepares nnU-Net WG output for zone input: reads probability maps,
    applies thresholding, resampling, dilation, and connected-component
    filtering. Exposes original and resampled path dictionaries via
    ``get_paths``.
    """

    def __init__(self, base_output_path: str, nnUNet_raw: str) -> None:
        self.base_output_path = base_output_path
        self.nnUNet_raw = nnUNet_raw
        self.wg_dict_original: dict[str, dict[str, str]] = {}
        self.wg_dict_resampled: dict[str, dict[str, str]] = {}
        self.setup_logging()

    def setup_logging(self) -> None:
        logging.basicConfig(
            level=logging.INFO,
            format="%(asctime)s - %(levelname)s - %(message)s",
        )

    def create_directories(self, key: str) -> None:
        try:
            os.makedirs(
                os.path.join(self.base_output_path, key, "Resampled"),
                exist_ok=True,
            )
            os.makedirs(
                os.path.join(self.base_output_path, key, "Original"),
                exist_ok=True,
            )
        except OSError as e:
            logging.error(f"Error creating directories for {key}: {e}")
            raise

    def process_images(
        self,
        pats_for_wg_inference: dict[str, Any],
        pats_for_wg: dict[str, Any],
        pats: dict[str, Any],
    ) -> None:
        for key, val in pats_for_wg_inference.items():
            try:
                wg_binary = sitk.ReadImage(val["binary"])
                self.create_directories(key)
                wg_binary = process_mask(wg_binary)
                wg_binary = remove_small_components(wg_binary)

                filtered_ser = filter_ser(pats_for_wg[key], mask_dilation(wg_binary))
                probs = np.load(val["probs"])["probabilities"]
                wg_probs = probs[1, :, :, :]
                wg_probs = sitk.GetImageFromArray(wg_probs)
                wg_probs.CopyInformation(wg_binary)

                wg_binary_resampled = sitk.Resample(
                    wg_binary,
                    pats[key],
                    sitk.Transform(),
                    sitk.sitkNearestNeighbor,
                )
                wg_probs_resampled = sitk.Resample(
                    wg_probs,
                    pats[key],
                    sitk.Transform(),
                    sitk.sitkNearestNeighbor,
                )

                output_paths = {
                    "Original": {
                        "wg_binary": os.path.join(
                            self.base_output_path, key, "Original", "wg_binary.nii.gz"
                        ),
                        "wg_probs": os.path.join(
                            self.base_output_path, key, "Original", "wg_probs.nii.gz"
                        ),
                    },
                    "Resampled": {
                        "wg_binary": os.path.join(
                            self.base_output_path, key, "Resampled", "wg_binary.nii.gz"
                        ),
                        "wg_probs": os.path.join(
                            self.base_output_path, key, "Resampled", "wg_probs.nii.gz"
                        ),
                    },
                }

                self.write_image(
                    wg_binary_resampled, output_paths["Resampled"]["wg_binary"]
                )
                self.write_image(
                    wg_probs_resampled, output_paths["Resampled"]["wg_probs"]
                )
                self.write_image(wg_binary, output_paths["Original"]["wg_binary"])
                self.write_image(wg_probs, output_paths["Original"]["wg_probs"])

                self.wg_dict_original[key] = output_paths["Original"]
                self.wg_dict_resampled[key] = output_paths["Resampled"]
                os.makedirs(
                    os.path.join(
                        nnUNet_raw,
                        "Dataset019_ProstateZonesSegmentationWgFilteredLessDilated",
                        "ImagesTs",
                    ),
                    exist_ok=True,
                )
                sitk.WriteImage(
                    filtered_ser,
                    os.path.join(
                        nnUNet_raw,
                        os.path.join(
                            os.path.join(
                                "Dataset019_ProstateZonesSegmentationWgFilteredLessDilated",
                                "ImagesTs",
                            ),
                            f"ProstateZonesFilteredLessDilated_ProstateZones_{key}_0000.nii.gz",
                        ),
                    ),
                )

            except Exception as e:
                logging.error(f"Error processing {key}: {e}")

    def write_image(self, image: Any, path: str) -> None:
        try:
            sitk.WriteImage(image, path)
        except Exception as e:
            logging.error(f"Error writing image to {path}: {e}")

    def get_paths(self) -> tuple[dict[str, dict[str, str]], dict[str, dict[str, str]]]:
        return self.wg_dict_original, self.wg_dict_resampled


class ZoneProcessor:
    """Zone inference results processor.

    Reads nnU-Net zone probability outputs, applies post-processing
    (thresholding, resampling to original space), and exposes original
    and resampled path dictionaries via ``get_paths``.
    """

    def __init__(self, base_output_path: str) -> None:
        self.base_output_path = base_output_path
        self.resampled: dict[str, dict[str, str]] = {}
        self.original: dict[str, dict[str, str]] = {}
        self.setup_logging()

    def setup_logging(self) -> None:
        logging.basicConfig(
            level=logging.INFO,
            format="%(asctime)s - %(levelname)s - %(message)s",
        )

    def create_directories(self, key: str) -> None:
        try:
            os.makedirs(
                os.path.join(self.base_output_path, key, "Resampled"),
                exist_ok=True,
            )
            os.makedirs(
                os.path.join(self.base_output_path, key, "Original"),
                exist_ok=True,
            )
        except OSError as e:
            logging.error(f"Error creating directories for {key}: {e}")
            raise

    def process_zones(
        self, pats_for_zones: dict[str, Any], pats: dict[str, Any]
    ) -> None:
        for key, val in pats_for_zones.items():
            try:
                zones = sitk.ReadImage(val["binary"])
                tz_binary, pz_binary = create_binary_masks(zones)
                tz_binary = process_mask(tz_binary)
                pz_binary = process_mask(pz_binary)
                tz_binary = remove_small_components(tz_binary)
                pz_binary = remove_small_components(pz_binary)

                probs = np.load(val["probs"])["probabilities"]
                tz, pz = probs[1, :, :, :], probs[2, :, :, :]
                tz = sitk.GetImageFromArray(tz)
                pz = sitk.GetImageFromArray(pz)
                tz.CopyInformation(tz_binary)
                pz.CopyInformation(pz_binary)

                self.create_directories(key)

                resampled_paths = {
                    "tz_binary": os.path.join(
                        "Outputs", key, "Resampled", "tz_binary.nii.gz"
                    ),
                    "tz_probs": os.path.join(
                        "Outputs", key, "Resampled", "tz_probs.nii.gz"
                    ),
                    "pz_binary": os.path.join(
                        "Outputs", key, "Resampled", "pz_binary.nii.gz"
                    ),
                    "pz_probs": os.path.join(
                        "Outputs", key, "Resampled", "pz_probs.nii.gz"
                    ),
                }

                original_paths = {
                    "tz_binary": os.path.join(
                        "Outputs", key, "Original", "tz_binary.nii.gz"
                    ),
                    "tz_probs": os.path.join(
                        "Outputs", key, "Original", "tz_probs.nii.gz"
                    ),
                    "pz_binary": os.path.join(
                        "Outputs", key, "Original", "pz_binary.nii.gz"
                    ),
                    "pz_probs": os.path.join(
                        "Outputs", key, "Original", "pz_probs.nii.gz"
                    ),
                }

                self.write_image(
                    sitk.Resample(
                        tz_binary,
                        pats[key],
                        sitk.Transform(),
                        sitk.sitkNearestNeighbor,
                    ),
                    resampled_paths["tz_binary"],
                )
                self.write_image(
                    sitk.Resample(
                        tz, pats[key], sitk.Transform(), sitk.sitkNearestNeighbor
                    ),
                    resampled_paths["tz_probs"],
                )
                self.write_image(
                    sitk.Resample(
                        pz_binary,
                        pats[key],
                        sitk.Transform(),
                        sitk.sitkNearestNeighbor,
                    ),
                    resampled_paths["pz_binary"],
                )
                self.write_image(
                    sitk.Resample(
                        pz, pats[key], sitk.Transform(), sitk.sitkNearestNeighbor
                    ),
                    resampled_paths["pz_probs"],
                )

                self.write_image(tz_binary, original_paths["tz_binary"])
                self.write_image(tz, original_paths["tz_probs"])
                self.write_image(pz_binary, original_paths["pz_binary"])
                self.write_image(pz, original_paths["pz_probs"])

                self.resampled[key] = resampled_paths
                self.original[key] = original_paths

            except Exception as e:
                logging.error(f"Error processing {key}: {e}")

    def write_image(self, image: Any, path: str) -> None:
        try:
            sitk.WriteImage(image, path)
        except Exception as e:
            logging.error(f"Error writing image to {path}: {e}")

    def get_paths(self) -> tuple[dict[str, dict[str, str]], dict[str, dict[str, str]]]:
        return self.original, self.resampled


class DeleteRedundantfiles:
    """Post-pipeline workspace cleaner.

    Removes intermediate nnU-Net inference files (probability maps,
    resampled intermediates, temporary patient directories) after
    the final segmentation output has been saved.
    """

    def __init__(self) -> None:
        pass

    @staticmethod
    def clean_workspace_wg(paths_dict: dict[str, dict[str, str]]) -> None:
        for key, val in paths_dict.items():
            for path in val.values():
                try:
                    os.remove(path)
                    print(f"File {path} deleted successfully.")
                except FileNotFoundError:
                    print(f"File {path} not found.")
                except Exception as e:
                    print(f"Error deleting file {path}: {e}")
            nnpaths = os.path.join("nnUnet_paths", "nnUNet_raw")
            file = os.path.join(
                nnpaths, os.path.join("OutcomesWG", f"ProstateWG_{key}.pkl")
            )
            try:
                os.remove(file)
            except FileNotFoundError:
                print(f"File {file} not found.")
            except Exception as e:
                print(f"Error deleting file {file}: {e}")

    @staticmethod
    def clean_workspace_zones(paths_dict: dict[str, dict[str, str]]) -> None:
        for key, val in paths_dict.items():
            for path in val.values():
                try:
                    os.remove(path)
                    print(f"File {path} deleted successfully.")
                except FileNotFoundError:
                    print(f"File {path} not found.")
                except Exception as e:
                    print(f"Error deleting file {path}: {e}")
            nnpaths = os.path.join("nnUnet_paths", "nnUNet_raw")
            file = os.path.join(
                nnpaths,
                os.path.join(
                    "OutcomesZones",
                    f"ProstateZonesFilteredLessDilated_ProstateZones_{key}.pkl",
                ),
            )
            try:
                os.remove(file)
            except FileNotFoundError:
                print(f"File {file} not found.")
            except Exception as e:
                print(f"Error deleting file {file}: {e}")

    @staticmethod
    def clean_patients_directory(wg_paths: str, zones_paths: str) -> None:
        try:
            for filename in os.listdir(wg_paths):
                file_path = os.path.join(wg_paths, filename)
                if os.path.isfile(file_path):
                    os.remove(file_path)
                    print(f"File {file_path} deleted successfully.")
        except Exception as e:
            print(f"Error cleaning directory {wg_paths}: {e}")

        try:
            for filename in os.listdir(zones_paths):
                file_path = os.path.join(zones_paths, filename)
                if os.path.isfile(file_path):
                    os.remove(file_path)
                    print(f"File {file_path} deleted successfully.")
        except Exception as e:
            print(f"Error cleaning directory {zones_paths}: {e}")


class MaskPostProcessor:
    """Final mask combiner.

    Reads WG, PZ, and TZ binary masks from the output-path JSON map and
    ORs them together into the whole-gland binary output, ensuring the
    WG mask includes any zone-level detections.
    """

    def __init__(self, out_dict_path: str) -> None:
        """Initialize the post-processor with a JSON map of output paths."""
        with open(out_dict_path, "r") as f:
            self.images: dict[str, dict[str, str]] = json.load(f)
        self.image: Any = None

    def batch_post_process(self) -> None:
        for k, v in self.images.items():
            image_wg = sitk.ReadImage(v["wg_binary"])
            image_pz = sitk.ReadImage(v["pz_binary"])
            image_tz = sitk.ReadImage(v["tz_binary"])
            combined_mask = sitk.Or(image_wg, image_pz)
            combined_mask = sitk.Or(combined_mask, image_tz)
            sitk.WriteImage(combined_mask, v["wg_binary"])


def process_masks(out_volume: str) -> None:
    mps = MaskPostProcessor(os.path.join(out_volume, "nnOutputSegmentationPaths.json"))
    mps.batch_post_process()
    mps = MaskPostProcessor(
        os.path.join(out_volume, "ResampledToOriginalSegmentationPaths.json")
    )
    mps.batch_post_process()
