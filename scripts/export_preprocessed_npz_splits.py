#!/usr/bin/env python3
"""Export copied nnU-Net 3D folds as framework-neutral preprocessed NPZ cases."""

from __future__ import annotations

import hashlib
import json
import os
import pickle
import shutil
from pathlib import Path
from typing import Any

import blosc2
import numpy as np


MINIMUM_FREE_BYTES = 5 * 1024**3
PROPERTY_KEYS = (
    "spacing",
    "bbox_used_for_cropping",
    "shape_before_cropping",
    "shape_after_cropping_and_before_resampling",
    "sitk_stuff",
)


def array_digest(array: np.ndarray) -> str:
    contiguous = np.ascontiguousarray(array)
    return hashlib.sha256(memoryview(contiguous).cast("B")).hexdigest()


def jsonable(value: Any) -> Any:
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, dict):
        return {str(key): jsonable(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [jsonable(item) for item in value]
    return value


def write_json_atomic(destination: Path, payload: Any) -> None:
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_name(f".{destination.name}.writing")
    try:
        temporary.write_text(
            json.dumps(payload, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        os.replace(temporary, destination)
    finally:
        if temporary.exists():
            temporary.unlink()


def load_b2nd(path: Path) -> np.ndarray:
    if not path.is_file():
        raise FileNotFoundError(f"Missing preprocessed array: {path}")
    return np.asarray(blosc2.open(str(path), mode="r")[:])


def verify_npz(
    destination: Path,
    image_digest: str,
    label_digest: str,
    image_shape: tuple[int, ...],
    label_shape: tuple[int, ...],
    image_dtype: np.dtype[Any],
    label_dtype: np.dtype[Any],
) -> int:
    if destination.is_symlink() or not destination.is_file():
        raise FileNotFoundError(f"Missing physical NPZ export: {destination}")

    with np.load(destination, allow_pickle=False) as archive:
        if set(archive.files) != {"image", "label"}:
            raise ValueError(f"Unexpected NPZ members in {destination}: {archive.files}")
        stored_image = archive["image"]
        stored_label = archive["label"]

    if stored_image.shape != image_shape or stored_image.dtype != image_dtype:
        raise ValueError(f"Image shape or dtype mismatch in {destination}")
    if stored_label.shape != label_shape or stored_label.dtype != label_dtype:
        raise ValueError(f"Label shape or dtype mismatch in {destination}")
    if array_digest(stored_image) != image_digest:
        raise ValueError(f"Image values do not match the source: {destination}")
    if array_digest(stored_label) != label_digest:
        raise ValueError(f"Label values do not match the source: {destination}")
    return destination.stat().st_size


def write_or_verify_npz(
    destination: Path,
    image: np.ndarray,
    label: np.ndarray,
    image_digest: str,
    label_digest: str,
) -> tuple[bool, int]:
    if destination.exists() or destination.is_symlink():
        size = verify_npz(
            destination,
            image_digest,
            label_digest,
            image.shape,
            label.shape,
            image.dtype,
            label.dtype,
        )
        return False, size

    required_buffer = image.nbytes + label.nbytes
    free_bytes = shutil.disk_usage(destination.parent.parent).free
    if free_bytes - required_buffer < MINIMUM_FREE_BYTES:
        raise OSError(
            f"Insufficient free space to safely export {destination}; "
            f"{free_bytes / 1024**3:.2f} GiB remains"
        )

    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_name(f".{destination.name}.writing")
    try:
        with temporary.open("wb") as stream:
            np.savez_compressed(stream, image=image, label=label)
        os.replace(temporary, destination)
    finally:
        if temporary.exists():
            temporary.unlink()

    size = verify_npz(
        destination,
        image_digest,
        label_digest,
        image.shape,
        label.shape,
        image.dtype,
        label.dtype,
    )
    return True, size


def export_case(subset: Path, case: str) -> tuple[dict[str, Any], bool]:
    source_directory = subset / "nnUNetPlans_3d_fullres"
    image_source = source_directory / f"{case}.b2nd"
    label_source = source_directory / f"{case}_seg.b2nd"
    properties_source = source_directory / f"{case}.pkl"

    image = load_b2nd(image_source)
    label = load_b2nd(label_source)
    if image.shape != label.shape:
        raise ValueError(
            f"Image/label shape mismatch for {case}: {image.shape} != {label.shape}"
        )

    with properties_source.open("rb") as stream:
        properties = pickle.load(stream)  # noqa: S301 - trusted local nnU-Net data

    image_hash = array_digest(image)
    label_hash = array_digest(label)
    destination = subset / "preprocessed" / f"{case}.npz"
    created, compressed_bytes = write_or_verify_npz(
        destination,
        image,
        label,
        image_hash,
        label_hash,
    )

    metadata = {
        "case_id": case,
        "preprocessed": True,
        "array_layout": "C,D,H,W",
        "image_shape": list(image.shape),
        "image_dtype": str(image.dtype),
        "label_shape": list(label.shape),
        "label_dtype": str(label.dtype),
        "label_values": np.unique(label).tolist(),
        "image_sha256": image_hash,
        "label_sha256": label_hash,
        "compressed_bytes": compressed_bytes,
        "properties": {
            key: jsonable(properties.get(key))
            for key in PROPERTY_KEYS
            if key in properties
        },
    }
    metadata_destination = subset / "metadata" / f"{case}.json"
    write_json_atomic(metadata_destination, metadata)

    manifest_entry = {
        "case_id": case,
        "data": str(destination.relative_to(subset)),
        "metadata": str(metadata_destination.relative_to(subset)),
        "image_shape": metadata["image_shape"],
        "image_dtype": metadata["image_dtype"],
        "label_shape": metadata["label_shape"],
        "label_dtype": metadata["label_dtype"],
        "label_values": metadata["label_values"],
        "image_sha256": image_hash,
        "label_sha256": label_hash,
    }
    return manifest_entry, created


def export_subset(subset: Path, expected_cases: list[str]) -> tuple[int, int]:
    patients_path = subset / "patients.json"
    patients = json.loads(patients_path.read_text(encoding="utf-8"))
    if patients != expected_cases:
        raise ValueError(f"Patient list does not match splits_final.json: {subset}")

    entries: list[dict[str, Any]] = []
    created_count = 0
    for index, case in enumerate(patients, start=1):
        entry, created = export_case(subset, case)
        entries.append(entry)
        created_count += created
        if index % 100 == 0 or index == len(patients):
            print(
                f"{subset}: {index}/{len(patients)} cases exported and verified",
                flush=True,
            )

    dataset = json.loads((subset / "dataset.json").read_text(encoding="utf-8"))
    manifest = {
        "format_version": 1,
        "preprocessed": True,
        "array_layout": "C,D,H,W",
        "case_count": len(entries),
        "channel_names": dataset.get("channel_names"),
        "labels": dataset.get("labels"),
        "cases": entries,
    }
    write_json_atomic(subset / "manifest.json", manifest)
    return len(entries), created_count


def write_readme(dataset_root: Path) -> None:
    readme = """# Framework-neutral preprocessed 3D folds

Each `foldN/train` and `foldN/val` directory contains:

- `preprocessed/<case_id>.npz`: compressed NumPy archive with `image` and `label` arrays.
- `metadata/<case_id>.json`: shapes, dtypes, spatial properties, and SHA-256 checksums.
- `manifest.json`: all cases in the subset and the dataset label/channel definitions.
- `patients.json`: the ordered patient identifiers for the subset.

Arrays preserve the nnU-Net-preprocessed, normalized and resampled values exactly. Both
arrays use channel-first `C,D,H,W` layout. Labels may contain `-1` for ignored voxels
outside the cropped region.

Load a case without nnU-Net:

```python
import numpy as np

with np.load("preprocessed/<case_id>.npz", allow_pickle=False) as case:
    image = case["image"]
    label = case["label"]
```
"""
    (dataset_root / "README.md").write_text(readme, encoding="utf-8")


def export_dataset(dataset_root: Path) -> None:
    splits = json.loads(
        (dataset_root / "splits_final.json").read_text(encoding="utf-8")
    )
    if len(splits) != 3:
        raise ValueError(f"Expected exactly three folds in {dataset_root}")

    total_cases = 0
    total_created = 0
    for fold_index, fold in enumerate(splits):
        for subset_name in ("train", "val"):
            subset = dataset_root / f"fold{fold_index}" / subset_name
            cases, created = export_subset(subset, fold[subset_name])
            total_cases += cases
            total_created += created

    write_readme(dataset_root)
    print(
        f"{dataset_root}: verified {total_cases} fold-case exports; "
        f"created {total_created} NPZ files",
        flush=True,
    )


def main() -> None:
    repository_root = Path(__file__).resolve().parents[1]
    for directory_name in (
        "wg_prostate_net_picai",
        "zonal_prostate_net_picai",
    ):
        export_dataset(repository_root / directory_name)


if __name__ == "__main__":
    main()
