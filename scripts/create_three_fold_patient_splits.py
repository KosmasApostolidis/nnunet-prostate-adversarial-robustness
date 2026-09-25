#!/usr/bin/env python3
"""Create deterministic three-fold patient-level nnU-Net split manifests."""

from __future__ import annotations

import hashlib
import json
import os
import shutil
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path


N_SPLITS = 3
SEED = 12345


@dataclass(frozen=True)
class DatasetSpec:
    source: Path
    output_directory: Path
    case_prefix: str

    @property
    def source_directory(self) -> Path:
        return self.source.parent


def load_case_universe(source: Path) -> list[str]:
    with source.open(encoding="utf-8") as stream:
        splits = json.load(stream)

    if not isinstance(splits, list) or not splits:
        raise ValueError(f"{source} must contain a non-empty list of folds")

    expected_universe: set[str] | None = None
    for fold_index, fold in enumerate(splits):
        if not isinstance(fold, dict):
            raise ValueError(f"{source}: fold {fold_index} is not an object")

        train = fold.get("train")
        val = fold.get("val")
        if not isinstance(train, list) or not isinstance(val, list):
            raise ValueError(
                f"{source}: fold {fold_index} must contain train and val lists"
            )
        if not all(isinstance(case, str) for case in train + val):
            raise ValueError(f"{source}: fold {fold_index} contains a non-string case")
        if len(train) != len(set(train)) or len(val) != len(set(val)):
            raise ValueError(f"{source}: fold {fold_index} contains duplicate cases")

        train_set = set(train)
        val_set = set(val)
        overlap = train_set & val_set
        if overlap:
            raise ValueError(
                f"{source}: fold {fold_index} has {len(overlap)} train/val overlaps"
            )

        universe = train_set | val_set
        if expected_universe is None:
            expected_universe = universe
        elif universe != expected_universe:
            raise ValueError(
                f"{source}: fold {fold_index} does not contain the same case universe"
            )

    assert expected_universe is not None
    return sorted(expected_universe)


def stable_patient_order(patient_ids: list[str]) -> list[str]:
    """Return a reproducible pseudo-random order without library RNG dependence."""

    def rank(patient_id: str) -> tuple[bytes, str]:
        digest = hashlib.sha256(f"{SEED}:{patient_id}".encode()).digest()
        return digest, patient_id

    return sorted(patient_ids, key=rank)


def create_splits(cases: list[str], case_prefix: str) -> list[dict[str, list[str]]]:
    cases_by_patient: dict[str, list[str]] = defaultdict(list)
    for case in cases:
        if not case.startswith(case_prefix):
            raise ValueError(f"Unexpected case identifier: {case}")
        patient_id = case.removeprefix(case_prefix)
        if not patient_id:
            raise ValueError(f"Empty patient identifier in case: {case}")
        cases_by_patient[patient_id].append(case)

    patients = stable_patient_order(sorted(cases_by_patient))
    validation_patients = [set(patients[index::N_SPLITS]) for index in range(N_SPLITS)]

    splits: list[dict[str, list[str]]] = []
    all_cases = set(cases)
    for val_patients in validation_patients:
        val = sorted(
            case
            for patient_id in val_patients
            for case in cases_by_patient[patient_id]
        )
        train = sorted(all_cases - set(val))
        splits.append({"train": train, "val": val})

    validation_cases = [case for fold in splits for case in fold["val"]]
    if len(validation_cases) != len(set(validation_cases)):
        raise AssertionError("A case appears in validation in more than one fold")
    if set(validation_cases) != all_cases:
        raise AssertionError("Not every case appears in validation exactly once")

    return splits


def source_files_for_case(spec: DatasetSpec, case: str) -> list[Path]:
    relative_files = [
        Path("nnUNetPlans_3d_fullres") / f"{case}.b2nd",
        Path("nnUNetPlans_3d_fullres") / f"{case}_seg.b2nd",
        Path("nnUNetPlans_3d_fullres") / f"{case}.pkl",
        Path("gt_segmentations") / f"{case}.nii.gz",
    ]
    source_files = [
        spec.source_directory / relative_file
        for relative_file in relative_files
    ]
    missing = [source_file for source_file in source_files if not source_file.is_file()]
    if missing:
        missing_list = "\n".join(str(path) for path in missing)
        raise FileNotFoundError(f"Missing 3D patient files:\n{missing_list}")
    return source_files


def copy_file(source: Path, destination: Path) -> bool:
    """Copy one file atomically; return whether a new copy was written."""

    if destination.is_symlink():
        raise FileExistsError(f"Refusing to replace symlink: {destination}")
    if destination.is_file() and destination.stat().st_size == source.stat().st_size:
        return False
    if destination.exists():
        raise FileExistsError(f"Refusing to replace existing path: {destination}")

    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_name(f".{destination.name}.copying")
    if temporary.exists():
        temporary.unlink()
    try:
        shutil.copy2(source, temporary)
        os.replace(temporary, destination)
    finally:
        if temporary.exists():
            temporary.unlink()
    return True


def write_subset(spec: DatasetSpec, subset_directory: Path, cases: list[str]) -> int:
    patient_files = {
        case: source_files_for_case(spec, case)
        for case in cases
    }

    subset_directory.mkdir(parents=True, exist_ok=True)
    (subset_directory / "patients.json").write_text(
        json.dumps(cases, indent=2) + "\n",
        encoding="utf-8",
    )

    for metadata_name in (
        "dataset.json",
        "dataset_fingerprint.json",
        "nnUNetPlans.json",
    ):
        source_metadata = spec.source_directory / metadata_name
        if not source_metadata.is_file():
            raise FileNotFoundError(f"Missing dataset metadata: {source_metadata}")
        copy_file(source_metadata, subset_directory / metadata_name)

    files_copied = 0
    for files in patient_files.values():
        for source_file in files:
            relative_file = source_file.relative_to(spec.source_directory)
            destination = subset_directory / relative_file
            files_copied += copy_file(source_file, destination)
    return files_copied


def write_splits(spec: DatasetSpec) -> None:
    cases = load_case_universe(spec.source)
    splits = create_splits(cases, spec.case_prefix)

    spec.output_directory.mkdir(parents=True, exist_ok=True)
    destination = spec.output_directory / "splits_final.json"
    destination.write_text(json.dumps(splits, indent=2) + "\n", encoding="utf-8")

    files_copied = 0
    for fold_index, fold in enumerate(splits):
        fold_directory = spec.output_directory / f"fold{fold_index}"
        files_copied += write_subset(spec, fold_directory / "train", fold["train"])
        files_copied += write_subset(spec, fold_directory / "val", fold["val"])

    fold_sizes = ", ".join(
        f"fold {index}: {len(fold['train'])} train/{len(fold['val'])} val"
        for index, fold in enumerate(splits)
    )
    print(
        f"{destination}: {len(cases)} patients; {fold_sizes}; "
        f"{files_copied} files copied"
    )


def main() -> None:
    repository_root = Path(__file__).resolve().parents[1]
    preprocessed = repository_root / "nnUnet_paths" / "nnUNet_preprocessed"

    specs = [
        DatasetSpec(
            source=(
                preprocessed
                / "Dataset016_WgSegmentationPNetAndPicai"
                / "splits_final.json"
            ),
            output_directory=repository_root / "wg_prostate_net_picai",
            case_prefix="ProstateWG_",
        ),
        DatasetSpec(
            source=(
                preprocessed
                / "Dataset019_ProstateZonesSegmentationWgFilteredLessDilated"
                / "splits_final.json"
            ),
            output_directory=repository_root / "zonal_prostate_net_picai",
            case_prefix="ProstateZonesFilteredLessDilated_ProstateZones_",
        ),
    ]

    for spec in specs:
        write_splits(spec)


if __name__ == "__main__":
    main()
