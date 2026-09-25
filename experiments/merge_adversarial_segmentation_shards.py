"""Validate and atomically merge deterministic segmentation-evaluation shards."""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path

import pandas as pd


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from experiments.evaluate_adversarial_quality_all_cases import (  # noqa: E402
    ATTACK_LABELS,
    DEFAULT_EPS_N,
    _available_case_ids,
    _dataset_paths,
)
from experiments.evaluate_adversarial_segmentation_all_cases import (  # noqa: E402
    DATASETS,
    DEFAULT_OUTPUT_DIR,
    FIELDNAMES,
    _write_same_pass_quality,
    _write_summary,
)


DEFAULT_SHARD_ROOT = DEFAULT_OUTPUT_DIR.parent / "boundary_shards"
IDENTITY_COLUMNS = ["dataset", "case_id", "epsilon_n", "attack", "class"]
COMMON_MANIFEST_KEYS = (
    "attack_steps",
    "batch_size",
    "boundary_metrics",
    "boundary_workers",
    "deterministic",
    "bf16_autocast",
    "max_cases",
    "max_ssim_slices",
    "seed",
    "device",
)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _read_shard(path: Path) -> pd.DataFrame:
    return pd.read_csv(
        path,
        dtype={
            "dataset": "string",
            "case_id": "string",
            "attack": "string",
            "class": "string",
        },
    )


def _validate_shard(
    frame: pd.DataFrame,
    *,
    dataset_key: str,
    attack_key: str,
    epsilon_n_values: list[int],
) -> tuple[int, int]:
    missing_columns = set(FIELDNAMES).difference(frame.columns)
    if missing_columns:
        raise ValueError(
            f"{dataset_key}/{attack_key}: missing columns {sorted(missing_columns)}"
        )
    if frame[IDENTITY_COLUMNS].isna().any().any():
        raise ValueError(f"{dataset_key}/{attack_key}: null identity values")
    if frame.duplicated(IDENTITY_COLUMNS).any():
        raise ValueError(f"{dataset_key}/{attack_key}: duplicate identity rows")

    attack_label = ATTACK_LABELS[attack_key]
    if set(frame["dataset"].astype(str)) != {dataset_key}:
        raise ValueError(f"{dataset_key}/{attack_key}: unexpected dataset values")
    if set(frame["attack"].astype(str)) != {attack_label}:
        raise ValueError(f"{dataset_key}/{attack_key}: unexpected attack values")
    if set(frame["epsilon_n"].astype(int)) != set(epsilon_n_values):
        raise ValueError(f"{dataset_key}/{attack_key}: incomplete epsilon values")

    case_ids = _available_case_ids(_dataset_paths(dataset_key)["data"])
    class_names = [*DATASETS[dataset_key]["class_names"], "macro"]
    expected = {
        (case_id, epsilon_n, class_name)
        for case_id in case_ids
        for epsilon_n in epsilon_n_values
        for class_name in class_names
    }
    actual = set(
        frame[["case_id", "epsilon_n", "class"]]
        .assign(
            case_id=lambda values: values["case_id"].astype(str),
            epsilon_n=lambda values: values["epsilon_n"].astype(int),
            **{"class": lambda values: values["class"].astype(str)},
        )
        .itertuples(index=False, name=None)
    )
    if actual != expected:
        missing = len(expected.difference(actual))
        unexpected = len(actual.difference(expected))
        raise ValueError(
            f"{dataset_key}/{attack_key}: incomplete grid "
            f"({missing} missing, {unexpected} unexpected rows)"
        )

    boundary_columns = (
        "clean_hd95_mm",
        "adversarial_hd95_mm",
        "clean_asd_mm",
        "adversarial_asd_mm",
    )
    foreground = frame[frame["class"].astype(str) != "macro"]
    if not any(foreground[column].notna().any() for column in boundary_columns):
        raise ValueError(f"{dataset_key}/{attack_key}: boundary metrics are absent")
    return len(case_ids) * len(epsilon_n_values), len(frame)


def _read_and_validate_manifest(
    path: Path,
    *,
    dataset_key: str,
    attack_key: str,
    epsilon_n_values: list[int],
) -> dict[str, object]:
    manifest = json.loads(path.read_text(encoding="utf-8"))
    expected_values = {
        "datasets": [dataset_key],
        "attacks": [attack_key],
        "epsilon_n_values": epsilon_n_values,
        "boundary_metrics": True,
        "deterministic": True,
        "max_cases": None,
    }
    for key, expected in expected_values.items():
        if manifest.get(key) != expected:
            raise ValueError(
                f"{dataset_key}/{attack_key}: manifest {key!r} is "
                f"{manifest.get(key)!r}, expected {expected!r}"
            )
    return manifest


def merge_shards(
    *,
    shard_root: Path,
    output_dir: Path,
    datasets: list[str],
    attacks: list[str],
    epsilon_n_values: list[int],
) -> None:
    frames: list[pd.DataFrame] = []
    shard_records: list[dict[str, object]] = []
    reference_config: dict[str, object] | None = None
    checkpoints: dict[str, str] = {}
    expected_keys = 0

    for dataset_key in datasets:
        for attack_key in attacks:
            shard_dir = shard_root / f"{dataset_key}_{attack_key}"
            csv_path = shard_dir / "adversarial_segmentation_all_cases.csv"
            manifest_path = shard_dir / "run_manifest.json"
            if not csv_path.is_file() or not manifest_path.is_file():
                raise FileNotFoundError(f"incomplete shard directory: {shard_dir}")

            manifest = _read_and_validate_manifest(
                manifest_path,
                dataset_key=dataset_key,
                attack_key=attack_key,
                epsilon_n_values=epsilon_n_values,
            )
            config = {key: manifest.get(key) for key in COMMON_MANIFEST_KEYS}
            if reference_config is None:
                reference_config = config
            elif config != reference_config:
                raise ValueError(
                    f"{dataset_key}/{attack_key}: run configuration differs from "
                    "the other shards"
                )

            frame = _read_shard(csv_path)
            shard_keys, shard_rows = _validate_shard(
                frame,
                dataset_key=dataset_key,
                attack_key=attack_key,
                epsilon_n_values=epsilon_n_values,
            )
            frames.append(frame[FIELDNAMES])
            expected_keys += shard_keys
            checkpoints.update(
                {str(key): str(value) for key, value in manifest["checkpoints"].items()}
            )
            shard_records.append(
                {
                    "dataset": dataset_key,
                    "attack": attack_key,
                    "path": str(csv_path.resolve()),
                    "run_manifest": str(manifest_path.resolve()),
                    "resume_from": manifest.get("resume_from"),
                    "rows": shard_rows,
                    "case_attack_epsilon_keys": shard_keys,
                    "sha256": _sha256(csv_path),
                    "run_manifest_sha256": _sha256(manifest_path),
                }
            )
            print(
                f"Validated {dataset_key}/{attack_key}: "
                f"{shard_rows:,} rows, {shard_keys:,} keys",
                flush=True,
            )

    merged = pd.concat(frames, ignore_index=True)
    if merged.duplicated(IDENTITY_COLUMNS).any():
        raise ValueError("duplicate identity rows across shards")
    actual_keys = merged[
        ["dataset", "case_id", "epsilon_n", "attack"]
    ].drop_duplicates()
    if len(actual_keys) != expected_keys:
        raise ValueError(
            f"merged key count {len(actual_keys):,} differs from "
            f"expected {expected_keys:,}"
        )
    merged = merged.sort_values(IDENTITY_COLUMNS, kind="stable").reset_index(drop=True)

    output_dir.mkdir(parents=True, exist_ok=True)
    output_csv = output_dir / "adversarial_segmentation_all_cases.csv"
    summary_csv = output_dir / "adversarial_segmentation_all_cases_summary.csv"
    quality_csv = output_dir / "adversarial_quality_same_pass.csv"
    output_tmp = output_csv.with_suffix(".csv.tmp")
    summary_tmp = summary_csv.with_suffix(".csv.tmp")
    quality_tmp = quality_csv.with_suffix(".csv.tmp")
    manifest_path = output_dir / "run_manifest.json"
    manifest_tmp = manifest_path.with_suffix(".json.tmp")

    merged.to_csv(output_tmp, index=False)
    output_tmp.replace(output_csv)
    _write_summary(output_csv, summary_tmp)
    summary_tmp.replace(summary_csv)
    _write_same_pass_quality(output_csv, quality_tmp)
    quality_tmp.replace(quality_csv)

    assert reference_config is not None
    canonical_manifest = {
        "datasets": datasets,
        "epsilon_n_values": epsilon_n_values,
        "attacks": attacks,
        **reference_config,
        "checkpoints": checkpoints,
        "merge": {
            "source_shard_root": str(shard_root.resolve()),
            "case_attack_epsilon_keys": int(len(actual_keys)),
            "rows": int(len(merged)),
            "output_sha256": _sha256(output_csv),
            "shards": shard_records,
        },
    }
    manifest_tmp.write_text(
        json.dumps(canonical_manifest, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    manifest_tmp.replace(manifest_path)
    print(
        f"Merged {len(merged):,} rows and {len(actual_keys):,} keys into {output_csv}",
        flush=True,
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Validate and merge deterministic boundary-evaluation shards"
    )
    parser.add_argument("--shard-root", type=Path, default=DEFAULT_SHARD_ROOT)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument(
        "--datasets", nargs="+", choices=list(DATASETS), default=list(DATASETS)
    )
    parser.add_argument(
        "--attacks", nargs="+", choices=list(ATTACK_LABELS), default=list(ATTACK_LABELS)
    )
    parser.add_argument("--eps-n", nargs="+", type=int, default=DEFAULT_EPS_N)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    epsilon_n_values = sorted(set(int(value) for value in args.eps_n))
    if not epsilon_n_values or any(value <= 0 for value in epsilon_n_values):
        raise ValueError("--eps-n values must be positive")
    merge_shards(
        shard_root=args.shard_root.resolve(),
        output_dir=args.output_dir.resolve(),
        datasets=list(dict.fromkeys(args.datasets)),
        attacks=list(dict.fromkeys(args.attacks)),
        epsilon_n_values=epsilon_n_values,
    )


if __name__ == "__main__":
    main()
