"""Load and resolve attack/model configuration from configs/ JSON files."""

from __future__ import annotations

import json
import os
from dataclasses import dataclass, field
from typing import Any


_CONFIG_DIR = os.path.join(
    os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))),
    "configs",
)


def _resolve_nnunet_paths() -> str:
    from mri_prostate_seg.paths import NNUNET_PATHS
    return NNUNET_PATHS


@dataclass(frozen=True)
class ModelConfig:
    key: str
    name: str
    dataset_folder: str
    trainer: str
    configuration: str
    num_classes: int
    class_names: list[str]
    epsilons: list[float]
    data_dir: str
    splits_json: str

    def checkpoint_path(self, fold: int) -> str:
        return os.path.join(
            _resolve_nnunet_paths(),
            "nnUNet_results",
            self.dataset_folder,
            self.trainer,
            f"fold_{fold}",
            "checkpoint_final.pth",
        )


@dataclass(frozen=True)
class AttackConfig:
    name: str
    type: str
    description: str
    n_steps: int
    step_size: float | None
    random_start: bool
    momentum: float
    normalized_grad: bool
    num_restarts: int
    adaptation_rate: float = 0.0
    checkpoints: list[float] = field(default_factory=lambda: [0.25, 0.5, 0.75])


@dataclass(frozen=True)
class ExperimentConfig:
    name: str
    description: str
    models: list[str]
    attacks: list[str]
    output_dir: str
    folds: list[int]
    max_samples: int | None
    seed: int
    shared_trajectory: bool
    batch_size: int


def load_model_config(model_key: str) -> ModelConfig:
    path = os.path.join(_CONFIG_DIR, "models", f"{model_key}.json")
    with open(path) as f:
        d = json.load(f)
    nnunet = _resolve_nnunet_paths()
    data_dir = os.path.join(
        nnunet, "nnUNet_preprocessed", d["dataset_folder"], f"nnUNetPlans_{d['configuration']}",
    )
    splits_json = os.path.join(
        nnunet, "nnUNet_preprocessed", d["dataset_folder"], "splits_final.json",
    )
    return ModelConfig(
        key=d["key"],
        name=d["name"],
        dataset_folder=d["dataset_folder"],
        trainer=d["trainer"],
        configuration=d["configuration"],
        num_classes=d["num_classes"],
        class_names=d["class_names"],
        epsilons=d["epsilons"],
        data_dir=data_dir,
        splits_json=splits_json,
    )


def load_attack_config(attack_key: str) -> AttackConfig:
    path = os.path.join(_CONFIG_DIR, "attacks", f"{attack_key}.json")
    with open(path) as f:
        d = json.load(f)
    return AttackConfig(
        name=d["name"],
        type=d["type"],
        description=d["description"],
        n_steps=d.get("n_steps", 1),
        step_size=d.get("step_size"),
        random_start=d.get("random_start", False),
        momentum=d.get("momentum", 0.0),
        normalized_grad=d.get("normalized_grad", False),
        num_restarts=d.get("num_restarts", 1),
        adaptation_rate=d.get("adaptation_rate", 0.0),
        checkpoints=d.get("checkpoints", [0.25, 0.5, 0.75]),
    )


def load_experiment_config(experiment_name: str) -> ExperimentConfig:
    path = os.path.join(_CONFIG_DIR, "experiments", f"{experiment_name}.json")
    with open(path) as f:
        d = json.load(f)
    return ExperimentConfig(
        name=d["name"],
        description=d["description"],
        models=d["models"],
        attacks=d["attacks"],
        output_dir=d.get("output_dir", "adversarial_eval_results"),
        folds=d.get("folds", [0]),
        max_samples=d.get("max_samples"),
        seed=d.get("seed", 42),
        shared_trajectory=d.get("shared_trajectory", True),
        batch_size=d.get("batch_size", 1),
    )


_LIST_CONFIGS = {
    "models": ["wg", "zones"],
    "attacks": ["fgsm", "pgd", "a_pgd"],
    "experiments": ["eval_wg_zones_pgd_apgd"],
}


def list_available() -> dict[str, list[str]]:
    return dict(_LIST_CONFIGS)
