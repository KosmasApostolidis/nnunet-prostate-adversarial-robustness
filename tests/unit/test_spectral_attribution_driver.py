"""Driver-level tests on the toy model: schema, stages, resume keys."""

from __future__ import annotations

import numpy as np
import pandas as pd
import torch

from experiments import evaluate_spectral_attribution_all_cases as drv
from mri_prostate_seg.experiments.edge_attribution.aggregate import (
    REQUIRED_SUMMARY_COLUMNS,
    check_summary_schema,
)
from tests.unit.test_edge_attribution_torch_ops import NUM_CLASSES, SHAPE, _Linear

FULL = ("energy", "causal", "interaction")


def _settings(stages):
    return drv.Settings(stages, 4, 4, 8, 1, 20, 10, 0.10, 2000, True, 20260721)


def test_summary_fieldnames_satisfy_the_aggregate_schema_guard() -> None:
    frame = pd.DataFrame({c: [0.0] for c in drv.SUMMARY_FIELDNAMES})
    check_summary_schema(frame, "test")
    assert set(REQUIRED_SUMMARY_COLUMNS) <= set(drv.SUMMARY_FIELDNAMES)
    assert drv.ATLAS_TYPE == "radial_shells_24"


def _toy_case_and_condition():
    rng = np.random.default_rng(5)
    x_np = -np.ones(SHAPE, dtype=np.float32)
    x_np[1:3, 4:12, 4:12] = 1.0
    seg = (x_np > 0).astype(np.int16)
    delta = rng.normal(scale=0.3, size=SHAPE)
    delta[1:3, 4:12, 4:12] -= 0.8  # damages the foreground
    delta = np.clip(delta, -1.0, 1.0).astype(np.float32)  # epsilon_n = 255 -> 1.0
    model = _Linear()
    case = drv.spectral_case(
        model, torch.device("cpu"), [1, 1, 5], NUM_CLASSES, "wg", "toy",  # pads W 16->20
        x_np, seg, (3.0, 0.5, 0.5), in_causal=True, in_interaction=True,
    )
    delta_pad = np.zeros(case.padded_shape, dtype=np.float32)
    delta_pad[: SHAPE[0], : SHAPE[1], : SHAPE[2]] = delta
    adv_pad = case.x_pad + torch.from_numpy(delta_pad)[None, None]
    cond = drv.build_spectral_condition(
        model, case, "pgd", "PGD-BCE", 255, adv_pad, 1, NUM_CLASSES, None
    )
    return model, case, cond


def test_process_condition_writes_every_table_with_24_shells(tmp_path, monkeypatch) -> None:
    monkeypatch.setattr(drv, "INTERACTION_EPS", (255,))
    model, case, cond = _toy_case_and_condition()
    drv.process_condition(model, case, cond, NUM_CLASSES, _settings(FULL), tmp_path)
    summary = pd.read_csv(tmp_path / drv.SUMMARY_CSV)
    shells = pd.read_csv(tmp_path / drv.SHELL_CSV)
    curves = pd.read_csv(tmp_path / drv.CURVES_CSV)
    controls = pd.read_csv(tmp_path / drv.CONTROL_CSV)
    assert len(summary) == 1 and summary.atlas_type[0] == drv.ATLAS_TYPE
    assert len(shells) == 24 and set(shells.band) == {"LF", "MF", "HF"}
    assert summary.parseval_error[0] < 1e-6
    live = ~shells.shell_empty.astype(bool)
    assert np.isfinite(shells.necessity_dice).sum() == live.sum()
    assert set(curves.ranking_type) == set(drv.RANKINGS)
    assert curves.evaluated.all()
    assert set(controls.control_type) >= {"greedy", "pairwise", "count_matched_joint"}
    assert set(controls.observed_band.dropna()) <= {"LF", "MF", "HF"}
    assert summary.stage[0] == "interaction"
    assert np.isfinite(summary.control_permutation_p[0])
    for col in REQUIRED_SUMMARY_COLUMNS:
        assert col in summary.columns
    check_summary_schema(summary, "toy")


def test_causal_only_settings_skip_interaction_rows(tmp_path) -> None:
    model, case, cond = _toy_case_and_condition()
    drv.process_condition(
        model, case, cond, NUM_CLASSES, _settings(("energy", "causal")), tmp_path
    )
    assert not (tmp_path / drv.CONTROL_CSV).exists()
    summary = pd.read_csv(tmp_path / drv.SUMMARY_CSV)
    assert summary.stage[0] == "causal"


def test_completed_keys_round_trip(tmp_path, monkeypatch) -> None:
    monkeypatch.setattr(drv, "INTERACTION_EPS", (255,))
    model, case, cond = _toy_case_and_condition()
    drv.process_condition(model, case, cond, NUM_CLASSES, _settings(FULL), tmp_path)
    assert drv.completed_keys(tmp_path / drv.SUMMARY_CSV) == {
        ("wg", "toy", 255, "PGD-BCE")
    }


def test_dataset_paths_default_is_the_clean_trainer_and_trainer_dir_overrides_it() -> None:
    default = drv._dataset_paths("wg")
    assert default["checkpoint"].parts[-3:] == (
        "nnUNetTrainer__nnUNetPlans__3d_fullres", "fold_all", "checkpoint_final.pth"
    )
    defended = drv._dataset_paths("wg", "nnUNetTrainerSmoothFieldFinetune__nnUNetPlans__3d_fullres")
    assert defended["checkpoint"].parts[-3:] == (
        "nnUNetTrainerSmoothFieldFinetune__nnUNetPlans__3d_fullres", "fold_all", "checkpoint_final.pth"
    )
    assert defended["checkpoint"].parent.parent.parent == default["checkpoint"].parent.parent.parent
    assert defended["data"] == default["data"] and defended["plans"] == default["plans"]


def test_trainer_dir_cli_defaults_to_clean_model(monkeypatch) -> None:
    monkeypatch.setattr("sys.argv", ["prog"])
    assert drv.parse_args().trainer_dir == drv.TRAINERS["unet"]
    monkeypatch.setattr("sys.argv", ["prog", "--trainer-dir", "X__Y__3d_fullres"])
    assert drv.parse_args().trainer_dir == "X__Y__3d_fullres"


def test_attack_valid_only_flag_defaults_off_and_is_recorded(monkeypatch) -> None:
    monkeypatch.setattr("sys.argv", ["prog"])
    assert drv.parse_args().attack_valid_only is False
    monkeypatch.setattr("sys.argv", ["prog", "--attack-valid-only"])
    assert drv.parse_args().attack_valid_only is True
    settings = drv.Settings(("energy",), 4, 4, 8, 1, 20, 10, 0.10, 2000, True, 1)
    assert settings.attack_valid_only is False  # positional construction unchanged
