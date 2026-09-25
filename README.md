# Adversarial Robustness of nnU-Net for Prostate MRI Segmentation

Code accompanying:

> K. K. Apostolidis, D. I. Zaridis, V. C. Pezoulas, N. S. Tachos, E. Mylona, K. Marias, M. Tsiknakis, and D. I. Fotiadis,
> "Evaluating the Adversarial Robustness of nnU-Net: A Slice-Level Vulnerability Analysis in Prostate MRI Segmentation."

The repository contains the two-stage nnU-Net v2 segmentation pipeline (whole gland → zones), the adversarial attack
implementations, the evaluation and slice-level analysis code, the nnU-Net plans, and the 5-fold cross-validation
assignments used in the paper.

## Contents

| Path | What it is |
|------|------------|
| `src/mri_prostate_seg/` | Installable package: attacks, evaluation, metrics, slice-vulnerability analysis, segmentation pipeline |
| `src/mri_prostate_seg/attacks/` | Attack implementations (see table below) |
| `configs/attacks/*.json` | Attack hyper-parameters, one file per attack |
| `configs/models/{wg,zones}.json` | Model definitions (dataset, trainer, ε grid) |
| `experiments/` | Runnable evaluation drivers and analysis/plotting scripts |
| `training/` | Custom nnU-Net trainers (reference copies) |
| `nnUnet_paths/nnUNet_results/<Dataset>/nnUNetTrainer__nnUNetPlans__3d_fullres/{plans.json,dataset.json}` | nnU-Net plans and dataset descriptors of the two trained models |
| `nnUnet_paths/nnUNet_preprocessed/<Dataset>/splits_final.json` | 5-fold cross-validation assignments (anonymised case IDs) |
| `tests/` | Unit and integration tests |

Models:

| Model | nnU-Net dataset | Configuration | Classes | Cases | Folds (train/val) |
|-------|-----------------|---------------|---------|-------|-------------------|
| Whole gland (WG) | `Dataset016_WgSegmentationPNetAndPicai` | `3d_fullres` | WG | 1,396 | ≈1,117 / 279 |
| Zones | `Dataset019_ProstateZonesSegmentationWgFilteredLessDilated` | `3d_fullres` | TZ+CZ, PZ | 603 | ≈482 / 121 |

## Attacks

| `--attack` | Module | Description |
|------------|--------|-------------|
| `fgsm` | `fgsm.py` | Fast Gradient Sign Method (single step, L∞) |
| `pgd` | `pgd.py` | Projected Gradient Descent (L∞) |
| `a_pgd` | `a_pgd.py` | Adaptive-step PGD — the paper's **AS-PGD** (not Croce & Hein's Auto-PGD) |
| `apgd_updated` | `apgd_updated.py` | AS-PGD with cosine step backbone |
| `auto_pgd` | `auto_pgd.py` | Auto-PGD (Croce & Hein, ICML 2020), ported to 3D segmentation |
| `segpgd` | `segpgd.py` | SegPGD (Gu et al., ECCV 2022) |
| `cospgd` | `cospgd.py` | CosPGD (Agnihotri et al., ICML 2024) |
| `dag` | `dag.py` | Dense Adversary Generation (Xie et al., ICCV 2017) |
| `sea` | `sea.py` | Segmentation Ensemble Attack (Croce et al., 2023) |
| `blade`, `blade1`, `blade_mm` | `blade_legacy.py`, `blade_updated.py`, `blade_mm.py` | Boundary-weighted ascending-ε ladder ensemble |
| — | `alma_prox.py` | ALMA prox minimum-norm attack (Rony et al., CVPR 2023) |

`a_pgd_campaign.py` is kept frozen for reproducibility of the published AS-PGD numbers; new work should use `a_pgd.py`.
`auto_pgd_plus.py` is a documented negative result.

## Setup

```bash
pip install -r requirements.txt      # CUDA 11.7 / torch 2.0.1 / nnunetv2 2.2.1 / numpy 1.26.2
pip install -e .                     # optional: installs the package and mri-* console scripts

export nnUNet_raw="$PWD/nnUnet_paths/nnUNet_raw"
export nnUNet_preprocessed="$PWD/nnUnet_paths/nnUNet_preprocessed"
export nnUNet_results="$PWD/nnUnet_paths/nnUNet_results"
```

Without `pip install -e .`, prefix commands with `PYTHONPATH=src`.

## Trained checkpoints

The trained weights are not included in this repository; they are available from the corresponding author on
reasonable request. Place each `checkpoint_final.pth` at

```
nnUnet_paths/nnUNet_results/<Dataset>/nnUNetTrainer__nnUNetPlans__3d_fullres/fold_<N>/checkpoint_final.pth
```

next to the `plans.json` and `dataset.json` already provided.

To retrain with the same folds, run `nnUNetv2_plan_and_preprocess` on your data and keep the provided
`splits_final.json`; nnU-Net uses it instead of generating new splits.

## Data

The imaging data are not redistributed. The whole-gland model was trained on 951 cases of the public PI-CAI
dataset ([Zenodo](https://zenodo.org/records/6624726)) and 445 cases of a multi-center in-house cohort; the zonal
model was trained on 603 in-house cases. The in-house data are not publicly available.

## Running an attack evaluation

```bash
PYTHONPATH=src python experiments/pgd_adversarial_evaluation.py \
  --attack pgd --model wg --fold 0 --max-samples 12 --output-dir results/scratch
```

Use `--max-samples` for a smoke test before full runs. Seed 42 is used throughout.

## Tests

```bash
PYTHONPATH=src python -m pytest
```

## License

Apache License 2.0 — see `LICENSE`.
