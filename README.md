# Adversarial Robustness of nnU-Net for Prostate MRI Segmentation

Code for the paper:

> K. K. Apostolidis, D. I. Zaridis, V. C. Pezoulas, N. S. Tachos, E. Mylona, K. Marias, M. Tsiknakis, and D. I. Fotiadis,
> **"Evaluating the Adversarial Robustness of nnU-Net: A Slice-Level Vulnerability Analysis in Prostate MRI Segmentation."**

We attack two 3D nnU-Net v2 models, one for the whole gland (WG) and one for the prostate zones (TZ+CZ and PZ), with
FGSM, PGD and adaptive-step PGD (AS-PGD). We then measure where along the gland (base, mid-gland, apex) the
segmentation breaks down.

## What is in this repository

| Included | Where |
|----------|-------|
| Source code (attacks, evaluation, metrics, slice-level analysis, segmentation pipeline) | `src/mri_prostate_seg/` |
| Attack implementations | `src/mri_prostate_seg/attacks/` |
| Attack hyper-parameters | `configs/attacks/*.json` |
| Model definitions (dataset, trainer, ε grid) | `configs/models/wg.json`, `configs/models/zones.json` |
| Runnable experiment scripts | `experiments/` |
| nnU-Net plans of the trained models | `nnUnet_paths/nnUNet_results/<Dataset>/nnUNetTrainer__nnUNetPlans__3d_fullres/plans.json` |
| 5-fold cross-validation assignments | `nnUnet_paths/nnUNet_preprocessed/<Dataset>/splits_final.json` |

**Not included:**

- **Imaging data.** The WG model was trained on 951 cases from the public PI-CAI dataset
  ([Zenodo](https://zenodo.org/records/6624726)) and 445 cases from a multi-center in-house cohort. The zonal model was
  trained on 603 in-house cases. The in-house data are not publicly available.
- **Trained checkpoints.** These are available from the corresponding author on reasonable request.

## The two models

| | Whole gland (WG) | Zones |
|--|------------------|-------|
| nnU-Net dataset | `Dataset016_WgSegmentationPNetAndPicai` | `Dataset019_ProstateZonesSegmentationWgFilteredLessDilated` |
| Configuration | `3d_fullres` | `3d_fullres` |
| Labels | 0 background, 1 WG | 0 background, 1 TZ+CZ, 2 PZ |
| Cases | 1,396 | 603 |
| Per fold (train / validation) | ≈1,117 / 279 | ≈482 / 121 |

Each model was trained once per fold (5 checkpoints per model). Each checkpoint is attacked on its own validation
fold, so every case is evaluated exactly once.

## Setup

**1. Install**

```bash
git clone https://github.com/KosmasApostolidis/nnunet-prostate-adversarial-robustness.git
cd nnunet-prostate-adversarial-robustness
pip install -r requirements.txt    # pinned: CUDA 11.7, torch 2.0.1, nnunetv2 2.2.1, numpy 1.26.2
pip install -e .                   # optional; without it, prefix commands with PYTHONPATH=src
```

An NVIDIA GPU is needed for the attack evaluations.

**2. Point nnU-Net at the bundled folders**

```bash
export nnUNet_raw="$PWD/nnUnet_paths/nnUNet_raw"
export nnUNet_preprocessed="$PWD/nnUnet_paths/nnUNet_preprocessed"
export nnUNet_results="$PWD/nnUnet_paths/nnUNet_results"
```

**3. Add the checkpoints and data**

Put each checkpoint next to the plans that are already there:

```
nnUnet_paths/nnUNet_results/<Dataset>/nnUNetTrainer__nnUNetPlans__3d_fullres/
├── plans.json            (provided)
├── dataset.json          (provided)
└── fold_<N>/checkpoint_final.pth
```

If you are retraining on your own data instead, run `nnUNetv2_plan_and_preprocess` and keep the provided
`splits_final.json`. nnU-Net then uses these folds instead of generating new ones.

## Reproducing the paper

All commands are run from the repository root. Add `--max-samples 5` to any of them for a quick smoke test.

| Paper result | Command |
|--------------|---------|
| FGSM | `python experiments/fgsm_adversarial_evaluation.py --model both --all-folds --output-dir results/fgsm` |
| PGD | `python experiments/pgd_adversarial_evaluation.py --attack pgd --model both --fold <N> --pgd-steps 20 --output-dir results/pgd` |
| AS-PGD | `python experiments/pgd_adversarial_evaluation.py --attack a_pgd --model both --fold <N> --pgd-steps 20 --output-dir results/as_pgd` |
| Slice-level vulnerability (base / mid / apex) | `python experiments/slice_vulnerability_analysis.py --model both --fold <N> --output-dir results/slice` |

Run the PGD, AS-PGD and slice-level commands for each fold `<N>` = 0–4. The default ε grid is
{0, 0.02, 0.04, 0.06, 0.08, 0.1}; `--epsilons-wg` and `--epsilons-zones` override it. The seed is 42 throughout. Full
runs take many GPU-hours per fold.

## Attacks

The paper uses the first three. The others are implemented, configured and runnable through the same script
(`--attack <name>`).

| `--attack` | Module | Attack |
|------------|--------|--------|
| — (separate script) | `fgsm.py` | FGSM: one signed-gradient step |
| `pgd` | `pgd.py` | PGD, L∞ |
| `a_pgd` | `a_pgd_campaign.py` | AS-PGD, the paper's adaptive-step PGD (not Croce & Hein's Auto-PGD) |
| `auto_pgd` | `auto_pgd.py` | Auto-PGD (Croce & Hein, ICML 2020), ported to 3D segmentation |
| `apgd_updated` | `apgd_updated.py` | AS-PGD with a cosine step-size schedule |
| `segpgd` | `segpgd.py` | SegPGD (Gu et al., ECCV 2022) |
| `cospgd` | `cospgd.py` | CosPGD (Agnihotri et al., ICML 2024) |
| `dag` | `dag.py` | Dense Adversary Generation (Xie et al., ICCV 2017) |
| `sea` | `sea.py` | Segmentation Ensemble Attack (Croce et al., 2023) |
| `alma_prox` | `alma_prox.py` | ALMA prox minimum-norm attack (Rony et al., CVPR 2023) |
| `blade`, `blade1`, `blade_mm`, `blade_boundary`, `blade_frontier` | `blade_legacy.py`, `blade_updated.py`, `blade_mm.py` | BLADE: boundary-weighted ascending-ε ladder ensemble |

Notes:

- `a_pgd_campaign.py` is frozen so that the published AS-PGD numbers stay reproducible. `a_pgd.py` is a corrected
  version with a different best-iterate rule, which makes it a weaker attack. Use it for new work.
- `auto_pgd_plus.py` is a documented negative result (weaker than `auto_pgd`). It is kept for reference only.

## Repository layout

```
├── src/mri_prostate_seg/       main package
│   ├── attacks/                attack implementations
│   ├── eval/, metrics/         evaluation loop, Dice / HD95 / ASD
│   ├── experiments/            slice-vulnerability and perturbation analyses
│   ├── models/, pipeline/      nnU-Net loading and two-stage (WG → zones) inference
│   └── cli/                    console-script entry points
├── experiments/                runnable scripts (evaluation, statistics, figures)
├── configs/                    attack and model configurations
├── nnUnet_paths/               nnU-Net plans, dataset descriptors, fold splits
├── training/                   custom nnU-Net trainers (reference copies)
├── scripts/                    data-preparation and bookkeeping utilities
├── Utils/, __main__.py         segmentation-app entry point (Docker)
└── tests/                      unit and integration tests
```

## Tests

```bash
PYTHONPATH=src python -m pytest
```

## License

Apache License 2.0. See `LICENSE`.
