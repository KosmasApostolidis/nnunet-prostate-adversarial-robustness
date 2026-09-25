# Running nnU-Net with the corpus AT-loss functions (NuAT / UIAT / LBGAT)

Step-by-step guide for training the prostate segmentation nnU-Nets with the
adversarial-training loss terms in `robust_loss_utils.py`. Each corpus loss is a
**separate trainer** selected by name with `-tr`; the plain TRADES baseline is
left untouched. Covers the smoke run first (cheap plumbing check), then a real run.

> **Scope.** Single-modality T2-weighted MRI. Whole-gland = `Dataset016`,
> zones = `Dataset019`.
>
> | Trainer (`-tr`) | Loss |
> |---|---|
> | `nnUNetTrainerRobustLoss` | TRADES-KL + gradient-difference baseline (unchanged) |
> | `nnUNetTrainerNuAT` | baseline + nuclear-norm penalty |
> | `nnUNetTrainerUIAT` | baseline + inverse-adversary KL |
> | `nnUNetTrainerLBGAT` | **stub** — raises `NotImplementedError` (frozen reference not wired) |
>
> Each corpus trainer subclasses `nnUNetTrainerRobustLoss` and adds its term via
> the base's generic `_extra_robust_term` seam. `NuAT` and `UIAT` are runnable today.

---

## 0. Prerequisites (one-time)

1. **The trainer + utils must be the files Python actually imports.** They live
   in the editable nnU-Net install, NOT only in this repo:

   ```
   nnUNet/nnunetv2/training/nnUNetTrainer/variants/loss/nnUNetTrainerRobustLoss.py
   nnUNet/nnunetv2/training/nnUNetTrainer/variants/loss/robust_loss_utils.py
   ```

   The corpus trainers live in `nnUNetTrainerCorpusLoss.py` in the same dir.
   Verify nnU-Net discovers them (from the loaded copies):

   ```bash
   python -c "
   from batchgenerators.utilities.file_and_folder_operations import join
   from nnunetv2.utilities.find_class_by_name import recursive_find_python_class
   import nnunetv2
   base = join(nnunetv2.__path__[0], 'training', 'nnUNetTrainer')
   for n in ['nnUNetTrainerRobustLoss','nnUNetTrainerNuAT','nnUNetTrainerUIAT','nnUNetTrainerLBGAT']:
       print(n, '->', 'FOUND' if recursive_find_python_class(base, n, 'nnunetv2.training.nnUNetTrainer') else 'MISSING')
   "
   ```

   > **Note:** the version-controlled copy in `MRI-.../training/nnUNetTrainerRobustLoss.py`
   > has diverged from this loaded copy (different defaults). The **loaded** copy
   > above is the source of truth for what runs. Do not assume editing the repo
   > copy changes training behavior.

2. **nnU-Net path env vars** (point at this repo's `nnUnet_paths/`):

   ```bash
   export nnUNet_raw="$PWD/nnUnet_paths/nnUNet_raw"
   export nnUNet_preprocessed="$PWD/nnUnet_paths/nnUNet_preprocessed"
   export nnUNet_results="$PWD/nnUnet_paths/nnUNet_results"
   ```

3. **Preprocessed data present.** `Dataset016` and `Dataset019` are already
   preprocessed under `nnUNet_preprocessed/`. If starting from raw:

   ```bash
   nnUNetv2_plan_and_preprocess -d 016 -c 3d_fullres --verify_dataset_integrity
   ```

4. **Run the unit + integration tests:**

   ```bash
   python training/test_corpus_losses.py        # CPU: loss terms + trainer glue
   python training/test_robust_trainer.py        # CPU: discovery + utils
   python training/integration_corpus_gpu.py     # GPU: real train_step per trainer, both datasets
   ```

   The GPU integration test builds the real network/loss from each dataset's plans
   and drives a synthetic batch through `train_step` for NuAT/UIAT/LBGAT on
   Dataset016 + Dataset019 — the most direct "runs in the framework" check.

---

## 1. Inject the `robust_loss` config

The trainer reads a `robust_loss` block from the dataset's `dataset.json` **or**
`nnUNetPlans.json` (dataset.json wins on conflicts). The least-invasive place is
the **preprocessed `dataset.json`** — it does not change the plans identifier or
the results-folder name. The SAME block works for every corpus trainer; which
corpus loss runs is chosen by `-tr`, not by a config key.

Open:

```
nnUnet_paths/nnUNet_preprocessed/Dataset016_WgSegmentationPNetAndPicai/dataset.json
```

Add a top-level `robust_loss` key. **Smoke** values (reaches the adversarial
phase immediately, nonzero `lambda_corpus`) — copy from
`training/robust_loss_smoke_config.json`:

```json
"robust_loss": {
  "lambda_corpus": 0.1,
  "nuat_max_voxels": 512,
  "inv_steps": 1,
  "warm_adv_epochs": 0,
  "fgsm_epochs": 1,
  "mi_fgsm_epochs": 0,
  "pgd_epochs": 1,
  "eps_min": 0.01,
  "eps_max": 0.02,
  "lambda_adv_base": 1.0,
  "lambda_gp_base": 0.1,
  "pgd_steps": 2,
  "pgd_restarts": 1,
  "mi_steps": 2
}
```

> Keep the existing `dataset.json` content; only ADD the `robust_loss` key
> (valid JSON — mind the comma). Back up the file first.

### Config keys that matter for the corpus losses

| key | meaning |
|---|---|
| `lambda_corpus` | weight of the corpus term in the total loss (0 ⇒ no-op; keep > 0) |
| `nuat_max_voxels` | NuAT: max boundary voxels in the per-sample SVD matrix (tractability cap) |
| `inv_steps` | UIAT: inverse-adversary ascent steps toward the true label |
| `warm_adv_epochs`/`fgsm_epochs`/`mi_fgsm_epochs`/`pgd_epochs` | 4-stage schedule lengths; total epochs = their sum |
| `eps_max` | L∞ budget (fraction of input intensity range) |

> The corpus term is **added on top of** the existing TRADES objective and is
> only active when the adversarial phase is active (`lambda_adv`/`lambda_gp` > 0).
> Set `warm_adv_epochs: 0` for the smoke run so it engages on epoch 0.

---

## 2. Smoke run (1 fold, ~1–2 epochs)

Pick the corpus trainer with `-tr`:

```bash
nnUNetv2_train 016 3d_fullres 0 -tr nnUNetTrainerNuAT
# or: -tr nnUNetTrainerUIAT   (inverse-adversary KL)
# or: -tr nnUNetTrainerRobustLoss   (plain TRADES baseline, no corpus term)
```

**What to check in the log** (`nnUNet_results/Dataset016_.../nnUNetTrainerNuAT__nnUNetPlans__3d_fullres/fold_0/training_log_*.txt`):

```
Epoch 0 [FGSM] | eps=0.0100 | r_adv: 0.0xxxx r_gp: 0.0xxxx | r_nuat: 0.0xxxx | Train Dice: 0.xx | Val Dice: 0.xx | Epoch time: .. s
```

Gate — the smoke run passes if:
- training starts, no crash, no NaN loss;
- the `r_<trainer>` field appears (e.g. `r_nuat` / `r_uiat`) and is finite/nonzero;
- the epoch completes and a checkpoint is written.

`nnUNetTrainerLBGAT` raises `NotImplementedError` (expected — reference model not wired).

---

## 3. Real run

Edit `dataset.json` `robust_loss` to real settings, e.g.:

```json
"robust_loss": {
  "lambda_corpus": 1.0,
  "nuat_max_voxels": 2048,
  "warm_adv_epochs": 50,
  "fgsm_epochs": 100,
  "mi_fgsm_epochs": 100,
  "pgd_epochs": 250,
  "eps_max": 0.03,
  "pgd_steps": 3,
  "pgd_restarts": 1
}
```

Train (repeat per fold 0–4 for cross-validation; per seed for variance):

```bash
nnUNetv2_train 016 3d_fullres 0 -tr nnUNetTrainerNuAT      # whole gland
nnUNetv2_train 019 3d_fullres 0 -tr nnUNetTrainerNuAT      # zones (region/sigmoid head)
```

> Zones (`Dataset019`) uses the region (multilabel sigmoid) head; whole gland
> (`Dataset016`) uses the non-region head. Both paths are handled automatically —
> no config difference needed.

Output goes to:

```
nnUNet_results/Dataset016_.../nnUNetTrainerNuAT__nnUNetPlans__3d_fullres/fold_0/
```

---

## 4. Evaluate adversarial robustness

The existing harness sweeps ε and reports per-case Dice / HD95 / ASD + ASR:

```bash
python experiments/pgd_adversarial_evaluation.py --model wg   --fold 0 --seed 0
python experiments/pgd_adversarial_evaluation.py --model zones --fold 0 --seed 0
```

> This harness currently uses PGD/FGSM only. A strong/adaptive (AutoAttack-class,
> loss-specific) evaluation — needed to rule out gradient masking per the
> literature-review G2 finding — is **not built yet**. Treat PGD-only robustness
> numbers as a lower bound on attack strength.

---

## 5. Comparing losses (the controlled study)

To compare NuAT vs UIAT vs the TRADES baseline fairly, hold **everything else
fixed** (arch, schedule, eps, seeds, split — same `robust_loss` block, same fold)
and vary only the trainer name (`-tr`). See
`research/outputs/adv_losses/10_experiment_plan.md` for the full grid (losses ×
seeds × ε), the phased gates, and the metrics schema.

---

## Gotchas

- **Edit the loaded copy, not the repo copy.** Behavior comes from
  `nnUNet/nnunetv2/.../`, not `MRI-.../training/`. (See §0.1.)
- **Pick the loss by trainer name (`-tr`), not a config key.** `nnUNetTrainerNuAT`,
  `nnUNetTrainerUIAT`, `nnUNetTrainerLBGAT`, or `nnUNetTrainerRobustLoss` (baseline).
- **`lambda_corpus: 0` is a silent no-op** — the term computes but contributes
  nothing. Keep it > 0.
- **`nnUNetTrainerLBGAT` is a stub** — raises `NotImplementedError` until a frozen
  clean reference model is wired in.
- **Corpus term is additive on top of TRADES**, not a replacement of the TRADES
  KL (a deliberate, flagged design choice; the papers replace it). Revisit before
  drawing mechanism conclusions.
- **`dataset.json` must stay valid JSON.** A trailing/missing comma will make
  nnU-Net fail to load the dataset before training starts.
- **3D AT is expensive.** Each adversarial step adds forward/backward passes.
  Budget accordingly; start with the smoke config.
