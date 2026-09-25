# All-Models Training Runbook — D016 + D019 (Clean → PGD-AT Fine-Tune)

Two-stage pipeline for all 8 architecture×dataset combinations: (1) clean `fold=all`
training, then (2) PGD-AT adversarial fine-tuning from the clean checkpoint.

## Status Overview

*Snapshot verified 2026-07-11 ~18:35 EEST — re-check with `ps aux | grep nnUNetv2_train`
and the relevant `fold_all/checkpoint_final.pth` before trusting "Running" rows or
launching a Part A command; training state changes continuously.*

**All 8 Stage 1 (clean `fold_all`) baselines are now complete** — both ResEnc XL
baselines finished 2026-07-11 (D016 11:37, D019 14:09), so every architecture is ready
for its Stage 2 PGD-AT fine-tune. Stage 2 status: **5 done, 2 running, 1 not started.**

| # | Dataset | Architecture | Plans Flag | Stage 1 (clean) | Stage 2 (PGD-AT) |
|---|---------|-------------|------------|-----------------|-------------------|
| 1 | D016 WG | Classical nnUNet | *(default)* | ✅ Done | ✅ Done (final 07-11 09:37) |
| 2 | D016 WG | ResEnc M | `nnUNetResEncUNetMPlans` | ✅ Done | ✅ Done (final 07-11 17:23, Dice 0.9738) |
| 3 | D016 WG | ResEnc L | `nnUNetResEncUNetLPlans` | ✅ Done | ✅ Done |
| 4 | D016 WG | ResEnc XL | `nnUNetResEncUNetXLPlans` | ✅ Done | 🔄 Running (epoch ~0/100, started 18:34) |
| 5 | D019 Zones | Classical nnUNet | *(default)* | ✅ Done | ✅ Done (Dice 0.9861) |
| 6 | D019 Zones | ResEnc M | `nnUNetResEncUNetMPlans` | ✅ Done | ✅ Done (final 07-11 15:35, Dice 0.9895) |
| 7 | D019 Zones | ResEnc L | `nnUNetResEncUNetLPlans` | ✅ Done | 🔄 Running (epoch ~12/100, ~638 s/epoch, started 14:46) |
| 8 | D019 Zones | ResEnc XL | `nnUNetResEncUNetXLPlans` | ✅ Done | ⬜ Run **B8** (not started) |

**PGD-AT hyperparameters** (both datasets, shared — eps = 5/255):
`eps=0.01961`, `pgd_steps=7`, `pgd_alpha=0.007`, `finetune_lr=1e-3`, `num_epochs=100`

---

## 0. Environment (required for every command)

```bash
conda activate prostagent

export nnUNet_raw="$REPO_ROOT/nnUnet_paths/nnUNet_raw"
export nnUNet_preprocessed="$REPO_ROOT/nnUnet_paths/nnUNet_preprocessed"
export nnUNet_results="$REPO_ROOT/nnUnet_paths/nnUNet_results"
export PYTHONPATH=/path/to/nnUNet:$PYTHONPATH
```

`PYTHONPATH` is critical: the pip-installed `nnunetv2==2.6.4` in `prostagent` does NOT
contain the custom AT trainers (`nnUNetTrainerPGDATFinetune`, etc.). Without it, the
trainer name won't be found.

---

## Output Directory Naming

Stage 1 → `$nnUNet_results/<Dataset>/nnUNetTrainer__<Plans>__3d_fullres/fold_all/`
Stage 2 → `$nnUNet_results/<Dataset>/nnUNetTrainerPGDATFinetune__<Plans>__3d_fullres/fold_all/`

Dataset names:
- D016 → `Dataset016_WgSegmentationPNetAndPicai`
- D019 → `Dataset019_ProstateZonesSegmentationWgFilteredLessDilated`

---

## Dataset Config Notes

### D016 (`Dataset016_WgSegmentationPNetAndPicai`)
- Labels: 2 (background, WG)
- Cases: 1396
- Raw `dataset.json` has `adv_training` block with eps/pgd_steps/pgd_alpha already populated.
- `finetune_checkpoint` must be set per-architecture before each Stage 2 run (see commands below).

### D019 (`Dataset019_ProstateZonesSegmentationWgFilteredLessDilated`)
- Labels: 3 (background, TZ+CZ, PZ)
- Cases: 603
- Raw `dataset.json` has `adv_training` block added (eps=0.01961, pgd_steps=7, pgd_alpha=0.007).
- `finetune_checkpoint` must be set per-architecture before each Stage 2 run.

---

## Part A — Stage 1: Clean `fold=all` Training

> ✅ **All Stage 1 baselines are complete as of 2026-07-11** — the commands below are
> retained for reference/re-runs only; nothing here needs launching.

Run for architectures missing `fold_all` (marked ⬜ **Need Stage 1** above).
These can run in **parallel** (independent GPU jobs).

Each uses the standard `nnUNetTrainer` (no adversarial component).

### A1. D016 — ResEnc M

```bash
nnUNetv2_train 016 3d_fullres all -tr nnUNetTrainer -p nnUNetResEncUNetMPlans
```

### A2. D016 — ResEnc XL

```bash
nnUNetv2_train 016 3d_fullres all -tr nnUNetTrainer -p nnUNetResEncUNetXLPlans
```

### A3. D019 — ResEnc M

```bash
nnUNetv2_train 019 3d_fullres all -tr nnUNetTrainer -p nnUNetResEncUNetMPlans
```

### A4. D019 — ResEnc L

```bash
nnUNetv2_train 019 3d_fullres all -tr nnUNetTrainer -p nnUNetResEncUNetLPlans
```

### A5. D019 — ResEnc XL

```bash
nnUNetv2_train 019 3d_fullres all -tr nnUNetTrainer -p nnUNetResEncUNetXLPlans
```

---

## Part B — Stage 2: PGD-AT Adversarial Fine-Tuning

Each command:
1. Writes the architecture-specific `finetune_checkpoint` path into the `nnUNet_preprocessed`
   `dataset.json` `adv_training` block (inline `python -c` one-liner).
2. Launches `nnUNetv2_train` with `nnUNetTrainerPGDATFinetune`.

**Check before running** — if
`$nnUNet_results/<Dataset>/nnUNetTrainerPGDATFinetune__<Plans>__3d_fullres/fold_all/checkpoint_final.pth`
already exists, that combo is already done (see Status Overview). Unlike
`run_5fold_pgdat_finetune.sh`, none of the commands below skip already-completed runs —
running one anyway will overwrite the existing checkpoint.

These **must** run after their Stage 1 completes (the trainer raises `ValueError` on
missing/shape-mismatched checkpoint). Run **sequentially** on a single GPU to avoid
CUDA-context contention.

### B1. D016 — Classical nnUNet

```bash
python -c "
import json
p = '$nnUNet_preprocessed/Dataset016_WgSegmentationPNetAndPicai/dataset.json'
d = json.load(open(p))
d.setdefault('adv_training', {})['finetune_checkpoint'] = '$nnUNet_results/Dataset016_WgSegmentationPNetAndPicai/nnUNetTrainer__nnUNetPlans__3d_fullres/fold_all/checkpoint_final.pth'
json.dump(d, open(p, 'w'), indent=2)
print('finetune_checkpoint set → classical nnUNet D016')
"
nnUNetv2_train 016 3d_fullres all -tr nnUNetTrainerPGDATFinetune
```

### B2. D016 — ResEnc M

```bash
python -c "
import json
p = '$nnUNet_preprocessed/Dataset016_WgSegmentationPNetAndPicai/dataset.json'
d = json.load(open(p))
d.setdefault('adv_training', {})['finetune_checkpoint'] = '$nnUNet_results/Dataset016_WgSegmentationPNetAndPicai/nnUNetTrainer__nnUNetResEncUNetMPlans__3d_fullres/fold_all/checkpoint_final.pth'
json.dump(d, open(p, 'w'), indent=2)
print('finetune_checkpoint set → ResEnc M D016')
"
nnUNetv2_train 016 3d_fullres all -tr nnUNetTrainerPGDATFinetune -p nnUNetResEncUNetMPlans
```

### B3. D016 — ResEnc L

```bash
python -c "
import json
p = '$nnUNet_preprocessed/Dataset016_WgSegmentationPNetAndPicai/dataset.json'
d = json.load(open(p))
d.setdefault('adv_training', {})['finetune_checkpoint'] = '$nnUNet_results/Dataset016_WgSegmentationPNetAndPicai/nnUNetTrainer__nnUNetResEncUNetLPlans__3d_fullres/fold_all/checkpoint_final.pth'
json.dump(d, open(p, 'w'), indent=2)
print('finetune_checkpoint set → ResEnc L D016')
"
nnUNetv2_train 016 3d_fullres all -tr nnUNetTrainerPGDATFinetune -p nnUNetResEncUNetLPlans
```

### B4. D016 — ResEnc XL

```bash
python -c "
import json
p = '$nnUNet_preprocessed/Dataset016_WgSegmentationPNetAndPicai/dataset.json'
d = json.load(open(p))
d.setdefault('adv_training', {})['finetune_checkpoint'] = '$nnUNet_results/Dataset016_WgSegmentationPNetAndPicai/nnUNetTrainer__nnUNetResEncUNetXLPlans__3d_fullres/fold_all/checkpoint_final.pth'
json.dump(d, open(p, 'w'), indent=2)
print('finetune_checkpoint set → ResEnc XL D016')
"
nnUNetv2_train 016 3d_fullres all -tr nnUNetTrainerPGDATFinetune -p nnUNetResEncUNetXLPlans
```

### B5. D019 — Classical nnUNet

```bash
python -c "
import json
p = '$nnUNet_preprocessed/Dataset019_ProstateZonesSegmentationWgFilteredLessDilated/dataset.json'
d = json.load(open(p))
d.setdefault('adv_training', {})['finetune_checkpoint'] = '$nnUNet_results/Dataset019_ProstateZonesSegmentationWgFilteredLessDilated/nnUNetTrainer__nnUNetPlans__3d_fullres/fold_all/checkpoint_final.pth'
json.dump(d, open(p, 'w'), indent=2)
print('finetune_checkpoint set → classical nnUNet D019')
"
nnUNetv2_train 019 3d_fullres all -tr nnUNetTrainerPGDATFinetune
```

### B6. D019 — ResEnc M

```bash
python -c "
import json
p = '$nnUNet_preprocessed/Dataset019_ProstateZonesSegmentationWgFilteredLessDilated/dataset.json'
d = json.load(open(p))
d.setdefault('adv_training', {})['finetune_checkpoint'] = '$nnUNet_results/Dataset019_ProstateZonesSegmentationWgFilteredLessDilated/nnUNetTrainer__nnUNetResEncUNetMPlans__3d_fullres/fold_all/checkpoint_final.pth'
json.dump(d, open(p, 'w'), indent=2)
print('finetune_checkpoint set → ResEnc M D019')
"
nnUNetv2_train 019 3d_fullres all -tr nnUNetTrainerPGDATFinetune -p nnUNetResEncUNetMPlans
```

### B7. D019 — ResEnc L

```bash
python -c "
import json
p = '$nnUNet_preprocessed/Dataset019_ProstateZonesSegmentationWgFilteredLessDilated/dataset.json'
d = json.load(open(p))
d.setdefault('adv_training', {})['finetune_checkpoint'] = '$nnUNet_results/Dataset019_ProstateZonesSegmentationWgFilteredLessDilated/nnUNetTrainer__nnUNetResEncUNetLPlans__3d_fullres/fold_all/checkpoint_final.pth'
json.dump(d, open(p, 'w'), indent=2)
print('finetune_checkpoint set → ResEnc L D019')
"
nnUNetv2_train 019 3d_fullres all -tr nnUNetTrainerPGDATFinetune -p nnUNetResEncUNetLPlans
```

### B8. D019 — ResEnc XL

```bash
python -c "
import json
p = '$nnUNet_preprocessed/Dataset019_ProstateZonesSegmentationWgFilteredLessDilated/dataset.json'
d = json.load(open(p))
d.setdefault('adv_training', {})['finetune_checkpoint'] = '$nnUNet_results/Dataset019_ProstateZonesSegmentationWgFilteredLessDilated/nnUNetTrainer__nnUNetResEncUNetXLPlans__3d_fullres/fold_all/checkpoint_final.pth'
json.dump(d, open(p, 'w'), indent=2)
print('finetune_checkpoint set → ResEnc XL D019')
"
nnUNetv2_train 019 3d_fullres all -tr nnUNetTrainerPGDATFinetune -p nnUNetResEncUNetXLPlans
```

---

## Resuming Interrupted Runs

Append `--c` to any `nnUNetv2_train` command to resume from the last saved checkpoint:

```bash
nnUNetv2_train 016 3d_fullres all -tr nnUNetTrainerPGDATFinetune -p nnUNetResEncUNetMPlans --c
```

---

## Inference

Use `-f all` to target the `fold_all` checkpoint:

```bash
# D016 — adversarially fine-tuned ResEnc L
nnUNetv2_predict -i INPUT_DIR -o OUTPUT_DIR -d 016 -c 3d_fullres \
  -tr nnUNetTrainerPGDATFinetune -p nnUNetResEncUNetLPlans -f all

# D019 — adversarially fine-tuned classical nnUNet
nnUNetv2_predict -i INPUT_DIR -o OUTPUT_DIR -d 019 -c 3d_fullres \
  -tr nnUNetTrainerPGDATFinetune -f all
```

---

## Recommended Execution Order

Stage 1 is fully complete, so all remaining work is Stage 2. As of 2026-07-11 ~18:35:

- **Done (5):** B1 (D016 classical), B3 (D016 ResEnc L), B5 (D019 classical), B6 (D019 ResEnc M, Dice 0.9895 — finished 15:35), B2 (D016 ResEnc M, Dice 0.9738 — finished 17:23).
- **Running (2):** B7 (D019 ResEnc L, ~epoch 12/100 at ~638 s/epoch ≈ 10.6 min/epoch → ~15.5 h remaining, ETA ~10:00 EEST 07-12) and B4 (D016 ResEnc XL, epoch ~0/100, started 18:34). The earlier "~33 min/epoch" for B7 was just the epoch-1 `torch.compile` warmup; steady-state is ~10.6 min/epoch.
- **Remaining to launch (1):** **B8** (D019 ResEnc XL).

With B4 (XL) + B7 (ResEnc L) both live the GPU is nearly full (**~130 GB / 143.8 GB used,
~14 GB free**), so **B8 cannot start concurrently** — an XL job needs ~65 GB. Launch B8
only after B7 finishes (frees ~47 GB, ETA ~10:00 EEST 07-12) or B4 finishes. Stagger any
new launch by ≥30 s from other `nnUNetv2_train` starts (see Gotchas) to avoid CUDA-context
contention.

Historical note (before Stage 1 finished): Stage 1 was 5 parallel jobs (A1–A5); Stage 2
was the sequential chain B1 → B2 → … → B8, one per GPU.

---

## Gotchas

- **`PYTHONPATH` must be set** — the pip-installed `nnunetv2==2.6.4` lacks AT trainers.
- **Edit `dataset.json` under `nnUNet_preprocessed`, not `nnUNet_raw`** — `nnUNetv2_train`
  loads `dataset_json` from `$nnUNet_preprocessed/<Dataset>/dataset.json`
  (`nnunetv2/run/run_training.py:66`); it never reads the raw copy. Writing
  `finetune_checkpoint`/`num_epochs` to raw's `dataset.json` has zero effect on
  training — the trainer silently keeps whatever was last written to preprocessed.
- **`finetune_checkpoint` must match the architecture** — ResEnc L weights loaded into an
  XL-plans run will silently succeed if `arch_kwargs` happen to match (same patch size),
  but produce garbage. Always verify the log line: *"loaded FULL pretrained weights
  (including seg_layers) from ..."* and confirm the path matches the architecture.
- **No two `nnUNetv2_train` jobs can start at the exact same time** — launching two jobs
  within ~30s causes the first to crash from CUDA-context contention. Stagger launches.
- **Stage 1 `fold=all` uses every case for training** — there is no validation split, so
  the validation Dice at the end is computed on the training set. This is intentional:
  the checkpoints are for deployment, not model selection.
- **`--c` resume picks up from the last saved checkpoint** — use it if a run is killed or
  crashes. It resumes the trainer/plans/fold combination that already exists on disk.
