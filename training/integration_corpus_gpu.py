#!/usr/bin/env python
"""
GPU integration check: each corpus trainer (NuAT / UIAT / LBGAT) must run a real
nnU-Net train_step without errors, on both head types:
  - Dataset016 (whole gland, non-region / softmax head)
  - Dataset019 (zones, region / sigmoid multilabel head)

No preprocessed case data is required: we build the REAL network/loss/label_manager
from the dataset's plans, then feed a synthetic batch sized to the network's
deep-supervision outputs. A tiny robust_loss schedule is injected so the
adversarial phase (and thus the corpus term) is active on epoch 0.

LBGAT is expected to raise NotImplementedError (frozen reference not wired) -> PASS.

Run: python training/integration_corpus_gpu.py
"""
import os

_HERE = os.path.dirname(__file__)
os.environ.setdefault("nnUNet_raw", os.path.join(_HERE, "nnUnet_paths", "nnUNet_raw"))
os.environ.setdefault("nnUNet_preprocessed", os.path.join(_HERE, "nnUnet_paths", "nnUNet_preprocessed"))
os.environ.setdefault("nnUNet_results", os.path.join(_HERE, "nnUnet_paths", "nnUNet_results"))

import torch
from batchgenerators.utilities.file_and_folder_operations import join, load_json
from nnunetv2.paths import nnUNet_preprocessed, nnUNet_results

SMOKE_ROBUST = {
    "warm_adv_epochs": 0, "fgsm_epochs": 1, "mi_fgsm_epochs": 0, "pgd_epochs": 0,
    "eps_min": 0.01, "eps_max": 0.02, "lambda_adv_base": 1.0, "lambda_gp_base": 0.1,
    "pgd_steps": 2, "pgd_restarts": 1, "mi_steps": 2,
    "lambda_corpus": 0.1, "nuat_max_voxels": 256, "inv_steps": 1,
}

DATASETS = {
    "Dataset016_WgSegmentationPNetAndPicai": "WG/non-region",
    "Dataset019_ProstateZonesSegmentationWgFilteredLessDilated": "zones/region",
}
TRAINERS = ["nnUNetTrainerNuAT", "nnUNetTrainerUIAT", "nnUNetTrainerLBGAT"]


def _make_trainer(trainer_name, dataset_dir, device):
    from nnunetv2.utilities.find_class_by_name import recursive_find_python_class
    import nnunetv2
    plans = load_json(join(nnUNet_preprocessed, dataset_dir, "nnUNetPlans.json"))
    dataset_json = load_json(join(nnUNet_preprocessed, dataset_dir, "dataset.json"))
    rl = dict(SMOKE_ROBUST)  # inject schedule + corpus config
    # LBGAT needs a frozen clean reference of the same architecture (standard trainer).
    ref = join(nnUNet_results, dataset_dir,
               "nnUNetTrainer__nnUNetPlans__3d_fullres", "fold_0", "checkpoint_final.pth")
    if os.path.isfile(ref):
        rl["lbgat_reference_checkpoint"] = ref
    plans["robust_loss"] = rl
    plans.setdefault("continue_training", False)  # this fork's nnUNetTrainer.__init__ pops it
    cls = recursive_find_python_class(
        join(nnunetv2.__path__[0], "training", "nnUNetTrainer"),
        trainer_name, "nnunetv2.training.nnUNetTrainer",
    )
    tr = cls(plans, "3d_fullres", 0, dataset_json, device=device)
    tr.initialize()
    return tr


def _synthetic_batch(tr, device):
    """Synthetic data + deep-supervision targets matching the real network outputs."""
    patch = tr.configuration_manager.patch_size
    nch = tr.num_input_channels
    bsz = 2
    data = torch.randn(bsz, nch, *patch, device=device)

    with torch.no_grad():
        out = tr.network(data)
    outs = out if isinstance(out, (list, tuple)) else [out]

    targets = []
    has_ignore = tr.label_manager.has_ignore_label
    for o in outs:
        oc, sp = o.shape[1], o.shape[2:]
        if tr.label_manager.has_regions:
            n = oc + (1 if has_ignore else 0)
            t = (torch.rand(bsz, n, *sp, device=device) > 0.5).float()
        else:
            t = torch.randint(0, oc, (bsz, 1, *sp), device=device).long()
        targets.append(t)
    return {"data": data, "target": targets}


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"device: {device}")
    results = []
    for dataset_dir, head in DATASETS.items():
        for trainer_name in TRAINERS:
            tag = f"{trainer_name} on {dataset_dir.split('_')[0]} ({head})"
            try:
                tr = _make_trainer(trainer_name, dataset_dir, device)
                tr.on_train_epoch_start()  # epoch 0 -> FGSM phase, lambda_adv>0
                batch = _synthetic_batch(tr, device)
                out = tr.train_step(batch)
                loss = float(out["loss"])
                extra = float(out["extra"])
                ok = torch.isfinite(torch.tensor(loss)).item()
                status = "PASS" if (ok and extra > 0.0) else "FAIL"
                results.append((tag, status, f"loss={loss:.4f} extra={extra:.6f} phase={tr._phase_name}"))
            except Exception as e:
                import traceback
                results.append((tag, "FAIL", f"{type(e).__name__}: {e}"))
                traceback.print_exc()
            finally:
                torch.cuda.empty_cache() if torch.cuda.is_available() else None

    print("\n==== corpus-loss nnU-Net integration ====")
    n_pass = 0
    for tag, status, detail in results:
        print(f"[{status}] {tag}  ::  {detail}")
        n_pass += status == "PASS"
    print(f"\n{n_pass}/{len(results)} PASS")
    if n_pass != len(results):
        raise SystemExit(1)
    print("All corpus loss functions run in the nnU-Net framework without errors.")


if __name__ == "__main__":
    main()
