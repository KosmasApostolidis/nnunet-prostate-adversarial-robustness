#!/usr/bin/env python
"""
GPU integration check: Fast-AT / Free-AT / YOPO each run a real nnU-Net train_step
without errors, on both head types (Dataset016 non-region, Dataset019 region).

No preprocessed cases needed: builds the real network/loss from each dataset's plans,
then drives a synthetic batch (sized to the network's deep-supervision outputs) through
each trainer's custom adversarial train_step. A tiny adv_training config keeps it fast.

Run: python training/integration_advtrain_gpu.py
"""
import os

_HERE = os.path.dirname(__file__)
os.environ.setdefault("nnUNet_raw", os.path.join(_HERE, "nnUnet_paths", "nnUNet_raw"))
os.environ.setdefault("nnUNet_preprocessed", os.path.join(_HERE, "nnUnet_paths", "nnUNet_preprocessed"))
os.environ.setdefault("nnUNet_results", os.path.join(_HERE, "nnUnet_paths", "nnUNet_results"))

import torch
from batchgenerators.utilities.file_and_folder_operations import join, load_json
from nnunetv2.paths import nnUNet_preprocessed
from nnunetv2.utilities.find_class_by_name import recursive_find_python_class
import nnunetv2

# Tiny config: small replay/step counts so the synthetic step is fast.
ADV_CFG = {"eps": 0.02, "fast_alpha": 0.025, "free_m": 2, "yopo_m": 2, "yopo_n": 2, "num_epochs": 1}

DATASETS = {
    "Dataset016_WgSegmentationPNetAndPicai": "WG/non-region",
    "Dataset019_ProstateZonesSegmentationWgFilteredLessDilated": "zones/region",
}
TRAINERS = ["nnUNetTrainerFastAT", "nnUNetTrainerFreeAT", "nnUNetTrainerYOPO"]


def _make_trainer(trainer_name, dataset_dir, device):
    plans = load_json(join(nnUNet_preprocessed, dataset_dir, "nnUNetPlans.json"))
    dataset_json = load_json(join(nnUNet_preprocessed, dataset_dir, "dataset.json"))
    plans["adv_training"] = dict(ADV_CFG)
    plans.setdefault("continue_training", False)  # this fork's nnUNetTrainer.__init__ pops it
    cls = recursive_find_python_class(
        join(nnunetv2.__path__[0], "training", "nnUNetTrainer"),
        trainer_name, "nnunetv2.training.nnUNetTrainer",
    )
    tr = cls(plans, "3d_fullres", 0, dataset_json, device=device)
    tr.initialize()
    return tr


def _synthetic_batch(tr, device):
    patch = tr.configuration_manager.patch_size
    data = torch.randn(2, tr.num_input_channels, *patch, device=device)
    with torch.no_grad():
        out = tr.network(data)
    outs = out if isinstance(out, (list, tuple)) else [out]
    has_ignore = tr.label_manager.has_ignore_label
    targets = []
    for o in outs:
        oc, sp = o.shape[1], o.shape[2:]
        if tr.label_manager.has_regions:
            ch = oc + (1 if has_ignore else 0)
            targets.append((torch.rand(2, ch, *sp, device=device) > 0.5).float())
        else:
            targets.append(torch.randint(0, oc, (2, 1, *sp), device=device).long())
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
                batch = _synthetic_batch(tr, device)
                out = tr.train_step(batch)
                loss = float(out["loss"])
                ok = bool(torch.isfinite(torch.tensor(loss)).item())
                results.append((tag, "PASS" if ok else "FAIL", f"loss={loss:.4f}"))
            except Exception as e:
                import traceback
                results.append((tag, "FAIL", f"{type(e).__name__}: {e}"))
                traceback.print_exc()
            finally:
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()

    print("\n==== Fast/Free/YOPO nnU-Net integration ====")
    n_pass = sum(s == "PASS" for _, s, _ in results)
    for tag, status, detail in results:
        print(f"[{status}] {tag}  ::  {detail}")
    print(f"\n{n_pass}/{len(results)} PASS")
    if n_pass != len(results):
        raise SystemExit(1)
    print("Fast-AT, Free-AT, and YOPO all run in the nnU-Net framework without errors.")


if __name__ == "__main__":
    main()
