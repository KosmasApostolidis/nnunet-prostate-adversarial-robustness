"""Attack-strength checks for the ResEnc EMBC paper (review items R9 and R10).

Runs Auto-PGD on a whole-gland fold-0 subset for nnU-Net, ResEnc-M and ResEnc-XL
at native per-epsilon budgets, with two objectives and several iteration counts:

* ``dice_ce``: the campaign objective (soft Dice + cross-entropy), at 20, 50 and
  100 iterations. R10: does the model ordering or the step sizes change when
  the 20-iteration budget of the paper is raised?
* ``margin``: a class-balanced per-voxel margin loss (Carlini-Wagner, kappa = 0):
  for each class, the mean over its voxels of min(max_{j!=y} z_j - z_y, 0),
  averaged over classes. It is linear in the logits for every still-correct
  voxel, so unlike cross-entropy it does not down-weight voxels with a large
  logit margin. R9: does the robustness gain survive an objective that does
  not saturate as margins grow?

Everything else is the campaign protocol: the driver's data loader, per-sample
intensity bounds, padding, single forward pass, Auto-PGD port (step 2*eps,
eps-sphere random start, highest-loss iterate), metrics and seed 42.

    PYTHONPATH=src python experiments/resenc_attack_sanity_subset.py \
        --arch resenc_xl --every 4 --output-dir results/.../attack_sanity
"""

from __future__ import annotations

import argparse
import csv
import os
import sys
import time

import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from fgsm_adversarial_evaluation import (  # noqa: E402
    IGNORE_LABEL,
    compute_metrics,
    dice_ce_loss,
    get_checkpoint_for_fold,
    load_model,
    load_sample,
    load_validation_ids,
    pad_to_divisible,
    samplewise_bounds,
    unpad,
    _make_eval_config,
)
from mri_prostate_seg.attacks.auto_pgd import auto_pgd_trajectory  # noqa: E402

TRAINERS = {
    "unet": "nnUNetTrainer__nnUNetPlans__3d_fullres",
    "resenc_m": "nnUNetTrainer__nnUNetResEncUNetMPlans__3d_fullres",
    "resenc_l": "nnUNetTrainer__nnUNetResEncUNetLPlans__3d_fullres",
    "resenc_xl": "nnUNetTrainer__nnUNetResEncUNetXLPlans__3d_fullres",
}
FIELDS = [
    "arch",
    "fold",
    "case_id",
    "loss",
    "iters",
    "eps",
    "dice",
    "hd95",
    "asd",
    "sec",
]


def margin_loss(
    logits: torch.Tensor, target: torch.Tensor, num_classes: int
) -> torch.Tensor:
    """Class-balanced CW margin loss (kappa = 0), to be maximised."""
    t = target.squeeze(1).long()
    valid = t != IGNORE_LABEL
    t_safe = torch.where(valid, t, torch.zeros_like(t))
    z_true = logits.gather(1, t_safe.unsqueeze(1)).squeeze(1)
    others = logits.scatter(1, t_safe.unsqueeze(1), float("-inf"))
    gap = (others.amax(dim=1) - z_true).clamp(max=0.0)
    per_class = [
        gap[valid & (t == c)].mean()
        for c in range(num_classes)
        if (valid & (t == c)).any()
    ]
    return torch.stack(per_class).mean()


LOSSES = {"dice_ce": dice_ce_loss, "margin": margin_loss}


def done_keys(path: str) -> set[tuple]:
    if not os.path.exists(path):
        return set()
    with open(path) as f:
        return {
            (r["case_id"], r["loss"], int(r["iters"]), float(r["eps"]))
            for r in csv.DictReader(f)
        }


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--arch", choices=sorted(TRAINERS), required=True)
    ap.add_argument("--fold", type=int, default=0)
    ap.add_argument(
        "--every", type=int, default=4, help="keep every k-th validation case"
    )
    ap.add_argument("--max-cases", type=int, default=None)
    ap.add_argument("--epsilons", type=float, nargs="+", default=[0.02, 0.06])
    ap.add_argument(
        "--runs",
        nargs="+",
        default=["dice_ce:20", "dice_ce:50", "dice_ce:100", "margin:20", "margin:100"],
        help="loss:iterations pairs",
    )
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--min-free-gb", type=float, default=12.0)
    ap.add_argument("--output-dir", required=True)
    args = ap.parse_args()

    runs = [(name, int(n)) for name, n in (r.split(":") for r in args.runs)]
    os.makedirs(args.output_dir, exist_ok=True)
    out_path = os.path.join(args.output_dir, f"wg_{args.arch}_fold{args.fold}.csv")
    done = done_keys(out_path)

    device = torch.device("cuda")
    # The GPU is shared with other campaigns: wait for room rather than OOM.
    while torch.cuda.mem_get_info()[0] / 1e9 < args.min_free_gb:
        print(f"waiting for {args.min_free_gb} GB free GPU memory", flush=True)
        time.sleep(60)
    cfg = {**_make_eval_config("wg", None), "trainer": TRAINERS[args.arch]}
    net, div = load_model(
        get_checkpoint_for_fold(cfg, args.fold),
        cfg["configuration"],
        cfg["num_classes"],
        device,
    )
    net.eval()
    num_classes = cfg["num_classes"]
    ids = load_validation_ids(cfg["splits_json"], args.fold)[:: args.every][
        : args.max_cases
    ]
    print(
        f"{args.arch} fold{args.fold}: {len(ids)} cases, runs={runs}, eps={args.epsilons}",
        flush=True,
    )

    new_file = not os.path.exists(out_path)
    with open(out_path, "a", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=FIELDS)
        if new_file:
            writer.writeheader()
        for case_id in ids:
            img, seg = load_sample(cfg["data_dir"], case_id)
            seg = np.asarray(seg, dtype=np.int64)
            x = torch.from_numpy(img[np.newaxis]).to(device)
            y = torch.from_numpy(seg[np.newaxis].astype(np.float32)).to(device)
            bounds = samplewise_bounds(x)
            xp, shape = pad_to_divisible(x, div)
            yp, _ = pad_to_divisible(y, div)
            for loss_name, iters in runs:
                for eps in args.epsilons:
                    if (case_id, loss_name, iters, float(eps)) in done:
                        continue
                    # Seed per (case, run, eps) so rows are reproducible and resumable.
                    torch.manual_seed(args.seed)
                    t0 = time.time()
                    with torch.enable_grad():
                        _, snaps = auto_pgd_trajectory(
                            net,
                            xp,
                            yp,
                            eps,
                            num_classes,
                            LOSSES[loss_name],
                            n_steps=iters,
                            random_start=True,
                            x_bounds=bounds,
                            snapshot_eps=[eps],
                        )
                    with torch.no_grad():
                        logits = unpad(net(snaps[eps]).float(), shape)
                    d, h, a = compute_metrics(logits, seg[np.newaxis], num_classes)[1]
                    writer.writerow(
                        dict(
                            arch=args.arch,
                            fold=args.fold,
                            case_id=case_id,
                            loss=loss_name,
                            iters=iters,
                            eps=eps,
                            dice=f"{d:.6f}",
                            hd95=f"{h:.6f}",
                            asd=f"{a:.6f}",
                            sec=f"{time.time() - t0:.1f}",
                        )
                    )
                    f.flush()
                    del snaps, logits
            print(f"{args.arch} {case_id} done", flush=True)
            del x, y, xp, yp
            torch.cuda.empty_cache()


if __name__ == "__main__":
    main()
