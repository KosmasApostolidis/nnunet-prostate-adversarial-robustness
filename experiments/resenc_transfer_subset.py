"""Transfer and large-epsilon checks for the ResEnc EMBC paper (review item REV-23).

On the same whole-gland fold-0 subset as ``resenc_attack_sanity_subset.py``,
Auto-PGD (campaign objective, 20 iterations) is run on each source model
(nnU-Net, ResEnc-M, ResEnc-XL) and the resulting adversarial input is scored by
every target model. The diagonal is the white-box result; the off-diagonal
cells are black-box transfer. If a model's robustness were gradient masking,
perturbations crafted on another model would hurt it more than its own.
epsilon = 0.3 (three times the paper's largest budget) checks that the
white-box attack can still drive every model towards zero Dice.

    PYTHONPATH=src python experiments/resenc_transfer_subset.py \
        --output-dir results/.../attack_sanity
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
from resenc_attack_sanity_subset import TRAINERS  # noqa: E402
from mri_prostate_seg.attacks.auto_pgd import auto_pgd_trajectory  # noqa: E402

ARCHS = ("unet", "resenc_m", "resenc_xl")
FIELDS = ["source", "target", "fold", "case_id", "eps", "dice", "hd95", "asd"]


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--fold", type=int, default=0)
    ap.add_argument("--every", type=int, default=9)
    ap.add_argument("--epsilons", type=float, nargs="+", default=[0.02, 0.06, 0.3])
    ap.add_argument("--iters", type=int, default=20)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--output-dir", required=True)
    args = ap.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)
    out_path = os.path.join(args.output_dir, f"wg_transfer_fold{args.fold}.csv")
    done: set[tuple] = set()
    if os.path.exists(out_path):
        with open(out_path) as f:
            done = {
                (r["source"], r["case_id"], float(r["eps"])) for r in csv.DictReader(f)
            }

    device = torch.device("cuda")
    base = _make_eval_config("wg", None)
    nets = {}
    divs = set()
    for arch in ARCHS:
        cfg = {**base, "trainer": TRAINERS[arch]}
        net, div = load_model(
            get_checkpoint_for_fold(cfg, args.fold),
            cfg["configuration"],
            cfg["num_classes"],
            device,
        )
        nets[arch] = net.eval()
        divs.add(tuple(div))
    # One padded input must be valid for every model.
    assert len(divs) == 1, f"models need different padding: {divs}"
    num_classes = base["num_classes"]
    ids = load_validation_ids(base["splits_json"], args.fold)[:: args.every]
    print(f"fold{args.fold}: {len(ids)} cases, eps={args.epsilons}", flush=True)

    new_file = not os.path.exists(out_path)
    with open(out_path, "a", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=FIELDS)
        if new_file:
            writer.writeheader()
        for case_id in ids:
            t0 = time.time()
            img, seg = load_sample(base["data_dir"], case_id)
            seg = np.asarray(seg, dtype=np.int64)
            x = torch.from_numpy(img[np.newaxis]).to(device)
            y = torch.from_numpy(seg[np.newaxis].astype(np.float32)).to(device)
            bounds = samplewise_bounds(x)
            xp, shape = pad_to_divisible(x, div)
            yp, _ = pad_to_divisible(y, div)
            for source in ARCHS:
                for eps in args.epsilons:
                    if (source, case_id, float(eps)) in done:
                        continue
                    torch.manual_seed(args.seed)
                    with torch.enable_grad():
                        _, snaps = auto_pgd_trajectory(
                            nets[source],
                            xp,
                            yp,
                            eps,
                            num_classes,
                            dice_ce_loss,
                            n_steps=args.iters,
                            random_start=True,
                            x_bounds=bounds,
                            snapshot_eps=[eps],
                        )
                    for target in ARCHS:
                        with torch.no_grad():
                            logits = unpad(nets[target](snaps[eps]).float(), shape)
                        d, h, a = compute_metrics(logits, seg[np.newaxis], num_classes)[
                            1
                        ]
                        writer.writerow(
                            dict(
                                source=source,
                                target=target,
                                fold=args.fold,
                                case_id=case_id,
                                eps=eps,
                                dice=f"{d:.6f}",
                                hd95=f"{h:.6f}",
                                asd=f"{a:.6f}",
                            )
                        )
                    f.flush()
                    del snaps
            print(f"{case_id} done ({time.time() - t0:.0f} s)", flush=True)
            del x, y, xp, yp
            torch.cuda.empty_cache()


if __name__ == "__main__":
    main()
