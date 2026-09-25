"""Replay campaign attacks on a case subset: where the damage lands, and how it scales.

Each (arm, step count) is replayed exactly as ``benchmark_attack_cost.py``
replays the campaign call (native per epsilon for step arms, the ladder for
BLADE arms, seed 42, confined for the zones), but with ``--steps`` overriding
the campaign's 20 and with the adversarial snapshots kept. For every snapshot
the script records:

* Dice per foreground target, gradients spent, and wall-clock (every run);
* with ``--direction``: the segmentation-direction summary of
  ``mri_prostate_seg.experiments.segmentation_direction`` (induced false
  positives and negatives, volume change, surface motion, centroid shift) per
  target, the signed-distance damage profile, and where the perturbation's
  energy sits relative to the ground-truth gland boundary.

Targets: ``WG`` for the whole-gland model; ``TZ+CZ``, ``PZ`` and their union
``gland`` for the zones model, plus the TZ<->PZ swap volume.

Rows are appended as they finish and ``(arm, steps, case)`` triples already on
disk are skipped, so an interrupted shard resumes where it stopped.

Usage::

    PYTHONPATH=src python experiments/run_blade_mechanism_study.py \
        --task wg --arms auto_pgd sea blade --steps 20 --cases 60 --direction \
        --output-dir results/blade_attack/mechanism/direction
"""

from __future__ import annotations

import argparse
import csv
import functools
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))
import pgd_adversarial_evaluation as pgd_mod  # noqa: E402
from benchmark_attack_cost import (  # noqa: E402
    RANDOM_START,
    Counting,
    attack_fn_for,
)
from fgsm_adversarial_evaluation import (  # noqa: E402
    ConfinedInput,
    _make_eval_config,
    attackable_mask,
    confine,
    fgsm_attack,
    get_checkpoint_for_fold,
    load_model,
    load_sample,
    load_validation_ids,
    pad_to_divisible,
    samplewise_bounds,
    unpad,
)
from run_twelve_arm_wg_campaign import (  # noqa: E402
    ARM_RESTARTS,
    ARMS,
    EPSILONS,
    LADDER_ARMS,
    NUM_RESTARTS,
    SEED,
    TRAINERS,
)

from mri_prostate_seg.experiments.perturbation_structure import (  # noqa: E402
    SIGNED_BAND_NAMES,
    signed_band_masks,
)
from mri_prostate_seg.experiments.segmentation_direction.case import (  # noqa: E402
    analyze_binary_target,
    target_geometry,
)

IGNORE_LABEL = -1
TOLERANCE_MM = 0.5  # same surface tolerance as the segmentation-direction study


# Factorial controls that are not campaign arms: wrapper name in the driver,
# whether it runs as a ladder (one call over all budgets) and its restarts.
# On the zones they use the BLADE-MC machinery, like the campaign's BLADE arms.
FACTORIAL_ARMS = {
    "blade_noladder": ("blade_noladder_with_restarts", False, 1),
    "sea_ladder": ("sea_ladder_with_restarts", True, 1),
    "blade1_r3": ("blade1_with_restarts", True, 3),
}


def _resolve(arm, task, cfg):
    """(attack function, runs as a ladder, restarts, random start) for an arm."""
    if arm not in FACTORIAL_ARMS:
        return (attack_fn_for(arm, task, cfg), arm in LADDER_ARMS,
                ARM_RESTARTS.get(arm, NUM_RESTARTS), RANDOM_START[arm])
    name, ladder, restarts = FACTORIAL_ARMS[arm]
    fn = getattr(pgd_mod, name)
    if task == "zones":
        fn = functools.partial(fn, blade_impl="mc", spacing=pgd_mod._plan_spacing(cfg))
    return fn, ladder, restarts, True


def snapshots_for(arm, attacked, x, y, eps_list, num_classes, task, cfg, n_steps):
    """The campaign's attack call(s) for one case; returns {eps: x_adv}."""
    if arm == "fgsm":
        # Single step, no restarts: the campaign's own fgsm_attack per epsilon.
        return {e: fgsm_attack(attacked, x, y, e, num_classes) for e in eps_list}
    fn, ladder, restarts, random_start = _resolve(arm, task, cfg)
    x_min, x_max = samplewise_bounds(x)
    common = dict(
        n_steps=n_steps,
        step_size=None,
        momentum=0.0,
        normalized_grad=False,
        x_bounds=(x_min, x_max),
        num_restarts=restarts,
        loss_log=None,
        random_start=random_start,
    )
    with torch.enable_grad():
        if ladder:
            snaps = fn(
                attacked,
                x,
                y,
                max(eps_list),
                num_classes,
                snapshot_eps=eps_list,
                **common,
            )
            return {e: snaps[e] for e in eps_list}
        return {
            e: fn(attacked, x, y, e, num_classes, snapshot_eps=[e], **common)[e]
            for e in eps_list
        }


def targets_of(seg: np.ndarray, task: str) -> dict[str, np.ndarray]:
    if task == "wg":
        return {"WG": seg == 1}
    return {"TZ+CZ": seg == 1, "PZ": seg == 2, "gland": seg >= 1}


def pred_targets(pred: np.ndarray, task: str) -> dict[str, np.ndarray]:
    return targets_of(pred, task)


def dice(a: np.ndarray, b: np.ndarray) -> float:
    denom = int(a.sum()) + int(b.sum())
    return float(2.0 * int((a & b).sum()) / denom) if denom else 1.0


def perturbation_bands(
    delta: np.ndarray, gland_distance: np.ndarray, valid: np.ndarray
) -> dict:
    """Share of the perturbation's energy (sum of delta^2) per signed band."""
    energy = np.where(valid, delta.astype(np.float64) ** 2, 0.0)
    total = float(energy.sum())
    bands = signed_band_masks(gland_distance)
    out = {
        "delta_linf": float(np.abs(delta[valid]).max())
        if valid.any()
        else float("nan"),
        "delta_l2": float(np.sqrt(total)),
        "delta_nonzero_frac": float(
            np.count_nonzero(delta[valid]) / max(valid.sum(), 1)
        ),
    }
    for name in SIGNED_BAND_NAMES:
        out[f"energy_share_{name}"] = (
            float(energy[bands[name]].sum() / total) if total else float("nan")
        )
        n = int((bands[name] & valid).sum())
        out[f"energy_density_{name}"] = (
            float(energy[bands[name]].sum() / n) if n else float("nan")
        )
    return out


def append_rows(path: Path, rows: list[dict]) -> None:
    if not rows:
        return
    new = not path.is_file()
    with path.open("a", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(rows[0]))
        if new:
            w.writeheader()
        w.writerows(rows)


def done_keys(path: Path) -> set[tuple]:
    if not path.is_file():
        return set()
    t = pd.read_csv(path, usecols=["arm", "steps", "case_id"])
    return set(zip(t["arm"], t["steps"], t["case_id"]))


def run(args: argparse.Namespace) -> None:
    device = torch.device("cuda")
    eps_list = [float(e) for e in EPSILONS if float(e) > 0]
    cfg = _make_eval_config(args.task, None)
    cfg["epsilons"] = [0.0, *eps_list]
    cfg = {**cfg, "trainer": TRAINERS[args.arch]}
    num_classes = cfg["num_classes"]
    spacing = pgd_mod._plan_spacing(cfg)
    confined = args.task == "zones"
    case_ids = load_validation_ids(cfg["splits_json"], args.fold)[: args.cases]
    case_ids = case_ids[args.shard :: args.num_shards]

    out = Path(args.output_dir)
    out.mkdir(parents=True, exist_ok=True)
    tag = f"{args.task}_{args.arch}_shard{args.shard}of{args.num_shards}"
    summary_path = out / f"summary_{tag}.csv"
    profile_path = out / f"profile_{tag}.csv"
    done = done_keys(summary_path)

    network, div = load_model(
        get_checkpoint_for_fold(cfg, args.fold),
        cfg["configuration"],
        num_classes,
        device,
    )
    network.eval()
    counter = Counting(network)

    for case_id in case_ids:
        todo = [
            (a, s) for s in args.steps for a in args.arms if (a, s, case_id) not in done
        ]
        if not todo:
            continue
        img, seg = load_sample(cfg["data_dir"], case_id)
        seg = np.asarray(seg, dtype=np.int64)
        x = torch.from_numpy(img[np.newaxis]).to(device)
        y = torch.from_numpy(seg[np.newaxis].astype(np.float32)).to(device)
        xp, orig_shape = pad_to_divisible(x, div)
        yp, _ = pad_to_divisible(y, div)
        mask = attackable_mask(y, div) if confined else None
        gt = seg[0]
        valid = gt != IGNORE_LABEL

        def predict(inp: torch.Tensor) -> np.ndarray:
            with torch.no_grad():
                logits = network(confine(inp, xp, mask))
            return unpad(logits.float(), orig_shape).argmax(1)[0].cpu().numpy()

        clean_pred = predict(xp)
        mask_dir = Path(args.save_masks) if args.save_masks else None
        if mask_dir is not None:
            mask_dir.mkdir(parents=True, exist_ok=True)
            ref = mask_dir / f"{args.task}_{case_id}_reference.npz"
            if not ref.is_file():
                np.savez_compressed(
                    ref, gt=gt.astype(np.int8), clean=clean_pred.astype(np.uint8),
                    spacing=np.asarray(spacing, dtype=np.float64),
                )
        gt_t, clean_t = targets_of(gt, args.task), pred_targets(clean_pred, args.task)
        gt_t = {k: v & valid for k, v in gt_t.items()}
        clean_t = {k: v & valid for k, v in clean_t.items()}
        geometry, gland_distance = {}, None
        if args.direction:
            geometry = {k: target_geometry(gt_t[k], clean_t[k], spacing) for k in gt_t}
            gland = "WG" if args.task == "wg" else "gland"
            gland_distance = geometry[gland].phi_gt
        x_clean = unpad(xp, orig_shape)[0, 0].cpu().numpy()

        for arm, steps in todo:
            att = ConfinedInput(counter, xp, mask) if mask is not None else counter
            torch.manual_seed(SEED)
            counter.reset()
            torch.cuda.synchronize()
            t0 = time.perf_counter()
            snaps = snapshots_for(
                arm, att, xp, yp, eps_list, num_classes, args.task, cfg, steps
            )
            torch.cuda.synchronize()
            seconds = time.perf_counter() - t0
            base = dict(
                task=args.task,
                arch=args.arch,
                arm=arm,
                steps=steps,
                case_id=case_id,
                n_backward=counter.n_backward,
                seconds=round(seconds, 3),
            )
            summary_rows, profile_rows = [], []
            for eps in eps_list:
                x_adv = confine(snaps[eps], xp, mask)
                adv_pred = predict(x_adv)
                if mask_dir is not None:
                    np.savez_compressed(
                        mask_dir / f"{args.task}_{case_id}_{arm}_steps{steps}_eps{eps:.2f}.npz",
                        adv=adv_pred.astype(np.uint8),
                    )
                adv_t = {
                    k: v & valid for k, v in pred_targets(adv_pred, args.task).items()
                }
                pert = {}
                if args.direction:
                    delta = unpad(x_adv, orig_shape)[0, 0].cpu().numpy() - x_clean
                    pert = perturbation_bands(delta, gland_distance, valid)
                    if args.task == "zones":
                        swap = (
                            valid
                            & (gt >= 1)
                            & (
                                ((clean_pred == 1) & (adv_pred == 2))
                                | ((clean_pred == 2) & (adv_pred == 1))
                            )
                        )
                        pert["zone_swap_mm3"] = float(swap.sum() * np.prod(spacing))
                for name in gt_t:
                    row = {
                        **base,
                        "epsilon": eps,
                        "target": name,
                        "clean_dice": dice(clean_t[name], gt_t[name]),
                        "adv_dice": dice(adv_t[name], gt_t[name]),
                    }
                    if args.direction:
                        summ, prof = analyze_binary_target(
                            gt_t[name],
                            clean_t[name],
                            adv_t[name],
                            valid=valid,
                            spacing=spacing,
                            tolerance_mm=TOLERANCE_MM,
                            geometry=geometry[name],
                        )
                        summ = {
                            k: v
                            for k, v in summ.items()
                            if k not in ("clean_dice", "adv_dice")
                        }
                        row.update(summ)
                        row.update(pert)
                        profile_rows += [
                            {**base, "epsilon": eps, "target": name, **p} for p in prof
                        ]
                    summary_rows.append(row)
            append_rows(summary_path, summary_rows)
            append_rows(profile_path, profile_rows)
            print(
                f"{tag} {case_id} {arm} steps={steps}: {seconds:.1f} s, "
                f"{counter.n_backward} grads",
                flush=True,
            )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--task", choices=["wg", "zones"], required=True)
    parser.add_argument("--arch", choices=list(TRAINERS), default="unet")
    parser.add_argument(
        "--arms", nargs="+", choices=[*ARMS, *FACTORIAL_ARMS], required=True
    )
    parser.add_argument("--steps", nargs="+", type=int, default=[20])
    parser.add_argument("--cases", type=int, default=60)
    parser.add_argument("--fold", type=int, default=0)
    parser.add_argument("--shard", type=int, default=0)
    parser.add_argument("--num-shards", type=int, default=1)
    parser.add_argument("--direction", action="store_true")
    parser.add_argument(
        "--save-masks",
        default=None,
        help="directory for argmax masks: <task>_<case>_reference.npz (ground truth, "
        "clean prediction, spacing) and <task>_<case>_<arm>_steps<N>_eps<e>.npz",
    )
    parser.add_argument("--output-dir", required=True)
    run(parser.parse_args())


if __name__ == "__main__":
    main()
