"""Computational cost of every campaign arm on every model: nominal and measured.

For each (task, architecture, arm) the campaign's attack call is replayed on
the first ``--cases`` validation cases of one fold, sequentially on an
otherwise idle GPU, under the campaign protocol (native per epsilon for the
step arms, the ladder for the BLADE arms, 20 steps, seed 42, confined for the
zones). Recorded per case:

* ``n_forward`` / ``n_backward``: network forward passes and input-gradient
  evaluations the attack itself made (the nominal cost; evaluation forwards
  are excluded),
* ``seconds``: wall-clock of the attack calls (CUDA-synchronised),
* ``peak_gb``: peak allocated CUDA memory during the attack.

Rows are appended to ``attack_cost_per_case.csv`` as they finish (rerunning
skips rows already there), and ``summarize()`` writes the mean tables.

Usage::

    PYTHONPATH=src python experiments/benchmark_attack_cost.py --cases 5
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
from fgsm_adversarial_evaluation import (  # noqa: E402
    ConfinedInput,
    _make_eval_config,
    attackable_mask,
    fgsm_attack,
    get_checkpoint_for_fold,
    load_model,
    load_sample,
    load_validation_ids,
    pad_to_divisible,
    samplewise_bounds,
)
from run_twelve_arm_wg_campaign import (  # noqa: E402
    ARM_ATTACK,
    ARM_LABELS,
    ARM_RESTARTS,
    ARMS,
    EPSILONS,
    LADDER_ARMS,
    N_STEPS,
    NUM_RESTARTS,
    SEED,
    TRAINERS,
)

ARCH_LABELS = {
    "unet": "nnU-Net", "resenc_m": "ResEnc-M", "resenc_l": "ResEnc-L", "resenc_xl": "ResEnc-XL",
}
TASK_LABELS = {"wg": "WG", "zones": "Zones"}
# Number of attacked cases in a full campaign (all five folds), for the
# GPU-hour extrapolation.
CAMPAIGN_CASES = {"wg": 1396, "zones": 603}
# The campaign driver's --random-start choice per arm (see the driver's main()):
# dag keeps a clean start, everything else starts randomly.
RANDOM_START = {arm: arm != "dag" for arm in ARMS}


class Counting(torch.nn.Module):
    """Counts forward passes and input-gradient backward passes of a network."""

    def __init__(self, network: torch.nn.Module) -> None:
        super().__init__()
        self.network = network
        self.n_forward = 0
        self.n_backward = 0

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        self.n_forward += 1
        out = self.network(x)
        if out.requires_grad:
            out.register_hook(self._count_backward)
        return out

    def _count_backward(self, grad: torch.Tensor) -> torch.Tensor:
        self.n_backward += 1
        return grad

    def reset(self) -> None:
        self.n_forward = self.n_backward = 0


def attack_fn_for(arm: str, task: str, cfg: dict):
    attack = ARM_ATTACK.get(arm, arm)
    fn = {
        "auto_pgd": pgd_mod.auto_pgd_with_restarts,
        "segpgd": pgd_mod.segpgd_with_restarts,
        "cospgd": pgd_mod.cospgd_with_restarts,
        "dag": pgd_mod.dag_with_restarts,
        "sea": pgd_mod.sea_with_restarts,
        "blade": pgd_mod.blade_with_restarts,
        "blade1": pgd_mod.blade1_with_restarts,
        "blade_mm": pgd_mod.blade_mm_with_restarts,
        "blade_boundary": pgd_mod.blade_boundary_with_restarts,
        "blade_frontier": pgd_mod.blade_frontier_with_restarts,
    }.get(attack, pgd_mod.pgd_with_restarts)
    if task == "zones" and attack in pgd_mod.BLADE_MC_ARMS:
        fn = functools.partial(fn, blade_impl="mc", spacing=pgd_mod._plan_spacing(cfg))
    return fn


def run_attack(arm, attacked, x, y, eps_list, num_classes, task, cfg):
    """One case's attack calls exactly as the campaign driver issues them."""
    if arm == "fgsm":
        for eps in eps_list:
            if eps > 0:
                fgsm_attack(attacked, x, y, eps, num_classes)
        return
    fn = attack_fn_for(arm, task, cfg)
    x_min, x_max = samplewise_bounds(x)
    common = dict(
        n_steps=N_STEPS, step_size=None, momentum=0.0, normalized_grad=False,
        x_bounds=(x_min, x_max), num_restarts=ARM_RESTARTS.get(arm, NUM_RESTARTS),
        loss_log=None, random_start=RANDOM_START[arm],
    )
    with torch.enable_grad():
        if arm in LADDER_ARMS:
            fn(attacked, x, y, max(eps_list), num_classes, snapshot_eps=eps_list, **common)
        else:
            for eps in eps_list:
                if eps > 0:
                    fn(attacked, x, y, eps, num_classes, snapshot_eps=[eps], **common)


def done_keys(path: Path) -> set[tuple]:
    if not path.is_file():
        return set()
    t = pd.read_csv(path)
    return set(zip(t["task"], t["arch"], t["arm"], t["case_id"]))


def benchmark(args: argparse.Namespace, per_case: Path) -> None:
    device = torch.device("cuda")
    eps_list = sorted(float(e) for e in EPSILONS)
    done = done_keys(per_case)
    write_header = not per_case.is_file()
    for task in args.models:
        cfg = _make_eval_config(task, None)
        cfg["epsilons"] = eps_list
        num_classes = cfg["num_classes"]
        confined = task == "zones"
        case_ids = load_validation_ids(cfg["splits_json"], args.fold)[: args.cases]
        for arch in args.archs:
            cfg_arch = {**cfg, "trainer": TRAINERS[arch]}
            ckpt = get_checkpoint_for_fold(cfg_arch, args.fold)
            network, div = load_model(ckpt, cfg["configuration"], num_classes, device)
            network.eval()
            counter = Counting(network)
            samples = []
            for case_id in case_ids:
                img, seg = load_sample(cfg["data_dir"], case_id)
                x = torch.from_numpy(img[np.newaxis]).to(device)
                y = torch.from_numpy(seg[np.newaxis].astype(np.float32)).to(device)
                xp, _ = pad_to_divisible(x, div)
                yp, _ = pad_to_divisible(y, div)
                mask = attackable_mask(y, div) if confined else None
                samples.append((case_id, xp, yp, mask))
            # warm-up: one gradient so cuDNN autotuning is not billed to an arm
            _, xp, yp, mask = samples[0]
            att = ConfinedInput(counter, xp, mask) if mask is not None else counter
            fgsm_attack(att, xp, yp, 0.02, num_classes)
            torch.cuda.synchronize()
            for arm in args.arms:
                for case_id, xp, yp, mask in samples:
                    if (task, arch, arm, case_id) in done:
                        continue
                    att = ConfinedInput(counter, xp, mask) if mask is not None else counter
                    torch.manual_seed(SEED)
                    counter.reset()
                    torch.cuda.reset_peak_memory_stats()
                    torch.cuda.synchronize()
                    t0 = time.perf_counter()
                    run_attack(arm, att, xp, yp, eps_list, num_classes, task, cfg_arch)
                    torch.cuda.synchronize()
                    row = dict(
                        task=task, arch=arch, arm=arm, case_id=case_id,
                        voxels=int(xp.numel()),
                        n_forward=counter.n_forward, n_backward=counter.n_backward,
                        seconds=round(time.perf_counter() - t0, 3),
                        peak_gb=round(torch.cuda.max_memory_allocated() / 1e9, 3),
                    )
                    with per_case.open("a", newline="") as fh:
                        w = csv.DictWriter(fh, fieldnames=list(row))
                        if write_header:
                            w.writeheader()
                            write_header = False
                        w.writerow(row)
                    print(f"{task} {arch} {arm} {case_id}: {row['seconds']} s, "
                          f"{row['n_backward']} grads, {row['peak_gb']} GB", flush=True)
            del network, counter, samples
            torch.cuda.empty_cache()


HIGHLIGHT = '<span style="color:blue">**{}**</span>'


def _wide(t: pd.DataFrame, value: str, fmt: str, lowest_best: bool = True) -> str:
    cols = [f"{ARCH_LABELS[a]} {TASK_LABELS[m]}" for m in TASK_LABELS for a in ARCH_LABELS]
    t = t.assign(col=t["arch"].map(ARCH_LABELS) + " " + t["task"].map(TASK_LABELS))
    wide = t.pivot(index="arm", columns="col", values=value).reindex(index=ARMS, columns=cols)
    wide = wide.dropna(how="all").dropna(axis=1, how="all")
    best = wide.idxmin() if lowest_best else wide.idxmax()
    out = wide.map(lambda v: fmt.format(v))
    for col, arm in best.items():
        out.loc[arm, col] = HIGHLIGHT.format(fmt.format(wide.loc[arm, col]))
    out.index = [ARM_LABELS[a] for a in out.index]
    return out.to_markdown()


def summarize(per_case: Path, out_dir: Path) -> None:
    raw = pd.read_csv(per_case)
    g = raw.groupby(["task", "arch", "arm"], as_index=False).agg(
        n_cases=("case_id", "count"), n_forward=("n_forward", "mean"),
        n_backward=("n_backward", "mean"), seconds=("seconds", "mean"),
        seconds_sd=("seconds", "std"), peak_gb=("peak_gb", "max"),
    )
    g["campaign_gpu_hours"] = g["seconds"] * g["task"].map(CAMPAIGN_CASES) / 3600
    g["arm_order"] = g["arm"].map({a: i for i, a in enumerate(ARMS)})
    g = g.sort_values(["task", "arch", "arm_order"]).drop(columns="arm_order")
    g.to_csv(out_dir / "attack_cost_summary.csv", index=False, float_format="%.3f")

    nominal = (
        raw.groupby("arm").agg(n_forward=("n_forward", "mean"), n_backward=("n_backward", "mean"))
        .reindex([a for a in ARMS if a in set(raw["arm"])])
    )
    same = raw.groupby("arm")[["n_forward", "n_backward"]].nunique().max().max() == 1
    nominal.index = [ARM_LABELS[a] for a in nominal.index]
    md = [
        "# Computational cost per attack and model",
        "",
        f"Sequential replay of the campaign attack call on the first {raw['case_id'].nunique()} "
        "validation cases of fold 0 (idle GPU, 20 steps, seed 42, native per-epsilon over "
        "eps = 0.02-0.1, ladders for the BLADE arms, zones confined). Evaluation forwards "
        "are not counted.",
        "",
        "## 1. Nominal cost: network passes per case (forward / input-gradient)",
        "",
        ("Identical for every model and case." if same else
         "Mean over models and cases (counts differ; see attack_cost_summary.csv)."),
        "",
        nominal.to_markdown(floatfmt=".0f"),
        "",
        "## 2. Measured wall-clock seconds per case (mean; lowest per column in blue)",
        "",
        _wide(g, "seconds", "{:.1f}"),
        "",
        "## 3. Peak CUDA memory per case, GB (max over cases; lowest per column in blue)",
        "",
        _wide(g, "peak_gb", "{:.1f}"),
        "",
        "## 4. Extrapolated GPU-hours for one full campaign tree "
        f"(seconds x {CAMPAIGN_CASES['wg']} WG / {CAMPAIGN_CASES['zones']} zones cases, "
        "single sequential worker)",
        "",
        _wide(g, "campaign_gpu_hours", "{:.1f}"),
        "",
    ]
    (out_dir / "attack_cost_tables.md").write_text("\n".join(md))
    print(f"Saved {out_dir / 'attack_cost_summary.csv'} and attack_cost_tables.md")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--models", nargs="+", default=["wg", "zones"], choices=["wg", "zones"])
    parser.add_argument("--archs", nargs="+", default=list(TRAINERS), choices=list(TRAINERS))
    parser.add_argument("--arms", nargs="+", default=list(ARMS), choices=list(ARMS))
    parser.add_argument("--cases", type=int, default=5)
    parser.add_argument("--fold", type=int, default=0)
    parser.add_argument(
        "--output-dir", type=Path, default=Path("results/blade_attack/computational_cost")
    )
    parser.add_argument("--summarize-only", action="store_true")
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    per_case = args.output_dir / "attack_cost_per_case.csv"
    if not args.summarize_only:
        benchmark(args, per_case)
    summarize(per_case, args.output_dir)


if __name__ == "__main__":
    main()
