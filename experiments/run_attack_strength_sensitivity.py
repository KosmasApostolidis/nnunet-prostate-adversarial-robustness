"""Attack-strength sensitivity sweep for Reviewer 1 question 4.

The reported campaign used PGD with 20 iterations and one trajectory started
from the clean image. The reviewer asked how the results change with 50-200
iterations and five or more restarts. This driver re-runs the same attack over
a fixed subsample, varying only the iteration count and the number of restarts,
so any change in Dice is attributable to attack strength alone.

Cases are drawn from every fold and attacked against that fold's own
checkpoint, matching the protocol used for Table I. Step size stays at
max_eps / steps, shared-trajectory snapshots stay on, and the seed is fixed,
exactly as in the reported campaign.

``--random-start`` is deliberately NOT passed: restart 0 then reproduces the
frozen campaign's trajectory and restarts 1..N-1 are randomized, so the
multi-restart arm searches a superset of the reported attack and can only be as
strong or stronger. Passing it would exclude the baseline trajectory from the
restart set and let a "stronger" configuration return a weaker example.
"""

from __future__ import annotations

import argparse
import subprocess
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
DRIVER = REPO_ROOT / "experiments" / "pgd_adversarial_evaluation.py"
EPSILONS = ["0.02", "0.04", "0.06", "0.08", "0.1"]
FOLDS = (0, 1, 2, 3, 4)

# (pgd_steps, num_restarts); the first entry reproduces the reported campaign.
CONFIGURATIONS = ((20, 1), (50, 1), (100, 1), (200, 1), (200, 5))


def run_one(
    model: str,
    fold: int,
    steps: int,
    restarts: int,
    cases: int,
    output_root: Path,
    env_path: str,
    deterministic: bool = False,
    log_loss_curve: bool = False,
    benchmark: bool = False,
    attack: str = "pgd",
) -> None:
    output_dir = output_root / f"steps{steps}_restarts{restarts}" / model
    output_dir.mkdir(parents=True, exist_ok=True)
    epsilon_flag = "--epsilons-wg" if model == "wg" else "--epsilons-zones"
    command = [
        sys.executable,
        str(DRIVER),
        "--attack", attack,
        "--model", model,
        "--fold", str(fold),
        "--max-samples", str(cases),
        "--pgd-steps", str(steps),
        "--num-restarts", str(restarts),
        "--shared-trajectory",
        "--seed", "42",
        epsilon_flag, *EPSILONS,
        "--output-dir", str(output_dir),
        "--resume",
    ]
    if deterministic:
        command.append("--deterministic")
    if log_loss_curve:
        command.append("--log-loss-curve")
    if benchmark:
        command.append("--benchmark")
    subprocess.run(
        command,
        check=True,
        cwd=REPO_ROOT,
        env={**dict(__import__("os").environ), "PYTHONPATH": env_path},
        stdout=subprocess.DEVNULL,
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cases-per-fold", type=int, default=8)
    parser.add_argument("--models", nargs="+", default=["wg", "zones"])
    parser.add_argument(
        "--output-root",
        type=Path,
        default=REPO_ROOT / "rebuttal_paper2" / "attack_strength_sensitivity",
    )
    parser.add_argument(
        "--deterministic",
        action="store_true",
        help="Run the driver with deterministic algorithms (reproducible, slower)",
    )
    parser.add_argument(
        "--log-loss-curve",
        action="store_true",
        help="Log the per-step attack loss so monotonicity can be checked",
    )
    parser.add_argument(
        "--benchmark",
        action="store_true",
        help="Enable cuDNN autotuning in the driver",
    )
    parser.add_argument(
        "--attack",
        choices=["pgd", "a_pgd"],
        default="pgd",
        help="Which attack to sweep. a_pgd is the manuscript's adaptive-step "
        "attack (Table I's APGD); give it its own --output-root.",
    )
    parser.add_argument(
        "--workers",
        type=int,
        default=1,
        help="Jobs to run concurrently. Each (arm, model, fold) job is "
        "independent and writes its own CSV, so concurrency changes throughput "
        "only, never results. Batch-1 3D inference leaves the GPU far from "
        "saturated, so 4-6 workers is typically several times faster.",
    )
    parser.add_argument(
        "--pythonpath",
        default="src",
        help="PYTHONPATH for the driver; append the shim directory if the "
        "legacy fgsm_adversarial_evaluation module is not on the path",
    )
    args = parser.parse_args()

    args.output_root.mkdir(parents=True, exist_ok=True)
    jobs = [
        (steps, restarts, model, fold)
        for steps, restarts in CONFIGURATIONS
        for model in args.models
        for fold in FOLDS
    ]
    total = len(jobs)
    # Arms differ by more than an order of magnitude in cost, so a job-count ETA
    # is meaningless. Weight each job by its forward/backward passes per case.
    weight = {job: job[0] * job[1] for job in jobs}
    total_weight = sum(weight.values())
    done_weight = 0.0
    started = time.time()
    workers = max(1, args.workers)
    print(f"{total} jobs, {workers} concurrent, attack={args.attack}", flush=True)

    def submit(job):
        steps, restarts, model, fold = job
        run_one(
            model,
            fold,
            steps,
            restarts,
            args.cases_per_fold,
            args.output_root,
            args.pythonpath,
            deterministic=args.deterministic,
            log_loss_curve=args.log_loss_curve,
            benchmark=args.benchmark,
            attack=args.attack,
        )
        return job

    failures = []
    with ThreadPoolExecutor(max_workers=workers) as pool:
        futures = {pool.submit(submit, job): job for job in jobs}
        for done, future in enumerate(as_completed(futures), start=1):
            steps, restarts, model, fold = futures[future]
            label = f"steps={steps} restarts={restarts} {model} fold{fold}"
            try:
                future.result()
            except Exception as exc:  # noqa: BLE001 - report and keep going
                failures.append((label, exc))
                print(f"[{done}/{total}] FAILED {label}: {exc}", flush=True)
                continue
            done_weight += weight[futures[future]]
            elapsed = (time.time() - started) / 60.0
            remaining = elapsed * (total_weight - done_weight) / max(done_weight, 1e-9)
            print(
                f"[{done}/{total}] {label}  elapsed={elapsed:.1f} min  "
                f"work={100 * done_weight / total_weight:.0f}%  "
                f"eta={remaining:.0f} min",
                flush=True,
            )

    if failures:
        print(f"sweep finished with {len(failures)} failed jobs:", flush=True)
        for label, exc in failures:
            print(f"  {label}: {exc}", flush=True)
        raise SystemExit(1)
    print("sweep complete", flush=True)


if __name__ == "__main__":
    main()
