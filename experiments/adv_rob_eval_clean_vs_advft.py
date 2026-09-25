"""
Adversarial robustness evaluation: clean vs. PGD-AT adversarially fine-tuned models.

Evaluates FGSM, PGD and AutoAttack at epsilon in {0, 2, 4, 8, 16, 32}/255 on every
clean-vs-PGD-AT model pair whose fold="all" checkpoints both exist, comparing the
clean and PGD-AT models trained under the same fold="all" regime (no held-out
split) so both sides see identical training data:

  - Dataset016 (WG, K=2)   : nnU-Net UNet / ResEnc-M / ResEnc-L / ResEnc-XL
  - Dataset019 (Zones, K=3): nnU-Net UNet / ResEnc-M / ResEnc-L / ResEnc-XL
    (ResEnc-XL PGD-AT not trained yet, so its pair is omitted)

The attacks are the multi-class PyTorch port of the seg3d-attacks suite
(``seg3d_attacks.pytorch``): faithful FGSM, PGD (Madry, multi-restart) and the
full 4-component AutoAttack ensemble (APGD-CE + APGD-margin + FAB-T + Square,
per-sample worst case by macro soft Dice). Model loading, data loading, the
per-fold evaluation loop and metrics are reused from
adv_rob_eval_nnunet_nnunetrecenc.py; outputs go to a dedicated directory so they
don't mix with the full architecture-comparison sweep.
"""

import argparse
import os
import sys

import torch

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
REPO_ROOT = os.path.dirname(SCRIPT_DIR)
sys.path.insert(0, SCRIPT_DIR)

# seg3d-attacks is a sibling repo, not pip-installed — put it on the path.
_SEG3D_DEFAULT = os.path.join(os.path.dirname(REPO_ROOT), "seg3d-attacks")
SEG3D_HOME = os.environ.get("SEG3D_ATTACKS_HOME", _SEG3D_DEFAULT)
if SEG3D_HOME not in sys.path:
    sys.path.insert(0, SEG3D_HOME)

from adv_rob_eval_nnunet_nnunetrecenc import run  # noqa: E402
from seg3d_attacks.pytorch import get_attack  # noqa: E402

# ε in n/255 units (L∞), the standard adversarial-robustness grid.
EPSILONS = [0.0, 2 / 255, 4 / 255, 8 / 255, 16 / 255, 32 / 255]

# (dataset_key, clean trainer key, PGD-AT trainer key)
COMPARISONS = [
    ("wg", "unet", "unet_advft"),
    ("wg", "resenc_m", "resenc_m_advft"),
    ("wg", "resenc_l", "resenc_l_advft"),
    ("wg", "resenc_xl", "resenc_xl_advft"),
    ("zones", "unet", "unet_advft"),
    ("zones", "resenc_m", "resenc_m_advft"),
    ("zones", "resenc_l", "resenc_l_advft"),
    ("zones", "resenc_xl", "resenc_xl_advft"),
]


def _adapt(attack):
    """Wrap a seg3d_attacks attack instance in the harness's attack signature.

    The harness calls ``fn(model, x, y, eps, num_classes) -> x_adv``; the seg3d
    attacks use ``attack.generate(model, x, eps, y=y)`` and infer the class count
    from the logits, so ``num_classes`` is unused here. ``y`` is the padded
    integer label map ``[1, 1, D, H, W]`` the harness already prepares.
    """

    def fn(model, x, y, eps, num_classes):
        return attack.generate(model, x, eps, y=y)

    return fn


# attack key (CLI / seg3d registry name) -> CSV/plot display name.
# "apgd" is Auto-PGD with the default objective="auto", which with a label y is
# cross-entropy — i.e. APGD-CE (the single white-box component, not the ensemble).
ATTACK_DISPLAY = {
    "fgsm": "FGSM",
    "pgd": "PGD",
    "apgd": "APGD-CE",
    "autoattack": "AutoAttack",
}

# Reduced AutoAttack: APGD-CE + APGD-margin only (drops FAB + Square, ~3× faster).
# Justified when validate_autoattack_components.py shows FAB/Square never lower
# the worst-case Dice on a case sample.
REDUCED_AUTOATTACK_COMPONENTS = ("apgd_ce", "apgd_m")


def build_attacks(names, reduced_autoattack=False):
    """Build the harness attack dict {display_name: adapter} for the given keys.

    ``autoattack`` is the full 4-component ensemble (seg3d-attacks defaults),
    or its reduced APGD-CE + APGD-margin subset when ``reduced_autoattack`` is
    set (the CSV/plot display name stays ``AutoAttack`` either way).
    """

    def make(name):
        if name == "autoattack" and reduced_autoattack:
            return get_attack("autoattack", components=REDUCED_AUTOATTACK_COMPONENTS)
        return get_attack(name)

    return {ATTACK_DISPLAY[n]: _adapt(make(n)) for n in names}


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Clean vs. PGD-AT adversarially fine-tuned robustness comparison"
    )
    parser.add_argument(
        "--output-dir", type=str, default="adv_rob_eval_results/clean_vs_advft"
    )
    parser.add_argument("--max-cases", type=int, default=None)
    parser.add_argument("--device", type=str, default=None)
    parser.add_argument(
        "--batch-size",
        type=int,
        default=1,
        help="cases attacked per GPU forward (throughput only; memory-bound)",
    )
    parser.add_argument(
        "--attacks",
        nargs="+",
        choices=list(ATTACK_DISPLAY),
        default=["autoattack"],
        help="attacks to evaluate (default: autoattack only)",
    )
    parser.add_argument(
        "--reduced-autoattack",
        action="store_true",
        help="run AutoAttack as APGD-CE + APGD-margin only (drop FAB + Square, "
        "~3× faster); validate first with validate_autoattack_components.py",
    )
    parser.add_argument(
        "--bf16",
        action="store_true",
        help="run attack generation under bf16 autocast (faster on GPU; metric "
        "forward stays fp32); validate drift with measure_bf16_autocast.py first",
    )
    args = parser.parse_args()

    device = (
        torch.device(args.device)
        if args.device
        else torch.device("cuda" if torch.cuda.is_available() else "cpu")
    )
    print(f"Device: {device}")

    attacks = build_attacks(args.attacks, reduced_autoattack=args.reduced_autoattack)
    aa_mode = (
        "reduced (APGD-CE+margin)" if args.reduced_autoattack else "full 4-component"
    )
    print(
        f"Attacks: {list(attacks)} | epsilons (·/255): {[round(e * 255) for e in EPSILONS]}"
    )
    if "autoattack" in args.attacks:
        print(f"AutoAttack: {aa_mode}")

    for dataset_key, clean_key, advft_key in COMPARISONS:
        run(
            dataset_key,
            clean_key,
            args,
            device,
            force_fold_all=True,
            attacks=attacks,
            epsilons=EPSILONS,
        )
        run(
            dataset_key,
            advft_key,
            args,
            device,
            force_fold_all=True,
            attacks=attacks,
            epsilons=EPSILONS,
        )

    print("\nAll clean-vs-advft evaluations complete.")


if __name__ == "__main__":
    main()
