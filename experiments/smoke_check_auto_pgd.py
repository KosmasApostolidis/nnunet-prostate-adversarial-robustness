"""Four-case gate before the full Auto-PGD campaign.

Reproduces the probe recorded in ``a_pgd_campaign``'s docstring (fold-0, four
cases, eps = 0.1: WG Dice 0.518 for the campaign APGD, 0.568 for plain PGD) and
adds the new Auto-PGD arm alongside. If the two known arms do not reproduce,
the harness has drifted and the campaign must not be started.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import torch

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))
sys.path.insert(0, str(REPO_ROOT / "experiments"))

from fgsm_adversarial_evaluation import (  # noqa: E402
    compute_metrics,
    dice_ce_loss,
    load_model,
    load_sample,
    load_validation_ids,
    pad_to_divisible,
    samplewise_bounds,
    unpad,
)
from mri_prostate_seg.attacks.a_pgd_campaign import a_pgd_trajectory  # noqa: E402
from mri_prostate_seg.attacks.auto_pgd import auto_pgd_trajectory  # noqa: E402
from mri_prostate_seg.attacks.config_loader import load_model_config  # noqa: E402
from mri_prostate_seg.attacks.pgd import pgd_trajectory  # noqa: E402

EPS = 0.1
N_STEPS = 20
N_CASES = 4
FOLD = 0
# From the a_pgd_campaign docstring: fold-0, four cases, eps = 0.1, WG Dice.
EXPECTED = {"pgd": 0.568, "a_pgd": 0.518}
# Wide enough to absorb random-start jitter on a four-case mean, tight enough
# that a wrong checkpoint, wrong split or wrong attack (all off by 0.2 or more)
# still trips it. Per-case Dice spread under these attacks is about 0.16, so a
# four-case mean carries roughly 0.08 of sampling noise across *different*
# cases; this gate compares the *same* four cases and varies only the random
# start, which the campaign's duplicate runs showed moves per-case Dice by at
# most 0.011.
TOLERANCE = 0.05


def _attack(name, network, x, y, num_classes, x_bounds):
    if name == "pgd":
        adv, _ = pgd_trajectory(
            network,
            x,
            y,
            EPS,
            num_classes,
            dice_ce_loss,
            n_steps=N_STEPS,
            random_start=True,
            x_bounds=x_bounds,
        )
        return adv
    if name == "a_pgd":
        adv, _ = a_pgd_trajectory(
            network,
            x,
            y,
            EPS,
            num_classes,
            dice_ce_loss,
            n_steps=N_STEPS,
            random_start=True,
            momentum=0.9,
            x_bounds=x_bounds,
            adaptation_rate=0.1,
            checkpoints=[0.25, 0.5, 0.75],
        )
        return adv
    adv, _ = auto_pgd_trajectory(
        network,
        x,
        y,
        EPS,
        num_classes,
        dice_ce_loss,
        n_steps=N_STEPS,
        random_start=True,
        x_bounds=x_bounds,
    )
    return adv


def main() -> None:
    torch.manual_seed(42)
    np.random.seed(42)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model_cfg = load_model_config("wg")
    cfg = {
        "configuration": model_cfg.configuration,
        "num_classes": model_cfg.num_classes,
        "data_dir": model_cfg.data_dir,
        "splits_json": model_cfg.splits_json,
    }
    network, div_factors = load_model(
        model_cfg.checkpoint_path(FOLD),
        cfg["configuration"],
        cfg["num_classes"],
        device,
    )
    network.eval()
    case_ids = load_validation_ids(cfg["splits_json"], FOLD)[:N_CASES]

    scores: dict[str, list[float]] = {"pgd": [], "a_pgd": [], "auto_pgd": []}
    for case_id in case_ids:
        img_np, seg_np = load_sample(cfg["data_dir"], case_id)
        seg_np = np.asarray(seg_np, dtype=np.int64)
        x = torch.from_numpy(img_np[np.newaxis]).to(device)
        y = torch.from_numpy(seg_np[np.newaxis].astype(np.float32)).to(device)
        x_bounds = samplewise_bounds(x)
        x_pad, orig_shape = pad_to_divisible(x, div_factors)
        y_pad, _ = pad_to_divisible(y, div_factors)
        for name in scores:
            with torch.enable_grad():
                adv = _attack(name, network, x_pad, y_pad, cfg["num_classes"], x_bounds)
            # Mirrors experiments/pgd_adversarial_evaluation.py:686-692 exactly:
            # compute_metrics takes unpadded LOGITS, a [1, 1, D, H, W] target and
            # the class count, and returns {class: (dice, hd95, asd)} tuples.
            with torch.no_grad():
                logits = unpad(network(adv).float(), orig_shape)
            dice_val, _, _ = compute_metrics(
                logits, seg_np[np.newaxis], cfg["num_classes"]
            )[1]
            scores[name].append(float(dice_val))
        print(f"{case_id}: " + "  ".join(f"{k}={v[-1]:.3f}" for k, v in scores.items()))

    means = {k: float(np.mean(v)) for k, v in scores.items()}
    print("\nmeans:", json.dumps(means, indent=2))

    out_dir = REPO_ROOT / "results" / "autopgd_comparison"
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "smoke_fold0.json").write_text(json.dumps(means, indent=2))

    failures = [
        f"{k}: got {means[k]:.3f}, expected {want:.3f} +/- {TOLERANCE}"
        for k, want in EXPECTED.items()
        if abs(means[k] - want) > TOLERANCE
    ]
    if failures:
        raise SystemExit(
            "Harness drift, do not start the campaign:\n  " + "\n  ".join(failures)
        )
    print(
        "\nKnown arms reproduce. Auto-PGD mean Dice at eps=0.1: "
        f"{means['auto_pgd']:.3f}"
    )


if __name__ == "__main__":
    main()
