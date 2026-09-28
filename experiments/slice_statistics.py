"""Recompute every slice-level statistic of the revised manuscript (FGSM vs noise).

Reads the per-fold per-slice CSVs written by ``slice_vulnerability_analysis.py``
(``<root>/<arm>/<model>_fold<k>/<model>_fold<k>_per_slice_<arm>.csv``) and the
prediction-flip CSVs written by ``prediction_flip_analysis.py``
(``<flip-root>/<arm>/<model>_fold<k>_prediction_flip.csv``), and writes, at one
epsilon:

(a) ``fe_dice_drop.csv``: patient fixed-effects OLS of per-slice Delta-Dice on
    region (base, apex vs mid), a cubic polynomial in log area and clean Dice,
    with a patient-cluster bootstrap;
(b) ``fe_fgsm_minus_noise.csv``: the same model on the noise Delta-Dice and on
    the paired FGSM - noise excess;
(c) ``matched_excess_contrasts.csv``: area-quartile Base-Mid / Apex-Mid
    contrasts on that excess;
(d) ``label_source_regional.csv`` and ``label_source_matched_contrasts.csv``:
    the WG split into PI-CAI and manually labelled in-house cases;
(e) ``fe_asd_change.csv``: the adjusted per-slice ASD change (mm), slices with
    an empty prediction (undefined ASD) excluded;
(f) ``flip_regional.csv`` and ``fe_prediction_flip.csv``: prediction flip
    (1 - flip Dice), raw regional means and the adjusted model;
(g) ``quartile_area_balance.csv``: base vs mid area and clean Dice within each
    area quartile.

The ``*_all_folds_*`` files are never read: they hold only the last fold.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))
sys.path.insert(0, str(REPO_ROOT / "experiments"))

from plot_fig2_area_stratified import (  # noqa: E402
    N_QUARTILES,
    SEED,
    _stratified_contrasts,
)

MODELS = ("wg", "zones")
CLASSES = ("WG", "TZ+CZ", "PZ")
REGIONS = ("Base", "Mid", "Apex")
N_FOLDS = 5
# Both models are 0.5 x 0.5 mm in-plane (3d_fullres plans); metrics.slice_asd is in pixels.
MM_PER_PIXEL = 0.5
MM2_PER_PIXEL = MM_PER_PIXEL * MM_PER_PIXEL
N_BOOT = 1000
PICAI_ID = r"ProstateWG_\d{5}"
PAIR_KEY = ["class", "case_id", "slice_idx"]
REPORTED_TERMS = ("base", "apex", "clean")


def label_source(case_ids: pd.Series) -> pd.Series:
    """PI-CAI for ``ProstateWG_<5 digits>``, otherwise the manually labelled in-house cohort."""
    is_picai = case_ids.astype(str).str.fullmatch(PICAI_ID)
    return pd.Series(np.where(is_picai, "PI-CAI", "in-house"), index=case_ids.index)


def _prepare(frame: pd.DataFrame, eps: float) -> pd.DataFrame:
    frame = frame[np.isclose(frame["epsilon"], eps) & (frame["gt_area"] > 0)].copy()
    frame["area_mm2"] = frame["gt_area"] * MM2_PER_PIXEL
    if "asd_change" in frame:
        frame["asd_change_mm"] = frame["asd_change"] * MM_PER_PIXEL
    frame["source"] = label_source(frame["case_id"])
    return frame


def _read_folds(paths: list[Path], expected: int, what: str) -> pd.DataFrame:
    if len(paths) != expected:
        raise FileNotFoundError(
            f"{what}: expected {expected} fold files, found {len(paths)}"
        )
    return pd.concat([pd.read_csv(p) for p in paths], ignore_index=True)


def load_slices(root: Path, arm: str, eps: float) -> pd.DataFrame:
    """Per-fold per-slice rows of both models at ``eps``, foreground slices only."""
    paths = [
        p
        for model in MODELS
        for p in sorted(
            (root / arm).glob(f"{model}_fold*/{model}_fold*_per_slice_{arm}.csv")
        )
    ]
    return _prepare(_read_folds(paths, N_FOLDS * len(MODELS), f"{root / arm}"), eps)


def load_flip(flip_root: Path, arm: str, eps: float) -> pd.DataFrame:
    """Per-fold prediction-flip rows at ``eps`` with ``flip = 1 - flip_dice``."""
    paths = sorted((flip_root / arm).glob("*_prediction_flip.csv"))
    frame = _prepare(
        _read_folds(paths, N_FOLDS * len(MODELS), f"{flip_root / arm}"), eps
    )
    frame["flip"] = 1.0 - frame["flip_dice"]
    return frame


def paired_excess(
    adversarial: pd.DataFrame, noise: pd.DataFrame, value: str
) -> pd.DataFrame:
    """Pair each adversarial slice with its noise slice; ``excess = adv - noise``.

    Raises ``ValueError`` on duplicated slices or on slices present in one arm only.
    """
    noise_col = f"noise_{value}"
    merged = adversarial.merge(
        noise[PAIR_KEY + [value]].rename(columns={value: noise_col}),
        on=PAIR_KEY,
        how="inner",
        validate="one_to_one",
    )
    if not len(merged) == len(adversarial) == len(noise):
        raise ValueError(
            f"unpaired slices: adversarial={len(adversarial)} noise={len(noise)} "
            f"paired={len(merged)}"
        )
    merged["excess"] = merged[value] - merged[noise_col]
    return merged


def fixed_effects_ols(
    frame: pd.DataFrame, outcome: str, n_boot: int = N_BOOT, seed: int = SEED
) -> pd.DataFrame:
    """Patient fixed-effects OLS with a patient-cluster bootstrap.

    Regressors: base and apex indicators (mid is the reference), log area, its
    square and cube, and clean Dice. The fixed effects are absorbed by demeaning
    within patient; per-patient cross-products make each bootstrap draw a
    weighted sum. Returns the base, apex and clean coefficients with 95% CIs.
    """
    log_area = np.log(frame["area_mm2"])
    design = pd.DataFrame(
        {
            "base": (frame["anatomical_region"] == "Base").astype(float),
            "apex": (frame["anatomical_region"] == "Apex").astype(float),
            "la": log_area,
            "la2": log_area**2,
            "la3": log_area**3,
            "clean": frame["dice_clean"],
        },
        index=frame.index,
    )
    patients = frame["case_id"]
    x = (design - design.groupby(patients).transform("mean")).to_numpy()
    y_raw = frame[outcome]
    y = (y_raw - y_raw.groupby(patients).transform("mean")).to_numpy()
    codes, unique = pd.factorize(patients)
    n_patients, k = len(unique), design.shape[1]
    xtx = np.zeros((n_patients, k, k))
    xty = np.zeros((n_patients, k))
    np.add.at(xtx, codes, x[:, :, None] * x[:, None, :])
    np.add.at(xty, codes, x * y[:, None])
    point = np.linalg.solve(xtx.sum(0), xty.sum(0))
    rng = np.random.default_rng(seed)
    draws = np.empty((n_boot, k))
    for b in range(n_boot):
        weights = np.bincount(
            rng.integers(0, n_patients, n_patients), minlength=n_patients
        )
        draws[b] = np.linalg.solve(np.tensordot(weights, xtx, 1), weights @ xty)
    rows = [
        {
            "term": name,
            "estimate": point[j],
            "ci95_low": np.percentile(draws[:, j], 2.5),
            "ci95_high": np.percentile(draws[:, j], 97.5),
        }
        for j, name in enumerate(design.columns)
        if name in REPORTED_TERMS
    ]
    result = pd.DataFrame(rows)
    result["n_patients"] = n_patients
    result["n_slices"] = len(frame)
    return result


def groups(frame: pd.DataFrame) -> list[tuple[str, pd.DataFrame]]:
    """Each class, then the WG split by label source."""
    out = [(cls, frame[frame["class"] == cls]) for cls in CLASSES]
    wg = frame[frame["class"] == "WG"]
    out += [(f"WG {src}", wg[wg["source"] == src]) for src in ("in-house", "PI-CAI")]
    return [(name, g) for name, g in out if not g.empty]


def add_area_quartile(frame: pd.DataFrame) -> pd.DataFrame:
    """Class-specific area quartiles over all foreground slices of the class."""
    frame = frame.copy()
    frame["area_quartile"] = -1
    for _, g in frame.groupby("class"):
        frame.loc[g.index, "area_quartile"] = pd.qcut(
            g["area_mm2"], N_QUARTILES, labels=False
        )
    return frame


def matched_contrasts(frame: pd.DataFrame, value: str) -> pd.DataFrame:
    """Within-quartile paired Base-Mid and Apex-Mid differences of ``value``."""
    subset = frame.assign(patient_id=frame["case_id"], dice_drop=frame[value])
    stats = _stratified_contrasts(subset, np.random.default_rng(SEED))
    return pd.DataFrame(
        [
            {
                "area_quartile": f"Q{q + 1}",
                "term": term,
                "difference": stats["point"][r, q],
                "ci95_low": stats["low"][r, q],
                "ci95_high": stats["high"][r, q],
                "n_patients": int(stats["n_patients"][r, q]),
            }
            for q in range(N_QUARTILES)
            for r, term in enumerate(("Base_vs_Mid", "Apex_vs_Mid"))
        ]
    )


def regional_patient_means(frame: pd.DataFrame, value: str) -> pd.DataFrame:
    """Mean over patients of each patient's mean ``value`` per region."""
    per_patient = (
        frame.groupby(["case_id", "anatomical_region"])[value].mean().unstack()
    )
    return pd.DataFrame(
        [
            {
                "region": region,
                value: per_patient[region].mean(),
                "n_patients": int(per_patient[region].notna().sum()),
            }
            for region in REGIONS
            if region in per_patient
        ]
    )


def quartile_area_balance(frame: pd.DataFrame) -> pd.DataFrame:
    """Base vs mid area and clean Dice per class and area quartile."""
    rows = []
    for (cls, q), g in frame.groupby(["class", "area_quartile"]):
        base = g[g["anatomical_region"] == "Base"]
        mid = g[g["anatomical_region"] == "Mid"]
        area_b = base.groupby("case_id")["area_mm2"].mean()
        area_m = mid.groupby("case_id")["area_mm2"].mean()
        both = area_b.index.intersection(area_m.index)
        rows.append(
            {
                "class": cls,
                "area_quartile": f"Q{int(q) + 1}",
                "median_area_base_mm2": base["area_mm2"].median(),
                "median_area_mid_mm2": mid["area_mm2"].median(),
                "paired_mean_area_base_mm2": area_b[both].mean(),
                "paired_mean_area_mid_mm2": area_m[both].mean(),
                "paired_area_gap_pct": 100
                * (1 - area_b[both].mean() / area_m[both].mean()),
                "paired_clean_base": base.groupby("case_id")["dice_clean"]
                .mean()[both]
                .mean(),
                "paired_clean_mid": mid.groupby("case_id")["dice_clean"]
                .mean()[both]
                .mean(),
                "n_pairs": len(both),
            }
        )
    return pd.DataFrame(rows)


def _stack(parts: list[tuple[dict, pd.DataFrame]]) -> pd.DataFrame:
    return pd.concat(
        [
            table.assign(**labels)[list(labels) + list(table.columns)]
            for labels, table in parts
        ],
        ignore_index=True,
    )


def dice_tables(fgsm: pd.DataFrame, noise: pd.DataFrame) -> dict[str, pd.DataFrame]:
    """Sections (a)-(e) and (g) from the per-slice Delta-Dice/ASD rows."""
    fgsm = add_area_quartile(fgsm)
    paired = paired_excess(fgsm, noise, "dice_drop")
    wg = fgsm[fgsm["class"] == "WG"]
    asd = fgsm.dropna(subset=["asd_change_mm"])
    return {
        "fe_dice_drop": _stack(
            [({"group": n}, fixed_effects_ols(g, "dice_drop")) for n, g in groups(fgsm)]
        ),
        "fe_fgsm_minus_noise": _stack(
            [
                ({"group": n, "outcome": y}, fixed_effects_ols(g, y))
                for n, g in groups(paired)
                for y in ("noise_dice_drop", "excess")
            ]
        ),
        "matched_excess_contrasts": _stack(
            [
                (
                    {"group": cls},
                    matched_contrasts(paired[paired["class"] == cls], "excess"),
                )
                for cls in CLASSES
            ]
        ),
        "label_source_regional": _stack(
            [
                (
                    {"source": src, "metric": v},
                    regional_patient_means(g, v).rename(columns={v: "mean"}),
                )
                for src, g in wg.groupby("source")
                for v in ("dice_drop", "dice_clean")
            ]
        ),
        "label_source_matched_contrasts": _stack(
            [
                ({"source": src}, matched_contrasts(g, "dice_drop"))
                for src, g in wg.groupby("source")
            ]
        ),
        "fe_asd_change": _stack(
            [
                (
                    {
                        "group": n,
                        "n_excluded_empty_prediction": int(
                            (fgsm["class"] == n).sum() - len(g)
                        ),
                    },
                    fixed_effects_ols(g, "asd_change_mm"),
                )
                for n, g in groups(asd)
                if n in CLASSES
            ]
        ),
        "quartile_area_balance": quartile_area_balance(fgsm),
    }


def flip_tables(fgsm: pd.DataFrame, noise: pd.DataFrame) -> dict[str, pd.DataFrame]:
    """Section (f): raw regional prediction flip and the adjusted model."""
    paired = paired_excess(fgsm, noise, "flip")
    regional = _stack(
        [
            (
                {"group": n},
                g.groupby("anatomical_region")[["flip", "noise_flip", "excess"]]
                .mean()
                .reindex(list(REGIONS))
                .rename_axis("region")
                .reset_index()
                .assign(
                    n_slices=g.groupby("anatomical_region")
                    .size()
                    .reindex(list(REGIONS))
                    .to_numpy()
                ),
            )
            for n, g in groups(paired)
        ]
    )
    adjusted = _stack(
        [
            ({"group": n, "outcome": y}, fixed_effects_ols(g, y))
            for n, g in groups(paired)
            for y in ("flip", "noise_flip", "excess")
        ]
    )
    return {"flip_regional": regional, "fe_prediction_flip": adjusted}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--root", type=Path, default=REPO_ROOT / "results" / "camera_ready_assd")
    parser.add_argument(
        "--flip-root", type=Path, default=REPO_ROOT / "results" / "prediction_flip"
    )
    parser.add_argument("--eps", type=float, default=0.1)
    parser.add_argument(
        "--out", type=Path, default=None, help="default: <root>/analysis"
    )
    args = parser.parse_args()
    out = args.out or args.root / "analysis"
    out.mkdir(parents=True, exist_ok=True)
    tables = dice_tables(
        load_slices(args.root, "fgsm", args.eps),
        load_slices(args.root, "noise", args.eps),
    )
    tables.update(
        flip_tables(
            load_flip(args.flip_root, "fgsm", args.eps),
            load_flip(args.flip_root, "noise", args.eps),
        )
    )
    for name, table in tables.items():
        path = out / f"slice_statistics_{name}.csv"
        table.insert(0, "epsilon", args.eps)
        table.to_csv(path, index=False)
        print(f"  {path}")


if __name__ == "__main__":
    main()
