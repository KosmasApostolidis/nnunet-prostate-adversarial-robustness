"""Revision (Tier-2 reanalysis) evidence for ``paper_resenc/manuscript.tex``.

Re-analyses existing per-sample outputs only (no attack runs, no training).
Mirrors ``analyze_resenc_paper_evidence.py``: the same per-fold loaders,
budget-averaged per-case metric (``mean_over_epsilons`` / ``arch_matrix``),
the same Hodges-Lehmann estimator and BH correction, and seed 42 with 2,000
resamples for every bootstrap. Bootstraps here resample cases *within fold*
(``paired_contrasts.stratified_bootstrap_indices``), or folds then cases
(hierarchical), and report percentile intervals.

Items (roadmap IDs): E02, E27/E28, E20, E21, E22, E18, E08, E23, E24, E14,
E19. Outputs go to ``--output-dir`` (default
``results/blade_attack/attack_comparison/stats_no_blade/revision``), one CSV
per item prefixed with its ID.

Usage::

    PYTHONPATH=src python experiments/analyze_resenc_revision_evidence.py
"""

from __future__ import annotations

import argparse
import os
import pickle
import re
import sys
from functools import lru_cache

import numpy as np
import pandas as pd
from scipy import stats

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import analyze_resenc_paper_evidence as arpe  # noqa: E402
from analyze_attack_architecture_stats import (  # noqa: E402
    ARCHS,
    CLASS_TAG,
    RESULTS,
    TREES,
    arch_matrix,
)
from analyze_main_comparison_stats import (  # noqa: E402
    ALPHA,
    _fdr_bh,
    _hodges_lehmann,
    build_matrix,
)

from mri_prostate_seg.experiments import paired_contrasts as pc  # noqa: E402

ARMS8 = arpe.ARMS8
NON_DAG = [a for a in ARMS8 if a != "dag"]
EPSILONS = arpe.EPSILONS
FOLDS = arpe.FOLDS
N_BOOT = 2000
SEED = 42
STATS_DIR = os.path.join(RESULTS, "attack_comparison", "stats_no_blade")
PREPROCESSED = {
    "WG": "nnUnet_paths/nnUNet_preprocessed/Dataset016_WgSegmentationPNetAndPicai/"
    "nnUNetPlans_3d_fullres",
    "zones": "nnUnet_paths/nnUNet_preprocessed/"
    "Dataset019_ProstateZonesSegmentationWgFilteredLessDilated/nnUNetPlans_3d_fullres",
}
# PI-CAI patient IDs are five-digit integers (10000-11475); the in-house WG
# cases carry long DICOM-UID-derived numeric IDs. Verified to split 951/445.
PICAI_ID = re.compile(r"ProstateWG_\d{5}")

Frames = dict[str, dict[str, dict[str, pd.DataFrame]]]


# --------------------------------------------------------------------------- data
def load_pooled_frames() -> Frames:
    """Per-sample frames per (class, arch, arm), all folds, with a ``fold`` column."""
    by_fold = arpe.load_frames_by_fold(ARMS8)
    return {
        c: {
            a: {
                arm: pd.concat(
                    [by_fold[f][c][a][arm].assign(fold=f) for f in FOLDS],
                    ignore_index=True,
                )
                for arm in ARMS8
            }
            for a in ARCHS
        }
        for c in TREES
    }


def case_fold_map(frames: Frames, class_name: str) -> pd.Series:
    """case_id -> fold, checked identical across the four architectures."""
    maps = []
    for arch in ARCHS:
        df = frames[class_name][arch]["pgd"]
        maps.append(
            df.drop_duplicates("case_id").set_index("case_id")["fold"].sort_index()
        )
    for other in maps[1:]:
        if not maps[0].equals(other):
            raise ValueError(
                f"{class_name}: fold assignment differs across architectures"
            )
    return maps[0]


class Data:
    """Frames plus cached case x architecture matrices and fold labels."""

    def __init__(self, frames: Frames) -> None:
        self.frames = frames
        self.folds = {c: case_fold_map(frames, c) for c in TREES}

    @lru_cache(maxsize=None)
    def mat(
        self, c: str, arm: str, metric: str, eps: float | None = None
    ) -> pd.DataFrame:
        """Budget-averaged (``eps=None``) or per-epsilon case x arch matrix."""
        return arch_matrix(self.frames[c], arm, metric, c, eps)

    @lru_cache(maxsize=None)
    def clean(self, c: str) -> pd.DataFrame:
        cols = {
            a: build_matrix({"pgd": self.frames[c][a]["pgd"]}, "dice", 0.0, ["pgd"], c)[
                "pgd"
            ]
            for a in ARCHS
        }
        return pd.DataFrame(cols).dropna()

    def fold_labels(self, c: str, index: pd.Index) -> np.ndarray:
        return self.folds[c].loc[index].to_numpy()


# ------------------------------------------------------------------------ helpers
def wilcoxon_p(diff: np.ndarray) -> float:
    nz = diff[np.isfinite(diff) & (diff != 0)]
    return float("nan") if nz.size == 0 else float(stats.wilcoxon(nz)[1])


def boot_mean(diff: np.ndarray, idx: np.ndarray | list[np.ndarray]) -> dict[str, float]:
    lo, hi = pc.percentile_ci(pc.resampled_means(diff, idx))
    return {"mean": float(np.mean(diff)), "ci_lo": lo, "ci_hi": hi}


def steps_with_ci(
    mat: pd.DataFrame, idx: np.ndarray | list[np.ndarray], prefix: str = ""
) -> dict[str, float]:
    """Encoder (M-nnU), batch (XL-M, L-M), total (XL-nnU) mean steps and the
    ratio-of-means batch/encoder shares, each with a percentile bootstrap CI."""
    u, m, l_, x = (mat[a].to_numpy() for a in ARCHS)
    diffs = {"encoder": m - u, "batch_xl": x - m, "batch_l": l_ - m, "total": x - u}
    row: dict[str, float] = {}
    boots = {}
    for name, d in diffs.items():
        b = boot_mean(d, idx)
        boots[name] = pc.resampled_means(d, idx)
        for k, v in b.items():
            row[f"{prefix}{name}_{k}"] = v
    for name in ("batch_xl", "encoder"):
        row[f"{prefix}share_{name}"] = pc.ratio_of_means(diffs[name], diffs["total"])
        lo, hi = pc.percentile_ci(boots[name] / boots["total"])
        row[f"{prefix}share_{name}_ci_lo"] = lo
        row[f"{prefix}share_{name}_ci_hi"] = hi
    return row


def strat_idx(data: Data, c: str, index: pd.Index) -> np.ndarray:
    return pc.stratified_bootstrap_indices(data.fold_labels(c, index), N_BOOT, SEED)


def hl_share_row(mat: pd.DataFrame) -> dict[str, float]:
    u, m, x = (
        mat["unet"].to_numpy(),
        mat["resenc_m"].to_numpy(),
        mat["resenc_xl"].to_numpy(),
    )
    hl_enc = _hodges_lehmann(m - u)[0]
    hl_batch = _hodges_lehmann(x - m)[0]
    hl_tot = _hodges_lehmann(x - u)[0]
    return {
        "hl_encoder": hl_enc,
        "hl_batch_xl": hl_batch,
        "hl_total": hl_tot,
        "hl_share_encoder": arpe.share_of_total(hl_enc, hl_tot),
        "hl_share_batch_xl": arpe.share_of_total(hl_batch, hl_tot),
    }


def vs_unet_family(mat: pd.DataFrame) -> list[dict[str, float]]:
    """HL + Wilcoxon for each ResEnc minus nnU-Net, BH within the 3-family."""
    rows = []
    for arch in ARCHS[1:]:
        diff = mat[arch].to_numpy() - mat["unet"].to_numpy()
        hl, lo, hi = _hodges_lehmann(diff)
        rows.append(
            {
                "comparator": arch,
                "n_cases": len(mat),
                "hl_shift": hl,
                "hl_ci_lo": lo,
                "hl_ci_hi": hi,
                "p_value": wilcoxon_p(diff),
            }
        )
    p_adj = _fdr_bh(np.array([r["p_value"] for r in rows]))
    for r, p in zip(rows, p_adj):
        r["p_adj_bh"] = float(p)
        r["significant"] = bool(p < ALPHA)
    return rows


# -------------------------------------------------------------------------- items
def e02_decomposition(data: Data) -> pd.DataFrame:
    rows = []
    for c in TREES:
        for arm in ARMS8:
            mat = data.mat(c, arm, "dice")
            u, m, x = (mat[a].to_numpy() for a in ("unet", "resenc_m", "resenc_xl"))
            dec = pc.additive_decomposition(u, m, x)
            contrast = (x - m) - (m - u)
            hl, lo, hi = _hodges_lehmann(contrast)
            rows.append(
                {
                    "class_name": c,
                    "arm": arm,
                    "n_cases": len(mat),
                    "additive_check": dec["mean_first_step"]
                    + dec["mean_second_step"]
                    - dec["mean_total"],
                    **steps_with_ci(mat, strat_idx(data, c, mat.index)),
                    "contrast_hl": hl,
                    "contrast_hl_ci_lo": lo,
                    "contrast_hl_ci_hi": hi,
                    "contrast_wilcoxon_p": wilcoxon_p(contrast),
                    **hl_share_row(mat),
                }
            )
    return pd.DataFrame(rows)


def e27_per_epsilon(data: Data) -> pd.DataFrame:
    rows = []
    for c in TREES:
        for arm in ARMS8:
            for eps in EPSILONS:
                mat = data.mat(c, arm, "dice", eps)
                rows.append(
                    {
                        "class_name": c,
                        "arm": arm,
                        "epsilon": eps,
                        "n_cases": len(mat),
                        **steps_with_ci(mat, strat_idx(data, c, mat.index)),
                    }
                )
    return pd.DataFrame(rows)


# E20 ---------------------------------------------------------------------------
def undefined_counts(data: Data) -> pd.DataFrame:
    rows = []
    for c in TREES:
        for arch in ARCHS:
            for arm in ARMS8:
                df = data.frames[c][arch][arm]
                df = df[df["class"] == c]
                for eps in [0.0, *EPSILONS]:
                    sub = df[np.isclose(df["epsilon"], eps)]
                    rows.append(
                        {
                            "class_name": c,
                            "arch": arch,
                            "arm": arm,
                            "epsilon": eps,
                            "n_rows": len(sub),
                            "n_hd95_undefined": int(sub["hd95"].isna().sum()),
                            "n_asd_undefined": int(sub["asd"].isna().sum()),
                            "n_undefined_with_dice0": int(
                                (sub["hd95"].isna() & (sub["dice"] == 0)).sum()
                            ),
                        }
                    )
    return pd.DataFrame(rows)


def image_diagonal_mm(class_name: str, case_id: str) -> float:
    """Physical diagonal of the case's original image grid (shape x spacing)."""
    key = "WG" if class_name == "WG" else "zones"
    with open(os.path.join(PREPROCESSED[key], f"{case_id}.pkl"), "rb") as f:
        props = pickle.load(f)
    extent = np.asarray(props["shape_before_cropping"]) * np.asarray(props["spacing"])
    return float(np.sqrt(np.sum(extent**2)))


def per_eps_pivot(data: Data, c: str, arch: str, arm: str, metric: str) -> pd.DataFrame:
    df = data.frames[c][arch][arm]
    df = df[(df["class"] == c) & df["epsilon"].round(4).isin(EPSILONS)]
    return df.pivot(index="case_id", columns="epsilon", values=metric)


def distance_matrix(
    data: Data, c: str, arm: str, metric: str, variant: str, diag: dict[str, float]
) -> pd.DataFrame:
    """Budget-averaged HD95/ASD case x arch matrix under an undefined-value rule."""
    if variant == "available":
        return data.mat(c, arm, metric)
    pivots = {a: per_eps_pivot(data, c, a, arm, metric) for a in ARCHS}
    if variant == "imputed":
        cols = {}
        for a, p in pivots.items():
            fill = pd.Series({cid: diag[cid] for cid in p.index[p.isna().any(axis=1)]})
            cols[a] = p.apply(lambda col: col.fillna(fill)).mean(axis=1)
        return pd.DataFrame(cols)
    complete = pd.concat([p.notna().all(axis=1) for p in pivots.values()], axis=1).all(
        axis=1
    )
    keep = complete[complete].index
    return pd.DataFrame({a: p.loc[keep].mean(axis=1) for a, p in pivots.items()})


def e20_distance_sensitivity(data: Data, counts: pd.DataFrame) -> pd.DataFrame:
    undefined_cases: dict[str, set[str]] = {}
    for c in TREES:
        ids: set[str] = set()
        for arch in ARCHS:
            for arm in ARMS8:
                df = data.frames[c][arch][arm]
                df = df[(df["class"] == c) & (df["hd95"].isna() | df["asd"].isna())]
                ids |= set(df["case_id"])
        undefined_cases[c] = ids
    diag = {
        cid: image_diagonal_mm(c, cid)
        for c, ids in undefined_cases.items()
        for cid in ids
    }
    rows = []
    for variant in ("available", "imputed", "complete"):
        for c in TREES:
            for arm in ARMS8:
                for metric in ("hd95", "asd"):
                    mat = distance_matrix(data, c, arm, metric, variant, diag)
                    shares = hl_share_row(mat)
                    for r in vs_unet_family(mat):
                        rows.append(
                            {
                                "variant": variant,
                                "class_name": c,
                                "arm": arm,
                                "metric": metric,
                                **r,
                                **shares,
                            }
                        )
    out = pd.DataFrame(rows)
    out.attrs["diag"] = diag
    return out


# E21 ---------------------------------------------------------------------------
def e21_per_fold(data: Data) -> tuple[pd.DataFrame, pd.DataFrame]:
    fold_rows, hier_rows = [], []
    for c in TREES:
        for arm in ARMS8:
            mat = data.mat(c, arm, "dice")
            folds = data.fold_labels(c, mat.index)
            for f in FOLDS:
                sub = mat[folds == f]
                idx = pc.stratified_bootstrap_indices(np.zeros(len(sub)), N_BOOT, SEED)
                fold_rows.append(
                    {
                        "class_name": c,
                        "arm": arm,
                        "fold": f,
                        "n_cases": len(sub),
                        **steps_with_ci(sub, idx),
                    }
                )
            hier = pc.hierarchical_bootstrap_indices(folds, N_BOOT, SEED)
            hier_rows.append(
                {
                    "class_name": c,
                    "arm": arm,
                    "n_cases": len(mat),
                    **steps_with_ci(mat, hier, prefix="hier_"),
                    **steps_with_ci(
                        mat, strat_idx(data, c, mat.index), prefix="strat_"
                    ),
                }
            )
    return pd.DataFrame(fold_rows), pd.DataFrame(hier_rows)


# E22 ---------------------------------------------------------------------------
def load_cross_arch_dice() -> pd.DataFrame:
    frames = []
    for c, tag in CLASS_TAG.items():
        df = pd.read_csv(os.path.join(STATS_DIR, f"cross_arch_wilcoxon_{tag}.csv"))
        frames.append(df[(df["metric"] == "dice") & (df["epsilon"] == "budget_avg")])
    return pd.concat(frames, ignore_index=True)


def e22_sesoi(dice72: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for sesoi in (0.01, 0.02):
        for subset, sub in (
            ("all_8", dice72),
            ("non_dag_7", dice72[dice72.arm != "dag"]),
        ):
            rows.append(
                {
                    "sesoi": sesoi,
                    "subset": subset,
                    "n_contrasts": len(sub),
                    "n_ci_lo_above": int((sub["hl_ci95_low"] > sesoi).sum()),
                    "n_hl_above": int((sub["hl_shift"] > sesoi).sum()),
                }
            )
    return pd.DataFrame(rows)


def e22_tost(data: Data, margin: float = 0.01) -> pd.DataFrame:
    rows = []
    for c in TREES:
        clean = data.clean(c)
        for arch in ARCHS[1:]:
            t = pc.tost_paired(
                clean[arch].to_numpy() - clean["unet"].to_numpy(), margin
            )
            rows.append(
                {
                    "class_name": c,
                    "comparator": arch,
                    "n_cases": len(clean),
                    "margin": margin,
                    **t,
                    "equivalent": t["p_tost"] < ALPHA,
                }
            )
    return pd.DataFrame(rows)


def e22_degradation(data: Data) -> pd.DataFrame:
    """Per-case clean minus attacked budget-averaged Dice, ResEnc vs nnU-Net."""
    rows = []
    for c in TREES:
        clean = data.clean(c)
        for arm in ARMS8:
            att = data.mat(c, arm, "dice")
            common = att.index.intersection(clean.index)
            deg = clean.loc[common] - att.loc[common]
            fam = vs_unet_family(deg)
            for r in fam:
                r.update(
                    {
                        "class_name": c,
                        "arm": arm,
                        "mean_deg_unet": float(deg["unet"].mean()),
                        "mean_deg_comparator": float(deg[r["comparator"]].mean()),
                        "resenc_degrades_less": bool(r["hl_shift"] < 0),
                    }
                )
            rows.extend(fam)
    return pd.DataFrame(rows)


# E18 ---------------------------------------------------------------------------
def e18_kendall(arm_sets: dict[str, list[str]]) -> pd.DataFrame:
    rows = []
    for c, tag in CLASS_TAG.items():
        rank = pd.read_csv(os.path.join(STATS_DIR, f"attack_ranking_by_arch_{tag}.csv"))
        for metric in ("dice", "hd95", "asd"):
            sub = rank[rank["metric"] == metric].set_index("arm")
            for label, arms in arm_sets.items():
                s = sub.loc[arms]
                for arch in ARCHS[1:]:
                    tau, p = stats.kendalltau(s["unet_median"], s[f"{arch}_median"])
                    rows.append(
                        {
                            "class_name": c,
                            "metric": metric,
                            "arm_set": label,
                            "comparator": arch,
                            "n_attacks": len(arms),
                            "kendall_tau": tau,
                            "p_value": p,
                        }
                    )
    return pd.DataFrame(rows)


def e18_dag_vs_clean(data: Data) -> pd.DataFrame:
    rows = []
    for c in TREES:
        clean = data.clean(c)
        dag = data.mat(c, "dag", "dice")
        common = dag.index.intersection(clean.index)
        for arch in ARCHS:
            d = dag.loc[common, arch].to_numpy() - clean.loc[common, arch].to_numpy()
            hl, lo, hi = _hodges_lehmann(d)
            row = {
                "class_name": c,
                "arch": arch,
                "n_cases": len(common),
                "median_abs_dag_minus_clean": float(np.median(np.abs(d))),
                "mean_dag_minus_clean": float(d.mean()),
                "hl_dag_minus_clean": hl,
                "hl_ci_lo": lo,
                "hl_ci_hi": hi,
                "wilcoxon_p": wilcoxon_p(d),
            }
            if arch != "unet":
                cd = (
                    clean.loc[common, arch].to_numpy()
                    - clean.loc[common, "unet"].to_numpy()
                )
                row["median_abs_clean_arch_minus_unet"] = float(np.median(np.abs(cd)))
            rows.append(row)
    return pd.DataFrame(rows)


# E08 ---------------------------------------------------------------------------
def e08_source(data: Data) -> pd.DataFrame:
    rows = []
    for arm in ARMS8:
        mat = data.mat("WG", arm, "dice")
        source = np.where(
            [bool(PICAI_ID.fullmatch(i)) for i in mat.index], "PI-CAI", "in-house"
        )
        for src in ("PI-CAI", "in-house"):
            sub = mat[source == src]
            rows.append(
                {
                    "arm": arm,
                    "source": src,
                    "n_cases": len(sub),
                    **steps_with_ci(sub, strat_idx(data, "WG", sub.index)),
                    **hl_share_row(sub),
                }
            )
    return pd.DataFrame(rows)


# E23 ---------------------------------------------------------------------------
def e23_global_bh(dice72: pd.DataFrame) -> pd.DataFrame:
    out = dice72[
        ["class", "arm", "comparator", "hl_shift", "p_value", "p_adj_bh", "significant"]
    ].copy()
    out["p_adj_bh_global72"] = _fdr_bh(out["p_value"].to_numpy())
    out["significant_global72"] = out["p_adj_bh_global72"] < ALPHA
    return out


# E24 ---------------------------------------------------------------------------
E24_PAIRS = (
    ("resenc_l", "resenc_m"),
    ("resenc_xl", "resenc_l"),
    ("resenc_xl", "resenc_m"),
    ("resenc_m", "unet"),
    ("resenc_l", "unet"),
    ("resenc_xl", "unet"),
)


def e24_eps010(data: Data) -> pd.DataFrame:
    rows = []
    for c in TREES:
        for arm in ("auto_pgd", "sea"):
            mat = data.mat(c, arm, "dice", 0.1)
            idx = strat_idx(data, c, mat.index)
            for a, b in E24_PAIRS:
                d = mat[a].to_numpy() - mat[b].to_numpy()
                hl, lo, hi = _hodges_lehmann(d)
                rows.append(
                    {
                        "class_name": c,
                        "arm": arm,
                        "epsilon": 0.1,
                        "contrast": f"{a}-{b}",
                        "n_cases": len(mat),
                        **{f"boot_{k}": v for k, v in boot_mean(d, idx).items()},
                        "hl": hl,
                        "hl_ci_lo": lo,
                        "hl_ci_hi": hi,
                        "wilcoxon_p": wilcoxon_p(d),
                    }
                )
    return pd.DataFrame(rows)


# E14 ---------------------------------------------------------------------------
def e14_intensity_range(data: Data) -> pd.DataFrame:
    import blosc2  # nnU-Net's preprocessed-array format; only this item needs it

    cohorts = {
        "WG": ("WG", list(data.folds["WG"].index)),
        "zones": ("zones", list(data.folds["TZ+CZ"].index)),
    }
    rows = []
    for cohort, (key, ids) in cohorts.items():
        for cid in ids:
            arr = blosc2.open(os.path.join(PREPROCESSED[key], f"{cid}.b2nd"), mode="r")[
                :
            ]
            lo, hi = float(arr.min()), float(arr.max())
            src = (
                ("PI-CAI" if PICAI_ID.fullmatch(cid) else "in-house")
                if key == "WG"
                else "in-house"
            )
            rows.append(
                {
                    "cohort": cohort,
                    "source": src,
                    "case_id": cid,
                    "zmin": lo,
                    "zmax": hi,
                    "range": hi - lo,
                }
            )
    return pd.DataFrame(rows)


def summarize_e14(per_case: pd.DataFrame) -> pd.DataFrame:
    rows = []
    groups = [
        ("WG", per_case[per_case.cohort == "WG"]),
        (
            "WG PI-CAI",
            per_case[(per_case.cohort == "WG") & (per_case.source == "PI-CAI")],
        ),
        (
            "WG in-house",
            per_case[(per_case.cohort == "WG") & (per_case.source == "in-house")],
        ),
        ("zones", per_case[per_case.cohort == "zones"]),
    ]
    for label, g in groups:
        for eps in EPSILONS:
            frac = eps / g["range"]
            q1, med, q3 = np.percentile(frac, [25, 50, 75])
            r1, rmed, r3 = np.percentile(g["range"], [25, 50, 75])
            rows.append(
                {
                    "group": label,
                    "n_cases": len(g),
                    "epsilon": eps,
                    "range_median": rmed,
                    "range_q1": r1,
                    "range_q3": r3,
                    "eps_frac_median": med,
                    "eps_frac_q1": q1,
                    "eps_frac_q3": q3,
                }
            )
    return pd.DataFrame(rows)


# E19 ---------------------------------------------------------------------------
def e19_worst_case(data: Data) -> pd.DataFrame:
    rows = []
    for c in TREES:
        per_arm = [data.mat(c, arm, "dice") for arm in NON_DAG]
        common = per_arm[0].index
        for m in per_arm[1:]:
            common = common.intersection(m.index)
        worst = pd.DataFrame(
            {
                a: np.min([m.loc[common, a].to_numpy() for m in per_arm], axis=0)
                for a in ARCHS
            },
            index=common,
        )
        argmin = {
            a: pd.Series(
                np.argmin([m.loc[common, a].to_numpy() for m in per_arm], axis=0)
            )
            .map(dict(enumerate(NON_DAG)))
            .value_counts()
            .to_dict()
            for a in ARCHS
        }
        fam = {r["comparator"]: r for r in vs_unet_family(worst)}
        for a in ARCHS:
            rows.append(
                {
                    "class_name": c,
                    "arch": a,
                    "n_cases": len(worst),
                    "mean_worst_dice": float(worst[a].mean()),
                    "median_worst_dice": float(worst[a].median()),
                    "argmin_counts": str(argmin[a]),
                    **fam.get(a, {}),
                }
            )
    return pd.DataFrame(rows)


# --------------------------------------------------------------------------- main
def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--output-dir", default=os.path.join(STATS_DIR, "revision"))
    parser.add_argument(
        "--skip-e14",
        action="store_true",
        help="skip the preprocessed-volume read (E14)",
    )
    args = parser.parse_args()
    os.makedirs(args.output_dir, exist_ok=True)

    def save(df: pd.DataFrame, name: str) -> None:
        df.to_csv(os.path.join(args.output_dir, name), index=False)
        print(f"wrote {name}: {len(df)} rows")

    data = Data(load_pooled_frames())
    save(e02_decomposition(data), "E02_additive_decomposition.csv")
    save(e27_per_epsilon(data), "E27_E28_per_epsilon_steps.csv")
    counts = undefined_counts(data)
    save(counts, "E20_undefined_distance_counts.csv")
    sens = e20_distance_sensitivity(data, counts)
    save(sens, "E20_distance_sensitivity.csv")
    save(
        pd.DataFrame(
            [{"case_id": k, "diagonal_mm": v} for k, v in sens.attrs["diag"].items()]
        ),
        "E20_imputation_diagonals.csv",
    )
    per_fold, hier = e21_per_fold(data)
    save(per_fold, "E21_per_fold_steps.csv")
    save(hier, "E21_hierarchical_bootstrap.csv")
    dice72 = load_cross_arch_dice()
    save(e22_sesoi(dice72), "E22_sesoi_counts.csv")
    save(e22_tost(data), "E22_clean_tost.csv")
    save(e22_degradation(data), "E22_degradation_endpoint.csv")
    save(e18_kendall({"all_8": ARMS8, "non_dag_7": NON_DAG}), "E18_kendall_tau.csv")
    save(e18_dag_vs_clean(data), "E18_dag_vs_clean.csv")
    save(e08_source(data), "E08_wg_by_source.csv")
    save(e23_global_bh(dice72), "E23_global_bh.csv")
    save(e24_eps010(data), "E24_eps010_pairwise.csv")
    save(e19_worst_case(data), "E19_worst_case_over_attacks.csv")
    if not args.skip_e14:
        per_case = e14_intensity_range(data)
        save(per_case, "E14_intensity_range_per_case.csv")
        save(summarize_e14(per_case), "E14_eps_fraction_of_range.csv")


if __name__ == "__main__":
    main()
