"""Driver helpers that need no GPU."""

from __future__ import annotations

import math

import numpy as np
import pandas as pd
import pytest
import torch

import experiments.evaluate_edge_attribution_all_cases as evaluator
from experiments.evaluate_edge_attribution_all_cases import (
    CONTROL_FIELDNAMES,
    CURVES_FIELDNAMES,
    EXPECTED_ATLASES,
    IDENTITY,
    PATCH_FIELDNAMES,
    QC_FIELDNAMES,
    SUMMARY_CSV,
    SUMMARY_FIELDNAMES,
    atlas_path,
    completed_keys,
    condition_damages,
    curve_k_values,
    expected_atlases,
    load_or_build_atlases,
    map_path,
    pad_masks,
    primary_fn,
    select_pool_top_q,
)
from mri_prostate_seg.experiments.edge_attribution import AtlasConfig, EdgeAtlas

SPACING = (3.0, 0.5, 0.5)
SHAPE = (10, 64, 64)


def test_expected_atlases_per_dataset() -> None:
    assert expected_atlases("wg") == {"native_edges", "outer_boundary_gt"}
    assert expected_atlases("zones") == {
        "native_edges",
        "outer_boundary_gt",
        "zone_interface_gt",
    }
    assert set(EXPECTED_ATLASES) == {"wg", "zones"}


def test_completed_keys_needs_every_atlas(tmp_path) -> None:
    rows = [
        {
            "dataset": "wg",
            "case_id": "a",
            "epsilon_n": 16,
            "attack": "APGD-BCE",
            "atlas_type": "native_edges",
        },
        {
            "dataset": "wg",
            "case_id": "a",
            "epsilon_n": 16,
            "attack": "APGD-BCE",
            "atlas_type": "outer_boundary_gt",
        },
        {
            "dataset": "wg",
            "case_id": "b",
            "epsilon_n": 16,
            "attack": "APGD-BCE",
            "atlas_type": "native_edges",
        },
    ]
    path = tmp_path / SUMMARY_CSV
    pd.DataFrame(rows).to_csv(path, index=False)
    assert completed_keys(path) == {("wg", "a", 16, "APGD-BCE")}
    assert completed_keys(tmp_path / "missing.csv") == set()


def test_paths_layout(tmp_path) -> None:
    assert atlas_path(tmp_path, "wg", "c1") == tmp_path / "atlases" / "wg" / "c1.npz"
    assert (
        map_path(tmp_path, "zones", "c2", "PGD-BCE", 8)
        == tmp_path / "maps" / "zones" / "c2" / "PGD-BCE_eps8.npz"
    )


def test_pad_masks_zero_pads_to_model_grid() -> None:
    masks = np.ones((2, 3, 4, 4), dtype=bool)
    out = pad_masks(masks, (4, 8, 8))
    assert out.shape == (2, 4, 8, 8) and out.dtype == bool
    assert out[:, :3, :4, :4].all() and not out[:, 3:].any() and not out[:, :, 4:].any()


def test_primary_fn_wg_and_zones() -> None:
    assert primary_fn("wg")(np.array([0.7])) == pytest.approx(0.7)
    assert primary_fn("zones")(np.array([0.6, 0.8])) == pytest.approx(0.7)


def test_condition_damages_signs_undefined_and_non_damaging() -> None:
    clean = {"dice": 0.9, "hd95": 2.0, "asd": 0.5, "loss": 0.1}
    adv = {"dice": 0.6, "hd95": math.inf, "asd": 0.7, "loss": 0.4}
    damages, undefined = condition_damages(clean, adv)
    assert damages["dice"] == pytest.approx(0.3) and damages["asd"] == pytest.approx(
        0.2
    )
    assert math.isnan(damages["hd95"]) and undefined == ["hd95"]
    same, none = condition_damages(clean, clean)
    assert same["dice"] == 0.0 and none == []


def test_curve_k_values_ladder_and_fractions() -> None:
    ks = curve_k_values(200)
    assert ks[:3] == [1, 2, 3] and 89 in ks and 200 in ks
    assert 2 in ks and 10 in ks and 20 in ks and 40 in ks  # 1/5/10/20 % of 200
    assert ks == sorted(set(ks))
    assert curve_k_values(3) == [1, 2, 3]


def test_load_or_build_atlases_caches_and_survives_corruption(tmp_path) -> None:
    seg = np.zeros(SHAPE, dtype=np.int16)
    seg[3:7, 20:44, 16:48] = 1
    rng = np.random.default_rng(0)
    image = rng.normal(size=SHAPE).astype(np.float32)
    image[seg > 0] += 3.0
    path = tmp_path / "wg" / "case.npz"
    config = AtlasConfig(min_core_voxels=3)
    atlases, roi, strength = load_or_build_atlases(
        path, image, seg, SPACING, config=config
    )
    assert path.is_file() and set(atlases) == {"native_edges", "outer_boundary_gt"}
    again, roi2, _ = load_or_build_atlases(path, image, seg, SPACING, config=config)
    assert np.array_equal(again["native_edges"].tubes, atlases["native_edges"].tubes)
    assert np.array_equal(roi, roi2)
    path.write_bytes(b"not an npz")
    rebuilt, _, _ = load_or_build_atlases(path, image, seg, SPACING, config=config)
    assert np.array_equal(rebuilt["native_edges"].tubes, atlases["native_edges"].tubes)


def test_fieldnames_start_with_identity_and_are_unique() -> None:
    for names in (
        SUMMARY_FIELDNAMES,
        PATCH_FIELDNAMES,
        CURVES_FIELDNAMES,
        CONTROL_FIELDNAMES,
        QC_FIELDNAMES,
    ):
        assert names[: len(IDENTITY)] == IDENTITY
        assert len(names) == len(set(names))


# ---------------------------------------------------------------------------
# Joint matched control: selection symmetry (C3)
#
# `interaction_stage` compares the removal of the q patches with the highest
# MEASURED necessity, chosen out of the whole screened set, against a joint
# matched control.  If the control arm is an unselected draw, the comparison is
# selection on outcome: under a null where every patch has identical true
# necessity the observed statistic is still the maximum of len(ordered) noisy
# measurements, so `control_permutation_p` is skewed small no matter how well
# volume, energy and band are matched.  The tests below run the real
# `interaction_stage` against a stub whose response to a mask is deterministic
# (which is what makes selection on outcome bite -- a stub that returns fresh
# noise per call would make the pre-fix code look unbiased).
#
# Every region, real patch or control draw, is PATCH_VOXELS voxels and the
# hidden per-voxel effect is one i.i.d. field, so every region's effect sum is a
# draw from one common distribution: a true null in which the arms are
# exchangeable, yet each region's measured response reproduces across calls.
# The per-voxel (rather than per-region) formulation is what makes this
# possible at all -- control regions are BFS draws that do not exist until the
# draw happens, so they cannot be enumerated and assigned effects in advance.
#
# That equal-volume world is blind to a second asymmetry, because it holds the
# covariate that one runs along constant: matched draws do not succeed
# uniformly, so the pool is the twin set of a size-truncated subset of the
# patches, and an observed arm sampled from anywhere else wins on union volume.
# `_volume_asymmetry_world` below is the harness for that one.
# ---------------------------------------------------------------------------
EFFECT_SHAPE = (4, 48, 48)
PATCH_VOXELS = 16
N_PATCHES = 60
N_ADV_VOXELS = 2000
DICE_GAIN = 50.0


class _EffectModel(torch.nn.Module):
    """Dice is a fixed, reproducible function of the removed region's effect sum."""

    def __init__(
        self,
        effect: np.ndarray,
        clean: torch.Tensor,
        fg_index: np.ndarray,
        gain: float = DICE_GAIN,
    ) -> None:
        super().__init__()
        self.gain = gain
        self.effect = torch.from_numpy(effect.astype(np.float32))
        self.clean = clean
        self.fg_index = torch.from_numpy(fg_index.astype(np.int64))
        self.n_fg = int(fg_index.size)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x_adv is x + 1 everywhere, so equality with the clean volume recovers
        # the removal mask exactly.
        removed = x[:, 0] == self.clean
        total = (removed.float() * self.effect).sum(dim=(1, 2, 3))
        n = torch.clamp(
            torch.round(N_ADV_VOXELS + self.gain * total), 0, self.n_fg
        ).long()
        logits = torch.zeros((int(x.shape[0]), 2, *self.clean.shape))
        flat = logits.view(int(x.shape[0]), 2, -1)
        flat[:, 0, :] = 1.0
        for i in range(int(x.shape[0])):
            flat[i, 1, self.fg_index[: int(n[i])]] = 2.0
        return logits


def _prefix_dice(n: int, n_fg: int) -> float:
    """Dice of a prediction covering the first ``n`` of ``n_fg`` GT voxels."""

    return 2.0 * n / (n + n_fg)


def _null_world(
    seed: int, *, roi: np.ndarray | None = None
) -> tuple[torch.nn.Module, object, object, dict, list[int]]:
    rng = np.random.default_rng(seed)
    tubes = np.zeros(EFFECT_SHAPE, dtype=np.int32)
    spots = [(y, x) for y in range(0, 46, 6) for x in range(0, 46, 6)][:N_PATCHES]
    for pid, (y, x) in enumerate(spots, start=1):
        tubes[:, y : y + 2, x : x + 2] = pid
    return _world_from(
        tubes,
        rng.normal(size=EFFECT_SHAPE),
        np.zeros(EFFECT_SHAPE),
        roi,
    )


# A world in which draw success covaries with patch volume -- the covariate the
# equal-volume null above holds constant, and therefore cannot see.
#
# Two properties are needed, and both were calibrated against the mechanism
# MEASURED on the pilot's own atlases rather than tuned until the fractions
# looked right (zones/native_edges: P(draw succeeds) 0.95 on the smallest
# volume quartile against 0.59 on the largest, and a top-q union 1.34x larger
# when the observed arm is sampled from the full size distribution instead of
# the drawable one):
#
#   * heavy-tailed volumes -- HETERO_SIZES puts four patches at 96 voxels among
#     56 of 8 or 16.  An evenly spread 3:1 range cannot express the defect at
#     q=4: whichever arm selects, both reach the same union volume because the
#     largest size class is far bigger than q, so the truncation has nothing to
#     take away.  A few rare large patches are what makes losing one cost real
#     volume, which is also the shape of the real atlases (median 418 voxels,
#     max 2574).
#   * a real band constraint -- every draw must land in ONE four-row
#     signed-distance band whose capacity (768 voxels) is below the pool's
#     demand (928), so the sweep exhausts it and the 96-voxel targets are the
#     ones that stop fitting.
#
# The per-voxel effect has a positive mean, so recovery grows with removed
# volume as it does in the real data, and HETERO_DICE_GAIN keeps the response
# off the Dice clamp that would otherwise flatten it.
HETERO_SIZES = ((8, 44), (16, 12), (96, 4))  # (voxels, how many patches)
HETERO_EFFECT_MEAN = 0.5
BAND_CENTRE_Y = 24
BAND_MM_PER_ROW = 0.5  # -> the [0, 2) mm band is four y-rows deep
HETERO_DICE_GAIN = 5.0


def _serpentine_cells(dy: int, dx: int):
    """(y, x) offsets in an order where consecutive cells stay 26-adjacent."""

    for j in range(dy):
        xs = range(dx) if j % 2 == 0 else range(dx - 1, -1, -1)
        for i in xs:
            yield j, i


def _volume_asymmetry_world(
    seed: int, *, roi: np.ndarray | None = None
) -> tuple[torch.nn.Module, object, object, dict, list[int]]:
    rng = np.random.default_rng(seed)
    sizes = [v for v, count in HETERO_SIZES for _ in range(count)]
    assert len(sizes) == N_PATCHES
    tubes = np.zeros(EFFECT_SHAPE, dtype=np.int32)
    spots = [(y, x) for y in range(0, 46, 6) for x in range(0, 46, 6)][:N_PATCHES]
    for pid, (y, x) in enumerate(spots, start=1):
        filled = 0
        for j, i in _serpentine_cells(5, 5):
            for z in range(EFFECT_SHAPE[0]):
                if filled >= sizes[pid - 1]:
                    break
                tubes[z, y + j, x + i] = pid
                filled += 1
            if filled >= sizes[pid - 1]:
                break
    # signed distance varies with y only; rows[*]["signed_distance_median_mm"]
    # is 0.0 for every patch, so distance_band puts every draw in [0, 2) mm.
    sd = np.broadcast_to(
        ((np.arange(EFFECT_SHAPE[1]) - BAND_CENTRE_Y) * BAND_MM_PER_ROW)[None, :, None],
        EFFECT_SHAPE,
    ).astype(np.float64)
    effect = rng.normal(loc=HETERO_EFFECT_MEAN, size=EFFECT_SHAPE)
    return _world_from(tubes, effect, sd, roi, HETERO_DICE_GAIN)


def _world_from(
    tubes: np.ndarray,
    effect: np.ndarray,
    signed_distance: np.ndarray,
    roi: np.ndarray | None,
    gain: float = DICE_GAIN,
) -> tuple[torch.nn.Module, object, object, dict, list[int]]:
    atlas = EdgeAtlas(name="native_edges", cores=tubes.copy(), tubes=tubes)

    seg = np.zeros(EFFECT_SHAPE, dtype=np.int16)
    seg[:, 4:44, 4:44] = 1
    fg_index = np.flatnonzero(seg.ravel() > 0)
    clean = torch.zeros((1, 1, *EFFECT_SHAPE), dtype=torch.float32)
    model = _EffectModel(effect, clean[0, 0], fg_index, gain)

    adv_dice = _prefix_dice(N_ADV_VOXELS, int(fg_index.size))
    sums = {pid: float(effect[tubes == pid].sum()) for pid in range(1, N_PATCHES + 1)}
    # The order causal_stage would produce under this stub: measured necessity
    # is strictly increasing in the patch's effect sum.
    ordered = sorted(sums, key=lambda pid: -sums[pid])
    rows = {
        pid: {
            "patch_id": pid,
            "signed_distance_median_mm": 0.0,
            "energy": float(int((tubes == pid).sum())),
            "fractional_recovery_dice": 0.0,
            "sufficiency_dice": sums[pid],
            # What causal_stage would have measured: the observed arm ranks its
            # matched subset on this, never on a fresh measurement.
            "necessity_dice": sums[pid],
        }
        for pid in ordered
    }
    case = evaluator.CaseContext(
        dataset_key="wg",
        case_id="synthetic",
        spacing=SPACING,
        seg=seg,
        valid=seg >= 0,
        gt_fg=seg > 0,
        roi=np.ones(EFFECT_SHAPE, dtype=bool) if roi is None else roi,
        strength=np.zeros(EFFECT_SHAPE),
        atlases={"native_edges": atlas},
        signed_distance=signed_distance,
        x_pad=clean,
        y_pad=torch.from_numpy(seg[None, None].astype(np.float32)),
        padded_shape=EFFECT_SHAPE,
        shape=EFFECT_SHAPE,
        clean_probs=np.zeros((2, *EFFECT_SHAPE)),
        clean_pred=np.zeros(EFFECT_SHAPE, dtype=np.int16),
        clean_loss=0.0,
        clean_metrics={"dice": 1.0},
        static={},
        in_causal=True,
        in_interaction=True,
    )
    cond = evaluator.Condition(
        attack_key="apgd",
        attack_label="APGD-BCE",
        epsilon_n=8,
        seed=0,
        adv_pad=clean + 1.0,
        delta=np.ones(EFFECT_SHAPE, dtype=np.float32),
        energy=np.ones(EFFECT_SHAPE),
        adv_pred=np.zeros(EFFECT_SHAPE, dtype=np.int16),
        adv_metrics={"dice": adv_dice},
        damages={"dice": 1.0 - adv_dice},
        undefined=[],
        linf=1.0,
        rms=1.0,
        saturation=0.0,
        rms_error=0.0,
    )
    return model, case, cond, rows, ordered


def _null_settings(*, pool: int = 0, trials: int = 2) -> object:
    return evaluator.Settings(
        stages=("interaction",),
        ig_steps=0,
        ig_batch=1,
        iv_batch=16,
        boundary_workers=1,
        greedy_limit=2,
        pairwise_limit=2,
        controls_per_patch=trials,
        control_pool_size=pool,
        local_radius_mm=0.0,
        save_maps=False,
        base_seed=42,
    )


def _run_interaction(
    seed: int,
    *,
    pool: int = 0,
    trials: int = 2,
    roi: np.ndarray | None = None,
    world=_null_world,
) -> tuple[list[dict], dict]:
    model, case, cond, rows, ordered = world(seed, roi=roi)
    summary: dict = {}
    out = evaluator.interaction_stage(
        model,
        case,
        cond,
        "native_edges",
        rows,
        summary,
        ordered,
        2,
        _null_settings(pool=pool, trials=trials),
        np.random.default_rng(seed + 1000),
    )
    return out, summary


def _joint_recoveries(out: list[dict]) -> list[float]:
    return [
        float(r["recovery_fraction"])
        for r in out
        if str(r["control_type"]).endswith("_joint")
    ]


def _observed_rows(out: list[dict]) -> list[dict]:
    return [r for r in out if r["control_type"] == "observed_matched_subset"]


def _observed_beats_control_median(reps: int, *, world=_null_world) -> float:
    wins = 0
    for rep in range(reps):
        out, summary = _run_interaction(rep, world=world)
        joints = _joint_recoveries(out)
        assert joints, "the null harness must produce joint controls"
        if float(summary["observed_matched_recovery"]) > float(np.median(joints)):
            wins += 1
    return wins / reps


def _cap_pool_draws(monkeypatch, m_target: int, *, trials: int) -> None:
    """Let only the first ``m_target`` draws of each JOINT trial succeed.

    Call arithmetic, per control type: the per-patch control block draws
    ``len(top10) * trials`` times first (top10 is 6 patches for N_PATCHES=60);
    after that each joint trial calls its matched-region function exactly
    ``N_PATCHES`` times -- once per walk entry, failures included -- so the call
    index alone locates a draw within its own trial's pool.
    """

    per_patch = 6 * trials

    def capped(name: str):
        real = getattr(evaluator, name)
        calls = {"n": 0}

        def fake(**kwargs: object) -> np.ndarray | None:
            i = calls["n"]
            calls["n"] += 1
            if i < per_patch or (i - per_patch) % N_PATCHES < m_target:
                return real(**kwargs)  # type: ignore[no-any-return]
            return None

        return fake

    for name in ("volume_matched_region", "energy_matched_region"):
        monkeypatch.setattr(evaluator, name, capped(name))


def test_joint_control_is_symmetric_under_a_null(monkeypatch) -> None:
    """Both arms select top-q by measured recovery, so neither wins systematically."""

    assert 0.25 <= _observed_beats_control_median(20) <= 0.75
    # The pre-fix construction: q matched draws that were never selected on any
    # measured quantity, against an observed arm that was.  Everything else --
    # the pool draw, the matching, the disjointness update, the stub -- is
    # identical, so a failure here can only be the selection asymmetry.  If this
    # ever lands inside the band the stub has stopped being able to see the
    # defect and the test above is worthless.
    monkeypatch.setattr(
        evaluator,
        "select_pool_top_q",
        lambda recoveries, regions, q: np.logical_or.reduce(list(regions)[:q]),
    )
    assert not 0.25 <= _observed_beats_control_median(20) <= 0.75


def test_joint_control_is_symmetric_when_draw_success_tracks_volume(
    monkeypatch,
) -> None:
    """The volume-asymmetry null: heterogeneous patches, one thin band.

    The equal-volume world above cannot see this defect at all -- it holds the
    covariate the bias runs along constant, so a uniform subsample of `ordered`
    and P_succ are the same set by construction and BOTH land inside the band.
    Here draw success falls with target volume, so the two differ: the observed
    arm must rank the pool's OWN id set, or it selects its top q out of the full
    size distribution while the control selects out of the drawable, size-
    truncated one, and wins on union volume alone.

    Measured: 0.35 for the fix, 0.95 for the pre-fix uniform subsample.
    """

    assert (
        0.25
        <= _observed_beats_control_median(20, world=_volume_asymmetry_world)
        <= 0.75
    )

    # The pre-fix construction: a uniform M-subsample of `ordered` instead of
    # P_succ.  It draws from its OWN generator, so the pool draws, the control
    # regions and every control_t are bit-identical to the run above and the
    # only thing that changed is which patches the observed arm may rank.
    def prefix_arm(_ordered: list[int], rng: np.random.Generator):
        def stub(pool_ids, necessity, q):
            sub = [
                int(p)
                for p in rng.choice(
                    np.asarray(_ordered), size=len(pool_ids), replace=False
                )
            ]
            return sorted(sub, key=lambda pid: -necessity[pid])[:q]

        return stub

    wins = 0
    for rep in range(20):
        model, case, cond, rows, ordered = _volume_asymmetry_world(rep)
        monkeypatch.setattr(
            evaluator,
            "select_observed_matched_subset",
            prefix_arm(ordered, np.random.default_rng(rep + 7)),
        )
        summary: dict = {}
        out = evaluator.interaction_stage(
            model,
            case,
            cond,
            "native_edges",
            rows,
            summary,
            ordered,
            2,
            _null_settings(),
            np.random.default_rng(rep + 1000),
        )
        joints = _joint_recoveries(out)
        assert joints, "the null harness must produce joint controls"
        if float(summary["observed_matched_recovery"]) > float(np.median(joints)):
            wins += 1
    assert not 0.25 <= wins / 20 <= 0.75, (
        "the harness has stopped seeing the volume asymmetry"
    )


def test_select_pool_top_q_takes_the_highest_measured_recoveries() -> None:
    regions = [np.zeros(6, dtype=bool) for _ in range(5)]
    for i, region in enumerate(regions):
        region[i] = True
    union = select_pool_top_q([0.1, 0.9, 0.2, 0.8, 0.0], regions, 2)
    assert list(np.flatnonzero(union)) == [1, 3]  # not the first q, not all of them


def test_select_pool_top_q_unions_fewer_than_q_when_pool_is_short() -> None:
    """The selected list's true length is ``min(q, len(regions))``, not q --
    this is what the driver's ``n_patches_in_joint`` must report. Only
    reachable directly on ``select_pool_top_q``: after the I1 fix,
    ``interaction_stage``'s gate requires ``len(pool) > q`` for any trial that
    contributes a joint control, so inside the driver this length is always
    exactly q. ``select_pool_top_q``'s own truncation is nonetheless what the
    driver's ``min(q, len(pool))`` computation relies on staying correct for a
    pool this short."""
    regions = [np.zeros(6, dtype=bool) for _ in range(3)]
    for i, region in enumerate(regions):
        region[i] = True
    union = select_pool_top_q([0.5, 0.9, 0.1], regions, 5)  # q=5 > len(regions)=3
    assert list(np.flatnonzero(union)) == [0, 1, 2]  # all 3, not padded to 5


def test_joint_control_removes_exactly_the_top_q_of_the_pool(monkeypatch) -> None:
    seen: list[tuple[np.ndarray, int]] = []
    real = evaluator.select_pool_top_q

    def spy(recoveries, regions, q):
        union = real(recoveries, regions, q)
        best = sorted(range(len(regions)), key=lambda i: -recoveries[i])[:q]
        expected = np.logical_or.reduce([regions[i] for i in best])
        assert np.array_equal(union, expected)
        assert not np.array_equal(union, np.logical_or.reduce(list(regions)[:q])), (
            "top-q must not coincide with first-q-drawn"
        )
        assert union.sum() < np.logical_or.reduce(list(regions)).sum()
        seen.append((union, q))
        return union

    monkeypatch.setattr(evaluator, "select_pool_top_q", spy)
    out, _ = _run_interaction(7)
    joints = [r for r in out if str(r["control_type"]).endswith("_joint")]
    assert len(seen) == len(joints) == 4  # 2 control types x 2 trials
    assert [int(r["control_voxels"]) for r in joints] == [
        int(union.sum()) for union, _ in seen
    ]
    assert {int(r["n_patches_in_joint"]) for r in joints} == {seen[0][1]}


def test_joint_rows_pair_trial_wise_with_the_observed_matched_subset(
    monkeypatch,
) -> None:
    """Each joint row is compared against ITS OWN trial's observed statistic."""

    _cap_pool_draws(monkeypatch, 20, trials=2)
    out, summary = _run_interaction(11)
    joints = [r for r in out if str(r["control_type"]).endswith("_joint")]
    observed = _observed_rows(out)
    assert len(joints) == len(observed) == 4  # 2 control types x 2 trials
    # M < len(ordered), so the per-trial observed statistics genuinely differ --
    # a pairing test against four identical numbers would prove nothing.
    assert len({float(r["recovery_fraction"]) for r in observed}) > 1
    assert [r["patch_recovery_fraction"] for r in joints] == [
        r["recovery_fraction"] for r in observed
    ]
    # The comparison column must describe the same mask as target_voxels.
    assert [r["target_voxels"] for r in joints] == [
        r["control_voxels"] for r in observed
    ]
    assert all(
        r["patch_recovery_fraction"] != summary["observed_top10_recovery"]
        for r in joints
    )
    wins = sum(
        1
        for j, o in zip(joints, observed)
        if float(j["recovery_fraction"]) >= float(o["recovery_fraction"])
    )
    assert summary["control_permutation_p"] == pytest.approx(
        (wins + 1) / (len(joints) + 1)
    )
    assert summary["observed_matched_recovery"] == pytest.approx(
        float(np.median([float(r["recovery_fraction"]) for r in observed]))
    )
    assert summary["matched_pool_size_median"] == 20.0
    assert summary["control_drawable_fraction"] == pytest.approx(20 / N_PATCHES)


def test_a_pool_far_below_the_request_still_contributes(monkeypatch) -> None:
    """The gate is M > q only -- the regression the 10-case pilot caught.

    Every trial there realised 64-78% of the requested pool, so the old
    ``len(pool) < 0.9 * requested_pool`` condition skipped all 20 of them and
    control_permutation_p was NaN on every row.
    """

    _cap_pool_draws(monkeypatch, 20, trials=2)
    out, summary = _run_interaction(3)
    assert 20 < 0.9 * N_PATCHES  # would have been skipped as undersized
    assert summary["control_pool_size"] == 20.0
    assert summary["n_trials_skipped_undersized"] == 0
    assert summary["control_pool_undersized"] is False
    assert len(_joint_recoveries(out)) == 4
    assert not math.isnan(float(summary["control_permutation_p"]))


def test_pool_at_or_below_q_still_skips_and_yields_nan(monkeypatch) -> None:
    """q = 4 here, so M = 4 leaves the control arm no selection to perform."""

    _cap_pool_draws(monkeypatch, 4, trials=2)
    out, summary = _run_interaction(5)
    assert not _joint_recoveries(out) and not _observed_rows(out)
    assert summary["n_trials_skipped_undersized"] == 4  # 2 control types x 2 trials
    assert summary["control_pool_undersized"] is True
    assert math.isnan(float(summary["control_permutation_p"]))
    assert math.isnan(float(summary["observed_matched_recovery"]))
    assert math.isnan(float(summary["matched_pool_size_median"]))
    # Still reported: the drawable fraction is exactly what says why it is NaN.
    assert summary["control_drawable_fraction"] == pytest.approx(4 / N_PATCHES)


def test_pool_holds_exactly_one_twin_per_patch(monkeypatch) -> None:
    """The matching argument rests on this: |P_succ| == |pool|, ids unique.

    If a trial ever drew two twins for one patch, or left one unpaired, the
    observed arm would rank a different set from the one the pool was matched
    to and the volume/band/energy marginals would silently stop agreeing.
    """

    real_top, real_obs = (
        evaluator.select_pool_top_q,
        (evaluator.select_observed_matched_subset),
    )
    pool_sizes: list[int] = []
    id_sets: list[list[int]] = []

    def spy_top(recoveries, regions, q):
        pool_sizes.append(len(regions))
        return real_top(recoveries, regions, q)

    def spy_obs(pool_ids, necessity, q):
        id_sets.append(list(pool_ids))
        return real_obs(pool_ids, necessity, q)

    monkeypatch.setattr(evaluator, "select_pool_top_q", spy_top)
    monkeypatch.setattr(evaluator, "select_observed_matched_subset", spy_obs)
    _cap_pool_draws(monkeypatch, 20, trials=2)
    _run_interaction(13)
    assert len(id_sets) == len(pool_sizes) == 4
    for ids, n_regions in zip(id_sets, pool_sizes):
        assert len(ids) == n_regions == 20  # one twin each, none unpaired
        assert len(set(ids)) == len(ids)  # and never two twins for one patch
        assert set(ids) <= set(range(1, N_PATCHES + 1))


def test_observed_subset_costs_one_extra_forward_per_trial(monkeypatch) -> None:
    """The observed arm reuses causal_stage's necessity instead of remeasuring.

    Every single-mask intervention in this stage is either the one top-10%
    removal or one observed matched subset, so the count pins the extra cost at
    exactly one forward per qualifying trial -- not a second pool sweep.
    """

    real = evaluator.evaluate_interventions
    single_mask = {"n": 0}

    def spy(*args, **kwargs):
        masks = kwargs["masks"] if "masks" in kwargs else args[5]
        if masks.shape[0] == 1:
            single_mask["n"] += 1
        return real(*args, **kwargs)

    monkeypatch.setattr(evaluator, "evaluate_interventions", spy)
    _cap_pool_draws(monkeypatch, 20, trials=2)
    out, _ = _run_interaction(17)
    assert single_mask["n"] == 1 + len(_observed_rows(out)) == 5


def test_pool_sizes_are_recorded_and_the_cap_is_honoured() -> None:
    _, summary = _run_interaction(3)
    assert summary["observed_pool_size"] == N_PATCHES
    assert summary["control_pool_size"] == float(N_PATCHES)
    assert summary["matched_pool_size_median"] == float(N_PATCHES)
    assert summary["control_drawable_fraction"] == 1.0
    assert summary["n_trials_skipped_undersized"] == 0
    assert summary["control_pool_undersized"] is False
    _, capped = _run_interaction(3, pool=20)
    assert capped["observed_pool_size"] == N_PATCHES
    assert capped["control_pool_size"] == 20.0
    # --control-pool-size caps the REQUESTED pool; the realised M is what both
    # arms use, and it is that M the drawable fraction reports.
    assert capped["matched_pool_size_median"] == 20.0
    assert capped["control_drawable_fraction"] == pytest.approx(20 / N_PATCHES)
    assert capped["control_pool_undersized"] is False


def test_undersized_pool_skips_the_trial_and_flags_it() -> None:
    roi = np.zeros(EFFECT_SHAPE, dtype=bool)
    # Room for at most three disjoint 16-voxel regions, i.e. M <= q = 4 once
    # the per-patch controls have taken their share.  (A roomier ROI that
    # merely falls short of the request is no longer a skip -- see
    # test_a_pool_far_below_the_request_still_contributes.)
    roi[:, 0:3, 0:6] = True
    out, summary = _run_interaction(5, roi=roi)
    assert not _joint_recoveries(out), "a short joint must not be contributed"
    assert summary["n_trials_skipped_undersized"] == 4  # 2 control types x 2 trials
    assert summary["control_pool_size"] < 0.9 * N_PATCHES
    assert summary["control_pool_undersized"] is True
    assert math.isnan(float(summary["control_permutation_p"]))
    # The per-patch controls still drew, so this is not `no_matched_controls`.
    assert summary["n_controls_volume"] > 0


def test_control_pool_at_or_below_q_skips_the_trial() -> None:
    """I1: ``--control-pool-size`` at or below q must not silently reopen the
    pre-fix selection bias.  q=4 in the null harness (60 patches -> top10 has
    6 -> q = ceil(0.6*6) = 4); pool=2 <= q means order[:q] would take the
    whole (unselected) pool, which is no selection at all."""
    out, summary = _run_interaction(5, pool=2)
    assert not _joint_recoveries(out), (
        "control_pool_size <= q must not emit an unselected joint control"
    )
    assert summary["n_trials_skipped_undersized"] == 4  # 2 control types x 2 trials
    assert math.isnan(float(summary["control_permutation_p"]))


def test_flag_fires_when_a_single_trial_is_skipped_while_median_stays_healthy(
    monkeypatch,
) -> None:
    """M4: the flag must fire on a single skipped trial even though the
    median realised pool size is unaffected -- a median-based trigger hides a
    materially thinned null distribution when only a minority of trials are
    skipped."""
    real = evaluator.energy_matched_region
    calls = {"n": 0}
    # Call order: the per-patch block calls energy_matched_region
    # len(top10) * trials times (6 * 3 = 18) before the joint block's
    # energy-matched arm begins; that arm's first trial then draws exactly
    # N_PATCHES times regardless of success. Failing calls 19-78 forces only
    # that one trial (of 6: 2 control types x 3 trials) to an empty pool.
    per_patch_calls = 6 * 3
    fail_until = per_patch_calls + N_PATCHES

    def fake(**kwargs: object) -> np.ndarray | None:
        calls["n"] += 1
        if per_patch_calls < calls["n"] <= fail_until:
            return None
        return real(**kwargs)

    monkeypatch.setattr(evaluator, "energy_matched_region", fake)
    _, summary = _run_interaction(5, trials=3)
    assert summary["n_trials_skipped_undersized"] == 1
    assert summary["control_pool_size"] == float(N_PATCHES)  # median unaffected
    assert summary["control_pool_undersized"] is True


def test_trainer_dir_cli_defaults_to_clean_model(monkeypatch) -> None:
    monkeypatch.setattr("sys.argv", ["prog"])
    assert evaluator.parse_args().trainer_dir == evaluator.TRAINERS["unet"]
    monkeypatch.setattr("sys.argv", ["prog", "--trainer-dir", "X__Y__3d_fullres"])
    assert evaluator.parse_args().trainer_dir == "X__Y__3d_fullres"


def test_attack_valid_only_flag_defaults_off_and_settings_stay_positional(monkeypatch) -> None:
    monkeypatch.setattr("sys.argv", ["prog"])
    assert evaluator.parse_args().attack_valid_only is False
    monkeypatch.setattr("sys.argv", ["prog", "--attack-valid-only"])
    assert evaluator.parse_args().attack_valid_only is True
    settings = evaluator.Settings(("energy",), 4, 4, 8, 1, 10, 10, 20, 500, 5.0, True, 1)
    assert settings.attack_valid_only is False


def test_attackable_mask_excludes_ignore_label_and_confines_the_attack(monkeypatch) -> None:
    y = torch.full((1, 1, 2, 4, 4), evaluator.IGNORE_LABEL, dtype=torch.long)
    y[..., 1:, 1:3, 1:3] = 1
    mask = evaluator.attackable_mask(y)
    assert mask.dtype == torch.bool and mask.shape == y.shape
    assert mask.sum().item() == 4
    seen: dict[str, object] = {}

    def fake_fgsm(model, x, y, eps, num_classes, perturbation_mask=None):
        seen["mask"] = perturbation_mask
        return x

    monkeypatch.setattr(evaluator, "fgsm_bce_independent_batch", fake_fgsm)
    case = type("C", (), {"dataset_key": "zones", "case_id": "c", "x_pad": torch.zeros_like(y, dtype=torch.float32), "y_pad": y})()
    evaluator.run_attack(None, case, "fgsm", 16, 3, attack_steps=1, base_seed=0)
    assert seen["mask"] is None
    evaluator.run_attack(None, case, "fgsm", 16, 3, attack_steps=1, base_seed=0, valid_only=True)
    assert torch.equal(seen["mask"], mask)
