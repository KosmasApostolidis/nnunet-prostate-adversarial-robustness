#!/usr/bin/env python
"""
Unit tests for the corpus AT-loss terms (NuAT / UIAT / LBGAT) in robust_loss_utils.

CPU-only, no GPU, no dataset. For each term and BOTH nnU-Net head types
(non-region softmax = WG, region sigmoid multilabel = zones) it asserts:
  - forward value is finite and strictly nonzero on dummy 3D tensors,
  - backward produces finite, nonzero gradients w.r.t. the robust logits
    (the half these losses actually fail on: NuAT's nuclear norm goes through
    an SVD whose backward NaNs on colliding singular values).

Run: python training/test_corpus_losses.py
"""
import torch

from nnunetv2.training.nnUNetTrainer.variants.loss.robust_loss_utils import (
    boundary_band_mask,
    inverse_adversary,
    inverse_adversary_kl_term,
    lbgat_logit_mse_term,
    nuclear_norm_penalty,
)

torch.manual_seed(0)

# Dummy 3D volumes: small, both head shapes. Band mask from a synthetic label.
B, D, H, W = 2, 6, 12, 12
Y = (torch.rand(B, 1, D, H, W) > 0.6).float()
BAND = boundary_band_mask(Y)
assert BAND.shape == Y.shape and BAND.isfinite().all()


def _assert_term_ok(name, term, logits_adv):
    """Forward finite+nonzero; backward finite+nonzero w.r.t. logits_adv."""
    assert torch.isfinite(term).all(), f"{name}: non-finite forward {term}"
    assert term.item() > 0.0, f"{name}: term not strictly positive ({term.item()})"
    term.backward()
    g = logits_adv.grad
    assert g is not None, f"{name}: no gradient flowed to logits_adv"
    assert torch.isfinite(g).all(), f"{name}: non-finite gradient"
    assert g.abs().sum().item() > 0.0, f"{name}: zero gradient"
    print(f"OK: {name}  term={term.item():.6f}  |grad|={g.abs().sum().item():.4f}")


def test_nuclear_norm_penalty():
    for c in (2, 3):  # 2-ch (WG-style) and 3-ch (zones-style) logit widths
        clean = torch.randn(B, c, D, H, W)
        adv = (clean + 0.3 * torch.randn(B, c, D, H, W)).requires_grad_(True)
        term = nuclear_norm_penalty(clean, adv, band_mask=BAND, max_voxels=256)
        _assert_term_ok(f"NuAT[C={c}]", term, adv)
        # band_mask=None path (importance = logit-shift magnitude) must also work
        adv2 = (clean + 0.3 * torch.randn(B, c, D, H, W)).requires_grad_(True)
        t2 = nuclear_norm_penalty(clean, adv2, band_mask=None, max_voxels=256)
        _assert_term_ok(f"NuAT[C={c},no-band]", t2, adv2)


def test_nuclear_norm_degenerate_backward():
    """Regression guard: the nuclear-norm SVD backward must stay finite on
    rank-collapsing inputs (mostly-background 3D patches → near-constant logit
    diff → repeated/zero singular values). Term may be 0; gradient must be finite."""
    c = 3
    clean = torch.randn(B, c, D, H, W)
    cases = {
        "zero-diff": clean.clone(),                                  # adv == clean -> diff 0
        "const-diff": clean + 0.5,                                   # rank-1 diff
        "near-rank1": clean + 0.5 + 1e-6 * torch.randn(B, c, D, H, W),
    }
    for name, adv_val in cases.items():
        adv = adv_val.detach().requires_grad_(True)
        term = nuclear_norm_penalty(clean, adv, band_mask=BAND, max_voxels=256)
        assert torch.isfinite(term).all(), f"NuAT degenerate {name}: non-finite term"
        term.backward()
        assert adv.grad is not None and torch.isfinite(adv.grad).all(), \
            f"NuAT degenerate {name}: non-finite gradient"
        print(f"OK: NuAT-degenerate[{name}]  term={term.item():.4f} grad-finite")


def test_uiat_kl_term():
    # multilabel sigmoid (zones)
    c = 3
    p_inv = torch.sigmoid(torch.randn(B, c, D, H, W))
    adv = torch.randn(B, c, D, H, W, requires_grad=True)
    term = inverse_adversary_kl_term(p_inv, adv, band_mask=BAND, multilabel_sigmoid=True)
    _assert_term_ok("UIAT[sigmoid-multilabel]", term, adv)

    # multiclass softmax (non-region)
    p_inv_mc = torch.softmax(torch.randn(B, c, D, H, W), dim=1)
    adv_mc = torch.randn(B, c, D, H, W, requires_grad=True)
    term_mc = inverse_adversary_kl_term(p_inv_mc, adv_mc, band_mask=BAND,
                                        binary=False, multilabel_sigmoid=False)
    _assert_term_ok("UIAT[softmax-multiclass]", term_mc, adv_mc)

    # binary path (p_clean is foreground prob [B,1], logits index channel 1)
    p_inv_b = torch.sigmoid(torch.randn(B, 1, D, H, W))
    adv_b = torch.randn(B, 2, D, H, W, requires_grad=True)
    term_b = inverse_adversary_kl_term(p_inv_b, adv_b, band_mask=BAND, binary=True)
    _assert_term_ok("UIAT[binary]", term_b, adv_b)


def test_lbgat_logit_mse_term():
    for c in (2, 3):
        ref = torch.randn(B, c, D, H, W)  # frozen reference logits
        rob = (ref + 0.5 * torch.randn(B, c, D, H, W)).requires_grad_(True)
        term = lbgat_logit_mse_term(rob, ref, band_mask=BAND)
        _assert_term_ok(f"LBGAT[C={c}]", term, rob)


def test_inverse_adversary_generator():
    """Generator stays inside the L_inf ball, returns finite detached output, AND
    actually moves the prediction toward the true label (p_inv closer to y than the
    clean prediction is) — guarding against the failure where it descends a ~0
    KL-to-clean objective and leaves p_inv ≈ p_clean (uiat collapsing to trades).
    Uses a fixed Conv3d as the dummy network forward."""
    c = 3
    net = torch.nn.Conv3d(c, c, kernel_size=3, padding=1)
    net.eval()

    def fwd(x):
        return net(x.float())

    x = torch.randn(B, c, D, H, W)
    # multilabel (zones-style) ground-truth mask, one channel per region.
    y = (torch.rand(B, c, D, H, W) > 0.5).float()
    eps = 0.05
    x_inv = inverse_adversary(fwd, x, eps, y, binary=False,
                              inv_steps=3, band_mask=BAND, multilabel_sigmoid=True)
    assert x_inv.shape == x.shape
    assert torch.isfinite(x_inv).all(), "inverse adversary produced non-finite values"
    assert not x_inv.requires_grad, "inverse adversary must be detached"
    max_dx = (x_inv - x).abs().max().item()
    assert max_dx <= eps + 1e-5, "inverse adversary escaped eps ball"

    # The inverse example must AGREE WITH THE LABEL more than the clean input does.
    with torch.no_grad():
        p_clean = torch.sigmoid(fwd(x))
        p_inv = torch.sigmoid(fwd(x_inv))
    agree_clean = (p_clean * y + (1 - p_clean) * (1 - y)).mean().item()
    agree_inv = (p_inv * y + (1 - p_inv) * (1 - y)).mean().item()
    moved = (p_inv - p_clean).abs().mean().item()
    assert agree_inv > agree_clean, \
        f"inverse adversary did not move toward label (agree {agree_clean:.4f}->{agree_inv:.4f})"
    assert moved > 1e-4, "p_inv ≈ p_clean: inverse adversary is a no-op (uiat would collapse to trades)"
    print(f"OK: inverse_adversary  max|dx|={max_dx:.5f}  label-agree {agree_clean:.4f}->{agree_inv:.4f}  |Δp|={moved:.4f}")


def test_corpus_trainer_terms():
    """Each corpus trainer's _corpus_term runs end-to-end on a dummy network
    (exercises the real trainer glue: eval/train toggle, inverse-adversary call,
    extra forwards). LBGAT raises ValueError without a reference checkpoint, and
    computes a finite term with an injected reference network."""
    from types import MethodType, SimpleNamespace
    from nnunetv2.training.nnUNetTrainer.variants.loss.nnUNetTrainerCorpusLoss import (
        nnUNetTrainerNuAT, nnUNetTrainerUIAT, nnUNetTrainerLBGAT,
    )

    c = 3
    net = torch.nn.Conv3d(c, c, kernel_size=3, padding=1)
    net.train()

    def fwd(x):
        return net(x.float())

    x = torch.randn(B, c, D, H, W)
    ctx = dict(
        x_f=x, logits_clean=fwd(x).detach(),
        x_adv=(x + 0.01 * torch.randn_like(x)).detach(),
        y_for_gp=(torch.rand(B, c, D, H, W) > 0.5).float(),
        band_mask=BAND, binary=False, multilabel_sigmoid=True,
    )

    # NuAT and UIAT _corpus_term run end-to-end on a dummy net (real trainer glue:
    # eval/train toggle, inverse-adversary call, extra forwards).
    stub = SimpleNamespace(
        _forward_logits_robust=fwd, network=net,
        nuat_max_voxels=256, inv_steps=2, _eps_curr=0.03,
    )
    t_nuat = nnUNetTrainerNuAT._corpus_term(stub, ctx)
    assert torch.isfinite(t_nuat).all() and t_nuat.item() > 0.0, f"nuat term bad {t_nuat}"
    assert net.training, "nuat: net train mode not restored"
    print(f"OK: NuAT._corpus_term  ={t_nuat.item():.6f}")

    t_uiat = nnUNetTrainerUIAT._corpus_term(stub, ctx)
    assert torch.isfinite(t_uiat).all() and t_uiat.item() > 0.0, f"uiat term bad {t_uiat}"
    assert net.training, "uiat: net train mode not restored"
    print(f"OK: UIAT._corpus_term  ={t_uiat.item():.6f}")

    # LBGAT with no reference configured -> ValueError
    noref = SimpleNamespace(lbgat_reference_checkpoint=None, _reference_network=None)
    noref._ensure_reference_network = MethodType(nnUNetTrainerLBGAT._ensure_reference_network, noref)
    try:
        nnUNetTrainerLBGAT._corpus_term(noref, ctx)
        raise AssertionError("LBGAT must raise without a reference checkpoint")
    except ValueError:
        print("OK: LBGAT._corpus_term  raises ValueError when no reference configured")

    # LBGAT with an injected (dummy) frozen reference -> finite term, grad to robust logits
    ref_net = torch.nn.Conv3d(c, c, kernel_size=3, padding=1).eval()
    for p in ref_net.parameters():
        p.requires_grad_(False)
    lbgat_self = SimpleNamespace(_forward_logits_robust=fwd, _reference_network=ref_net,
                                 lbgat_reference_checkpoint="<injected>")
    lbgat_self._ensure_reference_network = MethodType(
        nnUNetTrainerLBGAT._ensure_reference_network, lbgat_self)
    lbgat_self._reference_logits = MethodType(nnUNetTrainerLBGAT._reference_logits, lbgat_self)
    t_lbgat = nnUNetTrainerLBGAT._corpus_term(lbgat_self, ctx)
    assert torch.isfinite(t_lbgat).all() and t_lbgat.item() > 0.0, f"lbgat term bad {t_lbgat}"
    print(f"OK: LBGAT._corpus_term  ={t_lbgat.item():.6f} (injected dummy reference)")


def test_extra_term_routing():
    """_extra_robust_term: returns 0 (no-op) in the clean phase (_adv_ctx is None),
    and lambda_corpus * corpus_term when context is present. Confirms the base seam
    routes the subclass term into the loss and applies lambda_corpus."""
    from types import MethodType, SimpleNamespace
    from nnunetv2.training.nnUNetTrainer.variants.loss.nnUNetTrainerCorpusLoss import (
        nnUNetTrainerNuAT,
    )

    c = 3
    net = torch.nn.Conv3d(c, c, kernel_size=3, padding=1)
    net.train()

    def fwd(x):
        return net(x.float())

    x = torch.randn(B, c, D, H, W)
    ctx = dict(
        x_f=x, logits_clean=fwd(x).detach(),
        x_adv=(x + 0.01 * torch.randn_like(x)).detach(),
        y_for_gp=(torch.rand(B, c, D, H, W) > 0.5).float(),
        band_mask=BAND, binary=False, multilabel_sigmoid=True,
    )

    # clean phase: no context -> exactly zero, no-op
    clean_stub = SimpleNamespace(_adv_ctx=None, device=torch.device("cpu"))
    z = nnUNetTrainerNuAT._extra_robust_term(clean_stub)
    assert z.item() == 0.0, f"clean-phase extra term must be 0, got {z.item()}"

    # adv phase: lambda_corpus * corpus_term, and lambda actually scales it
    raw = nnUNetTrainerNuAT._corpus_term(
        SimpleNamespace(_forward_logits_robust=fwd, nuat_max_voxels=256), ctx
    ).item()
    adv_stub = SimpleNamespace(
        _adv_ctx=ctx, device=torch.device("cpu"),
        _forward_logits_robust=fwd, nuat_max_voxels=256, lambda_corpus=3.0,
    )
    adv_stub._corpus_term = MethodType(nnUNetTrainerNuAT._corpus_term, adv_stub)
    weighted = nnUNetTrainerNuAT._extra_robust_term(adv_stub).item()
    assert abs(weighted - 3.0 * raw) < 1e-4, f"lambda_corpus not applied: {weighted} vs 3*{raw}"
    print(f"OK: extra-term routing  clean=0.0  adv={weighted:.6f} (=3.0*{raw:.6f})")


if __name__ == "__main__":
    test_nuclear_norm_penalty()
    test_nuclear_norm_degenerate_backward()
    test_uiat_kl_term()
    test_lbgat_logit_mse_term()
    test_inverse_adversary_generator()
    test_corpus_trainer_terms()
    test_extra_term_routing()
    print("\nAll corpus-loss term tests passed.")
