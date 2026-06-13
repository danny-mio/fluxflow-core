"""Tests for ctx = f(img, z) conditional coupling."""

import torch

from fluxflow.models.v100.vae import FluxCompressor_v100


def _model(D: int = 32) -> FluxCompressor_v100:
    return FluxCompressor_v100(
        in_channels=3,
        d_model=D,
        downscales=4,
        max_hw=256,
        ctx_attn_layers=2,
        ctx_attn_heads=2,
        use_gradient_checkpointing=False,
    )


def test_ctx_depends_on_z():
    """ctx_features differ when z differs (same image, different samples).

    NB: depends on the z-injector scales being non-zero (they start at 0 at
    init). We force them to a small value to verify the coupling pathway works.
    """
    torch.manual_seed(0)
    m = _model(D=32).eval()
    # Force the z-injector scales away from zero so coupling is exercised.
    m.ctx_zinject_beta_scale.data.fill_(1.0)
    m.ctx_zinject_gamma_scale.data.fill_(0.5)
    img = torch.randn(1, 3, 128, 128)
    torch.manual_seed(1)
    packed_a = m(img)
    torch.manual_seed(2)
    packed_b = m(img)
    ctx_a = packed_a[:, :-1, 32:]
    ctx_b = packed_b[:, :-1, 32:]
    assert not torch.allclose(ctx_a, ctx_b, atol=1e-4)


def test_ctx_injection_identity_at_init():
    """At init (beta_scale=0, gamma_scale=0), ctx output should NOT depend on z."""
    torch.manual_seed(0)
    m = _model(D=32).eval()
    # Confirm zero-init scales.
    assert torch.equal(m.ctx_zinject_beta_scale.data, torch.zeros(1))
    assert torch.equal(m.ctx_zinject_gamma_scale.data, torch.zeros(1))
    img = torch.randn(1, 3, 128, 128)
    torch.manual_seed(1)
    packed_a = m(img)
    torch.manual_seed(2)
    packed_b = m(img)
    ctx_a = packed_a[:, :-1, 32:]
    ctx_b = packed_b[:, :-1, 32:]
    assert torch.allclose(ctx_a, ctx_b, atol=1e-5)


def test_ctx_attn_heads_min_head_dim_at_d32():
    """At D=32 with min head_dim=16, ctx_attn_heads is auto-reduced to 2."""
    m = _model(D=32)
    first = m.ctx_token_attn[0].attn
    assert first.num_heads == 2


def test_ctx_features_shape_unchanged():
    """Ctx tokens still have shape [B, T, D] in packed format."""
    m = _model(D=32).eval()
    img = torch.randn(1, 3, 128, 128)
    packed = m(img)
    T = (128 // 16) ** 2
    assert packed.shape == (1, T + 1, 64)


def test_ctx_no_tanh_unbounded():
    """ctx_tokens are not tanh-squashed; values can exceed |1|.

    Force non-zero injection scales so the coupling pulls ctx away from the
    LN unit-variance ball.
    """
    torch.manual_seed(0)
    m = _model(D=32).eval()
    m.ctx_zinject_beta_scale.data.fill_(5.0)
    m.ctx_zinject_gamma_scale.data.fill_(2.0)
    img = torch.randn(2, 3, 128, 128) * 5.0
    packed = m(img)
    ctx = packed[:, :-1, 32:]
    assert ctx.abs().max() > 1.0
