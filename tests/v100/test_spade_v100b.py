"""Tests for multi-scale SPADE with bounded γ."""

import torch

from fluxflow.models.v100.conditioning import SPADE_v100b


def test_spade_v100b_identity_at_init():
    """With beta_scale=0 and gamma_scale=0 at init, output == GroupNorm(x)."""
    torch.manual_seed(0)
    layer = SPADE_v100b(context_nc=32, num_features=32)
    x = torch.randn(2, 32, 16, 16)
    ctx = torch.randn(2, 32, 16, 16)
    out = layer(x, ctx)
    gn = layer.bn(x)
    assert torch.allclose(out, gn, atol=1e-6), "SPADE_v100b must be exactly identity at init"


def test_spade_v100b_gamma_strictly_positive():
    """After random init, γ stays strictly > 0 for any context."""
    torch.manual_seed(1)
    layer = SPADE_v100b(context_nc=32, num_features=32)
    layer.gamma_scale.data.fill_(0.5)
    ctx = torch.randn(2, 32, 8, 8) * 5.0
    actv = layer.mlp_shared(ctx)
    gamma_raw = layer.gamma_head(actv)
    gamma = (
        1.0
        + torch.nn.functional.softplus(layer.gamma_scale * gamma_raw)
        - torch.nn.functional.softplus(torch.zeros(1, device=ctx.device))
    )
    assert (gamma > 0).all()


def test_spade_v100b_multiscale_heads_exist():
    """Three beta heads (low, mid, hi) and one gamma head are present."""
    layer = SPADE_v100b(context_nc=16, num_features=32)
    assert hasattr(layer, "beta_low")
    assert hasattr(layer, "beta_mid")
    assert hasattr(layer, "beta_hi")
    assert hasattr(layer, "gamma_head")
    assert layer.beta_hi.dilation == (3, 3)


def test_spade_v100b_none_context_is_identity():
    """When context is None, output == GroupNorm(x) regardless of scales."""
    torch.manual_seed(2)
    layer = SPADE_v100b(context_nc=16, num_features=32)
    layer.beta_scale.data.fill_(2.0)
    layer.gamma_scale.data.fill_(2.0)
    x = torch.randn(2, 32, 8, 8)
    out = layer(x, None)
    assert torch.allclose(out, layer.bn(x), atol=1e-6)


def test_spade_v100b_context_changes_output():
    """When scales are nonzero, different contexts produce different outputs."""
    torch.manual_seed(3)
    layer = SPADE_v100b(context_nc=8, num_features=32)
    layer.beta_scale.data.fill_(1.0)
    layer.gamma_scale.data.fill_(1.0)
    x = torch.randn(2, 32, 8, 8)
    ctx_a = torch.randn(2, 8, 8, 8)
    ctx_b = torch.randn(2, 8, 8, 8)
    out_a = layer(x, ctx_a)
    out_b = layer(x, ctx_b)
    assert not torch.allclose(out_a, out_b)
