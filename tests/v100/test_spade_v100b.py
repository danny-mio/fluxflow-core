"""Tests for multi-scale SPADE with bounded γ."""

import pytest
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


def test_spade_v100b_scale_drift_zero_at_init():
    """gamma_scale/beta_scale start at 0, so drift is exactly 0 before training."""
    layer = SPADE_v100b(context_nc=16, num_features=32)
    gamma_drift, beta_drift = layer.scale_drift()
    assert gamma_drift == 0.0
    assert beta_drift == 0.0


def test_spade_v100b_scale_drift_tracks_parameter_change():
    """Drift reflects |current - init| once the scalars move away from 0."""
    layer = SPADE_v100b(context_nc=16, num_features=32)
    layer.gamma_scale.data.fill_(0.3)
    layer.beta_scale.data.fill_(-0.2)
    gamma_drift, beta_drift = layer.scale_drift()
    assert gamma_drift == pytest.approx(0.3)
    assert beta_drift == pytest.approx(0.2)


def test_spade_v100b_beta_scale_bounded_via_tanh():
    """beta_scale is unclamped; simulate aggressive drift and verify the
    *effective* scale consumed at forward()'s use-site stays within (-1, 1)
    (gamma_scale left at its zero-init default so gamma==1, isolating beta).
    """
    torch.manual_seed(0)
    layer = SPADE_v100b(context_nc=16, num_features=32)
    layer.beta_scale.data.fill_(8.0)  # fp32 tanh saturates to 1.0 above ~9
    x = torch.randn(2, 32, 8, 8)
    ctx = torch.randn(2, 16, 8, 8)
    with torch.no_grad():
        out = layer(x, ctx)
        normalized = layer.bn(x)
        actv = layer.mlp_shared(ctx)
        beta_raw = layer.beta_low(actv) + layer.beta_mid(actv) + layer.beta_hi(actv)

    assert torch.isfinite(out).all()
    effective_scale = torch.tanh(layer.beta_scale)
    assert effective_scale.abs().item() < 1.0
    torch.testing.assert_close(out, normalized + effective_scale * beta_raw, atol=1e-5, rtol=1e-4)


def test_spade_v100b_gamma_scale_bounded_via_tanh():
    """gamma_scale feeds softplus, which is asymptotically linear (not
    self-bounding); an unclamped gamma_scale can still blow up gamma without
    bound as it drifts. Verify the *effective* scale consumed at forward()'s
    use-site stays within (-1, 1) even after aggressive drift.
    """
    torch.manual_seed(1)
    layer = SPADE_v100b(context_nc=16, num_features=32)
    layer.gamma_scale.data.fill_(-8.0)  # fp32 tanh saturates to -1.0 below ~-9
    x = torch.randn(2, 32, 8, 8)
    ctx = torch.randn(2, 16, 8, 8)
    with torch.no_grad():
        out = layer(x, ctx)
        normalized = layer.bn(x)
        actv = layer.mlp_shared(ctx)
        gamma_raw = layer.gamma_head(actv)
        zero = torch.zeros((), device=x.device, dtype=x.dtype)
        effective_scale = torch.tanh(layer.gamma_scale)
        expected_gamma = (
            1.0
            + torch.nn.functional.softplus(effective_scale * gamma_raw)
            - torch.nn.functional.softplus(zero)
        )

    assert torch.isfinite(out).all()
    assert effective_scale.abs().item() < 1.0
    torch.testing.assert_close(out, expected_gamma * normalized, atol=1e-5, rtol=1e-4)


def test_spade_v100b_scale_drift_survives_state_dict_roundtrip():
    """Init buffers travel with state_dict, so drift stays correct after checkpoint load."""
    layer = SPADE_v100b(context_nc=16, num_features=32)
    layer.gamma_scale.data.fill_(0.5)
    state = layer.state_dict()

    layer2 = SPADE_v100b(context_nc=16, num_features=32)
    layer2.load_state_dict(state)

    gamma_drift, _ = layer2.scale_drift()
    assert gamma_drift == pytest.approx(0.5)
