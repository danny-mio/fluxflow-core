"""Init-only robustness fixes for FluxCompressor_v100 / FluxExpander_v100."""

import pytest
import torch

from fluxflow.models.v100.vae import FluxCompressor_v100, FluxExpander_v100

FAMILIES = ["bezier", "pade"]


def _comp(fam: str, d: int = 32) -> FluxCompressor_v100:
    torch.manual_seed(0)
    return FluxCompressor_v100(
        d_model=d, downscales=4, activation_type=fam, use_gradient_checkpointing=False
    ).eval()


def _exp(fam: str, d: int = 32) -> FluxExpander_v100:
    torch.manual_seed(0)
    return FluxExpander_v100(
        d_model=d, upscales=4, activation_type=fam, use_gradient_checkpointing=False
    ).eval()


def test_no_calibrate_init():
    assert not hasattr(FluxCompressor_v100, "_calibrate_init")


@pytest.mark.parametrize("fam", FAMILIES)
def test_initial_sigma_small(fam):
    c = _comp(fam)
    with torch.no_grad():
        _, mu, lv = c(torch.randn(2, 3, 128, 128).clamp(-1, 1), training=True)
    assert torch.isfinite(lv).all() and mu.std() > 0.05
    assert -8.0 < lv.mean() < -4.0
    assert (0.5 * lv).exp().mean() < 0.2


@pytest.mark.parametrize("fam", FAMILIES)
def test_logvar_head_not_saturated_at_init(fam):
    c = _comp(fam)
    with torch.no_grad():
        lv = c.logvar_activation(c.logvar_proj(c.latent_proj(torch.randn(2, 32, 8, 8))))
    assert lv.min() > -9.5 and lv.max() < -3.0


def test_bezier_heads_cover_data_range():
    e = _exp("bezier")
    with torch.no_grad():
        out = e.rgb_activation(torch.randn(2, 3, 64, 64) * 10)
    assert out.max() > 1.0 and out.min() < -1.0
    c = _comp("bezier")
    with torch.no_grad():
        out = c.mu_activation(torch.randn(2, 32, 8, 8) * 10)
    assert out.max() > 1.5 and out.min() < -1.5


def test_bezier_head_controls_monotone():
    for act in (_exp("bezier").rgb_activation, _comp("bezier").mu_activation):
        p = torch.stack([act.p0, act.p1, act.p2, act.p3]).flatten(1)
        assert (p[1:] > p[:-1]).all()


@pytest.mark.parametrize("scale", [0.5, 1.0, 3.0, 10.0])
def test_pade_rgb_head_bounded(scale):
    e = _exp("pade")
    with torch.no_grad():
        out = e.rgb_activation(torch.randn(4, 3, 64, 64) * scale)
    assert out.abs().max() <= 1.3


def test_pade_rgb_head_realistic_preactivations_bounded():
    e = _exp("pade")
    with torch.no_grad():
        out = e.rgb_activation(e.to_rgb_conv(torch.randn(2, e.to_rgb_conv[-1].in_channels, 32, 32)))
    assert out.abs().max() <= 1.3


def test_pade_logvar_init_near_minus_six():
    c = _comp("pade")
    with torch.no_grad():
        lv = c.logvar_activation(torch.zeros(1, 32, 4, 4))
    assert -6.5 < lv.mean() < -5.5


def test_pade_mu_head_identity_at_init():
    c = _comp("pade")
    x = torch.randn(1, 32, 4, 4)
    with torch.no_grad():
        assert torch.allclose(c.mu_activation(x), x, atol=1e-2)


@pytest.mark.parametrize("fam", FAMILIES)
def test_tiny_crop_roundtrip(fam):
    """16x64 px -> latent H==1; reflect padding would crash."""
    c, e = _comp(fam), _exp(fam)
    with torch.no_grad():
        out = e(c(torch.randn(1, 3, 16, 64).clamp(-1, 1)))
    assert out.shape == (1, 3, 16, 64)
    with torch.no_grad():
        out = e(c(torch.randn(1, 3, 64, 16).clamp(-1, 1)))
    assert out.shape == (1, 3, 64, 16)


def test_seam_smoother_keeps_reflect_for_normal_sizes():
    e = _exp("bezier")
    assert e.seam_smoother.padding_mode == "reflect"
    assert e.seam_smoother_ctx.padding_mode == "reflect"


def test_seam_smoother_zero_init_preserved():
    e = _exp("bezier")
    for m in (e.seam_smoother, e.seam_smoother_ctx):
        assert (m.weight == 0).all() and (m.bias == 0).all()


def test_zinject_identity_at_init():
    c = _comp("bezier")
    assert c.ctx_zinject_beta_scale.item() == 0 and c.ctx_zinject_gamma_scale.item() == 0
