"""Init-only robustness fixes for FluxCompressor_v100 / FluxExpander_v100."""

import pytest
import torch

from fluxflow.models.v100.vae import FluxCompressor_v100, FluxExpander_v100

FAMILIES = ["bezier", "pade"]


def _rms(t: torch.Tensor) -> float:
    """Typical magnitude (mean |t|); RMS is outlier-dominated for these activations."""
    return t.float().abs().mean().item()


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


@pytest.mark.parametrize("fam", FAMILIES)
@pytest.mark.parametrize("d", [32, 64])
def test_encoder_signal_preserved(fam, d):
    c = _comp(fam, d)
    img = torch.randn(4, 3, 128, 128).clamp(-1, 1)
    with torch.no_grad():
        x = c.add_coord_channels(img)
        first = None
        for i in range(c.downscales):
            x = c.encoder_first_step[i](x)
            first = _rms(x) if first is None else first
            x = c.encoder_z[i](x)
        cx = c.add_coord_channels(img)
        for i in range(c.downscales):
            cx = c.ctx_encoder_z[i](c.ctx_encoder_first_step[i](cx))
        cp = _rms(c.ctx_proj(cx))
        lat = _rms(c.latent_proj(x))
    for name, v in (("z", _rms(x)), ("ctx_proj", cp), ("latent", lat)):
        assert first / 30 < v < first * 30, (name, v, first)
    assert cp > 1e-3


@pytest.mark.parametrize("fam", FAMILIES)
def test_mu_logvar_not_degenerate(fam):
    c = _comp(fam)
    with torch.no_grad():
        _, mu, lv = c(torch.randn(2, 3, 128, 128).clamp(-1, 1), training=True)
    assert mu.std() > 0.05
    assert -3.0 < lv.mean() < 3.0


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


def test_pade_logvar_init_not_near_deterministic():
    c = _comp("pade")
    with torch.no_grad():
        lv = c.logvar_activation(torch.zeros(1, 32, 4, 4))
    assert -2.0 < lv.mean() < 1.0


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
