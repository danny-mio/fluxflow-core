"""Tests for the redesigned compressor z-path."""

import torch

from fluxflow.models.v100.vae import FluxCompressor_v100


def _model(D: int = 32) -> FluxCompressor_v100:
    return FluxCompressor_v100(
        in_channels=3,
        d_model=D,
        downscales=4,
        max_hw=512,
        ctx_attn_layers=2,
        ctx_attn_heads=2,
        use_gradient_checkpointing=False,
    )


def test_compressor_returns_packed_2d_width():
    """Packed token width is 2*D."""
    m = _model(D=32)
    img = torch.randn(1, 3, 256, 256)
    packed = m(img, training=False)
    assert packed.shape == (1, 16 * 16 + 1, 64)


def test_compressor_logvar_uses_wide_bezier():
    """logvar_activation is WideTrainableBezier with wide default range."""
    from fluxflow.models.activations import WideTrainableBezier

    m = _model(D=32)
    assert isinstance(m.logvar_activation, WideTrainableBezier)
    assert m.logvar_activation.p0.min().item() <= -7.0
    assert m.logvar_activation.p3.max().item() >= 3.5


def test_compressor_no_tanh_in_z_tokens():
    """z_tokens output is unbounded (no tanh squash) — values can exceed |1|."""
    torch.manual_seed(0)
    m = _model(D=32)
    img = torch.randn(2, 3, 256, 256) * 5.0
    packed = m(img)
    z_tokens = packed[:, :-1, :32]
    assert (
        z_tokens.abs().max() > 1.0
    ), "z_tokens should be unbounded post-LayerNorm (no tanh squash)"


def test_compressor_source_does_not_reference_pe_content():
    """Source no longer adds `pe_content` in the z forward path."""
    import inspect

    from fluxflow.models.v100 import vae

    src = inspect.getsource(vae)
    assert "+ pe_content" not in src
    assert "pe_content +" not in src


def test_compressor_training_returns_mu_logvar():
    """Training mode returns (packed, mu, logvar) for KL loss."""
    m = _model(D=32)
    img = torch.randn(1, 3, 256, 256)
    out = m(img, training=True)
    assert isinstance(out, tuple) and len(out) == 3
    packed, mu, logvar = out
    assert mu.shape == (1, 32, 16, 16)
    assert logvar.shape == (1, 32, 16, 16)
