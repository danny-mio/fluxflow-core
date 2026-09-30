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


def test_z_token_attn_layer_count_matches_attn_layers():
    """z_token_attn is a ModuleList sized by attn_layers (mirrors ctx_token_attn)."""
    m = _model(D=32)
    assert len(m.z_token_attn) == m.attn_layers == 4


def test_z_token_attn_head_fallback_at_small_d_model():
    """At d_model=16 with default attn_heads=8, heads auto-reduce to 2 (head_dim=8)."""
    m = FluxCompressor_v100(
        d_model=16,
        downscales=2,
        ctx_attn_layers=1,
        ctx_attn_heads=2,
        use_gradient_checkpointing=False,
    )
    assert m.z_token_attn[0].attn.num_heads == 2


def test_z_token_attn_head_fallback_capped_by_attn_heads_param():
    """The d_model//16 heuristic is still capped by the explicit attn_heads."""
    m = FluxCompressor_v100(
        d_model=64,
        downscales=2,
        attn_heads=1,
        attn_layers=1,
        ctx_attn_layers=1,
        ctx_attn_heads=2,
        use_gradient_checkpointing=False,
    )
    assert m.z_token_attn[0].attn.num_heads == 1


def test_z_token_attn_head_fallback_decrements_until_divisible():
    """max(2, d_model // 16) must itself evenly divide d_model, else decrement."""
    m = FluxCompressor_v100(
        d_model=17,
        downscales=2,
        attn_layers=1,
        ctx_attn_layers=1,
        ctx_attn_heads=1,
        use_gradient_checkpointing=False,
    )
    # max(2, 17 // 16) == 2, but 17 % 2 != 0 -> falls back to 1.
    assert m.z_token_attn[0].attn.num_heads == 1


def test_z_token_attn_receives_gradient():
    """z_token_attn parameters actually participate in the forward pass."""
    torch.manual_seed(0)
    m = _model(D=32)
    img = torch.randn(1, 3, 256, 256, requires_grad=True)
    _, mu, logvar = m(img, training=True)
    (mu.sum() + logvar.sum()).backward()
    for name, p in m.z_token_attn.named_parameters():
        assert p.grad is not None, f"{name} received no gradient"
        assert torch.isfinite(p.grad).all()


def test_encoder_z_last_stage_kernel_bump():
    """Only the last encoder_z stage uses kernel_size=12/padding=5."""
    m = _model(D=32)
    for i, stage in enumerate(m.encoder_z):
        conv = stage[0]
        if i == m.downscales - 1:
            assert conv.kernel_size == (12, 12)
            assert conv.padding == (5, 5)
        else:
            assert conv.kernel_size == (8, 8)
            assert conv.padding == (3, 3)
        assert conv.stride == (2, 2)


def test_encoder_z_last_stage_kernel_bump_preserves_spatial_size():
    """Kernel bump on the last stage must not change output spatial dims."""
    m = _model(D=32)
    for h, w in [(256, 256), (128, 128), (64, 64)]:
        img = torch.randn(1, 3, h, w)
        _, mu, _ = m(img, training=True)
        assert mu.shape[-2:] == (h // 16, w // 16)


def test_full_forward_d16_with_z_attention():
    """Full forward at the standardized d_model=16 target, with z-path attention."""
    torch.manual_seed(0)
    m = FluxCompressor_v100(
        in_channels=3,
        d_model=16,
        downscales=4,
        max_hw=256,
        ctx_attn_layers=2,
        ctx_attn_heads=2,
        use_gradient_checkpointing=False,
    )
    img = torch.randn(1, 3, 128, 128)
    packed, mu, logvar = m(img, training=True)
    assert mu.shape == (1, 16, 8, 8)
    assert logvar.shape == (1, 16, 8, 8)
    assert packed.shape == (1, 8 * 8 + 1, 32)
    assert torch.isfinite(packed).all()
    assert torch.isfinite(mu).all()
    assert torch.isfinite(logvar).all()
