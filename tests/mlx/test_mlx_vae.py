import pytest

mx = pytest.importorskip("mlx.core")


def test_flux_expander_output_shape():
    """FluxExpander must decode packed latent to [B, 3, H*2^upscales, W*2^upscales]."""
    from fluxflow.mlx.layers.vae import FluxExpander

    d_model = 32
    upscales = 2
    max_hw = 64
    dec = FluxExpander(d_model=d_model, upscales=upscales, max_hw=max_hw)

    # Packed: [B, T+1, D] where T=h*w, D=d_model+CONTEXT_DIMS(5)
    # h=2, w=2 spatial tokens at initial resolution
    B, h, w = 1, 2, 2
    T = h * w
    D = d_model + 5  # CONTEXT_DIMS=5
    img_tokens = mx.zeros((B, T, d_model))
    ctx_tokens = mx.zeros((B, T, 5))
    # HW token encodes h/max_hw, w/max_hw
    hw_token = mx.array([[[h / max_hw, w / max_hw] + [0.0] * (D - 2)]])
    img_seq = mx.concatenate([img_tokens, ctx_tokens], axis=-1)
    packed = mx.concatenate([img_seq, hw_token], axis=1)

    out = dec(packed)
    mx.eval(out)
    # 2 upscales: 2x2 → 4x4 → 8x8
    assert out.shape == (B, 3, h * (2**upscales), w * (2**upscales))


def test_flux_expander_output_range():
    """Output must be in [-1, 1] due to TrainableBezier rgb_activation."""
    from fluxflow.mlx.layers.vae import FluxExpander

    d_model = 16
    upscales = 1
    max_hw = 32
    dec = FluxExpander(d_model=d_model, upscales=upscales, max_hw=max_hw)

    B, h, w = 1, 2, 2
    T = h * w
    D = d_model + 5
    img_tokens = mx.random.normal((B, T, d_model))
    ctx_tokens = mx.random.normal((B, T, 5))
    hw_token = mx.array([[[h / max_hw, w / max_hw] + [0.0] * (D - 2)]])
    img_seq = mx.concatenate([img_tokens, ctx_tokens], axis=-1)
    packed = mx.concatenate([img_seq, hw_token], axis=1)

    out = dec(packed)
    mx.eval(out)
    assert float(mx.min(out).item()) >= -1.01
    assert float(mx.max(out).item()) <= 1.01
