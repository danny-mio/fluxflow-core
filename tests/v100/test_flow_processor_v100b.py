"""Tests for the redesigned FluxFlowProcessor_v100."""

import inspect
import logging

import pytest
import torch

from fluxflow.models.v100.flow import FluxFlowProcessor_v100, _nearest_hw_factorization
from fluxflow.models.v100.positional import build_axial_rope_2d

_LOGGER_NAME = "fluxflow.models.v100.flow"


def _model(d_model=128, vae_dim=32, n_layers=2, ctx_tokens=4, max_hw=256):
    return FluxFlowProcessor_v100(
        d_model=d_model,
        vae_dim=vae_dim,
        embedding_size=1024,
        n_head=4,
        n_layers=n_layers,
        max_hw=max_hw,
        ctx_tokens=ctx_tokens,
    )


def _packed(B, T, H, W, vae_dim, max_hw, seed=0):
    torch.manual_seed(seed)
    packed = torch.randn(B, T + 1, 2 * vae_dim)
    packed[:, -1, 0] = H / max_hw
    packed[:, -1, 1] = W / max_hw
    return packed


def _text_and_time(B, T_txt=5):
    text_seq = torch.randn(B, T_txt, 1024)
    text_mask = torch.ones(B, T_txt, dtype=torch.bool)
    timesteps = torch.full((B,), 0.5)
    return text_seq, text_mask, timesteps


def test_flow_forward_signature_text_seq_mask():
    """forward signature includes text_seq and text_mask."""
    sig = inspect.signature(FluxFlowProcessor_v100.forward)
    params = list(sig.parameters)
    assert "text_seq" in params
    assert "text_mask" in params
    assert "text_embeddings" not in params


def test_flow_uses_sinusoidal_time():
    """time_mlp is present; old Embedding(1000) time_embed is gone."""
    m = _model()
    assert hasattr(m, "time_mlp")
    assert not hasattr(
        m, "time_embed"
    ), "time_embed (Embedding(1000)) should be replaced by time_mlp"


def test_flow_has_gated_ctx_agg():
    """ctx_gate_proj and ctx_delta_proj exist (GRU-style residual)."""
    m = _model()
    assert hasattr(m, "ctx_gate_proj")
    assert hasattr(m, "ctx_delta_proj")


def test_flow_forward_runs_with_pertoken_text():
    torch.manual_seed(0)
    m = _model(d_model=64, vae_dim=16, n_layers=2).eval()
    D = 16
    T = 8 * 8
    packed = torch.randn(1, T + 1, 2 * D)
    packed[0, -1, 0] = 8 * 16 / 256.0
    packed[0, -1, 1] = 8 * 16 / 256.0
    text_seq = torch.randn(1, 5, 1024)
    text_mask = torch.ones(1, 5, dtype=torch.bool)
    timesteps = torch.tensor([0.5])
    out = m(packed, text_seq, text_mask, timesteps)
    assert out.shape == packed.shape


def test_flow_old_signature_raises_helpful_error():
    """Calling with the old 3-arg form raises a TypeError."""
    m = _model(d_model=64, vae_dim=16, n_layers=2)
    packed = torch.zeros(1, 65, 32)
    text_embeddings_old = torch.randn(1, 1024)
    timesteps = torch.tensor([0.5])
    with pytest.raises(TypeError):
        m(packed, text_embeddings_old, timesteps)


# ---------------------------------------------------------------------------
# Fix C: _nearest_hw_factorization + RoPE-fallback / spatial-post-processing
# hw_vec/T mismatch handling.
# ---------------------------------------------------------------------------


def test_hw_mismatch_fallback_logs_warning(caplog):
    """T=17 with hint (4, 4) is unrecoverable (17 is prime): RoPE pads 1
    trailing token with identity rotation and warns."""
    vae_dim = 16
    m = _model(d_model=64, vae_dim=vae_dim, n_layers=1, max_hw=64).eval()
    packed = _packed(1, 17, 4, 4, vae_dim, max_hw=64)
    text_seq, text_mask, timesteps = _text_and_time(1)
    with caplog.at_level(logging.WARNING, logger=_LOGGER_NAME):
        out = m(packed, text_seq, text_mask, timesteps)
    assert out.shape == packed.shape
    records = [r for r in caplog.records if r.name == _LOGGER_NAME]
    rope_msgs = [r.getMessage() for r in records if "identity RoPE rotation" in r.getMessage()]
    assert len(rope_msgs) == 1
    msg = rope_msgs[0]
    assert "T=17" in msg
    assert "padding 1 trailing" in msg


def test_same_hw_consistent_no_warning(caplog):
    """H*W == T exactly: no fallback, no warning."""
    vae_dim = 16
    m = _model(d_model=64, vae_dim=vae_dim, n_layers=1, max_hw=64).eval()
    packed = _packed(1, 16, 4, 4, vae_dim, max_hw=64)
    text_seq, text_mask, timesteps = _text_and_time(1)
    with caplog.at_level(logging.WARNING, logger=_LOGGER_NAME):
        out = m(packed, text_seq, text_mask, timesteps)
    assert out.shape == packed.shape
    assert [r for r in caplog.records if r.name == _LOGGER_NAME] == []


def test_rope_pad_collides_with_position_zero():
    """Documents the identity-rotation defect precisely: the sin=0/cos=1 pad
    used for unrecoverable trailing tokens is numerically identical to
    position (0, 0)'s real RoPE embedding, so padded tokens are
    indistinguishable from the origin token."""
    head_dim = 16
    h_eff, w_eff = _nearest_hw_factorization(4, 4, 17)
    assert h_eff * w_eff == 16  # unrecoverable: 17 is prime, falls back to sqrt(17)
    sin_w, cos_w, sin_h, cos_h = build_axial_rope_2d(
        h_eff, w_eff, head_dim, device=torch.device("cpu"), dtype=torch.float32
    )
    half = head_dim // 2
    pad_sin = torch.zeros(half)
    pad_cos = torch.ones(half)
    assert torch.equal(pad_sin, sin_w[0])
    assert torch.equal(pad_cos, cos_w[0])
    assert torch.equal(pad_sin, sin_h[0])
    assert torch.equal(pad_cos, cos_h[0])


def test_spatial_postprocessing_truncation_unrecoverable_logs_warning(caplog):
    """T=17 with hint (4, 4) is unrecoverable for spatial post-processing
    too: truncates 1 trailing token and warns, documenting the dropped
    token(s) and the missing ctx_delta update."""
    vae_dim = 16
    m = _model(d_model=64, vae_dim=vae_dim, n_layers=1, max_hw=64).eval()
    packed = _packed(1, 17, 4, 4, vae_dim, max_hw=64)
    text_seq, text_mask, timesteps = _text_and_time(1)
    with caplog.at_level(logging.WARNING, logger=_LOGGER_NAME):
        out = m(packed, text_seq, text_mask, timesteps)
    assert out.shape == packed.shape
    records = [r for r in caplog.records if r.name == _LOGGER_NAME]
    # One warning from the RoPE fallback, one from spatial post-processing.
    spatial_msgs = [r.getMessage() for r in records if "truncating" in r.getMessage()]
    assert len(spatial_msgs) == 1
    msg = spatial_msgs[0]
    assert "T=17" in msg
    assert "truncating 1 trailing" in msg
    assert "ctx_delta" in msg
    assert "NOT receive" in msg


def test_nearest_hw_factorization_resolves_bf16_rounding(caplog):
    """bf16 rounding of hw_vec commonly drifts the recovered (H, W) hint by
    a small amount from the true grid; the shared helper should still find
    the exact factorization of T nearby instead of falling back to a
    lossy sqrt(T) split."""
    # Realistic bf16 round-trip case near the project's max_hw=1024 ceiling:
    # true grid is (801, 8) -> T=6408; storing H/max_hw in bf16 and decoding
    # it back drifts the recovered hint to (800, 8).
    max_hw = 1024
    H0, W0, T = 801, 8, 6408
    hw_vec = torch.tensor([H0 / max_hw, W0 / max_hw], dtype=torch.float32)
    hw_bf16 = hw_vec.to(torch.bfloat16).to(torch.float32)
    h_hint = int(round(hw_bf16[0].item() * max_hw))
    w_hint = int(round(hw_bf16[1].item() * max_hw))
    assert (h_hint, w_hint) != (H0, W0)  # confirm bf16 rounding actually drifted
    h_r, w_r = _nearest_hw_factorization(h_hint, w_hint, T)
    assert (h_r, w_r) == (H0, W0)
    assert h_r * w_r == T

    # Downstream: occurrences 2/3 (spatial post-processing) must no longer
    # truncate for a small-scale analogue of this bf16-rounding mismatch.
    vae_dim = 16
    m = _model(d_model=64, vae_dim=vae_dim, n_layers=1, max_hw=16).eval()
    true_h, true_w, small_T = 8, 8, 64
    hint_h, hint_w = 7, 9  # off by +/-1 from the true grid, T still factors exactly
    assert _nearest_hw_factorization(hint_h, hint_w, small_T) == (true_h, true_w)
    packed = _packed(1, small_T, hint_h, hint_w, vae_dim, max_hw=16)
    text_seq, text_mask, timesteps = _text_and_time(1)
    with caplog.at_level(logging.WARNING, logger=_LOGGER_NAME):
        out = m(packed, text_seq, text_mask, timesteps)
    assert out.shape == packed.shape
    truncation_warnings = [
        r for r in caplog.records if r.name == _LOGGER_NAME and "truncating" in r.getMessage()
    ]
    assert truncation_warnings == []  # t_valid == T: no truncation occurred


def test_forward_strict_raises_on_rope_extra():
    """strict=True hard-errors instead of warn-and-pad when the RoPE
    fallback would need to pad trailing tokens with identity rotation."""
    vae_dim = 16
    m = _model(d_model=64, vae_dim=vae_dim, n_layers=1, max_hw=64).eval()
    packed = _packed(1, 17, 4, 4, vae_dim, max_hw=64)
    text_seq, text_mask, timesteps = _text_and_time(1)
    with pytest.raises(ValueError, match="hw_vec"):
        m(packed, text_seq, text_mask, timesteps, strict=True)


def test_forward_strict_does_not_raise_on_consistent_hw():
    """strict=True is a no-op when hw_vec already matches T exactly."""
    vae_dim = 16
    m = _model(d_model=64, vae_dim=vae_dim, n_layers=1, max_hw=64).eval()
    packed = _packed(1, 16, 4, 4, vae_dim, max_hw=64)
    text_seq, text_mask, timesteps = _text_and_time(1)
    out = m(packed, text_seq, text_mask, timesteps, strict=True)
    assert out.shape == packed.shape
