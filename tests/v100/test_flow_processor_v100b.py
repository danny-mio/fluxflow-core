"""Tests for the redesigned FluxFlowProcessor_v100."""

import inspect

import pytest
import torch

from fluxflow.models.v100.flow import FluxFlowProcessor_v100


def _model(d_model=128, vae_dim=32, n_layers=2, ctx_tokens=4):
    return FluxFlowProcessor_v100(
        d_model=d_model,
        vae_dim=vae_dim,
        embedding_size=1024,
        n_head=4,
        n_layers=n_layers,
        max_hw=256,
        ctx_tokens=ctx_tokens,
    )


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
