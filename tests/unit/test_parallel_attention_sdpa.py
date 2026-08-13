"""Tests for the opt-in SDPA attention backend (fluxflow.models.v070.flow.ParallelAttention)."""

import pytest
import torch

from fluxflow.models.v070.flow import ParallelAttention


def _id(x):
    return x


def _same_weights(d_model, n_head, seed):
    torch.manual_seed(seed)
    einsum_attn = ParallelAttention(d_model, n_head, attn_backend="einsum")
    sdpa_attn = ParallelAttention(d_model, n_head, attn_backend="sdpa")
    sdpa_attn.load_state_dict(einsum_attn.state_dict())
    return einsum_attn, sdpa_attn


def test_sdpa_matches_einsum_no_mask():
    einsum_attn, sdpa_attn = _same_weights(32, 4, seed=0)
    q = torch.randn(2, 5, 32)
    kv = torch.randn(2, 7, 32)
    out_einsum = einsum_attn(q, kv, _id, _id)
    out_sdpa = sdpa_attn(q, kv, _id, _id)
    assert torch.allclose(out_einsum, out_sdpa, atol=1e-4, rtol=1e-4)


def test_sdpa_matches_einsum_with_mask():
    einsum_attn, sdpa_attn = _same_weights(32, 4, seed=1)
    q = torch.randn(1, 4, 32)
    kv = torch.randn(1, 7, 32)
    mask = torch.tensor([[True, True, True, False, True, True, True]])
    out_einsum = einsum_attn(q, kv, _id, _id, attn_mask=mask)
    out_sdpa = sdpa_attn(q, kv, _id, _id, attn_mask=mask)
    assert torch.allclose(out_einsum, out_sdpa, atol=1e-4, rtol=1e-4)


def test_default_backend_is_einsum():
    attn = ParallelAttention(16, 4)
    assert attn.attn_backend == "einsum"


def test_invalid_backend_raises():
    with pytest.raises(ValueError, match="attn_backend"):
        ParallelAttention(16, 4, attn_backend="flash")


def test_state_dict_keys_identical_between_backends():
    """attn_backend must not change parameter shapes/names -- existing
    checkpoints must load under either backend with strict=True."""
    einsum_attn = ParallelAttention(16, 4, attn_backend="einsum")
    sdpa_attn = ParallelAttention(16, 4, attn_backend="sdpa")
    assert set(einsum_attn.state_dict().keys()) == set(sdpa_attn.state_dict().keys())
    for k, v in einsum_attn.state_dict().items():
        assert v.shape == sdpa_attn.state_dict()[k].shape


def test_int_mask_raises_type_error_for_both_backends():
    int_mask = torch.ones(1, 5, dtype=torch.int64)
    q = torch.randn(1, 3, 16)
    kv = torch.randn(1, 5, 16)
    for backend in ("einsum", "sdpa"):
        attn = ParallelAttention(16, 4, attn_backend=backend)
        with pytest.raises(TypeError, match="attn_mask must be a bool tensor"):
            attn(q, kv, _id, _id, attn_mask=int_mask)
