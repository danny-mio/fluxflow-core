"""Tests for mask-aware ParallelAttention."""

import torch

from fluxflow.models.v070.flow import ParallelAttention


def _id(x):
    return x


def test_parallel_attention_no_mask_unchanged():
    """attn_mask=None behaves exactly like the old implementation."""
    torch.manual_seed(0)
    attn = ParallelAttention(d_model=16, n_head=4)
    q = torch.randn(2, 5, 16)
    kv = torch.randn(2, 7, 16)
    out_no_mask = attn(q, kv, _id, _id)
    out_default = attn(q, kv, _id, _id, attn_mask=None)
    assert torch.allclose(out_no_mask, out_default)


def test_parallel_attention_mask_blocks_position():
    """Position-3 fully masked → attention output unchanged when V[:, 3] varies."""
    torch.manual_seed(1)
    attn = ParallelAttention(d_model=16, n_head=4)
    q = torch.randn(1, 4, 16)
    kv_a = torch.randn(1, 7, 16)
    kv_b = kv_a.clone()
    kv_b[:, 3, :] = 99.0  # different at position 3 only
    mask = torch.tensor([[True, True, True, False, True, True, True]])
    out_a = attn(q, kv_a, _id, _id, attn_mask=mask)
    out_b = attn(q, kv_b, _id, _id, attn_mask=mask)
    assert torch.allclose(out_a, out_b, atol=1e-5)


def test_parallel_attention_mask_all_true_equals_no_mask():
    """attn_mask all True == no mask."""
    torch.manual_seed(2)
    attn = ParallelAttention(d_model=16, n_head=4)
    q = torch.randn(2, 5, 16)
    kv = torch.randn(2, 7, 16)
    mask = torch.ones(2, 7, dtype=torch.bool)
    out_masked = attn(q, kv, _id, _id, attn_mask=mask)
    out_unmasked = attn(q, kv, _id, _id, attn_mask=None)
    assert torch.allclose(out_masked, out_unmasked, atol=1e-6)


def test_parallel_attention_mask_int_dtype_raises():
    """Passing an int mask (e.g. from HuggingFace tokenizer) raises TypeError."""
    import pytest

    torch.manual_seed(3)
    attn = ParallelAttention(d_model=16, n_head=4)
    q = torch.randn(2, 5, 16)
    kv = torch.randn(2, 7, 16)
    int_mask = torch.ones(2, 7, dtype=torch.int64)
    with pytest.raises(TypeError, match="bool tensor"):
        attn(q, kv, _id, _id, attn_mask=int_mask)
