"""Tests for the redesigned FluxTransformerBlock_v100."""

import inspect

import torch

from fluxflow.models.v100.flow import FluxTransformerBlock_v100


def _block(d=64, nh=4):
    return FluxTransformerBlock_v100(d_model=d, n_head=nh)


def test_block_forward_signature_has_text_mask_and_time_cond():
    sig = inspect.signature(FluxTransformerBlock_v100.forward)
    assert "text_mask" in sig.parameters
    assert "time_cond" in sig.parameters


def test_block_no_pillar_cross_attn():
    b = _block()
    assert not hasattr(b, "pillar_cross_attn")
    assert not hasattr(b, "norm_pillar")


def test_block_uses_widened_pillar():
    """First Linear in any pillar maps D → 2D (widened)."""
    b = _block(d=64)
    first_linear = next(m for m in b.p0.modules() if isinstance(m, torch.nn.Linear))
    assert first_linear.out_features == 128  # 2 * 64


def test_block_has_dual_film_per_pillar():
    b = _block()
    for i in range(4):
        assert hasattr(b, f"film_p{i}_text")
        assert hasattr(b, f"film_p{i}_time")


def test_block_has_split_norm2():
    b = _block()
    assert hasattr(b, "norm2_q")
    assert hasattr(b, "norm2_kv")
    assert not hasattr(b, "norm2")


def test_block_forward_runs():
    """Forward pass produces correct shape with new signature."""
    torch.manual_seed(0)
    d, nh = 64, 4
    b = _block(d=d, nh=nh)
    B, T_img, T_txt = 1, 16, 5
    img_seq = torch.randn(B, T_img, d)
    text_seq = torch.randn(B, T_txt, d)
    text_mask = torch.ones(B, T_txt, dtype=torch.bool)
    head_dim = d // nh
    half = head_dim // 2
    # Axial buffers for img (each half_dim)
    sin_w_img = torch.zeros(T_img, half)
    cos_w_img = torch.ones(T_img, half)
    sin_h_img = torch.zeros(T_img, half)
    cos_h_img = torch.ones(T_img, half)
    # Text uses 1D RoPE (full head_dim per the rotary helper)
    sin_txt = torch.zeros(T_txt, head_dim)
    cos_txt = torch.ones(T_txt, head_dim)
    text_cond = torch.randn(B, d)
    time_cond = torch.randn(B, d)
    out, p0, p1, p2, p3 = b(
        img_seq,
        text_seq,
        text_mask,
        sin_w_img,
        cos_w_img,
        sin_h_img,
        cos_h_img,
        sin_txt,
        cos_txt,
        None,
        None,
        None,
        None,
        text_cond,
        time_cond,
    )
    assert out.shape == img_seq.shape
    for p in (p0, p1, p2, p3):
        assert p.shape == img_seq.shape
