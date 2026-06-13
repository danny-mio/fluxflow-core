"""Tests for 2D axial Rotary Positional Embedding."""

import torch

from fluxflow.models.v100.positional import build_axial_rope_2d


def test_axial_rope_2d_shape():
    """Returns (sin, cos) each of shape [H*W, head_dim]."""
    sin, cos = build_axial_rope_2d(
        H=4, W=6, head_dim=16, device=torch.device("cpu"), dtype=torch.float32
    )
    assert sin.shape == (24, 16)
    assert cos.shape == (24, 16)


def test_axial_rope_2d_head_dim_split():
    """Top head_dim/2 dims encode W positions, bottom half encode H."""
    sin, cos = build_axial_rope_2d(
        H=3, W=3, head_dim=8, device=torch.device("cpu"), dtype=torch.float32
    )
    # Token at (h=0, w=0): both halves at position 0 → sin == 0, cos == 1
    assert torch.allclose(sin[0, :], torch.zeros(8), atol=1e-6)
    assert torch.allclose(cos[0, :], torch.ones(8), atol=1e-6)


def test_axial_rope_2d_spatial_locality():
    """Adjacent (h, w) tokens have smaller RoPE distance than far-apart ones."""
    sin, cos = build_axial_rope_2d(
        H=8, W=8, head_dim=16, device=torch.device("cpu"), dtype=torch.float32
    )
    # Token 0 = (0,0), token 1 = (0,1), token 63 = (7,7)
    dist_adj = torch.norm(torch.cat([sin[0], cos[0]]) - torch.cat([sin[1], cos[1]]))
    dist_far = torch.norm(torch.cat([sin[0], cos[0]]) - torch.cat([sin[63], cos[63]]))
    assert dist_adj < dist_far


def test_axial_rope_2d_head_dim_must_be_divisible_by_4():
    """Raises when head_dim isn't divisible by 4 (each axis needs even dim)."""
    import pytest

    with pytest.raises(AssertionError):
        build_axial_rope_2d(H=2, W=2, head_dim=6, device=torch.device("cpu"), dtype=torch.float32)
