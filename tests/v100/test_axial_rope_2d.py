"""Tests for 2D axial Rotary Positional Embedding."""

import pytest
import torch

from fluxflow.models.v100.positional import build_axial_rope_2d


def test_axial_rope_2d_returns_four_buffers_with_half_dim():
    """Returns (sin_w, cos_w, sin_h, cos_h) each of shape [H*W, head_dim // 2]."""
    sin_w, cos_w, sin_h, cos_h = build_axial_rope_2d(
        H=4,
        W=6,
        head_dim=16,
        device=torch.device("cpu"),
        dtype=torch.float32,
    )
    expected_shape = (24, 8)
    for buf in (sin_w, cos_w, sin_h, cos_h):
        assert buf.shape == expected_shape


def test_axial_rope_2d_origin_token_has_zero_rotation():
    """Token at (h=0, w=0): all sin == 0 and all cos == 1 (sin(0)=0, cos(0)=1)."""
    sin_w, cos_w, sin_h, cos_h = build_axial_rope_2d(
        H=3,
        W=3,
        head_dim=8,
        device=torch.device("cpu"),
        dtype=torch.float32,
    )
    assert torch.allclose(sin_w[0], torch.zeros(4), atol=1e-6)
    assert torch.allclose(cos_w[0], torch.ones(4), atol=1e-6)
    assert torch.allclose(sin_h[0], torch.zeros(4), atol=1e-6)
    assert torch.allclose(cos_h[0], torch.ones(4), atol=1e-6)


def test_axial_rope_2d_w_axis_independent_of_h():
    """At (h=0, w=1): sin_w is nonzero; sin_h is still zero (h hasn't moved)."""
    sin_w, _cos_w, sin_h, _cos_h = build_axial_rope_2d(
        H=4,
        W=4,
        head_dim=8,
        device=torch.device("cpu"),
        dtype=torch.float32,
    )
    # Token index 1 corresponds to (h=0, w=1).
    assert not torch.allclose(sin_w[1], torch.zeros(4), atol=1e-6)
    assert torch.allclose(sin_h[1], torch.zeros(4), atol=1e-6)


def test_axial_rope_2d_h_axis_independent_of_w():
    """At (h=1, w=0): sin_h is nonzero; sin_w is still zero (w hasn't moved)."""
    sin_w, _cos_w, sin_h, _cos_h = build_axial_rope_2d(
        H=4,
        W=4,
        head_dim=8,
        device=torch.device("cpu"),
        dtype=torch.float32,
    )
    # Token index W=4 corresponds to (h=1, w=0).
    assert torch.allclose(sin_w[4], torch.zeros(4), atol=1e-6)
    assert not torch.allclose(sin_h[4], torch.zeros(4), atol=1e-6)


def test_axial_rope_2d_spatial_locality():
    """Adjacent (h, w) tokens have smaller RoPE distance than far-apart ones.

    Concatenate all four buffers per token so the distance reflects both axes.
    """
    sin_w, cos_w, sin_h, cos_h = build_axial_rope_2d(
        H=8,
        W=8,
        head_dim=16,
        device=torch.device("cpu"),
        dtype=torch.float32,
    )
    feat = torch.cat([sin_w, cos_w, sin_h, cos_h], dim=-1)
    dist_adj = torch.norm(feat[0] - feat[1])
    dist_far = torch.norm(feat[0] - feat[63])
    assert dist_adj < dist_far


def _apply_split_half_rotation(
    x: torch.Tensor, sin: torch.Tensor, cos: torch.Tensor
) -> torch.Tensor:
    """Standard split-half RoPE rotation, matching v070 apply_rotary's
    `x.chunk(2, dim=-1)` convention that build_axial_rope_2d's buffers must
    pair against.
    """
    x1, x2 = x.chunk(2, dim=-1)
    return x * cos + torch.cat((-x2, x1), dim=-1) * sin


def test_axial_rope_2d_rotation_is_orthonormal():
    """Rotating a random vector with the _rope_1d-derived sin/cos buffers
    must preserve its norm (RoPE's core guarantee) — Fix A regression guard
    for the interleaved -> split-half layout correction.
    """
    torch.manual_seed(0)
    H, W, head_dim = 5, 7, 16
    sin_w, cos_w, sin_h, cos_h = build_axial_rope_2d(
        H=H,
        W=W,
        head_dim=head_dim,
        device=torch.device("cpu"),
        dtype=torch.float32,
    )
    half = head_dim // 2
    x = torch.randn(H * W, half)
    for sin_buf, cos_buf in ((sin_w, cos_w), (sin_h, cos_h)):
        rotated = _apply_split_half_rotation(x, sin_buf, cos_buf)
        assert torch.allclose(rotated.norm(dim=-1), x.norm(dim=-1), rtol=1e-4, atol=1e-4)


def test_axial_rope_2d_head_dim_must_be_divisible_by_4():
    """Raises when head_dim isn't divisible by 4 (each half-axis needs even dim)."""
    with pytest.raises(AssertionError):
        build_axial_rope_2d(
            H=2,
            W=2,
            head_dim=6,
            device=torch.device("cpu"),
            dtype=torch.float32,
        )
