"""Tests for WideTrainableBezier — wide-range learnable Bezier activation."""

import torch

from fluxflow.models.activations import WideTrainableBezier


def test_wide_trainable_bezier_default_range_init():
    """Default init produces output approximately in [-8, 4]."""
    layer = WideTrainableBezier(shape=(32,), channel_only=True)
    x = torch.randn(4, 32)
    out = layer(x)
    # Default p0=-8, p1=-2, p2=2, p3=4 → cubic Bezier on sigmoid(x) ∈ (0,1)
    # Output bounded by min/max of control points
    assert out.min() >= -8.0
    assert out.max() <= 4.0


def test_wide_trainable_bezier_shape_preserved():
    """Output shape matches input shape."""
    layer = WideTrainableBezier(shape=(32,), channel_only=True)
    x = torch.randn(2, 32, 8, 8)
    out = layer(x)
    assert out.shape == x.shape


def test_wide_trainable_bezier_gradients_flow():
    """All four control points receive gradients."""
    layer = WideTrainableBezier(shape=(8,), channel_only=True)
    x = torch.randn(2, 8, requires_grad=True)
    out = layer(x).sum()
    out.backward()
    for p in [layer.p0, layer.p1, layer.p2, layer.p3]:
        assert p.grad is not None
        assert not torch.allclose(p.grad, torch.zeros_like(p.grad))


def test_wide_trainable_bezier_custom_init():
    """Custom p0..p3 override defaults."""
    layer = WideTrainableBezier(shape=(4,), channel_only=True, p0=-10.0, p1=-5.0, p2=5.0, p3=10.0)
    assert torch.allclose(layer.p0, torch.full((4,), -10.0))
    assert torch.allclose(layer.p1, torch.full((4,), -5.0))
    assert torch.allclose(layer.p2, torch.full((4,), 5.0))
    assert torch.allclose(layer.p3, torch.full((4,), 10.0))
