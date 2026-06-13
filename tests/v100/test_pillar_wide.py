"""Tests for the widened pillarLayer."""

import torch
import torch.nn as nn

from fluxflow.models.v100.pillar import pillarLayerWide


def test_pillar_wide_shape_in_out():
    """D → hidden → … → D round-trips through the same shape."""
    m = pillarLayerWide(in_size=32, hidden=64, depth=3)
    x = torch.randn(2, 16, 32)
    y = m(x)
    assert y.shape == x.shape


def test_pillar_wide_intermediate_width():
    """Inner Linear layers have hidden width, outer maps in→hidden and hidden→in."""
    m = pillarLayerWide(in_size=32, hidden=64, depth=3)
    linears = [mod for mod in m.modules() if isinstance(mod, nn.Linear)]
    assert len(linears) == 3
    assert linears[0].in_features == 32 and linears[0].out_features == 64
    assert linears[1].in_features == 64 and linears[1].out_features == 64
    assert linears[2].in_features == 64 and linears[2].out_features == 32


def test_pillar_wide_default_hidden_is_2x_in():
    """Default hidden = 2 * in_size."""
    m = pillarLayerWide(in_size=32)
    linears = [mod for mod in m.modules() if isinstance(mod, nn.Linear)]
    assert linears[0].out_features == 64


def test_pillar_wide_gradients_flow():
    """Gradient flows from output through both Linear layers back to input."""
    m = pillarLayerWide(in_size=8, hidden=16, depth=3)
    x = torch.randn(2, 4, 8, requires_grad=True)
    m(x).sum().backward()
    assert x.grad is not None


def test_pillar_wide_depth_two_minimum():
    """At minimum depth=2, function emits exactly 2 Linears (no middle layer)."""
    m = pillarLayerWide(in_size=16, hidden=32, depth=2)
    linears = [mod for mod in m.modules() if isinstance(mod, nn.Linear)]
    assert len(linears) == 2
    assert linears[0].in_features == 16 and linears[0].out_features == 32
    assert linears[1].in_features == 32 and linears[1].out_features == 16


def test_pillar_wide_depth_one_raises():
    """depth=1 raises ValueError (not silently disabled by python -O)."""
    import pytest

    with pytest.raises(ValueError, match="depth must be >= 2"):
        pillarLayerWide(in_size=16, depth=1)
