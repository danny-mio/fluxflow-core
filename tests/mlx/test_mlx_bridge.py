import pytest

mx = pytest.importorskip("mlx.core")
torch = pytest.importorskip("torch")


def test_gradient_free_bridge_roundtrip():
    """mlx_forward must return a PyTorch tensor with same shape and dtype."""
    from fluxflow.mlx.bridge import mlx_forward

    x = torch.randn(2, 4)

    def identity_mlx(arr: mx.array) -> mx.array:
        return arr

    out = mlx_forward(x, identity_mlx)
    assert out.shape == x.shape
    assert out.dtype == x.dtype


def test_gradient_free_bridge_preserves_values():
    """mlx_forward must faithfully apply the MLX function."""
    from fluxflow.mlx.bridge import mlx_forward

    x = torch.ones(3, 3)
    out = mlx_forward(x, lambda a: a * 2.0)
    assert torch.allclose(out, torch.ones(3, 3) * 2.0)


def test_gradient_free_bridge_no_grad():
    """Output must not require grad (gradient-free contract)."""
    from fluxflow.mlx.bridge import mlx_forward

    x = torch.randn(4, 4, requires_grad=True)
    out = mlx_forward(x, lambda a: a * 3.0)
    assert not out.requires_grad


def test_gradient_free_bridge_device_preserved():
    """Output must be on the same device type as input (CPU for this test)."""
    from fluxflow.mlx.bridge import mlx_forward

    x = torch.randn(2, 2)
    out = mlx_forward(x, lambda a: a)
    assert out.device.type == x.device.type
