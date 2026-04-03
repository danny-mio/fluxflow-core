import pytest

mx = pytest.importorskip("mlx.core")
import numpy as np  # noqa: E402


def test_bezier_activation_output_shape():
    from fluxflow.mlx.layers.activations import BezierActivation

    # Input: [B, C*5, H, W] in NCHW (C groups of 5 channels each)
    x = mx.random.normal((2, 10, 4, 4))  # 2 features × 5 channels
    act = BezierActivation()
    out = act(x)
    mx.eval(out)
    assert out.shape == (2, 2, 4, 4)


def test_trainable_bezier_output_range():
    from fluxflow.mlx.layers.activations import TrainableBezier

    act = TrainableBezier(channels=3, p0=-1.0, p1=-0.05, p2=0.05, p3=1.0)
    x = mx.random.normal((1, 3, 8, 8))
    out = act(x)
    mx.eval(out)
    # With p0=-1, p3=1 and sigmoid(x) in (0,1), output must be in [-1, 1]
    assert float(mx.min(out).item()) >= -1.01
    assert float(mx.max(out).item()) <= 1.01


def test_trainable_bezier_channel_only_broadcasts():
    from fluxflow.mlx.layers.activations import TrainableBezier

    act = TrainableBezier(channels=3, p0=-1.0, p1=-0.05, p2=0.05, p3=1.0)
    x = mx.random.normal((2, 3, 16, 16))
    out = act(x)
    mx.eval(out)
    assert out.shape == (2, 3, 16, 16)


def test_cubic_bezier_matches_pytorch():
    """MLX Bezier output must match PyTorch Bezier within fp32 tolerance."""
    torch = pytest.importorskip("torch")
    from fluxflow.mlx.layers.activations import _cubic_bezier

    # Use fixed values to avoid randomness in comparison
    t_np = np.array([[0.1, 0.5, 0.9], [0.2, 0.7, 0.3]], dtype=np.float32)
    p0_np = np.full_like(t_np, -0.5)
    p1_np = np.full_like(t_np, -0.05)
    p2_np = np.full_like(t_np, 0.05)
    p3_np = np.full_like(t_np, 0.5)

    # PyTorch reference (using sigmoid as t)
    x_torch = torch.tensor(t_np)
    t_torch = torch.sigmoid(x_torch)
    ti = 1.0 - t_torch
    ref = (
        ti**3 * (-0.5)
        + 3 * ti**2 * t_torch * (-0.05)
        + 3 * ti * t_torch**2 * 0.05
        + t_torch**3 * 0.5
    )

    # MLX result
    out_mlx = _cubic_bezier(
        mx.array(t_np),
        mx.array(p0_np),
        mx.array(p1_np),
        mx.array(p2_np),
        mx.array(p3_np),
    )
    mx.eval(out_mlx)
    out_np = np.array(out_mlx)

    np.testing.assert_allclose(out_np, ref.numpy(), atol=1e-5)
