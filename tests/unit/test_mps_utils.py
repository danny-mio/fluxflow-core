import torch
from fluxflow.utils.mps import mps_safe_pool2d


def test_pool2d_divisible_size():
    x = torch.randn(2, 64, 8, 8)
    out = mps_safe_pool2d(x, output_size=(1, 1))
    assert out.shape == (2, 64, 1, 1)


def test_pool2d_non_divisible_size():
    # Non-divisible sizes fail on MPS; this test verifies the happy path on CPU.
    x = torch.randn(2, 64, 7, 7)
    out = mps_safe_pool2d(x, output_size=(1, 1))
    assert out.shape == (2, 64, 1, 1)


def test_pool2d_preserves_values_on_uniform_input():
    x = torch.ones(1, 4, 6, 6) * 3.0
    out = mps_safe_pool2d(x, output_size=(1, 1))
    assert torch.allclose(out, torch.ones(1, 4, 1, 1) * 3.0, atol=1e-5)


def test_pool2d_fallback_branch_via_mock():
    """Exercises the except branch by mocking adaptive_avg_pool2d to raise."""
    import unittest.mock as mock
    from fluxflow.utils import mps as mps_mod

    x = torch.ones(1, 4, 6, 6) * 2.0
    error = RuntimeError("MPS backend does not support non-divisible adaptive pooling")
    with mock.patch.object(mps_mod.F, "adaptive_avg_pool2d", side_effect=error):
        out = mps_safe_pool2d(x, output_size=(1, 1))
    assert out.shape == (1, 4, 1, 1)
    assert torch.allclose(out, torch.tensor([[[[2.0]]] * 4]))


def test_pool2d_fallback_tiled_mean():
    """Fallback tiled mean is exact for divisible sizes."""
    import unittest.mock as mock
    from fluxflow.utils import mps as mps_mod

    x = torch.arange(16, dtype=torch.float32).reshape(1, 1, 4, 4)
    # Expected: 2×2 average pooling → [[2.5, 4.5], [10.5, 12.5]]
    expected = torch.tensor([[[[2.5, 4.5], [10.5, 12.5]]]])
    error = RuntimeError("MPS divisible error")
    with mock.patch.object(mps_mod.F, "adaptive_avg_pool2d", side_effect=error):
        out = mps_safe_pool2d(x, output_size=(2, 2))
    assert torch.allclose(out, expected)
