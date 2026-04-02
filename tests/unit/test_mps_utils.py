import torch
from fluxflow.utils.mps import mps_safe_pool2d


def test_pool2d_divisible_size():
    x = torch.randn(2, 64, 8, 8)
    out = mps_safe_pool2d(x, output_size=(1, 1))
    assert out.shape == (2, 64, 1, 1)


def test_pool2d_non_divisible_size():
    # This is the case that fails on MPS with adaptive_avg_pool2d
    x = torch.randn(2, 64, 7, 7)
    out = mps_safe_pool2d(x, output_size=(1, 1))
    assert out.shape == (2, 64, 1, 1)


def test_pool2d_preserves_values_on_uniform_input():
    x = torch.ones(1, 4, 6, 6) * 3.0
    out = mps_safe_pool2d(x, output_size=(1, 1))
    assert torch.allclose(out, torch.ones(1, 4, 1, 1) * 3.0, atol=1e-5)
