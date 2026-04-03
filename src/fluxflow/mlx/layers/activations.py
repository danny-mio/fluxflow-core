"""MLX implementations of FluxFlow's Bezier activation functions."""

import mlx.core as mx
import mlx.nn as nn


class BezierActivation(nn.Module):
    """
    Input-based Bezier activation: 5→1 channel reduction.

    Expects input with channels divisible by 5.
    Every 5 consecutive channels are interpreted as [t, p0, p1, p2, p3].
    Input layout: [B, C, H, W] (NCHW, matching PyTorch convention).
    """

    def __call__(self, x: mx.array) -> mx.array:
        B, C, H, W = x.shape
        assert C % 5 == 0, f"Channel dimension must be divisible by 5, got {C}"
        F_ = C // 5
        # Reshape: [B, F, 5, H, W] — consecutive 5-channel blocks map to [t,p0,p1,p2,p3]
        x = x.reshape(B, F_, 5, H, W)
        t = x[:, :, 0]
        p0 = x[:, :, 1]
        p1 = x[:, :, 2]
        p2 = x[:, :, 3]
        p3 = x[:, :, 4]
        return _cubic_bezier(t, p0, p1, p2, p3)


class TrainableBezier(nn.Module):
    """
    Per-channel learnable Bezier activation.

    Args:
        channels: Number of channels
        p0, p1, p2, p3: Initial control point values
    """

    def __init__(
        self,
        channels: int,
        p0: float = -1.0,
        p1: float = -0.05,
        p2: float = 0.05,
        p3: float = 1.0,
    ):
        super().__init__()
        self.p0 = mx.ones((channels,)) * p0
        self.p1 = mx.ones((channels,)) * p1
        self.p2 = mx.ones((channels,)) * p2
        self.p3 = mx.ones((channels,)) * p3

    def __call__(self, x: mx.array) -> mx.array:
        # x: [B, C, H, W] — broadcast params over B, H, W
        p0 = self.p0[None, :, None, None]
        p1 = self.p1[None, :, None, None]
        p2 = self.p2[None, :, None, None]
        p3 = self.p3[None, :, None, None]
        return _cubic_bezier(x, p0, p1, p2, p3)


def _cubic_bezier(
    x: mx.array,
    p0: mx.array,
    p1: mx.array,
    p2: mx.array,
    p3: mx.array,
) -> mx.array:
    """Evaluate cubic Bezier at t=sigmoid(x) with given control points."""
    t = mx.sigmoid(x)
    t2 = t * t
    t3 = t2 * t
    ti = 1.0 - t
    ti2 = ti * ti
    ti3 = ti2 * ti
    return ti3 * p0 + 3.0 * ti2 * t * p1 + 3.0 * ti * t2 * p2 + t3 * p3
