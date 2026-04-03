"""MLX conditioning modules: SPADE (beta-only) and FiLM."""

import mlx.core as mx
import mlx.nn as nn


class SPADE(nn.Module):
    """
    Beta-only SPADE: out = GroupNorm(x) + beta.

    Context is bilinearly upsampled if spatial dims don't match x.
    Operates on NCHW tensors; transposes internally for MLX Conv2d calls.
    """

    def __init__(self, context_nc: int, num_features: int):
        super().__init__()
        # Find largest num_groups <= 32 that divides num_features
        num_groups = 1
        for ng in range(min(32, num_features), 0, -1):
            if num_features % ng == 0:
                num_groups = ng
                break
        self.gn = nn.GroupNorm(num_groups, num_features, affine=False)
        hidden = 128
        # MLX Conv2d: weights are [out, kH, kW, in] — NHWC
        self.mlp_shared = nn.Sequential(
            nn.Conv2d(context_nc, hidden, kernel_size=3, padding=1),
            nn.ReLU(),
        )
        self.mlp_beta = nn.Conv2d(hidden, num_features, kernel_size=3, padding=1)

    def __call__(self, x: mx.array, context: mx.array | None) -> mx.array:
        # x: [B, C, H, W] (NCHW)
        # GroupNorm in MLX operates on [..., num_features] — transpose to NHWC first
        x_nhwc = mx.transpose(x, (0, 2, 3, 1))
        normalized_nhwc = self.gn(x_nhwc)
        normalized = mx.transpose(normalized_nhwc, (0, 3, 1, 2))  # back to NCHW

        if context is None:
            return normalized

        # Upsample context if needed
        if context.shape[2] != x.shape[2] or context.shape[3] != x.shape[3]:
            # MLX image.interpolate expects NHWC
            ctx_nhwc = mx.transpose(context, (0, 2, 3, 1))
            scale_h = x.shape[2] / context.shape[2]
            scale_w = x.shape[3] / context.shape[3]
            ctx_nhwc = mx.image.interpolate(
                ctx_nhwc, scale_factor=[scale_h, scale_w], mode="bilinear"
            )
            context = mx.transpose(ctx_nhwc, (0, 3, 1, 2))  # back to NCHW

        # Apply convs (MLX conv expects NHWC)
        ctx_nhwc = mx.transpose(context, (0, 2, 3, 1))
        actv = self.mlp_shared(ctx_nhwc)
        beta_nhwc = self.mlp_beta(actv)
        beta = mx.transpose(beta_nhwc, (0, 3, 1, 2))  # NCHW
        return normalized + beta


class FiLM(nn.Module):
    """Feature-wise Linear Modulation for conditioning sequences."""

    def __init__(self, d_in: int, d_feat: int):
        super().__init__()
        self.proj = nn.Linear(d_in, d_feat * 2)

    def __call__(self, x: mx.array, cond: mx.array) -> mx.array:
        # cond: [B, d_in], x: [B, d_feat]
        gamma, beta = mx.split(self.proj(cond), 2, axis=-1)
        return x * (1.0 + gamma) + beta
