"""MLX VAE decoder components for FluxFlow (v070 architecture)."""

import mlx.core as mx
import mlx.nn as nn

from .activations import BezierActivation, TrainableBezier
from .conditioning import SPADE

# Number of context dimensions (matches PyTorch v070)
CONTEXT_DIMS = 5


def _nchw_to_nhwc(x: mx.array) -> mx.array:
    return mx.transpose(x, (0, 2, 3, 1))


def _nhwc_to_nchw(x: mx.array) -> mx.array:
    return mx.transpose(x, (0, 3, 1, 2))


def _nearest_upsample2x(x: mx.array) -> mx.array:
    """Nearest-neighbor 2x spatial upsample on NCHW tensor."""
    # x: [B, C, H, W]
    x = mx.repeat(x, 2, axis=2)  # [B, C, 2H, W]
    x = mx.repeat(x, 2, axis=3)  # [B, C, 2H, 2W]
    return x


class _Conv2dNCHW(nn.Module):
    """Thin wrapper: accepts NCHW, calls MLX Conv2d (NHWC), returns NCHW."""

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int,
        padding: int = 0,
        stride: int = 1,
        dilation: int = 1,
    ):
        super().__init__()
        self.conv = nn.Conv2d(
            in_channels,
            out_channels,
            kernel_size,
            padding=padding,
            stride=stride,
        )
        # MLX Conv2d does not expose dilation in older versions; store and apply manually
        # if dilation > 1 we use nn.Conv2d with dilation kwarg if available, else fall back
        self._dilation = dilation
        if dilation > 1:
            # Re-create conv with dilation
            self.conv = nn.Conv2d(
                in_channels,
                out_channels,
                kernel_size,
                padding=padding,
                stride=stride,
                dilation=dilation,
            )

    def __call__(self, x: mx.array) -> mx.array:
        # x: [B, C, H, W]
        x_nhwc = _nchw_to_nhwc(x)
        out_nhwc = self.conv(x_nhwc)
        return _nhwc_to_nchw(out_nhwc)


class _ConvTranspose2dNCHW(nn.Module):
    """Thin wrapper: accepts NCHW, calls MLX ConvTranspose2d (NHWC), returns NCHW."""

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int,
        stride: int = 1,
        padding: int = 0,
    ):
        super().__init__()
        self.conv = nn.ConvTranspose2d(
            in_channels,
            out_channels,
            kernel_size,
            stride=stride,
            padding=padding,
        )

    def __call__(self, x: mx.array) -> mx.array:
        x_nhwc = _nchw_to_nhwc(x)
        out_nhwc = self.conv(x_nhwc)
        return _nhwc_to_nchw(out_nhwc)


class _GroupNormNCHW(nn.Module):
    """GroupNorm wrapper: accepts NCHW, transposes to NHWC for MLX, returns NCHW."""

    def __init__(self, num_groups: int, num_features: int):
        super().__init__()
        self.gn = nn.GroupNorm(num_groups, num_features)

    def __call__(self, x: mx.array) -> mx.array:
        x_nhwc = _nchw_to_nhwc(x)
        out_nhwc = self.gn(x_nhwc)
        return _nhwc_to_nchw(out_nhwc)


class ResidualUpsampleBlock(nn.Module):
    """
    Residual block with 2x upsampling via transposed convolution.

    Mirrors PyTorch v070 ResidualUpsampleBlock with BezierActivation and SPADE.

    Args:
        channels: Number of input/output channels
        context_size: Context dimensionality for SPADE
        use_spade: Enable SPADE conditioning
    """

    def __init__(self, channels: int, context_size: int = 1024, use_spade: bool = True):
        super().__init__()
        self.use_spade = use_spade

        if self.use_spade:
            self.spade = SPADE(context_size, channels)

        # Transposed convolution upsampling path (mirrors PyTorch conv1 Sequential)
        self.deconv = _ConvTranspose2dNCHW(
            channels, channels * 5, kernel_size=16, stride=2, padding=7
        )
        self.act1 = BezierActivation()
        self.conv = _Conv2dNCHW(
            channels, channels * 5, kernel_size=5, padding=4, stride=1, dilation=2
        )
        self.act2 = BezierActivation()

    def __call__(self, x: mx.array, context: mx.array | None = None) -> mx.array:
        """
        Args:
            x: [B, channels, H, W]
            context: [B, context_size, H', W'] or None

        Returns:
            [B, channels, 2*H, 2*W]
        """
        identity = x

        if self.use_spade:
            x = self.spade(x, context)

        x = self.deconv(x)
        x = self.act1(x)
        x = self.conv(x)
        x = self.act2(x)

        identity_up = _nearest_upsample2x(identity)
        return x + 0.1 * identity_up


class ProgressiveUpscaler(nn.Module):
    """
    Progressive upsampling using stacked ResidualUpsampleBlocks.

    Args:
        channels: Number of channels
        steps: Number of upsampling steps
        context_size: Context dimension for SPADE
        use_spade: Enable SPADE conditioning
    """

    def __init__(
        self, channels: int = 3, steps: int = 2, context_size: int = 1024, use_spade: bool = True
    ):
        super().__init__()
        self.layers = [
            ResidualUpsampleBlock(channels, context_size, use_spade=use_spade) for _ in range(steps)
        ]

    def __call__(self, x: mx.array, context: mx.array | None) -> mx.array:
        for layer in self.layers:
            x = layer(x, context)
        return x


class FluxExpander(nn.Module):
    """
    MLX decoder that expands latent tokens to images via progressive upsampling.

    Mirrors PyTorch v070 FluxExpander architecture.

    Args:
        d_model: Latent dimension (default: 128)
        upscales: Number of 2x upsampling stages (default: 4)
        max_hw: Maximum spatial dimension for denormalization (default: 1024)
    """

    def __init__(self, d_model: int = 128, upscales: int = 4, max_hw: int = 1024):
        super().__init__()
        self.max_hw = max_hw

        self.upscale = ProgressiveUpscaler(
            channels=d_model,
            steps=upscales,
            context_size=CONTEXT_DIMS,
            use_spade=True,
        )

        # Final RGB conversion chain — mirrors PyTorch to_rgb_conv
        self.rgb_conv1 = _Conv2dNCHW(d_model, 96, kernel_size=3, padding=1)
        self.rgb_gn1 = _GroupNormNCHW(8, 96)
        self.rgb_conv2 = _Conv2dNCHW(96, 48, kernel_size=3, padding=1)
        self.rgb_gn2 = _GroupNormNCHW(8, 48)
        self.rgb_conv3 = _Conv2dNCHW(48, 3, kernel_size=1, padding=0)

        # Learnable per-channel (R/G/B) activation curves
        # p0=-1.0, p3=1.0 ensures output range [-1, 1]
        self.rgb_activation = TrainableBezier(
            channels=3,
            p0=-1.0,
            p1=-0.05,
            p2=0.05,
            p3=1.0,
        )

    def _to_rgb(self, x: mx.array) -> mx.array:
        """Apply RGB conversion chain (NCHW throughout)."""
        x = self.rgb_conv1(x)
        x = self.rgb_gn1(x)
        x = nn.silu(x)
        x = self.rgb_conv2(x)
        x = self.rgb_gn2(x)
        x = nn.silu(x)
        x = self.rgb_conv3(x)
        return x

    def unpack(self, packed: mx.array):
        """
        Extract image tokens, context, and spatial dims from packed representation.

        Args:
            packed: [B, T+1, D+CONTEXT_DIMS]

        Returns:
            img_seq: [B, T, D]
            context: [B, T, CONTEXT_DIMS]
            h: int
            w: int
        """
        img_seq_with_context = packed[:, :-1, :]  # [B, T, D+CONTEXT_DIMS]
        img_seq = img_seq_with_context[:, :, :-CONTEXT_DIMS]  # [B, T, D]
        context = img_seq_with_context[:, :, -CONTEXT_DIMS:]  # [B, T, CONTEXT_DIMS]

        hw_token = packed[:, -1, :]  # [B, D+CONTEXT_DIMS]
        # Read h/w from first two elements of HW token, assume batch-uniform dims
        h = int(round(float(hw_token[0, 0].item()) * self.max_hw))
        w = int(round(float(hw_token[0, 1].item()) * self.max_hw))
        h = max(h, 1)
        w = max(w, 1)
        return img_seq, context, h, w

    def __call__(self, packed: mx.array, use_context: bool = True) -> mx.array:
        """
        Args:
            packed: [B, T+1, D+CONTEXT_DIMS]
            use_context: Whether to pass context to SPADE

        Returns:
            [B, 3, H*2^upscales, W*2^upscales]
        """
        img_seq, context, h, w = self.unpack(packed)
        B, L, D = img_seq.shape
        t_valid = h * w
        assert t_valid <= L, f"Mismatch: tokens {L} < h*w {t_valid}"

        # Reshape tokens to spatial feature map: [B, D, h, w]
        feat = img_seq[:, :t_valid, :]  # [B, T, D]
        feat = feat.reshape(B, h, w, D)  # [B, h, w, D]
        feat = mx.transpose(feat, (0, 3, 1, 2))  # [B, D, h, w] NCHW

        # Reshape context for SPADE: [B, CONTEXT_DIMS, h, w]
        if use_context:
            ctx = context[:, :t_valid, :]  # [B, T, CONTEXT_DIMS]
            ctx = ctx.reshape(B, h, w, CONTEXT_DIMS)  # [B, h, w, CONTEXT_DIMS]
            ctx = mx.transpose(ctx, (0, 3, 1, 2))  # [B, CONTEXT_DIMS, h, w] NCHW
        else:
            ctx = None

        upscaled = self.upscale(feat, ctx)
        rgb = self._to_rgb(upscaled)
        return self.rgb_activation(rgb)
