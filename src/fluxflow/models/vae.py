"""
VAE (Variational Autoencoder) components for FluxFlow.

Contains:
- FluxCompressor: Variational encoder with self-attention
- FluxExpander: Decoder with progressive upsampling
- ResidualUpsampleBlock: Upsampling block with FiLM/SPADE
- ProgressiveUpscaler: Multi-stage upsampler
- Clamp: Output clamping layer
"""

import math
from functools import partial

import torch
import torch.nn as nn
from einops import rearrange
from torch.utils.checkpoint import checkpoint

from .activations import BezierActivation, TrainableBezier
from .conditioning import SPADE, ContextAttentionMixer


class Clamp(nn.Module):
    """Clamps tensor values to specified range."""

    def __init__(self, min=-1, max=1):
        super().__init__()
        self.min = min
        self.max = max

    def forward(self, x):
        return torch.clamp(x, self.min, self.max)


class ResidualUpsampleBlock(nn.Module):
    """
    Residual block with 2x upsampling via transposed convolution.

    Supports SPADE (spatial context) for spatially-adaptive normalization.

    Args:
        channels: Number of input/output channels
        context_size: Context dimensionality for SPADE
        use_spade: Enable SPADE conditioning
    """

    def __init__(self, channels, context_size=1024, use_spade=True):
        super().__init__()
        self.use_spade = use_spade

        if self.use_spade:
            self.spade = SPADE(context_size, channels)

        # Transposed convolution upsampling path
        self.conv1 = nn.Sequential(
            nn.ConvTranspose2d(
                channels, channels * 5, kernel_size=16, stride=2, padding=7
            ),  # doubles H,W
            BezierActivation(t_pre_activation="sigmoid", p_preactivation="silu"),
            nn.Conv2d(channels, channels * 5, kernel_size=5, padding=4, stride=1, dilation=2),
            BezierActivation(t_pre_activation="sigmoid", p_preactivation="silu"),
        )

        # Residual upsampling path - simple nearest neighbor upsampling
        self.skip_upsample = nn.Upsample(scale_factor=2, mode="nearest")

    def forward(self, x, context=None):
        """
        Args:
            x: Input features [B, channels, H, W]
            context: Spatial context for SPADE [B, context_size, H', W']
                     Pass None to disable SPADE conditioning

        Returns:
            Upsampled features [B, channels, 2*H, 2*W]
        """
        # Save input for residual connection
        identity = x

        if self.use_spade and context is not None:
            x = self.spade(x, context)

        # Main convolution path (no checkpoint here, done at higher level)
        x = self.conv1(x)

        # Add scaled residual connection with upsampling
        # Scale down residual to allow SPADE modulation to have more effect
        identity_up = self.skip_upsample(identity)
        return x + 0.1 * identity_up  # Reduced residual weight


class ProgressiveUpscaler(nn.Module):
    """
    Progressive upsampling using stacked ResidualUpsampleBlocks.

    Args:
        channels: Number of channels
        steps: Number of upsampling steps (each doubles resolution)
        context_size: Context dimension for SPADE
        use_spade: Enable SPADE conditioning
    """

    def __init__(
        self,
        channels=3,
        steps=2,
        context_size=1024,
        use_spade=True,
        use_gradient_checkpointing=True,
    ):
        super().__init__()
        self.use_gradient_checkpointing = use_gradient_checkpointing
        self.layers = nn.ModuleList(
            [
                ResidualUpsampleBlock(channels, context_size, use_spade=use_spade)
                for _ in range(steps)
            ]
        )

    def forward(self, x, context):
        """
        Args:
            x: Input [B, channels, H, W]
            context: Spatial context [B, context_size, H, W]
                     Pass same as x for self-modulation, or None to disable

        Returns:
            Upsampled output [B, channels, H*2^steps, W*2^steps]
        """

        # Checkpoint entire upscaling path as one block instead of per-layer
        def upscale_all(x, context):
            # Use provided context for SPADE at each stage
            for layer in self.layers:
                # Pass context for spatial modulation
                x = layer(x, context)
            return x

        if self.use_gradient_checkpointing:
            return checkpoint(partial(upscale_all), x, context, use_reentrant=False)
        else:
            return upscale_all(x, context)


class FluxCompressor(nn.Module):
    """
    Variational encoder with progressive downsampling and self-attention.

    Encodes images to latent tokens with optional KL divergence regularization.
    Uses hybrid positional encoding combining fixed sinusoidal patterns with
    content-based features from the deterministic latent representation.
    Output format: [img_tokens (H*W, D), hw_vector (1, D)]

    Args:
        in_channels: Input image channels (default: 3)
        d_model: Latent dimension (default: 128)
        downscales: Number of 2x downsampling stages (default: 4)
        max_hw: Maximum spatial dimension for normalization (default: 1024)
        use_attention: Enable self-attention over latent tokens
        attn_layers: Number of self-attention layers (default: 2)
        attn_heads: Number of attention heads (default: 8)
        attn_ff_mult: Feed-forward expansion multiplier (default: 2)
        attn_dropout: Attention dropout rate (default: 0.0)
    """

    def __init__(
        self,
        in_channels=3,
        d_model=128,
        downscales=4,
        max_hw=1024,
        use_attention=True,
        attn_layers=4,
        attn_heads=8,
        attn_ff_mult=2,
        attn_dropout=0.0,
        use_gradient_checkpointing=True,
    ):
        super().__init__()
        self.max_hw = max_hw
        self.downscales = downscales
        self.d_model = d_model
        self.attn_layers = attn_layers
        self.use_gradient_checkpointing = use_gradient_checkpointing
        assert d_model % attn_heads == 0, "d_model must be divisible by attn_heads"

        # Progressive channel increase
        input_ch = in_channels + 2  # includes coord channels
        self.stage_channels = [
            int(round(c))
            for c in torch.linspace(input_ch, max(d_model, 8), steps=downscales + 1).tolist()
        ]

        # Encoder (feature evolution & downsample path)
        self.encoder_first_step = nn.ModuleList(
            [
                nn.Sequential(
                    nn.Conv2d(
                        self.stage_channels[i],
                        self.stage_channels[i + 1] * 5,
                        kernel_size=3,
                        stride=1,
                        padding=1,
                        bias=False,
                    ),
                    BezierActivation(t_pre_activation="sigmoid", p_preactivation="silu"),
                )
                for i in range(downscales)
            ]
        )

        self.encoder_z = nn.ModuleList(
            [
                nn.Sequential(
                    nn.Conv2d(
                        self.stage_channels[i + 1],
                        self.stage_channels[i + 1] * 5,
                        kernel_size=8,
                        stride=2,
                        padding=3,
                    ),
                    BezierActivation(t_pre_activation="sigmoid", p_preactivation="silu"),
                )
                for i in range(downscales)
            ]
        )

        # Latent heads
        final_ch = self.stage_channels[-1]
        self.latent_proj = nn.Sequential(
            *[
                nn.Sequential(
                    nn.Conv2d(final_ch, d_model * 5, kernel_size=1),
                    BezierActivation(t_pre_activation="sigmoid", p_preactivation="silu"),
                )
                for _ in range(2)
            ]
        )
        self.mu_proj = nn.Sequential(
            *[
                nn.Sequential(
                    nn.Conv2d(d_model, d_model * 5, kernel_size=1),
                    BezierActivation(t_pre_activation="sigmoid", p_preactivation="silu"),
                )
                for _ in range(2)
            ]
        )
        self.logvar_proj = nn.Sequential(
            *[
                nn.Sequential(
                    nn.Conv2d(d_model, d_model * 5, kernel_size=1),
                    BezierActivation(t_pre_activation="sigmoid", p_preactivation="silu"),
                )
                for _ in range(2)
            ]
        )

        # Per-channel learnable activations for latent bottleneck
        # Each latent channel learns its optimal value range
        # Helps different channels specialize (color vs structure vs texture)
        self.mu_activation = TrainableBezier(
            shape=(d_model,),
            channel_only=True,
            p0=-2.0,  # Wider initial range for latent space
            p1=-0.5,
            p2=0.5,
            p3=2.0,
        )
        self.logvar_activation = TrainableBezier(
            shape=(d_model,),
            channel_only=True,
            p0=-3.0,  # Logvar typically has wider range
            p1=-1.0,
            p2=1.0,
            p3=3.0,
        )

        # Token self-attention blocks
        def ff(dim):
            hidden = dim * attn_ff_mult
            return nn.Sequential(
                nn.Linear(dim, hidden * 5),
                BezierActivation(t_pre_activation="sigmoid", p_preactivation="silu"),
                nn.Linear(hidden, dim),
            )

        class AttnBlock(nn.Module):
            def __init__(self, dim, heads, drop):
                super().__init__()
                self.norm1 = nn.LayerNorm(dim)
                self.attn = nn.MultiheadAttention(
                    embed_dim=dim, num_heads=heads, dropout=drop, batch_first=True
                )
                self.norm2 = nn.LayerNorm(dim)
                self.ff = ff(dim)

            def forward(self, x):
                # x: [B, T, D]
                h = self.norm1(x)
                attn_out, _ = self.attn(h, h, h, need_weights=False)
                x = x + attn_out
                x = x + self.ff(self.norm2(x))
                return x

        self.token_attn = nn.ModuleList(
            [AttnBlock(d_model, attn_heads, attn_dropout) for _ in range(attn_layers)]
        )

        # Fixed 2D sinusoidal positional encoding
        self.register_buffer("_pe_dummy", torch.zeros(1), persistent=False)
        self._pos_cache = {}  # {(H,W,D,device): tensor}

    @staticmethod
    def add_coord_channels(x):
        """Add normalized coordinate channels to input."""
        B, _, H, W = x.shape
        yy, xx = torch.meshgrid(
            torch.linspace(-1, 1, H, device=x.device),
            torch.linspace(-1, 1, W, device=x.device),
            indexing="ij",
        )
        coords = torch.stack([xx, yy], dim=0).unsqueeze(0).expand(B, -1, -1, -1)
        return torch.cat([x, coords], dim=1)

    def _build_2d_sincos_pe(self, H, W, D, device, dtype):
        """Build 2D sinusoidal positional encoding."""
        key = (H, W, D, device)
        if key in self._pos_cache:
            return self._pos_cache[key]

        # Split D across x and y equally
        half = D // 2

        def _pe_1d(L, C):
            pos = torch.arange(L, device=device, dtype=dtype).unsqueeze(1)
            div = torch.exp(
                torch.arange(0, C, 2, device=device, dtype=dtype) * (-math.log(10000.0) / C)
            )
            pe = torch.zeros(L, C, device=device, dtype=dtype)
            pe[:, 0::2] = torch.sin(pos * div)
            pe[:, 1::2] = torch.cos(pos * div)
            return pe  # [L, C]

        Cx = half
        Cy = D - Cx
        pe_x = _pe_1d(W, Cx)  # [W, Cx]
        pe_y = _pe_1d(H, Cy)  # [H, Cy]
        pe_y_expanded = pe_y[:, None, :].expand(H, W, Cy)
        pe_x_expanded = pe_x[None, :, :].expand(H, W, Cx)
        pe = torch.cat([pe_y_expanded, pe_x_expanded], dim=-1).reshape(H * W, D)  # [HW, D]
        self._pos_cache[key] = pe
        return pe

    def reparameterize(self, mu, logvar):
        """Reparameterization trick for VAE."""
        std = (0.5 * logvar).exp()
        eps = torch.randn_like(std)
        return mu + eps * std

    def forward(self, img, training=False):
        """
        Args:
            img: Input image [B, C, H, W]
            training: If True, returns (packed, mu, logvar) for KL loss

        Returns:
            packed: [B, T+1, D] where T=H*W tokens + 1 hw vector
            mu, logvar: (if training=True) [B, D, H_latent, W_latent]
        """
        # Input validation
        assert img.dim() == 4, f"Expected 4D input [B,C,H,W], got {img.dim()}D"
        assert (
            img.shape[1] % 5 == 0 or img.shape[1] == 3
        ), f"Channel dimension must be 3 or divisible by 5, got {img.shape[1]}"

        # Checkpoint entire encoding path as one block
        def encode_block(img):
            x = self.add_coord_channels(img)
            # Progressive downsampling
            for i in range(self.downscales):
                x = self.encoder_first_step[i](x)
                x = self.encoder_z[i](x)
            return x

        if self.use_gradient_checkpointing:
            x = checkpoint(encode_block, img, use_reentrant=False)
        else:
            x = encode_block(img)

        # Latent projections (small operations, no checkpoint needed)
        latent = self.latent_proj(x)
        mu = self.mu_proj(latent)
        logvar = self.logvar_proj(latent)

        # Apply per-channel learnable activation to latent bottleneck
        # Each latent channel learns its optimal encoding range
        mu = self.mu_activation(mu)
        logvar = self.logvar_activation(logvar)

        z = self.reparameterize(mu, logvar)  # [B, D, H, W]

        B, D, H, W = latent.shape
        # Flatten to tokens
        img_seq = z.flatten(2).permute(0, 2, 1)  # [B, T, D], T=H*W

        # Hybrid positional encoding: fixed spatial + content-based
        pe_fixed = self._build_2d_sincos_pe(H, W, D, device=img_seq.device, dtype=img_seq.dtype)
        pe_content = latent.flatten(2).permute(0, 2, 1)  # [B, T, D]
        img_seq = img_seq + pe_fixed.unsqueeze(0) + pe_content

        # Checkpoint attention layers as a group
        def attn_block(seq):
            for blk in self.token_attn:
                seq = blk(seq)
            return seq

        if self.use_gradient_checkpointing:
            img_seq = checkpoint(attn_block, img_seq, use_reentrant=False)
        else:
            img_seq = attn_block(img_seq)

        # HW token (encodes spatial dimensions)
        hw_vec = torch.zeros((B, 1, D), device=z.device, dtype=z.dtype)
        hw_vec[:, 0, 0] = H / float(self.max_hw)
        hw_vec[:, 0, 1] = W / float(self.max_hw)

        packed = torch.cat([img_seq, hw_vec], dim=1)  # [B, T+1, D]

        if training:
            return packed, mu, logvar
        return packed


class FluxExpander(nn.Module):
    """
    Decoder that expands latent tokens to images via progressive upsampling.

    Args:
        d_model: Latent dimension (default: 128)
        upscales: Number of 2x upsampling stages (default: 4)
        max_hw: Maximum spatial dimension for denormalization (default: 1024)
        ctx_tokens: Number of tokens to use for context (default: 4)
    """

    def __init__(
        self, d_model=128, upscales=4, max_hw=1024, ctx_tokens=4, use_gradient_checkpointing=True
    ):
        super().__init__()
        self.max_hw = max_hw
        self.ctx_tokens = ctx_tokens

        self.upscale = ProgressiveUpscaler(
            channels=d_model,
            steps=upscales,
            context_size=d_model,
            use_spade=True,
            use_gradient_checkpointing=use_gradient_checkpointing,
        )

        # Pool context from image tokens
        self.context_mixer = ContextAttentionMixer(
            d_model, n_head=max(4, d_model // 64), use_cls=True
        )

        # Final RGB conversion - CRITICAL FOR COLOR QUALITY
        # Wide channels (96 -> 48) preserve color information
        # TrainableBezier on RGB channels learns per-channel color correction
        self.to_rgb_conv = nn.Sequential(
            nn.Conv2d(d_model, 96, kernel_size=3, padding=1),
            nn.GroupNorm(8, 96),
            nn.SiLU(inplace=True),
            nn.Conv2d(96, 48, kernel_size=3, padding=1),
            nn.GroupNorm(8, 48),
            nn.SiLU(inplace=True),
            nn.Conv2d(48, 3, kernel_size=1, padding=0),
        )

        # Learnable per-channel (R/G/B) activation curves
        # channel_only=True allows variable input resolutions
        # Only 3 parameters per control point → 12 params total
        self.rgb_activation = TrainableBezier(
            shape=(3,),  # R, G, B channels only
            p0=-1.0,  # Output range start
            p3=1.0,  # Output range end
            channel_only=True,
        )

    def unpack(self, packed):
        """Extract image tokens and spatial dimensions from packed representation."""
        img_seq = packed[:, :-1, :].contiguous()  # [B, T, D]
        H = (packed[:, -1, 0] * self.max_hw).round().clamp(min=1).long()
        W = (packed[:, -1, 1] * self.max_hw).round().clamp(min=1).long()
        return img_seq, H, W

    def forward(self, packed, use_context=True):
        """
        Args:
            packed: Latent representation [B, T+1, D]
            use_context: Enable context conditioning (default: True)

        Returns:
            Generated images [B, 3, H, W]
        """
        img_seq, H, W = self.unpack(packed)
        B, L, D = img_seq.shape

        # Check if all samples have same H, W for batch optimization
        if B > 1 and (H == H[0]).all() and (W == W[0]).all():
            # Fast path: all samples have same dimensions, can batch process
            h, w = H[0].item(), W[0].item()
            t_valid = h * w
            assert t_valid <= L, f"Mismatch: tokens {L} < h*w {t_valid}"

            # Batch process all samples
            feat = rearrange(img_seq[:, :t_valid], "b (h w) d -> b d h w", h=h, w=w)

            # Use feat itself as spatial context for SPADE (spatially-adaptive)
            # This provides rich spatial information instead of a pooled vector
            if use_context:
                ctx = feat  # [B, D, h, w] - spatial context
            else:
                ctx = None  # Disable SPADE conditioning entirely

            upscaled = self.upscale(feat, ctx)
            rgb = self.to_rgb_conv(upscaled)
            return self.rgb_activation(rgb)

        else:
            # Slow path: variable dimensions, must process individually
            outputs = []
            for i in range(B):
                h, w = H[i].item(), W[i].item()
                t_valid = h * w
                assert t_valid <= L, f"Mismatch: tokens {L} < h*w {t_valid}"

                feat_i = rearrange(img_seq[i : i + 1, :t_valid], "b (h w) d -> b d h w", h=h, w=w)

                # Use feat_i itself as spatial context for SPADE
                if use_context:
                    ctx_i = feat_i  # [1, D, h, w] - spatial context
                else:
                    ctx_i = None  # Disable SPADE conditioning entirely

                upscaled = self.upscale(feat_i, ctx_i)
                rgb = self.to_rgb_conv(upscaled)
                out = self.rgb_activation(rgb)
                outputs.append(out)

            return torch.cat(outputs, dim=0)
