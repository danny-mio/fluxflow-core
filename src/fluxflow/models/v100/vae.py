"""
VAE components for FluxFlow v0.10.0.

Key changes vs v0.7.0/v0.8.0:
- Independent context branch with full conv stack + self-attention (no shared params with z).
- context_dims == d_model at construction time (no module-level CONTEXT_DIMS constant).
- Packed latent shape: [B, T+1, 2*D] (was [B, T+1, D+5]).
- SPADEWithLearnableScale (beta_scale=0 at init) in the expander.
- z-path token_attn and context_proj removed; replaced by ctx_token_attn in context branch.
- v0.10.0-bezier-coupled: WideTrainableBezier logvar, drop tanh on z_tokens, drop pe_content leak.

Full retraining is required; no weight migration from v0.8.0 is supported.
"""

import math
from typing import cast

import torch
import torch.nn as nn
from einops import rearrange
from torch.utils.checkpoint import checkpoint

from ..activations import BezierActivation, TrainableBezier, WideTrainableBezier, xavier_init
from .conditioning import SPADE_v100b


def _should_checkpoint(args: tuple) -> bool:
    return torch.is_grad_enabled() and any(
        isinstance(a, torch.Tensor) and a.requires_grad for a in args
    )


def _maybe_checkpoint(fn, *args, **kwargs):
    if _should_checkpoint(args):
        return checkpoint(fn, *args, **kwargs)
    kwargs.pop("use_reentrant", None)
    return fn(*args, **kwargs)


# ---------------------------------------------------------------------------
# Shared helpers (mirrors v070/vae.py but not imported to keep versions clean)
# ---------------------------------------------------------------------------


class _ResidualUpsampleBlock(nn.Module):
    """
    Residual block with 2x upsampling — v0.10.0 variant using SPADEWithLearnableScale.

    Args:
        channels: Number of input/output channels
        context_size: Context dimensionality for SPADE (= d_model in v0.10.0)
        use_spade: Enable SPADE conditioning
    """

    def __init__(self, channels: int, context_size: int = 1024, use_spade: bool = True) -> None:
        super().__init__()
        self.use_spade = use_spade
        if self.use_spade:
            self.spade = SPADE_v100b(context_size, channels)

        self.conv1 = nn.Sequential(
            nn.ConvTranspose2d(channels, channels * 5, kernel_size=16, stride=2, padding=7),
            BezierActivation(t_pre_activation="tanh", p_preactivation="silu"),
            nn.Conv2d(channels, channels * 5, kernel_size=5, padding=4, stride=1, dilation=2),
            BezierActivation(t_pre_activation="tanh", p_preactivation="silu"),
        )
        self.skip_upsample = nn.Upsample(scale_factor=2, mode="nearest")
        self.conv1_scale = nn.Parameter(torch.zeros(1))

    def forward(self, x: torch.Tensor, context: torch.Tensor | None = None) -> torch.Tensor:
        """
        Args:
            x: Input features [B, channels, H, W]
            context: Spatial context for SPADE [B, context_size, H', W'] or None

        Returns:
            torch.Tensor: Upsampled features [B, channels, 2*H, 2*W]
        """
        identity = x
        if self.use_spade:
            x = self.spade(x, context)
        x = self.conv1(x)
        identity_up = self.skip_upsample(identity)
        # conv1_scale is unclamped (see __init__); tanh at the use-site bounds
        # the effective scale to (-1, 1) so this branch can never out-grow the
        # identity path, while tanh(0)==0 keeps identity-at-init exact and
        # d/dx tanh(x)|_0==1 keeps gradient flow at init unaffected.
        return cast(torch.Tensor, identity_up + torch.tanh(self.conv1_scale) * x)


class _ProgressiveUpscaler(nn.Module):
    """
    Progressive upsampling using stacked _ResidualUpsampleBlocks (v0.10.0 variant).

    Args:
        channels: Number of channels
        steps: Number of upsampling steps (each doubles resolution)
        context_size: Context dimension for SPADE
        use_spade: Enable SPADE conditioning
    """

    def __init__(
        self,
        channels: int = 3,
        steps: int = 2,
        context_size: int = 1024,
        use_spade: bool = True,
        use_gradient_checkpointing: bool = True,
    ) -> None:
        super().__init__()
        self.use_gradient_checkpointing = use_gradient_checkpointing
        self.layers = nn.ModuleList(
            [
                _ResidualUpsampleBlock(channels, context_size, use_spade=use_spade)
                for _ in range(steps)
            ]
        )

    def forward(self, x: torch.Tensor, context: torch.Tensor | None) -> torch.Tensor:
        """
        Args:
            x: Input [B, channels, H, W]
            context: Spatial context [B, context_size, H, W] or None

        Returns:
            torch.Tensor: Upsampled output [B, channels, H*2^steps, W*2^steps]
        """
        if self.use_gradient_checkpointing and x.requires_grad:
            for layer in self.layers:
                x = _maybe_checkpoint(layer, x, context, use_reentrant=True)
            return x
        for layer in self.layers:
            x = layer(x, context)
        return x


# ---------------------------------------------------------------------------
# Attention block (mirrors v070/vae.py AttnBlock, defined at module level for pickling)
# ---------------------------------------------------------------------------


class _AttnBlock(nn.Module):
    """
    Self-attention block with pre-norm and feed-forward.

    Matches the AttnBlock idiom in v070/vae.py:314-330.

    Args:
        dim: Token dimension
        heads: Number of attention heads
        drop: Attention dropout rate
        ff_mult: Feed-forward expansion multiplier
    """

    def __init__(self, dim: int, heads: int, drop: float = 0.0, ff_mult: int = 2) -> None:
        super().__init__()
        hidden = dim * ff_mult
        self.norm1 = nn.LayerNorm(dim)
        self.attn = nn.MultiheadAttention(
            embed_dim=dim, num_heads=heads, dropout=drop, batch_first=True
        )
        self.norm2 = nn.LayerNorm(dim)
        self.ff = nn.Sequential(
            nn.Linear(dim, hidden * 5),
            BezierActivation(t_pre_activation="sigmoid", p_preactivation="silu"),
            nn.Linear(hidden, dim),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: Token sequence [B, T, D]

        Returns:
            torch.Tensor: Attended tokens [B, T, D]
        """
        h = self.norm1(x)
        attn_out, _ = self.attn(h, h, h, need_weights=False)
        x = x + attn_out
        x = x + self.ff(self.norm2(x))
        return x


# ---------------------------------------------------------------------------
# FluxCompressor_v100
# ---------------------------------------------------------------------------


class FluxCompressor_v100(nn.Module):
    """
    Variational encoder with independent context branch for FluxFlow v0.10.0.

    Differences vs FluxCompressor (v0.7.0/v0.8.0):
    - Independent context branch (ctx_encoder_first_step, ctx_encoder_z, ctx_proj,
      ctx_token_attn, ctx_final_norm) with no shared parameters with the z path.
    - context_dims == d_model (not a fixed 5-dimensional constant).
    - z-path token_attn and context_proj removed.
    - Packed output shape: [B, T+1, 2*D] (was [B, T+1, D+5]).

    Args:
        in_channels: Input image channels (default: 3)
        d_model: Latent dimension (default: 128)
        downscales: Number of 2x downsampling stages (default: 4)
        max_hw: Maximum spatial dimension for normalization (default: 1024)
        attn_layers: Number of z-path attention layers (kept for API compat, unused in v0.10.0)
        attn_heads: Number of attention heads in z-path (kept for API compat, unused in v0.10.0)
        attn_ff_mult: Feed-forward multiplier (default: 2)
        attn_dropout: Attention dropout rate (default: 0.0)
        ctx_attn_layers: Number of context branch attention layers (default: 4)
        ctx_attn_heads: Number of context branch attention heads (default: 8)
        use_attention: Retained for API compatibility (not used; context branch always has attn)
        use_gradient_checkpointing: Enable gradient checkpointing (default: True)
    """

    def get_context_dims(self) -> int:
        """Return context dimensions (always equals d_model in v0.10.0)."""
        return self.d_model

    def get_downscales(self) -> int:
        """Return the number of downsampling levels used."""
        return int(self.downscales)

    @classmethod
    def get_default_downscales(cls) -> int:
        """Return the default number of downsampling levels."""
        return 4

    def __init__(
        self,
        in_channels: int = 3,
        d_model: int = 128,
        downscales: int = 4,
        max_hw: int = 1024,
        use_attention: bool = True,  # kept for API compat
        attn_layers: int = 4,  # kept for API compat (unused)
        attn_heads: int = 8,  # kept for API compat (unused)
        attn_ff_mult: int = 2,
        attn_dropout: float = 0.0,
        ctx_attn_layers: int = 4,
        ctx_attn_heads: int = 8,
        use_gradient_checkpointing: bool = True,
    ) -> None:
        super().__init__()
        self.max_hw = max_hw
        self.downscales = downscales
        self.d_model = d_model
        self.attn_layers = attn_layers  # stored for metadata detection
        self.use_gradient_checkpointing = use_gradient_checkpointing

        assert d_model % ctx_attn_heads == 0, "d_model must be divisible by ctx_attn_heads"

        # ---- z path (identical to v070 except token_attn + context_proj removed) ----
        input_ch = in_channels + 2  # +2 coord channels
        self.stage_channels = [
            int(round(c))
            for c in torch.linspace(input_ch, max(d_model, 8), steps=downscales + 1).tolist()
        ]

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
                    BezierActivation(t_pre_activation="tanh", p_preactivation="silu"),
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
                    BezierActivation(t_pre_activation="tanh", p_preactivation="silu"),
                )
                for i in range(downscales)
            ]
        )

        final_ch = self.stage_channels[-1]
        self.latent_proj = nn.Sequential(
            *[
                nn.Sequential(
                    nn.Conv2d(final_ch, d_model * 5, kernel_size=1),
                    BezierActivation(t_pre_activation="tanh", p_preactivation="silu"),
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

        self.mu_activation = TrainableBezier(
            shape=(d_model,), channel_only=True, p0=-0.5, p1=-0.1, p2=0.1, p3=0.5
        )
        self.logvar_activation = WideTrainableBezier(
            shape=(d_model,),
            channel_only=True,
            p0=-8.0,
            p1=-2.0,
            p2=2.0,
            p3=4.0,
        )

        self.final_norm = nn.LayerNorm(d_model)

        # ---- context branch (independent from z) ----
        self.ctx_stage_channels = [
            int(round(c))
            for c in torch.linspace(input_ch, max(d_model, 8), steps=downscales + 1).tolist()
        ]

        self.ctx_encoder_first_step = nn.ModuleList(
            [
                nn.Sequential(
                    nn.Conv2d(
                        self.ctx_stage_channels[i],
                        self.ctx_stage_channels[i + 1] * 5,
                        kernel_size=3,
                        stride=1,
                        padding=1,
                        bias=False,
                    ),
                    BezierActivation(t_pre_activation="tanh", p_preactivation="silu"),
                )
                for i in range(downscales)
            ]
        )

        self.ctx_encoder_z = nn.ModuleList(
            [
                nn.Sequential(
                    nn.Conv2d(
                        self.ctx_stage_channels[i + 1],
                        self.ctx_stage_channels[i + 1] * 5,
                        kernel_size=8,
                        stride=2,
                        padding=3,
                    ),
                    BezierActivation(t_pre_activation="tanh", p_preactivation="silu"),
                )
                for i in range(downscales)
            ]
        )

        ctx_final_ch = self.ctx_stage_channels[-1]
        self.ctx_proj = nn.Sequential(
            *[
                nn.Sequential(
                    nn.Conv2d(ctx_final_ch, d_model * 5, kernel_size=1),
                    BezierActivation(t_pre_activation="tanh", p_preactivation="silu"),
                )
                for _ in range(2)
            ]
        )

        # ---- z → ctx coupling: SPADE-style injector at the bottleneck ----
        # γ_z = 1 + softplus(...); β_z = Conv1×1. Identity at init via zero-init
        # γ_inj_scale / β_inj_scale so untrained ctx behaviour matches v0.10.0-pre
        # for warm-start compatibility (M8 salvage script).
        self.ctx_zinject_proj = nn.Sequential(
            nn.Conv2d(d_model, d_model * 5, kernel_size=1),
            BezierActivation(t_pre_activation="tanh", p_preactivation="silu"),
        )
        self.ctx_zinject_beta = nn.Conv2d(d_model, d_model, kernel_size=1)
        self.ctx_zinject_gamma = nn.Conv2d(d_model, d_model, kernel_size=1)
        self.ctx_zinject_beta_scale = nn.Parameter(torch.zeros(1))
        self.ctx_zinject_gamma_scale = nn.Parameter(torch.zeros(1))
        self.ctx_zinject_norm = nn.GroupNorm(
            num_groups=min(32, d_model),
            num_channels=d_model,
            affine=False,
        )

        # Auto-fallback so head_dim stays ≥ 16 (silent quality killer at D=32).
        # With ctx_attn_heads=8 and d_model=32, head_dim = 4 — noise-only.
        # Reduce heads until head_dim ≥ 16 AND d_model is divisible by heads.
        effective_ctx_heads = max(2, d_model // 16)
        effective_ctx_heads = min(effective_ctx_heads, ctx_attn_heads)
        while d_model % effective_ctx_heads != 0 and effective_ctx_heads > 1:
            effective_ctx_heads -= 1
        self.ctx_token_attn = nn.ModuleList(
            [
                _AttnBlock(d_model, effective_ctx_heads, attn_dropout, attn_ff_mult)
                for _ in range(ctx_attn_layers)
            ]
        )
        self.ctx_final_norm = nn.LayerNorm(d_model)

        # Fixed PE cache
        self.register_buffer("_pe_dummy", torch.zeros(1), persistent=False)
        self._pos_cache: dict = {}

        # Established codebase convention (conditioning.py, encoders.py, every
        # flow.py) -- PyTorch's raw default Conv2d init was never validated
        # against this architecture's deep unnormalized Conv2d->BezierActivation
        # stacks and produces NaN/Inf on real image input from a fresh (no
        # checkpoint) random init, especially under fp16's narrow dynamic range.
        # Only touches Conv2d/Conv3d/Linear/ConvTranspose2d; the zero-init
        # ctx_zinject_beta_scale/ctx_zinject_gamma_scale nn.Parameters above are
        # bare Parameters (not Modules), so .apply() never visits them --
        # placement here is safe regardless of order.
        self.apply(xavier_init)

    @staticmethod
    def add_coord_channels(x: torch.Tensor) -> torch.Tensor:
        """Add normalized coordinate channels to input."""
        B, _, H, W = x.shape
        yy, xx = torch.meshgrid(
            torch.linspace(-1, 1, H, device=x.device),
            torch.linspace(-1, 1, W, device=x.device),
            indexing="ij",
        )
        coords = torch.stack([xx, yy], dim=0).unsqueeze(0).expand(B, -1, -1, -1)
        return torch.cat([x, coords], dim=1)

    def _build_2d_sincos_pe(
        self, H: int, W: int, D: int, device: torch.device, dtype: torch.dtype
    ) -> torch.Tensor:
        """Build 2D sinusoidal positional encoding [H*W, D]."""
        key = (H, W, D, device)
        if key in self._pos_cache:
            return cast(torch.Tensor, self._pos_cache[key])

        half = D // 2

        def _pe_1d(L: int, C: int) -> torch.Tensor:
            pos = torch.arange(L, device=device, dtype=dtype).unsqueeze(1)
            div = torch.exp(
                torch.arange(0, C, 2, device=device, dtype=dtype) * (-math.log(10000.0) / C)
            )
            pe = torch.zeros(L, C, device=device, dtype=dtype)
            pe[:, 0::2] = torch.sin(pos * div)
            pe[:, 1::2] = torch.cos(pos * div)
            return pe

        Cx = half
        Cy = D - Cx
        pe_x = _pe_1d(W, Cx)
        pe_y = _pe_1d(H, Cy)
        pe_y_expanded = pe_y[:, None, :].expand(H, W, Cy)
        pe_x_expanded = pe_x[None, :, :].expand(H, W, Cx)
        pe = torch.cat([pe_y_expanded, pe_x_expanded], dim=-1).reshape(H * W, D)
        self._pos_cache[key] = pe
        return pe

    def reparameterize(self, mu: torch.Tensor, logvar: torch.Tensor) -> torch.Tensor:
        """Reparameterization trick for VAE."""
        std = (0.5 * logvar).exp()
        eps = torch.randn_like(std)
        return mu + eps * std

    def forward(
        self, img: torch.Tensor, training: bool = False
    ) -> torch.Tensor | tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Encode image to packed latent representation.

        Args:
            img: Input image [B, C, H, W]
            training: If True, returns (packed, mu, logvar) for KL loss

        Returns:
            packed: [B, T+1, 2*D] where T = H_lat * W_lat tokens + 1 HW vector
            mu, logvar: (if training=True) [B, D, H_latent, W_latent]
        """
        assert img.dim() == 4, f"Expected 4D input [B,C,H,W], got {img.dim()}D"

        # ---- z branch ----
        def encode_block(img: torch.Tensor) -> torch.Tensor:
            x = self.add_coord_channels(img)
            for i in range(self.downscales):
                x = self.encoder_first_step[i](x)
                x = self.encoder_z[i](x)
            return x

        if self.use_gradient_checkpointing and img.requires_grad:
            x = _maybe_checkpoint(encode_block, img, use_reentrant=True)
        else:
            x = encode_block(img)

        latent = self.latent_proj(x)
        mu = self.mu_activation(self.mu_proj(latent))
        logvar = self.logvar_activation(self.logvar_proj(latent))
        z = self.reparameterize(mu, logvar)  # [B, D, H_lat, W_lat]

        # ---- z path: clean Gaussian, no tanh, no pe_content leak ----
        # See docs/plans/2026-06-13-v0.10.0-redesign-design.md §2.1
        B, D, H, W = z.shape
        pe_fixed = self._build_2d_sincos_pe(H, W, D, device=z.device, dtype=z.dtype)
        z_raw = z.flatten(2).permute(0, 2, 1) + pe_fixed.unsqueeze(0)
        # NB: pe_content REMOVED (was a deterministic skip around the bottleneck).
        z_tokens = self.final_norm(z_raw)
        # NB: tanh REMOVED. LayerNorm gives unit variance; squashing the tails
        # is what limited sharpness in v0.10.0-pre.

        # ---- context branch: conditional on z, no tanh ----
        def ctx_encode_block(img: torch.Tensor) -> torch.Tensor:
            cx = self.add_coord_channels(img)
            for i in range(self.downscales):
                cx = self.ctx_encoder_first_step[i](cx)
                cx = self.ctx_encoder_z[i](cx)
            return cx

        if self.use_gradient_checkpointing and img.requires_grad:
            cx = _maybe_checkpoint(ctx_encode_block, img, use_reentrant=True)
        else:
            cx = ctx_encode_block(img)

        cx = self.ctx_proj(cx)  # [B, D, H_lat, W_lat]

        # z → ctx coupling: condition ctx on the sampled z.
        z_proj = self.ctx_zinject_proj(z)  # [B, D, H, W]
        # ctx_zinject_beta_scale/ctx_zinject_gamma_scale are unclamped bare
        # Parameters (see __init__); tanh at the use-site bounds the effective
        # scale to (-1, 1) — identical rationale to SPADE_v100b.beta_scale/
        # gamma_scale in conditioning.py.
        beta_z = self.ctx_zinject_beta(z_proj) * torch.tanh(self.ctx_zinject_beta_scale)
        gamma_z = (
            1.0
            + torch.nn.functional.softplus(
                torch.tanh(self.ctx_zinject_gamma_scale) * self.ctx_zinject_gamma(z_proj)
            )
            - torch.nn.functional.softplus(torch.zeros((), device=z.device, dtype=z.dtype))
        )
        cx = gamma_z * self.ctx_zinject_norm(cx) + beta_z

        ctx_tokens = cx.flatten(2).permute(0, 2, 1)  # [B, T, D]

        def ctx_attn_block(seq: torch.Tensor) -> torch.Tensor:
            for blk in self.ctx_token_attn:
                seq = blk(seq)
            return seq

        if self.use_gradient_checkpointing and ctx_tokens.requires_grad:
            attended_ctx = _maybe_checkpoint(ctx_attn_block, ctx_tokens, use_reentrant=True)
        else:
            attended_ctx = ctx_attn_block(ctx_tokens)

        context_tokens = self.ctx_final_norm(attended_ctx)  # NO tanh

        # ---- pack ----
        img_seq_with_context = torch.cat([z_tokens, context_tokens], dim=-1)  # [B, T, 2D]
        hw_vec = torch.zeros((B, 1, D + D), device=z.device, dtype=z.dtype)
        hw_vec[:, 0, 0] = H / float(self.max_hw)
        hw_vec[:, 0, 1] = W / float(self.max_hw)
        packed = torch.cat([img_seq_with_context, hw_vec], dim=1)  # [B, T+1, 2D]

        if training:
            return packed, mu, logvar
        return packed


# ---------------------------------------------------------------------------
# FluxExpander_v100
# ---------------------------------------------------------------------------


class FluxExpander_v100(nn.Module):
    """
    Decoder that expands v0.10.0 packed latent tokens to images.

    Differences vs FluxExpander (v0.7.0/v0.8.0):
    - Expects packed [B, T+1, 2*D] (was [B, T+1, D+5]).
    - Uses SPADEWithLearnableScale (beta_scale=0 at init).
    - context_size = d_model (not CONTEXT_DIMS=5).

    Args:
        d_model: Latent dimension (default: 128)
        upscales: Number of 2x upsampling stages (default: 4)
        max_hw: Maximum spatial dimension for denormalization (default: 1024)
        ctx_tokens: Number of tokens used for context (default: 4, for API compat)
        use_gradient_checkpointing: Enable gradient checkpointing (default: True)
    """

    def __init__(
        self,
        d_model: int = 128,
        upscales: int = 4,
        max_hw: int = 1024,
        ctx_tokens: int = 4,
        use_gradient_checkpointing: bool = True,
    ) -> None:
        super().__init__()
        self.max_hw = max_hw
        self.d_model = d_model

        # v0.10.0: context_size = d_model (not the old CONTEXT_DIMS=5)
        self.upscale = _ProgressiveUpscaler(
            channels=d_model,
            steps=upscales,
            context_size=d_model,
            use_spade=True,
            use_gradient_checkpointing=use_gradient_checkpointing,
        )

        # RGB conversion identical to v070
        self.to_rgb_conv = nn.Sequential(
            nn.Conv2d(d_model, 96, kernel_size=3, padding=1),
            nn.GroupNorm(8, 96),
            nn.SiLU(inplace=True),
            nn.Conv2d(96, 48, kernel_size=3, padding=1),
            nn.GroupNorm(8, 48),
            nn.SiLU(inplace=True),
            nn.Conv2d(48, 3, kernel_size=1, padding=0),
        )

        self.rgb_activation = TrainableBezier(
            shape=(3,),
            p0=-0.5,
            p1=-0.05,
            p2=0.05,
            p3=0.5,
            channel_only=True,
        )

        # Established codebase convention (conditioning.py, encoders.py, every
        # flow.py) -- see FluxCompressor_v100.__init__ for full rationale.
        # MUST run before the seam-smoother zero-inits below: xavier_init
        # would otherwise clobber their intentional zero weight/bias back to
        # Xavier-uniform random values, breaking the "identity residual at
        # init" design (backward-compat with existing checkpoints).
        self.apply(xavier_init)

        # Seam-smoother: zero-init residual 3x3 conv with reflect padding.
        # Blends adjacent latent tokens to avoid 16-pixel grid artifacts at token
        # boundaries. Initialised to zero so the residual starts as a no-op,
        # preserving exact backward-compatibility with existing checkpoints; the
        # conv learns only the blending needed to reduce reconstruction/adversarial
        # loss at token seams.
        self.seam_smoother = nn.Conv2d(
            in_channels=d_model,
            out_channels=d_model,
            kernel_size=3,
            padding=1,
            padding_mode="reflect",
        )
        nn.init.zeros_(self.seam_smoother.weight)
        assert self.seam_smoother.bias is not None  # bias=True by default in nn.Conv2d
        nn.init.zeros_(self.seam_smoother.bias)

        # Seam-smoother for the context (SPADE conditioning) tokens.
        # Same zero-init residual design; context tokens have identical spatial
        # layout and exhibit the same token-boundary seam issue.
        self.seam_smoother_ctx = nn.Conv2d(
            in_channels=d_model,
            out_channels=d_model,
            kernel_size=3,
            padding=1,
            padding_mode="reflect",
        )
        nn.init.zeros_(self.seam_smoother_ctx.weight)
        assert self.seam_smoother_ctx.bias is not None  # bias=True by default in nn.Conv2d
        nn.init.zeros_(self.seam_smoother_ctx.bias)

    def _load_from_state_dict(
        self,
        state_dict,
        prefix,
        local_metadata,
        strict,
        missing_keys,
        unexpected_keys,
        error_msgs,
    ) -> None:
        # Detect legacy v0.10.0-pre SPADE keys and raise a clear error pointing
        # users at the salvage path.
        legacy_markers = ["spade.mlp_beta.weight", "spade.mlp_beta.bias"]
        for key in state_dict:
            if any(marker in key for marker in legacy_markers):
                from ...exceptions import IncompatibleCheckpointError

                raise IncompatibleCheckpointError(
                    f"Found legacy SPADE key '{key}' from v0.10.0-bezier-coupled "
                    "predecessor. The redesigned multi-scale SPADE has different "
                    "parameters. Run scripts/migrate_v0.10.0_to_redesign.py to "
                    "produce a warm-start checkpoint."
                )
        super()._load_from_state_dict(
            state_dict,
            prefix,
            local_metadata,
            strict,
            missing_keys,
            unexpected_keys,
            error_msgs,
        )

    def unpack(
        self, packed: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Extract image tokens, context tokens, and spatial dims from packed representation.

        Args:
            packed: [B, T+1, 2*D]

        Returns:
            img_seq: [B, T, D]
            context: [B, T, D]
            H: [B] height scalars (long)
            W: [B] width scalars (long)
        """
        img_seq_with_context = packed[:, :-1, :].contiguous()  # [B, T, 2D]
        D = self.d_model
        img_seq = img_seq_with_context[:, :, :D].contiguous()  # [B, T, D]
        context = img_seq_with_context[:, :, D:].contiguous()  # [B, T, D]
        H = (packed[:, -1, 0] * self.max_hw).round().clamp(min=1).long()
        W = (packed[:, -1, 1] * self.max_hw).round().clamp(min=1).long()
        return img_seq, context, H, W

    def forward(self, packed: torch.Tensor, use_context: bool = True) -> torch.Tensor:
        """
        Decode packed latent to image.

        Args:
            packed: [B, T+1, 2*D] — VAE tokens + context tokens + HW vector
            use_context: Whether to pass context to SPADE (default: True)

        Returns:
            torch.Tensor: Generated images [B, 3, H, W]
        """
        img_seq, context, H, W = self.unpack(packed)
        B, L, D = img_seq.shape

        if B > 1 and (H == H[0]).all() and (W == W[0]).all():
            h, w = int(H[0].item()), int(W[0].item())
            t_valid = h * w
            assert t_valid <= L, f"Mismatch: tokens {L} < h*w {t_valid}"

            feat = rearrange(img_seq[:, :t_valid], "b (h w) d -> b d h w", h=h, w=w)
            feat = feat + self.seam_smoother(
                feat
            )  # zero-init residual; learns to blend token seams

            if use_context:
                ctx = rearrange(context[:, :t_valid], "b (h w) c -> b c h w", h=h, w=w).contiguous()
                ctx = ctx + self.seam_smoother_ctx(ctx)  # zero-init residual; blends context seams
            else:
                ctx = None

            upscaled = self.upscale(feat, ctx)
            rgb = self.to_rgb_conv(upscaled)
            return cast(torch.Tensor, self.rgb_activation(rgb))
        else:
            outputs = []
            for i in range(B):
                h, w = int(H[i].item()), int(W[i].item())
                t_valid = h * w
                assert t_valid <= L, f"Mismatch: tokens {L} < h*w {t_valid}"

                feat_i = rearrange(img_seq[i : i + 1, :t_valid], "b (h w) d -> b d h w", h=h, w=w)
                feat_i = feat_i + self.seam_smoother(
                    feat_i
                )  # zero-init residual; blends token seams
                if use_context:
                    ctx_i = rearrange(
                        context[i : i + 1, :t_valid], "b (h w) c -> b c h w", h=h, w=w
                    ).contiguous()
                    ctx_i = ctx_i + self.seam_smoother_ctx(
                        ctx_i
                    )  # zero-init residual; blends context seams  # noqa: E501
                else:
                    ctx_i = None

                upscaled = self.upscale(feat_i, ctx_i)
                rgb = self.to_rgb_conv(upscaled)
                outputs.append(self.rgb_activation(rgb))

            return torch.cat(outputs, dim=0)
