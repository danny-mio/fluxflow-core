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

import logging
import math
import re
from typing import Any, Mapping, cast

import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange
from torch.func import functional_call
from torch.utils.checkpoint import checkpoint

from ..activations import make_activation, xavier_init
from ...exceptions import IncompatibleCheckpointError
from .conditioning import SPADE_v100b

logger = logging.getLogger(__name__)


def _should_checkpoint(args: tuple) -> bool:
    return torch.is_grad_enabled() and any(
        isinstance(a, torch.Tensor) and a.requires_grad for a in args
    )


def _maybe_checkpoint(fn, *args, **kwargs):
    if _should_checkpoint(args):
        return checkpoint(fn, *args, **kwargs)
    kwargs.pop("use_reentrant", None)
    return fn(*args, **kwargs)


def _frozen_call(layer: nn.Module):
    """Run `layer` with parameters snapshotted as detached constants.

    Frozen params (flags off, or swapped via `torch.func.functional_call`) are
    captured at forward time, so the checkpoint recompute sees the same
    grad-free tensors even if flags/swaps are restored before backward.
    """
    params = {n: p.detach() for n, p in layer.named_parameters()}
    return lambda x, ctx: functional_call(layer, params, (x, ctx))


# Minimum per-stage channel width for the decoder's upsample taper. 16 keeps
# full-resolution stages from being starved (4 channels could not carry
# texture) while staying cheap now that the x5 wide lanes sit behind 3x3 convs.
_MIN_UPSAMPLE_CHANNELS = 16

# Legacy (pre-release) decoder upsample blocks used ConvTranspose2d with this kernel.
_LEGACY_UPSAMPLE_KERNEL = 16
LEGACY_DECODER_MSG = (
    "Checkpoint has the pre-release v0.10.0 decoder layout "
    "(upscale.layers.{idx}.conv1.0 is a 16x16 ConvTranspose2d). The decoder upsample "
    "block changed to nearest-upsample + 3x3 convs and cannot load old weights: "
    "retrain the decoder, or use scripts/migrate_v0_10_0_to_redesign.py to salvage the "
    "encoder and flow weights."
)
_UPSAMPLE_CONV1_RE = re.compile(r"(?:^|\.)upscale\.layers\.(\d+)\.conv1\.0\.weight$")


def legacy_upsample_keys(state_dict: Mapping[str, Any]) -> list[str]:
    """Keys of ``upscale.layers.N.conv1.0.weight`` tensors that use the legacy ConvTranspose layout."""
    return [
        k
        for k, v in state_dict.items()
        if _UPSAMPLE_CONV1_RE.search(k)
        and getattr(v, "ndim", 0) == 4
        and v.shape[-1] == _LEGACY_UPSAMPLE_KERNEL
    ]


def check_decoder_layout(state_dict: Mapping[str, Any]) -> None:
    """Raise IncompatibleCheckpointError if the state dict holds legacy v0.10.0 upsample blocks."""
    bad = legacy_upsample_keys(state_dict)
    if bad:
        raise IncompatibleCheckpointError(f"{LEGACY_DECODER_MSG} (found: '{bad[0]}')")


def _taper_channels(
    d_model: int, upscales: int, floor_ch: int = _MIN_UPSAMPLE_CHANNELS
) -> list[int]:
    """Per-stage output channel widths for the decoder's progressive upsampler.

    Constant channel width through every upsample stage silently RE-EXPANDS
    small d_model configs (e.g. d_model=16) back up at each doubling of
    spatial resolution, which is expensive right where compute is most
    costly. Instead, taper the width down each stage: ``out_ch(stage_i) =
    max(floor_ch, d_model // 2**stage_i)`` for ``stage_i = 1..upscales``.

    Args:
        d_model: Latent dimension (stage 0's input width).
        upscales: Number of 2x upsampling stages.
        floor_ch: Minimum channel width for any stage (default: 16).

    Returns:
        list[int]: ``out_ch`` for each stage, length ``upscales``.
    """
    return [max(floor_ch, d_model // (2**stage_i)) for stage_i in range(1, upscales + 1)]


def _smooth(conv: nn.Conv2d, x: torch.Tensor) -> torch.Tensor:
    """Apply a 3x3 reflect-padded seam conv; replicate-pad when a dim < 2 (reflect needs >= 2)."""
    if min(x.shape[-2:]) >= 2:
        return cast(torch.Tensor, conv(x))
    return F.conv2d(F.pad(x, (1, 1, 1, 1), mode="replicate"), conv.weight, conv.bias)


_CAL_PROBE_SEED = 1234
_CAL_PROBE_SIDE = 128
_CAL_PROBE_BATCH = 6
_CAL_ITERS = 12
_CAL_TOL = 0.05  # |log(rms/target)|


def _mean_abs(t: torch.Tensor) -> float:
    """Calibration scale: geometric mean over samples of per-sample mean|t|.

    Bezier/Padé outputs are heavy-tailed across samples; RMS or a pooled mean is
    dominated by one outlier image and does not generalise to other inputs.
    """
    per_sample = t.float().abs().flatten(1).mean(1).clamp_min(1e-30)
    return float(per_sample.log().mean().exp().item())


def _calibrate_block(
    block: nn.Module, conv: nn.Conv2d, x: torch.Tensor, target: float
) -> torch.Tensor:
    """Scale `conv.weight` (init only) until block(x) has scale (_mean_abs) ~= target; return block(x).

    Bezier/Padé 5x-wide activations are strongly non-linear in conv scale, so a
    fixed gain is unstable across seeds/widths; secant search in log-space is robust.
    """
    slope, prev, s_prev = 1.0, None, 1.0
    with torch.no_grad():
        y = cast(torch.Tensor, block(x))
        for _ in range(_CAL_ITERS):
            r = _mean_abs(y)
            if not math.isfinite(r) or r <= 0:
                break
            err = math.log(target / r)
            if abs(err) < _CAL_TOL:
                break
            if prev is not None and abs(math.log(s_prev)) > 1e-6:
                slope = min(max(math.log(r / prev) / math.log(s_prev), 0.5), 6.0)
            s_prev = math.exp(max(min(err / slope, 1.5), -1.5))
            prev = r
            conv.weight.mul_(s_prev)
            y = cast(torch.Tensor, block(x))
    return y


def _calibrate_chain(
    blocks: list[nn.Module], x: torch.Tensor, target: float | None = None
) -> torch.Tensor:
    """Calibrate Sequential(Conv2d, act) blocks in order, each to preserve the incoming RMS."""
    for blk in blocks:
        conv = next(m for m in blk.modules() if isinstance(m, nn.Conv2d))
        x = _calibrate_block(blk, conv, x, target if target is not None else _mean_abs(x))
    return x


def _safe_group_norm(num_channels: int, max_groups: int = 8) -> nn.GroupNorm:
    """GroupNorm with the largest num_groups <= max_groups that divides num_channels.

    Mirrors the divisor search in v070/conditioning.py's SPADE, generalized so
    channel counts derived from `_taper_channels`
    never produce an invalid GroupNorm (num_groups must divide num_channels).
    """
    num_groups = 1
    for ng in range(min(max_groups, num_channels), 0, -1):
        if num_channels % ng == 0:
            num_groups = ng
            break
    return nn.GroupNorm(num_groups, num_channels)


# ---------------------------------------------------------------------------
# Shared helpers (mirrors v070/vae.py but not imported to keep versions clean)
# ---------------------------------------------------------------------------


class _ResidualUpsampleBlock(nn.Module):
    """
    Residual block with 2x upsampling (nearest + 3x3 convs) — v0.10.0 variant using SPADEWithLearnableScale.

    Per-stage channel width tapers (Problem 2 fix): `in_ch` and `out_ch` may
    differ, in which case the `nn.Upsample` skip path (which does not itself
    change channel count) gets a 1x1 channel-projection so it can be added to
    the channel-changing `conv1` path. SPADE conditions the
    `conv1` output, so its `num_features` is `out_ch` -- the width of the
    tensor actually being normalized -- while `context_size` (the ctx signal's
    channel count) stays decoupled at the caller's chosen width (= d_model in
    v0.10.0), never tapered.

    Args:
        in_ch: Number of input channels.
        out_ch: Number of output channels.
        context_size: Context dimensionality for SPADE (= d_model in v0.10.0)
        use_spade: Enable SPADE conditioning
    """

    def __init__(
        self,
        in_ch: int,
        out_ch: int,
        context_size: int = 1024,
        use_spade: bool = True,
        activation_type: str = "bezier",
    ) -> None:
        super().__init__()
        self.use_spade = use_spade
        self.activation_type = activation_type
        self.in_ch = in_ch
        self.out_ch = out_ch
        if self.use_spade:
            self.spade = SPADE_v100b(context_size, out_ch, activation_type=activation_type)

        # Resize-conv: nearest upsample BEFORE the convs, so every output parity
        # phase sees the same kernel (a stride-2 ConvTranspose2d gives each phase
        # disjoint taps -> period-2 checkerboard that later layers cannot mix).
        self.upsample = nn.Upsample(scale_factor=2, mode="nearest")
        self.conv1 = nn.Sequential(
            nn.Conv2d(in_ch, out_ch * 5, kernel_size=3, padding=1),
            make_activation(
                self.activation_type, "fixed", t_pre_activation="tanh", p_preactivation="silu"
            ),
            nn.Conv2d(out_ch, out_ch * 5, kernel_size=3, padding=1),
            make_activation(
                self.activation_type, "fixed", t_pre_activation="tanh", p_preactivation="silu"
            ),
        )
        self.skip_upsample = nn.Upsample(scale_factor=2, mode="nearest")
        # Only project the skip path when widths differ, to avoid wasted
        # params on stages where the taper happens to keep in_ch == out_ch.
        self.skip_proj = nn.Conv2d(in_ch, out_ch, kernel_size=1) if in_ch != out_ch else None
        self.conv1_scale = nn.Parameter(torch.zeros(1))

    def forward(self, x: torch.Tensor, context: torch.Tensor | None = None) -> torch.Tensor:
        """
        Args:
            x: Input features [B, in_ch, H, W]
            context: Spatial context for SPADE [B, context_size, H', W'] or None

        Returns:
            torch.Tensor: Upsampled features [B, out_ch, 2*H, 2*W]
        """
        identity = x
        x = self.conv1(self.upsample(x))
        if self.use_spade:
            x = self.spade(x, context)
        identity_up = self.skip_upsample(identity)
        if self.skip_proj is not None:
            identity_up = self.skip_proj(identity_up)
        # conv1_scale is unclamped (see __init__); tanh at the use-site bounds
        # the effective scale to (-1, 1) so this branch can never out-grow the
        # identity path, while tanh(0)==0 keeps identity-at-init exact and
        # d/dx tanh(x)|_0==1 keeps gradient flow at init unaffected.
        return cast(torch.Tensor, identity_up + torch.tanh(self.conv1_scale) * x)


class _ProgressiveUpscaler(nn.Module):
    """
    Progressive upsampling using stacked _ResidualUpsampleBlocks (v0.10.0 variant).

    Channel width tapers per stage via `_taper_channels` (Problem 2 fix)
    instead of staying constant, so small `d_model` configs (e.g. 16) don't
    re-expand at full image resolution. `context_size` (the SPADE ctx signal)
    stays decoupled at a fixed width for every stage.

    Args:
        channels: Input channel count (= d_model)
        steps: Number of upsampling steps (each doubles resolution)
        context_size: Context dimension for SPADE
        use_spade: Enable SPADE conditioning
        floor_ch: Minimum per-stage channel width (see `_taper_channels`)
    """

    def __init__(
        self,
        channels: int = 3,
        steps: int = 2,
        context_size: int = 1024,
        use_spade: bool = True,
        use_gradient_checkpointing: bool = True,
        activation_type: str = "bezier",
        floor_ch: int = _MIN_UPSAMPLE_CHANNELS,
    ) -> None:
        super().__init__()
        self.use_gradient_checkpointing = use_gradient_checkpointing
        stage_out_ch = _taper_channels(channels, steps, floor_ch)
        stage_in_ch = [channels] + stage_out_ch[:-1]
        self.layers = nn.ModuleList(
            [
                _ResidualUpsampleBlock(
                    in_ch,
                    out_ch,
                    context_size,
                    use_spade=use_spade,
                    activation_type=activation_type,
                )
                for in_ch, out_ch in zip(stage_in_ch, stage_out_ch)
            ]
        )
        self.out_channels = stage_out_ch[-1] if stage_out_ch else channels

    def forward(self, x: torch.Tensor, context: torch.Tensor | None) -> torch.Tensor:
        """
        Args:
            x: Input [B, channels, H, W]
            context: Spatial context [B, context_size, H, W] or None

        Returns:
            torch.Tensor: Upsampled output [B, channels, H*2^steps, W*2^steps]
        """
        if self.use_gradient_checkpointing and x.requires_grad:
            frozen = not any(p.requires_grad for p in self.parameters())
            for layer in self.layers:
                x = _maybe_checkpoint(
                    _frozen_call(layer) if frozen else layer, x, context, use_reentrant=False
                )
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

    def __init__(
        self,
        dim: int,
        heads: int,
        drop: float = 0.0,
        ff_mult: int = 2,
        activation_type: str = "bezier",
    ) -> None:
        super().__init__()
        self.activation_type = activation_type
        hidden = dim * ff_mult
        self.norm1 = nn.LayerNorm(dim)
        self.attn = nn.MultiheadAttention(
            embed_dim=dim, num_heads=heads, dropout=drop, batch_first=True
        )
        self.norm2 = nn.LayerNorm(dim)
        self.ff = nn.Sequential(
            nn.Linear(dim, hidden * 5),
            make_activation(
                self.activation_type, "fixed", t_pre_activation="sigmoid", p_preactivation="silu"
            ),
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
    - Packed output shape: [B, T+1, 2*D] (was [B, T+1, D+5]); z half is raw z.

    Args:
        in_channels: Input image channels (default: 3)
        d_model: Latent dimension (default: 128)
        downscales: Number of 2x downsampling stages (default: 4)
        max_hw: Maximum spatial dimension for normalization (default: 1024)
        attn_layers: Number of z-path attention layers (z_token_attn)
        attn_heads: Number of attention heads in z-path (auto-reduced for small d_model)
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
        activation_type: str = "bezier",
    ) -> None:
        super().__init__()
        self.max_hw = max_hw
        self.downscales = downscales
        self.d_model = d_model
        self.attn_layers = attn_layers  # stored for metadata detection
        self.use_gradient_checkpointing = use_gradient_checkpointing
        self.activation_type = activation_type

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
                    make_activation(
                        self.activation_type,
                        "fixed",
                        t_pre_activation="tanh",
                        p_preactivation="silu",
                    ),
                )
                for i in range(downscales)
            ]
        )

        # Last stage only: kernel=12/padding=5 widens the bottleneck stage's
        # receptive field. floor((N+2*3-8)/2)+1 == floor((N+2*5-12)/2)+1 for
        # all N (both reduce to floor((N-2)/2)+1), so output spatial size is
        # unchanged.
        self.encoder_z = nn.ModuleList(
            [
                nn.Sequential(
                    nn.Conv2d(
                        self.stage_channels[i + 1],
                        self.stage_channels[i + 1] * 5,
                        kernel_size=12 if i == downscales - 1 else 8,
                        stride=2,
                        padding=5 if i == downscales - 1 else 3,
                    ),
                    make_activation(
                        self.activation_type,
                        "fixed",
                        t_pre_activation="tanh",
                        p_preactivation="silu",
                    ),
                )
                for i in range(downscales)
            ]
        )

        final_ch = self.stage_channels[-1]
        self.latent_proj = nn.Sequential(
            *[
                nn.Sequential(
                    nn.Conv2d(final_ch, d_model * 5, kernel_size=1),
                    make_activation(
                        self.activation_type,
                        "fixed",
                        t_pre_activation="tanh",
                        p_preactivation="silu",
                    ),
                )
                for _ in range(2)
            ]
        )
        self.mu_proj = nn.Sequential(
            *[
                nn.Sequential(
                    nn.Conv2d(d_model, d_model * 5, kernel_size=1),
                    make_activation(
                        self.activation_type,
                        "fixed",
                        t_pre_activation="sigmoid",
                        p_preactivation="silu",
                    ),
                )
                for _ in range(2)
            ]
        )
        self.logvar_proj = nn.Sequential(
            *[
                nn.Sequential(
                    nn.Conv2d(d_model, d_model * 5, kernel_size=1),
                    make_activation(
                        self.activation_type,
                        "fixed",
                        t_pre_activation="sigmoid",
                        p_preactivation="silu",
                    ),
                )
                for _ in range(2)
            ]
        )

        if self.activation_type == "bezier":
            self.mu_activation = make_activation(
                "bezier",
                "trainable",
                shape=(d_model,),
                channel_only=True,
                p0=-2.0,
                p1=-0.6,
                p2=0.6,
                p3=2.0,
            )
            self.logvar_activation = make_activation(
                "bezier",
                "wide",
                shape=(d_model,),
                channel_only=True,
                p0=-8.0,
                p1=-2.0,
                p2=2.0,
                p3=4.0,
            )
        else:
            self.mu_activation = make_activation(
                self.activation_type, "trainable", shape=(d_model,), channel_only=True
            )
            # Bezier wide logvar init is ~-0.5 (Bernstein midpoint of -8,-2,2,4);
            # match it (x - 0.5) instead of x - 8 (std~0.018, near-deterministic AE).
            self.logvar_activation = make_activation(
                self.activation_type,
                "wide",
                shape=(d_model,),
                channel_only=True,
                a0=-0.5,
                b1=1e-3,
            )

        # z is the one signal with no path to global context: SPADE conditions
        # the decoder on `ctx` only, and `ctx` already gets FiLM-modulated by
        # `z` via ctx_zinject_* below, but `z` itself never gets anything back.
        # Give the z-path's final feature map (pre latent_proj) the same
        # flatten-to-tokens / self-attend / reshape-to-spatial step ctx already
        # has via ctx_token_attn -- own _AttnBlock instance, own head-count
        # auto-fallback (see ctx_token_attn below for the identical rationale).
        effective_z_heads = max(2, d_model // 16)
        effective_z_heads = min(effective_z_heads, attn_heads)
        while d_model % effective_z_heads != 0 and effective_z_heads > 1:
            effective_z_heads -= 1
        self.z_token_attn = nn.ModuleList(
            [
                _AttnBlock(
                    d_model,
                    effective_z_heads,
                    attn_dropout,
                    attn_ff_mult,
                    activation_type=self.activation_type,
                )
                for _ in range(attn_layers)
            ]
        )

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
                    make_activation(
                        self.activation_type,
                        "fixed",
                        t_pre_activation="tanh",
                        p_preactivation="silu",
                    ),
                )
                for i in range(downscales)
            ]
        )

        # Last stage only: same kernel/padding bump as encoder_z, for
        # architectural consistency between the z and ctx paths.
        self.ctx_encoder_z = nn.ModuleList(
            [
                nn.Sequential(
                    nn.Conv2d(
                        self.ctx_stage_channels[i + 1],
                        self.ctx_stage_channels[i + 1] * 5,
                        kernel_size=12 if i == downscales - 1 else 8,
                        stride=2,
                        padding=5 if i == downscales - 1 else 3,
                    ),
                    make_activation(
                        self.activation_type,
                        "fixed",
                        t_pre_activation="tanh",
                        p_preactivation="silu",
                    ),
                )
                for i in range(downscales)
            ]
        )

        ctx_final_ch = self.ctx_stage_channels[-1]
        self.ctx_proj = nn.Sequential(
            *[
                nn.Sequential(
                    nn.Conv2d(ctx_final_ch, d_model * 5, kernel_size=1),
                    make_activation(
                        self.activation_type,
                        "fixed",
                        t_pre_activation="tanh",
                        p_preactivation="silu",
                    ),
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
            make_activation(
                self.activation_type, "fixed", t_pre_activation="tanh", p_preactivation="silu"
            ),
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
                _AttnBlock(
                    d_model,
                    effective_ctx_heads,
                    attn_dropout,
                    attn_ff_mult,
                    activation_type=self.activation_type,
                )
                for _ in range(ctx_attn_layers)
            ]
        )
        self.ctx_final_norm = nn.LayerNorm(d_model)

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
        self._calibrate_init()

    def _calibrate_init(self) -> None:
        """Rescale conv inits so per-stage signal scale is preserved down both encoder paths (init only).

        Without it Bezier shrinks RMS ~3.5x/layer (bottleneck ~1e-5, ctx_proj below
        GroupNorm eps) and Padé ~2x/layer. Uses a local-generator probe so global RNG
        and zero-inits (seam smoothers / zinject scales) are untouched.
        """
        side = _CAL_PROBE_SIDE
        gen = torch.Generator().manual_seed(_CAL_PROBE_SEED)
        probe = torch.randn(
            _CAL_PROBE_BATCH, self.stage_channels[0] - 2, side, side, generator=gen
        ).clamp(-1, 1)
        x0 = self.add_coord_channels(probe)
        z = _calibrate_chain(
            [
                m
                for i in range(self.downscales)
                for m in (self.encoder_first_step[i], self.encoder_z[i])
            ],
            x0,
        )
        lat = _calibrate_chain(list(self.latent_proj), z)
        _calibrate_chain(list(self.mu_proj), lat)
        _calibrate_chain(list(self.logvar_proj), lat)
        cx = _calibrate_chain(
            [
                m
                for i in range(self.downscales)
                for m in (self.ctx_encoder_first_step[i], self.ctx_encoder_z[i])
            ],
            x0,
        )
        _calibrate_chain(list(self.ctx_proj), cx)

    @staticmethod
    def add_coord_channels(x: torch.Tensor) -> torch.Tensor:
        """Add normalized coordinate channels to input."""
        B, _, H, W = x.shape
        yy, xx = torch.meshgrid(
            torch.linspace(-1, 1, H, device=x.device, dtype=x.dtype),
            torch.linspace(-1, 1, W, device=x.device, dtype=x.dtype),
            indexing="ij",
        )
        coords = torch.stack([xx, yy], dim=0).unsqueeze(0).expand(B, -1, -1, -1)
        return torch.cat([x, coords], dim=1)

    def reparameterize(self, mu: torch.Tensor, logvar: torch.Tensor) -> torch.Tensor:
        """Reparameterization trick for VAE."""
        std = (0.5 * logvar).exp()
        eps = torch.randn_like(std)
        return mu + eps * std

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
        # Pre-release checkpoints carry a z-path LayerNorm that no longer exists.
        stale = [k for k in state_dict if k.startswith(prefix + "final_norm.")]
        if stale:
            for k in stale:
                del state_dict[k]
            logger.warning(
                "Dropped %d stale '%sfinal_norm.*' keys: the packed z format changed "
                "(raw z, no LayerNorm/PE). Encoder weights remain valid; flow models "
                "trained on the old packed format are incompatible.",
                len(stale),
                prefix,
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

    def forward(
        self, img: torch.Tensor, training: bool = False, sample: bool | None = None
    ) -> torch.Tensor | tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Encode image to packed latent representation.

        Args:
            img: Input image [B, C, H, W]
            training: If True, returns (packed, mu, logvar) for KL loss
            sample: Draw z = mu + eps*std if True, use the deterministic z = mu if
                False; None -> same as ``training``. The ctx z-injection uses the same z.

        Returns:
            packed: [B, T+1, 2*D] where T = H_lat * W_lat; first D channels are the raw
                z tokens (no PE, no LayerNorm), last D the context tokens; row T is HW.
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

        # z-path global-context step (see z_token_attn comment in __init__):
        # flatten to tokens, self-attend, reshape back to spatial for the
        # 1x1-conv mu/logvar projections below.
        Bx, Dx, Hx, Wx = x.shape
        z_seq = x.flatten(2).permute(0, 2, 1)  # [B, T, D]

        def z_attn_block(seq: torch.Tensor) -> torch.Tensor:
            for blk in self.z_token_attn:
                seq = blk(seq)
            return seq

        if self.use_gradient_checkpointing and z_seq.requires_grad:
            z_seq = _maybe_checkpoint(z_attn_block, z_seq, use_reentrant=True)
        else:
            z_seq = z_attn_block(z_seq)

        x = z_seq.permute(0, 2, 1).reshape(Bx, Dx, Hx, Wx)

        latent = self.latent_proj(x)
        mu = self.mu_activation(self.mu_proj(latent))
        logvar = self.logvar_activation(self.logvar_proj(latent))
        # Bezier's mu/logvar activations are inherently bounded (Bernstein convex
        # combination of control points); Pade's are not (see
        # docs/PADE-ACTIVATION-ANALYSIS.md's "Known trade-off, not yet resolved").
        # Observed live: unclamped logvar reaching ~66, giving
        # std=exp(0.5*66)=~2e14 in reparameterize() below -- overflows fp16's
        # 65504 ceiling and poisons every downstream consumer of the sampled z.
        # Clamp to the standard VAE-literature range (matches e.g. Stable
        # Diffusion's AutoencoderKL) so std stays well inside fp16 range even
        # with a several-sigma eps draw.
        logvar = logvar.clamp(-30.0, 20.0)
        if sample is None:
            sample = training
        z = self.reparameterize(mu, logvar) if sample else mu  # [B, D, H_lat, W_lat]

        # ---- z path: raw z (the KL-shaped N(0, I) signal) goes into the packed latent ----
        # No PE, no LayerNorm, no tanh: the flow sees exactly what the KL term shapes.
        # Position is carried by token order + the decoder's convs/padding.
        B, D, H, W = z.shape
        z_tokens = z.flatten(2).permute(0, 2, 1)  # [B, T, D]

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
        activation_type: str = "bezier",
    ) -> None:
        super().__init__()
        self.max_hw = max_hw
        self.d_model = d_model
        self.activation_type = activation_type

        # v0.10.0: context_size = d_model (not the old CONTEXT_DIMS=5)
        self.upscale = _ProgressiveUpscaler(
            channels=d_model,
            steps=upscales,
            context_size=d_model,
            use_spade=True,
            use_gradient_checkpointing=use_gradient_checkpointing,
            activation_type=activation_type,
        )

        # RGB conversion head: widths derive from the final upsample stage's
        # actual output channel count (final_ch), not a hardcoded 96/48 (which
        # at small d_model, e.g. 16, silently RE-EXPANDED 16->96 at full image
        # resolution -- expensive and tuned for a different d_model).
        final_ch = self.upscale.out_channels
        self.to_rgb_conv = nn.Sequential(
            nn.Conv2d(final_ch, final_ch * 2, kernel_size=3, padding=1),
            _safe_group_norm(final_ch * 2),
            nn.SiLU(inplace=True),
            nn.Conv2d(final_ch * 2, final_ch, kernel_size=3, padding=1),
            _safe_group_norm(final_ch),
            nn.SiLU(inplace=True),
            nn.Conv2d(final_ch, 3, kernel_size=1, padding=0),
        )

        if activation_type == "bezier":
            self.rgb_activation = make_activation(
                "bezier",
                "trainable",
                shape=(3,),
                channel_only=True,
                p0=-1.25,
                p1=-0.4,
                p2=0.4,
                p3=1.25,
            )
        else:
            # Softsign-like y = 1.25x/(1+|x|): bounded by +-1.25 at init (data is [-1, 1]).
            self.rgb_activation = make_activation(
                activation_type, "trainable", shape=(3,), channel_only=True, a1=1.25, b1=1.0
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
                raise IncompatibleCheckpointError(
                    f"Found legacy SPADE key '{key}' from v0.10.0-bezier-coupled "
                    "predecessor. The redesigned multi-scale SPADE has different "
                    "parameters. Run scripts/migrate_v0.10.0_to_redesign.py to "
                    "produce a warm-start checkpoint."
                )
        check_decoder_layout({k: v for k, v in state_dict.items() if k.startswith(prefix)})
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

    @property
    def use_gradient_checkpointing(self) -> bool:
        """Whether the progressive upscaler checkpoints its stages."""
        return bool(self.upscale.use_gradient_checkpointing)

    @use_gradient_checkpointing.setter
    def use_gradient_checkpointing(self, enabled: bool) -> None:
        self.upscale.use_gradient_checkpointing = enabled

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
            feat = feat + _smooth(
                self.seam_smoother, feat
            )  # zero-init residual; learns to blend token seams

            if use_context:
                ctx = rearrange(context[:, :t_valid], "b (h w) c -> b c h w", h=h, w=w).contiguous()
                ctx = ctx + _smooth(
                    self.seam_smoother_ctx, ctx
                )  # zero-init residual; blends context seams
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
                feat_i = feat_i + _smooth(
                    self.seam_smoother, feat_i
                )  # zero-init residual; blends token seams
                if use_context:
                    ctx_i = rearrange(
                        context[i : i + 1, :t_valid], "b (h w) c -> b c h w", h=h, w=w
                    ).contiguous()
                    ctx_i = ctx_i + _smooth(
                        self.seam_smoother_ctx, ctx_i
                    )  # zero-init residual; blends context seams  # noqa: E501
                else:
                    ctx_i = None

                upscaled = self.upscale(feat_i, ctx_i)
                rgb = self.to_rgb_conv(upscaled)
                outputs.append(self.rgb_activation(rgb))

            return torch.cat(outputs, dim=0)
