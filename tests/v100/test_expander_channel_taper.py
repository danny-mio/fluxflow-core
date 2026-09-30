"""Tests for the v0.10.0 expander decoder channel taper (memory + fidelity fix).

Covers:
- `_taper_channels` formula at several d_model/upscales combinations.
- `to_rgb_conv`'s width tracking the final upsample stage's output channels
  instead of the old hardcoded 96/48.
- A full `FluxExpander_v100` forward pass at d_model=16 (the new standard
  target) producing a valid [B, 3, H, W] RGB output.
"""

import torch

from fluxflow.models.v100.vae import (
    _MIN_UPSAMPLE_CHANNELS,
    FluxExpander_v100,
    _ProgressiveUpscaler,
    _taper_channels,
)


def test_taper_channels_d16_upscales4():
    """d_model=16, upscales=4, default floor_ch=4 -> [8, 4, 4, 4]."""
    assert _taper_channels(16, 4) == [8, 4, 4, 4]


def test_taper_channels_d32_upscales4():
    """d_model=32, upscales=4 -> [16, 8, 4, 4]."""
    assert _taper_channels(32, 4) == [16, 8, 4, 4]


def test_taper_channels_d128_upscales4():
    """d_model=128, upscales=4 -> [64, 32, 16, 8], no floor clamping needed."""
    assert _taper_channels(128, 4) == [64, 32, 16, 8]


def test_taper_channels_never_below_floor():
    """Every stage width is >= floor_ch, even for small d_model / many stages."""
    widths = _taper_channels(8, 6)
    assert all(w >= _MIN_UPSAMPLE_CHANNELS for w in widths)
    assert len(widths) == 6


def test_taper_channels_custom_floor():
    """A custom floor_ch is respected instead of the module default."""
    assert _taper_channels(16, 4, floor_ch=2) == [8, 4, 2, 2]


def test_progressive_upscaler_exposes_final_out_channels():
    """`_ProgressiveUpscaler.out_channels` tracks the last stage's taper width."""
    up = _ProgressiveUpscaler(channels=16, steps=4, context_size=16)
    assert up.out_channels == 4
    assert up.layers[0].in_ch == 16
    assert up.layers[0].out_ch == 8
    assert up.layers[-1].out_ch == 4


def test_to_rgb_conv_tracks_final_stage_width_not_hardcoded_96():
    """to_rgb_conv's first conv in_channels == final upsample stage out_channels,
    not the old hardcoded 96, for a small d_model."""
    m = FluxExpander_v100(d_model=16, upscales=4, use_gradient_checkpointing=False)
    final_ch = m.upscale.out_channels
    assert final_ch == 4  # d_model=16, upscales=4, floor_ch=4 -> [8,4,4,4]

    first_conv = m.to_rgb_conv[0]
    assert isinstance(first_conv, torch.nn.Conv2d)
    assert first_conv.in_channels == final_ch
    assert first_conv.out_channels == final_ch * 2

    last_conv = m.to_rgb_conv[-1]
    assert isinstance(last_conv, torch.nn.Conv2d)
    assert last_conv.out_channels == 3


def test_flux_expander_v100_forward_at_d_model_16():
    """Full forward pass at d_model=16 (new standard target) is shape-correct
    and numerically finite, with no shape errors from the tapered channels."""
    torch.manual_seed(0)
    D = 16
    upscales = 4
    m = FluxExpander_v100(
        d_model=D, upscales=upscales, max_hw=128, use_gradient_checkpointing=False
    ).eval()

    T = 4
    packed = torch.randn(1, T * T + 1, 2 * D)
    packed[0, -1, 0] = T / 128.0
    packed[0, -1, 1] = T / 128.0

    with torch.no_grad():
        out = m(packed)

    expected_hw = T * (2**upscales)
    assert out.shape == (1, 3, expected_hw, expected_hw)
    assert torch.isfinite(out).all()
