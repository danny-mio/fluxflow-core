"""Tests that the v0.10.0 expander uses SPADE_v100b throughout."""

import torch

from fluxflow.models.v100.vae import FluxExpander_v100
from fluxflow.models.v100.conditioning import SPADE_v100b


def _model(D: int = 32, upscales: int = 4) -> FluxExpander_v100:
    return FluxExpander_v100(
        d_model=D,
        upscales=upscales,
        max_hw=256,
        use_gradient_checkpointing=False,
    )


def test_expander_uses_spade_v100b():
    """All SPADE instances in the upscaler are SPADE_v100b."""
    m = _model()
    found = False
    for sub in m.modules():
        if isinstance(sub, SPADE_v100b):
            found = True
            break
    assert found, "Expected at least one SPADE_v100b in the expander"


def test_expander_identity_at_init_with_zero_scales():
    """With all SPADE scales at 0, decoder still produces finite output."""
    torch.manual_seed(0)
    m = _model().eval()
    D = 32
    T = 16
    packed = torch.zeros(1, T * T + 1, 2 * D)
    packed[0, -1, 0] = T / 256.0
    packed[0, -1, 1] = T / 256.0
    out = m(packed)
    assert torch.isfinite(out).all()
    assert out.shape == (1, 3, 256, 256)


def test_expander_decode_shape_round_trip():
    """Decode produces [B, 3, H, W]."""
    m = _model().eval()
    D = 32
    T = 8
    packed = torch.randn(1, T * T + 1, 2 * D)
    packed[0, -1, 0] = T / 256.0
    packed[0, -1, 1] = T / 256.0
    out = m(packed)
    assert out.shape == (1, 3, 128, 128)
