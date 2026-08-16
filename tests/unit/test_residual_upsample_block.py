"""Tests for _ResidualUpsampleBlock identity-at-init (v100 only, Fix 3).

conv1_scale gates the conv branch; at init the block must be an exact
identity: identity_up + 0 * conv1(x) == identity_up.
"""

import torch

from fluxflow.models.v100.vae import FluxExpander_v100, _ResidualUpsampleBlock


def test_residual_upsample_block_identity_at_init():
    """At init, conv1_scale == 0 → forward(x) == skip_upsample(x) exactly."""
    block = _ResidualUpsampleBlock(channels=8, use_spade=False)
    block.eval()

    assert torch.all(block.conv1_scale == 0), "conv1_scale must be zero at init"

    x = torch.randn(1, 8, 4, 4)
    with torch.no_grad():
        out = block(x, context=None)
        expected = block.skip_upsample(x)

    torch.testing.assert_close(out, expected, atol=0, rtol=0)


def test_state_dict_compat_missing_conv1_scale():
    """Legacy checkpoint missing conv1_scale loads cleanly via strict=False.

    Simulates a pre-fix checkpoint: save state_dict, strip conv1_scale keys,
    reload into a fresh FluxExpander_v100 with strict=False. Only conv1_scale
    keys should be missing, nothing unexpected, and the freshly-constructed
    conv1_scale parameters default to zero (identity-preserving).
    """
    exp = FluxExpander_v100(d_model=32, upscales=2)
    exp.eval()

    B, T, D = 1, 16, 32
    packed = torch.zeros(B, T + 1, 2 * D)
    packed[:, -1, 0] = 4 / 1024.0
    packed[:, -1, 1] = 4 / 1024.0
    with torch.no_grad():
        out_before = exp(packed)

    sd = {k: v for k, v in exp.state_dict().items() if "conv1_scale" not in k}

    fresh = FluxExpander_v100(d_model=32, upscales=2)
    fresh.eval()
    missing, unexpected = fresh.load_state_dict(sd, strict=False)

    assert all("conv1_scale" in k for k in missing), f"Unexpected missing keys: {missing}"
    assert unexpected == [], f"Unexpected keys: {unexpected}"

    for layer in fresh.upscale.layers:
        assert torch.all(layer.conv1_scale == 0), "conv1_scale must default to zero"

    with torch.no_grad():
        out_after = fresh(packed)

    assert torch.isfinite(out_after).all()
    torch.testing.assert_close(out_after, out_before, atol=0, rtol=0)


def test_residual_upsample_block_conv1_scale_bounded_via_tanh():
    """conv1_scale is unclamped; an aggressive optimizer step (or many steps
    of drift) can push it far from 0. The *effective* scale consumed in
    forward() must stay bounded to (-1, 1) via tanh so the branch can never
    contribute more than the identity path, regardless of how large the raw
    Parameter grows.
    """
    torch.manual_seed(0)
    block = _ResidualUpsampleBlock(channels=8, use_spade=False)
    block.eval()
    block.conv1_scale.data.fill_(
        8.0
    )  # simulate aggressive drift (fp32 tanh saturates to 1.0 above ~9)

    x = torch.randn(1, 8, 4, 4)
    with torch.no_grad():
        out = block(x, context=None)
        identity_up = block.skip_upsample(x)
        conv_out = block.conv1(x)

    assert torch.isfinite(out).all()
    effective_scale = torch.tanh(block.conv1_scale)
    assert effective_scale.abs().item() < 1.0
    torch.testing.assert_close(out, identity_up + effective_scale * conv_out, atol=1e-5, rtol=1e-4)
