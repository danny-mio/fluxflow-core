"""B5 (clean z in packed latent) + D5/D4 (resize-conv decoder, floor 16) + loud legacy failures."""

import logging

import pytest
import torch
import torch.nn as nn

from fluxflow.exceptions import IncompatibleCheckpointError
from fluxflow.models.pipeline import FluxPipeline
from fluxflow.models.v100.vae import (
    _MIN_UPSAMPLE_CHANNELS,
    FluxCompressor_v100,
    FluxExpander_v100,
    _ProgressiveUpscaler,
    _ResidualUpsampleBlock,
    _taper_channels,
)

D = 16


def _comp(**kw) -> FluxCompressor_v100:
    return FluxCompressor_v100(
        d_model=D,
        downscales=2,
        max_hw=64,
        attn_layers=1,
        ctx_attn_layers=1,
        ctx_attn_heads=2,
        use_gradient_checkpointing=False,
        **kw,
    )


# ---- B5 ----------------------------------------------------------------------------------------


def test_no_final_norm_param():
    m = _comp()
    assert not hasattr(m, "final_norm")
    assert not any(k.startswith("final_norm.") for k in m.state_dict())


def test_packed_z_half_equals_mu_tokens_when_not_sampling():
    m = _comp().eval()
    img = torch.randn(2, 3, 32, 32)
    with torch.no_grad():
        packed, mu, _ = m(img, training=True, sample=False)
    expected = mu.flatten(2).permute(0, 2, 1)
    assert torch.equal(packed[:, :-1, :D], expected)


def test_packed_z_half_is_raw_sample_when_sampling():
    m = _comp().eval()
    img = torch.randn(2, 3, 32, 32)
    with torch.no_grad():
        torch.manual_seed(0)
        p1, mu, lv = m(img, training=True, sample=True)
        torch.manual_seed(1)
        p2, _, _ = m(img, training=True, sample=True)
    assert not torch.equal(p1[:, :-1, :D], p2[:, :-1, :D])
    # z = mu + eps*std exactly (no PE / LayerNorm): mean over many tokens stays near mu
    std = (0.5 * lv).exp().flatten(2).permute(0, 2, 1)
    z = p1[:, :-1, :D]
    assert ((z - mu.flatten(2).permute(0, 2, 1)).abs() <= 6 * std + 1e-4).all()


def test_sample_defaults_to_training_flag():
    m = _comp().eval()
    img = torch.randn(1, 3, 32, 32)
    with torch.no_grad():
        a = m(img)
        b = m(img)
        c, mu, _ = m(img, training=True)
        d, _, _ = m(img, training=True)
    assert torch.equal(a, b)  # eval default: deterministic
    assert torch.equal(a[:, :-1, :D], mu.flatten(2).permute(0, 2, 1))
    assert not torch.equal(c[:, :-1, :D], d[:, :-1, :D])  # training default: stochastic


def test_sample_false_in_training_is_deterministic_and_ctx_uses_same_z():
    m = _comp().eval()
    img = torch.randn(1, 3, 32, 32)
    with torch.no_grad():
        a, _, _ = m(img, training=True, sample=False)
        b, _, _ = m(img, training=True, sample=False)
    assert torch.equal(a, b)


def test_sample_true_without_training_is_stochastic():
    m = _comp().eval()
    img = torch.randn(1, 3, 32, 32)
    with torch.no_grad():
        a = m(img, sample=True)
        b = m(img, sample=True)
    assert not torch.equal(a[:, :-1, :D], b[:, :-1, :D])


def test_compressor_drops_final_norm_keys_with_warning(caplog):
    m = _comp()
    sd = dict(m.state_dict())
    sd["final_norm.weight"] = torch.ones(D)
    sd["final_norm.bias"] = torch.zeros(D)
    with caplog.at_level(logging.WARNING):
        res = m.load_state_dict(sd, strict=True)
    assert not res.unexpected_keys
    assert any("final_norm" in r.message and "flow" in r.message for r in caplog.records)


def test_compressor_drops_prefixed_final_norm_via_pipeline_load(caplog):
    m = _comp()
    sd = {"compressor." + k: v for k, v in m.state_dict().items()}
    sd["compressor.final_norm.weight"] = torch.ones(D)
    holder = nn.ModuleDict({"compressor": _comp()})
    with caplog.at_level(logging.WARNING):
        res = holder.load_state_dict(sd, strict=True)
    assert not res.unexpected_keys
    assert any("final_norm" in r.message for r in caplog.records)


def test_ctx_final_norm_keys_not_dropped():
    m = _comp()
    assert any(k.startswith("ctx_final_norm.") for k in m.state_dict())
    m.load_state_dict(m.state_dict(), strict=True)


# ---- D5 / D4 -----------------------------------------------------------------------------------


def test_min_upsample_channels_is_16():
    assert _MIN_UPSAMPLE_CHANNELS == 16
    assert _taper_channels(16, 4) == [16, 16, 16, 16]
    assert _taper_channels(32, 4) == [16, 16, 16, 16]
    assert _taper_channels(128, 4) == [64, 32, 16, 16]
    assert _taper_channels(256, 4) == [128, 64, 32, 16]


@pytest.mark.parametrize("in_ch,out_ch", [(16, 16), (32, 16)])
def test_block_layout(in_ch, out_ch):
    b = _ResidualUpsampleBlock(in_ch, out_ch, context_size=8)
    assert isinstance(b.upsample, nn.Upsample)
    assert b.upsample.scale_factor == 2 and b.upsample.mode == "nearest"
    assert not any(isinstance(m, nn.ConvTranspose2d) for m in b.modules())
    c0, c2 = b.conv1[0], b.conv1[2]
    assert tuple(c0.weight.shape) == (out_ch * 5, in_ch, 3, 3)
    assert tuple(c2.weight.shape) == (out_ch * 5, out_ch, 3, 3)
    assert c0.padding == (1, 1) and c2.padding == (1, 1) and c2.dilation == (1, 1)
    assert (b.skip_proj is None) == (in_ch == out_ch)


def test_block_output_shape_exact_2x():
    for h, w in [(3, 5), (4, 4), (1, 7)]:
        b = _ResidualUpsampleBlock(16, 16, context_size=8, use_spade=False)
        assert b(torch.randn(2, 16, h, w)).shape == (2, 16, 2 * h, 2 * w)


def test_upscaler_each_stage_doubles():
    up = _ProgressiveUpscaler(
        channels=32, steps=3, context_size=8, use_gradient_checkpointing=False
    )
    x = torch.randn(1, 32, 3, 5)
    for layer in up.layers:
        h, w = x.shape[-2:]
        x = layer(x, torch.randn(1, 8, h, w))
        assert x.shape[-2:] == (2 * h, 2 * w)
    assert up.out_channels == 16


def test_no_checkerboard_for_constant_input_at_init():
    torch.manual_seed(0)
    b = _ResidualUpsampleBlock(16, 16, context_size=8, use_spade=False)
    x = torch.randn(1, 16, 1, 1).expand(1, 16, 16, 16).contiguous()
    with torch.no_grad():
        y = b.conv1(b.upsample(x))[:, :, 8:-8, 8:-8]  # interior: away from zero-pad borders
    phases = torch.stack([y[:, :, i::2, j::2].mean() for i in (0, 1) for j in (0, 1)])
    assert phases.var().item() < 1e-12
    assert y.abs().sum() > 0


def test_identity_at_init_preserved():
    b = _ResidualUpsampleBlock(16, 16, context_size=8, use_spade=False)
    x = torch.randn(1, 16, 4, 4)
    with torch.no_grad():
        assert torch.equal(b(x), b.skip_upsample(x))


def test_full_expander_forward_with_clean_z_packed():
    m = FluxExpander_v100(d_model=D, upscales=2, max_hw=64, use_gradient_checkpointing=False)
    c = _comp().eval()
    with torch.no_grad():
        out = m(c(torch.randn(1, 3, 32, 32)))
    assert out.shape == (1, 3, 32, 32) and torch.isfinite(out).all()


# ---- loud legacy failure -----------------------------------------------------------------------


def _legacy_expander_sd(m: FluxExpander_v100) -> dict:
    sd = dict(m.state_dict())
    k = "upscale.layers.0.conv1.0.weight"
    n_out5, n_in = sd[k].shape[0], sd[k].shape[1]
    sd[k] = torch.zeros(n_in, n_out5, 16, 16)  # legacy ConvTranspose2d layout
    return sd


def test_expander_legacy_upsample_layout_raises():
    m = FluxExpander_v100(d_model=D, upscales=2, max_hw=64)
    with pytest.raises(IncompatibleCheckpointError, match="decoder layout"):
        m.load_state_dict(_legacy_expander_sd(m), strict=False)


def test_expander_legacy_raises_when_nested_prefix():
    m = FluxExpander_v100(d_model=D, upscales=2, max_hw=64)
    holder = nn.ModuleDict({"expander": m})
    sd = {"expander." + k: v for k, v in _legacy_expander_sd(m).items()}
    with pytest.raises(IncompatibleCheckpointError):
        holder.load_state_dict(sd, strict=False)


def test_expander_new_layout_loads():
    m = FluxExpander_v100(d_model=D, upscales=2, max_hw=64)
    m.load_state_dict(m.state_dict(), strict=True)


# ---- key-based architecture detection ----------------------------------------------------------


def _v100_keys(conv_shape):
    return {
        "diffuser.flow_processor.vae_to_dmodel.weight": torch.zeros(8, 4),
        "diffuser.flow_processor.transformer_blocks.0.ctx_gate_proj.weight": torch.zeros(1),
        "diffuser.expander.upscale.layers.0.conv1.0.weight": torch.zeros(*conv_shape),
        "diffuser.expander.upscale.layers.1.conv1.0.weight": torch.zeros(*conv_shape),
    }


def test_pipeline_detect_config_new_layout():
    cfg = FluxPipeline._detect_config(_v100_keys((80, 16, 3, 3)))
    assert cfg["upscales"] == 2 and cfg["model_version"] == "0.10.0"


def test_pipeline_detect_config_legacy_layout_raises():
    with pytest.raises(IncompatibleCheckpointError, match="decoder layout"):
        FluxPipeline._detect_config(_v100_keys((16, 80, 16, 16)))


def test_pipeline_detect_config_old_versions_with_k16_still_ok():
    sd = {k: v for k, v in _v100_keys((16, 80, 16, 16)).items() if "ctx_gate_proj" not in k}
    assert FluxPipeline._detect_config(sd)["model_version"] == "0.3.0"


def test_diffusion_pipeline_detect_config_legacy_v100_raises():
    from fluxflow.models.diffusion_pipeline import FluxFlowPipeline

    with pytest.raises(IncompatibleCheckpointError):
        FluxFlowPipeline._detect_config(_v100_keys((16, 80, 16, 16)))
