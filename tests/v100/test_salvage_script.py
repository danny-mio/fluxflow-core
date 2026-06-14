"""Tests for the v0.10.0-pre -> v0.10.0-bezier-coupled salvage script."""

import safetensors.torch as st
import torch
import pytest

from scripts.migrate_v0_10_0_to_redesign import migrate_checkpoint


@pytest.fixture
def fake_old_checkpoint(tmp_path):
    """A minimal fake old-format checkpoint covering the direct-copy keys."""
    path = tmp_path / "fake_old.safetensors"
    tensors = {
        "diffuser.compressor.encoder_first_step.0.0.weight": torch.randn(60, 5, 3, 3),
        "diffuser.compressor.logvar_activation.p0": torch.full((32,), -1.0),
        "diffuser.compressor.logvar_activation.p3": torch.full((32,), 1.0),
        "diffuser.flow_processor.vae_to_dmodel.weight": torch.randn(512, 64),
        "diffuser.flow_processor.text_proj.weight": torch.randn(512, 1024),
        # Legacy keys that should be dropped:
        "diffuser.flow_processor.time_embed.0.weight": torch.randn(1000, 512),
        "diffuser.flow_processor.transformer_blocks.0.pillar_cross_attn.q_proj.weight": torch.randn(
            512, 512
        ),
    }
    st.save_file(tensors, str(path))
    return path


def test_migrate_direct_copy_keys(fake_old_checkpoint, tmp_path):
    out_path = tmp_path / "warm.safetensors"
    report = migrate_checkpoint(fake_old_checkpoint, out_path)
    new = st.load_file(str(out_path))
    # Direct-copy keys survive
    assert "diffuser.compressor.encoder_first_step.0.0.weight" in new
    assert "diffuser.flow_processor.vae_to_dmodel.weight" in new
    # Legacy dropped
    assert "diffuser.flow_processor.time_embed.0.weight" not in new
    assert "pillar_cross_attn" not in " ".join(new.keys())
    # Report mentions dropped keys
    assert any("time_embed" in k for k in report["dropped"])
    assert any("pillar_cross_attn" in k for k in report["dropped"])


def test_migrate_logvar_rescales_old_range_to_wide(fake_old_checkpoint, tmp_path):
    out_path = tmp_path / "warm.safetensors"
    report = migrate_checkpoint(fake_old_checkpoint, out_path)
    new = st.load_file(str(out_path))
    p0 = new["diffuser.compressor.logvar_activation.p0"]
    p3 = new["diffuser.compressor.logvar_activation.p3"]
    # Old range was [-1, 1]; new range is [-8, 4]; -1 -> -8, 1 -> 4.
    assert torch.allclose(p0, torch.full_like(p0, -8.0), atol=1e-5)
    assert torch.allclose(p3, torch.full_like(p3, 4.0), atol=1e-5)
    assert any("logvar_activation" in k for k in report["rescaled"])


def test_migrate_spade_partial_fill(tmp_path):
    """Old mlp_beta becomes the warm-start for beta_mid in the new SPADE."""
    src = tmp_path / "src.safetensors"
    st.save_file(
        {
            "diffuser.expander.upscale.layers.0.spade.mlp_beta.weight": torch.randn(32, 128, 3, 3),
            "diffuser.expander.upscale.layers.0.spade.mlp_beta.bias": torch.randn(32),
            "diffuser.expander.upscale.layers.0.spade.beta_scale": torch.tensor([0.5]),
            "diffuser.expander.upscale.layers.0.spade.mlp_shared.0.weight": torch.randn(
                128, 32, 3, 3
            ),
        },
        str(src),
    )
    dst = tmp_path / "dst.safetensors"
    report = migrate_checkpoint(src, dst)
    new = st.load_file(str(dst))
    # mlp_beta -> beta_mid
    assert "diffuser.expander.upscale.layers.0.spade.beta_mid.weight" in new
    assert "diffuser.expander.upscale.layers.0.spade.beta_mid.bias" in new
    # beta_scale preserved
    assert "diffuser.expander.upscale.layers.0.spade.beta_scale" in new
    # New heads zero-init (so identity behaviour preserved at warm-start)
    assert "diffuser.expander.upscale.layers.0.spade.beta_low.weight" in new
    assert new["diffuser.expander.upscale.layers.0.spade.beta_low.weight"].abs().sum() == 0
    assert "diffuser.expander.upscale.layers.0.spade.beta_hi.weight" in new
    assert "diffuser.expander.upscale.layers.0.spade.gamma_head.weight" in new
    # gamma_scale zero-init
    assert torch.allclose(
        new["diffuser.expander.upscale.layers.0.spade.gamma_scale"], torch.zeros(1)
    )
    # mlp_shared from old form is dropped (new deeper MLP doesn't fit)
    assert "diffuser.expander.upscale.layers.0.spade.mlp_shared.0.weight" not in new
    assert any(
        "diffuser.expander.upscale.layers.0.spade.mlp_shared" in k for k in report["dropped"]
    )
    assert any("spade" in k for k in report["partial_filled"])
