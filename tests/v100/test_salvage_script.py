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
