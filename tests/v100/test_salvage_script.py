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


def test_migrate_pillar_padding(tmp_path):
    """Old D->D pillar layers become D->2D->2D->D with upper-half embedding."""
    D = 8
    src = tmp_path / "src.safetensors"
    base = "diffuser.flow_processor.transformer_blocks.0.p0"
    # Old per-pillar layers were all D->D
    w0 = torch.randn(D, D)
    b0 = torch.randn(D)
    w1 = torch.randn(D, D)
    b1 = torch.randn(D)
    w2 = torch.randn(D, D)
    b2 = torch.randn(D)
    st.save_file(
        {
            f"{base}.0.0.weight": w0,
            f"{base}.0.0.bias": b0,
            f"{base}.1.0.weight": w1,
            f"{base}.1.0.bias": b1,
            f"{base}.2.0.weight": w2,
            f"{base}.2.0.bias": b2,
        },
        str(src),
    )
    dst = tmp_path / "dst.safetensors"
    report = migrate_checkpoint(src, dst)
    new = st.load_file(str(dst))

    # Layer 0: (D, D) -> (2D, D); upper D rows = old weights, lower D rows = 0.
    nw0 = new[f"{base}.0.0.weight"]
    assert nw0.shape == (2 * D, D)
    assert torch.allclose(nw0[:D], w0)
    assert torch.allclose(nw0[D:], torch.zeros_like(nw0[D:]))
    nb0 = new[f"{base}.0.0.bias"]
    assert nb0.shape == (2 * D,)
    assert torch.allclose(nb0[:D], b0)
    assert torch.allclose(nb0[D:], torch.zeros(D))

    # Layer 1: (D, D) -> (2D, 2D); upper-left D x D block = old weights, rest = 0.
    nw1 = new[f"{base}.1.0.weight"]
    assert nw1.shape == (2 * D, 2 * D)
    assert torch.allclose(nw1[:D, :D], w1)
    assert torch.allclose(nw1[D:, :], torch.zeros(D, 2 * D))
    assert torch.allclose(nw1[:, D:], torch.cat([torch.zeros(D, D), torch.zeros(D, D)], dim=0))
    nb1 = new[f"{base}.1.0.bias"]
    assert nb1.shape == (2 * D,)
    assert torch.allclose(nb1[:D], b1)
    assert torch.allclose(nb1[D:], torch.zeros(D))

    # Layer 2: (D, D) -> (D, 2D); left half = old weights, right half = 0.
    nw2 = new[f"{base}.2.0.weight"]
    assert nw2.shape == (D, 2 * D)
    assert torch.allclose(nw2[:, :D], w2)
    assert torch.allclose(nw2[:, D:], torch.zeros(D, D))
    # Layer 2 bias stays D-wide.
    nb2 = new[f"{base}.2.0.bias"]
    assert nb2.shape == (D,)
    assert torch.allclose(nb2, b2)

    assert any(".p0.0.0.weight" in k for k in report["padded"])
    assert any(".p0.2.0.weight" in k for k in report["padded"])


def test_migrate_film_duplication(tmp_path):
    """Old single FiLM per pillar warm-starts text head; time head zero-init."""
    D = 16
    src = tmp_path / "src.safetensors"
    base = "diffuser.flow_processor.transformer_blocks.0"
    fw = torch.randn(2 * D, D)
    fb = torch.randn(2 * D)
    st.save_file(
        {
            f"{base}.film_p0.weight": fw,
            f"{base}.film_p0.bias": fb,
        },
        str(src),
    )
    dst = tmp_path / "dst.safetensors"
    report = migrate_checkpoint(src, dst)
    new = st.load_file(str(dst))

    # text head: full copy
    assert torch.allclose(new[f"{base}.film_p0_text.weight"], fw)
    assert torch.allclose(new[f"{base}.film_p0_text.bias"], fb)
    # time head: zero-init
    assert torch.allclose(new[f"{base}.film_p0_time.weight"], torch.zeros_like(fw))
    assert torch.allclose(new[f"{base}.film_p0_time.bias"], torch.zeros_like(fb))
    # legacy single film_p0 key no longer present
    assert f"{base}.film_p0.weight" not in new

    assert any("film_p0_text" in k for k in report["duplicated"])
    assert any("film_p0_time" in k for k in report["duplicated"])


def test_migrate_norm2_duplication(tmp_path):
    """Old norm2 LayerNorm warm-starts both norm2_q and norm2_kv."""
    D = 16
    src = tmp_path / "src.safetensors"
    base = "diffuser.flow_processor.transformer_blocks.0"
    nw = torch.randn(D)
    nb = torch.randn(D)
    st.save_file(
        {
            f"{base}.norm2.weight": nw,
            f"{base}.norm2.bias": nb,
        },
        str(src),
    )
    dst = tmp_path / "dst.safetensors"
    report = migrate_checkpoint(src, dst)
    new = st.load_file(str(dst))

    assert torch.allclose(new[f"{base}.norm2_q.weight"], nw)
    assert torch.allclose(new[f"{base}.norm2_q.bias"], nb)
    assert torch.allclose(new[f"{base}.norm2_kv.weight"], nw)
    assert torch.allclose(new[f"{base}.norm2_kv.bias"], nb)
    assert f"{base}.norm2.weight" not in new

    assert any("norm2_q" in k for k in report["duplicated"])
    assert any("norm2_kv" in k for k in report["duplicated"])
