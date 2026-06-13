"""Integration tests for v0.10.0 pipeline round-trip.

Tests the full image → compress → decompress pipeline using v0.10.0 models,
and verifies versioned save/load.
"""

import pytest
import torch


def test_full_roundtrip():
    """Image → compress → decompress must yield correct output shape."""
    from fluxflow.models.factory import create_bezier_models

    comp, exp, flow, _ = create_bezier_models(
        vae_dim=64, flow_d_model=128, model_version="0.10.0", downscales=2
    )
    img = torch.randn(1, 3, 64, 64) * 0.5
    with torch.no_grad():
        packed = comp(img)
    assert packed.shape[-1] == 128  # 2 * 64

    with torch.no_grad():
        out_img = exp(packed)
    assert out_img.shape == (1, 3, 64, 64)


def test_packed_token_dim_is_2xd():
    """Packed token last dim must be 2*d_model for v0.10.0."""
    from fluxflow.models.factory import create_bezier_models

    comp, exp, flow, _ = create_bezier_models(
        vae_dim=32, flow_d_model=64, model_version="0.10.0", downscales=2
    )
    img = torch.randn(1, 3, 32, 32)
    with torch.no_grad():
        packed = comp(img)
    assert packed.shape[-1] == 64  # 2 * 32


def test_flow_forward_shape():
    """Flow processor must accept and return v0.10.0 packed shape."""
    from fluxflow.models.factory import create_bezier_models

    comp, exp, flow, _ = create_bezier_models(
        vae_dim=32, flow_d_model=64, model_version="0.10.0", downscales=2
    )
    img = torch.randn(1, 3, 32, 32)
    with torch.no_grad():
        packed = comp(img)
        # v0.10.0 flow processor consumes per-token text + bool mask.
        text_seq = torch.randn(1, 6, 1024)
        text_mask = torch.ones(1, 6, dtype=torch.bool)
        t = torch.tensor([0.5])
        out = flow(packed, text_seq, text_mask, t)
    assert out.shape == packed.shape


def test_versioned_save_load(tmp_path):
    """Save then load a v0.10.0 checkpoint; loaded model must be non-None."""
    from fluxflow.models.factory import create_bezier_models
    from fluxflow.models.pipeline import FluxPipeline
    from fluxflow.models.versioning import load_versioned_checkpoint, save_versioned_checkpoint

    comp, exp, flow, _ = create_bezier_models(
        vae_dim=32, flow_d_model=64, model_version="0.10.0", downscales=2
    )
    pipeline = FluxPipeline(comp, flow, exp)
    save_versioned_checkpoint(pipeline, tmp_path / "v010_ckpt", model_version="0.10.0")

    loaded = load_versioned_checkpoint(tmp_path / "v010_ckpt", device="cpu")
    assert loaded is not None


@pytest.mark.parametrize("version", ["0.7.0", "0.8.0", "0.10.0"])
def test_create_bezier_models_all_versions(version):
    """create_bezier_models must not raise for any supported version."""
    from fluxflow.models.factory import create_bezier_models

    comp, exp, flow, enc = create_bezier_models(
        vae_dim=32, flow_d_model=64, model_version=version, downscales=2
    )
    img = torch.randn(1, 3, 32, 32)
    with torch.no_grad():
        packed = comp(img)
    if version in ("0.7.0", "0.8.0"):
        assert packed.shape[-1] == 32 + 5  # D + CONTEXT_DIMS
    elif version == "0.10.0":
        assert packed.shape[-1] == 32 * 2  # 2D
