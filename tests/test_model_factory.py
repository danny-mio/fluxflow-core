"""
Tests for model factory (Bezier vs Baseline model creation with versioning).

Validates that ModelFactory can create compatible models for both
Bezier and Baseline variants, with support for different versions.
"""

import pytest
import torch
import torch.nn as nn

from fluxflow.models.factory import ModelFactory, create_baseline_models, create_bezier_models


class TestModelFactory:
    """Test ModelFactory basic functionality."""

    def test_factory_init_bezier_v030(self):
        """Test factory initialization for Bezier models v0.3.0."""
        factory = ModelFactory(model_type="bezier", model_version="0.3.0")

        assert factory.model_type == "bezier"
        assert factory.model_version == "0.3.0"
        assert factory.vae_dim == 128
        assert factory.bezier_flow_blocks == 10

    def test_factory_init_bezier_v060(self):
        """Test factory initialization for Bezier models v0.6.0."""
        factory = ModelFactory(model_type="bezier", model_version="0.6.0")

        assert factory.model_type == "bezier"
        assert factory.model_version == "0.6.0"
        assert factory.vae_dim == 128
        assert factory.bezier_flow_blocks == 10

    def test_factory_init_baseline(self):
        """Test factory initialization for Baseline models."""
        factory = ModelFactory(
            model_type="baseline",
            baseline_activation="silu",
            baseline_vae_width_mult=4.5,
            baseline_flow_blocks=17,
        )

        assert factory.model_type == "baseline"
        assert factory.vae_dim == 128
        assert factory.baseline_activation == "silu"
        assert factory.baseline_vae_width_mult == 4.5
        assert factory.baseline_flow_blocks == 17

    def test_factory_config_roundtrip(self):
        """Test config export/import."""
        factory1 = ModelFactory(
            model_type="baseline",
            vae_dim=256,
            baseline_flow_blocks=20,
        )

        config = factory1.get_config()
        factory2 = ModelFactory.from_config(config)

        assert factory2.model_type == factory1.model_type
        assert factory2.vae_dim == factory1.vae_dim
        assert factory2.baseline_flow_blocks == factory1.baseline_flow_blocks


class TestVAEEncoderCreation:
    """Test VAE encoder creation (same for both model types)."""

    def test_create_bezier_vae_encoder_v030(self):
        """Test Bezier VAE encoder creation v0.3.0."""
        factory = ModelFactory(model_type="bezier", model_version="0.3.0", vae_dim=128)
        encoder = factory.create_vae_encoder(downscales=4)  # v0.3.0 uses 4 downscales

        assert encoder is not None
        assert encoder.d_model == 128

    def test_create_bezier_vae_encoder_v060(self):
        """Test Bezier VAE encoder creation v0.6.0."""
        factory = ModelFactory(model_type="bezier", model_version="0.6.0", vae_dim=128)
        encoder = factory.create_vae_encoder(downscales=3)  # v0.6.0 uses 3 downscales

        assert encoder is not None
        assert encoder.d_model == 128

    def test_create_baseline_vae_encoder(self):
        """Test Baseline VAE encoder creation (uses same encoder as bezier)."""
        factory = ModelFactory(model_type="baseline", vae_dim=128)
        encoder = factory.create_vae_encoder()

        assert encoder is not None
        assert encoder.d_model == 128

    def test_vae_encoder_forward_v030(self):
        """Test VAE encoder forward pass v0.3.0."""
        factory = ModelFactory(model_type="bezier", model_version="0.3.0", vae_dim=128)
        encoder = factory.create_vae_encoder(downscales=4, use_gradient_checkpointing=False)

        # Create dummy input
        batch_size = 2
        x = torch.randn(batch_size, 3, 64, 64)

        with torch.no_grad():
            mu_logvar, deterministic = encoder(x)

        # Check output shapes - encoder returns flattened [T, D]
        # After 4 downscales: 64 -> 32 -> 16 -> 8 -> 4
        # expected_spatial = 4 * 4  # 16 tokens
        # Note: Encoder returns [T+1, D] format (flattened across batch)
        assert mu_logvar.shape[1] == 128  # D dimension
        assert deterministic.shape[1] == 128  # D dimension

    def test_vae_encoder_forward_v060(self):
        """Test VAE encoder forward pass v0.6.0."""
        factory = ModelFactory(model_type="bezier", model_version="0.6.0", vae_dim=128)
        encoder = factory.create_vae_encoder(downscales=3, use_gradient_checkpointing=False)

        # Create dummy input
        batch_size = 2
        x = torch.randn(batch_size, 3, 64, 64)

        with torch.no_grad():
            mu_logvar, deterministic = encoder(x)

        # Check output shapes - encoder returns flattened [T, D]
        # After 3 downscales: 64 -> 32 -> 16 -> 8
        # expected_spatial = 8 * 8  # 64 tokens
        # Note: Encoder returns [T+1, D] format (flattened across batch)
        assert mu_logvar.shape[1] == 128  # D dimension
        assert deterministic.shape[1] == 128  # D dimension


class TestTextEncoderCreation:
    """Test text encoder creation (shared for both model types)."""

    def test_create_text_encoder_frozen(self):
        """Test frozen text encoder creation."""
        factory = ModelFactory()
        encoder = factory.create_text_encoder(embed_dim=1024, frozen=True)

        assert encoder is not None

        # Verify frozen
        for param in encoder.parameters():
            assert not param.requires_grad

    def test_text_encoder_same_for_both_types(self):
        """Verify text encoder is IDENTICAL for Bezier and Baseline."""
        bezier_factory = ModelFactory(model_type="bezier")
        baseline_factory = ModelFactory(model_type="baseline")

        bezier_encoder = bezier_factory.create_text_encoder()
        baseline_encoder = baseline_factory.create_text_encoder()

        # Both should have same architecture
        bezier_params = sum(p.numel() for p in bezier_encoder.parameters())
        baseline_params = sum(p.numel() for p in baseline_encoder.parameters())

        assert bezier_params == baseline_params

    def test_text_encoder_output_shape(self):
        """Text encoder now outputs per-token (text_seq [B, T, E], text_mask [B, T])."""
        factory = ModelFactory()
        encoder = factory.create_text_encoder(embed_dim=1024, frozen=True)

        tokens = torch.randint(0, 30522, (2, 15))

        with torch.no_grad():
            text_seq, text_mask = encoder(tokens)

        assert text_seq.shape == (2, 15, 1024)  # [B, T, embed_dim]
        assert text_mask.shape == (2, 15)
        assert text_mask.dtype == torch.bool


class TestVAEDecoderCreation:
    """Test VAE decoder creation."""

    def test_create_bezier_vae_decoder_v030(self):
        """Test Bezier VAE decoder creation v0.3.0."""
        factory = ModelFactory(model_type="bezier", model_version="0.3.0", vae_dim=128)
        decoder = factory.create_vae_decoder(upscales=4)  # v0.3.0 uses 4 upscales

        assert decoder is not None

    def test_create_bezier_vae_decoder_v060(self):
        """Test Bezier VAE decoder creation v0.6.0."""
        factory = ModelFactory(model_type="bezier", model_version="0.6.0", vae_dim=128)
        decoder = factory.create_vae_decoder(upscales=3)  # v0.6.0 uses 3 upscales

        assert decoder is not None

    def test_create_baseline_vae_decoder(self):
        """Test Baseline VAE decoder creation (Phase 1.1 complete)."""
        factory = ModelFactory(model_type="baseline", vae_dim=128)
        decoder = factory.create_vae_decoder()

        assert decoder is not None
        assert isinstance(decoder, nn.Module)

    def test_bezier_decoder_forward_v030(self):
        """Test Bezier VAE decoder forward pass v0.3.0."""
        factory = ModelFactory(model_type="bezier", model_version="0.3.0", vae_dim=128)
        decoder = factory.create_vae_decoder(upscales=4, use_gradient_checkpointing=False)

        # Create dummy latent
        batch_size = 2
        spatial_tokens = 16  # 4×4 image
        packed = torch.randn(batch_size, spatial_tokens + 1, 128)

        # Set hw_vec (last token) to valid spatial dims
        packed[:, -1, 0] = 4 / 1024  # H normalized
        packed[:, -1, 1] = 4 / 1024  # W normalized

        with torch.no_grad():
            output = decoder(packed, use_context=False)

        # After 4 upscales: 4 -> 8 -> 16 -> 32 -> 64
        assert output.shape == (batch_size, 3, 64, 64)

    def test_bezier_decoder_forward_v060(self):
        """Test Bezier VAE decoder forward pass v0.6.0."""
        factory = ModelFactory(model_type="bezier", model_version="0.6.0", vae_dim=128)
        decoder = factory.create_vae_decoder(upscales=3, use_gradient_checkpointing=False)

        # Create dummy latent
        batch_size = 2
        spatial_tokens = 64  # 8×8 image
        packed = torch.randn(batch_size, spatial_tokens + 1, 128)

        # Set hw_vec (last token) to valid spatial dims
        packed[:, -1, 0] = 8 / 1024  # H normalized
        packed[:, -1, 1] = 8 / 1024  # W normalized

        with torch.no_grad():
            output = decoder(packed, use_context=False)

        # After 3 upscales: 8 -> 16 -> 32 -> 64
        assert output.shape == (batch_size, 3, 64, 64)


class TestFlowProcessorCreation:
    """Test Flow processor creation."""

    def test_create_bezier_flow_processor_v030(self):
        """Test Bezier Flow processor creation v0.3.0."""
        factory = ModelFactory(
            model_type="bezier", model_version="0.3.0", flow_d_model=512, vae_dim=128
        )
        flow = factory.create_flow_processor()

        assert flow is not None
        # Verify it has transformer blocks
        assert hasattr(flow, "transformer_blocks")
        assert len(flow.transformer_blocks) == 10  # Bezier uses 10 blocks

    def test_create_bezier_flow_processor_v060(self):
        """Test Bezier Flow processor creation v0.6.0."""
        factory = ModelFactory(
            model_type="bezier", model_version="0.6.0", flow_d_model=512, vae_dim=128
        )
        flow = factory.create_flow_processor()

        assert flow is not None
        # Verify it has transformer blocks
        assert hasattr(flow, "transformer_blocks")
        assert len(flow.transformer_blocks) == 10  # Bezier uses 10 blocks

    def test_create_baseline_flow_processor(self):
        """Test Baseline Flow processor creation (Phase 1.2 complete)."""
        factory = ModelFactory(model_type="baseline", baseline_flow_blocks=17)
        flow = factory.create_flow_processor()

        assert flow is not None
        assert hasattr(flow, "transformer_blocks")
        assert len(flow.transformer_blocks) == 17  # Baseline uses 17 blocks


class TestBaselineUpsamplerHelper:
    """Test baseline upsampler helper method."""

    def test_create_baseline_upsampler(self):
        """Test baseline upsampler creation."""
        factory = ModelFactory(
            model_type="baseline",
            baseline_activation="silu",
            baseline_vae_width_mult=4.5,
            baseline_vae_depth_mult=1.0,
        )

        upsampler = factory.create_baseline_upsampler(
            channels=64,
            steps=2,
            context_size=1024,
            use_gradient_checkpointing=False,
        )

        assert upsampler is not None
        assert len(upsampler.layers) == 2

    def test_baseline_upsampler_forward(self):
        """Test baseline upsampler forward pass."""
        factory = ModelFactory(model_type="baseline")
        upsampler = factory.create_baseline_upsampler(
            channels=64,
            steps=2,
            use_gradient_checkpointing=False,
        )

        x = torch.randn(2, 64, 8, 8)

        with torch.no_grad():
            output = upsampler(x, context=None)

        # 2 steps: 8 -> 16 -> 32
        assert output.shape == (2, 64, 32, 32)


class TestConvenienceFunctions:
    """Test convenience functions for model creation."""

    def test_create_bezier_models_v030(self):
        """Test create_bezier_models() convenience function v0.3.0."""
        vae_encoder, vae_decoder, flow, text_encoder = create_bezier_models(model_version="0.3.0")

        assert vae_encoder is not None
        assert vae_decoder is not None
        assert flow is not None
        assert text_encoder is not None

        # Verify dimensions match
        assert vae_encoder.d_model == 128
        # Flow processor has transformer_blocks
        assert hasattr(flow, "transformer_blocks")
        assert len(flow.transformer_blocks) == 10  # v0.3.0 uses 10 blocks

    def test_create_bezier_models_v060(self):
        """Test create_bezier_models() convenience function v0.6.0."""
        vae_encoder, vae_decoder, flow, text_encoder = create_bezier_models(model_version="0.6.0")

        assert vae_encoder is not None
        assert vae_decoder is not None
        assert flow is not None
        assert text_encoder is not None

        # Verify dimensions match
        assert vae_encoder.d_model == 128
        # Flow processor has transformer_blocks
        assert hasattr(flow, "transformer_blocks")
        assert len(flow.transformer_blocks) == 10  # v0.6.0 also uses 10 blocks

    def test_create_baseline_models_complete(self):
        """Test create_baseline_models() returns complete set (Phase 1.2 complete)."""
        vae_encoder, vae_decoder, flow, text_encoder = create_baseline_models()

        assert vae_encoder is not None
        assert vae_decoder is not None
        assert flow is not None  # Now implemented!
        assert text_encoder is not None

        # Verify baseline has 17 flow blocks
        assert len(flow.transformer_blocks) == 17


class TestParameterMatching:
    """Test that factory creates models with correct parameter counts."""

    def test_bezier_vae_encoder_params(self):
        """Test Bezier VAE encoder parameter count."""
        factory = ModelFactory(model_type="bezier", vae_dim=128)
        encoder = factory.create_vae_encoder()

        total_params = sum(p.numel() for p in encoder.parameters())

        # Should have reasonable param count (ballpark)
        assert total_params > 1_000_000  # At least 1M params
        assert total_params < 20_000_000  # Less than 20M params (encoder is large)

    def test_baseline_vae_encoder_matches_bezier(self):
        """Test Baseline VAE encoder has same params as Bezier (encoder is shared)."""
        bezier_factory = ModelFactory(model_type="bezier", vae_dim=128)
        baseline_factory = ModelFactory(model_type="baseline", vae_dim=128)

        bezier_encoder = bezier_factory.create_vae_encoder()
        baseline_encoder = baseline_factory.create_vae_encoder()

        bezier_params = sum(p.numel() for p in bezier_encoder.parameters())
        baseline_params = sum(p.numel() for p in baseline_encoder.parameters())

        # Encoders should be identical (same class)
        assert bezier_params == baseline_params


class TestModelFactoryV080:
    """Test ModelFactory with v0.8.0 (pillar-attention)."""

    def test_factory_init_v080(self):
        """Factory should accept model_version='0.8.0'."""
        factory = ModelFactory(model_type="bezier", model_version="0.8.0", vae_dim=128)
        assert factory.model_version == "0.8.0"

    def test_create_flow_processor_v080(self):
        """Factory should return FluxFlowProcessor_v080 for version 0.8.0."""
        from fluxflow.models.v080.flow import FluxFlowProcessor_v080

        factory = ModelFactory(
            model_type="bezier",
            model_version="0.8.0",
            vae_dim=32,
            flow_d_model=32,
            flow_embedding_size=32,
            bezier_flow_blocks=2,
        )
        proc = factory.create_flow_processor(n_head=4)
        assert isinstance(proc, FluxFlowProcessor_v080)

    def test_v080_flow_processor_has_film(self):
        """FluxFlowProcessor_v080 blocks must have FiLM layers."""
        from fluxflow.models.v080.flow import FluxTransformerBlock_v080

        factory = ModelFactory(
            model_type="bezier",
            model_version="0.8.0",
            vae_dim=32,
            flow_d_model=32,
            flow_embedding_size=32,
            bezier_flow_blocks=2,
        )
        proc = factory.create_flow_processor(n_head=4)
        blk = proc.transformer_blocks[0]
        assert isinstance(blk, FluxTransformerBlock_v080)
        assert hasattr(blk, "film_p0")
        assert hasattr(blk, "pillar_cross_attn")

    def test_v080_forward_pass(self):
        """v0.8.0 processor forward should preserve packed shape."""
        from fluxflow.models.v070.vae import CONTEXT_DIMS

        factory = ModelFactory(
            model_type="bezier",
            model_version="0.8.0",
            vae_dim=32,
            flow_d_model=32,
            flow_embedding_size=32,
            bezier_flow_blocks=2,
        )
        proc = factory.create_flow_processor(n_head=4, max_hw=64)

        T = 16
        vae_dim = 32
        B = 1
        packed = torch.randn(B, T + 1, vae_dim + CONTEXT_DIMS)
        packed[:, -1, :] = 0
        packed[:, -1, 0] = 8 / 64  # H
        packed[:, -1, 1] = 8 / 64  # W
        text_emb = torch.randn(B, 32)
        t = torch.tensor([0.5])

        with torch.no_grad():
            out = proc(packed, text_emb, t)

        assert out.shape == packed.shape


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
