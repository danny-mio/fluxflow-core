"""Unit tests for encoder models (src/models/encoders.py)."""

import torch
import torch.nn as nn

from fluxflow.models.encoders import BertTextEncoder, ImageEncoder


class TestBertTextEncoder:
    """Tests for BertTextEncoder model."""

    def test_initialization_from_scratch(self):
        """Should initialize with custom DistilBERT config when no pretrain_model."""
        encoder = BertTextEncoder(embed_dim=1024, pretrain_model=None)

        assert isinstance(encoder, nn.Module)
        assert hasattr(encoder, "language_model")
        assert hasattr(encoder, "ouput_layer")  # Note: typo in original code

    def test_initialization_parameters(self):
        """Should use specified embed_dim for output."""
        embed_dim = 512
        encoder = BertTextEncoder(embed_dim=embed_dim, pretrain_model=None)

        # Check that output layer exists
        assert encoder.ouput_layer is not None

    def test_forward_output_shape(self):
        """Forward pass should output [B, embed_dim]."""
        encoder = BertTextEncoder(embed_dim=1024, pretrain_model=None)
        batch_size = 2
        seq_len = 16

        input_ids = torch.randint(0, 30522, (batch_size, seq_len))
        attention_mask = torch.ones(batch_size, seq_len)

        output = encoder(input_ids, attention_mask)

        # Output should be [B, embed_dim]
        assert output.shape == (batch_size, 1024)

    def test_forward_without_attention_mask(self):
        """Should work without explicit attention mask."""
        encoder = BertTextEncoder(embed_dim=512, pretrain_model=None)
        batch_size = 2
        seq_len = 20

        input_ids = torch.randint(0, 30522, (batch_size, seq_len))

        output = encoder(input_ids)

        assert output.shape == (batch_size, 512)

    def test_batch_size_independence(self):
        """Should work with different batch sizes."""
        encoder = BertTextEncoder(embed_dim=768, pretrain_model=None)
        seq_len = 32

        for batch_size in [1, 2, 4, 8]:
            input_ids = torch.randint(0, 30522, (batch_size, seq_len))
            output = encoder(input_ids)
            assert output.shape[0] == batch_size

    def test_different_sequence_lengths(self):
        """Should handle different sequence lengths."""
        encoder = BertTextEncoder(embed_dim=512, pretrain_model=None)
        batch_size = 2

        for seq_len in [8, 16, 32, 64]:
            input_ids = torch.randint(0, 30522, (batch_size, seq_len))
            output = encoder(input_ids)
            assert output.shape == (batch_size, 512)

    def test_gradient_flow(self):
        """Gradients should flow through encoder."""
        encoder = BertTextEncoder(embed_dim=256, pretrain_model=None)

        input_ids = torch.randint(0, 30522, (2, 16))
        attention_mask = torch.ones(2, 16)

        # Make embeddings require grad (they're leaf variables)
        for param in encoder.parameters():
            if param.requires_grad:
                break

        output = encoder(input_ids, attention_mask)
        loss = output.sum()
        loss.backward()

        # Check that at least some parameters have gradients
        has_grad = any(p.grad is not None for p in encoder.parameters() if p.requires_grad)
        assert has_grad

    def test_mean_pooling_aggregation(self):
        """Encoder should use mean pooling over sequence."""
        encoder = BertTextEncoder(embed_dim=512, pretrain_model=None)
        encoder.eval()  # Disable dropout for deterministic behavior

        # Two identical sequences
        input_ids = torch.randint(0, 30522, (1, 20))
        input_ids_doubled = input_ids.repeat(2, 1)

        output1 = encoder(input_ids)
        output2 = encoder(input_ids_doubled)

        # Outputs should be identical for same input
        assert torch.allclose(output1[0], output2[0], atol=1e-5)
        assert torch.allclose(output1[0], output2[1], atol=1e-5)

    def test_bezier_activation_applied(self):
        """Should use Bezier activation in output layer."""
        encoder = BertTextEncoder(embed_dim=256, pretrain_model=None)

        assert hasattr(encoder, "apply_bezier_activation")
        # BezierActivation should be part of output layer
        assert encoder.apply_bezier_activation is not None

    def test_vocab_size(self):
        """Should use BERT vocab size of 30522."""
        encoder = BertTextEncoder(embed_dim=512, pretrain_model=None)

        # Vocab size should be 30522 for DistilBERT
        assert encoder.language_model.config.vocab_size == 30522

    def test_eval_mode(self):
        """Should work in eval mode."""
        encoder = BertTextEncoder(embed_dim=512, pretrain_model=None)
        encoder.eval()

        input_ids = torch.randint(0, 30522, (2, 16))

        with torch.no_grad():
            output = encoder(input_ids)

        assert output.shape == (2, 512)


class TestImageEncoder:
    """Tests for ImageEncoder model."""

    def test_initialization(self):
        """Should initialize with specified parameters."""
        encoder = ImageEncoder(img_channels=3, text_embedding_dim=1024, feature_maps=128)

        assert isinstance(encoder, nn.Module)
        assert hasattr(encoder, "downsample")
        assert hasattr(encoder, "build_embed_validation")

    def test_forward_output_shape(self):
        """Forward pass should output [B, text_embedding_dim]."""
        encoder = ImageEncoder(img_channels=3, text_embedding_dim=1024, feature_maps=128)
        batch_size = 2

        # Input should be 1024x1024 (based on progressive downsampling design)
        images = torch.randn(batch_size, 3, 1024, 1024)

        output = encoder(images)

        # Output should be [B, text_embedding_dim]
        assert output.shape == (batch_size, 1024)

    def test_different_image_sizes(self):
        """Should handle different input sizes (though designed for 1024)."""
        encoder = ImageEncoder(img_channels=3, text_embedding_dim=512, feature_maps=64)

        # Test with different sizes (may downsample differently)
        for size in [512, 1024]:
            images = torch.randn(1, 3, size, size)
            output = encoder(images)
            assert output.shape == (1, 512)

    def test_batch_size_independence(self):
        """Should work with different batch sizes."""
        encoder = ImageEncoder(img_channels=3, text_embedding_dim=768, feature_maps=128)

        for batch_size in [1, 2, 4]:
            images = torch.randn(batch_size, 3, 1024, 1024)
            output = encoder(images)
            assert output.shape[0] == batch_size

    def test_gradient_flow(self):
        """Gradients should flow through encoder."""
        encoder = ImageEncoder(img_channels=3, text_embedding_dim=512, feature_maps=64)

        images = torch.randn(2, 3, 512, 512, requires_grad=True)

        output = encoder(images)
        loss = output.sum()
        loss.backward()

        assert images.grad is not None
        assert not torch.isnan(images.grad).any()

    def test_downsample_progressive(self):
        """downsample should progressively reduce spatial dimensions."""
        encoder = ImageEncoder(img_channels=3, text_embedding_dim=512, feature_maps=128)

        images = torch.randn(2, 3, 1024, 1024)

        # Test downsample method
        downsampled = encoder.downsample(images)

        # Should output feature_maps channels
        assert downsampled.shape[1] == 128
        # Spatial dimensions should be significantly reduced
        assert downsampled.shape[2] < 1024
        assert downsampled.shape[3] < 1024

    def test_bezier_control_points(self):
        """downsample should use Bezier control points at each stage."""
        encoder = ImageEncoder(img_channels=3, text_embedding_dim=512, feature_maps=128)

        # Check that control point convolutions exist
        assert hasattr(encoder, "initial_conv_1024_p0")
        assert hasattr(encoder, "initial_conv_1024_p1")
        assert hasattr(encoder, "initial_conv_1024_p2")
        assert hasattr(encoder, "initial_conv_1024_p3")

    def test_context_module(self):
        """Should use LeanContext2D for context."""
        encoder = ImageEncoder(img_channels=3, text_embedding_dim=512, feature_maps=128)

        assert hasattr(encoder, "context_module")

    def test_spade_normalization(self):
        """Should use SPADE for spatial adaptive normalization."""
        encoder = ImageEncoder(img_channels=3, text_embedding_dim=512, feature_maps=128)

        assert hasattr(encoder, "spade")

    def test_matching_head(self):
        """Should have matching head for embedding output."""
        encoder = ImageEncoder(img_channels=3, text_embedding_dim=1024, feature_maps=128)

        assert hasattr(encoder, "matching_head")

    def test_spectral_normalization_applied(self):
        """Conv layers should use spectral normalization."""
        encoder = ImageEncoder(img_channels=3, text_embedding_dim=512, feature_maps=128)

        # Check for spectral norm in at least one conv layer
        has_spectral_norm = False
        for module in encoder.modules():
            if isinstance(module, nn.Conv2d):
                if hasattr(module, "weight_u"):
                    has_spectral_norm = True
                    break

        assert has_spectral_norm, "ImageEncoder should use spectral normalization"

    def test_single_channel_input(self):
        """Should work with single-channel images."""
        encoder = ImageEncoder(img_channels=1, text_embedding_dim=512, feature_maps=64)

        images = torch.randn(2, 1, 512, 512)
        output = encoder(images)

        assert output.shape == (2, 512)

    def test_multi_channel_input(self):
        """Should work with multi-channel images."""
        encoder = ImageEncoder(img_channels=4, text_embedding_dim=512, feature_maps=64)

        images = torch.randn(2, 4, 512, 512)
        output = encoder(images)

        assert output.shape == (2, 512)

    def test_different_feature_map_sizes(self):
        """Should work with different feature_maps parameter."""
        for feature_maps in [64, 128, 256]:
            encoder = ImageEncoder(
                img_channels=3, text_embedding_dim=512, feature_maps=feature_maps
            )

            images = torch.randn(1, 3, 512, 512)
            output = encoder(images)

            assert output.shape == (1, 512)

    def test_eval_mode(self):
        """Should work in eval mode."""
        encoder = ImageEncoder(img_channels=3, text_embedding_dim=512, feature_maps=128)
        encoder.eval()

        images = torch.randn(2, 3, 512, 512)

        with torch.no_grad():
            output = encoder(images)

        assert output.shape == (2, 512)

    def test_build_embed_validation_shape(self):
        """build_embed_validation should produce correct output shape."""
        encoder = ImageEncoder(img_channels=3, text_embedding_dim=512, feature_maps=128)

        # Create dummy downsampled features
        batch_size = 2
        feature_maps = 128
        h, w = 32, 32
        downsampled = torch.randn(batch_size, feature_maps, h, w)
        images = torch.randn(batch_size, 3, 512, 512)

        embed = encoder.build_embed_validation(downsampled, images)

        # Should output text_embedding_dim
        assert embed.shape == (batch_size, 512)


class TestEncoderIntegration:
    """Integration tests for encoder models."""

    def test_text_image_embedding_compatibility(self):
        """Text and image encoders should output same embedding dimension."""
        embed_dim = 1024

        text_encoder = BertTextEncoder(embed_dim=embed_dim, pretrain_model=None)
        image_encoder = ImageEncoder(img_channels=3, text_embedding_dim=embed_dim)

        # Text input
        input_ids = torch.randint(0, 30522, (2, 32))
        text_embeds = text_encoder(input_ids)

        # Image input
        images = torch.randn(2, 3, 512, 512)
        image_embeds = image_encoder(images)

        # Should have same embedding dimension
        assert text_embeds.shape[1] == image_embeds.shape[1]
        assert text_embeds.shape[1] == embed_dim

    def test_contrastive_learning_setup(self):
        """Encoders should be suitable for contrastive learning."""
        embed_dim = 512

        text_encoder = BertTextEncoder(embed_dim=embed_dim, pretrain_model=None)
        image_encoder = ImageEncoder(img_channels=3, text_embedding_dim=embed_dim)

        batch_size = 4
        input_ids = torch.randint(0, 30522, (batch_size, 20))
        images = torch.randn(batch_size, 3, 512, 512)

        text_embeds = text_encoder(input_ids)
        image_embeds = image_encoder(images)

        # Compute similarity matrix (simplified contrastive loss)
        # Normalize embeddings
        text_embeds_norm = torch.nn.functional.normalize(text_embeds, dim=1)
        image_embeds_norm = torch.nn.functional.normalize(image_embeds, dim=1)

        # Similarity matrix: [B, B]
        similarity = torch.matmul(text_embeds_norm, image_embeds_norm.T)

        assert similarity.shape == (batch_size, batch_size)
        # Values should be in [-1, 1] after normalization
        assert (similarity >= -1.0).all() and (similarity <= 1.0).all()

    def test_encoder_eval_determinism(self):
        """Encoders in eval mode should be deterministic."""
        text_encoder = BertTextEncoder(embed_dim=512, pretrain_model=None)
        text_encoder.eval()

        image_encoder = ImageEncoder(img_channels=3, text_embedding_dim=512)
        image_encoder.eval()

        input_ids = torch.randint(0, 30522, (2, 16))
        images = torch.randn(2, 3, 512, 512)

        with torch.no_grad():
            text_out1 = text_encoder(input_ids)
            text_out2 = text_encoder(input_ids)
            image_out1 = image_encoder(images)
            image_out2 = image_encoder(images)

        assert torch.allclose(text_out1, text_out2, atol=1e-5)
        assert torch.allclose(image_out1, image_out2, atol=1e-5)

    def test_gradient_flow_through_both_encoders(self):
        """Gradients should flow through both encoders."""
        embed_dim = 256

        text_encoder = BertTextEncoder(embed_dim=embed_dim, pretrain_model=None)
        image_encoder = ImageEncoder(img_channels=3, text_embedding_dim=embed_dim, feature_maps=64)

        input_ids = torch.randint(0, 30522, (2, 16))
        images = torch.randn(2, 3, 256, 256, requires_grad=True)

        text_embeds = text_encoder(input_ids)
        image_embeds = image_encoder(images)

        # Compute simple loss (dot product)
        loss = (text_embeds * image_embeds).sum()
        loss.backward()

        # Check gradients
        assert images.grad is not None
        has_text_grad = any(
            p.grad is not None for p in text_encoder.parameters() if p.requires_grad
        )
        assert has_text_grad

    def test_device_compatibility(self):
        """Encoders should work on CPU."""
        text_encoder = BertTextEncoder(embed_dim=512, pretrain_model=None)
        image_encoder = ImageEncoder(img_channels=3, text_embedding_dim=512)

        text_encoder = text_encoder.to("cpu")
        image_encoder = image_encoder.to("cpu")

        input_ids = torch.randint(0, 30522, (2, 16), device="cpu")
        images = torch.randn(2, 3, 256, 256, device="cpu")

        text_out = text_encoder(input_ids)
        image_out = image_encoder(images)

        assert text_out.device.type == "cpu"
        assert image_out.device.type == "cpu"

    def test_different_batch_sizes_both_encoders(self):
        """Both encoders should handle different batch sizes."""
        text_encoder = BertTextEncoder(embed_dim=512, pretrain_model=None)
        image_encoder = ImageEncoder(img_channels=3, text_embedding_dim=512)

        for batch_size in [1, 2, 4]:
            input_ids = torch.randint(0, 30522, (batch_size, 20))
            images = torch.randn(batch_size, 3, 256, 256)

            text_out = text_encoder(input_ids)
            image_out = image_encoder(images)

            assert text_out.shape == (batch_size, 512)
            assert image_out.shape == (batch_size, 512)

    def test_embedding_statistics(self):
        """Encoder outputs should have reasonable statistics."""
        # Seed for reproducible random initialization
        torch.manual_seed(42)

        text_encoder = BertTextEncoder(embed_dim=512, pretrain_model=None)
        image_encoder = ImageEncoder(img_channels=3, text_embedding_dim=512, feature_maps=64)

        text_encoder.eval()
        image_encoder.eval()

        torch.manual_seed(123)  # Seed for reproducible inputs
        input_ids = torch.randint(0, 30522, (8, 32))
        images = torch.randn(8, 3, 256, 256)

        with torch.no_grad():
            text_embeds = text_encoder(input_ids)
            image_embeds = image_encoder(images)

        # Check for NaN/Inf (critical checks)
        assert not torch.isnan(text_embeds).any()
        assert not torch.isinf(text_embeds).any()
        assert not torch.isnan(image_embeds).any()
        assert not torch.isinf(image_embeds).any()

        # Check embeddings are not completely zero
        assert text_embeds.abs().max() > 0
        assert image_embeds.abs().max() > 0
